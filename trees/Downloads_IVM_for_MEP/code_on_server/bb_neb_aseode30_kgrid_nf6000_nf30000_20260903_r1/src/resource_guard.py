"""Per-machine heavy-task lock and macOS resource fail-stop monitor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import threading
import time
from typing import Any, Iterable


_ACTIVE_GUARD: "ResourceGuard | None" = None


def assert_active_guard_safe() -> None:
    """Called immediately before each physical Force attempt."""

    guard = _ACTIVE_GUARD
    if guard is not None:
        guard.assert_safe()


class ResourceGuardError(RuntimeError):
    pass


class HeavyTaskAlreadyRunning(ResourceGuardError):
    pass


class ResourcePreflightRejected(ResourceGuardError):
    pass


class ResourceFailStop(ResourceGuardError):
    pass


@dataclass(frozen=True)
class ResourceSnapshot:
    utc: str
    monotonic_seconds: float
    machine: str
    pid: int
    batch_id: int | None
    memory_free_percent: int
    pressure_state: str
    available_bytes_estimate: int
    below_2gb_warning: bool
    page_size_bytes: int
    pageouts: int
    swapouts: int
    rss_bytes: int
    disk_free_bytes: int
    disk_total_bytes: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        raise ResourceGuardError(
            f"resource command failed ({completed.returncode}): {' '.join(command)}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def parse_memory_pressure_free_percent(text: str) -> int:
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", text)
    if not match:
        raise ResourceGuardError("could not parse memory_pressure -Q output")
    value = int(match.group(1))
    if not 0 <= value <= 100:
        raise ResourceGuardError("memory_pressure free percentage is outside [0,100]")
    return value


def parse_vm_stat(text: str) -> dict[str, int]:
    page_match = re.search(r"page size of\s+(\d+)\s+bytes", text)
    if not page_match:
        raise ResourceGuardError("could not parse vm_stat page size")
    result = {"page_size_bytes": int(page_match.group(1))}
    keys = {
        "Pages free": "pages_free",
        "Pages inactive": "pages_inactive",
        "Pages speculative": "pages_speculative",
        "Pages purgeable": "pages_purgeable",
        "Pageouts": "pageouts",
        "Swapouts": "swapouts",
    }
    for source_key, target_key in keys.items():
        match = re.search(rf'^"?{re.escape(source_key)}"?:\s+(\d+)\.', text, re.MULTILINE)
        if not match:
            raise ResourceGuardError(f"could not parse vm_stat field {source_key!r}")
        result[target_key] = int(match.group(1))
    return result


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # ru_maxrss is bytes on macOS and KiB on Linux.  Production is macOS, but
    # this branch keeps zero-Force unit tests meaningful on other hosts.
    if os.uname().sysname == "Darwin":
        return raw
    return raw * 1024


def _pressure_state(free_percent: int) -> str:
    if free_percent >= 20:
        return "normal"
    if free_percent >= 10:
        return "warning"
    return "critical"


class ResourceGuard:
    """Hold the project-wide heavy lock and audit resource state per batch.

    Thresholds are fixed code constants, not adaptive knobs:

    * new task rejected below 10% memory-pressure free percentage;
    * warning remains allowed for the one locked task, but a continuous
      warning episode of at least 60 seconds fail-stops;
    * critical fail-stops after at least two observations spanning 15 seconds;
    * rapid pageout/swap means both >=32768 pages and >=4096 pages/min;
    * one >=1 GiB peak-RSS jump, or three >=256 MiB jumps, fail-stops;
    * 2 GiB available is warning-only; disk below 5 GiB rejects start and
      below 2 GiB at runtime fail-stops.
    """

    START_DISK_FREE_BYTES = 5 * 1024**3
    RUNTIME_DISK_FREE_BYTES = 2 * 1024**3
    TWO_GB = 2 * 1024**3
    WARNING_STREAK_LIMIT = 3
    CRITICAL_STREAK_LIMIT = 2
    WARNING_SUSTAINED_SECONDS = 60.0
    CRITICAL_SUSTAINED_SECONDS = 15.0
    RAPID_VM_PAGES = 32768
    RAPID_VM_PAGES_PER_MINUTE = 4096.0
    RSS_ONE_JUMP_BYTES = 1024**3
    RSS_GROWTH_BYTES = 256 * 1024**2
    RSS_GROWTH_STREAK_LIMIT = 3
    BACKGROUND_INTERVAL_SECONDS = 15.0

    def __init__(
        self,
        run_dir: Path | str,
        machine: str,
        *,
        lock_paths: Iterable[Path | str] | None = None,
    ) -> None:
        if not machine or not isinstance(machine, str):
            raise ResourceGuardError("machine must be a nonempty identity string")
        self.run_dir = Path(run_dir).resolve()
        self.machine = machine
        self.project_root = Path(__file__).resolve().parent.parent
        requested_locks = tuple(Path(value).resolve() for value in (lock_paths or ()))
        if requested_locks and len(requested_locks) != 2:
            raise ResourceGuardError("explicit k-grid policy requires exactly two lock paths")
        if len(set(requested_locks)) != len(requested_locks):
            raise ResourceGuardError("host-local lock paths must be distinct")
        self.lock_paths = tuple(sorted(requested_locks, key=str))
        self.lock_path = (
            self.lock_paths[0] if self.lock_paths else self.project_root / ".heavy_task.lock"
        )
        self.monitor_path = self.run_dir / "resource_monitor.jsonl"
        self._lock_stream: Any | None = None
        self._lock_streams: list[Any] = []
        self._monitor_stream: Any | None = None
        self._preflight_done = False
        self._closed = False
        self._last: ResourceSnapshot | None = None
        self._warning_streak = 0
        self._critical_streak = 0
        self._rss_growth_streak = 0
        self._warning_since: float | None = None
        self._critical_since: float | None = None
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._async_failure: str | None = None
        self._last_batch_id = 0

    def _acquire_lock(self) -> None:
        if self._lock_streams:
            return
        paths = self.lock_paths or (self.lock_path,)
        try:
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    stream.seek(0)
                    owner = stream.read().strip()
                    stream.close()
                    raise HeavyTaskAlreadyRunning(
                        f"another local heavy task holds {path}; owner={owner!r}"
                    ) from error
                self._lock_streams.append(stream)
            owner_record = (
                json.dumps(
                    {
                        "schema": "bb-cleanroom-dual-heavy-lock-v1",
                        "pid": os.getpid(),
                        "machine": self.machine,
                        "run_dir": str(self.run_dir),
                        "lock_paths": [str(path) for path in paths],
                        "utc": _utc_now(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            for stream in self._lock_streams:
                stream.seek(0)
                stream.truncate()
                stream.write(owner_record)
                stream.flush()
                os.fsync(stream.fileno())
            self._lock_stream = self._lock_streams[0]
        except Exception:
            for stream in reversed(self._lock_streams):
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                finally:
                    stream.close()
            self._lock_streams = []
            self._lock_stream = None
            raise

    def _open_monitor(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            self.monitor_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
            0o600,
        )
        self._monitor_stream = os.fdopen(fd, "ab", buffering=0)

    def _append(self, event: str, snapshot: ResourceSnapshot, **fields: Any) -> None:
        if self._monitor_stream is None:
            raise ResourceGuardError("resource monitor is not open")
        payload = {
            "schema": "bb-cleanroom-resource-monitor-v1",
            "event": event,
            "snapshot": asdict(snapshot),
            **fields,
        }
        self._monitor_stream.write(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        os.fsync(self._monitor_stream.fileno())

    def _snapshot(self, batch_id: int | None) -> ResourceSnapshot:
        pressure_output = _run(["/usr/bin/memory_pressure", "-Q"])
        vm = parse_vm_stat(_run(["/usr/bin/vm_stat"]))
        free_percent = parse_memory_pressure_free_percent(pressure_output)
        available_pages = (
            vm["pages_free"]
            + vm["pages_inactive"]
            + vm["pages_speculative"]
            + vm["pages_purgeable"]
        )
        available_bytes = available_pages * vm["page_size_bytes"]
        disk = shutil.disk_usage(self.project_root)
        return ResourceSnapshot(
            utc=_utc_now(),
            monotonic_seconds=time.monotonic(),
            machine=self.machine,
            pid=os.getpid(),
            batch_id=batch_id,
            memory_free_percent=free_percent,
            pressure_state=_pressure_state(free_percent),
            available_bytes_estimate=available_bytes,
            below_2gb_warning=available_bytes < self.TWO_GB,
            page_size_bytes=vm["page_size_bytes"],
            pageouts=vm["pageouts"],
            swapouts=vm["swapouts"],
            rss_bytes=_peak_rss_bytes(),
            disk_free_bytes=disk.free,
            disk_total_bytes=disk.total,
        )

    def preflight(self) -> dict[str, Any]:
        if self._closed:
            raise ResourceGuardError("closed ResourceGuard cannot be reused")
        if self._preflight_done:
            raise ResourceGuardError("preflight may be called exactly once")
        self._acquire_lock()
        try:
            self._open_monitor()
            snapshot = self._snapshot(batch_id=None)
            rejection: str | None = None
            if snapshot.memory_free_percent < 10:
                rejection = (
                    f"memory_pressure free {snapshot.memory_free_percent}% is below the "
                    "10% new-task gate"
                )
            elif snapshot.disk_free_bytes < self.START_DISK_FREE_BYTES:
                rejection = (
                    f"disk free {snapshot.disk_free_bytes} is below the frozen 5 GiB start gate"
                )
            if rejection is not None:
                self._append("preflight_rejected", snapshot, reason=rejection)
                self.close()
                raise ResourcePreflightRejected(rejection)
            self._append(
                "preflight_pass",
                snapshot,
                mode="normal" if snapshot.memory_free_percent >= 20 else "warning_single_task",
                two_gb_is_warning_only=True,
                thresholds=self.thresholds(),
            )
            self._last = snapshot
            self._preflight_done = True
            global _ACTIVE_GUARD
            if _ACTIVE_GUARD is not None and _ACTIVE_GUARD is not self:
                raise HeavyTaskAlreadyRunning("another ResourceGuard is active in this process")
            _ACTIVE_GUARD = self
            self._monitor_thread = threading.Thread(
                target=self._background_loop,
                name="bb-cleanroom-resource-monitor",
                daemon=True,
            )
            self._monitor_thread.start()
            return asdict(snapshot)
        except Exception:
            if not self._preflight_done and not self._closed:
                self.close()
            raise

    def checkpoint(self, batch_id: int) -> dict[str, Any]:
        with self._state_lock:
            self.assert_safe()
            if not isinstance(batch_id, int) or batch_id < 0:
                raise ResourceGuardError("batch_id must be a nonnegative integer")
            self._last_batch_id = batch_id
            return self._checkpoint_locked(batch_id=batch_id, source="batch_boundary")

    def rebaseline(self, reason: str) -> dict[str, Any]:
        """Reset growth deltas after a known zero-Force initialization event."""

        if not isinstance(reason, str) or not reason:
            raise ResourceGuardError("rebaseline reason must be nonempty")
        with self._state_lock:
            self.assert_safe()
            current = self._snapshot(batch_id=self._last_batch_id)
            self._last = current
            self._rss_growth_streak = 0
            self._append("rebaseline", current, reason=reason)
            return asdict(current)

    def _checkpoint_locked(self, *, batch_id: int, source: str) -> dict[str, Any]:
        current = self._snapshot(batch_id=batch_id)
        previous = self._last
        assert previous is not None

        if current.pressure_state == "normal":
            self._warning_streak = 0
            self._critical_streak = 0
            self._warning_since = None
            self._critical_since = None
        elif current.pressure_state == "warning":
            self._warning_streak += 1
            self._critical_streak = 0
            if self._warning_since is None:
                self._warning_since = current.monotonic_seconds
            self._critical_since = None
        else:
            self._critical_streak += 1
            self._warning_streak = 0
            if self._critical_since is None:
                self._critical_since = current.monotonic_seconds
            self._warning_since = None

        warning_duration = (
            0.0
            if self._warning_since is None
            else current.monotonic_seconds - self._warning_since
        )
        critical_duration = (
            0.0
            if self._critical_since is None
            else current.monotonic_seconds - self._critical_since
        )

        elapsed_minutes = max(
            (current.monotonic_seconds - previous.monotonic_seconds) / 60.0,
            1.0 / 600.0,
        )
        pageout_delta = max(0, current.pageouts - previous.pageouts)
        swapout_delta = max(0, current.swapouts - previous.swapouts)
        pageout_rate = pageout_delta / elapsed_minutes
        swapout_rate = swapout_delta / elapsed_minutes
        rapid_pageout = (
            pageout_delta >= self.RAPID_VM_PAGES
            and pageout_rate >= self.RAPID_VM_PAGES_PER_MINUTE
        )
        rapid_swapout = (
            swapout_delta >= self.RAPID_VM_PAGES
            and swapout_rate >= self.RAPID_VM_PAGES_PER_MINUTE
        )

        rss_delta = max(0, current.rss_bytes - previous.rss_bytes)
        if rss_delta >= self.RSS_GROWTH_BYTES:
            self._rss_growth_streak += 1
        else:
            self._rss_growth_streak = 0

        reasons: list[str] = []
        if (
            self._warning_streak >= 2
            and warning_duration >= self.WARNING_SUSTAINED_SECONDS
        ):
            reasons.append(
                f"sustained warning pressure for {warning_duration:.1f} seconds "
                f"across {self._warning_streak} observations"
            )
        if (
            self._critical_streak >= self.CRITICAL_STREAK_LIMIT
            and critical_duration >= self.CRITICAL_SUSTAINED_SECONDS
        ):
            reasons.append(
                f"sustained critical pressure for {critical_duration:.1f} seconds "
                f"across {self._critical_streak} observations"
            )
        if rapid_pageout:
            reasons.append(
                f"rapid pageout delta={pageout_delta} rate={pageout_rate:.1f} pages/min"
            )
        if rapid_swapout:
            reasons.append(
                f"rapid swapout delta={swapout_delta} rate={swapout_rate:.1f} pages/min"
            )
        if rss_delta >= self.RSS_ONE_JUMP_BYTES:
            reasons.append(f"abnormal one-checkpoint RSS growth {rss_delta} bytes")
        if self._rss_growth_streak >= self.RSS_GROWTH_STREAK_LIMIT:
            reasons.append(
                f"sustained RSS growth for {self._rss_growth_streak} checkpoints"
            )
        if current.disk_free_bytes < self.RUNTIME_DISK_FREE_BYTES:
            reasons.append(
                f"disk free {current.disk_free_bytes} below frozen 2 GiB runtime gate"
            )

        fields = {
            "warning_streak": self._warning_streak,
            "critical_streak": self._critical_streak,
            "warning_duration_seconds": warning_duration,
            "critical_duration_seconds": critical_duration,
            "pageout_delta": pageout_delta,
            "swapout_delta": swapout_delta,
            "pageout_pages_per_minute": pageout_rate,
            "swapout_pages_per_minute": swapout_rate,
            "rss_delta_bytes": rss_delta,
            "rss_growth_streak": self._rss_growth_streak,
            "two_gb_is_warning_only": True,
        }
        self._last = current
        if reasons:
            self._append("runtime_fail_stop", current, reasons=reasons, source=source, **fields)
            raise ResourceFailStop("; ".join(reasons))
        self._append("checkpoint", current, source=source, **fields)
        return {**asdict(current), **fields}

    def _background_loop(self) -> None:
        while not self._stop_event.wait(self.BACKGROUND_INTERVAL_SECONDS):
            with self._state_lock:
                if self._closed or self._async_failure is not None:
                    return
                try:
                    self._checkpoint_locked(
                        batch_id=self._last_batch_id,
                        source="continuous_15s_monitor",
                    )
                except Exception as error:
                    self._async_failure = f"{type(error).__name__}: {error}"
                    return

    def assert_safe(self, batch_id: int | None = None, slot_id: int | None = None) -> None:
        """Cheap cached safety latch; performs no subprocess or Force work."""

        if not self._preflight_done or self._closed:
            raise ResourceGuardError("physical call requires a live successful ResourceGuard")
        if self._async_failure is not None:
            raise ResourceFailStop(self._async_failure)

    @classmethod
    def thresholds(cls) -> dict[str, Any]:
        return {
            "normal_memory_free_percent_min": 20,
            "warning_memory_free_percent_min": 10,
            "new_task_memory_free_percent_min": 10,
            "warning_streak_limit": cls.WARNING_STREAK_LIMIT,
            "critical_streak_limit": cls.CRITICAL_STREAK_LIMIT,
            "warning_sustained_seconds": cls.WARNING_SUSTAINED_SECONDS,
            "critical_sustained_seconds": cls.CRITICAL_SUSTAINED_SECONDS,
            "rapid_vm_pages": cls.RAPID_VM_PAGES,
            "rapid_vm_pages_per_minute": cls.RAPID_VM_PAGES_PER_MINUTE,
            "rss_one_jump_bytes": cls.RSS_ONE_JUMP_BYTES,
            "rss_growth_bytes": cls.RSS_GROWTH_BYTES,
            "rss_growth_streak_limit": cls.RSS_GROWTH_STREAK_LIMIT,
            "available_2gb_policy": "warning_only",
            "start_disk_free_bytes": cls.START_DISK_FREE_BYTES,
            "runtime_disk_free_bytes": cls.RUNTIME_DISK_FREE_BYTES,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._stop_event.set()
        if (
            self._monitor_thread is not None
            and self._monitor_thread is not threading.current_thread()
        ):
            self._monitor_thread.join(timeout=35.0)
        self._monitor_thread = None
        with self._state_lock:
            if self._monitor_stream is not None:
                self._monitor_stream.flush()
                os.fsync(self._monitor_stream.fileno())
                self._monitor_stream.close()
                self._monitor_stream = None
            for stream in reversed(self._lock_streams):
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                stream.close()
            self._lock_streams = []
            self._lock_stream = None
        global _ACTIVE_GUARD
        if _ACTIVE_GUARD is self:
            _ACTIVE_GUARD = None
        self._closed = True

    def __enter__(self) -> "ResourceGuard":
        self.preflight()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
