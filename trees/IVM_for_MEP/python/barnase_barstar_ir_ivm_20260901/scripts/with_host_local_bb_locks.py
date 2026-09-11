#!/usr/bin/env python3
"""Hold both host-local BB fcntl locks around one exact task.

The normal mode launches a child command only after both locks are acquired.
The guard mode is a transparent remediation for an already-running exact PID:
it verifies the PID command, acquires both locks, and holds them until that PID
exits.  Guard mode does not claim that locks were held before its receipt time.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import platform
from pathlib import Path
import shutil
import socket
import subprocess
import time


DEFAULT_LOCKS = (
    Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock"),
    Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pid_command(pid: int) -> str:
    result = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "command="],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def launch_facts() -> dict:
    processor = platform.processor()
    sysctl = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if sysctl.returncode == 0 and sysctl.stdout.strip():
        processor = sysctl.stdout.strip()
    disk_path = Path("/Users/wuzhiyou/IVM_for_MEP").resolve(strict=True)
    disk = shutil.disk_usage(disk_path)
    return {
        "host": {
            "hostname": socket.gethostname(),
            "architecture": platform.machine(),
            "processor": processor,
        },
        "working_directory": str(Path.cwd().resolve()),
        "launch_disk": {
            "path": str(disk_path),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }


def write_receipt(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def acquire(paths: tuple[Path, ...]):
    streams = []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a+")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except Exception:
                stream.close()
                raise
            streams.append(stream)
    except Exception:
        for stream in reversed(streams):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()
        raise
    return streams


def release(streams) -> None:
    for stream in reversed(streams):
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--guard-pid", type=int)
    parser.add_argument("--expected-command-substring")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--min-free-gib", type=float, default=0.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.guard_pid is not None:
        if args.command:
            raise RuntimeError("guard mode cannot launch a command")
        if not args.expected_command_substring:
            raise RuntimeError("guard mode requires an exact command substring")
        observed = pid_command(args.guard_pid)
        if args.expected_command_substring not in observed:
            raise RuntimeError("guard PID is absent or its command does not match")
    else:
        if not args.command:
            raise RuntimeError("launch mode requires a command after --")
        if args.command[0] == "--":
            args.command = args.command[1:]

    streams = acquire(DEFAULT_LOCKS)
    acquired = utc_now()
    try:
        if args.guard_pid is not None:
            observed = pid_command(args.guard_pid)
            if args.expected_command_substring not in observed:
                raise RuntimeError("guard target exited while locks were acquired")
            payload = {
                "schema": "bb-host-local-lock-guard-receipt-v1",
                "status": "LOCKS_ACQUIRED_AFTER_TARGET_START",
                "guard_pid": os.getpid(),
                "target_pid": args.guard_pid,
                "target_command": observed,
                "lock_paths": [str(path) for path in DEFAULT_LOCKS],
                "locks_acquired_utc": acquired,
                "pre_acquisition_interval_is_not_claimed_locked": True,
            }
            write_receipt(args.receipt, payload)
            while pid_command(args.guard_pid):
                time.sleep(2.0)
            payload["status"] = "TARGET_EXITED_LOCKS_RELEASED"
            payload["locks_released_utc"] = utc_now()
            write_receipt(args.receipt, payload)
            return

        facts = launch_facts()
        required_free_bytes = int(args.min_free_gib * 1024**3)
        if facts["launch_disk"]["free_bytes"] < required_free_bytes:
            raise RuntimeError(
                "launch disk gate failed: "
                f"{facts['launch_disk']['free_bytes']} < {required_free_bytes} bytes"
            )
        payload = {
            "schema": "bb-host-local-lock-launch-receipt-v1",
            "status": "LOCKS_ACQUIRED_BEFORE_CHILD_START",
            "wrapper_pid": os.getpid(),
            "command": args.command,
            "lock_paths": [str(path) for path in DEFAULT_LOCKS],
            "locks_acquired_utc": acquired,
            "required_free_bytes": required_free_bytes,
            **facts,
        }
        write_receipt(args.receipt, payload)
        child = subprocess.Popen(args.command)
        payload["status"] = "CHILD_STARTED_WITH_LOCKS"
        payload["child_pid"] = child.pid
        payload["child_started_utc"] = utc_now()
        write_receipt(args.receipt, payload)
        returncode = child.wait()
        payload["status"] = "CHILD_EXITED_LOCKS_RELEASED"
        payload["returncode"] = returncode
        payload["locks_released_utc"] = utc_now()
        write_receipt(args.receipt, payload)
        raise SystemExit(returncode)
    finally:
        release(streams)


if __name__ == "__main__":
    main()
