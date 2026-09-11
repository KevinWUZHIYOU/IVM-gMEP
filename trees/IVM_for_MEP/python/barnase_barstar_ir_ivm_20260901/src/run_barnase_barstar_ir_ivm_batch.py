#!/usr/bin/env python3
"""Run registered BB IR-IVM members serially under two host-local locks."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from run_barnase_barstar_ir_ivm_member import member_label


DEFAULT_CANONICAL_LOCK = "/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock"
DEFAULT_DOWNLOADS_LOCK = "/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock"


def parse_setting(text: str) -> Tuple[float, float, int]:
    fields = text.split(",")
    if len(fields) != 3:
        raise ValueError("each setting must be alpha,r,m_max")
    return float(fields[0]), float(fields[1]), int(fields[2])


def atomic_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@contextmanager
def acquire_all_host_locks(paths):
    """Acquire both checkout locks in fixed order or release all and fail."""
    ordered = sorted({Path(path).expanduser().resolve() for path in paths}, key=str)
    handles = []
    try:
        for path in ordered:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                raise RuntimeError(f"another BB task owns {path}") from error
            handles.append(handle)
        yield [str(path) for path in ordered]
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def result_glob(root: Path, tier: str, label: str) -> List[Path]:
    return sorted((root / "runs" / tier).glob(f"ivm_{label}_*.npz"))


def validate_result(path: Path, *, expected_label: str, expected_budget: int) -> None:
    audit_path = path.with_suffix(".audit.json")
    if not audit_path.is_file():
        raise RuntimeError(f"result has no audit: {path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("consistent") is not True:
        raise RuntimeError(f"result audit is not consistent: {path}")
    attempts = int(audit["backend_raw_joint_attempts"])
    successes = int(audit["backend_raw_joint_evaluations"])
    image_misses = int(audit["sum_image_joint_evaluations"])
    if attempts != successes or attempts != image_misses:
        raise RuntimeError(f"result physical-attempt ledger is invalid: {path}")
    if int(audit["endpoint_joint_ef_cache_misses"]) != 0:
        raise RuntimeError(f"result endpoint NoF ledger is invalid: {path}")
    if int(audit["endpoint_inherited_joint_ef_cache_entries"]) != 2:
        raise RuntimeError(f"result inherited endpoint cache ledger is invalid: {path}")
    inherited = list(audit["per_image_inherited_cache_entries"])
    if inherited[0] != 1 or inherited[-1] != 1:
        raise RuntimeError(f"result endpoint cache vectors are invalid: {path}")
    if int(audit["endpoint_budget_blocked_misses"]) != 0:
        raise RuntimeError(f"result attempted an uncached endpoint: {path}")
    if attempts != int(audit["optimization_internal_nf"]):
        raise RuntimeError(f"result Number of Forces is not internal-only: {path}")
    if attempts > int(expected_budget):
        raise RuntimeError(f"result overshot its registered budget: {path}")
    with np.load(path, allow_pickle=False) as result:
        required = {
            "run_label",
            "number_of_forces",
            "optimization_internal_nf",
            "total_internal_nf_budget",
            "total_internal_nf_consumed",
            "total_internal_nf_budget_exhausted",
            "total_internal_nf_remaining",
            "total_internal_nf_budget_stop_reason",
            "total_internal_nf_budget_no_overshoot",
            "total_internal_nf_full_band_preflight_zero_step_stops",
            "total_internal_nf_complete_band_budget_stop_count",
            "total_internal_nf_terminal_partial_tail_present",
            "total_internal_nf_terminal_partial_tail_quarantined",
        }
        missing = sorted(required.difference(result.files))
        if missing:
            raise RuntimeError(
                f"result terminal-budget metadata is missing {missing}: {path}"
            )
        if str(np.asarray(result["run_label"]).item()) != expected_label:
            raise RuntimeError(f"result label does not match registration: {path}")
        numeric = {
            name: int(np.asarray(result[name]).item())
            for name in (
                "number_of_forces",
                "optimization_internal_nf",
                "total_internal_nf_budget",
                "total_internal_nf_consumed",
            )
        }
        exhausted = bool(
            np.asarray(result["total_internal_nf_budget_exhausted"]).item()
        )
        remaining = int(np.asarray(result["total_internal_nf_remaining"]).item())
        stop_reason = str(
            np.asarray(result["total_internal_nf_budget_stop_reason"]).item()
        )
        no_overshoot = bool(
            np.asarray(result["total_internal_nf_budget_no_overshoot"]).item()
        )
        preflight_stops = int(
            np.asarray(
                result["total_internal_nf_full_band_preflight_zero_step_stops"]
            ).item()
        )
        complete_band_stops = int(
            np.asarray(
                result["total_internal_nf_complete_band_budget_stop_count"]
            ).item()
        )
        partial_tail = bool(
            np.asarray(result["total_internal_nf_terminal_partial_tail_present"]).item()
        )
        quarantined = bool(
            np.asarray(
                result["total_internal_nf_terminal_partial_tail_quarantined"]
            ).item()
        )
    if numeric["total_internal_nf_budget"] != int(expected_budget):
        raise RuntimeError(f"result NPZ budget ledger is not terminal: {path}")
    for key in ("number_of_forces", "optimization_internal_nf", "total_internal_nf_consumed"):
        if numeric[key] != attempts:
            raise RuntimeError(f"result NPZ/audit Number-of-Forces mismatch: {path}")
    if not no_overshoot or remaining != int(expected_budget) - attempts:
        raise RuntimeError(f"result budget remainder ledger is invalid: {path}")
    if attempts == int(expected_budget):
        if not exhausted:
            raise RuntimeError(f"result reports an unexhausted exact budget: {path}")
    else:
        if exhausted or not (0 < remaining < 30):
            raise RuntimeError(f"result atomic budget remainder is invalid: {path}")
        if stop_reason != "total_internal_joint_budget_insufficient_for_full_band":
            raise RuntimeError(f"result did not stop at an atomic full-band boundary: {path}")
        if preflight_stops < 1 or complete_band_stops < 1:
            raise RuntimeError(f"result lacks an atomic budget-stop witness: {path}")
        if not partial_tail or not quarantined:
            raise RuntimeError(f"result terminal partial tail is not quarantined: {path}")


def required_disk_free_bytes(tier: str) -> int:
    if tier not in {"smoke", "screen", "production"}:
        raise ValueError(f"unknown BB campaign tier: {tier}")
    return int((8 if tier == "production" else 2) * 1024**3)


def run_batch(args: argparse.Namespace) -> None:
    if int(args.jobs) != 1:
        raise ValueError("BB campaign jobs is frozen to 1")
    root = Path(args.root).expanduser().resolve()
    settings = [parse_setting(item) for item in args.settings]
    status_path = root / "reports" / "campaign_status" / f"{args.name}.json"
    records: List[Dict[str, object]] = []
    started = time.time()
    with acquire_all_host_locks(
        (args.global_lock_path, args.peer_global_lock_path)
    ) as held_locks:
        disk_stats = os.statvfs(root)
        disk_free_bytes = int(disk_stats.f_bavail * disk_stats.f_frsize)
        disk_free_threshold_bytes = required_disk_free_bytes(args.tier)

        def persist_status(status: str = "running") -> None:
            atomic_json(status_path, {
                "name": args.name,
                "status": status,
                "pid": os.getpid(),
                "jobs": 1,
                "held_locks": held_locks,
                "started_unix": started,
                "updated_unix": time.time(),
                "disk_free_bytes_at_gate": disk_free_bytes,
                "disk_free_threshold_bytes": disk_free_threshold_bytes,
                "records": records,
            })

        for index, (alpha, ratio, cap) in enumerate(settings):
            label = member_label(
                alpha,
                ratio,
                cap,
                args.internal_nf_budget,
                args.attempt_tag,
            )
            existing = result_glob(root, args.tier, label)
            record: Dict[str, object] = {
                "index": index,
                "alpha": alpha,
                "r": ratio,
                "m_max": cap,
                "label": label,
                "attempt_tag": args.attempt_tag,
            }
            if len(existing) == 1:
                validate_result(
                    existing[0],
                    expected_label=label,
                    expected_budget=args.internal_nf_budget,
                )
                record.update(status="already_complete", result=str(existing[0]))
                records.append(record)
                persist_status()
                continue
            if len(existing) > 1:
                raise RuntimeError(f"multiple results match {label}")
            disk_stats = os.statvfs(root)
            disk_free_bytes = int(disk_stats.f_bavail * disk_stats.f_frsize)
            if disk_free_bytes < disk_free_threshold_bytes:
                records.append({
                    "index": index,
                    "alpha": alpha,
                    "r": ratio,
                    "m_max": cap,
                    "label": label,
                    "status": "blocked_insufficient_disk",
                })
                persist_status("blocked_insufficient_disk")
                raise RuntimeError(
                    "insufficient disk for BB member: "
                    f"{disk_free_bytes} < {disk_free_threshold_bytes} bytes"
                )
            command = [
                sys.executable,
                str(root / "src" / "run_barnase_barstar_ir_ivm_member.py"),
                "--root", str(root), "--alpha", f"{alpha:g}",
                "--residual-ratio", f"{ratio:g}",
                "--inner-max-steps", str(cap),
                "--internal-nf-budget", str(args.internal_nf_budget),
                "--tier", args.tier,
                "--attempt-tag", args.attempt_tag,
            ]
            record.update(status="running", command=command, started_unix=time.time())
            records.append(record)
            persist_status()
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(root / "src")
            completed = subprocess.run(command, cwd=str(root), env=environment, check=False)
            record.update(returncode=int(completed.returncode), finished_unix=time.time())
            if completed.returncode != 0:
                record["status"] = "failed"
                persist_status("failed")
                raise RuntimeError(f"member failed: {label}")
            produced = result_glob(root, args.tier, label)
            if len(produced) != 1:
                raise RuntimeError(f"member did not produce exactly one result: {label}")
            validate_result(
                produced[0],
                expected_label=label,
                expected_budget=args.internal_nf_budget,
            )
            record.update(status="complete", result=str(produced[0]))
            persist_status("complete" if index == len(settings) - 1 else "running")
        persist_status("complete")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--name", required=True)
    parser.add_argument("--tier", choices=("smoke", "screen", "production"), required=True)
    parser.add_argument("--internal-nf-budget", type=int, required=True)
    parser.add_argument("--settings", nargs="+", required=True)
    parser.add_argument("--jobs", type=int, choices=(1,), default=1)
    parser.add_argument("--attempt-tag", default="r1")
    parser.add_argument("--global-lock-path", default=DEFAULT_CANONICAL_LOCK)
    parser.add_argument("--peer-global-lock-path", default=DEFAULT_DOWNLOADS_LOCK)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_batch(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
