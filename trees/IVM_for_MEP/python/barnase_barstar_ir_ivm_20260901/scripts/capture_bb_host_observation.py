#!/usr/bin/env python3
"""Seal a zero-OpenMM host/process/lock observation for one running BB member."""

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


LOCKS = (
    Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock"),
    Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock"),
)


def run_text(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")
    return completed.stdout.strip()


def pid_command(pid: int) -> str:
    return run_text(["/bin/ps", "-p", str(pid), "-o", "command="])


def lock_holders(path: Path) -> list[int]:
    completed = subprocess.run(
        ["/usr/sbin/lsof", "-t", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(f"lsof failed for {path}: {completed.stderr.strip()}")
    return sorted({int(line) for line in completed.stdout.splitlines() if line.strip()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--wrapper-pid", type=int, required=True)
    parser.add_argument("--child-pid", type=int, required=True)
    parser.add_argument("--expected-child-substring", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    wrapper_command = pid_command(args.wrapper_pid)
    child_command = pid_command(args.child_pid)
    if args.expected_child_substring not in child_command:
        raise RuntimeError("child command does not match the registered parameter tuple")
    holders = {str(path): lock_holders(path) for path in LOCKS}
    if any(args.wrapper_pid not in pids for pids in holders.values()):
        raise RuntimeError("wrapper PID does not hold both host-local locks")

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
    payload = {
        "schema": "bb-running-host-observation-v1",
        "status": "RUNNING_CHILD_AND_BOTH_LOCKS_OBSERVED",
        "observed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": args.label,
        "host": {
            "hostname": socket.gethostname(),
            "architecture": platform.machine(),
            "processor": processor,
        },
        "working_directory": str(Path.cwd().resolve()),
        "wrapper_pid": args.wrapper_pid,
        "wrapper_command": wrapper_command,
        "child_pid": args.child_pid,
        "child_command": child_command,
        "expected_child_substring": args.expected_child_substring,
        "lock_holders": holders,
        "launch_disk": {
            "path": str(disk_path),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "openmm_calls": 0,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        output.unlink(missing_ok=True)
        raise
    print(output)


if __name__ == "__main__":
    main()
