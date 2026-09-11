#!/usr/bin/env python3
"""Refresh the in-progress BB cost-to-precision PDF after each closed path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def closed_row_count(wal: Path) -> int:
    rows: dict[int, set[int]] = {}
    lines = wal.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        record = json.loads(line)
        if record.get("status") != "success":
            continue
        rows.setdefault(int(record["row_index"]), set()).add(int(record["point_index"]))
    return sum(points == set(range(500)) for points in rows.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ivm-contract", type=Path, required=True)
    parser.add_argument("--ivm-wal", type=Path, required=True)
    parser.add_argument("--ivm-final-result", type=Path, required=True)
    parser.add_argument("--ode-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=20.0)
    args = parser.parse_args()
    script = Path(__file__).with_name("plot_bb_cost_to_precision_preview.py")
    current = closed_row_count(args.ivm_wal)
    print(json.dumps({"event": "watcher_started", "closed_paths": current}), flush=True)
    while True:
        time.sleep(max(5.0, float(args.interval_seconds)))
        observed = closed_row_count(args.ivm_wal)
        if observed > current:
            environment = os.environ.copy()
            environment.setdefault("MPLCONFIGDIR", "/private/tmp/bb_ivm_mplconfig")
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--ivm-contract",
                    str(args.ivm_contract),
                    "--ivm-wal",
                    str(args.ivm_wal),
                    "--ode-result",
                    str(args.ode_result),
                    "--output",
                    str(args.output),
                ],
                check=True,
                env=environment,
            )
            current = observed
            print(json.dumps({"event": "preview_refreshed", "closed_paths": current}), flush=True)
        if args.ivm_final_result.exists() and current >= 11:
            print(json.dumps({"event": "watcher_complete", "closed_paths": current}), flush=True)
            return


if __name__ == "__main__":
    main()
