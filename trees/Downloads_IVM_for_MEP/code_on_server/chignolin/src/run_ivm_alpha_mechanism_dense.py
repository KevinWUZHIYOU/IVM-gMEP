"""Adaptive dense barriers for the alpha mechanism ablation endpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def run_one(
    python: Path,
    root: Path,
    record: dict,
    call: int,
    namespace: str,
) -> dict:
    label = (
        f"alpha_mechanism_{record['variant']}_a{tag(record['alpha'])}"
        f"_call{call}"
    )
    command = [
        str(python),
        str(root / "src" / "postprocess_dense_barrier.py"),
        "--root",
        str(root),
        "--result",
        record["result_file"],
        "--output-label",
        label,
        "--workers",
        "1",
        "--calls",
        str(call),
    ]
    output = root / "runs" / "diagnostics" / f"dense_barrier_{label}.npz"
    audit_file = output.with_suffix(".audit.json")
    log_dir = root / "logs" / namespace
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (
        log_dir / f"{record['variant']}_a{tag(record['alpha'])}.log"
    )
    if output.exists() and audit_file.exists():
        status = "reused"
        returncode = 0
    else:
        environment = os.environ.copy()
        environment.update(
            {
                "OPENMM_CPU_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        with log_file.open("w") as log:
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        returncode = int(process.returncode)
        status = (
            "completed"
            if returncode == 0 and output.exists() and audit_file.exists()
            else "failed"
        )
    row = {
        "variant": record["variant"],
        "alpha": record["alpha"],
        "call": call,
        "status": status,
        "returncode": returncode,
        "source_result": record["result_file"],
        "result_file": str(output),
        "audit_file": str(audit_file),
        "log_file": str(log_file),
        "command": command,
    }
    if status == "failed":
        return row
    audit = json.loads(audit_file.read_text())
    with np.load(output, allow_pickle=False) as saved:
        selected_calls = np.asarray(saved["calls"], dtype=int)
        barriers = np.asarray(saved["adaptive_barrier_kcal"], dtype=float)
        subdivisions = np.asarray(
            saved["selected_subdivisions"], dtype=int
        )
        if selected_calls.tolist() != [call]:
            raise RuntimeError(
                f"Dense checkpoint mismatch for {record['variant']} "
                f"alpha={record['alpha']}: {selected_calls.tolist()}"
            )
        row.update(
            {
                "dense_barrier_kcal": float(barriers[0]),
                "selected_subdivisions": int(subdivisions[0]),
                "diagnostic_joint_evaluations": int(
                    saved["diagnostic_backend_raw_joint_evaluations"]
                ),
            }
        )
    row["diagnostic_counter_consistent"] = bool(
        audit["diagnostic_backend_raw_joint_evaluations"]
        == audit["diagnostic_evaluator_cache_misses"]
        == row["diagnostic_joint_evaluations"]
    )
    row["diagnostic_calls_excluded"] = bool(
        audit["counts_excluded_from_optimization_axis"]
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--call", type=int, default=150)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--namespace",
        default="ivm_alpha_mechanism_dense_20260728a",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    manifest = json.loads(args.manifest.expanduser().resolve().read_text())
    records = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = [
            executor.submit(
                run_one,
                python,
                root,
                record,
                args.call,
                args.namespace,
            )
            for record in manifest["records"]
        ]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            row = future.result()
            records.append(row)
            print(
                f"[{completed}/{len(futures)}] {row['variant']} "
                f"alpha={row['alpha']:g} {row['status']}",
                flush=True,
            )
    records.sort(key=lambda row: (row["variant"], row["alpha"]))
    if any(
        row["status"] not in {"completed", "reused"}
        or not row.get("diagnostic_counter_consistent", False)
        or not row.get("diagnostic_calls_excluded", False)
        for row in records
    ):
        raise RuntimeError("Dense alpha mechanism audit failed")
    report_dir = root / "reports" / args.namespace
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "manifest.json"
    output.write_text(
        json.dumps(
            {
                "source_manifest": str(
                    args.manifest.expanduser().resolve()
                ),
                "checkpoint_call_per_internal_image": args.call,
                "diagnostic_calls_excluded_from_optimization_axis": True,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
