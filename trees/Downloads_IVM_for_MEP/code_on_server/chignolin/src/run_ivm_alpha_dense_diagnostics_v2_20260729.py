#!/usr/bin/env python3
"""Densified barriers at fixed total-force budgets for the IVM alpha sweep."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


FORCE_BUDGETS = (1002, 3002, 6002, 10002)
TRAILING_FRACTION = 0.30
TRAILING_SAMPLES = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alpha_tag(alpha: float) -> str:
    return f"{float(alpha):g}".replace(".", "p")


def checkpoint_mapping(
    result: Path,
    force_budgets: tuple[int, ...],
    trailing_fraction: float,
    trailing_samples: int,
) -> list[dict]:
    with np.load(result, allow_pickle=False) as saved:
        calls = np.asarray(saved["calls_per_image"], dtype=int)
        total_force = np.asarray(saved["counter_vectors"], dtype=int).sum(
            axis=1
        )
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=int)
    stored = set(int(value) for value in stored_calls)
    if not 0.0 < float(trailing_fraction) <= 1.0:
        raise ValueError("trailing_fraction must lie in (0, 1]")
    if int(trailing_samples) < 2:
        raise ValueError("trailing_samples must be at least two")
    rows = []
    for budget in force_budgets:
        endpoint_eligible = np.flatnonzero(total_force <= int(budget))
        endpoint_index = (
            int(endpoint_eligible[-1]) if endpoint_eligible.size else 0
        )
        endpoint_force = int(total_force[endpoint_index])
        window_start = max(
            int(total_force[0]),
            int(np.ceil((1.0 - trailing_fraction) * endpoint_force)),
        )
        targets = np.linspace(
            window_start,
            endpoint_force,
            int(trailing_samples),
        )
        for sample_ordinal, target in enumerate(targets):
            eligible = np.flatnonzero(total_force <= float(target))
            index = int(eligible[-1]) if eligible.size else 0
            call = int(calls[index])
            if call not in stored:
                raise RuntimeError(
                    f"Selected call {call} is not stored in {result}"
                )
            rows.append(
                {
                    "requested_force_budget": int(budget),
                    "trailing_fraction": float(trailing_fraction),
                    "sample_ordinal": int(sample_ordinal),
                    "sample_count": int(trailing_samples),
                    "requested_sample_force": float(target),
                    "selected_call_per_image": call,
                    "actual_number_of_force_evaluations": int(
                        total_force[index]
                    ),
                    "role": (
                        "budget_endpoint"
                        if sample_ordinal == int(trailing_samples) - 1
                        else "trailing_window_sample"
                    ),
                }
            )
    return rows


def run_one(
    python: Path,
    root: Path,
    alpha: float,
    result: Path,
    checkpoints: list[dict],
    workers: int,
    namespace: str,
    output_label_prefix: str,
) -> dict:
    label = f"{output_label_prefix}_a{alpha_tag(alpha)}"
    output = (
        root / "runs" / "diagnostics" / f"dense_barrier_{label}.npz"
    )
    audit = output.with_suffix(".audit.json")
    calls = sorted(
        {
            int(row["selected_call_per_image"])
            for row in checkpoints
        }
    )
    command = [
        str(python),
        str(root / "src" / "postprocess_dense_barrier.py"),
        "--root",
        str(root),
        "--result",
        str(result),
        "--output-label",
        label,
        "--workers",
        str(workers),
        "--calls",
        ",".join(str(call) for call in calls),
    ]
    log_dir = root / "logs" / namespace
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"alpha_{alpha_tag(alpha)}.log"
    if output.exists() and audit.exists():
        status = "reused"
        returncode = 0
    else:
        environment = os.environ.copy()
        for variable in (
            "OPENMM_CPU_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
        ):
            environment[variable] = "1"
        environment["PYTHONPATH"] = str(root / "src")
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        status = "completed" if completed.returncode == 0 else "failed"
        returncode = completed.returncode
    return {
        "alpha": float(alpha),
        "status": status,
        "returncode": int(returncode),
        "source_result": str(result.resolve()),
        "source_result_sha256": sha256(result),
        "checkpoints": checkpoints,
        "result_file": str(output.resolve()) if output.exists() else None,
        "audit_file": str(audit.resolve()) if audit.exists() else None,
        "log_file": str(log_path.resolve()),
        "command": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--outer-workers", type=int, default=3)
    parser.add_argument("--diagnostic-workers", type=int, default=1)
    parser.add_argument(
        "--force-budgets",
        default=",".join(str(value) for value in FORCE_BUDGETS),
    )
    parser.add_argument(
        "--trailing-fraction",
        type=float,
        default=TRAILING_FRACTION,
    )
    parser.add_argument(
        "--trailing-samples",
        type=int,
        default=TRAILING_SAMPLES,
    )
    parser.add_argument(
        "--report-namespace",
        default="ivm_fixed_alpha_sensitivity_total_nf_20260727a",
    )
    parser.add_argument(
        "--output-label-prefix",
        default="ivm_fixed_alpha_total_nf",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    report_dir = root / "reports" / args.report_namespace
    alpha_manifest = json.loads(
        (report_dir / "manifest.json").read_text()
    )
    force_budgets = tuple(
        int(value)
        for value in args.force_budgets.split(",")
        if value.strip()
    )
    if not force_budgets:
        raise ValueError("At least one force budget is required")
    jobs = []
    for record in alpha_manifest["records"]:
        if not record.get("valid", False):
            raise RuntimeError(
                f"Invalid fixed-alpha production: alpha={record['alpha']}"
            )
        result = Path(record["result_file"]).resolve()
        if sha256(result) != record["result_sha256"]:
            raise RuntimeError(f"Alpha result changed: {result}")
        checkpoints = checkpoint_mapping(
            result,
            force_budgets,
            args.trailing_fraction,
            args.trailing_samples,
        )
        jobs.append(
            (
                args.python.expanduser().resolve(),
                root,
                float(record["alpha"]),
                result,
                checkpoints,
                int(args.diagnostic_workers),
                args.report_namespace,
                args.output_label_prefix,
            )
        )

    records = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(args.outer_workers))
    ) as executor:
        futures = {
            executor.submit(run_one, *job): job[2] for job in jobs
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            records.append(record)
            print(
                f"[{completed}/{len(futures)}] alpha={record['alpha']:g} "
                f"{record['status']}",
                flush=True,
            )
    failed = [
        record["alpha"]
        for record in records
        if record["status"] == "failed"
    ]
    if failed:
        raise RuntimeError(f"Dense alpha diagnostics failed: {failed}")

    diagnostic_total = 0
    for record in records:
        result = Path(record["result_file"])
        audit_path = Path(record["audit_file"])
        audit = json.loads(audit_path.read_text())
        with np.load(result, allow_pickle=False) as saved:
            source_result = Path(str(saved["source_result"])).resolve()
            backend = int(
                saved["diagnostic_backend_raw_joint_evaluations"]
            )
            misses = int(saved["diagnostic_evaluator_cache_misses"])
            available_calls = set(
                int(value) for value in saved["calls"]
            )
        expected_calls = {
            int(row["selected_call_per_image"])
            for row in record["checkpoints"]
        }
        if (
            source_result != Path(record["source_result"]).resolve()
            or backend != misses
            or backend
            != audit["diagnostic_backend_raw_joint_evaluations"]
            or not audit["counts_excluded_from_optimization_axis"]
            or available_calls != expected_calls
        ):
            raise RuntimeError(
                f"Dense diagnostic audit failed for alpha={record['alpha']}"
            )
        record["result_sha256"] = sha256(result)
        record["audit_sha256"] = sha256(audit_path)
        record["diagnostic_joint_evaluations"] = backend
        record["selected_subdivision_counts"] = audit[
            "selected_subdivision_counts"
        ]
        diagnostic_total += backend

    output = report_dir / "dense_manifest.json"
    output.write_text(
        json.dumps(
            {
                "requested_total_force_budgets": list(force_budgets),
                "trailing_window_fraction": float(
                    args.trailing_fraction
                ),
                "densified_samples_per_trailing_window": int(
                    args.trailing_samples
                ),
                "barrier_protocol": (
                    "Real OpenMM energies on linearly densified adjacent "
                    "image segments: 10 and 20 subdivisions; double to "
                    "40/80 whenever successive barriers differ by more "
                    "than 0.05 kcal/mol."
                ),
                "optimization_cost_axis": "counter_vectors.sum(axis=1)",
                "cost_includes_all_22_images": True,
                "diagnostic_calls_excluded_from_optimization_axis": True,
                "total_diagnostic_joint_evaluations": diagnostic_total,
                "records": sorted(
                    records, key=lambda row: row["alpha"]
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
