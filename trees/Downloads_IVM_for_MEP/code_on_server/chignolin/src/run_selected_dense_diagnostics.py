#!/usr/bin/env python3
"""Run uniform-checkpoint adaptive dense barriers for selected methods."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def choose_calls(result: Path, maximum_call: int, maximum_points: int) -> list[int]:
    with np.load(result, allow_pickle=False) as saved:
        calls = np.asarray(saved["stored_path_calls"], dtype=int)
    calls = np.unique(calls[calls <= int(maximum_call)])
    if calls.size <= maximum_points:
        return calls.tolist()
    # Combine logarithmic early coverage with uniform whole-curve coverage.
    linear = np.linspace(0, calls.size - 1, maximum_points // 2)
    geometric = np.geomspace(1, calls.size, maximum_points // 2) - 1
    indices = np.unique(
        np.clip(
            np.rint(np.concatenate((linear, geometric))).astype(int),
            0,
            calls.size - 1,
        )
    )
    # Fill any vacancies with uniformly spaced unused indices.
    if indices.size < maximum_points:
        for index in np.rint(
            np.linspace(0, calls.size - 1, maximum_points * 3)
        ).astype(int):
            indices = np.unique(np.append(indices, index))
            if indices.size >= maximum_points:
                break
    return calls[np.sort(indices[:maximum_points])].tolist()


def run_one(
    python: Path,
    root: Path,
    name: str,
    result: Path,
    maximum_call: int,
    maximum_points: int,
    diagnostic_workers: int,
) -> dict:
    label = f"paper_total_nf_{name}_adaptive48"
    output = root / "runs" / "diagnostics" / f"dense_barrier_{label}.npz"
    audit = output.with_suffix(".audit.json")
    calls = choose_calls(result, maximum_call, maximum_points)
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
        str(diagnostic_workers),
        "--calls",
        ",".join(str(call) for call in calls),
    ]
    log_dir = root / "logs" / "paper_total_nf_dense_20260727a"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
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
        "name": name,
        "status": status,
        "returncode": returncode,
        "source_result": str(result.resolve()),
        "selected_calls_per_image": calls,
        "n_selected_paths": len(calls),
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
    parser.add_argument("--diagnostic-workers", type=int, default=2)
    parser.add_argument("--maximum-points", type=int, default=48)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    production_manifest = json.loads(
        (
            root
            / "reports"
            / "paper_method_productions_20260727a"
            / "manifest.json"
        ).read_text()
    )
    production = {
        row["name"]: Path(row["result_file"])
        for row in production_manifest["records"]
    }
    selected = {
        "ivm_alpha1": (
            root
            / "runs"
            / "ivm"
            / "ivm_ivm_equal_idpp_v52_fixed_alpha1_production_central_state_ivm_cfg31b81924ce51_budget500.npz",
            500,
        ),
        "ode_neb_k50": (
            root
            / "runs"
            / "neb"
            / "neb_equal_idpp_k50_budget5000_production_central_k50_alpha0p0001_ode_budget5000.npz",
            500,
        ),
        "static_neb": (production["static_neb_balanced"], 500),
        "original_string": (
            production["original_string_balanced"],
            500,
        ),
        "obsm_fire": (production["obsm_low_iperp"], 500),
    }
    jobs = [
        (
            args.python.expanduser().resolve(),
            root,
            name,
            result,
            maximum_call,
            int(args.maximum_points),
            int(args.diagnostic_workers),
        )
        for name, (result, maximum_call) in selected.items()
    ]
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
                f"[{completed}/{len(futures)}] {record['name']} "
                f"{record['status']}",
                flush=True,
            )
    if any(record["status"] == "failed" for record in records):
        raise RuntimeError(
            "Dense diagnostics failed: "
            + ", ".join(
                record["name"]
                for record in records
                if record["status"] == "failed"
            )
        )
    for record in records:
        audit = json.loads(Path(record["audit_file"]).read_text())
        if (
            audit["diagnostic_backend_raw_joint_evaluations"]
            != audit["diagnostic_evaluator_cache_misses"]
            or not audit["counts_excluded_from_optimization_axis"]
        ):
            raise RuntimeError(
                f"Dense diagnostic audit failed for {record['name']}"
            )
        record["diagnostic_joint_evaluations"] = audit[
            "diagnostic_backend_raw_joint_evaluations"
        ]
        record["selected_subdivision_counts"] = audit[
            "selected_subdivision_counts"
        ]
    report_dir = root / "reports" / "paper_total_nf_dense_20260727a"
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "manifest.json"
    output.write_text(
        json.dumps(
            {
                "checkpoint_selection": (
                    "union of logarithmic early and uniform whole-curve "
                    "indices, at most 48 paths per method"
                ),
                "optimization_cost_axis": "counter_vectors.sum(axis=1)",
                "diagnostic_calls_excluded_from_optimization_axis": True,
                "records": sorted(records, key=lambda row: row["name"]),
            },
            indent=2,
        )
        + "\n"
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
