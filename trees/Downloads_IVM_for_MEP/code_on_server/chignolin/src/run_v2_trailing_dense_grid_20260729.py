#!/usr/bin/env python3
"""Run and audit the preregistered dense-barrier diagnostic for a V2 grid."""

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


REFERENCE_BARRIER_KCAL = 20.373739515742795
EXPECTED_BUDGETS = (1000, 3000, 6000, 10000)
EXPECTED_SAMPLES = 7
EXPECTED_TRAILING_FRACTION = 0.30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alpha_tag(alpha: float) -> str:
    return f"{float(alpha):g}".replace(".", "p")


def run_one(
    python: Path,
    root: Path,
    report_dir: Path,
    record: dict,
) -> dict:
    alpha = float(record["alpha"])
    source = Path(record["result_file"]).resolve()
    if sha256(source) != record["result_sha256"]:
        raise RuntimeError(f"Optimization result changed: {source}")
    label = f"v2_preregistered_trailing_a{alpha_tag(alpha)}"
    output = (
        root / "runs" / "diagnostics" / f"dense_barrier_{label}.npz"
    )
    audit = output.with_suffix(".audit.json")
    log_dir = report_dir / "dense_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"alpha_{alpha_tag(alpha)}.log"
    command = [
        str(python),
        str(
            root
            / "src"
            / "postprocess_trailing_dense_barrier_v2_20260729.py"
        ),
        "--root",
        str(root),
        "--result",
        str(source),
        "--output-label",
        label,
        "--budgets",
        ",".join(str(value) for value in EXPECTED_BUDGETS),
        "--trailing-fraction",
        str(EXPECTED_TRAILING_FRACTION),
        "--samples",
        str(EXPECTED_SAMPLES),
        "--workers",
        "1",
        "--reference-barrier",
        repr(REFERENCE_BARRIER_KCAL),
    ]
    status = "reused"
    if not (output.exists() and audit.exists()):
        environment = os.environ.copy()
        for variable in (
            "OPENMM_CPU_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
        ):
            environment[variable] = "1"
        environment["PYTHONPATH"] = str(root / "src")
        with log.open("w") as stream:
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"Dense diagnostic failed for alpha={alpha:g}; see {log}"
            )
        status = "completed"

    report = json.loads(audit.read_text())
    with np.load(output, allow_pickle=False) as saved:
        source_in_dense = Path(str(saved["source_result"])).resolve()
        source_hash_in_dense = str(saved["source_result_sha256"])
        budgets = tuple(
            int(value)
            for value in saved["requested_total_force_budgets"]
        )
        fraction = float(saved["trailing_window_fraction"])
        samples = int(saved["samples_per_window"])
        barriers = np.asarray(
            saved["adaptive_barrier_kcal"], dtype=float
        )
        errors = np.asarray(saved["barrier_error_kcal"], dtype=float)
        subdivisions = np.asarray(
            saved["selected_subdivisions"], dtype=np.int64
        )
        actual_force = np.asarray(
            saved["actual_total_force"], dtype=np.int64
        )
        backend = int(saved["diagnostic_backend_raw_joint_evaluations"])
        misses = int(saved["diagnostic_evaluator_cache_misses"])
        reference = float(saved["reference_barrier_kcal"])
    expected_shape = (len(EXPECTED_BUDGETS), EXPECTED_SAMPLES)
    if (
        source_in_dense != source
        or source_hash_in_dense != record["result_sha256"]
        or budgets != EXPECTED_BUDGETS
        or not np.isclose(fraction, EXPECTED_TRAILING_FRACTION)
        or samples != EXPECTED_SAMPLES
        or barriers.shape != expected_shape
        or errors.shape != expected_shape
        or subdivisions.shape != expected_shape
        or actual_force.shape != expected_shape
        or backend != misses
        or backend != int(report["diagnostic_backend_raw_joint_evaluations"])
        or not report["counts_excluded_from_optimization_axis"]
        or report["barrier_protocol"]["energy_backend"]
        != "real OpenMM CPU double precision"
        or report["barrier_protocol"]["interpolation"]
        != "linear Cartesian interpolation per segment"
        or report["barrier_protocol"]["initial_subdivisions"] != 10
        or report["barrier_protocol"]["comparison_subdivisions"] != 20
        or report["barrier_protocol"]["adaptive_threshold_kcal"] != 0.05
        or report["barrier_protocol"]["maximum_subdivisions"] != 80
        or not np.isclose(reference, REFERENCE_BARRIER_KCAL)
        or not np.allclose(
            errors,
            np.abs(barriers - REFERENCE_BARRIER_KCAL),
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise RuntimeError(f"Dense audit failed for alpha={alpha:g}")
    rms = np.sqrt(np.mean(errors * errors, axis=1))
    return {
        "alpha": alpha,
        "status": status,
        "source_result": str(source),
        "source_result_sha256": record["result_sha256"],
        "dense_result": str(output.resolve()),
        "dense_result_sha256": sha256(output),
        "dense_audit": str(audit.resolve()),
        "dense_audit_sha256": sha256(audit),
        "log": str(log.resolve()),
        "requested_total_force_budgets": list(EXPECTED_BUDGETS),
        "samples_per_window": EXPECTED_SAMPLES,
        "trailing_fraction": EXPECTED_TRAILING_FRACTION,
        "rms_barrier_error_kcal_by_budget": rms.tolist(),
        "mean_dense_barrier_kcal_by_budget": np.mean(
            barriers, axis=1
        ).tolist(),
        "temporal_std_dense_barrier_kcal_by_budget": np.std(
            barriers, axis=1
        ).tolist(),
        "diagnostic_joint_evaluations": backend,
        "selected_subdivision_counts": {
            str(level): int(np.count_nonzero(subdivisions == level))
            for level in (20, 40, 80)
        },
        "command": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--outer-workers", type=int, default=4)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    records = manifest["records"]
    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(args.outer_workers))
    ) as executor:
        futures = [
            executor.submit(
                run_one,
                args.python.expanduser().resolve(),
                root,
                report_dir,
                record,
            )
            for record in records
        ]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            row = future.result()
            results.append(row)
            print(
                f"[{completed}/{len(futures)}] alpha={row['alpha']:g} "
                f"{row['status']}",
                flush=True,
            )

    results.sort(key=lambda row: row["alpha"])
    output = {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "protocol": {
            "energy_backend": "real OpenMM CPU double precision",
            "interpolation": "linear Cartesian interpolation per segment",
            "subdivisions": "10, 20, then 40/80 if successive difference > 0.05 kcal/mol",
            "reference_barrier_kcal": REFERENCE_BARRIER_KCAL,
            "requested_total_force_budgets": list(EXPECTED_BUDGETS),
            "trailing_fraction": EXPECTED_TRAILING_FRACTION,
            "samples_per_window": EXPECTED_SAMPLES,
            "diagnostic_calls_excluded_from_optimization_axis": True,
            "sparse_22_image_barrier_eligible": False,
        },
        "total_diagnostic_joint_evaluations": int(
            sum(row["diagnostic_joint_evaluations"] for row in results)
        ),
        "records": results,
    }
    output_path = report_dir / "dense_manifest.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    os.replace(temporary, output_path)
    print(output_path, flush=True)


if __name__ == "__main__":
    main()
