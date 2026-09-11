#!/usr/bin/env python3
"""Real-OpenMM dense barriers for complete V9 alpha-grid macrostates."""

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


FORCE_BUDGETS = (1000, 3000, 6000, 10000)
TRAILING_FRACTION = 0.30
TRAILING_SAMPLES = 7
REFERENCE_BARRIER_KCAL = 20.373739515742795


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alpha_token(alpha: float) -> str:
    return f"{float(alpha):g}".replace(".", "p")


def complete_macrostate_mask(saved) -> np.ndarray:
    free_polish = np.asarray(saved["state_free_polish_active"], dtype=bool)
    spring_committed = np.asarray(saved["reparam_committed"], dtype=bool)
    return (~free_polish) | spring_committed


def checkpoint_mapping(result: Path) -> list[dict]:
    with np.load(result, allow_pickle=False) as saved:
        calls = np.asarray(saved["calls_per_image"], dtype=int)
        total_force = np.asarray(
            saved["counter_vectors"], dtype=np.int64
        ).sum(axis=1)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=int)
        complete = complete_macrostate_mask(saved)
    if not complete[0]:
        raise RuntimeError(f"Initial state is incomplete: {result}")
    stored = set(int(value) for value in stored_calls)
    complete_indices = np.flatnonzero(complete)
    rows = []
    for budget in FORCE_BUDGETS:
        endpoint_candidates = complete_indices[
            total_force[complete_indices] <= int(budget)
        ]
        if not endpoint_candidates.size:
            raise RuntimeError(
                f"No complete macrostate at N_F <= {budget}: {result}"
            )
        endpoint_index = int(endpoint_candidates[-1])
        endpoint_force = int(total_force[endpoint_index])
        eligible_complete = complete_indices[
            complete_indices <= endpoint_index
        ]
        if eligible_complete.size < 2:
            raise RuntimeError(
                f"No produced complete macrostate through N_F={budget}"
            )
        starts = {
            "trailing": max(
                float(total_force[eligible_complete[0]]),
                endpoint_force - TRAILING_FRACTION * float(budget),
            ),
            "anytime": float(total_force[eligible_complete[1]]),
        }
        for window_kind, window_start in starts.items():
            targets = np.linspace(
                window_start, endpoint_force, TRAILING_SAMPLES
            )
            for ordinal, target in enumerate(targets):
                candidates = eligible_complete[
                    total_force[eligible_complete] <= float(target)
                ]
                index = (
                    int(candidates[-1])
                    if candidates.size
                    else int(eligible_complete[0])
                )
                call = int(calls[index])
                if call not in stored:
                    raise RuntimeError(
                        f"Complete call {call} is not stored in {result}"
                    )
                rows.append(
                    {
                        "requested_total_force_budget": int(budget),
                        "window_kind": window_kind,
                        "sample_ordinal": int(ordinal),
                        "sample_count": TRAILING_SAMPLES,
                        "requested_sample_total_force": float(target),
                        "source_record_index": index,
                        "selected_call_per_image": call,
                        "actual_total_force": int(total_force[index]),
                        "complete_operator_split_macrostate": True,
                        "role": (
                            "budget_endpoint"
                            if ordinal == TRAILING_SAMPLES - 1
                            else f"{window_kind}_window_sample"
                        ),
                    }
                )
    return rows


def run_one(
    python: Path,
    root: Path,
    alpha: float,
    result: Path,
    output_label_prefix: str,
    namespace: str,
    diagnostic_workers: int,
) -> dict:
    checkpoints = checkpoint_mapping(result)
    label = f"{output_label_prefix}_a{alpha_token(alpha)}"
    dense_output = (
        root / "runs" / "diagnostics" / f"dense_barrier_{label}.npz"
    )
    dense_audit = dense_output.with_suffix(".audit.json")
    calls = sorted(
        {int(row["selected_call_per_image"]) for row in checkpoints}
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
        str(diagnostic_workers),
        "--calls",
        ",".join(str(value) for value in calls),
    ]
    log_dir = root / "logs" / namespace
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"alpha_{alpha_token(alpha)}.log"
    if dense_output.exists() and dense_audit.exists():
        status = "reused"
        returncode = 0
    else:
        environment = os.environ.copy()
        environment["OPENMM_CPU_THREADS"] = "1"
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        environment["OMP_NUM_THREADS"] = "1"
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
        returncode = int(completed.returncode)
        status = "completed" if returncode == 0 else "failed"
    return {
        "alpha": float(alpha),
        "status": status,
        "returncode": returncode,
        "source_result": str(result),
        "source_result_sha256": sha256(result),
        "checkpoints": checkpoints,
        "dense_result": str(dense_output),
        "dense_audit": str(dense_audit),
        "log": str(log_path),
        "command": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--grid-summary", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-label-prefix", required=True)
    parser.add_argument("--outer-workers", type=int, default=3)
    parser.add_argument("--diagnostic-workers", type=int, default=1)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    summary_path = args.grid_summary.expanduser().resolve()
    summary = json.loads(summary_path.read_text())
    jobs = []
    for record in summary["records"]:
        result = Path(record["result"]).resolve()
        if sha256(result) != record["result_sha256"]:
            raise RuntimeError(f"Optimization result changed: {result}")
        jobs.append(
            (
                args.python.expanduser().resolve(),
                root,
                float(record["alpha"]),
                result,
                args.output_label_prefix,
                args.output_dir.name,
                int(args.diagnostic_workers),
            )
        )

    records = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(args.outer_workers))
    ) as executor:
        futures = [executor.submit(run_one, *job) for job in jobs]
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
        raise RuntimeError(f"Dense diagnostics failed: {failed}")

    diagnostic_total = 0
    for record in records:
        dense_result = Path(record["dense_result"])
        dense_audit = Path(record["dense_audit"])
        audit = json.loads(dense_audit.read_text())
        with np.load(dense_result, allow_pickle=False) as saved:
            if (
                Path(str(saved["source_result"])).resolve()
                != Path(record["source_result"]).resolve()
            ):
                raise RuntimeError("Dense source/result mismatch")
            available_calls = np.asarray(saved["calls"], dtype=int)
            barriers = np.asarray(
                saved["adaptive_barrier_kcal"], dtype=float
            )
            subdivisions = np.asarray(
                saved["selected_subdivisions"], dtype=int
            )
            final_differences = np.asarray(
                saved["final_level_difference_kcal"], dtype=float
            )
            backend = int(
                saved["diagnostic_backend_raw_joint_evaluations"]
            )
            misses = int(saved["diagnostic_evaluator_cache_misses"])
        if (
            backend != misses
            or backend
            != int(audit["diagnostic_backend_raw_joint_evaluations"])
            or not audit["counts_excluded_from_optimization_axis"]
            or np.any(
                (final_differences > 0.05 + 1.0e-12)
                & (subdivisions < 80)
            )
        ):
            raise RuntimeError(
                f"Dense audit/convergence failed for alpha={record['alpha']}"
            )
        barrier_by_call = {
            int(call): float(value)
            for call, value in zip(available_calls, barriers)
        }
        subdivision_by_call = {
            int(call): int(value)
            for call, value in zip(available_calls, subdivisions)
        }
        matrices = {}
        for window_kind in ("trailing", "anytime"):
            barrier_matrix = np.empty(
                (len(FORCE_BUDGETS), TRAILING_SAMPLES), dtype=float
            )
            matrices[window_kind] = {
                "barrier": barrier_matrix,
                "error": np.empty_like(barrier_matrix),
                "force": np.empty_like(barrier_matrix, dtype=np.int64),
                "subdivision": np.empty_like(
                    barrier_matrix, dtype=np.int64
                ),
            }
        for row in record["checkpoints"]:
            budget_index = FORCE_BUDGETS.index(
                int(row["requested_total_force_budget"])
            )
            ordinal = int(row["sample_ordinal"])
            call = int(row["selected_call_per_image"])
            value = barrier_by_call[call]
            matrix = matrices[row["window_kind"]]
            matrix["barrier"][budget_index, ordinal] = value
            matrix["error"][budget_index, ordinal] = abs(
                value - REFERENCE_BARRIER_KCAL
            )
            matrix["force"][budget_index, ordinal] = int(
                row["actual_total_force"]
            )
            matrix["subdivision"][budget_index, ordinal] = (
                subdivision_by_call[call]
            )
        trailing = matrices["trailing"]
        anytime = matrices["anytime"]
        derived = dense_result.with_name(
            dense_result.stem + "_budget_windows.npz"
        )
        np.savez_compressed(
            derived,
            source_dense_result=np.array(str(dense_result)),
            requested_total_force_budgets=np.asarray(
                FORCE_BUDGETS, dtype=np.int64
            ),
            trailing_adaptive_barrier_kcal=trailing["barrier"],
            trailing_barrier_error_kcal=trailing["error"],
            trailing_actual_total_force=trailing["force"],
            trailing_selected_subdivisions=trailing["subdivision"],
            trailing_rms_barrier_error_kcal_by_budget=np.sqrt(
                np.mean(trailing["error"] ** 2, axis=1)
            ),
            anytime_adaptive_barrier_kcal=anytime["barrier"],
            anytime_barrier_error_kcal=anytime["error"],
            anytime_actual_total_force=anytime["force"],
            anytime_selected_subdivisions=anytime["subdivision"],
            anytime_rms_barrier_error_kcal_by_budget=np.sqrt(
                np.mean(anytime["error"] ** 2, axis=1)
            ),
            reference_barrier_kcal=np.array(REFERENCE_BARRIER_KCAL),
        )
        record["dense_result_sha256"] = sha256(dense_result)
        record["dense_audit_sha256"] = sha256(dense_audit)
        record["derived_budget_windows"] = str(derived)
        record["derived_budget_windows_sha256"] = sha256(derived)
        record["diagnostic_joint_evaluations"] = backend
        record["trailing_rms_barrier_error_kcal_by_budget"] = np.sqrt(
            np.mean(trailing["error"] ** 2, axis=1)
        ).tolist()
        record["anytime_rms_barrier_error_kcal_by_budget"] = np.sqrt(
            np.mean(anytime["error"] ** 2, axis=1)
        ).tolist()
        record["endpoint_dense_barrier_kcal_by_budget"] = (
            trailing["barrier"][:, -1].tolist()
        )
        diagnostic_total += backend

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_grid_summary": str(summary_path),
        "source_grid_summary_sha256": sha256(summary_path),
        "requested_total_force_budgets": list(FORCE_BUDGETS),
        "checkpoint_policy": (
            "Seven latest complete operator-split macrostates at or before "
            "force-uniform targets in each of (i) the final 30% window and "
            "(ii) the full anytime window from the first algorithm-produced "
            "macrostate, including the endpoint."
        ),
        "trailing_fraction": TRAILING_FRACTION,
        "trailing_samples": TRAILING_SAMPLES,
        "reference_barrier_kcal": REFERENCE_BARRIER_KCAL,
        "barrier_protocol": (
            "Real OpenMM Cartesian-linear 10/20 subdivisions per segment; "
            "40/80 while successive barriers differ by >0.05 kcal/mol."
        ),
        "sparse_22_image_barrier_used": False,
        "diagnostic_calls_excluded_from_optimization_axis": True,
        "total_diagnostic_joint_evaluations": diagnostic_total,
        "records": sorted(records, key=lambda row: row["alpha"]),
    }
    output = args.output_dir / "dense_manifest.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
