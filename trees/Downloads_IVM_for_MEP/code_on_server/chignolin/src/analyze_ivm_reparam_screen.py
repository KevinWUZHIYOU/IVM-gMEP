#!/usr/bin/env python3
"""Dense-barrier analysis for the audited IVM reparameterization screen."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
from pathlib import Path

import numpy as np

from chignolin_core import TANGENT_DEFINITION, array_sha256, atomic_save_npz
from postprocess_dense_barrier import process_chunk, split_chunks


def forward_fill(calls, values, grid):
    calls = np.asarray(calls, dtype=int)
    values = np.asarray(values, dtype=float)
    order = np.argsort(calls)
    calls, values = calls[order], values[order]
    indices = np.searchsorted(calls, grid, side="right") - 1
    indices = np.clip(indices, 0, len(calls) - 1)
    return values[indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    result_paths = sorted(
        (root / "runs" / "ivm").glob("ivm_reparam_screen_*.npz")
    )
    if len(result_paths) != 18:
        raise RuntimeError(f"Expected 18 screen results, found {len(result_paths)}")

    records = []
    final_paths = []
    initial_hashes = set()
    grid = np.arange(1, 201, dtype=int)
    with np.load(
        root / "runs" / "neb" / "neb_pointwise_envelope_budget500.npz",
        allow_pickle=False,
    ) as saved:
        neb_i_envelope = np.asarray(saved["i_perp"][:200], dtype=float)
        neb_b_envelope = np.asarray(saved["barrier_kcal"][:200], dtype=float)
    with np.load(
        root
        / "runs"
        / "diagnostics"
        / "dense_barrier_neb_production_all_stored_calls.npz",
        allow_pickle=False,
    ) as saved:
        barrier_reference = float(saved["adaptive_barrier_kcal"][-1])

    for result_path in result_paths:
        with np.load(result_path, allow_pickle=False) as saved:
            if str(saved["run_label"]) != "reparam_screen":
                raise RuntimeError(f"Unexpected run label: {result_path}")
            if str(saved["tangent_definition"]) != TANGENT_DEFINITION:
                raise RuntimeError(f"Non-central tangent: {result_path}")
            initial_hashes.add(str(saved["initial_path_sha256"]))
            calls = np.asarray(saved["calls_per_image"], dtype=int)
            if int(calls[-1]) != 200:
                raise RuntimeError(f"Screen run did not reach 200 calls: {result_path}")
            stored_calls = np.asarray(saved["stored_path_calls"], dtype=int)
            selected = np.flatnonzero(stored_calls == 200)
            if selected.size != 1:
                raise RuntimeError(f"Missing unique call-200 path: {result_path}")
            final_paths.append(
                np.asarray(saved["stored_paths"][selected[0]], dtype=float)
            )
            iperp_curve = forward_fill(calls, saved["i_perp"], grid)
            barrier_curve = forward_fill(calls, saved["barrier_kcal"], grid)
            records.append(
                {
                    "result_path": str(result_path),
                    "spring_reparam_every": int(saved["reparam_every"]),
                    "objective_tolerance_kcal": float(
                        saved["reparam_objective_tolerance_kcal"]
                    ),
                    "energy_tolerance_kcal": float(
                        saved["reparam_energy_tolerance_kcal"]
                    ),
                    "force_rtol": float(saved["reparam_force_rtol"]),
                    "i_perp_at_200": float(saved["i_perp"][-1]),
                    "raw_barrier_at_200_kcal": float(
                        saved["barrier_kcal"][-1]
                    ),
                    "physical_fmax_at_200": float(
                        saved["physical_fmax"][-1]
                    ),
                    "mean_reparam_accepted_fraction": float(
                        np.nanmean(saved["reparam_accepted_fraction"])
                    ),
                    "i_perp_loss_count_vs_neb_envelope": int(
                        np.count_nonzero(
                            iperp_curve[1:] > neb_i_envelope[1:]
                        )
                    ),
                    "raw_barrier_loss_count_vs_neb_envelope": int(
                        np.count_nonzero(
                            barrier_curve[1:] > neb_b_envelope[1:]
                        )
                    ),
                }
            )
    if len(initial_hashes) != 1:
        raise RuntimeError(f"Screen used multiple initial paths: {initial_hashes}")

    workers = max(1, min(args.workers, len(records)))
    chunks = split_chunks(
        np.arange(len(records)),
        np.asarray(final_paths),
        workers,
    )
    payloads = []
    pdb_path = root / "inputs" / "chignolin_clean.pdb"
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_chunk, str(pdb_path), indices, paths)
            for indices, paths in chunks
        ]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            payloads.append(future.result())
            print(
                f"reparameterization-screen dense worker "
                f"{completed}/{len(futures)} complete",
                flush=True,
            )
    dense_rows = sorted(
        [row for rows, _ in payloads for row in rows],
        key=lambda item: item["index"],
    )
    if [row["index"] for row in dense_rows] != list(range(len(records))):
        raise RuntimeError("Dense workers did not cover every screen result")
    backend_total = sum(
        audit["backend_raw_joint_evaluations"] for _, audit in payloads
    )
    evaluator_total = sum(
        audit["evaluator_joint_evaluations"] for _, audit in payloads
    )
    row_total = sum(
        row["diagnostic_joint_evaluations"] for row in dense_rows
    )
    if not (backend_total == evaluator_total == row_total):
        raise RuntimeError("Dense diagnostic counter audit mismatch")

    for record, dense in zip(records, dense_rows):
        record["dense_barrier_at_200_kcal"] = dense["adaptive_barrier"]
        record["dense_barrier_error_at_200_kcal"] = abs(
            dense["adaptive_barrier"] - barrier_reference
        )
        record["selected_subdivisions"] = dense["subdivisions"]
        record["final_level_difference_kcal"] = dense[
            "final_level_difference"
        ]
        record["diagnostic_joint_evaluations"] = dense[
            "diagnostic_joint_evaluations"
        ]
    minimum_i = min(record["i_perp_at_200"] for record in records)
    minimum_error = max(
        min(record["dense_barrier_error_at_200_kcal"] for record in records),
        1e-12,
    )
    for record in records:
        record["score"] = float(
            np.log10(record["i_perp_at_200"] / minimum_i)
            + np.log10(
                max(record["dense_barrier_error_at_200_kcal"], 1e-12)
                / minimum_error
            )
            + 0.25
            * record["i_perp_loss_count_vs_neb_envelope"]
            / 199.0
        )
    records.sort(key=lambda item: item["score"])
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank

    output = (
        root
        / "runs"
        / "diagnostics"
        / "dense_barrier_ivm_reparam_screen_call200.npz"
    )
    atomic_save_npz(
        output,
        result_paths=np.asarray([record["result_path"] for record in records]),
        final_paths_sha256=np.asarray(
            [
                array_sha256(
                    final_paths[result_paths.index(Path(record["result_path"]))]
                )
                for record in records
            ]
        ),
        dense_barrier_kcal=np.asarray(
            [record["dense_barrier_at_200_kcal"] for record in records]
        ),
        selected_subdivisions=np.asarray(
            [record["selected_subdivisions"] for record in records],
            dtype=int,
        ),
        diagnostic_backend_raw_joint_evaluations=np.array(backend_total),
        diagnostic_evaluator_cache_misses=np.array(evaluator_total),
    )
    report = {
        "initial_path_sha256": next(iter(initial_hashes)),
        "barrier_reference_kcal": barrier_reference,
        "n_configurations": len(records),
        "diagnostic_backend_raw_joint_evaluations": backend_total,
        "diagnostic_evaluator_cache_misses": evaluator_total,
        "counts_excluded_from_optimization_axis": True,
        "ranking": records,
        "top_six": records[:6],
    }
    report_path = root / "reports" / "ivm_reparam_screen_analysis.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    csv_path = root / "reports" / "ivm_reparam_screen_ranking.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
