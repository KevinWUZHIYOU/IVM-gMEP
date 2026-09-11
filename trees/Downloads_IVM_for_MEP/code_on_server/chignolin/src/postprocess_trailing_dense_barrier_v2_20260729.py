#!/usr/bin/env python3
"""Audited adaptive dense barriers over trailing total-force windows."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from chignolin_core import (
    ImageEvaluator,
    OpenMMBackend,
    array_sha256,
    atomic_save_npz,
    densified_barrier,
)


DEFAULT_BUDGETS = (1000, 3000, 6000, 10000)
DEFAULT_TRAILING_FRACTION = 0.30
DEFAULT_SAMPLES = 7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_trailing_indices(
    total_force: np.ndarray,
    budgets: tuple[int, ...],
    trailing_fraction: float,
    samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select latest measured states at equally spaced trailing targets."""
    if not 0.0 < float(trailing_fraction) <= 1.0:
        raise ValueError("trailing fraction must lie in (0, 1]")
    if int(samples) < 2:
        raise ValueError("samples must be at least two")
    total_force = np.asarray(total_force, dtype=np.int64)
    selected = np.zeros((len(budgets), int(samples)), dtype=np.int64)
    actual = np.zeros_like(selected)
    targets = np.zeros_like(selected, dtype=float)
    bounds = np.zeros((len(budgets), 2), dtype=float)
    for budget_index, budget in enumerate(budgets):
        eligible = np.flatnonzero(total_force <= int(budget))
        endpoint_index = int(eligible[-1]) if eligible.size else 0
        endpoint_force = int(total_force[endpoint_index])
        start_force = max(
            int(total_force[0]),
            int(np.ceil((1.0 - trailing_fraction) * endpoint_force)),
        )
        bounds[budget_index] = (start_force, endpoint_force)
        targets[budget_index] = np.linspace(
            start_force, endpoint_force, int(samples)
        )
        for sample_index, target in enumerate(targets[budget_index]):
            sample_eligible = np.flatnonzero(total_force <= target)
            index = int(sample_eligible[-1]) if sample_eligible.size else 0
            selected[budget_index, sample_index] = index
            actual[budget_index, sample_index] = total_force[index]
    return selected, actual, targets, bounds


def process_chunk(pdb_path: str, row_indices, paths):
    backend = OpenMMBackend(
        Path(pdb_path),
        platform_preference=("CPU",),
        precision="double",
        label="trailing_dense_barrier_diagnostic",
    )
    evaluator = ImageEvaluator(backend, image_index=0)
    rows = []
    for row_index, path in zip(row_indices, paths):
        before = backend.raw_joint_evaluations
        barrier10, _ = densified_barrier(path, evaluator, 10)
        barrier20, _ = densified_barrier(path, evaluator, 20)
        chosen = barrier20
        subdivisions = 20
        previous = barrier20
        difference = abs(barrier20 - barrier10)
        while difference > 0.05 and subdivisions < 80:
            subdivisions *= 2
            chosen, _ = densified_barrier(path, evaluator, subdivisions)
            difference = abs(chosen - previous)
            previous = chosen
        rows.append(
            {
                "row_index": int(row_index),
                "barrier10": float(barrier10),
                "barrier20": float(barrier20),
                "adaptive_barrier": float(chosen),
                "subdivisions": int(subdivisions),
                "final_level_difference": float(difference),
                "diagnostic_joint_evaluations": int(
                    backend.raw_joint_evaluations - before
                ),
            }
        )
    return rows, {
        "backend": int(backend.raw_joint_evaluations),
        "misses": int(evaluator.counters.joint_evaluations),
        "energy_requests": int(evaluator.counters.energy_requests),
        "force_requests": int(evaluator.counters.force_requests),
        "cache_hits": int(evaluator.counters.cache_hits),
    }


def split_chunks(indices, paths, workers):
    chunks = []
    for worker in range(workers):
        selected = np.arange(worker, len(indices), workers)
        if selected.size:
            chunks.append(
                (
                    np.asarray(indices)[selected].tolist(),
                    np.asarray(paths)[selected],
                )
            )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-label", required=True)
    parser.add_argument(
        "--budgets",
        default=",".join(str(value) for value in DEFAULT_BUDGETS),
    )
    parser.add_argument(
        "--trailing-fraction",
        type=float,
        default=DEFAULT_TRAILING_FRACTION,
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--reference-barrier",
        type=float,
        default=np.nan,
        help=(
            "Optional independently registered dense barrier used only to "
            "also report absolute errors; raw dense barriers are always saved."
        ),
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    result = args.result.expanduser().resolve()
    budgets = tuple(
        int(item) for item in args.budgets.split(",") if item.strip()
    )
    if not budgets:
        raise ValueError("at least one total-force budget is required")
    with np.load(result, allow_pickle=False) as saved:
        total_force = np.asarray(
            saved["counter_vectors"], dtype=np.int64
        ).sum(axis=1)
        history_calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        stored_calls = np.asarray(
            saved["stored_path_calls"], dtype=np.int64
        )
        stored_paths = np.asarray(saved["stored_paths"], dtype=float)
        initial_hash = str(saved["initial_path_sha256"])
    if not (
        len(total_force)
        == len(history_calls)
        == len(stored_calls)
        == len(stored_paths)
    ):
        raise RuntimeError(
            "This diagnostic requires --save-all-paths so every measured "
            "history state has one stored path."
        )
    if not np.array_equal(history_calls, stored_calls):
        raise RuntimeError(
            "Stored path labels are not aligned with measured history states."
        )

    selected, actual, targets, bounds = select_trailing_indices(
        total_force,
        budgets,
        args.trailing_fraction,
        args.samples,
    )
    unique_indices = np.unique(selected)
    unique_paths = stored_paths[unique_indices]
    workers = max(1, min(int(args.workers), len(unique_indices)))
    chunks = split_chunks(
        np.arange(len(unique_indices)), unique_paths, workers
    )
    payloads = []
    pdb_path = root / "inputs" / "chignolin_clean.pdb"
    if workers == 1:
        row_indices, chunk_paths = chunks[0]
        payloads.append(
            process_chunk(str(pdb_path), row_indices, chunk_paths)
        )
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers
        ) as executor:
            futures = [
                executor.submit(
                    process_chunk,
                    str(pdb_path),
                    row_indices,
                    chunk_paths,
                )
                for row_indices, chunk_paths in chunks
            ]
            for future in concurrent.futures.as_completed(futures):
                payloads.append(future.result())

    rows = sorted(
        [row for payload, _ in payloads for row in payload],
        key=lambda row: row["row_index"],
    )
    if [row["row_index"] for row in rows] != list(
        range(len(unique_indices))
    ):
        raise RuntimeError("Diagnostic workers did not cover every path")
    backend_total = sum(audit["backend"] for _, audit in payloads)
    misses_total = sum(audit["misses"] for _, audit in payloads)
    row_total = sum(
        row["diagnostic_joint_evaluations"] for row in rows
    )
    if not backend_total == misses_total == row_total:
        raise RuntimeError(
            "Diagnostic backend/cache-miss audit mismatch: "
            f"{backend_total}, {misses_total}, {row_total}"
        )

    unique_barriers = np.asarray(
        [row["adaptive_barrier"] for row in rows]
    )
    lookup = {
        int(source_index): row_index
        for row_index, source_index in enumerate(unique_indices)
    }
    mapped = np.vectorize(lambda value: lookup[int(value)])(selected)
    adaptive = unique_barriers[mapped]
    subdivisions = np.asarray(
        [row["subdivisions"] for row in rows], dtype=np.int64
    )[mapped]
    final_difference = np.asarray(
        [row["final_level_difference"] for row in rows]
    )[mapped]
    reference = float(args.reference_barrier)
    errors = (
        np.abs(adaptive - reference)
        if np.isfinite(reference)
        else np.full_like(adaptive, np.nan)
    )

    output = (
        root
        / "runs"
        / "diagnostics"
        / f"dense_barrier_{args.output_label}.npz"
    )
    atomic_save_npz(
        output,
        source_result=np.array(str(result)),
        source_result_sha256=np.array(sha256(result)),
        source_initial_path_sha256=np.array(initial_hash),
        requested_total_force_budgets=np.asarray(budgets, dtype=int),
        trailing_window_fraction=np.array(args.trailing_fraction),
        samples_per_window=np.array(args.samples),
        window_bounds_total_force=bounds,
        target_total_force=targets,
        actual_total_force=actual,
        source_record_indices=selected,
        unique_source_record_indices=unique_indices,
        source_paths_sha256=np.array(array_sha256(unique_paths)),
        adaptive_barrier_kcal=adaptive,
        reference_barrier_kcal=np.array(reference),
        barrier_error_kcal=errors,
        trailing_mean_barrier_kcal=np.mean(adaptive, axis=1),
        trailing_mean_barrier_error_kcal=np.mean(errors, axis=1),
        selected_subdivisions=subdivisions,
        final_level_difference_kcal=final_difference,
        diagnostic_joint_evaluations_per_unique_path=np.asarray(
            [row["diagnostic_joint_evaluations"] for row in rows],
            dtype=int,
        ),
        diagnostic_backend_raw_joint_evaluations=np.array(backend_total),
        diagnostic_evaluator_cache_misses=np.array(misses_total),
    )
    audit = {
        "source_result": str(result),
        "source_result_sha256": sha256(result),
        "output": str(output),
        "optimization_cost_axis": "counter_vectors.sum(axis=1)",
        "cost_includes_all_22_images": True,
        "counts_excluded_from_optimization_axis": True,
        "barrier_protocol": {
            "energy_backend": "real OpenMM CPU double precision",
            "interpolation": "linear Cartesian interpolation per segment",
            "initial_subdivisions": 10,
            "comparison_subdivisions": 20,
            "adaptive_threshold_kcal": 0.05,
            "maximum_subdivisions": 80,
        },
        "requested_total_force_budgets": list(budgets),
        "trailing_window_fraction": float(args.trailing_fraction),
        "samples_per_window": int(args.samples),
        "unique_paths": int(len(unique_indices)),
        "selected_subdivision_counts": {
            str(level): int(np.count_nonzero(subdivisions == level))
            for level in (20, 40, 80)
        },
        "diagnostic_backend_raw_joint_evaluations": int(backend_total),
        "diagnostic_evaluator_cache_misses": int(misses_total),
        "diagnostic_row_evaluations": int(row_total),
        "reference_barrier_kcal": (
            reference if np.isfinite(reference) else None
        ),
    }
    audit_path = output.with_suffix(".audit.json")
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2) + "\n")
    os.replace(temporary, audit_path)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
