#!/usr/bin/env python3
"""Adaptive, separately audited dense 0 K barrier postprocessing."""

from __future__ import annotations

import argparse
import concurrent.futures
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


def process_chunk(pdb_path: str, indices, paths):
    backend = OpenMMBackend(
        Path(pdb_path),
        platform_preference=("CPU",),
        precision="double",
        label="dense_barrier_diagnostic",
    )
    evaluator = ImageEvaluator(backend, image_index=0)
    rows = []
    for call_index, path in zip(indices, paths):
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
                "index": int(call_index),
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
    audit = {
        "backend_raw_joint_evaluations": int(backend.raw_joint_evaluations),
        "evaluator_joint_evaluations": int(
            evaluator.counters.joint_evaluations
        ),
        "energy_requests": int(evaluator.counters.energy_requests),
        "force_requests": int(evaluator.counters.force_requests),
        "cache_hits": int(evaluator.counters.cache_hits),
    }
    return rows, audit


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-label", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--calls",
        default="",
        help="Optional comma-separated stored call labels; default processes all",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    result_path = Path(args.result).expanduser().resolve()
    with np.load(result_path, allow_pickle=False) as saved:
        calls = np.asarray(saved["stored_path_calls"], dtype=int)
        paths = np.asarray(saved["stored_paths"], dtype=float)
        result_initial_hash = str(saved["initial_path_sha256"])
    if len(calls) != len(paths):
        raise RuntimeError("Stored call/path length mismatch")
    source_path_records = len(calls)
    unique_calls, reverse_indices = np.unique(
        calls[::-1], return_index=True
    )
    keep = len(calls) - 1 - reverse_indices
    keep = keep[np.argsort(calls[keep], kind="stable")]
    duplicate_call_records_collapsed = int(len(calls) - len(keep))
    calls = calls[keep]
    paths = paths[keep]
    if len(unique_calls) != len(calls):
        raise RuntimeError("Failed to retain one final path per call label")
    if args.calls:
        requested = {
            int(item) for item in args.calls.split(",") if item.strip()
        }
        mask = np.asarray([int(call) in requested for call in calls])
        missing = requested.difference(int(call) for call in calls[mask])
        if missing:
            raise ValueError(f"Requested calls are not stored: {sorted(missing)}")
        calls = calls[mask]
        paths = paths[mask]

    workers = max(1, min(args.workers, len(calls)))
    chunks = split_chunks(np.arange(len(calls)), paths, workers)
    payloads = []
    pdb_path = root / "inputs" / "chignolin_clean.pdb"
    if workers == 1:
        # Avoid creating an unnecessary process/semaphore for the serial
        # diagnostic path.  This also lets the fully audited postprocessor run
        # inside restricted batch environments that prohibit POSIX semaphores.
        indices, chunk_paths = chunks[0]
        payloads.append(
            process_chunk(str(pdb_path), indices, chunk_paths)
        )
        print("dense-barrier worker 1/1 complete", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers
        ) as executor:
            futures = [
                executor.submit(
                    process_chunk,
                    str(pdb_path),
                    indices,
                    chunk_paths,
                )
                for indices, chunk_paths in chunks
            ]
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                rows, audit = future.result()
                payloads.append((rows, audit))
                print(
                    f"dense-barrier worker {completed}/{len(futures)} complete",
                    flush=True,
                )

    rows = sorted(
        [row for payload, _ in payloads for row in payload],
        key=lambda item: item["index"],
    )
    if [row["index"] for row in rows] != list(range(len(calls))):
        raise RuntimeError("Dense barrier workers did not cover every path")
    backend_total = sum(
        audit["backend_raw_joint_evaluations"] for _, audit in payloads
    )
    evaluator_total = sum(
        audit["evaluator_joint_evaluations"] for _, audit in payloads
    )
    row_total = sum(row["diagnostic_joint_evaluations"] for row in rows)
    if not (backend_total == evaluator_total == row_total):
        raise RuntimeError(
            "Diagnostic backend/cache-miss audit mismatch: "
            f"{backend_total}, {evaluator_total}, {row_total}"
        )

    output = (
        root
        / "runs"
        / "diagnostics"
        / f"dense_barrier_{args.output_label}.npz"
    )
    atomic_save_npz(
        output,
        source_result=np.array(str(result_path)),
        source_initial_path_sha256=np.array(result_initial_hash),
        source_stored_paths_sha256=np.array(array_sha256(paths)),
        calls=calls,
        barrier_10_kcal=np.asarray([row["barrier10"] for row in rows]),
        barrier_20_kcal=np.asarray([row["barrier20"] for row in rows]),
        adaptive_barrier_kcal=np.asarray(
            [row["adaptive_barrier"] for row in rows]
        ),
        selected_subdivisions=np.asarray(
            [row["subdivisions"] for row in rows], dtype=int
        ),
        final_level_difference_kcal=np.asarray(
            [row["final_level_difference"] for row in rows]
        ),
        diagnostic_joint_evaluations_per_record=np.asarray(
            [row["diagnostic_joint_evaluations"] for row in rows], dtype=int
        ),
        diagnostic_backend_raw_joint_evaluations=np.array(backend_total),
        diagnostic_evaluator_cache_misses=np.array(evaluator_total),
    )
    report = {
        "source_result": str(result_path),
        "output": str(output),
        "source_path_records": source_path_records,
        "n_paths": len(calls),
        "duplicate_call_records_collapsed": duplicate_call_records_collapsed,
        "duplicate_policy": "retain the final recorded state at each max-call label",
        "initial_subdivisions": 10,
        "comparison_subdivisions": 20,
        "adaptive_threshold_kcal": 0.05,
        "maximum_subdivisions": 80,
        "diagnostic_backend_raw_joint_evaluations": backend_total,
        "diagnostic_evaluator_cache_misses": evaluator_total,
        "counts_excluded_from_optimization_axis": True,
        "selected_subdivision_counts": {
            str(level): int(
                np.count_nonzero(
                    np.asarray([row["subdivisions"] for row in rows]) == level
                )
            )
            for level in (20, 40, 80)
        },
    }
    report_path = output.with_suffix(".audit.json")
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, report_path)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
