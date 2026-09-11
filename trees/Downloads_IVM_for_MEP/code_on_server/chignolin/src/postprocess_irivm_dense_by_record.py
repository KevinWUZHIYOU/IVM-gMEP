#!/usr/bin/env python3
"""Record-indexed dense barriers for committed Chignolin IR-IVM states.

The historical Chignolin dense-barrier postprocessor identifies paths by the
maximum number of calls made to any image.  IR-IVM has deliberately uneven
per-image counters, so that label is not a unique state identifier.  This
standalone postprocessor instead binds every diagnostic result to an explicit
optimization-history record and to the paper cost
``counter_vectors.sum(axis=1)``.

Only record zero (the initial path) and records whose reparameterization was
committed are eligible.  OpenMM is imported lazily inside the diagnostic
worker so the eligibility and mapping logic can be tested without creating an
OpenMM backend.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


N_IMAGES = 22
COMPLETE_STATE_DEFINITION = "initial_or_reparam_committed"
INITIAL_SUBDIVISIONS = 10
COMPARISON_SUBDIVISIONS = 20
ADAPTIVE_THRESHOLD_KCAL = 0.05
MAXIMUM_SUBDIVISIONS = 80


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def atomic_save_npz(destination: Path, **arrays) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)


def atomic_write_json(destination: Path, payload: Dict) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, destination)


def parse_record_indices(text: str) -> np.ndarray:
    fields = [field.strip() for field in str(text).split(",")]
    if not fields or any(not field for field in fields):
        raise ValueError(
            "--record-indices must be a non-empty comma-separated list"
        )
    try:
        indices = np.asarray([int(field) for field in fields], dtype=np.int64)
    except ValueError as error:
        raise ValueError("--record-indices must contain integers only") from error
    if np.any(indices < 0):
        raise ValueError("record indices must be non-negative")
    if len(indices) > 1 and np.any(np.diff(indices) <= 0):
        raise ValueError(
            "record indices must be unique and strictly increasing"
        )
    return indices


def build_record_selection(
    *,
    counter_vectors: np.ndarray,
    i_perp: np.ndarray,
    reparam_committed: np.ndarray,
    record_indices: Sequence[int],
    stored_paths: np.ndarray,
    complete_state_definition: str,
    calls_per_image: Optional[np.ndarray] = None,
    stored_path_calls: Optional[np.ndarray] = None,
    stored_path_record_indices: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Validate eligibility and map history records to exact stored paths.

    This function is intentionally independent of OpenMM.  If the result does
    not carry explicit ``stored_path_record_indices``, a one-to-one alignment
    between history records and stored paths is required and audited through
    the historical call labels.  No path is ever selected by a call label.
    """

    if str(complete_state_definition) != COMPLETE_STATE_DEFINITION:
        raise ValueError(
            "IR-IVM result has an incompatible complete-state definition: "
            f"{complete_state_definition!r}"
        )

    counters = np.asarray(counter_vectors)
    if counters.ndim != 2 or counters.shape[1] != N_IMAGES:
        raise ValueError(
            f"counter_vectors must have shape (records, {N_IMAGES})"
        )
    integer_counters = np.asarray(counters, dtype=np.int64)
    if not np.array_equal(counters, integer_counters):
        raise ValueError("counter_vectors must contain exact integers")
    if np.any(integer_counters < 0):
        raise ValueError("counter_vectors contain negative counts")
    if np.any(integer_counters[:, 0] < 1) or np.any(
        integer_counters[:, -1] < 1
    ):
        raise ValueError("fixed-endpoint first evaluations are not included")

    n_records = int(integer_counters.shape[0])
    metric = np.asarray(i_perp, dtype=float)
    committed = np.asarray(reparam_committed, dtype=bool)
    if metric.shape != (n_records,):
        raise ValueError("i_perp must contain one value per history record")
    if committed.shape != (n_records,):
        raise ValueError(
            "reparam_committed must contain one flag per history record"
        )

    indices = np.asarray(record_indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("at least one record index is required")
    if np.any(indices < 0) or np.any(indices >= n_records):
        raise IndexError(
            f"record index outside the available range 0..{n_records - 1}"
        )
    if len(indices) > 1 and np.any(np.diff(indices) <= 0):
        raise ValueError(
            "record indices must be unique and strictly increasing"
        )

    complete = committed.copy()
    complete[0] = True
    ineligible = indices[~complete[indices]]
    if ineligible.size:
        joined = ",".join(str(int(value)) for value in ineligible)
        raise ValueError(
            "requested records are neither the initial state nor committed "
            f"reparameterizations: {joined}"
        )

    total_nf = integer_counters.sum(axis=1, dtype=np.int64)
    if n_records > 1 and np.any(np.diff(total_nf) < 0):
        raise ValueError("paper total NF is not cumulative")
    if not np.all(np.isfinite(metric[indices])):
        raise ValueError("selected I_perp values must be finite")

    paths = np.asarray(stored_paths, dtype=float)
    if paths.ndim != 3 or paths.shape[1] != N_IMAGES:
        raise ValueError(
            f"stored_paths must have shape (stored records, {N_IMAGES}, dim)"
        )

    if stored_path_record_indices is not None:
        path_records = np.asarray(
            stored_path_record_indices, dtype=np.int64
        )
        if path_records.shape != (len(paths),):
            raise ValueError(
                "stored_path_record_indices and stored_paths differ in length"
            )
        if np.any(path_records < 0) or np.any(path_records >= n_records):
            raise ValueError("stored_path_record_indices are out of range")
        if len(np.unique(path_records)) != len(path_records):
            raise ValueError("stored_path_record_indices are not unique")
        record_to_path = {
            int(record): int(path_index)
            for path_index, record in enumerate(path_records)
        }
        missing = [
            int(record)
            for record in indices
            if int(record) not in record_to_path
        ]
        if missing:
            raise ValueError(
                "selected history records have no stored paths: "
                + ",".join(str(value) for value in missing)
            )
        path_indices = np.asarray(
            [record_to_path[int(record)] for record in indices],
            dtype=np.int64,
        )
        mapping_policy = "explicit_stored_path_record_indices"
    else:
        if len(paths) != n_records:
            raise ValueError(
                "without stored_path_record_indices, every history record "
                "must have exactly one stored path"
            )
        if calls_per_image is None or stored_path_calls is None:
            raise ValueError(
                "implicit one-to-one mapping requires calls_per_image and "
                "stored_path_calls for alignment audit"
            )
        history_calls = np.asarray(calls_per_image, dtype=np.int64)
        path_calls = np.asarray(stored_path_calls, dtype=np.int64)
        if history_calls.shape != (n_records,) or path_calls.shape != (
            n_records,
        ):
            raise ValueError("history and stored-path call labels are mis-sized")
        if not np.array_equal(history_calls, path_calls):
            raise ValueError(
                "stored paths are not aligned one-to-one with history records"
            )
        path_indices = indices.copy()
        mapping_policy = "audited_one_to_one_history_order"

    selected_paths = np.asarray(paths[path_indices], dtype=float)
    state_kind = np.asarray(
        [
            "initial" if int(index) == 0 else "reparam_committed"
            for index in indices
        ]
    )
    return {
        "record_index": indices,
        "stored_path_index": path_indices,
        "paper_total_nf": total_nf[indices],
        "i_perp": metric[indices],
        "reparam_committed": committed[indices],
        "complete_state_kind": state_kind,
        "paths": selected_paths,
        "selected_counter_vectors": integer_counters[indices],
        "final_counter_vector": integer_counters[-1],
        "mapping_policy": np.asarray(mapping_policy),
    }


def load_record_selection(
    result: Path, record_indices: Sequence[int]
) -> Dict[str, np.ndarray]:
    result = Path(result).expanduser().resolve()
    with np.load(result, allow_pickle=False) as saved:
        required = {
            "counter_vectors",
            "i_perp",
            "reparam_committed",
            "stored_paths",
            "calls_per_image",
            "stored_path_calls",
            "initial_path_sha256",
            "total_internal_nf_complete_state_definition",
        }
        missing = sorted(required.difference(saved.files))
        if missing:
            raise ValueError(
                "IR-IVM result is missing required arrays: " + ",".join(missing)
            )
        selection = build_record_selection(
            counter_vectors=saved["counter_vectors"],
            i_perp=saved["i_perp"],
            reparam_committed=saved["reparam_committed"],
            record_indices=record_indices,
            stored_paths=saved["stored_paths"],
            complete_state_definition=str(
                np.asarray(
                    saved["total_internal_nf_complete_state_definition"]
                ).item()
            ),
            calls_per_image=saved["calls_per_image"],
            stored_path_calls=saved["stored_path_calls"],
            stored_path_record_indices=(
                saved["stored_path_record_indices"]
                if "stored_path_record_indices" in saved.files
                else None
            ),
        )
        selection["initial_path_sha256"] = np.asarray(
            str(np.asarray(saved["initial_path_sha256"]).item())
        )
    return selection


def validate_source_audit(
    result: Path, final_counter_vector: np.ndarray
) -> Tuple[Path, str, Dict]:
    audit_path = Path(result).with_suffix(".audit.json")
    if not audit_path.is_file():
        raise FileNotFoundError(f"missing optimization counter audit: {audit_path}")
    audit = json.loads(audit_path.read_text())
    backend = int(audit.get("backend_raw_joint_evaluations", -1))
    summed = int(audit.get("sum_image_joint_evaluations", -2))
    per_image = np.asarray(
        audit.get("per_image_joint_evaluations", []), dtype=np.int64
    )
    if (
        not audit.get("consistent", False)
        or backend != summed
        or per_image.shape != (N_IMAGES,)
        or not np.array_equal(
            per_image, np.asarray(final_counter_vector, dtype=np.int64)
        )
    ):
        raise ValueError(f"optimization counter audit is inconsistent: {audit_path}")
    return audit_path, sha256_file(audit_path), audit


def process_chunk(
    pdb_path: str, row_indices: Sequence[int], paths: np.ndarray
) -> Tuple[List[Dict], Dict[str, int]]:
    # Lazy import is deliberate: mapping/eligibility tests must not initialize
    # or even require an OpenMM backend.
    from chignolin_core import (  # pylint: disable=import-outside-toplevel
        ImageEvaluator,
        OpenMMBackend,
        densified_barrier,
    )

    backend = OpenMMBackend(
        Path(pdb_path),
        platform_preference=("CPU",),
        precision="double",
        label="irivm_dense_barrier_by_record",
    )
    evaluator = ImageEvaluator(backend, image_index=0)
    rows = []
    for row_index, path in zip(row_indices, paths):
        before = int(backend.raw_joint_evaluations)
        barrier10, _ = densified_barrier(
            path, evaluator, INITIAL_SUBDIVISIONS
        )
        barrier20, _ = densified_barrier(
            path, evaluator, COMPARISON_SUBDIVISIONS
        )
        chosen = float(barrier20)
        subdivisions = COMPARISON_SUBDIVISIONS
        previous = float(barrier20)
        difference = abs(float(barrier20) - float(barrier10))
        while (
            difference > ADAPTIVE_THRESHOLD_KCAL
            and subdivisions < MAXIMUM_SUBDIVISIONS
        ):
            subdivisions *= 2
            chosen, _ = densified_barrier(path, evaluator, subdivisions)
            chosen = float(chosen)
            difference = abs(chosen - previous)
            previous = chosen
        rows.append(
            {
                "row_index": int(row_index),
                "barrier10": float(barrier10),
                "barrier20": float(barrier20),
                "adaptive_barrier": chosen,
                "subdivisions": int(subdivisions),
                "final_level_difference": float(difference),
                "diagnostic_joint_evaluations": int(
                    backend.raw_joint_evaluations - before
                ),
            }
        )
    return rows, {
        "backend_raw_joint_evaluations": int(
            backend.raw_joint_evaluations
        ),
        "evaluator_cache_misses": int(evaluator.counters.joint_evaluations),
        "energy_requests": int(evaluator.counters.energy_requests),
        "force_requests": int(evaluator.counters.force_requests),
        "cache_hits": int(evaluator.counters.cache_hits),
    }


def split_chunks(
    row_indices: np.ndarray, paths: np.ndarray, workers: int
) -> List[Tuple[List[int], np.ndarray]]:
    chunks = []
    for worker in range(int(workers)):
        selected = np.arange(worker, len(row_indices), int(workers))
        if selected.size:
            chunks.append(
                (
                    np.asarray(row_indices)[selected].tolist(),
                    np.asarray(paths)[selected],
                )
            )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--record-indices",
        required=True,
        help=(
            "Strictly increasing comma-separated optimization-history "
            "record indices. Only record 0 or reparam_committed records pass."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output and its audit explicitly.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    result = args.result.expanduser().resolve()
    output = args.output.expanduser().resolve()
    audit_output = output.with_suffix(".audit.json")
    if output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    if result == output:
        raise ValueError("diagnostic output must differ from the source result")
    if not args.overwrite and (output.exists() or audit_output.exists()):
        raise FileExistsError(
            f"refusing to overwrite {output} or {audit_output}"
        )

    record_indices = parse_record_indices(args.record_indices)
    selection = load_record_selection(result, record_indices)
    source_audit_path, source_audit_sha256, source_audit = (
        validate_source_audit(result, selection["final_counter_vector"])
    )

    pdb_path = root / "inputs" / "chignolin_clean.pdb"
    if not pdb_path.is_file():
        raise FileNotFoundError(f"missing Chignolin topology: {pdb_path}")
    paths = np.asarray(selection["paths"], dtype=float)
    workers = max(1, min(int(args.workers), len(paths)))
    row_indices = np.arange(len(paths), dtype=np.int64)
    chunks = split_chunks(row_indices, paths, workers)
    payloads = []
    if workers == 1:
        chunk_rows, chunk_paths = chunks[0]
        payloads.append(process_chunk(str(pdb_path), chunk_rows, chunk_paths))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers
        ) as executor:
            futures = [
                executor.submit(
                    process_chunk, str(pdb_path), chunk_rows, chunk_paths
                )
                for chunk_rows, chunk_paths in chunks
            ]
            for future in concurrent.futures.as_completed(futures):
                payloads.append(future.result())

    rows = sorted(
        [row for payload, _ in payloads for row in payload],
        key=lambda row: row["row_index"],
    )
    if [row["row_index"] for row in rows] != list(range(len(paths))):
        raise RuntimeError("diagnostic workers did not cover every selected path")

    backend_total = sum(
        audit["backend_raw_joint_evaluations"] for _, audit in payloads
    )
    misses_total = sum(
        audit["evaluator_cache_misses"] for _, audit in payloads
    )
    energy_requests = sum(
        audit["energy_requests"] for _, audit in payloads
    )
    force_requests = sum(
        audit["force_requests"] for _, audit in payloads
    )
    cache_hits = sum(audit["cache_hits"] for _, audit in payloads)
    per_record_calls = np.asarray(
        [row["diagnostic_joint_evaluations"] for row in rows],
        dtype=np.int64,
    )
    row_total = int(np.sum(per_record_calls))
    if not backend_total == misses_total == row_total:
        raise RuntimeError(
            "diagnostic backend/cache-miss audit mismatch: "
            f"{backend_total}, {misses_total}, {row_total}"
        )

    source_result_sha256 = sha256_file(result)
    selected_paths_sha256 = sha256_array(paths)
    adaptive = np.asarray(
        [row["adaptive_barrier"] for row in rows], dtype=float
    )
    subdivisions = np.asarray(
        [row["subdivisions"] for row in rows], dtype=np.int64
    )
    atomic_save_npz(
        output,
        source_result=np.asarray(str(result)),
        source_result_sha256=np.asarray(source_result_sha256),
        source_optimization_audit=np.asarray(str(source_audit_path)),
        source_optimization_audit_sha256=np.asarray(source_audit_sha256),
        source_initial_path_sha256=selection["initial_path_sha256"],
        source_selected_paths_sha256=np.asarray(selected_paths_sha256),
        record_index=selection["record_index"],
        stored_path_index=selection["stored_path_index"],
        paper_total_nf=selection["paper_total_nf"],
        i_perp=selection["i_perp"],
        reparam_committed=selection["reparam_committed"],
        complete_state_kind=selection["complete_state_kind"],
        complete_state_definition=np.asarray(COMPLETE_STATE_DEFINITION),
        path_mapping_policy=selection["mapping_policy"],
        selected_counter_vectors=selection["selected_counter_vectors"],
        barrier_10_kcal=np.asarray(
            [row["barrier10"] for row in rows], dtype=float
        ),
        barrier_20_kcal=np.asarray(
            [row["barrier20"] for row in rows], dtype=float
        ),
        adaptive_dense_barrier_kcal=adaptive,
        selected_subdivisions=subdivisions,
        final_level_difference_kcal=np.asarray(
            [row["final_level_difference"] for row in rows], dtype=float
        ),
        diagnostic_joint_evaluations_per_record=per_record_calls,
        diagnostic_backend_raw_joint_evaluations=np.asarray(backend_total),
        diagnostic_evaluator_cache_misses=np.asarray(misses_total),
        diagnostic_energy_requests=np.asarray(energy_requests),
        diagnostic_force_requests=np.asarray(force_requests),
        diagnostic_cache_hits=np.asarray(cache_hits),
        diagnostic_counts_excluded_from_paper_total_nf=np.asarray(True),
        paper_cost_axis=np.asarray("counter_vectors.sum(axis=1)"),
        paper_total_nf_includes_all_22_images=np.asarray(True),
        initial_subdivisions=np.asarray(INITIAL_SUBDIVISIONS),
        comparison_subdivisions=np.asarray(COMPARISON_SUBDIVISIONS),
        adaptive_threshold_kcal=np.asarray(ADAPTIVE_THRESHOLD_KCAL),
        maximum_subdivisions=np.asarray(MAXIMUM_SUBDIVISIONS),
    )
    output_sha256 = sha256_file(output)

    audit_rows = []
    for row_index, record_index in enumerate(selection["record_index"]):
        audit_rows.append(
            {
                "record_index": int(record_index),
                "stored_path_index": int(
                    selection["stored_path_index"][row_index]
                ),
                "complete_state_kind": str(
                    selection["complete_state_kind"][row_index]
                ),
                "reparam_committed": bool(
                    selection["reparam_committed"][row_index]
                ),
                "paper_total_nf": int(selection["paper_total_nf"][row_index]),
                "i_perp": float(selection["i_perp"][row_index]),
                "adaptive_dense_barrier_kcal": float(adaptive[row_index]),
                "selected_subdivisions": int(subdivisions[row_index]),
                "diagnostic_joint_evaluations": int(
                    per_record_calls[row_index]
                ),
            }
        )
    audit = {
        "source_result": str(result),
        "source_result_sha256": source_result_sha256,
        "source_optimization_audit": str(source_audit_path),
        "source_optimization_audit_sha256": source_audit_sha256,
        "source_optimization_backend_raw_joint_evaluations": int(
            source_audit["backend_raw_joint_evaluations"]
        ),
        "output": str(output),
        "output_sha256": output_sha256,
        "source_selected_paths_sha256": selected_paths_sha256,
        "complete_state_definition": COMPLETE_STATE_DEFINITION,
        "eligibility_rule": (
            "record_index == 0 or reparam_committed[record_index]"
        ),
        "all_selected_records_eligible": True,
        "path_mapping_policy": str(selection["mapping_policy"].item()),
        "optimization_cost_axis": "counter_vectors.sum(axis=1)",
        "cost_includes_all_22_images": True,
        "fixed_endpoint_first_evaluations_included": True,
        "diagnostic_counts_excluded_from_optimization_axis": True,
        "barrier_protocol": {
            "energy_backend": "real OpenMM CPU double precision",
            "interpolation": "linear Cartesian interpolation per segment",
            "initial_subdivisions": INITIAL_SUBDIVISIONS,
            "comparison_subdivisions": COMPARISON_SUBDIVISIONS,
            "adaptive_threshold_kcal": ADAPTIVE_THRESHOLD_KCAL,
            "maximum_subdivisions": MAXIMUM_SUBDIVISIONS,
        },
        "records": audit_rows,
        "selected_initial_states": int(
            np.count_nonzero(selection["record_index"] == 0)
        ),
        "selected_committed_reparameterizations": int(
            np.count_nonzero(selection["record_index"] != 0)
        ),
        "diagnostic_backend_raw_joint_evaluations": int(backend_total),
        "diagnostic_evaluator_cache_misses": int(misses_total),
        "diagnostic_row_joint_evaluations": int(row_total),
        "diagnostic_energy_requests": int(energy_requests),
        "diagnostic_force_requests": int(force_requests),
        "diagnostic_cache_hits": int(cache_hits),
        "selected_subdivision_counts": {
            str(level): int(np.count_nonzero(subdivisions == level))
            for level in (20, 40, 80)
        },
    }
    atomic_write_json(audit_output, audit)
    print(f"Saved {output}")
    print(f"Saved {audit_output}")
    print(f"output_sha256={output_sha256}")
    print(f"audit_sha256={sha256_file(audit_output)}")


if __name__ == "__main__":
    main()
