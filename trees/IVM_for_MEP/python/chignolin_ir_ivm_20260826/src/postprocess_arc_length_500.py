#!/usr/bin/env python3
"""Evaluate Chignolin paths on one fixed 500-point arc-length grid.

The spatial diagnostic is deliberately independent of the optimizer and of
the number of images in the stored path.  Each selected 22-image path is
treated as a piecewise-linear curve in the unweighted 3N Cartesian coordinate
space.  Exactly 500 points, including both endpoints, are sampled at
``s_j = j L / 499`` and evaluated with the same OpenMM CPU/double backend.

Selection of optimization-history records is frozen in a JSON contract.  The
contract is also responsible for declaring which records are eligible; this
script verifies the declaration against the raw result before doing any
OpenMM work.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


N_IMAGES = 22
N_DENSE_POINTS = 500
SPACE_DEFINITION = "unweighted_3N_cartesian"
INTERPOLATION = "piecewise_linear"
ENDPOINT_POLICY = "include_both_endpoints"
GRID_FORMULA = "s_j=j*L/499,j=0,...,499"
METHOD_ORDER = (
    "IVM",
    "ODE-NEB",
    "Static-NEB",
    "Original String",
    "OBSM-FIRE",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_save_npz(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def arc_length_resample(path: np.ndarray, count: int = N_DENSE_POINTS) -> Tuple[np.ndarray, Dict]:
    """Return exactly ``count`` points uniformly spaced in cumulative arc length."""

    curve = np.asarray(path, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[0] != N_IMAGES or curve.shape[1] % 3:
        raise ValueError(f"expected a ({N_IMAGES}, 3N) path, got {curve.shape}")
    if int(count) < 2:
        raise ValueError("the arc-length grid must contain at least two points")
    if not np.all(np.isfinite(curve)):
        raise ValueError("path coordinates must be finite")
    segment_lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    if np.any(segment_lengths <= 0.0) or not np.all(np.isfinite(segment_lengths)):
        raise ValueError("every piecewise-linear path segment must have positive length")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    targets = np.linspace(0.0, total_length, int(count), dtype=np.float64)
    segment_index = np.searchsorted(cumulative, targets, side="right") - 1
    segment_index = np.clip(segment_index, 0, len(segment_lengths) - 1)
    fraction = (
        targets - cumulative[segment_index]
    ) / segment_lengths[segment_index]
    dense = (
        (1.0 - fraction[:, None]) * curve[segment_index]
        + fraction[:, None] * curve[segment_index + 1]
    )
    dense[0] = curve[0]
    dense[-1] = curve[-1]
    return dense, {
        "total_arc_length": total_length,
        "minimum_original_segment_length": float(np.min(segment_lengths)),
        "maximum_original_segment_length": float(np.max(segment_lengths)),
        "dense_spacing": total_length / float(int(count) - 1),
    }


def _scalar(saved, key: str):
    return np.asarray(saved[key]).item()


def _resolve_contract_path(contract_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = contract_path.parent / path
    return path.resolve()


def resolve_record_indices(
    *,
    contract_path: Path,
    record: Dict,
    calls_per_image: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    """Resolve a frozen record-selection rule to explicit history indices."""

    if "record_indices" in record:
        indices = np.asarray(record["record_indices"], dtype=np.int64)
    else:
        selection = record.get("record_selection", {})
        policy = str(selection.get("policy", ""))
        if policy == "all_saved_states":
            indices = np.arange(len(calls_per_image), dtype=np.int64)
        elif policy == "all_eligible_states":
            indices = np.flatnonzero(np.asarray(eligible, dtype=bool))
        elif policy == "calls_from_dense_result":
            dense_path = _resolve_contract_path(
                contract_path, selection["dense_result"]
            )
            if sha256_file(dense_path) != str(selection["dense_result_sha256"]):
                raise ValueError("record-selection dense-result hash mismatch")
            with np.load(dense_path, allow_pickle=False) as dense:
                selected_calls = np.asarray(dense["calls"], dtype=np.int64)
            resolved = []
            for call in selected_calls:
                matching = np.flatnonzero(calls_per_image == int(call))
                if not matching.size:
                    raise ValueError(
                        f"record-selection call {int(call)} is absent from the source"
                    )
                resolved.append(int(matching[-1]))
            indices = np.asarray(resolved, dtype=np.int64)
        else:
            raise ValueError(f"unsupported record-selection policy {policy!r}")
    if indices.ndim != 1 or not len(indices):
        raise ValueError("record selection resolved to no records")
    if np.any(indices < 0) or (len(indices) > 1 and np.any(np.diff(indices) <= 0)):
        raise ValueError("resolved record indices must be increasing and unique")
    if int(indices[-1]) >= len(calls_per_image):
        raise IndexError("resolved record index exceeds the source history")
    if not np.all(np.asarray(eligible, dtype=bool)[indices]):
        raise ValueError("record selection includes an ineligible state")
    expected_hash = record.get("resolved_record_indices_sha256")
    if expected_hash is not None and sha256_array(indices) != str(expected_hash):
        raise ValueError("resolved record-index hash mismatch")
    return indices


def load_selected_paths(contract_path: Path) -> Tuple[Dict, List[Dict]]:
    contract_path = Path(contract_path).expanduser().resolve()
    contract = json.loads(contract_path.read_text())
    protocol = contract.get("arc_length_500_protocol", {})
    required_protocol = {
        "coordinate_space": SPACE_DEFINITION,
        "interpolation": INTERPOLATION,
        "number_of_points": N_DENSE_POINTS,
        "endpoint_policy": ENDPOINT_POLICY,
        "grid_formula": GRID_FORMULA,
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(
                f"contract protocol {key!r} must be {expected!r}, "
                f"got {protocol.get(key)!r}"
            )
    methods = contract.get("methods", {})
    if tuple(methods) != METHOD_ORDER:
        raise ValueError(f"contract method order must be {METHOD_ORDER}")

    rows: List[Dict] = []
    for method in METHOD_ORDER:
        record = methods[method]
        result = _resolve_contract_path(contract_path, record["source_result"])
        if not result.is_file():
            raise FileNotFoundError(result)
        actual_result_hash = sha256_file(result)
        if actual_result_hash != str(record["source_result_sha256"]):
            raise ValueError(f"{method} source-result hash mismatch")
        with np.load(result, allow_pickle=False) as saved:
            required = {
                "stored_paths",
                "stored_path_calls",
                "calls_per_image",
                "counter_vectors",
                "i_perp",
                "initial_path_sha256",
            }
            missing = sorted(required.difference(saved.files))
            if missing:
                raise ValueError(f"{method} source is missing {','.join(missing)}")
            paths = np.asarray(saved["stored_paths"], dtype=np.float64)
            calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
            stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
            counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
            i_perp = np.asarray(saved["i_perp"], dtype=np.float64)
            if (
                paths.ndim != 3
                or paths.shape[1] != N_IMAGES
                or len(paths) != len(calls)
                or len(paths) != len(stored_calls)
                or counters.shape != (len(paths), N_IMAGES)
                or i_perp.shape != (len(paths),)
                or not np.array_equal(calls, stored_calls)
            ):
                raise ValueError(f"{method} history/path arrays are not one-to-one")
            if np.any(counters < 0) or np.any(counters[:, (0, -1)] < 1):
                raise ValueError(f"{method} counter vectors are invalid")
            total_nf = counters.sum(axis=1, dtype=np.int64)
            if len(total_nf) > 1 and np.any(np.diff(total_nf) < 0):
                raise ValueError(f"{method} Number-of-Forces history is not cumulative")
            initial_hash = str(_scalar(saved, "initial_path_sha256"))
            if initial_hash != str(contract["common_initial_path_sha256"]):
                raise ValueError(f"{method} uses a different initial path")

            eligibility = str(record["eligibility"])
            if eligibility == "initial_or_reparam_committed":
                if "reparam_committed" not in saved.files:
                    raise ValueError("IVM source has no reparam_committed history")
                committed = np.asarray(saved["reparam_committed"], dtype=bool)
                eligible = committed.copy()
                eligible[0] = True
            elif eligibility in {
                "all_saved_states",
                "explicit_saved_states",
            }:
                eligible = np.ones(len(paths), dtype=bool)
            else:
                raise ValueError(f"unsupported eligibility rule {eligibility!r}")
            indices = resolve_record_indices(
                contract_path=contract_path,
                record=record,
                calls_per_image=calls,
                eligible=eligible,
            )

            for index in indices:
                index = int(index)
                selected_path = np.asarray(paths[index], dtype=np.float64)
                rows.append(
                    {
                        "method": method,
                        "source_result": str(result),
                        "source_result_sha256": actual_result_hash,
                        "source_record_index": index,
                        "calls_per_image": int(calls[index]),
                        "paper_total_nf": int(total_nf[index]),
                        "i_perp": float(i_perp[index]),
                        "counter_vector": np.asarray(counters[index], dtype=np.int64),
                        "path": selected_path,
                        "path_sha256": sha256_array(selected_path),
                        "selection_role": str(record["selection_role"]),
                        "eligibility": eligibility,
                    }
                )
    return contract, rows


def _evaluate_chunk(topology_path: str, chunk: Sequence[Tuple[int, np.ndarray]]) -> List[Dict]:
    from chignolin_core import (  # pylint: disable=import-outside-toplevel
        EV_TO_KCAL,
        OpenMMBackend,
    )

    backend = OpenMMBackend(
        Path(topology_path),
        platform_preference=("CPU",),
        precision="double",
        label="chignolin_arc_length_500",
    )
    output = []
    for row_index, path in chunk:
        dense, geometry = arc_length_resample(path, N_DENSE_POINTS)
        before = int(backend.raw_joint_evaluations)
        energies = np.asarray(
            [backend.evaluate_joint(point)[0] for point in dense],
            dtype=np.float64,
        )
        raw_calls = int(backend.raw_joint_evaluations) - before
        if raw_calls != N_DENSE_POINTS:
            raise RuntimeError("a 500-point diagnostic did not make exactly 500 fresh evaluations")
        output.append(
            {
                "row_index": int(row_index),
                "barrier_500_kcal": float(
                    (np.max(energies) - energies[0]) * EV_TO_KCAL
                ),
                "diagnostic_joint_evaluations": raw_calls,
                **geometry,
            }
        )
    return output


def split_chunks(rows: Sequence[Dict], workers: int) -> List[List[Tuple[int, np.ndarray]]]:
    chunks: List[List[Tuple[int, np.ndarray]]] = [[] for _ in range(int(workers))]
    for row_index, row in enumerate(rows):
        chunks[row_index % int(workers)].append((row_index, row["path"]))
    return [chunk for chunk in chunks if chunk]


def evaluate_rows(topology: Path, rows: Sequence[Dict], workers: int) -> List[Dict]:
    chunks = split_chunks(rows, max(1, min(int(workers), len(rows))))
    if len(chunks) == 1:
        results = _evaluate_chunk(str(topology), chunks[0])
    else:
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(_evaluate_chunk, str(topology), chunk)
                for chunk in chunks
            ]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
    results.sort(key=lambda row: row["row_index"])
    if [row["row_index"] for row in results] != list(range(len(rows))):
        raise RuntimeError("diagnostic workers did not return every selected row")
    return results


def load_precomputed(path: Path, rows: Sequence[Dict]) -> List[Dict]:
    """Import independently computed values after exact row/path validation."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as saved:
        required = {
            "method",
            "source_record_index",
            "paper_total_nf",
            "i_perp",
            "barrier_500_kcal",
            "total_arc_length",
            "selected_path_sha256",
            "diagnostic_joint_evaluations",
        }
        missing = sorted(required.difference(saved.files))
        if missing:
            raise ValueError("precomputed result is missing " + ",".join(missing))
        methods = np.asarray(saved["method"]).astype(str)
        methods[methods == "IR-IVM"] = "IVM"
        records = np.asarray(saved["source_record_index"], dtype=np.int64)
        nf = np.asarray(saved["paper_total_nf"], dtype=np.int64)
        i_perp = np.asarray(saved["i_perp"], dtype=np.float64)
        barrier = np.asarray(saved["barrier_500_kcal"], dtype=np.float64)
        arc_length = np.asarray(saved["total_arc_length"], dtype=np.float64)
        path_hash = np.asarray(saved["selected_path_sha256"]).astype(str)
        calls = np.asarray(saved["diagnostic_joint_evaluations"], dtype=np.int64)
    n = len(methods)
    if not all(
        len(array) == n
        for array in (records, nf, i_perp, barrier, arc_length, path_hash, calls)
    ):
        raise ValueError("precomputed arrays differ in length")
    lookup = {}
    for index, key in enumerate(zip(methods, records)):
        normalized = (str(key[0]), int(key[1]))
        if normalized in lookup:
            raise ValueError(f"duplicate precomputed row {normalized}")
        lookup[normalized] = index
    expected = {(row["method"], row["source_record_index"]) for row in rows}
    if set(lookup) != expected:
        missing = sorted(expected.difference(lookup))
        extra = sorted(set(lookup).difference(expected))
        raise ValueError(
            f"precomputed selection mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    evaluated = []
    for row_index, row in enumerate(rows):
        source_index = lookup[(row["method"], row["source_record_index"])]
        _, geometry = arc_length_resample(row["path"], N_DENSE_POINTS)
        if (
            int(nf[source_index]) != row["paper_total_nf"]
            or not np.isclose(i_perp[source_index], row["i_perp"], rtol=0.0, atol=0.0)
            or str(path_hash[source_index]) != row["path_sha256"]
            or int(calls[source_index]) != N_DENSE_POINTS
            or not np.isclose(
                arc_length[source_index], geometry["total_arc_length"], rtol=2e-14, atol=2e-14
            )
            or not np.isfinite(barrier[source_index])
        ):
            raise ValueError(
                f"precomputed row does not match {row['method']} record "
                f"{row['source_record_index']}"
            )
        evaluated.append(
            {
                "row_index": row_index,
                "barrier_500_kcal": float(barrier[source_index]),
                "diagnostic_joint_evaluations": N_DENSE_POINTS,
                **geometry,
            }
        )
    return evaluated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--precomputed",
        type=Path,
        help=(
            "Import independently computed values after exact method, record, "
            "NoF, I_perp, path-hash, arc-length, and 500-call validation."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    contract_path = args.contract.expanduser().resolve()
    output = args.output.expanduser().resolve()
    audit_output = output.with_suffix(".audit.json")
    if output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    if not args.overwrite and (output.exists() or audit_output.exists()):
        raise FileExistsError("refusing to overwrite an existing diagnostic")

    contract, rows = load_selected_paths(contract_path)
    topology = _resolve_contract_path(contract_path, contract["topology"]["path"])
    if sha256_file(topology) != str(contract["topology"]["sha256"]):
        raise ValueError("topology hash mismatch")
    precomputed = (
        None if args.precomputed is None else args.precomputed.expanduser().resolve()
    )
    evaluated = (
        evaluate_rows(topology, rows, args.workers)
        if precomputed is None
        else load_precomputed(precomputed, rows)
    )

    for row, metric in zip(rows, evaluated):
        row.update(metric)
    atomic_save_npz(
        output,
        method=np.asarray([row["method"] for row in rows]),
        source_result=np.asarray([row["source_result"] for row in rows]),
        source_result_sha256=np.asarray([row["source_result_sha256"] for row in rows]),
        source_record_index=np.asarray([row["source_record_index"] for row in rows], dtype=np.int64),
        calls_per_image=np.asarray([row["calls_per_image"] for row in rows], dtype=np.int64),
        paper_total_nf=np.asarray([row["paper_total_nf"] for row in rows], dtype=np.int64),
        i_perp=np.asarray([row["i_perp"] for row in rows], dtype=np.float64),
        selected_counter_vectors=np.asarray([row["counter_vector"] for row in rows], dtype=np.int64),
        selected_path_sha256=np.asarray([row["path_sha256"] for row in rows]),
        selection_role=np.asarray([row["selection_role"] for row in rows]),
        eligibility=np.asarray([row["eligibility"] for row in rows]),
        barrier_500_kcal=np.asarray([row["barrier_500_kcal"] for row in rows], dtype=np.float64),
        total_arc_length=np.asarray([row["total_arc_length"] for row in rows], dtype=np.float64),
        minimum_original_segment_length=np.asarray([row["minimum_original_segment_length"] for row in rows], dtype=np.float64),
        maximum_original_segment_length=np.asarray([row["maximum_original_segment_length"] for row in rows], dtype=np.float64),
        dense_spacing=np.asarray([row["dense_spacing"] for row in rows], dtype=np.float64),
        diagnostic_joint_evaluations_per_path=np.asarray([row["diagnostic_joint_evaluations"] for row in rows], dtype=np.int64),
        coordinate_space=np.asarray(SPACE_DEFINITION),
        interpolation=np.asarray(INTERPOLATION),
        number_of_points=np.asarray(N_DENSE_POINTS),
        endpoint_policy=np.asarray(ENDPOINT_POLICY),
        grid_formula=np.asarray(GRID_FORMULA),
        optimization_cost_axis=np.asarray("counter_vectors.sum(axis=1)"),
        diagnostic_counts_excluded_from_optimization_axis=np.asarray(True),
        contract=np.asarray(str(contract_path)),
        contract_sha256=np.asarray(sha256_file(contract_path)),
        topology=np.asarray(str(topology)),
        topology_sha256=np.asarray(sha256_file(topology)),
        computation_mode=np.asarray(
            "direct_openmm" if precomputed is None else "validated_precomputed_import"
        ),
        precomputed_source=np.asarray("" if precomputed is None else str(precomputed)),
        precomputed_source_sha256=np.asarray(
            "" if precomputed is None else sha256_file(precomputed)
        ),
    )
    output_hash = sha256_file(output)
    method_counts = {
        method: int(sum(row["method"] == method for row in rows))
        for method in METHOD_ORDER
    }
    audit_rows = [
        {
            "method": row["method"],
            "source_record_index": row["source_record_index"],
            "calls_per_image": row["calls_per_image"],
            "paper_total_nf": row["paper_total_nf"],
            "i_perp": row["i_perp"],
            "selected_path_sha256": row["path_sha256"],
            "barrier_500_kcal": row["barrier_500_kcal"],
            "total_arc_length": row["total_arc_length"],
            "diagnostic_joint_evaluations": row["diagnostic_joint_evaluations"],
            "selection_role": row["selection_role"],
        }
        for row in rows
    ]
    audit = {
        "status": "validated",
        "output": str(output),
        "output_sha256": output_hash,
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "topology": str(topology),
        "topology_sha256": sha256_file(topology),
        "protocol": contract["arc_length_500_protocol"],
        "method_counts": method_counts,
        "selected_path_count": len(rows),
        "diagnostic_joint_evaluations": int(N_DENSE_POINTS * len(rows)),
        "every_selected_path_has_exactly_500_fresh_evaluations": True,
        "diagnostic_counts_excluded_from_optimization_axis": True,
        "cost_includes_all_22_images": True,
        "computation_mode": (
            "direct_openmm" if precomputed is None else "validated_precomputed_import"
        ),
        "precomputed_source": None if precomputed is None else str(precomputed),
        "precomputed_source_sha256": (
            None if precomputed is None else sha256_file(precomputed)
        ),
        "precomputed_audit": (
            None
            if precomputed is None or not precomputed.with_suffix(".json").is_file()
            else str(precomputed.with_suffix(".json"))
        ),
        "precomputed_audit_sha256": (
            None
            if precomputed is None or not precomputed.with_suffix(".json").is_file()
            else sha256_file(precomputed.with_suffix(".json"))
        ),
        "records": audit_rows,
    }
    atomic_write_json(audit_output, audit)
    print(f"Saved {output}")
    print(f"Saved {audit_output}")
    print(f"output_sha256={output_hash}")
    print(f"audit_sha256={sha256_file(audit_output)}")


if __name__ == "__main__":
    main()
