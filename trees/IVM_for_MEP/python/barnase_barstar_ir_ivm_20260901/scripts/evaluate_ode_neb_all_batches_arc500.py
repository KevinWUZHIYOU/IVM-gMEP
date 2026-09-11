#!/usr/bin/env python3
"""Evaluate one resumable shard of all paid ODE--NEB paths at arc500."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_discrete_iperp_densified_barrier import (
    COORDINATE_SPACE,
    GRID_FORMULA,
    INTERPOLATION,
    N_DENSE_POINTS,
    N_IMAGES,
    DIMENSION,
    SAMPLING,
    acquire_host_locks,
    atomic_save_npz,
    atomic_write_json,
    evaluate_paths,
    load_exact_reuse_rows,
    sha256_array,
    sha256_file,
)


SCHEMA = "bb-ode-neb-all-paid-current-paths-arc500-contract-v2"


def official_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    metadata = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_reuse_rows(diagnostics: Path, audit_path: Path, topology_sha256: str) -> dict:
    """Accept the current schema or the hash-complete four-row ODE preview schema."""
    with np.load(diagnostics, allow_pickle=False) as saved:
        current_schema = "diagnostic_energy_successes_per_path" in saved.files
    if current_schema:
        return load_exact_reuse_rows(diagnostics, audit_path, topology_sha256)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit.get("schema") == "bb-ode-neb-arc500-preview-audit-v1", "legacy reuse audit schema mismatch")
    require(audit.get("status") == "PASS", "legacy reuse audit did not pass")
    require(audit.get("output_sha256") == sha256_file(diagnostics), "legacy reuse output hash mismatch")
    require(int(audit.get("path_count", -1)) == 4, "legacy reuse path count mismatch")
    require(int(audit.get("diagnostic_energy_attempts", -1)) == 2000, "legacy reuse attempt count mismatch")
    require(int(audit.get("diagnostic_energy_successes", -1)) == 2000, "legacy reuse success count mismatch")
    require(int(audit.get("diagnostic_force_requests", -1)) == 0, "legacy reuse requested a force")
    wal_path = Path(audit["energy_wal"]).resolve(strict=True)
    require(audit.get("energy_wal_sha256") == sha256_file(wal_path), "legacy reuse WAL hash mismatch")
    with np.load(diagnostics, allow_pickle=False) as saved:
        require(str(np.asarray(saved["topology_sha256"]).item()) == topology_sha256, "legacy reuse topology mismatch")
        require(str(np.asarray(saved["coordinate_space"]).item()) == COORDINATE_SPACE, "legacy reuse coordinate-space mismatch")
        require(str(np.asarray(saved["interpolation"]).item()) == INTERPOLATION, "legacy reuse interpolation mismatch")
        require(str(np.asarray(saved["sampling"]).item()) == SAMPLING, "legacy reuse sampling mismatch")
        require(int(np.asarray(saved["number_of_points"]).item()) == N_DENSE_POINTS, "legacy reuse point-count mismatch")
        require(str(np.asarray(saved["grid_formula"]).item()) == GRID_FORMULA, "legacy reuse grid mismatch")
        path_hashes = np.asarray(saved["selected_path_sha256"]).astype(str)
        dense_hashes = np.asarray(saved["dense_coordinate_sha256"]).astype(str)
        energies = np.asarray(saved["energy_eV"], dtype=np.float64)
        energy_hashes = np.asarray(saved["energy_sha256"]).astype(str)
        attempts = np.asarray(saved["diagnostic_energy_attempts_per_path"], dtype=np.int64)
        forces = np.asarray(saved["diagnostic_force_requests_per_path"], dtype=np.int64)
    require(energies.shape == (4, N_DENSE_POINTS), "legacy reuse energy array shape mismatch")
    require(np.all(attempts == N_DENSE_POINTS), "legacy reuse lacks 500 attempts per path")
    require(np.all(forces == 0), "legacy reuse contains a force request")
    result = {}
    source_sha = sha256_file(diagnostics)
    for index, path_hash in enumerate(path_hashes):
        require(sha256_array(energies[index]) == energy_hashes[index], "legacy reuse energy hash mismatch")
        result[str(path_hash)] = {
            "dense_coordinate_sha256": str(dense_hashes[index]),
            "energy_eV": energies[index].copy(),
            "energy_sha256": str(energy_hashes[index]),
            "source_diagnostic": str(diagnostics),
            "source_diagnostic_sha256": source_sha,
            "source_row_index": index,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument(
        "--global-lock-path",
        type=Path,
        default=Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock"),
    )
    parser.add_argument(
        "--peer-global-lock-path",
        type=Path,
        default=Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock"),
    )
    args = parser.parse_args()
    require(args.shard_count > 0, "shard count must be positive")
    require(0 <= args.shard_index < args.shard_count, "invalid shard index")

    contract_path = args.contract.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    audit_output = output.with_suffix(".audit.json")
    wal_output = output.with_suffix(".energy.wal.jsonl")
    if any(path.exists() for path in (output, audit_output, wal_output)):
        raise FileExistsError("refusing to overwrite an all-batch diagnostic shard")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(contract.get("schema") == SCHEMA, "unsupported all-batch contract")
    require(
        contract.get("status") == "FROZEN_BEFORE_ANY_NEW_ALL_BATCH_ARC500_ENERGY_CALL",
        "all-batch contract was not frozen prospectively",
    )
    protocol = contract["diagnostic_protocol"]
    expected_protocol = {
        "coordinate_space": COORDINATE_SPACE,
        "interpolation": INTERPOLATION,
        "sampling": SAMPLING,
        "number_of_points": N_DENSE_POINTS,
        "include_both_endpoints": True,
        "grid_formula": GRID_FORMULA,
        "energy_evaluations_per_path": N_DENSE_POINTS,
        "force_evaluations_per_path": 0,
        "optimization_N_F_contribution": 0,
    }
    for key, expected in expected_protocol.items():
        require(protocol.get(key) == expected, f"diagnostic protocol mismatch: {key}")
    bindings = contract["bindings"]
    for role, record in bindings.items():
        path = Path(record["path"]).expanduser().resolve(strict=True)
        require(sha256_file(path) == record["sha256"], f"binding mismatch: {role}")
    topology = Path(bindings["topology"]["path"]).resolve(strict=True)
    reuse_path = Path(bindings["reuse_diagnostics"]["path"]).resolve(strict=True)
    reuse_audit = Path(bindings["reuse_audit"]["path"]).resolve(strict=True)
    reuse_rows = load_reuse_rows(
        reuse_path, reuse_audit, str(bindings["topology"]["sha256"])
    )

    records = contract["paths"]
    require(len(records) == 333, "contract does not contain all 333 paid paths")
    unique_official = []
    for record in records:
        value = str(record["official_path32_sha256"])
        if value not in unique_official:
            unique_official.append(value)
    assigned = {
        value
        for index, value in enumerate(unique_official)
        if index % args.shard_count == args.shard_index
    }
    selected_records = [
        record
        for record in records
        if str(record["official_path32_sha256"]) in assigned
    ]
    require(bool(selected_records), "diagnostic shard has no assigned paths")

    # Evaluate each distinct retained path once.  Rows at rejected paid trials
    # are expanded below by exact path-hash reuse of the preceding current path.
    selected = []
    selected_hashes = set()
    for record in selected_records:
        value = str(record["official_path32_sha256"])
        if value not in selected_hashes:
            selected.append(record)
            selected_hashes.add(value)

    rows = []
    for record in selected:
        archive = Path(record["coordinate_archive"]).resolve(strict=True)
        require(
            sha256_file(archive) == record["coordinate_archive_sha256"],
            f"source batch {record['source_batch_id']} coordinate archive hash mismatch",
        )
        with np.load(archive, allow_pickle=False) as saved:
            coordinates = np.asarray(saved["coordinates30_A"], dtype=np.float64)
            endpoint0 = np.asarray(saved["endpoint0_A"], dtype=np.float64)
            endpoint1 = np.asarray(saved["endpoint1_A"], dtype=np.float64)
            embedded = str(np.asarray(saved["path32_sha256"]).item())
        path3 = np.concatenate((endpoint0[None], coordinates, endpoint1[None]), axis=0)
        require(path3.shape == (N_IMAGES, DIMENSION // 3, 3), "path shape mismatch")
        official = official_array_sha256(path3)
        require(
            official == embedded == record["official_path32_sha256"],
            f"source batch {record['source_batch_id']} official path identity mismatch",
        )
        path = np.ascontiguousarray(path3.reshape(N_IMAGES, DIMENSION))
        rows.append(
            {
                "label": f"ODE-NEB_current_source_batch_{int(record['source_batch_id']):04d}",
                "batch_id": int(record["source_batch_id"]),
                "optimization_N_F": int(record["optimization_N_F"]),
                "state_role": str(record["source_state_role"]),
                "path": path,
                "path_sha256": sha256_array(path),
                "official_path32_sha256": official,
                "source_result": str(archive),
                "source_result_sha256": str(record["coordinate_archive_sha256"]),
            }
        )

    with acquire_host_locks(
        (args.global_lock_path, args.peer_global_lock_path)
    ) as held_locks:
        evaluated = evaluate_paths(topology, rows, wal_output, reuse_rows)
    fresh_attempts = sum(
        int(row["diagnostic_energy_attempts_this_evaluation"]) for row in evaluated
    )
    evaluated_by_hash = {
        str(row["official_path32_sha256"]): row for row in evaluated
    }
    expanded = []
    first_occurrence = set()
    for record in selected_records:
        path_hash = str(record["official_path32_sha256"])
        row = dict(evaluated_by_hash[path_hash])
        is_duplicate = path_hash in first_occurrence
        first_occurrence.add(path_hash)
        row["batch_id"] = int(record["paid_batch_id"])
        row["source_batch_id"] = int(record["source_batch_id"])
        row["optimization_N_F"] = int(record["optimization_N_F"])
        row["disposition"] = str(record["disposition"])
        row["state_role"] = str(record["source_state_role"])
        if is_duplicate:
            row["diagnostic_energy_attempts_this_evaluation"] = 0
            row["diagnostic_energy_reused"] = True
        expanded.append(row)
    reused_count = sum(bool(row["diagnostic_energy_reused"]) for row in expanded)
    atomic_save_npz(
        output,
        batch_id=np.asarray([row["batch_id"] for row in expanded], dtype=np.int64),
        current_path_source_batch_id=np.asarray(
            [row["source_batch_id"] for row in expanded], dtype=np.int64
        ),
        optimization_N_F=np.asarray(
            [row["optimization_N_F"] for row in expanded], dtype=np.int64
        ),
        disposition=np.asarray([row["disposition"] for row in expanded]),
        state_role=np.asarray([row["state_role"] for row in expanded]),
        densified_Delta_E_kcal_per_mol=np.asarray(
            [row["densified_Delta_E_kcal_per_mol"] for row in expanded],
            dtype=np.float64,
        ),
        energy_eV=np.asarray([row["energy_eV"] for row in expanded]),
        selected_path_sha256=np.asarray([row["path_sha256"] for row in expanded]),
        official_path32_sha256=np.asarray(
            [row["official_path32_sha256"] for row in expanded]
        ),
        dense_coordinate_sha256=np.asarray(
            [row["dense_coordinate_sha256"] for row in expanded]
        ),
        energy_sha256=np.asarray([row["energy_sha256"] for row in expanded]),
        total_arclength_A=np.asarray(
            [row["total_arclength_A"] for row in expanded], dtype=np.float64
        ),
        dense_spacing_A=np.asarray(
            [row["dense_spacing_A"] for row in expanded], dtype=np.float64
        ),
        diagnostic_energy_attempts_per_path=np.full(
            len(expanded), N_DENSE_POINTS, dtype=np.int64
        ),
        diagnostic_energy_successes_per_path=np.full(
            len(expanded), N_DENSE_POINTS, dtype=np.int64
        ),
        diagnostic_energy_attempts_this_evaluation=np.asarray(
            [row["diagnostic_energy_attempts_this_evaluation"] for row in expanded],
            dtype=np.int64,
        ),
        diagnostic_energy_reused=np.asarray(
            [row["diagnostic_energy_reused"] for row in expanded], dtype=bool
        ),
        diagnostic_force_requests_per_path=np.zeros(len(expanded), dtype=np.int64),
        optimization_N_F_contribution=np.asarray(0, dtype=np.int64),
        coordinate_space=np.asarray(COORDINATE_SPACE),
        interpolation=np.asarray(INTERPOLATION),
        sampling=np.asarray(SAMPLING),
        number_of_points=np.asarray(N_DENSE_POINTS, dtype=np.int64),
        grid_formula=np.asarray(GRID_FORMULA),
        contract=np.asarray(str(contract_path)),
        contract_sha256=np.asarray(sha256_file(contract_path)),
        topology=np.asarray(str(topology)),
        topology_sha256=np.asarray(sha256_file(topology)),
        energy_wal=np.asarray(str(wal_output)),
        held_host_locks=np.asarray(held_locks),
        shard_index=np.asarray(args.shard_index, dtype=np.int64),
        shard_count=np.asarray(args.shard_count, dtype=np.int64),
    )
    audit = {
        "schema": "bb-ode-neb-all-paid-batches-arc500-shard-audit-v1",
        "status": "PASS",
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "energy_wal": str(wal_output),
        "energy_wal_sha256": sha256_file(wal_output),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "paid_N_F_state_count": len(expanded),
        "distinct_retained_path_count": len(evaluated),
        "fresh_diagnostic_energy_attempts": fresh_attempts,
        "fresh_diagnostic_energy_successes": fresh_attempts,
        "reused_path_count": reused_count,
        "lineage_energy_evaluations": N_DENSE_POINTS * len(expanded),
        "diagnostic_force_requests": 0,
        "optimization_N_F_contribution": 0,
        "every_included_path_has_exactly_500_successful_energy_evaluations": True,
        "held_host_locks": held_locks,
        "paid_batch_ids": [int(row["batch_id"]) for row in expanded],
        "current_path_source_batch_ids": [int(row["source_batch_id"]) for row in expanded],
    }
    atomic_write_json(audit_output, audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
