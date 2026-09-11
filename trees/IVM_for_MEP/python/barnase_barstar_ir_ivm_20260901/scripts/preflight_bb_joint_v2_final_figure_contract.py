#!/usr/bin/env python3
"""Read-only, zero-OpenMM preflight for the final joint-E/F v2 figure contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


N_IMAGES = 32
N_INTERNAL = 30
DIMENSION = 9396
N_DENSE = 500
METHODS = ("IVM", "ODE-NEB")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def uniform_cartesian_arc500(path: np.ndarray) -> np.ndarray:
    curve = np.asarray(path, dtype=np.float64)
    require(curve.shape == (N_IMAGES, DIMENSION), "selected path shape is invalid for arc500 reconstruction")
    lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    require(np.all(np.isfinite(lengths)) and np.all(lengths > 0.0), "selected path has an invalid segment")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, float(cumulative[-1]), N_DENSE)
    index = np.clip(np.searchsorted(cumulative, targets, side="right") - 1, 0, len(lengths) - 1)
    fraction = (targets - cumulative[index]) / lengths[index]
    dense = (1.0 - fraction[:, None]) * curve[index] + fraction[:, None] * curve[index + 1]
    dense[0] = curve[0]
    dense[-1] = curve[-1]
    return np.ascontiguousarray(dense)


def discrete_iperp(path: np.ndarray, internal_forces: np.ndarray) -> float:
    curve = np.asarray(path, dtype=np.float64)
    forces = np.asarray(internal_forces, dtype=np.float64)
    require(curve.shape == (N_IMAGES, DIMENSION) and forces.shape == (N_INTERNAL, DIMENSION), "selected path/internal force shape mismatch")
    chord = curve[2:] - curve[:-2]
    chord_norm = np.linalg.norm(chord, axis=1, keepdims=True)
    require(np.all(chord_norm > 1.0e-30), "selected path has a zero central chord")
    tangent = chord / chord_norm
    force_perp = forces - np.sum(forces * tangent, axis=1, keepdims=True) * tangent
    segment_lengths = np.linalg.norm(curve[1:] - curve[:-1], axis=1)
    weights = 0.5 * (segment_lengths[:-1] + segment_lengths[1:])
    per_image_l2 = np.linalg.norm(force_perp, axis=1)
    return float(np.sum(weights * per_image_l2 ** 2))


def bound_path(record: dict, role: str) -> Path:
    require(isinstance(record, dict) and set(record) >= {"path", "sha256"}, f"{role} binding missing")
    path = Path(record["path"]).expanduser().resolve(strict=True)
    require(sha256_file(path) == str(record["sha256"]), f"{role} SHA mismatch")
    return path


def validate_recursive_bindings(value, role: str = "figure contract") -> int:
    count = 0
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            bound_path(value, role)
            count += 1
        for key, child in value.items():
            count += validate_recursive_bindings(child, f"{role}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            count += validate_recursive_bindings(child, f"{role}[{index}]")
    return count


def scalar(saved, key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def read_wal_events(path: Path):
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid WAL JSON at {path}:{line_number}") from error
        require(isinstance(event, dict), f"WAL event is not an object at {path}:{line_number}")
        events.append(event)
    require(events, f"empty WAL: {path}")
    return events


def wal_lifetime_stats(events):
    starts = [event for event in events if event.get("event") == "attempt_started"]
    successes = [event for event in events if event.get("event") == "attempt_succeeded"]
    failures = [event for event in events if event.get("event") == "attempt_failed"]
    start_ids = {str(event["attempt_id"]) for event in starts}
    terminal_id_list = [str(event["attempt_id"]) for event in successes + failures]
    terminal_ids = set(terminal_id_list)
    require(len(start_ids) == len(starts), "duplicate WAL attempt_id")
    require(len(terminal_ids) == len(terminal_id_list), "WAL attempt_id has multiple terminal events")
    require(all(str(event["attempt_id"]) in start_ids for event in successes + failures), "WAL terminal without start")
    require(not ({str(event["attempt_id"]) for event in successes} & {str(event["attempt_id"]) for event in failures}), "WAL attempt has conflicting terminals")
    logical = {(int(event["row_index"]), str(event["stage"]), int(event["point_index"])) for event in starts}
    return {
        "attempt_started": len(starts), "attempt_succeeded": len(successes),
        "attempt_failed": len(failures), "attempt_open": len(start_ids - terminal_ids),
        "retry_attempts": len(starts) - len(logical),
    }


def validate_stage_wal_closure(events, row_index: int, row: dict, stage: str, coordinates, energies, forces):
    starts_at = [index for index, event in enumerate(events) if event.get("event") == "path_stage_started" and int(event.get("row_index", -1)) == row_index and event.get("stage", stage) == stage]
    completes_at = [index for index, event in enumerate(events) if event.get("event") == "path_stage_data_complete_uncommitted" and int(event.get("row_index", -1)) == row_index and event.get("stage") == stage]
    require(starts_at and completes_at, f"{row['label']} lacks a complete {stage} WAL stage")
    complete_index = completes_at[-1]
    start_candidates = [index for index in starts_at if index < complete_index]
    require(start_candidates, f"{row['label']} {stage} complete has no start")
    segment = events[start_candidates[-1]:complete_index + 1]
    starts = [event for event in segment if event.get("event") == "attempt_started"]
    successes = [event for event in segment if event.get("event") == "attempt_succeeded"]
    failures = [event for event in segment if event.get("event") == "attempt_failed"]
    require(len(starts) == len(successes) == len(coordinates) and not failures, f"{row['label']} {stage} accepted WAL stage is incomplete")
    start_by_id = {str(event["attempt_id"]): event for event in starts}
    require(len(start_by_id) == len(starts), f"{row['label']} {stage} duplicate start ids")
    success_by_point = {}
    for event in successes:
        attempt_id = str(event["attempt_id"])
        require(attempt_id in start_by_id, f"{row['label']} {stage} success without start")
        started = start_by_id[attempt_id]
        for key in ("stage", "row_index", "label", "optimization_N_F", "path_sha256", "point_index", "coordinate_sha256", "session_id"):
            require(event.get(key) == started.get(key), f"{row['label']} {stage} start/success {key} mismatch")
        point = int(event["point_index"])
        require(point not in success_by_point, f"{row['label']} {stage} duplicate point success")
        success_by_point[point] = event
    expected_points = list(range(1, N_INTERNAL + 1)) if stage == "original_discrete_internal" else list(range(N_DENSE))
    require(sorted(success_by_point) == expected_points, f"{row['label']} {stage} point coverage mismatch")
    for array_index, point in enumerate(expected_points):
        event = success_by_point[point]
        coordinate = np.asarray(coordinates[array_index])
        energy = float(energies[array_index])
        force = np.asarray(forces[array_index])
        require(event.get("coordinate_sha256") == sha256_array(coordinate), f"{row['label']} {stage} coordinate WAL/shard mismatch at {point}")
        require(float(event["energy_eV"]) == energy, f"{row['label']} {stage} energy WAL/shard mismatch at {point}")
        require(event.get("energy_scalar_sha256") == sha256_array(np.asarray(energy, dtype=np.float64)), f"{row['label']} {stage} energy hash mismatch at {point}")
        require(event.get("force_sha256") == sha256_array(force), f"{row['label']} {stage} force WAL/shard mismatch at {point}")
    return {"accepted_stage_attempts": len(starts), "accepted_stage_successes": len(successes), "accepted_stage_failures": 0, "accepted_stage_open": 0, "pointwise_WAL_shard_closure": "PASS"}


def validate_online_wal(rows, events):
    matched = []
    for row_index, row in enumerate(rows):
        candidates = [event for event in events if event.get("event") == "verified_online_paid_full_state_reused" and int(event.get("row_index", -1)) == row_index and event.get("path_sha256") == row["path_sha256"]]
        require(candidates, f"{row['label']} lacks online reuse WAL record")
        event = candidates[-1]
        require(event.get("online_energy_sha256") == row["online_energy_sha256"] and event.get("online_force_sha256") == row["online_force_sha256"], f"{row['label']} online reuse WAL hash mismatch")
        require(event.get("source_sha256") == row["online_full_state_source"]["sha256"] and int(event.get("new_physical_calls", -1)) == 0, f"{row['label']} online reuse WAL provenance mismatch")
        matched.append({"label": row["label"], "status": "PASS", "new_physical_calls": 0})
    return matched


def validate_method(method: str, closure: dict):
    result = bound_path(closure["result"], f"{method} result")
    audit_path = bound_path(closure["audit"], f"{method} audit")
    contract_path = bound_path(closure["contract"], f"{method} metric contract")
    manifest_path = bound_path(closure["manifest"], f"{method} manifest")
    shard_paths = [bound_path(record, f"{method} shard {index}") for index, record in enumerate(closure["authoritative_shards"])]
    wal_paths = [bound_path(record, f"{method} WAL {index}") for index, record in enumerate(closure["WALs"])]
    require(len(shard_paths) == 4 and len(wal_paths) == 2, f"{method} closure count mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_contract_schema = "bb-dense-four-ivm-joint-ef-contract-v2" if method == "IVM" else "bb-dense-four-path-joint-ef-contract-v2"
    require(contract.get("schema") == expected_contract_schema and contract.get("frozen") is True, f"{method} metric contract is not frozen v2")
    require(contract.get("method") == method and int(contract.get("jobs", -1)) == 1, f"{method} metric contract method/jobs mismatch")
    require(contract.get("physical_call_policy", {}).get("dense_joint_EF_per_path") == N_DENSE, f"{method} dense joint-call policy mismatch")
    require(contract.get("physical_call_policy", {}).get("dense_full_force_storage") is True, f"{method} dense force storage policy missing")
    require(contract.get("physical_call_policy", {}).get("optimization_N_F_contribution") == 0, f"{method} metric contract contaminates optimization N_F")
    require(audit.get("status") == "PASS" and audit.get("method") == method, f"{method} audit is not PASS")
    require(audit.get("output_sha256") == sha256_file(result) and audit.get("contract_sha256") == sha256_file(contract_path), f"{method} audit result/contract mismatch")
    require(int(audit.get("optimization_N_F_contribution", -1)) == 0 and int(audit.get("individual_incomplete_WAL_attempts_reused", -1)) == 0, f"{method} audit accounting mismatch")
    require(audit.get("shard_manifest_sha256") == sha256_file(manifest_path), f"{method} audit/manifest mismatch")
    require(manifest.get("status") == "PASS" and manifest.get("aggregate_sha256") == sha256_file(result), f"{method} manifest mismatch")
    require(len(manifest.get("shards", [])) == 4, f"{method} manifest shard count mismatch")
    wal_fields = ("online_paid_state_wal", "dense_joint_ef_wal") if method == "IVM" else ("discrete_joint_ef_wal", "dense_joint_ef_wal")
    for index, (wal_path, field) in enumerate(zip(wal_paths, wal_fields)):
        require(Path(str(audit.get(field, ""))).expanduser().resolve() == wal_path, f"{method} audit/{field} path mismatch")
        require(audit.get(field + "_sha256") == sha256_file(wal_path), f"{method} audit/{field} SHA mismatch")
    with np.load(result, allow_pickle=False) as saved:
        require(int(scalar(saved, "number_of_points")) == N_DENSE and int(scalar(saved, "optimization_N_F_contribution")) == 0, f"{method} aggregate protocol mismatch")
        require("dense_coordinates_A" not in saved.files and "dense_force_eV_per_A" not in saved.files, f"{method} aggregate duplicates dense coordinates/forces")
        aggregate_shards = [Path(value).expanduser().resolve(strict=True) for value in np.asarray(saved["shard_path"]).astype(str)]
        aggregate_shas = np.asarray(saved["shard_sha256"]).astype(str)
        aggregate_path_sha = np.asarray(saved["selected_path_sha256"]).astype(str)
        aggregate_dense_sha = np.asarray(saved["dense_coordinate_sha256"]).astype(str)
        aggregate_energy_sha = np.asarray(saved["energy_sha256"]).astype(str)
        aggregate_force_sha = np.asarray(saved["dense_force_sha256"]).astype(str)
        aggregate_delta = np.asarray(saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64)
        aggregate_i_perp = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
    require(aggregate_shards == shard_paths, f"{method} aggregate/direct shard paths mismatch")
    reports = []
    loaded_shards = []
    for index, shard_path in enumerate(shard_paths):
        require(sha256_file(shard_path) == aggregate_shas[index], f"{method} aggregate shard SHA mismatch")
        with np.load(shard_path, allow_pickle=False) as shard:
            required = {"selected_path", "selected_path_sha256", "internal_joint_energy_eV", "internal_diagnostic_force_eV_per_A", "discrete_I_perp", "dense_coordinates_A", "dense_energy_eV", "dense_force_eV_per_A", "dense_coordinate_sha256", "dense_energy_sha256", "dense_force_sha256", "densified_Delta_E_kcal_per_mol", "optimization_N_F_contribution"}
            require(required.issubset(shard.files), f"{method} shard {index} fields missing")
            selected = np.asarray(shard["selected_path"], dtype=np.float64)
            dense = np.asarray(shard["dense_coordinates_A"], dtype=np.float64)
            energy = np.asarray(shard["dense_energy_eV"], dtype=np.float64)
            force = np.asarray(shard["dense_force_eV_per_A"], dtype=np.float64)
            require(selected.shape == (N_IMAGES, DIMENSION), f"{method} shard {index} selected-path shape mismatch")
            require(dense.shape == force.shape == (N_DENSE, DIMENSION) and energy.shape == (N_DENSE,), f"{method} shard {index} dense shape mismatch")
            require(np.all(np.isfinite(selected)) and np.all(np.isfinite(dense)) and np.all(np.isfinite(energy)) and np.all(np.isfinite(force)), f"{method} shard {index} non-finite")
            path_sha = sha256_array(selected)
            dense_sha = sha256_array(dense)
            energy_sha = sha256_array(energy)
            force_sha = sha256_array(force)
            require(path_sha == str(scalar(shard, "selected_path_sha256")) == aggregate_path_sha[index], f"{method} shard {index} path hash closure failed")
            require(dense_sha == str(scalar(shard, "dense_coordinate_sha256")) == aggregate_dense_sha[index], f"{method} shard {index} dense-coordinate hash closure failed")
            require(energy_sha == str(scalar(shard, "dense_energy_sha256")) == aggregate_energy_sha[index], f"{method} shard {index} dense-energy hash closure failed")
            require(force_sha == str(scalar(shard, "dense_force_sha256")) == aggregate_force_sha[index], f"{method} shard {index} dense-force hash closure failed")
            rebuilt_dense = uniform_cartesian_arc500(selected)
            require(rebuilt_dense.dtype.str == dense.dtype.str and rebuilt_dense.shape == dense.shape and rebuilt_dense.tobytes(order="C") == dense.tobytes(order="C"), f"{method} shard {index} dense coordinates differ from deterministic Cartesian arc500")
            measured_i_perp = discrete_iperp(selected, np.asarray(shard["internal_diagnostic_force_eV_per_A"], dtype=np.float64))
            require(np.asarray(measured_i_perp).tobytes() == np.asarray(float(scalar(shard, "discrete_I_perp"))).tobytes() == np.asarray(aggregate_i_perp[index]).tobytes(), f"{method} shard {index} discrete I_perp reconstruction mismatch")
            require(np.array_equal(dense[0], selected[0]) and np.array_equal(dense[-1], selected[-1]), f"{method} shard {index} endpoint closure failed")
            delta = float((np.max(energy) - energy[0]) * 23.0605)
            require(np.asarray(delta).tobytes() == np.asarray(float(scalar(shard, "densified_Delta_E_kcal_per_mol"))).tobytes() == np.asarray(aggregate_delta[index]).tobytes(), f"{method} shard {index} Delta E closure failed")
            require(int(scalar(shard, "optimization_N_F_contribution")) == 0, f"{method} shard {index} contaminates optimization N_F")
            loaded_shards.append({key: np.asarray(shard[key]).copy() for key in shard.files})
        record = manifest["shards"][index]
        require(Path(record["path"]).expanduser().resolve() == shard_path and record["sha256"] == aggregate_shas[index], f"{method} manifest shard binding mismatch")
        require(record.get("dense_coordinate_sha256") == dense_sha and record.get("dense_energy_sha256") == energy_sha and record.get("dense_force_sha256") == force_sha, f"{method} manifest array hashes mismatch")
        reports.append({"index": index, "path_sha256": path_sha, "dense_coordinate_sha256": dense_sha, "dense_energy_sha256": energy_sha, "dense_force_sha256": force_sha, "endpoints_included": True, "optimization_N_F_contribution": 0})
    rows = contract.get("paths", [])
    require(len(rows) == 4, f"{method} contract path count mismatch")
    first_events = read_wal_events(wal_paths[0])
    dense_events = read_wal_events(wal_paths[1])
    if method == "ODE-NEB":
        recomputed_paths = []
        for row_index, (row, shard) in enumerate(zip(rows, loaded_shards)):
            discrete = validate_stage_wal_closure(first_events, row_index, row, "original_discrete_internal", shard["selected_path"][1:-1], shard["internal_joint_energy_eV"], shard["internal_diagnostic_force_eV_per_A"])
            dense_report = validate_stage_wal_closure(dense_events, row_index, row, "dense_arc500_joint_EF", shard["dense_coordinates_A"], shard["dense_energy_eV"], shard["dense_force_eV_per_A"])
            recomputed_paths.append({"label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "discrete": discrete, "dense": dense_report})
        recomputed = {"status": "PASS", "discrete_lifetime": wal_lifetime_stats(first_events), "dense_lifetime": wal_lifetime_stats(dense_events), "paths": recomputed_paths}
        require(recomputed == audit.get("WAL_shard_pointwise_closure"), f"{method} recomputed WAL/shard closure differs from audit")
    else:
        online_recomputed = validate_online_wal(rows, first_events)
        require(online_recomputed == audit.get("online_source_closure"), "IVM recomputed online-state WAL closure differs from audit")
        dense_paths = []
        for row_index, (row, shard) in enumerate(zip(rows, loaded_shards)):
            dense_paths.append({"label": row["label"], **validate_stage_wal_closure(dense_events, row_index, row, "dense_arc500_joint_EF", shard["dense_coordinates_A"], shard["dense_energy_eV"], shard["dense_force_eV_per_A"])})
        recomputed = {"status": "PASS", "lifetime": wal_lifetime_stats(dense_events), "paths": dense_paths}
        require(recomputed == audit.get("dense_WAL_shard_pointwise_closure"), "IVM recomputed dense WAL/shard closure differs from audit")
    return {"result": str(result), "contract": str(contract_path), "audit": str(audit_path), "manifest": str(manifest_path), "WALs": [str(path) for path in wal_paths], "paths": reports}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    contract_path = args.contract.expanduser().resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(contract.get("schema") == "bb-densified-cost-to-precision-figure-contract-v1" and contract.get("status") == "FROZEN_DENSIFIED_INPUT_BINDINGS", "final figure contract is not frozen")
    protocol = contract.get("metric_protocol", {})
    require(protocol.get("arc500_stored_arrays_per_path") == ["dense_coordinates_A", "dense_energy_eV", "dense_force_eV_per_A"], "figure contract lacks joint-E/F v2 arrays")
    require(protocol.get("authoritative_data_unit") == "complete atomic per-path shard" and int(protocol.get("optimization_N_F_contribution", -1)) == 0, "figure contract storage/NoF policy mismatch")
    binding_count = validate_recursive_bindings(contract)
    closure = contract.get("bindings", {}).get("joint_EF_v2_direct_data_closure", {})
    require(set(closure) == set(METHODS), "figure contract lacks two-method v2 closure")
    reports = {method: validate_method(method, closure[method]) for method in METHODS}
    output = Path(contract["figure_output"]).expanduser().resolve()
    require(not output.exists() and not output.with_suffix(".json").exists(), "preflight refuses an existing final figure/provenance")
    print(json.dumps({"status": "ZERO_OPENMM_FINAL_FIGURE_CONTRACT_PREFLIGHT_PASS", "contract": str(contract_path), "contract_sha256": sha256_file(contract_path), "recursive_file_bindings_verified": binding_count, "figure_output": str(output), "methods": reports, "physical_calls": 0}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
