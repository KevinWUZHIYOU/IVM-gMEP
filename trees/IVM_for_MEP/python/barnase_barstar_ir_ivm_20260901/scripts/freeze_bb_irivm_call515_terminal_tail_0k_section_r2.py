#!/usr/bin/env python3
"""Freeze a diagnostic 59x28 0 K section around returned R5B call 515.

The center is deliberately *not* promoted to a committed IVM state.  It is the
fully evaluated path returned at the end of the NF15000-budget run: call 515,
optimization NF14983, inside the terminal uncommitted charged tail.  This
coordinate-only stage imports no OpenMM and contributes zero optimization NoF.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONTRACT_SCHEMA = "bb-irivm-call515-terminal-tail-section-coordinate-contract-r2-v1"
CONTRACT_STATUS = "FROZEN_BEFORE_ANY_CALL515_SECTION_ENERGY_CALL"
MANIFEST_SCHEMA = "bb-irivm-call515-terminal-tail-section-coordinate-manifest-r2-v1"
MANIFEST_STATUS = "PASS_ZERO_OPENMM_RETURNED_TERMINAL_COORDINATE_FREEZE"
AUDIT_SCHEMA = "bb-irivm-call515-terminal-tail-section-coordinate-audit-r2-v1"
AUDIT_STATUS = "PASS_ZERO_OPENMM_RETURNED_TERMINAL_COORDINATE_AUDIT"
POINTS = 59
NORMAL_POINTS = 28
N_IMAGES = 32
ATOMS = 3132
DIMENSION = 3 * ATOMS
CALL = 515
NF = 14983
ZERO_INDEX = 4
N_GRID = np.arange(-0.12, 0.6901, 0.03, dtype=np.float64)
N_GRID[ZERO_INDEX] = 0.0

REPLAY_ROOT = ROOT / "runs/full_state_history_replay_r5b"
RESULT = REPLAY_ROOT / (
    "ivm_bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b_central_state_"
    "ivm_cfgf5e79e830f27_budget15000_storagelegacy.npz"
)
SHARD = REPLAY_ROOT / (
    "bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b_observer/shards/"
    "call_0515_stable_last.npz"
)
VERIFICATION = ROOT / (
    "reports/verifications/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF15000_20260902_R5B_PRIMARY.json"
)
REGISTRATION_DIR = ROOT / "registrations/diagnostics"
OUTPUT_DIR = ROOT / "runs/diagnostics"
CONTRACT = REGISTRATION_DIR / (
    "BB_IRIVM_CALL515_TERMINAL_UNCOMMITTED_TAIL_0K_SECTION_COORDINATE_"
    "CONTRACT_R2_20260903.json"
)
OUTPUT = OUTPUT_DIR / (
    "bb_irivm_call515_nf14983_terminal_uncommitted_tail_0k_section_"
    "coordinates_59x28_20260903_r2.npz"
)
AUDIT = OUTPUT.with_suffix(".audit.json")
LOCKS = tuple(sorted((
    Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock"),
    Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock"),
), key=str))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size}


def scalar(saved: Mapping[str, Any], key: str) -> Any:
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **payload)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_binding(record: Mapping[str, Any], role: str) -> Path:
    require(isinstance(record, Mapping) and {"path", "sha256"} <= set(record),
            f"{role} binding missing")
    path = Path(str(record["path"])).expanduser().resolve(strict=True)
    require(sha256_file(path) == str(record["sha256"]), f"{role} SHA-256 mismatch")
    return path


def probe_locks() -> list[str]:
    streams = []
    try:
        for path in LOCKS:
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a+")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                stream.close()
                raise RuntimeError(f"another BB task owns {path}") from error
            streams.append(stream)
        return [str(path) for path in LOCKS]
    finally:
        for stream in reversed(streams):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


def validate_sources(result_path: Path, shard_path: Path,
                     verification_path: Path) -> dict[str, np.ndarray]:
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    require(verification.get("status") ==
            "PASS_R5B_EXACT_PREFIX318_AND_DYNAMIC_STABLE_LAST_FULL_STATE_REPLAY",
            "R5B primary verification status mismatch")
    require(int(verification.get("terminal_optimization_N_F", -1)) == NF,
            "R5B terminal optimization NF mismatch")
    rows = [row for row in verification.get("path_inventory", [])
            if int(row.get("call_per_image", -1)) == CALL]
    require(len(rows) == 1, "verification lacks unique call515")
    row = rows[0]
    require(int(row.get("optimization_N_F", -1)) == NF
            and row.get("formally_reportable") is False
            and row.get("terminal_uncommitted_charged_tail") is True,
            "call515 is not the declared terminal uncommitted-tail state")
    require(Path(str(row.get("shard", ""))).resolve() == shard_path
            and row.get("shard_sha256") == sha256_file(shard_path),
            "call515 shard binding mismatch")
    with np.load(shard_path, allow_pickle=False) as saved:
        require(int(scalar(saved, "call_per_image")) == CALL
                and int(scalar(saved, "optimization_N_F")) == NF,
                "call515 shard counters mismatch")
        require(bool(scalar(saved, "fully_evaluated_current_path")),
                "call515 path is not fully evaluated")
        path = np.ascontiguousarray(saved["path32_A"], dtype=np.float64)
        energy = np.ascontiguousarray(saved["energy_eV"], dtype=np.float64)
        force = np.ascontiguousarray(saved["force_eV_per_A"], dtype=np.float64)
    require(path.shape == force.shape == (N_IMAGES, DIMENSION)
            and energy.shape == (N_IMAGES,), "call515 full-state shape mismatch")
    require(sha256_array(path) == row["path_sha256"]
            and sha256_array(energy) == row["energy_sha256"]
            and sha256_array(force) == row["force_sha256"],
            "call515 array hash mismatch")
    with np.load(result_path, allow_pickle=False) as saved:
        final_path = np.ascontiguousarray(saved["final_path"], dtype=np.float64)
        initial_path = np.ascontiguousarray(saved["initial_path"], dtype=np.float64)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
        stored_paths = np.asarray(saved["stored_paths"], dtype=np.float64)
        require(int(scalar(saved, "total_internal_nf_consumed")) == NF
                and int(scalar(saved, "total_internal_nf_last_complete_state_nf")) == 8982
                and bool(scalar(saved, "total_internal_nf_terminal_partial_tail_quarantined")),
                "R5B terminal-tail metadata mismatch")
        require(np.asarray(saved["reparam_committed"], dtype=bool)[-1] == np.bool_(False),
                "call515 unexpectedly committed")
    require(final_path.shape == initial_path.shape == (N_IMAGES, DIMENSION)
            and stored_calls[-1] == CALL and np.array_equal(stored_paths[-1], path)
            and np.array_equal(final_path, path),
            "result.final_path/stored call515/shard are not byte-identical")
    return {"path": path.reshape(N_IMAGES, ATOMS, 3),
            "energy": energy, "force": force,
            "initial": initial_path.reshape(N_IMAGES, ATOMS, 3)}


def arc(path: np.ndarray) -> np.ndarray:
    flat = np.asarray(path, dtype=np.float64).reshape(len(path), -1)
    lengths = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    require(np.all(np.isfinite(lengths)) and np.all(lengths > 0.0),
            "invalid path segment")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return cumulative / cumulative[-1]


def resample(path: np.ndarray, target: np.ndarray) -> np.ndarray:
    flat = path.reshape(len(path), -1)
    source = arc(path)
    result = np.stack([np.interp(target, source, flat[:, column])
                       for column in range(flat.shape[1])], axis=1)
    result = result.reshape(len(target), ATOMS, 3)
    result[0], result[-1] = path[0], path[-1]
    return np.ascontiguousarray(result, dtype=np.float64)


def basis(center: np.ndarray, initial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = center.reshape(POINTS, -1)
    tangent = np.empty_like(flat)
    tangent[1:-1] = flat[2:] - flat[:-2]
    tangent[0], tangent[-1] = flat[1] - flat[0], flat[-1] - flat[-2]
    norm = np.linalg.norm(tangent, axis=1)
    require(np.all(norm > 1.0e-14), "degenerate center tangent")
    tangent /= norm[:, None]
    displacement = initial.reshape(POINTS, -1) - flat
    displacement -= np.sum(displacement * tangent, axis=1)[:, None] * tangent
    size = np.linalg.norm(displacement, axis=1)
    valid = size > 1.0e-12
    require(np.count_nonzero(valid) >= 3, "insufficient normal anchors")
    normal = np.zeros_like(displacement)
    normal[valid] = displacement[valid] / size[valid, None]
    anchors = np.flatnonzero(valid)
    for index in np.flatnonzero(~valid):
        candidate = normal[anchors[np.argmin(np.abs(anchors - index))]].copy()
        candidate -= np.dot(candidate, tangent[index]) * tangent[index]
        candidate_norm = np.linalg.norm(candidate)
        require(candidate_norm > 1.0e-14, "degenerate borrowed normal")
        normal[index] = candidate / candidate_norm
    return tangent.reshape(center.shape), normal.reshape(center.shape)


def validate_coordinate_manifest_for_energy(path: Path,
                                            expected_sha: str | None = None) -> dict[str, Any]:
    if expected_sha is not None:
        require(sha256_file(path) == expected_sha, "coordinate manifest hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        require(str(scalar(saved, "schema")) == MANIFEST_SCHEMA
                and str(scalar(saved, "status")) == MANIFEST_STATUS,
                "coordinate manifest identity mismatch")
        require(int(scalar(saved, "center_source_call_per_image")) == CALL
                and int(scalar(saved, "center_source_optimization_N_F")) == NF
                and bool(scalar(saved, "terminal_uncommitted_charged_tail"))
                and not bool(scalar(saved, "formally_reportable")),
                "coordinate manifest state semantics mismatch")
        coordinates = np.ascontiguousarray(saved["coordinates_A"], dtype=np.float64)
        hashes = np.asarray(saved["coordinate_sha256"]).astype(str)
        center = np.ascontiguousarray(saved["center_path59_A"], dtype=np.float64)
        n_grid = np.ascontiguousarray(saved["normal_coordinate_rms_A"], dtype=np.float64)
        path32 = np.ascontiguousarray(saved["returned_terminal_path32_A"], dtype=np.float64)
        initial = np.ascontiguousarray(saved["common_initial_path59_A"], dtype=np.float64)
        normals = np.ascontiguousarray(saved["normal_directions"], dtype=np.float64)
        center_s = np.ascontiguousarray(saved["center_s"], dtype=np.float64)
        contract = Path(str(scalar(saved, "freeze_contract"))).resolve(strict=True)
        builder = Path(str(scalar(saved, "builder"))).resolve(strict=True)
        result = Path(str(scalar(saved, "source_result"))).resolve(strict=True)
        shard = Path(str(scalar(saved, "source_shard"))).resolve(strict=True)
        declared = {
            "contract": str(scalar(saved, "freeze_contract_sha256")),
            "builder": str(scalar(saved, "builder_sha256")),
            "result": str(scalar(saved, "source_result_sha256")),
            "shard": str(scalar(saved, "source_shard_sha256")),
            "center": str(scalar(saved, "center_path59_sha256")),
            "path": str(scalar(saved, "returned_terminal_path32_sha256")),
        }
    require(coordinates.shape == (POINTS, NORMAL_POINTS, ATOMS, 3)
            and hashes.shape == (POINTS, NORMAL_POINTS)
            and center.shape == initial.shape == normals.shape == (POINTS, ATOMS, 3)
            and path32.shape == (N_IMAGES, ATOMS, 3)
            and center_s.shape == (POINTS,) and np.array_equal(n_grid, N_GRID),
            "coordinate manifest dimensions mismatch")
    require(np.array_equal(coordinates[:, ZERO_INDEX], center),
            "n=0 surface column differs from center")
    require(sha256_file(contract) == declared["contract"]
            and sha256_file(builder) == declared["builder"]
            and sha256_file(result) == declared["result"]
            and sha256_file(shard) == declared["shard"]
            and sha256_array(center) == declared["center"]
            and sha256_array(path32) == declared["path"],
            "coordinate manifest source hash mismatch")
    for index in range(POINTS):
        for normal_index in range(NORMAL_POINTS):
            require(sha256_array(coordinates[index, normal_index]) ==
                    hashes[index, normal_index], "coordinate cell hash mismatch")
    return {
        "path": path, "path_sha256": sha256_file(path),
        "surface_coordinates": coordinates, "cell_hashes": hashes,
        "center": center, "n_grid": n_grid, "original_path32": path32,
        "original_path32_sha256": sha256_array(path32),
        "common_initial_path59_A": initial, "normal_directions": normals,
        "center_s": center_s, "source_result": result, "source_shard": shard,
        "freeze_contract": contract, "builder": builder,
    }


def prepare_contract() -> None:
    result, shard, verification = (path.resolve(strict=True)
                                   for path in (RESULT, SHARD, VERIFICATION))
    validate_sources(result, shard, verification)
    require(not OUTPUT.exists() and not AUDIT.exists(),
            "coordinate output already exists")
    locks = probe_locks()
    payload = {
        "schema": CONTRACT_SCHEMA, "status": CONTRACT_STATUS,
        "created_utc": utc_now(), "builder": binding(Path(__file__)),
        "source_result": binding(result), "source_shard": binding(shard),
        "primary_verification": binding(verification),
        "state_semantics": {
            "center_source_call_per_image": CALL,
            "center_source_optimization_N_F": NF,
            "returned_terminal_state": True,
            "fully_evaluated_current_path": True,
            "formally_reportable": False,
            "terminal_uncommitted_charged_tail": True,
            "must_not_be_described_as_committed_IVM": True,
        },
        "geometry": {"center_points": POINTS, "normal_points": NORMAL_POINTS,
                     "normal_start_A": -0.12, "normal_stop_A": 0.69,
                     "normal_step_A": 0.03, "zero_normal_index": ZERO_INDEX},
        "planned_coordinate_output": str(OUTPUT.resolve()),
        "planned_coordinate_audit": str(AUDIT.resolve()),
        "lock_probe": {"status": "PASS", "locks": locks},
        "physical_execution_authorized": False,
        "physical_calls_started": 0, "force_calls": 0,
        "optimization_N_F_contribution": 0,
    }
    atomic_json(CONTRACT, payload)
    print(json.dumps({"status": CONTRACT_STATUS, "contract": binding(CONTRACT),
                      "physical_calls_started": 0}, indent=2))


def build_coordinates(contract_path: Path, output: Path) -> None:
    contract_path = contract_path.resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(contract.get("schema") == CONTRACT_SCHEMA
            and contract.get("status") == CONTRACT_STATUS,
            "coordinate contract identity mismatch")
    require(resolve_binding(contract["builder"], "builder") == Path(__file__).resolve(),
            "coordinate builder binding mismatch")
    require(output.resolve() == Path(contract["planned_coordinate_output"]).resolve()
            and not output.exists() and not AUDIT.exists(),
            "coordinate output path/existence mismatch")
    result = resolve_binding(contract["source_result"], "source result")
    shard = resolve_binding(contract["source_shard"], "source shard")
    verification = resolve_binding(contract["primary_verification"], "verification")
    source = validate_sources(result, shard, verification)
    target = np.linspace(0.0, 1.0, POINTS, dtype=np.float64)
    center = resample(source["path"], target)
    initial = resample(source["initial"], target)
    tangents, normals = basis(center, initial)
    coordinates = center[:, None] + (
        N_GRID[None, :, None, None] * np.sqrt(float(ATOMS)) * normals[:, None]
    )
    coordinates[:, ZERO_INDEX] = center
    coordinates = np.ascontiguousarray(coordinates, dtype=np.float64)
    hashes = np.asarray([[sha256_array(coordinates[i, j])
                          for j in range(NORMAL_POINTS)] for i in range(POINTS)])
    payload = {
        "schema": np.asarray(MANIFEST_SCHEMA), "status": np.asarray(MANIFEST_STATUS),
        "freeze_contract": np.asarray(str(contract_path)),
        "freeze_contract_sha256": np.asarray(sha256_file(contract_path)),
        "builder": np.asarray(str(Path(__file__).resolve())),
        "builder_sha256": np.asarray(sha256_file(Path(__file__).resolve())),
        "source_result": np.asarray(str(result)),
        "source_result_sha256": np.asarray(sha256_file(result)),
        "source_shard": np.asarray(str(shard)),
        "source_shard_sha256": np.asarray(sha256_file(shard)),
        "primary_verification": np.asarray(str(verification)),
        "primary_verification_sha256": np.asarray(sha256_file(verification)),
        "center_source_call_per_image": np.asarray(CALL, dtype=np.int64),
        "center_source_optimization_N_F": np.asarray(NF, dtype=np.int64),
        "returned_terminal_state": np.asarray(True),
        "fully_evaluated_current_path": np.asarray(True),
        "formally_reportable": np.asarray(False),
        "terminal_uncommitted_charged_tail": np.asarray(True),
        "returned_terminal_path32_A": source["path"],
        "returned_terminal_path32_sha256": np.asarray(sha256_array(source["path"])),
        "returned_terminal_energy32_eV": source["energy"],
        "returned_terminal_energy32_sha256": np.asarray(sha256_array(source["energy"])),
        "center_path59_A": center,
        "center_path59_sha256": np.asarray(sha256_array(center)),
        "center_s": target, "common_initial_path59_A": initial,
        "central_tangents": tangents, "normal_directions": normals,
        "normal_coordinate_rms_A": N_GRID,
        "zero_normal_index": np.asarray(ZERO_INDEX, dtype=np.int64),
        "coordinates_A": coordinates, "coordinate_sha256": hashes,
        "coordinate_space": np.asarray("Cartesian Angstrom"),
        "surface_kind": np.asarray(
            "unrelaxed returned-terminal-call515-centered single-normal 0 K diagnostic section"
        ),
        "surface_energy_values": np.asarray(POINTS * NORMAL_POINTS, dtype=np.int64),
        "fresh_surface_energy_only_calls_planned": np.asarray(
            POINTS * NORMAL_POINTS - 2, dtype=np.int64),
        "inherited_fixed_endpoint_energy_values": np.asarray(2, dtype=np.int64),
        "required_force_evaluations": np.asarray(0, dtype=np.int64),
        "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
    }
    atomic_npz(output, payload)
    validate_coordinate_manifest_for_energy(output, sha256_file(output))
    audit_payload = {
        "schema": AUDIT_SCHEMA, "status": AUDIT_STATUS, "created_utc": utc_now(),
        "coordinate_contract": binding(contract_path),
        "coordinate_manifest": binding(output), "builder": binding(Path(__file__)),
        "source_result": binding(result), "source_shard": binding(shard),
        "primary_verification": binding(verification),
        "center_source_call_per_image": CALL,
        "center_source_optimization_N_F": NF,
        "state_role": "returned terminal state; terminal uncommitted charged tail; not committed IVM",
        "surface_shape": [POINTS, NORMAL_POINTS],
        "stored_coordinate_values": POINTS * NORMAL_POINTS,
        "force_calls": 0, "physical_calls_started": 0,
        "optimization_N_F_contribution": 0,
    }
    atomic_json(AUDIT, audit_payload)
    print(json.dumps({"status": AUDIT_STATUS, "manifest": binding(output),
                      "audit": binding(AUDIT), "shape": [POINTS, NORMAL_POINTS],
                      "physical_calls_started": 0}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-contract", action="store_true")
    mode.add_argument("--build", action="store_true")
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.prepare_contract:
        prepare_contract()
    else:
        build_coordinates(args.contract.expanduser(), args.output.expanduser())


if __name__ == "__main__":
    main()
