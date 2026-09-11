#!/usr/bin/env python3
"""Freeze the zero-physics 15x9 constrained-relaxation pilot coordinates.

The pilot is a deterministic subset of the call-515/NF-14983 returned-terminal
59x28 Cartesian section.  This program only reads and writes coordinates and
provenance; it never imports OpenMM and cannot make an energy/Force call.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import freeze_bb_irivm_call515_terminal_tail_0k_section_r2 as source_geometry


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
REGISTRATION_DIR = ROOT / "registrations/diagnostics"
OUTPUT_DIR = ROOT / "runs/diagnostics"

SOURCE = source_geometry.OUTPUT
SOURCE_AUDIT = source_geometry.AUDIT
SOURCE_SHA256 = "8d84d1a5e2881306e02cacfb035d16e21f8a02666911f90cc57d89dde6611bea"
SOURCE_AUDIT_SHA256 = "d8e1de581b4ae4c0ae4a742637cbee03658b464f21bf7169f3734b108a12b853"
SOURCE_PATH_SHA256 = "5a4ba3c5dc8f7063c1963f0bfd951f82356e60d0ad9c4fc17c0bc8b0d36d8569"

S_INDICES = np.asarray([0, 4, 8, 12, 17, 21, 25, 29, 33, 37, 41, 46, 50, 54, 58], dtype=np.int64)
N_INDICES = np.asarray([0, 2, 4, 8, 12, 16, 20, 24, 27], dtype=np.int64)
# Run the center column first so an interrupted pilot still has the most useful
# constrained profile.  Remaining columns expand deterministically in |n|.
N_EXECUTION_ORDER = np.asarray([2, 1, 0, 3, 4, 5, 6, 7, 8], dtype=np.int64)

CONTRACT_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-coordinate-contract-r1-v1"
CONTRACT_STATUS = "FROZEN_ZERO_PHYSICS_BEFORE_CONSTRAINED_RELAXATION"
MANIFEST_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-coordinate-manifest-r1-v1"
MANIFEST_STATUS = "PASS_ZERO_PHYSICS_FIXED_15X9_PILOT"
AUDIT_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-coordinate-audit-r1-v1"
AUDIT_STATUS = "PASS_ZERO_PHYSICS_FIXED_15X9_PILOT_AUDIT"

CONTRACT = REGISTRATION_DIR / (
    "BB_IRIVM_CALL515_TERMINAL_UNCOMMITTED_TAIL_CONSTRAINED_RELAXED_"
    "PILOT_COORDINATE_CONTRACT_R1_20260903.json"
)
OUTPUT = OUTPUT_DIR / (
    "bb_irivm_call515_nf14983_terminal_uncommitted_tail_constrained_relaxed_"
    "pilot_coordinates_15x9_20260903_r1.npz"
)
AUDIT = OUTPUT.with_suffix(".audit.json")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size}


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


def atomic_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    try:
        np.savez(temporary, **payload)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def scalar(saved: Mapping[str, np.ndarray], key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def validate_source() -> dict[str, np.ndarray]:
    source = SOURCE.resolve(strict=True)
    audit_path = SOURCE_AUDIT.resolve(strict=True)
    require(sha256_file(source) == SOURCE_SHA256, "source coordinate SHA mismatch")
    require(sha256_file(audit_path) == SOURCE_AUDIT_SHA256, "source coordinate audit SHA mismatch")
    source_geometry.validate_coordinate_manifest_for_energy(source, SOURCE_SHA256)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit.get("status") == source_geometry.AUDIT_STATUS,
            "source coordinate audit status mismatch")
    with np.load(source, allow_pickle=False) as saved:
        require(str(scalar(saved, "returned_terminal_path32_sha256")) == SOURCE_PATH_SHA256,
                "returned path SHA mismatch")
        require(int(scalar(saved, "center_source_call_per_image")) == 515
                and int(scalar(saved, "center_source_optimization_N_F")) == 14983
                and bool(scalar(saved, "terminal_uncommitted_charged_tail"))
                and not bool(scalar(saved, "formally_reportable")),
                "source state semantics mismatch")
        coordinates = np.ascontiguousarray(saved["coordinates_A"][S_INDICES][:, N_INDICES], dtype=np.float64)
        hashes = np.asarray(saved["coordinate_sha256"][S_INDICES][:, N_INDICES]).astype(str)
        tangents = np.ascontiguousarray(saved["central_tangents"][S_INDICES], dtype=np.float64)
        normals = np.ascontiguousarray(saved["normal_directions"][S_INDICES], dtype=np.float64)
        source_s = np.ascontiguousarray(saved["center_s"][S_INDICES], dtype=np.float64)
        source_n = np.ascontiguousarray(saved["normal_coordinate_rms_A"][N_INDICES], dtype=np.float64)
    require(coordinates.shape == (15, 9, source_geometry.ATOMS, 3),
            "pilot coordinate shape mismatch")
    require(tangents.shape == normals.shape == (15, source_geometry.ATOMS, 3),
            "pilot basis shape mismatch")
    flat_t = tangents.reshape(15, -1)
    flat_n = normals.reshape(15, -1)
    require(np.allclose(np.linalg.norm(flat_t, axis=1), 1.0, atol=2e-15, rtol=0.0)
            and np.allclose(np.linalg.norm(flat_n, axis=1), 1.0, atol=2e-15, rtol=0.0)
            and np.all(np.abs(np.sum(flat_t * flat_n, axis=1)) < 2e-15),
            "source tau/nu basis is not orthonormal")
    for i in range(15):
        for j in range(9):
            require(sha256_array(coordinates[i, j]) == hashes[i, j],
                    "selected coordinate hash mismatch")
    return {"coordinates": coordinates, "hashes": hashes,
            "tangents": tangents, "normals": normals,
            "s": source_s, "n": source_n}


def validate_manifest(path: Path, expected_sha: str | None = None) -> dict[str, np.ndarray]:
    path = Path(path).resolve(strict=True)
    if expected_sha is not None:
        require(sha256_file(path) == expected_sha, "pilot manifest SHA mismatch")
    with np.load(path, allow_pickle=False) as saved:
        require(str(scalar(saved, "schema")) == MANIFEST_SCHEMA
                and str(scalar(saved, "status")) == MANIFEST_STATUS,
                "pilot manifest identity mismatch")
        require(str(scalar(saved, "returned_terminal_path32_sha256")) == SOURCE_PATH_SHA256,
                "pilot path SHA mismatch")
        coordinates = np.ascontiguousarray(saved["initial_coordinates_A"], dtype=np.float64)
        hashes = np.asarray(saved["initial_coordinate_sha256"]).astype(str)
        tangents = np.ascontiguousarray(saved["fixed_tau"], dtype=np.float64)
        normals = np.ascontiguousarray(saved["fixed_nu"], dtype=np.float64)
        s_values = np.ascontiguousarray(saved["s_values"], dtype=np.float64)
        n_values = np.ascontiguousarray(saved["n_values_rms_A"], dtype=np.float64)
        execution = np.ascontiguousarray(saved["execution_grid_indices"], dtype=np.int64)
        s_indices = np.ascontiguousarray(saved["source_s_indices"], dtype=np.int64)
        n_indices = np.ascontiguousarray(saved["source_n_indices"], dtype=np.int64)
    require(coordinates.shape == (15, 9, source_geometry.ATOMS, 3)
            and hashes.shape == (15, 9)
            and tangents.shape == normals.shape == (15, source_geometry.ATOMS, 3)
            and execution.shape == (135, 2), "pilot manifest dimensions mismatch")
    require(np.array_equal(s_indices, S_INDICES) and np.array_equal(n_indices, N_INDICES),
            "pilot source index mismatch")
    expected_execution = np.asarray(
        [(i, j) for j in N_EXECUTION_ORDER for i in range(15)], dtype=np.int64)
    require(np.array_equal(execution, expected_execution)
            and len({tuple(item) for item in execution.tolist()}) == 135,
            "pilot execution order mismatch")
    require(np.array_equal(s_values, S_INDICES.astype(np.float64) / 58.0)
            and np.array_equal(n_values,
                               np.asarray([-0.12, -0.06, 0.0, 0.12, 0.24, 0.36, 0.48, 0.60, 0.69])),
            "pilot collective-coordinate values mismatch")
    for i in range(15):
        for j in range(9):
            require(sha256_array(coordinates[i, j]) == hashes[i, j],
                    "pilot coordinate hash mismatch")
    return {"coordinates": coordinates, "hashes": hashes,
            "tangents": tangents, "normals": normals,
            "s": s_values, "n": n_values, "execution": execution}


def prepare_contract() -> None:
    validate_source()
    require(not CONTRACT.exists() and not OUTPUT.exists() and not AUDIT.exists(),
            "pilot coordinate chain already exists")
    payload = {
        "schema": CONTRACT_SCHEMA, "status": CONTRACT_STATUS,
        "created_utc": utc_now(), "builder": binding(THIS_FILE),
        "source_coordinate_manifest": binding(SOURCE),
        "source_coordinate_audit": binding(SOURCE_AUDIT),
        "center_state": {
            "method": "IVM", "call_per_image": 515,
            "optimization_N_F": 14983, "path_sha256": SOURCE_PATH_SHA256,
            "state_role": "terminal_uncommitted_charged_tail",
            "formally_reportable": False,
        },
        "source_s_indices": S_INDICES.tolist(),
        "source_n_indices": N_INDICES.tolist(),
        "n_execution_order_within_pilot": N_EXECUTION_ORDER.tolist(),
        "surface_shape": [15, 9],
        "planned_coordinate_manifest": str(OUTPUT.resolve()),
        "planned_coordinate_audit": str(AUDIT.resolve()),
        "openmm_contexts_constructed": 0, "physical_force_calls": 0,
        "optimization_N_F_contribution": 0,
    }
    atomic_json(CONTRACT, payload)
    print(json.dumps({"status": CONTRACT_STATUS, "contract": binding(CONTRACT),
                      "physical_force_calls": 0}, indent=2))


def build() -> None:
    contract = json.loads(CONTRACT.resolve(strict=True).read_text(encoding="utf-8"))
    require(contract.get("schema") == CONTRACT_SCHEMA
            and contract.get("status") == CONTRACT_STATUS
            and contract.get("builder") == binding(THIS_FILE)
            and contract.get("source_coordinate_manifest") == binding(SOURCE)
            and contract.get("source_coordinate_audit") == binding(SOURCE_AUDIT),
            "pilot coordinate contract mismatch")
    require(not OUTPUT.exists() and not AUDIT.exists(), "pilot coordinate output exists")
    data = validate_source()
    execution = np.asarray([(i, j) for j in N_EXECUTION_ORDER for i in range(15)], dtype=np.int64)
    payload = {
        "schema": np.asarray(MANIFEST_SCHEMA), "status": np.asarray(MANIFEST_STATUS),
        "contract": np.asarray(str(CONTRACT.resolve())),
        "contract_sha256": np.asarray(sha256_file(CONTRACT)),
        "builder": np.asarray(str(THIS_FILE)), "builder_sha256": np.asarray(sha256_file(THIS_FILE)),
        "source_coordinate_manifest": np.asarray(str(SOURCE.resolve())),
        "source_coordinate_manifest_sha256": np.asarray(sha256_file(SOURCE)),
        "source_coordinate_audit": np.asarray(str(SOURCE_AUDIT.resolve())),
        "source_coordinate_audit_sha256": np.asarray(sha256_file(SOURCE_AUDIT)),
        "center_source_call_per_image": np.asarray(515, dtype=np.int64),
        "center_source_optimization_N_F": np.asarray(14983, dtype=np.int64),
        "returned_terminal_path32_sha256": np.asarray(SOURCE_PATH_SHA256),
        "terminal_uncommitted_charged_tail": np.asarray(True),
        "formally_reportable": np.asarray(False),
        "source_s_indices": S_INDICES, "source_n_indices": N_INDICES,
        "s_values": data["s"], "n_values_rms_A": data["n"],
        "execution_grid_indices": execution,
        "initial_coordinates_A": data["coordinates"],
        "initial_coordinate_sha256": data["hashes"],
        "fixed_tau": data["tangents"], "fixed_nu": data["normals"],
        "projector_definition": np.asarray("P(v)=v-tau*(tau dot v)-nu*(nu dot v); tau and nu frozen per s"),
        "openmm_contexts_constructed": np.asarray(0, dtype=np.int64),
        "physical_force_calls": np.asarray(0, dtype=np.int64),
        "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
    }
    atomic_npz(OUTPUT, payload)
    validate_manifest(OUTPUT, sha256_file(OUTPUT))
    audit = {
        "schema": AUDIT_SCHEMA, "status": AUDIT_STATUS, "created_utc": utc_now(),
        "contract": binding(CONTRACT), "builder": binding(THIS_FILE),
        "source_coordinate_manifest": binding(SOURCE),
        "source_coordinate_audit": binding(SOURCE_AUDIT),
        "coordinate_manifest": binding(OUTPUT),
        "center_state": contract["center_state"],
        "surface_shape": [15, 9], "stored_initial_coordinates": 135,
        "fixed_s_values": data["s"].tolist(), "fixed_n_values_rms_A": data["n"].tolist(),
        "openmm_contexts_constructed": 0, "physical_force_calls": 0,
        "optimization_N_F_contribution": 0,
    }
    atomic_json(AUDIT, audit)
    print(json.dumps({"status": AUDIT_STATUS, "manifest": binding(OUTPUT),
                      "audit": binding(AUDIT), "physical_force_calls": 0}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-contract", action="store_true")
    modes.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.prepare_contract:
        prepare_contract()
    else:
        build()


if __name__ == "__main__":
    main()
