#!/usr/bin/env python3
"""Run the hash-bound 15x9 constrained-relaxation pilot for BB.

Each grid point is relaxed independently in the frozen linear subspace
orthogonal to the call-515 tangent ``tau`` and section normal ``nu``.  Every
physical request is one joint OpenMM energy/Force call and is durably recorded
before the next request.  These diagnostic Force calls never contribute to the
IR-IVM optimization ``N_F``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np

import freeze_bb_irivm_call515_constrained_relaxed_pilot_r1b as geometry


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
SRC = ROOT / "src"
REGISTRATION_DIR = ROOT / "registrations/diagnostics"
OUTPUT_DIR = ROOT / "runs/diagnostics"
TOPOLOGY = ROOT / "inputs/complex_fixed.pdb"
ENDPOINT_CACHE = ROOT / "inputs/endpoint_joint_ef_cache.npz"
ENDPOINT_CACHE_SHA256 = "5cd4e757c99f0784d8b0781d05d8083070c8cfaad2670ead26999b3a4134e99a"
COORDINATE = geometry.base.OUTPUT
COORDINATE_AUDIT = geometry.base.AUDIT
COORDINATE_SHA256 = "1e0acc34ff629d3bb3c5eb7aa633d6af72eeaf8289aa98205acb8882caa754ec"
COORDINATE_AUDIT_SHA256 = "305ca970e862c331aa0f502f592c27d28d8a7d73f0c214c8e21b4a8a3f62fe16"

CONTRACT_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-run-contract-r1-v1"
CONTRACT_STATUS = "FROZEN_BEFORE_ANY_CONSTRAINED_PILOT_FORCE_CALL"
PREAUTH_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-preflight-r1-v1"
PREAUTH_STATUS = "PASS_ZERO_OPENMM_BEFORE_CONSTRAINED_PILOT_AUTHORIZATION"
AUTH_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-authorization-r1-v1"
AUTH_STATUS = "AUTHORIZED_AFTER_ZERO_OPENMM_PREFLIGHT"
WAL_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-wal-r1-v1"
ATTEMPT_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-attempt-r1-v1"
POINT_AUDIT_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-point-audit-r1-v1"
RESULT_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-result-r1-v1"
AUDIT_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-audit-r1-v1"

CONTRACT = REGISTRATION_DIR / (
    "BB_IRIVM_CALL515_TERMINAL_UNCOMMITTED_TAIL_CONSTRAINED_RELAXED_"
    "PILOT_RUN_CONTRACT_R1_20260903.json"
)
PREAUTH = REGISTRATION_DIR / (
    "BB_IRIVM_CALL515_TERMINAL_UNCOMMITTED_TAIL_CONSTRAINED_RELAXED_"
    "PILOT_PREAUTH_R1_20260903.json"
)
AUTHORIZATION = REGISTRATION_DIR / (
    "BB_IRIVM_CALL515_TERMINAL_UNCOMMITTED_TAIL_CONSTRAINED_RELAXED_"
    "PILOT_AUTHORIZATION_R1_20260903.json"
)
RUN_DIR = OUTPUT_DIR / (
    "bb_irivm_call515_nf14983_terminal_uncommitted_tail_constrained_relaxed_"
    "pilot_15x9_20260903_r1"
)
ATTEMPT_DIR = RUN_DIR / "attempts"
POINT_AUDIT_DIR = RUN_DIR / "point_audits"
WAL = RUN_DIR / "force.wal.jsonl"
RESULT = RUN_DIR / "result.npz"
AUDIT = RUN_DIR / "audit.json"

LOCKS = tuple(sorted(geometry.base.source_geometry.LOCKS, key=str))
THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENMM_CPU_THREADS",
)
N_ATOMS = 3132
GRID_SHAPE = (15, 9)
POINTS = 135
FMAX_EV_A = 0.10
MAX_UPDATE_STEPS = 50
MAX_FORCE_CALLS_PER_POINT = MAX_UPDATE_STEPS + 1
FIXED_ENDPOINT_ANCHORS = 2
FRESH_POINTS = POINTS - FIXED_ENDPOINT_ANCHORS
MAX_FORCE_CALLS = FRESH_POINTS * MAX_FORCE_CALLS_PER_POINT
MAX_CENTER_COLUMN_FORCE_CALLS = (GRID_SHAPE[0] - FIXED_ENDPOINT_ANCHORS) * MAX_FORCE_CALLS_PER_POINT
DT0 = 0.10
DTMAX = 1.00
FINC = 1.10
FDEC = 0.50
ALPHA0 = 0.10
FALPHA = 0.99
NMIN = 5
MAX_ATOM_STEP_A = 0.10
MINIMUM_FREE_BYTES = 12 * 1024**3
RUNTIME_FREE_BYTES_RESERVE = 8 * 1024**3
ESTIMATED_HISTORY_BYTES = 3_200_000_000


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


def resolve_binding(record: Mapping[str, Any], role: str) -> Path:
    require(isinstance(record, Mapping) and set(record) >= {"path", "sha256", "bytes"},
            f"{role} binding incomplete")
    path = Path(str(record["path"])).expanduser().resolve(strict=True)
    require(binding(path) == dict(record), f"{role} binding mismatch")
    return path


def fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        fsync_parent(path)
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
        fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_npz_payload(path: Path, expected: Mapping[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(expected), "published result field inventory mismatch")
        for key, wanted in expected.items():
            actual = np.asarray(saved[key])
            target = np.asarray(wanted)
            require(actual.dtype == target.dtype and np.array_equal(actual, target),
                    f"published result field mismatch: {key}")


def scalar(saved: Mapping[str, np.ndarray], key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def check_thread_caps() -> None:
    wrong = {name: os.environ.get(name) for name in THREAD_VARS
             if os.environ.get(name) != "1"}
    require(not wrong, f"all six thread caps must equal 1: {wrong}")


@contextmanager
def acquire_locks(blocking: bool = False):
    streams = []
    try:
        for path in LOCKS:
            stream = path.open("a+")
            mode = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(stream.fileno(), mode)
            except BlockingIOError as error:
                stream.close()
                raise RuntimeError(f"another BB task owns {path}") from error
            streams.append(stream)
        yield [str(path) for path in LOCKS]
    finally:
        for stream in reversed(streams):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


def validate_coordinates() -> dict[str, np.ndarray]:
    require(sha256_file(COORDINATE) == COORDINATE_SHA256,
            "pilot coordinate manifest SHA mismatch")
    require(sha256_file(COORDINATE_AUDIT) == COORDINATE_AUDIT_SHA256,
            "pilot coordinate audit SHA mismatch")
    data = geometry.base.validate_manifest(COORDINATE, COORDINATE_SHA256)
    audit = json.loads(COORDINATE_AUDIT.read_text(encoding="utf-8"))
    require(audit.get("status") == geometry.base.AUDIT_STATUS
            and audit.get("coordinate_manifest") == binding(COORDINATE),
            "pilot coordinate audit mismatch")
    return data


def project(value: np.ndarray, tau: np.ndarray, nu: np.ndarray) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    t = np.asarray(tau, dtype=np.float64).reshape(-1)
    n = np.asarray(nu, dtype=np.float64).reshape(-1)
    result = flat - np.dot(flat, t) * t - np.dot(flat, n) * n
    return np.ascontiguousarray(result.reshape(N_ATOMS, 3), dtype=np.float64)


def projected_fmax(force: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(np.asarray(force), axis=1)))


def constraint_drift(position: np.ndarray, origin: np.ndarray,
                     tau: np.ndarray, nu: np.ndarray) -> tuple[float, float]:
    displacement = (np.asarray(position, dtype=np.float64)
                    - np.asarray(origin, dtype=np.float64)).reshape(-1)
    tau_drift = abs(float(np.dot(displacement, np.asarray(tau).reshape(-1))))
    nu_drift = abs(float(np.dot(displacement, np.asarray(nu).reshape(-1))))
    return tau_drift, nu_drift


def validate_endpoint_cache(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    require(sha256_file(ENDPOINT_CACHE) == ENDPOINT_CACHE_SHA256,
            "canonical endpoint joint E/F cache SHA mismatch")
    with np.load(ENDPOINT_CACHE, allow_pickle=False) as saved:
        coordinates = np.ascontiguousarray(saved["coordinates_A"], dtype=np.float64).reshape(2, N_ATOMS, 3)
        energies = np.ascontiguousarray(saved["energies_eV"], dtype=np.float64)
        forces = np.ascontiguousarray(saved["forces_eV_per_A"], dtype=np.float64).reshape(2, N_ATOMS, 3)
    require(coordinates.shape == forces.shape == (2, N_ATOMS, 3)
            and energies.shape == (2,) and np.all(np.isfinite(coordinates))
            and np.all(np.isfinite(energies)) and np.all(np.isfinite(forces)),
            "canonical endpoint cache dimensions/non-finite value mismatch")
    require(np.array_equal(coordinates[0], data["coordinates"][0, 2])
            and np.array_equal(coordinates[1], data["coordinates"][-1, 2]),
            "fixed pilot anchors are not byte-identical to endpoint cache")
    return {"coordinates": coordinates, "energies": energies, "forces": forces}


def fire_advance(position: np.ndarray, origin: np.ndarray,
                 projected_force: np.ndarray, tau: np.ndarray, nu: np.ndarray,
                 velocity: np.ndarray, dt: float, alpha: float,
                 n_positive: int, first_step: bool) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    force = project(projected_force, tau, nu)
    v = project(velocity, tau, nu)
    if not first_step:
        power = float(np.vdot(force, v))
        if power > 0.0:
            force_norm = float(np.linalg.norm(force))
            velocity_norm = float(np.linalg.norm(v))
            if force_norm > 0.0 and velocity_norm > 0.0:
                v = (1.0 - alpha) * v + alpha * force * (velocity_norm / force_norm)
            if n_positive > NMIN:
                dt = min(dt * FINC, DTMAX)
                alpha *= FALPHA
            n_positive += 1
        else:
            v = np.zeros_like(v)
            alpha = ALPHA0
            dt *= FDEC
            n_positive = 0
    v = project(v + dt * force, tau, nu)
    displacement = project(dt * v, tau, nu)
    maximum = float(np.max(np.linalg.norm(displacement, axis=1)))
    if maximum > MAX_ATOM_STEP_A:
        displacement *= MAX_ATOM_STEP_A / maximum
    displacement = project(displacement, tau, nu)
    candidate = position + displacement
    candidate = origin + project(candidate - origin, tau, nu)
    require(np.all(np.isfinite(candidate)) and np.all(np.isfinite(v)),
            "FIRE produced a non-finite state")
    tau_drift, nu_drift = constraint_drift(candidate, origin, tau, nu)
    require(tau_drift <= 1e-10 and nu_drift <= 1e-10,
            "FIRE candidate escaped fixed s/n subspace")
    return candidate, v, float(dt), float(alpha), int(n_positive)


class DurableWal:
    def __init__(self, path: Path, header: Mapping[str, Any]):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            require(lines and json.loads(lines[0]) == dict(header),
                    "existing constrained-pilot WAL header mismatch")
            self.stream = path.open("a", encoding="utf-8", buffering=1)
        else:
            self.stream = path.open("x", encoding="utf-8", buffering=1)
            self.append(header)
            fsync_parent(path)

    def append(self, record: Mapping[str, Any]) -> None:
        self.stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


def wal_header(contract: Path, authorization: Path) -> dict[str, Any]:
    return {
        "schema": WAL_SCHEMA, "status": "OPEN_APPEND_ONLY_FSYNC",
        "contract": binding(contract), "authorization": binding(authorization),
        "coordinate_manifest": binding(COORDINATE),
        "coordinate_audit": binding(COORDINATE_AUDIT),
        "endpoint_joint_ef_cache": binding(ENDPOINT_CACHE),
        "grid_shape": list(GRID_SHAPE), "independent_points": POINTS,
        "fresh_relaxed_points": FRESH_POINTS,
        "fixed_cached_endpoint_anchors": FIXED_ENDPOINT_ANCHORS,
        "maximum_force_calls_per_point": MAX_FORCE_CALLS_PER_POINT,
        "maximum_fresh_center_column_force_calls": MAX_CENTER_COLUMN_FORCE_CALLS,
        "maximum_physical_force_attempts": MAX_FORCE_CALLS,
        "physical_call_kind": "joint OpenMM getState(getEnergy=True,getForces=True)",
        "optimization_N_F_contribution": 0,
    }


def read_wal(path: Path, header: Mapping[str, Any], *, validate_shards: bool = True) -> dict[str, Any]:
    if not path.exists():
        return {"starts": {}, "successes": {}, "failures": {},
                "point_completions": {}, "publications": []}
    lines = path.read_text(encoding="utf-8").splitlines()
    require(lines and json.loads(lines[0]) == dict(header), "WAL header mismatch")
    starts: dict[str, dict] = {}
    successes: dict[str, dict] = {}
    failures: dict[str, dict] = {}
    completions: dict[int, dict] = {}
    publications = []
    for raw in lines[1:]:
        event = json.loads(raw)
        kind = event.get("event")
        if kind == "attempt_started":
            attempt_id = str(event["attempt_id"])
            require(attempt_id not in starts, "duplicate attempt_started")
            starts[attempt_id] = event
        elif kind in ("attempt_succeeded", "attempt_failed"):
            attempt_id = str(event["attempt_id"])
            require(attempt_id in starts, "terminal event lacks start")
            target = successes if kind == "attempt_succeeded" else failures
            require(attempt_id not in successes and attempt_id not in failures,
                    "duplicate attempt terminal")
            target[attempt_id] = event
            if kind == "attempt_succeeded" and validate_shards:
                shard_path = resolve_binding(event["attempt_shard"], "attempt shard")
                shard = load_attempt(shard_path)
                require(str(scalar(shard, "attempt_id")) == attempt_id
                        and int(scalar(shard, "point_id")) == int(event["point_id"])
                        and int(scalar(shard, "optimizer_step")) == int(event["optimizer_step"])
                        and str(scalar(shard, "coordinate_sha256")) == event["coordinate_sha256"]
                        and str(scalar(shard, "energy_sha256")) == event["energy_sha256"]
                        and str(scalar(shard, "raw_force_sha256")) == event["raw_force_sha256"]
                        and str(scalar(shard, "projected_force_sha256")) == event["projected_force_sha256"]
                        and str(scalar(shard, "next_coordinate_sha256")) == event["next_coordinate_sha256"],
                        "attempt shard/WAL semantic hash mismatch")
        elif kind == "point_completed":
            point_id = int(event["point_id"])
            require(point_id not in completions, "duplicate point completion")
            if validate_shards:
                resolve_binding(event["point_audit"], "point audit")
            completions[point_id] = event
        elif kind == "result_published":
            publications.append(event)
        else:
            raise RuntimeError(f"unknown WAL event: {kind}")
    require(len(starts) <= MAX_FORCE_CALLS, "Force-call authorization exceeded")
    require(not failures, "failed Force attempt requires superseding authorization")
    require(set(starts) == set(successes),
            "open Force attempt requires superseding authorization")
    by_point: dict[int, list[dict]] = {}
    for event in successes.values():
        by_point.setdefault(int(event["point_id"]), []).append(event)
    for events in by_point.values():
        events.sort(key=lambda item: int(item["optimizer_step"]))
        for previous, current in zip(events[:-1], events[1:]):
            current_start = starts[str(current["attempt_id"])]
            require(current_start.get("coordinate_sha256")
                    == previous.get("next_coordinate_sha256"),
                    "resumed/continued coordinate differs from exact prior next state")
    return {"starts": starts, "successes": successes, "failures": failures,
            "point_completions": completions, "publications": publications}


def attempt_path(point_id: int, i: int, j: int, step: int) -> Path:
    return ATTEMPT_DIR / f"point_{point_id:03d}_s{i:02d}_n{j:02d}" / f"step_{step:03d}.npz"


def point_audit_path(point_id: int, i: int, j: int) -> Path:
    return POINT_AUDIT_DIR / f"point_{point_id:03d}_s{i:02d}_n{j:02d}.json"


def load_attempt(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        require(str(scalar(saved, "schema")) == ATTEMPT_SCHEMA
                and str(scalar(saved, "status")) == "PASS_PHYSICAL_JOINT_EF",
                "attempt shard identity mismatch")
        payload = {key: np.asarray(saved[key]) for key in saved.files}
    for array_key, hash_key in (
        ("coordinate_A", "coordinate_sha256"),
        ("raw_force_eV_A", "raw_force_sha256"),
        ("projected_force_eV_A", "projected_force_sha256"),
        ("next_coordinate_A", "next_coordinate_sha256"),
    ):
        require(sha256_array(payload[array_key]) == str(np.asarray(payload[hash_key]).item()),
                f"attempt shard {array_key} internal hash mismatch")
    update = np.ascontiguousarray(payload["next_coordinate_A"] - payload["coordinate_A"],
                                  dtype=np.float64)
    require(sha256_array(update) == str(np.asarray(payload["update_displacement_sha256"]).item()),
            "attempt shard derived update hash mismatch")
    require(sha256_array(np.asarray(payload["energy_eV"]))
            == str(np.asarray(payload["energy_sha256"]).item()),
            "attempt shard energy internal hash mismatch")
    return payload


def point_successes(state: Mapping[str, Any], point_id: int) -> list[dict]:
    values = [event for event in state["successes"].values()
              if int(event["point_id"]) == point_id]
    values.sort(key=lambda item: int(item["optimizer_step"]))
    require([int(item["optimizer_step"]) for item in values] == list(range(len(values))),
            "point attempt steps are not contiguous from zero")
    require(len(values) <= MAX_FORCE_CALLS_PER_POINT, "point Force-call cap exceeded")
    return values


def make_point_audit(point_id: int, i: int, j: int, events: Sequence[dict],
                     terminal_reason: str) -> Path:
    path = point_audit_path(point_id, i, j)
    bindings = [event["attempt_shard"] for event in events]
    for record in bindings:
        resolve_binding(record, "point attempt shard")
    payload = {
        "schema": POINT_AUDIT_SCHEMA, "status": "PASS_POINT_COMPLETE",
        "completed_utc": utc_now(), "point_id": point_id,
        "s_grid_index": i, "n_grid_index": j,
        "s_value": float(geometry.REGISTERED_S[i]),
        "n_value_rms_A": float(geometry.REGISTERED_N[j]),
        "terminal_reason": terminal_reason, "physical_force_calls": len(events),
        "fixed_endpoint_anchor": False, "optimizer_convergence_applicable": True,
        "attempt_shards": bindings, "optimization_N_F_contribution": 0,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        require(existing == payload or (
            {key: value for key, value in existing.items() if key != "completed_utc"}
            == {key: value for key, value in payload.items() if key != "completed_utc"}),
            "existing point audit mismatch")
    else:
        atomic_json(path, payload)
    return path


def make_anchor_audit(point_id: int, i: int, j: int, endpoint_index: int,
                      origin: np.ndarray, raw_force: np.ndarray,
                      projected_force: np.ndarray) -> Path:
    path = point_audit_path(point_id, i, j)
    projected_value = projected_fmax(projected_force)
    raw_value = projected_fmax(raw_force)
    payload = {
        "schema": POINT_AUDIT_SCHEMA, "status": "PASS_POINT_COMPLETE",
        "completed_utc": utc_now(), "point_id": point_id,
        "s_grid_index": i, "n_grid_index": j,
        "s_value": float(geometry.REGISTERED_S[i]),
        "n_value_rms_A": float(geometry.REGISTERED_N[j]),
        "terminal_reason": "fixed_endpoint_cached_below_threshold",
        "fixed_endpoint_anchor": True,
        "optimizer_convergence_applicable": False,
        "cached_force_below_common_threshold": projected_value <= FMAX_EV_A,
        "endpoint_cache_index": endpoint_index,
        "endpoint_joint_ef_cache": binding(ENDPOINT_CACHE),
        "coordinate_sha256": sha256_array(origin),
        "raw_force_sha256": sha256_array(raw_force),
        "projected_force_sha256": sha256_array(projected_force),
        "raw_fmax_eV_A": raw_value,
        "projected_fmax_eV_A": projected_value,
        "physical_force_calls": 0, "attempt_shards": [],
        "optimization_N_F_contribution": 0,
    }
    require(projected_value <= FMAX_EV_A,
            "cached endpoint force exceeds common pilot threshold")
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        require({key: value for key, value in existing.items() if key != "completed_utc"}
                == {key: value for key, value in payload.items() if key != "completed_utc"},
                "existing fixed-anchor audit mismatch")
    else:
        atomic_json(path, payload)
    return path


def prepare_contract() -> None:
    validate_coordinates()
    require(not CONTRACT.exists() and not PREAUTH.exists() and not AUTHORIZATION.exists()
            and not RUN_DIR.exists(), "constrained-pilot run chain already exists")
    payload = {
        "schema": CONTRACT_SCHEMA, "status": CONTRACT_STATUS,
        "created_utc": utc_now(), "evaluator": binding(THIS_FILE),
        "bb_core": binding(SRC / "bb_core.py"), "topology": binding(TOPOLOGY),
        "endpoint_joint_ef_cache": binding(ENDPOINT_CACHE),
        "coordinate_manifest": binding(COORDINATE),
        "coordinate_audit": binding(COORDINATE_AUDIT),
        "center_state": {
            "method": "IVM", "call_per_image": 515, "optimization_N_F": 14983,
            "path_sha256": geometry.base.SOURCE_PATH_SHA256,
            "state_role": "terminal_uncommitted_charged_tail",
            "formally_reportable": False,
        },
        "grid": {"shape": list(GRID_SHAPE),
                 "s_values": geometry.REGISTERED_S.tolist(),
                 "n_values_rms_A": geometry.REGISTERED_N.tolist(),
                 "execution_grid_indices": validate_coordinates()["execution"].tolist()},
        "constraint": {
            "kind": "frozen_linearized_s_n",
            "projector": "P(v)=v-tau*(tau dot v)-nu*(nu dot v)",
            "tau_nu_source": "coordinate manifest; fixed independently for each s",
            "projector_scope": "one flattened global 9396-dimensional projector per s; not per atom",
            "position_reprojection": "x <- x0 + P(x-x0) after every update",
            "convergence": "max_atom_norm(P(raw_Force))",
            "maximum_pre_oracle_and_post_update_absolute_tau_nu_drift_A": 1e-10,
        },
        "optimizer": {
            "name": "projected_FIRE_with_globally_scaled_per_atom_displacement_cap",
            "force_sign": "OpenMM physical Force=-gradient(E); update follows +P(Force)",
            "mass_policy": "unit-mass pseudo-velocity for every Cartesian coordinate",
            "downhill_check": False, "fmax_eV_per_A": FMAX_EV_A,
            "maximum_update_steps": MAX_UPDATE_STEPS,
            "maximum_force_calls_per_point": MAX_FORCE_CALLS_PER_POINT,
            "dt0": DT0, "dtmax": DTMAX, "finc": FINC, "fdec": FDEC,
            "alpha0": ALPHA0, "falpha": FALPHA, "Nmin": NMIN,
            "maximum_per_atom_step_A": MAX_ATOM_STEP_A,
        },
        "endpoint_policy": {
            "fixed_anchors": "s=0,n=0 and s=1,n=0 stay fixed and reuse canonical joint E/F",
            "fixed_anchor_fresh_force_calls": 0,
            "other_s_endpoint_points": "n!=0 values are surface probes, independently relaxed, and are not path endpoints",
            "endpoint_cache_evidence": binding(ENDPOINT_CACHE),
        },
        "physical_call_policy": {
            "kind": "joint energy+Force", "maximum_force_calls": MAX_FORCE_CALLS,
            "fresh_relaxed_points": FRESH_POINTS,
            "maximum_force_calls_per_fresh_point": MAX_FORCE_CALLS_PER_POINT,
            "maximum_fresh_center_column_force_calls": MAX_CENTER_COLUMN_FORCE_CALLS,
            "fixed_cached_endpoint_anchors": FIXED_ENDPOINT_ANCHORS,
            "energy_only_calls": 0, "inherited_joint_ef_values": 2,
            "optimization_N_F_contribution": 0,
        },
        "durability": {
            "wal": str(WAL.resolve()), "attempt_directory": str(ATTEMPT_DIR.resolve()),
            "point_audit_directory": str(POINT_AUDIT_DIR.resolve()),
            "result": str(RESULT.resolve()), "audit": str(AUDIT.resolve()),
            "attempt_payload": ["coordinate_A", "energy_eV", "raw_force_eV_A",
                                "projected_force_eV_A", "next_coordinate_A",
                                "update_displacement_SHA256_for_next_minus_current",
                                "fmax", "constraint_drift", "hashes",
                                "fire_velocity_out", "fire_scalar_state"],
        },
        "host_role": "local", "jobs": 1,
        "host_local_lock_paths": [str(path) for path in LOCKS],
        "thread_policy": {name: "1" for name in THREAD_VARS},
        "minimum_free_bytes_at_preflight": MINIMUM_FREE_BYTES,
        "minimum_free_bytes_before_each_physical_force_call": RUNTIME_FREE_BYTES_RESERVE,
        "estimated_maximum_history_bytes": ESTIMATED_HISTORY_BYTES,
    }
    atomic_json(CONTRACT, payload)
    print(json.dumps({"status": CONTRACT_STATUS, "contract": binding(CONTRACT),
                      "physical_force_calls_started": 0}, indent=2))


def load_contract() -> dict[str, Any]:
    path = CONTRACT.resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema") == CONTRACT_SCHEMA
            and payload.get("status") == CONTRACT_STATUS,
            "run contract identity mismatch")
    require(resolve_binding(payload["evaluator"], "evaluator") == THIS_FILE
            and resolve_binding(payload["bb_core"], "bb_core") == SRC / "bb_core.py"
            and resolve_binding(payload["topology"], "topology") == TOPOLOGY
            and resolve_binding(payload["endpoint_joint_ef_cache"], "endpoint cache") == ENDPOINT_CACHE
            and resolve_binding(payload["coordinate_manifest"], "coordinate manifest") == COORDINATE
            and resolve_binding(payload["coordinate_audit"], "coordinate audit") == COORDINATE_AUDIT,
            "run contract source path mismatch")
    require(payload["physical_call_policy"]["maximum_force_calls"] == MAX_FORCE_CALLS
            and payload["optimizer"]["maximum_force_calls_per_point"] == MAX_FORCE_CALLS_PER_POINT
            and payload["physical_call_policy"]["fixed_cached_endpoint_anchors"] == 2,
            "run contract Force budget mismatch")
    return payload


def preflight() -> None:
    contract = load_contract()
    check_thread_caps()
    coordinate_data = validate_coordinates()
    endpoint = validate_endpoint_cache(coordinate_data)
    require(not PREAUTH.exists() and not AUTHORIZATION.exists() and not RUN_DIR.exists(),
            "preflight output/run directory already exists")
    with acquire_locks() as locks:
        probed = locks
    free = shutil.disk_usage(OUTPUT_DIR).free
    require(free >= MINIMUM_FREE_BYTES, "insufficient disk for constrained pilot")
    payload = {
        "schema": PREAUTH_SCHEMA, "status": PREAUTH_STATUS,
        "created_utc": utc_now(), "contract": binding(CONTRACT),
        "evaluator": binding(THIS_FILE), "coordinate_manifest": binding(COORDINATE),
        "coordinate_audit": binding(COORDINATE_AUDIT),
        "endpoint_joint_ef_cache": binding(ENDPOINT_CACHE),
        "fixed_anchor_coordinate_hashes": [sha256_array(point) for point in endpoint["coordinates"]],
        "fixed_anchor_raw_force_fmax_eV_A": [projected_fmax(point) for point in endpoint["forces"]],
        "canonical_two_lock_nonblocking_probe": {"status": "PASS", "locks": probed},
        "thread_policy": {name: os.environ[name] for name in THREAD_VARS},
        "observed_free_bytes": free, "required_free_bytes": MINIMUM_FREE_BYTES,
        "maximum_physical_force_attempts": MAX_FORCE_CALLS,
        "maximum_fresh_center_column_force_attempts": MAX_CENTER_COLUMN_FORCE_CALLS,
        "fixed_cached_endpoint_anchors": 2,
        "openmm_contexts_constructed": 0, "physical_force_calls_started": 0,
        "optimization_N_F_contribution": 0,
    }
    atomic_json(PREAUTH, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def authorize() -> None:
    contract = load_contract()
    del contract
    check_thread_caps()
    require(PREAUTH.exists() and not AUTHORIZATION.exists() and not RUN_DIR.exists(),
            "authorization chain state mismatch")
    preauth = json.loads(PREAUTH.read_text(encoding="utf-8"))
    require(preauth.get("schema") == PREAUTH_SCHEMA
            and preauth.get("status") == PREAUTH_STATUS
            and preauth.get("contract") == binding(CONTRACT)
            and preauth.get("evaluator") == binding(THIS_FILE)
            and int(preauth.get("physical_force_calls_started", -1)) == 0,
            "preauthorization preflight mismatch")
    with acquire_locks() as locks:
        probed = locks
    payload = {
        "schema": AUTH_SCHEMA, "status": AUTH_STATUS, "enabled": True,
        "created_utc": utc_now(), "contract": binding(CONTRACT),
        "preauthorization_preflight": binding(PREAUTH), "evaluator": binding(THIS_FILE),
        "coordinate_manifest": binding(COORDINATE),
        "endpoint_joint_ef_cache": binding(ENDPOINT_CACHE),
        "result": str(RESULT.resolve()), "audit": str(AUDIT.resolve()),
        "maximum_physical_force_attempts": MAX_FORCE_CALLS,
        "maximum_force_calls_per_point": MAX_FORCE_CALLS_PER_POINT,
        "maximum_fresh_center_column_force_calls": MAX_CENTER_COLUMN_FORCE_CALLS,
        "fixed_cached_endpoint_anchors": 2,
        "energy_only_calls_authorized": 0, "optimization_N_F_contribution": 0,
        "canonical_two_lock_second_probe": {"status": "PASS", "locks": probed},
        "host_role": "local", "jobs": 1,
        "thread_policy": {name: "1" for name in THREAD_VARS},
    }
    atomic_json(AUTHORIZATION, payload)
    print(json.dumps({"status": AUTH_STATUS, "authorization": binding(AUTHORIZATION),
                      "physical_force_calls_started": 0}, indent=2))


def load_authorized() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    contract = load_contract()
    check_thread_caps()
    authorization = json.loads(AUTHORIZATION.resolve(strict=True).read_text(encoding="utf-8"))
    require(authorization.get("schema") == AUTH_SCHEMA
            and authorization.get("status") == AUTH_STATUS
            and authorization.get("enabled") is True
            and authorization.get("contract") == binding(CONTRACT)
            and authorization.get("preauthorization_preflight") == binding(PREAUTH)
            and authorization.get("evaluator") == binding(THIS_FILE)
            and authorization.get("endpoint_joint_ef_cache") == binding(ENDPOINT_CACHE)
            and int(authorization.get("maximum_physical_force_attempts", -1)) == MAX_FORCE_CALLS
            and int(authorization.get("fixed_cached_endpoint_anchors", -1)) == 2,
            "execution authorization mismatch")
    return contract, validate_coordinates()


def run() -> None:
    contract, data = load_authorized()
    del contract
    endpoint = validate_endpoint_cache(data)
    header = wal_header(CONTRACT, AUTHORIZATION)
    free = shutil.disk_usage(OUTPUT_DIR).free
    require(free >= MINIMUM_FREE_BYTES or RUN_DIR.exists(),
            "insufficient disk before constrained pilot")
    with acquire_locks() as held_locks:
        wal = DurableWal(WAL, header)
        backend = None
        try:
            state = read_wal(WAL, header)
            if AUDIT.exists():
                print(AUDIT.read_text(encoding="utf-8"))
                return
            if state["publications"]:
                require(RESULT.exists(), "published result is missing")
            execution = data["execution"]
            invocation_starts_before = len(state["starts"])
            for point_id, pair in enumerate(execution):
                i, j = (int(pair[0]), int(pair[1]))
                successes = point_successes(state, point_id)
                if point_id in state["point_completions"]:
                    require(successes or (j == 2 and i in (0, 14)),
                            "completed non-anchor point has no attempts")
                    continue
                origin = np.ascontiguousarray(data["coordinates"][i, j], dtype=np.float64)
                tau = np.ascontiguousarray(data["tangents"][i], dtype=np.float64)
                nu = np.ascontiguousarray(data["normals"][i], dtype=np.float64)
                if j == 2 and i in (0, 14):
                    endpoint_index = 0 if i == 0 else 1
                    raw_force = endpoint["forces"][endpoint_index]
                    force = project(raw_force, tau, nu)
                    point_audit = make_anchor_audit(
                        point_id, i, j, endpoint_index, origin, raw_force, force)
                    completion = {
                        "event": "point_completed", "utc": utc_now(),
                        "point_id": point_id, "s_grid_index": i,
                        "n_grid_index": j,
                        "terminal_reason": "fixed_endpoint_cached_below_threshold",
                        "fixed_endpoint_anchor": True, "physical_force_calls": 0,
                        "endpoint_joint_ef_cache": binding(ENDPOINT_CACHE),
                        "point_audit": binding(point_audit),
                    }
                    wal.append(completion)
                    state["point_completions"][point_id] = completion
                    continue
                if successes:
                    last_path = resolve_binding(successes[-1]["attempt_shard"], "last attempt")
                    last = load_attempt(last_path)
                    terminal = bool(scalar(last, "terminal"))
                    if terminal:
                        reason = str(scalar(last, "terminal_reason"))
                        point_audit = make_point_audit(point_id, i, j, successes, reason)
                        wal.append({"event": "point_completed", "utc": utc_now(),
                                    "point_id": point_id, "s_grid_index": i,
                                    "n_grid_index": j, "terminal_reason": reason,
                                    "physical_force_calls": len(successes),
                                    "point_audit": binding(point_audit)})
                        state["point_completions"][point_id] = {
                            "point_id": point_id, "point_audit": binding(point_audit),
                            "terminal_reason": reason,
                        }
                        continue
                    position = np.ascontiguousarray(last["next_coordinate_A"], dtype=np.float64)
                    velocity = np.ascontiguousarray(last["fire_velocity_out"], dtype=np.float64)
                    dt = float(scalar(last, "fire_dt_out"))
                    alpha = float(scalar(last, "fire_alpha_out"))
                    n_positive = int(scalar(last, "fire_n_positive_out"))
                    step = len(successes)
                else:
                    position = origin.copy()
                    velocity = np.zeros_like(origin)
                    dt, alpha, n_positive, step = DT0, ALPHA0, 0, 0
                while True:
                    require(shutil.disk_usage(OUTPUT_DIR).free >= RUNTIME_FREE_BYTES_RESERVE,
                            "runtime disk reserve reached before next Force call")
                    require(step <= MAX_UPDATE_STEPS, "optimizer step exceeds contract")
                    require(len(state["starts"]) < MAX_FORCE_CALLS,
                            "global physical Force-call authorization exhausted")
                    pre_tau_drift, pre_nu_drift = constraint_drift(
                        position, origin, tau, nu)
                    require(pre_tau_drift <= 1e-10 and pre_nu_drift <= 1e-10,
                            "pre-oracle coordinate escaped fixed s/n subspace")
                    attempt_id = f"p{point_id:03d}:step{step:03d}"
                    path = attempt_path(point_id, i, j, step)
                    ordinal = len(state["starts"]) + 1
                    start = {
                        "event": "attempt_started", "utc": utc_now(),
                        "attempt_id": attempt_id, "physical_force_call_ordinal": ordinal,
                        "point_id": point_id, "s_grid_index": i, "n_grid_index": j,
                        "s_value": float(data["s"][i]), "n_value_rms_A": float(data["n"][j]),
                        "optimizer_step": step, "coordinate_sha256": sha256_array(position),
                        "pre_oracle_tau_drift_A": pre_tau_drift,
                        "pre_oracle_nu_drift_A": pre_nu_drift,
                    }
                    if backend is None:
                        if str(SRC) not in sys.path:
                            sys.path.insert(0, str(SRC))
                        from bb_core import OpenMMBackend
                        backend = OpenMMBackend(TOPOLOGY, platform_preference=("CPU",),
                                                precision="double",
                                                label="call515_constrained_relaxed_pilot")
                    wal.append(start)
                    state["starts"][attempt_id] = start
                    try:
                        energy, raw_flat = backend.evaluate_joint(position.reshape(-1))
                        raw_force = np.ascontiguousarray(raw_flat.reshape(N_ATOMS, 3), dtype=np.float64)
                    except Exception as error:
                        wal.append({"event": "attempt_failed", "utc": utc_now(),
                                    **{key: value for key, value in start.items() if key != "event" and key != "utc"},
                                    "error": repr(error)})
                        raise
                    force = project(raw_force, tau, nu)
                    fmax = projected_fmax(force)
                    raw_fmax = projected_fmax(raw_force)
                    terminal = fmax <= FMAX_EV_A or step == MAX_UPDATE_STEPS
                    reason = "converged" if fmax <= FMAX_EV_A else (
                        "maximum_update_steps" if terminal else "continue")
                    if terminal:
                        next_position = position.copy()
                        velocity_out = velocity.copy()
                        dt_out, alpha_out, n_positive_out = dt, alpha, n_positive
                    else:
                        next_position, velocity_out, dt_out, alpha_out, n_positive_out = fire_advance(
                            position, origin, force, tau, nu, velocity, dt, alpha,
                            n_positive, first_step=(step == 0))
                    update = np.ascontiguousarray(next_position - position, dtype=np.float64)
                    post_tau_drift, post_nu_drift = constraint_drift(
                        next_position, origin, tau, nu)
                    require(post_tau_drift <= 1e-10 and post_nu_drift <= 1e-10,
                            "post-update coordinate escaped fixed s/n subspace")
                    maximum_update = float(np.max(np.linalg.norm(update, axis=1)))
                    require(maximum_update <= MAX_ATOM_STEP_A + 1e-12,
                            "globally scaled per-atom displacement cap violated")
                    payload = {
                        "schema": np.asarray(ATTEMPT_SCHEMA),
                        "status": np.asarray("PASS_PHYSICAL_JOINT_EF"),
                        "attempt_id": np.asarray(attempt_id),
                        "physical_force_call_ordinal": np.asarray(ordinal, dtype=np.int64),
                        "point_id": np.asarray(point_id, dtype=np.int64),
                        "s_grid_index": np.asarray(i, dtype=np.int64),
                        "n_grid_index": np.asarray(j, dtype=np.int64),
                        "s_value": np.asarray(data["s"][i], dtype=np.float64),
                        "n_value_rms_A": np.asarray(data["n"][j], dtype=np.float64),
                        "optimizer_step": np.asarray(step, dtype=np.int64),
                        "coordinate_A": position, "energy_eV": np.asarray(energy, dtype=np.float64),
                        "raw_force_eV_A": raw_force, "projected_force_eV_A": force,
                        "raw_fmax_eV_A": np.asarray(raw_fmax, dtype=np.float64),
                        "projected_fmax_eV_A": np.asarray(fmax, dtype=np.float64),
                        "pre_oracle_tau_drift_A": np.asarray(pre_tau_drift, dtype=np.float64),
                        "pre_oracle_nu_drift_A": np.asarray(pre_nu_drift, dtype=np.float64),
                        "post_update_tau_drift_A": np.asarray(post_tau_drift, dtype=np.float64),
                        "post_update_nu_drift_A": np.asarray(post_nu_drift, dtype=np.float64),
                        "maximum_atom_update_A": np.asarray(maximum_update, dtype=np.float64),
                        "fixed_tau_sha256": np.asarray(sha256_array(tau)),
                        "fixed_nu_sha256": np.asarray(sha256_array(nu)),
                        "coordinate_sha256": np.asarray(sha256_array(position)),
                        "energy_sha256": np.asarray(sha256_array(np.asarray(energy))),
                        "raw_force_sha256": np.asarray(sha256_array(raw_force)),
                        "projected_force_sha256": np.asarray(sha256_array(force)),
                        "update_displacement_sha256": np.asarray(sha256_array(update)),
                        "next_coordinate_A": next_position,
                        "next_coordinate_sha256": np.asarray(sha256_array(next_position)),
                        "fire_velocity_out": velocity_out,
                        "fire_dt_out": np.asarray(dt_out, dtype=np.float64),
                        "fire_alpha_out": np.asarray(alpha_out, dtype=np.float64),
                        "fire_n_positive_out": np.asarray(n_positive_out, dtype=np.int64),
                        "terminal": np.asarray(terminal), "terminal_reason": np.asarray(reason),
                        "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
                    }
                    atomic_npz(path, payload)
                    success = {
                        "event": "attempt_succeeded", "utc": utc_now(),
                        **{key: value for key, value in start.items() if key != "event" and key != "utc"},
                        "energy_eV": float(energy), "raw_fmax_eV_A": raw_fmax,
                        "projected_fmax_eV_A": fmax,
                        "energy_sha256": sha256_array(np.asarray(energy)),
                        "raw_force_sha256": sha256_array(raw_force),
                        "projected_force_sha256": sha256_array(force),
                        "next_coordinate_sha256": sha256_array(next_position),
                        "attempt_shard": binding(path), "terminal": terminal,
                        "terminal_reason": reason,
                    }
                    wal.append(success)
                    state["successes"][attempt_id] = success
                    successes = point_successes(state, point_id)
                    if terminal:
                        point_audit = make_point_audit(point_id, i, j, successes, reason)
                        completion = {"event": "point_completed", "utc": utc_now(),
                                      "point_id": point_id, "s_grid_index": i,
                                      "n_grid_index": j, "terminal_reason": reason,
                                      "fixed_endpoint_anchor": False,
                                      "physical_force_calls": len(successes),
                                      "point_audit": binding(point_audit)}
                        wal.append(completion)
                        state["point_completions"][point_id] = completion
                        break
                    position, velocity = next_position, velocity_out
                    dt, alpha, n_positive = dt_out, alpha_out, n_positive_out
                    step += 1
            state = read_wal(WAL, header)
            require(len(state["point_completions"]) == POINTS,
                    "not all constrained pilot points completed")
            final_coordinate = np.empty((15, 9, N_ATOMS, 3), dtype=np.float64)
            final_energy = np.empty((15, 9), dtype=np.float64)
            final_raw_force = np.empty_like(final_coordinate)
            final_projected_force = np.empty_like(final_coordinate)
            calls = np.empty((15, 9), dtype=np.int64)
            converged = np.empty((15, 9), dtype=np.bool_)
            terminal_reason = np.empty((15, 9), dtype="<U48")
            for point_id, pair in enumerate(data["execution"]):
                i, j = int(pair[0]), int(pair[1])
                events = point_successes(state, point_id)
                if j == 2 and i in (0, 14):
                    endpoint_index = 0 if i == 0 else 1
                    final_coordinate[i, j] = endpoint["coordinates"][endpoint_index]
                    final_energy[i, j] = endpoint["energies"][endpoint_index]
                    final_raw_force[i, j] = endpoint["forces"][endpoint_index]
                    final_projected_force[i, j] = project(
                        endpoint["forces"][endpoint_index], data["tangents"][i],
                        data["normals"][i])
                    calls[i, j] = 0
                    reason = "fixed_endpoint_cached_below_threshold"
                else:
                    require(events, "fresh point lacks final attempt")
                    last = load_attempt(resolve_binding(events[-1]["attempt_shard"], "final attempt"))
                    final_coordinate[i, j] = last["coordinate_A"]
                    final_energy[i, j] = float(scalar(last, "energy_eV"))
                    final_raw_force[i, j] = last["raw_force_eV_A"]
                    final_projected_force[i, j] = last["projected_force_eV_A"]
                    calls[i, j] = len(events)
                    reason = str(scalar(last, "terminal_reason"))
                terminal_reason[i, j] = reason
                converged[i, j] = reason in (
                    "converged", "fixed_endpoint_cached_below_threshold")
            result_payload = {
                "schema": np.asarray(RESULT_SCHEMA), "status": np.asarray("PASS"),
                "contract": np.asarray(str(CONTRACT.resolve())),
                "contract_sha256": np.asarray(sha256_file(CONTRACT)),
                "authorization": np.asarray(str(AUTHORIZATION.resolve())),
                "authorization_sha256": np.asarray(sha256_file(AUTHORIZATION)),
                "coordinate_manifest": np.asarray(str(COORDINATE.resolve())),
                "coordinate_manifest_sha256": np.asarray(sha256_file(COORDINATE)),
                "center_source_call_per_image": np.asarray(515, dtype=np.int64),
                "center_source_optimization_N_F": np.asarray(14983, dtype=np.int64),
                "terminal_uncommitted_charged_tail": np.asarray(True),
                "formally_reportable": np.asarray(False),
                "s_values": data["s"], "n_values_rms_A": data["n"],
                "initial_coordinates_A": data["coordinates"],
                "final_coordinates_A": final_coordinate,
                "final_energy_eV": final_energy, "final_raw_force_eV_A": final_raw_force,
                "final_projected_force_eV_A": final_projected_force,
                "physical_force_calls_per_point": calls,
                "converged": converged, "terminal_reason": terminal_reason,
                "fixed_endpoint_anchor_mask": np.asarray(
                    [[j == 2 and i in (0, 14) for j in range(9)] for i in range(15)],
                    dtype=np.bool_),
                "optimizer_convergence_applicable_mask": np.asarray(
                    [[not (j == 2 and i in (0, 14)) for j in range(9)] for i in range(15)],
                    dtype=np.bool_),
                "endpoint_joint_ef_cache": np.asarray(str(ENDPOINT_CACHE.resolve())),
                "endpoint_joint_ef_cache_sha256": np.asarray(sha256_file(ENDPOINT_CACHE)),
                "fixed_cached_endpoint_values": np.asarray(2, dtype=np.int64),
                "physical_force_calls_total": np.asarray(len(state["starts"]), dtype=np.int64),
                "maximum_authorized_force_calls": np.asarray(MAX_FORCE_CALLS, dtype=np.int64),
                "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
                "force_wal": np.asarray(str(WAL.resolve())),
                "held_host_locks": np.asarray(held_locks),
            }
            if not RESULT.exists():
                atomic_npz(RESULT, result_payload)
            validate_npz_payload(RESULT, result_payload)
            if not state["publications"]:
                publication = {"event": "result_published", "utc": utc_now(),
                               "result": binding(RESULT)}
                wal.append(publication)
                state["publications"].append(publication)
            require(len(state["publications"]) == 1, "result publication count mismatch")
            if backend is not None:
                require(backend.raw_joint_attempts == backend.raw_joint_evaluations,
                        "backend joint attempt/success ledger mismatch")
                require(backend.raw_joint_attempts
                        == len(state["starts"]) - invocation_starts_before,
                        "backend/WAL invocation Force-call ledger mismatch")
            point_audits = [binding(point_audit_path(point_id, int(pair[0]), int(pair[1])))
                            for point_id, pair in enumerate(data["execution"])]
            audit_payload = {
                "schema": AUDIT_SCHEMA, "status": "PASS", "completed_utc": utc_now(),
                "contract": binding(CONTRACT), "preauthorization_preflight": binding(PREAUTH),
                "authorization": binding(AUTHORIZATION),
                "coordinate_manifest": binding(COORDINATE),
                "coordinate_audit": binding(COORDINATE_AUDIT),
                "endpoint_joint_ef_cache": binding(ENDPOINT_CACHE),
                "result": binding(RESULT), "force_wal": binding(WAL),
                "point_audits": point_audits, "held_host_locks": held_locks,
                "independent_points": POINTS,
                "physical_force_calls_total": len(state["starts"]),
                "physical_force_calls_succeeded": len(state["successes"]),
                "physical_force_calls_failed": len(state["failures"]),
                "maximum_authorized_force_calls": MAX_FORCE_CALLS,
                "fixed_cached_endpoint_anchors": 2,
                "fresh_relaxed_points": FRESH_POINTS,
                "fresh_converged_points": int(np.count_nonzero(terminal_reason == "converged")),
                "fixed_cached_below_threshold_points": int(np.count_nonzero(
                    terminal_reason == "fixed_endpoint_cached_below_threshold")),
                "max_step_points": int(np.count_nonzero(
                    terminal_reason == "maximum_update_steps")),
                "energy_only_calls": 0, "optimization_N_F_contribution": 0,
                "center_state": {
                    "method": "IVM", "call_per_image": 515, "optimization_N_F": 14983,
                    "path_sha256": geometry.base.SOURCE_PATH_SHA256,
                    "state_role": "terminal_uncommitted_charged_tail",
                    "formally_reportable": False,
                },
            }
            atomic_json(AUDIT, audit_payload)
            print(json.dumps(audit_payload, indent=2, sort_keys=True))
        finally:
            wal.close()


def self_test() -> None:
    rng = np.random.default_rng(20260903)
    tau = rng.normal(size=(N_ATOMS, 3)); tau /= np.linalg.norm(tau)
    nu = rng.normal(size=(N_ATOMS, 3)); nu = project(nu, tau, np.zeros_like(tau)); nu /= np.linalg.norm(nu)
    value = rng.normal(size=(N_ATOMS, 3))
    projected = project(value, tau, nu)
    require(abs(float(np.vdot(projected, tau))) < 1e-10
            and abs(float(np.vdot(projected, nu))) < 1e-10,
            "projector self-test failed")
    origin = rng.normal(size=(N_ATOMS, 3))
    quadratic_force = project(-origin, tau, nu)
    candidate, velocity, dt, alpha, count = fire_advance(
        origin.copy(), origin, quadratic_force, tau, nu, np.zeros_like(origin),
        DT0, ALPHA0, 0, True)
    require(candidate.shape == velocity.shape == origin.shape and dt > 0 and alpha > 0
            and count == 0, "FIRE self-test failed")
    require(0.5 * float(np.vdot(candidate, candidate))
            < 0.5 * float(np.vdot(origin, origin)),
            "FIRE did not follow +Force downhill on E=||x||^2/2")
    print(json.dumps({"status": "PASS_ZERO_PHYSICS_SELF_TEST",
                      "openmm_contexts_constructed": 0,
                      "physical_force_calls_started": 0}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-contract", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--authorize", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--self-test-zero-physics", action="store_true")
    args = parser.parse_args()
    if args.prepare_contract:
        prepare_contract()
    elif args.preflight:
        preflight()
    elif args.authorize:
        authorize()
    elif args.run:
        run()
    else:
        self_test()


if __name__ == "__main__":
    main()
