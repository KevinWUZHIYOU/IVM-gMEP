#!/usr/bin/env python3
"""Evaluate the frozen IR-IVM NF8982 59x28 section with energy-only calls.

The 1,652 surface coordinates and the 30 internal images of the exact source
path are separate diagnostic stages.  No force is requested and no call is
charged to optimization ``N_F``.  An append-only fsync WAL makes completed
energy calls reusable after interruption without interpolating any value.
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
from typing import Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bb_core import OpenMMBackend  # noqa: E402


CONTRACT_SCHEMA = "bb-irivm-nf8982-0k-section-energy-contract-v2"
CONTRACT_STATUS = "FROZEN_R3_CALL_POLICY_BEFORE_FIRST_SECTION_ENERGY_CALL"
AUTH_SCHEMA = "bb-irivm-nf8982-0k-section-energy-authorization-v2"
AUTH_STATUS = "AUTHORIZED_AFTER_ZERO_OPENMM_PREFLIGHT"
PREAUTH_SCHEMA = "bb-irivm-nf8982-0k-section-energy-preauthorization-preflight-v1"
PREAUTH_STATUS = "PASS_ZERO_OPENMM_BEFORE_AUTHORIZATION"
COORDINATE_SCHEMA = "bb-irivm-nf8982-0k-section-coordinate-manifest-v2"
COORDINATE_STATUS = "FROZEN_BEFORE_ANY_SECTION_ENERGY_CALL"
RESULT_SCHEMA = "bb-irivm-nf8982-0k-section-energy-result-v2"
AUDIT_SCHEMA = "bb-irivm-nf8982-0k-section-energy-audit-v2"
WAL_SCHEMA = "bb-irivm-nf8982-0k-section-energy-wal-v2"
SURFACE_STAGE = "surface"
PATH_STAGE = "original_path32"
POINTS = 59
NORMAL_POINTS = 28
ZERO_INDEX = 4
N_IMAGES = 32
N_INTERNAL = 30
N_ATOMS = 3132
DIMENSION = 3 * N_ATOMS
SURFACE_CALLS = POINTS * NORMAL_POINTS
PATH_INTERNAL_CALLS = N_INTERNAL
SURFACE_FRESH_CALLS = SURFACE_CALLS - 2
TOTAL_FRESH_CALLS = SURFACE_FRESH_CALLS + PATH_INTERNAL_CALLS
ENDPOINT_CACHE = ROOT / "inputs/endpoint_joint_ef_cache.npz"
ENDPOINT_CACHE_SHA256 = "5cd4e757c99f0784d8b0781d05d8083070c8cfaad2670ead26999b3a4134e99a"
CALL_POLICY_SCHEMA = "bb-irivm-nf8982-0k-section-energy-call-policy-v3"
CALL_POLICY_STATUS = "FROZEN_BEFORE_ANY_SECTION_ENERGY_CALL"
MINIMUM_FREE_BYTES_RESERVE = 8 * 1024**3
ESTIMATED_FINAL_BYTES = 320 * 1024**2
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENMM_CPU_THREADS",
)
CANONICAL_LOCKS = {
    Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock").resolve(),
    Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock").resolve(),
}
EXPECTED_N_GRID = np.arange(-0.12, 0.6901, 0.03, dtype=np.float64)
EXPECTED_N_GRID[ZERO_INDEX] = 0.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def scalar(saved: Mapping[str, np.ndarray], key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def file_binding(path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def resolve_binding(base: Path, record: Mapping[str, object], role: str) -> Path:
    require(isinstance(record, Mapping), f"{role} binding is not an object")
    require(set(record) >= {"path", "sha256"}, f"{role} lacks path/SHA-256")
    path = Path(str(record["path"])).expanduser()
    if not path.is_absolute():
        path = base.parent / path
    path = path.resolve(strict=True)
    require(sha256_file(path) == str(record["sha256"]), f"{role} SHA-256 mismatch")
    return path


def check_thread_caps() -> None:
    wrong = {
        name: os.environ.get(name)
        for name in THREAD_VARS
        if os.environ.get(name) != "1"
    }
    require(not wrong, f"all six thread caps must equal '1': {wrong}")


@contextmanager
def acquire_two_locks(paths: Sequence[Path]):
    ordered = sorted({Path(path).expanduser().resolve() for path in paths}, key=str)
    require(len(ordered) == 2 and set(ordered) == CANONICAL_LOCKS, "canonical two-lock set required")
    streams = []
    try:
        for path in ordered:
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a+")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                stream.close()
                raise RuntimeError(f"another BB task owns {path}") from error
            streams.append(stream)
        yield [str(path) for path in ordered]
    finally:
        for stream in reversed(streams):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_npz_no_overwrite(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    require(not temporary.exists(), f"stale temporary output: {temporary}")
    try:
        np.savez_compressed(temporary, **payload)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_no_overwrite(path: Path, payload: Mapping[str, object]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    require(not temporary.exists(), f"stale temporary output: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class DurableWal:
    def __init__(self, path: Path, expected: Mapping[str, object]):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = dict(expected)
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            require(lines, "existing energy WAL is empty")
            header = json.loads(lines[0])
            require(header == expected, "existing energy WAL header mismatch")
            self.stream = path.open("a", encoding="utf-8", buffering=1)
        else:
            self.stream = path.open("x", encoding="utf-8", buffering=1)
            self.append(expected)
            _fsync_parent(path)

    def append(self, record: Mapping[str, object]) -> None:
        self.stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


def validate_coordinate_manifest(path: Path, expected_sha: str | None = None) -> dict:
    if expected_sha is not None:
        require(sha256_file(path) == expected_sha, "coordinate manifest SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "schema",
            "status",
            "freeze_contract",
            "freeze_contract_sha256",
            "builder",
            "builder_sha256",
            "center_source_optimization_N_F",
            "center_source_call_per_image",
            "terminal_partial_tail_excluded",
            "original_path32_A",
            "original_path32_sha256",
            "center_path59_A",
            "center_path59_sha256",
            "center_s",
            "central_tangents",
            "normal_directions",
            "normal_coordinate_rms_A",
            "zero_normal_index",
            "coordinates_A",
            "coordinate_sha256",
            "surface_energy_only_calls_planned",
            "original_path_internal_energy_only_calls_planned",
            "total_energy_only_calls_planned",
            "required_force_evaluations",
            "optimization_N_F_contribution",
            "selected_result",
            "selected_result_sha256",
        }
        require(required.issubset(saved.files), "coordinate manifest fields missing")
        require(
            str(scalar(saved, "schema")) == COORDINATE_SCHEMA
            and str(scalar(saved, "status")) == COORDINATE_STATUS,
            "coordinate manifest identity mismatch",
        )
        require(int(scalar(saved, "center_source_optimization_N_F")) == 8982, "center NoF mismatch")
        require(int(scalar(saved, "center_source_call_per_image")) == 318, "center call mismatch")
        require(bool(scalar(saved, "terminal_partial_tail_excluded")), "terminal tail was not excluded")
        require(int(scalar(saved, "zero_normal_index")) == ZERO_INDEX, "zero-normal index mismatch")
        require(int(scalar(saved, "surface_energy_only_calls_planned")) == SURFACE_CALLS, "surface count mismatch")
        require(int(scalar(saved, "original_path_internal_energy_only_calls_planned")) == PATH_INTERNAL_CALLS, "path count mismatch")
        # This is a historical planning field in the already frozen R2 geometry.
        # The prospective R3 execution policy inherits the two fixed endpoints,
        # so its effective fresh-call total is 1,680 rather than this old 1,682.
        require(
            int(scalar(saved, "total_energy_only_calls_planned"))
            == SURFACE_CALLS + PATH_INTERNAL_CALLS,
            "historical total count mismatch",
        )
        require(int(scalar(saved, "required_force_evaluations")) == 0, "coordinate contract requests force")
        require(int(scalar(saved, "optimization_N_F_contribution")) == 0, "coordinate contract charges optimization")
        path32 = np.ascontiguousarray(saved["original_path32_A"], dtype=np.float64)
        center = np.ascontiguousarray(saved["center_path59_A"], dtype=np.float64)
        tangent = np.ascontiguousarray(saved["central_tangents"], dtype=np.float64)
        normal = np.ascontiguousarray(saved["normal_directions"], dtype=np.float64)
        n_grid = np.ascontiguousarray(saved["normal_coordinate_rms_A"], dtype=np.float64)
        coordinates = np.ascontiguousarray(saved["coordinates_A"], dtype=np.float64)
        hashes = np.asarray(saved["coordinate_sha256"]).astype(str)
        source_result = Path(str(scalar(saved, "selected_result"))).expanduser().resolve(strict=True)
        source_result_sha = str(scalar(saved, "selected_result_sha256"))
        freeze_contract = Path(str(scalar(saved, "freeze_contract"))).expanduser().resolve(strict=True)
        builder = Path(str(scalar(saved, "builder"))).expanduser().resolve(strict=True)
        freeze_contract_sha = str(scalar(saved, "freeze_contract_sha256"))
        builder_sha = str(scalar(saved, "builder_sha256"))
    require(path32.shape == (N_IMAGES, N_ATOMS, 3), "original path32 shape mismatch")
    require(center.shape == tangent.shape == normal.shape == (POINTS, N_ATOMS, 3), "basis shape mismatch")
    require(coordinates.shape == (POINTS, NORMAL_POINTS, N_ATOMS, 3), "surface shape mismatch")
    require(hashes.shape == (POINTS, NORMAL_POINTS), "cell hash table shape mismatch")
    require(np.array_equal(n_grid, EXPECTED_N_GRID), "normal grid mismatch")
    require(n_grid[ZERO_INDEX].tobytes() == np.float64(0.0).tobytes(), "n=0 is not exact +0.0")
    require(np.array_equal(coordinates[:, ZERO_INDEX], center), "n=0 column is not exact center")
    require(np.array_equal(center[[0, -1]], path32[[0, -1]]), "center endpoints differ from source path")
    require(np.all(np.isfinite(path32)) and np.all(np.isfinite(coordinates)), "non-finite coordinates")
    require(sha256_array(path32) == str(scalar_from_file(path, "original_path32_sha256")), "path32 hash mismatch")
    require(sha256_array(center) == str(scalar_from_file(path, "center_path59_sha256")), "center hash mismatch")
    flat_tangent = tangent.reshape(POINTS, -1)
    flat_normal = normal.reshape(POINTS, -1)
    require(np.allclose(np.linalg.norm(flat_tangent, axis=1), 1.0, rtol=0.0, atol=2.0e-12), "tangent norm mismatch")
    require(np.allclose(np.linalg.norm(flat_normal, axis=1), 1.0, rtol=0.0, atol=2.0e-12), "normal norm mismatch")
    require(np.allclose(np.sum(flat_tangent * flat_normal, axis=1), 0.0, rtol=0.0, atol=2.0e-12), "tangent/normal not orthogonal")
    for i in range(POINTS):
        for j in range(NORMAL_POINTS):
            expected = center[i] if j == ZERO_INDEX else center[i] + n_grid[j] * np.sqrt(float(N_ATOMS)) * normal[i]
            require(np.array_equal(coordinates[i, j], expected), f"surface coordinate formula mismatch at {i},{j}")
            require(sha256_array(coordinates[i, j]) == hashes[i, j], f"cell hash mismatch at {i},{j}")
    require(sha256_file(source_result) == source_result_sha, "selected result binding drift")
    require(sha256_file(freeze_contract) == freeze_contract_sha, "coordinate freeze contract drift")
    require(sha256_file(builder) == builder_sha, "coordinate builder drift")
    return {
        "path": path,
        "path_sha256": sha256_file(path),
        "surface_coordinates": coordinates,
        "cell_hashes": hashes,
        "center": center,
        "n_grid": n_grid,
        "original_path32": path32,
        "original_path32_sha256": sha256_array(path32),
        "source_result": source_result,
        "source_result_sha256": source_result_sha,
        "freeze_contract": freeze_contract,
        "builder": builder,
    }


def scalar_from_file(path: Path, key: str):
    with np.load(path, allow_pickle=False) as saved:
        return scalar(saved, key)


def load_endpoint_cache(path: Path, data: dict, expected_sha: str) -> dict:
    require(sha256_file(path) == expected_sha == ENDPOINT_CACHE_SHA256, "endpoint cache SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as saved:
        require(
            {"coordinates_A", "energies_eV", "forces_eV_per_A"}.issubset(saved.files),
            "endpoint cache fields missing",
        )
        coordinates = np.ascontiguousarray(saved["coordinates_A"], dtype=np.float64)
        energies = np.ascontiguousarray(saved["energies_eV"], dtype=np.float64)
        forces = np.ascontiguousarray(saved["forces_eV_per_A"], dtype=np.float64)
    require(coordinates.shape == (2, DIMENSION), "endpoint cache coordinate shape mismatch")
    require(energies.shape == (2,) and forces.shape == (2, DIMENSION), "endpoint cache E/F shape mismatch")
    require(np.all(np.isfinite(coordinates)) and np.all(np.isfinite(energies)) and np.all(np.isfinite(forces)), "endpoint cache contains non-finite values")
    path_endpoints = data["original_path32"][[0, -1]].reshape(2, DIMENSION)
    surface_endpoints = data["surface_coordinates"][[0, -1], ZERO_INDEX].reshape(2, DIMENSION)
    require(np.array_equal(coordinates, path_endpoints), "endpoint cache differs from exact path endpoints")
    require(np.array_equal(coordinates, surface_endpoints), "endpoint cache differs from surface n=0 endpoints")
    require(sha256_array(coordinates) == "5788e22f86df91ed5500e88a8dc053f7408bd2fb39414de121a6eec36b833db1", "endpoint coordinate-pair hash mismatch")
    require(sha256_array(energies) == "24884695bc3bbecbf1e5007bf8f245bfe6d7eb5bf81725e986b8dd0ece088a31", "endpoint energy-pair hash mismatch")
    require(sha256_array(forces) == "c28ab9e8ad515beb3cfd5de8cc09af542ff9c80a96815d8d1b8b17d03fd8e01c", "endpoint force-pair hash mismatch")
    return {
        "path": path,
        "sha256": expected_sha,
        "coordinates": coordinates,
        "energies": energies,
        "forces": forces,
        "coordinate_sha256": sha256_array(coordinates),
        "energy_sha256": sha256_array(energies),
        "force_sha256": sha256_array(forces),
    }


def _logical_coordinates(data: dict, stage: str):
    require(stage in {SURFACE_STAGE, PATH_STAGE}, "unknown energy stage")
    if stage == SURFACE_STAGE:
        for flat_index in range(SURFACE_CALLS):
            i, j = divmod(flat_index, NORMAL_POINTS)
            if (i, j) in {(0, ZERO_INDEX), (POINTS - 1, ZERO_INDEX)}:
                continue
            yield {
                "stage": SURFACE_STAGE,
                "logical_index": flat_index,
                "center_index": i,
                "normal_index": j,
                "coordinate": data["surface_coordinates"][i, j].reshape(-1),
                "coordinate_sha256": data["cell_hashes"][i, j],
            }
        return
    if stage == PATH_STAGE:
        for internal_index, coordinate in enumerate(data["original_path32"][1:-1], start=1):
            yield {
                "stage": PATH_STAGE,
                "logical_index": internal_index - 1,
                "original_image_index": internal_index,
                "coordinate": coordinate.reshape(-1),
                "coordinate_sha256": sha256_array(coordinate),
            }
        return


def read_wal(path: Path, expected_header: Mapping[str, object]) -> tuple[list[dict], dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    require(lines, "energy WAL is empty")
    header = json.loads(lines[0])
    require(header == expected_header, "energy WAL header mismatch")
    events = [json.loads(line) for line in lines[1:]]
    starts = {}
    terminals = {}
    publications = []
    for event in events:
        kind = event.get("event")
        require(
            kind
            in {
                "attempt_started",
                "attempt_succeeded",
                "attempt_failed",
                "result_published",
            },
            "unknown WAL event",
        )
        if kind == "result_published":
            publications.append(event)
            require(len(publications) == 1, "duplicate result publication footer")
            continue
        require(not publications, "attempt event appears after result publication footer")
        attempt_id = str(event.get("attempt_id", ""))
        require(attempt_id, "WAL event lacks attempt_id")
        if kind == "attempt_started":
            require(attempt_id not in starts, "duplicate WAL attempt start")
            starts[attempt_id] = event
        else:
            require(attempt_id in starts, "WAL terminal without start")
            require(attempt_id not in terminals, "duplicate/conflicting WAL terminal")
            terminals[attempt_id] = event
            for key in ("stage", "logical_index", "coordinate_sha256"):
                require(event.get(key) == starts[attempt_id].get(key), "WAL terminal/start mismatch")
    successes = {}
    for attempt_id, terminal in terminals.items():
        if terminal["event"] != "attempt_succeeded":
            continue
        key = (str(terminal["stage"]), int(terminal["logical_index"]))
        require(key not in successes, "multiple successful physical calls for one logical coordinate")
        value = float(terminal.get("energy_eV", np.nan))
        require(np.isfinite(value), "non-finite WAL energy")
        require(
            sha256_array(np.asarray(value, dtype=np.float64))
            == str(terminal.get("energy_scalar_sha256", "")),
            "WAL energy scalar hash mismatch",
        )
        successes[key] = terminal
    lifetime = {
        "attempt_started": len(starts),
        "attempt_succeeded": sum(event["event"] == "attempt_succeeded" for event in terminals.values()),
        "attempt_failed": sum(event["event"] == "attempt_failed" for event in terminals.values()),
        "attempt_open": len(set(starts) - set(terminals)),
        "accepted_unique_successes": len(successes),
        "retry_attempts": len(starts) - len({(str(event["stage"]), int(event["logical_index"])) for event in starts.values()}),
    }
    require(lifetime["attempt_started"] <= int(expected_header["maximum_attempt_started_records"]), "WAL lifetime attempt bound exceeded")
    require(lifetime["attempt_failed"] == 0 and lifetime["attempt_open"] == 0, "failed/open WAL requires a superseding authorization and output")
    require(lifetime["retry_attempts"] == 0, "same-authorization retry is forbidden")
    return events, {
        "starts": starts,
        "terminals": terminals,
        "successes": successes,
        "publications": publications,
        "lifetime": lifetime,
    }


class EnergyOnlyBackend:
    def __init__(self, topology: Path):
        self.backend = OpenMMBackend(
            topology,
            platform_preference=("CPU",),
            precision="double",
            label="bb_irivm_nf8982_section_energy_only",
        )
        self.raw_energy_attempts = 0
        self.raw_energy_successes = 0

    def evaluate_energy(self, coordinate: np.ndarray) -> float:
        from openmm.unit import angstrom, kilojoule_per_mole

        point = np.asarray(coordinate, dtype=np.float64)
        require(point.shape == (DIMENSION,), "energy coordinate shape mismatch")
        require(np.all(np.isfinite(point)), "energy coordinate is non-finite")
        self.backend.context.setPositions(point.reshape(N_ATOMS, 3) * angstrom)
        self.raw_energy_attempts += 1
        state = self.backend.context.getState(getEnergy=True, getForces=False)
        value = float(
            state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        ) / self.backend.KJ_PER_MOL_PER_EV
        require(np.isfinite(value), "non-finite diagnostic energy")
        self.raw_energy_successes += 1
        return value


def expected_wal_header(state: dict, stage: str) -> dict:
    require(stage in {SURFACE_STAGE, PATH_STAGE}, "unknown WAL stage")
    maximum = SURFACE_FRESH_CALLS if stage == SURFACE_STAGE else PATH_INTERNAL_CALLS
    return {
        "schema": WAL_SCHEMA,
        "status": "OPEN_APPEND_ONLY_FSYNC",
        "stage": stage,
        "contract": str(state["contract_path"]),
        "contract_sha256": state["contract_sha256"],
        "authorization": str(state["authorization_path"]),
        "authorization_sha256": state["authorization_sha256"],
        "coordinate_manifest": str(state["coordinate_manifest"]),
        "coordinate_manifest_sha256": state["coordinate_manifest_sha256"],
        "energy_call_policy": str(state["energy_call_policy"]),
        "energy_call_policy_sha256": sha256_file(state["energy_call_policy"]),
        "endpoint_cache": str(state["endpoint"]["path"]),
        "endpoint_cache_sha256": state["endpoint"]["sha256"],
        "fresh_energy_only_attempts_planned": maximum,
        "maximum_attempt_started_records": maximum,
        "force_calls_planned": 0,
        "optimization_N_F_contribution": 0,
    }


def load_contract(
    contract_path: Path,
    authorization_path: Path,
    host_role: str,
    output: Path,
    stage: str,
) -> dict:
    require(stage in {SURFACE_STAGE, PATH_STAGE}, "unknown requested stage")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(contract.get("schema") == CONTRACT_SCHEMA and contract.get("status") == CONTRACT_STATUS, "energy contract is not frozen")
    require(contract.get("host_role") == host_role == "local", "host role mismatch")
    require(int(contract.get("jobs", -1)) == 1, "jobs must equal one")
    require(resolve_binding(contract_path, contract["evaluator"], "evaluator") == Path(__file__).resolve(), "evaluator path mismatch")
    require(resolve_binding(contract_path, contract["bb_core"], "bb_core") == (SRC / "bb_core.py").resolve(), "bb_core path mismatch")
    topology = resolve_binding(contract_path, contract["topology"], "topology")
    require(topology == (ROOT / "inputs/complex_fixed.pdb").resolve(strict=True), "noncanonical topology")
    require(sha256_file(topology) == OpenMMBackend.EXPECTED_PDB_SHA256, "topology/backend identity mismatch")
    coordinate_manifest = resolve_binding(contract_path, contract["coordinate_manifest"], "coordinate manifest")
    data = validate_coordinate_manifest(coordinate_manifest, str(contract["coordinate_manifest"]["sha256"]))
    require(resolve_binding(contract_path, contract["coordinate_freeze_contract"], "coordinate freeze contract") == data["freeze_contract"], "coordinate freeze contract mismatch")
    call_policy_path = resolve_binding(contract_path, contract["energy_call_policy"], "energy call policy")
    call_policy = json.loads(call_policy_path.read_text(encoding="utf-8"))
    require(call_policy.get("schema") == CALL_POLICY_SCHEMA and call_policy.get("status") == CALL_POLICY_STATUS, "energy call policy is not frozen")
    require(int(call_policy.get("physical_energy_or_force_calls_started", -1)) == 0, "energy policy was not frozen before physical calls")
    endpoint_path = resolve_binding(contract_path, contract["endpoint_cache"], "endpoint cache")
    require(endpoint_path == ENDPOINT_CACHE.resolve(strict=True), "noncanonical endpoint cache")
    endpoint = load_endpoint_cache(endpoint_path, data, str(contract["endpoint_cache"]["sha256"]))
    locks = [Path(value).expanduser().resolve() for value in contract.get("host_local_lock_paths", [])]
    require(len(locks) == 2 and set(locks) == CANONICAL_LOCKS, "contract lacks canonical two-lock set")
    policy = contract.get("physical_call_policy", {})
    require(
        policy
        == {
            "surface_energy_values_stored": SURFACE_CALLS,
            "surface_fresh_energy_only_attempts": SURFACE_FRESH_CALLS,
            "surface_inherited_fixed_endpoint_values": 2,
            "exact_original_path_internal_fresh_energy_only_attempts": PATH_INTERNAL_CALLS,
            "exact_original_path_inherited_fixed_endpoint_values": 2,
            "total_fresh_energy_only_attempts": TOTAL_FRESH_CALLS,
            "force_calls": 0,
            "getState_getEnergy": True,
            "getState_getForces": False,
            "optimization_N_F_contribution": 0,
        },
        "physical call policy mismatch",
    )
    expected_output = Path(str(contract.get("result_output", ""))).expanduser().resolve()
    path_output = Path(str(contract.get("original_path32_result_output", ""))).expanduser().resolve()
    require(path_output != expected_output and path_output.suffix == ".npz", "invalid original path stage output")
    requested_output = expected_output if stage == SURFACE_STAGE else path_output
    require(output == requested_output, "stage output differs from frozen contract")

    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    require(authorization.get("schema") == AUTH_SCHEMA and authorization.get("status") == AUTH_STATUS and authorization.get("enabled") is True, "physical execution is not authorized")
    require(authorization.get("host_role") == host_role and int(authorization.get("jobs", -1)) == 1, "authorization host/jobs mismatch")
    require(authorization.get("result_output") == str(expected_output), "authorization surface output mismatch")
    require(
        authorization.get("original_path32_result_output") == str(path_output),
        "authorization original-path output mismatch",
    )
    require(
        authorization.get("authorized_stages") == [PATH_STAGE, SURFACE_STAGE],
        "authorization stages mismatch",
    )
    require(authorization.get("contract") == file_binding(contract_path), "authorization contract binding mismatch")
    require(authorization.get("evaluator") == file_binding(Path(__file__)), "authorization evaluator binding mismatch")
    preauth_path = resolve_binding(
        authorization_path,
        authorization.get("zero_openmm_preflight", {}),
        "zero-OpenMM preauthorization preflight",
    )
    preauth = json.loads(preauth_path.read_text(encoding="utf-8"))
    require(
        preauth.get("schema") == PREAUTH_SCHEMA
        and preauth.get("status") == PREAUTH_STATUS,
        "preauthorization preflight identity mismatch",
    )
    require(preauth.get("contract") == file_binding(contract_path), "preflight contract binding mismatch")
    require(preauth.get("evaluator") == file_binding(Path(__file__)), "preflight evaluator binding mismatch")
    require(
        preauth_path
        == Path(str(contract.get("zero_openmm_preflight_output", ""))).expanduser().resolve(),
        "preflight path differs from frozen contract",
    )
    require(preauth.get("original_path32_result_output") == str(path_output), "preflight path-output mismatch")
    require(preauth.get("surface_result_output") == str(expected_output), "preflight surface-output mismatch")
    require(preauth.get("canonical_two_lock_nonblocking_probe", {}).get("status") == "PASS", "preflight lock probe did not pass")
    require(int(preauth.get("physical_calls_started", -1)) == 0, "preflight was not zero-OpenMM")
    require(int(authorization.get("maximum_energy_only_attempts", -1)) == TOTAL_FRESH_CALLS, "authorization total attempt bound mismatch")
    require(int(authorization.get("maximum_surface_energy_only_attempts", -1)) == SURFACE_FRESH_CALLS, "authorization surface attempt bound mismatch")
    require(int(authorization.get("maximum_original_internal_energy_only_attempts", -1)) == PATH_INTERNAL_CALLS, "authorization path attempt bound mismatch")
    require(int(authorization.get("maximum_force_calls", -1)) == 0 and int(authorization.get("optimization_N_F_contribution", -1)) == 0, "authorization force/optimization policy mismatch")
    require(int(authorization.get("physical_calls_started_before_authorization", -1)) == 0, "authorization was not prospective")
    require(authorization.get("thread_policy") == {name: "1" for name in THREAD_VARS}, "authorization thread policy mismatch")
    require(
        authorization.get("canonical_two_lock_second_probe", {}).get("status") == "PASS"
        and set(authorization["canonical_two_lock_second_probe"].get("locks", []))
        == {str(path) for path in CANONICAL_LOCKS},
        "authorization canonical lock probe mismatch",
    )
    return {
        "contract": contract,
        "contract_path": contract_path,
        "contract_sha256": sha256_file(contract_path),
        "authorization": authorization,
        "authorization_path": authorization_path,
        "authorization_sha256": sha256_file(authorization_path),
        "preauthorization_preflight": preauth_path,
        "topology": topology,
        "coordinate_manifest": coordinate_manifest,
        "coordinate_manifest_sha256": sha256_file(coordinate_manifest),
        "data": data,
        "energy_call_policy": call_policy_path,
        "endpoint": endpoint,
        "locks": locks,
        "output": expected_output,
        "path_output": path_output,
        "requested_stage": stage,
        "requested_output": requested_output,
    }


def stage_artifacts(state: dict, stage: str) -> tuple[Path, Path, Path]:
    require(stage in {SURFACE_STAGE, PATH_STAGE}, "unknown energy stage")
    output = state["output"] if stage == SURFACE_STAGE else state["path_output"]
    audit = output.with_suffix(".audit.json")
    wal = output.with_suffix(".energy.wal.jsonl")
    return output, audit, wal


def stage_maximum_attempts(stage: str) -> int:
    require(stage in {SURFACE_STAGE, PATH_STAGE}, "unknown energy stage")
    return SURFACE_FRESH_CALLS if stage == SURFACE_STAGE else PATH_INTERNAL_CALLS


def validate_publication(
    wal_state: Mapping[str, object],
    output: Path,
    stage: str,
    *,
    required: bool,
) -> None:
    publications = list(wal_state["publications"])
    if not publications:
        require(not required, "completed result lacks WAL publication footer")
        return
    require(output.exists(), "WAL publication footer points to missing result")
    event = publications[0]
    require(event.get("event") == "result_published", "invalid publication event")
    require(event.get("stage") == stage, "publication stage mismatch")
    require(event.get("result") == str(output), "publication result path mismatch")
    require(event.get("result_sha256") == sha256_file(output), "publication result SHA-256 mismatch")
    require(int(event.get("result_bytes", -1)) == output.stat().st_size, "publication result size mismatch")
    require(event.get("result_schema") == RESULT_SCHEMA, "publication result schema mismatch")


def _inspect_existing_stage_wal(state: dict, stage: str) -> dict | None:
    output, audit, wal = stage_artifacts(state, stage)
    if not wal.exists():
        require(not output.exists() and not audit.exists(), f"orphan {stage} result/audit lacks WAL")
        return None
    _, wal_state = read_wal(wal, expected_wal_header(state, stage))
    # A crash can occur after the atomically published NPZ and before the WAL
    # footer.  That state is recoverable only after full NPZ/WAL validation in
    # run(); an already published audit always requires the footer.
    validate_publication(wal_state, output, stage, required=bool(audit.exists()))
    require(not audit.exists() or output.exists(), f"orphan {stage} audit lacks result")
    return wal_state


def preflight(state: dict, stage: str) -> dict:
    output, audit, wal = stage_artifacts(state, stage)
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    required_free = MINIMUM_FREE_BYTES_RESERVE + (0 if output.exists() else ESTIMATED_FINAL_BYTES)
    require(free >= required_free, f"insufficient free disk: {free} < {required_free}")
    stage_states = {
        candidate: _inspect_existing_stage_wal(state, candidate)
        for candidate in (PATH_STAGE, SURFACE_STAGE)
    }
    if stage == SURFACE_STAGE:
        require(
            state["path_output"].exists()
            and state["path_output"].with_suffix(".audit.json").exists(),
            "surface stage requires the independently published original_path32 result and audit",
        )
        validate_result(
            state["path_output"],
            state,
            PATH_STAGE,
            stage_states[PATH_STAGE],
        )
        validate_audit(
            state["path_output"].with_suffix(".audit.json"),
            state,
            PATH_STAGE,
            stage_states[PATH_STAGE],
        )
    with acquire_two_locks(state["locks"]) as probed_locks:
        lock_probe = list(probed_locks)
    maximum = stage_maximum_attempts(stage)
    existing = stage_states[stage]
    total_existing_starts = sum(
        candidate_state["lifetime"]["attempt_started"]
        for candidate_state in stage_states.values()
        if candidate_state is not None
    )
    require(total_existing_starts <= TOTAL_FRESH_CALLS, "authorization lifetime total attempt bound exceeded")
    return {
        "status": "ZERO_OPENMM_PREFLIGHT_PASS",
        "stage": stage,
        "stage_energy_values": SURFACE_CALLS if stage == SURFACE_STAGE else N_IMAGES,
        "stage_fresh_energy_only_attempts_maximum": maximum,
        "total_fresh_energy_only_attempts_authorized": TOTAL_FRESH_CALLS,
        "surface_inherited_fixed_endpoint_values": 2 if stage == SURFACE_STAGE else 0,
        "path_inherited_fixed_endpoint_values": 2 if stage == PATH_STAGE else 0,
        "force_calls": 0,
        "optimization_N_F_contribution": 0,
        "canonical_two_lock_nonblocking_probe": {"status": "PASS", "locks": lock_probe},
        "result_exists_pending_audit": output.exists(),
        "audit_already_exists": audit.exists(),
        "resume_wal_exists": wal.exists(),
        "stage_existing_attempt_started": 0 if existing is None else existing["lifetime"]["attempt_started"],
        "authorization_existing_attempt_started_across_both_stages": total_existing_starts,
        "observed_free_bytes": free,
        "required_free_bytes": required_free,
    }


def _attempt_energy(wal: DurableWal, backend, item: dict, prior_start_count: int) -> float:
    attempt_id = f"{item['stage']}:{int(item['logical_index']):04d}:attempt{prior_start_count + 1:03d}"
    base = {
        "attempt_id": attempt_id,
        "stage": item["stage"],
        "logical_index": int(item["logical_index"]),
        "coordinate_sha256": item["coordinate_sha256"],
    }
    for key in ("center_index", "normal_index", "original_image_index"):
        if key in item:
            base[key] = int(item[key])
    wal.append({"event": "attempt_started", "utc": utc_now(), **base})
    try:
        energy = float(backend.evaluate_energy(item["coordinate"]))
        require(np.isfinite(energy), "backend returned non-finite energy")
    except Exception as error:
        wal.append({"event": "attempt_failed", "utc": utc_now(), **base, "error": repr(error)})
        raise
    wal.append(
        {
            "event": "attempt_succeeded",
            "utc": utc_now(),
            **base,
            "energy_eV": energy,
            "energy_scalar_sha256": sha256_array(np.asarray(energy, dtype=np.float64)),
        }
    )
    return energy


def _expected_stage_items(state: dict, stage: str) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    items = list(_logical_coordinates(state["data"], stage))
    expected_count = stage_maximum_attempts(stage)
    require(len(items) == expected_count, "logical coordinate inventory count mismatch")
    mapping = {(item["stage"], int(item["logical_index"])): item for item in items}
    require(len(mapping) == expected_count, "logical coordinate inventory is not unique")
    return items, mapping


def validate_stage_wal(state: dict, stage: str, wal_state: dict, *, complete: bool) -> None:
    _, expected = _expected_stage_items(state, stage)
    for event in wal_state["starts"].values():
        key = (str(event.get("stage", "")), int(event.get("logical_index", -1)))
        require(key in expected, "WAL start refers to an unregistered logical coordinate")
        item = expected[key]
        require(event.get("coordinate_sha256") == item["coordinate_sha256"], "WAL start coordinate hash mismatch")
        for name in ("center_index", "normal_index", "original_image_index"):
            require(event.get(name) == item.get(name), f"WAL start {name} mismatch")
    require(set(wal_state["successes"]).issubset(expected), "WAL success inventory is outside frozen stage")
    for key, event in wal_state["successes"].items():
        item = expected[key]
        require(event.get("coordinate_sha256") == item["coordinate_sha256"], "WAL success coordinate hash mismatch")
        for name in ("center_index", "normal_index", "original_image_index"):
            require(event.get(name) == item.get(name), f"WAL success {name} mismatch")
    if complete:
        require(set(wal_state["successes"]) == set(expected), "stage WAL is incomplete")
        require(
            wal_state["lifetime"]["attempt_started"] == stage_maximum_attempts(stage),
            "completed stage attempt count mismatch",
        )


def _common_result_payload(state: dict, stage: str, held_locks: list[str]) -> dict:
    _, _, wal_path = stage_artifacts(state, stage)
    return {
        "schema": np.asarray(RESULT_SCHEMA),
        "status": np.asarray("PASS"),
        "stage": np.asarray(stage),
        "method": np.asarray("IR-IVM"),
        "center_source_optimization_N_F": np.asarray(8982, dtype=np.int64),
        "center_source_call_per_image": np.asarray(318, dtype=np.int64),
        "coordinate_manifest": np.asarray(str(state["coordinate_manifest"])),
        "coordinate_manifest_sha256": np.asarray(state["coordinate_manifest_sha256"]),
        "energy_call_policy": np.asarray(str(state["energy_call_policy"])),
        "energy_call_policy_sha256": np.asarray(sha256_file(state["energy_call_policy"])),
        "endpoint_cache": np.asarray(str(state["endpoint"]["path"])),
        "endpoint_cache_sha256": np.asarray(state["endpoint"]["sha256"]),
        "endpoint_coordinate_pair_sha256": np.asarray(state["endpoint"]["coordinate_sha256"]),
        "endpoint_energy_pair_sha256": np.asarray(state["endpoint"]["energy_sha256"]),
        "contract": np.asarray(str(state["contract_path"])),
        "contract_sha256": np.asarray(state["contract_sha256"]),
        "authorization": np.asarray(str(state["authorization_path"])),
        "authorization_sha256": np.asarray(state["authorization_sha256"]),
        "preauthorization_preflight": np.asarray(str(state["preauthorization_preflight"])),
        "preauthorization_preflight_sha256": np.asarray(sha256_file(state["preauthorization_preflight"])),
        "evaluator_sha256": np.asarray(sha256_file(Path(__file__))),
        "bb_core_sha256": np.asarray(sha256_file(SRC / "bb_core.py")),
        "topology_sha256": np.asarray(sha256_file(state["topology"])),
        "energy_wal": np.asarray(str(wal_path)),
        "held_host_locks": np.asarray(held_locks),
        "fresh_fixed_endpoint_energy_calls": np.asarray(0, dtype=np.int64),
        "force_calls": np.asarray(0, dtype=np.int64),
        "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
    }


def build_result_payload(state: dict, stage: str, wal_state: dict, held_locks: list[str]) -> dict:
    validate_stage_wal(state, stage, wal_state, complete=True)
    payload = _common_result_payload(state, stage, held_locks)
    successes = wal_state["successes"]
    endpoint_energy = state["endpoint"]["energies"]
    if stage == PATH_STAGE:
        coordinates = state["data"]["original_path32"]
        coordinate_hashes = np.asarray([sha256_array(point) for point in coordinates])
        energy = np.empty(N_IMAGES, dtype=np.float64)
        energy[[0, -1]] = endpoint_energy
        for logical_index in range(PATH_INTERNAL_CALLS):
            energy[logical_index + 1] = float(successes[(stage, logical_index)]["energy_eV"])
        fresh_mask = np.zeros(N_IMAGES, dtype=np.bool_)
        fresh_mask[1:-1] = True
        inherited_mask = ~fresh_mask
        lineage = np.full(N_IMAGES, "fresh_exact_original_path_internal", dtype="<U48")
        lineage[[0, -1]] = "inherited_canonical_endpoint_cache"
        payload.update(
            {
                "original_path32_A": coordinates,
                "original_path32_sha256": np.asarray(state["data"]["original_path32_sha256"]),
                "original_path32_coordinate_sha256": coordinate_hashes,
                "original_path32_energy_eV": energy,
                "original_path32_energy_sha256": np.asarray(sha256_array(energy)),
                "fresh_evaluated_mask": fresh_mask,
                "inherited_endpoint_mask": inherited_mask,
                "energy_value_lineage": lineage,
                "stored_energy_values": np.asarray(N_IMAGES, dtype=np.int64),
                "fresh_energy_only_attempts": np.asarray(PATH_INTERNAL_CALLS, dtype=np.int64),
                "inherited_endpoint_energy_values": np.asarray(2, dtype=np.int64),
            }
        )
    else:
        coordinates = state["data"]["surface_coordinates"]
        energy = np.empty((POINTS, NORMAL_POINTS), dtype=np.float64)
        fresh_mask = np.ones((POINTS, NORMAL_POINTS), dtype=np.bool_)
        fresh_mask[0, ZERO_INDEX] = False
        fresh_mask[-1, ZERO_INDEX] = False
        inherited_mask = ~fresh_mask
        lineage = np.full((POINTS, NORMAL_POINTS), "fresh_unrelaxed_cartesian_surface", dtype="<U48")
        lineage[0, ZERO_INDEX] = "inherited_canonical_endpoint_cache"
        lineage[-1, ZERO_INDEX] = "inherited_canonical_endpoint_cache"
        energy[0, ZERO_INDEX] = endpoint_energy[0]
        energy[-1, ZERO_INDEX] = endpoint_energy[1]
        for item in _logical_coordinates(state["data"], stage):
            i = int(item["center_index"])
            j = int(item["normal_index"])
            energy[i, j] = float(successes[(stage, int(item["logical_index"]))]["energy_eV"])
        path_result = state["path_output"]
        path_audit = path_result.with_suffix(".audit.json")
        require(path_result.exists() and path_audit.exists(), "surface result requires published original-path result")
        payload.update(
            {
                "surface_coordinates_A": coordinates,
                "surface_coordinate_sha256": state["data"]["cell_hashes"],
                "surface_energy_eV": energy,
                "surface_energy_sha256": np.asarray(sha256_array(energy)),
                "center_path59_A": state["data"]["center"],
                "normal_coordinate_rms_A": state["data"]["n_grid"],
                "zero_normal_index": np.asarray(ZERO_INDEX, dtype=np.int64),
                "fresh_evaluated_mask": fresh_mask,
                "inherited_endpoint_mask": inherited_mask,
                "energy_value_lineage": lineage,
                "stored_energy_values": np.asarray(SURFACE_CALLS, dtype=np.int64),
                "fresh_energy_only_attempts": np.asarray(SURFACE_FRESH_CALLS, dtype=np.int64),
                "inherited_endpoint_energy_values": np.asarray(2, dtype=np.int64),
                "original_path32_result": np.asarray(str(path_result)),
                "original_path32_result_sha256": np.asarray(sha256_file(path_result)),
                "original_path32_audit": np.asarray(str(path_audit)),
                "original_path32_audit_sha256": np.asarray(sha256_file(path_audit)),
            }
        )
    require(np.all(np.isfinite(energy)), "non-finite final energy")
    require(np.array_equal(fresh_mask ^ inherited_mask, np.ones_like(fresh_mask)), "energy masks do not partition values")
    return payload


def validate_result(path: Path, state: dict, stage: str, wal_state: dict | None) -> None:
    require(wal_state is not None, "result validation requires its WAL")
    validate_stage_wal(state, stage, wal_state, complete=True)
    expected = build_result_payload(
        state,
        stage,
        wal_state,
        [str(path) for path in sorted(state["locks"], key=str)],
    )
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(expected), "result field inventory mismatch")
        for key, expected_value in expected.items():
            actual = np.asarray(saved[key])
            wanted = np.asarray(expected_value)
            require(actual.dtype == wanted.dtype, f"result {key} dtype mismatch")
            require(np.array_equal(actual, wanted), f"result {key} value mismatch")


def publication_record(output: Path, stage: str) -> dict:
    return {
        "event": "result_published",
        "utc": utc_now(),
        "stage": stage,
        "result": str(output),
        "result_sha256": sha256_file(output),
        "result_bytes": output.stat().st_size,
        "result_schema": RESULT_SCHEMA,
    }


def build_audit(
    state: dict,
    stage: str,
    wal_state: dict,
    held_locks: list[str],
    preflight_report: dict,
    fresh: int,
    reused: int,
) -> dict:
    output, _, wal_path = stage_artifacts(state, stage)
    return {
        "schema": AUDIT_SCHEMA,
        "status": "PASS",
        "stage": stage,
        "completed_utc": utc_now(),
        "contract": file_binding(state["contract_path"]),
        "authorization": file_binding(state["authorization_path"]),
        "preauthorization_preflight": file_binding(state["preauthorization_preflight"]),
        "coordinate_manifest": file_binding(state["coordinate_manifest"]),
        "energy_call_policy": file_binding(state["energy_call_policy"]),
        "endpoint_cache": file_binding(state["endpoint"]["path"]),
        "result": file_binding(output),
        "energy_wal": file_binding(wal_path),
        "held_host_locks": held_locks,
        "stored_energy_values": SURFACE_CALLS if stage == SURFACE_STAGE else N_IMAGES,
        "fresh_energy_only_attempts_lineage": stage_maximum_attempts(stage),
        "inherited_endpoint_energy_values": 2,
        "fresh_fixed_endpoint_energy_calls": 0,
        "fresh_energy_only_calls_this_invocation": fresh,
        "reused_successful_energy_only_calls_this_invocation": reused,
        "energy_WAL_lifetime": wal_state["lifetime"],
        "force_calls": 0,
        "optimization_N_F_contribution": 0,
        "endpoint_policy": "two byte-identical fixed endpoint values inherited from the canonical endpoint cache; no fresh endpoint call",
        "preflight": preflight_report,
    }


def validate_audit(path: Path, state: dict, stage: str, wal_state: dict | None) -> dict:
    require(wal_state is not None, "audit validation requires its WAL")
    output, _, wal_path = stage_artifacts(state, stage)
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema") == AUDIT_SCHEMA and payload.get("status") == "PASS", "audit identity mismatch")
    require(payload.get("stage") == stage, "audit stage mismatch")
    for key, source in (
        ("contract", state["contract_path"]),
        ("authorization", state["authorization_path"]),
        ("preauthorization_preflight", state["preauthorization_preflight"]),
        ("coordinate_manifest", state["coordinate_manifest"]),
        ("energy_call_policy", state["energy_call_policy"]),
        ("endpoint_cache", state["endpoint"]["path"]),
        ("result", output),
        ("energy_wal", wal_path),
    ):
        require(payload.get(key) == file_binding(source), f"audit {key} binding mismatch")
    require(payload.get("held_host_locks") == [str(item) for item in sorted(state["locks"], key=str)], "audit lock set mismatch")
    require(int(payload.get("stored_energy_values", -1)) == (SURFACE_CALLS if stage == SURFACE_STAGE else N_IMAGES), "audit stored-value count mismatch")
    require(int(payload.get("fresh_energy_only_attempts_lineage", -1)) == stage_maximum_attempts(stage), "audit attempt lineage mismatch")
    require(int(payload.get("inherited_endpoint_energy_values", -1)) == 2, "audit inherited endpoint count mismatch")
    require(int(payload.get("fresh_fixed_endpoint_energy_calls", -1)) == 0, "audit fresh endpoint call mismatch")
    require(payload.get("energy_WAL_lifetime") == wal_state["lifetime"], "audit/WAL lifetime mismatch")
    require(int(payload.get("force_calls", -1)) == 0 and int(payload.get("optimization_N_F_contribution", -1)) == 0, "audit force/optimization mismatch")
    return payload


def run(state: dict, stage: str, backend_class=EnergyOnlyBackend) -> dict:
    preflight_report = preflight(state, stage)
    output, audit_path, wal_path = stage_artifacts(state, stage)
    fresh = 0
    reused = 0
    backend = None
    with acquire_two_locks(state["locks"]) as held_locks:
        wal = DurableWal(wal_path, expected_wal_header(state, stage))
        try:
            _, wal_state = read_wal(wal_path, expected_wal_header(state, stage))
            validate_stage_wal(state, stage, wal_state, complete=False)
            if audit_path.exists():
                require(output.exists(), "orphan audit lacks result")
                validate_publication(wal_state, output, stage, required=True)
                validate_result(output, state, stage, wal_state)
                return validate_audit(audit_path, state, stage, wal_state)

            if not output.exists():
                require(not wal_state["publications"], "publication footer exists without result")
                for item in _logical_coordinates(state["data"], stage):
                    key = (stage, int(item["logical_index"]))
                    existing = wal_state["successes"].get(key)
                    if existing is not None:
                        require(existing["coordinate_sha256"] == item["coordinate_sha256"], "reused WAL coordinate mismatch")
                        reused += 1
                        continue
                    if backend is None:
                        backend = backend_class(state["topology"])
                    _attempt_energy(wal, backend, item, 0)
                    fresh += 1
                _, wal_state = read_wal(wal_path, expected_wal_header(state, stage))
                validate_stage_wal(state, stage, wal_state, complete=True)
                payload = build_result_payload(state, stage, wal_state, held_locks)
                atomic_npz_no_overwrite(output, payload)
            else:
                validate_stage_wal(state, stage, wal_state, complete=True)

            validate_result(output, state, stage, wal_state)
            if not wal_state["publications"]:
                wal.append(publication_record(output, stage))
                _, wal_state = read_wal(wal_path, expected_wal_header(state, stage))
            validate_publication(wal_state, output, stage, required=True)
            validate_result(output, state, stage, wal_state)
            if backend is not None:
                require(backend.raw_energy_attempts == fresh, "fresh energy attempt ledger mismatch")
                require(backend.raw_energy_successes == fresh, "fresh energy success ledger mismatch")
                require(backend.backend.raw_joint_attempts == 0, "energy-only evaluator touched joint force-attempt ledger")
                require(backend.backend.raw_joint_evaluations == 0, "energy-only evaluator touched joint force-success ledger")
        finally:
            wal.close()

        other_stage = PATH_STAGE if stage == SURFACE_STAGE else SURFACE_STAGE
        other_state = _inspect_existing_stage_wal(state, other_stage)
        total_starts = wal_state["lifetime"]["attempt_started"] + (
            0 if other_state is None else other_state["lifetime"]["attempt_started"]
        )
        require(total_starts <= TOTAL_FRESH_CALLS, "authorization total attempt bound exceeded")
        audit = build_audit(
            state,
            stage,
            wal_state,
            held_locks,
            preflight_report,
            fresh,
            reused,
        )
        audit["authorization_lifetime_attempt_started_across_both_stages"] = total_starts
        atomic_json_no_overwrite(audit_path, audit)
        return audit


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--host-role", choices=("local",), required=True)
    parser.add_argument("--stage", choices=(PATH_STAGE, SURFACE_STAGE), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    check_thread_caps()
    contract_path = args.contract.expanduser().resolve(strict=True)
    authorization_path = args.authorization.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    require(output.suffix == ".npz", "result output must be NPZ")
    state = load_contract(contract_path, authorization_path, args.host_role, output, args.stage)
    if args.preflight_only:
        print(json.dumps(preflight(state, args.stage), indent=2, sort_keys=True))
        return
    print(json.dumps(run(state, args.stage), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
