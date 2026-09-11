#!/usr/bin/env python3
"""Evaluate the discrete 32-image geometric-action comparison.

The reported quantity is the fixed-endpoint, path-dependent part of the
geometric action in Eq. ``intro-geometric-action`` of the manuscript,

    S_hat(path) = integral_path |grad V| d ell.

For the Barnase--Barstar Cartesian calculation ``G = Id`` and OpenMM stores
``force = -grad V``.  On the original 32 images only, the cumulative discrete
quantity is therefore the segmentwise trapezoidal rule

    S_i = sum_{k < i} 0.5 (|F_k| + |F_{k+1}|) |x_{k+1} - x_k|.

There is no path densification, interpolation, smoothing, mass weighting, or
energy-reference term.  The ODE--NEB data already contain exact joint E/F.
Only the 30 internal images of the selected IR-IVM path require new diagnostic
joint E/F calls; the two fixed endpoints are inherited from the canonical
cache.  Diagnostic calls never change the optimization Number of Forces.
"""

from __future__ import annotations

import argparse
import base64
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


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import bb_core as bb_core_module  # noqa: E402
from bb_core import OpenMMBackend  # noqa: E402


SCHEMA = "bb-discrete32-path-dependent-geometric-action-result-v1"
AUDIT_SCHEMA = "bb-discrete32-path-dependent-geometric-action-audit-v1"
WAL_SCHEMA = "bb-discrete32-path-dependent-geometric-action-joint-ef-wal-v1"
CONTRACT_SCHEMA = "bb-discrete32-path-dependent-geometric-action-contract-v1"
CONTRACT_STATUS = "FROZEN_BEFORE_FIRST_GEOMETRIC_ACTION_DIAGNOSTIC_CALL"
PREAUTH_SCHEMA = "bb-discrete32-path-dependent-geometric-action-preflight-v1"
PREAUTH_STATUS = "PASS_ZERO_OPENMM_BEFORE_AUTHORIZATION"
AUTH_SCHEMA = "bb-discrete32-path-dependent-geometric-action-authorization-v1"
AUTH_STATUS = "AUTHORIZED_AFTER_ZERO_OPENMM_PREFLIGHT"

N_IMAGES = 32
N_INTERNAL = 30
N_ATOMS = 3132
DIMENSION = N_ATOMS * 3
EV_TO_KCAL_PER_MOL = 23.0605
IR_OPTIMIZATION_NF = 8982
ODE_OPTIMIZATION_NF = 9990
IR_CALL_PER_IMAGE = 318
FRESH_IR_INTERNAL_JOINT_EF_CALLS = 30

IR_ENERGY_RESULT = ROOT / (
    "runs/diagnostics/"
    "bb_irivm_nf8982_original_path32_energy_20260902_r4.npz"
)
IR_ENERGY_AUDIT = IR_ENERGY_RESULT.with_suffix(".audit.json")
ENDPOINT_CACHE = ROOT / "inputs/endpoint_joint_ef_cache.npz"
ODE_SOURCE = ROOT / (
    "inputs/baselines/ode_neb_source_paths_nf240_480_990_9990/"
    "nf09990_batch_0333_EF30.npz"
)
TOPOLOGY = ROOT / "inputs/complex_fixed.pdb"
BB_CORE = ROOT / "src/bb_core.py"
FREEZER = ROOT / "scripts/freeze_bb_discrete32_geometric_action_contract.py"
AUTHORIZER = ROOT / "scripts/authorize_bb_discrete32_geometric_action.py"
PLOTTER = ROOT / "scripts/plot_bb_discrete32_geometric_action.py"
TEST_FILE = ROOT / "tests/test_bb_discrete32_geometric_action.py"
MANUSCRIPT = Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/main.tex")

CONTRACT = ROOT / (
    "registrations/diagnostics/"
    "BB_DISCRETE32_GEOMETRIC_ACTION_CONTRACT_20260902_R1.json"
)
PREAUTH = ROOT / (
    "registrations/diagnostics/"
    "BB_DISCRETE32_GEOMETRIC_ACTION_PREAUTH_PREFLIGHT_20260902_R1.json"
)
AUTHORIZATION = ROOT / (
    "registrations/diagnostics/"
    "BB_DISCRETE32_GEOMETRIC_ACTION_EXECUTION_AUTHORIZATION_20260902_R1.json"
)
RESULT = ROOT / (
    "runs/diagnostics/"
    "bb_ivm_ode_neb_discrete32_geometric_action_20260902_r1.npz"
)
RESULT_PUBLICATION = RESULT.with_suffix(".publication.json")
FIGURE = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/"
    "active_candidates/barnase_barstar_near_nf9990_discrete32_geometric_action.pdf"
)
PROVENANCE = FIGURE.with_name(FIGURE.stem + "_provenance.json")
FIGURE_PUBLICATION = FIGURE.with_name(FIGURE.stem + "_publication.json")

EXPECTED_IR_RESULT_SHA256 = (
    "7f654f6aeda97698f8919e9c5e0275531c85108cd83dc5b8eb94fb920ae8e58b"
)
EXPECTED_IR_AUDIT_SHA256 = (
    "e4a121e6cb01de5b46b24ae1c1407024196a3078c7696c5de92e5265ebdbb79d"
)
EXPECTED_IR_PATH_SHA256 = (
    "46f54b316a6138d294150c810c14b5eedb14921a4a02717f61239cce9b7965bb"
)
EXPECTED_IR_ENERGY_SHA256 = (
    "a0acb7ab91f9326d807ccb37eae6f5de9a626aeac36b217705c79a2b344b73c7"
)
EXPECTED_ENDPOINT_CACHE_SHA256 = (
    "5cd4e757c99f0784d8b0781d05d8083070c8cfaad2670ead26999b3a4134e99a"
)
EXPECTED_ENDPOINT_COORDINATE_SHA256 = (
    "5788e22f86df91ed5500e88a8dc053f7408bd2fb39414de121a6eec36b833db1"
)
EXPECTED_ENDPOINT_ENERGY_SHA256 = (
    "24884695bc3bbecbf1e5007bf8f245bfe6d7eb5bf81725e986b8dd0ece088a31"
)
EXPECTED_ENDPOINT_FORCE_SHA256 = (
    "c28ab9e8ad515beb3cfd5de8cc09af542ff9c80a96815d8d1b8b17d03fd8e01c"
)
EXPECTED_ODE_SOURCE_SHA256 = (
    "0f8c84b0e11fa191db9163c13fca8f10504aa685952a4c728b4043b5eba40ae0"
)
EXPECTED_ODE_PATH_SHA256 = (
    "4c40f30af44fdbda87684d887136bcb1b71f6b20f6785892d50fd10ffcfdff09"
)
EXPECTED_ODE_ENERGY_SHA256 = (
    "e0d31606c4a1081cee028a6d9ca553866441387027fe8309c46300eb3a61b61f"
)
EXPECTED_ODE_FORCE_SHA256 = (
    "4fa1e1ccddb17084f591c64ce8e3380e3a5e71e6ce353ce6913ccc0af39d0b39"
)

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
MINIMUM_FREE_BYTES = 2 * 1024**3
ESTIMATED_FINAL_BYTES = 64 * 1024**2
# The new joint E/F call stores its exact energy.  This tolerance is only a
# same-Hamiltonian identity check against an earlier energy-only call; it is
# not used in the action and allows harmless CPU reduction-order roundoff.
MAX_ENERGY_RECHECK_DRIFT_EV = 1.0e-6

FORMULA_ID = "original32-segment-trapezoid-int-norm-gradV-dell-v1"
FORMULA_TEXT = (
    "S_hat[0]=0; S_hat[i]=sum_{k<i} "
    "0.5*(||F_k||_2+||F_{k+1}||_2)*||x_{k+1}-x_k||_2"
)


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
    require(value.shape == (), f"{key} must be scalar")
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
    wrong = {name: os.environ.get(name) for name in THREAD_VARS if os.environ.get(name) != "1"}
    require(not wrong, f"all six thread caps must equal '1': {wrong}")


def check_imported_bb_core() -> None:
    imported = Path(bb_core_module.__file__).expanduser().resolve(strict=True)
    require(imported == BB_CORE.resolve(strict=True), f"noncanonical imported bb_core: {imported}")


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
            stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class DurableWal:
    def __init__(self, path: Path, header: Mapping[str, object]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header = dict(header)
        if self.path.exists():
            lines = self.path.read_text(encoding="utf-8").splitlines()
            require(lines and json.loads(lines[0]) == self.header, "existing joint E/F WAL header mismatch")
            self.stream = self.path.open("a", encoding="utf-8", buffering=1)
        else:
            self.stream = self.path.open("x", encoding="utf-8", buffering=1)
            self.append(self.header)
            _fsync_parent(self.path)

    def append(self, record: Mapping[str, object]) -> None:
        self.stream.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


def discrete_path_dependent_action(
    coordinates_A: np.ndarray,
    forces_eV_per_A: np.ndarray,
) -> dict:
    """Return the frozen original-image trapezoidal action and its ingredients."""

    coordinates = np.ascontiguousarray(np.asarray(coordinates_A, dtype=np.float64))
    forces = np.ascontiguousarray(np.asarray(forces_eV_per_A, dtype=np.float64))
    require(coordinates.ndim == 2 and coordinates.shape == forces.shape, "coordinate/force shape mismatch")
    require(coordinates.shape[0] >= 2, "action requires at least two images")
    require(np.all(np.isfinite(coordinates)) and np.all(np.isfinite(forces)), "non-finite action input")
    segment_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    require(np.all(np.isfinite(segment_lengths)) and np.all(segment_lengths > 0.0), "degenerate path segment")
    force_norms = np.linalg.norm(forces, axis=1)
    increments = 0.5 * (force_norms[:-1] + force_norms[1:]) * segment_lengths
    require(np.all(np.isfinite(increments)) and np.all(increments >= 0.0), "invalid action increment")
    arc = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(segment_lengths)))
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(increments)))
    return {
        "segment_length_A": segment_lengths,
        "force_norm_eV_per_A": force_norms,
        "action_increment_eV": increments,
        "arc_length_A": arc,
        "cumulative_action_eV": cumulative,
    }


def _load_ir() -> dict:
    require(sha256_file(IR_ENERGY_RESULT) == EXPECTED_IR_RESULT_SHA256, "IR energy result file drift")
    require(sha256_file(IR_ENERGY_AUDIT) == EXPECTED_IR_AUDIT_SHA256, "IR energy audit file drift")
    audit = json.loads(IR_ENERGY_AUDIT.read_text(encoding="utf-8"))
    require(audit.get("status") == "PASS", "IR energy audit is not PASS")
    require(audit.get("result") == file_binding(IR_ENERGY_RESULT), "IR energy audit/result binding mismatch")
    with np.load(IR_ENERGY_RESULT, allow_pickle=False) as saved:
        required = {
            "status",
            "center_source_optimization_N_F",
            "center_source_call_per_image",
            "original_path32_A",
            "original_path32_sha256",
            "original_path32_energy_eV",
            "original_path32_energy_sha256",
        }
        require(required.issubset(saved.files), "IR energy result fields missing")
        require(str(scalar(saved, "status")) == "PASS", "IR energy result is not PASS")
        require(int(scalar(saved, "center_source_optimization_N_F")) == IR_OPTIMIZATION_NF, "IR N_F drift")
        require(int(scalar(saved, "center_source_call_per_image")) == IR_CALL_PER_IMAGE, "IR call drift")
        path = np.ascontiguousarray(saved["original_path32_A"], dtype=np.float64).reshape(N_IMAGES, DIMENSION)
        energy = np.ascontiguousarray(saved["original_path32_energy_eV"], dtype=np.float64)
        embedded_path_hash = str(scalar(saved, "original_path32_sha256"))
        embedded_energy_hash = str(scalar(saved, "original_path32_energy_sha256"))
    require(path.shape == (N_IMAGES, DIMENSION) and energy.shape == (N_IMAGES,), "IR data shape drift")
    require(np.all(np.isfinite(path)) and np.all(np.isfinite(energy)), "IR data non-finite")
    require(embedded_path_hash == EXPECTED_IR_PATH_SHA256 == sha256_array(path), "IR path hash drift")
    require(embedded_energy_hash == EXPECTED_IR_ENERGY_SHA256 == sha256_array(energy), "IR energy hash drift")
    return {"path": path, "energy": energy}


def _load_endpoint() -> dict:
    require(sha256_file(ENDPOINT_CACHE) == EXPECTED_ENDPOINT_CACHE_SHA256, "endpoint cache file drift")
    with np.load(ENDPOINT_CACHE, allow_pickle=False) as saved:
        coordinates = np.ascontiguousarray(saved["coordinates_A"], dtype=np.float64)
        energy = np.ascontiguousarray(saved["energies_eV"], dtype=np.float64)
        force = np.ascontiguousarray(saved["forces_eV_per_A"], dtype=np.float64)
    require(coordinates.shape == (2, DIMENSION), "endpoint coordinate shape drift")
    require(energy.shape == (2,) and force.shape == (2, DIMENSION), "endpoint E/F shape drift")
    require(sha256_array(coordinates) == EXPECTED_ENDPOINT_COORDINATE_SHA256, "endpoint coordinate hash drift")
    require(sha256_array(energy) == EXPECTED_ENDPOINT_ENERGY_SHA256, "endpoint energy hash drift")
    require(sha256_array(force) == EXPECTED_ENDPOINT_FORCE_SHA256, "endpoint force hash drift")
    return {"coordinates": coordinates, "energy": energy, "force": force}


def _load_ode(endpoint: dict) -> dict:
    require(sha256_file(ODE_SOURCE) == EXPECTED_ODE_SOURCE_SHA256, "ODE--NEB source file drift")
    with np.load(ODE_SOURCE, allow_pickle=False) as saved:
        internal_coordinates = np.ascontiguousarray(saved["coordinates30_A"], dtype=np.float64).reshape(N_INTERNAL, DIMENSION)
        internal_energy = np.ascontiguousarray(saved["energies_eV"], dtype=np.float64)
        internal_force = np.ascontiguousarray(saved["forces_eV_per_A"], dtype=np.float64).reshape(N_INTERNAL, DIMENSION)
        nf = int(scalar(saved, "nf_after"))
    path = np.concatenate((endpoint["coordinates"][:1], internal_coordinates, endpoint["coordinates"][1:]), axis=0)
    energy = np.concatenate((endpoint["energy"][:1], internal_energy, endpoint["energy"][1:]))
    force = np.concatenate((endpoint["force"][:1], internal_force, endpoint["force"][1:]), axis=0)
    require(nf == ODE_OPTIMIZATION_NF, "ODE--NEB N_F drift")
    require(sha256_array(path) == EXPECTED_ODE_PATH_SHA256, "ODE--NEB path hash drift")
    require(sha256_array(energy) == EXPECTED_ODE_ENERGY_SHA256, "ODE--NEB energy hash drift")
    require(sha256_array(force) == EXPECTED_ODE_FORCE_SHA256, "ODE--NEB force hash drift")
    return {"path": path, "energy": energy, "force": force}


def load_sources() -> dict:
    ir = _load_ir()
    endpoint = _load_endpoint()
    require(np.array_equal(ir["path"][[0, -1]], endpoint["coordinates"]), "IR endpoints differ from canonical cache")
    require(np.array_equal(ir["energy"][[0, -1]], endpoint["energy"]), "IR endpoint energies differ from canonical cache")
    ode = _load_ode(endpoint)
    return {"ir": ir, "ode": ode, "endpoint": endpoint}


def expected_wal_header(state: Mapping[str, object]) -> dict:
    return {
        "schema": WAL_SCHEMA,
        "contract": file_binding(state["contract_path"]),
        "authorization": file_binding(state["authorization_path"]),
        "output": str(state["output"]),
        "source_path_sha256": EXPECTED_IR_PATH_SHA256,
        "fresh_internal_joint_ef_attempt_limit": FRESH_IR_INTERNAL_JOINT_EF_CALLS,
        "fixed_endpoint_fresh_attempt_limit": 0,
        "optimization_N_F_contribution": 0,
    }


def _decode_force(event: Mapping[str, object]) -> np.ndarray:
    require(event.get("force_dtype") == "<f8", "WAL force dtype mismatch")
    require(event.get("force_shape") == [DIMENSION], "WAL force shape mismatch")
    try:
        raw = base64.b64decode(str(event["force_le_f8_base64"]), validate=True)
    except Exception as error:
        raise ValueError("WAL force base64 is invalid") from error
    require(len(raw) == DIMENSION * 8, "WAL force byte count mismatch")
    force = np.frombuffer(raw, dtype="<f8").copy()
    require(sha256_array(force) == event.get("force_sha256"), "WAL force SHA-256 mismatch")
    require(np.all(np.isfinite(force)), "WAL force is non-finite")
    return force


def read_wal(path: Path, header: Mapping[str, object]) -> dict:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    require(lines and json.loads(lines[0]) == dict(header), "joint E/F WAL header mismatch")
    starts = {}
    terminals = {}
    for line in lines[1:]:
        event = json.loads(line)
        kind = event.get("event")
        require(kind in {"attempt_started", "attempt_succeeded", "attempt_failed"}, "unknown WAL event")
        attempt_id = str(event.get("attempt_id", ""))
        require(attempt_id, "WAL event lacks attempt_id")
        if kind == "attempt_started":
            require(attempt_id not in starts, "duplicate attempt start")
            starts[attempt_id] = event
        else:
            require(attempt_id in starts and attempt_id not in terminals, "orphan/duplicate attempt terminal")
            terminal = event
            for key in ("logical_index", "original_image_index", "coordinate_sha256"):
                require(terminal.get(key) == starts[attempt_id].get(key), "attempt start/terminal mismatch")
            terminals[attempt_id] = terminal
    open_attempts = set(starts).difference(terminals)
    failures = [event for event in terminals.values() if event["event"] == "attempt_failed"]
    require(
        not failures and not open_attempts,
        "failed/open R1 WAL cannot resume; create a separately versioned evaluator, "
        "contract, authorization, WAL, and output",
    )
    successes = {}
    for event in terminals.values():
        if event["event"] != "attempt_succeeded":
            continue
        index = int(event["logical_index"])
        require(0 <= index < N_INTERNAL and index not in successes, "duplicate/out-of-range WAL success")
        image_index = index + 1
        require(int(event["original_image_index"]) == image_index, "WAL image index mismatch")
        energy = float(event.get("energy_eV", np.nan))
        require(np.isfinite(energy), "WAL energy non-finite")
        require(sha256_array(np.asarray(energy)) == event.get("energy_scalar_sha256"), "WAL energy hash mismatch")
        _decode_force(event)
        successes[index] = event
    require(len(starts) <= FRESH_IR_INTERNAL_JOINT_EF_CALLS, "WAL attempt bound exceeded")
    return {
        "starts": starts,
        "terminals": terminals,
        "successes": successes,
        "attempt_started": len(starts),
        "attempt_succeeded": len(successes),
        "attempt_failed": len(failures),
        "open_attempts": len(open_attempts),
    }


def validate_success_prefix(
    wal_state: Mapping[str, object],
    ir_path: np.ndarray,
) -> None:
    """Bind every reusable WAL success to its frozen original IR image."""

    for logical_index, event in wal_state["successes"].items():
        expected_hash = sha256_array(ir_path[int(logical_index) + 1])
        require(
            event.get("coordinate_sha256") == expected_hash,
            f"reusable WAL/source coordinate mismatch at internal image {int(logical_index) + 1}",
        )


def load_contract(contract_path: Path, authorization_path: Path, output: Path) -> dict:
    check_imported_bb_core()
    contract_path = contract_path.expanduser().resolve(strict=True)
    authorization_path = authorization_path.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    require(contract_path == CONTRACT.resolve(), "noncanonical geometric-action contract")
    require(authorization_path == AUTHORIZATION.resolve(), "noncanonical geometric-action authorization")
    require(output == RESULT.resolve(), "noncanonical geometric-action output")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    require(contract.get("schema") == CONTRACT_SCHEMA and contract.get("status") == CONTRACT_STATUS, "contract identity mismatch")
    require(contract.get("host_role") == "local" and contract.get("jobs") == 1, "contract host/jobs mismatch")
    require(contract.get("evaluator") == file_binding(THIS_FILE), "contract evaluator binding mismatch")
    require(
        resolve_binding(contract_path, contract.get("contract_freezer", contract.get("code_and_topology", {}).get("contract_freezer", {})), "contract freezer")
        == FREEZER.resolve(strict=True),
        "noncanonical contract freezer",
    )
    require(resolve_binding(contract_path, contract.get("execution_authorizer", {}), "execution authorizer") == AUTHORIZER.resolve(strict=True), "noncanonical execution authorizer")
    code = contract.get("code_and_topology", {})
    require(isinstance(code, Mapping), "contract code/topology block missing")
    expected_code = {
        "evaluator": THIS_FILE,
        "contract_freezer": FREEZER,
        "execution_authorizer": AUTHORIZER,
        "plotter": PLOTTER,
        "tests": TEST_FILE,
        "bb_core": BB_CORE,
        "topology": TOPOLOGY,
    }
    for role, expected_path in expected_code.items():
        require(
            resolve_binding(contract_path, code.get(role, {}), role) == expected_path.resolve(strict=True),
            f"noncanonical {role}",
        )
    sources = contract.get("source_artifacts", {})
    require(isinstance(sources, Mapping), "contract source-artifact block missing")
    expected_sources = {
        "IVM_exact_path_and_energy": IR_ENERGY_RESULT,
        "IVM_exact_path_and_energy_audit": IR_ENERGY_AUDIT,
        "fixed_endpoint_joint_EF_cache": ENDPOINT_CACHE,
        "ODE_NEB_internal_joint_EF": ODE_SOURCE,
    }
    for role, expected_path in expected_sources.items():
        require(
            resolve_binding(contract_path, sources.get(role, {}), role) == expected_path.resolve(strict=True),
            f"noncanonical {role}",
        )
    manuscript = contract.get("manuscript", {})
    require(
        resolve_binding(contract_path, manuscript.get("source", {}), "manuscript")
        == MANUSCRIPT.resolve(strict=True),
        "noncanonical manuscript",
    )
    require(contract.get("result_output") == str(output), "contract output mismatch")
    require(contract.get("formula", {}).get("id") == FORMULA_ID and contract["formula"].get("text") == FORMULA_TEXT, "contract formula mismatch")
    require(contract.get("sampling") == {"images": 32, "segments": 31, "coordinate_metric": "unweighted Cartesian Euclidean", "densification": "none", "interpolation": "none", "smoothing": "none"}, "contract sampling mismatch")
    require(contract.get("source_paths") == {"IVM": {"optimization_N_F": 8982, "call_per_image": 318, "path_sha256": EXPECTED_IR_PATH_SHA256}, "ODE-NEB": {"optimization_N_F": 9990, "path_sha256": EXPECTED_ODE_PATH_SHA256}}, "contract source-path identity mismatch")
    require(
        contract.get("physical_call_policy")
        == {
            "IVM_internal_fresh_joint_EF_attempt_maximum": N_INTERNAL,
            "IVM_fixed_endpoint_fresh_joint_EF_attempt_maximum": 0,
            "ODE_NEB_fresh_joint_EF_attempt_maximum": 0,
            "joint_getState_getEnergy_and_getForces": True,
            "diagnostic_calls_only": True,
            "optimization_N_F_contribution": 0,
            "maximum_joint_vs_energy_only_abs_drift_eV": MAX_ENERGY_RECHECK_DRIFT_EV,
        },
        "contract physical-call policy mismatch",
    )
    require(contract.get("thread_policy") == {name: "1" for name in THREAD_VARS}, "contract thread policy mismatch")
    require(contract.get("joint_EF_WAL_output") == str(output.with_suffix(".joint_ef.wal.jsonl")), "contract WAL output mismatch")
    require(contract.get("result_publication_output") == str(RESULT_PUBLICATION.resolve()), "contract result-publication output mismatch")
    require(contract.get("audit_output") == str(output.with_suffix(".audit.json")), "contract audit output mismatch")
    require(contract.get("figure_output") == str(FIGURE.resolve()), "contract figure output mismatch")
    require(contract.get("figure_provenance_output") == str(PROVENANCE.resolve()), "contract provenance output mismatch")
    require(contract.get("figure_publication_output") == str(FIGURE_PUBLICATION.resolve()), "contract figure-publication output mismatch")
    require(contract.get("zero_openmm_preflight_output") == str(PREAUTH.resolve()), "contract preflight output mismatch")
    require(contract.get("execution_authorization_output") == str(AUTHORIZATION.resolve()), "contract authorization output mismatch")
    require(contract.get("physical_execution_requires_separate_authorization") is True, "contract authorization gate missing")
    require(authorization.get("schema") == AUTH_SCHEMA and authorization.get("status") == AUTH_STATUS and authorization.get("enabled") is True, "authorization identity mismatch")
    require(authorization.get("contract") == file_binding(contract_path), "authorization contract binding mismatch")
    require(authorization.get("evaluator") == file_binding(THIS_FILE), "authorization evaluator binding mismatch")
    require(authorization.get("authorizer") == file_binding(AUTHORIZER), "authorization authorizer binding mismatch")
    require(authorization.get("host_role") == "local" and authorization.get("jobs") == 1, "authorization host/jobs mismatch")
    require(authorization.get("result_output") == str(output), "authorization output mismatch")
    require(
        authorization.get("result_publication_output") == str(RESULT_PUBLICATION.resolve()),
        "authorization result-publication mismatch",
    )
    require(int(authorization.get("maximum_fresh_internal_joint_ef_attempts", -1)) == FRESH_IR_INTERNAL_JOINT_EF_CALLS, "authorization attempt bound mismatch")
    require(int(authorization.get("maximum_fresh_endpoint_joint_ef_attempts", -1)) == 0, "authorization endpoint-call policy mismatch")
    require(int(authorization.get("optimization_N_F_contribution", -1)) == 0, "authorization optimization accounting mismatch")
    require(int(authorization.get("maximum_fresh_ODE_NEB_joint_ef_attempts", -1)) == 0, "authorization ODE-call policy mismatch")
    require(authorization.get("thread_policy") == {name: "1" for name in THREAD_VARS}, "authorization thread policy mismatch")
    require(int(authorization.get("physical_calls_started_before_authorization", -1)) == 0, "authorization predates a physical call")
    require(int(authorization.get("openmm_contexts_constructed_before_authorization", -1)) == 0, "authorization predates an OpenMM Context")
    preauth_path = resolve_binding(authorization_path, authorization.get("zero_openmm_preflight", {}), "zero-OpenMM preflight")
    require(preauth_path == PREAUTH.resolve(strict=True), "noncanonical zero-OpenMM preflight")
    preauth = json.loads(preauth_path.read_text(encoding="utf-8"))
    require(preauth.get("schema") == PREAUTH_SCHEMA and preauth.get("status") == PREAUTH_STATUS, "preflight identity mismatch")
    require(preauth.get("contract") == file_binding(contract_path), "preflight contract binding mismatch")
    require(preauth.get("evaluator") == file_binding(THIS_FILE), "preflight evaluator binding mismatch")
    require(int(preauth.get("physical_calls_started", -1)) == 0 and int(preauth.get("openmm_contexts_constructed", -1)) == 0, "preflight was not zero-OpenMM")
    require(preauth.get("thread_policy") == {name: "1" for name in THREAD_VARS}, "preflight thread policy mismatch")
    require(preauth.get("result_output") == str(output), "preflight output mismatch")
    require(int(preauth.get("maximum_fresh_internal_joint_ef_attempts", -1)) == N_INTERNAL, "preflight attempt bound mismatch")
    require(int(preauth.get("maximum_fresh_endpoint_joint_ef_attempts", -1)) == 0, "preflight endpoint-call policy mismatch")
    require(int(preauth.get("optimization_N_F_contribution", -1)) == 0, "preflight optimization accounting mismatch")
    locks = [Path(value).expanduser().resolve() for value in contract.get("host_local_lock_paths", [])]
    require(len(set(locks)) == 2 and set(locks) == CANONICAL_LOCKS, "contract lock set mismatch")
    expected_lock_strings = {str(path) for path in CANONICAL_LOCKS}
    preauth_probe = preauth.get("canonical_two_lock_nonblocking_probe", {})
    require(
        preauth_probe.get("status") == "PASS" and set(preauth_probe.get("locks", [])) == expected_lock_strings,
        "preflight lock probe mismatch",
    )
    authorization_probe = authorization.get("canonical_two_lock_second_probe", {})
    require(
        authorization_probe.get("status") == "PASS" and set(authorization_probe.get("locks", [])) == expected_lock_strings,
        "authorization lock probe mismatch",
    )
    return {
        "contract": contract,
        "contract_path": contract_path,
        "authorization": authorization,
        "authorization_path": authorization_path,
        "preauth_path": preauth_path,
        "output": output,
        "wal": output.with_suffix(".joint_ef.wal.jsonl"),
        "result_publication": output.with_suffix(".publication.json"),
        "audit": output.with_suffix(".audit.json"),
        "locks": locks,
    }


def _build_payload(state: Mapping[str, object], sources: Mapping[str, object], successes: Mapping[int, Mapping[str, object]], held_locks: Sequence[str]) -> dict:
    require(set(successes) == set(range(N_INTERNAL)), "IR joint E/F inventory is incomplete")
    ir_path = sources["ir"]["path"]
    ir_energy_reference = sources["ir"]["energy"]
    endpoint = sources["endpoint"]
    ir_energy = np.empty(N_IMAGES, dtype=np.float64)
    ir_force = np.empty((N_IMAGES, DIMENSION), dtype=np.float64)
    ir_energy[[0, -1]] = endpoint["energy"]
    ir_force[[0, -1]] = endpoint["force"]
    for logical_index in range(N_INTERNAL):
        event = successes[logical_index]
        require(
            event["coordinate_sha256"] == sha256_array(ir_path[logical_index + 1]),
            f"IR WAL/source coordinate mismatch at internal image {logical_index + 1}",
        )
        ir_energy[logical_index + 1] = float(event["energy_eV"])
        ir_force[logical_index + 1] = _decode_force(event)
    energy_drift = np.abs(ir_energy - ir_energy_reference)
    require(float(np.max(energy_drift)) <= MAX_ENERGY_RECHECK_DRIFT_EV, "IR joint-call energy disagrees with frozen exact-energy result")
    methods = np.asarray(["IVM", "ODE-NEB"])
    nfs = np.asarray([IR_OPTIMIZATION_NF, ODE_OPTIMIZATION_NF], dtype=np.int64)
    paths = np.stack((ir_path, sources["ode"]["path"]), axis=0)
    energies = np.stack((ir_energy, sources["ode"]["energy"]), axis=0)
    forces = np.stack((ir_force, sources["ode"]["force"]), axis=0)
    reports = [discrete_path_dependent_action(paths[i], forces[i]) for i in range(2)]
    arc = np.stack([report["arc_length_A"] for report in reports])
    segment = np.stack([report["segment_length_A"] for report in reports])
    norms = np.stack([report["force_norm_eV_per_A"] for report in reports])
    increments = np.stack([report["action_increment_eV"] for report in reports])
    cumulative_eV = np.stack([report["cumulative_action_eV"] for report in reports])
    cumulative_kcal = cumulative_eV * EV_TO_KCAL_PER_MOL
    return {
        "schema": np.asarray(SCHEMA),
        "status": np.asarray("PASS"),
        "manuscript_quantity": np.asarray("fixed-endpoint path-dependent part of the geometric action"),
        "manuscript_equation_label": np.asarray("eq:intro-geometric-action"),
        "riemannian_metric": np.asarray("G=Id_9396"),
        "formula_id": np.asarray(FORMULA_ID),
        "formula_text": np.asarray(FORMULA_TEXT),
        "method": methods,
        "optimization_N_F": nfs,
        "source_call_per_image": np.asarray([IR_CALL_PER_IMAGE, -1], dtype=np.int64),
        "coordinates_A": paths,
        "energy_eV": energies,
        "force_eV_per_A": forces,
        "path_sha256": np.asarray([sha256_array(value) for value in paths]),
        "energy_sha256": np.asarray([sha256_array(value) for value in energies]),
        "force_sha256": np.asarray([sha256_array(value) for value in forces]),
        "segment_length_A": segment,
        "force_norm_eV_per_A": norms,
        "action_increment_eV": increments,
        "arc_length_A": arc,
        "cumulative_path_dependent_geometric_action_eV": cumulative_eV,
        "cumulative_path_dependent_geometric_action_kcal_per_mol": cumulative_kcal,
        "total_path_dependent_geometric_action_eV": cumulative_eV[:, -1],
        "total_path_dependent_geometric_action_kcal_per_mol": cumulative_kcal[:, -1],
        "number_of_original_images": np.asarray(N_IMAGES, dtype=np.int64),
        "number_of_original_segments": np.asarray(N_IMAGES - 1, dtype=np.int64),
        "densification": np.asarray("none"),
        "interpolation": np.asarray("none"),
        "smoothing": np.asarray("none"),
        "coordinate_metric": np.asarray("unweighted Cartesian Euclidean"),
        "mass_weighting": np.asarray(False),
        "energy_reference_used_by_metric": np.asarray(False),
        "ir_max_abs_joint_vs_energy_only_drift_eV": np.asarray(float(np.max(energy_drift))),
        "fresh_diagnostic_joint_EF_attempts_by_method": np.asarray([N_INTERNAL, 0], dtype=np.int64),
        "fresh_fixed_endpoint_joint_EF_attempts_by_method": np.asarray([0, 0], dtype=np.int64),
        "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
        "contract": np.asarray(str(state["contract_path"])),
        "contract_sha256": np.asarray(sha256_file(state["contract_path"])),
        "authorization": np.asarray(str(state["authorization_path"])),
        "authorization_sha256": np.asarray(sha256_file(state["authorization_path"])),
        "zero_openmm_preflight": np.asarray(str(state["preauth_path"])),
        "zero_openmm_preflight_sha256": np.asarray(sha256_file(state["preauth_path"])),
        "evaluator_sha256": np.asarray(sha256_file(THIS_FILE)),
        "bb_core_sha256": np.asarray(sha256_file(BB_CORE)),
        "topology_sha256": np.asarray(sha256_file(TOPOLOGY)),
        "ir_energy_result": np.asarray(str(IR_ENERGY_RESULT)),
        "ir_energy_result_sha256": np.asarray(EXPECTED_IR_RESULT_SHA256),
        "ir_energy_audit": np.asarray(str(IR_ENERGY_AUDIT)),
        "ir_energy_audit_sha256": np.asarray(EXPECTED_IR_AUDIT_SHA256),
        "endpoint_cache": np.asarray(str(ENDPOINT_CACHE)),
        "endpoint_cache_sha256": np.asarray(EXPECTED_ENDPOINT_CACHE_SHA256),
        "ode_source": np.asarray(str(ODE_SOURCE)),
        "ode_source_sha256": np.asarray(EXPECTED_ODE_SOURCE_SHA256),
        "joint_ef_wal": np.asarray(str(state["wal"])),
        "held_host_locks": np.asarray(list(held_locks)),
    }


def validate_result(path: Path, state: Mapping[str, object], sources: Mapping[str, object], wal_state: Mapping[str, object]) -> None:
    expected = _build_payload(state, sources, wal_state["successes"], [str(path) for path in sorted(state["locks"], key=str)])
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(expected), "result field inventory mismatch")
        for key, value in expected.items():
            actual = np.asarray(saved[key])
            wanted = np.asarray(value)
            require(actual.dtype == wanted.dtype and np.array_equal(actual, wanted), f"result {key} mismatch")


def expected_audit_fields(state: Mapping[str, object], wal_state: Mapping[str, object]) -> dict:
    return {
        "schema": AUDIT_SCHEMA,
        "status": "PASS",
        "contract": file_binding(state["contract_path"]),
        "authorization": file_binding(state["authorization_path"]),
        "zero_openmm_preflight": file_binding(state["preauth_path"]),
        "result": file_binding(state["output"]),
        "result_publication": file_binding(state["result_publication"]),
        "joint_ef_wal": file_binding(state["wal"]),
        "formula_id": FORMULA_ID,
        "formula_text": FORMULA_TEXT,
        "original_images_per_path": 32,
        "direct_segments_per_path": 31,
        "densification": "none",
        "interpolation": "none",
        "fresh_IR_internal_joint_EF_attempts": wal_state["attempt_started"],
        "successful_IR_internal_joint_EF_attempts": wal_state["attempt_succeeded"],
        "failed_attempts": wal_state["attempt_failed"],
        "open_attempts": wal_state["open_attempts"],
        "fresh_fixed_endpoint_joint_EF_attempts": 0,
        "ODE_NEB_reused_exact_joint_EF_values": 32,
        "optimization_N_F_contribution": 0,
        "held_host_locks": [str(path) for path in sorted(state["locks"], key=str)],
    }


def validate_audit(path: Path, state: Mapping[str, object], wal_state: Mapping[str, object]) -> dict:
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = expected_audit_fields(state, wal_state)
    require(set(audit) == set(expected) | {"completed_utc"}, "audit field inventory mismatch")
    require(isinstance(audit.get("completed_utc"), str) and audit["completed_utc"], "audit completion time missing")
    for key, value in expected.items():
        require(audit.get(key) == value, f"audit {key} mismatch")
    return audit


def expected_result_publication(state: Mapping[str, object]) -> dict:
    return {
        "schema": "bb-discrete32-path-dependent-geometric-action-result-publication-v1",
        "status": "PASS_ATOMIC_NO_OVERWRITE_PUBLICATION",
        "result": file_binding(state["output"]),
        "joint_ef_wal": file_binding(state["wal"]),
        "contract": file_binding(state["contract_path"]),
        "authorization": file_binding(state["authorization_path"]),
        "result_schema": SCHEMA,
    }


def validate_result_publication(state: Mapping[str, object]) -> dict:
    publication = json.loads(state["result_publication"].read_text(encoding="utf-8"))
    expected = expected_result_publication(state)
    require(set(publication) == set(expected) | {"published_utc"}, "result-publication field inventory mismatch")
    require(isinstance(publication.get("published_utc"), str) and publication["published_utc"], "result publication time missing")
    for key, value in expected.items():
        require(publication.get(key) == value, f"result-publication {key} mismatch")
    return publication


def preflight(state: Mapping[str, object]) -> dict:
    check_thread_caps()
    sources = load_sources()
    output = state["output"]
    wal_path = state["wal"]
    publication_path = state["result_publication"]
    audit_path = state["audit"]
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    require(free >= MINIMUM_FREE_BYTES + (0 if output.exists() else ESTIMATED_FINAL_BYTES), "insufficient free disk")
    wal_state = None
    if wal_path.exists():
        wal_state = read_wal(wal_path, expected_wal_header(state))
        validate_success_prefix(wal_state, sources["ir"]["path"])
    else:
        require(
            not output.exists() and not publication_path.exists() and not audit_path.exists(),
            "orphan result/publication/audit without WAL",
        )
    if output.exists():
        require(wal_state is not None, "result lacks joint E/F WAL")
        require(wal_state["attempt_succeeded"] == N_INTERNAL, "published result lacks all 30 joint E/F successes")
        validate_result(output, state, sources, wal_state)
        if publication_path.exists():
            validate_result_publication(state)
        if audit_path.exists():
            require(publication_path.exists(), "audit exists before atomic result publication")
            validate_audit(audit_path, state, wal_state)
    else:
        require(not audit_path.exists(), "orphan audit without result")
        require(not publication_path.exists(), "publication exists without result")
    with acquire_two_locks(state["locks"]) as held:
        lock_probe = list(held)
    return {
        "status": "PASS_ZERO_OPENMM_EXECUTION_PREFLIGHT",
        "physical_calls_started": 0,
        "openmm_contexts_constructed": 0,
        "fresh_internal_joint_EF_attempt_limit": N_INTERNAL,
        "fresh_fixed_endpoint_joint_EF_attempt_limit": 0,
        "optimization_N_F_contribution": 0,
        "existing_successes": 0 if wal_state is None else wal_state["attempt_succeeded"],
        "result_exists": output.exists(),
        "publication_complete": publication_path.exists(),
        "audit_complete": audit_path.exists(),
        "canonical_two_lock_probe": lock_probe,
        "observed_free_bytes": free,
    }


def _success_event(index: int, coordinate: np.ndarray, energy: float, force: np.ndarray, attempt_id: str) -> dict:
    force = np.ascontiguousarray(force, dtype="<f8")
    require(force.shape == (DIMENSION,) and np.all(np.isfinite(force)), "backend force shape/finite mismatch")
    return {
        "event": "attempt_succeeded",
        "utc": utc_now(),
        "attempt_id": attempt_id,
        "logical_index": index,
        "original_image_index": index + 1,
        "coordinate_sha256": sha256_array(coordinate),
        "energy_eV": float(energy),
        "energy_scalar_sha256": sha256_array(np.asarray(energy, dtype=np.float64)),
        "force_dtype": "<f8",
        "force_shape": [DIMENSION],
        "force_sha256": sha256_array(force),
        "force_le_f8_base64": base64.b64encode(force.tobytes(order="C")).decode("ascii"),
    }


def run(state: Mapping[str, object], backend_class=OpenMMBackend) -> dict:
    report = preflight(state)
    if report["audit_complete"]:
        current = read_wal(state["wal"], expected_wal_header(state))
        return validate_audit(state["audit"], state, current)
    sources = load_sources()
    with acquire_two_locks(state["locks"]) as held_locks:
        wal = DurableWal(state["wal"], expected_wal_header(state))
        try:
            current = read_wal(state["wal"], expected_wal_header(state))
            validate_success_prefix(current, sources["ir"]["path"])
            if state["output"].exists():
                require(current["attempt_succeeded"] == N_INTERNAL, "existing result lacks all 30 successes")
                validate_result(state["output"], state, sources, current)
            else:
                remaining = [index for index in range(N_INTERNAL) if index not in current["successes"]]
                backend = None if not remaining else backend_class(TOPOLOGY, label="discrete32_geometric_action")
                starting_raw_attempts = 0 if backend is None else int(backend.raw_joint_attempts)
                starting_raw_evaluations = 0 if backend is None else int(backend.raw_joint_evaluations)
                fresh_this_invocation = 0
                for index in remaining:
                    coordinate = sources["ir"]["path"][index + 1]
                    attempt_id = f"ir_internal:{index:02d}:attempt001"
                    base = {
                        "attempt_id": attempt_id,
                        "logical_index": index,
                        "original_image_index": index + 1,
                        "coordinate_sha256": sha256_array(coordinate),
                    }
                    wal.append({"event": "attempt_started", "utc": utc_now(), **base})
                    try:
                        energy, force = backend.evaluate_joint(coordinate)
                        require(np.isfinite(energy), "backend energy non-finite")
                        event = _success_event(index, coordinate, float(energy), force, attempt_id)
                    except Exception as error:
                        wal.append({"event": "attempt_failed", "utc": utc_now(), "error": repr(error), **base})
                        raise
                    wal.append(event)
                    fresh_this_invocation += 1
                if backend is not None:
                    require(int(backend.raw_joint_attempts) - starting_raw_attempts == fresh_this_invocation, "backend/WAL attempt ledger mismatch")
                    require(int(backend.raw_joint_evaluations) - starting_raw_evaluations == fresh_this_invocation, "backend success ledger mismatch")
                current = read_wal(state["wal"], expected_wal_header(state))
                validate_success_prefix(current, sources["ir"]["path"])
                require(current["attempt_started"] == N_INTERNAL and current["attempt_succeeded"] == N_INTERNAL, "joint E/F WAL did not close exactly 30 calls")
                payload = _build_payload(state, sources, current["successes"], held_locks)
                atomic_npz_no_overwrite(state["output"], payload)
                validate_result(state["output"], state, sources, current)
        finally:
            wal.close()
    final_wal = read_wal(state["wal"], expected_wal_header(state))
    if not state["result_publication"].exists():
        atomic_json_no_overwrite(
            state["result_publication"],
            {**expected_result_publication(state), "published_utc": utc_now()},
        )
    validate_result_publication(state)
    audit = {**expected_audit_fields(state, final_wal), "completed_utc": utc_now()}
    atomic_json_no_overwrite(state["audit"], audit)
    return validate_audit(state["audit"], state, final_wal)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION)
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    state = load_contract(args.contract, args.authorization, args.output)
    if args.preflight_only:
        print(json.dumps(preflight(state), indent=2, sort_keys=True))
    else:
        print(json.dumps(run(state), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
