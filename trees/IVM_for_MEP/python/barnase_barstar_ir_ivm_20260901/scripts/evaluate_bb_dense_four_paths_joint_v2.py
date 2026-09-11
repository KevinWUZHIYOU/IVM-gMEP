#!/usr/bin/env python3
"""Audited joint-E/F evaluation of four preregistered BB ODE--NEB paths.

The article I_perp is evaluated on each original 32-image path from fresh
forces at its 30 internal images.  Separately, the same path is interpolated
to 500 uniformly spaced points in unweighted Cartesian arc length.  Every
dense point receives exactly one joint energy/full-force call.  Dense forces
are archival diagnostics: they do not enter either metric or optimization N_F.

Completed paths are published as immutable atomic NPZ shards.  A restart may
reuse a shard only after validating its complete data and provenance closure;
individual WAL successes from an interrupted path are never silently reused.
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
import tempfile
from typing import Dict, Iterable, Sequence, Tuple
import uuid

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bb_core import OpenMMBackend, path_diagnostics  # noqa: E402


SCHEMA = "bb-dense-four-path-joint-ef-contract-v2"
AUDIT_SCHEMA = "bb-dense-four-path-joint-ef-audit-v2"
SHARD_SCHEMA = "bb-dense-path-joint-ef-shard-v2"
DISCRETE_WAL_SCHEMA = "bb-discrete-path-joint-ef-wal-v2"
DENSE_WAL_SCHEMA = "bb-dense-path-joint-ef-wal-v2"
N_IMAGES = 32
N_INTERNAL = 30
N_ATOMS = 3132
DIMENSION = 9396
N_DENSE = 500
EV_TO_KCAL = 23.0605
ODE_NF = (60, 150, 300, 1320)
MINIMUM_FREE_BYTES_RESERVE = 8 * 1024 ** 3
ESTIMATED_SHARD_BYTES = (2 * N_DENSE * DIMENSION + N_INTERNAL * DIMENSION + N_IMAGES * DIMENSION) * 8 + 8 * 1024 ** 2
THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENMM_CPU_THREADS",
)
METRIC_POLICY = {
    "coordinate_space": "unweighted_3N_cartesian",
    "sampling": "uniform_cartesian_arclength",
    "interpolation": "piecewise_linear",
    "number_of_points": 500,
    "grid_formula": "s_j=j*L/499,j=0,...,499",
    "include_both_endpoints": True,
    "dense_physical_call": "one_joint_energy_full_force_getState_per_point",
    "dense_force_use": "stored_only_not_used_by_metrics_or_optimization",
    "discrete_I_perp_implementation": "bb_core.path_diagnostics",
    "tangent": "normalized central difference: x[i+1] - x[i-1]",
    "weight": "half sum of adjacent Cartesian chord lengths",
    "original_fixed_endpoint_diagnostic_force_evaluations": 0,
    "dense_fixed_endpoint_diagnostic_force_evaluations": 2,
    "optimization_N_F_contribution": 0,
}


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
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def official_path32_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    require(array.shape == (N_IMAGES, N_ATOMS, 3), "official path shape mismatch")
    require(np.all(np.isfinite(array)), "official path contains non-finite values")
    metadata = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base.parent / path
    return path.resolve(strict=True)


def require_file(record: Dict[str, object], base: Path, role: str) -> Path:
    require(set(record) >= {"path", "sha256"}, f"{role} lacks path/SHA-256")
    path = resolve(base, str(record["path"]))
    require(sha256_file(path) == str(record["sha256"]), f"{role} SHA-256 mismatch")
    return path


def check_thread_caps() -> None:
    wrong = {name: os.environ.get(name) for name in THREAD_VARS if os.environ.get(name) != "1"}
    if wrong:
        raise RuntimeError(f"all six thread variables must equal '1': {wrong}")


def _scalar(saved, key: str):
    return np.asarray(saved[key]).item()


def _load_ode_path(source: Path) -> np.ndarray:
    with np.load(source, allow_pickle=False) as saved:
        required = {"coordinates30_A", "endpoint0_A", "endpoint1_A", "path32_sha256"}
        require(required.issubset(saved.files), "ODE archive fields missing")
        internal = np.asarray(saved["coordinates30_A"], dtype=np.float64)
        endpoint0 = np.asarray(saved["endpoint0_A"], dtype=np.float64)
        endpoint1 = np.asarray(saved["endpoint1_A"], dtype=np.float64)
        embedded = str(_scalar(saved, "path32_sha256"))
    require(internal.shape == (N_INTERNAL, N_ATOMS, 3), "ODE internal coordinate shape mismatch")
    require(endpoint0.shape == endpoint1.shape == (N_ATOMS, 3), "ODE endpoint shape mismatch")
    path3 = np.concatenate((endpoint0[None], internal, endpoint1[None]), axis=0)
    require(official_path32_sha256(path3) == embedded, "ODE embedded official path SHA mismatch")
    return np.ascontiguousarray(path3.reshape(N_IMAGES, DIMENSION))


def uniform_arc500(path: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    curve = np.asarray(path, dtype=np.float64)
    require(curve.shape == (N_IMAGES, DIMENSION), f"wrong path shape {curve.shape}")
    require(np.all(np.isfinite(curve)), "path contains non-finite coordinates")
    lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    require(np.all(np.isfinite(lengths)) and np.all(lengths > 0), "invalid path segment")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, float(cumulative[-1]), N_DENSE)
    index = np.clip(np.searchsorted(cumulative, targets, side="right") - 1, 0, len(lengths) - 1)
    fraction = (targets - cumulative[index]) / lengths[index]
    dense = (1.0 - fraction[:, None]) * curve[index] + fraction[:, None] * curve[index + 1]
    dense[0] = curve[0]
    dense[-1] = curve[-1]
    return np.ascontiguousarray(dense), {
        "total_arclength_A": float(cumulative[-1]),
        "dense_spacing_A": float(cumulative[-1]) / 499.0,
    }


def validate_and_load(contract_path: Path, host_role: str):
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(contract.get("schema") == SCHEMA, "unsupported contract schema")
    require(contract.get("frozen") is True, "contract is not frozen")
    require(contract.get("method") == "ODE-NEB", "v2 ODE evaluator requires method ODE-NEB")
    require(int(contract.get("jobs", -1)) == 1, "contract jobs must equal 1")
    require(contract.get("metric_policy") == METRIC_POLICY, "metric policy mismatch")
    evaluator = require_file(contract["evaluator"], contract_path, "evaluator")
    require(evaluator == Path(__file__).resolve(), "contract binds a different evaluator")
    bb_core = require_file(contract["bb_core"], contract_path, "bb_core")
    require(bb_core == (SRC / "bb_core.py").resolve(), "contract binds a different bb_core")
    topology = require_file(contract["topology"], contract_path, "topology")
    selection_path = require_file(contract["targeted_R2_selection"], contract_path, "targeted R2 selection")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    official = {int(row["optimization_N_F"]): row for row in selection["official_paid_states"]}
    locks = [Path(value).expanduser().resolve() for value in contract["host_local_lock_paths"]]
    require(len(locks) == 2 and len(set(locks)) == 2, "exactly two distinct lock paths required")
    rows = contract.get("paths")
    require(isinstance(rows, list) and len(rows) == 4, "contract must contain four paths")
    require(tuple(int(row["optimization_N_F"]) for row in rows) == ODE_NF, "paths must be ordered NF60/150/300/1320")
    loaded = []
    for row in rows:
        nf = int(row["optimization_N_F"])
        require(row.get("host_role") == host_role, f"NF{nf} host-role mismatch")
        require(row.get("method") == "ODE-NEB", f"NF{nf} method mismatch")
        require(row.get("discrete_force_policy") == "fresh_30_internal_joint_EF", f"NF{nf} discrete force policy mismatch")
        require(row.get("dense_policy") == "fresh_or_complete_validated_shard_500_joint_EF", f"NF{nf} dense policy mismatch")
        source = require_file(row["source"], contract_path, f"NF{nf} source")
        selected = official.get(nf)
        require(selected is not None, f"NF{nf} absent from targeted R2")
        require(resolve(selection_path, str(selected["coordinate_archive"])) == source, f"NF{nf} archive differs from targeted R2")
        require(str(selected["coordinate_archive_sha256"]) == str(row["source"]["sha256"]), f"NF{nf} archive SHA differs from targeted R2")
        path = _load_ode_path(source)
        require(sha256_array(path) == str(row["path_sha256"]), f"NF{nf} raw path SHA mismatch")
        official_sha = official_path32_sha256(path.reshape(N_IMAGES, N_ATOMS, 3))
        require(official_sha == str(row["official_path32_sha256"]), f"NF{nf} official path SHA mismatch")
        require(official_sha == str(selected["official_path32_sha256"]), f"NF{nf} differs from targeted R2 official path")
        loaded.append({**row, "source_path": source, "path": path})
    physical = contract.get("physical_call_policy", {})
    require(physical.get("original_internal_joint_EF_per_path") == N_INTERNAL, "wrong discrete call policy")
    require(physical.get("dense_joint_EF_per_path") == N_DENSE, "wrong dense call policy")
    require(physical.get("dense_full_force_storage") is True, "dense full-force storage not required")
    require(physical.get("optimization_N_F_contribution") == 0, "diagnostics must not enter optimization N_F")
    storage = contract.get("storage_policy", {})
    require(storage.get("authoritative_unit") == "complete_atomic_path_shard", "shard authority policy mismatch")
    require(storage.get("aggregate_role") == "index_metrics_and_plot_compatible_dense_energy_not_dense_coordinates_or_forces", "aggregate role policy mismatch")
    require(int(storage.get("minimum_free_bytes_reserve", -1)) == MINIMUM_FREE_BYTES_RESERVE, "free-disk reserve policy mismatch")
    require(int(storage.get("estimated_uncompressed_shard_bytes_per_path", -1)) == ESTIMATED_SHARD_BYTES, "shard disk estimate mismatch")
    durability = contract.get("durability_policy", {})
    for key in ("separate_discrete_and_dense_append_only_fsync_WAL", "atomic_complete_path_shard", "shard_contains_dense_coordinates_energy_full_force", "manifest_binds_every_shard_and_array_hash", "refuse_final_output_overwrite"):
        require(durability.get(key) is True, f"durability policy {key} must be true")
    require(durability.get("restart_reuse_unit") == "complete_fully_validated_path_shard", "restart reuse policy mismatch")
    require(durability.get("reuse_individual_incomplete_WAL_successes") is False, "incomplete WAL reuse must be false")
    return contract, topology, locks, loaded


class DurableWal:
    """Append-only fsync WAL which can be reopened after interruption."""

    def __init__(self, path: Path, schema: str, contract_path: Path, contract_sha: str):
        self.path = path
        self.session_id = uuid.uuid4().hex
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            require(lines, f"empty existing WAL {path}")
            header = json.loads(lines[0])
            require(header.get("event") == "wal_header", f"bad WAL header {path}")
            require(header.get("schema") == schema, f"wrong WAL schema {path}")
            require(header.get("contract_sha256") == contract_sha, f"WAL contract mismatch {path}")
            require(Path(header["contract"]).resolve() == contract_path, f"WAL contract path mismatch {path}")
            self.stream = path.open("a", encoding="utf-8", buffering=1)
            self.append({"event": "resume_session_started", "utc": utc_now(), "contract_sha256": contract_sha, "session_id": self.session_id})
        else:
            self.stream = path.open("x", encoding="utf-8", buffering=1)
            self.append({
                "event": "wal_header", "schema": schema, "created_utc": utc_now(),
                "contract": str(contract_path), "contract_sha256": contract_sha,
                "optimization_N_F_contribution": 0,
            })
            self.append({"event": "initial_session_started", "utc": utc_now(), "contract_sha256": contract_sha, "session_id": self.session_id})

    def append(self, payload: Dict[str, object]) -> None:
        self.stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


@contextmanager
def acquire_two_locks(paths: Sequence[Path]):
    ordered = sorted({Path(path).expanduser().resolve() for path in paths}, key=str)
    require(len(ordered) == 2, "contract must bind two distinct host-local locks")
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


def _atomic_save_npz_no_overwrite(path: Path, arrays: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".tmp.", suffix=".npz", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        # Both host-local locks are held, so this same-filesystem rename is an
        # atomic publication and cannot race another conforming BB evaluator.
        os.rename(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json_no_overwrite(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def shard_path_for(output: Path, nf: int) -> Path:
    return output.with_suffix("").with_name(output.stem + ".shards") / f"ode_neb_nf{nf}_joint_ef_arc500_v2.npz"


def _validate_shard(path: Path, row: Dict[str, object], contract_sha: str, topology_sha: str, evaluator_sha: str, bb_core_sha: str):
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "schema", "status", "method", "optimization_N_F", "label", "path_identifier_json",
            "contract_sha256", "evaluator_sha256", "bb_core_sha256", "topology_sha256",
            "selected_path", "selected_path_sha256", "official_path32_sha256",
            "internal_joint_energy_eV", "internal_diagnostic_force_eV_per_A",
            "internal_joint_energy_sha256", "internal_diagnostic_force_sha256", "discrete_I_perp",
            "dense_coordinates_A", "dense_energy_eV", "dense_force_eV_per_A",
            "dense_coordinate_sha256", "dense_energy_sha256", "dense_force_sha256",
            "densified_Delta_E_kcal_per_mol", "number_of_points", "optimization_N_F_contribution",
        }
        require(required.issubset(saved.files), f"incomplete shard {path}")
        require(str(_scalar(saved, "schema")) == SHARD_SCHEMA and str(_scalar(saved, "status")) == "PASS", "shard is not PASS v2")
        require(str(_scalar(saved, "method")) == "ODE-NEB", "shard method mismatch")
        require(int(_scalar(saved, "optimization_N_F")) == int(row["optimization_N_F"]), "shard N_F mismatch")
        require(str(_scalar(saved, "label")) == str(row["label"]), "shard label mismatch")
        require(json.loads(str(_scalar(saved, "path_identifier_json"))) == row["path_identifier"], "shard path identifier mismatch")
        require(str(_scalar(saved, "contract_sha256")) == contract_sha, "shard contract mismatch")
        require(str(_scalar(saved, "evaluator_sha256")) == evaluator_sha, "shard evaluator mismatch")
        require(str(_scalar(saved, "bb_core_sha256")) == bb_core_sha, "shard bb_core mismatch")
        require(str(_scalar(saved, "topology_sha256")) == topology_sha, "shard topology mismatch")
        selected = np.asarray(saved["selected_path"], dtype=np.float64)
        internal_e = np.asarray(saved["internal_joint_energy_eV"], dtype=np.float64)
        internal_f = np.asarray(saved["internal_diagnostic_force_eV_per_A"], dtype=np.float64)
        dense = np.asarray(saved["dense_coordinates_A"], dtype=np.float64)
        dense_e = np.asarray(saved["dense_energy_eV"], dtype=np.float64)
        dense_f = np.asarray(saved["dense_force_eV_per_A"], dtype=np.float64)
        require(selected.shape == (N_IMAGES, DIMENSION), "shard selected path shape mismatch")
        require(internal_e.shape == (N_INTERNAL,) and internal_f.shape == (N_INTERNAL, DIMENSION), "shard discrete arrays shape mismatch")
        require(dense.shape == dense_f.shape == (N_DENSE, DIMENSION) and dense_e.shape == (N_DENSE,), "shard dense arrays shape mismatch")
        require(np.all(np.isfinite(internal_e)) and np.all(np.isfinite(internal_f)), "shard discrete arrays non-finite")
        require(np.all(np.isfinite(dense)) and np.all(np.isfinite(dense_e)) and np.all(np.isfinite(dense_f)), "shard dense arrays non-finite")
        require(sha256_array(selected) == str(row["path_sha256"]) == str(_scalar(saved, "selected_path_sha256")), "shard selected path hash mismatch")
        require(str(_scalar(saved, "official_path32_sha256")) == str(row["official_path32_sha256"]), "shard official path hash mismatch")
        require(sha256_array(internal_e) == str(_scalar(saved, "internal_joint_energy_sha256")), "shard internal energy hash mismatch")
        require(sha256_array(internal_f) == str(_scalar(saved, "internal_diagnostic_force_sha256")), "shard internal force hash mismatch")
        expected_dense, _ = uniform_arc500(selected)
        require(np.array_equal(dense, expected_dense), "shard dense coordinates differ from deterministic arc500")
        require(sha256_array(dense) == str(_scalar(saved, "dense_coordinate_sha256")), "shard dense coordinate hash mismatch")
        require(sha256_array(dense_e) == str(_scalar(saved, "dense_energy_sha256")), "shard dense energy hash mismatch")
        require(sha256_array(dense_f) == str(_scalar(saved, "dense_force_sha256")), "shard dense force hash mismatch")
        diagnostic = path_diagnostics(selected, np.zeros(N_IMAGES, dtype=np.float64), internal_f)
        require(float(_scalar(saved, "discrete_I_perp")) == float(diagnostic["i_perp"]), "shard I_perp mismatch")
        delta_e = float((np.max(dense_e) - dense_e[0]) * EV_TO_KCAL)
        require(float(_scalar(saved, "densified_Delta_E_kcal_per_mol")) == delta_e, "shard Delta E mismatch")
        require(int(_scalar(saved, "number_of_points")) == N_DENSE, "shard point count mismatch")
        require(int(_scalar(saved, "optimization_N_F_contribution")) == 0, "shard optimization N_F contribution mismatch")
        return {key: np.asarray(saved[key]).copy() for key in saved.files}


def _attempt(wal: DurableWal, backend: OpenMMBackend, row_index: int, row: Dict[str, object], stage: str, point_index: int, coordinate: np.ndarray):
    coordinate_sha = sha256_array(coordinate)
    attempt_id = f"{wal.session_id}:{row_index}:{stage}:{point_index}:{int(backend.raw_joint_attempts) + 1}"
    started = {
        "event": "attempt_started", "utc": utc_now(), "stage": stage,
        "row_index": row_index, "label": str(row["label"]),
        "optimization_N_F": int(row["optimization_N_F"]), "path_identifier": row["path_identifier"],
        "path_sha256": str(row["path_sha256"]), "point_index": point_index,
        "coordinate_sha256": coordinate_sha, "backend_attempt_ordinal": int(backend.raw_joint_attempts) + 1,
        "session_id": wal.session_id, "attempt_id": attempt_id,
    }
    wal.append(started)
    try:
        energy, force = backend.evaluate_joint(coordinate)
    except Exception as error:
        wal.append({**started, "event": "attempt_failed", "utc": utc_now(), "error": repr(error)})
        raise
    force = np.ascontiguousarray(force, dtype=np.float64)
    require(force.shape == (DIMENSION,) and np.all(np.isfinite(force)) and np.isfinite(energy), "bad joint E/F result")
    wal.append({
        **started, "event": "attempt_succeeded", "utc": utc_now(),
        "energy_eV": float(energy), "energy_scalar_sha256": sha256_array(np.asarray(float(energy), dtype=np.float64)),
        "force_sha256": sha256_array(force),
    })
    return float(energy), force


def _evaluate_one(row_index: int, row: Dict[str, object], backend: OpenMMBackend, discrete_wal: DurableWal, dense_wal: DurableWal, contract_sha: str, topology_sha: str, evaluator_sha: str, bb_core_sha: str, shard_path: Path):
    path = row["path"]
    internal_e = np.empty(N_INTERNAL, dtype=np.float64)
    internal_f = np.empty((N_INTERNAL, DIMENSION), dtype=np.float64)
    discrete_wal.append({"event": "path_stage_started", "utc": utc_now(), "stage": "original_discrete_internal", "row_index": row_index, "label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "path_sha256": row["path_sha256"]})
    for index, coordinate in enumerate(path[1:-1], start=1):
        internal_e[index - 1], internal_f[index - 1] = _attempt(discrete_wal, backend, row_index, row, "original_discrete_internal", index, coordinate)
    diagnostic = path_diagnostics(path, np.zeros(N_IMAGES, dtype=np.float64), internal_f)
    i_perp = float(diagnostic["i_perp"])
    discrete_wal.append({"event": "path_stage_data_complete_uncommitted", "utc": utc_now(), "stage": "original_discrete_internal", "row_index": row_index, "label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "path_sha256": row["path_sha256"], "joint_attempts": N_INTERNAL, "joint_successes": N_INTERNAL, "internal_joint_energy_sha256": sha256_array(internal_e), "internal_diagnostic_force_sha256": sha256_array(internal_f), "discrete_I_perp": i_perp})

    dense, geometry = uniform_arc500(path)
    dense_e = np.empty(N_DENSE, dtype=np.float64)
    dense_f = np.empty((N_DENSE, DIMENSION), dtype=np.float64)
    dense_wal.append({"event": "path_stage_started", "utc": utc_now(), "stage": "dense_arc500_joint_EF", "row_index": row_index, "label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "path_sha256": row["path_sha256"], "dense_coordinate_sha256": sha256_array(dense), "number_of_points": N_DENSE})
    for index, coordinate in enumerate(dense):
        dense_e[index], dense_f[index] = _attempt(dense_wal, backend, row_index, row, "dense_arc500_joint_EF", index, coordinate)
    delta_e = float((np.max(dense_e) - dense_e[0]) * EV_TO_KCAL)
    dense_wal.append({"event": "path_stage_data_complete_uncommitted", "utc": utc_now(), "stage": "dense_arc500_joint_EF", "row_index": row_index, "label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "path_sha256": row["path_sha256"], "joint_attempts": N_DENSE, "joint_successes": N_DENSE, "dense_coordinate_sha256": sha256_array(dense), "dense_energy_sha256": sha256_array(dense_e), "dense_force_sha256": sha256_array(dense_f), "densified_Delta_E_kcal_per_mol": delta_e})

    arrays = {
        "schema": np.asarray(SHARD_SCHEMA), "status": np.asarray("PASS"), "method": np.asarray("ODE-NEB"),
        "optimization_N_F": np.asarray(int(row["optimization_N_F"]), dtype=np.int64), "label": np.asarray(str(row["label"])),
        "path_identifier_json": np.asarray(json.dumps(row["path_identifier"], sort_keys=True)),
        "contract_sha256": np.asarray(contract_sha), "evaluator_sha256": np.asarray(evaluator_sha), "bb_core_sha256": np.asarray(bb_core_sha), "topology_sha256": np.asarray(topology_sha),
        "selected_path": path, "selected_path_sha256": np.asarray(str(row["path_sha256"])), "official_path32_sha256": np.asarray(str(row["official_path32_sha256"])),
        "internal_joint_energy_eV": internal_e, "internal_diagnostic_force_eV_per_A": internal_f,
        "internal_joint_energy_sha256": np.asarray(sha256_array(internal_e)), "internal_diagnostic_force_sha256": np.asarray(sha256_array(internal_f)), "discrete_I_perp": np.asarray(i_perp),
        "dense_coordinates_A": dense, "dense_energy_eV": dense_e, "dense_force_eV_per_A": dense_f,
        "dense_coordinate_sha256": np.asarray(sha256_array(dense)), "dense_energy_sha256": np.asarray(sha256_array(dense_e)), "dense_force_sha256": np.asarray(sha256_array(dense_f)),
        "densified_Delta_E_kcal_per_mol": np.asarray(delta_e), "total_arclength_A": np.asarray(geometry["total_arclength_A"]), "dense_spacing_A": np.asarray(geometry["dense_spacing_A"]),
        "number_of_points": np.asarray(N_DENSE, dtype=np.int64), "original_discrete_joint_EF_attempts": np.asarray(N_INTERNAL, dtype=np.int64), "dense_joint_EF_attempts": np.asarray(N_DENSE, dtype=np.int64), "dense_endpoint_joint_EF_attempts": np.asarray(2, dtype=np.int64), "original_fixed_endpoint_diagnostic_force_attempts": np.asarray(0, dtype=np.int64), "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
    }
    _atomic_save_npz_no_overwrite(shard_path, arrays)
    return _validate_shard(shard_path, row, contract_sha, topology_sha, evaluator_sha, bb_core_sha)


def _aggregate(output: Path, rows, shards, contract_path: Path, topology: Path, held_locks, discrete_wal_path: Path, dense_wal_path: Path):
    shard_paths = [shard_path_for(output, int(row["optimization_N_F"])) for row in rows]
    arrays = {
        "schema": np.asarray("bb-dense-four-path-joint-ef-result-v2"), "status": np.asarray("PASS"), "method": np.asarray("ODE-NEB"), "authoritative_data_unit": np.asarray("complete_atomic_path_shard"),
        "labels": np.asarray([str(_scalar(shard, "label")) for shard in shards]),
        "optimization_N_F": np.asarray([int(_scalar(shard, "optimization_N_F")) for shard in shards], dtype=np.int64),
        "path_identifier_json": np.asarray([str(_scalar(shard, "path_identifier_json")) for shard in shards]),
        "selected_path": np.asarray([shard["selected_path"] for shard in shards]), "selected_path_sha256": np.asarray([str(_scalar(shard, "selected_path_sha256")) for shard in shards]), "official_path32_sha256": np.asarray([str(_scalar(shard, "official_path32_sha256")) for shard in shards]),
        "discrete_I_perp": np.asarray([float(_scalar(shard, "discrete_I_perp")) for shard in shards]),
        "internal_diagnostic_force_eV_per_A": np.asarray([shard["internal_diagnostic_force_eV_per_A"] for shard in shards]), "internal_diagnostic_force_sha256": np.asarray([str(_scalar(shard, "internal_diagnostic_force_sha256")) for shard in shards]),
        "internal_joint_energy_eV": np.asarray([shard["internal_joint_energy_eV"] for shard in shards]), "internal_joint_energy_sha256": np.asarray([str(_scalar(shard, "internal_joint_energy_sha256")) for shard in shards]),
        "diagnostic_force_attempts_per_path": np.full(4, N_INTERNAL, dtype=np.int64), "fixed_endpoint_diagnostic_force_attempts_per_path": np.zeros(4, dtype=np.int64),
        "energy_eV": np.asarray([shard["dense_energy_eV"] for shard in shards]), "energy_sha256": np.asarray([str(_scalar(shard, "dense_energy_sha256")) for shard in shards]),
        "dense_coordinate_sha256": np.asarray([str(_scalar(shard, "dense_coordinate_sha256")) for shard in shards]), "dense_force_sha256": np.asarray([str(_scalar(shard, "dense_force_sha256")) for shard in shards]),
        "densified_Delta_E_kcal_per_mol": np.asarray([float(_scalar(shard, "densified_Delta_E_kcal_per_mol")) for shard in shards]),
        "diagnostic_energy_attempts_lineage_per_path": np.full(4, N_DENSE, dtype=np.int64), "diagnostic_energy_attempts_fresh_per_path": np.full(4, N_DENSE, dtype=np.int64),
        "dense_joint_EF_attempts_per_path": np.full(4, N_DENSE, dtype=np.int64), "dense_full_force_stored_per_path": np.ones(4, dtype=bool), "dense_endpoint_joint_EF_attempts_per_path": np.full(4, 2, dtype=np.int64),
        "total_arclength_A": np.asarray([float(_scalar(shard, "total_arclength_A")) for shard in shards]), "dense_spacing_A": np.asarray([float(_scalar(shard, "dense_spacing_A")) for shard in shards]),
        "coordinate_space": np.asarray(METRIC_POLICY["coordinate_space"]), "sampling": np.asarray(METRIC_POLICY["sampling"]), "interpolation": np.asarray(METRIC_POLICY["interpolation"]), "number_of_points": np.asarray(N_DENSE, dtype=np.int64), "grid_formula": np.asarray(METRIC_POLICY["grid_formula"]),
        "shard_path": np.asarray([str(path) for path in shard_paths]), "shard_sha256": np.asarray([sha256_file(path) for path in shard_paths]),
        "contract": np.asarray(str(contract_path)), "contract_sha256": np.asarray(sha256_file(contract_path)), "evaluator_sha256": np.asarray(sha256_file(Path(__file__))), "bb_core_sha256": np.asarray(sha256_file(SRC / "bb_core.py")), "topology": np.asarray(str(topology)), "topology_sha256": np.asarray(sha256_file(topology)),
        "force_wal": np.asarray(str(discrete_wal_path)), "energy_wal": np.asarray(str(dense_wal_path)), "dense_joint_wal": np.asarray(str(dense_wal_path)), "held_host_locks": np.asarray(held_locks), "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
    }
    _atomic_save_npz_no_overwrite(output, arrays)


def _validate_aggregate(output: Path, rows, shards, contract_path: Path) -> None:
    with np.load(output, allow_pickle=False) as saved:
        required = {"schema", "status", "method", "authoritative_data_unit", "optimization_N_F", "labels", "selected_path", "selected_path_sha256", "internal_joint_energy_eV", "internal_joint_energy_sha256", "internal_diagnostic_force_eV_per_A", "internal_diagnostic_force_sha256", "discrete_I_perp", "energy_eV", "densified_Delta_E_kcal_per_mol", "dense_coordinate_sha256", "energy_sha256", "dense_force_sha256", "shard_path", "shard_sha256", "contract_sha256", "optimization_N_F_contribution"}
        require(required.issubset(saved.files), "preexisting aggregate is incomplete")
        require(str(_scalar(saved, "schema")) == "bb-dense-four-path-joint-ef-result-v2" and str(_scalar(saved, "status")) == "PASS", "preexisting aggregate is not PASS v2")
        require(str(_scalar(saved, "method")) == "ODE-NEB" and str(_scalar(saved, "authoritative_data_unit")) == "complete_atomic_path_shard", "preexisting aggregate authority mismatch")
        require(str(_scalar(saved, "contract_sha256")) == sha256_file(contract_path), "preexisting aggregate contract mismatch")
        require(int(_scalar(saved, "optimization_N_F_contribution")) == 0, "preexisting aggregate optimization N_F mismatch")
        expected_nf = np.asarray([int(row["optimization_N_F"]) for row in rows], dtype=np.int64)
        require(np.array_equal(np.asarray(saved["optimization_N_F"]), expected_nf), "preexisting aggregate N_F mismatch")
        require(np.array_equal(np.asarray(saved["labels"]).astype(str), np.asarray([row["label"] for row in rows])), "preexisting aggregate labels mismatch")
        bindings = (
            ("selected_path_sha256", "selected_path_sha256"),
            ("dense_coordinate_sha256", "dense_coordinate_sha256"),
            ("energy_sha256", "dense_energy_sha256"),
            ("dense_force_sha256", "dense_force_sha256"),
        )
        for aggregate_key, shard_key in bindings:
            expected = np.asarray([str(_scalar(shard, shard_key)) for shard in shards])
            require(np.array_equal(np.asarray(saved[aggregate_key]).astype(str), expected), f"preexisting aggregate {aggregate_key} mismatch")
        array_bindings = (
            ("selected_path", "selected_path"),
            ("internal_joint_energy_eV", "internal_joint_energy_eV"),
            ("internal_diagnostic_force_eV_per_A", "internal_diagnostic_force_eV_per_A"),
            ("energy_eV", "dense_energy_eV"),
        )
        for aggregate_key, shard_key in array_bindings:
            expected = np.asarray([shard[shard_key] for shard in shards])
            actual = np.asarray(saved[aggregate_key])
            require(actual.dtype.str == expected.dtype.str and actual.shape == expected.shape and actual.tobytes(order="C") == expected.tobytes(order="C"), f"preexisting aggregate {aggregate_key} value mismatch")
        scalar_bindings = (
            ("discrete_I_perp", "discrete_I_perp"),
            ("densified_Delta_E_kcal_per_mol", "densified_Delta_E_kcal_per_mol"),
        )
        for aggregate_key, shard_key in scalar_bindings:
            expected = np.asarray([float(_scalar(shard, shard_key)) for shard in shards], dtype=np.float64)
            actual = np.asarray(saved[aggregate_key], dtype=np.float64)
            require(actual.shape == expected.shape and actual.tobytes(order="C") == expected.tobytes(order="C"), f"preexisting aggregate {aggregate_key} value mismatch")
        internal_hash_bindings = (
            ("internal_joint_energy_sha256", "internal_joint_energy_sha256"),
            ("internal_diagnostic_force_sha256", "internal_diagnostic_force_sha256"),
        )
        for aggregate_key, shard_key in internal_hash_bindings:
            expected = np.asarray([str(_scalar(shard, shard_key)) for shard in shards])
            require(np.array_equal(np.asarray(saved[aggregate_key]).astype(str), expected), f"preexisting aggregate {aggregate_key} mismatch")
        expected_paths = np.asarray([str(shard_path_for(output, int(row["optimization_N_F"]))) for row in rows])
        expected_shas = np.asarray([sha256_file(Path(path)) for path in expected_paths])
        require(np.array_equal(np.asarray(saved["shard_path"]).astype(str), expected_paths), "preexisting aggregate shard path mismatch")
        require(np.array_equal(np.asarray(saved["shard_sha256"]).astype(str), expected_shas), "preexisting aggregate shard SHA mismatch")
        require("dense_coordinates_A" not in saved.files and "dense_force_eV_per_A" not in saved.files, "aggregate must not duplicate authoritative dense coordinates/forces")


def _count_wal(path: Path, event: str) -> int:
    return sum(json.loads(line).get("event") == event for line in path.read_text(encoding="utf-8").splitlines())


def _read_wal_events(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _wal_lifetime_stats(events):
    starts = [event for event in events if event.get("event") == "attempt_started"]
    successes = [event for event in events if event.get("event") == "attempt_succeeded"]
    failures = [event for event in events if event.get("event") == "attempt_failed"]
    start_ids = {str(event["attempt_id"]) for event in starts}
    terminal_ids = {str(event["attempt_id"]) for event in successes + failures}
    require(len(start_ids) == len(starts), "duplicate WAL attempt_id")
    require(all(str(event["attempt_id"]) in start_ids for event in successes + failures), "WAL terminal without start")
    require(not ({str(event["attempt_id"]) for event in successes} & {str(event["attempt_id"]) for event in failures}), "WAL attempt has conflicting terminals")
    logical = {(int(event["row_index"]), str(event["stage"]), int(event["point_index"])) for event in starts}
    return {
        "attempt_started": len(starts), "attempt_succeeded": len(successes),
        "attempt_failed": len(failures), "attempt_open": len(start_ids - terminal_ids),
        "retry_attempts": len(starts) - len(logical),
    }


def _validate_stage_wal_closure(events, row_index: int, row: Dict[str, object], stage: str, coordinates: np.ndarray, energies: np.ndarray, forces: np.ndarray):
    starts_at = [index for index, event in enumerate(events) if event.get("event") == "path_stage_started" and int(event.get("row_index", -1)) == row_index and event.get("stage", stage) == stage]
    completes_at = [index for index, event in enumerate(events) if event.get("event") == "path_stage_data_complete_uncommitted" and int(event.get("row_index", -1)) == row_index and event.get("stage") == stage]
    require(starts_at and completes_at, f"{row['label']} lacks a complete {stage} WAL stage")
    complete_index = completes_at[-1]
    start_candidates = [index for index in starts_at if index < complete_index]
    require(start_candidates, f"{row['label']} {stage} complete has no start")
    start_index = start_candidates[-1]
    segment = events[start_index:complete_index + 1]
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
        coordinate = coordinates[array_index]
        energy = float(energies[array_index])
        force = forces[array_index]
        require(event["coordinate_sha256"] == sha256_array(coordinate), f"{row['label']} {stage} coordinate WAL/shard mismatch at {point}")
        require(float(event["energy_eV"]) == energy, f"{row['label']} {stage} energy WAL/shard mismatch at {point}")
        require(event["energy_scalar_sha256"] == sha256_array(np.asarray(energy, dtype=np.float64)), f"{row['label']} {stage} energy hash mismatch at {point}")
        require(event["force_sha256"] == sha256_array(force), f"{row['label']} {stage} force WAL/shard mismatch at {point}")
    return {"accepted_stage_attempts": len(starts), "accepted_stage_successes": len(successes), "accepted_stage_failures": 0, "accepted_stage_open": 0, "pointwise_WAL_shard_closure": "PASS"}


def validate_all_wal_shard_closure(rows, shards, discrete_wal_path: Path, dense_wal_path: Path):
    discrete_events = _read_wal_events(discrete_wal_path)
    dense_events = _read_wal_events(dense_wal_path)
    path_reports = []
    for row_index, (row, shard) in enumerate(zip(rows, shards)):
        discrete = _validate_stage_wal_closure(
            discrete_events, row_index, row, "original_discrete_internal",
            np.asarray(shard["selected_path"])[1:-1], np.asarray(shard["internal_joint_energy_eV"]), np.asarray(shard["internal_diagnostic_force_eV_per_A"]),
        )
        dense = _validate_stage_wal_closure(
            dense_events, row_index, row, "dense_arc500_joint_EF",
            np.asarray(shard["dense_coordinates_A"]), np.asarray(shard["dense_energy_eV"]), np.asarray(shard["dense_force_eV_per_A"]),
        )
        path_reports.append({"label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "discrete": discrete, "dense": dense})
    return {"status": "PASS", "discrete_lifetime": _wal_lifetime_stats(discrete_events), "dense_lifetime": _wal_lifetime_stats(dense_events), "paths": path_reports}


def run(contract_path: Path, topology: Path, locks: Sequence[Path], rows, output: Path) -> None:
    contract_sha = sha256_file(contract_path)
    topology_sha = sha256_file(topology)
    evaluator_sha = sha256_file(Path(__file__))
    bb_core_sha = sha256_file(SRC / "bb_core.py")
    discrete_wal_path = output.with_suffix(".discrete_joint_ef.wal.jsonl")
    dense_wal_path = output.with_suffix(".dense_joint_ef.wal.jsonl")
    manifest_path = output.with_suffix(".manifest.json")
    audit_path = output.with_suffix(".audit.json")
    if audit_path.exists():
        raise FileExistsError("refusing to overwrite completed final audit")
    if manifest_path.exists() and not output.exists():
        raise RuntimeError("orphan manifest exists without aggregate")
    if output.exists() and any(not shard_path_for(output, int(row["optimization_N_F"])).exists() for row in rows):
        raise RuntimeError("preexisting aggregate exists without all authoritative shards")
    output.parent.mkdir(parents=True, exist_ok=True)
    missing_shards = sum(not shard_path_for(output, int(row["optimization_N_F"])).exists() for row in rows)
    required_free = MINIMUM_FREE_BYTES_RESERVE + missing_shards * ESTIMATED_SHARD_BYTES
    observed_free = shutil.disk_usage(output.parent).free
    if observed_free < required_free:
        raise RuntimeError(f"insufficient free disk: {observed_free} < required {required_free}")
    discrete_wal = DurableWal(discrete_wal_path, DISCRETE_WAL_SCHEMA, contract_path, contract_sha)
    dense_wal = DurableWal(dense_wal_path, DENSE_WAL_SCHEMA, contract_path, contract_sha)
    shards = []
    reused = []
    fresh = []
    backend = None
    held_locks = []
    try:
        with acquire_two_locks(locks) as held_locks:
            for row_index, row in enumerate(rows):
                shard_path = shard_path_for(output, int(row["optimization_N_F"]))
                if shard_path.exists():
                    shard = _validate_shard(shard_path, row, contract_sha, topology_sha, evaluator_sha, bb_core_sha)
                    reused.append(int(row["optimization_N_F"]))
                    payload = {"event": "complete_validated_shard_reused", "utc": utc_now(), "row_index": row_index, "label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "path_sha256": row["path_sha256"], "shard": str(shard_path), "shard_sha256": sha256_file(shard_path), "individual_incomplete_WAL_attempts_reused": 0}
                    discrete_wal.append(payload)
                    dense_wal.append(payload)
                else:
                    discrete_wal.append({"event": "path_restart_from_zero", "utc": utc_now(), "row_index": row_index, "label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "path_sha256": row["path_sha256"], "individual_incomplete_WAL_attempts_reused": 0})
                    dense_wal.append({"event": "path_restart_from_zero", "utc": utc_now(), "row_index": row_index, "label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "path_sha256": row["path_sha256"], "individual_incomplete_WAL_attempts_reused": 0})
                    if backend is None:
                        backend = OpenMMBackend(topology, platform_preference=("CPU",), precision="double", label="bb_dense_joint_ef_v2")
                    before = int(backend.raw_joint_attempts)
                    shard = _evaluate_one(row_index, row, backend, discrete_wal, dense_wal, contract_sha, topology_sha, evaluator_sha, bb_core_sha, shard_path)
                    require(int(backend.raw_joint_attempts) - before == N_INTERNAL + N_DENSE, "backend attempt ledger mismatch for path")
                    require(int(backend.raw_joint_evaluations) == int(backend.raw_joint_attempts), "backend success ledger mismatch")
                    fresh.append(int(row["optimization_N_F"]))
                    published = {"event": "complete_shard_published", "utc": utc_now(), "row_index": row_index, "label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "path_sha256": row["path_sha256"], "shard": str(shard_path), "shard_sha256": sha256_file(shard_path)}
                    discrete_wal.append(published)
                    dense_wal.append(published)
                shards.append(shard)
            wal_shard_closure = validate_all_wal_shard_closure(rows, shards, discrete_wal_path, dense_wal_path)
            if output.exists():
                _validate_aggregate(output, rows, shards, contract_path)
                aggregate_event = "preexisting_aggregate_validated"
            else:
                _aggregate(output, rows, shards, contract_path, topology, held_locks, discrete_wal_path, dense_wal_path)
                _validate_aggregate(output, rows, shards, contract_path)
                aggregate_event = "aggregate_published"
            discrete_wal.append({"event": aggregate_event, "utc": utc_now(), "output": str(output), "output_sha256": sha256_file(output)})
            dense_wal.append({"event": aggregate_event, "utc": utc_now(), "output": str(output), "output_sha256": sha256_file(output)})
    finally:
        discrete_wal.close()
        dense_wal.close()
    manifest = {
        "schema": "bb-dense-four-path-joint-ef-shard-manifest-v2", "status": "PASS",
        "created_utc": utc_now(), "contract": str(contract_path), "contract_sha256": contract_sha,
        "evaluator_sha256": evaluator_sha, "bb_core_sha256": bb_core_sha, "topology_sha256": topology_sha,
        "aggregate": str(output), "aggregate_sha256": sha256_file(output),
        "WAL_shard_pointwise_closure": wal_shard_closure,
        "shards": [
            {
                "label": str(row["label"]), "optimization_N_F": int(row["optimization_N_F"]),
                "path_identifier": row["path_identifier"], "path_sha256": str(row["path_sha256"]),
                "path": str(shard_path_for(output, int(row["optimization_N_F"]))),
                "sha256": sha256_file(shard_path_for(output, int(row["optimization_N_F"]))),
                "dense_coordinate_sha256": str(_scalar(shard, "dense_coordinate_sha256")),
                "dense_energy_sha256": str(_scalar(shard, "dense_energy_sha256")),
                "dense_force_sha256": str(_scalar(shard, "dense_force_sha256")),
            }
            for row, shard in zip(rows, shards)
        ],
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("schema", "status", "contract", "contract_sha256", "evaluator_sha256", "bb_core_sha256", "topology_sha256", "aggregate", "aggregate_sha256", "shards"):
            require(existing_manifest.get(key) == manifest.get(key), f"preexisting manifest {key} mismatch")
    else:
        _atomic_write_json_no_overwrite(manifest_path, manifest)
    paths = []
    for row, shard in zip(rows, shards):
        shard_path = shard_path_for(output, int(row["optimization_N_F"]))
        paths.append({
            "label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "path_identifier": row["path_identifier"],
            "path_sha256": row["path_sha256"], "official_path32_sha256": row["official_path32_sha256"], "shard": str(shard_path), "shard_sha256": sha256_file(shard_path),
            "discrete_I_perp": float(_scalar(shard, "discrete_I_perp")), "densified_Delta_E_kcal_per_mol": float(_scalar(shard, "densified_Delta_E_kcal_per_mol")),
            "original_discrete_joint_EF_attempts_lineage": N_INTERNAL, "original_fixed_endpoint_diagnostic_force_attempts": 0,
            "dense_joint_EF_attempts_lineage": N_DENSE, "dense_endpoint_joint_EF_attempts_lineage": 2, "dense_full_force_vectors_stored": N_DENSE,
            "dense_coordinate_sha256": str(_scalar(shard, "dense_coordinate_sha256")), "dense_energy_sha256": str(_scalar(shard, "dense_energy_sha256")), "dense_force_sha256": str(_scalar(shard, "dense_force_sha256")),
            "fresh_this_invocation": int(row["optimization_N_F"]) in fresh, "complete_validated_shard_reused": int(row["optimization_N_F"]) in reused,
            "individual_incomplete_WAL_attempts_reused": 0, "optimization_N_F_contribution": 0,
        })
    audit = {
        "schema": AUDIT_SCHEMA, "status": "PASS", "completed_utc": utc_now(), "method": "ODE-NEB",
        "contract": str(contract_path), "contract_sha256": contract_sha, "evaluator_sha256": evaluator_sha, "bb_core_sha256": bb_core_sha, "topology_sha256": topology_sha,
        "output": str(output), "output_sha256": sha256_file(output), "discrete_joint_ef_wal": str(discrete_wal_path), "discrete_joint_ef_wal_sha256": sha256_file(discrete_wal_path), "dense_joint_ef_wal": str(dense_wal_path), "dense_joint_ef_wal_sha256": sha256_file(dense_wal_path), "held_host_locks": held_locks,
        "shard_manifest": str(manifest_path), "shard_manifest_sha256": sha256_file(manifest_path),
        "path_count": 4, "fresh_paths_this_invocation": fresh, "complete_validated_shards_reused": reused,
        "free_disk_gate": {"observed_free_bytes": observed_free, "required_free_bytes": required_free, "minimum_free_bytes_reserve": MINIMUM_FREE_BYTES_RESERVE, "missing_shards_at_start": missing_shards, "status": "PASS"},
        "WAL_shard_pointwise_closure": wal_shard_closure,
        "original_discrete_joint_EF_attempts_lineage": 4 * N_INTERNAL, "dense_joint_EF_attempts_lineage": 4 * N_DENSE, "dense_full_force_vectors_stored": 4 * N_DENSE,
        "dense_endpoint_joint_EF_attempts_lineage": 8, "original_fixed_endpoint_diagnostic_force_attempts": 0,
        "lifetime_discrete_attempt_started_events": _count_wal(discrete_wal_path, "attempt_started"), "lifetime_discrete_attempt_succeeded_events": _count_wal(discrete_wal_path, "attempt_succeeded"),
        "lifetime_dense_attempt_started_events": _count_wal(dense_wal_path, "attempt_started"), "lifetime_dense_attempt_succeeded_events": _count_wal(dense_wal_path, "attempt_succeeded"),
        "individual_incomplete_WAL_attempts_reused": 0, "optimization_N_F_contribution": 0, "paths": paths,
    }
    _atomic_write_json_no_overwrite(audit_path, audit)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host-role", choices=("local", "remote"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    check_thread_caps()
    contract_path = args.contract.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    require(output.suffix == ".npz", "--output must end in .npz")
    contract, topology, locks, rows = validate_and_load(contract_path, args.host_role)
    summary = {
        "status": "ZERO_OPENMM_PREFLIGHT_PASS", "contract": str(contract_path), "contract_sha256": sha256_file(contract_path), "method": "ODE-NEB", "host_role": args.host_role,
        "path_count": 4, "optimization_N_F": [int(row["optimization_N_F"]) for row in rows], "path_sha256": [str(row["path_sha256"]) for row in rows],
        "original_discrete_joint_EF_attempts_planned": N_INTERNAL * 4, "dense_joint_EF_attempts_planned": N_DENSE * 4,
        "dense_full_force_vectors_planned_for_storage": N_DENSE * 4, "dense_endpoint_joint_EF_attempts_planned": 8,
        "restart_granularity": "complete_validated_path_shard_only", "individual_incomplete_WAL_attempts_reused": 0, "optimization_N_F_contribution": 0,
    }
    if args.preflight_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    run(contract_path, topology, locks, rows, output)


if __name__ == "__main__":
    main()
