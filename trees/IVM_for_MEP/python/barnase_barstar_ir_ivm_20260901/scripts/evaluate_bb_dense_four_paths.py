#!/usr/bin/env python3
"""Purpose-built dual-metric evaluator for four frozen BB paths.

This is deliberately not a general campaign runner.  It accepts exactly one
four-path ODE--NEB or IR-IVM contract, verifies every bound artifact, and then
computes the manuscript diagnostics without contributing to optimization N_F.
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
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bb_core import OpenMMBackend, path_diagnostics  # noqa: E402


SCHEMA = "bb-dense-four-path-dual-metric-contract-v1"
AUDIT_SCHEMA = "bb-dense-four-path-dual-metric-audit-v1"
FORCE_WAL_SCHEMA = "bb-dense-four-path-force-wal-v1"
ENERGY_WAL_SCHEMA = "bb-dense-four-path-energy-wal-v1"
N_IMAGES = 32
N_INTERNAL = 30
N_ATOMS = 3132
DIMENSION = 9396
N_DENSE = 500
EV_TO_KCAL = 23.0605
ODE_NF = (60, 150, 300, 1320)
IVM_CALLS = (2, 5, 10, 50)
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENMM_CPU_THREADS",
)
METRIC_POLICY = {
    "coordinate_space": "unweighted_3N_cartesian",
    "sampling": "uniform_cartesian_arclength",
    "interpolation": "piecewise_linear",
    "number_of_points": 500,
    "grid_formula": "s_j=j*L/499,j=0,...,499",
    "include_both_endpoints": True,
    "discrete_I_perp_implementation": "bb_core.path_diagnostics",
    "tangent": "normalized central difference: x[i+1] - x[i-1]",
    "weight": "half sum of adjacent Cartesian chord lengths",
    "fixed_endpoint_diagnostic_force_evaluations": 0,
    "optimization_N_F_contribution": 0,
}


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
    """Hash exactly as the sealed ODE--NEB paid-state ledger does."""
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_file(record: Dict[str, object], base: Path, role: str) -> Path:
    require(set(record) >= {"path", "sha256"}, f"{role} lacks path/SHA-256")
    path = resolve(base, str(record["path"]))
    actual = sha256_file(path)
    require(actual == str(record["sha256"]), f"{role} SHA-256 mismatch")
    return path


def check_thread_caps() -> None:
    wrong = {name: os.environ.get(name) for name in THREAD_VARS if os.environ.get(name) != "1"}
    if wrong:
        raise RuntimeError(f"all six thread variables must equal '1': {wrong}")


def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class DurableWal:
    def __init__(self, path: Path, header: Dict[str, object]):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", encoding="utf-8", buffering=1)
        self.append(header)

    def append(self, payload: Dict[str, object]) -> None:
        self.stream.write(json.dumps(payload, sort_keys=True) + "\n")
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


def _load_ode_path(source: Path) -> np.ndarray:
    with np.load(source, allow_pickle=False) as saved:
        require(set(saved.files) >= {"coordinates30_A", "endpoint0_A", "endpoint1_A", "path32_sha256"}, "ODE archive fields missing")
        internal = np.asarray(saved["coordinates30_A"], dtype=np.float64)
        endpoint0 = np.asarray(saved["endpoint0_A"], dtype=np.float64)
        endpoint1 = np.asarray(saved["endpoint1_A"], dtype=np.float64)
        embedded = str(np.asarray(saved["path32_sha256"]).item())
    require(internal.shape == (N_INTERNAL, N_ATOMS, 3), "ODE internal coordinate shape mismatch")
    require(endpoint0.shape == endpoint1.shape == (N_ATOMS, 3), "ODE endpoint shape mismatch")
    path3 = np.concatenate((endpoint0[None], internal, endpoint1[None]), axis=0)
    require(official_path32_sha256(path3) == embedded, "ODE embedded official path SHA mismatch")
    return np.ascontiguousarray(path3.reshape(N_IMAGES, DIMENSION))


def _scalar(saved, key: str):
    return np.asarray(saved[key]).item()


def _load_ivm_path(source: Path, record: Dict[str, object]) -> Tuple[np.ndarray, float, int]:
    keys = record.get("replay_keys", {})
    defaults = {
        "stored_paths": "stored_paths", "stored_path_calls": "stored_path_calls",
        "history_calls": "calls_per_image", "counter_vectors": "counter_vectors",
        "i_perp": "i_perp", "reparam_committed": "reparam_committed",
    }
    names = {key: str(keys.get(key, value)) for key, value in defaults.items()}
    with np.load(source, allow_pickle=False) as saved:
        require(set(names.values()).issubset(saved.files), "IVM replay history fields missing")
        paths = np.asarray(saved[names["stored_paths"]], dtype=np.float64)
        stored_calls = np.asarray(saved[names["stored_path_calls"]], dtype=np.int64)
        history_calls = np.asarray(saved[names["history_calls"]], dtype=np.int64)
        counters = np.asarray(saved[names["counter_vectors"]], dtype=np.int64)
        values = np.asarray(saved[names["i_perp"]], dtype=np.float64)
        committed = np.asarray(saved[names["reparam_committed"]], dtype=bool)
    call = int(record["path_identifier"]["calls_per_image"])
    stored_index = np.flatnonzero(stored_calls == call)
    history_index = np.flatnonzero(history_calls == call)
    require(stored_index.size == history_index.size == 1, "IVM replay call is not unique")
    hi = int(history_index[0])
    require(bool(committed[hi]), "IVM replay path was not committed")
    require(counters.shape == (len(history_calls), N_IMAGES), "IVM counter history shape mismatch")
    require(np.all(counters[:, (0, -1)] == 0), "IVM endpoint counter is nonzero")
    path = np.ascontiguousarray(paths[int(stored_index[0])], dtype=np.float64)
    return path, float(values[hi]), int(counters[hi].sum(dtype=np.int64))


def _selection_rows(selection: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    rows = selection.get("official_paid_states")
    require(isinstance(rows, list), "targeted R2 selection lacks official_paid_states")
    return {int(row["optimization_N_F"]): row for row in rows}


def validate_and_load(contract_path: Path, host_role: str):
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(contract.get("schema") == SCHEMA, "unsupported contract schema")
    require(contract.get("frozen") is True, "contract is not frozen")
    require(int(contract.get("jobs", -1)) == 1, "contract jobs must equal 1")
    require(host_role in {"local", "remote"}, "invalid host role")
    require(contract.get("metric_policy") == METRIC_POLICY, "metric policy mismatch")
    evaluator = require_file(contract["evaluator"], contract_path, "evaluator")
    require(evaluator == Path(__file__).resolve(), "contract binds a different evaluator")
    bb_core = require_file(contract["bb_core"], contract_path, "bb_core")
    require(bb_core == (SRC / "bb_core.py").resolve(), "contract binds a different bb_core")
    topology = require_file(contract["topology"], contract_path, "topology")
    locks = [Path(value).expanduser().resolve() for value in contract["host_local_lock_paths"]]
    require(len(locks) == 2 and len(set(locks)) == 2, "exactly two distinct lock paths required")
    paths = contract.get("paths")
    require(isinstance(paths, list) and len(paths) == 4, "contract must contain exactly four paths")
    methods = {str(row.get("method")) for row in paths}
    require(methods in ({"ODE-NEB"}, {"IVM"}), "contract must be method-homogeneous")
    method = next(iter(methods))
    require(all(row.get("host_role") == host_role for row in paths), "host-role assignment mismatch")
    selection_rows = None
    if method == "ODE-NEB":
        selection_path = require_file(contract["targeted_R2_selection"], contract_path, "targeted R2 selection")
        selection_rows = _selection_rows(json.loads(selection_path.read_text(encoding="utf-8")))
        require(tuple(sorted(int(row["optimization_N_F"]) for row in paths)) == ODE_NF, "ODE contract must bind NF60/150/300/1320")
    else:
        require(tuple(sorted(int(row["path_identifier"]["calls_per_image"]) for row in paths)) == IVM_CALLS, "IVM contract must bind replay calls2/5/10/50")
    loaded = []
    for row in paths:
        source = require_file(row["source"], contract_path, f"{method} source")
        if method == "ODE-NEB":
            nf = int(row["optimization_N_F"])
            selected = selection_rows.get(nf)
            require(selected is not None, f"NF{nf} absent from targeted R2 inventory")
            require(resolve(selection_path, str(selected["coordinate_archive"])) == source, f"NF{nf} archive differs from targeted R2")
            require(str(selected["coordinate_archive_sha256"]) == str(row["source"]["sha256"]), f"NF{nf} archive SHA differs from targeted R2")
            path = _load_ode_path(source)
            online_i = np.nan
            computed_nf = nf
            require(row.get("force_policy") == "fresh_30_internal_joint_EF", "ODE force policy mismatch")
        else:
            path, online_i, computed_nf = _load_ivm_path(source, row)
            nf = int(row["optimization_N_F"])
            require(computed_nf == nf, "IVM replay history N_F mismatch")
            require(row.get("force_policy") == "online_paid_history_zero_diagnostic_force", "IVM force policy mismatch")
        require(path.shape == (N_IMAGES, DIMENSION), "selected path shape mismatch")
        path_sha = sha256_array(path)
        require(path_sha == str(row["path_sha256"]), "selected path SHA mismatch")
        if method == "ODE-NEB":
            require(official_path32_sha256(path.reshape(N_IMAGES, N_ATOMS, 3)) == str(row["official_path32_sha256"]), f"NF{nf} contract official path SHA mismatch")
            require(str(row["official_path32_sha256"]) == str(selected["official_path32_sha256"]), f"NF{nf} official path SHA differs from targeted R2")
        loaded.append({**row, "source_path": source, "path": path, "online_i_perp": online_i, "computed_nf": computed_nf})
    reuse = {}
    for bundle in contract.get("allowed_pass_energy_reuse_bundles", []):
        result = require_file(bundle["result"], contract_path, "energy reuse result")
        audit_path = require_file(bundle["audit"], contract_path, "energy reuse audit")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        require(audit.get("status") == "PASS", "energy reuse audit is not PASS")
        require(audit.get("output_sha256") == sha256_file(result), "energy reuse audit/output mismatch")
        with np.load(result, allow_pickle=False) as saved:
            required = {"selected_path_sha256", "dense_coordinate_sha256", "energy_eV", "energy_sha256", "number_of_points", "topology_sha256", "diagnostic_force_requests_per_path"}
            require(required.issubset(saved.files), "energy reuse result lacks required arrays")
            require(int(_scalar(saved, "number_of_points")) == N_DENSE, "energy reuse point count mismatch")
            require(str(_scalar(saved, "topology_sha256")) == sha256_file(topology), "energy reuse topology mismatch")
            require(np.all(np.asarray(saved["diagnostic_force_requests_per_path"]) == 0), "energy reuse requested force")
            for index, path_sha in enumerate(np.asarray(saved["selected_path_sha256"]).astype(str)):
                energies = np.asarray(saved["energy_eV"][index], dtype=np.float64)
                energy_sha = str(np.asarray(saved["energy_sha256"])[index])
                require(energies.shape == (N_DENSE,) and sha256_array(energies) == energy_sha, "energy reuse vector hash mismatch")
                entry = {
                    "dense_sha256": str(np.asarray(saved["dense_coordinate_sha256"])[index]),
                    "energies": energies,
                    "energy_sha256": energy_sha,
                    "source_result": str(result),
                    "source_result_sha256": sha256_file(result),
                    "source_audit": str(audit_path),
                    "source_audit_sha256": sha256_file(audit_path),
                    "source_row_index": index,
                }
                previous = reuse.get(path_sha)
                require(previous is None or (previous["dense_sha256"] == entry["dense_sha256"] and previous["energy_sha256"] == energy_sha), "conflicting energy reuse bundles")
                reuse[path_sha] = entry
    return contract, method, topology, locks, loaded, reuse


class EnergyOnly:
    def __init__(self, topology: Path):
        self.backend = OpenMMBackend(topology, platform_preference=("CPU",), precision="double", label="bb_dense_four_energy_only")

    def evaluate(self, coordinate: np.ndarray) -> float:
        from openmm.unit import angstrom, kilojoule_per_mole
        point = np.asarray(coordinate, dtype=np.float64)
        self.backend.context.setPositions(point.reshape(N_ATOMS, 3) * angstrom)
        state = self.backend.context.getState(getEnergy=True, getForces=False)
        value = float(state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)) / self.backend.KJ_PER_MOL_PER_EV
        require(np.isfinite(value), "non-finite diagnostic energy")
        return value


def run_evaluation(contract_path: Path, contract, method: str, topology: Path, locks: Sequence[Path], rows, reuse, output: Path):
    force_wal_path = output.with_suffix(".force.wal.jsonl")
    energy_wal_path = output.with_suffix(".energy.wal.jsonl")
    audit_path = output.with_suffix(".audit.json")
    force_wal = DurableWal(force_wal_path, {"schema": FORCE_WAL_SCHEMA, "created_utc": utc_now(), "contract": str(contract_path), "contract_sha256": sha256_file(contract_path), "optimization_N_F_contribution": 0})
    energy_wal = DurableWal(energy_wal_path, {"schema": ENERGY_WAL_SCHEMA, "created_utc": utc_now(), "contract": str(contract_path), "contract_sha256": sha256_file(contract_path), "optimization_N_F_contribution": 0})
    force_backend = None
    energy_evaluator = None
    evaluated = []
    try:
        with acquire_two_locks(locks) as held_locks:
            for row_index, row in enumerate(rows):
                path = row["path"]
                path_sha = str(row["path_sha256"])
                internal_forces = np.full((N_INTERNAL, DIMENSION), np.nan, dtype=np.float64)
                internal_joint_energy = np.full(N_INTERNAL, np.nan, dtype=np.float64)
                if method == "ODE-NEB":
                    if force_backend is None:
                        force_backend = OpenMMBackend(topology, platform_preference=("CPU",), precision="double", label="bb_dense_four_force")
                    for internal_index, coordinate in enumerate(path[1:-1]):
                        attempt = int(force_backend.raw_joint_attempts) + 1
                        start = {"event": "attempt_started", "utc": utc_now(), "row_index": row_index, "internal_image_index": internal_index + 1, "attempt": attempt, "path_sha256": path_sha, "coordinate_sha256": sha256_array(coordinate)}
                        force_wal.append(start)
                        try:
                            energy, force = force_backend.evaluate_joint(coordinate)
                        except Exception as error:
                            force_wal.append({**start, "event": "attempt_failed", "utc": utc_now(), "error": repr(error)})
                            raise
                        internal_joint_energy[internal_index] = energy
                        internal_forces[internal_index] = force
                        force_wal.append({**start, "event": "attempt_succeeded", "utc": utc_now(), "energy_eV": float(energy), "force_sha256": sha256_array(force)})
                    diagnostic = path_diagnostics(path, np.zeros(N_IMAGES, dtype=np.float64), internal_forces)
                    i_perp = float(diagnostic["i_perp"])
                    force_attempts = N_INTERNAL
                else:
                    i_perp = float(row["online_i_perp"])
                    require(np.isfinite(i_perp) and i_perp >= 0, "invalid replay-history I_perp")
                    force_attempts = 0
                    force_wal.append({"event": "online_paid_history_reuse", "utc": utc_now(), "row_index": row_index, "path_sha256": path_sha, "diagnostic_force_attempts": 0})
                dense, geometry = uniform_arc500(path)
                dense_sha = sha256_array(dense)
                reused = reuse.get(path_sha)
                if reused is not None:
                    require(reused["dense_sha256"] == dense_sha, "exact energy reuse dense-coordinate SHA mismatch")
                    energies = reused["energies"].copy()
                    fresh_energy = 0
                    lineage = "exact_PASS_reuse"
                    energy_wal.append({"event": "exact_PASS_reuse", "utc": utc_now(), "row_index": row_index, "path_sha256": path_sha, "dense_coordinate_sha256": dense_sha, "energy_sha256": reused["energy_sha256"], "source_result": reused["source_result"], "source_result_sha256": reused["source_result_sha256"], "source_audit": reused["source_audit"], "source_audit_sha256": reused["source_audit_sha256"], "source_row_index": reused["source_row_index"]})
                else:
                    if energy_evaluator is None:
                        energy_evaluator = EnergyOnly(topology)
                    energies = np.empty(N_DENSE, dtype=np.float64)
                    for point_index, coordinate in enumerate(dense):
                        attempt = sum(int(item.get("fresh_energy_attempts", 0)) for item in evaluated) + point_index + 1
                        start = {"event": "attempt_started", "utc": utc_now(), "row_index": row_index, "point_index": point_index, "attempt": attempt, "path_sha256": path_sha, "coordinate_sha256": sha256_array(coordinate)}
                        energy_wal.append(start)
                        try:
                            energies[point_index] = energy_evaluator.evaluate(coordinate)
                        except Exception as error:
                            energy_wal.append({**start, "event": "attempt_failed", "utc": utc_now(), "error": repr(error)})
                            raise
                        energy_wal.append({**start, "event": "attempt_succeeded", "utc": utc_now(), "energy_eV": float(energies[point_index])})
                    fresh_energy = N_DENSE
                    lineage = "fresh_energy_only"
                require(np.all(np.isfinite(energies)), "non-finite dense energy vector")
                energy_sha = sha256_array(energies)
                if reused is not None:
                    require(energy_sha == reused["energy_sha256"], "reused energy SHA mismatch")
                evaluated.append({**row, **geometry, "dense_sha256": dense_sha, "energies": energies, "energy_sha256": energy_sha, "delta_E": float((np.max(energies) - energies[0]) * EV_TO_KCAL), "i_perp": i_perp, "internal_forces": internal_forces, "internal_joint_energy": internal_joint_energy, "force_sha256": "" if force_attempts == 0 else sha256_array(internal_forces), "joint_energy_sha256": "" if force_attempts == 0 else sha256_array(internal_joint_energy), "force_attempts": force_attempts, "fresh_energy_attempts": fresh_energy, "energy_lineage": lineage, "energy_reuse": reused})
        if force_backend is not None:
            require(force_backend.raw_joint_attempts == force_backend.raw_joint_evaluations == N_INTERNAL * len(rows), "force backend ledger mismatch")
        if energy_evaluator is not None:
            require(energy_evaluator.backend.raw_joint_attempts == 0, "energy-only diagnostic incremented joint force ledger")
        atomic_save_npz(
            output,
            method=np.asarray(method), labels=np.asarray([str(row["label"]) for row in evaluated]),
            optimization_N_F=np.asarray([int(row["optimization_N_F"]) for row in evaluated], dtype=np.int64),
            path_identifier_json=np.asarray([json.dumps(row["path_identifier"], sort_keys=True) for row in evaluated]),
            selected_path=np.asarray([row["path"] for row in evaluated]),
            selected_path_sha256=np.asarray([str(row["path_sha256"]) for row in evaluated]),
            discrete_I_perp=np.asarray([row["i_perp"] for row in evaluated], dtype=np.float64),
            internal_diagnostic_force_eV_per_A=np.asarray([row["internal_forces"] for row in evaluated]),
            internal_diagnostic_force_sha256=np.asarray([row["force_sha256"] for row in evaluated]),
            internal_joint_energy_eV=np.asarray([row["internal_joint_energy"] for row in evaluated]),
            internal_joint_energy_sha256=np.asarray([row["joint_energy_sha256"] for row in evaluated]),
            diagnostic_force_attempts_per_path=np.asarray([row["force_attempts"] for row in evaluated], dtype=np.int64),
            fixed_endpoint_diagnostic_force_attempts_per_path=np.zeros(len(evaluated), dtype=np.int64),
            dense_coordinate_sha256=np.asarray([row["dense_sha256"] for row in evaluated]),
            energy_eV=np.asarray([row["energies"] for row in evaluated]),
            energy_sha256=np.asarray([row["energy_sha256"] for row in evaluated]),
            densified_Delta_E_kcal_per_mol=np.asarray([row["delta_E"] for row in evaluated], dtype=np.float64),
            diagnostic_energy_lineage=np.asarray([row["energy_lineage"] for row in evaluated]),
            diagnostic_energy_attempts_lineage_per_path=np.full(len(evaluated), N_DENSE, dtype=np.int64),
            diagnostic_energy_attempts_fresh_per_path=np.asarray([row["fresh_energy_attempts"] for row in evaluated], dtype=np.int64),
            total_arclength_A=np.asarray([row["total_arclength_A"] for row in evaluated]),
            dense_spacing_A=np.asarray([row["dense_spacing_A"] for row in evaluated]),
            coordinate_space=np.asarray(METRIC_POLICY["coordinate_space"]), sampling=np.asarray(METRIC_POLICY["sampling"]), interpolation=np.asarray(METRIC_POLICY["interpolation"]), number_of_points=np.asarray(N_DENSE, dtype=np.int64), grid_formula=np.asarray(METRIC_POLICY["grid_formula"]),
            contract=np.asarray(str(contract_path)), contract_sha256=np.asarray(sha256_file(contract_path)), evaluator_sha256=np.asarray(sha256_file(Path(__file__))), bb_core_sha256=np.asarray(sha256_file(SRC / "bb_core.py")), topology=np.asarray(str(topology)), topology_sha256=np.asarray(sha256_file(topology)), force_wal=np.asarray(str(force_wal_path)), energy_wal=np.asarray(str(energy_wal_path)), held_host_locks=np.asarray(held_locks), optimization_N_F_contribution=np.asarray(0, dtype=np.int64),
        )
    finally:
        force_wal.close()
        energy_wal.close()
    details = []
    for row in evaluated:
        reused = row["energy_reuse"]
        details.append({
            "label": str(row["label"]), "optimization_N_F": int(row["optimization_N_F"]), "path_identifier": row["path_identifier"], "path_sha256": str(row["path_sha256"]), "dense_coordinate_sha256": row["dense_sha256"], "discrete_I_perp": row["i_perp"], "densified_Delta_E_kcal_per_mol": row["delta_E"], "diagnostic_force_attempts": row["force_attempts"], "diagnostic_force_successes": row["force_attempts"], "fixed_endpoint_diagnostic_force_attempts": 0, "force_sha256": row["force_sha256"], "joint_energy_sha256": row["joint_energy_sha256"], "diagnostic_energy_attempts_lineage": N_DENSE, "diagnostic_energy_attempts_fresh": row["fresh_energy_attempts"], "diagnostic_energy_attempts_reused": N_DENSE - row["fresh_energy_attempts"], "energy_sha256": row["energy_sha256"], "energy_lineage": row["energy_lineage"], "energy_reuse_source": None if reused is None else {key: reused[key] for key in ("source_result", "source_result_sha256", "source_audit", "source_audit_sha256", "source_row_index")}, "optimization_N_F_contribution": 0,
        })
    audit = {
        "schema": AUDIT_SCHEMA, "status": "PASS", "completed_utc": utc_now(), "method": method,
        "contract": str(contract_path), "contract_sha256": sha256_file(contract_path), "evaluator_sha256": sha256_file(Path(__file__)), "bb_core_sha256": sha256_file(SRC / "bb_core.py"), "topology_sha256": sha256_file(topology),
        "output": str(output), "output_sha256": sha256_file(output), "force_wal": str(force_wal_path), "force_wal_sha256": sha256_file(force_wal_path), "energy_wal": str(energy_wal_path), "energy_wal_sha256": sha256_file(energy_wal_path), "held_host_locks": held_locks,
        "path_count": len(evaluated), "diagnostic_force_attempts": sum(row["force_attempts"] for row in evaluated), "diagnostic_force_successes": sum(row["force_attempts"] for row in evaluated), "fixed_endpoint_diagnostic_force_attempts": 0, "diagnostic_energy_attempts_lineage": N_DENSE * len(evaluated), "diagnostic_energy_attempts_fresh": sum(row["fresh_energy_attempts"] for row in evaluated), "diagnostic_energy_attempts_reused": N_DENSE * len(evaluated) - sum(row["fresh_energy_attempts"] for row in evaluated), "optimization_N_F_contribution": 0, "paths": details,
    }
    atomic_write_json(audit_path, audit)


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
    artifacts = (output, output.with_suffix(".audit.json"), output.with_suffix(".force.wal.jsonl"), output.with_suffix(".energy.wal.jsonl"))
    if any(path.exists() for path in artifacts):
        raise FileExistsError("refusing to overwrite an output/audit/WAL artifact")
    contract, method, topology, locks, rows, reuse = validate_and_load(contract_path, args.host_role)
    summary = {"status": "ZERO_OPENMM_PREFLIGHT_PASS", "contract": str(contract_path), "contract_sha256": sha256_file(contract_path), "method": method, "host_role": args.host_role, "path_count": len(rows), "optimization_N_F": [int(row["optimization_N_F"]) for row in rows], "path_sha256": [str(row["path_sha256"]) for row in rows], "exact_energy_reuse_count": sum(str(row["path_sha256"]) in reuse for row in rows), "fresh_force_attempts_planned": (N_INTERNAL * len(rows) if method == "ODE-NEB" else 0), "fresh_energy_attempts_planned": sum(str(row["path_sha256"]) not in reuse for row in rows) * N_DENSE, "optimization_N_F_contribution": 0}
    if args.preflight_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    run_evaluation(contract_path, contract, method, topology, locks, rows, reuse, output)


if __name__ == "__main__":
    main()
