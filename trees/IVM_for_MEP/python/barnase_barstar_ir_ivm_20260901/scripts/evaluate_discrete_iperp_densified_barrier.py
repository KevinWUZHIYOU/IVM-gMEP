#!/usr/bin/env python3
"""Evaluate the two principal path diagnostics for BB IR-IVM.

The reported diagnostics follow the terminology of the manuscript:

* the discrete, unnormalized normal residual ``I_perp`` on the 32-image
  chain; and
* the densified 0 K forward barrier ``Delta E`` on exactly 500 points placed
  uniformly in Cartesian arclength along the same eligible image chain.

The densified diagnostic requests energy only.  It never requests force and
its attempts are recorded separately from the optimization Number of Forces
``N_F``.  The script is strictly offline and refuses to select an open,
rolled-back, or terminal partial path state.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


N_IMAGES = 32
N_INTERNAL = 30
N_DENSE_POINTS = 500
DIMENSION = 9396
EV_TO_KCAL_PER_MOL = 23.0605
COORDINATE_SPACE = "unweighted_3N_cartesian"
INTERPOLATION = "piecewise_linear"
SAMPLING = "uniform_cartesian_arclength"
GRID_FORMULA = "s_j=j*L/499,j=0,...,499"
CONTRACT_SCHEMA = "bb-irivm-discrete-iperp-densified-barrier-contract-v1"
METRIC_AMENDMENT_SHA256 = (
    "ca547dd86a119e0dd4b6049e296ae9616d335248c3df1ce1f77f7fcb91500bc5"
)
ALPHA_SCOPE_R2_SHA256 = (
    "9dd4dda96d428df1e7acd07533061cc77f8e8fe55c1d38bdda14f24510be6e0f"
)
DEFAULT_CANONICAL_LOCK = "/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock"
DEFAULT_DOWNLOADS_LOCK = "/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_save_npz(path: Path, **arrays) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)


def atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


@contextmanager
def acquire_host_locks(paths: Sequence[Path]):
    """Hold both host-local BB locks during every physical energy evaluation."""

    ordered = sorted({Path(path).expanduser().resolve() for path in paths}, key=str)
    handles = []
    try:
        for path in ordered:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                raise RuntimeError(f"another BB task owns {path}") from error
            handles.append(handle)
        yield [str(path) for path in ordered]
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def resolve_path(contract_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = contract_path.parent / candidate
    return candidate.resolve()


def uniform_arclength_resample(
    path: np.ndarray, count: int = N_DENSE_POINTS
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Sample a piecewise-linear image chain at equal Cartesian arclength."""

    curve = np.asarray(path, dtype=np.float64)
    if curve.shape != (N_IMAGES, DIMENSION):
        raise ValueError(
            f"expected a ({N_IMAGES}, {DIMENSION}) image chain, got {curve.shape}"
        )
    if int(count) != N_DENSE_POINTS:
        raise ValueError(f"the registered diagnostic requires {N_DENSE_POINTS} points")
    if not np.all(np.isfinite(curve)):
        raise ValueError("image-chain coordinates must be finite")
    segment_lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    if np.any(segment_lengths <= 0.0) or not np.all(np.isfinite(segment_lengths)):
        raise ValueError("every image-chain segment must have positive finite length")
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
    return np.ascontiguousarray(dense), {
        "total_arclength_A": total_length,
        "minimum_original_segment_length_A": float(np.min(segment_lengths)),
        "maximum_original_segment_length_A": float(np.max(segment_lengths)),
        "dense_spacing_A": total_length / float(int(count) - 1),
    }


def _scalar(saved: np.lib.npyio.NpzFile, key: str):
    return np.asarray(saved[key]).item()


def _require_hash(path: Path, expected: str, role: str) -> None:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{role} SHA-256 mismatch: {actual} != {expected}")


def _validate_contract(contract_path: Path) -> Dict[str, object]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unsupported diagnostic contract schema")
    protocol = contract.get("diagnostic_protocol", {})
    expected_protocol = {
        "coordinate_space": COORDINATE_SPACE,
        "interpolation": INTERPOLATION,
        "sampling": SAMPLING,
        "number_of_points": N_DENSE_POINTS,
        "include_both_endpoints": True,
        "grid_formula": GRID_FORMULA,
        "force_evaluations_per_path": 0,
        "optimization_N_F_contribution": 0,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(
                f"diagnostic protocol {key!r} must be {expected!r}, "
                f"got {protocol.get(key)!r}"
            )
    bindings = contract.get("bindings", {})
    metric = bindings.get("metric_policy_amendment", {})
    alpha_scope = bindings.get("alpha_scope_R2", {})
    if metric.get("sha256") != METRIC_AMENDMENT_SHA256:
        raise ValueError("metric-policy amendment binding mismatch")
    if alpha_scope.get("sha256") != ALPHA_SCOPE_R2_SHA256:
        raise ValueError("alpha-scope R2 binding mismatch")
    for role, record in (
        ("metric-policy amendment", metric),
        ("alpha-scope R2", alpha_scope),
    ):
        bound_path = resolve_path(contract_path, str(record["path"]))
        _require_hash(bound_path, str(record["sha256"]), role)
    runs = contract.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("diagnostic contract contains no optimization runs")
    labels = [str(record.get("label", "")) for record in runs]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("optimization-run labels must be nonempty and unique")
    return contract


def _validate_optimization_audit(audit: Dict[str, object], budget: int) -> int:
    required_equal = (
        "backend_raw_joint_attempts",
        "backend_raw_joint_evaluations",
        "sum_image_joint_evaluations",
        "optimization_internal_nf",
        "number_of_forces",
    )
    if audit.get("consistent") is not True:
        raise ValueError("optimization result has an inconsistent NoF audit")
    actual = int(audit[required_equal[0]])
    if any(int(audit[key]) != actual for key in required_equal):
        raise ValueError("optimization result has inconsistent physical NoF counters")
    if actual > int(budget):
        raise ValueError("optimization result overshot its registered NoF budget")
    if int(audit["endpoint_getstate_attempts"]) != 0:
        raise ValueError("optimization result charged a fixed endpoint")
    if int(audit["endpoint_joint_ef_cache_misses"]) != 0:
        raise ValueError("optimization result missed a fixed-endpoint cache entry")
    if int(audit["endpoint_inherited_joint_ef_cache_entries"]) != 2:
        raise ValueError("optimization result does not contain both inherited endpoints")
    return actual


def _validate_terminal_budget_metadata(saved, budget: int, actual: int) -> None:
    """Accept equality or a witnessed atomic full-band remainder below 30."""

    if int(_scalar(saved, "number_of_forces")) != actual:
        raise ValueError("result/audit Number of Forces mismatch")
    if int(_scalar(saved, "optimization_internal_nf")) != actual:
        raise ValueError("result optimization Number of Forces mismatch")
    if int(_scalar(saved, "total_internal_nf_consumed")) != actual:
        raise ValueError("result consumed Number of Forces mismatch")
    if int(_scalar(saved, "total_internal_nf_budget")) != int(budget):
        raise ValueError("result registered budget mismatch")
    remaining = int(_scalar(saved, "total_internal_nf_remaining"))
    if remaining != int(budget) - actual:
        raise ValueError("result budget remainder mismatch")
    if not bool(_scalar(saved, "total_internal_nf_budget_no_overshoot")):
        raise ValueError("result overshot its budget")
    exhausted = bool(_scalar(saved, "total_internal_nf_budget_exhausted"))
    if actual == int(budget):
        if not exhausted:
            raise ValueError("result exact budget is not marked exhausted")
        return
    if exhausted or not (0 < remaining < 30):
        raise ValueError("result atomic budget remainder is invalid")
    if str(_scalar(saved, "total_internal_nf_budget_stop_reason")) != (
        "total_internal_joint_budget_insufficient_for_full_band"
    ):
        raise ValueError("result did not stop at an atomic full-band boundary")
    if int(
        _scalar(saved, "total_internal_nf_full_band_preflight_zero_step_stops")
    ) < 1:
        raise ValueError("result lacks a full-band preflight stop")
    if int(_scalar(saved, "total_internal_nf_complete_band_budget_stop_count")) < 1:
        raise ValueError("result lacks a complete-band budget stop")
    if not bool(_scalar(saved, "total_internal_nf_terminal_partial_tail_present")):
        raise ValueError("atomic remainder result has no terminal partial tail")
    if not bool(
        _scalar(saved, "total_internal_nf_terminal_partial_tail_quarantined")
    ):
        raise ValueError("atomic remainder result did not quarantine its partial tail")


def _history_index_for_call(calls: np.ndarray, call: int) -> int:
    matching = np.flatnonzero(calls == int(call))
    if matching.size != 1:
        raise ValueError(f"stored path call {call} is not unique in the history")
    return int(matching[0])


def load_eligible_rows(
    contract_path: Path, contract: Dict[str, object]
) -> Tuple[np.ndarray, float, int, List[Dict[str, object]]]:
    """Load every registered post-initial reparameterized image chain."""

    common_record = contract["common_initial_path"]
    common_file = resolve_path(contract_path, str(common_record["path"]))
    _require_hash(common_file, str(common_record["file_sha256"]), "common initial path")
    with np.load(common_file, allow_pickle=False) as saved:
        if tuple(saved.files) != ("initial_path",):
            raise ValueError("common initial-path archive has unexpected fields")
        common_path = np.asarray(saved["initial_path"], dtype=np.float64)
    if common_path.shape != (N_IMAGES, DIMENSION):
        raise ValueError("common initial path has the wrong shape")
    if sha256_array(common_path) != str(common_record["array_sha256"]):
        raise ValueError("common initial-path array hash mismatch")

    initial_i_perp: float | None = None
    initial_nf: int | None = None
    rows: List[Dict[str, object]] = []
    for run in contract["runs"]:
        label = str(run["label"])
        result = resolve_path(contract_path, str(run["source_result"]))
        result_audit = resolve_path(contract_path, str(run["source_audit"]))
        _require_hash(result, str(run["source_result_sha256"]), f"{label} result")
        _require_hash(
            result_audit, str(run["source_audit_sha256"]), f"{label} audit"
        )
        audit = json.loads(result_audit.read_text(encoding="utf-8"))
        budget = int(run["registered_optimization_N_F"])
        actual_nf = _validate_optimization_audit(audit, budget)
        with np.load(result, allow_pickle=False) as saved:
            required = {
                "run_label",
                "initial_path",
                "stored_paths",
                "stored_path_calls",
                "calls_per_image",
                "counter_vectors",
                "i_perp",
                "reparam_committed",
                "number_of_forces",
                "total_internal_nf_budget_exhausted",
                "total_internal_nf_budget_no_overshoot",
                "total_internal_nf_budget",
                "total_internal_nf_consumed",
                "total_internal_nf_remaining",
                "total_internal_nf_budget_stop_reason",
                "total_internal_nf_full_band_preflight_zero_step_stops",
                "total_internal_nf_complete_band_budget_stop_count",
                "total_internal_nf_terminal_partial_tail_present",
                "total_internal_nf_terminal_partial_tail_quarantined",
            }
            missing = sorted(required.difference(saved.files))
            if missing:
                raise ValueError(f"{label} result is missing {missing}")
            if str(_scalar(saved, "run_label")) != label:
                raise ValueError(f"{label} result label mismatch")
            _validate_terminal_budget_metadata(saved, budget, actual_nf)

            source_initial = np.asarray(saved["initial_path"], dtype=np.float64)
            if not np.array_equal(source_initial, common_path):
                raise ValueError(f"{label} does not use the common initial path")
            paths = np.asarray(saved["stored_paths"], dtype=np.float64)
            stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
            calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
            counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
            i_perp = np.asarray(saved["i_perp"], dtype=np.float64)
            committed = np.asarray(saved["reparam_committed"], dtype=bool)

        if paths.shape != (len(stored_calls), N_IMAGES, DIMENSION):
            raise ValueError(f"{label} stored path array has the wrong shape")
        if (
            calls.ndim != 1
            or counters.shape != (len(calls), N_IMAGES)
            or i_perp.shape != (len(calls),)
            or committed.shape != (len(calls),)
        ):
            raise ValueError(f"{label} history arrays are inconsistent")
        if np.any(counters < 0) or np.any(counters[:, (0, -1)] != 0):
            raise ValueError(f"{label} counter vectors violate endpoint accounting")
        total_nf = counters.sum(axis=1, dtype=np.int64)
        if np.any(np.diff(total_nf) < 0):
            raise ValueError(f"{label} Number-of-Forces history is not cumulative")
        initial_index = _history_index_for_call(calls, int(stored_calls[0]))
        if int(stored_calls[0]) != 1 or initial_index != 0:
            raise ValueError(f"{label} does not store the initial image chain first")
        this_initial_i = float(i_perp[initial_index])
        this_initial_nf = int(total_nf[initial_index])
        if initial_i_perp is None:
            initial_i_perp = this_initial_i
            initial_nf = this_initial_nf
        elif this_initial_i != initial_i_perp or this_initial_nf != initial_nf:
            raise ValueError("optimization runs disagree on the common initial state")

        selected_calls = []
        for stored_index, call in enumerate(stored_calls):
            history_index = _history_index_for_call(calls, int(call))
            if int(call) == 1:
                continue
            if not bool(committed[history_index]):
                continue
            path = np.asarray(paths[stored_index], dtype=np.float64)
            selected_calls.append(int(call))
            rows.append(
                {
                    "label": label,
                    "alpha": float(run["alpha"]),
                    "r": float(run["r"]),
                    "m_max": int(run["m_max"]),
                    "source_result": str(result),
                    "source_result_sha256": str(run["source_result_sha256"]),
                    "source_audit": str(result_audit),
                    "source_audit_sha256": str(run["source_audit_sha256"]),
                    "history_index": history_index,
                    "calls_per_image": int(call),
                    "optimization_N_F": int(total_nf[history_index]),
                    "discrete_I_perp": float(i_perp[history_index]),
                    "path": path,
                    "path_sha256": sha256_array(path),
                }
            )
        expected_calls = [int(value) for value in run["eligible_committed_calls"]]
        if selected_calls != expected_calls:
            raise ValueError(
                f"{label} eligible committed calls {selected_calls} "
                f"do not match the frozen contract {expected_calls}"
            )
    if initial_i_perp is None or initial_nf is None:
        raise RuntimeError("no common initial state was recovered")
    return common_path, initial_i_perp, initial_nf, rows


class EnergyOnlyOpenMMEvaluator:
    """Canonical BB CPU/double evaluator that never requests a force."""

    def __init__(self, topology: Path):
        from bb_core import OpenMMBackend  # pylint: disable=import-outside-toplevel

        self.backend = OpenMMBackend(
            topology,
            platform_preference=("CPU",),
            precision="double",
            label="bb_densified_energy_only",
        )
        self.attempts = 0
        self.successes = 0
        self.failures = 0

    def evaluate(self, positions_flat: np.ndarray) -> float:
        from openmm.unit import angstrom, kilojoule_per_mole

        positions = np.asarray(positions_flat, dtype=np.float64)
        if positions.shape != (self.backend.dimension,):
            raise ValueError("diagnostic coordinate has the wrong shape")
        if not np.all(np.isfinite(positions)):
            raise ValueError("diagnostic coordinate contains a non-finite value")
        self.backend.context.setPositions(
            positions.reshape(self.backend.n_atoms, 3) * angstrom
        )
        self.attempts += 1
        try:
            state = self.backend.context.getState(getEnergy=True, getForces=False)
            energy_kj = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
            energy = float(energy_kj) / self.backend.KJ_PER_MOL_PER_EV
            if not np.isfinite(energy):
                raise FloatingPointError("OpenMM returned non-finite energy")
        except Exception:
            self.failures += 1
            raise
        self.successes += 1
        return energy


def evaluate_paths(
    topology: Path,
    rows: Sequence[Dict[str, object]],
    wal_path: Path,
    reuse_rows: Dict[str, Dict[str, object]] | None = None,
) -> List[Dict[str, object]]:
    reusable = {} if reuse_rows is None else dict(reuse_rows)
    evaluator: EnergyOnlyOpenMMEvaluator | None = None
    evaluated: List[Dict[str, object]] = []
    with wal_path.open("x", encoding="utf-8", buffering=1) as wal:
        header = {
            "schema": "bb-densified-energy-wal-v1",
            "number_of_points_per_path": N_DENSE_POINTS,
            "force_requests": 0,
            "optimization_N_F_contribution": 0,
            "exact_path_reuse_enabled": bool(reusable),
        }
        wal.write(json.dumps(header, sort_keys=True) + "\n")
        for row_index, row in enumerate(rows):
            dense, geometry = uniform_arclength_resample(row["path"])
            dense_sha = sha256_array(dense)
            reused = reusable.get(str(row["path_sha256"]))
            if reused is not None:
                if str(reused["dense_coordinate_sha256"]) != dense_sha:
                    raise ValueError("exact-path reuse dense-coordinate hash mismatch")
                energies = np.asarray(reused["energy_eV"], dtype=np.float64).copy()
                wal.write(
                    json.dumps(
                        {
                            "row_index": row_index,
                            "status": "exact_path_energy_reuse",
                            "selected_path_sha256": row["path_sha256"],
                            "dense_coordinate_sha256": dense_sha,
                            "energy_sha256": reused["energy_sha256"],
                            "source_diagnostic": reused["source_diagnostic"],
                            "source_diagnostic_sha256": reused[
                                "source_diagnostic_sha256"
                            ],
                            "source_row_index": reused["source_row_index"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                current_attempts = 0
            else:
                if evaluator is None:
                    evaluator = EnergyOnlyOpenMMEvaluator(topology)
                energies = np.empty(N_DENSE_POINTS, dtype=np.float64)
                for point_index, point in enumerate(dense):
                    attempt = {
                        "row_index": row_index,
                        "point_index": point_index,
                        "attempt": evaluator.attempts + 1,
                    }
                    try:
                        energies[point_index] = evaluator.evaluate(point)
                    except Exception as error:
                        attempt.update(status="failed", error=repr(error))
                        wal.write(json.dumps(attempt, sort_keys=True) + "\n")
                        wal.flush()
                        os.fsync(wal.fileno())
                        raise
                    attempt.update(
                        status="success", energy_eV=float(energies[point_index])
                    )
                    wal.write(json.dumps(attempt, sort_keys=True) + "\n")
                current_attempts = N_DENSE_POINTS
            wal.flush()
            os.fsync(wal.fileno())
            if evaluator is not None and evaluator.backend.raw_joint_attempts != 0:
                raise RuntimeError("densified diagnostic requested a joint energy--force state")
            energy_sha = sha256_array(energies)
            if reused is not None and energy_sha != str(reused["energy_sha256"]):
                raise ValueError("exact-path reuse energy hash mismatch")
            if reused is None:
                # A contract can contain the same exact committed path under
                # more than one parameter setting.  Once its 500-point energy
                # vector has been evaluated in this invocation, reuse it for
                # all later hash-identical rows instead of spending another
                # 500 diagnostic energy calls.  This changes no optimization
                # state and contributes zero Number of Forces.
                reusable[str(row["path_sha256"])] = {
                    "dense_coordinate_sha256": dense_sha,
                    "energy_eV": energies.copy(),
                    "energy_sha256": energy_sha,
                    "source_diagnostic": str(wal_path),
                    "source_diagnostic_sha256": sha256_file(wal_path),
                    "source_row_index": row_index,
                }
            evaluated.append(
                {
                    **row,
                    **geometry,
                    "dense_coordinate_sha256": dense_sha,
                    "energy_eV": energies,
                    "energy_sha256": energy_sha,
                    "densified_Delta_E_kcal_per_mol": float(
                        (np.max(energies) - energies[0]) * EV_TO_KCAL_PER_MOL
                    ),
                    "diagnostic_energy_attempts": N_DENSE_POINTS,
                    "diagnostic_energy_successes": N_DENSE_POINTS,
                    "diagnostic_force_requests": 0,
                    "diagnostic_energy_reused": reused is not None,
                    "diagnostic_energy_attempts_this_evaluation": current_attempts,
                    "diagnostic_energy_source": (
                        str(reused["source_diagnostic"])
                        if reused is not None
                        else "fresh_current_evaluation"
                    ),
                }
            )
    expected_fresh = sum(
        int(row["diagnostic_energy_attempts_this_evaluation"])
        for row in evaluated
    )
    actual_attempts = 0 if evaluator is None else evaluator.attempts
    if actual_attempts != expected_fresh:
        raise RuntimeError("diagnostic energy-attempt ledger is incomplete")
    if evaluator is not None and (
        evaluator.successes != evaluator.attempts or evaluator.failures != 0
    ):
        raise RuntimeError("diagnostic energy evaluator recorded a failure")
    return evaluated


def load_exact_reuse_rows(
    diagnostics: Path,
    audit_path: Path,
    topology_sha256: str,
) -> Dict[str, Dict[str, object]]:
    """Load only fully audited, hash-identical 500-energy path records."""

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ValueError("reuse diagnostic audit did not pass")
    if audit.get("output_sha256") != sha256_file(diagnostics):
        raise ValueError("reuse diagnostic audit/output hash mismatch")
    if int(audit.get("diagnostic_force_requests", -1)) != 0:
        raise ValueError("reuse diagnostic requested a force")
    with np.load(diagnostics, allow_pickle=False) as saved:
        required = {
            "selected_path_sha256",
            "dense_coordinate_sha256",
            "energy_eV",
            "energy_sha256",
            "diagnostic_energy_attempts_per_path",
            "diagnostic_energy_successes_per_path",
            "diagnostic_force_requests_per_path",
            "coordinate_space",
            "interpolation",
            "sampling",
            "number_of_points",
            "grid_formula",
            "topology_sha256",
        }
        missing = sorted(required.difference(saved.files))
        if missing:
            raise ValueError(f"reuse diagnostic is missing {missing}")
        if str(_scalar(saved, "coordinate_space")) != COORDINATE_SPACE:
            raise ValueError("reuse diagnostic coordinate-space mismatch")
        if str(_scalar(saved, "interpolation")) != INTERPOLATION:
            raise ValueError("reuse diagnostic interpolation mismatch")
        if str(_scalar(saved, "sampling")) != SAMPLING:
            raise ValueError("reuse diagnostic sampling mismatch")
        if int(_scalar(saved, "number_of_points")) != N_DENSE_POINTS:
            raise ValueError("reuse diagnostic point-count mismatch")
        if str(_scalar(saved, "grid_formula")) != GRID_FORMULA:
            raise ValueError("reuse diagnostic grid mismatch")
        if str(_scalar(saved, "topology_sha256")) != topology_sha256:
            raise ValueError("reuse diagnostic topology mismatch")
        paths = np.asarray(saved["selected_path_sha256"]).astype(str)
        dense_hashes = np.asarray(saved["dense_coordinate_sha256"]).astype(str)
        energies = np.asarray(saved["energy_eV"], dtype=np.float64)
        energy_hashes = np.asarray(saved["energy_sha256"]).astype(str)
        attempts = np.asarray(saved["diagnostic_energy_attempts_per_path"], dtype=int)
        successes = np.asarray(saved["diagnostic_energy_successes_per_path"], dtype=int)
        forces = np.asarray(saved["diagnostic_force_requests_per_path"], dtype=int)
    n_rows = len(paths)
    if energies.shape != (n_rows, N_DENSE_POINTS):
        raise ValueError("reuse diagnostic energy array has the wrong shape")
    if any(len(array) != n_rows for array in (dense_hashes, energy_hashes, attempts, successes, forces)):
        raise ValueError("reuse diagnostic row arrays differ in length")
    if np.any(attempts != N_DENSE_POINTS) or np.any(successes != N_DENSE_POINTS):
        raise ValueError("reuse diagnostic lacks 500 successful energy evaluations")
    if np.any(forces != 0):
        raise ValueError("reuse diagnostic contains a force request")
    result: Dict[str, Dict[str, object]] = {}
    source_sha = sha256_file(diagnostics)
    for index, path_sha in enumerate(paths):
        if sha256_array(energies[index]) != energy_hashes[index]:
            raise ValueError("reuse diagnostic energy-array hash mismatch")
        record = {
            "dense_coordinate_sha256": dense_hashes[index],
            "energy_eV": energies[index],
            "energy_sha256": energy_hashes[index],
            "source_diagnostic": str(diagnostics),
            "source_diagnostic_sha256": source_sha,
            "source_row_index": index,
        }
        previous = result.get(path_sha)
        if previous is not None and (
            previous["dense_coordinate_sha256"] != record["dense_coordinate_sha256"]
            or previous["energy_sha256"] != record["energy_sha256"]
        ):
            raise ValueError("reuse diagnostic has conflicting duplicate path hashes")
        result[path_sha] = record
    return result


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-diagnostics", type=Path)
    parser.add_argument("--reuse-diagnostic-audit", type=Path)
    parser.add_argument("--fresh-shard-index", type=int, default=0)
    parser.add_argument("--fresh-shard-count", type=int, default=1)
    parser.add_argument(
        "--global-lock-path", type=Path, default=Path(DEFAULT_CANONICAL_LOCK)
    )
    parser.add_argument(
        "--peer-global-lock-path", type=Path, default=Path(DEFAULT_DOWNLOADS_LOCK)
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    contract_path = args.contract.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    audit_output = output.with_suffix(".audit.json")
    wal_output = output.with_suffix(".energy.wal.jsonl")
    if output.exists() or audit_output.exists() or wal_output.exists():
        raise FileExistsError("refusing to overwrite an existing diagnostic artifact")

    contract = _validate_contract(contract_path)
    common_path, common_i_perp, common_nf, committed_rows = load_eligible_rows(
        contract_path, contract
    )
    rows: List[Dict[str, object]] = [
        {
            "label": "common_initial_path",
            "alpha": np.nan,
            "r": np.nan,
            "m_max": -1,
            "source_result": "",
            "source_result_sha256": "",
            "source_audit": "",
            "source_audit_sha256": "",
            "history_index": 0,
            "calls_per_image": 1,
            "optimization_N_F": int(common_nf),
            "discrete_I_perp": float(common_i_perp),
            "path": common_path,
            "path_sha256": sha256_array(common_path),
        },
        *committed_rows,
    ]
    topology_record = contract["topology"]
    topology = resolve_path(contract_path, str(topology_record["path"]))
    _require_hash(topology, str(topology_record["sha256"]), "topology")
    if (args.reuse_diagnostics is None) != (args.reuse_diagnostic_audit is None):
        raise ValueError("both reuse diagnostic paths must be provided together")
    reuse_rows: Dict[str, Dict[str, object]] = {}
    reuse_diagnostics: Path | None = None
    reuse_audit: Path | None = None
    if args.reuse_diagnostics is not None:
        reuse_diagnostics = args.reuse_diagnostics.expanduser().resolve(strict=True)
        reuse_audit = args.reuse_diagnostic_audit.expanduser().resolve(strict=True)
        reuse_rows = load_exact_reuse_rows(
            reuse_diagnostics,
            reuse_audit,
            str(topology_record["sha256"]),
        )
    if args.fresh_shard_count < 1 or not (
        0 <= args.fresh_shard_index < args.fresh_shard_count
    ):
        raise ValueError("invalid fresh-path shard index/count")
    fresh_hashes = []
    for row in rows:
        path_sha = str(row["path_sha256"])
        if path_sha not in reuse_rows and path_sha not in fresh_hashes:
            fresh_hashes.append(path_sha)
    assigned_fresh = {
        path_sha
        for index, path_sha in enumerate(fresh_hashes)
        if index % args.fresh_shard_count == args.fresh_shard_index
    }
    rows = [
        row
        for row in rows
        if str(row["path_sha256"]) in assigned_fresh
        or (
            args.fresh_shard_index == 0
            and str(row["path_sha256"]) in reuse_rows
        )
    ]
    if not rows:
        raise ValueError("diagnostic shard has no assigned paths")
    with acquire_host_locks(
        (args.global_lock_path, args.peer_global_lock_path)
    ) as held_locks:
        evaluated = evaluate_paths(topology, rows, wal_output, reuse_rows)

    atomic_save_npz(
        output,
        label=np.asarray([row["label"] for row in evaluated]),
        alpha=np.asarray([row["alpha"] for row in evaluated], dtype=np.float64),
        r=np.asarray([row["r"] for row in evaluated], dtype=np.float64),
        m_max=np.asarray([row["m_max"] for row in evaluated], dtype=np.int64),
        source_result=np.asarray([row["source_result"] for row in evaluated]),
        source_result_sha256=np.asarray(
            [row["source_result_sha256"] for row in evaluated]
        ),
        source_audit=np.asarray([row["source_audit"] for row in evaluated]),
        source_audit_sha256=np.asarray(
            [row["source_audit_sha256"] for row in evaluated]
        ),
        history_index=np.asarray(
            [row["history_index"] for row in evaluated], dtype=np.int64
        ),
        calls_per_image=np.asarray(
            [row["calls_per_image"] for row in evaluated], dtype=np.int64
        ),
        optimization_N_F=np.asarray(
            [row["optimization_N_F"] for row in evaluated], dtype=np.int64
        ),
        discrete_I_perp=np.asarray(
            [row["discrete_I_perp"] for row in evaluated], dtype=np.float64
        ),
        selected_path_sha256=np.asarray(
            [row["path_sha256"] for row in evaluated]
        ),
        dense_coordinate_sha256=np.asarray(
            [row["dense_coordinate_sha256"] for row in evaluated]
        ),
        energy_eV=np.asarray([row["energy_eV"] for row in evaluated]),
        energy_sha256=np.asarray([row["energy_sha256"] for row in evaluated]),
        densified_Delta_E_kcal_per_mol=np.asarray(
            [row["densified_Delta_E_kcal_per_mol"] for row in evaluated],
            dtype=np.float64,
        ),
        total_arclength_A=np.asarray(
            [row["total_arclength_A"] for row in evaluated], dtype=np.float64
        ),
        minimum_original_segment_length_A=np.asarray(
            [row["minimum_original_segment_length_A"] for row in evaluated],
            dtype=np.float64,
        ),
        maximum_original_segment_length_A=np.asarray(
            [row["maximum_original_segment_length_A"] for row in evaluated],
            dtype=np.float64,
        ),
        dense_spacing_A=np.asarray(
            [row["dense_spacing_A"] for row in evaluated], dtype=np.float64
        ),
        diagnostic_energy_attempts_per_path=np.full(
            len(evaluated), N_DENSE_POINTS, dtype=np.int64
        ),
        diagnostic_energy_successes_per_path=np.full(
            len(evaluated), N_DENSE_POINTS, dtype=np.int64
        ),
        diagnostic_force_requests_per_path=np.zeros(len(evaluated), dtype=np.int64),
        diagnostic_energy_reused_per_path=np.asarray(
            [row["diagnostic_energy_reused"] for row in evaluated], dtype=bool
        ),
        diagnostic_energy_attempts_this_evaluation_per_path=np.asarray(
            [row["diagnostic_energy_attempts_this_evaluation"] for row in evaluated],
            dtype=np.int64,
        ),
        diagnostic_energy_source=np.asarray(
            [row["diagnostic_energy_source"] for row in evaluated]
        ),
        optimization_N_F_contribution=np.asarray(0, dtype=np.int64),
        coordinate_space=np.asarray(COORDINATE_SPACE),
        interpolation=np.asarray(INTERPOLATION),
        sampling=np.asarray(SAMPLING),
        number_of_points=np.asarray(N_DENSE_POINTS, dtype=np.int64),
        grid_formula=np.asarray(GRID_FORMULA),
        contract=np.asarray(str(contract_path)),
        contract_sha256=np.asarray(sha256_file(contract_path)),
        metric_policy_amendment_sha256=np.asarray(METRIC_AMENDMENT_SHA256),
        alpha_scope_R2_sha256=np.asarray(ALPHA_SCOPE_R2_SHA256),
        topology=np.asarray(str(topology)),
        topology_sha256=np.asarray(sha256_file(topology)),
        energy_wal=np.asarray(str(wal_output)),
        held_host_locks=np.asarray(held_locks),
        exact_reuse_diagnostics=np.asarray(
            "" if reuse_diagnostics is None else str(reuse_diagnostics)
        ),
        exact_reuse_diagnostics_sha256=np.asarray(
            "" if reuse_diagnostics is None else sha256_file(reuse_diagnostics)
        ),
        exact_reuse_diagnostic_audit=np.asarray(
            "" if reuse_audit is None else str(reuse_audit)
        ),
        exact_reuse_diagnostic_audit_sha256=np.asarray(
            "" if reuse_audit is None else sha256_file(reuse_audit)
        ),
        fresh_shard_index=np.asarray(args.fresh_shard_index, dtype=np.int64),
        fresh_shard_count=np.asarray(args.fresh_shard_count, dtype=np.int64),
        assigned_fresh_path_sha256=np.asarray(sorted(assigned_fresh)),
    )
    audit = {
        "schema": "bb-irivm-discrete-iperp-densified-barrier-audit-v1",
        "status": "PASS",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "energy_wal": str(wal_output),
        "energy_wal_sha256": sha256_file(wal_output),
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "metric_policy_amendment_sha256": METRIC_AMENDMENT_SHA256,
        "alpha_scope_R2_sha256": ALPHA_SCOPE_R2_SHA256,
        "selected_path_count_including_common_initial": len(evaluated),
        "diagnostic_energy_attempts": N_DENSE_POINTS * len(evaluated),
        "diagnostic_energy_successes": N_DENSE_POINTS * len(evaluated),
        "diagnostic_energy_attempts_this_evaluation": sum(
            int(row["diagnostic_energy_attempts_this_evaluation"])
            for row in evaluated
        ),
        "diagnostic_energy_successes_this_evaluation": sum(
            int(row["diagnostic_energy_attempts_this_evaluation"])
            for row in evaluated
        ),
        "exact_reused_path_count": sum(
            bool(row["diagnostic_energy_reused"]) for row in evaluated
        ),
        "freshly_evaluated_path_count": sum(
            not bool(row["diagnostic_energy_reused"]) for row in evaluated
        ),
        "exact_reuse_diagnostics": (
            None if reuse_diagnostics is None else str(reuse_diagnostics)
        ),
        "exact_reuse_diagnostics_sha256": (
            None if reuse_diagnostics is None else sha256_file(reuse_diagnostics)
        ),
        "exact_reuse_diagnostic_audit": (
            None if reuse_audit is None else str(reuse_audit)
        ),
        "exact_reuse_diagnostic_audit_sha256": (
            None if reuse_audit is None else sha256_file(reuse_audit)
        ),
        "fresh_shard_index": args.fresh_shard_index,
        "fresh_shard_count": args.fresh_shard_count,
        "assigned_fresh_path_sha256": sorted(assigned_fresh),
        "diagnostic_force_requests": 0,
        "optimization_N_F_contribution": 0,
        "held_host_locks": held_locks,
        "every_path_has_exactly_500_fresh_energy_evaluations": not bool(reuse_rows),
        "every_path_is_bound_to_exactly_500_successful_energy_evaluations": True,
        "sparse_barrier_used_for_selection": False,
        "dense_I_perp_used_for_selection": False,
        "records": [
            {
                "label": row["label"],
                "history_index": row["history_index"],
                "optimization_N_F": row["optimization_N_F"],
                "discrete_I_perp": row["discrete_I_perp"],
                "selected_path_sha256": row["path_sha256"],
                "dense_coordinate_sha256": row["dense_coordinate_sha256"],
                "energy_sha256": row["energy_sha256"],
                "densified_Delta_E_kcal_per_mol": row[
                    "densified_Delta_E_kcal_per_mol"
                ],
                "diagnostic_energy_attempts": N_DENSE_POINTS,
                "diagnostic_energy_attempts_this_evaluation": row[
                    "diagnostic_energy_attempts_this_evaluation"
                ],
                "diagnostic_energy_reused": row["diagnostic_energy_reused"],
                "diagnostic_energy_source": row["diagnostic_energy_source"],
                "diagnostic_force_requests": 0,
            }
            for row in evaluated
        ],
    }
    atomic_write_json(audit_output, audit)
    print(f"Saved {output}")
    print(f"Saved {audit_output}")
    print(f"Saved {wal_output}")


if __name__ == "__main__":
    main()
