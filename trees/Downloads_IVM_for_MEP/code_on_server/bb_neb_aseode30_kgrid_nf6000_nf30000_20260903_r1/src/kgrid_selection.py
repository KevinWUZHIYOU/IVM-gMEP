"""Prospectively frozen, zero-Force selection rule for the BB ODE--NEB k grid.

The public selector does not accept precomputed metric columns. It accepts
complete, audited paid-batch evidence for every candidate, reconstructs the
accepted/current state at every exact 30-force budget, and computes both
article metrics from that same original 32-image path.

A rejected ODE trial remains in the evidence and in the force count, but its
trial geometry is never metric-eligible. At that budget the metric state is
the preceding accepted path and its already-recorded energy/force arrays.
No physical query, densification, or interpolation occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Dict, Mapping, Sequence

import numpy as np

from .accounting import array_sha256
from .kgrid_spec import INITIAL_ARRAY_SHA256, K_VALUES, build_run


SCHEMA = "bb-ase-ode-kgrid-selection-v1"
ANALYSIS_SCHEMA = "bb-ase-ode-kgrid-selection-analysis-v1"
SCREENING_TERMINAL_NF = 6000
NF_PER_BATCH = 30
BB_DOF = 9396
BUDGET_GRID = np.arange(
    NF_PER_BATCH,
    SCREENING_TERMINAL_NF + NF_PER_BATCH,
    NF_PER_BATCH,
    dtype=np.int64,
)
METRICS = ("i_perp", "delta_e32_eV")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _strict_json_number(value: float):
    """Keep the receipt strict JSON even for an exact zero reference."""

    value = float(value)
    if math.isfinite(value):
        return value
    if value == math.inf:
        return "positive_infinity"
    raise ValueError("selection result contains an invalid non-finite score")


def _array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{label} shape {result.shape} != {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} contains a non-finite value")
    return np.ascontiguousarray(result)


def _metric_pair(
    path32_A: np.ndarray,
    energy32_eV: np.ndarray,
    full_force32_eV_per_A: np.ndarray,
) -> tuple[float, float]:
    """Compute both frozen metrics on one identical 32-image path."""

    tangent = path32_A[2:] - path32_A[:-2]
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    if np.any(tangent_norm <= 1.0e-30):
        raise ValueError("central-difference tangent has zero norm")
    tangent = tangent / tangent_norm
    force_internal = full_force32_eV_per_A[1:-1]
    force_perp = force_internal - (
        np.sum(force_internal * tangent, axis=1, keepdims=True) * tangent
    )
    segments = np.linalg.norm(path32_A[1:] - path32_A[:-1], axis=1)
    weights = 0.5 * (segments[:-1] + segments[1:])
    # Keep the operation sequence identical to bb_core.path_diagnostics:
    # norm first, then square, then apply the local Voronoi weights.
    per_image_l2 = np.linalg.norm(force_perp, axis=1)
    i_perp = float(np.sum(weights * per_image_l2**2))
    delta_e32 = float(np.max(energy32_eV) - energy32_eV[0])
    if not math.isfinite(i_perp) or not math.isfinite(delta_e32):
        raise ValueError("computed metric is non-finite")
    if i_perp < 0.0 or delta_e32 < 0.0:
        raise ValueError("computed metric is negative")
    return i_perp, delta_e32


@dataclass(frozen=True)
class PaidBatchEvidence:
    """One batch already verified against optimization and full-state WALs."""

    batch_id: int
    optimization_nf: int
    spring_k_eV_per_A2: float
    role: str
    disposition: str
    paid_path32_A: object
    energy32_eV: object
    full_force32_eV_per_A: object
    current_path32_A: object
    paid_force_attempt_indices: object
    endpoint_optimization_nf: int
    source_artifact_sha256: str
    ledger_commit_event_sha256: str
    full_state_wal_event_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PaidBatchEvidence":
        required = set(cls.__dataclass_fields__)
        if set(value) != required:
            raise ValueError(
                "paid-batch evidence fields must be exactly " + repr(sorted(required))
            )
        return cls(**value)


@dataclass(frozen=True)
class CandidateTrace:
    spring_k_eV_per_A2: float
    degrees_of_freedom: int
    optimization_nf: np.ndarray
    i_perp: np.ndarray
    delta_e32_eV: np.ndarray
    dispositions: tuple[str, ...]
    current_path_sha256: tuple[str, ...]
    paid_batch_source_closure_sha256: str
    terminal_audit_sha256: str
    common_initial_path_sha256: str

    def validate(self) -> None:
        if float(self.spring_k_eV_per_A2) not in K_VALUES:
            raise ValueError("candidate trace has an invalid k")
        if int(self.degrees_of_freedom) <= 0 or int(self.degrees_of_freedom) % 3:
            raise ValueError("candidate degrees of freedom are invalid")
        for name in ("optimization_nf", "i_perp", "delta_e32_eV"):
            values = getattr(self, name)
            if not isinstance(values, np.ndarray) or values.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional array")
            if values.shape != BUDGET_GRID.shape:
                raise ValueError(
                    f"{name} must contain all {BUDGET_GRID.size} paid batches"
                )
        if not np.array_equal(self.optimization_nf, BUDGET_GRID):
            raise ValueError("optimization_nf must be exactly 30, 60, ..., 6000")
        for name in METRICS:
            values = getattr(self, name)
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} contains an invalid value")
        if len(self.dispositions) != BUDGET_GRID.size:
            raise ValueError("disposition history is incomplete")
        if self.dispositions[0] != "current_initial" or any(
            item not in {"accepted", "rejected"} for item in self.dispositions[1:]
        ):
            raise ValueError("disposition history is invalid")
        if len(self.current_path_sha256) != BUDGET_GRID.size or any(
            not _SHA256.fullmatch(value) for value in self.current_path_sha256
        ):
            raise ValueError("current-path hash history is invalid")
        for value in (
            self.paid_batch_source_closure_sha256,
            self.terminal_audit_sha256,
            self.common_initial_path_sha256,
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError("candidate source binding is invalid")

    def lower_envelope(self, metric: str) -> np.ndarray:
        if metric not in METRICS:
            raise ValueError(f"unknown selection metric: {metric}")
        return np.minimum.accumulate(getattr(self, metric))


def _candidate_from_paid_batches(
    *,
    spring_k: float,
    terminal_audit_sha256: str,
    batches: Sequence[Mapping[str, object]],
    expected_dof: int,
) -> CandidateTrace:
    k = float(spring_k)
    if k not in K_VALUES:
        raise ValueError(f"unexpected spring coefficient: {k}")
    if not _SHA256.fullmatch(str(terminal_audit_sha256)):
        raise ValueError("terminal audit SHA-256 is missing or malformed")
    if len(batches) != BUDGET_GRID.size:
        raise ValueError("candidate must contain exactly 200 paid batches")

    dof = int(expected_dof)
    if dof <= 0 or dof % 3:
        raise ValueError("expected_dof must be a positive Cartesian dimension")
    endpoints = None
    endpoint_energy = None
    endpoint_force = None
    current_path = None
    current_energy = None
    current_force = None
    metric_i = []
    metric_e = []
    dispositions = []
    current_hashes = []
    source_rows = []

    for expected_batch, raw in enumerate(batches, start=1):
        row = PaidBatchEvidence.from_mapping(raw)
        if int(row.batch_id) != expected_batch:
            raise ValueError("paid batch ids must be exactly 1, ..., 200")
        expected_nf = NF_PER_BATCH * expected_batch
        if int(row.optimization_nf) != expected_nf:
            raise ValueError("paid batch optimization N_F is not 30*batch_id")
        if float(row.spring_k_eV_per_A2) != k:
            raise ValueError("paid batch spring coefficient differs from candidate")
        role = str(row.role)
        disposition = str(row.disposition)
        if expected_batch == 1:
            if role != "current_initial" or disposition != "current_initial":
                raise ValueError("batch 1 must be the unique current_initial")
        elif role != "trial" or disposition not in {"accepted", "rejected"}:
            raise ValueError("later batches must be accepted/rejected ODE trials")
        attempts = tuple(int(value) for value in row.paid_force_attempt_indices)
        if attempts != tuple(range(expected_nf - 29, expected_nf + 1)):
            raise ValueError("batch does not prove exactly 30 contiguous paid attempts")
        if int(row.endpoint_optimization_nf) != 0:
            raise ValueError("fixed endpoints may not contribute optimization N_F")
        for digest in (
            str(row.source_artifact_sha256),
            str(row.ledger_commit_event_sha256),
            str(row.full_state_wal_event_sha256),
        ):
            if not _SHA256.fullmatch(digest):
                raise ValueError("paid batch lacks a valid source/WAL hash binding")

        paid_path = _array(row.paid_path32_A, (32, dof), "paid_path32_A")
        energy = _array(row.energy32_eV, (32,), "energy32_eV")
        force = _array(
            row.full_force32_eV_per_A,
            (32, dof),
            "full_force32_eV_per_A",
        )
        declared_current = _array(
            row.current_path32_A, (32, dof), "current_path32_A"
        )
        if endpoints is None:
            endpoints = np.array(paid_path[[0, -1]], copy=True)
            endpoint_energy = np.array(energy[[0, -1]], copy=True)
            endpoint_force = np.array(force[[0, -1]], copy=True)
        elif not (
            np.array_equal(paid_path[[0, -1]], endpoints)
            and np.array_equal(declared_current[[0, -1]], endpoints)
            and np.array_equal(energy[[0, -1]], endpoint_energy)
            and np.array_equal(force[[0, -1]], endpoint_force)
        ):
            raise ValueError("fixed endpoint coordinates/energy/force changed")

        if disposition in {"current_initial", "accepted"}:
            if not np.array_equal(declared_current, paid_path):
                raise ValueError("accepted/current path is not its paid path")
            current_path = paid_path
            current_energy = energy
            current_force = force
        else:
            if current_path is None or current_energy is None or current_force is None:
                raise ValueError("rejected trial precedes an accepted current state")
            if not np.array_equal(declared_current, current_path):
                raise ValueError("rejected trial did not restore the preceding accepted path")
            if np.array_equal(paid_path, current_path):
                raise ValueError("rejected paid trial duplicates the accepted current path")

        assert current_path is not None
        assert current_energy is not None
        assert current_force is not None
        i_perp, delta_e32 = _metric_pair(current_path, current_energy, current_force)
        metric_i.append(i_perp)
        metric_e.append(delta_e32)
        dispositions.append(disposition)
        current_hashes.append(array_sha256(current_path))
        source_rows.append(
            {
                "batch_id": expected_batch,
                "optimization_N_F": expected_nf,
                "role": role,
                "disposition": disposition,
                "paid_attempt_first": attempts[0],
                "paid_attempt_last": attempts[-1],
                "endpoint_optimization_N_F": 0,
                "source_artifact_sha256": str(row.source_artifact_sha256),
                "ledger_commit_event_sha256": str(row.ledger_commit_event_sha256),
                "full_state_wal_event_sha256": str(row.full_state_wal_event_sha256),
                "paid_path_sha256": array_sha256(paid_path),
                "energy32_sha256": array_sha256(energy),
                "full_force32_sha256": array_sha256(force),
                "current_path_sha256": current_hashes[-1],
            }
        )

    trace = CandidateTrace(
        spring_k_eV_per_A2=k,
        degrees_of_freedom=dof,
        optimization_nf=BUDGET_GRID.copy(),
        i_perp=np.asarray(metric_i, dtype=np.float64),
        delta_e32_eV=np.asarray(metric_e, dtype=np.float64),
        dispositions=tuple(dispositions),
        current_path_sha256=tuple(current_hashes),
        paid_batch_source_closure_sha256=_canonical_json_sha256(source_rows),
        terminal_audit_sha256=str(terminal_audit_sha256),
        common_initial_path_sha256=current_hashes[0],
    )
    trace.validate()
    return trace


def _log_regret(value: float, best: float) -> float:
    if best < 0.0 or value < best:
        raise ValueError("invalid regret operands")
    if best == 0.0:
        return 0.0 if value == 0.0 else math.inf
    return math.log(value / best)


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right)
    )


def _select_canonical_traces(canonical: Mapping[float, CandidateTrace]):
    """The sole frozen Pareto/minimax decision implementation."""

    if set(canonical) != set(K_VALUES):
        raise ValueError(f"selection requires exactly the four k values {K_VALUES}")
    if len({trace.degrees_of_freedom for trace in canonical.values()}) != 1:
        raise ValueError("candidate Cartesian dimensions differ")
    initial_hashes = {trace.common_initial_path_sha256 for trace in canonical.values()}
    if len(initial_hashes) != 1:
        raise ValueError("four candidates do not share the exact same initial path")
    audit_hashes = {trace.terminal_audit_sha256 for trace in canonical.values()}
    if len(audit_hashes) != len(K_VALUES):
        raise ValueError("candidate terminal audits are not four distinct artifacts")

    envelopes = {
        k: {metric: trace.lower_envelope(metric) for metric in METRICS}
        for k, trace in canonical.items()
    }
    best = {
        metric: np.min(
            np.stack([envelopes[k][metric] for k in K_VALUES], axis=0), axis=0
        )
        for metric in METRICS
    }
    candidates = {}
    for k in K_VALUES:
        metric_regrets = {}
        terminal_regrets = {}
        for metric in METRICS:
            regrets = np.asarray(
                [
                    _log_regret(float(value), float(reference))
                    for value, reference in zip(envelopes[k][metric], best[metric])
                ],
                dtype=np.float64,
            )
            metric_regrets[metric] = float(np.mean(regrets))
            terminal_regrets[metric] = float(regrets[-1])
        candidates[k] = {
            "mean_regret_pair": tuple(metric_regrets[m] for m in METRICS),
            "terminal_regret_pair": tuple(terminal_regrets[m] for m in METRICS),
            "mean_log_regret": metric_regrets,
            "terminal_log_regret": terminal_regrets,
        }

    nondominated = []
    for k in K_VALUES:
        pair = candidates[k]["mean_regret_pair"]
        if not any(
            other != k
            and _dominates(candidates[other]["mean_regret_pair"], pair)
            for other in K_VALUES
        ):
            nondominated.append(k)

    def tie_break_key(k: float) -> tuple[float, float, float, float, float]:
        mean_pair = candidates[k]["mean_regret_pair"]
        terminal_pair = candidates[k]["terminal_regret_pair"]
        return (
            max(mean_pair),
            sum(mean_pair) / len(mean_pair),
            max(terminal_pair),
            sum(terminal_pair) / len(terminal_pair),
            k,
        )

    selected = min(nondominated, key=tie_break_key)
    return envelopes, candidates, nondominated, selected, tie_break_key


def select_best_k(
    candidate_evidence: Mapping[float, Mapping[str, object]],
    *,
    expected_dof: int = BB_DOF,
) -> dict:
    """Recompute both metrics from evidence and select one k prospectively."""

    canonical: Dict[float, CandidateTrace] = {}
    for raw_k, evidence in candidate_evidence.items():
        k = float(raw_k)
        if k in canonical:
            raise ValueError(f"duplicate spring coefficient after normalization: {k}")
        if k not in K_VALUES:
            raise ValueError(f"unexpected spring coefficient: {k}")
        if set(evidence) != {"terminal_audit_sha256", "batches"}:
            raise ValueError("candidate evidence fields are not exact")
        canonical[k] = _candidate_from_paid_batches(
            spring_k=k,
            terminal_audit_sha256=str(evidence["terminal_audit_sha256"]),
            batches=evidence["batches"],
            expected_dof=expected_dof,
        )
    envelopes, candidates, nondominated, selected, tie_break_key = \
        _select_canonical_traces(canonical)
    initial_hashes = {trace.common_initial_path_sha256 for trace in canonical.values()}
    serializable = {}
    for k in K_VALUES:
        trace = canonical[k]
        serializable[str(k)] = {
            "terminal_audit_sha256": trace.terminal_audit_sha256,
            "paid_batch_source_closure_sha256": trace.paid_batch_source_closure_sha256,
            "common_initial_path_sha256": trace.common_initial_path_sha256,
            "accepted_trials": trace.dispositions.count("accepted"),
            "rejected_trials": trace.dispositions.count("rejected"),
            "metric_history": {
                "optimization_N_F": trace.optimization_nf.tolist(),
                "i_perp": trace.i_perp.tolist(),
                "delta_e32_eV": trace.delta_e32_eV.tolist(),
            },
            "mean_log_regret": {
                metric: _strict_json_number(candidates[k]["mean_log_regret"][metric])
                for metric in METRICS
            },
            "terminal_log_regret": {
                metric: _strict_json_number(
                    candidates[k]["terminal_log_regret"][metric]
                )
                for metric in METRICS
            },
            "terminal_lower_envelope": {
                metric: float(envelopes[k][metric][-1]) for metric in METRICS
            },
            "pareto_nondominated": k in nondominated,
            "selection_key": [
                _strict_json_number(value) for value in tie_break_key(k)
            ],
        }
    result = {
        "schema": ANALYSIS_SCHEMA,
        "status": "SELECTION_COMPLETE",
        "screening_terminal_optimization_N_F": SCREENING_TERMINAL_NF,
        "optimization_N_F_grid": BUDGET_GRID.tolist(),
        "degrees_of_freedom": int(expected_dof),
        "common_initial_path_sha256": next(iter(initial_hashes)),
        "eligible_state": "current accepted path after each paid batch",
        "rejected_trial_metric_eligible": False,
        "rejected_trial_paid_N_F_retained": True,
        "endpoint_optimization_N_F": 0,
        "metric_source": "recomputed from path32, energy32, full_force32",
        "metric_interpolation": False,
        "densification": False,
        "metrics": list(METRICS),
        "selected_spring_k_eV_per_A2": float(selected),
        "pareto_nondominated_spring_k_eV_per_A2": [
            float(k) for k in nondominated
        ],
        "candidates": serializable,
    }
    _canonical_json_sha256(result)
    return result


def select_verified_compact_metric_traces(
    verified_candidates: Sequence[Mapping[str, object]],
    *,
    selection_protocol: Mapping[str, str],
) -> dict:
    """Select from four compact traces after their provenance was verified.

    This boundary accepts metrics and immutable receipt bindings only.  It
    deliberately has no path argument and cannot read or trust unaudited raw
    arrays.  The caller must first verify each compact trace against its
    origin-host compact-audit receipt.
    """

    if not isinstance(selection_protocol, Mapping) \
            or set(selection_protocol) != {"path", "sha256"} \
            or not isinstance(selection_protocol["path"], str) \
            or not selection_protocol["path"] \
            or not _SHA256.fullmatch(str(selection_protocol["sha256"])):
        raise ValueError("selection protocol binding is malformed")
    canonical: Dict[float, CandidateTrace] = {}
    receipt_records = []
    expected_candidate_keys = {
        "spring_k_eV_per_A2", "optimization_N_F", "i_perp",
        "delta_e32_eV", "dispositions", "current_path_sha256",
        "common_initial_path_sha256", "source_closure_sha256",
        "terminal_audit_sha256", "receipt_record",
    }
    expected_record_keys = {
        "spring_k_eV_per_A2", "candidate", "authorization",
        "terminal_audit", "trace", "compact_audit", "origin_machine_id",
        "terminal_tree_closure_sha256",
    }
    for raw in verified_candidates:
        if not isinstance(raw, Mapping) or set(raw) != expected_candidate_keys:
            raise ValueError("verified compact candidate fields are not exact")
        k = float(raw["spring_k_eV_per_A2"])
        if k not in K_VALUES or k in canonical:
            raise ValueError("verified compact candidate k is invalid or duplicated")
        record = raw["receipt_record"]
        if not isinstance(record, Mapping) or set(record) != expected_record_keys \
                or float(record["spring_k_eV_per_A2"]) != k:
            raise ValueError("verified compact receipt record is malformed")
        candidate = record["candidate"]
        if not isinstance(candidate, Mapping):
            raise ValueError("verified compact candidate binding is absent")
        built = build_run(
            spring_k=float(candidate.get("spring_k_eV_per_A2", -1)),
            phase=str(candidate.get("phase", "")),
            run_label=str(candidate.get("run_label", "")),
        )
        if candidate != built.as_dict() or built.phase != "screening" \
                or built.spring_k != k or built.terminal_nf != SCREENING_TERMINAL_NF:
            raise ValueError("verified compact candidate is not an exact screening run")
        for name in ("authorization", "terminal_audit", "trace", "compact_audit"):
            binding = record[name]
            if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"} \
                    or not isinstance(binding["path"], str) or not binding["path"] \
                    or not _SHA256.fullmatch(str(binding["sha256"])):
                raise ValueError(f"verified compact {name} binding is malformed")
        if not isinstance(record["origin_machine_id"], str) \
                or not record["origin_machine_id"] \
                or not _SHA256.fullmatch(str(record["terminal_tree_closure_sha256"])):
            raise ValueError("verified compact origin/tree binding is malformed")
        if str(raw["common_initial_path_sha256"]) != INITIAL_ARRAY_SHA256:
            raise ValueError("verified compact trace is not the frozen common initial path")
        trace = CandidateTrace(
            spring_k_eV_per_A2=k,
            degrees_of_freedom=BB_DOF,
            optimization_nf=np.asarray(raw["optimization_N_F"], dtype=np.int64),
            i_perp=np.asarray(raw["i_perp"], dtype=np.float64),
            delta_e32_eV=np.asarray(raw["delta_e32_eV"], dtype=np.float64),
            dispositions=tuple(str(value) for value in raw["dispositions"]),
            current_path_sha256=tuple(
                str(value) for value in raw["current_path_sha256"]
            ),
            paid_batch_source_closure_sha256=str(raw["source_closure_sha256"]),
            terminal_audit_sha256=str(raw["terminal_audit_sha256"]),
            common_initial_path_sha256=str(raw["common_initial_path_sha256"]),
        )
        trace.validate()
        if trace.terminal_audit_sha256 != record["terminal_audit"]["sha256"]:
            raise ValueError("verified compact terminal audit hashes disagree")
        canonical[k] = trace
        receipt_records.append(dict(record))
    _, _, nondominated, selected, _ = _select_canonical_traces(canonical)
    return {
        "schema": SCHEMA,
        "status": "SELECTION_COMPLETE",
        "screening_terminal_optimization_N_F": SCREENING_TERMINAL_NF,
        "selected_spring_k_eV_per_A2": float(selected),
        "selection_protocol": dict(selection_protocol),
        "screening_traces": sorted(
            receipt_records, key=lambda item: float(item["spring_k_eV_per_A2"])
        ),
        "optimization_N_F_grid": BUDGET_GRID.tolist(),
        "pareto_nondominated_spring_k_eV_per_A2": [
            float(k) for k in nondominated
        ],
    }
