#!/usr/bin/env python3
"""Audit and summarize Chignolin IR-IVM optimization results.

The paper cost convention is the sum of real joint energy/force cache misses
over all 22 images.  The optimization budget excludes the two fixed endpoint
initializations, so at every recorded state

    total Number of Forces = internal Number of Forces + 2.

Only the initial path and transactionally committed ``state_curve``
reparameterizations are scientific path states.  Charged records after the
last such state remain in the accounting ledger, but are never eligible for a
first-passage result or for the reported final complete state.

The program accepts one or more result/companion-audit pairs and writes one
row per parameter setting to both CSV and JSON.  Dense-path barrier evaluation
is intentionally outside this tool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


IPERP_THRESHOLD = 0.21242215133173672
N_IMAGES = 22
N_ENDPOINTS = 2
COMPLETE_STATE_DEFINITION = "initial_or_reparam_committed"
USABLE_STATE_RULE = "record_index_0_or_reparam_committed"
EXPECTED_REPARAM_MODE = "state_curve"
EXPECTED_SELECTOR_MODE = "one_way_takeover"
EXPECTED_ETA = 1.1

_LABEL = re.compile(
    r"^chig_irivm_a1e(?P<alpha_exp>m?[0-9]+)_r(?P<ratio>[0-9]{2})_"
    r"m(?P<cap>[0-9]{2})_inf(?P<budget>[0-9]+)"
    r"(?:_r1|_cpu1_[a-z0-9][a-z0-9_-]{0,31})$"
)


class AuditError(RuntimeError):
    """Raised when a result fails a scientific or accounting invariant."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _require_keys(container: Iterable[str], required: Iterable[str], where: str) -> None:
    present = set(container)
    missing = sorted(set(required).difference(present))
    if missing:
        raise AuditError("{} is missing: {}".format(where, ", ".join(missing)))


def _scalar(saved: np.lib.npyio.NpzFile, key: str) -> Any:
    value = np.asarray(saved[key])
    if value.shape != ():
        raise AuditError("NPZ field {!r} is not scalar".format(key))
    return value.item()


def _exact_scalar(saved: np.lib.npyio.NpzFile, key: str, expected: Any) -> None:
    observed = _scalar(saved, key)
    if isinstance(expected, float):
        try:
            matches = math.isfinite(float(observed)) and float(observed) == expected
        except (TypeError, ValueError):
            matches = False
    else:
        matches = observed == expected
    if not matches:
        raise AuditError(
            "NPZ metadata drift at {}: observed {!r}, expected {!r}".format(
                key, observed, expected
            )
        )


def _json_int(payload: Mapping[str, Any], key: str) -> int:
    if key not in payload or isinstance(payload[key], bool):
        raise AuditError("companion audit lacks integer field {!r}".format(key))
    try:
        result = int(payload[key])
    except (TypeError, ValueError):
        raise AuditError("companion audit field {!r} is not an integer".format(key))
    if result != payload[key]:
        raise AuditError("companion audit field {!r} is not exact".format(key))
    return result


def _label_controls(label: str) -> Tuple[float, float, int, int]:
    match = _LABEL.fullmatch(label)
    if match is None:
        raise AuditError("run label does not encode the registered controls: " + label)
    exponent_text = match.group("alpha_exp")
    exponent = (
        -int(exponent_text[1:])
        if exponent_text.startswith("m")
        else int(exponent_text)
    )
    return (
        float(10**exponent),
        float(int(match.group("ratio"))) / 10.0,
        int(match.group("cap")),
        int(match.group("budget")),
    )


def _companion_path(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.stem + ".audit.json")


def _recovery_saved_record_evidence(
    cost: np.ndarray,
    attempted: np.ndarray,
    committed: np.ndarray,
    direct_nf: np.ndarray,
    recovery_nf: np.ndarray,
    attempt_count: int,
    scalar_recovery_nf: int,
) -> Dict[str, Any]:
    """Certify any recovery work missing from saved trajectory annotations.

    The production writer records the exact recovery total in a scalar.  A
    later history compaction keeps only the last row for a per-image call
    label, so it can replace a completed rollback closure row with an ordinary
    row having that same label.  Intermediate recovery rows remain, but their
    per-record recovery field is deliberately zero.

    Equality with the sum of saved recovery fields remains the preferred
    case.  A positive difference is accepted only when the saved records show
    complete attempted-to-ordinary transitions for the missing closure(s), the
    scalar attempt count closes exactly, and the counter-derived cost intervals
    can contain the whole difference.  A terminal open attempt is never used
    as evidence because it may be the quarantined partial tail.
    """

    assigned = int(np.sum(recovery_nf))
    if scalar_recovery_nf < assigned:
        raise AuditError(
            "recovery NF scalar is smaller than the sum of trajectory records"
        )
    unassigned = int(scalar_recovery_nf - assigned)
    explicit_closure = committed | (recovery_nf > 0)
    explicit_closure_count = int(np.count_nonzero(explicit_closure))

    if unassigned == 0:
        return {
            "assigned": assigned,
            "unassigned": 0,
            "explicit_closure_count": explicit_closure_count,
            "missing_closure_count": 0,
            "counter_lower_bound": 0,
            "counter_upper_bound": 0,
            "evidence": "exact_saved_record_sum",
            "missing_closure_intervals": [],
        }

    if np.any(explicit_closure & (~attempted | (direct_nf <= 0))):
        raise AuditError(
            "positive recovery classification gap has an invalid saved closure record"
        )

    missing_intervals: List[Dict[str, int]] = []
    open_start: Optional[int] = None
    last_trace: Optional[int] = None

    for index in range(int(cost.size)):
        if bool(explicit_closure[index]):
            open_start = None
            last_trace = None
            continue

        if bool(attempted[index]):
            if int(direct_nf[index]) > 0:
                if open_start is not None:
                    raise AuditError(
                        "positive recovery classification gap contains nested open attempts"
                    )
                if index == 0 or int(cost[index] - cost[index - 1]) != int(
                    direct_nf[index]
                ):
                    raise AuditError(
                        "positive recovery classification gap lacks a counter-exact direct start"
                    )
                open_start = index
                last_trace = index
            else:
                if open_start is None:
                    raise AuditError(
                        "positive recovery classification gap contains an orphan recovery trace"
                    )
                last_trace = index
            continue

        if open_start is not None:
            assert last_trace is not None
            lower_bound = int(cost[last_trace] - cost[open_start])
            upper_bound = int(cost[index] - cost[open_start])
            if lower_bound < 0 or upper_bound < lower_bound:
                raise AuditError(
                    "positive recovery classification gap has an invalid counter interval"
                )
            missing_intervals.append(
                {
                    "start_index": int(open_start),
                    "last_trace_index": int(last_trace),
                    "first_post_attempt_index": int(index),
                    "counter_lower_bound": lower_bound,
                    "counter_upper_bound": upper_bound,
                }
            )
            open_start = None
            last_trace = None

    if open_start is not None:
        raise AuditError(
            "positive recovery classification gap reaches a terminal open attempt"
        )

    missing_closure_count = len(missing_intervals)
    if missing_closure_count == 0:
        raise AuditError(
            "recovery NF scalar exceeds trajectory records without a saved closure gap"
        )
    if attempt_count != explicit_closure_count + missing_closure_count:
        raise AuditError(
            "recovery closure records and certified gaps do not match the attempt count"
        )

    lower_total = int(
        sum(interval["counter_lower_bound"] for interval in missing_intervals)
    )
    upper_total = int(
        sum(interval["counter_upper_bound"] for interval in missing_intervals)
    )
    if not lower_total <= unassigned <= upper_total:
        raise AuditError(
            "recovery classification gap is not supported by saved counter intervals"
        )

    return {
        "assigned": assigned,
        "unassigned": unassigned,
        "explicit_closure_count": explicit_closure_count,
        "missing_closure_count": missing_closure_count,
        "counter_lower_bound": lower_total,
        "counter_upper_bound": upper_total,
        "evidence": "complete_attempt_closure_gap_with_counter_bounds",
        "missing_closure_intervals": missing_intervals,
    }


def audit_result(
    npz_path: Path,
    audit_json_path: Optional[Path] = None,
    threshold: float = IPERP_THRESHOLD,
) -> Dict[str, Any]:
    """Validate one result pair and return its deterministic summary row."""

    npz_path = Path(npz_path).expanduser().resolve()
    audit_json_path = (
        _companion_path(npz_path)
        if audit_json_path is None
        else Path(audit_json_path).expanduser().resolve()
    )
    if not npz_path.is_file():
        raise AuditError("result NPZ does not exist: {}".format(npz_path))
    if not audit_json_path.is_file():
        raise AuditError("companion audit JSON does not exist: {}".format(audit_json_path))
    if not math.isfinite(float(threshold)) or float(threshold) < 0.0:
        raise AuditError("I_perp threshold must be finite and nonnegative")

    try:
        companion = json.loads(audit_json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuditError("cannot read companion audit JSON: {}".format(exc))
    if not isinstance(companion, dict):
        raise AuditError("companion audit JSON must contain an object")

    required_npz = {
        "active_alpha",
        "active_reparam_every",
        "alpha",
        "alpha_mode",
        "budget",
        "calls_per_image",
        "configuration_sha256",
        "counter_vectors",
        "discrete_all_returned_states_evaluated",
        "discrete_ffrbk_selector_alpha",
        "discrete_ffrbk_selector_final_implementation_id",
        "discrete_ffrbk_selector_history",
        "discrete_ffrbk_selector_inner_max_steps",
        "discrete_ffrbk_selector_mode",
        "discrete_ffrbk_selector_post_select_eta_composed",
        "discrete_ffrbk_selector_reparameterization_inside_subproblem",
        "discrete_ffrbk_selector_residual_ratio",
        "discrete_ffrbk_selector_eta",
        "discrete_history",
        "discrete_implementation_id",
        "discrete_inner_max_steps",
        "discrete_residual_ratio",
        "discrete_step_multiplier_eta",
        "discrete_step_multiplier_implementation_id",
        "frozen_subproblem",
        "i_perp",
        "initial_path",
        "max_step",
        "reparam_attempted",
        "reparam_committed",
        "reparam_direct_joint_evaluations",
        "reparam_every",
        "reparam_mode",
        "reparam_recovery_joint_evaluations",
        "run_label",
        "state_curve_direct_objective_acceptance",
        "state_curve_direct_solve",
        "state_reparam_accepts",
        "state_reparam_attempts",
        "state_reparam_recovery_joint_evaluations",
        "stored_path_calls",
        "stored_paths",
        "total_internal_joint_evaluations",
        "total_internal_nf_budget",
        "total_internal_nf_budget_blocked_miss_count",
        "total_internal_nf_budget_exhausted",
        "total_internal_nf_budget_no_overshoot",
        "total_internal_nf_complete_state_definition",
        "total_internal_nf_consumed",
        "total_internal_nf_endpoint_evaluations_excluded",
        "total_internal_nf_last_complete_state_index",
        "total_internal_nf_last_complete_state_nf",
        "total_internal_nf_remaining",
        "total_internal_nf_terminal_partial_tail_present",
        "total_internal_nf_terminal_partial_tail_quarantined",
        "total_internal_nf_terminal_record_is_complete_outer_state",
        "total_internal_nf_terminal_uncommitted_charged_tail_nf",
    }
    required_companion = {
        "backend_label",
        "backend_raw_joint_evaluations",
        "consistent",
        "discrete_history",
        "discrete_implementation_id",
        "discrete_inner_max_steps",
        "discrete_residual_ratio",
        "per_image_budget_blocked_misses",
        "per_image_joint_evaluations",
        "reparam_mode",
        "result_file",
        "state_reparam_accepts",
        "state_reparam_attempts",
        "state_reparam_recovery_joint_evaluations",
        "sum_image_joint_evaluations",
    }
    _require_keys(companion.keys(), required_companion, "companion audit JSON")

    try:
        archive = np.load(npz_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise AuditError("cannot read result NPZ: {}".format(exc))

    with archive as saved:
        _require_keys(saved.files, required_npz, "result NPZ")

        label = str(_scalar(saved, "run_label"))
        label_alpha, label_ratio, label_cap, label_budget = _label_controls(label)
        alpha = float(_scalar(saved, "alpha"))
        ratio = float(_scalar(saved, "discrete_residual_ratio"))
        cap = int(_scalar(saved, "discrete_inner_max_steps"))
        eta = float(_scalar(saved, "discrete_step_multiplier_eta"))
        budget = int(_scalar(saved, "total_internal_nf_budget"))

        for key, expected in (
            ("alpha", label_alpha),
            ("alpha_mode", "fixed"),
            ("budget", budget),
            ("discrete_ffrbk_selector_alpha", alpha),
            ("discrete_ffrbk_selector_eta", eta),
            ("discrete_ffrbk_selector_history", 7),
            ("discrete_ffrbk_selector_inner_max_steps", cap),
            ("discrete_ffrbk_selector_mode", EXPECTED_SELECTOR_MODE),
            ("discrete_ffrbk_selector_post_select_eta_composed", True),
            ("discrete_ffrbk_selector_reparameterization_inside_subproblem", False),
            ("discrete_ffrbk_selector_residual_ratio", ratio),
            ("discrete_history", 7),
            ("discrete_inner_max_steps", label_cap),
            ("discrete_residual_ratio", label_ratio),
            ("discrete_step_multiplier_eta", EXPECTED_ETA),
            ("frozen_subproblem", "discrete_b040"),
            ("max_step", 1.0),
            ("reparam_every", 1),
            ("reparam_mode", EXPECTED_REPARAM_MODE),
            ("state_curve_direct_objective_acceptance", "per_image"),
            ("state_curve_direct_solve", True),
            ("total_internal_nf_budget", label_budget),
            ("total_internal_nf_budget_no_overshoot", True),
            ("total_internal_nf_complete_state_definition", COMPLETE_STATE_DEFINITION),
            ("total_internal_nf_endpoint_evaluations_excluded", True),
        ):
            _exact_scalar(saved, key, expected)

        implementation_id = str(_scalar(saved, "discrete_implementation_id"))
        for key in (
            "discrete_ffrbk_selector_final_implementation_id",
            "discrete_step_multiplier_implementation_id",
        ):
            if str(_scalar(saved, key)) != implementation_id:
                raise AuditError("selector/eta implementation metadata disagree at " + key)

        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=np.int64)
        counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
        calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        attempted_records = np.asarray(saved["reparam_attempted"], dtype=bool)
        direct_nf = np.asarray(saved["reparam_direct_joint_evaluations"], dtype=np.int64)
        recovery_nf = np.asarray(
            saved["reparam_recovery_joint_evaluations"], dtype=np.int64
        )
        active_alpha = np.asarray(saved["active_alpha"], dtype=float)
        active_reparam_every = np.asarray(saved["active_reparam_every"], dtype=np.int64)
        stored_paths = np.asarray(saved["stored_paths"])
        initial_path = np.asarray(saved["initial_path"])

        n_records = cost.size
        aligned = (
            calls,
            stored_calls,
            i_perp,
            committed,
            attempted_records,
            direct_nf,
            recovery_nf,
            active_alpha,
            active_reparam_every,
        )
        if cost.ndim != 1 or n_records < 1:
            raise AuditError("trajectory cost must be a nonempty vector")
        if any(array.shape != (n_records,) for array in aligned):
            raise AuditError("trajectory arrays are not aligned")
        if counters.shape != (n_records, N_IMAGES):
            raise AuditError("counter_vectors must have shape (records, 22)")
        if stored_paths.ndim != 3 or stored_paths.shape[:2] != (n_records, N_IMAGES):
            raise AuditError("stored_paths must contain one 22-image path per record")
        if initial_path.shape != stored_paths.shape[1:]:
            raise AuditError("initial_path shape differs from stored path shape")
        if not np.array_equal(initial_path, stored_paths[0]):
            raise AuditError("record zero is not the registered initial path")
        if not np.array_equal(stored_calls, calls):
            raise AuditError("stored path labels differ from trajectory call labels")
        if np.any(counters < 0) or np.any(direct_nf < 0) or np.any(recovery_nf < 0):
            raise AuditError("negative evaluation count found")
        if not np.all(counters[:, 0] == 1) or not np.all(counters[:, -1] == 1):
            raise AuditError("fixed endpoint count must remain exactly one per endpoint")
        if not np.array_equal(cost, np.sum(counters[:, 1:-1], axis=1)):
            raise AuditError("internal NF differs from the 20 internal image counters")
        total_nf = np.sum(counters, axis=1)
        if not np.array_equal(total_nf, cost + N_ENDPOINTS):
            raise AuditError("total NF is not internal NF plus two endpoints")
        if np.any(np.diff(cost) <= 0):
            raise AuditError("internal NF trajectory is not strictly increasing")
        if not np.all(np.isfinite(i_perp)) or np.any(i_perp < 0.0):
            raise AuditError("I_perp trajectory is not finite and nonnegative")
        if not np.all(active_alpha == alpha):
            raise AuditError("alpha was not fixed throughout the trajectory")
        if not np.all(active_reparam_every == 1):
            raise AuditError("state_curve reparameterization was not active every iteration")
        if not bool(_scalar(saved, "discrete_all_returned_states_evaluated")):
            raise AuditError("result includes an unevaluated returned state")

        consumed = int(_scalar(saved, "total_internal_nf_consumed"))
        remaining = int(_scalar(saved, "total_internal_nf_remaining"))
        exhausted = bool(_scalar(saved, "total_internal_nf_budget_exhausted"))
        if consumed != int(cost[-1]):
            raise AuditError("terminal internal NF differs from the budget ledger")
        if consumed > budget or int(np.max(cost)) > budget:
            raise AuditError("internal NF budget was overshot")
        if remaining != budget - consumed or remaining < 0:
            raise AuditError("remaining internal NF budget is inconsistent")
        if exhausted != (consumed == budget):
            raise AuditError("budget exhausted flag is inconsistent")

        complete = committed.copy()
        complete[0] = True
        complete_indices = np.flatnonzero(complete)
        last_complete_index = int(complete_indices[-1])
        last_complete_internal_nf = int(cost[last_complete_index])
        terminal_complete = bool(complete[-1])
        terminal_tail_nf = consumed - last_complete_internal_nf
        partial_tail = bool((not terminal_complete) and terminal_tail_nf > 0)
        tail_checks = {
            "total_internal_nf_last_complete_state_index": last_complete_index,
            "total_internal_nf_last_complete_state_nf": last_complete_internal_nf,
            "total_internal_nf_terminal_record_is_complete_outer_state": terminal_complete,
            "total_internal_nf_terminal_uncommitted_charged_tail_nf": terminal_tail_nf,
            "total_internal_nf_terminal_partial_tail_present": partial_tail,
            "total_internal_nf_terminal_partial_tail_quarantined": partial_tail,
        }
        for key, expected in tail_checks.items():
            _exact_scalar(saved, key, expected)
        if partial_tail and np.any(complete[last_complete_index + 1 :]):
            raise AuditError("a quarantined tail contains a complete state")

        state_reparam_attempts = int(_scalar(saved, "state_reparam_attempts"))
        state_reparam_accepts = int(_scalar(saved, "state_reparam_accepts"))
        state_recovery_nf = int(
            _scalar(saved, "state_reparam_recovery_joint_evaluations")
        )
        if state_reparam_attempts < state_reparam_accepts:
            raise AuditError("reparameterization accepts exceed attempts")
        if state_reparam_accepts != int(np.count_nonzero(committed)):
            raise AuditError("reparameterization accept count differs from committed states")
        recovery_evidence = _recovery_saved_record_evidence(
            cost=cost,
            attempted=attempted_records,
            committed=committed,
            direct_nf=direct_nf,
            recovery_nf=recovery_nf,
            attempt_count=state_reparam_attempts,
            scalar_recovery_nf=state_recovery_nf,
        )

        hits = np.flatnonzero(complete & (i_perp <= float(threshold)))
        first_index: Optional[int] = int(hits[0]) if hits.size else None
        first_internal_nf: Optional[int] = (
            int(cost[first_index]) if first_index is not None else None
        )
        first_total_nf: Optional[int] = (
            int(total_nf[first_index]) if first_index is not None else None
        )
        first_iperp: Optional[float] = (
            float(i_perp[first_index]) if first_index is not None else None
        )

        last_path = stored_paths[last_complete_index]
        configuration_sha256 = str(_scalar(saved, "configuration_sha256"))
        if len(configuration_sha256) != 64:
            raise AuditError("configuration_sha256 is malformed")

        blocked_npz = int(_scalar(saved, "total_internal_nf_budget_blocked_miss_count"))

    if str(companion["backend_label"]) != "ivm_optimization":
        raise AuditError("companion audit uses an unexpected backend label")
    if companion["consistent"] is not True:
        raise AuditError("companion backend/evaluator audit is not consistent")
    companion_result = Path(str(companion["result_file"])).expanduser().resolve()
    if companion_result != npz_path:
        raise AuditError("companion audit points to a different NPZ")
    per_image = np.asarray(companion["per_image_joint_evaluations"], dtype=np.int64)
    blocked_per_image = np.asarray(
        companion["per_image_budget_blocked_misses"], dtype=np.int64
    )
    if per_image.shape != (N_IMAGES,) or blocked_per_image.shape != (N_IMAGES,):
        raise AuditError("companion audit must contain 22 image counters")
    if np.any(per_image < 0) or np.any(blocked_per_image < 0):
        raise AuditError("companion audit contains a negative counter")
    backend_total = _json_int(companion, "backend_raw_joint_evaluations")
    image_total = _json_int(companion, "sum_image_joint_evaluations")
    if backend_total != int(np.sum(per_image)) or image_total != int(np.sum(per_image)):
        raise AuditError("backend count differs from the sum of 22 image counters")
    if not np.array_equal(per_image, counters[-1]):
        raise AuditError("companion image counters differ from the terminal NPZ counters")
    if backend_total != consumed + N_ENDPOINTS:
        raise AuditError("paper total NF is not internal NF plus two endpoint calls")
    if per_image[0] != 1 or per_image[-1] != 1:
        raise AuditError("companion endpoint convention drifted")
    if int(np.sum(blocked_per_image[1:-1])) != blocked_npz:
        raise AuditError("blocked-miss companion count differs from NPZ metadata")
    if blocked_per_image[0] != 0 or blocked_per_image[-1] != 0:
        raise AuditError("a fixed endpoint was incorrectly budget-blocked")

    companion_scalar_checks = {
        "reparam_mode": EXPECTED_REPARAM_MODE,
        "discrete_implementation_id": implementation_id,
        "discrete_inner_max_steps": cap,
        "discrete_residual_ratio": ratio,
        "discrete_history": 7,
        "state_reparam_attempts": state_reparam_attempts,
        "state_reparam_accepts": state_reparam_accepts,
        "state_reparam_recovery_joint_evaluations": state_recovery_nf,
    }
    for key, expected in companion_scalar_checks.items():
        if companion[key] != expected:
            raise AuditError("companion/NPZ metadata disagree at " + key)

    return {
        "audit_status": "passed",
        "run_label": label,
        "alpha": alpha,
        "residual_ratio_r": ratio,
        "inner_max_steps_m_max": cap,
        "eta": eta,
        "selector_mode": EXPECTED_SELECTOR_MODE,
        "reparam_mode": EXPECTED_REPARAM_MODE,
        "internal_nf_budget": budget,
        "terminal_internal_nf": consumed,
        "terminal_total_nf": backend_total,
        "i_perp_threshold": float(threshold),
        "first_threshold_reached": first_index is not None,
        "first_threshold_state_index": first_index,
        "first_threshold_state_kind": (
            "initial" if first_index == 0 else "reparam_committed"
        )
        if first_index is not None
        else None,
        "first_threshold_internal_nf": first_internal_nf,
        "first_threshold_total_nf": first_total_nf,
        "first_threshold_i_perp": first_iperp,
        "usable_state_rule": USABLE_STATE_RULE,
        "usable_state_count": int(complete_indices.size),
        "usable_state_indices": [int(value) for value in complete_indices],
        "last_complete_state_index": last_complete_index,
        "last_complete_state_kind": (
            "initial" if last_complete_index == 0 else "reparam_committed"
        ),
        "last_complete_internal_nf": last_complete_internal_nf,
        "last_complete_total_nf": last_complete_internal_nf + N_ENDPOINTS,
        "last_complete_i_perp": float(i_perp[last_complete_index]),
        "last_complete_path_npz_key": "stored_paths",
        "last_complete_path_sha256": _sha256_array(last_path),
        "terminal_partial_tail_present": partial_tail,
        "terminal_partial_tail_quarantined": partial_tail,
        "terminal_partial_tail_internal_nf": terminal_tail_nf,
        "reparam_attempt_count": state_reparam_attempts,
        "reparam_accept_count": state_reparam_accepts,
        "reparam_trial_record_count": int(np.count_nonzero(attempted_records)),
        "reparam_direct_proposal_event_count": int(np.count_nonzero(direct_nf)),
        "reparam_direct_joint_evaluations": int(np.sum(direct_nf)),
        "recovery_event_count": int(np.count_nonzero(recovery_nf)),
        "recovery_event_count_is_lower_bound": bool(
            recovery_evidence["unassigned"] > 0
        ),
        "recovery_joint_evaluations": state_recovery_nf,
        "recovery_joint_evaluations_assigned_to_saved_record": int(
            recovery_evidence["assigned"]
        ),
        "recovery_joint_evaluations_unassigned_to_saved_record": int(
            recovery_evidence["unassigned"]
        ),
        "recovery_saved_record_explicit_closure_count": int(
            recovery_evidence["explicit_closure_count"]
        ),
        "recovery_saved_record_missing_closure_count": int(
            recovery_evidence["missing_closure_count"]
        ),
        "recovery_saved_record_gap_counter_lower_bound": int(
            recovery_evidence["counter_lower_bound"]
        ),
        "recovery_saved_record_gap_counter_upper_bound": int(
            recovery_evidence["counter_upper_bound"]
        ),
        "recovery_saved_record_gap_evidence": str(recovery_evidence["evidence"]),
        "recovery_saved_record_missing_closure_intervals": recovery_evidence[
            "missing_closure_intervals"
        ],
        "reparam_accept_count_through_last_complete": int(
            np.count_nonzero(committed[: last_complete_index + 1])
        ),
        "recovery_event_count_through_last_complete": int(
            np.count_nonzero(recovery_nf[: last_complete_index + 1])
        ),
        "recovery_joint_evaluations_through_last_complete": int(
            np.sum(recovery_nf[: last_complete_index + 1])
        ),
        "configuration_sha256": configuration_sha256,
        "result_npz": str(npz_path),
        "result_npz_sha256": _sha256_file(npz_path),
        "companion_audit_json": str(audit_json_path),
        "companion_audit_json_sha256": _sha256_file(audit_json_path),
    }


CSV_FIELDS = (
    "audit_status",
    "run_label",
    "alpha",
    "residual_ratio_r",
    "inner_max_steps_m_max",
    "eta",
    "selector_mode",
    "reparam_mode",
    "internal_nf_budget",
    "terminal_internal_nf",
    "terminal_total_nf",
    "i_perp_threshold",
    "first_threshold_reached",
    "first_threshold_state_index",
    "first_threshold_state_kind",
    "first_threshold_internal_nf",
    "first_threshold_total_nf",
    "first_threshold_i_perp",
    "usable_state_rule",
    "usable_state_count",
    "last_complete_state_index",
    "last_complete_state_kind",
    "last_complete_internal_nf",
    "last_complete_total_nf",
    "last_complete_i_perp",
    "last_complete_path_npz_key",
    "last_complete_path_sha256",
    "terminal_partial_tail_present",
    "terminal_partial_tail_quarantined",
    "terminal_partial_tail_internal_nf",
    "reparam_attempt_count",
    "reparam_accept_count",
    "reparam_trial_record_count",
    "reparam_direct_proposal_event_count",
    "reparam_direct_joint_evaluations",
    "recovery_event_count",
    "recovery_event_count_is_lower_bound",
    "recovery_joint_evaluations",
    "recovery_joint_evaluations_assigned_to_saved_record",
    "recovery_joint_evaluations_unassigned_to_saved_record",
    "recovery_saved_record_explicit_closure_count",
    "recovery_saved_record_missing_closure_count",
    "recovery_saved_record_gap_counter_lower_bound",
    "recovery_saved_record_gap_counter_upper_bound",
    "recovery_saved_record_gap_evidence",
    "reparam_accept_count_through_last_complete",
    "recovery_event_count_through_last_complete",
    "recovery_joint_evaluations_through_last_complete",
    "configuration_sha256",
    "result_npz",
    "result_npz_sha256",
    "companion_audit_json",
    "companion_audit_json_sha256",
)


def _atomic_write(path: Path, text: str) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_outputs(rows: Sequence[Mapping[str, Any]], csv_path: Path, json_path: Path) -> None:
    """Write deterministic CSV/JSON summaries after all inputs pass audit."""

    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["alpha"]),
            float(row["residual_ratio_r"]),
            int(row["inner_max_steps_m_max"]),
            int(row["internal_nf_budget"]),
            str(row["run_label"]),
        ),
    )
    labels = [str(row["run_label"]) for row in ordered]
    if len(labels) != len(set(labels)):
        raise AuditError("duplicate run labels would make the summary ambiguous")

    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in ordered:
        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in CSV_FIELDS})
    _atomic_write(Path(csv_path), buffer.getvalue())

    document = {
        "schema_version": "chignolin-ir-ivm-result-audit-v1",
        "i_perp_threshold": IPERP_THRESHOLD,
        "paper_total_nf_definition": "sum of all 22 image joint E/F cache misses",
        "paper_total_nf_relation": "total_nf = internal_nf + 2 fixed endpoint initializations",
        "usable_state_rule": USABLE_STATE_RULE,
        "dense_barrier_computed": False,
        "result_count": len(ordered),
        "results": ordered,
    }
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write(Path(json_path), serialized)


def _pairs(args: argparse.Namespace) -> List[Tuple[Path, Path]]:
    result: List[Tuple[Path, Path]] = []
    for npz_text in args.npz:
        npz = Path(npz_text).expanduser().resolve()
        result.append((npz, _companion_path(npz)))
    for npz_text, audit_text in args.pair or []:
        result.append(
            (
                Path(npz_text).expanduser().resolve(),
                Path(audit_text).expanduser().resolve(),
            )
        )
    if not result:
        raise AuditError("provide at least one NPZ or one --pair NPZ AUDIT_JSON")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "npz",
        nargs="*",
        help="result NPZ(s); each companion is inferred as <stem>.audit.json",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("NPZ", "AUDIT_JSON"),
        help="explicit result/companion pair; may be repeated",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    rows = [audit_result(npz, audit) for npz, audit in _pairs(args)]
    write_outputs(rows, Path(args.output_csv), Path(args.output_json))
    print(
        "audited {} Chignolin IR-IVM result(s); CSV={} JSON={}".format(
            len(rows),
            Path(args.output_csv).expanduser().resolve(),
            Path(args.output_json).expanduser().resolve(),
        )
    )


if __name__ == "__main__":
    main()
