#!/usr/bin/env python3
"""Run and fail-closed audit one exact-like/no-reparam Ala alpha cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import run_ivm_v9_discrete_ffrbk_takeover_exact_noreparam_shim as runtime
import run_ivm_v9_discrete_ffrbk_takeover_selector_shim as selector
import run_ivm_v9_discrete_exact_candidate_budget_solver as candidate_budget
import run_ivm_v9_total_internal_nf_budget as total_budget
import run_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0 as frozen_current
import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0 as command_helper
import run_stage4_uniform_optimized as legacy


MEMBER_POLICY_VERSION = "stage4_ala_discrete_exact_noreparam_alpha_grid_v1"
FAMILY = "exact_no_reparam"
ALPHAS = tuple(float(10**power) for power in range(-1, 7))
ALPHA_TEXT = {
    float(10**power): f"1e{power}" for power in range(-1, 7)
}
HOST_SHARDS = ("local", "remote")
COMMON_INITIAL_ARRAY_SHA256 = (
    "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
)
TOTAL_INTERNAL_NF_BUDGET = runtime.TOTAL_INTERNAL_NF_BUDGET
PER_IMAGE_SAFETY_BUDGET = runtime.PER_IMAGE_SAFETY_BUDGET
RESIDUAL_RATIO = runtime.RESIDUAL_RATIO
INNER_MAX_STEPS = runtime.INNER_MAX_STEPS
ETA = runtime.ETA
HISTORY = runtime.HISTORY
KKT_ATOL = runtime.KKT_ATOL
PHYSICAL_MAX_STEP = runtime.PHYSICAL_MAX_STEP
REPLICATE = "r1"
EARLY_STOP_REASONS = ("cached_coordinate_stall", "stalled_cached_rejection")

ROUTING_OPTIONS = frozenset(("--output-dir", "--run-label"))
SCIENTIFIC_VALUE_CHANGES = frozenset(
    (
        "--alpha",
        "--budget",
        "--plbfgs-kkt-rtol",
        "--reparam-every",
        "--discrete-inner-max-steps",
        "--discrete-residual-ratio",
    )
)

REQUIRED_RESULT_KEYS = frozenset(
    {
        "active_alpha",
        "active_reparam_every",
        "alpha",
        "alpha_mode",
        "calls_per_image",
        "configuration_sha256",
        "counter_vectors",
        "discrete_all_returned_states_evaluated",
        "discrete_audit_global_relative_residual",
        "discrete_audit_history_count",
        "discrete_audit_outer",
        "discrete_audit_selection",
        "discrete_audit_step",
        "discrete_audit_stop_reason",
        "discrete_ffrbk_selector_aborted_selection_count",
        "discrete_ffrbk_selector_aborted_solver_calls",
        "discrete_ffrbk_selector_apply_h0_inverse_preconditioner_none_calls",
        "discrete_ffrbk_selector_committed_selection_count",
        "discrete_ffrbk_selector_energy_backtracking",
        "discrete_ffrbk_selector_extra_physical_evaluations",
        "discrete_ffrbk_selector_fixed_solver_calls",
        "discrete_ffrbk_selector_inner_max_steps",
        "discrete_ffrbk_selector_latch_reset_count",
        "discrete_ffrbk_selector_mode",
        "discrete_ffrbk_selector_one_way_latch_is_preference",
        "discrete_ffrbk_selector_residual_ratio",
        "discrete_ffrbk_selector_uncommitted_selection_count",
        "discrete_history",
        "discrete_h0_selection_count",
        "discrete_inner_max_steps",
        "discrete_implementation_id",
        "discrete_lbfgs_selection_count",
        "discrete_residual_ratio",
        "discrete_step_multiplier_energy_backtracking",
        "discrete_step_multiplier_eta",
        "discrete_step_multiplier_extra_physical_evaluations",
        "discrete_step_multiplier_aborted_selection_count",
        "discrete_step_multiplier_uncommitted_selection_count",
        "discrete_stop_mode",
        "exact_no_reparam_all_recorded_states_complete",
        "exact_no_reparam_candidate_budget_policy_id",
        "exact_no_reparam_candidate_joint_steps_before",
        "exact_no_reparam_candidate_remaining_internal_nf_before",
        "exact_no_reparam_candidate_required_internal_misses",
        "exact_no_reparam_candidate_stopped_before_query",
        "exact_no_reparam_charged_noncomplete_state_mask",
        "exact_no_reparam_completed_cap_stop_allowed",
        "exact_no_reparam_complete_band_preflight",
        "exact_no_reparam_complete_band_preflight_policy_id",
        "exact_no_reparam_complete_state_definition",
        "exact_no_reparam_complete_state_indices",
        "exact_no_reparam_complete_state_mask",
        "exact_no_reparam_complete_state_nf",
        "exact_no_reparam_enabled",
        "exact_no_reparam_family",
        "exact_no_reparam_implementation_safety_inner_cap",
        "exact_no_reparam_initial_nf",
        "exact_no_reparam_initial_path",
        "exact_no_reparam_initial_path_sha256",
        "exact_no_reparam_inner_cap_required_nonbinding",
        "exact_no_reparam_operational_definition",
        "exact_no_reparam_residual_ratio",
        "exact_no_reparam_runtime_policy_id",
        "exact_no_reparam_scientific_maxstep_unbounded_required",
        "exact_no_reparam_solve_exact_endpoint_aligned_to_saved_state",
        "exact_no_reparam_solve_exact_endpoint_nf",
        "exact_no_reparam_solve_global_relative_residual",
        "exact_no_reparam_solve_joint_steps",
        "exact_no_reparam_solve_published_exact_endpoint_path",
        "exact_no_reparam_solve_published_exact_endpoint_path_sha256",
        "exact_no_reparam_solve_raw_exact_endpoint_path",
        "exact_no_reparam_solve_raw_exact_endpoint_path_sha256",
        "exact_no_reparam_solve_stop_reason",
        "exact_no_reparam_terminal_charged_tail_nf",
        "exact_no_reparam_terminal_record_is_complete",
        "frozen_subproblem",
        "geometric_reparam_joint_evaluations",
        "i_perp",
        "initial_path_sha256",
        "max_step",
        "plbfgs_kkt_atol",
        "plbfgs_kkt_rtol",
        "reparam_attempted",
        "reparam_committed",
        "reparam_direct_joint_evaluations",
        "reparam_every",
        "reparam_every_schedule",
        "reparam_mode",
        "reparam_recovery_joint_evaluations",
        "state_reparam_accepts",
        "state_reparam_attempts",
        "state_reparam_recovery_joint_evaluations",
        "stop_complete_iperp",
        "stop_iperp",
        "stop_reason",
        "stored_path_calls",
        "stored_paths",
        "total_internal_joint_evaluations",
        "total_internal_nf_budget",
        "total_internal_nf_budget_blocked_image_indices",
        "total_internal_nf_budget_blocked_miss_count",
        "total_internal_nf_budget_exhausted",
        "total_internal_nf_budget_exhaustion_observed_at_cache_miss",
        "total_internal_nf_budget_no_overshoot",
        "total_internal_nf_budget_policy_id",
        "total_internal_nf_budget_stop_reason",
        "total_internal_nf_complete_state_definition",
        "total_internal_nf_complete_state_mask_key",
        "total_internal_nf_complete_band_budget_stop_count",
        "total_internal_nf_consumed",
        "total_internal_nf_endpoint_evaluations_excluded",
        "total_internal_nf_full_band_preflight_calls",
        "total_internal_nf_full_band_preflight_crops",
        "total_internal_nf_full_band_preflight_zero_step_stops",
        "total_internal_nf_last_complete_state_index",
        "total_internal_nf_last_complete_state_nf",
        "total_internal_nf_remaining",
        "total_internal_nf_terminal_partial_tail_present",
        "total_internal_nf_terminal_partial_tail_quarantined",
        "total_internal_nf_terminal_record_is_complete_outer_state",
        "total_internal_nf_terminal_uncommitted_charged_tail_nf",
    }
)


def _require_alpha(value: Any) -> float:
    alpha = float(value)
    if not math.isfinite(alpha) or alpha not in ALPHAS:
        raise ValueError("alpha must be one of 1e-1,...,1e6")
    return alpha


def alpha_text(value: Any) -> str:
    return ALPHA_TEXT[_require_alpha(value)]


def alpha_tag(value: Any) -> str:
    return alpha_text(value).replace("-", "m").replace("+", "p")


def _require_execution_host(value: Any) -> str:
    host = str(value)
    if host not in HOST_SHARDS:
        raise ValueError("execution_host must be local or remote")
    return host


def assigned_host(value: Any, execution_host: Any = None) -> str:
    alpha = _require_alpha(value)
    if execution_host is not None:
        return _require_execution_host(execution_host)
    return HOST_SHARDS[ALPHAS.index(alpha) % len(HOST_SHARDS)]


def canonical_cell_spec(
    value: Any, execution_host: Any = None
) -> dict[str, object]:
    alpha = _require_alpha(value)
    # Routing is deliberately validated but excluded from the scientific
    # cell identity.  A manifest override executes the same preregistered
    # cell on another host rather than creating a second scientific cell.
    assigned_host(alpha, execution_host)
    return {
        "algorithm": (
            "discrete_b040+one_way_takeover+ffrbk_operator_lbfgs+eta1.1"
        ),
        "alpha": alpha_text(alpha),
        "family": FAMILY,
        "history": HISTORY,
        "kkt_atol": KKT_ATOL,
        "maxstep": "unbounded_subject_to_total_nf_budget",
        "implementation_safety_inner_cap": INNER_MAX_STEPS,
        "inner_cap_required_nonbinding": True,
        "physical_max_step": PHYSICAL_MAX_STEP,
        "reparameterization": "disabled",
        "residual_ratio": RESIDUAL_RATIO,
        "total_internal_nf_budget": TOTAL_INTERNAL_NF_BUDGET,
    }


def cell_id(value: Any, execution_host: Any = None) -> str:
    payload = json.dumps(
        canonical_cell_spec(value, execution_host),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def run_label(value: Any, execution_host: Any = None) -> str:
    alpha = _require_alpha(value)
    return (
        f"{MEMBER_POLICY_VERSION}_{FAMILY}_"
        f"{assigned_host(alpha, execution_host)}_"
        f"{cell_id(alpha, execution_host)}_a{alpha_tag(alpha)}_r01_unbounded_"
        f"NF{TOTAL_INTERNAL_NF_BUDGET}_{REPLICATE}"
    )


def _replace_option(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"expected exactly one valued {flag}")
    command[positions[0] + 1] = value


def _append_unique_option(command: list[str], flag: str, value: str) -> None:
    if flag in command:
        raise RuntimeError(f"refusing duplicate option {flag}")
    command.extend((flag, value))


def compact_command_sha256(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def single_run_command(
    root: Path, value: Any, execution_host: Any = None
) -> tuple[list[str], str]:
    root = root.expanduser().resolve()
    alpha = _require_alpha(value)
    command = list(frozen_current.current0_command(root))
    command[1] = str(
        (
            root
            / "src/run_ivm_v9_discrete_ffrbk_takeover_exact_noreparam_shim.py"
        ).resolve()
    )
    host = assigned_host(alpha, execution_host)
    label = run_label(alpha, host)
    _replace_option(
        command,
        "--output-dir",
        f"runs/{MEMBER_POLICY_VERSION}/members/{host}",
    )
    _replace_option(command, "--run-label", label)
    _replace_option(command, "--alpha", f"{alpha:g}")
    _replace_option(command, "--budget", str(PER_IMAGE_SAFETY_BUDGET))
    _replace_option(command, "--plbfgs-kkt-rtol", f"{RESIDUAL_RATIO:g}")
    _replace_option(command, "--reparam-every", "0")
    _replace_option(command, "--discrete-inner-max-steps", str(INNER_MAX_STEPS))
    _replace_option(
        command, "--discrete-residual-ratio", f"{RESIDUAL_RATIO:g}"
    )
    _append_unique_option(command, selector.MODE_FLAG, "one_way_takeover")
    _append_unique_option(
        command,
        total_budget.TOTAL_INTERNAL_NF_FLAG,
        str(TOTAL_INTERNAL_NF_BUDGET),
    )
    assert_registered_command(command, root, alpha, host)
    return command, label


def assert_registered_command(
    command: list[str],
    root: Path,
    value: Any,
    execution_host: Any = None,
) -> None:
    root = root.expanduser().resolve()
    alpha = _require_alpha(value)
    host = assigned_host(alpha, execution_host)
    current = list(frozen_current.current0_command(root))
    if len(command) != len(current) + 4:
        raise RuntimeError("exact/no-reparam command length drifted")
    if command[-4:] != [
        selector.MODE_FLAG,
        "one_way_takeover",
        total_budget.TOTAL_INTERNAL_NF_FLAG,
        str(TOTAL_INTERNAL_NF_BUDGET),
    ]:
        raise RuntimeError("exact/no-reparam runtime options drifted")

    allowed = {1}
    for flag in ROUTING_OPTIONS | SCIENTIFIC_VALUE_CHANGES:
        positions = [index for index, token in enumerate(current) if token == flag]
        if len(positions) != 1:
            raise RuntimeError(f"frozen current command has invalid {flag}")
        allowed.add(positions[0] + 1)
    drift = [
        index
        for index, (old, new) in enumerate(zip(current, command[:-4]))
        if old != new and index not in allowed
    ]
    if drift:
        raise RuntimeError(
            "exact/no-reparam changed an unregistered command token: "
            + ",".join(str(index) for index in drift)
        )

    options = command_helper.command_options(command)
    required = {
        "--root": str(root),
        "--output-dir": f"runs/{MEMBER_POLICY_VERSION}/members/{host}",
        "--run-label": run_label(alpha, host),
        "--initial-path-file": "inputs/common_initial_path_equidistant.npz",
        "--solver": "state_ivm",
        "--alpha": f"{alpha:g}",
        "--alpha-mode": "fixed",
        "--max-step": "1",
        "--history": "7",
        "--budget": str(PER_IMAGE_SAFETY_BUDGET),
        "--save-all-paths": "",
        "--state-globalization": "iperp_monotone",
        "--plbfgs-kkt-rtol": "0.1",
        "--reparam-mode": "state_curve",
        "--reparam-every": "0",
        "--state-curve-direct-solve": "",
        "--state-curve-direct-objective-acceptance": "per_image",
        "--stop-iperp": "-1",
        "--stop-complete-iperp": "-1",
        "--frozen-subproblem": "discrete_b040",
        "--discrete-inner-max-steps": "5000",
        "--discrete-stop-mode": "r08c10",
        "--discrete-residual-ratio": "0.1",
        "--discrete-history": "7",
        "--discrete-step-multiplier-eta": "1.1",
        "--plbfgs-kkt-atol": "0",
        selector.MODE_FLAG: "one_way_takeover",
        total_budget.TOTAL_INTERNAL_NF_FLAG: "5000",
    }
    for flag, expected in required.items():
        if options.get(flag) != expected:
            raise RuntimeError(f"exact/no-reparam option drift at {flag}")


def member_paths(
    root: Path, value: Any, execution_host: Any = None
) -> dict[str, Path | str]:
    root = root.expanduser().resolve()
    alpha = _require_alpha(value)
    host = assigned_host(alpha, execution_host)
    label = run_label(alpha, host)
    return {
        "alpha": alpha,
        "alpha_text": alpha_text(alpha),
        "assigned_host": host,
        "cell_id": cell_id(alpha, host),
        "label": label,
        "result_dir": root / "runs" / MEMBER_POLICY_VERSION / "members" / host,
        "audit": root / "runs" / MEMBER_POLICY_VERSION / "audits" / host / f"{label}.json",
        "member_log": root / "logs" / MEMBER_POLICY_VERSION / "members" / host / label,
        "worker_log": root / "logs" / MEMBER_POLICY_VERSION / "workers" / host / label,
    }


def expected_result_glob(value: Any, execution_host: Any = None) -> str:
    return (
        f"ivm_{run_label(value, execution_host)}_central_state_ivm_cfg*_"
        f"budget{PER_IMAGE_SAFETY_BUDGET}.npz"
    )


def _scalar(saved: np.lib.npyio.NpzFile, key: str):
    return np.asarray(saved[key]).item()


def _require_keys(
    saved: np.lib.npyio.NpzFile, keys: Iterable[str], result: Path
) -> None:
    missing = sorted(set(keys).difference(saved.files))
    if missing:
        raise RuntimeError(f"{result} is missing: {', '.join(missing)}")


def _require_scalar(saved: np.lib.npyio.NpzFile, key: str, expected: Any) -> None:
    if _scalar(saved, key) != expected:
        raise RuntimeError(f"result scalar drift at {key}")


def validate_result(
    result: Path, value: Any, execution_host: Any = None
) -> dict[str, object]:
    """Validate a complete synchronized prefix under the hard 5000-NF cap."""

    result = result.expanduser().resolve()
    alpha = _require_alpha(value)
    assigned_host(alpha, execution_host)
    with np.load(result, allow_pickle=False) as saved:
        _require_keys(saved, REQUIRED_RESULT_KEYS, result)
        scalar_expectations = {
            "alpha": alpha,
            "alpha_mode": "fixed",
            "initial_path_sha256": COMMON_INITIAL_ARRAY_SHA256,
            "max_step": PHYSICAL_MAX_STEP,
            "plbfgs_kkt_rtol": RESIDUAL_RATIO,
            "plbfgs_kkt_atol": KKT_ATOL,
            "reparam_mode": "state_curve",
            "reparam_every": 0,
            "reparam_every_schedule": "",
            "state_reparam_attempts": 0,
            "state_reparam_accepts": 0,
            "state_reparam_recovery_joint_evaluations": 0,
            "geometric_reparam_joint_evaluations": 0,
            "stop_iperp": -1.0,
            "stop_complete_iperp": -1.0,
            "frozen_subproblem": "discrete_b040",
            "discrete_inner_max_steps": INNER_MAX_STEPS,
            "discrete_stop_mode": "r08c10",
            "discrete_residual_ratio": RESIDUAL_RATIO,
            "discrete_history": HISTORY,
            "discrete_all_returned_states_evaluated": True,
            "discrete_ffrbk_selector_mode": "one_way_takeover",
            "discrete_ffrbk_selector_residual_ratio": RESIDUAL_RATIO,
            "discrete_ffrbk_selector_inner_max_steps": INNER_MAX_STEPS,
            "discrete_ffrbk_selector_energy_backtracking": False,
            "discrete_ffrbk_selector_extra_physical_evaluations": 0,
            "discrete_ffrbk_selector_one_way_latch_is_preference": True,
            "discrete_ffrbk_selector_apply_h0_inverse_preconditioner_none_calls": 0,
            "discrete_step_multiplier_eta": ETA,
            "discrete_step_multiplier_energy_backtracking": False,
            "discrete_step_multiplier_extra_physical_evaluations": 0,
            "exact_no_reparam_runtime_policy_id": runtime.POLICY_ID,
            "exact_no_reparam_family": FAMILY,
            "exact_no_reparam_operational_definition": (
                runtime.OPERATIONAL_DEFINITION
            ),
            "exact_no_reparam_enabled": True,
            "exact_no_reparam_residual_ratio": RESIDUAL_RATIO,
            "exact_no_reparam_scientific_maxstep_unbounded_required": True,
            "exact_no_reparam_implementation_safety_inner_cap": INNER_MAX_STEPS,
            "exact_no_reparam_inner_cap_required_nonbinding": True,
            "exact_no_reparam_completed_cap_stop_allowed": False,
            "exact_no_reparam_complete_state_definition": (
                "initial_plus_published_outer_states_after_normalized_"
                "residual_converged_frozen_solves"
            ),
            "exact_no_reparam_complete_band_preflight": True,
            "exact_no_reparam_complete_band_preflight_policy_id": (
                runtime.COMPLETE_BAND_PREFLIGHT_POLICY_ID
            ),
            "exact_no_reparam_candidate_budget_policy_id": (
                candidate_budget.POLICY_ID
            ),
            "exact_no_reparam_initial_nf": 20,
            "exact_no_reparam_initial_path_sha256": COMMON_INITIAL_ARRAY_SHA256,
            "total_internal_nf_budget_policy_id": total_budget.POLICY_ID,
            "total_internal_nf_budget": TOTAL_INTERNAL_NF_BUDGET,
            "total_internal_nf_budget_no_overshoot": True,
            "total_internal_nf_endpoint_evaluations_excluded": True,
            "total_internal_nf_budget_blocked_miss_count": 0,
            "total_internal_nf_budget_exhaustion_observed_at_cache_miss": False,
            "total_internal_nf_complete_state_definition": (
                total_budget.COMPLETENESS_KEYED_MASK
            ),
            "total_internal_nf_complete_state_mask_key": (
                runtime.EXACT_COMPLETE_MASK_KEY
            ),
        }
        for key, expected in scalar_expectations.items():
            _require_scalar(saved, key, expected)
        if candidate_budget.POLICY_ID not in str(
            _scalar(saved, "discrete_implementation_id")
        ):
            raise RuntimeError("result implementation ID omits candidate budget policy")

        active_alpha = np.asarray(saved["active_alpha"], dtype=float)
        active_reparam = np.asarray(saved["active_reparam_every"], dtype=int)
        calls = np.asarray(saved["calls_per_image"], dtype=int)
        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        counters = np.asarray(saved["counter_vectors"], dtype=int)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=int)
        stored_paths = np.asarray(saved["stored_paths"], dtype=float)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        record_count = int(cost.size)
        if record_count < 1:
            raise RuntimeError("exact/no-reparam trajectory is empty")
        trajectory_arrays = (active_alpha, active_reparam, calls, cost, i_perp)
        if any(array.shape != (record_count,) for array in trajectory_arrays):
            raise RuntimeError("exact/no-reparam trajectory arrays are misaligned")
        if not np.all(active_alpha == alpha):
            raise RuntimeError("alpha changed within exact/no-reparam trajectory")
        if np.any(active_reparam != 0):
            raise RuntimeError("a no-reparameterization state enabled reparameterization")
        if not np.array_equal(stored_calls, calls):
            raise RuntimeError("save-all path labels differ from trajectory labels")
        if stored_paths.ndim != 3 or stored_paths.shape[0] != record_count:
            raise RuntimeError("save-all paths are not trajectory-aligned")
        if counters.shape != (record_count, 22):
            raise RuntimeError("counter-vector trajectory has the wrong shape")
        if not np.all(counters[:, 0] == 1) or not np.all(counters[:, -1] == 1):
            raise RuntimeError("fixed endpoint counters changed")
        internal_counters = counters[:, 1:-1]
        if not np.all(internal_counters[0] == 1):
            raise RuntimeError("initial path did not evaluate all 20 internal images")
        if np.any(np.diff(internal_counters, axis=0) < 0):
            raise RuntimeError("an internal image counter decreased")
        if not np.array_equal(
            cost, np.sum(internal_counters, axis=1, dtype=np.int64)
        ):
            raise RuntimeError("NoF differs from summed internal counters")
        if not np.array_equal(calls, np.max(internal_counters, axis=1)):
            raise RuntimeError("calls_per_image differs from the maximum counter")
        if cost[0] != 20 or np.any(np.diff(cost) < 0):
            raise RuntimeError("NoF is not a nondecreasing prefix from 20")
        if cost[-1] > TOTAL_INTERNAL_NF_BUDGET:
            raise RuntimeError("trajectory overshot the total internal NoF cap")
        if np.any(np.diff(calls) < 0):
            raise RuntimeError("call labels decreased")
        if not np.all(np.isfinite(i_perp)) or np.any(i_perp < 0.0):
            raise RuntimeError("I_perp trajectory is invalid")

        consumed = int(cost[-1])
        remaining_nf = TOTAL_INTERNAL_NF_BUDGET - consumed
        budget_exhausted = consumed == TOTAL_INTERNAL_NF_BUDGET
        dynamic_scalar_expectations = {
            "total_internal_nf_consumed": consumed,
            "total_internal_nf_remaining": remaining_nf,
            "total_internal_nf_budget_exhausted": budget_exhausted,
        }
        for key, expected in dynamic_scalar_expectations.items():
            _require_scalar(saved, key, expected)
        for key in ("reparam_attempted", "reparam_committed"):
            if np.any(np.asarray(saved[key], dtype=bool)):
                raise RuntimeError(f"no-reparam result has true {key}")
        for key in (
            "reparam_direct_joint_evaluations",
            "reparam_recovery_joint_evaluations",
        ):
            if np.any(np.asarray(saved[key], dtype=int) != 0):
                raise RuntimeError(f"no-reparam result charged {key}")

        outer = np.asarray(saved["discrete_audit_outer"], dtype=int)
        steps = np.asarray(saved["discrete_audit_step"], dtype=int)
        selections = np.asarray(saved["discrete_audit_selection"], dtype="U16")
        histories = np.asarray(saved["discrete_audit_history_count"], dtype=int)
        stop_reasons = np.asarray(saved["discrete_audit_stop_reason"], dtype="U64")
        residuals = np.asarray(
            saved["discrete_audit_global_relative_residual"], dtype=float
        )
        if not (
            outer.shape == steps.shape == selections.shape == histories.shape
        ):
            raise RuntimeError("per-step discrete audit arrays differ in shape")
        if residuals.shape != stop_reasons.shape:
            raise RuntimeError("per-solve residual audit arrays are invalid")
        if np.any(steps < 0) or np.any(steps >= INNER_MAX_STEPS):
            raise RuntimeError(
                "a discrete step index reached the implementation safety cap"
            )
        if np.any(histories < 0) or np.any(histories > HISTORY):
            raise RuntimeError("L-BFGS history audit is invalid")
        if not np.all(np.isin(selections, ("H0", "LBFGS"))):
            raise RuntimeError("unknown selector decision")
        for outer_index in np.unique(outer):
            observed = steps[outer == outer_index]
            if not np.array_equal(observed, np.arange(observed.size)):
                raise RuntimeError("per-solve step indices are not contiguous")
        exact_stops = (
            "evaluated-relative-natural-residual",
            "evaluated-absolute-natural-residual",
        )
        if not np.all(np.isin(stop_reasons, exact_stops)):
            raise RuntimeError(
                "a returned frozen solve did not attain normalized residual 0.1"
            )
        if np.any(stop_reasons == "evaluated-inner-max-steps"):
            raise RuntimeError("the implementation safety cap constrained a solve")
        relative_mask = stop_reasons == "evaluated-relative-natural-residual"
        if np.any(residuals > RESIDUAL_RATIO + 1e-12):
            raise RuntimeError("a residual-converged solve stopped above r=0.1")
        if np.any(~np.isfinite(residuals)) or np.any(residuals < 0.0):
            raise RuntimeError("a completed solve has an invalid residual")

        exact_mask_raw = np.asarray(saved[runtime.EXACT_COMPLETE_MASK_KEY])
        if exact_mask_raw.dtype.kind != "b" or exact_mask_raw.shape != cost.shape:
            raise RuntimeError("exact-complete mask is not boolean/cost-aligned")
        exact_mask = exact_mask_raw.astype(bool, copy=False)
        charged_mask = np.asarray(
            saved["exact_no_reparam_charged_noncomplete_state_mask"], dtype=bool
        )
        complete_indices = np.asarray(
            saved["exact_no_reparam_complete_state_indices"], dtype=int
        )
        complete_nf = np.asarray(
            saved["exact_no_reparam_complete_state_nf"], dtype=int
        )
        if (
            not np.any(exact_mask)
            or not np.array_equal(charged_mask, ~exact_mask)
            or not np.array_equal(complete_indices, np.flatnonzero(exact_mask))
            or not np.array_equal(complete_nf, cost[exact_mask])
        ):
            raise RuntimeError("exact complete/charged state partition drifted")

        initial_exact_path = np.asarray(
            saved["exact_no_reparam_initial_path"], dtype=float
        )
        if (
            initial_exact_path.shape != stored_paths.shape[1:]
            or legacy.array_sha256(initial_exact_path)
            != str(_scalar(saved, "exact_no_reparam_initial_path_sha256"))
        ):
            raise RuntimeError("exact initial-path identity is invalid")

        solve_stop = np.asarray(
            saved["exact_no_reparam_solve_stop_reason"], dtype="U64"
        )
        solve_steps = np.asarray(
            saved["exact_no_reparam_solve_joint_steps"], dtype=int
        )
        solve_residuals = np.asarray(
            saved["exact_no_reparam_solve_global_relative_residual"], dtype=float
        )
        solve_endpoint_nf = np.asarray(
            saved["exact_no_reparam_solve_exact_endpoint_nf"], dtype=int
        )
        solve_aligned = np.asarray(
            saved[
                "exact_no_reparam_solve_exact_endpoint_aligned_to_saved_state"
            ],
            dtype=bool,
        )
        raw_endpoint_paths = np.asarray(
            saved["exact_no_reparam_solve_raw_exact_endpoint_path"], dtype=float
        )
        published_endpoint_paths = np.asarray(
            saved[
                "exact_no_reparam_solve_published_exact_endpoint_path"
            ],
            dtype=float,
        )
        raw_hashes = np.asarray(
            saved["exact_no_reparam_solve_raw_exact_endpoint_path_sha256"],
            dtype="U64",
        )
        published_hashes = np.asarray(
            saved[
                "exact_no_reparam_solve_published_exact_endpoint_path_sha256"
            ],
            dtype="U64",
        )
        solve_shape = stop_reasons.shape
        if not (
            solve_stop.shape
            == solve_steps.shape
            == solve_residuals.shape
            == solve_endpoint_nf.shape
            == solve_aligned.shape
            == raw_hashes.shape
            == published_hashes.shape
            == solve_shape
            and raw_endpoint_paths.shape
            == (solve_stop.size,) + stored_paths.shape[1:]
            and published_endpoint_paths.shape == raw_endpoint_paths.shape
            and np.array_equal(solve_stop, stop_reasons)
            and np.array_equal(solve_residuals, residuals)
        ):
            raise RuntimeError("exact per-solve endpoint audit is misaligned")
        ordered_outers = list(dict.fromkeys(outer.tolist()))
        observed_solve_steps = np.asarray(
            [np.count_nonzero(outer == item) for item in ordered_outers],
            dtype=int,
        )
        if not np.array_equal(solve_steps, observed_solve_steps):
            raise RuntimeError("exact solve step counts differ from runner audit")

        record_hashes = np.asarray(
            [legacy.array_sha256(path) for path in stored_paths], dtype="U64"
        )
        expected_exact_mask = np.zeros(cost.shape, dtype=bool)
        initial_nf = int(_scalar(saved, "exact_no_reparam_initial_nf"))
        initial_hash = legacy.array_sha256(initial_exact_path)
        initial_matches = (cost == initial_nf) & (record_hashes == initial_hash)
        if np.count_nonzero(initial_matches) < 1:
            raise RuntimeError("saved trajectory lost the path-identical initial state")
        expected_exact_mask |= initial_matches
        for index in range(solve_stop.size):
            if solve_endpoint_nf[index] < 0:
                raise RuntimeError("a relative-converged solve lacks endpoint NF")
            if (
                legacy.array_sha256(raw_endpoint_paths[index]) != raw_hashes[index]
                or legacy.array_sha256(published_endpoint_paths[index])
                != published_hashes[index]
            ):
                raise RuntimeError("exact endpoint path hash drifted")
            matches = (cost == solve_endpoint_nf[index]) & (
                record_hashes == published_hashes[index]
            )
            if solve_aligned[index] != bool(np.any(matches)):
                raise RuntimeError("exact endpoint saved-state alignment flag drifted")
            expected_exact_mask |= matches
        if not np.array_equal(exact_mask, expected_exact_mask):
            raise RuntimeError("exact-complete mask lacks path/NF identity proof")

        last_complete_index = int(np.flatnonzero(exact_mask)[-1])
        last_complete_nf = int(cost[last_complete_index])
        terminal_complete = bool(exact_mask[-1])
        charged_tail_nf = consumed - last_complete_nf
        partial_tail = bool(not terminal_complete and charged_tail_nf > 0)
        exact_dynamic = {
            "exact_no_reparam_all_recorded_states_complete": bool(
                np.all(exact_mask)
            ),
            "exact_no_reparam_terminal_record_is_complete": terminal_complete,
            "exact_no_reparam_terminal_charged_tail_nf": charged_tail_nf,
            "total_internal_nf_last_complete_state_index": last_complete_index,
            "total_internal_nf_last_complete_state_nf": last_complete_nf,
            "total_internal_nf_terminal_record_is_complete_outer_state": (
                terminal_complete
            ),
            "total_internal_nf_terminal_uncommitted_charged_tail_nf": (
                charged_tail_nf
            ),
            "total_internal_nf_terminal_partial_tail_present": partial_tail,
            "total_internal_nf_terminal_partial_tail_quarantined": partial_tail,
        }
        for key, expected in exact_dynamic.items():
            _require_scalar(saved, key, expected)

        candidate_required = np.asarray(
            saved["exact_no_reparam_candidate_required_internal_misses"],
            dtype=int,
        )
        candidate_remaining = np.asarray(
            saved["exact_no_reparam_candidate_remaining_internal_nf_before"],
            dtype=int,
        )
        candidate_stopped = np.asarray(
            saved["exact_no_reparam_candidate_stopped_before_query"], dtype=bool
        )
        candidate_joint_before = np.asarray(
            saved["exact_no_reparam_candidate_joint_steps_before"], dtype=int
        )
        if not (
            candidate_required.shape
            == candidate_remaining.shape
            == candidate_stopped.shape
            == candidate_joint_before.shape
            and candidate_required.ndim == 1
            and np.all((candidate_required >= 0) & (candidate_required <= 20))
            and np.all(candidate_remaining >= 0)
            and np.array_equal(
                candidate_stopped, candidate_required > candidate_remaining
            )
        ):
            raise RuntimeError("candidate-aware band-preflight arrays are invalid")
        stopped_count = int(np.count_nonzero(candidate_stopped))
        if stopped_count > 1 or (
            stopped_count == 1 and not bool(candidate_stopped[-1])
        ):
            raise RuntimeError("candidate budget refusal was not terminal")
        budget_limited = stopped_count == 1
        expected_budget_stop = (
            "total_internal_joint_budget_exhausted"
            if budget_limited and consumed == TOTAL_INTERNAL_NF_BUDGET
            else "total_internal_joint_budget_insufficient_for_full_band"
            if budget_limited
            else "runner_stopped_before_total_internal_joint_budget"
        )
        _require_scalar(
            saved, "total_internal_nf_budget_stop_reason", expected_budget_stop
        )
        outer_stop_reason = str(_scalar(saved, "stop_reason"))
        if budget_limited:
            if outer_stop_reason != "evaluation_budget":
                raise RuntimeError("candidate budget refusal did not terminate runner")
        elif outer_stop_reason not in EARLY_STOP_REASONS:
            raise RuntimeError("unregistered early exact/no-reparam stop reason")

        h0_count = int(np.count_nonzero(selections == "H0"))
        lbfgs_count = int(np.count_nonzero(selections == "LBFGS"))
        if h0_count != int(_scalar(saved, "discrete_h0_selection_count")):
            raise RuntimeError("H0 selection count mismatch")
        if lbfgs_count != int(_scalar(saved, "discrete_lbfgs_selection_count")):
            raise RuntimeError("L-BFGS selection count mismatch")
        if int(
            _scalar(saved, "discrete_ffrbk_selector_committed_selection_count")
        ) != selections.size:
            raise RuntimeError("selector committed-event count mismatch")
        fixed_calls = int(_scalar(saved, "discrete_ffrbk_selector_fixed_solver_calls"))
        latch_resets = int(_scalar(saved, "discrete_ffrbk_selector_latch_reset_count"))
        aborted_calls = int(_scalar(saved, "discrete_ffrbk_selector_aborted_solver_calls"))
        aborted_selections = int(
            _scalar(saved, "discrete_ffrbk_selector_aborted_selection_count")
        )
        selector_uncommitted = int(
            _scalar(saved, "discrete_ffrbk_selector_uncommitted_selection_count")
        )
        eta_aborted = int(
            _scalar(saved, "discrete_step_multiplier_aborted_selection_count")
        )
        eta_uncommitted = int(
            _scalar(saved, "discrete_step_multiplier_uncommitted_selection_count")
        )
        if (
            fixed_calls != latch_resets
            or fixed_calls != stop_reasons.size + aborted_calls
            or aborted_calls != stopped_count
            or candidate_required.size != selections.size + aborted_selections
            or selector_uncommitted != 0
            or eta_aborted != aborted_selections
            or eta_uncommitted != 0
        ):
            raise RuntimeError("selector solve accounting is inconsistent")
        blocked = int(_scalar(saved, "total_internal_nf_budget_blocked_miss_count"))
        blocked_images = np.asarray(
            saved["total_internal_nf_budget_blocked_image_indices"], dtype=int
        )
        preflight_calls = int(
            _scalar(saved, "total_internal_nf_full_band_preflight_calls")
        )
        preflight_crops = int(
            _scalar(saved, "total_internal_nf_full_band_preflight_crops")
        )
        preflight_zero_stops = int(
            _scalar(saved, "total_internal_nf_full_band_preflight_zero_step_stops")
        )
        complete_band_budget_stops = int(
            _scalar(saved, "total_internal_nf_complete_band_budget_stop_count")
        )
        expected_zero_stops = int(
            np.count_nonzero(candidate_stopped & (candidate_joint_before == 0))
        )
        if (
            blocked != 0
            or blocked_images.shape != (0,)
            or preflight_calls != candidate_required.size
            or preflight_crops != stopped_count
            or preflight_zero_stops != expected_zero_stops
            or complete_band_budget_stops != stopped_count
        ):
            raise RuntimeError("complete-band total-NF preflight audit is inconsistent")

        summary = {
            "configuration_sha256": str(_scalar(saved, "configuration_sha256")),
            "initial_path_sha256": str(_scalar(saved, "initial_path_sha256")),
            "trajectory_record_count": int(cost.size),
            "final_optimization_NF": int(cost[-1]),
            "initial_state_NF": int(cost[0]),
            "evaluated_candidate_band_count": int(
                np.count_nonzero(~candidate_stopped)
            ),
            "discrete_inner_step_count_including_cache_hits": int(
                np.count_nonzero(~candidate_stopped)
            ),
            "direction_proposal_count": int(candidate_required.size),
            "frozen_subproblem_solve_count": int(stop_reasons.size),
            "relative_residual_stop_count": int(np.count_nonzero(relative_mask)),
            "absolute_residual_stop_count": int(
                np.count_nonzero(
                    stop_reasons == "evaluated-absolute-natural-residual"
                )
            ),
            "budget_interrupted_solve_count": int(
                stopped_count
            ),
            "total_internal_nf_budget_exhausted": budget_exhausted,
            "remaining_total_internal_nf_budget": remaining_nf,
            "inner_cap_stop_count": 0,
            "h0_selection_count": h0_count,
            "lbfgs_selection_count": lbfgs_count,
            "selector_aborted_solver_calls_at_budget": aborted_calls,
            "total_budget_blocked_miss_count": blocked,
            "full_band_preflight_calls": preflight_calls,
            "full_band_preflight_crops": preflight_crops,
            "full_band_preflight_zero_step_stops": preflight_zero_stops,
            "exact_complete_state_count": int(np.count_nonzero(exact_mask)),
            "charged_noncomplete_state_count": int(
                np.count_nonzero(~exact_mask)
            ),
            "last_exact_complete_NF": last_complete_nf,
            "terminal_charged_tail_NF": charged_tail_nf,
            "terminal_i_perp": float(i_perp[-1]),
            "minimum_i_perp": float(np.min(i_perp)),
            "all_recorded_states_exact_complete": bool(np.all(exact_mask)),
            "terminal_record_is_exact_complete": terminal_complete,
            "no_reparameterization_cost": True,
            "nof_decomposition": {
                "initial_internal_path": 20,
                "charged_internal_after_initial": consumed - 20,
                "reparameterization": 0,
                "total_internal_nf": consumed,
            },
        }

    companion = result.with_suffix(".audit.json")
    if not companion.is_file():
        raise RuntimeError(f"missing runner counter audit: {companion}")
    counter_audit = json.loads(companion.read_text(encoding="utf-8"))
    expected_vector = [1] + counters[-1, 1:-1].astype(int).tolist() + [1]
    expected_backend_raw = consumed + 2
    if (
        counter_audit.get("consistent") is not True
        or int(counter_audit["backend_raw_joint_evaluations"])
        != expected_backend_raw
        or int(counter_audit["sum_image_joint_evaluations"])
        != expected_backend_raw
        or list(counter_audit["per_image_joint_evaluations"]) != expected_vector
        or int(counter_audit["state_reparam_attempts"]) != 0
        or int(counter_audit["state_reparam_accepts"]) != 0
        or int(counter_audit["state_reparam_recovery_joint_evaluations"]) != 0
        or int(counter_audit["geometric_reparam_joint_evaluations"]) != 0
        or counter_audit.get("discrete_all_returned_states_evaluated") is not True
    ):
        raise RuntimeError("runner counter audit violates exact/no-reparam protocol")
    summary.update(
        {
            "runner_counter_audit": str(companion),
            "runner_counter_audit_sha256": legacy.file_sha256(companion),
            "backend_raw_joint_evaluations": expected_backend_raw,
            "endpoint_joint_evaluations_excluded_from_NF": 2,
        }
    )
    return summary


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--execution-host", choices=HOST_SHARDS)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    alpha = _require_alpha(args.alpha)
    host = assigned_host(alpha, args.execution_host)
    command, label = single_run_command(root, alpha, host)
    paths = member_paths(root, alpha, host)
    result_dir = paths["result_dir"]
    audit_path = paths["audit"]
    log_dir = paths["member_log"]
    assert isinstance(result_dir, Path)
    assert isinstance(audit_path, Path)
    assert isinstance(log_dir, Path)
    existing = sorted(result_dir.glob(expected_result_glob(alpha, host)))
    if existing or audit_path.exists() or log_dir.exists():
        target = existing[0] if existing else audit_path if audit_path.exists() else log_dir
        raise FileExistsError(f"refusing to overwrite exact/no-reparam cell: {target}")
    result_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    wall_seconds = legacy.run_command(
        command, root, log_dir / "stdout.txt", log_dir / "stderr.txt"
    )
    result = legacy.find_result(
        result_dir, label, PER_IMAGE_SAFETY_BUDGET
    )
    summary = validate_result(result, alpha, host)
    payload: dict[str, object] = {
        "status": "complete",
        "member_policy_version": MEMBER_POLICY_VERSION,
        "family": FAMILY,
        "priority": "secondary_after_inexact_reparam_grid",
        "alpha": alpha,
        "alpha_text": alpha_text(alpha),
        "assigned_host": host,
        "default_assigned_host": assigned_host(alpha),
        "execution_host_override": args.execution_host,
        "cell_id": cell_id(alpha, host),
        "canonical_cell_spec": canonical_cell_spec(alpha, host),
        "residual_ratio": RESIDUAL_RATIO,
        "scientific_maxstep": "unbounded_subject_to_total_NF_budget",
        "implementation_safety_inner_cap": INNER_MAX_STEPS,
        "inner_cap_required_nonbinding": True,
        "physical_max_step": PHYSICAL_MAX_STEP,
        "total_internal_nf_budget": TOTAL_INTERNAL_NF_BUDGET,
        "per_image_runner_budget_is_safety_only": PER_IMAGE_SAFETY_BUDGET,
        "one_way_ffrbk_operator_lbfgs": True,
        "eta": ETA,
        "history": HISTORY,
        "kkt_atol": KKT_ATOL,
        "reparameterization_enabled": False,
        "energy_backtracking_enabled": False,
        "result": str(result),
        "result_sha256": legacy.file_sha256(result),
        "scientific_command_sha256": compact_command_sha256(command),
        "command": command,
        "wall_seconds": wall_seconds,
        **summary,
    }
    _write_json_exclusive(audit_path, payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
