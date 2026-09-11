#!/usr/bin/env python3
"""Run one preregistered Ala scalar-LBFGS mechanism-ablation cell.

The three cells share alpha=15, eta=1.1, r=0.8 OR k=10, m=7,
``kkt_atol=0``, N254, the common initial path, and the existing
``state_curve/per_image`` reparameterization.  ``current_ffrbk_selector`` is
the matched FFrbk-selector rerun.  ``scalar_selector`` and
``direct_lbfgs_only`` use the isolated scalar-LBFGS runtime shim.

This wrapper never indexes a physical energy or barrier array.  The Current0
cell must first reproduce every registered, barrier-blind scientific
trajectory field of the older alpha=15/eta=1.1 run; only routing,
configuration identity, and the now-explicit zero absolute-KKT tolerance are
allowed to differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import run_ivm_v9_discrete_scalar_lbfgs_shim as scalar_runtime
import run_ivm_v9_discrete_selected_eta_shim as eta_runtime
import run_stage4_uniform_optimized as legacy


MEMBER_POLICY_VERSION = (
    "stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1"
)
ARMS = (
    "current_ffrbk_selector",
    "scalar_selector",
    "direct_lbfgs_only",
)
SCALAR_ARMS = frozenset(("scalar_selector", "direct_lbfgs_only"))
CANONICAL_ALPHA = 15.0
CANONICAL_ETA = 1.1
CANONICAL_RATIO = 0.8
CANONICAL_MAX_EVALUATED_STEPS = 10
CANONICAL_HISTORY = 7
CANONICAL_KKT_ATOL = 0.0
CANONICAL_BUDGET = 254
CANONICAL_REPLICATE = "r1"
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
COMMON_INITIAL_ARRAY_SHA256 = (
    "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
)
BASE_DISCRETE_IMPLEMENTATION_ID = (
    "ala-frozen-discrete-ffrbk-dsel-m7-evaluated-v1"
)

OLD_CURRENT_COMMAND = (
    "logs/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1/members/"
    "stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_eta1p1_"
    "r08c10_a15_N254_discrete_b040_r1/command.json"
)
OLD_CURRENT_COMMAND_SHA256 = (
    "a3f622b4be15b64574479d5832bd76fc2a0de8fb3e2032ef4103e776129fb61f"
)
OLD_CURRENT_RESULT = (
    "runs/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1/members/"
    "ivm_stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_eta1p1_"
    "r08c10_a15_N254_discrete_b040_r1_central_state_ivm_"
    "cfg02c3f3990307_budget254.npz"
)
OLD_CURRENT_RESULT_SHA256 = (
    "bb189235868d63be23d335c0766db59e0315b82fd11448077a492966ce5eaa55"
)

ROUTING_OPTIONS = frozenset(("--output-dir", "--run-label"))

COMMON_RESULT_KEYS = frozenset(
    {
        "active_alpha",
        "alpha",
        "alpha_mode",
        "configuration_sha256",
        "i_perp",
        "initial_path_sha256",
        "physical_fmax",
        "plbfgs_kkt_atol",
        "reparam_committed",
        "reparam_mode",
        "segment_length_cv",
        "segment_length_ratio",
        "state_curve_direct_objective_acceptance",
        "stop_complete_iperp",
        "stored_path_calls",
        "stored_paths",
        "subproblem_line_search_backtracks",
        "total_internal_joint_evaluations",
    }
)
DISCRETE_RESULT_KEYS = frozenset(
    {
        "frozen_subproblem",
        "discrete_implementation_id",
        "discrete_inner_max_steps",
        "discrete_stop_mode",
        "discrete_residual_ratio",
        "discrete_history",
        "discrete_all_returned_states_evaluated",
        "discrete_audit_outer",
        "discrete_audit_step",
        "discrete_audit_selection",
        "discrete_audit_h0_raw_predicted_decrease",
        "discrete_audit_lbfgs_raw_predicted_decrease",
        "discrete_audit_h0_capped_predicted_decrease",
        "discrete_audit_lbfgs_capped_predicted_decrease",
        "discrete_audit_h0_cap_scale",
        "discrete_audit_lbfgs_cap_scale",
        "discrete_audit_history_count",
        "discrete_audit_curvature_status",
        "discrete_audit_global_relative_residual",
        "discrete_audit_stop_reason",
        "discrete_h0_selection_count",
        "discrete_lbfgs_selection_count",
    }
)
ETA_RESULT_KEYS = frozenset(
    {
        "discrete_step_multiplier_policy_id",
        "discrete_step_multiplier_base_implementation_id",
        "discrete_step_multiplier_implementation_id",
        "discrete_step_multiplier_eta",
        "discrete_step_multiplier_application_stage",
        "discrete_step_multiplier_selector_isolated",
        "discrete_step_multiplier_dynamic_cap_reapplied",
        "discrete_step_multiplier_energy_backtracking",
        "discrete_step_multiplier_extra_physical_evaluations",
        "discrete_step_multiplier_audit_selection",
        "discrete_step_multiplier_audit_requested",
        "discrete_step_multiplier_audit_applied",
        "discrete_step_multiplier_audit_cap_hit",
        "discrete_step_multiplier_audit_dynamic_max_step",
        "discrete_step_multiplier_audit_base_max_atomic_displacement",
        "discrete_step_multiplier_audit_requested_max_atomic_displacement",
        "discrete_step_multiplier_audit_applied_max_atomic_displacement",
        "discrete_step_multiplier_uncommitted_selection_count",
        "discrete_step_multiplier_aborted_selection_count",
    }
)
SCALAR_STATIC_RESULT_KEYS = frozenset(
    {
        "discrete_scalar_lbfgs_runtime_policy_id",
        "discrete_scalar_lbfgs_mode",
        "discrete_scalar_lbfgs_mode_policy_id",
        "discrete_scalar_lbfgs_scalar_initial_inverse_id",
        "discrete_scalar_lbfgs_post_scale_policy_id",
        "discrete_scalar_lbfgs_base_implementation_id",
        "discrete_scalar_lbfgs_implementation_id",
        "discrete_scalar_lbfgs_alpha",
        "discrete_scalar_lbfgs_eta",
        "discrete_scalar_lbfgs_empty_history_inverse_scale",
        "discrete_scalar_lbfgs_nonempty_history_gamma_floor",
        "discrete_scalar_lbfgs_history",
        "discrete_scalar_lbfgs_stop_mode",
        "discrete_scalar_lbfgs_residual_ratio",
        "discrete_scalar_lbfgs_inner_max_steps",
        "discrete_scalar_lbfgs_kkt_atol",
        "discrete_scalar_lbfgs_absolute_kkt_stop_enabled",
        "discrete_scalar_lbfgs_selector_competition",
        "discrete_scalar_lbfgs_direct_only",
        "discrete_scalar_lbfgs_fixed_solver_preconditioner_is_none",
        "discrete_scalar_lbfgs_ffrbk_applied_inside_fixed_solver",
        "discrete_scalar_lbfgs_ffrbk_solve_normal_calls_inside_fixed_solver",
        "discrete_scalar_lbfgs_ffrbk_solve_normal_calls_outside_fixed_solver",
        "discrete_scalar_lbfgs_runner_bank_argument_intercept_count",
        "discrete_scalar_lbfgs_bank_instance_count",
        "discrete_scalar_lbfgs_bank_build_count",
        "discrete_scalar_lbfgs_bank_init_raw_joint_evaluation_delta",
        "discrete_scalar_lbfgs_bank_build_raw_joint_evaluation_delta",
        "discrete_scalar_lbfgs_bank_build_zero_nf",
        "discrete_scalar_lbfgs_energy_backtracking",
        "discrete_scalar_lbfgs_extra_physical_evaluations",
        "discrete_scalar_lbfgs_reparameterization_inside_subproblem",
        "discrete_scalar_lbfgs_aborted_solver_call_count",
        "discrete_scalar_lbfgs_fixed_solver_call_count",
        "discrete_scalar_lbfgs_uncommitted_selection_count",
    }
)
SCALAR_AUDIT_SUFFIXES = frozenset(
    {
        "selection",
        "history_before",
        "scale_source",
        "initial_inverse_scale",
        "raw_initial_inverse_scale",
        "gamma_floor_applied",
        "last_sy",
        "last_yy",
        "h0_constructed",
        "selector_used",
        "direct_fallback",
        "dynamic_max_step",
        "base_max_atomic_displacement",
        "requested_max_atomic_displacement",
        "applied_max_atomic_displacement",
        "requested_multiplier",
        "applied_multiplier",
        "cap_hit",
        "base_predicted_decrease",
        "applied_predicted_decrease",
    }
)
SCALAR_AUDIT_RESULT_KEYS = frozenset(
    "discrete_scalar_lbfgs_audit_" + suffix
    for suffix in SCALAR_AUDIT_SUFFIXES
)
SCALAR_RESULT_KEYS = SCALAR_STATIC_RESULT_KEYS | SCALAR_AUDIT_RESULT_KEYS

CURRENT0_PARITY_KEYS = frozenset(
    (COMMON_RESULT_KEYS - {"configuration_sha256", "plbfgs_kkt_atol"})
    | DISCRETE_RESULT_KEYS
    | ETA_RESULT_KEYS
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_command_sha256(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _arm(value: str) -> str:
    if value not in ARMS:
        raise ValueError("unknown mechanism-ablation arm")
    return value


def run_label(arm: str) -> str:
    arm = _arm(arm)
    return (
        f"{MEMBER_POLICY_VERSION}_{arm}_eta1p1_r08c10_kkt0_a15_"
        f"N254_discrete_b040_{CANONICAL_REPLICATE}"
    )


def _replace_option(command: list[str], flag: str, value: str) -> None:
    positions = [i for i, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"expected exactly one valued {flag}")
    command[positions[0] + 1] = value


def command_options(command: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    index = 2
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            raise RuntimeError(f"unexpected command token: {token}")
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            if token in options:
                raise RuntimeError(f"duplicate command option: {token}")
            options[token] = command[index + 1]
            index += 2
        else:
            if token in options:
                raise RuntimeError(f"duplicate command option: {token}")
            options[token] = ""
            index += 1
    return options


def canonical_old_current_command(root: Path) -> list[str]:
    path = (root / OLD_CURRENT_COMMAND).resolve()
    if _file_sha256(path) != OLD_CURRENT_COMMAND_SHA256:
        raise RuntimeError("old alpha15/eta1.1 command hash drifted")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(x, str) for x in payload):
        raise RuntimeError("old control command is not a string vector")
    return payload


def assert_registered_command(
    command: list[str], old_command: list[str], root: Path, arm: str
) -> None:
    arm = _arm(arm)
    expected = list(old_command)
    _replace_option(
        expected, "--output-dir", f"runs/{MEMBER_POLICY_VERSION}/members"
    )
    _replace_option(expected, "--run-label", run_label(arm))
    expected.extend(["--plbfgs-kkt-atol", "0"])
    if arm in SCALAR_ARMS:
        expected[1] = str(
            (root / "src/run_ivm_v9_discrete_scalar_lbfgs_shim.py").resolve()
        )
        expected.extend([scalar_runtime.MODE_FLAG, arm])
    if command != expected:
        raise RuntimeError(f"{arm} command is not the registered exact clone")

    options = command_options(command)
    required = {
        "--initial-path-file": "inputs/common_initial_path_equidistant.npz",
        "--solver": "state_ivm",
        "--alpha": "15",
        "--alpha-mode": "fixed",
        "--max-step": "1",
        "--history": "7",
        "--budget": "254",
        "--reparam-mode": "state_curve",
        "--state-curve-direct-objective-acceptance": "per_image",
        "--stop-complete-iperp": "-1",
        "--frozen-subproblem": "discrete_b040",
        "--discrete-inner-max-steps": "10",
        "--discrete-stop-mode": "r08c10",
        "--discrete-residual-ratio": "0.8",
        "--discrete-history": "7",
        "--discrete-step-multiplier-eta": "1.1",
        "--plbfgs-kkt-atol": "0",
    }
    for flag, value in required.items():
        if options.get(flag) != value:
            raise RuntimeError(f"{arm} changed registered option {flag}")
    if arm in SCALAR_ARMS:
        if options.get(scalar_runtime.MODE_FLAG) != arm:
            raise RuntimeError(f"{arm} runtime mode drifted")
    elif scalar_runtime.MODE_FLAG in options:
        raise RuntimeError("Current0 unexpectedly contains scalar runtime mode")


def single_run_command(root: Path, arm: str) -> tuple[list[str], str]:
    arm = _arm(arm)
    old = canonical_old_current_command(root)
    command = list(old)
    _replace_option(
        command, "--output-dir", f"runs/{MEMBER_POLICY_VERSION}/members"
    )
    _replace_option(command, "--run-label", run_label(arm))
    command.extend(["--plbfgs-kkt-atol", "0"])
    if arm in SCALAR_ARMS:
        command[1] = str(
            (root / "src/run_ivm_v9_discrete_scalar_lbfgs_shim.py").resolve()
        )
        command.extend([scalar_runtime.MODE_FLAG, arm])
    assert_registered_command(command, old, root, arm)
    return command, run_label(arm)


def member_paths(root: Path, arm: str) -> dict[str, Any]:
    label = run_label(arm)
    return {
        "label": label,
        "result_dir": root / "runs" / MEMBER_POLICY_VERSION / "members",
        "audit": root / "runs" / MEMBER_POLICY_VERSION / "audits" / f"{label}.json",
        "member_log": root / "logs" / MEMBER_POLICY_VERSION / "members" / label,
        "launcher_log": root / "logs" / MEMBER_POLICY_VERSION / "launcher" / label,
    }


def _scalar(saved: np.lib.npyio.NpzFile, key: str):
    return np.asarray(saved[key]).item()


def _equal_nan_exact(left: Any, right: Any) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _assert_current0_parity(root: Path, saved: np.lib.npyio.NpzFile) -> list[str]:
    oracle = (root / OLD_CURRENT_RESULT).resolve()
    if _file_sha256(oracle) != OLD_CURRENT_RESULT_SHA256:
        raise RuntimeError("old alpha15/eta1.1 result hash drifted")
    with np.load(oracle, allow_pickle=False) as reference:
        for key in sorted(CURRENT0_PARITY_KEYS):
            if key not in saved.files or key not in reference.files:
                raise RuntimeError(f"Current0 parity field is missing: {key}")
            if not _equal_nan_exact(saved[key], reference[key]):
                raise RuntimeError(f"Current0 whole-run parity failed at {key}")
    return sorted(CURRENT0_PARITY_KEYS)


def _validate_common(saved: np.lib.npyio.NpzFile, arm: str) -> dict[str, Any]:
    if float(_scalar(saved, "alpha")) != CANONICAL_ALPHA or str(
        _scalar(saved, "alpha_mode")
    ) != "fixed":
        raise RuntimeError(f"{arm} changed fixed alpha")
    active = np.asarray(saved["active_alpha"], dtype=float)
    if active.shape != (CANONICAL_BUDGET,) or not np.all(active == CANONICAL_ALPHA):
        raise RuntimeError(f"{arm} alpha is not fixed throughout N254")
    expected = {
        "frozen_subproblem": "discrete_b040",
        "discrete_inner_max_steps": CANONICAL_MAX_EVALUATED_STEPS,
        "discrete_stop_mode": "r08c10",
        "discrete_residual_ratio": CANONICAL_RATIO,
        "discrete_history": CANONICAL_HISTORY,
        "plbfgs_kkt_atol": CANONICAL_KKT_ATOL,
        "reparam_mode": "state_curve",
        "state_curve_direct_objective_acceptance": "per_image",
        "stop_complete_iperp": -1.0,
        "initial_path_sha256": COMMON_INITIAL_ARRAY_SHA256,
    }
    for key, value in expected.items():
        if _scalar(saved, key) != value:
            raise RuntimeError(f"{arm} result drifted at {key}")
    if not bool(_scalar(saved, "discrete_all_returned_states_evaluated")):
        raise RuntimeError(f"{arm} returned an unevaluated state")
    if np.any(np.asarray(saved["subproblem_line_search_backtracks"], int) != 0):
        raise RuntimeError(f"{arm} contains forbidden energy backtracking")

    cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
    i_perp = np.asarray(saved["i_perp"], dtype=float)
    committed = np.asarray(saved["reparam_committed"], dtype=bool)
    cv = np.asarray(saved["segment_length_cv"], dtype=float)
    ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
    arrays = (cost, i_perp, committed, cv, ratio)
    if any(array.shape != (CANONICAL_BUDGET,) for array in arrays):
        raise RuntimeError(f"{arm} does not contain one N254 trajectory")
    if np.any(np.diff(cost) <= 0) or not all(
        np.all(np.isfinite(array)) for array in (i_perp, cv, ratio)
    ):
        raise RuntimeError(f"{arm} trajectory axes are invalid")

    selection = np.asarray(saved["discrete_audit_selection"], dtype="U8")
    outer = np.asarray(saved["discrete_audit_outer"], dtype=int)
    step = np.asarray(saved["discrete_audit_step"], dtype=int)
    if selection.shape != outer.shape or step.shape != outer.shape:
        raise RuntimeError(f"{arm} discrete step audit is misaligned")
    ordered_outers = list(dict.fromkeys(outer.tolist()))
    reasons = np.asarray(saved["discrete_audit_stop_reason"], dtype="U64")
    relatives = np.asarray(
        saved["discrete_audit_global_relative_residual"], dtype=float
    )
    if reasons.shape != relatives.shape or reasons.size != len(ordered_outers):
        raise RuntimeError(f"{arm} stop audit is misaligned")
    allowed_reasons = {
        "evaluated-relative-natural-residual",
        "evaluated-inner-max-steps",
        "per-image-new-call-limit",
        "preflight-per-image-new-call-limit",
    }
    if any(reason not in allowed_reasons for reason in reasons.tolist()):
        raise RuntimeError(f"{arm} used a nonregistered inner stop")
    for index, outer_value in enumerate(ordered_outers):
        local_steps = step[outer == outer_value]
        if not np.array_equal(local_steps, np.arange(local_steps.size)):
            raise RuntimeError(f"{arm} inner evaluated-step numbering drifted")
        if local_steps.size > CANONICAL_MAX_EVALUATED_STEPS:
            raise RuntimeError(f"{arm} exceeded k=10")
        reason = str(reasons[index])
        if reason == "evaluated-relative-natural-residual" and not (
            relatives[index] <= CANONICAL_RATIO
        ):
            raise RuntimeError(f"{arm} relative stop occurred above r=0.8")
        if reason == "evaluated-inner-max-steps" and local_steps.size != 10:
            raise RuntimeError(f"{arm} kmax stop did not occur at k=10")
    if any("absolute" in reason for reason in reasons.astype(str).tolist()):
        raise RuntimeError(f"{arm} used forbidden absolute-KKT stopping")

    complete = committed.copy()
    complete[0] = True
    eligible = complete & (cv <= MESH_CV_MAX) & (ratio <= MESH_RATIO_MAX)
    budget_truncations = int(
        np.count_nonzero(
            np.isin(
                reasons,
                ["per-image-new-call-limit", "preflight-per-image-new-call-limit"],
            )
        )
    )
    return {
        "configuration_sha256": str(_scalar(saved, "configuration_sha256")),
        "initial_path_sha256": str(_scalar(saved, "initial_path_sha256")),
        "trajectory_record_count": int(cost.size),
        "final_optimization_NF": int(cost[-1]),
        "complete_state_NF": cost[complete].tolist(),
        "mesh_eligible_complete_state_NF": cost[eligible].tolist(),
        "mesh_eligible_complete_state_count": int(np.count_nonzero(eligible)),
        "inner_subproblem_count": int(reasons.size),
        "relative_stop_count": int(
            np.count_nonzero(reasons == "evaluated-relative-natural-residual")
        ),
        "k10_stop_count": int(
            np.count_nonzero(reasons == "evaluated-inner-max-steps")
        ),
        "budget_truncation_count": budget_truncations,
        "budget_truncation_counted_as_converged": False,
    }


def _validate_current0(
    root: Path, saved: np.lib.npyio.NpzFile
) -> dict[str, Any]:
    expected_id = eta_runtime.shim_implementation_id(
        BASE_DISCRETE_IMPLEMENTATION_ID, CANONICAL_ETA
    )
    scalar_expectations = {
        "discrete_implementation_id": expected_id,
        "discrete_step_multiplier_policy_id": eta_runtime.SHIM_POLICY_ID,
        "discrete_step_multiplier_base_implementation_id": (
            BASE_DISCRETE_IMPLEMENTATION_ID
        ),
        "discrete_step_multiplier_implementation_id": expected_id,
        "discrete_step_multiplier_eta": CANONICAL_ETA,
        "discrete_step_multiplier_application_stage": eta_runtime.APPLICATION_STAGE,
    }
    for key, expected in scalar_expectations.items():
        if _scalar(saved, key) != expected:
            raise RuntimeError(f"Current0 identity mismatch at {key}")
    for key in (
        "discrete_step_multiplier_selector_isolated",
        "discrete_step_multiplier_dynamic_cap_reapplied",
    ):
        if not bool(_scalar(saved, key)):
            raise RuntimeError(f"Current0 false mechanism flag: {key}")
    if bool(_scalar(saved, "discrete_step_multiplier_energy_backtracking")) or int(
        _scalar(saved, "discrete_step_multiplier_extra_physical_evaluations")
    ) != 0:
        raise RuntimeError("Current0 enabled energy fallback or extra NF")
    if int(_scalar(saved, "discrete_step_multiplier_uncommitted_selection_count")):
        raise RuntimeError("Current0 contains an uncommitted selection")
    if int(_scalar(saved, "discrete_step_multiplier_aborted_selection_count")):
        raise RuntimeError("Current0 contains an aborted selection")

    selection = np.asarray(saved["discrete_audit_selection"], dtype="U8")
    eta_selection = np.asarray(
        saved["discrete_step_multiplier_audit_selection"], dtype="U8"
    )
    requested = np.asarray(
        saved["discrete_step_multiplier_audit_requested"], dtype=float
    )
    applied = np.asarray(saved["discrete_step_multiplier_audit_applied"], dtype=float)
    if not np.array_equal(selection, eta_selection) or not np.all(
        requested == CANONICAL_ETA
    ):
        raise RuntimeError("Current0 eta audit is misaligned")
    if np.any(applied <= 0.0) or np.any(
        applied > CANONICAL_ETA * (1.0 + 1.0e-12)
    ):
        raise RuntimeError("Current0 applied eta is invalid")
    parity = _assert_current0_parity(root, saved)
    return {
        "old_alpha15_eta1p1_whole_run_safe_parity": True,
        "old_alpha15_eta1p1_parity_fields": parity,
        "h0_selection_count": int(np.count_nonzero(selection == "H0")),
        "lbfgs_selection_count": int(np.count_nonzero(selection == "LBFGS")),
    }


def _validate_scalar(saved: np.lib.npyio.NpzFile, arm: str) -> dict[str, Any]:
    expected_id = scalar_runtime.shim_implementation_id(
        BASE_DISCRETE_IMPLEMENTATION_ID, arm
    )
    expected_scalars = {
        "discrete_implementation_id": expected_id,
        "discrete_scalar_lbfgs_runtime_policy_id": scalar_runtime.RUNTIME_POLICY_ID,
        "discrete_scalar_lbfgs_mode": arm,
        "discrete_scalar_lbfgs_mode_policy_id": scalar_runtime.MODE_POLICY_IDS[arm],
        "discrete_scalar_lbfgs_scalar_initial_inverse_id": (
            scalar_runtime.SCALAR_INITIAL_INVERSE_ID
        ),
        "discrete_scalar_lbfgs_post_scale_policy_id": (
            scalar_runtime.POST_SCALE_POLICY_ID
        ),
        "discrete_scalar_lbfgs_base_implementation_id": (
            BASE_DISCRETE_IMPLEMENTATION_ID
        ),
        "discrete_scalar_lbfgs_implementation_id": expected_id,
        "discrete_scalar_lbfgs_alpha": CANONICAL_ALPHA,
        "discrete_scalar_lbfgs_eta": CANONICAL_ETA,
        "discrete_scalar_lbfgs_empty_history_inverse_scale": 1.0,
        "discrete_scalar_lbfgs_nonempty_history_gamma_floor": 1.0e-12,
        "discrete_scalar_lbfgs_history": CANONICAL_HISTORY,
        "discrete_scalar_lbfgs_stop_mode": "r08c10",
        "discrete_scalar_lbfgs_residual_ratio": CANONICAL_RATIO,
        "discrete_scalar_lbfgs_inner_max_steps": CANONICAL_MAX_EVALUATED_STEPS,
        "discrete_scalar_lbfgs_kkt_atol": CANONICAL_KKT_ATOL,
    }
    for key, expected in expected_scalars.items():
        if _scalar(saved, key) != expected:
            raise RuntimeError(f"{arm} scalar identity mismatch at {key}")
    expected_bools = {
        "discrete_scalar_lbfgs_absolute_kkt_stop_enabled": False,
        "discrete_scalar_lbfgs_selector_competition": arm == "scalar_selector",
        "discrete_scalar_lbfgs_direct_only": arm == "direct_lbfgs_only",
        "discrete_scalar_lbfgs_fixed_solver_preconditioner_is_none": True,
        "discrete_scalar_lbfgs_ffrbk_applied_inside_fixed_solver": False,
        "discrete_scalar_lbfgs_bank_build_zero_nf": True,
        "discrete_scalar_lbfgs_energy_backtracking": False,
        "discrete_scalar_lbfgs_reparameterization_inside_subproblem": False,
    }
    for key, expected in expected_bools.items():
        if bool(_scalar(saved, key)) is not expected:
            raise RuntimeError(f"{arm} scalar boolean mismatch at {key}")
    if int(
        _scalar(
            saved,
            "discrete_scalar_lbfgs_ffrbk_solve_normal_calls_inside_fixed_solver",
        )
    ) != 0:
        raise RuntimeError(f"{arm} applied FFrbk inside the fixed solver")
    fixed_calls = int(_scalar(saved, "discrete_scalar_lbfgs_fixed_solver_call_count"))
    intercepts = int(
        _scalar(saved, "discrete_scalar_lbfgs_runner_bank_argument_intercept_count")
    )
    if fixed_calls <= 0 or intercepts != fixed_calls:
        raise RuntimeError(f"{arm} did not isolate every fixed solver bank argument")
    for key in (
        "discrete_scalar_lbfgs_bank_init_raw_joint_evaluation_delta",
        "discrete_scalar_lbfgs_bank_build_raw_joint_evaluation_delta",
    ):
        if np.any(np.asarray(saved[key], dtype=int) != 0):
            raise RuntimeError(f"{arm} bank construction changed physical NF")
    for key in (
        "discrete_scalar_lbfgs_extra_physical_evaluations",
        "discrete_scalar_lbfgs_aborted_solver_call_count",
        "discrete_scalar_lbfgs_uncommitted_selection_count",
    ):
        if int(_scalar(saved, key)) != 0:
            raise RuntimeError(f"{arm} nonzero forbidden counter at {key}")

    selection = np.asarray(saved["discrete_audit_selection"], dtype="U8")
    audit_selection = np.asarray(
        saved["discrete_scalar_lbfgs_audit_selection"], dtype="U8"
    )
    if not np.array_equal(selection, audit_selection):
        raise RuntimeError(f"{arm} scalar selection audit is misaligned")
    aligned = {
        suffix: np.asarray(saved["discrete_scalar_lbfgs_audit_" + suffix])
        for suffix in SCALAR_AUDIT_SUFFIXES
    }
    if any(array.shape != selection.shape for array in aligned.values()):
        raise RuntimeError(f"{arm} scalar step arrays are misaligned")
    history_before = aligned["history_before"].astype(int)
    source = aligned["scale_source"].astype("U64")
    gamma = aligned["initial_inverse_scale"].astype(float)
    raw_gamma = aligned["raw_initial_inverse_scale"].astype(float)
    sy = aligned["last_sy"].astype(float)
    yy = aligned["last_yy"].astype(float)
    floor = aligned["gamma_floor_applied"].astype(bool)
    empty = history_before == 0
    nonempty = ~empty
    if np.any(history_before < 0) or np.any(history_before > CANONICAL_HISTORY):
        raise RuntimeError(f"{arm} history count left m7")
    if not np.all(source[empty] == "identity-empty-history") or not np.all(
        gamma[empty] == 1.0
    ):
        raise RuntimeError(f"{arm} empty-history inverse is not I")
    if np.any(nonempty):
        expected_raw = sy[nonempty] / yy[nonempty]
        if not np.allclose(raw_gamma[nonempty], expected_raw, rtol=1e-13, atol=0.0):
            raise RuntimeError(f"{arm} standard sy/yy scale drifted")
        if not np.allclose(
            gamma[nonempty], np.maximum(expected_raw, 1.0e-12), rtol=0.0, atol=0.0
        ):
            raise RuntimeError(f"{arm} floored scalar inverse drifted")
    fallback = aligned["direct_fallback"].astype(bool)

    h0 = aligned["h0_constructed"].astype(bool)
    selector = aligned["selector_used"].astype(bool)
    if arm == "scalar_selector":
        if not np.all(h0) or not np.all(selector):
            raise RuntimeError("scalar_selector skipped its registered selector")
        if np.any(selection[empty] != "H0"):
            raise RuntimeError("scalar_selector empty-history tie did not choose H0")
    else:
        if np.any(h0) or np.any(selector) or np.any(selection != "LBFGS"):
            raise RuntimeError("direct_lbfgs_only constructed/called a selector arm")

    base = aligned["base_max_atomic_displacement"].astype(float)
    requested_max = aligned["requested_max_atomic_displacement"].astype(float)
    applied_max = aligned["applied_max_atomic_displacement"].astype(float)
    requested = aligned["requested_multiplier"].astype(float)
    applied = aligned["applied_multiplier"].astype(float)
    dynamic = aligned["dynamic_max_step"].astype(float)
    cap_hit = aligned["cap_hit"].astype(bool)
    if not np.all(requested == CANONICAL_ETA) or not np.allclose(
        requested_max, CANONICAL_ETA * base, rtol=1e-13, atol=1e-14
    ):
        raise RuntimeError(f"{arm} eta request audit drifted")
    if np.any(applied <= 0.0) or np.any(applied > CANONICAL_ETA * (1 + 1e-12)):
        raise RuntimeError(f"{arm} eta application is invalid")
    if np.any(applied_max > dynamic * (1 + 1e-12)) or not np.array_equal(
        cap_hit, requested_max > dynamic
    ):
        raise RuntimeError(f"{arm} dynamic recap audit drifted")
    return {
        "scalar_implementation_id": expected_id,
        "fixed_solver_call_count": fixed_calls,
        "runner_bank_argument_intercept_count": intercepts,
        "bank_instance_count": int(
            _scalar(saved, "discrete_scalar_lbfgs_bank_instance_count")
        ),
        "bank_build_count": int(
            _scalar(saved, "discrete_scalar_lbfgs_bank_build_count")
        ),
        "ffrbk_solve_normal_calls_inside_fixed_solver": 0,
        "ffrbk_solve_normal_calls_outside_fixed_solver": int(
            _scalar(
                saved,
                "discrete_scalar_lbfgs_ffrbk_solve_normal_calls_outside_fixed_solver",
            )
        ),
        "h0_selection_count": int(np.count_nonzero(selection == "H0")),
        "lbfgs_selection_count": int(np.count_nonzero(selection == "LBFGS")),
        "direct_fallback_count": int(np.count_nonzero(fallback)),
        "gamma_floor_count": int(np.count_nonzero(floor)),
        "fallback_or_gamma_floor_requires_mechanism_sensitivity_report": bool(
            np.any(fallback) or np.any(floor)
        ),
        "eta_cap_hit_count": int(np.count_nonzero(cap_hit)),
    }


def validate_result_iperp_only(root: Path, result: Path, arm: str) -> dict[str, Any]:
    arm = _arm(arm)
    required = COMMON_RESULT_KEYS | DISCRETE_RESULT_KEYS
    required |= ETA_RESULT_KEYS if arm == "current_ffrbk_selector" else SCALAR_RESULT_KEYS
    forbidden = [key for key in required if "barrier" in key.lower() or "densif" in key.lower()]
    if forbidden:
        raise RuntimeError(f"result allowlist contains barrier data: {forbidden}")
    with np.load(result, allow_pickle=False) as saved:
        missing = sorted(required.difference(saved.files))
        if missing:
            raise RuntimeError(f"{arm} result is missing: {', '.join(missing)}")
        common = _validate_common(saved, arm)
        mechanism = (
            _validate_current0(root, saved)
            if arm == "current_ffrbk_selector"
            else _validate_scalar(saved, arm)
        )
    return {
        **common,
        **mechanism,
        "result_fields_accessed": sorted(required),
        "physical_energy_or_barrier_arrays_accessed": False,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    root = args.root.expanduser().resolve()
    arm = _arm(args.arm)
    command, label = single_run_command(root, arm)
    paths = member_paths(root, arm)
    result_dir = paths["result_dir"]
    audit_path = paths["audit"]
    log_dir = paths["member_log"]
    assert isinstance(result_dir, Path)
    assert isinstance(audit_path, Path)
    assert isinstance(log_dir, Path)
    existing = sorted(
        result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget254.npz")
    )
    companions = sorted(
        result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget254.audit.json")
    )
    if existing or companions or audit_path.exists() or log_dir.exists():
        target = existing[0] if existing else companions[0] if companions else audit_path
        raise FileExistsError(f"refusing to overwrite mechanism cell: {target}")
    result_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    wall_seconds = legacy.run_command(
        command, root, log_dir / "stdout.txt", log_dir / "stderr.txt"
    )
    result = legacy.find_result(result_dir, label, CANONICAL_BUDGET)
    summary = validate_result_iperp_only(root, result, arm)
    audit = {
        "member_policy_version": MEMBER_POLICY_VERSION,
        "arm": arm,
        "alpha": CANONICAL_ALPHA,
        "eta": CANONICAL_ETA,
        "residual_ratio": CANONICAL_RATIO,
        "max_evaluated_steps": CANONICAL_MAX_EVALUATED_STEPS,
        "lbfgs_history": CANONICAL_HISTORY,
        "kkt_atol": CANONICAL_KKT_ATOL,
        "absolute_kkt_stop_enabled": False,
        "budget_per_image": CANONICAL_BUDGET,
        "replicate": CANONICAL_REPLICATE,
        "stop_rule": "R/R0 <= 0.8 OR evaluated accepted steps == 10",
        "reparameterization_policy": "state_curve/per_image from first state",
        "energy_or_objective_backtracking": False,
        "barrier_fields_accessed": False,
        "barrier_computation_or_densification_performed_by_wrapper": False,
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
        "result": str(result),
        "result_sha256": _file_sha256(result),
        "wall_seconds": wall_seconds,
        "scientific_command_sha256": _compact_command_sha256(command),
        "command": command,
        **summary,
    }
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
