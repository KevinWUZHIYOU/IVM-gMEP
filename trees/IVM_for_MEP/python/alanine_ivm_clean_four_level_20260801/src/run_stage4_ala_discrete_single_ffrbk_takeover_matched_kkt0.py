#!/usr/bin/env python3
"""Run one matched formal FFrbk selector-policy cell.

Both arms clone the frozen matched Current0 alpha=15/eta=1.1/kkt_atol=0
command.  The only scientific-command differences are the runtime entry point
and its selector-mode flag; output routing is independent per arm.  This
wrapper is deliberately I_perp-only and accepts a physically stopped
trajectory with 1--254 stored records as a valid *execution*, never as
scientific convergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import run_ivm_v9_discrete_ffrbk_takeover_selector_shim as runtime
import run_ivm_v9_discrete_selected_eta_shim as eta_runtime
import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0 as base
import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_direct_continuation as flexible
import run_stage4_uniform_optimized as legacy


MEMBER_POLICY_VERSION = (
    "stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1"
)
ARMS = ("one_way_takeover", "forced_operator_lbfgs_selection")
CANONICAL_BUDGET = 254
CANONICAL_REPLICATE = "r1"

CURRENT0_COMMAND = (
    "logs/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1/"
    "members/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
    "current_ffrbk_selector_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1/"
    "command.json"
)
CURRENT0_COMMAND_SHA256 = (
    "e33ca25d88bcc6e13867f074f0b6ff242ac0822f948fb758edb52ea62a640863"
)
ROUTING_OPTIONS = frozenset(("--output-dir", "--run-label"))

TAKEOVER_STATIC_RESULT_KEYS = frozenset(
    {
        "discrete_ffrbk_selector_runtime_policy_id",
        "discrete_ffrbk_selector_mode_policy_id",
        "discrete_ffrbk_selector_mode",
        "discrete_ffrbk_selector_operator_policy_id",
        "discrete_ffrbk_selector_base_implementation_id",
        "discrete_ffrbk_selector_pre_eta_implementation_id",
        "discrete_ffrbk_selector_final_implementation_id",
        "discrete_ffrbk_selector_alpha",
        "discrete_ffrbk_selector_eta",
        "discrete_ffrbk_selector_stop_mode",
        "discrete_ffrbk_selector_residual_ratio",
        "discrete_ffrbk_selector_inner_max_steps",
        "discrete_ffrbk_selector_history",
        "discrete_ffrbk_selector_kkt_atol",
        "discrete_ffrbk_selector_fixed_solver_calls",
        "discrete_ffrbk_selector_latch_reset_count",
        "discrete_ffrbk_selector_aborted_solver_calls",
        "discrete_ffrbk_selector_apply_h0_inverse_calls",
        "discrete_ffrbk_selector_apply_h0_inverse_preconditioner_none_calls",
        "discrete_ffrbk_selector_joint_lbfgs_direction_calls",
        "discrete_ffrbk_selector_committed_selection_count",
        "discrete_ffrbk_selector_uncommitted_selection_count",
        "discrete_ffrbk_selector_aborted_selection_count",
        "discrete_ffrbk_selector_energy_backtracking",
        "discrete_ffrbk_selector_extra_physical_evaluations",
        "discrete_ffrbk_selector_operator_preserved",
        "discrete_ffrbk_selector_post_select_eta_composed",
        "discrete_ffrbk_selector_reparameterization_inside_subproblem",
        "discrete_ffrbk_selector_selector_only_change",
        "discrete_ffrbk_selector_h0_shadow_constructed",
        "discrete_ffrbk_selector_true_lbfgs_only_construction",
        "discrete_ffrbk_selector_invalid_lbfgs_h0_fallback_enabled",
        "discrete_ffrbk_selector_one_way_latch_is_preference",
    }
)
TAKEOVER_AUDIT_SUFFIXES = frozenset(
    {
        "selection",
        "solver_call",
        "selector_call",
        "latched_before",
        "latched_after",
        "strict_win_trigger",
        "forced_lbfgs",
        "empty_history_tie",
        "h0_failover",
        "h0_valid",
        "lbfgs_valid",
        "h0_capped_decrease",
        "lbfgs_capped_decrease",
    }
)
TAKEOVER_AUDIT_RESULT_KEYS = frozenset(
    "discrete_ffrbk_selector_audit_" + suffix
    for suffix in TAKEOVER_AUDIT_SUFFIXES
)
TAKEOVER_RESULT_KEYS = TAKEOVER_STATIC_RESULT_KEYS | TAKEOVER_AUDIT_RESULT_KEYS


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
        raise ValueError("unknown FFrbk selector-policy arm")
    return value


def run_label(arm: str) -> str:
    arm = _arm(arm)
    return (
        f"{MEMBER_POLICY_VERSION}_{arm}_eta1p1_r08c10_kkt0_a15_"
        f"N254_discrete_b040_{CANONICAL_REPLICATE}"
    )


def _replace_option(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"expected exactly one valued {flag}")
    command[positions[0] + 1] = value


def current0_command(root: Path) -> list[str]:
    path = (root / CURRENT0_COMMAND).resolve()
    if _file_sha256(path) != CURRENT0_COMMAND_SHA256:
        raise RuntimeError("matched Current0 command hash drifted")
    command = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(command, list) or not all(
        isinstance(token, str) for token in command
    ):
        raise RuntimeError("matched Current0 command is not a string vector")
    return command


def assert_registered_command(
    command: list[str], current0: list[str], root: Path, arm: str
) -> None:
    arm = _arm(arm)
    expected = list(current0)
    expected[1] = str(
        (root / "src/run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py").resolve()
    )
    _replace_option(
        expected, "--output-dir", f"runs/{MEMBER_POLICY_VERSION}/members"
    )
    _replace_option(expected, "--run-label", run_label(arm))
    expected.extend([runtime.MODE_FLAG, arm])
    if command != expected:
        raise RuntimeError(f"{arm} is not the exact matched Current0 clone")

    # Apart from the inserted mode pair, exactly entry point and two routing
    # values may differ from the pinned Current0 vector.
    if len(command) != len(current0) + 2:
        raise RuntimeError(f"{arm} command length drifted")
    if command[-2:] != [runtime.MODE_FLAG, arm]:
        raise RuntimeError(f"{arm} mode routing drifted")
    differences = [
        index
        for index, (old, new) in enumerate(zip(current0, command[:-2]))
        if old != new
    ]
    allowed = sorted(
        [1]
        + [current0.index(flag) + 1 for flag in sorted(ROUTING_OPTIONS)]
    )
    if differences != allowed:
        raise RuntimeError(f"{arm} changed a nonregistered Current0 token")

    options = base.command_options(command)
    required = {
        "--root": str(root.resolve()),
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
        runtime.MODE_FLAG: arm,
    }
    for flag, value in required.items():
        if options.get(flag) != value:
            raise RuntimeError(f"{arm} changed registered option {flag}")


def single_run_command(root: Path, arm: str) -> tuple[list[str], str]:
    arm = _arm(arm)
    current0 = current0_command(root)
    command = list(current0)
    command[1] = str(
        (root / "src/run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py").resolve()
    )
    _replace_option(
        command, "--output-dir", f"runs/{MEMBER_POLICY_VERSION}/members"
    )
    _replace_option(command, "--run-label", run_label(arm))
    command.extend([runtime.MODE_FLAG, arm])
    assert_registered_command(command, current0, root, arm)
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


def _validate_eta(saved: np.lib.npyio.NpzFile) -> dict[str, Any]:
    expected = {
        "discrete_step_multiplier_policy_id": eta_runtime.SHIM_POLICY_ID,
        "discrete_step_multiplier_eta": base.CANONICAL_ETA,
        "discrete_step_multiplier_application_stage": eta_runtime.APPLICATION_STAGE,
    }
    for key, value in expected.items():
        if _scalar(saved, key) != value:
            raise RuntimeError(f"eta composition drifted at {key}")
    for key in (
        "discrete_step_multiplier_selector_isolated",
        "discrete_step_multiplier_dynamic_cap_reapplied",
    ):
        if not bool(_scalar(saved, key)):
            raise RuntimeError(f"eta composition false mechanism flag: {key}")
    if bool(_scalar(saved, "discrete_step_multiplier_energy_backtracking")):
        raise RuntimeError("eta composition enabled energy backtracking")
    if int(_scalar(saved, "discrete_step_multiplier_extra_physical_evaluations")):
        raise RuntimeError("eta composition added physical evaluations")
    if int(_scalar(saved, "discrete_step_multiplier_aborted_selection_count")):
        raise RuntimeError("eta composition contains an aborted selection")
    selection = np.asarray(saved["discrete_audit_selection"], dtype="U16")
    eta_selection = np.asarray(
        saved["discrete_step_multiplier_audit_selection"], dtype="U16"
    )
    requested = np.asarray(
        saved["discrete_step_multiplier_audit_requested"], dtype=float
    )
    applied = np.asarray(
        saved["discrete_step_multiplier_audit_applied"], dtype=float
    )
    if not np.array_equal(selection, eta_selection):
        raise RuntimeError("eta selection audit is misaligned")
    if not np.all(requested == base.CANONICAL_ETA):
        raise RuntimeError("eta request is not fixed at 1.1")
    if np.any(applied <= 0.0) or np.any(
        applied > base.CANONICAL_ETA * (1.0 + 1.0e-12)
    ):
        raise RuntimeError("eta application is invalid")
    return {"eta_cap_hit_count": int(np.count_nonzero(
        np.asarray(saved["discrete_step_multiplier_audit_cap_hit"], dtype=bool)
    ))}


def _validate_takeover(saved: np.lib.npyio.NpzFile, arm: str) -> dict[str, Any]:
    arm = _arm(arm)
    policy_id = runtime.RUNTIME_POLICY_ID
    mode_ids = runtime.MODE_POLICY_IDS
    expected = {
        "discrete_ffrbk_selector_runtime_policy_id": policy_id,
        "discrete_ffrbk_selector_mode_policy_id": mode_ids[arm],
        "discrete_ffrbk_selector_mode": arm,
        "discrete_ffrbk_selector_operator_policy_id": runtime.OPERATOR_POLICY_ID,
        "discrete_ffrbk_selector_base_implementation_id": (
            base.BASE_DISCRETE_IMPLEMENTATION_ID
        ),
        "discrete_ffrbk_selector_pre_eta_implementation_id": (
            runtime.selector_implementation_id(
                base.BASE_DISCRETE_IMPLEMENTATION_ID, arm
            )
        ),
        "discrete_ffrbk_selector_final_implementation_id": (
            runtime.final_implementation_id(
                base.BASE_DISCRETE_IMPLEMENTATION_ID, arm
            )
        ),
        "discrete_ffrbk_selector_alpha": base.CANONICAL_ALPHA,
        "discrete_ffrbk_selector_eta": base.CANONICAL_ETA,
        "discrete_ffrbk_selector_stop_mode": "r08c10",
        "discrete_ffrbk_selector_residual_ratio": base.CANONICAL_RATIO,
        "discrete_ffrbk_selector_inner_max_steps": (
            base.CANONICAL_MAX_EVALUATED_STEPS
        ),
        "discrete_ffrbk_selector_history": base.CANONICAL_HISTORY,
        "discrete_ffrbk_selector_kkt_atol": base.CANONICAL_KKT_ATOL,
    }
    for key, value in expected.items():
        if _scalar(saved, key) != value:
            raise RuntimeError(f"{arm} runtime identity drifted at {key}")

    pre_eta = str(_scalar(saved, "discrete_ffrbk_selector_pre_eta_implementation_id"))
    final = str(_scalar(saved, "discrete_ffrbk_selector_final_implementation_id"))
    if not pre_eta or not final or str(_scalar(saved, "discrete_implementation_id")) != final:
        raise RuntimeError(f"{arm} implementation identity chain drifted")
    if str(_scalar(saved, "discrete_step_multiplier_base_implementation_id")) != pre_eta:
        raise RuntimeError(f"{arm} eta was not composed after selector policy")
    if str(_scalar(saved, "discrete_step_multiplier_implementation_id")) != final:
        raise RuntimeError(f"{arm} final eta identity drifted")

    expected_bools = {
        "discrete_ffrbk_selector_energy_backtracking": False,
        "discrete_ffrbk_selector_operator_preserved": True,
        "discrete_ffrbk_selector_post_select_eta_composed": True,
        "discrete_ffrbk_selector_reparameterization_inside_subproblem": False,
        "discrete_ffrbk_selector_selector_only_change": True,
        "discrete_ffrbk_selector_h0_shadow_constructed": True,
        "discrete_ffrbk_selector_true_lbfgs_only_construction": False,
        "discrete_ffrbk_selector_invalid_lbfgs_h0_fallback_enabled": True,
        "discrete_ffrbk_selector_one_way_latch_is_preference": (
            arm == "one_way_takeover"
        ),
    }
    for key, value in expected_bools.items():
        if bool(_scalar(saved, key)) is not value:
            raise RuntimeError(f"{arm} runtime boolean drifted at {key}")
    for key in (
        "discrete_ffrbk_selector_extra_physical_evaluations",
        "discrete_ffrbk_selector_aborted_solver_calls",
        "discrete_ffrbk_selector_apply_h0_inverse_preconditioner_none_calls",
        "discrete_ffrbk_selector_uncommitted_selection_count",
        "discrete_ffrbk_selector_aborted_selection_count",
    ):
        if int(_scalar(saved, key)) != 0:
            raise RuntimeError(f"{arm} nonzero forbidden counter at {key}")

    selection = np.asarray(saved["discrete_audit_selection"], dtype="U16")
    aligned = {
        suffix: np.asarray(saved["discrete_ffrbk_selector_audit_" + suffix])
        for suffix in TAKEOVER_AUDIT_SUFFIXES
    }
    if any(array.shape != selection.shape for array in aligned.values()):
        raise RuntimeError(f"{arm} selector audit arrays are misaligned")
    if not np.array_equal(aligned["selection"].astype("U16"), selection):
        raise RuntimeError(f"{arm} selector identity audit is misaligned")
    if np.any(~np.isin(selection, ["H0", "LBFGS"])):
        raise RuntimeError(f"{arm} contains an unknown selection")
    h0_valid = aligned["h0_valid"].astype(bool)
    lbfgs_valid = aligned["lbfgs_valid"].astype(bool)
    if np.any((selection == "H0") & ~h0_valid) or np.any(
        (selection == "LBFGS") & ~lbfgs_valid
    ):
        raise RuntimeError(f"{arm} selected an invalid candidate")

    before = aligned["latched_before"].astype(bool)
    after = aligned["latched_after"].astype(bool)
    trigger = aligned["strict_win_trigger"].astype(bool)
    forced = aligned["forced_lbfgs"].astype(bool)
    empty_tie = aligned["empty_history_tie"].astype(bool)
    failover = aligned["h0_failover"].astype(bool)
    solver_call = aligned["solver_call"].astype(int)
    selector_call = aligned["selector_call"].astype(int)
    if np.any(solver_call < 0) or np.any(selector_call < 0):
        raise RuntimeError(f"{arm} selector call indices are invalid")
    for call in dict.fromkeys(solver_call.tolist()):
        local = selector_call[solver_call == call]
        if not np.array_equal(local, np.arange(local.size)):
            raise RuntimeError(f"{arm} selector calls are not local and contiguous")
        if before[solver_call == call][0]:
            raise RuntimeError(f"{arm} latch was not reset at a fixed subproblem")

    if arm == "one_way_takeover":
        if np.any(before & ~after):
            raise RuntimeError("one-way takeover latch moved backwards")
        if np.any(trigger & (~after | (selection != "LBFGS"))):
            raise RuntimeError("strict L-BFGS win did not trigger takeover")
        if np.any(forced & (~before | (selection != "LBFGS"))):
            raise RuntimeError("latched takeover did not force L-BFGS")
        if np.any(before & (selection != "LBFGS") & ~failover):
            raise RuntimeError("latched one-way mode returned to H0")
    else:
        if np.any(before) or np.any(after) or np.any(trigger):
            raise RuntimeError("forced operator-LBFGS selection unexpectedly used a latch")
        if np.any(lbfgs_valid & (selection != "LBFGS")):
            raise RuntimeError("forced operator-LBFGS mode did not select L-BFGS")
        if np.any(empty_tie & lbfgs_valid & (selection != "LBFGS")):
            raise RuntimeError("forced operator-LBFGS mode lost its empty-history tie")
        if np.any((selection == "H0") & ~failover):
            raise RuntimeError("forced operator-LBFGS H0 was not a failover")
        if np.any(forced != (selection == "LBFGS")):
            raise RuntimeError("forced operator-LBFGS selection audit drifted")

    fixed_calls = int(_scalar(saved, "discrete_ffrbk_selector_fixed_solver_calls"))
    latch_resets = int(_scalar(saved, "discrete_ffrbk_selector_latch_reset_count"))
    if fixed_calls <= 0 or latch_resets != fixed_calls:
        raise RuntimeError(f"{arm} did not reset selector state per fixed subproblem")
    apply_calls = int(
        _scalar(saved, "discrete_ffrbk_selector_apply_h0_inverse_calls")
    )
    joint_calls = int(
        _scalar(saved, "discrete_ffrbk_selector_joint_lbfgs_direction_calls")
    )
    # A one-selection prefix has empty history: it must apply FFrbk H0, while
    # the two-loop joint-LBFGS operator correctly has zero calls.  This case is
    # admissible under the preregistered 1--254 early-stop policy.
    if apply_calls < 0 or joint_calls < 0 or (selection.size and apply_calls <= 0):
        raise RuntimeError(f"{arm} FFrbk operator call audit is invalid")
    committed = int(
        _scalar(saved, "discrete_ffrbk_selector_committed_selection_count")
    )
    if committed != selection.size:
        raise RuntimeError(f"{arm} committed-selection count is misaligned")
    return {
        "selector_policy_id": policy_id,
        "selector_mode_policy_id": mode_ids[arm],
        "selector_pre_eta_implementation_id": pre_eta,
        "selector_final_implementation_id": final,
        "fixed_solver_call_count": fixed_calls,
        "latch_reset_count": latch_resets,
        "apply_h0_inverse_call_count": apply_calls,
        "joint_lbfgs_direction_call_count": joint_calls,
        "h0_selection_count": int(np.count_nonzero(selection == "H0")),
        "lbfgs_selection_count": int(np.count_nonzero(selection == "LBFGS")),
        "strict_win_trigger_count": int(np.count_nonzero(trigger)),
        "forced_lbfgs_count": int(np.count_nonzero(forced)),
        "empty_history_tie_count": int(np.count_nonzero(empty_tie)),
        "h0_failover_count": int(np.count_nonzero(failover)),
    }


def validate_result_iperp_only(root: Path, result: Path, arm: str) -> dict[str, Any]:
    arm = _arm(arm)
    required = (
        base.COMMON_RESULT_KEYS
        | base.DISCRETE_RESULT_KEYS
        | base.ETA_RESULT_KEYS
        | TAKEOVER_RESULT_KEYS
        | {"stop_reason"}
    )
    # The two mechanism flags mentioning energy are scalar negative controls,
    # not physical-objective arrays.  No physical energy/barrier name belongs
    # to the allowlist.
    allowed_negative_controls = {
        "discrete_step_multiplier_energy_backtracking",
        "discrete_step_multiplier_extra_physical_evaluations",
        "discrete_ffrbk_selector_energy_backtracking",
        "discrete_ffrbk_selector_extra_physical_evaluations",
    }
    forbidden = [
        key
        for key in required
        if (
            "barrier" in key.lower()
            or "densif" in key.lower()
            or "energy" in key.lower()
        )
        and key not in allowed_negative_controls
    ]
    if forbidden:
        raise RuntimeError(f"Iperp-only allowlist contains forbidden data: {forbidden}")
    with np.load(result, allow_pickle=False) as saved:
        missing = sorted(required.difference(saved.files))
        if missing:
            raise RuntimeError(f"{arm} result is missing: {', '.join(missing)}")
        common = flexible._validate_common_flexible(saved)
        takeover = _validate_takeover(saved, arm)
        eta = _validate_eta(saved)
    return {
        **common,
        **takeover,
        **eta,
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
        raise FileExistsError(f"refusing to overwrite selector cell: {target}")

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
        "alpha": base.CANONICAL_ALPHA,
        "eta": base.CANONICAL_ETA,
        "residual_ratio": base.CANONICAL_RATIO,
        "max_evaluated_steps": base.CANONICAL_MAX_EVALUATED_STEPS,
        "lbfgs_history": base.CANONICAL_HISTORY,
        "kkt_atol": base.CANONICAL_KKT_ATOL,
        "absolute_kkt_stop_enabled": False,
        "budget_per_image": CANONICAL_BUDGET,
        "trajectory_record_count_allowed_range": [1, CANONICAL_BUDGET],
        "replicate": CANONICAL_REPLICATE,
        "stop_rule": "R/R0 <= 0.8 OR evaluated accepted steps == 10",
        "reparameterization_policy": "state_curve/per_image from first state",
        "early_stop_is_scientific_convergence": False,
        "energy_or_objective_backtracking": False,
        "barrier_fields_accessed": False,
        "barrier_computation_or_densification_performed_by_wrapper": False,
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
        "result": str(result),
        "result_sha256": _file_sha256(result),
        "wall_seconds": wall_seconds,
        "scientific_command_sha256": _compact_command_sha256(command),
        "matched_current0_command_sha256": CURRENT0_COMMAND_SHA256,
        "command": command,
        **summary,
    }
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
