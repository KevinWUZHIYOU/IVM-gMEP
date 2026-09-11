#!/usr/bin/env python3
"""One isolated max-evaluated-steps cap ablation member for Ala.

Relative to the frozen alpha=12.5, r=0.8, cap=10 candidate, only
``--discrete-inner-max-steps`` changes scientifically.  Output routing changes
to an independent namespace.  The historical core selector ``r08c10`` is
retained and means the generic rule ``R/R0 <= 0.8 OR evaluated steps == cap``.
No energy-barrier field is indexed, propagated, or ranked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run_stage4_ala_discrete_single_alpha_refinement as frozen_refinement
import run_stage4_ala_discrete_single_fixedalpha as coarse_single
import run_stage4_uniform_optimized as legacy


MEMBER_POLICY_VERSION = (
    "stage4_ala_discrete_single_max_evaluated_steps_cap_ablation_v1"
)
CANONICAL_ALPHA = 12.5
CANONICAL_RATIO = 0.8
REFERENCE_CAP = 10
NEW_CAPS = (1, 2, 3)
CAP_TAGS = {1: "c01", 2: "c02", 3: "c03"}
PRIMARY_LOCK_ELIGIBLE_CAPS = (2, 3, 10)
CANONICAL_BUDGET = 254
CANONICAL_REPLICATE = "r1"
CORE_STOP_MODE_SELECTOR = "r08c10"
LBFGS_HISTORY = 7
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
ROUTING_OPTIONS = frozenset({"--output-dir", "--run-label"})
SCIENTIFIC_ABLATION_OPTION = "--discrete-inner-max-steps"
COMMON_RESULT_KEYS = frozenset(
    {
        "active_alpha",
        "alpha",
        "alpha_mode",
        "configuration_sha256",
        "i_perp",
        "initial_path_sha256",
        "reparam_committed",
        "reparam_mode",
        "segment_length_cv",
        "segment_length_ratio",
        "state_curve_direct_objective_acceptance",
        "stop_complete_iperp",
        "total_internal_joint_evaluations",
    }
)
DISCRETE_RESULT_KEYS = frozenset(
    {
        "discrete_all_returned_states_evaluated",
        "discrete_audit_global_relative_residual",
        "discrete_audit_history_count",
        "discrete_audit_outer",
        "discrete_audit_selection",
        "discrete_audit_step",
        "discrete_audit_stop_reason",
        "discrete_history",
        "discrete_h0_selection_count",
        "discrete_implementation_id",
        "discrete_inner_max_steps",
        "discrete_lbfgs_selection_count",
        "discrete_residual_ratio",
        "discrete_stop_mode",
        "frozen_subproblem",
    }
)


def _require_cap(value: int) -> int:
    if type(value) is not int or value not in NEW_CAPS:
        raise ValueError(
            "Evaluated-step cap must be one of "
            + ",".join(str(item) for item in NEW_CAPS)
        )
    return value


def cap_tag(value: int) -> str:
    return CAP_TAGS[_require_cap(value)]


def run_label(value: int) -> str:
    cap = _require_cap(value)
    return (
        f"{MEMBER_POLICY_VERSION}_{cap_tag(cap)}_r08_a12p5_"
        f"N{CANONICAL_BUDGET}_discrete_b040_{CANONICAL_REPLICATE}"
    )


def _replace_option(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"Expected one valued {flag} option")
    command[positions[0] + 1] = value


def canonical_reference_command(root: Path) -> list[str]:
    command, _ = frozen_refinement.single_run_command(
        root,
        CANONICAL_ALPHA,
        CANONICAL_BUDGET,
        CANONICAL_REPLICATE,
        "candidate",
    )
    return command


def assert_cap_only_scientific_change(
    ablation: list[str], canonical: list[str], value: int
) -> None:
    cap = _require_cap(value)
    if len(ablation) != len(canonical) or ablation[:2] != canonical[:2]:
        raise RuntimeError("Cap ablation changed the Python/core entry point")
    allowed_value_indices = set()
    for flag in ROUTING_OPTIONS | {SCIENTIFIC_ABLATION_OPTION}:
        positions = [index for index, token in enumerate(canonical) if token == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(canonical):
            raise RuntimeError(f"Canonical command has invalid {flag}")
        allowed_value_indices.add(positions[0] + 1)
    drift = [
        index
        for index, (left, right) in enumerate(zip(ablation, canonical))
        if left != right and index not in allowed_value_indices
    ]
    if drift:
        raise RuntimeError(
            "Cap ablation changed a frozen command token at "
            + ",".join(str(index) for index in drift)
        )
    options = coarse_single.command_options(ablation)
    canonical_options = coarse_single.command_options(canonical)
    required = {
        "--alpha": "12.5",
        "--alpha-mode": "fixed",
        "--budget": "254",
        "--frozen-subproblem": "discrete_b040",
        "--discrete-stop-mode": CORE_STOP_MODE_SELECTOR,
        "--discrete-residual-ratio": "0.8",
        "--discrete-history": "7",
        "--history": "7",
        "--reparam-mode": "state_curve",
        "--state-curve-direct-objective-acceptance": "per_image",
        "--stop-complete-iperp": "-1",
    }
    for flag, expected in required.items():
        if options.get(flag) != expected:
            raise RuntimeError(f"Cap ablation changed frozen option {flag}")
    if int(options[SCIENTIFIC_ABLATION_OPTION]) != cap:
        raise RuntimeError("Cap ablation carries the wrong evaluated-step cap")
    for flag, expected in canonical_options.items():
        if flag in ROUTING_OPTIONS | {SCIENTIFIC_ABLATION_OPTION}:
            continue
        if options.get(flag) != expected:
            raise RuntimeError(f"Cap ablation changed frozen option {flag}")


def single_run_command(root: Path, value: int) -> tuple[list[str], str]:
    cap = _require_cap(value)
    canonical = canonical_reference_command(root)
    command = list(canonical)
    label = run_label(cap)
    _replace_option(
        command,
        "--output-dir",
        f"runs/{MEMBER_POLICY_VERSION}/members",
    )
    _replace_option(command, "--run-label", label)
    _replace_option(command, SCIENTIFIC_ABLATION_OPTION, str(cap))
    assert_cap_only_scientific_change(command, canonical, cap)
    return command, label


def compact_command_sha256(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def member_paths(root: Path, value: int) -> dict[str, Path | str]:
    label = run_label(value)
    return {
        "label": label,
        "result_dir": root / "runs" / MEMBER_POLICY_VERSION / "members",
        "audit": root / "runs" / MEMBER_POLICY_VERSION / "audits" / f"{label}.json",
        "member_log": root / "logs" / MEMBER_POLICY_VERSION / "members" / label,
        "launcher_log": root / "logs" / MEMBER_POLICY_VERSION / "launcher" / label,
    }


def _require_keys(saved: np.lib.npyio.NpzFile, required, result: Path) -> None:
    missing = sorted(set(required).difference(saved.files))
    if missing:
        raise RuntimeError(f"{result} is missing: {', '.join(missing)}")


def _validate_result_iperp_only(result: Path, value: int) -> dict[str, object]:
    cap = _require_cap(value)
    with np.load(result, allow_pickle=False) as saved:
        _require_keys(saved, COMMON_RESULT_KEYS | DISCRETE_RESULT_KEYS, result)
        if float(legacy.scalar(saved, "alpha")) != CANONICAL_ALPHA:
            raise RuntimeError("Result did not retain alpha=12.5")
        if str(legacy.scalar(saved, "alpha_mode")) != "fixed":
            raise RuntimeError("Result alpha mode is not fixed")
        active = np.asarray(saved["active_alpha"], dtype=float)
        if not active.size or not np.all(active == CANONICAL_ALPHA):
            raise RuntimeError("Result changed alpha within the trajectory")
        if str(legacy.scalar(saved, "reparam_mode")) != "state_curve":
            raise RuntimeError("Result changed reparameterization")
        if str(legacy.scalar(saved, "state_curve_direct_objective_acceptance")) != "per_image":
            raise RuntimeError("Result changed per-image state acceptance")
        if float(legacy.scalar(saved, "stop_complete_iperp")) != -1.0:
            raise RuntimeError("Result used an I_perp early stop")
        if str(legacy.scalar(saved, "discrete_implementation_id")) != coarse_single.EXPECTED_IMPLEMENTATION_ID:
            raise RuntimeError("Result used the wrong discrete implementation")
        if str(legacy.scalar(saved, "frozen_subproblem")) != "discrete_b040":
            raise RuntimeError("Result changed the frozen subproblem")
        if str(legacy.scalar(saved, "discrete_stop_mode")) != CORE_STOP_MODE_SELECTOR:
            raise RuntimeError("Result changed the core stop selector")
        if int(legacy.scalar(saved, "discrete_inner_max_steps")) != cap:
            raise RuntimeError("Result used the wrong evaluated-step cap")
        if float(legacy.scalar(saved, "discrete_residual_ratio")) != CANONICAL_RATIO:
            raise RuntimeError("Result changed the residual ratio")
        if int(legacy.scalar(saved, "discrete_history")) != LBFGS_HISTORY:
            raise RuntimeError("Result changed L-BFGS history")
        if not bool(legacy.scalar(saved, "discrete_all_returned_states_evaluated")):
            raise RuntimeError("Result returned an unevaluated state")

        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        mesh_ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        if len({array.shape for array in (cost, i_perp, committed, cv, mesh_ratio)}) != 1 or cost.ndim != 1 or cost.size != CANONICAL_BUDGET:
            raise RuntimeError("Result trajectory arrays have invalid shapes")
        if np.any(np.diff(cost) <= 0):
            raise RuntimeError("Optimization NF is not strictly increasing")
        if not np.all(np.isfinite(i_perp)) or not np.all(np.isfinite(cv)) or not np.all(np.isfinite(mesh_ratio)):
            raise RuntimeError("I_perp or mesh diagnostics are non-finite")
        if np.any(i_perp < 0.0) or np.any(cv < 0.0) or np.any(mesh_ratio < 1.0):
            raise RuntimeError("I_perp or mesh diagnostics are invalid")

        outer = np.asarray(saved["discrete_audit_outer"], dtype=int)
        steps = np.asarray(saved["discrete_audit_step"], dtype=int)
        selections = np.asarray(saved["discrete_audit_selection"], dtype="U16")
        histories = np.asarray(saved["discrete_audit_history_count"], dtype=int)
        residuals = np.asarray(saved["discrete_audit_global_relative_residual"], dtype=float)
        stop_reasons = np.asarray(saved["discrete_audit_stop_reason"], dtype="U64")
        if len({array.shape for array in (outer, steps, selections, histories)}) != 1:
            raise RuntimeError("Per-step mechanism audit lengths differ")
        if residuals.shape != stop_reasons.shape:
            raise RuntimeError("Per-solve mechanism audit lengths differ")
        if np.any(steps < 0) or np.any(steps >= cap):
            raise RuntimeError("A recorded evaluated step exceeds the cap")
        for outer_index in np.unique(outer):
            observed = steps[outer == outer_index].tolist()
            if observed != list(range(len(observed))):
                raise RuntimeError("Per-solve evaluated step indices are not contiguous")
        if np.any(histories < 0) or np.any(histories > LBFGS_HISTORY):
            raise RuntimeError("Invalid L-BFGS history audit")
        if not np.all(np.isfinite(residuals)) or np.any(residuals < 0.0):
            raise RuntimeError("Invalid residual audit")
        h0_count = int(np.count_nonzero(selections == "H0"))
        lbfgs_count = int(np.count_nonzero(selections == "LBFGS"))
        if h0_count != int(legacy.scalar(saved, "discrete_h0_selection_count")) or lbfgs_count != int(legacy.scalar(saved, "discrete_lbfgs_selection_count")):
            raise RuntimeError("Direction selection count mismatch")
        if h0_count + lbfgs_count != selections.size:
            raise RuntimeError("Unknown direction selection")
        if cap == 1 and (lbfgs_count != 0 or not np.all(selections == "H0")):
            raise RuntimeError("Cap=1 must be the H0-only mechanism endpoint")

        complete = committed.copy()
        complete[0] = True
        eligible = complete & (cv <= MESH_CV_MAX) & (mesh_ratio <= MESH_RATIO_MAX)
        return {
            "configuration_sha256": str(legacy.scalar(saved, "configuration_sha256")),
            "initial_path_sha256": str(legacy.scalar(saved, "initial_path_sha256")),
            "trajectory_record_count": int(cost.size),
            "final_optimization_NF": int(cost[-1]),
            "complete_state_definition": "initial_plus_reparam_committed",
            "complete_state_NF": cost[complete].tolist(),
            "mesh_eligibility": "CV<=0.0065 and ratio<=1.03",
            "mesh_eligible_complete_state_NF": cost[eligible].tolist(),
            "mesh_eligible_complete_state_count": int(np.count_nonzero(eligible)),
            "reparam_committed_state_count": int(np.count_nonzero(committed)),
            "frozen_subproblem_solve_count": int(stop_reasons.size),
            "discrete_joint_step_count": int(selections.size),
            "h0_selection_count": h0_count,
            "lbfgs_selection_count": lbfgs_count,
            "lbfgs_takeover_observed": bool(lbfgs_count > 0),
            "lbfgs_eligible_step_count": int(np.count_nonzero(steps >= 1)),
            "maximum_history_count": int(np.max(histories)) if histories.size else 0,
            "relative_residual_stop_count": int(np.count_nonzero(stop_reasons == "evaluated-relative-natural-residual")),
            "inner_cap_stop_count": int(np.count_nonzero(stop_reasons == "evaluated-inner-max-steps")),
            "result_fields_accessed": sorted(COMMON_RESULT_KEYS | DISCRETE_RESULT_KEYS),
        }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-evaluated-steps", type=int, choices=NEW_CAPS, required=True)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    root = args.root.expanduser().resolve()
    cap = _require_cap(args.max_evaluated_steps)
    command, label = single_run_command(root, cap)
    paths = member_paths(root, cap)
    result_dir, audit_path, log_dir = paths["result_dir"], paths["audit"], paths["member_log"]
    assert isinstance(result_dir, Path) and isinstance(audit_path, Path) and isinstance(log_dir, Path)
    existing = sorted(result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget{CANONICAL_BUDGET}.npz"))
    if existing or audit_path.exists() or log_dir.exists():
        target = existing[0] if existing else audit_path
        raise FileExistsError(f"Refusing to overwrite cap member: {target}")
    result_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    wall_seconds = legacy.run_command(command, root, log_dir / "stdout.txt", log_dir / "stderr.txt")
    result = legacy.find_result(result_dir, label, CANONICAL_BUDGET)
    summary = _validate_result_iperp_only(result, cap)
    audit: dict[str, object] = {
        "member_policy_version": MEMBER_POLICY_VERSION,
        "member_role": "new_max_evaluated_steps_candidate",
        "alpha": CANONICAL_ALPHA,
        "residual_ratio": CANONICAL_RATIO,
        "max_evaluated_steps": cap,
        "cap_tag": cap_tag(cap),
        "budget_per_image": CANONICAL_BUDGET,
        "replicate": CANONICAL_REPLICATE,
        "lbfgs_history": LBFGS_HISTORY,
        "core_stop_mode_selector": CORE_STOP_MODE_SELECTOR,
        "stop_rule": f"R/R0 <= 0.8 OR evaluated accepted steps == {cap}",
        "scientific_change_from_frozen_cap10": "only --discrete-inner-max-steps",
        "routing_changes_from_frozen_cap10": ["--output-dir", "--run-label"],
        "scientific_command_sha256": compact_command_sha256(command),
        "primary_lock_eligible_cap": cap in PRIMARY_LOCK_ELIGIBLE_CAPS,
        "cap1_mechanism_only_h0_endpoint": cap == 1,
        "primary_requires_lbfgs_takeover": True,
        "single_run_from_canonical_common_path": True,
        "reparameterization_policy": "state_curve/per_image from first state",
        "iperp_only_scientific_analysis": True,
        "barrier_fields_accessed": False,
        "barrier_computation_or_densification_performed_by_wrapper": False,
        "barrier_embargo": "all energy-barrier fields remain unread and unranked",
        "formal_ranking_deferred_to_complete_mesh_iperp_ctp": True,
        "result": str(result),
        "result_sha256": legacy.file_sha256(result),
        "wall_seconds": wall_seconds,
        "command": command,
        **summary,
    }
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
