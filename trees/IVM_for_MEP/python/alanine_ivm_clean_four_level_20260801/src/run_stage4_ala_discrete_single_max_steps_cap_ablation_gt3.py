#!/usr/bin/env python3
"""One max-evaluated-steps>3 cap-ablation member for Ala v2.

Only output routing and ``--discrete-inner-max-steps`` differ from the frozen
alpha=12.5, r=0.8, cap=10 command.  Energy-barrier fields remain unread.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run_stage4_ala_discrete_single_max_steps_cap_ablation as superseded_v1
import run_stage4_uniform_optimized as legacy


MEMBER_POLICY_VERSION = (
    "stage4_ala_discrete_single_max_evaluated_steps_cap_ablation_gt3_v2"
)
CANONICAL_ALPHA = 12.5
CANONICAL_RATIO = 0.8
REFERENCE_CAP = 10
NEW_CAPS = (4, 5, 6)
CAP_TAGS = {4: "c04", 5: "c05", 6: "c06"}
PRIMARY_LOCK_ELIGIBLE_CAPS = NEW_CAPS + (REFERENCE_CAP,)
CANONICAL_BUDGET = 254
CANONICAL_REPLICATE = "r1"
CORE_STOP_MODE_SELECTOR = "r08c10"
LBFGS_HISTORY = 7
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
ROUTING_OPTIONS = frozenset({"--output-dir", "--run-label"})
SCIENTIFIC_ABLATION_OPTION = "--discrete-inner-max-steps"
COMMON_RESULT_KEYS = superseded_v1.COMMON_RESULT_KEYS
DISCRETE_RESULT_KEYS = superseded_v1.DISCRETE_RESULT_KEYS


def _require_cap(value: int) -> int:
    if type(value) is not int or value not in NEW_CAPS:
        raise ValueError("Evaluated-step cap must be one of 4,5,6")
    return value


def cap_tag(value: int) -> str:
    return CAP_TAGS[_require_cap(value)]


def run_label(value: int) -> str:
    cap = _require_cap(value)
    return (
        f"{MEMBER_POLICY_VERSION}_{cap_tag(cap)}_r08_a12p5_"
        f"N254_discrete_b040_r1"
    )


def canonical_reference_command(root: Path) -> list[str]:
    return superseded_v1.canonical_reference_command(root)


def _replace_option(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"Expected one valued {flag} option")
    command[positions[0] + 1] = value


def assert_cap_only_scientific_change(
    ablation: list[str], canonical: list[str], value: int
) -> None:
    cap = _require_cap(value)
    if len(ablation) != len(canonical) or ablation[:2] != canonical[:2]:
        raise RuntimeError("GT3 ablation changed the Python/core entry point")
    allowed = set()
    for flag in ROUTING_OPTIONS | {SCIENTIFIC_ABLATION_OPTION}:
        positions = [index for index, token in enumerate(canonical) if token == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(canonical):
            raise RuntimeError(f"Canonical command has invalid {flag}")
        allowed.add(positions[0] + 1)
    drift = [
        index
        for index, (left, right) in enumerate(zip(ablation, canonical))
        if left != right and index not in allowed
    ]
    if drift:
        raise RuntimeError(
            "GT3 cap ablation changed frozen token indices "
            + ",".join(str(index) for index in drift)
        )
    options = superseded_v1.coarse_single.command_options(ablation)
    canonical_options = superseded_v1.coarse_single.command_options(canonical)
    required = {
        "--alpha": "12.5",
        "--alpha-mode": "fixed",
        "--budget": "254",
        "--frozen-subproblem": "discrete_b040",
        "--discrete-stop-mode": "r08c10",
        "--discrete-residual-ratio": "0.8",
        "--discrete-history": "7",
        "--history": "7",
        "--reparam-mode": "state_curve",
        "--state-curve-direct-objective-acceptance": "per_image",
        "--stop-complete-iperp": "-1",
    }
    for flag, expected in required.items():
        if options.get(flag) != expected:
            raise RuntimeError(f"GT3 cap ablation changed frozen option {flag}")
    if int(options[SCIENTIFIC_ABLATION_OPTION]) != cap:
        raise RuntimeError("GT3 cap command carries the wrong cap")
    for flag, expected in canonical_options.items():
        if flag in ROUTING_OPTIONS | {SCIENTIFIC_ABLATION_OPTION}:
            continue
        if options.get(flag) != expected:
            raise RuntimeError(f"GT3 cap ablation changed frozen option {flag}")


def single_run_command(root: Path, value: int) -> tuple[list[str], str]:
    cap = _require_cap(value)
    canonical = canonical_reference_command(root)
    command = list(canonical)
    label = run_label(cap)
    _replace_option(command, "--output-dir", f"runs/{MEMBER_POLICY_VERSION}/members")
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


def _validate_result_iperp_only(result: Path, value: int) -> dict[str, object]:
    cap = _require_cap(value)
    with np.load(result, allow_pickle=False) as saved:
        missing = sorted((COMMON_RESULT_KEYS | DISCRETE_RESULT_KEYS).difference(saved.files))
        if missing:
            raise RuntimeError(f"Result is missing: {', '.join(missing)}")
        scalar = legacy.scalar
        if float(scalar(saved, "alpha")) != 12.5 or str(scalar(saved, "alpha_mode")) != "fixed":
            raise RuntimeError("Result did not retain fixed alpha=12.5")
        active = np.asarray(saved["active_alpha"], dtype=float)
        if not active.size or not np.all(active == 12.5):
            raise RuntimeError("Result changed alpha within the trajectory")
        if str(scalar(saved, "reparam_mode")) != "state_curve" or str(
            scalar(saved, "state_curve_direct_objective_acceptance")
        ) != "per_image":
            raise RuntimeError("Result changed state_curve/per_image")
        if float(scalar(saved, "stop_complete_iperp")) != -1.0:
            raise RuntimeError("Result used I_perp early stopping")
        if str(scalar(saved, "discrete_implementation_id")) != superseded_v1.coarse_single.EXPECTED_IMPLEMENTATION_ID:
            raise RuntimeError("Result used the wrong discrete implementation")
        if str(scalar(saved, "frozen_subproblem")) != "discrete_b040":
            raise RuntimeError("Result changed frozen subproblem")
        if str(scalar(saved, "discrete_stop_mode")) != "r08c10":
            raise RuntimeError("Result changed core stop selector")
        if int(scalar(saved, "discrete_inner_max_steps")) != cap:
            raise RuntimeError("Result used the wrong cap")
        if float(scalar(saved, "discrete_residual_ratio")) != 0.8:
            raise RuntimeError("Result changed residual ratio")
        if int(scalar(saved, "discrete_history")) != 7:
            raise RuntimeError("Result changed history")
        if not bool(scalar(saved, "discrete_all_returned_states_evaluated")):
            raise RuntimeError("Result returned an unevaluated state")

        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        mesh_ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        if len({array.shape for array in (cost, i_perp, committed, cv, mesh_ratio)}) != 1 or cost.ndim != 1 or cost.size != 254:
            raise RuntimeError("Result trajectory arrays have invalid shapes")
        if np.any(np.diff(cost) <= 0):
            raise RuntimeError("Optimization NF is not increasing")
        if not np.all(np.isfinite(i_perp)) or not np.all(np.isfinite(cv)) or not np.all(np.isfinite(mesh_ratio)):
            raise RuntimeError("I_perp or mesh is non-finite")
        if np.any(i_perp < 0.0) or np.any(cv < 0.0) or np.any(mesh_ratio < 1.0):
            raise RuntimeError("I_perp or mesh is invalid")

        outer = np.asarray(saved["discrete_audit_outer"], dtype=int)
        steps = np.asarray(saved["discrete_audit_step"], dtype=int)
        selections = np.asarray(saved["discrete_audit_selection"], dtype="U16")
        histories = np.asarray(saved["discrete_audit_history_count"], dtype=int)
        residuals = np.asarray(saved["discrete_audit_global_relative_residual"], dtype=float)
        reasons = np.asarray(saved["discrete_audit_stop_reason"], dtype="U64")
        if len({array.shape for array in (outer, steps, selections, histories)}) != 1 or residuals.shape != reasons.shape:
            raise RuntimeError("Mechanism audit arrays differ")
        if np.any(steps < 0) or np.any(steps >= cap):
            raise RuntimeError("Recorded step exceeds cap")
        for outer_index in np.unique(outer):
            observed = steps[outer == outer_index].tolist()
            if observed != list(range(len(observed))):
                raise RuntimeError("Evaluated steps are not contiguous")
        h0 = int(np.count_nonzero(selections == "H0"))
        lbfgs = int(np.count_nonzero(selections == "LBFGS"))
        if h0 != int(scalar(saved, "discrete_h0_selection_count")) or lbfgs != int(scalar(saved, "discrete_lbfgs_selection_count")) or h0 + lbfgs != selections.size:
            raise RuntimeError("Direction selection audit mismatch")
        if (
            np.any(histories < 0)
            or np.any(histories > 7)
            or not np.all(np.isfinite(residuals))
            or np.any(residuals < 0.0)
        ):
            raise RuntimeError("History/residual audit is invalid")
        complete = committed.copy()
        complete[0] = True
        eligible = complete & (cv <= 0.0065) & (mesh_ratio <= 1.03)
        return {
            "configuration_sha256": str(scalar(saved, "configuration_sha256")),
            "initial_path_sha256": str(scalar(saved, "initial_path_sha256")),
            "trajectory_record_count": int(cost.size),
            "final_optimization_NF": int(cost[-1]),
            "complete_state_NF": cost[complete].tolist(),
            "mesh_eligible_complete_state_NF": cost[eligible].tolist(),
            "mesh_eligible_complete_state_count": int(np.count_nonzero(eligible)),
            "reparam_committed_state_count": int(np.count_nonzero(committed)),
            "frozen_subproblem_solve_count": int(reasons.size),
            "discrete_joint_step_count": int(selections.size),
            "h0_selection_count": h0,
            "lbfgs_selection_count": lbfgs,
            "lbfgs_takeover_observed": bool(lbfgs > 0),
            "maximum_history_count": int(np.max(histories)) if histories.size else 0,
            "relative_residual_stop_count": int(np.count_nonzero(reasons == "evaluated-relative-natural-residual")),
            "inner_cap_stop_count": int(np.count_nonzero(reasons == "evaluated-inner-max-steps")),
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
    existing = sorted(result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget254.npz"))
    if existing or audit_path.exists() or log_dir.exists():
        target = existing[0] if existing else audit_path
        raise FileExistsError(f"Refusing to overwrite GT3 cap member: {target}")
    result_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    wall_seconds = legacy.run_command(command, root, log_dir / "stdout.txt", log_dir / "stderr.txt")
    result = legacy.find_result(result_dir, label, 254)
    summary = _validate_result_iperp_only(result, cap)
    lbfgs_takeover_observed = bool(summary["lbfgs_takeover_observed"])
    audit: dict[str, object] = {
        "member_policy_version": MEMBER_POLICY_VERSION,
        "member_role": "new_gt3_max_evaluated_steps_candidate",
        "alpha": 12.5,
        "residual_ratio": 0.8,
        "max_evaluated_steps": cap,
        "cap_tag": cap_tag(cap),
        "budget_per_image": 254,
        "replicate": "r1",
        "lbfgs_history": 7,
        "core_stop_mode_selector": "r08c10",
        "stop_rule": f"R/R0 <= 0.8 OR evaluated accepted steps == {cap}",
        "scientific_change_from_frozen_cap10": "only --discrete-inner-max-steps",
        "routing_changes_from_frozen_cap10": ["--output-dir", "--run-label"],
        "scientific_command_sha256": compact_command_sha256(command),
        "cap_is_primary_lock_eligible": True,
        "primary_requires_lbfgs_takeover": True,
        "primary_mechanism_requirement_met": lbfgs_takeover_observed,
        "primary_lock_eligible_after_mechanism_gate": lbfgs_takeover_observed,
        "single_run_from_canonical_common_path": True,
        "reparameterization_policy": "state_curve/per_image from first state",
        "iperp_only_scientific_analysis": True,
        "barrier_fields_accessed": False,
        "barrier_computation_or_densification_performed_by_wrapper": False,
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
