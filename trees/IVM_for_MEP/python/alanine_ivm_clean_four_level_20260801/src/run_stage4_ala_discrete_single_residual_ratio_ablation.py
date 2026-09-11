#!/usr/bin/env python3
"""One member of the preregistered Ala residual-ratio ablation.

This wrapper is intentionally a one-factor derivative of the completed
alpha=12.5 refinement candidate.  The only scientific command value that may
change is ``--discrete-residual-ratio``.  Output routing changes as well so the
new members cannot overwrite the frozen alpha-refinement data.  In
particular, alpha=12.5, N254, the ten evaluated-step cap, m7, and the complete
``state_curve/per_image`` outer policy remain byte-for-byte fixed.

The core's historical stop-mode selector is still named ``r08c10``.  Here it
means the generic rule ``R/R0 <= registered ratio OR evaluated steps == 10``;
the actual ratio is carried by ``--discrete-residual-ratio`` and by the
unambiguous member tags ``rr090c10`` and ``rr050c10``.

Result auditing is I_perp/mesh/mechanism only.  No energy-barrier array is
indexed, propagated, or ranked.
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
    "stage4_ala_discrete_single_residual_ratio_ablation_v1"
)
CANONICAL_ALPHA = 12.5
CANONICAL_BUDGET = 254
CANONICAL_REPLICATE = "r1"
REFERENCE_RATIO = 0.8
NEW_RATIOS = (0.9, 0.5)
RATIO_TAGS = {0.9: "rr090c10", 0.5: "rr050c10"}
CORE_STOP_MODE_SELECTOR = "r08c10"
EVALUATED_STEP_CAP = 10
LBFGS_HISTORY = 7
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
ROUTING_OPTIONS = frozenset({"--output-dir", "--run-label"})
SCIENTIFIC_ABLATION_OPTION = "--discrete-residual-ratio"

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
        "discrete_audit_selection",
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


def _require_ratio(value: float) -> float:
    ratio = float(value)
    if ratio not in NEW_RATIOS:
        raise ValueError(
            "Residual ratio must be one of "
            + ",".join(f"{item:g}" for item in NEW_RATIOS)
        )
    return ratio


def ratio_tag(value: float) -> str:
    return RATIO_TAGS[_require_ratio(value)]


def run_label(value: float) -> str:
    return (
        f"{MEMBER_POLICY_VERSION}_{ratio_tag(value)}_a12p5_"
        f"N{CANONICAL_BUDGET}_discrete_b040_{CANONICAL_REPLICATE}"
    )


def _replace_option(command: list[str], flag: str, value: str) -> None:
    occurrences = [index for index, token in enumerate(command) if token == flag]
    if len(occurrences) != 1:
        raise RuntimeError(f"Expected exactly one {flag} option")
    index = occurrences[0]
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise RuntimeError(f"Option {flag} has no value")
    command[index + 1] = value


def canonical_reference_command(root: Path) -> list[str]:
    command, _ = frozen_refinement.single_run_command(
        root,
        CANONICAL_ALPHA,
        CANONICAL_BUDGET,
        CANONICAL_REPLICATE,
        "candidate",
    )
    return command


def assert_ratio_only_scientific_change(
    ablation: list[str], canonical: list[str], value: float
) -> None:
    ratio = _require_ratio(value)
    if len(ablation) != len(canonical) or ablation[:2] != canonical[:2]:
        raise RuntimeError("Ablation changed the Python/core entry point")
    allowed_value_indices = set()
    for flag in ROUTING_OPTIONS | {SCIENTIFIC_ABLATION_OPTION}:
        occurrences = [index for index, token in enumerate(canonical) if token == flag]
        if len(occurrences) != 1 or occurrences[0] + 1 >= len(canonical):
            raise RuntimeError(f"Canonical command has invalid {flag}")
        allowed_value_indices.add(occurrences[0] + 1)
    drift = [
        index
        for index, (left, right) in enumerate(zip(ablation, canonical))
        if left != right and index not in allowed_value_indices
    ]
    if drift:
        raise RuntimeError(
            "Residual-ratio ablation changed a frozen command token at "
            + ",".join(str(index) for index in drift)
        )

    options = coarse_single.command_options(ablation)
    canonical_options = coarse_single.command_options(canonical)
    if float(options["--alpha"]) != CANONICAL_ALPHA:
        raise RuntimeError("Ablation alpha is not fixed at 12.5")
    if int(options["--budget"]) != CANONICAL_BUDGET:
        raise RuntimeError("Ablation budget is not N254")
    if options["--frozen-subproblem"] != "discrete_b040":
        raise RuntimeError("Ablation changed the frozen subproblem")
    if options["--discrete-stop-mode"] != CORE_STOP_MODE_SELECTOR:
        raise RuntimeError("Ablation changed the core stop-mode selector")
    if int(options["--discrete-inner-max-steps"]) != EVALUATED_STEP_CAP:
        raise RuntimeError("Ablation changed the evaluated-step cap")
    if int(options["--discrete-history"]) != LBFGS_HISTORY:
        raise RuntimeError("Ablation changed discrete L-BFGS history")
    if int(options["--history"]) != LBFGS_HISTORY:
        raise RuntimeError("Ablation changed outer L-BFGS history")
    if float(options[SCIENTIFIC_ABLATION_OPTION]) != ratio:
        raise RuntimeError("Ablation command carries the wrong residual ratio")
    if options["--reparam-mode"] != "state_curve":
        raise RuntimeError("Ablation changed state_curve reparameterization")
    if options["--state-curve-direct-objective-acceptance"] != "per_image":
        raise RuntimeError("Ablation changed per-image state acceptance")
    if float(options["--stop-complete-iperp"]) != -1.0:
        raise RuntimeError("Ablation introduced an I_perp early stop")
    for flag, canonical_value in canonical_options.items():
        if flag in ROUTING_OPTIONS | {SCIENTIFIC_ABLATION_OPTION}:
            continue
        if options.get(flag) != canonical_value:
            raise RuntimeError(f"Ablation changed frozen option {flag}")


def single_run_command(root: Path, value: float) -> tuple[list[str], str]:
    ratio = _require_ratio(value)
    canonical = canonical_reference_command(root)
    command = list(canonical)
    label = run_label(ratio)
    _replace_option(
        command,
        "--output-dir",
        f"runs/{MEMBER_POLICY_VERSION}/members",
    )
    _replace_option(command, "--run-label", label)
    _replace_option(command, SCIENTIFIC_ABLATION_OPTION, f"{ratio:g}")
    assert_ratio_only_scientific_change(command, canonical, ratio)
    return command, label


def compact_command_sha256(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def member_paths(root: Path, value: float) -> dict[str, Path | str]:
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


def _validate_result_iperp_only(result: Path, value: float) -> dict[str, object]:
    ratio_value = _require_ratio(value)
    with np.load(result, allow_pickle=False) as saved:
        _require_keys(saved, COMMON_RESULT_KEYS | DISCRETE_RESULT_KEYS, result)
        if float(legacy.scalar(saved, "alpha")) != CANONICAL_ALPHA:
            raise RuntimeError("Result did not retain alpha=12.5")
        if str(legacy.scalar(saved, "alpha_mode")) != "fixed":
            raise RuntimeError("Result alpha mode is not fixed")
        active_alpha = np.asarray(saved["active_alpha"], dtype=float)
        if not active_alpha.size or not np.all(active_alpha == CANONICAL_ALPHA):
            raise RuntimeError("Result changed alpha within the trajectory")
        if str(legacy.scalar(saved, "reparam_mode")) != "state_curve":
            raise RuntimeError("Result used the wrong reparameterization mode")
        if str(
            legacy.scalar(saved, "state_curve_direct_objective_acceptance")
        ) != "per_image":
            raise RuntimeError("Result used the wrong state-curve acceptance")
        if float(legacy.scalar(saved, "stop_complete_iperp")) != -1.0:
            raise RuntimeError("Result used an I_perp early-stop threshold")
        if str(legacy.scalar(saved, "discrete_implementation_id")) != (
            coarse_single.EXPECTED_IMPLEMENTATION_ID
        ):
            raise RuntimeError("Result used the wrong discrete implementation")
        if str(legacy.scalar(saved, "frozen_subproblem")) != "discrete_b040":
            raise RuntimeError("Result used the wrong frozen subproblem")
        if str(legacy.scalar(saved, "discrete_stop_mode")) != (
            CORE_STOP_MODE_SELECTOR
        ):
            raise RuntimeError("Result used the wrong core stop-mode selector")
        if int(legacy.scalar(saved, "discrete_inner_max_steps")) != (
            EVALUATED_STEP_CAP
        ):
            raise RuntimeError("Result used the wrong evaluated-step cap")
        if float(legacy.scalar(saved, "discrete_residual_ratio")) != ratio_value:
            raise RuntimeError("Result used the wrong residual ratio")
        if int(legacy.scalar(saved, "discrete_history")) != LBFGS_HISTORY:
            raise RuntimeError("Result used the wrong L-BFGS history")
        if not bool(
            legacy.scalar(saved, "discrete_all_returned_states_evaluated")
        ):
            raise RuntimeError("Result returned an unevaluated state")

        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        mesh_ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        shapes = {
            array.shape for array in (cost, i_perp, committed, cv, mesh_ratio)
        }
        if (
            len(shapes) != 1
            or cost.ndim != 1
            or cost.size != CANONICAL_BUDGET
        ):
            raise RuntimeError("Result trajectory arrays have invalid shapes")
        if np.any(np.diff(cost) <= 0):
            raise RuntimeError("Optimization NF is not strictly increasing")
        if not (
            np.all(np.isfinite(i_perp))
            and np.all(np.isfinite(cv))
            and np.all(np.isfinite(mesh_ratio))
        ):
            raise RuntimeError("I_perp or mesh diagnostics are non-finite")
        if (
            np.any(i_perp < 0.0)
            or np.any(cv < 0.0)
            or np.any(mesh_ratio < 1.0)
        ):
            raise RuntimeError("I_perp or mesh diagnostics are invalid")

        selections = np.asarray(saved["discrete_audit_selection"], dtype="U16")
        histories = np.asarray(saved["discrete_audit_history_count"], dtype=int)
        residuals = np.asarray(
            saved["discrete_audit_global_relative_residual"], dtype=float
        )
        stop_reasons = np.asarray(
            saved["discrete_audit_stop_reason"], dtype="U64"
        )
        if histories.shape != selections.shape:
            raise RuntimeError("Direction/history audit lengths differ")
        if residuals.shape != stop_reasons.shape:
            raise RuntimeError("Residual/stop audit lengths differ")
        if np.any(histories < 0) or np.any(histories > LBFGS_HISTORY):
            raise RuntimeError("L-BFGS audit history is invalid")
        if not np.all(np.isfinite(residuals)) or np.any(residuals < 0.0):
            raise RuntimeError("Residual-ratio audit is invalid")
        h0_count = int(np.count_nonzero(selections == "H0"))
        lbfgs_count = int(np.count_nonzero(selections == "LBFGS"))
        if h0_count != int(legacy.scalar(saved, "discrete_h0_selection_count")):
            raise RuntimeError("H0 selection count is inconsistent")
        if lbfgs_count != int(
            legacy.scalar(saved, "discrete_lbfgs_selection_count")
        ):
            raise RuntimeError("L-BFGS selection count is inconsistent")
        if h0_count + lbfgs_count != selections.size:
            raise RuntimeError("Unknown discrete direction selection")

        complete = committed.copy()
        complete[0] = True
        mesh = (cv <= MESH_CV_MAX) & (mesh_ratio <= MESH_RATIO_MAX)
        eligible = complete & mesh
        return {
            "configuration_sha256": str(
                legacy.scalar(saved, "configuration_sha256")
            ),
            "initial_path_sha256": str(
                legacy.scalar(saved, "initial_path_sha256")
            ),
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
            "maximum_history_count": (
                int(np.max(histories)) if histories.size else 0
            ),
            "relative_residual_stop_count": int(
                np.count_nonzero(
                    stop_reasons == "evaluated-relative-natural-residual"
                )
            ),
            "inner_cap_stop_count": int(
                np.count_nonzero(stop_reasons == "evaluated-inner-max-steps")
            ),
            "result_fields_accessed": sorted(
                COMMON_RESULT_KEYS | DISCRETE_RESULT_KEYS
            ),
        }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--residual-ratio",
        type=float,
        choices=NEW_RATIOS,
        required=True,
    )
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    root = args.root.expanduser().resolve()
    ratio_value = _require_ratio(args.residual_ratio)
    command, label = single_run_command(root, ratio_value)
    paths = member_paths(root, ratio_value)
    result_dir = paths["result_dir"]
    audit_path = paths["audit"]
    log_dir = paths["member_log"]
    assert isinstance(result_dir, Path)
    assert isinstance(audit_path, Path)
    assert isinstance(log_dir, Path)
    existing = sorted(
        result_dir.glob(
            f"ivm_{label}_central_state_ivm_cfg*_budget{CANONICAL_BUDGET}.npz"
        )
    )
    if existing or audit_path.exists() or log_dir.exists():
        target = existing[0] if existing else audit_path
        raise FileExistsError(f"Refusing to overwrite ratio member: {target}")
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
    summary = _validate_result_iperp_only(result, ratio_value)
    audit: dict[str, object] = {
        "member_policy_version": MEMBER_POLICY_VERSION,
        "member_role": "new_residual_ratio_candidate",
        "alpha": CANONICAL_ALPHA,
        "alpha_mode": "fixed",
        "fixed_alpha_throughout": True,
        "residual_ratio": ratio_value,
        "ratio_tag": ratio_tag(ratio_value),
        "budget_per_image": CANONICAL_BUDGET,
        "replicate": CANONICAL_REPLICATE,
        "evaluated_step_cap": EVALUATED_STEP_CAP,
        "lbfgs_history": LBFGS_HISTORY,
        "core_stop_mode_selector": CORE_STOP_MODE_SELECTOR,
        "core_stop_mode_selector_note": (
            "legacy API name; actual scientific ratio is discrete_residual_ratio"
        ),
        "stop_rule": (
            f"R/R0 <= {ratio_value:g} OR evaluated accepted steps == "
            f"{EVALUATED_STEP_CAP}"
        ),
        "scientific_change_from_frozen_alpha12p5_r08": (
            "only --discrete-residual-ratio"
        ),
        "routing_changes_from_frozen_alpha12p5_r08": [
            "--output-dir",
            "--run-label",
        ],
        "scientific_command_sha256": compact_command_sha256(command),
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
