#!/usr/bin/env python3
"""Isolated alpha-only refinement member for the frozen Ala v2 protocol.

This wrapper changes only routing (run label and output directories) relative
to the frozen coarse single-run wrapper.  The fixed alpha, direct discrete
candidate, matched legacy control, R08C10 candidate stop, N254 budget, and
``state_curve/per_image`` outer policy are otherwise byte-for-byte command
equivalent.  Result auditing reads I_perp, optimization cost, completeness,
mesh diagnostics, and protocol metadata only; energy-barrier arrays remain
embargoed and are never indexed or propagated.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import run_stage4_ala_discrete_single_fixedalpha as coarse_single
import run_stage4_uniform_optimized as legacy


MEMBER_POLICY_VERSION = (
    "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1"
)
REFINEMENT_ALPHAS = (5.0, 7.5, 12.5, 15.0)
CANONICAL_BUDGET = 254
CANONICAL_REPLICATE = "r1"
MEMBER_ROLES = ("candidate", "matched_control")
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
ROUTING_OPTIONS = frozenset({"--output-dir", "--run-label"})
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
CANDIDATE_RESULT_KEYS = frozenset(
    {
        "discrete_all_returned_states_evaluated",
        "discrete_history",
        "discrete_implementation_id",
        "discrete_inner_max_steps",
        "discrete_residual_ratio",
        "discrete_stop_mode",
        "frozen_subproblem",
    }
)
CONTROL_FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "ala_dsel_implementation_id",
        "ala_dsel_audit_selection",
        "ala_dsel_first_activation_nf",
        "discrete_implementation_id",
        "discrete_audit_selection",
    }
)


def _alpha_text(alpha: float) -> str:
    return f"{float(alpha):g}"


def _require_registered_member(
    alpha: float, budget: int, replicate: str, member_role: str
) -> None:
    if float(alpha) not in REFINEMENT_ALPHAS:
        raise ValueError(
            "Refinement alpha must be one of "
            + ",".join(_alpha_text(value) for value in REFINEMENT_ALPHAS)
        )
    if int(budget) != CANONICAL_BUDGET:
        raise ValueError(f"Refinement budget must equal {CANONICAL_BUDGET}")
    if replicate != CANONICAL_REPLICATE:
        raise ValueError(f"Refinement replicate must equal {CANONICAL_REPLICATE}")
    if member_role not in MEMBER_ROLES:
        raise ValueError(f"Unknown member role: {member_role}")


def run_label(alpha: float, member_role: str) -> str:
    _require_registered_member(
        alpha, CANONICAL_BUDGET, CANONICAL_REPLICATE, member_role
    )
    solver = "discrete_b040" if member_role == "candidate" else "legacy"
    stop = "r08c10" if member_role == "candidate" else "legacy_stop"
    return (
        f"{MEMBER_POLICY_VERSION}_{stop}_a{legacy.label_number(alpha)}_"
        f"N{CANONICAL_BUDGET}_{solver}_{CANONICAL_REPLICATE}"
    )


def _replace_option(command: list[str], flag: str, value: str) -> None:
    occurrences = [index for index, token in enumerate(command) if token == flag]
    if len(occurrences) != 1:
        raise RuntimeError(f"Expected exactly one {flag} option")
    index = occurrences[0]
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise RuntimeError(f"Option {flag} has no value")
    command[index + 1] = value


def assert_routing_only_change(
    refinement: list[str], canonical: list[str], alpha: float, member_role: str
) -> None:
    coarse_single.assert_single_run_policy(refinement, alpha, member_role)
    if refinement[:2] != canonical[:2]:
        raise RuntimeError("Refinement changed the Python/core entry point")
    refinement_options = coarse_single.command_options(refinement)
    canonical_options = coarse_single.command_options(canonical)
    for flag in ROUTING_OPTIONS:
        refinement_options.pop(flag, None)
        canonical_options.pop(flag, None)
    if refinement_options != canonical_options:
        differing = sorted(set(refinement_options) | set(canonical_options))
        raise RuntimeError(
            "Refinement drifted from the frozen coarse scientific command: "
            + ", ".join(
                flag
                for flag in differing
                if refinement_options.get(flag) != canonical_options.get(flag)
            )
        )


def single_run_command(
    root: Path,
    alpha: float,
    budget: int,
    replicate: str,
    member_role: str,
) -> tuple[list[str], str]:
    _require_registered_member(alpha, budget, replicate, member_role)
    canonical, _ = coarse_single.single_run_command(
        root, alpha, budget, replicate, member_role
    )
    command = list(canonical)
    label = run_label(alpha, member_role)
    _replace_option(
        command,
        "--output-dir",
        f"runs/{MEMBER_POLICY_VERSION}/members",
    )
    _replace_option(command, "--run-label", label)
    assert_routing_only_change(command, canonical, alpha, member_role)
    return command, label


def assert_matched_pair_commands(
    candidate: list[str], control: list[str], alpha: float
) -> None:
    coarse_single.assert_matched_pair_commands(candidate, control, alpha)
    if coarse_single.command_value(candidate, "--output-dir") != (
        f"runs/{MEMBER_POLICY_VERSION}/members"
    ):
        raise RuntimeError("Candidate uses the wrong refinement output directory")
    if coarse_single.command_value(control, "--output-dir") != (
        f"runs/{MEMBER_POLICY_VERSION}/members"
    ):
        raise RuntimeError("Control uses the wrong refinement output directory")


def member_paths(
    root: Path, alpha: float, member_role: str
) -> dict[str, Path | str]:
    label = run_label(alpha, member_role)
    return {
        "label": label,
        "result_dir": root / "runs" / MEMBER_POLICY_VERSION / "members",
        "audit": (
            root
            / "runs"
            / MEMBER_POLICY_VERSION
            / "audits"
            / f"{label}.json"
        ),
        "member_log": root / "logs" / MEMBER_POLICY_VERSION / "members" / label,
        "launcher_log": root
        / "logs"
        / MEMBER_POLICY_VERSION
        / "launcher"
        / label,
    }


def _require_keys(saved: np.lib.npyio.NpzFile, required, result: Path) -> None:
    missing = sorted(set(required).difference(saved.files))
    if missing:
        raise RuntimeError(f"{result} is missing: {', '.join(missing)}")


def _validate_result_iperp_only(
    result: Path, alpha: float, member_role: str
) -> dict[str, object]:
    with np.load(result, allow_pickle=False) as saved:
        _require_keys(saved, COMMON_RESULT_KEYS, result)
        if float(legacy.scalar(saved, "alpha")) != float(alpha):
            raise RuntimeError("Result did not retain the registered alpha")
        if str(legacy.scalar(saved, "alpha_mode")) != "fixed":
            raise RuntimeError("Result alpha mode is not fixed")
        active_alpha = np.asarray(saved["active_alpha"], dtype=float)
        if not active_alpha.size or not np.all(active_alpha == float(alpha)):
            raise RuntimeError("Result changed alpha within the trajectory")
        if str(legacy.scalar(saved, "reparam_mode")) != "state_curve":
            raise RuntimeError("Result used the wrong reparameterization mode")
        if str(
            legacy.scalar(saved, "state_curve_direct_objective_acceptance")
        ) != "per_image":
            raise RuntimeError("Result used the wrong state-curve acceptance")
        if float(legacy.scalar(saved, "stop_complete_iperp")) != -1.0:
            raise RuntimeError("Result used an I_perp early-stop threshold")

        cost = np.asarray(
            saved["total_internal_joint_evaluations"], dtype=int
        )
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        shapes = {array.shape for array in (cost, i_perp, committed, cv, ratio)}
        if len(shapes) != 1 or cost.ndim != 1 or cost.size != CANONICAL_BUDGET:
            raise RuntimeError("Result trajectory arrays have invalid shapes")
        if np.any(np.diff(cost) <= 0):
            raise RuntimeError("Optimization NF is not strictly increasing")
        if not (
            np.all(np.isfinite(i_perp))
            and np.all(np.isfinite(cv))
            and np.all(np.isfinite(ratio))
        ):
            raise RuntimeError("I_perp or mesh diagnostics are non-finite")
        if np.any(i_perp < 0.0) or np.any(cv < 0.0) or np.any(ratio < 1.0):
            raise RuntimeError("I_perp or mesh diagnostics are invalid")

        if member_role == "candidate":
            _require_keys(saved, CANDIDATE_RESULT_KEYS, result)
            if str(legacy.scalar(saved, "discrete_implementation_id")) != (
                coarse_single.EXPECTED_IMPLEMENTATION_ID
            ):
                raise RuntimeError("Candidate used the wrong implementation")
            if str(legacy.scalar(saved, "frozen_subproblem")) != "discrete_b040":
                raise RuntimeError("Candidate used the wrong frozen subproblem")
            if str(legacy.scalar(saved, "discrete_stop_mode")) != "r08c10":
                raise RuntimeError("Candidate used the wrong stop mode")
            if int(legacy.scalar(saved, "discrete_inner_max_steps")) != 10:
                raise RuntimeError("Candidate used the wrong evaluated-step cap")
            if float(legacy.scalar(saved, "discrete_residual_ratio")) != 0.8:
                raise RuntimeError("Candidate used the wrong residual ratio")
            if int(legacy.scalar(saved, "discrete_history")) != 7:
                raise RuntimeError("Candidate used the wrong L-BFGS history")
            if not bool(
                legacy.scalar(saved, "discrete_all_returned_states_evaluated")
            ):
                raise RuntimeError("Candidate returned an unevaluated state")
        elif member_role == "matched_control":
            leaked = CONTROL_FORBIDDEN_RESULT_KEYS.intersection(saved.files)
            if leaked:
                raise RuntimeError(
                    "Control contains discrete/DSEL outputs: "
                    + ", ".join(sorted(leaked))
                )
        else:
            raise RuntimeError(f"Unknown member role: {member_role}")

        complete = committed.copy()
        complete[0] = True
        mesh = (cv <= MESH_CV_MAX) & (ratio <= MESH_RATIO_MAX)
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
            "result_fields_accessed": sorted(
                COMMON_RESULT_KEYS
                | (CANDIDATE_RESULT_KEYS if member_role == "candidate" else set())
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--budget", type=int, default=CANONICAL_BUDGET)
    parser.add_argument("--member-role", choices=MEMBER_ROLES, required=True)
    parser.add_argument("--replicate", default=CANONICAL_REPLICATE)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    alpha = float(args.alpha)
    _require_registered_member(
        alpha, args.budget, args.replicate, args.member_role
    )
    command, label = single_run_command(
        root, alpha, args.budget, args.replicate, args.member_role
    )
    paths = member_paths(root, alpha, args.member_role)
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
        raise FileExistsError(f"Refusing to overwrite refinement member: {target}")
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
    summary = _validate_result_iperp_only(result, alpha, args.member_role)
    audit: dict[str, object] = {
        "member_policy_version": MEMBER_POLICY_VERSION,
        "member_role": args.member_role,
        "alpha": alpha,
        "alpha_mode": "fixed",
        "fixed_alpha_throughout": True,
        "budget_per_image": CANONICAL_BUDGET,
        "replicate": CANONICAL_REPLICATE,
        "scientific_command_equal_to_frozen_coarse_v2": True,
        "routing_only_isolated": True,
        "single_run_from_canonical_common_path": True,
        "candidate_stop_rule": (
            "R/R0 <= 0.8 OR evaluated accepted steps == 10"
            if args.member_role == "candidate"
            else "not_applicable_legacy_control"
        ),
        "reparameterization_policy": "state_curve/per_image from first state",
        "iperp_only_scientific_analysis": True,
        "barrier_fields_accessed": False,
        "barrier_computation_or_densification_performed_by_wrapper": False,
        "barrier_embargo": "all energy-barrier fields ignored until alpha lock",
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
