#!/usr/bin/env python3
"""Isolated N30 single-run reparameterization pilot for discrete Ala IVM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import run_stage4_ala_dsel_single_fixedalpha as single
import run_stage4_uniform_optimized as legacy


POLICY_VERSION = "stage4_ala_discrete_single_policy_pilot_v1"
PILOT_ALPHA = 20.0
PILOT_BUDGET = 30
POLICIES = ("precision_state_curve", "projected_pathsum")
SOLVER_ROLES = ("discrete_b040", "ala_dsel", "legacy")
EXPECTED_DISCRETE_IMPLEMENTATION = "ala-frozen-discrete-ffrbk-dsel-m7-evaluated-v1"
EXPECTED_ALA_DSEL_IMPLEMENTATION = (
    "ala-legacy-ivm-fixedalpha-triggered-ffrbk-vs-h0-lbfgs-"
    "postgeometry-dsel-v3"
)
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
DSEL_OPTIONS = frozenset(
    {
        "--frozen-subproblem",
        "--discrete-inner-max-steps",
        "--discrete-stop-mode",
        "--discrete-residual-ratio",
        "--discrete-history",
        "--ala-dsel-trigger-iperp",
    }
)


def _policy_options(policy: str) -> list[str]:
    if policy == "precision_state_curve":
        return [
            "--state-globalization", "iperp_monotone",
            "--state-trust-fraction", "2",
            "--state-polish-trust-fraction", "2",
            "--state-polish-alpha-factor", "0.5",
            "--inner-calls", "15",
            "--plbfgs-kkt-rtol", "0.8",
            "--reparam-mode", "state_curve",
            "--reparam-every", "1",
            "--state-curve-direct-solve",
            "--state-curve-direct-trial-calls", "15",
            "--state-curve-direct-objective-acceptance", "per_image",
            "--state-curve-immediate-block-relock",
            "--state-curve-stall-free-polish",
            "--state-curve-stall-free-polish-count", "1",
            "--reparam-ready-relative-iperp", "2",
            "--reparam-cv-trigger", "0.001",
            "--reparam-strict-cv-trigger", "0.001",
            "--reparam-retry-reduction", "0.8",
            "--reparam-recovery-calls", "320",
            "--reparam-recovery-inner-calls", "15",
            "--reparam-recovery-merge", "bicriteria",
            "--reparam-recovery-iperp-rtol", "0",
            "--reparam-equidistance-contraction", "0.2",
            "--stop-complete-iperp", "-1",
        ]
    if policy == "projected_pathsum":
        return [
            "--state-globalization", "iperp_monotone",
            "--state-trust-fraction", "1.5",
            "--state-polish-trust-fraction", "2",
            "--state-polish-alpha-factor", "0.5",
            "--state-strict-polish-fmax", "0.005",
            "--inner-calls", "15",
            "--plbfgs-kkt-rtol", "0.96",
            "--state-kkt-stall-tighten-target", "0.95",
            "--reparam-mode", "state_curve_projected",
            "--reparam-every", "1",
            "--state-curve-direct-solve",
            "--state-curve-direct-trial-calls", "15",
            "--state-curve-direct-objective-acceptance", "path_sum",
            "--state-curve-immediate-block-relock",
            "--state-curve-stall-free-polish",
            "--state-curve-stall-free-polish-count", "1",
            "--reparam-ready-relative-iperp", "2",
            "--reparam-cv-trigger", "-1",
            "--reparam-strict-cv-trigger", "0.15",
            "--reparam-recovery-calls", "30",
            "--reparam-recovery-inner-calls", "10",
            "--reparam-recovery-merge", "collective",
            "--reparam-recovery-iperp-rtol", "0",
            "--reparam-equidistance-contraction", "0.75",
            "--stop-complete-iperp", "-1",
        ]
    raise ValueError(f"Unknown policy: {policy}")


def _solver_options(solver_role: str) -> list[str]:
    if solver_role == "discrete_b040":
        return [
            "--frozen-subproblem", "discrete_b040",
            "--discrete-inner-max-steps", "10",
            "--discrete-stop-mode", "r08c10",
            "--discrete-residual-ratio", "0.8",
            "--discrete-history", "7",
        ]
    if solver_role == "ala_dsel":
        return [
            "--frozen-subproblem", "ala_dsel",
            "--discrete-inner-max-steps", "10",
            "--discrete-stop-mode", "r08c10",
            "--discrete-residual-ratio", "0.8",
            "--discrete-history", "7",
            "--ala-dsel-trigger-iperp", "inf",
        ]
    if solver_role == "legacy":
        return ["--frozen-subproblem", "legacy"]
    raise ValueError(f"Unknown solver role: {solver_role}")


def pilot_label(policy: str, solver_role: str, replicate: str) -> str:
    return (
        f"{POLICY_VERSION}_{policy}_{solver_role}_a20_"
        f"N{PILOT_BUDGET}_{replicate}"
    )


def pilot_command(
    root: Path,
    policy: str,
    solver_role: str,
    replicate: str,
    budget: int = PILOT_BUDGET,
) -> tuple[list[str], str]:
    label = (
        pilot_label(policy, solver_role, replicate)
        if budget == PILOT_BUDGET
        else f"{POLICY_VERSION}_{policy}_{solver_role}_a20_N{budget}_{replicate}"
    )
    command = legacy.runner_base(
        root,
        single.COMMON_INITIAL,
        f"runs/{POLICY_VERSION}/members",
        label,
        PILOT_ALPHA,
        budget,
    )
    command += _policy_options(policy)
    command += _solver_options(solver_role)
    assert_pilot_command(command, policy, solver_role, budget)
    return command, label


def assert_pilot_command(
    command: list[str], policy: str, solver_role: str, budget: int
) -> None:
    options = single.command_options(command)
    if float(options["--alpha"]) != PILOT_ALPHA:
        raise RuntimeError("Pilot alpha drift")
    if options["--alpha-mode"] != "fixed":
        raise RuntimeError("Pilot alpha is not fixed")
    if options["--initial-path-file"] != single.COMMON_INITIAL:
        raise RuntimeError("Pilot initial path drift")
    if int(options["--budget"]) != budget:
        raise RuntimeError("Pilot budget drift")
    expected_mode = (
        "state_curve" if policy == "precision_state_curve"
        else "state_curve_projected"
    )
    expected_acceptance = (
        "per_image" if policy == "precision_state_curve" else "path_sum"
    )
    if options["--reparam-mode"] != expected_mode:
        raise RuntimeError("Pilot reparameterization mode drift")
    if options["--state-curve-direct-objective-acceptance"] != expected_acceptance:
        raise RuntimeError("Pilot reparameterization acceptance drift")
    if options["--frozen-subproblem"] != solver_role:
        raise RuntimeError("Pilot solver role drift")
    if solver_role != "legacy":
        if options["--discrete-stop-mode"] != "r08c10":
            raise RuntimeError("Pilot stop rule drift")
        if int(options["--discrete-inner-max-steps"]) != 10:
            raise RuntimeError("Pilot evaluated-step cap drift")
        if float(options["--discrete-residual-ratio"]) != 0.8:
            raise RuntimeError("Pilot residual target drift")
    if "--ala-dsel-alpha" in options or "--alpha-schedule" in options:
        raise RuntimeError("Pilot contains an alpha switch")


def assert_policy_triple(
    commands: dict[str, list[str]], policy: str, budget: int
) -> None:
    for role in SOLVER_ROLES:
        assert_pilot_command(commands[role], policy, role, budget)
    stripped: dict[str, dict[str, str | bool]] = {}
    for role, command in commands.items():
        options = single.command_options(command)
        for flag in DSEL_OPTIONS | {"--run-label"}:
            options.pop(flag, None)
        stripped[role] = options
    if not (
        stripped["discrete_b040"]
        == stripped["ala_dsel"]
        == stripped["legacy"]
    ):
        raise RuntimeError("Pilot triple differs outside solver package")


def complete_state_mask(policy: str, committed: np.ndarray) -> np.ndarray:
    committed = np.asarray(committed, dtype=bool)
    if committed.ndim != 1 or not committed.size:
        raise RuntimeError("Invalid pilot complete-state array")
    if policy == "projected_pathsum":
        # Every published state has already passed the zero-call projected
        # curve construction; reparam_committed is not its completeness flag.
        return np.ones_like(committed, dtype=bool)
    if policy == "precision_state_curve":
        complete = committed.copy()
        complete[0] = True
        return complete
    raise ValueError(f"Unknown policy: {policy}")


def _validate_result(
    result: Path, policy: str, solver_role: str, budget: int
) -> dict[str, object]:
    with np.load(result, allow_pickle=False) as saved:
        if float(legacy.scalar(saved, "alpha")) != PILOT_ALPHA:
            raise RuntimeError("Pilot result alpha drift")
        if str(legacy.scalar(saved, "alpha_mode")) != "fixed":
            raise RuntimeError("Pilot result alpha mode drift")
        expected_mode = (
            "state_curve" if policy == "precision_state_curve"
            else "state_curve_projected"
        )
        if str(legacy.scalar(saved, "reparam_mode")) != expected_mode:
            raise RuntimeError("Pilot result policy drift")
        if solver_role == "discrete_b040":
            if str(legacy.scalar(saved, "frozen_subproblem")) != solver_role:
                raise RuntimeError("Standalone discrete result role drift")
            if str(legacy.scalar(saved, "discrete_implementation_id")) != (
                EXPECTED_DISCRETE_IMPLEMENTATION
            ):
                raise RuntimeError("Standalone discrete implementation drift")
            if not bool(
                legacy.scalar(saved, "discrete_all_returned_states_evaluated")
            ):
                raise RuntimeError("Standalone discrete result is incomplete")
            selections = np.asarray(
                saved["discrete_audit_selection"], dtype="U16"
            )
        elif solver_role == "ala_dsel":
            if str(legacy.scalar(saved, "frozen_subproblem")) != solver_role:
                raise RuntimeError("Ala DSEL result role drift")
            if str(legacy.scalar(saved, "ala_dsel_implementation_id")) != (
                EXPECTED_ALA_DSEL_IMPLEMENTATION
            ):
                raise RuntimeError("Ala DSEL implementation drift")
            selections = np.asarray(
                saved["ala_dsel_audit_selection"], dtype="U16"
            )
        elif solver_role == "legacy":
            if "frozen_subproblem" in saved.files:
                raise RuntimeError("Legacy result contains extension output")
            selections = np.asarray([], dtype="U16")
        else:
            raise RuntimeError(f"Unknown solver role: {solver_role}")

        costs = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        complete = complete_state_mask(policy, committed)
        mesh = (cv <= MESH_CV_MAX) & (ratio <= MESH_RATIO_MAX)
        eligible = complete & mesh
        complete_rows = [
            {
                "NF": int(costs[index]),
                "i_perp": float(i_perp[index]),
                "segment_length_cv": float(cv[index]),
                "segment_length_ratio": float(ratio[index]),
                "mesh_eligible": bool(mesh[index]),
            }
            for index in np.flatnonzero(complete)
        ]
        eligible_indices = np.flatnonzero(eligible)
        if eligible_indices.size:
            best_index = int(
                eligible_indices[np.argmin(i_perp[eligible_indices])]
            )
            best_iperp: float | None = float(i_perp[best_index])
            best_nf = int(costs[best_index])
        else:
            best_iperp = None
            best_nf = -1
        return {
            "configuration_sha256": str(
                legacy.scalar(saved, "configuration_sha256")
            ),
            "initial_path_sha256": str(
                legacy.scalar(saved, "initial_path_sha256")
            ),
            "configured_budget_per_image": budget,
            "terminal_NF": int(costs[-1]),
            "terminal_i_perp_diagnostic_only": float(i_perp[-1]),
            "terminal_segment_length_cv_diagnostic_only": float(cv[-1]),
            "terminal_segment_length_ratio_diagnostic_only": float(ratio[-1]),
            "complete_state_definition": (
                "all_published_projected_states"
                if policy == "projected_pathsum"
                else "initial_plus_reparam_committed"
            ),
            "complete_states": complete_rows,
            "mesh_eligible_complete_state_count": int(eligible_indices.size),
            "best_mesh_eligible_complete_i_perp": best_iperp,
            "best_mesh_eligible_complete_NF": best_nf,
            "h0_selection_count": int(np.count_nonzero(selections == "H0")),
            "lbfgs_selection_count": int(
                np.count_nonzero(selections == "LBFGS")
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--solver-role", choices=SOLVER_ROLES, required=True)
    parser.add_argument("--replicate", default="r1")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not args.replicate.startswith("r") or not args.replicate[1:].isdigit():
        raise ValueError("replicate must be r followed by digits")
    command, label = pilot_command(
        root, args.policy, args.solver_role, args.replicate
    )
    result_dir = root / "runs" / POLICY_VERSION / "members"
    audit_dir = root / "runs" / POLICY_VERSION / "audits"
    log_dir = root / "logs" / POLICY_VERSION / label
    for directory in (result_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        result_dir.glob(
            f"ivm_{label}_central_state_ivm_cfg*_budget{PILOT_BUDGET}.npz"
        )
    )
    audit_path = audit_dir / f"{label}.json"
    if existing or audit_path.exists() or log_dir.exists():
        target = existing[0] if existing else audit_path
        raise FileExistsError(f"Refusing to overwrite pilot member: {target}")
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    wall_seconds = legacy.run_command(
        command,
        root,
        log_dir / "stdout.txt",
        log_dir / "stderr.txt",
    )
    result = legacy.find_result(result_dir, label, PILOT_BUDGET)
    summary = _validate_result(
        result, args.policy, args.solver_role, PILOT_BUDGET
    )
    audit = {
        "policy_version": POLICY_VERSION,
        "pilot_only_not_formal_grid": True,
        "alpha": PILOT_ALPHA,
        "alpha_mode": "fixed",
        "policy": args.policy,
        "solver_role": args.solver_role,
        "primary_requested_solver": args.solver_role == "discrete_b040",
        "stop_rule": (
            "legacy"
            if args.solver_role == "legacy"
            else "R/R0 <= 0.8 OR evaluated accepted steps == 10"
        ),
        "matched_estimand": (
            "whole solver plus R08C10 package versus same-policy legacy"
        ),
        "mesh_gate": {
            "segment_length_cv_max": MESH_CV_MAX,
            "segment_length_ratio_max": MESH_RATIO_MAX,
        },
        "selection_state_definition": (
            "all_published_projected_states"
            if args.policy == "projected_pathsum"
            else "initial_plus_reparam_committed"
        ),
        "terminal_and_sparse_barrier_role": "diagnostic_only_not_selection",
        "command": command,
        "result": str(result),
        "result_sha256": legacy.file_sha256(result),
        "wall_seconds": wall_seconds,
        **summary,
    }
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
        handle.write("\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
