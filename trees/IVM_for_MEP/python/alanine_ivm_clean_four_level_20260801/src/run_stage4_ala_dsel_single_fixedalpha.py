#!/usr/bin/env python3
"""Single-run fixed-alpha Ala IVM with standalone direct-coordinate DSEL.

The path starts from the canonical common Alanine configuration and uses one
unchanged alpha, one optimizer process, and one reparameterization policy for
the full trajectory.  The fixed-path subproblem directly parameterizes all
discrete Cartesian path coordinates and is active from the first state.  Its
registered stop is exactly
``R/R0 <= 0.8 OR evaluated accepted steps == 10``.  Precision ``state_curve``
reparameterization is used from the first state because it is the existing
production policy that protects the mesh while targeting the low-I_perp tail;
mixing it with the fast path-summed policy would reintroduce a policy switch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import run_stage4_uniform_optimized as legacy


POLICY_VERSION = "stage4_ala_discrete_single_fixedalpha_v2"
EXPECTED_IMPLEMENTATION_ID = "ala-frozen-discrete-ffrbk-dsel-m7-evaluated-v1"
COMMON_INITIAL = "inputs/common_initial_path_equidistant.npz"
STOP_RULE_ID = "r08c10"
IMMEDIATE_TRIGGER_IPERP = math.inf
REPARAM_POLICY_ID = "precision_state_curve_from_start"
DSEL_COMMAND_OPTIONS = frozenset(
    {
        "--frozen-subproblem",
        "--discrete-inner-max-steps",
        "--discrete-stop-mode",
        "--discrete-residual-ratio",
        "--discrete-history",
        "--ala-dsel-trigger-iperp",
    }
)


def command_options(command: list[str]) -> dict[str, str | bool]:
    options: dict[str, str | bool] = {}
    index = 0
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            index += 1
            continue
        if token in options:
            raise RuntimeError(f"Duplicate command option: {token}")
        if index + 1 == len(command) or command[index + 1].startswith("--"):
            options[token] = True
            index += 1
        else:
            options[token] = command[index + 1]
            index += 2
    return options


def command_value(command: list[str], flag: str) -> str | bool:
    options = command_options(command)
    if flag not in options:
        raise RuntimeError(f"Missing command option: {flag}")
    return options[flag]


def run_label(
    alpha: float,
    budget: int,
    replicate: str,
    member_role: str,
) -> str:
    algorithm_tag = "discrete_b040" if member_role == "candidate" else "legacy"
    stop_tag = STOP_RULE_ID if member_role == "candidate" else "legacy_stop"
    return (
        f"{POLICY_VERSION}_{stop_tag}_a{legacy.label_number(alpha)}_"
        f"N{budget}_{algorithm_tag}_{replicate}"
    )


def single_run_command(
    root: Path,
    alpha: float,
    budget: int,
    replicate: str,
    member_role: str = "candidate",
) -> tuple[list[str], str]:
    label = run_label(alpha, budget, replicate, member_role)
    command = legacy.runner_base(
        root,
        COMMON_INITIAL,
        f"runs/{POLICY_VERSION}/members",
        label,
        alpha,
        budget,
    )
    command += [
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
    if member_role == "candidate":
        command += [
            "--frozen-subproblem", "discrete_b040",
            "--discrete-inner-max-steps", "10",
            "--discrete-stop-mode", "r08c10",
            "--discrete-residual-ratio", "0.8",
            "--discrete-history", "7",
        ]
    elif member_role == "matched_control":
        command += ["--frozen-subproblem", "legacy"]
    else:
        raise ValueError(f"Unknown member role: {member_role}")
    assert_single_run_policy(command, alpha, member_role)
    return command, label


def assert_single_run_policy(
    command: list[str], alpha: float, member_role: str
) -> None:
    options = command_options(command)
    expected_alpha = math.inf if math.isinf(alpha) else float(alpha)
    if float(options["--alpha"]) != expected_alpha:
        raise RuntimeError("Command does not use the registered fixed alpha")
    if options["--alpha-mode"] != "fixed":
        raise RuntimeError("Command alpha mode is not fixed")
    if options["--initial-path-file"] != COMMON_INITIAL:
        raise RuntimeError("Command does not start from the canonical common path")
    if member_role == "candidate":
        if options["--frozen-subproblem"] != "discrete_b040":
            raise RuntimeError("Candidate does not select standalone discrete B040")
        if options["--discrete-stop-mode"] != "r08c10":
            raise RuntimeError("Candidate does not use the sole R08C10 rule")
        if int(options["--discrete-inner-max-steps"]) != 10:
            raise RuntimeError("Evaluated-step cap must equal 10")
        if float(options["--discrete-residual-ratio"]) != 0.8:
            raise RuntimeError("Residual-ratio threshold must equal 0.8")
        if int(options["--discrete-history"]) != int(options["--history"]):
            raise RuntimeError("DSEL and legacy L-BFGS history must agree")
    elif member_role == "matched_control":
        if options["--frozen-subproblem"] != "legacy":
            raise RuntimeError("Matched control does not use the legacy solver")
        forbidden_control = {
            "--discrete-inner-max-steps",
            "--discrete-stop-mode",
            "--discrete-residual-ratio",
            "--discrete-history",
            "--ala-dsel-trigger-iperp",
        }
        leaked = forbidden_control.intersection(options)
        if leaked:
            raise RuntimeError(
                "Matched control contains DSEL options: "
                + ", ".join(sorted(leaked))
            )
    else:
        raise RuntimeError(f"Unknown member role: {member_role}")
    if options["--reparam-mode"] != "state_curve":
        raise RuntimeError("Command does not use precision state_curve throughout")
    if options["--state-curve-direct-objective-acceptance"] != "per_image":
        raise RuntimeError("Command changed the registered per-image policy")
    if float(options["--stop-complete-iperp"]) != -1.0:
        raise RuntimeError("Equal-budget run cannot stop on a terminal I_perp")
    forbidden = {"--ala-dsel-alpha", "--alpha-schedule"}
    present = forbidden.intersection(options)
    if present:
        raise RuntimeError(
            "Fixed-alpha command contains a forbidden switch: "
            + ", ".join(sorted(present))
        )


def assert_matched_pair_commands(
    candidate: list[str], control: list[str], alpha: float
) -> None:
    assert_single_run_policy(candidate, alpha, "candidate")
    assert_single_run_policy(control, alpha, "matched_control")
    candidate_options = command_options(candidate)
    control_options = command_options(control)
    for flag in DSEL_COMMAND_OPTIONS | {"--run-label"}:
        candidate_options.pop(flag, None)
        control_options.pop(flag, None)
    if candidate_options != control_options:
        differing = sorted(set(candidate_options) | set(control_options))
        raise RuntimeError(
            "Matched pair command drift outside the registered DSEL package: "
            + ", ".join(
                flag
                for flag in differing
                if candidate_options.get(flag) != control_options.get(flag)
            )
        )


def complete_candidate_calls(saved: np.lib.npyio.NpzFile) -> np.ndarray:
    """Single-run adaptation of the Stage-IV complete-state protocol.

    The common initial evaluated state is eligible, followed only by states
    whose precision reparameterization was committed.  Mesh gates and I_perp
    cost-to-precision filtering are applied later, never by terminal rank.
    """
    costs = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
    committed = np.asarray(saved["reparam_committed"], dtype=bool)
    if costs.ndim != 1 or committed.shape != costs.shape or not costs.size:
        raise RuntimeError("Invalid single-run complete-state arrays")
    eligible = committed.copy()
    eligible[0] = True
    return costs[eligible]


def _validate_result(
    result: Path,
    alpha: float,
    budget: int,
    member_role: str,
) -> dict[str, object]:
    with np.load(result, allow_pickle=False) as saved:
        if float(legacy.scalar(saved, "alpha")) != float(alpha):
            raise RuntimeError("Result did not keep the registered alpha")
        if str(legacy.scalar(saved, "alpha_mode")) != "fixed":
            raise RuntimeError("Result alpha mode is not fixed")
        if str(legacy.scalar(saved, "reparam_mode")) != "state_curve":
            raise RuntimeError("Result used the wrong reparameterization policy")
        if member_role == "candidate":
            if str(legacy.scalar(saved, "discrete_implementation_id")) != (
                EXPECTED_IMPLEMENTATION_ID
            ):
                raise RuntimeError("Result uses the wrong discrete implementation")
            if str(legacy.scalar(saved, "frozen_subproblem")) != "discrete_b040":
                raise RuntimeError("Candidate result uses the wrong subproblem")
            if str(legacy.scalar(saved, "discrete_stop_mode")) != "r08c10":
                raise RuntimeError("Candidate result used the wrong stop rule")
            if int(legacy.scalar(saved, "discrete_inner_max_steps")) != 10:
                raise RuntimeError("Result used the wrong evaluated-step cap")
            if float(legacy.scalar(saved, "discrete_residual_ratio")) != 0.8:
                raise RuntimeError("Result used the wrong residual threshold")
            all_returned_evaluated = bool(
                legacy.scalar(saved, "discrete_all_returned_states_evaluated")
            )
            if not all_returned_evaluated:
                raise RuntimeError("Candidate returned an unevaluated state")
            selections = np.asarray(
                saved["discrete_audit_selection"], dtype="U16"
            )
        elif member_role == "matched_control":
            forbidden_result_keys = {
                "ala_dsel_implementation_id",
                "ala_dsel_audit_selection",
                "ala_dsel_first_activation_nf",
                "discrete_implementation_id",
                "discrete_audit_selection",
            }
            leaked = forbidden_result_keys.intersection(saved.files)
            if leaked:
                raise RuntimeError(
                    "Legacy control contains DSEL output: "
                    + ", ".join(sorted(leaked))
                )
            all_returned_evaluated = True
            selections = np.asarray([], dtype="U16")
        else:
            raise RuntimeError(f"Unknown member role: {member_role}")
        costs = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        complete_calls = complete_candidate_calls(saved)
        return {
            "configuration_sha256": str(
                legacy.scalar(saved, "configuration_sha256")
            ),
            "initial_path_sha256": str(
                legacy.scalar(saved, "initial_path_sha256")
            ),
            "terminal_NF": int(costs[-1]),
            "terminal_calls_per_image": int(saved["calls_per_image"][-1]),
            "terminal_i_perp": float(saved["i_perp"][-1]),
            "minimum_i_perp": float(np.min(saved["i_perp"])),
            "terminal_sparse_barrier_kcal": float(saved["barrier_kcal"][-1]),
            "terminal_physical_fmax": float(saved["physical_fmax"][-1]),
            "terminal_segment_length_cv": float(saved["segment_length_cv"][-1]),
            "terminal_segment_length_ratio": float(
                saved["segment_length_ratio"][-1]
            ),
            "discrete_active_from_first_state": member_role == "candidate",
            "discrete_all_returned_states_evaluated": all_returned_evaluated,
            "discrete_joint_step_count": int(selections.size),
            "h0_selection_count": int(np.count_nonzero(selections == "H0")),
            "lbfgs_selection_count": int(
                np.count_nonzero(selections == "LBFGS")
            ),
            "safeguard_selection_count": int(
                np.count_nonzero(selections == "SAFEGUARD")
            ),
            "configured_budget_per_image": budget,
            "complete_state_definition": "initial_plus_reparam_committed",
            "complete_candidate_state_count": int(complete_calls.size),
            "complete_candidate_state_NF": complete_calls.tolist(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--budget", type=int, default=254)
    parser.add_argument(
        "--member-role",
        choices=("candidate", "matched_control"),
        default="candidate",
    )
    parser.add_argument("--replicate", default="r1")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    alpha = (
        math.inf
        if args.alpha.lower() in {"inf", "infinity"}
        else float(args.alpha)
    )
    if not alpha > 0.0:
        raise ValueError("alpha must be positive or inf")
    if args.budget < 2:
        raise ValueError("budget must be at least two evaluated states")
    if not args.replicate.startswith("r") or not args.replicate[1:].isdigit():
        raise ValueError("replicate must be r followed by digits")
    command, label = single_run_command(
        root,
        alpha,
        args.budget,
        args.replicate,
        args.member_role,
    )
    result_dir = root / "runs" / POLICY_VERSION / "members"
    audit_dir = root / "runs" / POLICY_VERSION / "audits"
    log_dir = root / "logs" / POLICY_VERSION / label
    for directory in (result_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        result_dir.glob(
            f"ivm_{label}_central_state_ivm_cfg*_budget{args.budget}.npz"
        )
    )
    audit_path = audit_dir / f"{label}.json"
    if existing or audit_path.exists() or log_dir.exists():
        target = existing[0] if existing else audit_path
        raise FileExistsError(f"Refusing to overwrite member: {target}")
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
    result = legacy.find_result(result_dir, label, args.budget)
    summary = _validate_result(
        result,
        alpha,
        args.budget,
        args.member_role,
    )
    audit: dict[str, object] = {
        "policy_version": POLICY_VERSION,
        "implementation_id": (
            EXPECTED_IMPLEMENTATION_ID
            if args.member_role == "candidate"
            else "legacy_state_ivm"
        ),
        "member_role": args.member_role,
        "single_run_from_canonical_common_path": True,
        "common_initial_path": COMMON_INITIAL,
        "alpha": "inf" if math.isinf(alpha) else alpha,
        "alpha_mode": "fixed",
        "fixed_alpha_throughout": True,
        "frozen_subproblem": (
            "discrete_b040" if args.member_role == "candidate" else "legacy"
        ),
        "discrete_solver_activation": (
            "standalone_from_first_state"
            if args.member_role == "candidate"
            else "not_applicable"
        ),
        "stop_rule_id": (
            STOP_RULE_ID if args.member_role == "candidate" else "legacy"
        ),
        "stop_rule": (
            "R/R0 <= 0.8 OR evaluated accepted steps == 10"
            if args.member_role == "candidate"
            else "legacy state-IV inner stopping"
        ),
        "tested_stop_variants": False,
        "reparameterization_policy_id": REPARAM_POLICY_ID,
        "reparameterization_policy": "state_curve per_image from first state",
        "reparameterization_choice_reason": (
            "existing production precision policy; one policy avoids an outer-"
            "algorithm switch and protects the low-I_perp mesh"
        ),
        "sparse_barrier_role": "blind_diagnostic_only_not_selection_or_ranking",
        "matched_pair_estimand": (
            "whole standalone discrete-coordinate DSEL plus R08C10 package "
            "versus legacy at identical alpha"
        ),
        "result": str(result),
        "result_sha256": legacy.file_sha256(result),
        "wall_seconds": wall_seconds,
        "command": command,
        **summary,
    }
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
        handle.write("\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
