#!/usr/bin/env python3
"""Clean Stage-IV run with a triggered B040-style accelerator for Ala IVM.

The Phase-A/Phase-B outer and reparameterization options are copied exactly
from ``run_stage4_uniform_optimized.py``.  The registered changes are the
H0-vs-legacy-H0-LBFGS selector, its I_perp state trigger, and the C10/R08
inner stopping rule.  Each member uses one fixed alpha throughout both
phases.  Before the trigger the core follows the legacy solver path; after
the trigger it enables DSEL with that same alpha.  Every member starts Phase
A from the common path and starts Phase B in a fresh process from that
member's evaluated Phase-A terminal path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

import run_stage4_uniform_optimized as legacy


POLICY_VERSION = "stage4_ala_dsel_ffrbk_v3"
EXPECTED_IMPLEMENTATION_ID = (
    "ala-legacy-ivm-fixedalpha-triggered-ffrbk-vs-h0-lbfgs-"
    "postgeometry-dsel-v3"
)


def runner_base(
    root: Path,
    initial: str,
    output_dir: str,
    run_label: str,
    alpha: float,
    budget: int,
    stop_mode: str,
    trigger_iperp: float = math.inf,
) -> list[str]:
    command = legacy.runner_base(
        root, initial, output_dir, run_label, alpha, budget
    )
    command += [
        "--frozen-subproblem",
        "ala_dsel",
        "--discrete-inner-max-steps",
        "10",
        "--discrete-stop-mode",
        stop_mode,
        "--discrete-residual-ratio",
        "0.8",
        "--discrete-history",
        "7",
        "--ala-dsel-trigger-iperp",
        f"{trigger_iperp:g}",
    ]
    return command


def phase_a_command(
    root: Path,
    alpha: float,
    budget: int,
    stop_mode: str,
    replicate: str,
    execution_tag: str = "",
    trigger_iperp: float = math.inf,
) -> tuple[list[str], str]:
    alpha_tag = legacy.label_number(alpha)
    trigger_tag = legacy.label_number(trigger_iperp)
    suffix = f"_{execution_tag}" if execution_tag else ""
    run_label = (
        f"{POLICY_VERSION}_phaseA_fast_{stop_mode}_a{alpha_tag}_"
        f"t{trigger_tag}_{replicate}"
        f"{suffix}"
    )
    command = runner_base(
        root,
        "inputs/common_initial_path_equidistant.npz",
        f"runs/{POLICY_VERSION}/phases",
        run_label,
        alpha,
        budget,
        stop_mode,
        trigger_iperp,
    )
    command += [
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
    ]
    return command, run_label


def phase_b_command(
    root: Path,
    alpha: float,
    budget: int,
    continuation: Path,
    stop_mode: str,
    replicate: str,
    stop_complete_iperp: float,
    execution_tag: str = "",
    trigger_iperp: float = math.inf,
) -> tuple[list[str], str]:
    alpha_tag = legacy.label_number(alpha)
    trigger_tag = legacy.label_number(trigger_iperp)
    suffix = f"_{execution_tag}" if execution_tag else ""
    run_label = (
        f"{POLICY_VERSION}_phaseB_precision_{stop_mode}_a{alpha_tag}_"
        f"t{trigger_tag}_{replicate}"
        f"{suffix}"
    )
    command = runner_base(
        root,
        str(continuation),
        f"runs/{POLICY_VERSION}/phases",
        run_label,
        alpha,
        budget,
        stop_mode,
        trigger_iperp,
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
        "--stop-complete-iperp", f"{stop_complete_iperp:g}",
    ]
    return command, run_label


def command_value(command: list[str], flag: str) -> str | bool:
    indices = [index for index, value in enumerate(command) if value == flag]
    if len(indices) != 1:
        raise RuntimeError(f"Expected one {flag}; found {len(indices)}")
    index = indices[0]
    if index + 1 == len(command) or command[index + 1].startswith("--"):
        return True
    return command[index + 1]


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


def assert_frozen_outer_policy(
    root: Path,
    alpha: float,
    phase_a: list[str],
    phase_b: list[str],
    continuation: Path,
) -> None:
    expected_alpha = math.inf if math.isinf(alpha) else float(alpha)
    trigger_values: list[float] = []
    for command, phase in ((phase_a, "A"), (phase_b, "B")):
        command_alpha = float(command_value(command, "--alpha"))
        if command_alpha != expected_alpha:
            raise RuntimeError(
                f"Phase {phase} alpha is not the registered fixed alpha"
            )
        if command_value(command, "--alpha-mode") != "fixed":
            raise RuntimeError(f"Phase {phase} alpha mode is not fixed")
        trigger_values.append(
            float(command_value(command, "--ala-dsel-trigger-iperp"))
        )
    if trigger_values[0] != trigger_values[1]:
        raise RuntimeError("Phase A and Phase B use different DSEL triggers")

    legacy_a, _ = legacy.phase_a_command(root, alpha, int(command_value(phase_a, "--budget")))
    legacy_b, _ = legacy.phase_b_command(
        root,
        alpha,
        int(command_value(phase_b, "--budget")),
        continuation,
        stop_complete_iperp=float(command_value(phase_b, "--stop-complete-iperp")),
    )
    extensions = {
        "--frozen-subproblem",
        "--discrete-inner-max-steps",
        "--discrete-stop-mode",
        "--discrete-residual-ratio",
        "--discrete-history",
        "--ala-dsel-trigger-iperp",
    }
    for new_command, old_command, phase in (
        (phase_a, legacy_a, "A"),
        (phase_b, legacy_b, "B"),
    ):
        new_options = command_options(new_command)
        old_options = command_options(old_command)
        for routing in ("--run-label", "--output-dir"):
            new_options.pop(routing)
            old_options.pop(routing)
        extension_values = {
            key: new_options.pop(key, None) for key in sorted(extensions)
        }
        if set(extension_values) != extensions or any(
            value is None for value in extension_values.values()
        ):
            raise RuntimeError(f"Phase {phase} is missing a DSEL extension")
        if new_options != old_options:
            differing = sorted(set(new_options) | set(old_options))
            raise RuntimeError(
                f"Phase {phase} outer/reparameterization command drift: "
                + ", ".join(
                    key
                    for key in differing
                    if new_options.get(key) != old_options.get(key)
                )
            )
        expected_initial = (
            "inputs/common_initial_path_equidistant.npz"
            if phase == "A"
            else str(continuation)
        )
        if new_options["--initial-path-file"] != expected_initial:
            raise RuntimeError(f"Phase {phase} initial path is not canonical")


def _concat_optional(saved_a, saved_b, key, dtype, phase_b_outer_offset=0):
    left = np.asarray(saved_a[key], dtype=dtype)
    right = np.asarray(saved_b[key], dtype=dtype)
    if key == "ala_dsel_audit_outer":
        right = right + int(phase_b_outer_offset)
    return np.concatenate((left, right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--stop-mode", choices=("c10", "r08"), required=True)
    parser.add_argument("--phase-a-budget", type=int, default=34)
    parser.add_argument("--phase-b-budget", type=int, default=220)
    parser.add_argument("--stop-complete-iperp", type=float, default=-1.0)
    parser.add_argument(
        "--ala-dsel-trigger-iperp",
        type=float,
        default=8e-4,
        help=(
            "Enable fixed-alpha DSEL once the current accepted I_perp is at "
            "or below this positive state threshold."
        ),
    )
    parser.add_argument("--replicate", default="r1")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    alpha = math.inf if args.alpha.lower() in {"inf", "infinity"} else float(args.alpha)
    if not (alpha > 0.0):
        raise ValueError("alpha must be positive or inf")
    trigger_iperp = float(args.ala_dsel_trigger_iperp)
    if not trigger_iperp > 0.0:
        raise ValueError("Ala DSEL trigger I_perp must be positive")
    if not args.replicate.startswith("r") or not args.replicate[1:].isdigit():
        raise ValueError("replicate must be r followed by digits")
    alpha_tag = legacy.label_number(alpha)
    trigger_tag = legacy.label_number(trigger_iperp)
    member_id = (
        f"{args.stop_mode}_a{alpha_tag}_t{trigger_tag}_{args.replicate}"
    )
    execution_tag = f"A{args.phase_a_budget}_B{args.phase_b_budget}"
    execution_id = f"{member_id}_{execution_tag}"
    result_dir = root / "runs" / POLICY_VERSION
    phase_dir = result_dir / "phases"
    continuation_dir = result_dir / "continuations"
    log_dir = root / "logs" / POLICY_VERSION / execution_id
    for directory in (result_dir, phase_dir, continuation_dir):
        directory.mkdir(parents=True, exist_ok=True)
    combined = result_dir / (
        f"{POLICY_VERSION}_{member_id}_phaseA{args.phase_a_budget}_"
        f"phaseB{args.phase_b_budget}.npz"
    )
    audit_path = combined.with_suffix(".json")
    if combined.exists() or audit_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing member; use a new replicate: {combined}"
        )
    # This exclusive directory is also the execution claim, preventing two
    # identical concurrent launches from racing past the file checks.
    log_dir.mkdir(parents=True, exist_ok=False)

    command_a, run_label_a = phase_a_command(
        root,
        alpha,
        args.phase_a_budget,
        args.stop_mode,
        args.replicate,
        execution_tag,
        trigger_iperp,
    )
    provisional_continuation = continuation_dir / (
        f"phaseA{args.phase_a_budget}_phaseB{args.phase_b_budget}_"
        f"{member_id}.npz"
    )
    command_b_preview, _ = phase_b_command(
        root,
        alpha,
        args.phase_b_budget,
        provisional_continuation,
        args.stop_mode,
        args.replicate,
        args.stop_complete_iperp,
        execution_tag,
        trigger_iperp,
    )
    assert_frozen_outer_policy(
        root, alpha, command_a, command_b_preview, provisional_continuation
    )
    (log_dir / "phaseA_command.json").write_text(
        json.dumps(command_a, indent=2) + "\n"
    )
    wall_a = legacy.run_command(
        command_a,
        root,
        log_dir / "phaseA_stdout.txt",
        log_dir / "phaseA_stderr.txt",
    )
    result_a = legacy.find_result(
        phase_dir, run_label_a, args.phase_a_budget
    )

    with np.load(result_a, allow_pickle=False) as saved_a:
        terminal = np.asarray(saved_a["final_path"], dtype=float)
        initial_sha = str(legacy.scalar(saved_a, "initial_path_sha256"))
        configuration_a = str(legacy.scalar(saved_a, "configuration_sha256"))
        phase_a_final_cost = int(saved_a["total_internal_joint_evaluations"][-1])
        phase_a_final_iperp = float(saved_a["i_perp"][-1])
    continuation = provisional_continuation
    np.savez_compressed(
        continuation,
        initial_path=terminal,
        path_sha256=np.array(legacy.array_sha256(terminal)),
        source_result=np.array(str(result_a)),
        source_result_sha256=np.array(legacy.file_sha256(result_a)),
        source_initial_path_sha256=np.array(initial_sha),
        source_configuration_sha256=np.array(configuration_a),
        source_final_force_evaluations=np.array(phase_a_final_cost),
        source_final_iperp=np.array(phase_a_final_iperp),
        continuation_policy=np.array(
            "fresh optimizer/evaluator state from evaluated phase-A terminal; "
            "phase-B initialization is charged"
        ),
    )

    command_b, run_label_b = phase_b_command(
        root,
        alpha,
        args.phase_b_budget,
        continuation,
        args.stop_mode,
        args.replicate,
        args.stop_complete_iperp,
        execution_tag,
        trigger_iperp,
    )
    assert_frozen_outer_policy(root, alpha, command_a, command_b, continuation)
    (log_dir / "phaseB_command.json").write_text(
        json.dumps(command_b, indent=2) + "\n"
    )
    wall_b = legacy.run_command(
        command_b,
        root,
        log_dir / "phaseB_stdout.txt",
        log_dir / "phaseB_stderr.txt",
    )
    result_b = legacy.find_result(
        phase_dir,
        run_label_b,
        args.phase_b_budget,
        initial_path_sha256=legacy.array_sha256(terminal),
    )

    with np.load(result_a, allow_pickle=False) as a, np.load(
        result_b, allow_pickle=False
    ) as b:
        expected_implementation = EXPECTED_IMPLEMENTATION_ID
        implementation_a = str(legacy.scalar(a, "ala_dsel_implementation_id"))
        implementation_b = str(legacy.scalar(b, "ala_dsel_implementation_id"))
        if implementation_a != expected_implementation or implementation_b != expected_implementation:
            raise RuntimeError("Phase results use the wrong Ala DSEL implementation")
        for saved, phase_name in ((a, "A"), (b, "B")):
            if float(legacy.scalar(saved, "alpha")) != float(alpha):
                raise RuntimeError(
                    f"Phase {phase_name} result did not use the fixed alpha"
                )
            if str(legacy.scalar(saved, "alpha_mode")) != "fixed":
                raise RuntimeError(
                    f"Phase {phase_name} result alpha mode is not fixed"
                )
            if (
                float(legacy.scalar(saved, "ala_dsel_trigger_iperp"))
                != trigger_iperp
            ):
                raise RuntimeError(
                    f"Phase {phase_name} result used the wrong DSEL trigger"
                )
        if not (
            str(legacy.scalar(a, "frozen_subproblem")) == "ala_dsel"
            and str(legacy.scalar(b, "frozen_subproblem")) == "ala_dsel"
            and bool(legacy.scalar(a, "ala_dsel_preserves_legacy_direction_candidate"))
            and bool(legacy.scalar(b, "ala_dsel_preserves_legacy_direction_candidate"))
            and bool(legacy.scalar(a, "ala_dsel_preserves_outer_reparameterization"))
            and bool(legacy.scalar(b, "ala_dsel_preserves_outer_reparameterization"))
        ):
            raise RuntimeError("Phase results fail the Ala DSEL provenance contract")
        decisions_before_activation_a = int(
            legacy.scalar(a, "ala_dsel_decisions_before_activation")
        )
        decisions_before_activation_b = int(
            legacy.scalar(b, "ala_dsel_decisions_before_activation")
        )
        decisions_before_activation = (
            decisions_before_activation_a + decisions_before_activation_b
        )
        if decisions_before_activation != 0:
            raise RuntimeError("Ala DSEL made a decision before state activation")
        for saved, phase_name in ((a, "A"), (b, "B")):
            if not (
                bool(
                    legacy.scalar(
                        saved, "ala_dsel_legacy_direction_before_activation"
                    )
                )
                and bool(
                    legacy.scalar(
                        saved, "ala_dsel_legacy_inner_stop_before_activation"
                    )
                )
                and bool(
                    legacy.scalar(
                        saved, "ala_dsel_registered_stop_only_after_activation"
                    )
                )
            ):
                raise RuntimeError(
                    f"Phase {phase_name} violated the pre-activation legacy contract"
                )
        cost_a = np.asarray(a["total_internal_joint_evaluations"], dtype=int)
        local_cost_b = np.asarray(b["total_internal_joint_evaluations"], dtype=int)
        cost_b = local_cost_b + cost_a[-1]
        transition_initialization_charged = bool(
            local_cost_b.size > 0
            and int(local_cost_b[0]) == int(terminal.shape[0] - 2)
            and int(cost_b[0] - cost_a[-1]) == int(terminal.shape[0] - 2)
        )
        if not transition_initialization_charged:
            raise RuntimeError("Phase-B initialization was not charged exactly once")
        macro_a = np.asarray(a["calls_per_image"], dtype=int)
        macro_b_local = np.asarray(b["calls_per_image"], dtype=int)
        macro_b = macro_b_local + macro_a[-1]
        stored_cost_a = legacy.cumulative_stored_cost(a)
        stored_cost_b = legacy.cumulative_stored_cost(b, int(cost_a[-1]))
        stored_macro_a = np.asarray(a["stored_path_calls"], dtype=int)
        stored_macro_b = np.asarray(b["stored_path_calls"], dtype=int) + macro_a[-1]
        i_perp = np.concatenate((a["i_perp"], b["i_perp"]))
        audit_outer_a = np.asarray(a["ala_dsel_audit_outer"], dtype=int)
        phase_b_outer_offset = int(np.max(audit_outer_a) + 1) if audit_outer_a.size else 0
        selection = _concat_optional(a, b, "ala_dsel_audit_selection", "U16")
        activation_a_nf = int(
            legacy.scalar(a, "ala_dsel_first_activation_nf")
        )
        activation_b_nf_local = int(
            legacy.scalar(b, "ala_dsel_first_activation_nf")
        )
        activation_a_calls = int(
            legacy.scalar(a, "ala_dsel_first_activation_calls_per_image")
        )
        activation_b_calls_local = int(
            legacy.scalar(b, "ala_dsel_first_activation_calls_per_image")
        )
        if activation_a_nf >= 0:
            activation_phase = 0
            activation_nf = activation_a_nf
            activation_calls = activation_a_calls
            activation_iperp = float(
                legacy.scalar(a, "ala_dsel_first_activation_iperp")
            )
        elif activation_b_nf_local >= 0:
            activation_phase = 1
            activation_nf = int(cost_a[-1]) + activation_b_nf_local
            activation_calls = int(macro_a[-1]) + activation_b_calls_local
            activation_iperp = float(
                legacy.scalar(b, "ala_dsel_first_activation_iperp")
            )
        else:
            activation_phase = -1
            activation_nf = -1
            activation_calls = -1
            activation_iperp = math.nan
        np.savez_compressed(
            combined,
            policy_version=np.array(POLICY_VERSION),
            frozen_subproblem=np.array("ala_dsel"),
            ala_dsel_implementation_id=np.array(expected_implementation),
            alpha=np.array(alpha),
            alpha_mode=np.array("fixed"),
            alpha_scope=np.array("one_fixed_alpha_throughout_both_phases"),
            fixed_alpha_throughout=np.array(True),
            ala_dsel_trigger_iperp=np.array(trigger_iperp),
            ala_dsel_activation_policy=np.array(
                "legacy_before_current_iperp_le_trigger_then_dsel_same_alpha"
            ),
            stop_mode=np.array(args.stop_mode),
            configured_relative_residual=np.array(0.8),
            relative_residual_stop_active=np.array(args.stop_mode == "r08"),
            inner_max_steps=np.array(10),
            lbfgs_history=np.array(7),
            common_initial_path=np.array("inputs/common_initial_path_equidistant.npz"),
            common_initial_path_sha256=np.array(initial_sha),
            phase_labels=np.asarray(["fast_pathsum", "precision_resample"]),
            phase_a_budget=np.array(args.phase_a_budget),
            phase_b_budget=np.array(args.phase_b_budget),
            phase_a_result=np.array(str(result_a)),
            phase_b_result=np.array(str(result_b)),
            phase_a_result_sha256=np.array(legacy.file_sha256(result_a)),
            phase_b_result_sha256=np.array(legacy.file_sha256(result_b)),
            phase_a_configuration_sha256=np.array(configuration_a),
            phase_b_configuration_sha256=np.array(
                str(legacy.scalar(b, "configuration_sha256"))
            ),
            phase_transition_macro_index=np.array(len(cost_a)),
            phase_transition_force_offset=np.array(cost_a[-1]),
            phase_transition_initialization_NF=np.array(local_cost_b[0]),
            phase_transition_initialization_charged=np.array(
                transition_initialization_charged
            ),
            macro_calls=np.concatenate((macro_a, macro_b)),
            calls_per_image=np.concatenate((macro_a, macro_b)),
            phase_a_calls_per_image_local=macro_a,
            phase_b_calls_per_image_local=macro_b_local,
            total_force_evaluations=np.concatenate((cost_a, cost_b)),
            i_perp=i_perp,
            sparse_barrier_kcal=np.concatenate((a["barrier_kcal"], b["barrier_kcal"])),
            physical_fmax=np.concatenate((a["physical_fmax"], b["physical_fmax"])),
            segment_length_cv=np.concatenate((a["segment_length_cv"], b["segment_length_cv"])),
            equidistance_relative_defect=np.concatenate((a["equidistance_relative_defect"], b["equidistance_relative_defect"])),
            segment_length_ratio=np.concatenate((a["segment_length_ratio"], b["segment_length_ratio"])),
            reparam_committed=np.concatenate((a["reparam_committed"], b["reparam_committed"])),
            phase=np.concatenate((np.zeros(len(cost_a), dtype=int), np.ones(len(cost_b), dtype=int))),
            stored_force_evaluations=np.concatenate((stored_cost_a, stored_cost_b)),
            stored_macro_calls=np.concatenate((stored_macro_a, stored_macro_b)),
            stored_phase=np.concatenate((np.zeros(len(stored_cost_a), dtype=int), np.ones(len(stored_cost_b), dtype=int))),
            stored_paths=np.concatenate((a["stored_paths"], b["stored_paths"]), axis=0),
            final_path=np.asarray(b["final_path"], dtype=float),
            ala_dsel_audit_outer=_concat_optional(
                a, b, "ala_dsel_audit_outer", int, phase_b_outer_offset
            ),
            ala_dsel_audit_selection=selection,
            ala_dsel_audit_h0_capped_predicted_decrease=_concat_optional(
                a, b, "ala_dsel_audit_h0_capped_predicted_decrease", float
            ),
            ala_dsel_audit_lbfgs_capped_predicted_decrease=_concat_optional(
                a, b, "ala_dsel_audit_lbfgs_capped_predicted_decrease", float
            ),
            ala_dsel_h0_selection_count=np.array(np.count_nonzero(selection == "H0")),
            ala_dsel_lbfgs_selection_count=np.array(np.count_nonzero(selection == "LBFGS")),
            ala_dsel_safeguard_selection_count=np.array(
                np.count_nonzero(selection == "SAFEGUARD")
            ),
            ala_dsel_decisions_before_activation=np.array(
                decisions_before_activation
            ),
            phase_a_ala_dsel_decisions_before_activation=np.array(
                decisions_before_activation_a
            ),
            phase_b_ala_dsel_decisions_before_activation=np.array(
                decisions_before_activation_b
            ),
            ala_dsel_first_activation_phase=np.array(activation_phase),
            ala_dsel_first_activation_nf=np.array(activation_nf),
            ala_dsel_first_activation_calls_per_image=np.array(
                activation_calls
            ),
            ala_dsel_first_activation_iperp=np.array(activation_iperp),
            phase_a_ala_dsel_first_activation_nf_local=np.array(
                activation_a_nf
            ),
            phase_b_ala_dsel_first_activation_nf_local=np.array(
                activation_b_nf_local
            ),
            phase_a_ala_dsel_first_activation_calls_per_image_local=np.array(
                activation_a_calls
            ),
            phase_b_ala_dsel_first_activation_calls_per_image_local=np.array(
                activation_b_calls_local
            ),
            ala_dsel_legacy_direction_before_activation=np.array(True),
            ala_dsel_legacy_inner_stop_before_activation=np.array(True),
            ala_dsel_registered_stop_only_after_activation=np.array(True),
            ala_dsel_preserves_legacy_direction_candidate=np.array(True),
            ala_dsel_preserves_outer_reparameterization=np.array(True),
            ala_dsel_changes_registered_inner_stop=np.array(True),
            ala_dsel_curve_selection_score_stage=np.array(
                "after_cap_and_zero_call_spring_projection"
            ),
            ala_dsel_selector_extra_physical_candidate_evaluations=np.array(0),
            stop_complete_iperp=np.array(args.stop_complete_iperp),
            phase_b_stop_reason=np.array(str(legacy.scalar(b, "stop_reason"))),
            terminal_reparam_committed=np.array(bool(b["reparam_committed"][-1])),
            materialization_mode=np.array("independent_online_execution"),
        )

    audit = {
        "policy_version": POLICY_VERSION,
        "member_id": member_id,
        "alpha": "inf" if math.isinf(alpha) else alpha,
        "alpha_mode": "fixed",
        "alpha_scope": "one_fixed_alpha_throughout_both_phases",
        "ala_dsel_trigger_iperp": trigger_iperp,
        "ala_dsel_activation_policy": (
            "legacy_before_current_iperp_le_trigger_then_dsel_same_alpha"
        ),
        "stop_mode": args.stop_mode,
        "inner_max_steps": 10,
        "configured_relative_residual": 0.8,
        "relative_residual_stop_active": args.stop_mode == "r08",
        "lbfgs_history": 7,
        "common_initial_path_sha256": initial_sha,
        "phase_a_result": str(result_a),
        "phase_a_result_sha256": legacy.file_sha256(result_a),
        "phase_b_result": str(result_b),
        "phase_b_result_sha256": legacy.file_sha256(result_b),
        "combined_result": str(combined),
        "combined_result_sha256": legacy.file_sha256(combined),
        "phase_a_wall_seconds": wall_a,
        "phase_b_wall_seconds": wall_b,
        "phase_a_final_force_evaluations": phase_a_final_cost,
        "phase_transition_initialization_NF": int(local_cost_b[0]),
        "phase_transition_initialization_charged": bool(
            transition_initialization_charged
        ),
        "combined_final_force_evaluations": int(cost_b[-1]),
        "combined_final_iperp": float(i_perp[-1]),
        "combined_min_iperp": float(np.min(i_perp)),
        "h0_selection_count": int(np.count_nonzero(selection == "H0")),
        "lbfgs_selection_count": int(np.count_nonzero(selection == "LBFGS")),
        "safeguard_selection_count": int(
            np.count_nonzero(selection == "SAFEGUARD")
        ),
        "ala_dsel_decisions_before_activation": decisions_before_activation,
        "phase_a_ala_dsel_decisions_before_activation": (
            decisions_before_activation_a
        ),
        "phase_b_ala_dsel_decisions_before_activation": (
            decisions_before_activation_b
        ),
        "ala_dsel_first_activation_phase": activation_phase,
        "ala_dsel_first_activation_nf": activation_nf,
        "ala_dsel_first_activation_calls_per_image": activation_calls,
        "ala_dsel_first_activation_iperp": (
            None if math.isnan(activation_iperp) else activation_iperp
        ),
        "phase_a_ala_dsel_first_activation_nf_local": activation_a_nf,
        "phase_b_ala_dsel_first_activation_nf_local": activation_b_nf_local,
        "phase_a_ala_dsel_first_activation_calls_per_image_local": (
            activation_a_calls
        ),
        "phase_b_ala_dsel_first_activation_calls_per_image_local": (
            activation_b_calls_local
        ),
        "legacy_direction_before_activation": True,
        "legacy_inner_stop_before_activation": True,
        "registered_stop_only_after_activation": True,
        "frozen_subproblem": "ala_dsel",
        "ala_dsel_implementation_id": expected_implementation,
        "legacy_direction_candidate_preserved": True,
        "legacy_outer_reparameterization_command_preserved": True,
        "registered_inner_stop_changed": True,
        "curve_selection_score_stage": (
            "after_cap_and_zero_call_spring_projection"
        ),
        "selector_extra_physical_candidate_evaluations": 0,
        "fixed_alpha_in_both_phases": True,
        "fixed_alpha_throughout": True,
        "cross_member_state_shared": False,
        "outer_policy_command_audit": "exact_bidirectional_comparison_passed",
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
