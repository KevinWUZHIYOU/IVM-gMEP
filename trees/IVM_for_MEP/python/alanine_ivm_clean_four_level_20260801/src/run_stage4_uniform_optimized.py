#!/usr/bin/env python3
"""Uniform two-phase production IVM, always restarted from the clean path.

The same state machine is used for every alpha.  Phase A is the fast,
path-summed projected-curve policy.  Phase B starts from phase A's accepted
terminal Cartesian path with fresh optimizer memory and uses the precision
resampling policy.  Alpha never changes.  The phase-B initialization is
counted rather than deduplicated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


POLICY_VERSION = "stage4_uniform_v3"


def label_number(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:g}".replace(".", "p").replace("-", "m")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def scalar(saved, key):
    return np.asarray(saved[key]).item()


def run_command(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path):
    environment = os.environ.copy()
    environment.update(
        {
            "OPENMM_CPU_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            + " ".join(command)
        )
    return time.time() - started


def runner_base(root: Path, initial: str, output_dir: str, run_label: str,
                alpha: float, budget: int) -> list[str]:
    return [
        sys.executable,
        str(root / "src/run_ivm_v9_spring_projected_production_20260730.py"),
        "--root", str(root),
        "--initial-path-file", initial,
        "--output-dir", output_dir,
        "--run-label", run_label,
        "--solver", "state_ivm",
        "--alpha", "inf" if math.isinf(alpha) else f"{alpha:g}",
        "--alpha-mode", "fixed",
        "--alpha-min", "0.01",
        "--alpha-max", "inf",
        "--max-step", "1",
        "--history", "7",
        "--stop-iperp", "-1",
        "--budget", str(budget),
        "--save-all-paths",
    ]


def phase_a_command(root: Path, alpha: float, budget: int) -> tuple[list[str], str]:
    alpha_tag = label_number(alpha)
    run_label = f"clean_stage4_v3_phaseA_fast_a{alpha_tag}"
    command = runner_base(
        root,
        "inputs/common_initial_path_equidistant.npz",
        "runs/stage4_inexact_optimized_resample/phases",
        run_label,
        alpha,
        budget,
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
    stop_complete_iperp: float = 0.00015,
    run_label_suffix: str = "",
) -> tuple[list[str], str]:
    alpha_tag = label_number(alpha)
    run_label = f"clean_stage4_v3_phaseB_precision_a{alpha_tag}{run_label_suffix}"
    command = runner_base(
        root,
        str(continuation),
        "runs/stage4_inexact_optimized_resample/phases",
        run_label,
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
        "--stop-complete-iperp", f"{stop_complete_iperp:g}",
    ]
    return command, run_label


def find_result(
    directory: Path,
    run_label: str,
    budget: int,
    initial_path_sha256=None,
) -> Path:
    matches = sorted(
        directory.glob(
            f"ivm_{run_label}_central_state_ivm_cfg*_budget{budget}.npz"
        )
    )
    if initial_path_sha256 is not None:
        filtered = []
        for candidate in matches:
            with np.load(candidate, allow_pickle=False) as saved:
                if str(scalar(saved, "initial_path_sha256")) == initial_path_sha256:
                    filtered.append(candidate)
        matches = filtered
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one result for {run_label}; found {matches}"
        )
    return matches[0]


def cumulative_stored_cost(saved, offset: int = 0) -> np.ndarray:
    calls = np.asarray(saved["calls_per_image"], dtype=int)
    costs = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
    mapping = {int(call): int(cost) for call, cost in zip(calls, costs)}
    stored_calls = np.asarray(saved["stored_path_calls"], dtype=int)
    return np.asarray([mapping[int(call)] + offset for call in stored_calls])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--phase-a-budget", type=int, default=80)
    parser.add_argument("--phase-b-budget", type=int, default=220)
    parser.add_argument("--stop-complete-iperp", type=float, default=0.00015)
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional filename/run-label suffix for non-production audit runs.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    alpha = math.inf if args.alpha.lower() in {"inf", "infinity"} else float(args.alpha)
    alpha_tag = label_number(alpha)
    if args.output_suffix and not args.output_suffix.replace("_", "").isalnum():
        raise ValueError("--output-suffix must contain only letters, digits, or underscores")
    filename_suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    run_label_suffix = filename_suffix
    result_dir = root / "runs/stage4_inexact_optimized_resample"
    phase_dir = result_dir / "phases"
    continuation_dir = root / "inputs/stage4_uniform_v3_continuations"
    log_dir = root / "logs" / (
        f"stage4_uniform_v3_a{alpha_tag}_phaseA{args.phase_a_budget}_"
        f"phaseB{args.phase_b_budget}{filename_suffix}"
    )
    for directory in (result_dir, phase_dir, continuation_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    combined = result_dir / (
            f"stage4_uniform_v3_a{alpha_tag}_phaseA{args.phase_a_budget}_"
        f"phaseB{args.phase_b_budget}{filename_suffix}.npz"
    )
    audit_path = combined.with_suffix(".json")
    if combined.exists() and audit_path.exists() and not args.force:
        print(combined)
        return

    command_a, run_label_a = phase_a_command(root, alpha, args.phase_a_budget)
    (log_dir / "phaseA_command.json").write_text(json.dumps(command_a, indent=2) + "\n")
    wall_a = run_command(
        command_a, root, log_dir / "phaseA_stdout.txt", log_dir / "phaseA_stderr.txt"
    )
    result_a = find_result(phase_dir, run_label_a, args.phase_a_budget)

    continuation = continuation_dir / (
        f"phaseA{args.phase_a_budget}_terminal_a{alpha_tag}.npz"
    )
    with np.load(result_a, allow_pickle=False) as saved_a:
        terminal = np.asarray(saved_a["final_path"], dtype=float)
        initial_sha = str(scalar(saved_a, "initial_path_sha256"))
        configuration_a = str(scalar(saved_a, "configuration_sha256"))
        phase_a_final_cost = int(np.asarray(saved_a["total_internal_joint_evaluations"])[-1])
        phase_a_final_iperp = float(np.asarray(saved_a["i_perp"])[-1])
    np.savez_compressed(
        continuation,
        initial_path=terminal,
        path_sha256=np.array(array_sha256(terminal)),
        source_result=np.array(str(result_a)),
        source_result_sha256=np.array(file_sha256(result_a)),
        source_initial_path_sha256=np.array(initial_sha),
        source_configuration_sha256=np.array(configuration_a),
        source_final_force_evaluations=np.array(phase_a_final_cost),
        source_final_iperp=np.array(phase_a_final_iperp),
        continuation_policy=np.array(
            "fresh optimizer memory from accepted phase-A terminal path; "
            "phase-B initialization remains charged"
        ),
    )

    command_b, run_label_b = phase_b_command(
        root,
        alpha,
        args.phase_b_budget,
        continuation,
        stop_complete_iperp=args.stop_complete_iperp,
        run_label_suffix=run_label_suffix,
    )
    (log_dir / "phaseB_command.json").write_text(json.dumps(command_b, indent=2) + "\n")
    wall_b = run_command(
        command_b, root, log_dir / "phaseB_stdout.txt", log_dir / "phaseB_stderr.txt"
    )
    result_b = find_result(
        phase_dir,
        run_label_b,
        args.phase_b_budget,
        initial_path_sha256=array_sha256(terminal),
    )

    with np.load(result_a, allow_pickle=False) as a, np.load(result_b, allow_pickle=False) as b:
        cost_a = np.asarray(a["total_internal_joint_evaluations"], dtype=int)
        cost_b = np.asarray(b["total_internal_joint_evaluations"], dtype=int) + cost_a[-1]
        macro_a = np.asarray(a["calls_per_image"], dtype=int)
        macro_b = np.asarray(b["calls_per_image"], dtype=int) + macro_a[-1]
        i_perp = np.concatenate((a["i_perp"], b["i_perp"]))
        raw_rises = int(np.count_nonzero(np.diff(i_perp) > 1e-12))
        stored_cost_a = cumulative_stored_cost(a)
        stored_cost_b = cumulative_stored_cost(b, int(cost_a[-1]))
        stored_macro_a = np.asarray(a["stored_path_calls"], dtype=int)
        stored_macro_b = np.asarray(b["stored_path_calls"], dtype=int) + macro_a[-1]
        configuration_b = str(scalar(b, "configuration_sha256"))
        initial_b_sha = str(scalar(b, "initial_path_sha256"))
        phase_b_stop_reason = str(scalar(b, "stop_reason"))
        phase_b_stop_target = float(scalar(b, "stop_complete_iperp"))
        phase_b_stop_rule_fired = (
            phase_b_stop_reason == "complete_reparameterized_i_perp_target"
        )
        terminal_reparam_committed = bool(
            np.asarray(b["reparam_committed"], dtype=bool)[-1]
        )
        if initial_b_sha != array_sha256(terminal):
            raise RuntimeError("Phase-B initial-path hash does not match phase-A terminal path")
        if not np.all(np.diff(np.concatenate((cost_a, cost_b))) > 0):
            raise RuntimeError("Combined force counter is not strictly increasing")
        np.savez_compressed(
            combined,
            policy_version=np.array(POLICY_VERSION),
            alpha=np.array(alpha),
            alpha_mode=np.array("fixed_across_both_phases"),
            common_initial_path=np.array("inputs/common_initial_path_equidistant.npz"),
            common_initial_path_sha256=np.array(initial_sha),
            phase_labels=np.asarray(["fast_pathsum", "precision_resample"]),
            phase_a_budget=np.array(args.phase_a_budget),
            phase_b_budget=np.array(args.phase_b_budget),
            phase_a_result=np.array(str(result_a)),
            phase_b_result=np.array(str(result_b)),
            phase_a_result_sha256=np.array(file_sha256(result_a)),
            phase_b_result_sha256=np.array(file_sha256(result_b)),
            phase_a_configuration_sha256=np.array(configuration_a),
            phase_b_configuration_sha256=np.array(configuration_b),
            phase_transition_macro_index=np.array(len(cost_a)),
            phase_transition_force_offset=np.array(cost_a[-1]),
            macro_calls=np.concatenate((macro_a, macro_b)),
            total_force_evaluations=np.concatenate((cost_a, cost_b)),
            i_perp=i_perp,
            sparse_barrier_kcal=np.concatenate((a["barrier_kcal"], b["barrier_kcal"])),
            physical_fmax=np.concatenate((a["physical_fmax"], b["physical_fmax"])),
            segment_length_cv=np.concatenate((a["segment_length_cv"], b["segment_length_cv"])),
            equidistance_relative_defect=np.concatenate((a["equidistance_relative_defect"], b["equidistance_relative_defect"])),
            phase=np.concatenate((np.zeros(len(cost_a), dtype=int), np.ones(len(cost_b), dtype=int))),
            stored_force_evaluations=np.concatenate((stored_cost_a, stored_cost_b)),
            stored_macro_calls=np.concatenate((stored_macro_a, stored_macro_b)),
            stored_phase=np.concatenate((np.zeros(len(stored_cost_a), dtype=int), np.ones(len(stored_cost_b), dtype=int))),
            stored_paths=np.concatenate((a["stored_paths"], b["stored_paths"]), axis=0),
            final_path=np.asarray(b["final_path"], dtype=float),
            raw_iperp_rises=np.array(raw_rises),
            stop_complete_iperp=np.array(phase_b_stop_target),
            stop_rule_fired=np.array(phase_b_stop_rule_fired),
            stop_reason=np.array(phase_b_stop_reason),
            selected_phase_b_call=np.array(int(b["calls_per_image"][-1])),
            selected_phase_b_force_evaluations=np.array(
                int(b["total_internal_joint_evaluations"][-1])
            ),
            terminal_reparam_committed=np.array(terminal_reparam_committed),
            materialization_mode=np.array("independent_online_execution"),
            independent_online_rerun_status=np.array("confirmed"),
        )

    audit = {
        "policy_version": POLICY_VERSION,
        "alpha": "inf" if math.isinf(alpha) else alpha,
        "common_initial_path_sha256": initial_sha,
        "phase_a_result": str(result_a),
        "phase_a_result_sha256": file_sha256(result_a),
        "phase_b_result": str(result_b),
        "phase_b_result_sha256": file_sha256(result_b),
        "combined_result": str(combined),
        "combined_result_sha256": file_sha256(combined),
        "phase_a_wall_seconds": wall_a,
        "phase_b_wall_seconds": wall_b,
        "phase_a_final_force_evaluations": phase_a_final_cost,
        "phase_transition_initialization_charged": True,
        "combined_final_force_evaluations": int(cost_b[-1]),
        "combined_final_iperp": float(i_perp[-1]),
        "combined_min_iperp": float(np.min(i_perp)),
        "raw_iperp_rises": raw_rises,
        "phase_b_stop_complete_iperp": phase_b_stop_target,
        "phase_b_stop_rule_fired": phase_b_stop_rule_fired,
        "phase_b_stop_reason": phase_b_stop_reason,
        "terminal_reparam_committed": terminal_reparam_committed,
        "materialization_mode": "independent_online_execution",
        "independent_online_rerun_status": "confirmed",
        "fixed_alpha_in_both_phases": True,
        "cross_alpha_state_shared": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
