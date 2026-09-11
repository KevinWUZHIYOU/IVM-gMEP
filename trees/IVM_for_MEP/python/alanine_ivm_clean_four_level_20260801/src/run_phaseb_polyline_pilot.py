#!/usr/bin/env python3
"""Pilot the phase-B policy with curve-preserving polyline resampling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import run_stage4_uniform_optimized as stage4


def replace_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alpha", default="20")
    parser.add_argument("--budget", type=int, default=220)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    alpha = math.inf if args.alpha.lower() in {"inf", "infinity"} else float(args.alpha)
    alpha_tag = stage4.label_number(alpha)
    continuation = (
        root
        / "inputs/stage4_uniform_v3_continuations"
        / f"phaseA80_terminal_a{alpha_tag}.npz"
    )
    if not continuation.exists():
        raise FileNotFoundError(continuation)
    command, _ = stage4.phase_b_command(root, alpha, args.budget, continuation)
    run_label = f"clean_stage4_pilot_phaseB_polyline_a{alpha_tag}"
    replace_option(command, "--output-dir", "runs/stage4_policy_pilots/phaseb_polyline")
    replace_option(command, "--run-label", run_label)
    replace_option(command, "--reparam-mode", "state_polyline")
    log_dir = root / "logs/stage4_policy_pilots" / run_label
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    wall_seconds = stage4.run_command(
        command,
        root,
        log_dir / "stdout.txt",
        log_dir / "stderr.txt",
    )
    result = stage4.find_result(
        root / "runs/stage4_policy_pilots/phaseb_polyline",
        run_label,
        args.budget,
    )
    with np.load(result, allow_pickle=False) as saved:
        report = {
            "status": "completed",
            "policy": "phase-B fixed-alpha state-IVM with equal-arclength polyline resampling",
            "alpha": "inf" if math.isinf(alpha) else alpha,
            "budget": args.budget,
            "continuation": str(continuation),
            "result": str(result),
            "wall_seconds": wall_seconds,
            "terminal_force_evaluations": int(saved["total_internal_joint_evaluations"][-1]),
            "terminal_i_perp": float(saved["i_perp"][-1]),
            "minimum_i_perp": float(np.min(saved["i_perp"])),
            "raw_i_perp_rises": int(np.count_nonzero(np.diff(saved["i_perp"]) > 1e-12)),
            "stop_reason": str(np.asarray(saved["stop_reason"]).item()),
            "reparameterization_commits": int(np.count_nonzero(saved["reparam_committed"])),
        }
    (log_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
