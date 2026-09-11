#!/usr/bin/env python3
"""Pilot a fresh-memory continuation of the projected-curve phase-A policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import run_stage4_uniform_optimized as stage4


def replace_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alpha", default="20")
    parser.add_argument("--budget", type=int, default=160)
    parser.add_argument("--continuation", type=Path)
    parser.add_argument("--label", default="phaseA_terminal")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    alpha = math.inf if args.alpha.lower() in {"inf", "infinity"} else float(args.alpha)
    alpha_tag = stage4.label_number(alpha)
    continuation = (
        args.continuation.expanduser().resolve()
        if args.continuation is not None
        else root
        / "inputs/stage4_uniform_v3_continuations"
        / f"phaseA80_terminal_a{alpha_tag}.npz"
    )
    if not continuation.exists():
        raise FileNotFoundError(continuation)
    command, _ = stage4.phase_a_command(root, alpha, args.budget)
    run_label = (
        f"clean_stage4_pilot_projected_restart_{args.label}_a{alpha_tag}"
    )
    replace_option(command, "--initial-path-file", str(continuation))
    replace_option(command, "--output-dir", "runs/stage4_policy_pilots/projected_restart")
    replace_option(command, "--run-label", run_label)
    command += ["--stop-complete-iperp", "0.00015"]
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
        root / "runs/stage4_policy_pilots/projected_restart",
        run_label,
        args.budget,
    )
    report = {
        "status": "completed",
        "policy": "fresh-memory restart of phase-A projected-curve path-sum IVM",
        "alpha": "inf" if math.isinf(alpha) else alpha,
        "budget": args.budget,
        "continuation": str(continuation),
        "result": str(result),
        "wall_seconds": wall_seconds,
    }
    (log_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
