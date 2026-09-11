#!/usr/bin/env python3
"""Run one audited phase-B precision/stability pilot from a chosen continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import run_stage4_uniform_optimized as stage4


def replace_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def number_label(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def continuation_path_sha256(path: Path) -> str:
    with np.load(path, allow_pickle=False) as saved:
        initial_path = np.asarray(saved["initial_path"], dtype=float)
    return hashlib.sha256(np.ascontiguousarray(initial_path).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--alpha", default="20")
    parser.add_argument("--budget", type=int, default=320)
    parser.add_argument("--kkt-rtol", type=float, required=True)
    parser.add_argument("--trust", type=float, default=1.0)
    parser.add_argument("--cv-trigger", type=float, default=0.001)
    parser.add_argument("--target", type=float, default=1.5e-4)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    continuation = args.continuation.expanduser().resolve()
    if not continuation.exists():
        raise FileNotFoundError(continuation)
    alpha = math.inf if args.alpha.lower() in {"inf", "infinity"} else float(args.alpha)
    alpha_tag = stage4.label_number(alpha)
    command, _ = stage4.phase_b_command(root, alpha, args.budget, continuation)
    run_label = (
        f"clean_stage4_pilot_phaseB_precision_{args.label}_"
        f"r{number_label(args.kkt_rtol)}_trust{number_label(args.trust)}_"
        f"cv{number_label(args.cv_trigger)}_"
        f"a{alpha_tag}"
    )
    replace_option(command, "--output-dir", "runs/stage4_policy_pilots/phaseb_precision")
    replace_option(command, "--run-label", run_label)
    replace_option(command, "--plbfgs-kkt-rtol", f"{args.kkt_rtol:g}")
    replace_option(command, "--state-trust-fraction", f"{args.trust:g}")
    replace_option(command, "--state-polish-trust-fraction", f"{args.trust:g}")
    replace_option(command, "--reparam-cv-trigger", f"{args.cv_trigger:g}")
    replace_option(command, "--reparam-strict-cv-trigger", f"{args.cv_trigger:g}")
    replace_option(command, "--stop-complete-iperp", f"{args.target:g}")

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
        root / "runs/stage4_policy_pilots/phaseb_precision",
        run_label,
        args.budget,
        initial_path_sha256=continuation_path_sha256(continuation),
    )
    with np.load(result, allow_pickle=False) as saved:
        report = {
            "status": "completed",
            "policy": "fixed-alpha phase-B state-IVM precision/stability pilot",
            "alpha": "inf" if math.isinf(alpha) else alpha,
            "budget": args.budget,
            "kkt_rtol": args.kkt_rtol,
            "trust": args.trust,
            "cv_trigger": args.cv_trigger,
            "target": args.target,
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
