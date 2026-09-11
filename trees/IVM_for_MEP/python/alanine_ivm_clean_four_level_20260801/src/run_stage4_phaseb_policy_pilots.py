#!/usr/bin/env python3
"""Controlled alpha=20 pilots for the stage-4 precision-resampling phase."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time


VARIANTS = {
    "curve_eq075_perimage": ("state_curve", "per_image", "0.75", 1),
    "curve_eq05_perimage": ("state_curve", "per_image", "0.5", 1),
    "curve_eq075_pathsum": ("state_curve", "path_sum", "0.75", 1),
    "projected_eq075_pathsum": ("state_curve_projected", "path_sum", "0.75", 1),
    "projected_eq05_pathsum": ("state_curve_projected", "path_sum", "0.5", 1),
}

ADDITIONAL_VARIANTS = {
    "curve_eq095_perimage_e1": ("state_curve", "per_image", "0.95", 1),
    "curve_eq075_perimage_e5": ("state_curve", "per_image", "0.75", 5),
    "curve_eq05_perimage_e5": ("state_curve", "per_image", "0.5", 5),
    "geometric_e1": ("geometric", "per_image", "0.75", 1),
    "geometric_e5": ("geometric", "per_image", "0.75", 5),
}


def command(root: Path, variant: str, specification, budget: int) -> list[str]:
    mode, objective, contraction, reparam_every = specification
    continuation = root / "inputs/stage4_uniform_v2_continuations/phaseA_terminal_a20.npz"
    runner = root / "src/run_ivm_v9_spring_projected_production_20260730.py"
    command = [
        sys.executable, str(runner),
        "--root", str(root),
        "--initial-path-file", str(continuation),
        "--output-dir", "runs/stage4_inexact_optimized_resample/pilots",
        "--run-label", f"clean_stage4_phaseB_pilot_{variant}_a20",
        "--solver", "state_ivm",
        "--alpha", "20",
        "--alpha-mode", "fixed",
        "--alpha-min", "0.01",
        "--alpha-max", "inf",
        "--max-step", "1",
        "--history", "7",
        "--stop-iperp", "-1",
        "--budget", str(budget),
        "--save-all-paths",
        "--state-globalization", "iperp_monotone",
        "--state-trust-fraction", "2",
        "--state-polish-trust-fraction", "2",
        "--state-polish-alpha-factor", "0.5",
        "--inner-calls", "15",
        "--plbfgs-kkt-rtol", "0.8",
        "--reparam-mode", mode,
        "--reparam-every", str(reparam_every),
        "--reparam-ready-relative-iperp", "2",
        "--reparam-cv-trigger", "0.001",
        "--reparam-strict-cv-trigger", "0.001",
        "--reparam-retry-reduction", "0.8",
        "--reparam-recovery-calls", "320",
        "--reparam-recovery-inner-calls", "15",
        "--reparam-recovery-merge", "bicriteria",
        "--reparam-recovery-iperp-rtol", "0",
        "--reparam-equidistance-contraction", contraction,
    ]
    if mode != "geometric":
        command += [
            "--state-curve-direct-solve",
            "--state-curve-direct-trial-calls", "15",
            "--state-curve-direct-objective-acceptance", objective,
            "--state-curve-immediate-block-relock",
            "--state-curve-stall-free-polish",
            "--state-curve-stall-free-polish-count", "1",
        ]
    return command


def run_one(root: Path, variant: str, specification, budget: int):
    log_dir = root / "logs/stage4_phaseb_pilots" / variant
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = command(root, variant, specification, budget)
    (log_dir / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    started = time.time()
    completed = subprocess.run(
        cmd, cwd=root, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False
    )
    (log_dir / "stdout.txt").write_text(completed.stdout)
    (log_dir / "stderr.txt").write_text(completed.stderr)
    matches = sorted(
        (root / "runs/stage4_inexact_optimized_resample/pilots").glob(
            f"ivm_clean_stage4_phaseB_pilot_{variant}_a20_central_state_ivm_cfg*_budget{budget}.npz"
        )
    )
    payload = {
        "variant": variant,
        "status": "completed" if completed.returncode == 0 and len(matches) == 1 else "failed",
        "returncode": completed.returncode,
        "result": str(matches[0]) if len(matches) == 1 else None,
        "wall_seconds": time.time() - started,
    }
    (log_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=220)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--suite", choices=("original", "additional"), default="original")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not (root / "inputs/stage4_uniform_v2_continuations/phaseA_terminal_a20.npz").exists():
        raise RuntimeError("Run stage4 uniform v2 alpha=20 first")
    variants = VARIANTS if args.suite == "original" else ADDITIONAL_VARIANTS
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(run_one, root, variant, specification, args.budget)
            for variant, specification in variants.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            payload = future.result()
            results.append(payload)
            print(json.dumps(payload, sort_keys=True), flush=True)
    output = root / "reports/stage4_phaseb_policy_pilots.json"
    output = output.with_name(f"{output.stem}_{args.suite}.json")
    output.write_text(json.dumps({"variants": variants, "results": results}, indent=2) + "\n")
    if any(item["status"] == "failed" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
