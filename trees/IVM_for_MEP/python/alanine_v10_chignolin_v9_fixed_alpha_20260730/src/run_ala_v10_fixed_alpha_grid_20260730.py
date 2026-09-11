#!/usr/bin/env python3
"""Run one selected, fixed production policy over the declared Ala alpha grid."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run_ala_v10_policy_20260730 import POLICIES


DEFAULT_ALPHAS = (
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    math.inf,
)


def alpha_label(alpha: float) -> str:
    if math.isinf(alpha):
        return "inf"
    return f"{alpha:g}".replace(".", "p").replace("-", "m")


def existing_result(
    root: Path,
    policy: str,
    alpha: float,
    budget: int,
    study_label: str,
):
    matches = sorted(
        (root / "runs" / "ivm").glob(
            f"ivm_{study_label}_{policy}_a{alpha_label(alpha)}_central_state_ivm_*_budget{budget}.npz"
        )
    )
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous existing result for alpha={alpha}: {matches}")
    return matches[0] if matches else None


def run_one(
    root: Path,
    policy: str,
    alpha: float,
    budget: int,
    initial_path_file: str,
    study_label: str,
) -> dict:
    prior = existing_result(root, policy, alpha, budget, study_label)
    if prior is not None:
        return {
            "alpha": "inf" if math.isinf(alpha) else alpha,
            "status": "skipped_existing",
            "result": str(prior),
            "returncode": 0,
            "wall_seconds": 0.0,
        }
    logs = root / "logs" / "fixed_alpha_grid" / study_label
    logs.mkdir(parents=True, exist_ok=True)
    token = alpha_label(alpha)
    log = logs / f"{policy}_a{token}_budget{budget}.log"
    command = [
        sys.executable,
        str(root / "src" / "run_ala_v10_policy_20260730.py"),
        "--root", str(root),
        "--policy", policy,
        "--alpha", "inf" if math.isinf(alpha) else f"{alpha:g}",
        "--budget", str(budget),
        "--initial-path-file", initial_path_file,
        "--study-label", study_label,
    ]
    environment = os.environ.copy()
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENMM_CPU_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    })
    started = time.time()
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, cwd=root, env=environment, stdout=stream,
            stderr=subprocess.STDOUT, check=False,
        )
    result = existing_result(root, policy, alpha, budget, study_label)
    return {
        "alpha": "inf" if math.isinf(alpha) else alpha,
        "status": "completed" if completed.returncode == 0 else "failed",
        "result": None if result is None else str(result),
        "returncode": int(completed.returncode),
        "wall_seconds": time.time() - started,
        "log": str(log),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--policy",
        choices=tuple(POLICIES),
        required=True,
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--budget", type=int, default=650)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--initial-path-file",
        default="inputs/common_initial_path_equidistant.npz",
    )
    parser.add_argument("--study-label", default="ala_v10")
    args = parser.parse_args()
    if args.max_workers < 1:
        raise ValueError("max-workers must be positive")
    if not args.study_label or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in args.study_label
    ):
        raise ValueError("study-label must contain only letters, digits, '_' or '-'")
    root = Path(args.root).expanduser().resolve()
    started = time.time()
    reports = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(
                run_one,
                root,
                args.policy,
                alpha,
                args.budget,
                args.initial_path_file,
                args.study_label,
            )
            for alpha in args.alphas
        ]
        for future in concurrent.futures.as_completed(futures):
            report = future.result()
            reports.append(report)
            print(
                f"alpha={report['alpha']} status={report['status']} "
                f"returncode={report['returncode']} "
                f"wall_seconds={report['wall_seconds']:.1f}",
                flush=True,
            )
    reports.sort(key=lambda item: math.inf if item["alpha"] == "inf" else float(item["alpha"]))
    manifest = {
        "policy": args.policy,
        "study_label": args.study_label,
        "initial_path_file": args.initial_path_file,
        "only_grid_variable": "alpha",
        "budget_per_internal_image": int(args.budget),
        "max_workers": int(args.max_workers),
        "wall_seconds": time.time() - started,
        "runs": reports,
    }
    output = (
        root
        / "logs"
        / "fixed_alpha_grid"
        / args.study_label
        / f"{args.policy}_manifest_budget{args.budget}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if any(item["returncode"] != 0 for item in reports):
        raise SystemExit("one or more grid members failed")
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
