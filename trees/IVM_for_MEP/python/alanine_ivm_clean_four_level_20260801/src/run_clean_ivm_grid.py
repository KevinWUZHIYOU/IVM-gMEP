#!/usr/bin/env python3
"""Run the clean-start four-level fixed-alpha Ala IVM audit grid.

Each run starts from the same frozen input NPZ.  No path, force cache, or
L-BFGS history is shared between alpha members.  Any history option is strictly
run-local and is reset when a new subprocess starts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 20.0, 30.0, 100.0, 1000.0, 10000.0, math.inf)
EXACT_RTOLS = (0.2, 0.1)

STAGES = {
    "stage1": {
        "output_dir": "runs/stage1_exact_no_resample",
        "budget": 160,
        "rtols": EXACT_RTOLS,
        "description": "exact-like frozen-normal IVM; no reparameterization",
    },
    "stage2": {
        "output_dir": "runs/stage2_exact_resample",
        "budget": 160,
        "rtols": EXACT_RTOLS,
        "description": "exact-like frozen-normal IVM; geometric equal-arclength reparameterization every outer step",
    },
    "stage3": {
        "output_dir": "runs/stage3_inexact_baseline_resample",
        "budget": 220,
        "rtols": (0.8,),
        "description": "baseline inexact IVM; at most five inner iterations; geometric resampling",
    },
    "stage4": {
        "output_dir": "runs/stage4_inexact_optimized_resample",
        "budget": 300,
        "rtols": (0.8,),
        "description": "uniform optimized two-phase inexact IVM; fast projected path-sum phase then precision resampling",
    },
}


def alpha_label(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:g}".replace(".", "p").replace("-", "m")


def number_label(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def common_command(root: Path, stage: str, alpha: float, rtol: float, budget: int) -> list[str]:
    if stage == "stage4":
        phase_a_budget = 80
        phase_b_budget = budget - phase_a_budget
        if phase_b_budget <= 0:
            raise ValueError("Stage 4 total budget must exceed its fixed 80-call phase A")
        return [
            sys.executable,
            str(root / "src/run_stage4_uniform_optimized.py"),
            "--root", str(root),
            "--alpha", "inf" if math.isinf(alpha) else f"{alpha:g}",
            "--phase-a-budget", str(phase_a_budget),
            "--phase-b-budget", str(phase_b_budget),
        ]
    runner = root / "src/run_ivm_v9_spring_projected_production_20260730.py"
    run_label = f"clean_{stage}_r{number_label(rtol)}_a{alpha_label(alpha)}"
    spec = STAGES[stage]
    command = [
        sys.executable,
        str(runner),
        "--root", str(root),
        "--initial-path-file", "inputs/common_initial_path_equidistant.npz",
        "--output-dir", spec["output_dir"],
        "--run-label", run_label,
        "--solver", "state_ivm",
        "--alpha", "inf" if math.isinf(alpha) else f"{alpha:g}",
        "--alpha-mode", "fixed",
        "--alpha-min", "0.01",
        "--alpha-max", "inf",
        "--max-step", "1",
        "--state-polish-alpha-factor", "1",
        "--history", "7",
        "--plbfgs-kkt-rtol", f"{rtol:g}",
        "--stop-iperp", "-1",
        "--budget", str(budget),
        "--save-all-paths",
    ]
    if stage in {"stage1", "stage2"}:
        command += [
            "--state-globalization", "proximal_only",
            "--state-trust-fraction", "2",
            "--state-polish-trust-fraction", "2",
            "--inner-calls", "80",
            "--plbfgs-min-accepted-steps", "2",
            "--reparam-mode", "geometric",
            "--reparam-every", "0" if stage == "stage1" else "1",
        ]
    elif stage == "stage3":
        command += [
            "--state-globalization", "proximal_only",
            "--state-trust-fraction", "2",
            "--state-polish-trust-fraction", "2",
            "--inner-calls", "5",
            "--plbfgs-min-accepted-steps", "3",
            "--reparam-mode", "geometric",
            "--reparam-every", "1",
        ]
    else:  # pragma: no cover
        raise ValueError(stage)
    return command


def expected_glob(stage: str, rtol: float, alpha: float, budget: int) -> str:
    if stage == "stage4":
        return (
            f"stage4_uniform_v3_a{alpha_label(alpha)}_phaseA80_"
            f"phaseB{budget - 80}.npz"
        )
    run_label = f"clean_{stage}_r{number_label(rtol)}_a{alpha_label(alpha)}"
    return f"ivm_{run_label}_central_state_ivm_cfg*_budget{budget}.npz"


def run_one(root: Path, stage: str, alpha: float, rtol: float, budget: int, force: bool):
    spec = STAGES[stage]
    output_dir = root / spec["output_dir"]
    matches = sorted(output_dir.glob(expected_glob(stage, rtol, alpha, budget)))
    if matches and not force:
        return {
            "stage": stage,
            "alpha": alpha,
            "rtol": rtol,
            "budget": budget,
            "status": "skipped_existing",
            "result": str(matches[-1]),
            "result_sha256": sha256(matches[-1]),
        }

    command = common_command(root, stage, alpha, rtol, budget)
    label = f"{stage}_r{number_label(rtol)}_a{alpha_label(alpha)}_b{budget}"
    log_dir = root / "logs" / label
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
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
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.time() - started
    (log_dir / "stdout.txt").write_text(completed.stdout)
    (log_dir / "stderr.txt").write_text(completed.stderr)
    (log_dir / "exit_code.txt").write_text(f"{completed.returncode}\n")
    matches = sorted(output_dir.glob(expected_glob(stage, rtol, alpha, budget)))
    result = matches[-1] if matches else None
    payload = {
        "stage": stage,
        "alpha": alpha,
        "rtol": rtol,
        "budget": budget,
        "status": "completed" if completed.returncode == 0 and result else "failed",
        "returncode": completed.returncode,
        "wall_seconds": elapsed,
        "result": None if result is None else str(result),
        "result_sha256": None if result is None else sha256(result),
        "log_dir": str(log_dir),
    }
    (log_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def parse_number_list(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip().lower()
        values.append(math.inf if item in {"inf", "infinity"} else float(item))
    return values


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGES) + ("all",), required=True)
    parser.add_argument("--alphas", default=",".join("inf" if math.isinf(x) else f"{x:g}" for x in DEFAULT_ALPHAS))
    parser.add_argument("--rtols", default="", help="Override the stage r grid")
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stages = tuple(STAGES) if args.stage == "all" else (args.stage,)
    alphas = parse_number_list(args.alphas)
    tasks = []
    for stage in stages:
        rtols = parse_number_list(args.rtols) if args.rtols else STAGES[stage]["rtols"]
        budget = args.budget or STAGES[stage]["budget"]
        for rtol in rtols:
            for alpha in alphas:
                tasks.append((root, stage, alpha, rtol, budget, args.force))

    manifest = {
        "protocol": "clean-start four-level fixed-alpha Ala IVM audit",
        "root": str(root),
        "common_initial_path": "inputs/common_initial_path_equidistant.npz",
        "common_initial_file_sha256": sha256(root / "inputs/common_initial_path_equidistant.npz"),
        "runner_sha256": sha256(root / "src/run_ivm_v9_spring_projected_production_20260730.py"),
        "orchestrator_sha256": sha256(Path(__file__)),
        "stages": {stage: STAGES[stage] for stage in stages},
        "alphas": alphas,
        "tasks": len(tasks),
        "jobs": args.jobs,
        "started_unix": time.time(),
    }
    manifest_path = root / "reports" / f"launch_{args.stage}_{int(manifest['started_unix'])}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        future_to_task = {pool.submit(run_one, *task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            payload = future.result()
            results.append(payload)
            print(json.dumps(payload, sort_keys=True), flush=True)

    results.sort(key=lambda item: (item["stage"], item["rtol"], item["alpha"]))
    manifest["finished_unix"] = time.time()
    manifest["results"] = results
    manifest["successful"] = sum(item["status"] in {"completed", "skipped_existing"} for item in results)
    manifest["failed"] = sum(item["status"] == "failed" for item in results)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {manifest_path}")
    if manifest["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
