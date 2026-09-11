#!/usr/bin/env python3
"""Run every fixed alpha through the entire Stage-IV audit budget.

The optimizer and resampling policy are identical to the production Stage-IV
protocol.  Only the complete-I_perp early-stop threshold is disabled so the
full A34/B440 dynamics can be compared without unequal censoring.  Results are
stored with a ``_fullbudget`` suffix and never overwrite production runs.
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

import numpy as np


ALPHAS = (0.01, 0.1, 1.0, 10.0, 20.0, 30.0, 100.0, 1000.0, 10000.0, math.inf)


def tag(alpha: float) -> str:
    return "inf" if math.isinf(alpha) else f"{alpha:g}".replace(".", "p")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(saved, key):
    return np.asarray(saved[key]).item()


def run_one(root: Path, alpha: float, phase_a: int, phase_b: int):
    output = root / "runs/stage4_inexact_optimized_resample" / (
        f"stage4_uniform_v3_a{tag(alpha)}_phaseA{phase_a}_phaseB{phase_b}_fullbudget.npz"
    )
    command = [
        sys.executable,
        str(root / "src/run_stage4_uniform_optimized.py"),
        "--root",
        str(root),
        "--alpha",
        "inf" if math.isinf(alpha) else f"{alpha:g}",
        "--phase-a-budget",
        str(phase_a),
        "--phase-b-budget",
        str(phase_b),
        "--stop-complete-iperp",
        "-1",
        "--output-suffix",
        "fullbudget",
    ]
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
    result = {
        "alpha": "inf" if math.isinf(alpha) else alpha,
        "command": command,
        "returncode": completed.returncode,
        "wall_seconds": time.time() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "result": str(output),
        "result_exists": output.exists(),
    }
    if completed.returncode == 0 and output.exists():
        with np.load(output, allow_pickle=False) as saved:
            result.update(
                {
                    "result_sha256": sha256(output),
                    "terminal_force": int(saved["total_force_evaluations"][-1]),
                    "terminal_i_perp": float(saved["i_perp"][-1]),
                    "minimum_i_perp": float(np.min(saved["i_perp"])),
                    "stop_reason": str(scalar(saved, "stop_reason")),
                    "stop_complete_iperp": float(scalar(saved, "stop_complete_iperp")),
                    "raw_iperp_rises": int(scalar(saved, "raw_iperp_rises")),
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a", type=int, default=34)
    parser.add_argument("--phase-b", type=int, default=440)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_one, root, alpha, args.phase_a, args.phase_b): alpha
            for alpha in ALPHAS
        }
        results = []
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            results.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)
    results.sort(key=lambda row: math.inf if row["alpha"] == "inf" else float(row["alpha"]))
    report = {
        "status": "complete" if all(item["returncode"] == 0 for item in results) else "failed",
        "purpose": "uncensored equal-budget Stage-IV alpha audit",
        "optimizer_policy": "same uniform clean Stage-IV fixed-alpha A34/B440 policy",
        "only_change": "complete-I_perp early stop disabled (threshold=-1)",
        "phase_a_budget": args.phase_a,
        "phase_b_budget": args.phase_b,
        "alpha_grid": ["inf" if math.isinf(alpha) else alpha for alpha in ALPHAS],
        "started_unix": started,
        "finished_unix": time.time(),
        "results": results,
    }
    output = root / "reports" / (
        f"launch_stage4_fullbudget_A{args.phase_a}_B{args.phase_b}_{int(started)}.json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
