#!/usr/bin/env python3
"""Apply one charged reference-free Pareto screen to all full-budget runs."""

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


ALPHAS = (0.01, 0.1, 1.0, 10.0, 20.0, 30.0, 100.0, 1000.0, 10000.0, math.inf)


def tag(alpha: float) -> str:
    return "inf" if math.isinf(alpha) else f"{alpha:g}".replace(".", "p")


def build_one(
    root: Path,
    alpha: float,
    phase_a: int,
    phase_b: int,
    label_prefix: str,
    continue_after_target: bool,
):
    source = root / "runs/stage4_inexact_optimized_resample" / (
        f"stage4_uniform_v3_a{tag(alpha)}_phaseA{phase_a}_phaseB{phase_b}_fullbudget.npz"
    )
    label = f"{label_prefix}_a{tag(alpha)}_phaseA{phase_a}_phaseB{phase_b}"
    output = root / "runs/pareto_archive" / f"pareto_{label}.npz"
    command = [
        sys.executable,
        str(root / "src/build_pareto_archive.py"),
        "--root",
        str(root),
        "--result",
        str(source),
        "--output-label",
        label,
    ]
    if continue_after_target:
        command.append("--continue-after-target")
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
    audit = output.with_suffix(".audit.json")
    result = {
        "alpha": "inf" if math.isinf(alpha) else alpha,
        "source": str(source),
        "output": str(output),
        "audit": str(audit),
        "returncode": completed.returncode,
        "wall_seconds": time.time() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_exists": output.exists(),
        "audit_exists": audit.exists(),
    }
    if audit.exists():
        result["audit_record"] = json.loads(audit.read_text())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a", type=int, default=34)
    parser.add_argument("--phase-b", type=int, default=440)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--label-prefix", default="clean_stage4_v6_fullbudget")
    parser.add_argument("--continue-after-target", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                build_one,
                root,
                alpha,
                args.phase_a,
                args.phase_b,
                args.label_prefix,
                args.continue_after_target,
            ): alpha
            for alpha in ALPHAS
        }
        results = []
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            results.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)
    results.sort(key=lambda row: math.inf if row["alpha"] == "inf" else float(row["alpha"]))
    passed = all(
        item["returncode"] == 0 and item["output_exists"] and item["audit_exists"]
        for item in results
    )
    report = {
        "status": "complete" if passed else "failed",
        "policy": "identical reference-free charged Pareto archive on uncensored runs",
        "label_prefix": args.label_prefix,
        "continue_after_target": args.continue_after_target,
        "phase_a_budget": args.phase_a,
        "phase_b_budget": args.phase_b,
        "started_unix": started,
        "finished_unix": time.time(),
        "results": results,
    }
    safe_prefix = args.label_prefix.replace("/", "_")
    output = root / "reports" / (
        f"build_{safe_prefix}_pareto_A{args.phase_a}_B{args.phase_b}_{int(started)}.json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
