#!/usr/bin/env python3
"""Densify common, explicitly selected checkpoints for the clean audit grid."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


STAGE_GLOBS = {
    "stage1": "runs/stage1_exact_no_resample/*budget160.npz",
    "stage2": "runs/stage2_exact_resample/*budget160.npz",
    "stage3": "runs/stage3_inexact_baseline_resample/*budget220.npz",
    "stage4": "runs/stage4_inexact_optimized_resample/stage4_uniform_v3_a*_phaseA80_phaseB220.npz",
}


def number_label(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:g}".replace(".", "p").replace("-", "m")


def selected_labels(result: Path, stage: str) -> tuple[str, str, list[int], float, float]:
    with np.load(result, allow_pickle=False) as saved:
        alpha = float(np.asarray(saved["alpha"]).item())
        if stage == "stage4":
            rtol = 0.8
            all_labels = np.asarray(saved["stored_force_evaluations"], dtype=int)
            metric_cost = np.asarray(saved["total_force_evaluations"], dtype=int)
            metric = np.asarray(saved["i_perp"], dtype=float)
            calls_key = "stored_force_evaluations"
        else:
            rtol = float(np.asarray(saved["plbfgs_kkt_rtol"]).item())
            all_labels = np.asarray(saved["stored_path_calls"], dtype=int)
            metric_cost = np.asarray(saved["calls_per_image"], dtype=int)
            metric = np.asarray(saved["i_perp"], dtype=float)
            calls_key = "stored_path_calls"
    unique_labels = np.unique(all_labels)
    targets = np.geomspace(max(1, unique_labels[0]), unique_labels[-1], 12)
    selected = {
        int(unique_labels[np.argmin(np.abs(unique_labels - target))])
        for target in targets
    }
    selected.update((int(unique_labels[0]), int(unique_labels[-1])))
    thresholds = (1.0, 0.1, 0.01, 0.003, 0.001, 5e-4, 2.1e-4, 1.5e-4)
    for threshold in thresholds:
        indices = np.flatnonzero(metric <= threshold)
        if indices.size:
            target_cost = int(metric_cost[indices[0]])
            selected.add(
                int(unique_labels[np.argmin(np.abs(unique_labels - target_cost))])
            )
    min_index = int(np.argmin(metric))
    selected.add(
        int(unique_labels[np.argmin(np.abs(unique_labels - int(metric_cost[min_index])))])
    )
    return calls_key, "stored_paths", sorted(selected), alpha, rtol


def run_one(root: Path, stage: str, result: Path, force: bool):
    calls_key, paths_key, calls, alpha, rtol = selected_labels(result, stage)
    policy_label = "_v3" if stage == "stage4" else ""
    output_label = (
        f"clean_{stage}{policy_label}_r{number_label(rtol)}_a{number_label(alpha)}_"
        "common_checkpoints"
    )
    output = root / "runs/diagnostics" / f"dense_barrier_{output_label}.npz"
    if output.exists() and not force:
        return {"stage": stage, "alpha": alpha, "rtol": rtol, "status": "skipped", "output": str(output)}
    command = [
        sys.executable,
        str(root / "src/postprocess_dense_barrier.py"),
        "--root", str(root),
        "--result", str(result),
        "--output-label", output_label,
        "--workers", "1",
        "--calls-key", calls_key,
        "--paths-key", paths_key,
        "--calls", ",".join(str(value) for value in calls),
    ]
    log_dir = root / "logs" / "dense_audit" / output_label
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (log_dir / "stdout.txt").write_text(completed.stdout)
    (log_dir / "stderr.txt").write_text(completed.stderr)
    payload = {
        "stage": stage,
        "alpha": "inf" if math.isinf(alpha) else alpha,
        "rtol": rtol,
        "status": "completed" if completed.returncode == 0 and output.exists() else "failed",
        "returncode": completed.returncode,
        "source": str(result),
        "output": str(output),
        "selected_labels": calls,
        "wall_seconds": time.time() - started,
    }
    (log_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGE_GLOBS) + ("all",), default="all")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    stages = tuple(STAGE_GLOBS) if args.stage == "all" else (args.stage,)
    tasks = []
    for stage in stages:
        results = sorted(root.glob(STAGE_GLOBS[stage]))
        if not results:
            raise RuntimeError(f"No results found for {stage}")
        tasks.extend((root, stage, result, args.force) for result in results)
    report = {
        "protocol": "common log-cost and I_perp first-passage checkpoints",
        "started_unix": time.time(),
        "tasks": len(tasks),
        "results": [],
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(run_one, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            payload = future.result()
            report["results"].append(payload)
            print(json.dumps(payload, sort_keys=True), flush=True)
    report["finished_unix"] = time.time()
    report["failed"] = sum(item["status"] == "failed" for item in report["results"])
    output = root / "reports/dense_audit_grid.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
