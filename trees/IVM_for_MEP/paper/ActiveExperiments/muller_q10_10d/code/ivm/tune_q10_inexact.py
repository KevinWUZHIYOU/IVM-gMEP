#!/usr/bin/env python3
"""Grid-select one alpha-independent full-normal inexact IVM policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from q10_ivm_core import save_run
from run_q10_ivm_groups import ALPHAS


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_ALPHAS = (1e-4, 3e-4, 1e-3, 1.3e-3, 3e-3, 1e-2)
CHECKPOINTS = (10, 20, 50, 100, 150, 200, 300)


def policy_id(policy):
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def policies():
    for accepted_steps in (2, 4, 8):
        for kkt_rtol in (0.8, 0.5, 0.2):
            for resample_interval in (1, 3, 5):
                yield {
                    "accepted_step_cap": accepted_steps,
                    "new_call_cap": 4 * accepted_steps + 4,
                    "kkt_rtol": kkt_rtol,
                    "kkt_atol": 1e-8,
                    "min_accepted_steps": 2,
                    "history_size": 7,
                    "warm_history": True,
                    "harmonic_preconditioner": True,
                    "active_curvature": 0.0,
                    "max_step": 3.0,
                    "armijo": 1e-4,
                    "backtrack": 0.5,
                    "resample_interval": resample_interval,
                }


def make_config(policy, alpha, budget, group="inexact_tuned_resample"):
    return {
        "group": group,
        "solver": "full_normal_lbfgs",
        "alpha": float(alpha),
        **policy,
        "resample": True,
        "resample_mode": "cubic",
        "resample_passes": 3,
        "reset_history_after_resample": True,
        "budget_per_image": int(budget),
        "outer_limit": 10000,
        "movement_tolerance": 1e-16,
        "alpha_policy": "fixed",
        "tangent": "improved",
        "normal_dimension": 9,
        "policy_id": policy_id(policy),
    }


def worker(payload):
    root, config, role = payload
    return save_run(root, config, role)


def write_summary(role, rows):
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "{}_summary.json".format(role)).open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    flat = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "config"}
        item["config_json"] = json.dumps(row["config"], sort_keys=True)
        flat.append(item)
    with (reports / "{}_summary.csv".format(role)).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def run_parallel(role, configs, workers):
    payloads = [(str(ROOT), config, role) for config in configs]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 10 == 0 or index == len(futures):
                print(role, index, len(futures), flush=True)
    rows.sort(key=lambda row: (row["config"]["policy_id"], row["config"]["alpha"]))
    write_summary(role, rows)
    return rows


def checkpoint_value(x, y, checkpoint, stable, floor):
    if checkpoint > x[-1]:
        return float(y[-1]) if stable else math.inf
    return float(
        10.0
        ** np.interp(
            math.log10(checkpoint),
            np.log10(x),
            np.log10(np.maximum(y, floor)),
        )
    )


def score_run(row):
    with np.load(ROOT / row["output"], allow_pickle=False) as data:
        calls = np.asarray(data["calls_per_image"], dtype=float)
        i_perp = np.asarray(data["i_perp"], dtype=float)
        barrier = np.abs(
            np.asarray(data["delta_e"], dtype=float)
            - float(np.asarray(data["reference_barrier"])[0])
        )
    keep = (
        np.isfinite(calls)
        & np.isfinite(i_perp)
        & np.isfinite(barrier)
        & (calls > 0)
    )
    calls, i_perp, barrier = calls[keep], i_perp[keep], barrier[keep]
    indices = np.unique(calls, return_index=True)[1]
    indices.sort()
    calls, i_perp, barrier = calls[indices], i_perp[indices], barrier[indices]
    stable = row["stop_reason"] == "stationary"
    values = []
    for checkpoint in CHECKPOINTS:
        ip = checkpoint_value(calls, i_perp, checkpoint, stable, 1e-30)
        be = checkpoint_value(calls, barrier, checkpoint, stable, 1e-12)
        if not np.isfinite(ip) or not np.isfinite(be):
            values.append(40.0)
        else:
            values.append(math.log10(max(ip, 1e-30)) + math.log10(max(be, 1e-12)))
    return float(np.mean(values)), values


def select(rows):
    candidates = []
    for row in rows:
        score, checkpoint_scores = score_run(row)
        row["selection_score"] = score
        row["checkpoint_scores"] = checkpoint_scores
        if (
            row["stop_reason"] not in ("certification_failure", "numerical_failure")
            and row["alpha_fixed_audit"]
            and row["full_normal_dimension_audit"]
        ):
            candidates.append(row)
    if not candidates:
        raise RuntimeError("no valid tuned IVM candidate")
    candidates.sort(key=lambda row: (row["selection_score"], row["final_i_perp"]))
    winner = candidates[0]
    policy_keys = tuple(policies().__next__().keys())
    policy = {key: winner["config"][key] for key in policy_keys}
    selection = {
        "rule": "minimum mean checkpoint log10(I_perp)+log10(dense barrier error); unfinished nonstationary checkpoints receive a fixed penalty",
        "checkpoints_calls_per_image": CHECKPOINTS,
        "selected_policy_id": winner["config"]["policy_id"],
        "selected_anchor_alpha": winner["config"]["alpha"],
        "selected_score": winner["selection_score"],
        "selected_policy": policy,
        "top_candidates": [
            {
                "policy_id": row["config"]["policy_id"],
                "alpha": row["config"]["alpha"],
                "selection_score": row["selection_score"],
                "final_i_perp": row["final_i_perp"],
                "final_barrier_error": row["final_barrier_error"],
                "stop_reason": row["stop_reason"],
            }
            for row in candidates[:30]
        ],
    }
    with (ROOT / "reports" / "q10_tuned_policy_selection.json").open("w") as handle:
        json.dump(selection, handle, indent=2, sort_keys=True)
    write_summary("q10_tuned_policy_screen_scored", rows)
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "production", "all"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage in ("screen", "all"):
        configs = [
            make_config(policy, alpha, 300)
            for policy in policies()
            for alpha in ANCHOR_ALPHAS
        ]
        rows = run_parallel("q10_tuned_policy_screen", configs, args.workers)
        selection = select(rows)
        print("selected", selection, flush=True)
    if args.stage in ("production", "all"):
        if args.stage == "production":
            with (ROOT / "reports" / "q10_tuned_policy_selection.json").open() as handle:
                selection = json.load(handle)
        configs = [
            make_config(selection["selected_policy"], alpha, 2000)
            for alpha in ALPHAS
        ]
        run_parallel("q10_inexact_tuned_resample", configs, args.workers)


if __name__ == "__main__":
    main()
