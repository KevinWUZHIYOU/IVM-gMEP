#!/usr/bin/env python3
"""Run the locked alpha-independent tuned inexact IVM policy."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from q10_ivm_core import save_run


ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (
    1e-4, 3e-4, 6e-4, 8e-4, 1e-3, 1.1e-3, 1.2e-3, 1.25e-3,
    1.3e-3, 1.35e-3, 1.4e-3, 1.5e-3, 3e-3, 1e-2, 3e-2,
    1e-1, 3e-1, 1.0, 3.0, 1e1, 1e2, 1e3, 1e4,
)


def configs():
    for alpha in ALPHAS:
        yield {
            "group": "inexact_tuned_resample",
            "solver": "full_normal_lbfgs",
            "alpha": alpha,
            "accepted_step_cap": 1,
            "new_call_cap": 12,
            "kkt_rtol": None,
            "kkt_atol": 1e-14,
            "min_accepted_steps": 1,
            "history_size": 15,
            "warm_history": True,
            "harmonic_preconditioner": True,
            "active_curvature": 0.0,
            "initial_max_step": 0.065,
            "initial_max_step_outer_limit": 5,
            "max_step": 0.5,
            "armijo": 1e-2,
            "backtrack": 0.5,
            "resample": True,
            "resample_mode": "cubic",
            "resample_interval": 5,
            "resample_spacing_trigger": -1.0,
            "resample_passes": 1,
            "reset_history_after_resample": False,
            "budget_per_image": 700,
            "outer_limit": 10000,
            "movement_tolerance": 1e-14,
            "alpha_policy": "fixed",
            "tangent": "improved",
            "normal_dimension": 9,
        }


def worker(config):
    return save_run(ROOT, config, "inexact_tuned_resample")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, config): config for config in configs()}
        for future in as_completed(futures):
            row = future.result(); rows.append(row); c = row["config"]
            print(
                "alpha={:.3e} calls={} I={:.6e} dE={:.6e} stop={}".format(
                    c["alpha"], row["final_calls_per_image"], row["final_i_perp"],
                    row["final_barrier_error"], row["stop_reason"]), flush=True,
            )
    rows.sort(key=lambda row: row["config"]["alpha"])
    reports = ROOT / "reports"; reports.mkdir(parents=True, exist_ok=True)
    with (reports / "q10_inexact_tuned_resample_summary.json").open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    flat = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "config"}
        item["config_json"] = json.dumps(row["config"], sort_keys=True)
        flat.append(item)
    with (reports / "q10_inexact_tuned_resample_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)


if __name__ == "__main__":
    main()
