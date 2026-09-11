#!/usr/bin/env python3
"""Run the first three fixed-alpha Muller-10 IVM groups."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from q10_ivm_core import save_run


ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (
    1e-4,
    3e-4,
    1e-3,
    1.3e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    3.0,
    1e1,
    1e2,
    1e3,
    1e4,
)
R_VALUES = (0.2, 0.1, 0.05)


def exact_configs(resample):
    for r_value in R_VALUES:
        for alpha in ALPHAS:
            yield {
                "group": "exact_resample" if resample else "exact_no_resample",
                "solver": "exact_multistart",
                "alpha": alpha,
                "kkt_rtol": r_value,
                "kkt_atol": 1e-3,
                "resample": bool(resample),
                "resample_mode": "cubic" if resample else "none",
                "resample_interval": 1,
                "resample_passes": 3,
                "spring_constant": 1.0,
                "spring_dtau": 0.15,
                "spring_max_iter": 20000,
                "spring_tolerance": 1e-9,
                "budget_per_image": 5000,
                "outer_limit": 80,
                "movement_tolerance": 1e-14,
                "alpha_policy": "fixed",
                "tangent": "improved",
                "normal_dimension": 9,
            }


def baseline_configs():
    for alpha in ALPHAS:
        yield {
            "group": "inexact_baseline_resample",
            "solver": "full_normal_lbfgs",
            "alpha": alpha,
            "accepted_step_cap": 2,
            "new_call_cap": 12,
            "kkt_rtol": None,
            "kkt_atol": 1e-3,
            "min_accepted_steps": 2,
            "history_size": 0,
            "warm_history": False,
            "harmonic_preconditioner": False,
            "active_curvature": 0.0,
            "max_step": 3.0,
            "armijo": 1e-4,
            "backtrack": 0.5,
            "resample": True,
            "resample_mode": "cubic",
            "resample_interval": 1,
            "resample_passes": 3,
            "spring_constant": 1.0,
            "spring_dtau": 0.15,
            "spring_max_iter": 20000,
            "spring_tolerance": 1e-9,
            "reset_history_after_resample": True,
            "budget_per_image": 2000,
            "outer_limit": 10000,
            "movement_tolerance": 1e-14,
            "alpha_policy": "fixed",
            "tangent": "improved",
            "normal_dimension": 9,
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


def run_role(role, configs, workers):
    payloads = [(str(ROOT), config, role) for config in configs]
    rows = []
    if workers == 1:
        for index, payload in enumerate(payloads, 1):
            rows.append(worker(payload))
            print(role, index, len(payloads), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, payload) for payload in payloads]
            for index, future in enumerate(as_completed(futures), 1):
                rows.append(future.result())
                if index % 5 == 0 or index == len(futures):
                    print(role, index, len(futures), flush=True)
    rows.sort(
        key=lambda row: (
            float(row["config"].get("kkt_rtol") or 0.0),
            float(row["config"]["alpha"]),
        )
    )
    write_summary(role, rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        choices=(
            "exact_no_resample",
            "exact_resample",
            "inexact_baseline_resample",
            "all",
        ),
        required=True,
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.group in ("exact_no_resample", "all"):
        run_role(
            "q10_exact_no_resample", list(exact_configs(False)), args.workers
        )
    if args.group in ("exact_resample", "all"):
        run_role("q10_exact_resample", list(exact_configs(True)), args.workers)
    if args.group in ("inexact_baseline_resample", "all"):
        run_role(
            "q10_inexact_baseline_resample", list(baseline_configs()), args.workers
        )


if __name__ == "__main__":
    main()
