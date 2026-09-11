#!/usr/bin/env python3
"""Targeted long-horizon refinement for the fixed-Q Muller-10 baselines.

This driver deliberately imports the already-audited implementations from
``run_q10_baselines``.  It only expands parameter ranges that the broad screen
identified as resolution- or horizon-limited; no method definition is changed.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from run_q10_baselines import save_run


METHOD_ORDER = (
    "original_string",
    "obsm_fire",
    "obsm_lbfgs",
    "static_neb",
    "ode_neb",
)


def refinement_specs():
    specs = []

    # Smaller string time steps probe the asymptotic discretization error.
    for dt, interval in itertools.product(
        (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4), (1, 2, 5)
    ):
        specs.append((
            "original_string",
            {
                "dt": dt,
                "reparam_interval": interval,
                "diagnostic_interval": 25,
            },
            5000,
        ))

    # The broad FIRE screen already reached machine-level normal residuals.
    # This local grid resolves speed around its winning region.
    for lam, inner, dt, interval in itertools.product(
        (0.75, 0.8, 0.85, 0.9, 0.95), (2, 4), (5e-3, 1e-2), (1, 5)
    ):
        specs.append((
            "obsm_fire",
            {
                "solver": "fire",
                "lambda": lam,
                "inner": inner,
                "fire_dt": dt,
                "fire_beta": 1.0,
                "reparam_interval": interval,
                "diagnostic_interval": 5,
            },
            500,
        ))

    # Existing lambda >= 0.1 candidates reached small residuals and then
    # rebounded.  Resolve smaller damping factors without changing OBSM.
    for lam, inner, interval in itertools.product(
        (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25),
        (1, 2, 4),
        (1, 2, 5),
    ):
        specs.append((
            "obsm_lbfgs",
            {
                "solver": "lbfgs",
                "lambda": lam,
                "inner": inner,
                "reparam_interval": interval,
                "diagnostic_interval": 5,
            },
            500,
        ))

    # Static NEB: refine the stable alpha band over weak to stiff springs.
    for k, alpha in itertools.product(
        (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 50.0, 75.0,
         100.0, 150.0, 200.0, 300.0, 500.0),
        (3e-5, 1e-4, 3e-4, 1e-3),
    ):
        specs.append((
            "static_neb",
            {
                "optimizer_method": "static",
                "k": k,
                "alpha": alpha,
                "fmax": 1e-14,
                "diagnostic_interval": 10,
            },
            500,
        ))

    # ODE NEB: the long run moved the optimum from k=1 toward k=100.
    for k, rtol, c2 in itertools.product(
        (30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0),
        (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
        (2.0, 10.0),
    ):
        specs.append((
            "ode_neb",
            {
                "k": k,
                "rtol": rtol,
                "C1": 0.01,
                "C2": c2,
                "fmax": 1e-14,
                "diagnostic_interval": 10,
            },
            500,
        ))

    return specs


def execute(payload):
    root, method, config, budget, role = payload
    return save_run(Path(root), method, config, budget, role)


def run_parallel(payloads, workers, label):
    rows = []
    if workers == 1:
        for index, payload in enumerate(payloads, 1):
            rows.append(execute(payload))
            if index % 10 == 0 or index == len(payloads):
                print("{} {}/{}".format(label, index, len(payloads)), flush=True)
        return rows

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 10 == 0 or index == len(payloads):
                print("{} {}/{}".format(label, index, len(payloads)), flush=True)
    return rows


def write_report(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    csv_path = path.with_suffix(".csv")
    fields = sorted({key for row in rows for key in row if key != "config"})
    fields.insert(3, "config_json")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: value for key, value in row.items() if key != "config"}
            flat["config_json"] = json.dumps(row["config"], sort_keys=True)
            writer.writerow(flat)


def diverse_candidates(rows, count):
    finite = [
        row for row in rows
        if row.get("finite") and row.get("stop_reason") != "numerical_failure"
    ]
    if not finite:
        finite = [row for row in rows if row.get("finite")]
    orderings = (
        lambda row: row["trajectory_score"],
        lambda row: row["final_i_perp"],
        lambda row: row["barrier_error"],
    )
    selected = []
    seen = set()
    for key in orderings:
        for row in sorted(finite, key=key)[:count]:
            marker = row["config_id"]
            if marker not in seen:
                seen.add(marker)
                selected.append(row)
    return selected


def run_screen(root, workers, methods):
    specs = [spec for spec in refinement_specs() if spec[0] in methods]
    payloads = [
        (str(root), method, config, budget, "refinement_screen")
        for method, config, budget in specs
    ]
    rows = run_parallel(payloads, workers, "refinement_screen")
    write_report(root / "reports" / "baseline_refinement_screen.json", rows)
    return rows


def run_production(root, workers, candidates_per_objective):
    screen_path = root / "reports" / "baseline_refinement_screen.json"
    rows = json.loads(screen_path.read_text())
    production_budget = {
        "original_string": 20000,
        "obsm_fire": 4000,
        "obsm_lbfgs": 5000,
        "static_neb": 5000,
        "ode_neb": 5000,
    }
    chosen = []
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        chosen.extend(diverse_candidates(method_rows, candidates_per_objective))
    payloads = [
        (
            str(root),
            row["method"],
            row["config"],
            production_budget[row["method"]],
            "refinement_production",
        )
        for row in chosen
    ]
    result = run_parallel(payloads, workers, "refinement_production")
    write_report(root / "reports" / "baseline_refinement_production.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("screen", "production", "all"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--candidates-per-objective", type=int, default=4)
    parser.add_argument(
        "--methods", nargs="+", choices=METHOD_ORDER, default=METHOD_ORDER
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.stage in ("screen", "all"):
        run_screen(root, args.workers, set(args.methods))
    if args.stage in ("production", "all"):
        run_production(root, args.workers, args.candidates_per_objective)


if __name__ == "__main__":
    main()
