#!/usr/bin/env python3
"""Tune only k for ASE NEBOptimizer(neb, method="ode")."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_q10_baselines import save_run


K_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 20.0, 30.0, 100.0, 300.0,
          1000.0, 3000.0, 5000.0, 10000.0)
FIXED_CONFIG = {
    "fmax": 1.0e-14,
    "diagnostic_interval": 10,
}
CONSTRUCTOR = 'NEBOptimizer(neb, method="ode")'
FORMAL_TOTAL_FORCE_BUDGET = 5000


def config_for(k: float) -> dict:
    return {"k": float(k), **FIXED_CONFIG}


def execute(payload: tuple[Path, float, int, str]) -> dict:
    root, k, budget, role = payload
    return save_run(root, "ode_neb", config_for(k), budget, role)


def run_grid(
    root: Path, ks: list[float], budget: int, role: str, workers: int
) -> list[dict]:
    payloads = [(root, k, budget, role) for k in ks]
    rows: list[dict] = []
    if workers <= 1:
        for payload in payloads:
            row = execute(payload)
            rows.append(row)
            print(
                f"[{role}] k={row['config']['k']:g} "
                f"score={row.get('trajectory_score', math.inf):.6g} "
                f"I_perp={row.get('final_i_perp', math.inf):.6g}",
                flush=True,
            )
        return sorted(rows, key=lambda row: float(row["config"]["k"]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, payload): payload[1] for payload in payloads}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"[{role}] k={row['config']['k']:g} "
                f"score={row.get('trajectory_score', math.inf):.6g} "
                f"I_perp={row.get('final_i_perp', math.inf):.6g}",
                flush=True,
            )
    return sorted(rows, key=lambda row: float(row["config"]["k"]))


def ranking_key(row: dict) -> tuple:
    return (
        not bool(row.get("eligible", False)),
        float(row.get("trajectory_score", math.inf)),
        float(row.get("i_perp_log_auc", math.inf)),
        float(row.get("barrier_log_auc", math.inf)),
        float(row.get("final_i_perp", math.inf)),
    )


def formal_window_key(root: Path, row: dict) -> tuple:
    with np.load(root / row["output"], allow_pickle=False) as archive:
        forces = np.asarray(archive["total_force_evaluations"], dtype=float)
        i_perp = np.asarray(archive["i_perp"], dtype=float)
        reference = float(np.ravel(archive["reference_barrier"])[0])
        barrier = np.abs(np.asarray(archive["delta_e"], dtype=float) - reference)
    eligible = np.flatnonzero(forces <= FORMAL_TOTAL_FORCE_BUDGET)
    index = int(eligible[-1]) if len(eligible) else 0
    joint = math.log10(max(float(i_perp[index]), 1.0e-300))
    joint += math.log10(max(float(barrier[index]), 1.0e-300))
    return joint, float(i_perp[index]), float(barrier[index])


def update_formal_selection(root: Path, selected: dict) -> None:
    path = root / "reports" / "baseline_final_selection.json"
    report = json.loads(path.read_text())
    report["selected"]["ode_neb"] = {
        "config": selected["config"],
        "ase_optimizer_constructor": CONSTRUCTOR,
        "only_tuned_parameter": "k",
        "dense_barrier_error": selected["barrier_error"],
        "final_calls_per_image": selected["final_calls_per_image"],
        "final_total_force_evaluations": selected["final_total_force_evaluations"],
        "final_i_perp": selected["final_i_perp"],
        "run": selected["output"],
        "stop_reason": selected["stop_reason"],
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def write_reports(
    root: Path, screen: list[dict], production: list[dict], selected: dict,
    screen_budget: int, production_budget: int,
) -> None:
    report = {
        "case": "fixed-Q unperturbed 10D Muller-Brown, 22 images",
        "ase_optimizer_constructor": CONSTRUCTOR,
        "only_tuned_parameter": "k",
        "ase_defaults_not_overridden": ["alpha", "verbose", "rtol", "C1", "C2"],
        "fixed_settings": FIXED_CONFIG,
        "k_grid": list(K_GRID),
        "screen_budget_per_image": screen_budget,
        "production_budget_per_image": production_budget,
        "force_cost_definition": (
            "total uncached force calls summed over all 20 internal images"
        ),
        "screen_rows": screen,
        "production_rows": production,
        "selected": selected,
    }
    json_path = root / "reports" / "q10_ode_default_k_sweep.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    ranked_screen = sorted(screen, key=ranking_key)
    ranked_production = sorted(production, key=ranking_key)
    lines = [
        "# Q10 Muller ODE-NEB default-constructor k sweep",
        "",
        f"- Constructor: `{CONSTRUCTOR}`",
        "- Only tuned parameter: `k`",
        f"- Fixed settings: `fmax={FIXED_CONFIG['fmax']}`, "
        f"`diagnostic_interval={FIXED_CONFIG['diagnostic_interval']}`",
        "- Force cost: total uncached calls summed over 20 internal images",
        "",
        "## Screen ranking",
        "",
        "| rank | k | trajectory score | final I_perp | total forces | stop |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(ranked_screen, 1):
        lines.append(
            f"| {rank} | {row['config']['k']:g} | "
            f"{row.get('trajectory_score', math.inf):.6e} | "
            f"{row.get('final_i_perp', math.inf):.6e} | "
            f"{row.get('final_total_force_evaluations', 0)} | "
            f"{row.get('stop_reason', '')} |"
        )
    lines.extend([
        "", "## Production ranking", "",
        "| rank | k | trajectory score | final I_perp | barrier error | total forces | stop |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ])
    for rank, row in enumerate(ranked_production, 1):
        lines.append(
            f"| {rank} | {row['config']['k']:g} | "
            f"{row.get('trajectory_score', math.inf):.6e} | "
            f"{row.get('final_i_perp', math.inf):.6e} | "
            f"{row.get('barrier_error', math.inf):.6e} | "
            f"{row.get('final_total_force_evaluations', 0)} | "
            f"{row.get('stop_reason', '')} |"
        )
    lines.extend([
        "", "## Selected", "",
        f"Selected `k={selected['config']['k']:g}` by joint endpoint performance "
        f"at total force budget {FORMAL_TOTAL_FORCE_BUDGET}.",
        "",
    ])
    (root / "reports" / "ODE_DEFAULT_K_SELECTION.md").write_text(
        "\n".join(lines)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--screen-budget", type=int, default=500)
    parser.add_argument("--production-budget", type=int, default=5000)
    parser.add_argument("--production-candidates", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()

    screen = run_grid(
        root, list(K_GRID), args.screen_budget, "ode_default_screen", args.workers
    )
    ranked_screen = sorted(screen, key=ranking_key)
    finalists = [
        float(row["config"]["k"])
        for row in ranked_screen[:args.production_candidates]
    ]
    production = run_grid(
        root, finalists, args.production_budget,
        "ode_default_production", args.workers,
    )
    selected = min(production, key=lambda row: formal_window_key(root, row))
    update_formal_selection(root, selected)
    write_reports(
        root, screen, production, selected,
        args.screen_budget, args.production_budget,
    )
    print(
        f"SELECTED k={selected['config']['k']:g} "
        f"run={selected['output']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
