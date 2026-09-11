#!/usr/bin/env python3
"""Preregistered Phase-A control grid for the Ala DSEL experiment.

Each cell changes only fixed ``alpha`` and the evaluated-state inner stopping
rule.  It starts from the canonical common path, preserves the Stage-IV
Phase-A outer/reparameterization policy, and writes an immutable result plus
an audit row.  The post-hoc dense barrier reference is deliberately absent
from selection and ranking.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import run_stage4_ala_dsel_uniform as dsel
import run_stage4_uniform_optimized as legacy


PROTOCOL_ID = "stage4_ala_dsel_ffrbk_v2_phaseA_control_grid"
DEFAULT_ALPHAS = (
    0.1,
    0.625,
    1.25,
    2.5,
    5.0,
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
    320.0,
    1000.0,
)
DEFAULT_STOPS = ("c10", "r08")
LOWER_IS_BETTER = (
    "i_perp",
    "barrier_kcal",
    "physical_fmax",
)


def parse_alphas(text: str) -> tuple[float, ...]:
    values = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        value = math.inf if item in {"inf", "infinity"} else float(item)
        if not value > 0.0:
            raise ValueError("Every alpha must be positive or inf")
        values.append(value)
    if not values:
        raise ValueError("The alpha grid cannot be empty")
    if len(set(values)) != len(values):
        raise ValueError("The alpha grid contains duplicates")
    return tuple(values)


def parse_stops(text: str) -> tuple[str, ...]:
    values = tuple(item.strip().lower() for item in text.split(",") if item.strip())
    if not values or any(item not in DEFAULT_STOPS for item in values):
        raise ValueError("Stop modes must be a nonempty subset of c10,r08")
    if len(set(values)) != len(values):
        raise ValueError("Stop modes contain duplicates")
    return values


def run_subprocess(
    command: list[str],
    root: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, float, str]:
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
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    return completed.returncode, time.time() - started, completed.stderr


def baseline_at_cost(saved, costs: np.ndarray, key: str) -> np.ndarray:
    baseline_cost = np.asarray(
        saved["total_internal_joint_evaluations"], dtype=int
    )
    baseline_value = np.asarray(saved[key], dtype=float)
    indices = np.searchsorted(baseline_cost, costs, side="right") - 1
    if np.any(indices < 0):
        raise RuntimeError("Candidate contains a cost before baseline initialization")
    return baseline_value[indices]


def summarize_result(result: Path, baseline: Path) -> dict[str, object]:
    with np.load(result, allow_pickle=False) as saved, np.load(
        baseline, allow_pickle=False
    ) as old:
        if str(legacy.scalar(saved, "frozen_subproblem")) != "ala_dsel":
            raise RuntimeError(f"Wrong frozen subproblem in {result}")
        costs = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        if costs.ndim != 1 or costs.size == 0 or np.any(np.diff(costs) <= 0):
            raise RuntimeError(f"Invalid cost sequence in {result}")
        regressions: dict[str, int] = {}
        maximum_ratios: dict[str, float] = {}
        for key in LOWER_IS_BETTER:
            values = np.asarray(saved[key], dtype=float)
            old_values = baseline_at_cost(old, costs, key)
            tolerance = 1e-12 * np.maximum(1.0, np.abs(old_values))
            regressions[key] = int(np.count_nonzero(values > old_values + tolerance))
            maximum_ratios[key] = float(
                np.max(values / np.maximum(np.abs(old_values), 1e-300))
            )
        selections = np.asarray(saved["ala_dsel_audit_selection"], dtype="U16")
        common_first_two = costs <= 40
        first_two_exact = True
        for key in (
            *LOWER_IS_BETTER,
            "segment_length_cv",
            "equidistance_relative_defect",
            "total_internal_joint_evaluations",
        ):
            candidate_values = np.asarray(saved[key])[common_first_two]
            old_cost = np.asarray(old["total_internal_joint_evaluations"], dtype=int)
            old_values = np.asarray(old[key])[np.isin(old_cost, costs[common_first_two])]
            first_two_exact &= np.array_equal(candidate_values, old_values)
        return {
            "result": str(result),
            "result_sha256": legacy.file_sha256(result),
            "configuration_sha256": str(legacy.scalar(saved, "configuration_sha256")),
            "terminal_NF": int(costs[-1]),
            "terminal_i_perp": float(saved["i_perp"][-1]),
            "terminal_sparse_barrier_kcal": float(saved["barrier_kcal"][-1]),
            "terminal_physical_fmax": float(saved["physical_fmax"][-1]),
            "terminal_segment_length_cv": float(saved["segment_length_cv"][-1]),
            "terminal_equidistance_defect": float(
                saved["equidistance_relative_defect"][-1]
            ),
            "h0_selection_count": int(np.count_nonzero(selections == "H0")),
            "lbfgs_selection_count": int(np.count_nonzero(selections == "LBFGS")),
            "first_two_records_bitwise_equal_to_legacy": bool(first_two_exact),
            "regression_counts_vs_legacy_at_same_NF": regressions,
            "maximum_ratios_vs_legacy_at_same_NF": maximum_ratios,
            "weakly_dominates_legacy_on_screened_metrics": bool(
                all(count == 0 for count in regressions.values())
            ),
        }


def run_cell(
    root: Path,
    baseline: Path,
    alpha: float,
    stop_mode: str,
    budget: int,
    replicate: str,
) -> dict[str, object]:
    command, run_label = dsel.phase_a_command(
        root, alpha, budget, stop_mode, replicate
    )
    phase_dir = root / "runs" / dsel.POLICY_VERSION / "phases"
    existing = sorted(
        phase_dir.glob(f"ivm_{run_label}_central_state_ivm_cfg*_budget{budget}.npz")
    )
    if existing:
        raise FileExistsError(
            "Refusing to reuse or overwrite a grid cell: " + str(existing[0])
        )
    member = f"{stop_mode}_a{legacy.label_number(alpha)}_{replicate}_A{budget}"
    log_dir = root / "logs" / dsel.POLICY_VERSION / "phaseA_grid" / member
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    returncode, elapsed, stderr = run_subprocess(
        command,
        root,
        log_dir / "stdout.txt",
        log_dir / "stderr.txt",
    )
    payload: dict[str, object] = {
        "alpha": "inf" if math.isinf(alpha) else alpha,
        "stop_mode": stop_mode,
        "budget_per_image": budget,
        "replicate": replicate,
        "command": command,
        "wall_seconds": elapsed,
        "returncode": returncode,
    }
    if returncode:
        payload["error"] = stderr[-4000:]
        return payload
    result = legacy.find_result(phase_dir, run_label, budget)
    payload.update(summarize_result(result, baseline))
    return payload


def write_csv(path: Path, results: list[dict[str, object]]) -> None:
    fields = (
        "alpha",
        "stop_mode",
        "budget_per_image",
        "terminal_NF",
        "terminal_i_perp",
        "terminal_sparse_barrier_kcal",
        "terminal_physical_fmax",
        "terminal_segment_length_cv",
        "terminal_equidistance_defect",
        "h0_selection_count",
        "lbfgs_selection_count",
        "first_two_records_bitwise_equal_to_legacy",
        "weakly_dominates_legacy_on_screened_metrics",
        "returncode",
        "wall_seconds",
        "result",
        "result_sha256",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--alphas",
        default=",".join(f"{value:g}" for value in DEFAULT_ALPHAS),
    )
    parser.add_argument("--stop-modes", default=",".join(DEFAULT_STOPS))
    parser.add_argument("--phase-a-budget", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--replicate", default="r201")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    alphas = parse_alphas(args.alphas)
    stop_modes = parse_stops(args.stop_modes)
    if args.phase_a_budget < 2:
        raise ValueError("Phase-A budget must be at least two paid states")
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    if not args.replicate.startswith("r") or not args.replicate[1:].isdigit():
        raise ValueError("replicate must be r followed by digits")

    baseline = root / (
        "runs/stage4_inexact_optimized_resample/phases/"
        "ivm_clean_stage4_v3_phaseA_fast_a20_central_state_ivm_"
        "cfg09d74897c1ef_budget34.npz"
    )
    common = root / "inputs/common_initial_path_equidistant.npz"
    if not baseline.is_file() or not common.is_file():
        raise FileNotFoundError("Canonical baseline or common initial path is missing")
    with np.load(common, allow_pickle=False) as saved:
        common_path = np.asarray(saved["initial_path"], dtype=float)

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{PROTOCOL_ID}_A{args.phase_a_budget}_{args.replicate}_"
        f"{int(time.time())}"
    )
    preregistration = reports / f"{stem}_preregistered.json"
    final_report = reports / f"{stem}.json"
    csv_report = reports / f"{stem}.csv"
    if any(path.exists() for path in (preregistration, final_report, csv_report)):
        raise FileExistsError("Refusing to overwrite a protocol report")

    core = root / "src/run_ivm_v9_spring_projected_production_20260730.py"
    wrapper = root / "src/run_stage4_ala_dsel_uniform.py"
    wrapper_implementation = root / "src/run_stage4_discrete_b040_uniform.py"
    launcher = Path(__file__).resolve()
    protocol = {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "scientific_question": (
            "Does adding capped H0-vs-legacy-H0-LBFGS direction selection "
            "improve the frozen-path Ala solve under an otherwise fixed Stage-IV policy?"
        ),
        "selection_leakage_guard": (
            "No densified reference barrier or barrier error is used for alpha selection"
        ),
        "alpha_grid": ["inf" if math.isinf(value) else value for value in alphas],
        "stop_modes": list(stop_modes),
        "stop_definitions": {
            "c10": "evaluated accepted-step cap 10; absolute KKT may stop earlier",
            "r08": "evaluated R/R0 <= 0.8 or evaluated accepted-step cap 10",
        },
        "phase_a_budget_per_image": args.phase_a_budget,
        "jobs": args.jobs,
        "replicate": args.replicate,
        "common_initial_file": str(common),
        "common_initial_file_sha256": legacy.file_sha256(common),
        "common_initial_array_sha256": legacy.array_sha256(common_path),
        "legacy_baseline": str(baseline),
        "legacy_baseline_sha256": legacy.file_sha256(baseline),
        "source_sha256": {
            "core": legacy.file_sha256(core),
            "wrapper": legacy.file_sha256(wrapper),
            "wrapper_implementation": legacy.file_sha256(
                wrapper_implementation
            ),
            "launcher": legacy.file_sha256(launcher),
        },
        "frozen_policy": (
            "Stage-IV Phase-A command flags and Ala reparameterization are unchanged; "
            "only alpha, DSEL, and the registered inner stop vary"
        ),
        "ranking_metrics": list(LOWER_IS_BETTER),
        "mesh_role": "audit only, not a scalar ranking objective",
        "created_unix": time.time(),
    }
    preregistration.write_text(json.dumps(protocol, indent=2) + "\n")

    cells = [
        (alpha, stop_mode)
        for stop_mode in stop_modes
        for alpha in alphas
    ]
    results = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                run_cell,
                root,
                baseline,
                alpha,
                stop_mode,
                args.phase_a_budget,
                args.replicate,
            ): (alpha, stop_mode)
            for alpha, stop_mode in cells
        }
        for future in concurrent.futures.as_completed(futures):
            alpha, stop_mode = futures[future]
            try:
                item = future.result()
            except Exception as exc:  # Preserve the complete grid audit.
                item = {
                    "alpha": "inf" if math.isinf(alpha) else alpha,
                    "stop_mode": stop_mode,
                    "budget_per_image": args.phase_a_budget,
                    "replicate": args.replicate,
                    "returncode": -1,
                    "error": repr(exc),
                }
            results.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)

    order = {name: index for index, name in enumerate(stop_modes)}
    results.sort(
        key=lambda row: (
            order[str(row["stop_mode"])],
            math.inf if row["alpha"] == "inf" else float(row["alpha"]),
        )
    )
    successful = [row for row in results if row.get("returncode") == 0]
    report = {
        **protocol,
        "status": (
            "complete" if len(successful) == len(results) else "incomplete"
        ),
        "started_unix": started,
        "finished_unix": time.time(),
        "preregistration": str(preregistration),
        "preregistration_sha256": legacy.file_sha256(preregistration),
        "results": results,
        "screen_winners_by_terminal_i_perp": {
            stop_mode: min(
                (
                    row
                    for row in successful
                    if row["stop_mode"] == stop_mode
                ),
                key=lambda row: float(row["terminal_i_perp"]),
                default=None,
            )
            for stop_mode in stop_modes
        },
    }
    final_report.write_text(json.dumps(report, indent=2) + "\n")
    write_csv(csv_report, results)
    print(final_report)
    print(csv_report)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
