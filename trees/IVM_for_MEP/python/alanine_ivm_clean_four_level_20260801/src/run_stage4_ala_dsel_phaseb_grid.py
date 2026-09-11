#!/usr/bin/env python3
"""Preregistered DSEL alpha grid on the shared legacy Ala Phase-A prefix.

The canonical legacy alpha=20 Phase A is not rerun or modified.  Every grid
cell starts a fresh Phase-B process from its evaluated terminal path, charges
the 20-image initialization, and changes only Phase-B alpha plus the registered
DSEL stop mode.  Thus the full cost curve through NF=680 is exactly legacy.
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
from run_stage4_ala_dsel_grid import (
    DEFAULT_ALPHAS,
    DEFAULT_STOPS,
    LOWER_IS_BETTER,
    parse_alphas,
    parse_stops,
)


PROTOCOL_ID = "stage4_ala_dsel_ffrbk_v2_sharedA34_phaseB_control_grid"
LEGACY_PHASE_A_RELATIVE = (
    "runs/stage4_inexact_optimized_resample/phases/"
    "ivm_clean_stage4_v3_phaseA_fast_a20_central_state_ivm_"
    "cfg09d74897c1ef_budget34.npz"
)
LEGACY_PHASE_B_RELATIVE = (
    "runs/stage4_inexact_optimized_resample/phases/"
    "ivm_clean_stage4_v3_phaseB_precision_a20_central_state_ivm_"
    "cfg21ee8dded116_budget220.npz"
)
CONTINUATION_RELATIVE = (
    "inputs/stage4_uniform_v3_continuations/phaseA34_terminal_a20.npz"
)


def run_process(command, root, stdout_path, stderr_path):
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


def old_value_at_cost(old, costs, key):
    old_cost = np.asarray(old["total_internal_joint_evaluations"], dtype=int)
    indices = np.searchsorted(old_cost, costs, side="right") - 1
    if np.any(indices < 0):
        raise RuntimeError("Candidate cost precedes charged Phase-B initialization")
    return np.asarray(old[key], dtype=float)[indices]


def summarize(result, legacy_phase_b, phase_a_prefix_nf):
    with np.load(result, allow_pickle=False) as saved, np.load(
        legacy_phase_b, allow_pickle=False
    ) as old:
        local_cost = np.asarray(
            saved["total_internal_joint_evaluations"], dtype=int
        )
        if int(local_cost[0]) != 20:
            raise RuntimeError("Phase-B initialization was not charged 20 NF")
        regressions = {}
        maximum_ratios = {}
        for key in LOWER_IS_BETTER:
            values = np.asarray(saved[key], dtype=float)
            old_values = old_value_at_cost(old, local_cost, key)
            tolerance = 1e-12 * np.maximum(1.0, np.abs(old_values))
            regressions[key] = int(
                np.count_nonzero(values > old_values + tolerance)
            )
            maximum_ratios[key] = float(
                np.max(values / np.maximum(np.abs(old_values), 1e-300))
            )
        selections = np.asarray(
            saved["ala_dsel_audit_selection"], dtype="U16"
        )
        first_exact = all(
            np.array_equal(np.asarray(saved[key])[:1], np.asarray(old[key])[:1])
            for key in (
                *LOWER_IS_BETTER,
                "segment_length_cv",
                "equidistance_relative_defect",
                "total_internal_joint_evaluations",
            )
        )
        return {
            "result": str(result),
            "result_sha256": legacy.file_sha256(result),
            "configuration_sha256": str(
                legacy.scalar(saved, "configuration_sha256")
            ),
            "local_terminal_NF": int(local_cost[-1]),
            "full_curve_terminal_NF": int(phase_a_prefix_nf + local_cost[-1]),
            "terminal_i_perp": float(saved["i_perp"][-1]),
            "terminal_sparse_barrier_kcal": float(saved["barrier_kcal"][-1]),
            "terminal_physical_fmax": float(saved["physical_fmax"][-1]),
            "terminal_segment_length_cv": float(
                saved["segment_length_cv"][-1]
            ),
            "terminal_equidistance_defect": float(
                saved["equidistance_relative_defect"][-1]
            ),
            "h0_selection_count": int(np.count_nonzero(selections == "H0")),
            "lbfgs_selection_count": int(
                np.count_nonzero(selections == "LBFGS")
            ),
            "safeguard_selection_count": int(
                np.count_nonzero(selections == "SAFEGUARD")
            ),
            "phase_b_initial_record_bitwise_equal_to_legacy": bool(first_exact),
            "legacy_prefix_through_NF": int(phase_a_prefix_nf),
            "regression_counts_vs_legacy_phase_b_at_same_local_NF": regressions,
            "maximum_ratios_vs_legacy_phase_b_at_same_local_NF": maximum_ratios,
            "weakly_dominates_legacy_phase_b_on_screened_metrics": bool(
                all(value == 0 for value in regressions.values())
            ),
        }


def run_cell(
    root,
    continuation,
    legacy_phase_b,
    phase_a_prefix_nf,
    alpha,
    stop_mode,
    budget,
    replicate,
):
    execution_tag = f"sharedA34_B{budget}"
    command, run_label = dsel.phase_b_command(
        root,
        alpha,
        budget,
        continuation,
        stop_mode,
        replicate,
        -1.0,
        execution_tag,
    )
    phase_a_command, _ = dsel.phase_a_command(
        root, alpha, 34, stop_mode, replicate, execution_tag
    )
    dsel.assert_frozen_outer_policy(
        root, alpha, phase_a_command, command, continuation
    )
    phase_dir = root / "runs" / dsel.POLICY_VERSION / "phases"
    existing = sorted(
        phase_dir.glob(f"ivm_{run_label}_central_state_ivm_cfg*_budget{budget}.npz")
    )
    if existing:
        raise FileExistsError("Refusing to overwrite " + str(existing[0]))
    member = f"{stop_mode}_a{legacy.label_number(alpha)}_{replicate}_{execution_tag}"
    log_dir = root / "logs" / dsel.POLICY_VERSION / "phaseB_grid" / member
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    returncode, elapsed, stderr = run_process(
        command,
        root,
        log_dir / "stdout.txt",
        log_dir / "stderr.txt",
    )
    payload = {
        "alpha": "inf" if math.isinf(alpha) else alpha,
        "stop_mode": stop_mode,
        "budget_per_image": budget,
        "replicate": replicate,
        "command": command,
        "returncode": returncode,
        "wall_seconds": elapsed,
    }
    if returncode:
        payload["error"] = stderr[-4000:]
        return payload
    result = legacy.find_result(
        phase_dir,
        run_label,
        budget,
        initial_path_sha256="4418a113c37836930685545c2a87a2e5e4e185836aba3e0afab5d5a835020726",
    )
    payload.update(summarize(result, legacy_phase_b, phase_a_prefix_nf))
    return payload


def write_csv(path, results):
    fields = (
        "alpha",
        "stop_mode",
        "budget_per_image",
        "local_terminal_NF",
        "full_curve_terminal_NF",
        "terminal_i_perp",
        "terminal_sparse_barrier_kcal",
        "terminal_physical_fmax",
        "h0_selection_count",
        "lbfgs_selection_count",
        "safeguard_selection_count",
        "phase_b_initial_record_bitwise_equal_to_legacy",
        "weakly_dominates_legacy_phase_b_on_screened_metrics",
        "returncode",
        "wall_seconds",
        "result",
        "result_sha256",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--alphas",
        default=",".join(f"{value:g}" for value in DEFAULT_ALPHAS),
    )
    parser.add_argument("--stop-modes", default=",".join(DEFAULT_STOPS))
    parser.add_argument("--phase-b-budget", type=int, default=80)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--replicate", default="r501")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    alphas = parse_alphas(args.alphas)
    stops = parse_stops(args.stop_modes)
    if args.phase_b_budget < 2 or args.jobs < 1:
        raise ValueError("Budget must be >=2 and jobs must be positive")
    if not args.replicate.startswith("r") or not args.replicate[1:].isdigit():
        raise ValueError("replicate must be r followed by digits")

    phase_a = root / LEGACY_PHASE_A_RELATIVE
    phase_b = root / LEGACY_PHASE_B_RELATIVE
    continuation = root / CONTINUATION_RELATIVE
    for path in (phase_a, phase_b, continuation):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(phase_a, allow_pickle=False) as saved_a, np.load(
        continuation, allow_pickle=False
    ) as saved_c:
        terminal = np.asarray(saved_a["final_path"], dtype=float)
        continuation_path = np.asarray(saved_c["initial_path"], dtype=float)
        if not np.array_equal(terminal, continuation_path):
            raise RuntimeError("Frozen continuation differs from Phase-A terminal")
        phase_a_prefix_nf = int(saved_a["total_internal_joint_evaluations"][-1])
        phase_a_terminal_i_perp = float(saved_a["i_perp"][-1])
        terminal_sha = legacy.array_sha256(terminal)
    if terminal_sha != "4418a113c37836930685545c2a87a2e5e4e185836aba3e0afab5d5a835020726":
        raise RuntimeError("Unexpected canonical Phase-A terminal path")

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{PROTOCOL_ID}_B{args.phase_b_budget}_{args.replicate}_"
        f"{int(time.time())}"
    )
    preregistered = reports / f"{stem}_preregistered.json"
    final = reports / f"{stem}.json"
    csv_path = reports / f"{stem}.csv"
    protocol = {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "scientific_question": (
            "Can post-geometry H0-vs-legacy-LBFGS DSEL accelerate the expensive "
            "Phase-B tail while preserving the complete legacy Phase-A prefix?"
        ),
        "legacy_prefix_is_reused_without_rerun": True,
        "legacy_prefix_through_NF": phase_a_prefix_nf,
        "legacy_prefix_terminal_i_perp": phase_a_terminal_i_perp,
        "legacy_phase_a": str(phase_a),
        "legacy_phase_a_sha256": legacy.file_sha256(phase_a),
        "legacy_phase_b_comparator": str(phase_b),
        "legacy_phase_b_sha256": legacy.file_sha256(phase_b),
        "continuation": str(continuation),
        "continuation_sha256": legacy.file_sha256(continuation),
        "continuation_path_sha256": terminal_sha,
        "alpha_grid": ["inf" if math.isinf(value) else value for value in alphas],
        "stop_modes": list(stops),
        "phase_b_budget_per_image": args.phase_b_budget,
        "selection_leakage_guard": (
            "No densified reference barrier or barrier error is used for selection"
        ),
        "ranking_metrics": list(LOWER_IS_BETTER),
        "source_sha256": {
            "core": legacy.file_sha256(
                root / "src/run_ivm_v9_spring_projected_production_20260730.py"
            ),
            "wrapper": legacy.file_sha256(
                root / "src/run_stage4_ala_dsel_uniform.py"
            ),
            "wrapper_implementation": legacy.file_sha256(
                root / "src/run_stage4_discrete_b040_uniform.py"
            ),
            "launcher": legacy.file_sha256(Path(__file__).resolve()),
        },
        "jobs": args.jobs,
        "replicate": args.replicate,
        "created_unix": time.time(),
    }
    preregistered.write_text(json.dumps(protocol, indent=2) + "\n")

    cells = [(alpha, stop) for stop in stops for alpha in alphas]
    results = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                run_cell,
                root,
                continuation,
                phase_b,
                phase_a_prefix_nf,
                alpha,
                stop,
                args.phase_b_budget,
                args.replicate,
            ): (alpha, stop)
            for alpha, stop in cells
        }
        for future in concurrent.futures.as_completed(futures):
            alpha, stop = futures[future]
            try:
                item = future.result()
            except Exception as exc:
                item = {
                    "alpha": "inf" if math.isinf(alpha) else alpha,
                    "stop_mode": stop,
                    "budget_per_image": args.phase_b_budget,
                    "replicate": args.replicate,
                    "returncode": -1,
                    "error": repr(exc),
                }
            results.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)
    stop_order = {value: index for index, value in enumerate(stops)}
    results.sort(
        key=lambda row: (
            stop_order[str(row["stop_mode"])],
            math.inf if row["alpha"] == "inf" else float(row["alpha"]),
        )
    )
    successful = [row for row in results if row.get("returncode") == 0]
    report = {
        **protocol,
        "status": "complete" if len(successful) == len(results) else "incomplete",
        "started_unix": started,
        "finished_unix": time.time(),
        "preregistration": str(preregistered),
        "preregistration_sha256": legacy.file_sha256(preregistered),
        "results": results,
        "screen_winners_by_terminal_i_perp": {
            stop: min(
                (row for row in successful if row["stop_mode"] == stop),
                key=lambda row: float(row["terminal_i_perp"]),
                default=None,
            )
            for stop in stops
        },
    }
    final.write_text(json.dumps(report, indent=2) + "\n")
    write_csv(csv_path, results)
    print(final)
    print(csv_path)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
