#!/usr/bin/env python3
"""Preregistered fixed-alpha, state-triggered Ala DSEL control grid.

Every cell runs the unchanged Stage-IV Phase-A/Phase-B protocol from the
canonical common path.  One alpha is fixed for the entire two-phase member.
The only registered algorithm extensions are the I_perp state trigger,
post-geometry H0-vs-legacy-H0-LBFGS DSEL, and either C10 or R08 stopping.
No densified barrier value is available to the launcher or used for ranking.
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
from run_stage4_ala_dsel_grid import DEFAULT_STOPS, parse_alphas, parse_stops


PROTOCOL_ID = "stage4_ala_dsel_ffrbk_v3_fixedalpha_trigger_control_grid"
DEFAULT_ALPHAS = (5.0, 10.0, 20.0, 40.0, 80.0)
DEFAULT_TRIGGER_IPERP = 8e-4
LEGACY_COMBINED_RELATIVE = (
    "runs/stage4_inexact_optimized_resample/"
    "stage4_uniform_v3_a20_phaseA34_phaseB220.npz"
)
EXACT_PREFIX_KEYS = (
    "total_force_evaluations",
    "i_perp",
    "sparse_barrier_kcal",
    "physical_fmax",
    "segment_length_cv",
    "equidistance_relative_defect",
    "phase",
)


def _number(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:g}"


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _run_process(
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
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return completed.returncode, time.time() - started, completed.stderr


def _member_paths(
    root: Path,
    alpha: float,
    trigger_iperp: float,
    stop_mode: str,
    replicate: str,
    phase_a_budget: int,
    phase_b_budget: int,
) -> tuple[str, Path, Path]:
    member_id = (
        f"{stop_mode}_a{legacy.label_number(alpha)}_"
        f"t{legacy.label_number(trigger_iperp)}_{replicate}"
    )
    combined = root / "runs" / dsel.POLICY_VERSION / (
        f"{dsel.POLICY_VERSION}_{member_id}_phaseA{phase_a_budget}_"
        f"phaseB{phase_b_budget}.npz"
    )
    return member_id, combined, combined.with_suffix(".json")


def _exact_alpha20_prefix(
    candidate: np.lib.npyio.NpzFile,
    baseline: np.lib.npyio.NpzFile,
    trigger_iperp: float,
) -> dict[str, object]:
    candidate_i = np.asarray(candidate["i_perp"], dtype=float)
    crossed = np.flatnonzero(candidate_i <= trigger_iperp)
    prefix_size = int(crossed[0] + 1) if crossed.size else int(candidate_i.size)
    if prefix_size > np.asarray(baseline["i_perp"]).size:
        return {
            "applicable": True,
            "bitwise_equal": False,
            "records_checked": prefix_size,
            "reason": "candidate prefix is longer than canonical baseline",
        }
    mismatched = []
    for key in EXACT_PREFIX_KEYS:
        left = np.asarray(candidate[key])[:prefix_size]
        right = np.asarray(baseline[key])[:prefix_size]
        if not np.array_equal(left, right):
            mismatched.append(key)
    costs = np.asarray(candidate["total_force_evaluations"], dtype=int)
    return {
        "applicable": True,
        "bitwise_equal": not mismatched,
        "records_checked": prefix_size,
        "through_NF": int(costs[prefix_size - 1]),
        "trigger_crossed_in_checked_curve": bool(crossed.size),
        "mismatched_keys": mismatched,
    }


def _summarize(
    combined: Path,
    audit_path: Path,
    baseline_path: Path,
    alpha: float,
    trigger_iperp: float,
    stop_mode: str,
    phase_a_budget: int,
    phase_b_budget: int,
) -> dict[str, object]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    with np.load(combined, allow_pickle=False) as saved, np.load(
        baseline_path, allow_pickle=False
    ) as baseline:
        if float(legacy.scalar(saved, "alpha")) != float(alpha):
            raise RuntimeError("Combined member alpha differs from its grid cell")
        if str(legacy.scalar(saved, "alpha_mode")) != "fixed":
            raise RuntimeError("Combined member did not use --alpha-mode fixed")
        if not bool(legacy.scalar(saved, "fixed_alpha_throughout")):
            raise RuntimeError("Combined member failed its fixed-alpha audit")
        if (
            float(legacy.scalar(saved, "ala_dsel_trigger_iperp"))
            != trigger_iperp
        ):
            raise RuntimeError("Combined member used the wrong DSEL trigger")
        if int(legacy.scalar(saved, "phase_a_budget")) != phase_a_budget:
            raise RuntimeError("Combined member used the wrong Phase-A budget")
        if int(legacy.scalar(saved, "phase_b_budget")) != phase_b_budget:
            raise RuntimeError("Combined member used the wrong Phase-B budget")
        if str(legacy.scalar(saved, "stop_mode")) != stop_mode:
            raise RuntimeError("Combined member used the wrong stop mode")
        if int(
            legacy.scalar(saved, "ala_dsel_decisions_before_activation")
        ) != 0:
            raise RuntimeError("Combined member selected DSEL before activation")
        costs = np.asarray(saved["total_force_evaluations"], dtype=int)
        selections = np.asarray(saved["ala_dsel_audit_selection"], dtype="U16")
        if alpha == 20.0:
            prefix = _exact_alpha20_prefix(saved, baseline, trigger_iperp)
        else:
            prefix = {
                "applicable": False,
                "reason": "official Stage-IV baseline uses alpha=20",
            }
        return {
            "combined_result": str(combined),
            "combined_result_sha256": legacy.file_sha256(combined),
            "member_audit": str(audit_path),
            "member_audit_sha256": legacy.file_sha256(audit_path),
            "terminal_NF": int(costs[-1]),
            "terminal_i_perp": float(saved["i_perp"][-1]),
            "minimum_i_perp": float(np.min(saved["i_perp"])),
            "terminal_sparse_barrier_kcal": float(
                saved["sparse_barrier_kcal"][-1]
            ),
            "terminal_physical_fmax": float(saved["physical_fmax"][-1]),
            "terminal_segment_length_cv": float(
                saved["segment_length_cv"][-1]
            ),
            "h0_selection_count": int(np.count_nonzero(selections == "H0")),
            "lbfgs_selection_count": int(
                np.count_nonzero(selections == "LBFGS")
            ),
            "safeguard_selection_count": int(
                np.count_nonzero(selections == "SAFEGUARD")
            ),
            "ala_dsel_decisions_before_activation": int(
                legacy.scalar(saved, "ala_dsel_decisions_before_activation")
            ),
            "ala_dsel_first_activation_phase": int(
                legacy.scalar(saved, "ala_dsel_first_activation_phase")
            ),
            "ala_dsel_first_activation_nf": int(
                legacy.scalar(saved, "ala_dsel_first_activation_nf")
            ),
            "ala_dsel_first_activation_calls_per_image": int(
                legacy.scalar(
                    saved, "ala_dsel_first_activation_calls_per_image"
                )
            ),
            "alpha_mode": str(legacy.scalar(saved, "alpha_mode")),
            "fixed_alpha_throughout": bool(
                legacy.scalar(saved, "fixed_alpha_throughout")
            ),
            "alpha20_exact_legacy_prefix": prefix,
            "outer_policy_command_audit": audit[
                "outer_policy_command_audit"
            ],
        }


def _run_cell(
    root: Path,
    baseline_path: Path,
    alpha: float,
    trigger_iperp: float,
    stop_mode: str,
    phase_a_budget: int,
    phase_b_budget: int,
    replicate: str,
) -> dict[str, object]:
    member_id, combined, audit_path = _member_paths(
        root,
        alpha,
        trigger_iperp,
        stop_mode,
        replicate,
        phase_a_budget,
        phase_b_budget,
    )
    if combined.exists() or audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite {combined}")
    log_dir = (
        root
        / "logs"
        / dsel.POLICY_VERSION
        / "fixedalpha_trigger_grid"
        / f"{member_id}_A{phase_a_budget}_B{phase_b_budget}"
    )
    log_dir.mkdir(parents=True, exist_ok=False)
    wrapper = root / "src/run_stage4_ala_dsel_uniform.py"
    command = [
        sys.executable,
        str(wrapper),
        "--root",
        str(root),
        "--alpha",
        _number(alpha),
        "--stop-mode",
        stop_mode,
        "--phase-a-budget",
        str(phase_a_budget),
        "--phase-b-budget",
        str(phase_b_budget),
        "--stop-complete-iperp",
        "-1",
        "--ala-dsel-trigger-iperp",
        _number(trigger_iperp),
        "--replicate",
        replicate,
    ]
    (log_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    returncode, elapsed, stderr = _run_process(
        command,
        root,
        log_dir / "stdout.txt",
        log_dir / "stderr.txt",
    )
    payload: dict[str, object] = {
        "alpha": "inf" if math.isinf(alpha) else alpha,
        "alpha_mode": "fixed",
        "fixed_alpha_throughout": True,
        "trigger_iperp": trigger_iperp,
        "stop_mode": stop_mode,
        "phase_a_budget_per_image": phase_a_budget,
        "phase_b_budget_per_image": phase_b_budget,
        "replicate": replicate,
        "command": command,
        "returncode": returncode,
        "wall_seconds": elapsed,
    }
    if returncode:
        payload["error"] = stderr[-4000:]
        return payload
    if not combined.is_file() or not audit_path.is_file():
        raise RuntimeError("Wrapper succeeded without immutable result and audit")
    payload.update(
        _summarize(
            combined,
            audit_path,
            baseline_path,
            alpha,
            trigger_iperp,
            stop_mode,
            phase_a_budget,
            phase_b_budget,
        )
    )
    return payload


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "alpha",
        "alpha_mode",
        "fixed_alpha_throughout",
        "trigger_iperp",
        "stop_mode",
        "phase_a_budget_per_image",
        "phase_b_budget_per_image",
        "terminal_NF",
        "terminal_i_perp",
        "minimum_i_perp",
        "terminal_sparse_barrier_kcal",
        "terminal_physical_fmax",
        "terminal_segment_length_cv",
        "h0_selection_count",
        "lbfgs_selection_count",
        "safeguard_selection_count",
        "returncode",
        "wall_seconds",
        "combined_result",
        "combined_result_sha256",
    )
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--alphas",
        default=",".join(f"{value:g}" for value in DEFAULT_ALPHAS),
    )
    parser.add_argument("--stop-modes", default=",".join(DEFAULT_STOPS))
    parser.add_argument(
        "--ala-dsel-trigger-iperp",
        type=float,
        default=DEFAULT_TRIGGER_IPERP,
    )
    parser.add_argument("--phase-a-budget", type=int, default=34)
    parser.add_argument("--phase-b-budget", type=int, default=220)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--replicate", default="r701")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    alphas = parse_alphas(args.alphas)
    stop_modes = parse_stops(args.stop_modes)
    trigger_iperp = float(args.ala_dsel_trigger_iperp)
    if 20.0 not in alphas:
        raise ValueError("The fixed-alpha grid must include alpha=20 control")
    if not trigger_iperp > 0.0 or not math.isfinite(trigger_iperp):
        raise ValueError("Trigger I_perp must be finite and positive")
    if args.phase_a_budget < 2 or args.phase_b_budget < 2:
        raise ValueError("Both per-image phase budgets must be at least two")
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    if not args.replicate.startswith("r") or not args.replicate[1:].isdigit():
        raise ValueError("replicate must be r followed by digits")

    common = root / "inputs/common_initial_path_equidistant.npz"
    baseline = root / LEGACY_COMBINED_RELATIVE
    core = root / "src/run_ivm_v9_spring_projected_production_20260730.py"
    wrapper = root / "src/run_stage4_ala_dsel_uniform.py"
    implementation = root / "src/run_stage4_discrete_b040_uniform.py"
    launcher = Path(__file__).resolve()
    for path in (common, baseline, core, wrapper, implementation, launcher):
        if not path.is_file():
            raise FileNotFoundError(path)

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    unique = time.time_ns()
    stem = (
        f"{PROTOCOL_ID}_A{args.phase_a_budget}_B{args.phase_b_budget}_"
        f"t{legacy.label_number(trigger_iperp)}_{args.replicate}_{unique}"
    )
    preregistration = reports / f"{stem}_preregistered.json"
    final_report = reports / f"{stem}.json"
    csv_report = reports / f"{stem}.csv"
    protocol: dict[str, object] = {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "scientific_question": (
            "With one alpha fixed throughout both phases, does legacy-until-"
            "trigger then same-alpha DSEL improve Stage-IV cost-to-precision?"
        ),
        "fixed_alpha_throughout_each_member": True,
        "alpha_mode": "fixed",
        "tail_alpha_or_alpha_switch_allowed": False,
        "alpha_grid": ["inf" if math.isinf(x) else x for x in alphas],
        "alpha20_control_required": True,
        "trigger_iperp": trigger_iperp,
        "trigger_definition": (
            "DSEL activates when current accepted I_perp <= trigger; the same "
            "fixed alpha remains in force"
        ),
        "stop_modes": list(stop_modes),
        "stop_definitions": {
            "c10": "evaluated accepted-step cap 10; absolute KKT may stop earlier",
            "r08": "evaluated R/R0 <= 0.8 or evaluated accepted-step cap 10",
        },
        "phase_a_budget_per_image": args.phase_a_budget,
        "phase_b_budget_per_image": args.phase_b_budget,
        "common_initial_path": str(common),
        "common_initial_path_sha256": legacy.file_sha256(common),
        "legacy_alpha20_comparator": str(baseline),
        "legacy_alpha20_comparator_sha256": legacy.file_sha256(baseline),
        "alpha20_exact_prefix_rule": (
            "bitwise compare all recorded diagnostics through and including "
            "the first accepted state with I_perp <= trigger"
        ),
        "selection_leakage_guard": (
            "No densified energy barrier or densified barrier error is read "
            "or used for selection or ranking"
        ),
        "screening_metrics": (
            "I_perp, sparse barrier, physical fmax, and mesh diagnostics only"
        ),
        "source_sha256": {
            "core": legacy.file_sha256(core),
            "wrapper_alias": legacy.file_sha256(wrapper),
            "wrapper_implementation": legacy.file_sha256(implementation),
            "launcher": legacy.file_sha256(launcher),
        },
        "jobs": args.jobs,
        "replicate": args.replicate,
        "created_unix": time.time(),
    }
    _write_json_exclusive(preregistration, protocol)

    cells = [(alpha, stop) for stop in stop_modes for alpha in alphas]
    rows: list[dict[str, object]] = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                _run_cell,
                root,
                baseline,
                alpha,
                trigger_iperp,
                stop,
                args.phase_a_budget,
                args.phase_b_budget,
                args.replicate,
            ): (alpha, stop)
            for alpha, stop in cells
        }
        for future in concurrent.futures.as_completed(futures):
            alpha, stop = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "alpha": "inf" if math.isinf(alpha) else alpha,
                    "alpha_mode": "fixed",
                    "fixed_alpha_throughout": True,
                    "trigger_iperp": trigger_iperp,
                    "stop_mode": stop,
                    "phase_a_budget_per_image": args.phase_a_budget,
                    "phase_b_budget_per_image": args.phase_b_budget,
                    "replicate": args.replicate,
                    "returncode": -1,
                    "error": repr(exc),
                }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    stop_order = {name: index for index, name in enumerate(stop_modes)}
    rows.sort(
        key=lambda row: (
            stop_order[str(row["stop_mode"])],
            math.inf if row["alpha"] == "inf" else float(row["alpha"]),
        )
    )
    successful = [row for row in rows if row.get("returncode") == 0]
    alpha20_rows = [row for row in successful if float(row["alpha"]) == 20.0]
    alpha20_prefix_pass = bool(alpha20_rows) and all(
        bool(row["alpha20_exact_legacy_prefix"]["bitwise_equal"])
        for row in alpha20_rows
    )
    report = {
        **protocol,
        "status": "complete" if len(successful) == len(rows) else "incomplete",
        "started_unix": started,
        "finished_unix": time.time(),
        "preregistration": str(preregistration),
        "preregistration_sha256": legacy.file_sha256(preregistration),
        "alpha20_exact_legacy_prefix_pass": alpha20_prefix_pass,
        "results": rows,
        "screen_winners_by_terminal_i_perp": {
            stop: min(
                (row for row in successful if row["stop_mode"] == stop),
                key=lambda row: float(row["terminal_i_perp"]),
                default=None,
            )
            for stop in stop_modes
        },
    }
    _write_json_exclusive(final_report, report)
    _write_csv(csv_report, rows)
    print(final_report)
    print(csv_report)
    if report["status"] != "complete" or not alpha20_prefix_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
