#!/usr/bin/env python3
"""Reproduce the LEPS NEB scan and IVM hyperparameter ablations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


NEB_K_VALUES = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000)
IVM_ALPHA_VALUES = (1, 3, 10, 20, 30, 50, 70, 100, 150, 300)
PRECISION_MODES = {
    "inexact": {
        "kkt_rtol": 0.8,
        "kkt_atol": 1e-10,
        "inner_calls": 15,
        "budget": 200,
    },
    "tight": {
        "kkt_rtol": 0.05,
        "kkt_atol": 1e-10,
        "inner_calls": 80,
        "budget": 200,
    },
    "near_exact": {
        # 500x tighter in relative KKT residual than "tight"; the 1e-7
        # absolute floor avoids asking a double-precision LEPS line search to
        # resolve forces below meaningful numerical accuracy.
        "kkt_rtol": 1e-4,
        "kkt_atol": 1e-7,
        "inner_calls": 300,
        # The analysis compares every mode at the common 200-call cutoff.
        # Extra budget only lets the inner solve demonstrate its own KKT stop.
        "budget": 500,
    },
}
REPARAM_MODES = ("none", "post_hermite")
STATIC_K_VALUES = (0.1, 0.3, 1, 3, 10, 30, 100)
STATIC_STEP_VALUES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
ORIGINAL_DT_VALUES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1)
OBSM_LAMBDA_VALUES = (0.25, 0.5, 0.75, 0.9, 0.95)
OBSM_INNER_VALUES = (1, 2, 5, 10, 20)
OBSM_FIRE_DT_VALUES = (0.001, 0.003, 0.01, 0.03, 0.1)


def run_command(command, cwd):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(cwd / "src")
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def result_exists(root, method, label, budget):
    return any(
        (root / "runs" / method).glob(f"{label}_cfg*_budget{budget}.npz")
    )


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def step_log_auc(data, key, budget, floor):
    grid = np.arange(1, budget + 1)
    calls = np.asarray(data["calls_per_image"], dtype=int)
    values = np.asarray(data[key], dtype=float)
    indices = np.searchsorted(calls, grid, side="right") - 1
    sampled = values[np.clip(indices, 0, len(values) - 1)]
    return float(np.mean(np.log10(np.maximum(sampled, floor))))


def select_neb_finalists(root):
    rows = []
    for path in sorted(
        (root / "runs" / "neb").glob("neb_scan_*_budget500.npz")
    ):
        data = load_npz(path)
        feasible = (
            float(data["segment_length_cv"][-1]) <= 0.10
            and float(data["barrier_error"][-1]) <= 0.10
        )
        score = step_log_auc(data, "i_perp", 500, 1e-14)
        score += step_log_auc(data, "barrier_error", 500, 1e-8)
        rows.append((feasible, score, float(data["spring_constant"])))
    eligible = sorted((score, k) for feasible, score, k in rows if feasible)
    if len(eligible) < 3:
        raise RuntimeError("Fewer than three feasible NEB finalists")
    return [k for _, k in eligible[:3]]


def select_ivm_alpha(root):
    rows = []
    for path in sorted(
        (root / "runs" / "ivm").glob(
            "ivm_alpha_*_tight_reparam_*_budget200.npz"
        )
    ):
        data = load_npz(path)
        score = step_log_auc(data, "i_perp", 200, 1e-14)
        score += step_log_auc(data, "barrier_error", 200, 1e-8)
        rows.append((score, float(data["alpha_initial"])))
    if not rows:
        raise RuntimeError("No tight, reparameterized IVM grid runs")
    return min(rows)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "all", "prepare", "neb", "ivm", "baselines", "production"
        ),
        default="all",
    )
    parser.add_argument("--neb-budget", type=int, default=500)
    parser.add_argument("--ivm-budget", type=int, default=200)
    parser.add_argument("--production-neb-budget", type=int, default=5000)
    parser.add_argument("--production-ivm-budget", type=int, default=500)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    python = sys.executable

    if args.stage in {"all", "prepare"}:
        run_command(
            [
                python,
                str(root / "src" / "prepare_initial_path.py"),
                "--root",
                str(root),
            ],
            root,
        )

    if args.stage in {"all", "neb"}:
        for k in NEB_K_VALUES:
            label = f"neb_scan_k{k:g}"
            if (
                not args.rerun
                and result_exists(root, "neb", label, args.neb_budget)
            ):
                continue
            run_command(
                [
                    python,
                    str(root / "src" / "run_neb.py"),
                    "--root",
                    str(root),
                    "--run-label",
                    label,
                    "--k",
                    str(k),
                    "--budget",
                    str(args.neb_budget),
                    "--max-ode-steps",
                    "100000",
                ],
                root,
            )

    if args.stage in {"all", "ivm"}:
        for precision_name, precision in PRECISION_MODES.items():
            for reparameterize in REPARAM_MODES:
                for alpha in IVM_ALPHA_VALUES:
                    experiment_budget = (
                        args.ivm_budget
                        if precision_name != "near_exact"
                        else max(args.ivm_budget, precision["budget"])
                    )
                    reparam_label = (
                        "reparam" if reparameterize == "post_hermite" else "none"
                    )
                    label = (
                        f"ivm_alpha_{alpha:g}_{precision_name}_{reparam_label}"
                    )
                    if (
                        not args.rerun
                        and result_exists(
                            root, "ivm", label, experiment_budget
                        )
                    ):
                        continue
                    run_command(
                        [
                            python,
                            str(root / "src" / "run_ivm.py"),
                            "--root",
                            str(root),
                            "--run-label",
                            label,
                            "--alpha",
                            str(alpha),
                            "--alpha-mode",
                            "fixed",
                            "--kkt-rtol",
                            str(precision["kkt_rtol"]),
                            "--kkt-atol",
                            str(precision["kkt_atol"]),
                            "--inner-calls",
                            str(precision["inner_calls"]),
                            "--subproblem-solver",
                            (
                                "independent_near_exact"
                                if precision_name == "near_exact"
                                else "independent"
                            ),
                            "--reparameterize",
                            reparameterize,
                            "--budget",
                            str(experiment_budget),
                        ],
                        root,
                    )

    if args.stage in {"all", "baselines"}:
        for k in STATIC_K_VALUES:
            for step in STATIC_STEP_VALUES:
                label = f"static_neb_k{k:g}_dt{step:g}"
                if not args.rerun and result_exists(
                    root, "neb", label, args.neb_budget
                ):
                    continue
                run_command(
                    [
                        python, str(root / "src" / "run_neb.py"),
                        "--root", str(root), "--run-label", label,
                        "--k", str(k), "--optimizer", "static",
                        "--alpha", str(step), "--budget", str(args.neb_budget),
                    ],
                    root,
                )
        for dt in ORIGINAL_DT_VALUES:
            label = f"original_string_dt{dt:g}_r1"
            if not args.rerun and result_exists(
                root, "original_string", label, args.ivm_budget
            ):
                continue
            run_command(
                [
                    python, str(root / "src" / "run_string_baseline.py"),
                    "--root", str(root), "--run-label", label,
                    "--method", "original", "--dt", str(dt),
                    "--reparam-interval", "1", "--budget",
                    str(args.ivm_budget),
                ],
                root,
            )
        for solver in ("fire", "lbfgs"):
            for lam in OBSM_LAMBDA_VALUES:
                for inner in OBSM_INNER_VALUES:
                    fire_steps = (
                        OBSM_FIRE_DT_VALUES if solver == "fire" else (0.01,)
                    )
                    for fire_dt in fire_steps:
                        label = (
                            f"obsm_{solver}_lam{lam:g}_inner{inner}_"
                            f"dt{fire_dt:g}_r1"
                        )
                        if solver == "lbfgs":
                            label += "_h7_s0.01"
                        family = f"obsm_{solver}"
                        if not args.rerun and result_exists(
                            root, family, label, args.ivm_budget
                        ):
                            continue
                        run_command(
                            [
                                python,
                                str(root / "src" / "run_string_baseline.py"),
                                "--root", str(root), "--run-label", label,
                                "--method", "obsm", "--solver", solver,
                                "--lambda-mix", str(lam),
                                "--inner-steps", str(inner),
                                "--fire-dt", str(fire_dt),
                                "--reparam-interval", "1",
                                "--budget", str(args.ivm_budget),
                            ],
                            root,
                        )

    if args.stage in {"all", "production"}:
        finalists = select_neb_finalists(root)
        for k in finalists:
            label = f"neb_production_k{k:g}"
            if (
                not args.rerun
                and result_exists(
                    root, "neb", label, args.production_neb_budget
                )
            ):
                continue
            run_command(
                [
                    python,
                    str(root / "src" / "run_neb.py"),
                    "--root",
                    str(root),
                    "--run-label",
                    label,
                    "--k",
                    str(k),
                    "--budget",
                    str(args.production_neb_budget),
                    "--max-ode-steps",
                    "1000000",
                ],
                root,
            )

        alpha = select_ivm_alpha(root)
        label = f"ivm_production_alpha{alpha:g}_tight_reparam"
        if args.rerun or not result_exists(
            root, "ivm", label, args.production_ivm_budget
        ):
            run_command(
                [
                    python,
                    str(root / "src" / "run_ivm.py"),
                    "--root",
                    str(root),
                    "--run-label",
                    label,
                    "--alpha",
                    str(alpha),
                    "--alpha-mode",
                    "fixed",
                    "--kkt-rtol",
                    str(PRECISION_MODES["tight"]["kkt_rtol"]),
                    "--kkt-atol",
                    str(PRECISION_MODES["tight"]["kkt_atol"]),
                    "--inner-calls",
                    str(PRECISION_MODES["tight"]["inner_calls"]),
                    "--subproblem-solver",
                    "independent",
                    "--reparameterize",
                    "post_hermite",
                    "--budget",
                    str(args.production_ivm_budget),
                ],
                root,
            )


if __name__ == "__main__":
    main()
