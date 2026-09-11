#!/usr/bin/env python3
"""Independent long-budget LEPS baseline production runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run_study import result_exists, run_command


def launch(root, family, label, budget, arguments, rerun):
    if rerun or not result_exists(root, family, label, budget):
        save_paths = [] if family == "neb" else ["--save-all-paths"]
        run_command(
            [
                sys.executable,
                str(root / arguments[0]),
                "--root",
                str(root),
                "--run-label",
                label,
                *arguments[1:],
                "--budget",
                str(budget),
                *save_paths,
            ],
            root,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    launch(
        root,
        "neb",
        "production_static_k10_dt0.05",
        args.budget,
        ("src/run_neb.py", "--k", "10", "--optimizer", "static", "--alpha", "0.05"),
        args.rerun,
    )
    launch(
        root,
        "original_string",
        "production_original_dt0.07_r1",
        args.budget,
        (
            "src/run_string_baseline.py",
            "--method",
            "original",
            "--dt",
            "0.07",
            "--reparam-interval",
            "1",
        ),
        args.rerun,
    )
    for label, lam, inner, dt in (
        ("production_fire_fast_l0.8_n4_dt0.22", 0.8, 4, 0.22),
        ("production_fire_stable_l0.85_n4_dt0.18", 0.85, 4, 0.18),
    ):
        launch(
            root,
            "obsm_fire",
            label,
            args.budget,
            (
                "src/run_string_baseline.py",
                "--method",
                "obsm",
                "--solver",
                "fire",
                "--lambda-mix",
                str(lam),
                "--inner-steps",
                str(inner),
                "--fire-dt",
                str(dt),
                "--reparam-interval",
                "1",
            ),
            args.rerun,
        )
    launch(
        root,
        "obsm_lbfgs",
        "production_lbfgs_r5_l0.95_n1_s0.1",
        args.budget,
        (
            "src/run_string_baseline.py",
            "--method",
            "obsm",
            "--solver",
            "lbfgs",
            "--lambda-mix",
            "0.95",
            "--inner-steps",
            "1",
            "--lbfgs-initial-scale",
            "0.1",
            "--lbfgs-history",
            "7",
            "--reparam-interval",
            "5",
        ),
        args.rerun,
    )


if __name__ == "__main__":
    main()
