#!/usr/bin/env python3
"""Focused second-stage LEPS baseline scan after the 200-call coarse grid."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

from run_study import result_exists, run_command


STATIC_K = (3, 5, 7, 10, 15, 20)
STATIC_DT = (0.02, 0.025, 0.03, 0.035, 0.04)
ORIGINAL_DT = (0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06)
FIRE_LAMBDA = (0.8, 0.85, 0.9, 0.925, 0.95, 0.975)
FIRE_INNER = (3, 4, 5, 6, 7)
FIRE_DT = (0.07, 0.09, 0.11, 0.13, 0.15, 0.18)
LBFGS_LAMBDA = (0.5, 0.65, 0.75, 0.85, 0.9, 0.95)
LBFGS_INNER = (1, 2, 3, 5)
LBFGS_SCALE = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument(
        "--family",
        choices=("all", "static", "original", "fire", "lbfgs"),
        default="all",
    )
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    python = sys.executable

    if args.family in {"all", "static"}:
        for k, dt in itertools.product(STATIC_K, STATIC_DT):
            label = f"refine_static_k{k:g}_dt{dt:g}"
            if not args.rerun and result_exists(root, "neb", label, args.budget):
                continue
            run_command(
                [
                    python, str(root / "src/run_neb.py"), "--root", str(root),
                    "--run-label", label, "--k", str(k), "--optimizer", "static",
                    "--alpha", str(dt), "--budget", str(args.budget),
                ],
                root,
            )

    if args.family in {"all", "original"}:
        for dt in ORIGINAL_DT:
            label = f"refine_v2dense_original_dt{dt:g}_r1"
            if not args.rerun and result_exists(
                root, "original_string", label, args.budget
            ):
                continue
            run_command(
                [
                    python, str(root / "src/run_string_baseline.py"),
                    "--root", str(root), "--run-label", label,
                    "--method", "original", "--dt", str(dt),
                    "--reparam-interval", "1", "--budget", str(args.budget),
                ],
                root,
            )

    if args.family in {"all", "fire"}:
        for lam, inner, dt in itertools.product(
            FIRE_LAMBDA, FIRE_INNER, FIRE_DT
        ):
            label = f"refine_v2dense_obsm_fire_l{lam:g}_n{inner}_dt{dt:g}_r1"
            if not args.rerun and result_exists(
                root, "obsm_fire", label, args.budget
            ):
                continue
            run_command(
                [
                    python, str(root / "src/run_string_baseline.py"),
                    "--root", str(root), "--run-label", label,
                    "--method", "obsm", "--solver", "fire",
                    "--lambda-mix", str(lam), "--inner-steps", str(inner),
                    "--fire-dt", str(dt), "--fire-beta", "0.1",
                    "--reparam-interval", "1", "--budget", str(args.budget),
                ],
                root,
            )

    if args.family in {"all", "lbfgs"}:
        for lam, inner, scale in itertools.product(
            LBFGS_LAMBDA, LBFGS_INNER, LBFGS_SCALE
        ):
            label = (
                f"refine_v2dense_obsm_lbfgs_l{lam:g}_n{inner}_"
                f"s{scale:g}_h7_r1"
            )
            if not args.rerun and result_exists(
                root, "obsm_lbfgs", label, args.budget
            ):
                continue
            run_command(
                [
                    python, str(root / "src/run_string_baseline.py"),
                    "--root", str(root), "--run-label", label,
                    "--method", "obsm", "--solver", "lbfgs",
                    "--lambda-mix", str(lam), "--inner-steps", str(inner),
                    "--lbfgs-initial-scale", str(scale),
                    "--lbfgs-history", "7", "--reparam-interval", "1",
                    "--budget", str(args.budget),
                ],
                root,
            )


if __name__ == "__main__":
    main()
