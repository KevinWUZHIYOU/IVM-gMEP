#!/usr/bin/env python3
"""Independent 1000-call validation of focused LEPS baseline candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run_study import result_exists, run_command


STATIC = ((10, 0.04), (7, 0.04), (10, 0.045), (10, 0.05), (15, 0.04))
ORIGINAL = (0.04, 0.05, 0.06, 0.07)
FIRE = (
    (0.8, 4, 0.18),
    (0.8, 5, 0.18),
    (0.85, 4, 0.18),
    (0.8, 4, 0.22),
    (0.8, 4, 0.25),
)
LBFGS = (
    (0.95, 5, 0.03),
    (0.9, 5, 0.03),
    (0.95, 3, 0.03),
    (0.95, 5, 0.01),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    python = sys.executable

    for k, dt in STATIC:
        label = f"extension1000_static_k{k:g}_dt{dt:g}"
        if args.rerun or not result_exists(root, "neb", label, args.budget):
            run_command(
                [
                    python, str(root / "src/run_neb.py"), "--root", str(root),
                    "--run-label", label, "--k", str(k), "--optimizer", "static",
                    "--alpha", str(dt), "--budget", str(args.budget),
                ],
                root,
            )
    for dt in ORIGINAL:
        label = f"extension1000_v2dense_original_dt{dt:g}_r1"
        if args.rerun or not result_exists(
            root, "original_string", label, args.budget
        ):
            run_command(
                [
                    python, str(root / "src/run_string_baseline.py"),
                    "--root", str(root), "--run-label", label,
                    "--method", "original", "--dt", str(dt),
                    "--reparam-interval", "1", "--budget", str(args.budget),
                ],
                root,
            )
    for lam, inner, dt in FIRE:
        label = f"extension1000_v2dense_fire_l{lam:g}_n{inner}_dt{dt:g}"
        if args.rerun or not result_exists(root, "obsm_fire", label, args.budget):
            run_command(
                [
                    python, str(root / "src/run_string_baseline.py"),
                    "--root", str(root), "--run-label", label,
                    "--method", "obsm", "--solver", "fire",
                    "--lambda-mix", str(lam), "--inner-steps", str(inner),
                    "--fire-dt", str(dt), "--reparam-interval", "1",
                    "--budget", str(args.budget),
                ],
                root,
            )
    for lam, inner, scale in LBFGS:
        label = f"extension1000_v2dense_lbfgs_l{lam:g}_n{inner}_s{scale:g}"
        if args.rerun or not result_exists(root, "obsm_lbfgs", label, args.budget):
            run_command(
                [
                    python, str(root / "src/run_string_baseline.py"),
                    "--root", str(root), "--run-label", label,
                    "--method", "obsm", "--solver", "lbfgs",
                    "--lambda-mix", str(lam), "--inner-steps", str(inner),
                    "--lbfgs-initial-scale", str(scale), "--lbfgs-history", "7",
                    "--reparam-interval", "1", "--budget", str(args.budget),
                ],
                root,
            )


if __name__ == "__main__":
    main()
