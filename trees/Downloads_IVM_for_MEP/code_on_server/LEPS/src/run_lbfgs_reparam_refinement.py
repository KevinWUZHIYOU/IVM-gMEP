#!/usr/bin/env python3
"""Test whether less-frequent reparameterization lets OBSM-L-BFGS change branch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run_study import result_exists, run_command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    for interval in (2, 5, 10):
        for lam in (0.8, 0.95):
            for inner in (1, 3):
                for scale in (0.03, 0.1):
                    label = (
                        f"lbfgs_reparam_r{interval}_l{lam:g}_"
                        f"n{inner}_s{scale:g}"
                    )
                    if not args.rerun and result_exists(
                        root, "obsm_lbfgs", label, args.budget
                    ):
                        continue
                    run_command(
                        [
                            sys.executable,
                            str(root / "src/run_string_baseline.py"),
                            "--root",
                            str(root),
                            "--run-label",
                            label,
                            "--method",
                            "obsm",
                            "--solver",
                            "lbfgs",
                            "--lambda-mix",
                            str(lam),
                            "--inner-steps",
                            str(inner),
                            "--lbfgs-initial-scale",
                            str(scale),
                            "--lbfgs-history",
                            "7",
                            "--reparam-interval",
                            str(interval),
                            "--budget",
                            str(args.budget),
                        ],
                        root,
                    )


if __name__ == "__main__":
    main()
