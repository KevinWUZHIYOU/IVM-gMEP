#!/usr/bin/env python3
"""Run only fixed-alpha cases whose trajectory can enter the new stall branch."""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path


ALPHAS_REQUIRING_RERUN = (20, 30, 50, 70, 100, 300, 1000)


def audited_result(root: Path, label: str) -> Path | None:
    matches = sorted(
        glob.glob(
            str(
                root
                / "runs"
                / "ivm"
                / f"ivm_{label}_central_state_ivm_*_budget500.npz"
            )
        )
    )
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous completed results for {label}: {matches}")
    if not matches:
        return None
    result = Path(matches[0])
    audit = result.with_suffix(".audit.json")
    if not audit.exists() or not json.loads(audit.read_text()).get("consistent"):
        raise RuntimeError(f"Existing result is not strictly audited: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    runner = args.runner.expanduser().resolve()

    for alpha in ALPHAS_REQUIRING_RERUN:
        label = f"production_escape_relock_a{alpha}"
        existing = audited_result(root, label)
        if existing is not None:
            print(f"SKIP audited {existing}", flush=True)
            continue
        command = [
            args.python,
            str(runner),
            "--root",
            str(root),
            "--initial-path-file",
            "inputs/common_initial_path_equidistant.npz",
            "--run-label",
            label,
            "--solver",
            "state_ivm",
            "--alpha",
            str(alpha),
            "--alpha-mode",
            "fixed",
            "--alpha-min",
            "0.03",
            "--alpha-max",
            "1000",
            "--max-step",
            "1",
            "--state-trust-fraction",
            "2",
            "--state-polish-trust-fraction",
            "2",
            "--state-globalization",
            "proximal_only",
            "--reparam-mode",
            "state_curve_projected",
            "--state-curve-direct-solve",
            "--state-curve-direct-trial-calls",
            "15",
            "--state-curve-stall-free-polish",
            "--state-curve-stall-free-polish-count",
            "1",
            "--state-curve-relock-escape-trust-factor",
            "0.0001",
            "--state-free-polish-reparam-strategy",
            "hermite_relock",
            "--reparam-ready-relative-iperp",
            "2",
            "--reparam-cv-trigger",
            "-1",
            "--reparam-recovery-calls",
            "30",
            "--inner-calls",
            "15",
            "--plbfgs-kkt-rtol",
            "0.8",
            "--budget",
            "500",
            "--save-all-paths",
        ]
        print("RUN " + " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=root)


if __name__ == "__main__":
    main()
