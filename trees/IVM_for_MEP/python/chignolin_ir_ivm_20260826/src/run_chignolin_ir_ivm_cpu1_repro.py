#!/usr/bin/env python3
"""Run a preregistered single-thread reproducibility check for Chignolin IR-IVM.

This adapter changes only the OpenMM CPU execution setting from two threads to
one.  It reuses the registered IR-IVM command builder and runtime unchanged, so
state-curve reparameterization and recovery remain mandatory at every outer
iteration.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
from pathlib import Path
from typing import Optional, Sequence

import run_chignolin_ir_ivm_member as base


EXPERIMENT_ID = "chignolin_ir_ivm_cpu1_prefix_repro_20260826_v1"
OPENMM_CPU_THREADS = "1"
TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def _tag(value: str) -> str:
    value = str(value).strip().lower()
    if not TAG_PATTERN.fullmatch(value):
        raise ValueError("replicate tag must be lowercase ASCII letters, digits, _ or -")
    return value


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    alpha = base._alpha(args.alpha)  # pylint: disable=protected-access
    ratio = base._ratio(args.residual_ratio)  # pylint: disable=protected-access
    cap = base._cap(args.inner_max_steps)  # pylint: disable=protected-access
    budget = base._budget(args.internal_nf_budget)  # pylint: disable=protected-access
    tag = _tag(args.replicate_tag)
    label = base.member_label(alpha, ratio, cap, budget).replace(
        "_r1", f"_cpu1_{tag}"
    )
    output_subdir = "runs/repro_cpu1"
    command = base.runner_argv(
        root, alpha, ratio, cap, budget, output_subdir, label
    )
    command[0] = str(Path(__file__).resolve())

    source_names = (
        "chignolin_core.py",
        "run_ivm_v9_spring_projected_production_20260730.py",
        "frozen_discrete_b040.py",
        "run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py",
        "run_ivm_v9_discrete_selected_eta_shim.py",
        "run_ivm_v9_discrete_ffrbk_takeover_grid_shim.py",
        "run_ivm_v9_total_internal_nf_budget.py",
        "run_chignolin_ir_ivm_member.py",
        Path(__file__).name,
    )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "purpose": "single-thread prefix reproducibility and fair execution comparison to Chignolin V9",
        "replicate_tag": tag,
        "label": label,
        "alpha": alpha,
        "residual_ratio_r": ratio,
        "inner_max_steps_m_max": cap,
        "physical_max_step_angstrom": 1.0,
        "eta": 1.1,
        "history": 7,
        "reparameterization": "state_curve every outer iteration with per_image direct acceptance and recovery",
        "internal_nf_budget": budget,
        "paper_total_nf_definition": "sum of all 22 image joint E/F cache misses; equals internal NF plus two initial endpoint calls",
        "openmm_cpu_threads": 1,
        "only_change_from_main_campaign_execution": "OPENMM_CPU_THREADS: 2 -> 1; no IR-IVM algorithm change",
        "command": command,
        "source_sha256": {
            name: base.sha256(root / "src" / name) for name in source_names
        },
        "input_sha256": {
            name: base.sha256(root / "inputs" / name)
            for name in (
                "chignolin_clean.pdb",
                "common_initial_path_equidistant.npz",
                "folded_min.traj",
                "misfolded_min.traj",
            )
        },
    }

    lock_path = root / "logs" / "chignolin_ir_ivm_host.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Chignolin host lock is already held: {lock_path}") from error
        registration = root / "registrations" / "repro_cpu1" / f"{label}.json"
        base._write_registration(registration, payload)  # pylint: disable=protected-access
        os.environ["OPENMM_CPU_THREADS"] = OPENMM_CPU_THREADS
        base.runtime.main(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--residual-ratio", required=True)
    parser.add_argument("--inner-max-steps", required=True)
    parser.add_argument("--internal-nf-budget", required=True)
    parser.add_argument("--replicate-tag", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
