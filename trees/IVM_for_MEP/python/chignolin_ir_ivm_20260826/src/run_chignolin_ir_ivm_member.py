#!/usr/bin/env python3
"""Run one audited Chignolin IR-IVM parameter setting.

This is a thin experiment adapter around the copied Ala IR-IVM runtime.  It
changes only the molecular system, input/output routing, the three registered
controls ``alpha``, ``r``, and ``m_max``, and the run budget.  In particular,
the state-curve reparameterization and recovery transaction are mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import run_ivm_v9_discrete_ffrbk_takeover_grid_shim as runtime


EXPERIMENT_ID = "chignolin_ir_ivm_transfer_20260826_v1"
ALPHAS = tuple(float(10**power) for power in range(-1, 7))
RESIDUAL_RATIOS = (0.1, 0.3, 0.5, 0.7, 0.8)
INNER_MAX_STEPS = (5, 8, 10, 15, 20, 50)
PRODUCTION_INTERNAL_NF_BUDGET = 5000
OPENMM_CPU_THREADS = "2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _alpha(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result not in ALPHAS:
        raise ValueError("alpha must be one of 1e-1, 1e0, ..., 1e6")
    return result


def _ratio(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result not in RESIDUAL_RATIOS:
        raise ValueError("r must be one of 0.1, 0.3, 0.5, 0.7, 0.8")
    return result


def _cap(value: Any) -> int:
    result = int(value)
    if str(result) != str(value).strip() or result not in INNER_MAX_STEPS:
        raise ValueError("m_max must be one of 5, 8, 10, 15, 20, 50")
    return result


def _budget(value: Any) -> int:
    result = int(value)
    if str(result) != str(value).strip() or result < 60:
        raise ValueError("internal NF budget must be an integer of at least 60")
    return result


def alpha_tag(value: Any) -> str:
    exponent = int(round(math.log10(_alpha(value))))
    return "a1e" + ("m" + str(-exponent) if exponent < 0 else str(exponent))


def ratio_tag(value: Any) -> str:
    return f"r{int(round(10 * _ratio(value))):02d}"


def member_label(alpha: Any, ratio: Any, cap: Any, budget: Any) -> str:
    return "_".join(
        (
            "chig_irivm",
            alpha_tag(alpha),
            ratio_tag(ratio),
            f"m{_cap(cap):02d}",
            f"inf{_budget(budget)}",
            "r1",
        )
    )


def runner_argv(
    root: Path,
    alpha: float,
    ratio: float,
    cap: int,
    budget: int,
    output_subdir: str,
    label: str,
) -> List[str]:
    return [
        str(Path(__file__).resolve()),
        "--root",
        str(root),
        "--initial-path-file",
        "inputs/common_initial_path_equidistant.npz",
        "--output-dir",
        output_subdir,
        "--run-label",
        label,
        "--solver",
        "state_ivm",
        "--alpha",
        f"{alpha:g}",
        "--alpha-mode",
        "fixed",
        "--alpha-min",
        "0.01",
        "--alpha-max",
        "inf",
        "--max-step",
        "1",
        "--history",
        "7",
        "--stop-iperp",
        "-1",
        "--budget",
        str(budget),
        "--save-all-paths",
        "--state-globalization",
        "iperp_monotone",
        "--state-trust-fraction",
        "2",
        "--state-polish-trust-fraction",
        "2",
        "--state-polish-alpha-factor",
        "0.5",
        "--inner-calls",
        "15",
        "--plbfgs-kkt-rtol",
        "0.8",
        "--reparam-mode",
        "state_curve",
        "--reparam-every",
        "1",
        "--state-curve-direct-solve",
        "--state-curve-direct-trial-calls",
        "15",
        "--state-curve-direct-objective-acceptance",
        "per_image",
        "--state-curve-immediate-block-relock",
        "--state-curve-stall-free-polish",
        "--state-curve-stall-free-polish-count",
        "1",
        "--reparam-ready-relative-iperp",
        "2",
        "--reparam-cv-trigger",
        "0.001",
        "--reparam-strict-cv-trigger",
        "0.001",
        "--reparam-retry-reduction",
        "0.8",
        "--reparam-recovery-calls",
        "320",
        "--reparam-recovery-inner-calls",
        "15",
        "--reparam-recovery-merge",
        "bicriteria",
        "--reparam-recovery-iperp-rtol",
        "0",
        "--reparam-equidistance-contraction",
        "0.2",
        "--stop-complete-iperp",
        "-1",
        "--frozen-subproblem",
        "discrete_b040",
        "--discrete-inner-max-steps",
        str(cap),
        "--discrete-stop-mode",
        "r08c10",
        "--discrete-residual-ratio",
        f"{ratio:g}",
        "--discrete-history",
        "7",
        "--discrete-step-multiplier-eta",
        "1.1",
        "--plbfgs-kkt-atol",
        "0",
        "--discrete-ffrbk-selector-mode",
        "one_way_takeover",
        "--total-internal-nf-budget",
        str(budget),
    ]


def _write_registration(path: Path, payload: Dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"registration already exists with drift: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def run_member(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    alpha = _alpha(args.alpha)
    ratio = _ratio(args.residual_ratio)
    cap = _cap(args.inner_max_steps)
    budget = _budget(args.internal_nf_budget)
    if args.tier == "production" and budget != PRODUCTION_INTERNAL_NF_BUDGET:
        raise ValueError("production members require exactly 5000 internal NF")
    label = member_label(alpha, ratio, cap, budget)
    output_subdir = f"runs/{args.tier}"
    command = runner_argv(
        root, alpha, ratio, cap, budget, output_subdir, label
    )
    source_names = (
        "chignolin_core.py",
        "run_ivm_v9_spring_projected_production_20260730.py",
        "frozen_discrete_b040.py",
        "run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py",
        "run_ivm_v9_discrete_selected_eta_shim.py",
        "run_ivm_v9_discrete_ffrbk_takeover_grid_shim.py",
        "run_ivm_v9_total_internal_nf_budget.py",
        Path(__file__).name,
    )
    payload: Dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "tier": args.tier,
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
        "openmm_cpu_threads": int(OPENMM_CPU_THREADS),
        "command": command,
        "source_sha256": {
            name: sha256(root / "src" / name) for name in source_names
        },
        "input_sha256": {
            name: sha256(root / "inputs" / name)
            for name in (
                "chignolin_clean.pdb",
                "common_initial_path_equidistant.npz",
                "folded_min.traj",
                "misfolded_min.traj",
            )
        },
    }
    registration = root / "registrations" / args.tier / f"{label}.json"
    _write_registration(registration, payload)
    os.environ["OPENMM_CPU_THREADS"] = OPENMM_CPU_THREADS
    runtime.main(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--residual-ratio", required=True)
    parser.add_argument("--inner-max-steps", required=True)
    parser.add_argument("--internal-nf-budget", required=True)
    parser.add_argument("--tier", choices=("smoke", "screen", "production"), required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_member(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
