#!/usr/bin/env python3
"""Registered Ala fixed-alpha runtime with reparameterization and a c5000 cap.

This is a narrow adapter around the frozen parametric grid runtime.  It keeps
the production one-way FFrbk/L-BFGS selector, eta=1.1, physical max step 1,
the shared 5000 internal-force budget, and ``state_curve/per_image``
reparameterization unchanged.  Only the registered control surface is
narrowed to eight decade alphas, residual ratio 0.1, and implementation safety
cap 5000.  Result validation requires that this cap never triggers.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import sys
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import run_ivm_v9_discrete_ffrbk_takeover_grid_shim as grid
import run_ivm_v9_discrete_ffrbk_takeover_selector_shim as fixed_selector
import run_ivm_v9_discrete_selected_eta_shim as selected_eta
import run_ivm_v9_total_internal_nf_budget as total_budget


POLICY_ID = "ala-discrete-one-way-selector-alpha-r01-c5000-reparam-runtime-v1"
ALPHAS = grid.ALLOWED_ALPHAS
FIXED_RESIDUAL_RATIO = 0.1
FIXED_INNER_MAX_STEPS = 5000
FIXED_TOTAL_INTERNAL_NF_BUDGET = 5000
FIXED_ETA = grid.FIXED_ETA
FIXED_STOP_MODE = grid.FIXED_STOP_MODE


class AlphaRuntimeSafetyError(RuntimeError):
    """Fail-closed error for scientific-identity or composition drift."""


def validate_controls(controls: grid.GridControls) -> grid.GridControls:
    checked = grid.validate_grid_controls(controls)
    if checked.alpha not in ALPHAS:
        raise ValueError("alpha is not in the registered decade grid")
    if checked.residual_ratio != FIXED_RESIDUAL_RATIO:
        raise ValueError("alpha campaign requires residual ratio 0.1")
    if checked.inner_max_steps != FIXED_INNER_MAX_STEPS:
        raise ValueError("alpha campaign requires inner max steps 5000")
    if checked.total_internal_nf_budget != FIXED_TOTAL_INTERNAL_NF_BUDGET:
        raise ValueError("alpha campaign requires total internal NF 5000")
    return checked


def controls_from_runner_argv(
    runner_argv: Sequence[str], total_internal_nf_budget: int
) -> grid.GridControls:
    try:
        controls = grid.GridControls(
            alpha=float(grid._required_option(runner_argv, "--alpha")),
            residual_ratio=float(
                grid._required_option(
                    runner_argv, "--discrete-residual-ratio"
                )
            ),
            inner_max_steps=int(
                grid._required_option(
                    runner_argv, "--discrete-inner-max-steps"
                )
            ),
            total_internal_nf_budget=total_internal_nf_budget,
        )
    except ValueError as error:
        raise ValueError("alpha runtime received invalid registered controls") from error
    return validate_controls(controls)


def extract_runtime_arguments(
    argv: Sequence[str],
) -> Tuple[grid.GridControls, str, float, List[str]]:
    declared_budget, without_budget = (
        total_budget.extract_total_internal_nf_budget(argv)
    )
    mode, eta, runner_argv = fixed_selector.extract_runtime_arguments(
        without_budget
    )
    controls = controls_from_runner_argv(runner_argv, declared_budget)
    return controls, mode, eta, runner_argv


@contextmanager
def configured_selector_policy(
    selector_module: Any, controls: grid.GridControls
) -> Iterator[Any]:
    """Reuse the frozen grid installer under this adapter's injective ID."""

    controls = validate_controls(controls)
    previous_policy_id = grid.POLICY_ID
    grid.POLICY_ID = POLICY_ID
    try:
        with grid.configured_selector_policy(
            selector_module, controls
        ) as configured:
            yield configured
    finally:
        grid.POLICY_ID = previous_policy_id


def validate_runner_argv(
    selector_module: Any,
    runner_argv: Sequence[str],
    controls: grid.GridControls,
    mode: str,
    eta: float,
) -> None:
    """Validate the unchanged production algorithm before OpenMM setup."""

    controls = validate_controls(controls)
    if mode != "one_way_takeover":
        raise ValueError("alpha runtime requires one_way_takeover")
    if float(eta) != FIXED_ETA:
        raise ValueError("alpha runtime requires eta=1.1")
    selector_module.validate_runner_argv(runner_argv)
    exact = {
        "--solver": "state_ivm",
        "--alpha-mode": "fixed",
        "--budget": "5000",
        "--max-step": "1",
        "--history": "7",
        "--state-globalization": "iperp_monotone",
        "--reparam-mode": "state_curve",
        "--reparam-every": "1",
        "--state-curve-direct-objective-acceptance": "per_image",
        "--frozen-subproblem": "discrete_b040",
        "--discrete-inner-max-steps": "5000",
        "--discrete-stop-mode": FIXED_STOP_MODE,
        "--discrete-residual-ratio": "0.1",
        "--discrete-history": "7",
        "--plbfgs-kkt-atol": "0",
    }
    for flag, expected in exact.items():
        if grid._required_option(runner_argv, flag) != expected:
            raise ValueError(f"alpha runtime requires {flag}={expected}")
    if list(runner_argv).count("--state-curve-direct-solve") != 1:
        raise ValueError("alpha runtime requires state-curve direct solve")


def _print_help() -> None:
    print(
        "usage: this audited runtime is invoked by "
        "run_stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_member.py\n"
        "fixed controls: r=0.1, inner-max-steps=5000, "
        "total-internal-nf-budget=5000, state_curve/per_image reparameterization"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    process_argv = list(sys.argv if argv is None else argv)
    if process_argv[1:] in (["--help"], ["-h"]):
        _print_help()
        return
    controls, mode, eta, runner_argv = extract_runtime_arguments(process_argv)
    old_argv = sys.argv
    budget_installed = None
    preflight_installed = None
    selector_installed = None
    eta_installed = None
    with configured_selector_policy(fixed_selector, controls) as selector_module:
        validate_runner_argv(
            selector_module, runner_argv, controls, mode, eta
        )
        sys.argv = list(runner_argv)
        ala_core = importlib.import_module("ala_core")
        budget_installed = total_budget.install_total_internal_nf_budget(
            ala_core,
            controls.total_internal_nf_budget,
            completeness_policy=(
                total_budget.COMPLETENESS_INITIAL_OR_REPARAM_COMMITTED
            ),
        )
        try:
            frozen = importlib.import_module("frozen_discrete_b040")
            preflight_installed = grid.install_full_band_budget_preflight(
                frozen, budget_installed
            )
            selector_installed = selector_module.install_ffrbk_selector_shim(
                frozen, mode
            )
            eta_installed = selected_eta.install_frozen_eta_shim(frozen, eta)
            expected_final = selector_module.final_implementation_id(
                selector_installed.base_implementation_id, mode
            )
            if eta_installed.implementation_id != expected_final:
                raise AlphaRuntimeSafetyError(
                    "unexpected selector/eta implementation identity"
                )
            runner = importlib.import_module(
                "run_ivm_v9_spring_projected_production_20260730"
            )
            eta_installed.attach_runner(runner)
            selector_installed.attach_runner(
                runner, eta_installed.implementation_id
            )
            budget_installed.attach_runner(runner)
            runner.main()
        finally:
            if budget_installed is not None:
                budget_installed.restore()
            if eta_installed is not None:
                eta_installed.restore()
            if selector_installed is not None:
                selector_installed.restore()
            if preflight_installed is not None:
                preflight_installed.restore()
            sys.argv = old_argv


if __name__ == "__main__":
    main()
