#!/usr/bin/env python3
"""Parametric one-way FFrbk/L-BFGS selector for the Chignolin grid.

This adapter reuses the already tested selector implementation and the locked
post-selection eta shim without changing either source file.  In one fresh
process it configures the selector's formerly fixed alpha/residual/cap audit
contract, installs eta=1.1, and composes the true total-internal-NF limiter.

The reusable :func:`configured_selector_policy` context contains no
reparameterization assumption.  The CLI entry point deliberately retains the
current inexact+reparameterization scientific identity and validates
``state_curve/per_image`` before OpenMM setup.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import importlib
import inspect
import math
import sys
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import run_ivm_v9_discrete_ffrbk_takeover_selector_shim as fixed_selector
import run_ivm_v9_discrete_selected_eta_shim as selected_eta
import run_ivm_v9_total_internal_nf_budget as total_budget


POLICY_ID = "chignolin-discrete-one-way-selector-parametric-grid-runtime-v2"
FIXED_ETA = 1.1
FIXED_HISTORY = 7
FIXED_KKT_ATOL = 0.0
FIXED_PHYSICAL_MAX_STEP = 1.0
FIXED_TOTAL_INTERNAL_NF_BUDGET = 5000
FIXED_STOP_MODE = "r08c10"
ALLOWED_ALPHAS = tuple(float(10**power) for power in range(-1, 7))
ALLOWED_RESIDUAL_RATIOS = (0.1, 0.3, 0.5, 0.7, 0.8)
INEXACT_GRID_INNER_MAX_STEPS = (5, 8, 10, 15, 20, 50)
# The reusable installer also admits the exact/no-reparameterization arm's
# finite implementation safety cap.  Its CLI describes that cap as effectively
# infinite relative to the 5000-NF run; it is not an inexact-grid cell.
ALLOWED_INNER_MAX_STEPS = INEXACT_GRID_INNER_MAX_STEPS + (250, 5000)


class ParametricSelectorSafetyError(RuntimeError):
    """Fail-closed error for a grid-control or composition drift."""


@dataclass(frozen=True)
class GridControls:
    alpha: float
    residual_ratio: float
    inner_max_steps: int
    total_internal_nf_budget: int = FIXED_TOTAL_INTERNAL_NF_BUDGET


@dataclass
class InstalledFullBandBudgetPreflight:
    frozen_module: Any
    budget_installed: Any
    original_solve: Any
    solve_wrapper: Any = None
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        if self.frozen_module.solve_frozen_discrete_b040_band is self.solve_wrapper:
            self.frozen_module.solve_frozen_discrete_b040_band = self.original_solve
        self.restored = True


def install_full_band_budget_preflight(
    frozen_module: Any, budget_installed: Any
) -> InstalledFullBandBudgetPreflight:
    """Crop primary joint steps to complete 20-image evaluations.

    Recovery/reparameterization has its own production-runner exception
    transaction.  The primary frozen solver instead receives only as many
    whole-band steps as the remaining shared budget can fund.  With fewer than
    20 calls left it returns its already evaluated input state without issuing
    an OpenMM query, allowing the runner to save a normal terminal artifact.
    """

    original = frozen_module.solve_frozen_discrete_b040_band
    signature = inspect.signature(original)
    n_internal = int(budget_installed.ala_core_module.N_INTERNAL)
    installed = InstalledFullBandBudgetPreflight(
        frozen_module=frozen_module,
        budget_installed=budget_installed,
        original_solve=original,
    )

    @wraps(original)
    def solve_with_total_budget_preflight(*args: Any, **kwargs: Any):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        requested = int(bound.arguments["max_new_calls"])
        remaining = max(
            0,
            int(budget_installed.budget)
            - int(budget_installed.successful_internal_joint_evaluations),
        )
        affordable_joint_steps = remaining // n_internal
        effective = min(requested, affordable_joint_steps)
        budget_installed.full_band_preflight_calls += 1
        if effective < requested:
            budget_installed.full_band_preflight_crops += 1
        if effective == 0 and requested > 0:
            budget_installed.full_band_preflight_zero_step_stops += 1
            budget_installed.complete_band_budget_stop_count += 1
        bound.arguments["max_new_calls"] = effective
        return original(*bound.args, **bound.kwargs)

    installed.solve_wrapper = solve_with_total_budget_preflight
    frozen_module.solve_frozen_discrete_b040_band = (
        solve_with_total_budget_preflight
    )
    return installed


def validate_grid_controls(controls: GridControls) -> GridControls:
    alpha = float(controls.alpha)
    ratio = float(controls.residual_ratio)
    cap = int(controls.inner_max_steps)
    budget = total_budget.validate_total_internal_nf_budget(
        controls.total_internal_nf_budget
    )
    if not math.isfinite(alpha) or alpha not in ALLOWED_ALPHAS:
        raise ValueError("alpha is not in the registered decade grid")
    if not math.isfinite(ratio) or ratio not in ALLOWED_RESIDUAL_RATIOS:
        raise ValueError("residual ratio is not in the registered grid")
    if (
        str(cap) != str(controls.inner_max_steps).strip()
        or cap not in ALLOWED_INNER_MAX_STEPS
    ):
        raise ValueError("inner max steps is not in the registered grid")
    return GridControls(alpha, ratio, cap, budget)


def _float_id_token(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("implementation-ID float must be finite")
    return value.hex().replace(".", "d").replace("+", "p").replace("-", "m")


def selector_control_token(controls: GridControls) -> str:
    controls = validate_grid_controls(controls)
    return "-".join(
        (
            "alpha" + _float_id_token(controls.alpha),
            "r" + _float_id_token(controls.residual_ratio),
            "c" + str(controls.inner_max_steps),
            "m7",
            "kkt0",
            "totalinternalnf" + str(controls.total_internal_nf_budget),
        )
    )


@contextmanager
def configured_selector_policy(
    selector_module: Any, controls: GridControls
) -> Iterator[Any]:
    """Temporarily parameterize the old selector's audit contract.

    Selection numerics are not reimplemented here.  The old module's install
    function still wraps the same frozen solver and selector; only its
    formerly hard-coded scientific constants and injective implementation-ID
    builder are supplied from the registered grid cell.
    """

    controls = validate_grid_controls(controls)
    names = (
        "FIXED_ALPHA",
        "FIXED_ETA",
        "FIXED_HISTORY",
        "FIXED_INNER_MAX_STEPS",
        "FIXED_STOP_MODE",
        "FIXED_RESIDUAL_RATIO",
        "FIXED_KKT_ATOL",
        "FIXED_MAX_STEP",
        "RUNTIME_POLICY_ID",
        "selector_implementation_id",
        "final_implementation_id",
    )
    prior: Dict[str, Any] = {name: getattr(selector_module, name) for name in names}

    def implementation_id(base_implementation_id: str, mode: str) -> str:
        base = str(base_implementation_id)
        if not base:
            raise ValueError("base implementation ID must be nonempty")
        mode = selector_module.validate_mode(mode)
        return "+".join(
            (
                base,
                POLICY_ID,
                selector_module.MODE_POLICY_IDS[mode],
                selector_module.OPERATOR_POLICY_ID,
                selector_control_token(controls),
            )
        )

    def final_implementation_id(base_implementation_id: str, mode: str) -> str:
        return selected_eta.shim_implementation_id(
            implementation_id(base_implementation_id, mode), FIXED_ETA
        )

    replacements = {
        "FIXED_ALPHA": controls.alpha,
        "FIXED_ETA": FIXED_ETA,
        "FIXED_HISTORY": FIXED_HISTORY,
        "FIXED_INNER_MAX_STEPS": controls.inner_max_steps,
        "FIXED_STOP_MODE": FIXED_STOP_MODE,
        "FIXED_RESIDUAL_RATIO": controls.residual_ratio,
        "FIXED_KKT_ATOL": FIXED_KKT_ATOL,
        "FIXED_MAX_STEP": FIXED_PHYSICAL_MAX_STEP,
        "RUNTIME_POLICY_ID": POLICY_ID,
        "selector_implementation_id": implementation_id,
        "final_implementation_id": final_implementation_id,
    }
    for name, value in replacements.items():
        setattr(selector_module, name, value)
    try:
        yield selector_module
    finally:
        for name, value in prior.items():
            setattr(selector_module, name, value)


def _option_values(argv: Sequence[str], flag: str) -> List[str]:
    values: List[str] = []
    tokens = list(argv)
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == flag:
            if index + 1 >= len(tokens):
                raise ValueError(f"{flag} requires one value")
            values.append(tokens[index + 1])
            index += 2
            continue
        prefix = flag + "="
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
        index += 1
    return values


def _required_option(argv: Sequence[str], flag: str) -> str:
    values = _option_values(argv, flag)
    if len(values) != 1:
        raise ValueError(f"grid runtime requires exactly one {flag} option")
    return values[0]


def controls_from_runner_argv(
    runner_argv: Sequence[str], total_internal_nf_budget: int
) -> GridControls:
    try:
        cap = int(_required_option(runner_argv, "--discrete-inner-max-steps"))
    except ValueError as error:
        raise ValueError("grid runtime requires an integer inner cap") from error
    return validate_grid_controls(
        GridControls(
            alpha=float(_required_option(runner_argv, "--alpha")),
            residual_ratio=float(
                _required_option(runner_argv, "--discrete-residual-ratio")
            ),
            inner_max_steps=cap,
            total_internal_nf_budget=total_internal_nf_budget,
        )
    )


def extract_runtime_arguments(
    argv: Sequence[str],
) -> Tuple[GridControls, str, float, List[str]]:
    declared_budget, without_budget = (
        total_budget.extract_total_internal_nf_budget(argv)
    )
    mode, eta, runner_argv = fixed_selector.extract_runtime_arguments(
        without_budget
    )
    controls = controls_from_runner_argv(runner_argv, declared_budget)
    return controls, mode, eta, runner_argv


def validate_inexact_reparam_runner_argv(
    selector_module: Any,
    runner_argv: Sequence[str],
    controls: GridControls,
    mode: str,
    eta: float,
) -> None:
    """Validate the current algorithm identity before any OpenMM setup."""

    controls = validate_grid_controls(controls)
    if controls.inner_max_steps not in INEXACT_GRID_INNER_MAX_STEPS:
        raise ValueError("inexact grid runtime received a non-grid inner cap")
    if mode != "one_way_takeover":
        raise ValueError("grid runtime requires one_way_takeover")
    if float(eta) != FIXED_ETA:
        raise ValueError("grid runtime requires eta=1.1")
    selector_module.validate_runner_argv(runner_argv)
    if int(_required_option(runner_argv, "--budget")) != int(
        controls.total_internal_nf_budget
    ):
        raise ValueError(
            "runner per-image safety budget must equal the total budget; "
            "the shared limiter remains authoritative"
        )
    exact = {
        "--reparam-mode": "state_curve",
        "--reparam-every": "1",
        "--state-curve-direct-objective-acceptance": "per_image",
        "--plbfgs-kkt-atol": "0",
        "--max-step": "1",
        "--discrete-stop-mode": FIXED_STOP_MODE,
    }
    for flag, expected in exact.items():
        if _required_option(runner_argv, flag) != expected:
            raise ValueError(f"grid runtime requires {flag}={expected}")
    if list(runner_argv).count("--state-curve-direct-solve") != 1:
        raise ValueError("grid runtime requires state-curve direct solve")


def main(argv: Optional[Sequence[str]] = None) -> None:
    process_argv = list(sys.argv if argv is None else argv)
    controls, mode, eta, runner_argv = extract_runtime_arguments(process_argv)
    old_argv = sys.argv
    budget_installed = None
    preflight_installed = None
    selector_installed = None
    eta_installed = None
    with configured_selector_policy(fixed_selector, controls) as selector_module:
        validate_inexact_reparam_runner_argv(
            selector_module, runner_argv, controls, mode, eta
        )
        sys.argv = list(runner_argv)
        chignolin_core = importlib.import_module("chignolin_core")
        budget_installed = total_budget.install_total_internal_nf_budget(
            chignolin_core,
            controls.total_internal_nf_budget,
            completeness_policy=(
                total_budget.COMPLETENESS_INITIAL_OR_REPARAM_COMMITTED
            ),
        )
        try:
            frozen = importlib.import_module("frozen_discrete_b040")
            preflight_installed = install_full_band_budget_preflight(
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
                raise ParametricSelectorSafetyError(
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
