#!/usr/bin/env python3
"""Isolated scalar-L-BFGS runtime arms for the frozen Ala subproblem.

This module leaves :mod:`frozen_discrete_b040` and the production runner
unchanged on disk.  At runtime it replaces only the frozen discrete solver
with one of two registered ablation arms:

``scalar_selector``
    Replace the FFrbk initial inverse by ``gamma I`` while retaining the
    original, cap-aware H0/L-BFGS selector.  With empty history ``gamma=1``;
    otherwise ``gamma=(s.T@y)/(y.T@y)`` from the newest accepted pair.

``direct_lbfgs_only``
    Use the same scalar two-loop recursion but construct only its L-BFGS
    proposal.  There is no H0 proposal and no selector competition.

Both arms solve the same fixed-path direct-Cartesian proximal problem, use
the same normal projection and dynamic whole-band cap, then multiply the
chosen/base step by the fixed ``eta=1.1`` and reapply that same cap.  There
is no objective/energy backtracking and no extra physical evaluation.  The
production runner may still build its zero-NF force-field bank for unchanged
outer recovery machinery, but this fixed-path solver receives
``preconditioner_bank=None`` and never applies FFrbk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import run_ivm_v9_discrete_selected_eta_shim as selected_eta


MODE_FLAG = "--discrete-scalar-lbfgs-mode"
VALID_MODES = ("scalar_selector", "direct_lbfgs_only")
FIXED_ALPHA = 15.0
FIXED_ETA = 1.1
FIXED_HISTORY = 7
FIXED_INNER_MAX_STEPS = 10
FIXED_STOP_MODE = "r08c10"
FIXED_RESIDUAL_RATIO = 0.8
FIXED_KKT_ATOL = 0.0
SCALAR_GAMMA_FLOOR = 1.0e-12
RUNTIME_POLICY_ID = "ala-discrete-scalar-lbfgs-runtime-ablation-v1"
SCALAR_INITIAL_INVERSE_ID = (
    "identity-I-empty_last-sy-over-yy-I-nonempty-floor1e-12-v1"
)
POST_SCALE_POLICY_ID = "post-base-step-eta-recap-v1"
MODE_POLICY_IDS = {
    "scalar_selector": "scalar-h0-vs-lbfgs-original-selector-v1",
    "direct_lbfgs_only": "direct-lbfgs-only-no-selector-v1",
}


class ScalarLBFGSSafetyError(RuntimeError):
    """Fail-closed error for this isolated runtime policy."""


class ScalarLBFGSDirectionError(ScalarLBFGSSafetyError):
    """Numerical/non-descent direction error eligible for direct fallback."""


def validate_mode(value: Any) -> str:
    mode = str(value)
    if mode not in VALID_MODES:
        raise ValueError(
            "scalar L-BFGS mode must be scalar_selector or direct_lbfgs_only"
        )
    return mode


def validate_fixed_eta(value: Any) -> float:
    eta = selected_eta.validate_eta(value)
    if eta != FIXED_ETA:
        raise ValueError("scalar L-BFGS runtime requires fixed eta=1.1")
    return eta


def _extract_exactly_one_option(
    argv: Sequence[str], flag: str
) -> Tuple[str, List[str]]:
    values: List[str] = []
    cleaned: List[str] = []
    argv = list(argv)
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == flag:
            if index + 1 >= len(argv):
                raise ValueError(f"{flag} requires one value")
            values.append(argv[index + 1])
            index += 2
            continue
        prefix = flag + "="
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
            index += 1
            continue
        cleaned.append(token)
        index += 1
    if len(values) != 1:
        raise ValueError(f"exactly one {flag} option is required")
    return values[0], cleaned


def extract_runtime_arguments(
    argv: Sequence[str],
) -> Tuple[str, float, List[str]]:
    """Remove the two shim-only options and validate their fixed values."""
    eta, without_eta = selected_eta.extract_eta_argument(argv)
    mode_text, cleaned = _extract_exactly_one_option(without_eta, MODE_FLAG)
    return validate_mode(mode_text), validate_fixed_eta(eta), cleaned


def _option_values(argv: Sequence[str], flag: str) -> List[str]:
    values: List[str] = []
    argv = list(argv)
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == flag:
            if index + 1 >= len(argv):
                raise ValueError(f"{flag} requires one value")
            values.append(argv[index + 1])
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
        raise ValueError(f"runtime requires exactly one {flag} option")
    return values[0]


def _require_empty_or_absent(argv: Sequence[str], flag: str) -> None:
    values = _option_values(argv, flag)
    if len(values) > 1 or (values and values[0] != ""):
        raise ValueError(f"runtime forbids a nonempty {flag}")


def validate_runner_argv(argv: Sequence[str]) -> None:
    """Fail before OpenMM setup if the registered control variables drift."""
    exact_text = {
        "--solver": "state_ivm",
        "--frozen-subproblem": "discrete_b040",
        "--alpha-mode": "fixed",
        "--discrete-stop-mode": FIXED_STOP_MODE,
        "--reparam-mode": "state_curve",
        "--state-curve-direct-objective-acceptance": "per_image",
    }
    for flag, expected in exact_text.items():
        if _required_option(argv, flag) != expected:
            raise ValueError(f"runtime requires {flag}={expected}")
    exact_float = {
        "--alpha": FIXED_ALPHA,
        "--discrete-residual-ratio": FIXED_RESIDUAL_RATIO,
        "--plbfgs-kkt-atol": FIXED_KKT_ATOL,
    }
    for flag, expected in exact_float.items():
        value = float(_required_option(argv, flag))
        if not math.isfinite(value) or value != expected:
            raise ValueError(f"runtime requires {flag}={expected:g}")
    exact_int = {
        "--history": FIXED_HISTORY,
        "--discrete-history": FIXED_HISTORY,
        "--discrete-inner-max-steps": FIXED_INNER_MAX_STEPS,
    }
    for flag, expected in exact_int.items():
        text = _required_option(argv, flag)
        try:
            value = int(text)
        except ValueError as error:
            raise ValueError(f"runtime requires integer {flag}") from error
        if str(value) != text.strip() or value != expected:
            raise ValueError(f"runtime requires {flag}={expected}")
    max_step = float(_required_option(argv, "--max-step"))
    if not math.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("runtime requires a finite positive --max-step")
    for flag in (
        "--alpha-schedule",
        "--alpha-precision-caps",
        "--alpha-relative-iperp-caps",
        "--max-step-schedule",
    ):
        _require_empty_or_absent(argv, flag)


def _float_token(value: float) -> str:
    return float(value).hex().replace(".", "d").replace("+", "p").replace(
        "-", "m"
    )


def shim_implementation_id(base_implementation_id: str, mode: str) -> str:
    base = str(base_implementation_id)
    if not base:
        raise ValueError("base implementation ID must be nonempty")
    mode = validate_mode(mode)
    return "+".join(
        (
            base,
            RUNTIME_POLICY_ID,
            MODE_POLICY_IDS[mode],
            SCALAR_INITIAL_INVERSE_ID,
            POST_SCALE_POLICY_ID,
            "alpha-" + _float_token(FIXED_ALPHA),
            "eta-" + _float_token(FIXED_ETA),
            "m7-r08c10-kkt0",
        )
    )


def _inner(left: Any, right: Any) -> float:
    result = float(
        np.sum(
            np.asarray(left, dtype=np.float64)
            * np.asarray(right, dtype=np.float64)
        )
    )
    if not math.isfinite(result):
        raise ScalarLBFGSSafetyError("non-finite scalar-LBFGS inner product")
    return result


def _norm(value: Any) -> float:
    result = float(np.linalg.norm(np.asarray(value, dtype=np.float64)))
    if not math.isfinite(result):
        raise ScalarLBFGSSafetyError("non-finite scalar-LBFGS norm")
    return result


def _max_atomic_displacement(value: Any) -> float:
    value = np.asarray(value, dtype=np.float64)
    result = float(np.max(np.linalg.norm(value.reshape(-1, 3), axis=1)))
    if not math.isfinite(result):
        raise ScalarLBFGSSafetyError("non-finite maximum atom displacement")
    return result


@dataclass(frozen=True)
class ScaleAudit:
    history_before: int
    source: str
    gamma: float
    raw_gamma: float
    floor_applied: bool
    last_sy: float
    last_yy: float


def scalar_initial_inverse_scale(
    history_s: Sequence[Any], history_y: Sequence[Any]
) -> ScaleAudit:
    """Return the registered identity-first/standard-last-pair scale."""
    if len(history_s) != len(history_y):
        raise ScalarLBFGSSafetyError("L-BFGS history lengths differ")
    if len(history_s) > FIXED_HISTORY:
        raise ScalarLBFGSSafetyError("L-BFGS history exceeds registered m=7")
    if not history_s:
        return ScaleAudit(
            0,
            "identity-empty-history",
            1.0,
            1.0,
            False,
            np.nan,
            np.nan,
        )
    last_s = np.asarray(history_s[-1], dtype=np.float64)
    last_y = np.asarray(history_y[-1], dtype=np.float64)
    if last_s.shape != last_y.shape or last_s.ndim != 2:
        raise ScalarLBFGSSafetyError("last L-BFGS pair shape differs")
    if not np.all(np.isfinite(last_s)) or not np.all(np.isfinite(last_y)):
        raise ScalarLBFGSSafetyError("last L-BFGS pair is non-finite")
    sy = _inner(last_s, last_y)
    yy = _inner(last_y, last_y)
    if sy <= 0.0 or yy <= 0.0:
        raise ScalarLBFGSSafetyError("last L-BFGS pair has invalid curvature")
    raw_gamma = sy / yy
    if not math.isfinite(raw_gamma) or raw_gamma <= 0.0:
        raise ScalarLBFGSSafetyError("scalar initial inverse is invalid")
    gamma = max(raw_gamma, SCALAR_GAMMA_FLOOR)
    return ScaleAudit(
        len(history_s),
        "last-sy-over-yy-with-1e-12-floor",
        float(gamma),
        float(raw_gamma),
        bool(raw_gamma < SCALAR_GAMMA_FLOOR),
        sy,
        yy,
    )


def direct_scalar_lbfgs_direction(
    gradient: Any,
    history_s: Sequence[Any],
    history_y: Sequence[Any],
    tangents: Any,
    discrete_module: Any,
) -> Tuple[np.ndarray, ScaleAudit]:
    """Joint projected standard two-loop direction with scalar initial H0."""
    gradient = np.asarray(gradient, dtype=np.float64)
    tangents = np.asarray(tangents, dtype=np.float64)
    if gradient.ndim != 2 or gradient.shape != tangents.shape:
        raise ScalarLBFGSSafetyError("gradient/tangent shape differs")
    gradient = discrete_module.project_normal(gradient, tangents)
    if not np.all(np.isfinite(gradient)):
        raise ScalarLBFGSSafetyError("projected gradient is non-finite")
    if len(history_s) != len(history_y):
        raise ScalarLBFGSSafetyError("L-BFGS history lengths differ")
    if len(history_s) > FIXED_HISTORY:
        raise ScalarLBFGSSafetyError("L-BFGS history exceeds registered m=7")

    usable: List[Tuple[np.ndarray, np.ndarray]] = []
    for raw_s, raw_y in zip(history_s, history_y):
        s_value = np.asarray(raw_s, dtype=np.float64)
        y_value = np.asarray(raw_y, dtype=np.float64)
        if s_value.shape != gradient.shape or y_value.shape != gradient.shape:
            raise ScalarLBFGSSafetyError("L-BFGS pair shape differs")
        if not np.all(np.isfinite(s_value)) or not np.all(np.isfinite(y_value)):
            raise ScalarLBFGSSafetyError("L-BFGS pair is non-finite")
        s_value = discrete_module.project_normal(s_value, tangents)
        y_value = discrete_module.project_normal(y_value, tangents)
        snorm = _norm(s_value)
        ynorm = _norm(y_value)
        curvature = _inner(s_value, y_value)
        if not (
            snorm > 0.0
            and ynorm > 0.0
            and curvature
            > float(discrete_module.CURVATURE_COSINE_MIN) * snorm * ynorm
        ):
            raise ScalarLBFGSSafetyError(
                "stored L-BFGS pair fails the registered curvature gate"
            )
        usable.append((s_value, y_value))

    scale = scalar_initial_inverse_scale(
        [pair[0] for pair in usable], [pair[1] for pair in usable]
    )
    q_value = gradient.copy()
    coefficients: List[Tuple[float, float, np.ndarray, np.ndarray]] = []
    for s_value, y_value in reversed(usable):
        rho = 1.0 / _inner(y_value, s_value)
        coefficient = rho * _inner(s_value, q_value)
        q_value -= coefficient * y_value
        coefficients.append((coefficient, rho, s_value, y_value))
    result = scale.gamma * q_value
    for coefficient, rho, s_value, y_value in reversed(coefficients):
        beta = rho * _inner(y_value, result)
        result += (coefficient - beta) * s_value
    direction = discrete_module.project_normal(-result, tangents)
    if not np.all(np.isfinite(direction)):
        raise ScalarLBFGSDirectionError(
            "scalar L-BFGS direction is non-finite"
        )
    if _inner(gradient, direction) >= 0.0:
        raise ScalarLBFGSDirectionError(
            "scalar L-BFGS direction is not descent"
        )
    return direction, scale


def _validate_fixed_solver_arguments(
    *,
    alpha: Any,
    max_step: Any,
    inner_max_steps: Any,
    stop_mode: Any,
    relative_residual: Any,
    history_size: Any,
    kkt_atol: Any,
    preconditioner_bank: Any,
) -> None:
    if float(alpha) != FIXED_ALPHA:
        raise ValueError("scalar L-BFGS solver requires fixed alpha=15")
    if not math.isfinite(float(max_step)) or float(max_step) <= 0.0:
        raise ValueError("scalar L-BFGS solver requires a positive finite cap")
    if type(inner_max_steps) is not int or inner_max_steps != FIXED_INNER_MAX_STEPS:
        raise ValueError("scalar L-BFGS solver requires inner_max_steps=10")
    if stop_mode != FIXED_STOP_MODE:
        raise ValueError("scalar L-BFGS solver requires stop_mode=r08c10")
    if float(relative_residual) != FIXED_RESIDUAL_RATIO:
        raise ValueError("scalar L-BFGS solver requires relative_residual=0.8")
    if type(history_size) is not int or history_size != FIXED_HISTORY:
        raise ValueError("scalar L-BFGS solver requires history_size=7")
    if float(kkt_atol) != FIXED_KKT_ATOL:
        raise ValueError("scalar L-BFGS solver requires kkt_atol=0")
    if preconditioner_bank is not None:
        raise ScalarLBFGSSafetyError(
            "scalar L-BFGS fixed solver requires preconditioner_bank=None"
        )


def solve_frozen_discrete_scalar_lbfgs_band(
    path: Any,
    energies: Any,
    forces: Any,
    tangents: Any,
    evaluators: Sequence[Any],
    alpha: float,
    max_step: float,
    max_new_calls: int,
    *,
    inner_max_steps: int = FIXED_INNER_MAX_STEPS,
    stop_mode: str = FIXED_STOP_MODE,
    relative_residual: float = FIXED_RESIDUAL_RATIO,
    history_size: int = FIXED_HISTORY,
    kkt_atol: float = FIXED_KKT_ATOL,
    preconditioner_bank: Any = None,
    active_images: Any = None,
    mode: str,
    eta: float,
    implementation_id: str,
    discrete_module: Any,
) -> Tuple[Any, ...]:
    """Solve one registered scalar-LBFGS frozen direct-Cartesian band."""
    mode = validate_mode(mode)
    eta = validate_fixed_eta(eta)
    _validate_fixed_solver_arguments(
        alpha=alpha,
        max_step=max_step,
        inner_max_steps=inner_max_steps,
        stop_mode=stop_mode,
        relative_residual=relative_residual,
        history_size=history_size,
        kkt_atol=kkt_atol,
        preconditioner_bank=preconditioner_bank,
    )
    path = np.asarray(path, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64)
    tangents = np.asarray(tangents, dtype=np.float64)
    n_internal = path.shape[0] - 2
    if path.ndim != 2 or path.shape[0] < 3:
        raise ValueError("fixed path must contain endpoints and internal images")
    if energies.shape != (path.shape[0],) or forces.shape != path.shape:
        raise ValueError("fixed path energy/force shape differs")
    if tangents.shape != (n_internal, path.shape[1]):
        raise ValueError("frozen tangent shape differs")
    if not (
        np.all(np.isfinite(path))
        and np.all(np.isfinite(energies))
        and np.all(np.isfinite(forces))
        and np.all(np.isfinite(tangents))
    ):
        raise ValueError("fixed path state must be finite")
    tangent_norms = np.linalg.norm(tangents, axis=1)
    if not np.allclose(tangent_norms, 1.0, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("frozen central tangents must be unit vectors")
    if active_images is not None:
        active = np.asarray(active_images, dtype=bool)
        if active.shape != (n_internal,) or not np.all(active):
            raise ValueError("joint scalar L-BFGS solver requires every image")
    if type(max_new_calls) is not int and not isinstance(max_new_calls, np.integer):
        raise ValueError("max_new_calls must be an integer")
    if int(max_new_calls) < 0:
        raise ValueError("max_new_calls must be nonnegative")
    inverse_alpha = 1.0 / FIXED_ALPHA
    reference_path = path.copy()
    current_path = path.copy()
    current_energies = energies.copy()
    current_forces = forces.copy()
    starting_counters = discrete_module.counter_arrays(evaluators)
    start_calls = starting_counters["joint_evaluations"][1:-1].copy()
    displacement = np.zeros_like(path[1:-1])
    gradient = discrete_module.project_normal(-forces[1:-1], tangents)
    residual = discrete_module.project_normal(
        gradient + inverse_alpha * displacement, tangents
    )
    initial_per_image = np.linalg.norm(residual, axis=1)
    initial_residual = _norm(residual) / np.sqrt(float(n_internal))
    if initial_residual <= 0.0:
        raise ScalarLBFGSSafetyError(
            "initial natural residual must be positive"
        )

    history_s: List[np.ndarray] = []
    history_y: List[np.ndarray] = []
    joint_steps = 0
    synchronized_paths: List[np.ndarray] = []
    synchronized_energies: List[np.ndarray] = []
    synchronized_forces: List[np.ndarray] = []
    synchronized_counters: List[Dict[str, np.ndarray]] = []
    direction_selection: List[str] = []
    h0_raw_decrease: List[float] = []
    lbfgs_raw_decrease: List[float] = []
    h0_capped_decrease: List[float] = []
    lbfgs_capped_decrease: List[float] = []
    h0_cap_scale: List[float] = []
    lbfgs_cap_scale: List[float] = []
    history_count: List[int] = []
    curvature_status: List[str] = []
    audit_history_before: List[int] = []
    audit_scale_source: List[str] = []
    audit_gamma: List[float] = []
    audit_raw_gamma: List[float] = []
    audit_gamma_floor_applied: List[bool] = []
    audit_last_sy: List[float] = []
    audit_last_yy: List[float] = []
    audit_h0_constructed: List[bool] = []
    audit_selector_used: List[bool] = []
    audit_direct_fallback: List[bool] = []
    audit_dynamic_max_step: List[float] = []
    audit_base_max: List[float] = []
    audit_requested_max: List[float] = []
    audit_applied_max: List[float] = []
    audit_requested_multiplier: List[float] = []
    audit_applied_multiplier: List[float] = []
    audit_cap_hit: List[bool] = []
    audit_base_predicted_decrease: List[float] = []
    audit_applied_predicted_decrease: List[float] = []
    uncommitted_selection_count = 0
    stop_reason = "not-started"

    def used_calls() -> np.ndarray:
        return (
            discrete_module.counter_arrays(evaluators)["joint_evaluations"][
                1:-1
            ]
            - start_calls
        ).astype(int)

    while True:
        residual_l2 = _norm(residual) / np.sqrt(float(n_internal))
        relative = residual_l2 / initial_residual
        if joint_steps > 0 and relative <= FIXED_RESIDUAL_RATIO:
            stop_reason = "evaluated-relative-natural-residual"
            break
        if joint_steps >= FIXED_INNER_MAX_STEPS:
            stop_reason = "evaluated-inner-max-steps"
            break
        current_used = used_calls()
        if np.any(current_used >= int(max_new_calls)):
            stop_reason = "per-image-new-call-limit"
            break

        direct_fallback = False
        try:
            raw_lbfgs, scale = direct_scalar_lbfgs_direction(
                residual,
                history_s,
                history_y,
                tangents,
                discrete_module,
            )
        except ScalarLBFGSDirectionError:
            if mode != "direct_lbfgs_only" or not history_s:
                raise
            # The old direct solver's only numerical safeguard: discard an
            # invalid curvature model and use the unique identity step.  This
            # is not an H0/FFrbk candidate and performs no physical query.
            history_s.clear()
            history_y.clear()
            raw_lbfgs = discrete_module.project_normal(-residual, tangents)
            scale = scalar_initial_inverse_scale(history_s, history_y)
            direct_fallback = True

        if mode == "scalar_selector":
            raw_h0 = discrete_module.project_normal(
                -scale.gamma * residual, tangents
            )
            if not history_s:
                # Exact byte tie: the original selector must choose H0.
                raw_lbfgs = np.array(raw_h0, dtype=np.float64, copy=True)
            h0 = discrete_module.screen_direction(
                "H0", raw_h0, residual, tangents, max_step
            )
            lbfgs = discrete_module.screen_direction(
                "LBFGS", raw_lbfgs, residual, tangents, max_step
            )
            selected = discrete_module.select_direction(h0, lbfgs)
            h0_constructed = True
            selector_used = True
        else:
            lbfgs = discrete_module.screen_direction(
                "LBFGS", raw_lbfgs, residual, tangents, max_step
            )
            if not lbfgs["valid"]:
                if history_s:
                    history_s.clear()
                    history_y.clear()
                    raw_lbfgs = discrete_module.project_normal(
                        -residual, tangents
                    )
                    scale = scalar_initial_inverse_scale(history_s, history_y)
                    direct_fallback = True
                    lbfgs = discrete_module.screen_direction(
                        "LBFGS", raw_lbfgs, residual, tangents, max_step
                    )
                if not lbfgs["valid"]:
                    raise ScalarLBFGSSafetyError(
                        "unique direct L-BFGS proposal is invalid: "
                        + str(lbfgs["error"])
                    )
            selected = lbfgs
            h0 = {
                "raw_decrease": 0.0,
                "capped_decrease": 0.0,
                "cap_scale": 0.0,
            }
            h0_constructed = False
            selector_used = False

        base_step = np.asarray(selected["step"], dtype=np.float64)
        base_norm = _norm(base_step)
        if base_norm <= 0.0:
            raise ScalarLBFGSSafetyError("selected/base step is zero")
        requested_step = FIXED_ETA * base_step
        applied_step, _ = discrete_module.cap_joint_direction(
            requested_step, tangents, max_step
        )
        applied_multiplier = _norm(applied_step) / base_norm
        if not (
            math.isfinite(applied_multiplier)
            and applied_multiplier > 0.0
            and applied_multiplier <= FIXED_ETA * (1.0 + 1.0e-12)
        ):
            raise ScalarLBFGSSafetyError("invalid applied eta multiplier")
        applied_decrease = -_inner(residual, applied_step)
        if applied_decrease <= 0.0:
            raise ScalarLBFGSSafetyError("eta-adjusted step is not descent")
        base_max = _max_atomic_displacement(base_step)
        requested_max = _max_atomic_displacement(requested_step)
        applied_max = _max_atomic_displacement(applied_step)
        cap_hit = bool(requested_max > float(max_step))

        trial_displacement = discrete_module.project_normal(
            displacement + applied_step, tangents
        )
        trial_path = reference_path.copy()
        trial_path[1:-1] = reference_path[1:-1] + trial_displacement
        if not np.array_equal(
            trial_path[0], reference_path[0]
        ) or not np.array_equal(trial_path[-1], reference_path[-1]):
            raise ScalarLBFGSSafetyError("fixed endpoints changed")

        blocked = np.asarray(
            [
                current_used[index] >= int(max_new_calls)
                and not evaluators[index + 1].has_cached(
                    trial_path[index + 1]
                )
                for index in range(n_internal)
            ],
            dtype=bool,
        )
        if np.any(blocked):
            uncommitted_selection_count += 1
            stop_reason = "preflight-per-image-new-call-limit"
            break

        trial_energies, trial_forces = discrete_module.evaluate_path(
            trial_path, evaluators
        )
        trial_gradient = discrete_module.project_normal(
            -trial_forces[1:-1], tangents
        )
        trial_residual = discrete_module.project_normal(
            trial_gradient + inverse_alpha * trial_displacement, tangents
        )
        if not (
            np.all(np.isfinite(trial_energies))
            and np.all(np.isfinite(trial_forces))
            and np.all(np.isfinite(trial_residual))
        ):
            raise ScalarLBFGSSafetyError("evaluated trial is non-finite")

        s_value = trial_displacement - displacement
        y_value = trial_residual - residual
        status = discrete_module.append_curvature(
            history_s, history_y, s_value, y_value, FIXED_HISTORY
        )
        displacement = trial_displacement
        current_path = trial_path
        current_energies = trial_energies
        current_forces = trial_forces
        gradient = trial_gradient
        residual = trial_residual
        joint_steps += 1

        direction_selection.append(str(selected["label"]))
        h0_raw_decrease.append(float(h0["raw_decrease"]))
        lbfgs_raw_decrease.append(float(lbfgs["raw_decrease"]))
        h0_capped_decrease.append(float(h0["capped_decrease"]))
        lbfgs_capped_decrease.append(float(lbfgs["capped_decrease"]))
        h0_cap_scale.append(float(h0["cap_scale"]))
        lbfgs_cap_scale.append(float(lbfgs["cap_scale"]))
        history_count.append(len(history_s))
        curvature_status.append(status)
        audit_history_before.append(scale.history_before)
        audit_scale_source.append(scale.source)
        audit_gamma.append(scale.gamma)
        audit_raw_gamma.append(scale.raw_gamma)
        audit_gamma_floor_applied.append(scale.floor_applied)
        audit_last_sy.append(scale.last_sy)
        audit_last_yy.append(scale.last_yy)
        audit_h0_constructed.append(h0_constructed)
        audit_selector_used.append(selector_used)
        audit_direct_fallback.append(direct_fallback)
        audit_dynamic_max_step.append(float(max_step))
        audit_base_max.append(base_max)
        audit_requested_max.append(requested_max)
        audit_applied_max.append(applied_max)
        audit_requested_multiplier.append(FIXED_ETA)
        audit_applied_multiplier.append(applied_multiplier)
        audit_cap_hit.append(cap_hit)
        audit_base_predicted_decrease.append(
            float(selected["capped_decrease"])
        )
        audit_applied_predicted_decrease.append(applied_decrease)
        synchronized_paths.append(current_path.copy())
        synchronized_energies.append(current_energies.copy())
        synchronized_forces.append(current_forces.copy())
        synchronized_counters.append(
            {
                name: values.copy()
                for name, values in discrete_module.counter_arrays(
                    evaluators
                ).items()
            }
        )

    final_per_image = np.linalg.norm(residual, axis=1)
    final_residual = _norm(residual) / np.sqrt(float(n_internal))
    relative_per_image = final_per_image / np.maximum(initial_per_image, 1.0e-30)
    global_relative = final_residual / initial_residual
    global_converged = bool(global_relative <= FIXED_RESIDUAL_RATIO)
    used = used_calls()
    hit_call_limit = used >= int(max_new_calls)
    per_image_target = FIXED_RESIDUAL_RATIO * initial_per_image
    report = {
        "initial_kkt_l2": initial_per_image,
        "final_kkt_l2": final_per_image,
        "relative_kkt": relative_per_image,
        "converged": final_per_image <= per_image_target,
        "hit_call_limit": hit_call_limit,
        "accepted_steps": np.full(n_internal, joint_steps, dtype=int),
        "backtracks": np.zeros(n_internal, dtype=int),
        "curvature_restarts": np.zeros(n_internal, dtype=int),
        "discrete_b040": True,
        "implementation_id": str(implementation_id),
        "global_initial_residual_l2": float(initial_residual),
        "global_final_residual_l2": float(final_residual),
        "global_relative_residual": float(global_relative),
        "global_converged": global_converged,
        "stop_reason": str(stop_reason),
        "joint_steps": int(joint_steps),
        "paid_trials": int(len(synchronized_paths)),
        "direction_selection": np.asarray(direction_selection, dtype="U8"),
        "h0_raw_predicted_decrease": np.asarray(h0_raw_decrease, dtype=float),
        "lbfgs_raw_predicted_decrease": np.asarray(
            lbfgs_raw_decrease, dtype=float
        ),
        "h0_capped_predicted_decrease": np.asarray(
            h0_capped_decrease, dtype=float
        ),
        "lbfgs_capped_predicted_decrease": np.asarray(
            lbfgs_capped_decrease, dtype=float
        ),
        "h0_cap_scale": np.asarray(h0_cap_scale, dtype=float),
        "lbfgs_cap_scale": np.asarray(lbfgs_cap_scale, dtype=float),
        "history_count": np.asarray(history_count, dtype=int),
        "curvature_status": np.asarray(curvature_status, dtype="U64"),
        "frozen_path_sha256": hashlib.sha256(
            np.ascontiguousarray(reference_path).tobytes()
        ).hexdigest(),
        "frozen_tangent_sha256": hashlib.sha256(
            np.ascontiguousarray(tangents).tobytes()
        ).hexdigest(),
        "all_returned_states_evaluated": True,
        "endpoints_bitwise_fixed": bool(
            np.array_equal(current_path[0], reference_path[0])
            and np.array_equal(current_path[-1], reference_path[-1])
        ),
        "scalar_lbfgs_mode": mode,
        "scalar_lbfgs_history_before": np.asarray(
            audit_history_before, dtype=int
        ),
        "scalar_lbfgs_scale_source": np.asarray(
            audit_scale_source, dtype="U32"
        ),
        "scalar_lbfgs_initial_inverse_scale": np.asarray(
            audit_gamma, dtype=float
        ),
        "scalar_lbfgs_raw_initial_inverse_scale": np.asarray(
            audit_raw_gamma, dtype=float
        ),
        "scalar_lbfgs_gamma_floor_applied": np.asarray(
            audit_gamma_floor_applied, dtype=bool
        ),
        "scalar_lbfgs_last_sy": np.asarray(audit_last_sy, dtype=float),
        "scalar_lbfgs_last_yy": np.asarray(audit_last_yy, dtype=float),
        "scalar_lbfgs_h0_constructed": np.asarray(
            audit_h0_constructed, dtype=bool
        ),
        "scalar_lbfgs_selector_used": np.asarray(
            audit_selector_used, dtype=bool
        ),
        "scalar_lbfgs_direct_fallback": np.asarray(
            audit_direct_fallback, dtype=bool
        ),
        "scalar_lbfgs_dynamic_max_step": np.asarray(
            audit_dynamic_max_step, dtype=float
        ),
        "scalar_lbfgs_base_max_atomic_displacement": np.asarray(
            audit_base_max, dtype=float
        ),
        "scalar_lbfgs_requested_max_atomic_displacement": np.asarray(
            audit_requested_max, dtype=float
        ),
        "scalar_lbfgs_applied_max_atomic_displacement": np.asarray(
            audit_applied_max, dtype=float
        ),
        "scalar_lbfgs_requested_multiplier": np.asarray(
            audit_requested_multiplier, dtype=float
        ),
        "scalar_lbfgs_applied_multiplier": np.asarray(
            audit_applied_multiplier, dtype=float
        ),
        "scalar_lbfgs_cap_hit": np.asarray(audit_cap_hit, dtype=bool),
        "scalar_lbfgs_base_predicted_decrease": np.asarray(
            audit_base_predicted_decrease, dtype=float
        ),
        "scalar_lbfgs_applied_predicted_decrease": np.asarray(
            audit_applied_predicted_decrease, dtype=float
        ),
        "scalar_lbfgs_uncommitted_selection_count": int(
            uncommitted_selection_count
        ),
        "scalar_lbfgs_fixed_solver_preconditioner_is_none": True,
        "scalar_lbfgs_energy_backtracking": False,
        "scalar_lbfgs_extra_physical_evaluations": 0,
        "scalar_lbfgs_kkt_atol": FIXED_KKT_ATOL,
        "scalar_lbfgs_absolute_kkt_stop_enabled": False,
    }
    return (
        current_path,
        current_energies,
        current_forces,
        used,
        synchronized_paths,
        synchronized_energies,
        synchronized_forces,
        synchronized_counters,
        report,
    )


@dataclass
class RuntimeAudit:
    reports: List[Dict[str, Any]] = field(default_factory=list)
    aborted_solver_calls: int = 0
    fixed_solver_calls: int = 0
    runner_bank_argument_intercepts: int = 0
    bank_instance_count: int = 0
    bank_init_raw_joint_evaluation_delta: List[int] = field(default_factory=list)
    bank_build_count: int = 0
    bank_build_raw_joint_evaluation_delta: List[int] = field(default_factory=list)
    bank_solve_normal_inside_fixed_solver: int = 0
    bank_solve_normal_outside_fixed_solver: int = 0
    fixed_solver_depth: int = 0


def _concat_reports(
    reports: Sequence[Dict[str, Any]], key: str, dtype: Any
) -> np.ndarray:
    values = [np.asarray(report[key], dtype=dtype) for report in reports]
    if not values:
        return np.asarray([], dtype=dtype)
    return np.concatenate(values)


@dataclass
class InstalledScalarLBFGSShim:
    mode: str
    eta: float
    base_implementation_id: str
    implementation_id: str
    frozen_module: Any
    original_solve: Any
    audit: RuntimeAudit
    runner_module: Any = None
    original_runner_implementation_id: Any = None
    original_runner_solve: Any = None
    runner_had_solve: bool = False
    original_atomic_save_npz: Any = None
    bank_class: Any = None
    original_bank_init: Any = None
    original_bank_build: Any = None
    original_bank_solve_normal: Any = None
    restored: bool = False

    def attach_runner(self, runner_module: Any) -> None:
        if self.runner_module is not None:
            raise RuntimeError("scalar L-BFGS shim already has a runner")
        self.runner_module = runner_module
        self.original_runner_implementation_id = (
            runner_module.DISCRETE_B040_IMPLEMENTATION_ID
        )
        self.runner_had_solve = hasattr(
            runner_module, "solve_frozen_discrete_b040_band"
        )
        if self.runner_had_solve:
            self.original_runner_solve = (
                runner_module.solve_frozen_discrete_b040_band
            )
            runner_module.solve_frozen_discrete_b040_band = (
                self.frozen_module.solve_frozen_discrete_b040_band
            )
        self.original_atomic_save_npz = runner_module.atomic_save_npz
        runner_module.DISCRETE_B040_IMPLEMENTATION_ID = self.implementation_id

        self.bank_class = runner_module.ForceFieldPreconditionerBank
        self.original_bank_init = self.bank_class.__init__
        self.original_bank_build = self.bank_class.build
        self.original_bank_solve_normal = self.bank_class.solve_normal
        runtime_audit = self.audit
        original_init = self.original_bank_init
        original_build = self.original_bank_build
        original_solve_normal = self.original_bank_solve_normal

        def audited_init(bank_self: Any, backend: Any, *args: Any, **kwargs: Any) -> None:
            before = int(backend.raw_joint_evaluations)
            original_init(bank_self, backend, *args, **kwargs)
            after = int(backend.raw_joint_evaluations)
            delta = after - before
            runtime_audit.bank_instance_count += 1
            runtime_audit.bank_init_raw_joint_evaluation_delta.append(delta)
            bank_self._scalar_lbfgs_runtime_backend = backend
            if delta != 0:
                raise ScalarLBFGSSafetyError(
                    "force-field bank construction changed physical NF"
                )

        def audited_build(bank_self: Any, path: Any) -> None:
            backend = getattr(bank_self, "_scalar_lbfgs_runtime_backend", None)
            if backend is None:
                raise ScalarLBFGSSafetyError(
                    "force-field bank lacks scalar-runtime backend audit"
                )
            before = int(backend.raw_joint_evaluations)
            original_build(bank_self, path)
            after = int(backend.raw_joint_evaluations)
            delta = after - before
            runtime_audit.bank_build_count += 1
            runtime_audit.bank_build_raw_joint_evaluation_delta.append(delta)
            if delta != 0:
                raise ScalarLBFGSSafetyError(
                    "force-field bank build changed physical NF"
                )

        def audited_solve_normal(bank_self: Any, *args: Any, **kwargs: Any) -> Any:
            if runtime_audit.fixed_solver_depth > 0:
                runtime_audit.bank_solve_normal_inside_fixed_solver += 1
                raise ScalarLBFGSSafetyError(
                    "FFrbk solve_normal is forbidden inside scalar fixed solver"
                )
            runtime_audit.bank_solve_normal_outside_fixed_solver += 1
            return original_solve_normal(bank_self, *args, **kwargs)

        self.bank_class.__init__ = audited_init
        self.bank_class.build = audited_build
        self.bank_class.solve_normal = audited_solve_normal

        def atomic_save_with_scalar(destination: Path, **arrays: Any) -> None:
            if str(np.asarray(arrays.get("frozen_subproblem", "")).item()) != (
                "discrete_b040"
            ):
                raise ScalarLBFGSSafetyError(
                    "scalar runtime requires frozen_subproblem=discrete_b040"
                )
            if str(
                np.asarray(arrays.get("discrete_implementation_id", "")).item()
            ) != self.implementation_id:
                raise ScalarLBFGSSafetyError(
                    "scalar runtime implementation ID drift"
                )
            fixed_scalars = {
                "alpha": FIXED_ALPHA,
                "alpha_mode": "fixed",
                "discrete_inner_max_steps": FIXED_INNER_MAX_STEPS,
                "discrete_stop_mode": FIXED_STOP_MODE,
                "discrete_residual_ratio": FIXED_RESIDUAL_RATIO,
                "discrete_history": FIXED_HISTORY,
                "plbfgs_kkt_atol": FIXED_KKT_ATOL,
                "reparam_mode": "state_curve",
                "state_curve_direct_objective_acceptance": "per_image",
            }
            for key, expected in fixed_scalars.items():
                if key not in arrays or np.asarray(arrays[key]).item() != expected:
                    raise ScalarLBFGSSafetyError(
                        f"scalar runtime result drift at {key}"
                    )
            active_alpha = np.asarray(arrays.get("active_alpha", []), dtype=float)
            if not active_alpha.size or not np.all(active_alpha == FIXED_ALPHA):
                raise ScalarLBFGSSafetyError(
                    "scalar runtime alpha was not fixed throughout"
                )
            selections = np.asarray(
                arrays.get("discrete_audit_selection", []), dtype="U8"
            )
            audit_selection = _concat_reports(
                self.audit.reports, "direction_selection", "U8"
            )
            if not np.array_equal(selections, audit_selection):
                raise ScalarLBFGSSafetyError(
                    "scalar runtime audit selections are misaligned"
                )
            if self.mode == "direct_lbfgs_only" and (
                np.any(selections != "LBFGS")
                or np.any(
                    _concat_reports(
                        self.audit.reports,
                        "scalar_lbfgs_h0_constructed",
                        bool,
                    )
                )
            ):
                raise ScalarLBFGSSafetyError(
                    "direct_lbfgs_only constructed/selected a forbidden H0 arm"
                )
            if self.audit.bank_solve_normal_inside_fixed_solver != 0:
                raise ScalarLBFGSSafetyError(
                    "FFrbk was applied inside scalar fixed solver"
                )
            if any(
                delta != 0
                for delta in (
                    self.audit.bank_init_raw_joint_evaluation_delta
                    + self.audit.bank_build_raw_joint_evaluation_delta
                )
            ):
                raise ScalarLBFGSSafetyError(
                    "bank construction/build changed physical NF"
                )

            additions = {
                "discrete_scalar_lbfgs_runtime_policy_id": np.array(
                    RUNTIME_POLICY_ID
                ),
                "discrete_scalar_lbfgs_mode": np.array(self.mode),
                "discrete_scalar_lbfgs_mode_policy_id": np.array(
                    MODE_POLICY_IDS[self.mode]
                ),
                "discrete_scalar_lbfgs_scalar_initial_inverse_id": np.array(
                    SCALAR_INITIAL_INVERSE_ID
                ),
                "discrete_scalar_lbfgs_post_scale_policy_id": np.array(
                    POST_SCALE_POLICY_ID
                ),
                "discrete_scalar_lbfgs_base_implementation_id": np.array(
                    self.base_implementation_id
                ),
                "discrete_scalar_lbfgs_implementation_id": np.array(
                    self.implementation_id
                ),
                "discrete_scalar_lbfgs_alpha": np.array(FIXED_ALPHA),
                "discrete_scalar_lbfgs_eta": np.array(FIXED_ETA),
                "discrete_scalar_lbfgs_empty_history_inverse_scale": np.array(
                    1.0
                ),
                "discrete_scalar_lbfgs_nonempty_history_gamma_floor": np.array(
                    SCALAR_GAMMA_FLOOR
                ),
                "discrete_scalar_lbfgs_history": np.array(FIXED_HISTORY),
                "discrete_scalar_lbfgs_stop_mode": np.array(FIXED_STOP_MODE),
                "discrete_scalar_lbfgs_residual_ratio": np.array(
                    FIXED_RESIDUAL_RATIO
                ),
                "discrete_scalar_lbfgs_inner_max_steps": np.array(
                    FIXED_INNER_MAX_STEPS
                ),
                "discrete_scalar_lbfgs_kkt_atol": np.array(FIXED_KKT_ATOL),
                "discrete_scalar_lbfgs_absolute_kkt_stop_enabled": np.array(
                    False
                ),
                "discrete_scalar_lbfgs_selector_competition": np.array(
                    self.mode == "scalar_selector"
                ),
                "discrete_scalar_lbfgs_direct_only": np.array(
                    self.mode == "direct_lbfgs_only"
                ),
                "discrete_scalar_lbfgs_fixed_solver_preconditioner_is_none": (
                    np.array(True)
                ),
                "discrete_scalar_lbfgs_ffrbk_applied_inside_fixed_solver": (
                    np.array(False)
                ),
                "discrete_scalar_lbfgs_ffrbk_solve_normal_calls_inside_fixed_solver": (
                    np.array(self.audit.bank_solve_normal_inside_fixed_solver)
                ),
                "discrete_scalar_lbfgs_ffrbk_solve_normal_calls_outside_fixed_solver": (
                    np.array(self.audit.bank_solve_normal_outside_fixed_solver)
                ),
                "discrete_scalar_lbfgs_runner_bank_argument_intercept_count": (
                    np.array(self.audit.runner_bank_argument_intercepts)
                ),
                "discrete_scalar_lbfgs_bank_instance_count": np.array(
                    self.audit.bank_instance_count
                ),
                "discrete_scalar_lbfgs_bank_build_count": np.array(
                    self.audit.bank_build_count
                ),
                "discrete_scalar_lbfgs_bank_init_raw_joint_evaluation_delta": (
                    np.asarray(
                        self.audit.bank_init_raw_joint_evaluation_delta, dtype=int
                    )
                ),
                "discrete_scalar_lbfgs_bank_build_raw_joint_evaluation_delta": (
                    np.asarray(
                        self.audit.bank_build_raw_joint_evaluation_delta, dtype=int
                    )
                ),
                "discrete_scalar_lbfgs_bank_build_zero_nf": np.array(
                    all(
                        delta == 0
                        for delta in self.audit.bank_build_raw_joint_evaluation_delta
                    )
                ),
                "discrete_scalar_lbfgs_energy_backtracking": np.array(False),
                "discrete_scalar_lbfgs_extra_physical_evaluations": np.array(0),
                "discrete_scalar_lbfgs_reparameterization_inside_subproblem": (
                    np.array(False)
                ),
                "discrete_scalar_lbfgs_aborted_solver_call_count": np.array(
                    self.audit.aborted_solver_calls
                ),
                "discrete_scalar_lbfgs_fixed_solver_call_count": np.array(
                    self.audit.fixed_solver_calls
                ),
                "discrete_scalar_lbfgs_audit_selection": audit_selection,
            }
            report_array_keys = {
                "history_before": ("scalar_lbfgs_history_before", int),
                "scale_source": ("scalar_lbfgs_scale_source", "U32"),
                "initial_inverse_scale": (
                    "scalar_lbfgs_initial_inverse_scale",
                    float,
                ),
                "raw_initial_inverse_scale": (
                    "scalar_lbfgs_raw_initial_inverse_scale",
                    float,
                ),
                "gamma_floor_applied": (
                    "scalar_lbfgs_gamma_floor_applied",
                    bool,
                ),
                "last_sy": ("scalar_lbfgs_last_sy", float),
                "last_yy": ("scalar_lbfgs_last_yy", float),
                "h0_constructed": ("scalar_lbfgs_h0_constructed", bool),
                "selector_used": ("scalar_lbfgs_selector_used", bool),
                "direct_fallback": ("scalar_lbfgs_direct_fallback", bool),
                "dynamic_max_step": (
                    "scalar_lbfgs_dynamic_max_step",
                    float,
                ),
                "base_max_atomic_displacement": (
                    "scalar_lbfgs_base_max_atomic_displacement",
                    float,
                ),
                "requested_max_atomic_displacement": (
                    "scalar_lbfgs_requested_max_atomic_displacement",
                    float,
                ),
                "applied_max_atomic_displacement": (
                    "scalar_lbfgs_applied_max_atomic_displacement",
                    float,
                ),
                "requested_multiplier": (
                    "scalar_lbfgs_requested_multiplier",
                    float,
                ),
                "applied_multiplier": (
                    "scalar_lbfgs_applied_multiplier",
                    float,
                ),
                "cap_hit": ("scalar_lbfgs_cap_hit", bool),
                "base_predicted_decrease": (
                    "scalar_lbfgs_base_predicted_decrease",
                    float,
                ),
                "applied_predicted_decrease": (
                    "scalar_lbfgs_applied_predicted_decrease",
                    float,
                ),
            }
            for suffix, (report_key, dtype) in report_array_keys.items():
                value = _concat_reports(self.audit.reports, report_key, dtype)
                if value.shape != selections.shape:
                    raise ScalarLBFGSSafetyError(
                        f"scalar audit array {suffix} is misaligned"
                    )
                additions[
                    "discrete_scalar_lbfgs_audit_" + suffix
                ] = value
            additions[
                "discrete_scalar_lbfgs_uncommitted_selection_count"
            ] = np.array(
                sum(
                    int(report["scalar_lbfgs_uncommitted_selection_count"])
                    for report in self.audit.reports
                )
            )
            collisions = sorted(set(additions).intersection(arrays))
            if collisions:
                raise ScalarLBFGSSafetyError(
                    "scalar metadata key collision: " + ", ".join(collisions)
                )
            self.original_atomic_save_npz(
                destination, **{**arrays, **additions}
            )

        runner_module.atomic_save_npz = atomic_save_with_scalar

    def restore(self) -> None:
        if self.restored:
            return
        if self.runner_module is not None:
            # ``main`` installs into frozen first and imports the runner only
            # afterwards.  A fresh runner therefore captures the shim ID and
            # wrapper by value; those captured values are not valid restore
            # targets.  Always restore the authoritative frozen base pair.
            self.runner_module.DISCRETE_B040_IMPLEMENTATION_ID = (
                self.base_implementation_id
            )
            if self.runner_had_solve:
                self.runner_module.solve_frozen_discrete_b040_band = (
                    self.original_solve
                )
            self.runner_module.atomic_save_npz = self.original_atomic_save_npz
            self.bank_class.__init__ = self.original_bank_init
            self.bank_class.build = self.original_bank_build
            self.bank_class.solve_normal = self.original_bank_solve_normal
        self.frozen_module.solve_frozen_discrete_b040_band = self.original_solve
        self.frozen_module.IMPLEMENTATION_ID = self.base_implementation_id
        self.restored = True


def install_scalar_lbfgs_shim(
    frozen_module: Any, mode: str, eta: float = FIXED_ETA
) -> InstalledScalarLBFGSShim:
    """Install one mode into a module instance without editing locked source."""
    mode = validate_mode(mode)
    eta = validate_fixed_eta(eta)
    base_id = str(frozen_module.IMPLEMENTATION_ID)
    implementation_id = shim_implementation_id(base_id, mode)
    original_solve = frozen_module.solve_frozen_discrete_b040_band
    original_signature = inspect.signature(original_solve)
    audit = RuntimeAudit()

    def solve_wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, ...]:
        bound = original_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        _validate_fixed_solver_arguments(
            alpha=arguments["alpha"],
            max_step=arguments["max_step"],
            inner_max_steps=arguments["inner_max_steps"],
            stop_mode=arguments["stop_mode"],
            relative_residual=arguments["relative_residual"],
            history_size=arguments["history_size"],
            kkt_atol=arguments["kkt_atol"],
            preconditioner_bank=None,
        )
        if arguments["preconditioner_bank"] is not None:
            audit.runner_bank_argument_intercepts += 1
        arguments["preconditioner_bank"] = None
        audit.fixed_solver_calls += 1
        audit.fixed_solver_depth += 1
        try:
            result = solve_frozen_discrete_scalar_lbfgs_band(
                *bound.args,
                **bound.kwargs,
                mode=mode,
                eta=eta,
                implementation_id=implementation_id,
                discrete_module=frozen_module,
            )
        except BaseException:
            audit.aborted_solver_calls += 1
            raise
        finally:
            audit.fixed_solver_depth -= 1
            if audit.fixed_solver_depth < 0:
                raise ScalarLBFGSSafetyError(
                    "scalar fixed-solver audit stack underflow"
                )
        report = result[8]
        if report["scalar_lbfgs_mode"] != mode:
            raise ScalarLBFGSSafetyError("scalar solver report mode drift")
        audit.reports.append(report)
        return result

    frozen_module.solve_frozen_discrete_b040_band = solve_wrapper
    frozen_module.IMPLEMENTATION_ID = implementation_id
    return InstalledScalarLBFGSShim(
        mode=mode,
        eta=eta,
        base_implementation_id=base_id,
        implementation_id=implementation_id,
        frozen_module=frozen_module,
        original_solve=original_solve,
        audit=audit,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    process_argv = list(sys.argv if argv is None else argv)
    mode, eta, runner_argv = extract_runtime_arguments(process_argv)
    validate_runner_argv(runner_argv)
    sys.argv = runner_argv
    frozen = importlib.import_module("frozen_discrete_b040")
    installed = install_scalar_lbfgs_shim(frozen, mode, eta)
    runner = importlib.import_module(
        "run_ivm_v9_spring_projected_production_20260730"
    )
    installed.attach_runner(runner)
    runner.main()


if __name__ == "__main__":
    main()
