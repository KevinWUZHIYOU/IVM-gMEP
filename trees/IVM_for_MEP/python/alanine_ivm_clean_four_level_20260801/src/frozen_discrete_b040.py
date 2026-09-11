"""Discrete fixed-path IVM subproblem inspired by BB B040.

This module deliberately contains no spline/DeepLR representation and no
reparameterization.  One outer state freezes the alanine path, its central
tangents, and a force-field preconditioner bank.  The optimizer acts on the
joint ``(n_internal, n_cartesian)`` normal displacement array.

At every fully evaluated state, a pure force-field ``H0`` direction and an
``H0``-initialized m-history L-BFGS direction are independently capped by the
same whole-band atom-displacement limit.  The direction with the larger
realized first-order predicted decrease is evaluated; an exact tie selects
``H0``.  Only the selected, physically evaluated step can enter a secant
pair.  Unlike BB B040's POST30 convention, every returned alanine state has a
complete energy/force evaluation.
"""

from __future__ import annotations

import hashlib

import numpy as np

IMPLEMENTATION_ID = "ala-frozen-discrete-ffrbk-dsel-m7-evaluated-v1"
CANONICAL_STOP_MODE = "r08c10"
VALID_STOP_MODES = (CANONICAL_STOP_MODE, "r08", "c10")
RESIDUAL_STOP_MODES = frozenset((CANONICAL_STOP_MODE, "r08"))
CURVATURE_COSINE_MIN = 1.0e-4
COUNTER_NAMES = (
    "joint_evaluations",
    "energy_requests",
    "force_requests",
    "cache_hits",
    "total_query_events",
    "both_requests",
    "budget_blocked_misses",
)


class DiscreteB040SafetyError(RuntimeError):
    """Fail-closed error for an invalid fixed-path optimizer state."""


def project_normal(vectors, tangents):
    """Project rows using the runner's frozen unit-tangent convention."""
    vectors = np.asarray(vectors, dtype=np.float64)
    tangents = np.asarray(tangents, dtype=np.float64)
    return vectors - np.sum(vectors * tangents, axis=1, keepdims=True) * tangents


def clip_max_atomic_displacement(direction, max_step):
    """Apply one common scale from the largest atom displacement."""
    direction = np.asarray(direction, dtype=np.float64)
    largest = float(
        np.max(np.linalg.norm(direction.reshape(-1, 3), axis=1))
    )
    if largest > float(max_step):
        return direction * (float(max_step) / largest)
    return direction


def counter_arrays(evaluators):
    """Read the same counter vectors as ``ala_core.counter_arrays``."""
    return {
        name: np.asarray(
            [getattr(evaluator.counters, name) for evaluator in evaluators],
            dtype=int,
        )
        for name in COUNTER_NAMES
    }


def evaluate_path(path, evaluators):
    """Issue one cached joint energy/force request per path image."""
    path = np.asarray(path, dtype=np.float64)
    if len(evaluators) != path.shape[0]:
        raise ValueError("evaluator count differs from path image count")
    energies = np.empty(path.shape[0], dtype=np.float64)
    forces = np.empty_like(path)
    for image_index, (position, evaluator) in enumerate(zip(path, evaluators)):
        energy, force = evaluator.query(position, "both")
        energies[image_index] = energy
        forces[image_index] = force
    return energies, forces


def _inner(left, right) -> float:
    value = float(np.sum(np.asarray(left) * np.asarray(right)))
    if not np.isfinite(value):
        raise DiscreteB040SafetyError("non-finite optimizer inner product")
    return value


def _norm(value) -> float:
    result = float(np.linalg.norm(np.asarray(value)))
    if not np.isfinite(result):
        raise DiscreteB040SafetyError("non-finite optimizer norm")
    return result


def _array_sha256(value) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value, dtype=np.float64)).tobytes()
    ).hexdigest()


def _proximal_inverse_alpha(alpha: float) -> float:
    value = float(alpha)
    if value <= 0.0 or np.isnan(value):
        raise ValueError("alpha must be positive or +inf")
    return 0.0 if np.isinf(value) else 1.0 / value


def apply_block_h0_inverse(
    covector,
    tangents,
    preconditioner_bank,
    alpha,
):
    """Apply the frozen block-diagonal FFrbk-plus-proximal inverse."""
    covector = np.asarray(covector, dtype=np.float64)
    tangents = np.asarray(tangents, dtype=np.float64)
    if covector.shape != tangents.shape or covector.ndim != 2:
        raise ValueError("joint covector/tangent shape differs")
    if preconditioner_bank is None:
        raise DiscreteB040SafetyError("frozen FFrbk bank is required")
    shift = _proximal_inverse_alpha(alpha)
    result = np.empty_like(covector)
    for image_index in range(covector.shape[0]):
        result[image_index] = preconditioner_bank.solve_normal(
            image_index,
            covector[image_index],
            tangents[image_index],
            diagonal_shift=shift,
        )
    result = project_normal(result, tangents)
    if not np.all(np.isfinite(result)):
        raise DiscreteB040SafetyError("H0 inverse returned non-finite values")
    return result


def joint_lbfgs_direction(
    gradient,
    history_s,
    history_y,
    apply_initial_inverse,
):
    """Return the joint L-BFGS direction with an operator-valued H0."""
    q = np.asarray(gradient, dtype=np.float64).copy()
    if len(history_s) != len(history_y):
        raise DiscreteB040SafetyError("L-BFGS history lengths differ")
    coefficients = []
    for s_value, y_value in reversed(tuple(zip(history_s, history_y))):
        if np.asarray(s_value).shape != q.shape or np.asarray(y_value).shape != q.shape:
            raise DiscreteB040SafetyError("L-BFGS pair shape differs")
        ys = _inner(y_value, s_value)
        if ys <= 0.0:
            raise DiscreteB040SafetyError("stored curvature is not positive")
        rho = 1.0 / ys
        coefficient = rho * _inner(s_value, q)
        q -= coefficient * y_value
        coefficients.append((coefficient, rho, s_value, y_value))
    result = np.asarray(apply_initial_inverse(q), dtype=np.float64)
    if result.shape != q.shape:
        raise DiscreteB040SafetyError("H0 inverse changed joint shape")
    for coefficient, rho, s_value, y_value in reversed(coefficients):
        beta = rho * _inner(y_value, result)
        result += (coefficient - beta) * s_value
    return -result


def cap_joint_direction(direction, tangents, max_step):
    """Project and cap one whole-band direction with one common scale."""
    direction = np.asarray(direction, dtype=np.float64)
    tangents = np.asarray(tangents, dtype=np.float64)
    if direction.shape != tangents.shape or direction.ndim != 2:
        raise DiscreteB040SafetyError("direction/tangent shape differs")
    if not np.all(np.isfinite(direction)):
        raise DiscreteB040SafetyError("raw direction is non-finite")
    max_step = float(max_step)
    if not np.isfinite(max_step) or max_step <= 0.0:
        raise DiscreteB040SafetyError("whole-band cap must be positive")
    projected = project_normal(direction, tangents)
    capped = clip_max_atomic_displacement(projected, max_step)
    capped = project_normal(capped, tangents)
    raw_norm = _norm(projected)
    capped_norm = _norm(capped)
    if raw_norm <= 0.0 or capped_norm <= 0.0:
        raise DiscreteB040SafetyError("direction is identically zero")
    scale = capped_norm / raw_norm
    if not np.isfinite(scale) or not 0.0 < scale <= 1.0 + 1.0e-14:
        raise DiscreteB040SafetyError("invalid whole-band cap scale")
    return capped, float(min(scale, 1.0))


def screen_direction(label, raw_direction, gradient, tangents, max_step):
    """Return a fail-closed proposal after the common Ala cap."""
    raw_decrease = 0.0
    try:
        raw = project_normal(
            np.asarray(raw_direction, dtype=np.float64), tangents
        )
        if raw.shape != np.asarray(gradient).shape or not np.all(np.isfinite(raw)):
            raise DiscreteB040SafetyError(label + " raw direction is invalid")
        raw_decrease = -_inner(gradient, raw)
        if raw_decrease <= 0.0:
            raise DiscreteB040SafetyError(label + " raw direction is not descent")
        step, scale = cap_joint_direction(raw, tangents, max_step)
        capped_decrease = -_inner(gradient, step)
        if capped_decrease <= 0.0:
            raise DiscreteB040SafetyError(label + " capped direction is not descent")
        return {
            "label": str(label),
            "valid": True,
            "error": "none",
            "raw": raw,
            "step": step,
            "raw_decrease": float(raw_decrease),
            "capped_decrease": float(capped_decrease),
            "cap_scale": float(scale),
        }
    except (ValueError, RuntimeError, FloatingPointError) as error:
        return {
            "label": str(label),
            "valid": False,
            "error": type(error).__name__ + ":" + str(error),
            "raw": None,
            "step": None,
            "raw_decrease": float(raw_decrease),
            "capped_decrease": 0.0,
            "cap_scale": 0.0,
        }


def select_direction(h0_proposal, lbfgs_proposal):
    """Implement B040's dynamic capped-decrease selection exactly."""
    if h0_proposal["valid"] and lbfgs_proposal["valid"]:
        if (
            h0_proposal["capped_decrease"]
            >= lbfgs_proposal["capped_decrease"]
        ):
            return h0_proposal
        return lbfgs_proposal
    if h0_proposal["valid"]:
        return h0_proposal
    if lbfgs_proposal["valid"]:
        return lbfgs_proposal
    raise DiscreteB040SafetyError(
        "both H0 and LBFGS capped direction proposals are invalid: "
        f"H0={h0_proposal['error']}; LBFGS={lbfgs_proposal['error']}"
    )


def append_curvature(history_s, history_y, s_value, y_value, history_size):
    """Append one same-frozen-path pair through B040's cosine gate."""
    s_value = np.asarray(s_value, dtype=np.float64)
    y_value = np.asarray(y_value, dtype=np.float64)
    snorm = _norm(s_value)
    ynorm = _norm(y_value)
    curvature = _inner(s_value, y_value)
    accepted = bool(
        snorm > 0.0
        and ynorm > 0.0
        and curvature > CURVATURE_COSINE_MIN * snorm * ynorm
    )
    if accepted:
        history_s.append(s_value.copy())
        history_y.append(y_value.copy())
        while len(history_s) > int(history_size):
            history_s.pop(0)
            history_y.pop(0)
    return (
        "accepted-same-frozen-path-plus-exact-prox"
        if accepted
        else "rejected-positive-curvature-cosine-gate"
    )


def solve_frozen_discrete_b040_band(
    path,
    energies,
    forces,
    tangents,
    evaluators,
    alpha,
    max_step,
    max_new_calls,
    *,
    inner_max_steps=10,
    stop_mode="r08",
    relative_residual=0.8,
    history_size=7,
    kkt_atol=1.0e-8,
    preconditioner_bank=None,
    active_images=None,
):
    """Solve one joint discrete fixed-path proximal subproblem.

    The return value intentionally matches the legacy Ala solver's 9-tuple.
    ``max_new_calls`` is a per-image physical-query limit for this outer;
    ``inner_max_steps`` counts complete joint candidate evaluations.
    """
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
            raise ValueError(
                "joint discrete solver requires every internal image"
            )
    if type(inner_max_steps) is not int or inner_max_steps < 1:
        raise ValueError("inner_max_steps must be a positive exact int")
    if type(history_size) is not int or history_size < 1:
        raise ValueError("history_size must be a positive exact int")
    if stop_mode not in VALID_STOP_MODES:
        raise ValueError("stop_mode must be r08c10, r08, or c10")
    relative_residual = float(relative_residual)
    if not 0.0 < relative_residual < 1.0:
        raise ValueError("relative_residual must lie in (0,1)")
    if int(max_new_calls) < 0:
        raise ValueError("max_new_calls must be nonnegative")
    inverse_alpha = _proximal_inverse_alpha(alpha)

    reference_path = path.copy()
    current_path = path.copy()
    current_energies = energies.copy()
    current_forces = forces.copy()
    starting_counters = counter_arrays(evaluators)
    start_calls = starting_counters["joint_evaluations"][1:-1].copy()
    displacement = np.zeros_like(path[1:-1])
    gradient = project_normal(-forces[1:-1], tangents)
    residual = project_normal(gradient + inverse_alpha * displacement, tangents)
    initial_per_image = np.linalg.norm(residual, axis=1)
    initial_residual = _norm(residual) / np.sqrt(float(n_internal))
    if initial_residual <= 0.0:
        raise DiscreteB040SafetyError("initial natural residual must be positive")

    history_s = []
    history_y = []
    joint_steps = 0
    synchronized_paths = []
    synchronized_energies = []
    synchronized_forces = []
    synchronized_counters = []
    direction_selection = []
    h0_raw_decrease = []
    lbfgs_raw_decrease = []
    h0_capped_decrease = []
    lbfgs_capped_decrease = []
    h0_cap_scale = []
    lbfgs_cap_scale = []
    history_count = []
    curvature_status = []
    stop_reason = "not-started"

    def used_calls():
        return (
            counter_arrays(evaluators)["joint_evaluations"][1:-1]
            - start_calls
        ).astype(int)

    while True:
        residual_l2 = _norm(residual) / np.sqrt(float(n_internal))
        relative = residual_l2 / initial_residual
        if joint_steps > 0 and residual_l2 <= float(kkt_atol):
            stop_reason = "evaluated-absolute-natural-residual"
            break
        if (
            stop_mode in RESIDUAL_STOP_MODES
            and joint_steps > 0
            and relative <= relative_residual
        ):
            stop_reason = "evaluated-relative-natural-residual"
            break
        if joint_steps >= int(inner_max_steps):
            stop_reason = "evaluated-inner-max-steps"
            break
        current_used = used_calls()
        if np.any(current_used >= int(max_new_calls)):
            stop_reason = "per-image-new-call-limit"
            break

        def apply_h0(value):
            return apply_block_h0_inverse(
                value, tangents, preconditioner_bank, alpha
            )

        h0_inverse_gradient = apply_h0(residual)
        raw_h0 = -np.asarray(h0_inverse_gradient, dtype=np.float64)
        if history_s:
            raw_lbfgs = joint_lbfgs_direction(
                residual, history_s, history_y, apply_h0
            )
        else:
            # Preserve exact bytes so the first deterministic tie selects H0.
            raw_lbfgs = np.array(raw_h0, dtype=np.float64, copy=True)
        h0 = screen_direction("H0", raw_h0, residual, tangents, max_step)
        lbfgs = screen_direction(
            "LBFGS", raw_lbfgs, residual, tangents, max_step
        )
        selected = select_direction(h0, lbfgs)
        step = np.asarray(selected["step"], dtype=np.float64)
        trial_displacement = project_normal(displacement + step, tangents)
        trial_path = reference_path.copy()
        trial_path[1:-1] = reference_path[1:-1] + trial_displacement
        if not np.array_equal(trial_path[0], reference_path[0]) or not np.array_equal(
            trial_path[-1], reference_path[-1]
        ):
            raise DiscreteB040SafetyError("fixed endpoints changed")

        # Prevent a partial band evaluation caused only by this solver-local
        # per-image limit.  The evaluator's campaign limit remains the final
        # authority and is allowed to raise before any state is committed.
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
            stop_reason = "preflight-per-image-new-call-limit"
            break

        trial_energies, trial_forces = evaluate_path(trial_path, evaluators)
        trial_gradient = project_normal(-trial_forces[1:-1], tangents)
        trial_residual = project_normal(
            trial_gradient + inverse_alpha * trial_displacement,
            tangents,
        )
        if not (
            np.all(np.isfinite(trial_energies))
            and np.all(np.isfinite(trial_forces))
            and np.all(np.isfinite(trial_residual))
        ):
            raise DiscreteB040SafetyError("evaluated trial is non-finite")

        s_value = trial_displacement - displacement
        y_value = trial_residual - residual
        status = append_curvature(
            history_s, history_y, s_value, y_value, history_size
        )
        displacement = trial_displacement
        current_path = trial_path
        current_energies = trial_energies
        current_forces = trial_forces
        gradient = trial_gradient
        residual = trial_residual
        joint_steps += 1

        direction_selection.append(selected["label"])
        h0_raw_decrease.append(h0["raw_decrease"])
        lbfgs_raw_decrease.append(lbfgs["raw_decrease"])
        h0_capped_decrease.append(h0["capped_decrease"])
        lbfgs_capped_decrease.append(lbfgs["capped_decrease"])
        h0_cap_scale.append(h0["cap_scale"])
        lbfgs_cap_scale.append(lbfgs["cap_scale"])
        history_count.append(len(history_s))
        curvature_status.append(status)
        synchronized_paths.append(current_path.copy())
        synchronized_energies.append(current_energies.copy())
        synchronized_forces.append(current_forces.copy())
        synchronized_counters.append(
            {
                name: values.copy()
                for name, values in counter_arrays(evaluators).items()
            }
        )

    final_per_image = np.linalg.norm(residual, axis=1)
    final_residual = _norm(residual) / np.sqrt(float(n_internal))
    relative_per_image = final_per_image / np.maximum(initial_per_image, 1.0e-30)
    global_relative = final_residual / initial_residual
    global_converged = bool(
        final_residual <= float(kkt_atol)
        or (
            stop_mode in RESIDUAL_STOP_MODES
            and global_relative <= relative_residual
        )
    )
    used = used_calls()
    hit_call_limit = used >= int(max_new_calls)
    per_image_target = np.maximum(
        float(kkt_atol), relative_residual * initial_per_image
    )
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
        "implementation_id": IMPLEMENTATION_ID,
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
        "frozen_path_sha256": _array_sha256(reference_path),
        "frozen_tangent_sha256": _array_sha256(tangents),
        "all_returned_states_evaluated": True,
        "endpoints_bitwise_fixed": bool(
            np.array_equal(current_path[0], reference_path[0])
            and np.array_equal(current_path[-1], reference_path[-1])
        ),
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
