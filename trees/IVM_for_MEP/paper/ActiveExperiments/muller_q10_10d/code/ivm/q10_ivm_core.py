#!/usr/bin/env python3
"""Theory-consistent fixed-alpha IVM on the fixed-Q 10D Muller case.

Every frozen-tangent proximal subproblem is represented in the complete
9-dimensional normal space.  No scalar-ray reduction, relaxation/mixing, or
adaptive-alpha controller is used.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from q10_potential import A, B, C, D, Q, X, Y, energy_values
from run_q10_baselines import (
    BudgetReached,
    DIMENSION,
    N_TOTAL_IMAGES,
    common_path,
    cubic_arclength_reparameterize,
    diagnostics,
    evaluate_path,
    improved_tangents,
    make_evaluators,
    normal_basis,
    reference_barrier,
)


N_INTERNAL = N_TOTAL_IMAGES - 2
NORMAL_DIMENSION = DIMENSION - 1
Q_SHA256 = "372962a47504e6df28a20d8b022482a6835680765434b4385a6dba0e826974bc"
REFERENCE_BARRIER = reference_barrier()


def improved_tangents_arbitrary(path, energies):
    """ASE improved-tangent formula for any path length, internal images only."""
    points = np.asarray(path, dtype=np.float64)
    values = np.asarray(energies, dtype=np.float64)
    tangents = []
    for index in range(1, points.shape[0] - 1):
        forward = points[index + 1] - points[index]
        backward = points[index] - points[index - 1]
        if values[index + 1] > values[index] > values[index - 1]:
            tangent = forward
        elif values[index + 1] < values[index] < values[index - 1]:
            tangent = backward
        else:
            delta_forward = abs(values[index + 1] - values[index])
            delta_backward = abs(values[index - 1] - values[index])
            delta_max = max(delta_forward, delta_backward)
            delta_min = min(delta_forward, delta_backward)
            if values[index + 1] > values[index - 1]:
                tangent = forward * delta_max + backward * delta_min
            else:
                tangent = forward * delta_min + backward * delta_max
        tangents.append(tangent / max(np.linalg.norm(tangent), 1.0e-15))
    return np.asarray(tangents, dtype=np.float64)

_NATURAL_LANDMARKS = np.asarray(
    [
        [0.623499399, 0.0280377512],
        [-0.558223673, 1.44172580],
        [-0.0500108, 0.4666941],
        [0.2124866, 0.2929883],
        [-0.8220016, 0.6243128],
        [-1.0, 0.2],
        [0.5, 1.2],
    ],
    dtype=float,
)


class CertificationFailure(RuntimeError):
    pass


def config_id(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def project_normal(vector, tangent):
    vector = np.asarray(vector, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    return vector - float(np.dot(vector, tangent)) * tangent


def _calls(evaluators):
    internal = evaluators[1:-1]
    values = np.asarray([item.calls for item in internal], dtype=int)
    return int(np.max(values)), float(np.mean(values)), int(np.sum(values))


def append_diagnostics(
    path,
    energies,
    gradients,
    evaluators,
    history,
    paths,
    normal_forces,
    alpha,
    outer,
    audit,
):
    row, normal = diagnostics(path, energies, gradients)
    maximum, mean, total = _calls(evaluators)
    row.update(
        {
            "alpha": float(alpha),
            "outer": int(outer),
            "calls_per_image": maximum,
            "mean_calls_per_image": mean,
            "total_force_evaluations": total,
            "reference_barrier": float(REFERENCE_BARRIER),
        }
    )
    row.update(audit)
    history.append(row)
    paths.append(np.asarray(path, dtype=float).copy())
    normal_forces.append(normal.copy())


def spring_reparameterize_improved(
    path,
    spring_constant=1.0,
    dtau=0.15,
    max_iter=20000,
    tolerance=1e-9,
):
    """Relax identical springs along the current improved NEB tangents.

    Only energies are used to select the improved tangent.  This geometry/gauge
    solve makes no force query and therefore does not change Number of Forces.
    """
    result = np.asarray(path, dtype=float).copy()
    energy_batches = 0
    residual = math.inf
    for iteration in range(int(max_iter) + 1):
        energies = energy_values(result)
        energy_batches += 1
        tangents = improved_tangents(result, energies)
        lengths = np.linalg.norm(np.diff(result, axis=0), axis=1)
        differences = lengths[1:] - lengths[:-1]
        spring_force = float(spring_constant) * differences[:, None] * tangents
        residual = float(np.max(np.linalg.norm(spring_force, axis=1)))
        if residual <= float(tolerance):
            return result, {
                "spring_residual": residual,
                "spring_iterations": iteration,
                "spring_energy_batches": energy_batches,
                "spring_converged": True,
            }
        result[1:-1] += float(dtau) * spring_force
        result[0] = path[0]
        result[-1] = path[-1]
        if not np.all(np.isfinite(result)):
            break
    return result, {
        "spring_residual": residual,
        "spring_iterations": int(max_iter),
        "spring_energy_batches": energy_batches,
        "spring_converged": residual <= float(tolerance),
    }


def _stable_energy_difference_normal(reference, basis, displacement):
    """Evaluate the normal-slice energy difference before float64 x+Bu."""
    dtype = np.longdouble
    matrix = np.asarray(Q, dtype=dtype)
    reference_z = matrix @ np.asarray(reference, dtype=dtype)
    delta_z = (
        matrix
        @ np.asarray(basis, dtype=dtype)
        @ np.asarray(displacement, dtype=dtype)
    )
    position_z = reference_z + delta_z
    aa = np.asarray(A, dtype=dtype)
    bb = np.asarray(B, dtype=dtype)
    cc = np.asarray(C, dtype=dtype)
    dd = np.asarray(D, dtype=dtype)
    xx = np.asarray(X, dtype=dtype)
    yy = np.asarray(Y, dtype=dtype)
    reference_dx = reference_z[0] - xx
    reference_dy = reference_z[1] - yy
    delta_x = delta_z[0]
    delta_y = delta_z[1]
    reference_exponent = (
        aa * reference_dx * reference_dx
        + bb * reference_dx * reference_dy
        + cc * reference_dy * reference_dy
    )
    exponent_difference = (
        aa * (dtype(2.0) * reference_dx * delta_x + delta_x * delta_x)
        + bb
        * (
            reference_dx * delta_y
            + reference_dy * delta_x
            + delta_x * delta_y
        )
        + cc * (dtype(2.0) * reference_dy * delta_y + delta_y * delta_y)
    )
    position_exponent = reference_exponent + exponent_difference
    if (
        not np.all(np.isfinite(reference_exponent))
        or not np.all(np.isfinite(position_exponent))
        or np.max(position_exponent) > dtype(1000.0)
    ):
        raise FloatingPointError("non-finite stable Muller energy difference")
    muller_difference = np.sum(
        dd
        * np.exp(reference_exponent)
        * np.expm1(exponent_difference)
    )
    harmonic_difference = dtype(2.0) * np.sum(
        delta_z[2:] * (dtype(2.0) * reference_z[2:] + delta_z[2:])
    )
    difference = muller_difference + harmonic_difference
    if not np.isfinite(difference):
        raise FloatingPointError("non-finite stable potential difference")
    return difference


def _proximal_query(evaluator, reference, basis, alpha, displacement):
    displacement = np.asarray(displacement, dtype=float)
    position = reference + basis @ displacement
    energy, gradient = evaluator.query(position)
    difference = _stable_energy_difference_normal(
        reference, basis, displacement
    )
    objective = float(
        difference
        + np.longdouble(0.5)
        * np.dot(
            np.asarray(displacement, dtype=np.longdouble),
            np.asarray(displacement, dtype=np.longdouble),
        )
        / np.longdouble(alpha)
    )
    reduced_gradient = basis.T @ gradient + displacement / float(alpha)
    return objective, reduced_gradient, energy, gradient, position


def _landmark_starts(reference, basis, alpha, initial_gradient):
    starts = [np.zeros(NORMAL_DIMENSION, dtype=float)]
    steepest = -float(alpha) * np.asarray(initial_gradient, dtype=float)
    norm = float(np.linalg.norm(steepest))
    if norm > 3.0:
        steepest *= 3.0 / norm
    starts.append(steepest)
    if float(alpha) >= 3e-2:
        for active in _NATURAL_LANDMARKS:
            natural = np.zeros(DIMENSION, dtype=float)
            natural[:2] = active
            target = natural @ Q
            starts.append(basis.T @ (target - reference))
    unique = []
    for start in starts:
        if not any(np.linalg.norm(start - old) <= 1e-10 for old in unique):
            unique.append(np.asarray(start, dtype=float))
    return unique


def solve_exact_multistart(
    evaluator,
    reference,
    tangent,
    alpha,
    kkt_rtol,
    kkt_atol,
):
    """Near-global multistart solve with an explicit 9D KKT certificate."""
    basis = normal_basis(tangent)
    zero = np.zeros(NORMAL_DIMENSION, dtype=float)
    zero_value, zero_gradient, _, _, _ = _proximal_query(
        evaluator, reference, basis, alpha, zero
    )
    initial_kkt = float(np.linalg.norm(zero_gradient))
    target = max(float(kkt_atol), float(kkt_rtol) * initial_kkt)
    best = None
    candidates = 0

    def objective(displacement):
        try:
            value, gradient, _, _, _ = _proximal_query(
                evaluator, reference, basis, alpha, displacement
            )
            return value, gradient
        except FloatingPointError:
            displacement = np.asarray(displacement, dtype=float)
            penalty = 1e100 + float(np.dot(displacement, displacement))
            return penalty, 2e50 * displacement

    for start in _landmark_starts(reference, basis, alpha, zero_gradient):
        try:
            result = minimize(
                objective,
                start,
                jac=True,
                method="L-BFGS-B",
                bounds=[(-5.0, 5.0)] * NORMAL_DIMENSION,
                options={
                    "maxiter": 1000,
                    "maxfun": 4000,
                    "maxls": 50,
                    "ftol": 1e-15,
                    "gtol": 1e-10,
                    "maxcor": 20,
                },
            )
            value, gradient, energy, physical_gradient, position = _proximal_query(
                evaluator, reference, basis, alpha, result.x
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            continue
        candidates += 1
        item = (
            float(value),
            float(np.linalg.norm(gradient)),
            np.asarray(result.x, dtype=float),
            float(energy),
            np.asarray(physical_gradient, dtype=float),
            np.asarray(position, dtype=float),
        )
        if best is None or item[0] < best[0]:
            best = item

    if best is None:
        raise CertificationFailure("exact multistart produced no finite candidate")
    value, final_kkt, displacement, energy, gradient, position = best
    objective_tolerance = 1e-24
    if value > zero_value + objective_tolerance:
        raise CertificationFailure("exact proximal objective increased")
    if final_kkt > target * (1.0 + 1e-6):
        raise CertificationFailure("exact proximal KKT certificate failed")
    return position, energy, gradient, {
        "initial_kkt": initial_kkt,
        "final_kkt": final_kkt,
        "relative_kkt": final_kkt / max(initial_kkt, 1e-300),
        "proximal_change": value - zero_value,
        "accepted_steps": math.nan,
        "new_calls": math.nan,
        "warm_pairs_used": 0,
        "multistart_candidates": candidates,
        "normal_dimension": basis.shape[1],
    }


def _harmonic_preconditioner(basis, alpha, active_curvature, enabled):
    if not enabled:
        return lambda vector: float(alpha) * np.asarray(vector, dtype=float)
    harmonic_projector = Q.T @ np.diag([0.0, 0.0] + [1.0] * 8) @ Q
    active_projector = np.eye(DIMENSION) - harmonic_projector
    model = (
        4.0 * harmonic_projector
        + float(active_curvature) * active_projector
    )
    reduced = basis.T @ model @ basis
    reduced += np.eye(NORMAL_DIMENSION) / float(alpha)
    eigenvalues, eigenvectors = np.linalg.eigh(reduced)
    eigenvalues = np.maximum(eigenvalues, 1e-12)

    def apply(vector):
        vector = np.asarray(vector, dtype=float)
        return eigenvectors @ ((eigenvectors.T @ vector) / eigenvalues)

    return apply


def _usable_warm_pairs(memory, basis, alpha, history_size):
    pairs = []
    for raw_s, raw_y_physical in memory[-int(history_size) :]:
        s = basis.T @ raw_s
        y = basis.T @ raw_y_physical + s / float(alpha)
        snorm = float(np.linalg.norm(s))
        ynorm = float(np.linalg.norm(y))
        curvature = float(np.dot(s, y))
        if (
            snorm > 1e-14
            and ynorm > 1e-14
            and np.isfinite(curvature)
            and curvature > 1e-6 * snorm * ynorm
        ):
            pairs.append((s.copy(), y.copy()))
    return pairs


def _lbfgs_direction(gradient, pairs, apply_initial_inverse):
    q = np.asarray(gradient, dtype=float).copy()
    coefficients = []
    for s, y in reversed(pairs):
        rho = 1.0 / float(np.dot(s, y))
        coefficient = rho * float(np.dot(s, q))
        coefficients.append((coefficient, rho, s, y))
        q -= coefficient * y
    result = np.asarray(apply_initial_inverse(q), dtype=float)
    for coefficient, rho, s, y in reversed(coefficients):
        beta = rho * float(np.dot(y, result))
        result += (coefficient - beta) * s
    direction = -result
    if (
        not np.all(np.isfinite(direction))
        or float(np.dot(direction, gradient)) >= 0.0
    ):
        direction = -np.asarray(apply_initial_inverse(gradient), dtype=float)
    return direction


def solve_inexact_lbfgs(
    evaluator,
    reference,
    tangent,
    alpha,
    accepted_step_cap,
    new_call_cap,
    kkt_rtol,
    kkt_atol,
    min_accepted_steps,
    history_size,
    potential_memory,
    warm_history,
    harmonic_preconditioner,
    active_curvature,
    max_step,
    armijo,
    backtrack,
    kkt_contraction_trigger,
    kkt_contraction_limit,
):
    basis = normal_basis(tangent)
    displacement = np.zeros(NORMAL_DIMENSION, dtype=float)
    value, reduced_gradient, energy, gradient, position = _proximal_query(
        evaluator, reference, basis, alpha, displacement
    )
    zero_value = value
    initial_kkt = float(np.linalg.norm(reduced_gradient))
    target = (
        -math.inf
        if kkt_rtol is None
        else max(float(kkt_atol), float(kkt_rtol) * initial_kkt)
    )
    start_calls = evaluator.calls
    apply_h0 = _harmonic_preconditioner(
        basis, alpha, active_curvature, harmonic_preconditioner
    )
    warm_pairs = (
        _usable_warm_pairs(potential_memory, basis, alpha, history_size)
        if warm_history and history_size > 0
        else []
    )
    local_pairs = []
    accepted = 0
    backtracks = 0
    restarts = 0
    kkt_contraction_steps = 0

    while accepted < int(accepted_step_cap):
        residual = float(np.linalg.norm(reduced_gradient))
        if (
            kkt_rtol is not None
            and residual <= target
            and accepted >= int(min_accepted_steps)
        ):
            break
        if evaluator.calls - start_calls >= int(new_call_cap):
            break
        pairs = (warm_pairs + local_pairs)[-int(history_size) :] if history_size else []
        direction = _lbfgs_direction(reduced_gradient, pairs, apply_h0)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > float(max_step):
            direction *= float(max_step) / direction_norm
        directional_derivative = float(np.dot(reduced_gradient, direction))
        if directional_derivative >= 0.0:
            warm_pairs = []
            local_pairs = []
            direction = -np.asarray(apply_h0(reduced_gradient), dtype=float)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > float(max_step):
                direction *= float(max_step) / direction_norm
            directional_derivative = float(np.dot(reduced_gradient, direction))
            restarts += 1
        if directional_derivative >= 0.0:
            break

        accepted_trial = False
        step_length = 1.0
        for _ in range(64):
            if evaluator.calls - start_calls >= int(new_call_cap):
                break
            trial_displacement = displacement + step_length * direction
            trial = _proximal_query(
                evaluator, reference, basis, alpha, trial_displacement
            )
            trial_value, trial_reduced, trial_energy, trial_gradient, trial_position = trial
            armijo_ok = trial_value <= (
                value
                + float(armijo) * step_length * directional_derivative
                + 1e-30
            )
            contraction_mode = (
                float(kkt_contraction_trigger) > 0.0
                and initial_kkt <= float(kkt_contraction_trigger)
            )
            contraction_ok = (
                contraction_mode
                and np.linalg.norm(trial_reduced)
                <= float(kkt_contraction_limit)
                * np.linalg.norm(reduced_gradient)
                and trial_value <= zero_value + 1e-24
            )
            if armijo_ok or contraction_ok:
                old_displacement = displacement.copy()
                old_reduced = reduced_gradient.copy()
                old_position = position.copy()
                old_gradient = gradient.copy()
                displacement = trial_displacement
                value = float(trial_value)
                reduced_gradient = np.asarray(trial_reduced, dtype=float)
                energy = float(trial_energy)
                gradient = np.asarray(trial_gradient, dtype=float)
                position = np.asarray(trial_position, dtype=float)
                s = displacement - old_displacement
                y = reduced_gradient - old_reduced
                curvature = float(np.dot(s, y))
                if curvature > 1e-6 * np.linalg.norm(s) * np.linalg.norm(y):
                    local_pairs.append((s.copy(), y.copy()))
                    if len(local_pairs) > int(history_size):
                        local_pairs.pop(0)
                physical_s = position - old_position
                physical_y = gradient - old_gradient
                physical_curvature = float(np.dot(physical_s, physical_y))
                if (
                    warm_history
                    and physical_curvature
                    > -0.99 * np.dot(physical_s, physical_s) / float(alpha)
                ):
                    potential_memory.append(
                        (physical_s.copy(), physical_y.copy())
                    )
                    if len(potential_memory) > int(history_size):
                        potential_memory.pop(0)
                accepted += 1
                if contraction_ok and not armijo_ok:
                    kkt_contraction_steps += 1
                accepted_trial = True
                break
            step_length *= float(backtrack)
            backtracks += 1
            if step_length < 2.0**-40:
                break
        if not accepted_trial:
            break

    final_kkt = float(np.linalg.norm(reduced_gradient))
    objective_tolerance = 1e-24
    if value > zero_value + objective_tolerance:
        raise CertificationFailure("inexact proximal objective increased")
    return position, energy, gradient, {
        "initial_kkt": initial_kkt,
        "final_kkt": final_kkt,
        "relative_kkt": final_kkt / max(initial_kkt, 1e-300),
        "proximal_change": value - zero_value,
        "accepted_steps": accepted,
        "new_calls": evaluator.calls - start_calls,
        "warm_pairs_used": len(warm_pairs),
        "multistart_candidates": 0,
        "normal_dimension": basis.shape[1],
        "line_search_backtracks": backtracks,
        "curvature_restarts": restarts,
        "kkt_contraction_steps": kkt_contraction_steps,
    }


def _aggregate_audits(audits, spring_audit, resampled):
    def finite_max(key, default=math.nan):
        values = np.asarray([item.get(key, math.nan) for item in audits], dtype=float)
        values = values[np.isfinite(values)]
        return float(np.max(values)) if len(values) else float(default)

    return {
        "max_inner_kkt_ratio": finite_max("relative_kkt"),
        "max_inner_kkt_residual": finite_max("final_kkt"),
        "max_inner_proximal_change": finite_max("proximal_change"),
        "max_inner_accepted_steps": finite_max("accepted_steps", 0.0),
        "max_inner_new_calls": finite_max("new_calls", 0.0),
        "max_warm_pairs_used": finite_max("warm_pairs_used", 0.0),
        "min_normal_dimension": float(
            min(item["normal_dimension"] for item in audits)
        ),
        "max_normal_dimension": float(
            max(item["normal_dimension"] for item in audits)
        ),
        "resampled": bool(resampled),
        "spring_residual_after": float(
            spring_audit.get("spring_residual", math.nan)
        ),
        "spring_iterations": int(spring_audit.get("spring_iterations", 0)),
        "spring_energy_batches": int(
            spring_audit.get("spring_energy_batches", 0)
        ),
        "spring_converged": bool(spring_audit.get("spring_converged", False)),
    }


def run(config):
    alpha = float(config["alpha"])
    evaluators = make_evaluators(int(config["budget_per_image"]))
    path = common_path()
    history = []
    paths = []
    normal_forces = []
    potential_memories = [[] for _ in range(N_INTERNAL)]
    energies, gradients = evaluate_path(path, evaluators)
    initial_audit = {
        "max_inner_kkt_ratio": math.nan,
        "max_inner_kkt_residual": math.nan,
        "max_inner_proximal_change": 0.0,
        "max_inner_accepted_steps": 0.0,
        "max_inner_new_calls": 0.0,
        "max_warm_pairs_used": 0.0,
        "min_normal_dimension": NORMAL_DIMENSION,
        "max_normal_dimension": NORMAL_DIMENSION,
        "resampled": False,
        "spring_residual_after": math.nan,
        "spring_iterations": 0,
        "spring_energy_batches": 0,
        "spring_converged": False,
    }
    append_diagnostics(
        path,
        energies,
        gradients,
        evaluators,
        history,
        paths,
        normal_forces,
        alpha,
        0,
        initial_audit,
    )
    stop_reason = "outer_limit"
    failure_message = ""

    for outer in range(1, int(config["outer_limit"]) + 1):
        try:
            energies, gradients = evaluate_path(path, evaluators)
            tangents = improved_tangents(path, energies)
            late_trigger = float(config.get("late_iperp_trigger", -1.0))
            late_active = (
                config["solver"] != "exact_multistart"
                and late_trigger > 0.0
                and float(history[-1]["i_perp"]) <= late_trigger
            )
            accepted_step_cap = int(
                config.get("late_accepted_step_cap", config.get("accepted_step_cap", 0))
                if late_active
                else config.get("accepted_step_cap", 0)
            )
            new_call_cap = int(
                config.get("late_new_call_cap", config.get("new_call_cap", 0))
                if late_active
                else config.get("new_call_cap", 0)
            )
            kkt_rtol = (
                config.get("late_kkt_rtol", config.get("kkt_rtol"))
                if late_active
                else config.get("kkt_rtol")
            )
            min_accepted_steps = int(
                config.get("late_min_accepted_steps", config.get("min_accepted_steps", 0))
                if late_active
                else config.get("min_accepted_steps", 0)
            )
            history_size = int(
                config.get("late_history_size", config.get("history_size", 0))
                if late_active
                else config.get("history_size", 0)
            )
            max_step = float(config.get("max_step", 3.0))
            if outer == 1 and "first_max_step" in config:
                max_step = float(config["first_max_step"])
            elif outer <= int(config.get("initial_max_step_outer_limit", 0)):
                max_step = float(config.get("initial_max_step", max_step))
            proposed = path.copy()
            proposed_energies = energies.copy()
            proposed_gradients = gradients.copy()
            audits = []
            for image_index in range(1, N_TOTAL_IMAGES - 1):
                if config["solver"] == "exact_multistart":
                    solved = solve_exact_multistart(
                        evaluators[image_index],
                        path[image_index],
                        tangents[image_index - 1],
                        alpha,
                        config["kkt_rtol"],
                        config["kkt_atol"],
                    )
                else:
                    solved = solve_inexact_lbfgs(
                        evaluators[image_index],
                        path[image_index],
                        tangents[image_index - 1],
                        alpha,
                        accepted_step_cap,
                        new_call_cap,
                        kkt_rtol,
                        config.get("kkt_atol", 1e-3),
                        min_accepted_steps,
                        history_size,
                        potential_memories[image_index - 1],
                        config.get("warm_history", False),
                        config.get("harmonic_preconditioner", False),
                        config.get("active_curvature", 0.0),
                        max_step,
                        config.get("armijo", 1e-4),
                        config.get("backtrack", 0.5),
                        config.get("kkt_contraction_trigger", -1.0),
                        config.get("kkt_contraction_limit", 0.9),
                    )
                position, energy, gradient, audit = solved
                proposed[image_index] = position
                proposed_energies[image_index] = energy
                proposed_gradients[image_index] = gradient
                audits.append(audit)

            if not np.all(np.isfinite(proposed)):
                raise FloatingPointError("non-finite proposed IVM band")
            resample_interval = int(config.get("resample_interval", 1))
            resample_start_outer = int(
                config.get("resample_start_outer", resample_interval)
            )
            resample_scheduled = (
                bool(config.get("resample", False))
                and not (late_active and bool(config.get("late_disable_resample", False)))
                and outer >= resample_start_outer
                and (outer - resample_start_outer) % resample_interval == 0
            )
            proposed_lengths = np.linalg.norm(np.diff(proposed, axis=0), axis=1)
            proposed_spacing_residual = float(
                np.max(np.abs(proposed_lengths[1:] - proposed_lengths[:-1]))
            )
            spacing_trigger = float(
                config.get("resample_spacing_trigger", -1.0)
            )
            resampled = resample_scheduled and (
                spacing_trigger <= 0.0
                or proposed_spacing_residual > spacing_trigger
            )
            spring_audit = {}
            if resampled:
                if config.get("resample_mode", "cubic") == "spring":
                    updated, spring_audit = spring_reparameterize_improved(
                        proposed,
                        spring_constant=config.get("spring_constant", 1.0),
                        dtau=config.get("spring_dtau", 0.15),
                        max_iter=config.get("spring_max_iter", 20000),
                        tolerance=config.get("spring_tolerance", 1e-9),
                    )
                    if not spring_audit["spring_converged"]:
                        raise CertificationFailure(
                            "improved-tangent spring reparameterization did not converge"
                        )
                else:
                    updated = proposed.copy()
                    for _ in range(int(config.get("resample_passes", 3))):
                        updated = cubic_arclength_reparameterize(updated)
                    lengths = np.linalg.norm(np.diff(updated, axis=0), axis=1)
                    residual = float(np.max(np.abs(lengths[1:] - lengths[:-1])))
                    spring_audit = {
                        "spring_residual": residual,
                        "spring_iterations": 0,
                        "spring_energy_batches": 0,
                        "spring_converged": True,
                    }
                if config.get("reset_history_after_resample", True):
                    potential_memories = [[] for _ in range(N_INTERNAL)]
                energies, gradients = evaluate_path(updated, evaluators)
            else:
                updated = proposed
                energies = proposed_energies
                gradients = proposed_gradients
            movement = float(np.max(np.linalg.norm(updated - path, axis=1)))
            path = updated
            append_diagnostics(
                path,
                energies,
                gradients,
                evaluators,
                history,
                paths,
                normal_forces,
                alpha,
                outer,
                _aggregate_audits(audits, spring_audit, resampled),
            )
            if movement <= float(config.get("movement_tolerance", 1e-14)):
                stop_reason = "stationary"
                break
        except BudgetReached:
            stop_reason = "budget"
            break
        except CertificationFailure as exc:
            stop_reason = "certification_failure"
            failure_message = str(exc)
            break
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            stop_reason = "numerical_failure"
            break

    return {
        "config": config,
        "config_id": config_id(config),
        "stop_reason": stop_reason,
        "failure_message": failure_message,
        "history": history,
        "paths": paths,
        "normal_forces": normal_forces,
        "final_path": path,
        "alpha_fixed_audit": all(row["alpha"] == alpha for row in history),
        "full_normal_dimension_audit": all(
            row["min_normal_dimension"] == NORMAL_DIMENSION
            and row["max_normal_dimension"] == NORMAL_DIMENSION
            for row in history
        ),
    }


def save_run(root, config, role):
    root = Path(root)
    result = run(config)
    output_dir = root / "runs" / role
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "{}.npz".format(result["config_id"])
    history = result["history"]
    keys = list(history[0])
    np.savez_compressed(
        output,
        case=np.array("fixed-Q unperturbed 10D Muller-Brown"),
        q_sha256=np.array(Q_SHA256),
        q_matrix=Q,
        coordinate_convention=np.array("z=Q@x; x=Q.T@z"),
        tangent_definition=np.array("ASE Henkelman-Jonsson improved tangent"),
        proximal_subproblem=np.array(
            "full 9D frozen-normal U(x_ref+B u)+||u||^2/(2 alpha)"
        ),
        config_json=np.array(json.dumps(config, sort_keys=True)),
        config_id=np.array(result["config_id"]),
        stop_reason=np.array(result["stop_reason"]),
        failure_message=np.array(result["failure_message"]),
        alpha_fixed_audit=np.array(result["alpha_fixed_audit"]),
        full_normal_dimension_audit=np.array(
            result["full_normal_dimension_audit"]
        ),
        transformed_initial_path=common_path(),
        final_path=result["final_path"],
        path_history=np.asarray(result["paths"]),
        normal_force_history=np.asarray(result["normal_forces"]),
        **{key: np.asarray([row[key] for row in history]) for key in keys},
    )
    final = history[-1]
    return {
        "case": "fixed-Q unperturbed 10D Muller-Brown",
        "q_sha256": Q_SHA256,
        "config": config,
        "config_id": result["config_id"],
        "output": str(output.relative_to(root)),
        "stop_reason": result["stop_reason"],
        "failure_message": result["failure_message"],
        "alpha_fixed_audit": result["alpha_fixed_audit"],
        "full_normal_dimension_audit": result["full_normal_dimension_audit"],
        "history_length": len(history),
        "final_i_perp": float(final["i_perp"]),
        "final_delta_e": float(final["delta_e"]),
        "final_barrier_error": float(
            abs(final["delta_e"] - REFERENCE_BARRIER)
        ),
        "final_total_force_evaluations": int(
            final["total_force_evaluations"]
        ),
        "final_calls_per_image": int(final["calls_per_image"]),
        "final_segment_ratio": float(final["segment_ratio"]),
        "final_spring_residual": float(final["spring_residual"]),
        "max_recorded_inner_proximal_change": float(
            np.nanmax([row["max_inner_proximal_change"] for row in history])
        ),
    }
