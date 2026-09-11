#!/usr/bin/env python3
"""Run the audited LEPS IVM and its alpha/reparameterization ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from leps_core import (
    N_INTERNAL,
    TANGENT_DEFINITION,
    EvaluationBudgetReached,
    LEPSBackend,
    array_sha256,
    atomic_save_npz,
    audit_counter_consistency,
    clip_max_atomic_displacement,
    counter_arrays,
    densified_barrier,
    evaluate_path,
    improved_hermite_equal_chord,
    improved_tangents,
    load_common_path,
    make_evaluators,
    path_diagnostics,
    project_normal,
    project_relative,
    reference_saddle,
    rms_segment_trust_radius,
    write_json,
)


def projected_lbfgs_direction(
    gradient,
    tangent,
    pairs,
    initial_inverse_scale,
):
    """Intrinsic L-BFGS direction in one frozen improved-tangent normal space."""
    gradient = project_normal(
        np.asarray(gradient)[None, :], np.asarray(tangent)[None, :]
    )[0]
    usable = []
    for raw_s, raw_y in pairs:
        s = project_normal(raw_s[None, :], tangent[None, :])[0]
        y = project_normal(raw_y[None, :], tangent[None, :])[0]
        snorm = np.linalg.norm(s)
        ynorm = np.linalg.norm(y)
        ys = float(np.dot(y, s))
        if (
            snorm > 1e-14
            and ynorm > 1e-14
            and np.isfinite(ys)
            and ys > 1e-6 * snorm * ynorm
        ):
            usable.append((s, y))
    q = gradient.copy()
    coefficients = []
    for s, y in reversed(usable):
        rho = 1.0 / float(np.dot(y, s))
        coefficient = rho * float(np.dot(s, q))
        coefficients.append((coefficient, rho, s, y))
        q -= coefficient * y
    if usable:
        last_s, last_y = usable[-1]
        gamma = float(np.dot(last_s, last_y) / np.dot(last_y, last_y))
    else:
        gamma = float(initial_inverse_scale)
    result = max(gamma, 1e-12) * q
    for coefficient, rho, s, y in reversed(coefficients):
        beta = rho * float(np.dot(y, result))
        result += (coefficient - beta) * s
    direction = project_normal((-result)[None, :], tangent[None, :])[0]
    if (
        not np.all(np.isfinite(direction))
        or float(np.dot(direction, gradient)) >= -1e-18
    ):
        direction = project_normal((-gradient)[None, :], tangent[None, :])[0]
    return direction


def proximal_quality(
    reference_path,
    reference_energies,
    reference_forces,
    candidate_path,
    candidate_energies,
    frozen_tangents,
    alpha,
    trust_radius,
):
    displacement = project_normal(
        candidate_path[1:-1] - reference_path[1:-1], frozen_tangents
    )
    gradient = project_normal(-reference_forces[1:-1], frozen_tangents)
    quadratic = 0.5 * np.sum(displacement**2, axis=1) / float(alpha)
    predicted = -np.sum(gradient * displacement, axis=1) - quadratic
    actual = reference_energies[1:-1] - candidate_energies[1:-1] - quadratic
    moved = np.linalg.norm(displacement, axis=1) > 1e-14
    valid = moved & np.isfinite(predicted) & np.isfinite(actual) & (predicted > 1e-14)
    ratios = actual[valid] / predicted[valid]
    ratios = ratios[np.isfinite(ratios)]
    quality = float(np.median(ratios)) if ratios.size else np.nan
    atom_steps = np.linalg.norm(
        displacement.reshape(displacement.shape[0], -1, 3), axis=2
    )
    utilization = (
        float(np.max(atom_steps)) / max(float(trust_radius), 1e-30)
        if atom_steps.size
        else 0.0
    )
    return quality, utilization, int(np.count_nonzero(moved))


def adapt_alpha(
    alpha,
    quality,
    utilization,
    moved_count,
    alpha_min,
    alpha_max,
):
    accepted_fraction = float(moved_count) / float(N_INTERNAL)
    value = float(alpha)
    action = "hold"
    if (
        not np.isfinite(quality)
        or quality < 0.25
        or accepted_fraction < 0.5
    ):
        value *= 0.5
        action = "shrink"
    elif quality > 0.75 and utilization > 0.8 and accepted_fraction >= 0.8:
        value *= 1.5
        action = "grow"
    clipped = float(np.clip(value, alpha_min, alpha_max))
    if clipped == float(alpha):
        action = "hold"
    return clipped, action


class Trace:
    def __init__(self, evaluators, saddle_barrier):
        self.evaluators = evaluators
        self.saddle_barrier = float(saddle_barrier)
        self.rows = []

    def record(
        self,
        path,
        energies,
        forces,
        alpha,
        accepted,
        alpha_action,
        kkt_max,
        kkt_relative_max,
        trust_radius,
        spring_residual,
        subproblem_kkt_fraction=np.nan,
        subproblem_kkt_all=False,
    ):
        counters = counter_arrays(self.evaluators)
        internal = counters["joint_evaluations"][1:-1]
        call = int(np.max(internal))
        diagnostic = path_diagnostics(path, energies, forces[1:-1])
        dense, _ = densified_barrier(path, subdivisions=80)
        row = {
            "call": call,
            "path": np.asarray(path, dtype=float).copy(),
            "energies": np.asarray(energies, dtype=float).copy(),
            "forces": np.asarray(forces, dtype=float).copy(),
            "i_perp": diagnostic["i_perp"],
            "fmax": diagnostic["fmax"],
            "barrier": diagnostic["barrier"],
            "dense_barrier": dense,
            "barrier_error": abs(dense - self.saddle_barrier),
            "i_arc": diagnostic["i_arc"],
            "equidistance_integral": diagnostic["equidistance_integral"],
            "spring_force_max_unit": diagnostic["spring_force_max_unit"],
            "equidistance_relative_defect": diagnostic[
                "equidistance_relative_defect"
            ],
            "segment_length_cv": diagnostic["segment_length_cv"],
            "segment_length_ratio": diagnostic["segment_length_ratio"],
            "alpha": float(alpha),
            "accepted": bool(accepted),
            "alpha_action": str(alpha_action),
            "kkt_max": float(kkt_max),
            "kkt_relative_max": float(kkt_relative_max),
            "trust_radius": float(trust_radius),
            "spring_residual": float(spring_residual),
            "subproblem_kkt_fraction": float(subproblem_kkt_fraction),
            "subproblem_kkt_all": bool(subproblem_kkt_all),
            "counter_vector": counters["joint_evaluations"].copy(),
            "energy_request_vector": counters["energy_requests"].copy(),
            "force_request_vector": counters["force_requests"].copy(),
            "cache_hit_vector": counters["cache_hits"].copy(),
            "mean_calls": float(np.mean(internal)),
            "total_internal_calls": int(np.sum(internal)),
        }
        if self.rows and self.rows[-1]["call"] == call:
            self.rows[-1] = row
        else:
            self.rows.append(row)

    def arrays(self):
        keys = [key for key in self.rows[0] if key != "call"]
        payload = {"calls_per_image": np.asarray([r["call"] for r in self.rows])}
        for key in keys:
            payload[key] = np.asarray([r[key] for r in self.rows])
        return payload


def solve_frozen_band(
    path,
    energies,
    forces,
    tangents,
    evaluators,
    trace,
    alpha,
    trust_radius,
    inner_calls,
    kkt_rtol,
    kkt_atol,
    history_size,
    integrated_reparameterize,
    budget,
):
    reference_path = path.copy()
    reference_energies = energies.copy()
    current_path = path.copy()
    current_energies = energies.copy()
    current_forces = forces.copy()
    current_diag = path_diagnostics(path, energies, forces[1:-1])
    displacement = np.zeros_like(path[1:-1])
    residual = project_normal(-forces[1:-1], tangents)
    objective = energies[1:-1].copy()
    initial_kkt = np.linalg.norm(residual, axis=1)
    target = np.maximum(float(kkt_atol), float(kkt_rtol) * initial_kkt)
    local_pairs = [[] for _ in range(N_INTERNAL)]
    start_counts = counter_arrays(evaluators)["joint_evaluations"][1:-1].copy()
    accepted_any = False
    spring_residual = 0.0

    def used_calls():
        return (
            counter_arrays(evaluators)["joint_evaluations"][1:-1]
            - start_counts
        )

    while True:
        residual_norm = np.linalg.norm(residual, axis=1)
        if np.all(residual_norm <= target):
            reason = "kkt"
            break
        if int(np.max(used_calls())) >= int(inner_calls):
            reason = "inner_call_limit"
            break
        if (
            int(
                np.max(
                    counter_arrays(evaluators)["joint_evaluations"][1:-1]
                )
            )
            >= int(budget)
        ):
            reason = "global_budget"
            break

        directions = np.empty_like(residual)
        for i in range(N_INTERNAL):
            direction = projected_lbfgs_direction(
                residual[i], tangents[i], local_pairs[i], alpha
            )
            direction = clip_max_atomic_displacement(direction, trust_radius)
            directions[i] = project_normal(
                direction[None, :], tangents[i][None, :]
            )[0]

        accepted = False
        scale = 1.0
        for _ in range(64):
            proposed = project_normal(
                displacement + scale * directions, tangents
            )
            raw_trial = reference_path.copy()
            raw_trial[1:-1] += proposed
            if integrated_reparameterize:
                trial_path, spring_residual = improved_hermite_equal_chord(
                    raw_trial, tangents
                )
            else:
                trial_path = raw_trial
                spring_residual = 0.0

            current_counts = counter_arrays(evaluators)[
                "joint_evaluations"
            ][1:-1]
            blocked = [
                current_counts[i] >= int(budget)
                and not evaluators[i + 1].has_cached(trial_path[i + 1])
                for i in range(N_INTERNAL)
            ]
            if any(blocked):
                reason = "global_budget"
                break

            try:
                trial_energies, trial_forces = evaluate_path(
                    trial_path, evaluators
                )
            except EvaluationBudgetReached:
                reason = "global_budget"
                break
            trial_displacement = project_normal(
                trial_path[1:-1] - reference_path[1:-1], tangents
            )
            trial_gradient = project_normal(-trial_forces[1:-1], tangents)
            trial_residual = trial_gradient + trial_displacement / float(alpha)
            trial_objective = (
                trial_energies[1:-1]
                + 0.5
                * np.sum(trial_displacement**2, axis=1)
                / float(alpha)
            )
            trial_diag = path_diagnostics(
                trial_path, trial_energies, trial_forces[1:-1]
            )
            objective_ok = np.all(
                trial_objective
                <= objective + 1e-13 * np.maximum(1.0, np.abs(objective))
            )
            global_ok = (
                trial_diag["i_perp"] <= current_diag["i_perp"] + 1e-13
            )
            accepted = bool(objective_ok and global_ok)
            if accepted:
                for i in range(N_INTERNAL):
                    s = trial_displacement[i] - displacement[i]
                    y = trial_residual[i] - residual[i]
                    sy = float(np.dot(s, y))
                    if (
                        np.linalg.norm(s) > 1e-14
                        and np.linalg.norm(y) > 1e-14
                        and sy
                        > 1e-6 * np.linalg.norm(s) * np.linalg.norm(y)
                    ):
                        local_pairs[i].append((s.copy(), y.copy()))
                        if len(local_pairs[i]) > int(history_size):
                            local_pairs[i].pop(0)
                current_path = trial_path
                current_energies = trial_energies
                current_forces = trial_forces
                current_diag = trial_diag
                displacement = trial_displacement
                residual = trial_residual
                objective = trial_objective
                accepted_any = True

            residual_norm_for_trace = np.linalg.norm(residual, axis=1)
            trace.record(
                current_path,
                current_energies,
                current_forces,
                alpha=alpha,
                accepted=accepted,
                alpha_action="hold",
                kkt_max=np.max(residual_norm_for_trace),
                kkt_relative_max=np.max(
                    residual_norm_for_trace / np.maximum(initial_kkt, 1e-30)
                ),
                trust_radius=trust_radius,
                spring_residual=spring_residual,
            )
            if accepted:
                break
            scale *= 0.5
            if scale < 2.0**-40:
                reason = "line_search"
                break
        if not accepted:
            if "reason" not in locals():
                reason = "line_search"
            break

    final_norm = np.linalg.norm(residual, axis=1)
    return {
        "path": current_path,
        "energies": current_energies,
        "forces": current_forces,
        "accepted_any": accepted_any,
        "reason": reason,
        "initial_kkt": initial_kkt,
        "final_kkt": final_norm,
        "relative_kkt": final_norm / np.maximum(initial_kkt, 1e-30),
        "used_calls": used_calls(),
        "spring_residual": spring_residual,
        "reference_path": reference_path,
        "reference_energies": reference_energies,
    }


def projected_scalar_kkt(z, gradient, lower, upper):
    """Projected-gradient KKT residual for a one-dimensional box."""
    projected = float(np.clip(float(z) - float(gradient), lower, upper))
    return abs(float(z) - projected)


def solve_independent_band(
    path,
    energies,
    forces,
    tangents,
    evaluators,
    trace,
    alpha,
    trust_radius,
    inner_calls,
    kkt_rtol,
    kkt_atol,
    budget,
    near_exact=False,
):
    """Solve the 20 frozen-normal LEPS subproblems independently.

    In the two physical LEPS coordinates each normal slice is one-dimensional.
    The scalar inverse-Hessian estimate is initialized by alpha and then updated
    by the exact one-dimensional BFGS secant.  Every trial uses an Armijo test
    on the genuine proximal objective.
    """
    reference_path = path.copy()
    reference_energies = energies.copy()
    reference_forces = forces.copy()
    reference_diag = path_diagnostics(path, energies, forces[1:-1])
    candidate_path = path.copy()
    candidate_energies = energies.copy()
    candidate_forces = forces.copy()
    initial_kkt = np.zeros(N_INTERNAL, dtype=float)
    final_kkt = np.zeros(N_INTERNAL, dtype=float)
    used_calls = np.zeros(N_INTERNAL, dtype=int)
    accepted_images = np.zeros(N_INTERNAL, dtype=bool)

    for internal_index in range(N_INTERNAL):
        image_index = internal_index + 1
        tangent = tangents[internal_index]
        # The potential depends only on flat coordinates 0 and 3.  ASE's
        # improved tangent has support in the same plane, so this is the unique
        # energy-relevant unit normal.
        physical_tangent = np.array([tangent[0], tangent[3]], dtype=float)
        tangent_norm = float(np.linalg.norm(physical_tangent))
        if tangent_norm <= 1e-30:
            raise RuntimeError("Improved tangent has no LEPS-plane component")
        physical_tangent /= tangent_norm
        normal = np.zeros_like(tangent)
        normal[0] = -physical_tangent[1]
        normal[3] = physical_tangent[0]
        normal /= np.linalg.norm(normal)
        z_bound = float(trust_radius) / max(abs(normal[0]), abs(normal[3]), 1e-30)

        reference = path[image_index].copy()
        energy = float(energies[image_index])
        force = forces[image_index].copy()
        z = 0.0
        gradient = -float(np.dot(force, normal)) + z / float(alpha)
        lower_bound = -z_bound
        upper_bound = z_bound
        kkt_residual = projected_scalar_kkt(
            z, gradient, lower_bound, upper_bound
        )
        initial_kkt[internal_index] = kkt_residual
        target = max(float(kkt_atol), float(kkt_rtol) * kkt_residual)
        objective = energy
        inverse_hessian = float(alpha)
        start_calls = evaluators[image_index].counters.joint_evaluations

        if near_exact:
            # Use the mature box-constrained L-BFGS-B line search for the
            # deliberately near-exact ablation.  The objective remains exactly
            # the same frozen-normal IVM subproblem; only its inner stopping
            # accuracy differs from the inexpensive production solver.
            state = {
                "z": z,
                "energy": energy,
                "force": force.copy(),
                "gradient": gradient,
                "objective": objective,
            }
            best = dict(state)

            def objective_and_gradient(raw_z):
                trial_z = float(raw_z[0])
                trial_position = reference + trial_z * normal
                evaluator = evaluators[image_index]
                if (
                    evaluator.counters.joint_evaluations >= int(budget)
                    and not evaluator.has_cached(trial_position)
                ):
                    raise EvaluationBudgetReached(
                        "near-exact IVM evaluation budget reached"
                    )
                trial_energy, trial_force = evaluator.query(
                    trial_position, "both"
                )
                trial_gradient = (
                    -float(np.dot(trial_force, normal))
                    + trial_z / float(alpha)
                )
                trial_objective = float(
                    trial_energy + 0.5 * trial_z**2 / float(alpha)
                )
                state.update(
                    z=trial_z,
                    energy=float(trial_energy),
                    force=np.asarray(trial_force, dtype=float).copy(),
                    gradient=float(trial_gradient),
                    objective=trial_objective,
                )
                if trial_objective < best["objective"]:
                    best.update(state)
                return trial_objective, np.asarray([trial_gradient])

            optimization_result = None
            try:
                optimization_result = minimize(
                    objective_and_gradient,
                    np.asarray([0.0]),
                    method="L-BFGS-B",
                    jac=True,
                    bounds=((lower_bound, upper_bound),),
                    options={
                        "gtol": target,
                        "ftol": 0.0,
                        "maxiter": int(inner_calls),
                        "maxfun": int(inner_calls),
                        "maxls": 64,
                    },
                )
            except EvaluationBudgetReached:
                pass
            chosen = best
            if optimization_result is not None:
                try:
                    objective_and_gradient(optimization_result.x)
                    if (
                        state["objective"]
                        <= reference_energies[image_index]
                        + 1e-13
                        * max(1.0, abs(reference_energies[image_index]))
                    ):
                        chosen = state
                except EvaluationBudgetReached:
                    pass
            z = float(chosen["z"])
            energy = float(chosen["energy"])
            force = np.asarray(chosen["force"], dtype=float).copy()
            gradient = float(chosen["gradient"])
            objective = float(chosen["objective"])
            kkt_residual = projected_scalar_kkt(
                z, gradient, lower_bound, upper_bound
            )
            accepted_images[internal_index] = bool(
                abs(z) > 1e-18
                and objective
                <= reference_energies[image_index]
                + 1e-13
                * max(1.0, abs(reference_energies[image_index]))
            )
            final_kkt[internal_index] = kkt_residual
            candidate_path[image_index] = reference + z * normal
            candidate_energies[image_index] = energy
            candidate_forces[image_index] = force
            used_calls[internal_index] = (
                evaluators[image_index].counters.joint_evaluations
                - start_calls
            )
            continue

        while kkt_residual > target:
            used = (
                evaluators[image_index].counters.joint_evaluations - start_calls
            )
            if used >= int(inner_calls):
                break
            direction = -inverse_hessian * gradient
            if not np.isfinite(direction) or gradient * direction >= -1e-18:
                direction = -gradient
                inverse_hessian = float(alpha)
            direction = float(
                np.clip(direction, -z_bound - z, z_bound - z)
            )
            if abs(direction) <= 1e-18:
                break

            accepted = False
            step_length = 1.0
            for _ in range(64):
                trial_z = float(
                    np.clip(z + step_length * direction, -z_bound, z_bound)
                )
                trial_position = reference + trial_z * normal
                evaluator = evaluators[image_index]
                if (
                    evaluator.counters.joint_evaluations >= int(budget)
                    and not evaluator.has_cached(trial_position)
                ):
                    break
                before = evaluator.counters.joint_evaluations
                try:
                    trial_energy, trial_force = evaluator.query(
                        trial_position, "both"
                    )
                except EvaluationBudgetReached:
                    break
                trial_gradient = (
                    -float(np.dot(trial_force, normal))
                    + trial_z / float(alpha)
                )
                trial_objective = float(
                    trial_energy + 0.5 * trial_z**2 / float(alpha)
                )
                trace.record(
                    reference_path,
                    reference_energies,
                    reference_forces,
                    alpha=alpha,
                    accepted=False,
                    alpha_action="hold",
                    kkt_max=float(np.max(initial_kkt)),
                    kkt_relative_max=1.0,
                    trust_radius=trust_radius,
                    spring_residual=0.0,
                )
                armijo = trial_objective <= (
                    objective
                    + 1e-4 * step_length * gradient * direction
                    + 1e-13 * max(1.0, abs(objective))
                )
                if armijo:
                    s = trial_z - z
                    y = trial_gradient - gradient
                    if s * y > 1e-12 * abs(s) * abs(y):
                        inverse_hessian = float(
                            np.clip(s / y, 1e-8, 1e8)
                        )
                    else:
                        inverse_hessian = float(alpha)
                    z = trial_z
                    energy = float(trial_energy)
                    force = np.asarray(trial_force, dtype=float).copy()
                    gradient = float(trial_gradient)
                    kkt_residual = projected_scalar_kkt(
                        z, gradient, lower_bound, upper_bound
                    )
                    objective = float(trial_objective)
                    accepted = True
                    accepted_images[internal_index] = True
                    break
                if evaluator.counters.joint_evaluations == before:
                    # A cached rejected coordinate would repeat forever.
                    step_length *= 0.5
                else:
                    step_length *= 0.5
                if step_length < 2.0**-40:
                    break
            if not accepted:
                break

        final_kkt[internal_index] = kkt_residual
        candidate_path[image_index] = reference + z * normal
        candidate_energies[image_index] = energy
        candidate_forces[image_index] = force
        used_calls[internal_index] = (
            evaluators[image_index].counters.joint_evaluations - start_calls
        )

    candidate_diag = path_diagnostics(
        candidate_path, candidate_energies, candidate_forces[1:-1]
    )
    accepted_band = bool(
        np.any(accepted_images)
        and candidate_diag["i_perp"] <= reference_diag["i_perp"] + 1e-13
    )
    if not accepted_band and np.any(accepted_images):
        scale = 0.5
        for _ in range(40):
            trial_path = reference_path + scale * (
                candidate_path - reference_path
            )
            current_counts = counter_arrays(evaluators)[
                "joint_evaluations"
            ][1:-1]
            blocked = [
                current_counts[i] >= int(budget)
                and not evaluators[i + 1].has_cached(trial_path[i + 1])
                for i in range(N_INTERNAL)
            ]
            if any(blocked):
                break
            try:
                trial_energies, trial_forces = evaluate_path(
                    trial_path, evaluators
                )
            except EvaluationBudgetReached:
                break
            trial_diag = path_diagnostics(
                trial_path, trial_energies, trial_forces[1:-1]
            )
            trial_displacement = project_normal(
                trial_path[1:-1] - reference_path[1:-1], tangents
            )
            trial_objective = (
                trial_energies[1:-1]
                + 0.5
                * np.sum(trial_displacement**2, axis=1)
                / float(alpha)
            )
            objective_ok = np.all(
                trial_objective
                <= reference_energies[1:-1]
                + 1e-13 * np.maximum(1.0, np.abs(reference_energies[1:-1]))
            )
            accepted_band = bool(
                objective_ok
                and trial_diag["i_perp"] <= reference_diag["i_perp"] + 1e-13
            )
            trace.record(
                trial_path if accepted_band else reference_path,
                trial_energies if accepted_band else reference_energies,
                trial_forces if accepted_band else reference_forces,
                alpha=alpha,
                accepted=accepted_band,
                alpha_action="hold",
                kkt_max=float(np.max(final_kkt)),
                kkt_relative_max=float(
                    np.max(final_kkt / np.maximum(initial_kkt, 1e-30))
                ),
                trust_radius=trust_radius,
                spring_residual=0.0,
            )
            if accepted_band:
                candidate_path = trial_path
                candidate_energies = trial_energies
                candidate_forces = trial_forces
                break
            scale *= 0.5

    if accepted_band:
        trace.record(
            candidate_path,
            candidate_energies,
            candidate_forces,
            alpha=alpha,
            accepted=True,
            alpha_action="hold",
            kkt_max=float(np.max(final_kkt)),
            kkt_relative_max=float(
                np.max(final_kkt / np.maximum(initial_kkt, 1e-30))
            ),
            trust_radius=trust_radius,
            spring_residual=0.0,
        )
    else:
        candidate_path = reference_path
        candidate_energies = reference_energies
        candidate_forces = reference_forces

    return {
        "path": candidate_path,
        "energies": candidate_energies,
        "forces": candidate_forces,
        "accepted_any": accepted_band,
        "reason": "kkt_or_inner_limit" if accepted_band else "global_guard",
        "initial_kkt": initial_kkt,
        "final_kkt": final_kkt,
        "relative_kkt": final_kkt / np.maximum(initial_kkt, 1e-30),
        "used_calls": used_calls,
        "spring_residual": 0.0,
        "reference_path": reference_path,
        "reference_energies": reference_energies,
    }


def run(args):
    root = args.root.expanduser().resolve()
    initial_path_file = args.initial_path_file.expanduser()
    if not initial_path_file.is_absolute():
        initial_path_file = root / initial_path_file
    path = load_common_path(initial_path_file)
    backend = LEPSBackend(label=args.run_label)
    evaluators = make_evaluators(backend, internal_limit=args.budget)
    energies, forces = evaluate_path(path, evaluators)
    saddle = reference_saddle()
    trace = Trace(evaluators, saddle["forward_barrier"])
    alpha = float(args.alpha)
    diagnostic = path_diagnostics(path, energies, forces[1:-1])
    trace.record(
        path,
        energies,
        forces,
        alpha=alpha,
        accepted=True,
        alpha_action="initial",
        kkt_max=float(np.max(diagnostic["per_image_l2"])),
        kkt_relative_max=1.0,
        trust_radius=rms_segment_trust_radius(
            path, args.trust_fraction, args.max_step
        ),
        spring_residual=0.0,
    )

    stop_reason = "unknown"
    outer_iterations = 0
    consecutive_stalls = 0
    consecutive_no_new_evaluation = 0
    while True:
        current_call = int(
            np.max(
                counter_arrays(evaluators)["joint_evaluations"][1:-1]
            )
        )
        if current_call >= args.budget:
            stop_reason = "evaluation_budget"
            break
        outer_start_call = current_call
        diagnostic = path_diagnostics(path, energies, forces[1:-1])
        if diagnostic["i_perp"] <= args.iperp_tolerance:
            stop_reason = "iperp_tolerance"
            break
        tangents = improved_tangents(path, energies)
        trust_radius = rms_segment_trust_radius(
            path, args.trust_fraction, args.max_step
        )
        before_path = path.copy()
        before_energies = energies.copy()
        before_forces = forces.copy()
        subproblem_budget = (
            args.budget - 1
            if args.reparameterize == "post_hermite"
            else args.budget
        )
        if args.subproblem_solver in {"independent", "independent_near_exact"}:
            result = solve_independent_band(
                path,
                energies,
                forces,
                tangents,
                evaluators,
                trace,
                alpha,
                trust_radius,
                args.inner_calls,
                args.kkt_rtol,
                args.kkt_atol,
                subproblem_budget,
                near_exact=(
                    args.subproblem_solver == "independent_near_exact"
                ),
            )
        else:
            result = solve_frozen_band(
                path,
                energies,
                forces,
                tangents,
                evaluators,
                trace,
                alpha,
                trust_radius,
                args.inner_calls,
                args.kkt_rtol,
                args.kkt_atol,
                args.history,
                args.reparameterize == "integrated_hermite",
                subproblem_budget,
            )
        kkt_targets = np.maximum(
            float(args.kkt_atol),
            float(args.kkt_rtol) * np.asarray(result["initial_kkt"]),
        )
        kkt_satisfied = (
            np.asarray(result["final_kkt"]) <= kkt_targets * (1.0 + 1e-10)
        )
        subproblem_kkt_fraction = float(np.mean(kkt_satisfied))
        subproblem_kkt_all = bool(np.all(kkt_satisfied))
        if trace.rows:
            trace.rows[-1]["subproblem_kkt_fraction"] = (
                subproblem_kkt_fraction
            )
            trace.rows[-1]["subproblem_kkt_all"] = subproblem_kkt_all
        raw_path = result["path"]
        raw_energies = result["energies"]
        raw_forces = result["forces"]
        quality, utilization, moved_count = proximal_quality(
            before_path,
            before_energies,
            before_forces,
            raw_path,
            raw_energies,
            tangents,
            alpha,
            trust_radius,
        )
        path = raw_path
        energies = raw_energies
        forces = raw_forces
        if result["accepted_any"] and args.reparameterize == "post_hermite":
            reparam_tangents = improved_tangents(raw_path, raw_energies)
            projected_path, spring_residual = improved_hermite_equal_chord(
                raw_path, reparam_tangents
            )
            current_counts = counter_arrays(evaluators)[
                "joint_evaluations"
            ][1:-1]
            blocked = [
                current_counts[i] >= int(args.budget)
                and not evaluators[i + 1].has_cached(projected_path[i + 1])
                for i in range(N_INTERNAL)
            ]
            if not any(blocked):
                try:
                    projected_energies, projected_forces = evaluate_path(
                        projected_path, evaluators
                    )
                    path = projected_path
                    energies = projected_energies
                    forces = projected_forces
                    projected_diag = path_diagnostics(
                        path, energies, forces[1:-1]
                    )
                    projected_residual = project_normal(
                        -forces[1:-1], improved_tangents(path, energies)
                    )
                    trace.record(
                        path,
                        energies,
                        forces,
                        alpha=alpha,
                        accepted=True,
                        alpha_action="reparameterize",
                        kkt_max=float(
                            np.max(np.linalg.norm(projected_residual, axis=1))
                        ),
                        kkt_relative_max=float(
                            np.max(np.linalg.norm(projected_residual, axis=1))
                            / max(
                                np.max(
                                    np.linalg.norm(
                                        project_normal(
                                            -before_forces[1:-1], tangents
                                        ),
                                        axis=1,
                                    )
                                ),
                                1e-30,
                            )
                        ),
                        trust_radius=trust_radius,
                        spring_residual=spring_residual,
                        subproblem_kkt_fraction=subproblem_kkt_fraction,
                        subproblem_kkt_all=subproblem_kkt_all,
                    )
                    del projected_diag
                except EvaluationBudgetReached:
                    pass
        old_alpha = alpha
        alpha_action = "fixed"
        if args.alpha_mode == "adaptive":
            alpha, alpha_action = adapt_alpha(
                alpha,
                quality,
                utilization,
                moved_count,
                args.alpha_min,
                args.alpha_max,
            )
        if trace.rows:
            trace.rows[-1]["alpha_action"] = alpha_action

        if result["accepted_any"]:
            consecutive_stalls = 0
        else:
            consecutive_stalls += 1
        outer_iterations += 1

        current_call = int(
            np.max(
                counter_arrays(evaluators)["joint_evaluations"][1:-1]
            )
        )
        if current_call >= args.budget:
            stop_reason = "evaluation_budget"
            break
        if current_call <= outer_start_call:
            consecutive_no_new_evaluation += 1
        else:
            consecutive_no_new_evaluation = 0
        if consecutive_no_new_evaluation >= 3:
            stop_reason = "no_new_evaluation_cycle"
            break
        if not result["accepted_any"]:
            if args.alpha_mode == "fixed":
                stop_reason = "stalled_fixed_alpha"
                break
            if alpha == old_alpha or consecutive_stalls >= args.max_stalls:
                stop_reason = "stalled_adaptive_alpha"
                break

    audit = audit_counter_consistency(backend, evaluators)
    if not audit["consistent"]:
        raise RuntimeError("LEPS IVM counter audit failed")
    configuration = {
        "method": "IVM",
        "run_label": args.run_label,
        "initial_path_file": project_relative(initial_path_file, root),
        "initial_path_sha256": array_sha256(load_common_path(initial_path_file)),
        "tangent_definition": TANGENT_DEFINITION,
        "alpha": args.alpha,
        "alpha_mode": args.alpha_mode,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "kkt_rtol": args.kkt_rtol,
        "kkt_atol": args.kkt_atol,
        "kkt_definition": (
            "one-dimensional box-constrained projected-gradient residual"
        ),
        "inner_calls": args.inner_calls,
        "history": args.history,
        "reparameterize": args.reparameterize,
        "trust_fraction": args.trust_fraction,
        "max_step": args.max_step,
        "budget": args.budget,
        "global_iperp_guard": True,
        "zero_new_evaluation_outer_guard": 3,
        "solver": (
            (
                "independent one-dimensional SciPy L-BFGS-B "
                "near-exact solve"
                if args.subproblem_solver == "independent_near_exact"
                else "independent one-dimensional projected L-BFGS"
            )
            if args.subproblem_solver
            in {"independent", "independent_near_exact"}
            else "whole-band projected L-BFGS"
        ),
        "subproblem_solver": args.subproblem_solver,
        "stop_reason": stop_reason,
        "outer_iterations": outer_iterations,
    }
    config_sha = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()
    output = (
        root
        / "runs"
        / "ivm"
        / f"{args.run_label}_cfg{config_sha[:12]}_budget{args.budget}.npz"
    )
    payload = trace.arrays()
    atomic_save_npz(
        output,
        method=np.asarray("IVM"),
        run_label=np.asarray(args.run_label),
        configuration_sha256=np.asarray(config_sha),
        tangent_definition=np.asarray(TANGENT_DEFINITION),
        initial_path=load_common_path(initial_path_file),
        initial_path_sha256=np.asarray(configuration["initial_path_sha256"]),
        final_path=path,
        stop_reason=np.asarray(stop_reason),
        outer_iterations=np.asarray(outer_iterations),
        reference_barrier=np.asarray(saddle["forward_barrier"]),
        alpha_initial=np.asarray(args.alpha),
        alpha_mode=np.asarray(args.alpha_mode),
        kkt_rtol=np.asarray(args.kkt_rtol),
        kkt_atol=np.asarray(args.kkt_atol),
        kkt_definition=np.asarray(configuration["kkt_definition"]),
        inner_calls=np.asarray(args.inner_calls),
        reparameterize=np.asarray(args.reparameterize),
        subproblem_solver=np.asarray(args.subproblem_solver),
        counter_audit_consistent=np.asarray(audit["consistent"]),
        backend_raw_joint_evaluations=np.asarray(
            audit["backend_raw_joint_evaluations"]
        ),
        **payload,
    )
    write_json(output.with_suffix(".audit.json"), {
        "configuration": configuration,
        "configuration_sha256": config_sha,
        "counter_audit": audit,
        "result": project_relative(output, root),
    })
    print(output)
    return output


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--initial-path-file",
        type=Path,
        default=Path("inputs/common_initial_path.npz"),
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument(
        "--alpha-mode", choices=("fixed", "adaptive"), default="fixed"
    )
    parser.add_argument("--alpha-min", type=float, default=0.03)
    parser.add_argument("--alpha-max", type=float, default=300.0)
    parser.add_argument("--kkt-rtol", type=float, default=0.8)
    parser.add_argument("--kkt-atol", type=float, default=1e-10)
    parser.add_argument("--inner-calls", type=int, default=15)
    parser.add_argument("--history", type=int, default=7)
    parser.add_argument(
        "--subproblem-solver",
        choices=("independent", "independent_near_exact", "whole_band"),
        default="independent",
    )
    parser.add_argument(
        "--reparameterize",
        choices=("none", "post_hermite", "integrated_hermite"),
        default="post_hermite",
    )
    parser.add_argument("--trust-fraction", type=float, default=2.0)
    parser.add_argument("--max-step", type=float, default=1.0)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--max-stalls", type=int, default=12)
    parser.add_argument("--iperp-tolerance", type=float, default=1e-14)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
