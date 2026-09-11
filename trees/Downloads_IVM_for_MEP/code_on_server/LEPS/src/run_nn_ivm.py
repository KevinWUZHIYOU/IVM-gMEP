"""Audited neural-network IVM on the two-dimensional LEPS potential."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator

from continuous_paths import (
    FixedEndpointPathNet,
    arclength_parameters_for_fractions,
    dense_neural_metrics,
    equal_arclength_parameters,
    fit_path_targets,
    load_parameter_vector,
    model_coordinates,
    normalized_autograd_tangent,
    parameter_vector,
    resample_polyline_equal_arclength,
    stratified_jitter_fractions,
)
from leps_core import (
    N_IMAGES,
    EvaluationBudgetReached,
    LEPSBackend,
    array_sha256,
    atomic_save_npz,
    audit_counter_consistency,
    counter_arrays,
    load_common_path,
    make_evaluators,
    physical_to_flat,
    reference_saddle,
    write_json,
)


def query_physical(evaluator, point):
    energy, force_flat = evaluator.query(physical_to_flat(point), "both")
    force = np.asarray(force_flat, dtype=np.float64)[[0, 3]]
    return float(energy), force


def sampled_i_perp(points, tangents, forces, endpoints):
    """Discrete arclength quadrature used only as a counted acceptance guard."""
    points = np.asarray(points, dtype=np.float64)
    tangents = np.asarray(tangents, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64)
    endpoints = np.asarray(endpoints, dtype=np.float64)
    band = np.vstack((endpoints[0], points, endpoints[1]))
    segment_lengths = np.linalg.norm(np.diff(band, axis=0), axis=1)
    weights = 0.5 * (segment_lengths[:-1] + segment_lengths[1:])
    normal_forces = forces - np.sum(
        forces * tangents,
        axis=1,
        keepdims=True,
    ) * tangents
    return float(np.sum(weights * np.sum(normal_forces**2, axis=1)))


def stratified_quadrature_weights(fractions):
    """Voronoi quadrature weights for sorted fractions on [0, 1]."""
    fractions = np.asarray(fractions, dtype=np.float64)
    boundaries = np.empty(fractions.size + 1, dtype=np.float64)
    boundaries[0] = 0.0
    boundaries[-1] = 1.0
    boundaries[1:-1] = 0.5 * (fractions[:-1] + fractions[1:])
    return np.diff(boundaries)


def solve_scalar_normal_subproblem(
    evaluator,
    point,
    normal,
    *,
    alpha,
    trust_radius,
    maxfun,
    kkt_rtol,
    kkt_atol,
):
    """Solve one frozen-normal IVM subproblem with joint value/gradient calls."""
    point = np.asarray(point, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    initial_energy, initial_force = query_physical(evaluator, point)
    initial_gradient = -float(np.dot(initial_force, normal))
    kkt_scale = max(abs(initial_gradient), float(kkt_atol))

    def objective_and_gradient(z_array):
        z = float(np.asarray(z_array).reshape(-1)[0])
        energy, force = query_physical(evaluator, point + z * normal)
        objective = energy + 0.5 * z * z / float(alpha)
        gradient = -float(np.dot(force, normal)) + z / float(alpha)
        return float(objective), np.asarray([gradient], dtype=np.float64)

    result = minimize(
        objective_and_gradient,
        np.zeros(1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-float(trust_radius), float(trust_radius))],
        options={
            "maxiter": int(maxfun),
            "maxfun": int(maxfun),
            "ftol": 1e-14,
            "gtol": max(float(kkt_atol), float(kkt_rtol) * kkt_scale),
            "maxls": 20,
        },
    )
    z = float(np.clip(result.x[0], -trust_radius, trust_radius))
    final_energy, final_force = query_physical(evaluator, point + z * normal)
    final_gradient = -float(np.dot(final_force, normal)) + z / float(alpha)
    objective = final_energy + 0.5 * z * z / float(alpha)
    bound_tolerance = 1e-10 * max(1.0, float(trust_radius))
    if z <= -float(trust_radius) + bound_tolerance:
        projected_gradient = min(final_gradient, 0.0)
    elif z >= float(trust_radius) - bound_tolerance:
        projected_gradient = max(final_gradient, 0.0)
    else:
        projected_gradient = final_gradient
    kkt_relative = abs(projected_gradient) / kkt_scale
    accepted = bool(
        np.isfinite(objective)
        and objective <= initial_energy + 1e-10
        and (
            abs(projected_gradient) <= float(kkt_atol)
            or kkt_relative <= float(kkt_rtol)
            or result.success
        )
    )
    if not accepted:
        z = 0.0
        final_energy = initial_energy
        objective = initial_energy
    return {
        "point": point,
        "target": point + z * normal,
        "z": z,
        "initial_energy": initial_energy,
        "initial_force": initial_force,
        "target_energy": final_energy,
        "target_force": final_force,
        "target_objective": objective,
        "initial_gradient": initial_gradient,
        "final_gradient": final_gradient,
        "kkt_relative": kkt_relative,
        "accepted": accepted,
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
    }


class NeuralTrace:
    def __init__(self):
        self.rows = []

    def append(
        self,
        model,
        evaluators,
        metrics,
        *,
        accepted,
        outer_iteration,
        fit_rms,
        fit_max,
        relaxation,
        kkt_relative_max,
        sampled_i_perp_guard=np.nan,
        reference_sampled_i_perp=np.nan,
        integrated_objective_change=np.nan,
        maximum_point_objective_increase=np.nan,
        collocation_arc_fractions=None,
        effective_fit_tolerance=np.nan,
        target_max_displacement=np.nan,
    ):
        counters = counter_arrays(evaluators)
        internal = counters["joint_evaluations"][1:-1]
        self.rows.append(
            {
                "call": int(np.max(internal)),
                "mean_calls": float(np.mean(internal)),
                "total_internal_calls": int(np.sum(internal)),
                "counter_vector": counters["joint_evaluations"].copy(),
                "energy_request_vector": counters["energy_requests"].copy(),
                "force_request_vector": counters["force_requests"].copy(),
                "cache_hit_vector": counters["cache_hits"].copy(),
                "dense_i_perp": float(metrics["dense_i_perp"]),
                "dense_fmax": float(metrics["dense_fmax"]),
                "dense_barrier": float(metrics["dense_barrier"]),
                "path_length": float(metrics["path_length"]),
                "minimum_speed": float(metrics["minimum_speed"]),
                "spring_force_max_unit": float(
                    metrics["spring_force_max_unit"]
                ),
                "segment_length_cv": float(metrics["segment_length_cv"]),
                "sample_s": np.asarray(metrics["sample_s"], dtype=np.float64),
                "sample_path": np.asarray(
                    metrics["sample_path"],
                    dtype=np.float64,
                ),
                "model_state": parameter_vector(model),
                "accepted": bool(accepted),
                "outer_iteration": int(outer_iteration),
                "fit_rms": float(fit_rms),
                "fit_max": float(fit_max),
                "relaxation": float(relaxation),
                "kkt_relative_max": float(kkt_relative_max),
                "sampled_i_perp_guard": float(sampled_i_perp_guard),
                "reference_sampled_i_perp": float(
                    reference_sampled_i_perp
                ),
                "integrated_objective_change": float(
                    integrated_objective_change
                ),
                "maximum_point_objective_increase": float(
                    maximum_point_objective_increase
                ),
                "collocation_arc_fractions": np.asarray(
                    (
                        metrics["sample_s"][1:-1]
                        if collocation_arc_fractions is None
                        else collocation_arc_fractions
                    ),
                    dtype=np.float64,
                ),
                "effective_fit_tolerance": float(effective_fit_tolerance),
                "target_max_displacement": float(target_max_displacement),
            }
        )

    def arrays(self):
        scalar_float = (
            "mean_calls",
            "dense_i_perp",
            "dense_fmax",
            "dense_barrier",
            "path_length",
            "minimum_speed",
            "spring_force_max_unit",
            "segment_length_cv",
            "fit_rms",
            "fit_max",
            "relaxation",
            "kkt_relative_max",
            "sampled_i_perp_guard",
            "reference_sampled_i_perp",
            "integrated_objective_change",
            "maximum_point_objective_increase",
            "effective_fit_tolerance",
            "target_max_displacement",
        )
        scalar_int = ("call", "total_internal_calls", "outer_iteration")
        payload = {}
        for key in scalar_float:
            payload[key] = np.asarray(
                [row[key] for row in self.rows],
                dtype=np.float64,
            )
        for key in scalar_int:
            payload[key] = np.asarray(
                [row[key] for row in self.rows],
                dtype=np.int64,
            )
        payload["calls_per_image"] = payload.pop("call")
        payload["accepted"] = np.asarray(
            [row["accepted"] for row in self.rows],
            dtype=bool,
        )
        for key in (
            "counter_vector",
            "energy_request_vector",
            "force_request_vector",
            "cache_hit_vector",
            "sample_s",
            "sample_path",
            "model_state",
            "collocation_arc_fractions",
        ):
            payload[key] = np.asarray([row[key] for row in self.rows])
        return payload


def configuration_hash(configuration: dict) -> str:
    encoded = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run(args):
    root = Path(args.root).expanduser().resolve()
    common_path = load_common_path(root / "inputs" / "common_initial_path.npz")
    model = FixedEndpointPathNet(
        width=args.width,
        depth=args.depth,
        fourier_order=args.fourier_order,
        seed=args.seed,
    )

    initial_sample = model_coordinates(model, np.linspace(0.0, 1.0, N_IMAGES))
    initial_flat = physical_to_flat(initial_sample)
    if not np.allclose(initial_flat, common_path, atol=2e-14, rtol=0.0):
        raise RuntimeError("NN-IVM does not start from the common initial path")

    backend = LEPSBackend(label=f"nn_ivm:{args.run_label}")
    collocation_count = int(args.collocation_count)
    if collocation_count != 20:
        raise ValueError(
            "This comparison requires exactly 20 internal collocation points"
        )
    if int(args.sampled_iperp_patience) < 1:
        raise ValueError("sampled_iperp_patience must be positive")
    evaluators = make_evaluators(
        backend,
        count=collocation_count + 2,
        internal_limit=args.budget,
    )
    # Endpoints are part of the run's evaluator audit and are cached once.
    query_physical(evaluators[0], initial_sample[0])
    query_physical(evaluators[-1], initial_sample[-1])

    trace = NeuralTrace()
    saddle = reference_saddle()
    metrics = dense_neural_metrics(
        model,
        dense_count=args.diagnostic_points,
    )
    metrics["dense_barrier_error"] = abs(
        metrics["dense_barrier"] - saddle["forward_barrier"]
    )
    initial_s = equal_arclength_parameters(
        model,
        collocation_count,
        dense_count=args.arclength_points,
    )
    initial_arc_fractions = stratified_jitter_fractions(
        collocation_count,
        0,
        jitter=args.collocation_jitter,
        seed=args.sampling_seed,
    )
    initial_points = model_coordinates(model, initial_s)
    for evaluator, point in zip(evaluators[1:-1], initial_points):
        query_physical(evaluator, point)
    trace.append(
        model,
        evaluators,
        metrics,
        accepted=True,
        outer_iteration=0,
        fit_rms=0.0,
        fit_max=0.0,
        relaxation=0.0,
        kkt_relative_max=np.nan,
        sampled_i_perp_guard=np.nan,
        reference_sampled_i_perp=np.nan,
        integrated_objective_change=np.nan,
        maximum_point_objective_increase=np.nan,
        collocation_arc_fractions=initial_arc_fractions,
    )

    stop_reason = "budget"
    outer_iteration = 0
    consecutive_stalls = 0
    consecutive_sampled_convergence = 0
    while True:
        internal_counts = counter_arrays(evaluators)["joint_evaluations"][1:-1]
        if int(np.max(internal_counts)) >= int(args.budget):
            stop_reason = "budget"
            break
        if metrics["dense_i_perp"] <= float(args.iperp_tolerance):
            stop_reason = "dense_i_perp"
            break
        if outer_iteration >= int(args.max_outer):
            stop_reason = "outer_limit"
            break

        outer_iteration += 1
        arc_fractions = stratified_jitter_fractions(
            collocation_count,
            outer_iteration,
            jitter=args.collocation_jitter,
            seed=args.sampling_seed,
        )
        sample_s = arclength_parameters_for_fractions(
            model,
            arc_fractions,
            dense_count=args.arclength_points,
        )
        old_points, tangents, _ = normalized_autograd_tangent(model, sample_s)
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
        subproblems = []
        budget_hit = False
        try:
            for evaluator, point, normal in zip(
                evaluators[1:-1],
                old_points,
                normals,
            ):
                subproblems.append(
                    solve_scalar_normal_subproblem(
                        evaluator,
                        point,
                        normal,
                        alpha=args.alpha,
                        trust_radius=args.trust_radius,
                        maxfun=args.inner_calls,
                        kkt_rtol=args.kkt_rtol,
                        kkt_atol=args.kkt_atol,
                    )
                )
        except EvaluationBudgetReached:
            budget_hit = True

        if budget_hit or len(subproblems) != collocation_count:
            stop_reason = "budget"
            # Preserve all real calls made before the budget interruption.
            trace.append(
                model,
                evaluators,
                metrics,
                accepted=False,
                outer_iteration=outer_iteration,
                fit_rms=np.nan,
                fit_max=np.nan,
                relaxation=0.0,
                kkt_relative_max=np.nan,
                sampled_i_perp_guard=np.nan,
                reference_sampled_i_perp=np.nan,
                integrated_objective_change=np.nan,
                maximum_point_objective_increase=np.nan,
                collocation_arc_fractions=arc_fractions,
            )
            break

        normal_steps = np.asarray(
            [result["z"] for result in subproblems],
            dtype=np.float64,
        )
        reference_energies = np.asarray(
            [result["initial_energy"] for result in subproblems],
            dtype=np.float64,
        )
        reference_forces = np.asarray(
            [result["initial_force"] for result in subproblems],
            dtype=np.float64,
        )
        endpoints = model_coordinates(model, [0.0, 1.0])
        reference_sampled_i_perp = sampled_i_perp(
            old_points,
            tangents,
            reference_forces,
            endpoints,
        )
        kkt_relative_max = float(
            np.max([result["kkt_relative"] for result in subproblems])
        )
        if (
            float(args.sampled_iperp_tolerance) > 0.0
            and reference_sampled_i_perp
            <= float(args.sampled_iperp_tolerance)
        ):
            consecutive_sampled_convergence += 1
        else:
            consecutive_sampled_convergence = 0
        if (
            consecutive_sampled_convergence
            >= int(args.sampled_iperp_patience)
        ):
            stop_reason = "sampled_i_perp"
            trace.append(
                model,
                evaluators,
                metrics,
                accepted=False,
                outer_iteration=outer_iteration,
                fit_rms=np.nan,
                fit_max=np.nan,
                relaxation=0.0,
                kkt_relative_max=kkt_relative_max,
                sampled_i_perp_guard=reference_sampled_i_perp,
                reference_sampled_i_perp=reference_sampled_i_perp,
                integrated_objective_change=np.nan,
                maximum_point_objective_increase=np.nan,
                collocation_arc_fractions=arc_fractions,
            )
            break
        quadrature_weights = stratified_quadrature_weights(arc_fractions)
        old_vector = parameter_vector(model)
        accepted = False
        fit_result = None
        accepted_scale = 0.0
        accepted_integrated_objective_change = np.nan
        accepted_maximum_point_objective_increase = np.nan
        effective_fit_tolerance = np.nan
        target_max_displacement = np.nan
        fit_s = np.linspace(
            0.0,
            1.0,
            int(args.fit_points) + 2,
        )[1:-1]
        fit_old_points, fit_tangents, _ = normalized_autograd_tangent(
            model,
            fit_s,
        )
        fit_normals = np.column_stack(
            (-fit_tangents[:, 1], fit_tangents[:, 0])
        )
        step_interpolant = PchipInterpolator(
            np.concatenate(([0.0], sample_s, [1.0])),
            np.concatenate(([0.0], normal_steps, [0.0])),
            extrapolate=False,
        )
        dense_normal_steps = np.asarray(
            step_interpolant(fit_s),
            dtype=np.float64,
        )

        for attempt, scale in enumerate(args.fit_relaxations):
            load_parameter_vector(model, old_vector)
            fit_targets = fit_old_points + (
                float(scale)
                * dense_normal_steps[:, None]
                * fit_normals
            )
            full_target = np.vstack(
                (endpoints[0], fit_targets, endpoints[1])
            )
            if args.fit_arclength_gauge:
                full_target = resample_polyline_equal_arclength(
                    full_target,
                )
                fit_targets = full_target[1:-1]
            target_max_displacement = float(
                np.max(np.linalg.norm(fit_targets - fit_old_points, axis=1))
            )
            effective_fit_tolerance = min(
                float(args.fit_tolerance),
                float(args.fit_absolute_tolerance)
                + float(args.fit_relative_tolerance)
                * target_max_displacement,
            )
            full_fit_s = np.linspace(0.0, 1.0, fit_s.size + 2)
            target_derivatives = np.gradient(
                full_target,
                full_fit_s,
                axis=0,
                edge_order=2,
            )[1:-1]
            fit_result = fit_path_targets(
                model,
                fit_s,
                fit_targets,
                target_tangents=target_derivatives,
                tangent_weight=args.fit_tangent_weight,
                smoothness_weight=args.smoothness,
                adam_steps=(
                    args.fit_adam_first
                    if outer_iteration == 1 and attempt == 0
                    else args.fit_adam
                ),
                adam_lr=args.fit_lr,
                lbfgs_steps=args.fit_lbfgs,
                polish_lbfgs_steps=args.fit_polish_lbfgs,
                polish_tangent_weight=args.fit_polish_tangent_weight,
            )
            candidate_s = arclength_parameters_for_fractions(
                model,
                arc_fractions,
                dense_count=args.arclength_points,
            )
            candidate_points = model_coordinates(model, candidate_s)
            if (
                not np.all(np.isfinite(candidate_points))
                or fit_result.max_error > effective_fit_tolerance
            ):
                continue
            candidate_energies = []
            candidate_forces = []
            try:
                for evaluator, point in zip(
                    evaluators[1:-1],
                    candidate_points,
                ):
                    energy, force = query_physical(evaluator, point)
                    candidate_energies.append(energy)
                    candidate_forces.append(force)
            except EvaluationBudgetReached:
                budget_hit = True
                break
            candidate_energies = np.asarray(
                candidate_energies,
                dtype=np.float64,
            )
            candidate_forces = np.asarray(
                candidate_forces,
                dtype=np.float64,
            )
            _, candidate_tangents, _ = normalized_autograd_tangent(
                model,
                candidate_s,
            )
            candidate_sampled_i_perp = sampled_i_perp(
                candidate_points,
                candidate_tangents,
                candidate_forces,
                endpoints,
            )
            displacement = candidate_points - old_points
            candidate_objective = candidate_energies + np.sum(
                displacement**2,
                axis=1,
            ) / (2.0 * float(args.alpha))
            objective_change = candidate_objective - reference_energies
            integrated_objective_change = float(
                np.sum(quadrature_weights * objective_change)
            )
            maximum_point_objective_increase = float(
                np.max(objective_change)
            )
            objective_ok = bool(
                integrated_objective_change
                < -float(args.acceptance_tolerance)
                and maximum_point_objective_increase
                <= float(args.pointwise_objective_guard)
            )
            i_perp_ok = bool(
                np.isfinite(candidate_sampled_i_perp)
                and candidate_sampled_i_perp
                <= (
                    (1.0 + float(args.iperp_guard_relative_growth))
                    * reference_sampled_i_perp
                    + float(args.iperp_guard_tolerance)
                    * max(1.0, reference_sampled_i_perp)
                )
            )
            geometry_probe = model_coordinates(
                model,
                np.linspace(0.0, 1.0, 1025),
            )
            candidate_length = float(
                np.sum(
                    np.linalg.norm(
                        np.diff(geometry_probe, axis=0),
                        axis=1,
                    )
                )
            )
            geometry_ok = bool(
                np.min(candidate_points) > 0.25
                and np.max(candidate_points) < 8.5
                and candidate_length < float(args.max_path_length)
            )
            if objective_ok and i_perp_ok and geometry_ok:
                accepted = True
                accepted_scale = float(scale)
                accepted_integrated_objective_change = (
                    integrated_objective_change
                )
                accepted_maximum_point_objective_increase = (
                    maximum_point_objective_increase
                )
                break

        if budget_hit:
            load_parameter_vector(model, old_vector)
            stop_reason = "budget"
        elif not accepted:
            load_parameter_vector(model, old_vector)
            consecutive_stalls += 1
            stop_reason = "fit_or_global_guard"
        else:
            consecutive_stalls = 0

        metrics = dense_neural_metrics(
            model,
            dense_count=args.diagnostic_points,
        )
        metrics["dense_barrier_error"] = abs(
            metrics["dense_barrier"] - saddle["forward_barrier"]
        )
        trace.append(
            model,
            evaluators,
            metrics,
            accepted=accepted,
            outer_iteration=outer_iteration,
            fit_rms=np.nan if fit_result is None else fit_result.rms_error,
            fit_max=np.nan if fit_result is None else fit_result.max_error,
            relaxation=accepted_scale,
            kkt_relative_max=kkt_relative_max,
            sampled_i_perp_guard=(
                candidate_sampled_i_perp
                if accepted
                else reference_sampled_i_perp
            ),
            reference_sampled_i_perp=reference_sampled_i_perp,
            integrated_objective_change=(
                accepted_integrated_objective_change
            ),
            maximum_point_objective_increase=(
                accepted_maximum_point_objective_increase
            ),
            collocation_arc_fractions=arc_fractions,
            effective_fit_tolerance=effective_fit_tolerance,
            target_max_displacement=target_max_displacement,
        )

        if budget_hit:
            break
        if consecutive_stalls >= int(args.max_stalls):
            stop_reason = "stalled"
            break

    audit = audit_counter_consistency(backend, evaluators)
    if not audit["consistent"]:
        raise RuntimeError(f"NN-IVM counter audit failed: {audit}")

    configuration = {
        "method": "NN-IVM",
        "run_label": args.run_label,
        "alpha": float(args.alpha),
        "width": int(args.width),
        "depth": int(args.depth),
        "fourier_order": int(args.fourier_order),
        "seed": int(args.seed),
        "trust_radius": float(args.trust_radius),
        "inner_calls": int(args.inner_calls),
        "kkt_rtol": float(args.kkt_rtol),
        "kkt_atol": float(args.kkt_atol),
        "fit_relaxations": [float(value) for value in args.fit_relaxations],
        "fit_tolerance": float(args.fit_tolerance),
        "fit_absolute_tolerance": float(args.fit_absolute_tolerance),
        "fit_relative_tolerance": float(args.fit_relative_tolerance),
        "smoothness": float(args.smoothness),
        "fit_tangent_weight": float(args.fit_tangent_weight),
        "fit_polish_lbfgs": int(args.fit_polish_lbfgs),
        "fit_polish_tangent_weight": float(
            args.fit_polish_tangent_weight
        ),
        "fit_arclength_gauge": bool(args.fit_arclength_gauge),
        "iperp_guard_relative_growth": float(
            args.iperp_guard_relative_growth
        ),
        "pointwise_objective_guard": float(
            args.pointwise_objective_guard
        ),
        "sampled_iperp_tolerance": float(
            args.sampled_iperp_tolerance
        ),
        "sampled_iperp_patience": int(args.sampled_iperp_patience),
        "fit_points": int(args.fit_points),
        "budget": int(args.budget),
        "collocation_count": collocation_count,
        "collocation_jitter": float(args.collocation_jitter),
        "sampling_seed": int(args.sampling_seed),
        "sampling": (
            "stratified equal-arclength collocation with reproducible "
            "within-stratum jitter"
        ),
        "tangent": "normalized torch autograd dphi/ds",
        "reparameterization": "none",
    }
    arrays = trace.arrays()
    arrays.update(
        {
            "method": np.asarray("NN-IVM"),
            "run_label": np.asarray(args.run_label),
            "alpha": np.asarray(args.alpha),
            "width": np.asarray(args.width),
            "depth": np.asarray(args.depth),
            "fourier_order": np.asarray(args.fourier_order),
            "seed": np.asarray(args.seed),
            "initial_path": common_path,
            "initial_path_sha256": np.asarray(array_sha256(common_path)),
            "configuration_sha256": np.asarray(
                configuration_hash(configuration)
            ),
            "reference_barrier": np.asarray(saddle["forward_barrier"]),
            "dense_barrier_error": np.abs(
                arrays["dense_barrier"] - saddle["forward_barrier"]
            ),
            "backend_raw_joint_evaluations": np.asarray(
                backend.raw_joint_evaluations
            ),
            "counter_audit_consistent": np.asarray(audit["consistent"]),
            "stop_reason": np.asarray(stop_reason),
            "tangent_definition": np.asarray(
                "normalized torch autograd dphi/ds"
            ),
            "sampling_definition": np.asarray(
                "stratified equal-arclength collocation with reproducible "
                "within-stratum jitter"
            ),
            "diagnostic_points": np.asarray(args.diagnostic_points),
            "collocation_count": np.asarray(collocation_count),
            "collocation_jitter": np.asarray(args.collocation_jitter),
            "sampling_seed": np.asarray(args.sampling_seed),
        }
    )

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    atomic_save_npz(output, **arrays)
    audit_path = output.with_suffix(".audit.json")
    write_json(
        audit_path,
        {
            "configuration": configuration,
            "configuration_sha256": configuration_hash(configuration),
            "stop_reason": stop_reason,
            "counter_audit": audit,
            "final": {
                "calls_per_image": int(arrays["calls_per_image"][-1]),
                "dense_i_perp": float(arrays["dense_i_perp"][-1]),
                "dense_barrier": float(arrays["dense_barrier"][-1]),
                "dense_barrier_error": float(
                    arrays["dense_barrier_error"][-1]
                ),
                "spring_force_max_unit": float(
                    arrays["spring_force_max_unit"][-1]
                ),
            },
        },
    )
    return output


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--alpha", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--fourier-order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--trust-radius", type=float, default=1.0)
    parser.add_argument("--inner-calls", type=int, default=40)
    parser.add_argument("--kkt-rtol", type=float, default=0.05)
    parser.add_argument("--kkt-atol", type=float, default=1e-8)
    parser.add_argument("--fit-tolerance", type=float, default=5e-4)
    parser.add_argument("--fit-absolute-tolerance", type=float, default=5e-4)
    parser.add_argument("--fit-relative-tolerance", type=float, default=0.10)
    parser.add_argument("--smoothness", type=float, default=1e-10)
    parser.add_argument("--fit-tangent-weight", type=float, default=0.1)
    gauge_group = parser.add_mutually_exclusive_group()
    gauge_group.add_argument(
        "--fit-arclength-gauge",
        dest="fit_arclength_gauge",
        action="store_true",
        help=(
            "Represent the fitted target curve with an equal-arclength "
            "parameter gauge; this changes only s, never the geometric path."
        ),
    )
    gauge_group.add_argument(
        "--no-fit-arclength-gauge",
        dest="fit_arclength_gauge",
        action="store_false",
    )
    parser.set_defaults(fit_arclength_gauge=True)
    parser.add_argument("--fit-points", type=int, default=127)
    parser.add_argument("--fit-adam-first", type=int, default=600)
    parser.add_argument("--fit-adam", type=int, default=120)
    parser.add_argument("--fit-lr", type=float, default=2e-2)
    parser.add_argument("--fit-lbfgs", type=int, default=80)
    parser.add_argument("--fit-polish-lbfgs", type=int, default=40)
    parser.add_argument(
        "--fit-polish-tangent-weight",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--fit-relaxations",
        type=float,
        nargs="+",
        default=(
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
        ),
    )
    parser.add_argument("--acceptance-tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--pointwise-objective-guard",
        type=float,
        default=5e-2,
        help=(
            "Maximum local proximal-objective increase allowed when the "
            "collocation-average path-space objective decreases."
        ),
    )
    parser.add_argument("--iperp-guard-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--iperp-guard-relative-growth",
        type=float,
        default=0.25,
        help=(
            "Safety cap for one projected band's counted sampled I_perp; "
            "the proximal objective remains the actual IVM acceptance test."
        ),
    )
    parser.add_argument("--max-path-length", type=float, default=18.0)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--collocation-count", type=int, default=20)
    parser.add_argument("--collocation-jitter", type=float, default=0.30)
    parser.add_argument("--sampling-seed", type=int, default=20260726)
    parser.add_argument("--max-outer", type=int, default=200)
    parser.add_argument("--max-stalls", type=int, default=4)
    parser.add_argument("--iperp-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--sampled-iperp-tolerance",
        type=float,
        default=0.0,
        help=(
            "Optional convergence threshold for the current path's "
            "20-point stratified I_perp estimator; zero disables it."
        ),
    )
    parser.add_argument(
        "--sampled-iperp-patience",
        type=int,
        default=3,
        help=(
            "Number of consecutive independently jittered 20-point "
            "estimates required below --sampled-iperp-tolerance."
        ),
    )
    parser.add_argument("--arclength-points", type=int, default=4097)
    parser.add_argument("--diagnostic-points", type=int, default=4097)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
