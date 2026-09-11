"""Neural IVM whose subminimization variables are network parameters.

At outer iteration k this program solves, with exactly 20 collocation streams,

    min_theta sum_j w_j [
        U(phi_k(s_j) + P_perp,k(s_j) delta_theta(s_j))
        + |P_perp,k delta_theta(s_j)|^2 / (2 alpha)
    ].

The accepted continuous path is the projected neural trial itself.  There is
no pointwise minimization followed by a separate path fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from continuous_paths import (
    FixedEndpointPathNet,
    ProjectedUpdatePath,
    ScalarEndpointUpdateNet,
    arclength_parameters_for_fractions,
    dense_neural_metrics,
    fit_path_targets,
    load_parameter_vector,
    model_coordinates,
    normalized_autograd_tangent,
    parameter_vector,
    stratified_jitter_fractions,
)
from leps_core import (
    N_IMAGES,
    EvaluationBudgetReached,
    LEPSBackend,
    atomic_save_npz,
    audit_counter_consistency,
    counter_arrays,
    load_common_path,
    make_evaluators,
    physical_to_flat,
    reference_saddle,
    write_json,
)
from torch_leps import torch_leps_energy_force


TORCH_DTYPE = torch.float64


def configuration_hash(configuration):
    encoded = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stratified_quadrature_weights(fractions):
    fractions = np.asarray(fractions, dtype=np.float64).reshape(-1)
    boundaries = np.empty(fractions.size + 1, dtype=np.float64)
    boundaries[0] = 0.0
    boundaries[-1] = 1.0
    boundaries[1:-1] = 0.5 * (fractions[:-1] + fractions[1:])
    return np.diff(boundaries)


def query_auditor(evaluator, point):
    energy, force_flat = evaluator.query(physical_to_flat(point), "both")
    force = np.asarray(force_flat, dtype=np.float64)[[0, 3]]
    return float(energy), force


def audit_torch_batch(evaluators, points, torch_energy, torch_force):
    auditor_energy = []
    auditor_force = []
    for evaluator, point in zip(evaluators, points):
        energy, force = query_auditor(evaluator, point)
        auditor_energy.append(energy)
        auditor_force.append(force)
    auditor_energy = np.asarray(auditor_energy, dtype=np.float64)
    auditor_force = np.asarray(auditor_force, dtype=np.float64)
    energy_error = float(
        np.max(np.abs(auditor_energy - torch_energy.detach().cpu().numpy()))
    )
    force_error = float(
        np.max(np.abs(auditor_force - torch_force.detach().cpu().numpy()))
    )
    if energy_error > 5e-12 or force_error > 5e-11:
        raise RuntimeError(
            "Torch LEPS disagrees with the audited reference backend: "
            f"energy={energy_error}, force={force_error}"
        )
    return energy_error, force_error


def sampled_i_perp(points, tangents, forces, endpoints):
    full = np.vstack((endpoints[0], points, endpoints[1]))
    left = np.linalg.norm(points - full[:-2], axis=1)
    right = np.linalg.norm(full[2:] - points, axis=1)
    weights = 0.5 * (left + right)
    normal_forces = forces - (
        np.sum(forces * tangents, axis=1)[:, None] * tangents
    )
    return float(np.sum(weights * np.sum(normal_forces**2, axis=1)))


class ThetaTrace:
    def __init__(self):
        self.rows = []

    def append(
        self,
        path,
        evaluators,
        metrics,
        *,
        outer_iteration,
        accepted,
        arc_fractions,
        objective_initial,
        objective_final,
        sampled_i_perp_value,
        lbfgs_closures,
        parameter_gradient_norm,
        update_max_amplitude,
        energy_validation_error,
        force_validation_error,
        accepted_update_count,
        compression_coordinate_error=np.nan,
        compression_tangent_error=np.nan,
    ):
        counters = counter_arrays(evaluators)
        internal = counters["joint_evaluations"][1:-1]
        self.rows.append(
            {
                "calls_per_image": int(np.max(internal)),
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
                "outer_iteration": int(outer_iteration),
                "accepted": bool(accepted),
                "collocation_arc_fractions": np.asarray(
                    arc_fractions,
                    dtype=np.float64,
                ),
                "objective_initial": float(objective_initial),
                "objective_final": float(objective_final),
                "sampled_i_perp": float(sampled_i_perp_value),
                "lbfgs_closures": int(lbfgs_closures),
                "parameter_gradient_norm": float(parameter_gradient_norm),
                "update_max_amplitude": float(update_max_amplitude),
                "energy_validation_error": float(energy_validation_error),
                "force_validation_error": float(force_validation_error),
                "accepted_update_count": int(accepted_update_count),
                "compression_coordinate_error": float(
                    compression_coordinate_error
                ),
                "compression_tangent_error": float(
                    compression_tangent_error
                ),
                "model_state": parameter_vector(path),
            }
        )

    def arrays(self):
        float_keys = (
            "mean_calls",
            "dense_i_perp",
            "dense_fmax",
            "dense_barrier",
            "path_length",
            "minimum_speed",
            "spring_force_max_unit",
            "segment_length_cv",
            "objective_initial",
            "objective_final",
            "sampled_i_perp",
            "parameter_gradient_norm",
            "update_max_amplitude",
            "energy_validation_error",
            "force_validation_error",
            "compression_coordinate_error",
            "compression_tangent_error",
        )
        int_keys = (
            "calls_per_image",
            "total_internal_calls",
            "outer_iteration",
            "lbfgs_closures",
            "accepted_update_count",
        )
        payload = {}
        for key in float_keys:
            payload[key] = np.asarray(
                [row[key] for row in self.rows],
                dtype=np.float64,
            )
        for key in int_keys:
            payload[key] = np.asarray(
                [row[key] for row in self.rows],
                dtype=np.int64,
            )
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
            "collocation_arc_fractions",
            "model_state",
        ):
            payload[key] = np.asarray([row[key] for row in self.rows])
        return payload


def gradient_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad.detach() ** 2))
    return float(np.sqrt(total))


def evaluate_subproblem(
    candidate,
    update,
    sample_s,
    weights,
    alpha,
    evaluators,
    *,
    backward,
):
    s_tensor = torch.as_tensor(sample_s, dtype=TORCH_DTYPE)
    weights_tensor = torch.as_tensor(weights, dtype=TORCH_DTYPE)
    points = candidate(s_tensor)
    energies, forces = torch_leps_energy_force(points, create_graph=backward)
    energy_error, force_error = audit_torch_batch(
        evaluators,
        points.detach().cpu().numpy(),
        energies,
        forces,
    )
    amplitudes = update(s_tensor).reshape(-1)
    objective = torch.sum(
        weights_tensor
        * (energies + amplitudes**2 / (2.0 * float(alpha)))
    )
    if backward:
        objective.backward()
    return objective, energies, forces, energy_error, force_error


def run(args):
    root = Path(args.root).expanduser().resolve()
    collocation_count = int(args.collocation_count)
    if collocation_count != 20:
        raise ValueError(
            "This comparison requires exactly 20 internal collocation points"
        )
    if args.alpha <= 0.0:
        raise ValueError("alpha must be positive")

    common_path = load_common_path(root / "inputs" / "common_initial_path.npz")
    current_path = FixedEndpointPathNet(
        width=args.path_width,
        depth=args.path_depth,
        fourier_order=args.path_fourier_order,
        seed=args.seed,
    )
    for parameter in current_path.parameters():
        parameter.requires_grad_(False)
    initial_sample = model_coordinates(
        current_path,
        np.linspace(0.0, 1.0, N_IMAGES),
    )
    if not np.allclose(
        physical_to_flat(initial_sample),
        common_path,
        atol=2e-14,
        rtol=0.0,
    ):
        raise RuntimeError("Theta-NN-IVM does not start from the common path")

    backend = LEPSBackend(label=f"nn_ivm_theta:{args.run_label}")
    evaluators = make_evaluators(
        backend,
        count=collocation_count + 2,
        internal_limit=args.budget,
    )
    initial_fractions = stratified_jitter_fractions(
        collocation_count,
        0,
        jitter=args.collocation_jitter,
        seed=args.sampling_seed,
    )
    for evaluator, point in zip(evaluators, initial_sample):
        query_auditor(evaluator, point)

    saddle = reference_saddle()
    metrics = dense_neural_metrics(
        current_path,
        dense_count=args.diagnostic_points,
    )
    metrics["dense_barrier_error"] = abs(
        metrics["dense_barrier"] - saddle["forward_barrier"]
    )
    trace = ThetaTrace()
    trace.append(
        current_path,
        evaluators,
        metrics,
        outer_iteration=0,
        accepted=True,
        arc_fractions=initial_fractions,
        objective_initial=np.nan,
        objective_final=np.nan,
        sampled_i_perp_value=np.nan,
        lbfgs_closures=0,
        parameter_gradient_norm=np.nan,
        update_max_amplitude=0.0,
        energy_validation_error=0.0,
        force_validation_error=0.0,
        accepted_update_count=0,
        compression_coordinate_error=0.0,
        compression_tangent_error=0.0,
    )

    accepted_updates = []
    stop_reason = "budget"
    consecutive_rejections = 0
    outer_iteration = 0
    while outer_iteration < int(args.max_outer):
        counts = counter_arrays(evaluators)["joint_evaluations"][1:-1]
        if int(np.max(counts)) >= int(args.budget):
            stop_reason = "budget"
            break
        outer_iteration += 1
        fractions = stratified_jitter_fractions(
            collocation_count,
            outer_iteration,
            jitter=args.collocation_jitter,
            seed=args.sampling_seed,
        )
        sample_s = arclength_parameters_for_fractions(
            current_path,
            fractions,
            dense_count=args.arclength_points,
        )
        old_points, old_tangents, _ = normalized_autograd_tangent(
            current_path,
            sample_s,
        )
        endpoints = model_coordinates(current_path, [0.0, 1.0])
        weights = stratified_quadrature_weights(fractions)

        update = ScalarEndpointUpdateNet(
            width=args.width,
            depth=args.depth,
            fourier_order=args.fourier_order,
            seed=args.seed + outer_iteration,
            max_amplitude=args.trust_radius,
        )
        candidate = ProjectedUpdatePath(current_path, update)
        trainable = list(update.parameters())
        optimizer = torch.optim.LBFGS(
            trainable,
            lr=float(args.lbfgs_lr),
            max_iter=int(args.inner_iterations),
            max_eval=int(args.inner_evaluations),
            tolerance_grad=float(args.gradient_tolerance),
            tolerance_change=float(args.change_tolerance),
            history_size=int(args.history_size),
            line_search_fn="strong_wolfe",
        )
        closure_count = 0
        objective_initial = np.nan
        objective_final = np.nan
        final_gradient_norm = np.nan
        energy_error = 0.0
        force_error = 0.0
        budget_hit = False

        def closure():
            nonlocal closure_count, objective_initial
            nonlocal final_gradient_norm, energy_error, force_error
            optimizer.zero_grad(set_to_none=True)
            result = evaluate_subproblem(
                candidate,
                update,
                sample_s,
                weights,
                args.alpha,
                evaluators[1:-1],
                backward=True,
            )
            objective, _, _, e_error, f_error = result
            closure_count += 1
            value = float(objective.detach())
            if not np.isfinite(objective_initial):
                objective_initial = value
            final_gradient_norm = gradient_norm(trainable)
            energy_error = max(energy_error, e_error)
            force_error = max(force_error, f_error)
            return objective

        try:
            optimizer.step(closure)
            optimizer.zero_grad(set_to_none=True)
            final_result = evaluate_subproblem(
                candidate,
                update,
                sample_s,
                weights,
                args.alpha,
                evaluators[1:-1],
                backward=True,
            )
            final_objective, _, final_forces, e_error, f_error = final_result
            closure_count += 1
            objective_final = float(final_objective.detach())
            final_gradient_norm = gradient_norm(trainable)
            energy_error = max(energy_error, e_error)
            force_error = max(force_error, f_error)
        except EvaluationBudgetReached:
            budget_hit = True
            final_forces = None

        accepted = False
        sampled_value = np.nan
        update_amplitude = np.nan
        compression_coordinate_error = np.nan
        compression_tangent_error = np.nan
        compressed_path = None
        if not budget_hit:
            candidate_points, candidate_tangents, _ = (
                normalized_autograd_tangent(candidate, sample_s)
            )
            sampled_value = sampled_i_perp(
                candidate_points,
                candidate_tangents,
                final_forces.detach().cpu().numpy(),
                endpoints,
            )
            with torch.no_grad():
                update_amplitude = float(
                    torch.max(
                        torch.abs(
                            update(
                                torch.as_tensor(
                                    np.linspace(0.0, 1.0, 1025),
                                    dtype=TORCH_DTYPE,
                                )
                            )
                        )
                    )
                )
            probe = model_coordinates(
                candidate,
                np.linspace(0.0, 1.0, int(args.geometry_points)),
            )
            candidate_length = float(
                np.sum(np.linalg.norm(np.diff(probe, axis=0), axis=1))
            )
            geometry_ok = bool(
                np.all(np.isfinite(probe))
                and np.min(probe) > float(args.coordinate_min)
                and np.max(probe) < float(args.coordinate_max)
                and candidate_length < float(args.max_path_length)
            )
            accepted = bool(
                np.isfinite(objective_final)
                and objective_final
                < objective_initial - float(args.acceptance_tolerance)
                and geometry_ok
            )
            if accepted:
                compression_s = np.linspace(
                    0.0,
                    1.0,
                    int(args.compression_points) + 2,
                )[1:-1]
                compression_targets, compression_tangents, _ = (
                    normalized_autograd_tangent(candidate, compression_s)
                )
                compressed_path = FixedEndpointPathNet(
                    width=args.path_width,
                    depth=args.path_depth,
                    fourier_order=args.path_fourier_order,
                    seed=args.seed,
                )
                load_parameter_vector(
                    compressed_path,
                    parameter_vector(current_path),
                )
                fit_path_targets(
                    compressed_path,
                    compression_s,
                    compression_targets,
                    target_tangents=compression_tangents,
                    tangent_weight=args.compression_tangent_weight,
                    smoothness_weight=0.0,
                    adam_steps=args.compression_adam,
                    adam_lr=args.compression_lr,
                    lbfgs_steps=args.compression_lbfgs,
                    polish_lbfgs_steps=0,
                )
                verification_s = np.linspace(
                    0.0,
                    1.0,
                    int(args.compression_verification_points),
                )
                target_coordinates, target_tangents, _ = (
                    normalized_autograd_tangent(candidate, verification_s)
                )
                compressed_coordinates, compressed_tangents, _ = (
                    normalized_autograd_tangent(
                        compressed_path,
                        verification_s,
                    )
                )
                compression_coordinate_error = float(
                    np.max(
                        np.linalg.norm(
                            compressed_coordinates - target_coordinates,
                            axis=1,
                        )
                    )
                )
                compression_tangent_error = float(
                    np.max(
                        np.linalg.norm(
                            compressed_tangents - target_tangents,
                            axis=1,
                        )
                    )
                )
                accepted = bool(
                    compression_coordinate_error
                    <= float(args.compression_coordinate_tolerance)
                    and compression_tangent_error
                    <= float(args.compression_tangent_tolerance)
                )

        if accepted:
            current_path = compressed_path
            for parameter in current_path.parameters():
                parameter.requires_grad_(False)
            accepted_updates.append(parameter_vector(update))
            consecutive_rejections = 0
        else:
            consecutive_rejections += 1

        metrics = dense_neural_metrics(
            current_path,
            dense_count=args.diagnostic_points,
        )
        metrics["dense_barrier_error"] = abs(
            metrics["dense_barrier"] - saddle["forward_barrier"]
        )
        trace.append(
            current_path,
            evaluators,
            metrics,
            outer_iteration=outer_iteration,
            accepted=accepted,
            arc_fractions=fractions,
            objective_initial=objective_initial,
            objective_final=objective_final,
            sampled_i_perp_value=sampled_value,
            lbfgs_closures=closure_count,
            parameter_gradient_norm=final_gradient_norm,
            update_max_amplitude=update_amplitude,
            energy_validation_error=energy_error,
            force_validation_error=force_error,
            accepted_update_count=len(accepted_updates),
            compression_coordinate_error=compression_coordinate_error,
            compression_tangent_error=compression_tangent_error,
        )
        if budget_hit:
            stop_reason = "budget"
            break
        if consecutive_rejections >= int(args.max_rejections):
            stop_reason = "rejected"
            break
        if (
            float(args.dense_iperp_tolerance) > 0.0
            and metrics["dense_i_perp"]
            <= float(args.dense_iperp_tolerance)
        ):
            stop_reason = "dense_i_perp"
            break

    audit = audit_counter_consistency(backend, evaluators)
    if not audit["consistent"]:
        raise RuntimeError(f"Theta-NN-IVM counter audit failed: {audit}")

    configuration = {
        "method": "NN-IVM-theta",
        "run_label": args.run_label,
        "subproblem": (
            "direct minimization over theta of stratified quadrature of "
            "U(phi_k + P_perp,k delta_theta) + "
            "|P_perp,k delta_theta|^2/(2 alpha)"
        ),
        "alpha": float(args.alpha),
        "width": int(args.width),
        "depth": int(args.depth),
        "fourier_order": int(args.fourier_order),
        "path_width": int(args.path_width),
        "path_depth": int(args.path_depth),
        "path_fourier_order": int(args.path_fourier_order),
        "seed": int(args.seed),
        "trust_radius": float(args.trust_radius),
        "optimizer": "torch.optim.LBFGS(strong_wolfe)",
        "inner_iterations": int(args.inner_iterations),
        "inner_evaluations": int(args.inner_evaluations),
        "gradient_tolerance": float(args.gradient_tolerance),
        "change_tolerance": float(args.change_tolerance),
        "history_size": int(args.history_size),
        "collocation_count": collocation_count,
        "collocation_jitter": float(args.collocation_jitter),
        "sampling_seed": int(args.sampling_seed),
        "sampling": (
            "20 reproducible stratified equal-arclength samples per "
            "subproblem; no dense optimization samples"
        ),
        "tangent": "normalized torch autograd dphi/ds",
        "post_subproblem_fit": (
            "no pointwise subproblem targets; only a force-free continuous "
            "representation compression of the already optimized trial"
        ),
        "representation_compression": (
            "accepted projected neural trial is fitted into a fixed-depth "
            "endpoint-constrained NN without LEPS calls; both coordinate "
            "and autograd-tangent errors are gated"
        ),
        "compression_coordinate_tolerance": float(
            args.compression_coordinate_tolerance
        ),
        "compression_tangent_tolerance": float(
            args.compression_tangent_tolerance
        ),
        "reparameterization": "none",
        "budget": int(args.budget),
    }
    arrays = trace.arrays()
    update_parameter_count = (
        len(parameter_vector(
            ScalarEndpointUpdateNet(
                width=args.width,
                depth=args.depth,
                fourier_order=args.fourier_order,
                seed=args.seed + 1,
                max_amplitude=args.trust_radius,
            )
        ))
    )
    final_updates = (
        np.asarray(accepted_updates, dtype=np.float64)
        if accepted_updates
        else np.empty((0, update_parameter_count), dtype=np.float64)
    )
    arrays.update(
        {
            "method": np.asarray("NN-IVM-theta"),
            "run_label": np.asarray(args.run_label),
            "alpha": np.asarray(args.alpha),
            "initial_path": np.asarray(common_path, dtype=np.float64),
            "initial_path_sha256": np.asarray(
                hashlib.sha256(
                    np.ascontiguousarray(common_path).tobytes()
                ).hexdigest()
            ),
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
                "20-point reproducible stratified equal-arclength jitter"
            ),
            "collocation_count": np.asarray(collocation_count),
            "collocation_jitter": np.asarray(args.collocation_jitter),
            "sampling_seed": np.asarray(args.sampling_seed),
            "diagnostic_points": np.asarray(args.diagnostic_points),
            "final_update_parameters": final_updates,
        }
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    atomic_save_npz(output, **arrays)
    write_json(
        output.with_suffix(".audit.json"),
        {
            "configuration": configuration,
            "configuration_sha256": configuration_hash(configuration),
            "stop_reason": stop_reason,
            "counter_audit": audit,
            "final": {
                "calls_per_image": int(arrays["calls_per_image"][-1]),
                "mean_calls": float(arrays["mean_calls"][-1]),
                "total_internal_calls": int(
                    arrays["total_internal_calls"][-1]
                ),
                "accepted_updates": int(len(accepted_updates)),
                "dense_i_perp": float(arrays["dense_i_perp"][-1]),
                "dense_barrier": float(arrays["dense_barrier"][-1]),
                "dense_barrier_error": float(
                    arrays["dense_barrier_error"][-1]
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
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--fourier-order", type=int, default=6)
    parser.add_argument("--path-width", type=int, default=64)
    parser.add_argument("--path-depth", type=int, default=3)
    parser.add_argument("--path-fourier-order", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--trust-radius", type=float, default=1.0)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--inner-iterations", type=int, default=40)
    parser.add_argument("--inner-evaluations", type=int, default=60)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-7)
    parser.add_argument("--change-tolerance", type=float, default=1e-10)
    parser.add_argument("--history-size", type=int, default=30)
    parser.add_argument("--acceptance-tolerance", type=float, default=1e-10)
    parser.add_argument("--collocation-count", type=int, default=20)
    parser.add_argument("--collocation-jitter", type=float, default=0.45)
    parser.add_argument("--sampling-seed", type=int, default=20260726)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--max-outer", type=int, default=20)
    parser.add_argument("--max-rejections", type=int, default=3)
    parser.add_argument("--dense-iperp-tolerance", type=float, default=0.0)
    parser.add_argument("--arclength-points", type=int, default=2049)
    parser.add_argument("--diagnostic-points", type=int, default=2049)
    parser.add_argument("--geometry-points", type=int, default=1025)
    parser.add_argument("--coordinate-min", type=float, default=0.25)
    parser.add_argument("--coordinate-max", type=float, default=8.5)
    parser.add_argument("--max-path-length", type=float, default=18.0)
    parser.add_argument("--compression-points", type=int, default=255)
    parser.add_argument("--compression-adam", type=int, default=300)
    parser.add_argument("--compression-lr", type=float, default=1e-2)
    parser.add_argument("--compression-lbfgs", type=int, default=100)
    parser.add_argument(
        "--compression-tangent-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--compression-verification-points",
        type=int,
        default=513,
    )
    parser.add_argument(
        "--compression-coordinate-tolerance",
        type=float,
        default=2e-3,
    )
    parser.add_argument(
        "--compression-tangent-tolerance",
        type=float,
        default=2e-2,
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
