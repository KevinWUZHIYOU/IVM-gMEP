"""Adaptive-AL neural IVM with direct theta-space subminimization.

The path-space proximal problem is discretized by exactly 20 reproducibly
jittered equal-arclength collocation points.  The trainable variables are all
parameters theta of one fixed-endpoint neural path.  An augmented Lagrangian
enforces tau_k dot (phi_theta - phi_k) = 0 at the collocation points, so the
accepted update is normal to the old path without a post-subproblem fit.

Unlike the fixed-cycle reference implementation, this variant keeps the
L-BFGS curvature history across closely related multiplier cycles and grows
the penalty only when the constraint residual fails a prescribed reduction
test.  The rule is fixed before a run and does not use barrier information.
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
    arclength_parameters_for_fractions,
    dense_neural_metrics,
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
DTYPE = torch.float64


def configuration_hash(configuration):
    encoded = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def quadrature_weights(fractions):
    fractions = np.asarray(fractions, dtype=np.float64)
    boundaries = np.empty(fractions.size + 1, dtype=np.float64)
    boundaries[0] = 0.0
    boundaries[-1] = 1.0
    boundaries[1:-1] = 0.5 * (fractions[:-1] + fractions[1:])
    return np.diff(boundaries)


def scheduled_jitter(base_jitter, iteration, schedule):
    base_jitter = float(base_jitter)
    iteration = int(iteration)
    if schedule == "constant" or iteration <= 1:
        return base_jitter
    if schedule == "inverse_sqrt":
        return base_jitter / np.sqrt(float(iteration))
    raise ValueError(f"Unsupported jitter schedule: {schedule}")


def query_auditor(evaluator, point):
    energy, force_flat = evaluator.query(physical_to_flat(point), "both")
    return float(energy), np.asarray(force_flat, dtype=np.float64)[[0, 3]]


class AuditedJointEnergy(torch.autograd.Function):
    """One cached joint energy-force oracle call with force-based backward."""

    @staticmethod
    def forward(ctx, points, evaluators):
        energies = []
        forces = []
        for evaluator, point in zip(
            evaluators,
            points.detach().cpu().numpy(),
        ):
            energy, force = query_auditor(evaluator, point)
            energies.append(energy)
            forces.append(force)
        energy_tensor = torch.as_tensor(
            np.asarray(energies, dtype=np.float64),
            dtype=points.dtype,
            device=points.device,
        )
        force_tensor = torch.as_tensor(
            np.asarray(forces, dtype=np.float64),
            dtype=points.dtype,
            device=points.device,
        )
        ctx.save_for_backward(force_tensor)
        ctx.mark_non_differentiable(force_tensor)
        return energy_tensor, force_tensor

    @staticmethod
    def backward(ctx, energy_gradient, force_gradient):
        del force_gradient
        (forces,) = ctx.saved_tensors
        coordinate_gradient = -energy_gradient[:, None] * forces
        return coordinate_gradient, None


def audited_joint_energy_force(points, evaluators):
    return AuditedJointEnergy.apply(points, evaluators)


def sampled_i_perp(points, tangents, forces, endpoints):
    full = np.vstack((endpoints[0], points, endpoints[1]))
    weights = 0.5 * (
        np.linalg.norm(points - full[:-2], axis=1)
        + np.linalg.norm(full[2:] - points, axis=1)
    )
    perpendicular = forces - (
        np.sum(forces * tangents, axis=1)[:, None] * tangents
    )
    return float(np.sum(weights * np.sum(perpendicular**2, axis=1)))


def gradient_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad.detach() ** 2))
    return float(np.sqrt(total))


class Trace:
    def __init__(self):
        self.rows = []

    def append(self, path, evaluators, metrics, **fields):
        counters = counter_arrays(evaluators)
        internal = counters["joint_evaluations"][1:-1]
        row = {
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
            "model_state": parameter_vector(path),
        }
        row.update(fields)
        self.rows.append(row)

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
            "constraint_max",
            "sampled_i_perp",
            "parameter_gradient_norm",
            "energy_validation_error",
            "force_validation_error",
            "al_penalty_final",
            "multiplier_max",
        )
        int_keys = (
            "calls_per_image",
            "total_internal_calls",
            "outer_iteration",
            "lbfgs_closures",
            "al_cycles",
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
            "model_state",
            "collocation_arc_fractions",
        ):
            payload[key] = np.asarray([row[key] for row in self.rows])
        return payload


def run(args):
    root = Path(args.root).expanduser().resolve()
    if int(args.collocation_count) != 20:
        raise ValueError("Exactly 20 internal collocation points are required")
    if args.alpha <= 0.0 or args.al_penalty <= 0.0:
        raise ValueError("alpha and al_penalty must be positive")
    if not 0.0 < float(args.constraint_reduction_target) < 1.0:
        raise ValueError("constraint_reduction_target must lie in (0, 1)")
    if float(args.al_penalty_max) < float(args.al_penalty):
        raise ValueError("al_penalty_max must not be below al_penalty")

    common_path = load_common_path(root / "inputs" / "common_initial_path.npz")
    current = FixedEndpointPathNet(
        width=args.width,
        depth=args.depth,
        fourier_order=args.fourier_order,
        seed=args.seed,
    )
    initial = model_coordinates(current, np.linspace(0.0, 1.0, N_IMAGES))
    if not np.allclose(
        physical_to_flat(initial),
        common_path,
        atol=2e-14,
        rtol=0.0,
    ):
        raise RuntimeError("Theta-AL NN-IVM does not start from common path")
    for parameter in current.parameters():
        parameter.requires_grad_(False)

    backend = LEPSBackend(label=f"nn_ivm_theta_al:{args.run_label}")
    evaluators = make_evaluators(
        backend,
        count=22,
        internal_limit=args.budget,
    )
    initial_fractions = stratified_jitter_fractions(
        20,
        0,
        jitter=args.collocation_jitter,
        seed=args.sampling_seed,
    )
    for evaluator, point in zip(evaluators, initial):
        query_auditor(evaluator, point)

    saddle = reference_saddle()
    metrics = dense_neural_metrics(
        current,
        dense_count=args.diagnostic_points,
    )
    trace = Trace()
    trace.append(
        current,
        evaluators,
        metrics,
        outer_iteration=0,
        accepted=True,
        collocation_arc_fractions=initial_fractions,
        objective_initial=np.nan,
        objective_final=np.nan,
        constraint_max=0.0,
        sampled_i_perp=np.nan,
        parameter_gradient_norm=np.nan,
        lbfgs_closures=0,
        al_cycles=0,
        energy_validation_error=0.0,
        force_validation_error=0.0,
        al_penalty_final=float(args.al_penalty),
        multiplier_max=0.0,
    )

    stop_reason = "budget"
    outer_iteration = 0
    consecutive_rejections = 0
    while outer_iteration < int(args.max_outer):
        counts = counter_arrays(evaluators)["joint_evaluations"][1:-1]
        if int(np.max(counts)) >= int(args.budget):
            stop_reason = "budget"
            break
        outer_iteration += 1
        fractions = stratified_jitter_fractions(
            20,
            outer_iteration,
            jitter=scheduled_jitter(
                args.collocation_jitter,
                outer_iteration,
                args.jitter_schedule,
            ),
            seed=args.sampling_seed,
        )
        sample_s = arclength_parameters_for_fractions(
            current,
            fractions,
            dense_count=args.arclength_points,
        )
        old_points, old_tangents, _ = normalized_autograd_tangent(
            current,
            sample_s,
        )
        endpoints = model_coordinates(current, [0.0, 1.0])
        weights = quadrature_weights(fractions)
        s_tensor = torch.as_tensor(sample_s, dtype=DTYPE)
        old_tensor = torch.as_tensor(old_points, dtype=DTYPE)
        tangent_tensor = torch.as_tensor(old_tangents, dtype=DTYPE)
        weight_tensor = torch.as_tensor(weights, dtype=DTYPE)
        multipliers = torch.zeros(20, dtype=DTYPE)
        penalty = float(args.al_penalty)

        candidate = FixedEndpointPathNet(
            width=args.width,
            depth=args.depth,
            fourier_order=args.fourier_order,
            seed=args.seed,
        )
        load_parameter_vector(candidate, parameter_vector(current))
        parameters = list(candidate.parameters())
        closure_count = 0
        al_cycles_done = 0
        objective_initial = np.nan
        objective_final = np.nan
        constraint_max = np.inf
        final_gradient = np.nan
        energy_error = 0.0
        force_error = 0.0
        budget_hit = False
        final_forces = None
        previous_constraint_max = np.inf

        def evaluate(backward, augmented):
            nonlocal energy_error, force_error, objective_initial
            points = candidate(s_tensor)
            energies, forces = audited_joint_energy_force(
                points,
                evaluators[1:-1],
            )
            displacement = points - old_tensor
            tangential = torch.sum(
                displacement * tangent_tensor,
                dim=1,
            )
            normal = displacement - tangential[:, None] * tangent_tensor
            physical_terms = (
                energies
                + torch.sum(normal**2, dim=1) / (2.0 * float(args.alpha))
            )
            physical_objective = torch.sum(weight_tensor * physical_terms)
            if not np.isfinite(objective_initial):
                objective_initial = float(physical_objective.detach())
            if augmented:
                objective = physical_objective + torch.sum(
                    weight_tensor
                    * (
                        multipliers * tangential
                        + 0.5 * penalty * tangential**2
                    )
                )
            else:
                objective = physical_objective
            if backward:
                objective.backward()
            return objective, physical_objective, tangential, forces

        optimizer = torch.optim.LBFGS(
            parameters,
            lr=float(args.lbfgs_lr),
            max_iter=int(args.inner_iterations),
            max_eval=int(args.inner_evaluations),
            tolerance_grad=float(args.gradient_tolerance),
            tolerance_change=float(args.change_tolerance),
            history_size=int(args.history_size),
            line_search_fn="strong_wolfe",
        )

        def closure():
            nonlocal closure_count, final_gradient
            optimizer.zero_grad(set_to_none=True)
            objective, _, _, _ = evaluate(True, True)
            closure_count += 1
            final_gradient = gradient_norm(parameters)
            return objective

        try:
            for al_cycle in range(int(args.al_cycles)):
                al_cycles_done = al_cycle + 1
                if al_cycle > 0 and not args.reuse_lbfgs_history:
                    optimizer = torch.optim.LBFGS(
                        parameters,
                        lr=float(args.lbfgs_lr),
                        max_iter=int(args.inner_iterations),
                        max_eval=int(args.inner_evaluations),
                        tolerance_grad=float(args.gradient_tolerance),
                        tolerance_change=float(args.change_tolerance),
                        history_size=int(args.history_size),
                        line_search_fn="strong_wolfe",
                    )
                optimizer.step(closure)
                optimizer.zero_grad(set_to_none=True)
                _, physical, constraints, final_forces = evaluate(
                    True,
                    False,
                )
                closure_count += 1
                objective_final = float(physical.detach())
                constraint_max = float(
                    torch.max(torch.abs(constraints)).detach()
                )
                final_gradient = gradient_norm(parameters)
                if constraint_max <= float(args.constraint_tolerance):
                    break
                multipliers = (
                    multipliers + penalty * constraints.detach()
                )
                if (
                    not args.adaptive_penalty
                    or constraint_max
                    > (
                        float(args.constraint_reduction_target)
                        * previous_constraint_max
                    )
                ):
                    penalty = min(
                        penalty * float(args.al_penalty_growth),
                        float(args.al_penalty_max),
                    )
                    if args.reset_history_on_penalty_growth:
                        optimizer.state.clear()
                previous_constraint_max = constraint_max
        except EvaluationBudgetReached:
            budget_hit = True

        accepted = False
        sampled_value = np.nan
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
            probe = model_coordinates(
                candidate,
                np.linspace(0.0, 1.0, int(args.geometry_points)),
            )
            length = float(
                np.sum(np.linalg.norm(np.diff(probe, axis=0), axis=1))
            )
            geometry_ok = bool(
                np.all(np.isfinite(probe))
                and np.min(probe) > float(args.coordinate_min)
                and np.max(probe) < float(args.coordinate_max)
                and length < float(args.max_path_length)
            )
            accepted = bool(
                objective_final
                < objective_initial - float(args.acceptance_tolerance)
                and constraint_max <= float(args.constraint_tolerance)
                and geometry_ok
            )

        if accepted:
            for parameter in candidate.parameters():
                parameter.requires_grad_(False)
            current = candidate
            consecutive_rejections = 0
        else:
            consecutive_rejections += 1

        metrics = dense_neural_metrics(
            current,
            dense_count=args.diagnostic_points,
        )
        trace.append(
            current,
            evaluators,
            metrics,
            outer_iteration=outer_iteration,
            accepted=accepted,
            collocation_arc_fractions=fractions,
            objective_initial=objective_initial,
            objective_final=objective_final,
            constraint_max=constraint_max,
            sampled_i_perp=sampled_value,
            parameter_gradient_norm=final_gradient,
            lbfgs_closures=closure_count,
            al_cycles=al_cycles_done,
            energy_validation_error=energy_error,
            force_validation_error=force_error,
            al_penalty_final=penalty,
            multiplier_max=float(torch.max(torch.abs(multipliers))),
        )
        if budget_hit:
            stop_reason = "budget"
            break
        if consecutive_rejections >= int(args.max_rejections):
            stop_reason = "rejected"
            break

    audit = audit_counter_consistency(backend, evaluators)
    if not audit["consistent"]:
        raise RuntimeError(f"Theta-AL NN-IVM counter audit failed: {audit}")

    configuration = {
        "method": "NN-IVM-theta-adaptive-AL",
        "run_label": args.run_label,
        "subproblem": (
            "minimize over all neural path parameters theta: "
            "sum_j w_j [U(phi_theta) + "
            "|P_perp,k(phi_theta-phi_k)|^2/(2 alpha)], subject to "
            "tau_k dot (phi_theta-phi_k)=0"
        ),
        "constraint_solver": "adaptive augmented Lagrangian",
        "alpha": float(args.alpha),
        "al_penalty": float(args.al_penalty),
        "al_penalty_growth": float(args.al_penalty_growth),
        "al_cycles": int(args.al_cycles),
        "adaptive_penalty": bool(args.adaptive_penalty),
        "constraint_reduction_target": float(
            args.constraint_reduction_target
        ),
        "al_penalty_max": float(args.al_penalty_max),
        "reuse_lbfgs_history": bool(args.reuse_lbfgs_history),
        "reset_history_on_penalty_growth": bool(
            args.reset_history_on_penalty_growth
        ),
        "constraint_tolerance": float(args.constraint_tolerance),
        "optimizer": "torch.optim.LBFGS(strong_wolfe)",
        "joint_oracle": (
            "one run-local historically cached joint energy-force query; "
            "autograd backward uses the cached force as -grad(U)"
        ),
        "energy_hessian": "not formed or finite-differenced",
        "width": int(args.width),
        "depth": int(args.depth),
        "fourier_order": int(args.fourier_order),
        "seed": int(args.seed),
        "inner_iterations": int(args.inner_iterations),
        "inner_evaluations": int(args.inner_evaluations),
        "gradient_tolerance": float(args.gradient_tolerance),
        "change_tolerance": float(args.change_tolerance),
        "history_size": int(args.history_size),
        "collocation_count": 20,
        "collocation_jitter": float(args.collocation_jitter),
        "jitter_schedule": str(args.jitter_schedule),
        "sampling_seed": int(args.sampling_seed),
        "post_subproblem_fit": "none",
        "reparameterization": "none",
        "tangent": "normalized torch autograd dphi/ds",
        "budget": int(args.budget),
    }
    arrays = trace.arrays()
    arrays.update(
        {
            "method": np.asarray("NN-IVM-theta-adaptive-AL"),
            "run_label": np.asarray(args.run_label),
            "alpha": np.asarray(args.alpha),
            "width": np.asarray(args.width),
            "depth": np.asarray(args.depth),
            "fourier_order": np.asarray(args.fourier_order),
            "seed": np.asarray(args.seed),
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
            "collocation_count": np.asarray(20),
            "collocation_jitter": np.asarray(args.collocation_jitter),
            "jitter_schedule": np.asarray(args.jitter_schedule),
            "sampling_seed": np.asarray(args.sampling_seed),
            "diagnostic_points": np.asarray(args.diagnostic_points),
            "joint_oracle_definition": np.asarray(
                "single cached joint energy-force query with force-based "
                "custom-autograd backward"
            ),
        }
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    atomic_save_npz(output, **arrays)
    accepted_indices = np.flatnonzero(arrays["accepted"])
    final_accepted_index = int(accepted_indices[-1])
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
                "dense_i_perp": float(arrays["dense_i_perp"][-1]),
                "dense_barrier": float(arrays["dense_barrier"][-1]),
                "dense_barrier_error": float(
                    arrays["dense_barrier_error"][-1]
                ),
                "constraint_max": float(
                    arrays["constraint_max"][final_accepted_index]
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
    parser.add_argument("--fourier-order", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--inner-iterations", type=int, default=20)
    parser.add_argument("--inner-evaluations", type=int, default=30)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-7)
    parser.add_argument("--change-tolerance", type=float, default=1e-10)
    parser.add_argument("--history-size", type=int, default=30)
    parser.add_argument("--al-penalty", type=float, default=100.0)
    parser.add_argument("--al-penalty-growth", type=float, default=10.0)
    parser.add_argument("--al-penalty-max", type=float, default=1e6)
    parser.add_argument("--al-cycles", type=int, default=3)
    parser.add_argument(
        "--constraint-reduction-target",
        type=float,
        default=0.5,
    )
    parser.set_defaults(
        adaptive_penalty=True,
        reuse_lbfgs_history=True,
        reset_history_on_penalty_growth=True,
    )
    parser.add_argument(
        "--no-adaptive-penalty",
        dest="adaptive_penalty",
        action="store_false",
    )
    parser.add_argument(
        "--no-reuse-lbfgs-history",
        dest="reuse_lbfgs_history",
        action="store_false",
    )
    parser.add_argument(
        "--keep-history-on-penalty-growth",
        dest="reset_history_on_penalty_growth",
        action="store_false",
    )
    parser.add_argument("--constraint-tolerance", type=float, default=1e-3)
    parser.add_argument("--acceptance-tolerance", type=float, default=1e-10)
    parser.add_argument("--collocation-count", type=int, default=20)
    parser.add_argument("--collocation-jitter", type=float, default=0.45)
    parser.add_argument(
        "--jitter-schedule",
        choices=("constant", "inverse_sqrt"),
        default="constant",
    )
    parser.add_argument("--sampling-seed", type=int, default=20260726)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--max-outer", type=int, default=30)
    parser.add_argument("--max-rejections", type=int, default=3)
    parser.add_argument("--arclength-points", type=int, default=2049)
    parser.add_argument("--diagnostic-points", type=int, default=2049)
    parser.add_argument("--geometry-points", type=int, default=1025)
    parser.add_argument("--coordinate-min", type=float, default=0.25)
    parser.add_argument("--coordinate-max", type=float, default=8.5)
    parser.add_argument("--max-path-length", type=float, default=18.0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
