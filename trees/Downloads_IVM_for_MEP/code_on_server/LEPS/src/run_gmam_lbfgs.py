"""Audited neural gMAM with finite-difference Hessians and Armijo L-BFGS."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from continuous_paths import (
    FixedEndpointPathNet,
    TORCH_DTYPE,
    coordinates_and_derivatives,
    dense_neural_metrics,
    equal_arclength_parameters,
    load_parameter_vector,
    model_coordinates,
    parameter_vector,
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
from run_gmam_nn import (
    GMAMTrace,
    configuration_hash,
    finite_difference_gradient_norm,
    query_gradient,
)


def _parameter_gradient_vector(model):
    return (
        torch.nn.utils.parameters_to_vector(
            [
                torch.zeros_like(parameter)
                if parameter.grad is None
                else parameter.grad
                for parameter in model.parameters()
            ]
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=True)
    )


def gmam_value_and_gradient(
    model,
    evaluators,
    all_s,
    *,
    hessian_step,
    gradient_norm_floor,
    speed_floor,
):
    """Evaluate the action and its parameter gradient with audited FD Hessians."""
    s_tensor, coordinates, first = coordinates_and_derivatives(
        model,
        all_s,
        create_graph=True,
    )
    points = coordinates.detach().cpu().numpy()
    count = len(evaluators)
    norms = np.zeros(count, dtype=np.float64)
    spatial_derivatives = np.zeros((count, 2), dtype=np.float64)
    symmetry_defects = []

    # Endpoint coordinates are fixed.  Their base forces enter the trapezoid
    # value, while no endpoint Hessian is needed for parameter differentiation.
    for index in (0, count - 1):
        _, gradient = query_gradient(evaluators[index], points[index])
        norms[index] = np.sqrt(
            np.dot(gradient, gradient) + gradient_norm_floor**2
        )

    for index, (evaluator, point) in enumerate(
        zip(evaluators[1:-1], points[1:-1]),
        start=1,
    ):
        (
            norms[index],
            spatial_derivatives[index],
            _,
            symmetry,
        ) = finite_difference_gradient_norm(
            evaluator,
            point,
            step=hessian_step,
            norm_floor=gradient_norm_floor,
        )
        symmetry_defects.append(symmetry)

    norm_tensor = torch.as_tensor(norms, dtype=TORCH_DTYPE)
    spatial_tensor = torch.as_tensor(
        spatial_derivatives,
        dtype=TORCH_DTYPE,
    )
    norm_surrogate = norm_tensor + torch.sum(
        spatial_tensor * (coordinates - coordinates.detach()),
        dim=1,
    )
    speed = torch.sqrt(torch.sum(first**2, dim=1) + speed_floor**2)
    objective = torch.trapz(norm_surrogate * speed, s_tensor)
    model.zero_grad(set_to_none=True)
    objective.backward()
    return (
        float(objective.detach()),
        _parameter_gradient_vector(model),
        float(np.max(symmetry_defects)) if symmetry_defects else 0.0,
    )


def gmam_value_only(
    model,
    evaluators,
    all_s,
    *,
    gradient_norm_floor,
    speed_floor,
):
    """Action-only line-search evaluation; each new point costs one force call."""
    _, coordinates, first = coordinates_and_derivatives(model, all_s)
    points = coordinates.detach().cpu().numpy()
    speed = np.sqrt(
        np.sum(first.detach().cpu().numpy() ** 2, axis=1) + speed_floor**2
    )
    norms = np.empty(len(evaluators), dtype=np.float64)
    for index, (evaluator, point) in enumerate(zip(evaluators, points)):
        _, gradient = query_gradient(evaluator, point)
        norms[index] = np.sqrt(
            np.dot(gradient, gradient) + gradient_norm_floor**2
        )
    return float(np.trapz(norms * speed, np.asarray(all_s)))


def lbfgs_direction(gradient, history):
    """Return a descent direction from positive-curvature L-BFGS pairs."""
    gradient = np.asarray(gradient, dtype=np.float64)
    q = gradient.copy()
    coefficients = []
    for displacement, gradient_change in reversed(history):
        inverse_curvature = 1.0 / float(
            np.dot(gradient_change, displacement)
        )
        coefficient = inverse_curvature * float(np.dot(displacement, q))
        q -= coefficient * gradient_change
        coefficients.append(
            (
                coefficient,
                inverse_curvature,
                displacement,
                gradient_change,
            )
        )
    if history:
        displacement, gradient_change = history[-1]
        scale = float(
            np.dot(displacement, gradient_change)
            / np.dot(gradient_change, gradient_change)
        )
        scale = float(np.clip(scale, 1e-8, 1e8))
    else:
        scale = 1.0
    result = scale * q
    for coefficient, inverse_curvature, displacement, gradient_change in (
        reversed(coefficients)
    ):
        beta = inverse_curvature * float(
            np.dot(gradient_change, result)
        )
        result += (coefficient - beta) * displacement
    direction = -result
    if (
        not np.all(np.isfinite(direction))
        or float(np.dot(direction, gradient)) >= -1e-18
    ):
        direction = -gradient
    return direction


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
    if not np.allclose(
        physical_to_flat(initial_sample),
        common_path,
        atol=2e-14,
        rtol=0.0,
    ):
        raise RuntimeError("gMAM does not start from the common initial path")

    collocation_count = int(args.collocation_count)
    backend = LEPSBackend(label=f"gmam_lbfgs:{args.run_label}")
    evaluators = make_evaluators(
        backend,
        count=collocation_count + 2,
        internal_limit=args.budget,
    )
    initial_s = equal_arclength_parameters(
        model,
        collocation_count,
        dense_count=args.arclength_points,
    )
    initial_all_s = np.concatenate(([0.0], initial_s, [1.0]))
    for evaluator, point in zip(
        evaluators,
        model_coordinates(model, initial_all_s),
    ):
        query_gradient(evaluator, point)

    trace = GMAMTrace()
    metrics = dense_neural_metrics(
        model,
        dense_count=args.diagnostic_points,
    )
    trace.append(
        model,
        evaluators,
        metrics,
        outer_iteration=0,
        objective=np.nan,
        parameter_gradient_norm=np.nan,
        hessian_symmetry_max=np.nan,
        learning_rate=0.0,
        accepted=True,
    )

    history = []
    previous_vector = None
    previous_gradient = None
    stop_reason = "budget"
    outer_iteration = 0
    while outer_iteration < int(args.max_outer):
        counts = counter_arrays(evaluators)["joint_evaluations"][1:-1]
        if int(np.max(counts)) + 4 > int(args.budget):
            stop_reason = "budget"
            break
        outer_iteration += 1
        internal_s = equal_arclength_parameters(
            model,
            collocation_count,
            dense_count=args.arclength_points,
        )
        all_s = np.concatenate(([0.0], internal_s, [1.0]))
        try:
            objective, gradient, symmetry = gmam_value_and_gradient(
                model,
                evaluators,
                all_s,
                hessian_step=args.hessian_step,
                gradient_norm_floor=args.gradient_norm_floor,
                speed_floor=args.speed_floor,
            )
        except EvaluationBudgetReached:
            stop_reason = "budget"
            break

        current_vector = parameter_vector(model)
        if previous_vector is not None:
            displacement = current_vector - previous_vector
            gradient_change = gradient - previous_gradient
            curvature = float(np.dot(displacement, gradient_change))
            if curvature > 1e-8 * np.linalg.norm(displacement) * max(
                np.linalg.norm(gradient_change),
                1e-30,
            ):
                history.append((displacement.copy(), gradient_change.copy()))
                if len(history) > int(args.history_size):
                    history.pop(0)

        direction = lbfgs_direction(gradient, history)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > float(args.max_direction_norm):
            direction *= float(args.max_direction_norm) / direction_norm
        directional_derivative = float(np.dot(gradient, direction))
        if directional_derivative >= 0.0:
            direction = -gradient
            directional_derivative = -float(np.dot(gradient, gradient))

        accepted = False
        accepted_step = 0.0
        step = float(args.initial_step)
        budget_hit = False
        for _ in range(int(args.max_backtracks)):
            load_parameter_vector(
                model,
                current_vector + step * direction,
            )
            probe = model_coordinates(model, np.linspace(0.0, 1.0, 1025))
            geometry_ok = bool(
                np.all(np.isfinite(probe))
                and np.min(probe) > 0.25
                and np.max(probe) < 8.5
                and np.sum(np.linalg.norm(np.diff(probe, axis=0), axis=1))
                < float(args.max_path_length)
            )
            if geometry_ok:
                try:
                    trial_objective = gmam_value_only(
                        model,
                        evaluators,
                        all_s,
                        gradient_norm_floor=args.gradient_norm_floor,
                        speed_floor=args.speed_floor,
                    )
                except EvaluationBudgetReached:
                    budget_hit = True
                    break
                if trial_objective <= (
                    objective
                    + float(args.armijo_c1)
                    * step
                    * directional_derivative
                ):
                    accepted = True
                    accepted_step = step
                    break
            step *= 0.5

        if not accepted:
            load_parameter_vector(model, current_vector)
        metrics = dense_neural_metrics(
            model,
            dense_count=args.diagnostic_points,
        )
        trace.append(
            model,
            evaluators,
            metrics,
            outer_iteration=outer_iteration,
            objective=objective,
            parameter_gradient_norm=float(np.linalg.norm(gradient)),
            hessian_symmetry_max=symmetry,
            learning_rate=accepted_step,
            accepted=accepted,
        )
        if budget_hit:
            stop_reason = "budget"
            break
        if not accepted:
            stop_reason = "line_search"
            break
        previous_vector = current_vector
        previous_gradient = gradient.copy()
        if metrics["dense_i_perp"] <= float(args.iperp_tolerance):
            stop_reason = "dense_i_perp"
            break
    else:
        stop_reason = "outer_limit"

    audit = audit_counter_consistency(backend, evaluators)
    if not audit["consistent"]:
        raise RuntimeError(f"gMAM counter audit failed: {audit}")
    saddle = reference_saddle()
    configuration = {
        "method": "NN-gMAM-FD-LBFGS",
        "run_label": args.run_label,
        "width": int(args.width),
        "depth": int(args.depth),
        "fourier_order": int(args.fourier_order),
        "seed": int(args.seed),
        "collocation_count": collocation_count,
        "hessian_step": float(args.hessian_step),
        "history_size": int(args.history_size),
        "initial_step": float(args.initial_step),
        "max_direction_norm": float(args.max_direction_norm),
        "armijo_c1": float(args.armijo_c1),
        "budget": int(args.budget),
        "objective": "integral |grad U(phi)| |phi'(s)| ds",
        "hessian": "2D central finite difference; four displaced calls",
        "optimizer": "parameter-space L-BFGS with audited Armijo line search",
    }
    arrays = trace.arrays()
    arrays.update(
        {
            "method": np.asarray("NN-gMAM-FD-LBFGS"),
            "run_label": np.asarray(args.run_label),
            "width": np.asarray(args.width),
            "depth": np.asarray(args.depth),
            "fourier_order": np.asarray(args.fourier_order),
            "seed": np.asarray(args.seed),
            "collocation_count": np.asarray(collocation_count),
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
            "hessian_definition": np.asarray(
                "2D central finite difference, 4 displaced force calls"
            ),
            "sampling_definition": np.asarray(
                "equal-arclength collocation without geometric "
                "reparameterization"
            ),
            "diagnostic_points": np.asarray(args.diagnostic_points),
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
                "equivalent_20_image_calls": float(
                    arrays["total_internal_calls"][-1] / 20.0
                ),
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
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--fourier-order", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--collocation-count", type=int, default=64)
    parser.add_argument("--hessian-step", type=float, default=1e-4)
    parser.add_argument("--gradient-norm-floor", type=float, default=1e-10)
    parser.add_argument("--speed-floor", type=float, default=1e-10)
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument("--initial-step", type=float, default=1.0)
    parser.add_argument("--max-direction-norm", type=float, default=0.1)
    parser.add_argument("--armijo-c1", type=float, default=1e-4)
    parser.add_argument("--max-backtracks", type=int, default=16)
    parser.add_argument("--max-path-length", type=float, default=30.0)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--max-outer", type=int, default=1000)
    parser.add_argument("--iperp-tolerance", type=float, default=0.0)
    parser.add_argument("--arclength-points", type=int, default=4097)
    parser.add_argument("--diagnostic-points", type=int, default=4097)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
