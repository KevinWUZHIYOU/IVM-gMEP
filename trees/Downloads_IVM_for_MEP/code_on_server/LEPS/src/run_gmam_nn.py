"""Neural-path gMAM baseline with explicitly counted finite-difference Hessians.

For a gradient system and fixed endpoints, minimizing

    integral |grad U(phi)| |phi'(s)| ds

is equivalent to the geometric minimum-action objective up to the constant
endpoint energy term.  Its path derivative needs H grad(U) / |grad(U)|.
Here every Hessian column is computed by a central difference of force/gradient
calls and all five joint evaluations per two-dimensional collocation point
(one base plus four displaced points) are audited.
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


def query_gradient(evaluator, point):
    energy, force_flat = evaluator.query(physical_to_flat(point), "both")
    force = np.asarray(force_flat, dtype=np.float64)[[0, 3]]
    return float(energy), -force


def finite_difference_gradient_norm(
    evaluator,
    point,
    *,
    step,
    norm_floor,
):
    """Return |grad U| and its spatial derivative using a 2D FD Hessian."""
    point = np.asarray(point, dtype=np.float64)
    _, gradient = query_gradient(evaluator, point)
    hessian = np.empty((2, 2), dtype=np.float64)
    for coordinate in range(2):
        offset = np.zeros(2, dtype=np.float64)
        offset[coordinate] = float(step)
        _, gradient_plus = query_gradient(evaluator, point + offset)
        _, gradient_minus = query_gradient(evaluator, point - offset)
        hessian[:, coordinate] = (
            gradient_plus - gradient_minus
        ) / (2.0 * float(step))
    norm = float(np.sqrt(np.dot(gradient, gradient) + norm_floor**2))
    spatial_derivative = hessian.T @ gradient / norm
    symmetry_defect = float(np.linalg.norm(hessian - hessian.T))
    return norm, spatial_derivative, hessian, symmetry_defect


class GMAMTrace:
    def __init__(self):
        self.rows = []

    def append(
        self,
        model,
        evaluators,
        metrics,
        *,
        outer_iteration,
        objective,
        parameter_gradient_norm,
        hessian_symmetry_max,
        learning_rate,
        accepted,
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
                "outer_iteration": int(outer_iteration),
                "gmam_objective": float(objective),
                "parameter_gradient_norm": float(parameter_gradient_norm),
                "hessian_symmetry_max": float(hessian_symmetry_max),
                "learning_rate": float(learning_rate),
                "accepted": bool(accepted),
            }
        )

    def arrays(self):
        payload = {}
        float_keys = (
            "mean_calls",
            "dense_i_perp",
            "dense_fmax",
            "dense_barrier",
            "path_length",
            "minimum_speed",
            "spring_force_max_unit",
            "segment_length_cv",
            "gmam_objective",
            "parameter_gradient_norm",
            "hessian_symmetry_max",
            "learning_rate",
        )
        int_keys = ("call", "total_internal_calls", "outer_iteration")
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
        ):
            payload[key] = np.asarray([row[key] for row in self.rows])
        return payload


def configuration_hash(configuration):
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
        raise RuntimeError("gMAM does not start from the common initial path")

    backend = LEPSBackend(label=f"gmam_nn:{args.run_label}")
    collocation_count = int(args.collocation_count)
    evaluator_count = collocation_count + 2
    evaluators = make_evaluators(
        backend,
        count=evaluator_count,
        internal_limit=args.budget,
    )
    query_gradient(evaluators[0], initial_sample[0])
    query_gradient(evaluators[-1], initial_sample[-1])
    initial_s = equal_arclength_parameters(
        model,
        collocation_count,
        dense_count=args.arclength_points,
    )
    initial_points = model_coordinates(model, initial_s)
    for evaluator, point in zip(evaluators[1:-1], initial_points):
        query_gradient(evaluator, point)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    trace = GMAMTrace()
    saddle = reference_saddle()
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
        learning_rate=args.lr,
        accepted=True,
    )

    stop_reason = "budget"
    outer_iteration = 0
    rejected_steps = 0
    lr_scale = 1.0
    current_lr = float(args.lr)
    while True:
        internal_counts = counter_arrays(evaluators)["joint_evaluations"][1:-1]
        if int(np.max(internal_counts)) + 4 > int(args.budget):
            stop_reason = "budget"
            break
        if metrics["dense_i_perp"] <= float(args.iperp_tolerance):
            stop_reason = "dense_i_perp"
            break
        if outer_iteration >= int(args.max_outer):
            stop_reason = "outer_limit"
            break

        outer_iteration += 1
        current_lr = float(args.lr) * lr_scale / (
            1.0 + float(args.lr_decay) * max(outer_iteration - 1, 0)
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        internal_s = equal_arclength_parameters(
            model,
            collocation_count,
            dense_count=args.arclength_points,
        )
        all_s = np.concatenate(([0.0], internal_s, [1.0]))
        s_tensor, coordinates, first = coordinates_and_derivatives(
            model,
            all_s,
            create_graph=True,
        )
        detached_points = coordinates.detach().cpu().numpy()
        norms = np.zeros(evaluator_count, dtype=np.float64)
        spatial_derivatives = np.zeros(
            (evaluator_count, 2),
            dtype=np.float64,
        )
        symmetry_defects = []
        budget_hit = False
        try:
            for evaluator, point, index in zip(
                evaluators[1:-1],
                detached_points[1:-1],
                range(1, evaluator_count - 1),
            ):
                (
                    norms[index],
                    spatial_derivatives[index],
                    _,
                    symmetry,
                ) = finite_difference_gradient_norm(
                    evaluator,
                    point,
                    step=args.hessian_step,
                    norm_floor=args.gradient_norm_floor,
                )
                symmetry_defects.append(symmetry)
        except EvaluationBudgetReached:
            budget_hit = True

        if budget_hit:
            stop_reason = "budget"
            trace.append(
                model,
                evaluators,
                metrics,
                outer_iteration=outer_iteration,
                objective=np.nan,
                parameter_gradient_norm=np.nan,
                hessian_symmetry_max=(
                    np.nan
                    if not symmetry_defects
                    else float(np.max(symmetry_defects))
                ),
                learning_rate=current_lr,
                accepted=False,
            )
            break

        norm_tensor = torch.as_tensor(norms, dtype=TORCH_DTYPE)
        spatial_tensor = torch.as_tensor(
            spatial_derivatives,
            dtype=TORCH_DTYPE,
        )
        # This surrogate has the exact value |grad U| and the explicitly
        # finite-differenced spatial derivative required by backpropagation.
        norm_surrogate = norm_tensor + torch.sum(
            spatial_tensor * (coordinates - coordinates.detach()),
            dim=1,
        )
        speed = torch.sqrt(
            torch.sum(first**2, dim=1)
            + float(args.speed_floor) ** 2
        )
        objective = torch.trapz(norm_surrogate * speed, s_tensor)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        parameter_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(args.gradient_clip),
            )
        )
        old_vector = parameter_vector(model)
        optimizer.step()

        dense_probe = model_coordinates(
            model,
            np.linspace(0.0, 1.0, 1025),
        )
        geometry_ok = bool(
            np.all(np.isfinite(dense_probe))
            and np.min(dense_probe) > 0.25
            and np.max(dense_probe) < 8.5
            and np.sum(np.linalg.norm(np.diff(dense_probe, axis=0), axis=1))
            < float(args.max_path_length)
        )
        if not geometry_ok:
            load_parameter_vector(model, old_vector)
            rejected_steps += 1
            # Clear momentum from the rejected step and retry more cautiously.
            lr_scale *= 0.5
            current_lr *= 0.5
            optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
            accepted = False
        else:
            accepted = True

        metrics = dense_neural_metrics(
            model,
            dense_count=args.diagnostic_points,
        )
        trace.append(
            model,
            evaluators,
            metrics,
            outer_iteration=outer_iteration,
            objective=float(objective.detach()),
            parameter_gradient_norm=parameter_gradient_norm,
            hessian_symmetry_max=float(np.max(symmetry_defects)),
            learning_rate=current_lr,
            accepted=accepted,
        )
        if rejected_steps >= int(args.max_rejected_steps):
            stop_reason = "geometry_guard"
            break

    audit = audit_counter_consistency(backend, evaluators)
    if not audit["consistent"]:
        raise RuntimeError(f"gMAM counter audit failed: {audit}")

    configuration = {
        "method": "NN-gMAM",
        "run_label": args.run_label,
        "width": int(args.width),
        "depth": int(args.depth),
        "fourier_order": int(args.fourier_order),
        "seed": int(args.seed),
        "lr": float(args.lr),
        "lr_decay": float(args.lr_decay),
        "gradient_clip": float(args.gradient_clip),
        "hessian_step": float(args.hessian_step),
        "budget": int(args.budget),
        "collocation_count": collocation_count,
        "sampling": "equal-arclength collocation on unchanged neural path",
        "tangent": "torch autograd dphi/ds",
        "hessian": (
            "central finite difference of the two gradient columns; "
            "one base plus four displaced joint evaluations per point"
        ),
        "objective": "integral |grad U(phi)| |phi'(s)| ds",
    }
    arrays = trace.arrays()
    arrays.update(
        {
            "method": np.asarray("NN-gMAM"),
            "run_label": np.asarray(args.run_label),
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
            "hessian_definition": np.asarray(
                "2D central finite difference, 4 displaced force calls"
            ),
            "sampling_definition": np.asarray(
                "equal-arclength collocation without path reparameterization"
            ),
            "diagnostic_points": np.asarray(args.diagnostic_points),
            "collocation_count": np.asarray(collocation_count),
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
                "dense_i_perp": float(arrays["dense_i_perp"][-1]),
                "dense_barrier": float(arrays["dense_barrier"][-1]),
                "dense_barrier_error": float(
                    arrays["dense_barrier_error"][-1]
                ),
                "spring_force_max_unit": float(
                    arrays["spring_force_max_unit"][-1]
                ),
                "rejected_steps": int(rejected_steps),
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
    parser.add_argument("--fourier-order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lr-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--hessian-step", type=float, default=1e-4)
    parser.add_argument("--gradient-norm-floor", type=float, default=1e-10)
    parser.add_argument("--speed-floor", type=float, default=1e-10)
    parser.add_argument("--max-path-length", type=float, default=30.0)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--collocation-count", type=int, default=64)
    parser.add_argument("--max-outer", type=int, default=1000)
    parser.add_argument("--max-rejected-steps", type=int, default=8)
    parser.add_argument("--iperp-tolerance", type=float, default=1e-12)
    parser.add_argument("--arclength-points", type=int, default=4097)
    parser.add_argument("--diagnostic-points", type=int, default=4097)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
