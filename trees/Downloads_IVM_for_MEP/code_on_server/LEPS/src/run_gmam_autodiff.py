"""Direct-autodiff neural gMAM with exactly 20 jittered collocation streams."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from continuous_paths import (
    FixedEndpointPathNet,
    coordinates_and_derivatives,
    dense_neural_metrics,
    load_parameter_vector,
    model_coordinates,
    parameter_vector,
    arclength_parameters_for_fractions,
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
from torch_leps import torch_leps_energy_force


def configuration_hash(configuration):
    encoded = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def query_auditor(evaluator, point):
    energy, force_flat = evaluator.query(physical_to_flat(point), "both")
    force = np.asarray(force_flat, dtype=np.float64)[[0, 3]]
    return float(energy), force


def audit_torch_batch(evaluators, points, torch_energy, torch_force):
    """Count one joint query per point and validate the Torch implementation."""
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


class AutodiffTrace:
    def __init__(self):
        self.rows = []

    def append(
        self,
        model,
        evaluators,
        metrics,
        *,
        outer_iteration,
        batch_objective,
        parameter_gradient_norm,
        learning_rate,
        accepted,
        arc_fractions,
        energy_validation_error,
        force_validation_error,
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
                "model_state": parameter_vector(model),
                "outer_iteration": int(outer_iteration),
                "gmam_objective": float(batch_objective),
                "parameter_gradient_norm": float(parameter_gradient_norm),
                "learning_rate": float(learning_rate),
                "accepted": bool(accepted),
                "collocation_arc_fractions": np.asarray(
                    arc_fractions,
                    dtype=np.float64,
                ),
                "energy_validation_error": float(energy_validation_error),
                "force_validation_error": float(force_validation_error),
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
            "gmam_objective",
            "parameter_gradient_norm",
            "learning_rate",
            "energy_validation_error",
            "force_validation_error",
        )
        int_keys = (
            "calls_per_image",
            "total_internal_calls",
            "outer_iteration",
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
    common_path = load_common_path(root / "inputs" / "common_initial_path.npz")
    collocation_count = int(args.collocation_count)
    if collocation_count != 20:
        raise ValueError(
            "This comparison requires exactly 20 internal collocation points"
        )
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
        raise RuntimeError("Autodiff gMAM does not start from the common path")

    backend = LEPSBackend(label=f"gmam_autodiff:{args.run_label}")
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
    initial_all_s = np.concatenate(([0.0], initial_fractions, [1.0]))
    for evaluator, point in zip(evaluators, initial_sample):
        query_auditor(evaluator, point)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    trace = AutodiffTrace()
    metrics = dense_neural_metrics(
        model,
        dense_count=args.diagnostic_points,
    )
    trace.append(
        model,
        evaluators,
        metrics,
        outer_iteration=0,
        batch_objective=np.nan,
        parameter_gradient_norm=np.nan,
        learning_rate=args.lr,
        accepted=True,
        arc_fractions=initial_fractions,
        energy_validation_error=0.0,
        force_validation_error=0.0,
    )

    stop_reason = "budget"
    rejected_steps = 0
    lr_scale = 1.0
    outer_iteration = 0
    while outer_iteration < int(args.max_outer):
        counts = counter_arrays(evaluators)["joint_evaluations"][1:-1]
        if int(np.max(counts)) >= int(args.budget):
            stop_reason = "budget"
            break
        outer_iteration += 1
        arc_fractions = stratified_jitter_fractions(
            collocation_count,
            outer_iteration,
            jitter=args.collocation_jitter,
            seed=args.sampling_seed,
        )
        internal_s = arclength_parameters_for_fractions(
            model,
            arc_fractions,
            dense_count=args.arclength_points,
        )
        all_s = np.concatenate(([0.0], internal_s, [1.0]))
        s_tensor, coordinates, first = coordinates_and_derivatives(
            model,
            all_s,
            create_graph=True,
        )
        try:
            torch_energy, torch_force = torch_leps_energy_force(
                coordinates,
                create_graph=True,
            )
            energy_error, force_error = audit_torch_batch(
                evaluators,
                coordinates.detach().cpu().numpy(),
                torch_energy,
                torch_force,
            )
        except EvaluationBudgetReached:
            stop_reason = "budget"
            break

        gradient_norm = torch.linalg.norm(-torch_force, dim=1)
        speed = torch.sqrt(
            torch.sum(first**2, dim=1) + float(args.speed_floor) ** 2
        )
        objective = torch.trapz(gradient_norm * speed, s_tensor)
        current_lr = float(args.lr) * lr_scale / (
            1.0 + float(args.lr_decay) * max(outer_iteration - 1, 0)
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
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

        probe = model_coordinates(model, np.linspace(0.0, 1.0, 1025))
        geometry_ok = bool(
            np.all(np.isfinite(probe))
            and np.min(probe) > 0.25
            and np.max(probe) < 8.5
            and np.sum(np.linalg.norm(np.diff(probe, axis=0), axis=1))
            < float(args.max_path_length)
        )
        if geometry_ok:
            accepted = True
        else:
            load_parameter_vector(model, old_vector)
            rejected_steps += 1
            lr_scale *= 0.5
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(args.lr) * lr_scale,
            )
            accepted = False

        metrics = dense_neural_metrics(
            model,
            dense_count=args.diagnostic_points,
        )
        trace.append(
            model,
            evaluators,
            metrics,
            outer_iteration=outer_iteration,
            batch_objective=float(objective.detach()),
            parameter_gradient_norm=parameter_gradient_norm,
            learning_rate=current_lr,
            accepted=accepted,
            arc_fractions=arc_fractions,
            energy_validation_error=energy_error,
            force_validation_error=force_error,
        )
        if rejected_steps >= int(args.max_rejected_steps):
            stop_reason = "geometry_guard"
            break
        if metrics["dense_i_perp"] <= float(args.iperp_tolerance):
            stop_reason = "dense_i_perp"
            break
    else:
        stop_reason = "outer_limit"

    audit = audit_counter_consistency(backend, evaluators)
    if not audit["consistent"]:
        raise RuntimeError(f"Autodiff gMAM counter audit failed: {audit}")
    saddle = reference_saddle()
    configuration = {
        "method": "NN-gMAM-direct-autodiff",
        "run_label": args.run_label,
        "width": int(args.width),
        "depth": int(args.depth),
        "fourier_order": int(args.fourier_order),
        "seed": int(args.seed),
        "collocation_count": collocation_count,
        "collocation_jitter": float(args.collocation_jitter),
        "sampling_seed": int(args.sampling_seed),
        "lr": float(args.lr),
        "lr_decay": float(args.lr_decay),
        "gradient_clip": float(args.gradient_clip),
        "budget": int(args.budget),
        "objective": "integral |grad U(phi)| |phi'(s)| ds",
        "potential": "pure PyTorch analytic LEPS energy",
        "force": "torch autograd of LEPS energy",
        "parameter_gradient": (
            "direct torch backward through force norm and path speed"
        ),
        "hessian": (
            "not formed or finite-differenced; second derivatives are "
            "implicit in torch autograd"
        ),
    }
    arrays = trace.arrays()
    arrays.update(
        {
            "method": np.asarray("NN-gMAM-direct-autodiff"),
            "run_label": np.asarray(args.run_label),
            "width": np.asarray(args.width),
            "depth": np.asarray(args.depth),
            "fourier_order": np.asarray(args.fourier_order),
            "seed": np.asarray(args.seed),
            "collocation_count": np.asarray(collocation_count),
            "collocation_jitter": np.asarray(args.collocation_jitter),
            "sampling_seed": np.asarray(args.sampling_seed),
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
                "20 stratified equal-arclength points with reproducible "
                "within-stratum jitter"
            ),
            "hessian_definition": np.asarray(
                "implicit torch autograd only; no explicit or FD Hessian"
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
                "dense_i_perp": float(arrays["dense_i_perp"][-1]),
                "dense_barrier": float(arrays["dense_barrier"][-1]),
                "dense_barrier_error": float(
                    arrays["dense_barrier_error"][-1]
                ),
                "spring_force_max_unit": float(
                    arrays["spring_force_max_unit"][-1]
                ),
                "maximum_torch_energy_validation_error": float(
                    np.max(arrays["energy_validation_error"])
                ),
                "maximum_torch_force_validation_error": float(
                    np.max(arrays["force_validation_error"])
                ),
                "rejected_geometry_steps": int(rejected_steps),
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
    parser.add_argument("--collocation-count", type=int, default=20)
    parser.add_argument("--collocation-jitter", type=float, default=0.30)
    parser.add_argument("--sampling-seed", type=int, default=20260726)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lr-decay", type=float, default=0.005)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--speed-floor", type=float, default=1e-10)
    parser.add_argument("--max-path-length", type=float, default=30.0)
    parser.add_argument("--max-rejected-steps", type=int, default=8)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--max-outer", type=int, default=1000)
    parser.add_argument("--iperp-tolerance", type=float, default=0.0)
    parser.add_argument("--arclength-points", type=int, default=4097)
    parser.add_argument("--diagnostic-points", type=int, default=4097)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
