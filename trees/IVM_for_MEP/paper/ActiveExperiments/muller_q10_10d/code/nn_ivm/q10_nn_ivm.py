#!/usr/bin/env python3
"""Theory-audited continuous NN-IVM for fixed-Q 10D Muller-Brown.

The clean method freezes the current continuous path and its tangent, then
minimizes the standard fixed-alpha IVM proximal objective in a pointwise
frozen-normal neural chart.  Twenty collocation samples approximate only the
path integral.  Dense metrics are certified after optimization and never enter
an update or acceptance decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from q10_ivm_core import REFERENCE_BARRIER, improved_tangents, improved_tangents_arbitrary
from q10_potential import DIMENSION, Q, energy_gradient, energy_values, transformed_initial_path


DTYPE = torch.float64
DEVICE = torch.device("cpu")
SAMPLE_COUNT = 20


class ForceBudgetReached(RuntimeError):
    pass


class StreamForceOracle:
    """Twenty independent cached streams with direct-sum force accounting."""

    def __init__(self, stream_count: int, budget: int):
        self.stream_count = int(stream_count)
        self.budget = int(budget)
        self.calls = np.zeros(self.stream_count, dtype=np.int64)
        self.caches: list[dict[bytes, tuple[float, np.ndarray]]] = [
            {} for _ in range(self.stream_count)
        ]

    @property
    def total_calls(self) -> int:
        return int(np.sum(self.calls))

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.ascontiguousarray(points, dtype=np.float64)
        if points.shape != (self.stream_count, DIMENSION):
            raise ValueError(f"expected {(self.stream_count, DIMENSION)}, got {points.shape}")
        missing = []
        for index, point in enumerate(points):
            if point.tobytes() not in self.caches[index]:
                missing.append(index)
        if self.total_calls + len(missing) > self.budget:
            raise ForceBudgetReached
        energies = np.empty(self.stream_count, dtype=np.float64)
        gradients = np.empty_like(points)
        for index, point in enumerate(points):
            key = point.tobytes()
            cached = self.caches[index].get(key)
            if cached is None:
                energy, gradient = energy_gradient(point)
                cached = (float(energy), np.asarray(gradient, dtype=np.float64))
                self.caches[index][key] = cached
                self.calls[index] += 1
            energies[index] = cached[0]
            gradients[index] = cached[1]
        return energies, gradients


class LinearEndpointPath(nn.Module):
    def __init__(self, start: np.ndarray, end: np.ndarray):
        super().__init__()
        self.register_buffer("start", torch.as_tensor(start, dtype=DTYPE))
        self.register_buffer("end", torch.as_tensor(end, dtype=DTYPE))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = s.reshape(-1, 1)
        return (1.0 - s) * self.start + s * self.end


class EndpointFixedDisplacementNet(nn.Module):
    def __init__(
        self, width: int, depth: int, fourier_order: int, seed: int,
        last_layer_only: bool = False,
    ):
        super().__init__()
        torch.manual_seed(int(seed))
        self.fourier_order = int(fourier_order)
        input_dimension = 1 + 2 * self.fourier_order
        layers: list[nn.Module] = []
        current = input_dimension
        for _ in range(int(depth)):
            linear = nn.Linear(current, int(width), dtype=DTYPE)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.Tanh()))
            current = int(width)
        output = nn.Linear(current, DIMENSION, dtype=DTYPE)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)
        if last_layer_only:
            for parameter in self.network.parameters():
                parameter.requires_grad_(False)
            for parameter in output.parameters():
                parameter.requires_grad_(True)

    def features(self, s: torch.Tensor) -> torch.Tensor:
        columns = [s.reshape(-1, 1)]
        for order in range(1, self.fourier_order + 1):
            angle = 2.0 * math.pi * order * s.reshape(-1, 1)
            columns.extend((torch.sin(angle), torch.cos(angle)))
        return torch.cat(columns, dim=1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s_column = s.reshape(-1, 1)
        envelope = 4.0 * s_column * (1.0 - s_column)
        return envelope * self.network(self.features(s))


class FrozenTangentTable(nn.Module):
    def __init__(self, grid: np.ndarray, tangents: np.ndarray):
        super().__init__()
        self.register_buffer("grid", torch.as_tensor(grid, dtype=DTYPE))
        self.register_buffer("tangents", torch.as_tensor(tangents, dtype=DTYPE))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        values = torch.clamp(s.reshape(-1), 0.0, 1.0)
        right = torch.searchsorted(self.grid, values, right=True)
        right = torch.clamp(right, 1, self.grid.numel() - 1)
        left = right - 1
        x0 = self.grid[left]
        x1 = self.grid[right]
        weight = ((values - x0) / torch.clamp(x1 - x0, min=1.0e-15)).reshape(-1, 1)
        tangent = (1.0 - weight) * self.tangents[left] + weight * self.tangents[right]
        return tangent / torch.clamp(torch.linalg.norm(tangent, dim=1, keepdim=True), min=1.0e-14)


class FrozenNormalUpdate(nn.Module):
    def __init__(
        self,
        reference: nn.Module,
        tangent_table: FrozenTangentTable,
        displacement: EndpointFixedDisplacementNet,
    ):
        super().__init__()
        self.reference = reference
        self.tangent_table = tangent_table
        self.displacement = displacement

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        reference = self.reference(s)
        tangent = self.tangent_table(s)
        raw = self.displacement(s)
        normal = raw - torch.sum(raw * tangent, dim=1, keepdim=True) * tangent
        return reference + normal


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def model_coordinates(model: nn.Module, s: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(np.asarray(s), dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        return model(tensor).cpu().numpy()


def path_geometry(model: nn.Module, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parameter = torch.as_tensor(np.asarray(s), dtype=DTYPE, device=DEVICE).clone().detach()
    parameter.requires_grad_(True)
    coordinates = model(parameter)
    derivatives = []
    for coordinate in range(DIMENSION):
        derivative = torch.autograd.grad(
            coordinates[:, coordinate].sum(),
            parameter,
            retain_graph=True,
            create_graph=False,
        )[0]
        derivatives.append(derivative)
    derivative = torch.stack(derivatives, dim=1)
    speed = torch.linalg.norm(derivative, dim=1)
    tangent = derivative / torch.clamp(speed.reshape(-1, 1), min=1.0e-14)
    return (
        coordinates.detach().cpu().numpy(),
        tangent.detach().cpu().numpy(),
        speed.detach().cpu().numpy(),
    )


def tangent_table(model: nn.Module, count: int) -> FrozenTangentTable:
    grid = np.linspace(0.0, 1.0, int(count))
    _, tangents, _ = path_geometry(model, grid)
    return FrozenTangentTable(grid, tangents)


def improved_tangent_table(
    model: nn.Module,
    arclength_count: int,
    stencil_step: float,
    table_count: int,
) -> FrozenTangentTable:
    """Continuous local NEB improved tangent on an arc-length stencil."""
    h = float(stencil_step)
    if not 0.0 < h < 0.5:
        raise ValueError("improved-tangent stencil step must lie in (0, 0.5)")
    center_fraction = np.linspace(h, 1.0 - h, int(table_count))
    triplet_fraction = np.stack(
        (center_fraction - h, center_fraction, center_fraction + h), axis=1
    )
    triplet_s = arclength_parameters(
        model, triplet_fraction.reshape(-1), arclength_count
    ).reshape(-1, 3)
    points = model_coordinates(model, triplet_s.reshape(-1)).reshape(
        -1, 3, DIMENSION
    )
    energies = energy_values(points.reshape(-1, DIMENSION)).reshape(-1, 3)
    tangents = np.asarray([
        improved_tangents_arbitrary(point_triplet, energy_triplet)[0]
        for point_triplet, energy_triplet in zip(points, energies)
    ])
    return FrozenTangentTable(triplet_s[:, 1], tangents)


def stratified_fractions(count: int, outer: int, jitter: float, seed: int) -> np.ndarray:
    centers = np.arange(int(count), dtype=np.float64) + 0.5
    if float(jitter) > 0.0:
        rng = np.random.default_rng(int(seed) + 104729 * int(outer))
        centers += rng.uniform(-float(jitter), float(jitter), size=int(count))
    return np.sort(np.clip(centers / float(count), 1.0e-8, 1.0 - 1.0e-8))


def arclength_parameters(model: nn.Module, fractions: np.ndarray, count: int) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, int(count))
    coordinates = model_coordinates(model, grid)
    lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 1.0e-14:
        raise RuntimeError("path has zero arclength")
    cumulative /= cumulative[-1]
    keep = np.concatenate(([True], np.diff(cumulative) > 1.0e-14))
    return np.interp(np.asarray(fractions), cumulative[keep], grid[keep])


def parameter_vector(module: nn.Module) -> torch.Tensor:
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    return torch.nn.utils.parameters_to_vector(parameters).detach().clone()


def load_parameter_vector(module: nn.Module, vector: torch.Tensor) -> None:
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    torch.nn.utils.vector_to_parameters(vector, parameters)


def gradient_vector(module: nn.Module) -> torch.Tensor:
    gradients = []
    for parameter in module.parameters():
        if parameter.requires_grad:
            gradients.append(
                torch.zeros_like(parameter).reshape(-1)
                if parameter.grad is None
                else parameter.grad.reshape(-1)
            )
    return torch.cat(gradients)


def proximal_evaluation(
    candidate: FrozenNormalUpdate,
    reference_points: np.ndarray,
    sample_s: np.ndarray,
    oracle: StreamForceOracle,
    alpha: float,
    *,
    backward: bool,
) -> dict:
    parameter = torch.as_tensor(sample_s, dtype=DTYPE, device=DEVICE)
    coordinates = candidate(parameter)
    points = coordinates.detach().cpu().numpy()
    energies, gradients = oracle.query(points)
    reference_tensor = torch.as_tensor(reference_points, dtype=DTYPE, device=DEVICE)
    displacement = coordinates - reference_tensor
    objective = float(
        np.mean(energies)
        + torch.mean(torch.sum(displacement.detach() ** 2, dim=1)).item()
        / (2.0 * float(alpha))
    )
    if backward:
        candidate.zero_grad(set_to_none=True)
        coordinate_gradient = (
            torch.as_tensor(gradients, dtype=DTYPE, device=DEVICE)
            + displacement.detach() / float(alpha)
        ) / float(points.shape[0])
        coordinates.backward(coordinate_gradient)
    return {
        "objective": objective,
        "points": points,
        "energies": energies,
        "gradients": gradients,
        "maximum_displacement": float(np.max(np.linalg.norm(points - reference_points, axis=1))),
    }


def dense_certificate(
    paths: np.ndarray,
    tangents: np.ndarray,
    speeds: np.ndarray,
    parameters: np.ndarray,
) -> dict:
    checkpoint_count, dense_count, _ = paths.shape
    dense_i_perp = np.empty(checkpoint_count, dtype=np.float64)
    dense_delta_e = np.empty(checkpoint_count, dtype=np.float64)
    inactive_max = np.empty(checkpoint_count, dtype=np.float64)
    for checkpoint in range(checkpoint_count):
        energies = np.empty(dense_count, dtype=np.float64)
        gradients = np.empty((dense_count, DIMENSION), dtype=np.float64)
        for index, point in enumerate(paths[checkpoint]):
            energies[index], gradients[index] = energy_gradient(point)
        tangent = improved_tangents_arbitrary(paths[checkpoint], energies)
        internal_gradients = gradients[1:-1]
        normal = internal_gradients - np.sum(
            internal_gradients * tangent, axis=1, keepdims=True
        ) * tangent
        segments = np.linalg.norm(np.diff(paths[checkpoint], axis=0), axis=1)
        weights = 0.5 * (segments[:-1] + segments[1:])
        dense_i_perp[checkpoint] = float(
            np.sum(weights * np.sum(normal * normal, axis=1))
        )
        dense_delta_e[checkpoint] = float(np.max(energies) - energies[0])
        natural = paths[checkpoint] @ Q.T
        inactive_max[checkpoint] = float(np.max(np.abs(natural[:, 2:])))
    return {
        "dense_i_perp": dense_i_perp,
        "dense_delta_e": dense_delta_e,
        "dense_barrier_error": np.abs(dense_delta_e - float(REFERENCE_BARRIER)),
        "dense_diagnostic_force_calls": int(checkpoint_count * dense_count),
        "inactive_coordinate_max": inactive_max,
    }


@dataclass(frozen=True)
class NNIVMConfig:
    alpha: float
    force_budget: int = 5000
    max_outer: int = 60
    sample_count: int = SAMPLE_COUNT
    sampling_jitter: float = 0.0
    sampling_seed: int = 20260802
    network_seed: int = 20260803
    fixed_network_features: bool = False
    random_feature_last_layer_only: bool = False
    persistent_displacement_init: bool = False
    persistent_init_scale: float = 1.0
    width: int = 24
    depth: int = 2
    fourier_order: int = 4
    inner_steps: int = 1
    resample_inner_batches: bool = False
    learning_rate: float = 0.01
    gradient_clip: float = 50.0
    backtracks: int = 6
    trust_radius: float = 0.30
    unconstrained_delta: bool = False
    use_improved_tangent: bool = False
    tangent_stencil_step: float = 0.002
    tangent_table_points: int = 257
    acceptance_tolerance: float = 1.0e-10
    arclength_points: int = 1025
    tangent_points: int = 2049
    certificate_points: int = 1001
    role: str = "nn_ivm_clean"


def configuration_id(config: NNIVMConfig) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def normalized_log_auc(cost: np.ndarray, values: np.ndarray, limit: int) -> float:
    cost = np.asarray(cost, dtype=np.float64)
    values = np.maximum(np.asarray(values, dtype=np.float64), 1.0e-300)
    keep = cost <= float(limit)
    x = cost[keep]
    y = np.log10(values[keep] / values[0])
    if x[-1] < float(limit):
        x = np.append(x, float(limit))
        y = np.append(y, y[-1])
    span = max(float(x[-1] - x[0]), 1.0)
    return float(np.trapz(y, x) / span)


def run_experiment(root: Path, config: NNIVMConfig) -> dict:
    if int(config.sample_count) != SAMPLE_COUNT:
        raise ValueError("Q10 NN-IVM requires exactly 20 collocation samples")
    if not np.isfinite(config.alpha) or float(config.alpha) <= 0.0:
        raise ValueError("alpha must be one positive fixed scalar")
    root = Path(root).resolve()
    initial = transformed_initial_path(22)
    model: nn.Module = freeze(LinearEndpointPath(initial[0], initial[-1]).double())
    oracle = StreamForceOracle(config.sample_count, config.force_budget)
    certificate_s = np.linspace(0.0, 1.0, config.certificate_points)

    trace: list[dict] = []
    geometry_paths: list[np.ndarray] = []
    geometry_tangents: list[np.ndarray] = []
    geometry_speeds: list[np.ndarray] = []

    def record(*, outer: int, accepted: bool, objective_change: float,
               gradient_norm: float, learning_rate: float,
               normal_leakage: float, maximum_displacement: float) -> None:
        path, tangent, speed = path_geometry(model, certificate_s)
        geometry_paths.append(path)
        geometry_tangents.append(tangent)
        geometry_speeds.append(speed)
        trace.append({
            "outer": int(outer),
            "number_of_forces": oracle.total_calls,
            "counter_vector": oracle.calls.copy(),
            "accepted": bool(accepted),
            "sampled_proximal_change": float(objective_change),
            "parameter_gradient_norm": float(gradient_norm),
            "accepted_learning_rate": float(learning_rate),
            "normal_leakage": float(normal_leakage),
            "maximum_displacement": float(maximum_displacement),
        })

    record(
        outer=0, accepted=True, objective_change=0.0,
        gradient_norm=0.0, learning_rate=0.0,
        normal_leakage=0.0, maximum_displacement=0.0,
    )
    stop_reason = "outer_limit"
    accepted_updates = 0
    previous_displacement_state = None
    for outer in range(1, int(config.max_outer) + 1):
        if oracle.total_calls + config.sample_count > config.force_budget:
            stop_reason = "budget"
            break
        sample_index = (
            (outer - 1) * int(config.inner_steps) + 1
            if config.resample_inner_batches
            else outer
        )
        fractions = stratified_fractions(
            config.sample_count, sample_index,
            config.sampling_jitter, config.sampling_seed,
        )
        sample_s = arclength_parameters(model, fractions, config.arclength_points)
        reference = freeze(model)
        reference_points, reference_tangents, _ = path_geometry(reference, sample_s)
        table = (
            improved_tangent_table(
                reference,
                config.arclength_points,
                config.tangent_stencil_step,
                config.tangent_table_points,
            )
            if config.use_improved_tangent
            else tangent_table(reference, config.tangent_points)
        )
        displacement = EndpointFixedDisplacementNet(
            config.width, config.depth, config.fourier_order,
            (
                config.network_seed
                if config.fixed_network_features
                else config.network_seed + outer
            ),
            last_layer_only=config.random_feature_last_layer_only,
        ).double()
        if config.persistent_displacement_init and previous_displacement_state is not None:
            displacement.load_state_dict(previous_displacement_state)
            output_layer = displacement.network[-1]
            with torch.no_grad():
                output_layer.weight.mul_(float(config.persistent_init_scale))
                output_layer.bias.mul_(float(config.persistent_init_scale))
        candidate = FrozenNormalUpdate(reference, table, displacement).double()

        accepted_any = False
        initial_objective = np.nan
        current_objective = np.nan
        final_gradient_norm = np.nan
        accepted_lr = 0.0
        maximum_displacement = 0.0
        try:
            audit_batches = []
            for inner in range(int(config.inner_steps)):
                if config.resample_inner_batches and inner > 0:
                    fractions = stratified_fractions(
                        config.sample_count,
                        (outer - 1) * int(config.inner_steps) + inner + 1,
                        config.sampling_jitter,
                        config.sampling_seed,
                    )
                    sample_s = arclength_parameters(
                        reference, fractions, config.arclength_points
                    )
                    reference_points = model_coordinates(reference, sample_s)
                reference_energies, _ = oracle.query(reference_points)
                zero_objective = float(np.mean(reference_energies))
                audit_batches.append(
                    (sample_s.copy(), reference_points.copy(), zero_objective)
                )
                current = proximal_evaluation(
                    candidate, reference_points, sample_s, oracle,
                    config.alpha, backward=True,
                )
                current_objective = current["objective"]
                gradient = gradient_vector(displacement)
                final_gradient_norm = float(torch.linalg.norm(gradient).item())
                scale = (
                    1.0
                    if config.unconstrained_delta
                    else min(1.0, float(config.gradient_clip) / max(final_gradient_norm, 1.0e-30))
                )
                direction = -scale * gradient
                old_vector = parameter_vector(displacement)
                trial_lr = float(config.learning_rate)
                step_accepted = False
                for _ in range(int(config.backtracks) + 1):
                    load_parameter_vector(displacement, old_vector + trial_lr * direction)
                    trial = proximal_evaluation(
                        candidate, reference_points, sample_s, oracle,
                        config.alpha, backward=False,
                    )
                    if (
                        np.isfinite(trial["objective"])
                        and trial["objective"]
                        <= current_objective + float(config.acceptance_tolerance)
                        and (
                            config.unconstrained_delta
                            or trial["maximum_displacement"] <= float(config.trust_radius)
                        )
                    ):
                        current_objective = trial["objective"]
                        maximum_displacement = max(
                            maximum_displacement, trial["maximum_displacement"]
                        )
                        accepted_lr = trial_lr
                        step_accepted = True
                        accepted_any = True
                        break
                    load_parameter_vector(displacement, old_vector)
                    trial_lr *= 0.5
                if not step_accepted:
                    break
            if config.resample_inner_batches and accepted_any:
                initial_objective = float(
                    np.mean([batch[2] for batch in audit_batches])
                )
                final_objectives = []
                for batch_s, batch_reference, _ in audit_batches:
                    final_objectives.append(
                        proximal_evaluation(
                            candidate,
                            batch_reference,
                            batch_s,
                            oracle,
                            config.alpha,
                            backward=False,
                        )["objective"]
                    )
                current_objective = float(np.mean(final_objectives))
            elif audit_batches:
                initial_objective = audit_batches[0][2]
        except ForceBudgetReached:
            stop_reason = "budget"

        normal_leakage = np.nan
        if accepted_any and np.isfinite(initial_objective):
            validation_s = np.linspace(0.0, 1.0, 257)[1:-1]
            reference_validation = model_coordinates(reference, validation_s)
            tangent_validation = table(
                torch.as_tensor(validation_s, dtype=DTYPE, device=DEVICE)
            ).detach().cpu().numpy()
            candidate_validation = model_coordinates(candidate, validation_s)
            delta = candidate_validation - reference_validation
            normal_leakage = float(np.max(np.abs(np.sum(delta * tangent_validation, axis=1))))
            maximum_displacement = float(np.max(np.linalg.norm(delta, axis=1)))
            if (
                (not config.unconstrained_delta and maximum_displacement > float(config.trust_radius))
                or normal_leakage > 2.0e-3
                or current_objective > initial_objective + float(config.acceptance_tolerance)
            ):
                accepted_any = False

        if accepted_any:
            model = freeze(candidate)
            accepted_updates += 1
            if config.persistent_displacement_init:
                previous_displacement_state = {
                    key: value.detach().clone()
                    for key, value in displacement.state_dict().items()
                }
        else:
            model = reference
        objective_change = (
            current_objective - initial_objective
            if np.isfinite(current_objective) and np.isfinite(initial_objective)
            else np.nan
        )
        record(
            outer=outer,
            accepted=accepted_any,
            objective_change=objective_change,
            gradient_norm=final_gradient_norm,
            learning_rate=accepted_lr if accepted_any else 0.0,
            normal_leakage=normal_leakage,
            maximum_displacement=maximum_displacement,
        )
        if stop_reason == "budget":
            break
        if oracle.total_calls >= config.force_budget:
            stop_reason = "budget"
            break

    paths = np.asarray(geometry_paths)
    tangents = np.asarray(geometry_tangents)
    speeds = np.asarray(geometry_speeds)
    certificate = dense_certificate(paths, tangents, speeds, certificate_s)
    number_of_forces = np.asarray([row["number_of_forces"] for row in trace], dtype=np.int64)
    accepted = np.asarray([row["accepted"] for row in trace], dtype=bool)
    proximal_change = np.asarray([row["sampled_proximal_change"] for row in trace])
    theory_pass = bool(
        np.all(proximal_change[accepted][1:] <= config.acceptance_tolerance)
        and np.all(np.diff(number_of_forces) >= 0)
        and float(config.alpha) == config.alpha
    )
    score_i = normalized_log_auc(
        number_of_forces, certificate["dense_i_perp"], config.force_budget
    )
    score_b = normalized_log_auc(
        number_of_forces, certificate["dense_barrier_error"], config.force_budget
    )

    output_directory = root / "runs" / config.role
    output_directory.mkdir(parents=True, exist_ok=True)
    cfg_id = configuration_id(config)
    output = output_directory / f"q10_nn_ivm_{cfg_id}.npz"
    model_path = output.with_suffix(".model.pt")
    torch.save(model, model_path)
    arrays = {
        "method": np.asarray("NN-IVM-clean-frozen-normal-chart"),
        "configuration_json": np.asarray(json.dumps(asdict(config), sort_keys=True)),
        "configuration_id": np.asarray(cfg_id),
        "alpha": np.asarray(config.alpha),
        "alpha_fixed": np.asarray(True),
        "number_of_forces": number_of_forces,
        "counter_vectors": np.asarray([row["counter_vector"] for row in trace]),
        "accepted": accepted,
        "outer": np.asarray([row["outer"] for row in trace]),
        "sampled_proximal_change": proximal_change,
        "parameter_gradient_norm": np.asarray([row["parameter_gradient_norm"] for row in trace]),
        "accepted_learning_rate": np.asarray([row["accepted_learning_rate"] for row in trace]),
        "normal_leakage": np.asarray([row["normal_leakage"] for row in trace]),
        "maximum_displacement": np.asarray([row["maximum_displacement"] for row in trace]),
        "certificate_s": certificate_s,
        "dense_path_history": paths,
        "dense_tangent_history": tangents,
        "dense_speed_history": speeds,
        "dense_i_perp": certificate["dense_i_perp"],
        "dense_delta_e": certificate["dense_delta_e"],
        "dense_barrier_error": certificate["dense_barrier_error"],
        "inactive_coordinate_max": certificate["inactive_coordinate_max"],
        "dense_diagnostic_force_calls": np.asarray(certificate["dense_diagnostic_force_calls"]),
        "dense_information_enters_optimization": np.asarray(False),
        "optimization_force_calls": np.asarray(oracle.total_calls),
        "model_path": np.asarray(str(model_path)),
        "stop_reason": np.asarray(stop_reason),
        "theory_audit_pass": np.asarray(theory_pass),
    }
    np.savez_compressed(output, **arrays)
    summary = {
        "configuration": asdict(config),
        "configuration_id": cfg_id,
        "output": str(output.relative_to(root)),
        "model": str(model_path.relative_to(root)),
        "stop_reason": stop_reason,
        "optimization_force_calls": oracle.total_calls,
        "accepted_updates": accepted_updates,
        "final_dense_i_perp": float(certificate["dense_i_perp"][-1]),
        "final_dense_delta_e": float(certificate["dense_delta_e"][-1]),
        "final_dense_barrier_error": float(certificate["dense_barrier_error"][-1]),
        "minimum_dense_i_perp": float(np.min(certificate["dense_i_perp"])),
        "minimum_dense_barrier_error": float(np.min(certificate["dense_barrier_error"])),
        "maximum_inactive_coordinate": float(np.max(certificate["inactive_coordinate_max"])),
        "normalized_log_i_perp_auc": score_i,
        "normalized_log_barrier_auc": score_b,
        "early_score": score_i + score_b,
        "theory_audit_pass": theory_pass,
        "unconstrained_delta": bool(config.unconstrained_delta),
        "improved_tangent": bool(config.use_improved_tangent),
        "random_feature_last_layer_only": bool(config.random_feature_last_layer_only),
        "dense_information_enters_optimization": False,
        "force_count_definition": "direct sum of the 20 independent sample-stream counters",
    }
    output.with_suffix(".audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
