#!/usr/bin/env python3
"""Unconstrained full-MLP and random-feature NN-IVM with L-BFGS subsolves."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize

import q10_nn_ivm as base
from q10_potential import Q, transformed_initial_path


class RandomRBFDisplacementNet(nn.Module):
    """Frozen random local features with one trainable linear output layer."""

    def __init__(self, width: int, sigma: float, seed: int):
        super().__init__()
        rng = np.random.default_rng(int(seed))
        centers = (
            np.arange(int(width), dtype=np.float64)
            + 0.5
            + rng.uniform(-0.35, 0.35, size=int(width))
        ) / float(width)
        self.register_buffer(
            "centers", torch.as_tensor(np.sort(centers), dtype=base.DTYPE)
        )
        self.sigma = float(sigma)
        self.output = nn.Linear(int(width), base.DIMENSION, dtype=base.DTYPE)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        column = s.reshape(-1, 1)
        feature = torch.exp(
            -0.5 * ((column - self.centers.reshape(1, -1)) / self.sigma) ** 2
        )
        envelope = 4.0 * column * (1.0 - column)
        return envelope * self.output(feature)


class RandomFourierDisplacementNet(nn.Module):
    """Frozen low-bandwidth random Fourier features with a trainable readout."""

    def __init__(self, width: int, bandwidth: float, seed: int):
        super().__init__()
        rng = np.random.default_rng(int(seed))
        self.register_buffer(
            "frequency",
            torch.as_tensor(
                rng.normal(0.0, float(bandwidth), size=int(width)),
                dtype=base.DTYPE,
            ),
        )
        self.register_buffer(
            "phase",
            torch.as_tensor(
                rng.uniform(0.0, 2.0 * np.pi, size=int(width)),
                dtype=base.DTYPE,
            ),
        )
        self.scale = float(np.sqrt(2.0 / int(width)))
        self.output = nn.Linear(int(width), base.DIMENSION, dtype=base.DTYPE)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        column = s.reshape(-1, 1)
        feature = self.scale * torch.cos(
            column * self.frequency.reshape(1, -1) + self.phase.reshape(1, -1)
        )
        envelope = 4.0 * column * (1.0 - column)
        return envelope * self.output(feature)


class ExactActiveSubspace(nn.Module):
    """Map unconstrained network output to the exact Q10 invariant subspace."""

    def __init__(self, displacement: nn.Module):
        super().__init__()
        self.displacement = displacement
        self.register_buffer("q", torch.as_tensor(Q, dtype=base.DTYPE))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        raw = self.displacement(s)
        natural = raw @ self.q.T
        return natural[:, :2] @ self.q[:2, :]


class ArcLengthReparameterized(nn.Module):
    """Frozen coordinate-preserving map from uniform arc fraction to old parameter."""

    def __init__(self, reference: nn.Module, count: int):
        super().__init__()
        self.reference = reference
        grid = np.linspace(0.0, 1.0, int(count))
        points = base.model_coordinates(reference, grid)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        cumulative /= cumulative[-1]
        keep = np.concatenate(([True], np.diff(cumulative) > 1.0e-14))
        self.register_buffer(
            "arc", torch.as_tensor(cumulative[keep], dtype=base.DTYPE)
        )
        self.register_buffer(
            "parameter", torch.as_tensor(grid[keep], dtype=base.DTYPE)
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        value = torch.clamp(s.reshape(-1), 0.0, 1.0)
        right = torch.searchsorted(self.arc, value, right=True)
        right = torch.clamp(right, 1, self.arc.numel() - 1)
        left = right - 1
        arc0 = self.arc[left]
        arc1 = self.arc[right]
        weight = (value - arc0) / torch.clamp(arc1 - arc0, min=1.0e-15)
        old_s = (1.0 - weight) * self.parameter[left] + weight * self.parameter[right]
        return self.reference(old_s)


@dataclass(frozen=True)
class MLPConfig:
    alpha: float = 1.0e-3
    force_budget: int = 5000
    max_outer: int = 100
    width: int = 24
    depth: int = 2
    fourier_order: int = 0
    random_feature_last_layer_only: bool = False
    random_feature_kind: str = "tanh"
    random_feature_sigma: float = 0.12
    random_feature_bandwidth: float = 3.0
    exact_active_subspace: bool = False
    fixed_feature_seed: bool = True
    network_seed: int = 20260803
    sampling_jitter: float = 0.3
    sampling_seed: int = 20260802
    inner_iterations: int = 5
    maximum_line_search: int = 12
    gradient_tolerance: float = 1.0e-7
    function_tolerance: float = 1.0e-12
    acceptance_tolerance: float = 1.0e-10
    tangent_stencil_step: float = 0.002
    tangent_table_points: int = 257
    arclength_points: int = 2049
    arc_length_reparameterization: bool = True
    certificate_points: int = 501
    role: str = "nn_ivm_mlp_lbfgs"


def configuration_id(config: MLPConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def run_experiment(root: Path, config: MLPConfig) -> dict:
    if not np.isfinite(config.alpha) or config.alpha <= 0.0:
        raise ValueError("alpha must be one positive fixed scalar")
    root = Path(root).resolve()
    initial = transformed_initial_path(22)
    model = base.freeze(base.LinearEndpointPath(initial[0], initial[-1]).double())
    oracle = base.StreamForceOracle(base.SAMPLE_COUNT, int(config.force_budget))
    certificate_s = np.linspace(0.0, 1.0, int(config.certificate_points))

    trace: list[dict] = []
    path_history: list[np.ndarray] = []
    tangent_history: list[np.ndarray] = []
    speed_history: list[np.ndarray] = []
    tangent_energy_evaluations = 0

    def record(outer: int, accepted: bool, initial_objective: float,
               final_objective: float, gradient_norm: float,
               optimizer_iterations: int, optimizer_evaluations: int,
               normal_leakage: float, maximum_displacement: float) -> None:
        path, tangent, speed = base.path_geometry(model, certificate_s)
        path_history.append(path)
        tangent_history.append(tangent)
        speed_history.append(speed)
        trace.append({
            "outer": int(outer),
            "number_of_forces": oracle.total_calls,
            "counter_vector": oracle.calls.copy(),
            "accepted": bool(accepted),
            "initial_objective": float(initial_objective),
            "final_objective": float(final_objective),
            "objective_change": float(final_objective - initial_objective),
            "gradient_norm": float(gradient_norm),
            "optimizer_iterations": int(optimizer_iterations),
            "optimizer_evaluations": int(optimizer_evaluations),
            "normal_leakage": float(normal_leakage),
            "maximum_displacement": float(maximum_displacement),
        })

    record(0, True, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0)
    accepted_updates = 0
    stop_reason = "outer_limit"

    for outer in range(1, int(config.max_outer) + 1):
        if oracle.total_calls + base.SAMPLE_COUNT > config.force_budget:
            stop_reason = "budget"
            break
        fractions = base.stratified_fractions(
            base.SAMPLE_COUNT, outer, config.sampling_jitter, config.sampling_seed
        )
        if config.arc_length_reparameterization:
            reference = base.freeze(
                ArcLengthReparameterized(model, int(config.arclength_points)).double()
            )
            sample_s = fractions
        else:
            sample_s = base.arclength_parameters(
                model, fractions, int(config.arclength_points)
            )
            reference = base.freeze(model)
        reference_points = base.model_coordinates(reference, sample_s)
        try:
            reference_energies, _ = oracle.query(reference_points)
        except base.ForceBudgetReached:
            stop_reason = "budget"
            break
        initial_objective = float(np.mean(reference_energies))
        tangent_table = base.improved_tangent_table(
            reference,
            int(config.arclength_points),
            float(config.tangent_stencil_step),
            int(config.tangent_table_points),
        )
        if config.exact_active_subspace:
            table_grid = tangent_table.grid.detach().cpu().numpy()
            table_tangent = tangent_table.tangents.detach().cpu().numpy()
            tangent_natural = table_tangent @ Q.T
            tangent_natural[:, 2:] = 0.0
            table_tangent = tangent_natural @ Q
            table_tangent /= np.maximum(
                np.linalg.norm(table_tangent, axis=1, keepdims=True), 1.0e-15
            )
            tangent_table = base.FrozenTangentTable(
                table_grid, table_tangent
            ).double()
        tangent_energy_evaluations += 3 * int(config.tangent_table_points)
        seed = (
            int(config.network_seed)
            if config.fixed_feature_seed
            else int(config.network_seed) + outer
        )
        if config.random_feature_kind == "rbf":
            if not config.random_feature_last_layer_only:
                raise ValueError("RBF random features require last-layer-only training")
            displacement: nn.Module = RandomRBFDisplacementNet(
                int(config.width), float(config.random_feature_sigma), seed
            ).double()
        elif config.random_feature_kind == "fourier":
            if not config.random_feature_last_layer_only:
                raise ValueError("Fourier random features require last-layer-only training")
            displacement = RandomFourierDisplacementNet(
                int(config.width), float(config.random_feature_bandwidth), seed
            ).double()
        else:
            displacement = base.EndpointFixedDisplacementNet(
                int(config.width), int(config.depth), int(config.fourier_order), seed,
                last_layer_only=bool(config.random_feature_last_layer_only),
            ).double()
        if config.exact_active_subspace:
            displacement = ExactActiveSubspace(displacement).double()
        candidate = base.FrozenNormalUpdate(
            reference, tangent_table, displacement
        ).double()
        x0 = base.parameter_vector(displacement).cpu().numpy().astype(np.float64)
        best = {
            "objective": initial_objective,
            "vector": x0.copy(),
            "gradient_norm": np.inf,
            "maximum_displacement": 0.0,
        }
        evaluation_count = 0
        budget_hit = False

        def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal evaluation_count
            base.load_parameter_vector(
                displacement,
                torch.as_tensor(vector, dtype=base.DTYPE, device=base.DEVICE),
            )
            result = base.proximal_evaluation(
                candidate, reference_points, sample_s, oracle,
                float(config.alpha), backward=True,
            )
            gradient = base.gradient_vector(displacement).detach().cpu().numpy()
            gradient_norm = float(np.linalg.norm(gradient))
            evaluation_count += 1
            if result["objective"] < best["objective"]:
                best.update({
                    "objective": float(result["objective"]),
                    "vector": np.asarray(vector, dtype=np.float64).copy(),
                    "gradient_norm": gradient_norm,
                    "maximum_displacement": float(result["maximum_displacement"]),
                })
            return float(result["objective"]), gradient

        optimizer_iterations = 0
        try:
            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": int(config.inner_iterations),
                    "maxls": int(config.maximum_line_search),
                    "gtol": float(config.gradient_tolerance),
                    "ftol": float(config.function_tolerance),
                    "maxcor": 10,
                },
            )
            optimizer_iterations = int(result.nit)
        except base.ForceBudgetReached:
            budget_hit = True

        accepted = bool(
            np.isfinite(best["objective"])
            and best["objective"] <= initial_objective + config.acceptance_tolerance
            and np.any(best["vector"] != x0)
        )
        normal_leakage = np.inf
        maximum_displacement = float(best["maximum_displacement"])
        if accepted:
            base.load_parameter_vector(
                displacement,
                torch.as_tensor(best["vector"], dtype=base.DTYPE, device=base.DEVICE),
            )
            validation_s = np.linspace(0.0, 1.0, 257)[1:-1]
            reference_validation = base.model_coordinates(reference, validation_s)
            tangent_validation = tangent_table(
                torch.as_tensor(validation_s, dtype=base.DTYPE)
            ).detach().cpu().numpy()
            candidate_validation = base.model_coordinates(candidate, validation_s)
            delta = candidate_validation - reference_validation
            normal_leakage = float(
                np.max(np.abs(np.sum(delta * tangent_validation, axis=1)))
            )
            maximum_displacement = float(np.max(np.linalg.norm(delta, axis=1)))
            if normal_leakage > 1.0e-8:
                accepted = False
        if accepted:
            model = base.freeze(candidate)
            accepted_updates += 1
        else:
            model = reference

        record(
            outer, accepted, initial_objective, float(best["objective"]),
            float(best["gradient_norm"]), optimizer_iterations,
            evaluation_count, normal_leakage, maximum_displacement,
        )
        if budget_hit or oracle.total_calls >= config.force_budget:
            stop_reason = "budget"
            break

    paths = np.asarray(path_history)
    tangents = np.asarray(tangent_history)
    speeds = np.asarray(speed_history)
    certificate = base.dense_certificate(paths, tangents, speeds, certificate_s)
    number_of_forces = np.asarray(
        [row["number_of_forces"] for row in trace], dtype=np.int64
    )
    accepted = np.asarray([row["accepted"] for row in trace], dtype=bool)
    objective_change = np.asarray([row["objective_change"] for row in trace])
    inactive = np.asarray(certificate["inactive_coordinate_max"])
    theory_pass = bool(
        np.all(objective_change[accepted][1:] <= config.acceptance_tolerance)
        and np.all(np.diff(number_of_forces) >= 0)
        and np.max(inactive) <= 1.0e-8
    )

    cfg_id = configuration_id(config)
    output_directory = root / "runs" / config.role
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / f"q10_nn_ivm_mlp_lbfgs_{cfg_id}.npz"
    model_path = output.with_suffix(".model.pt")
    torch.save(model, model_path)
    np.savez_compressed(
        output,
        method=np.asarray(
            "NN-IVM-random-feature-LBFGS"
            if config.random_feature_last_layer_only
            else "NN-IVM-full-MLP-LBFGS"
        ),
        configuration_json=np.asarray(json.dumps(asdict(config), sort_keys=True)),
        number_of_forces=number_of_forces,
        counter_vectors=np.asarray([row["counter_vector"] for row in trace]),
        accepted=accepted,
        outer=np.asarray([row["outer"] for row in trace]),
        objective_change=objective_change,
        gradient_norm=np.asarray([row["gradient_norm"] for row in trace]),
        optimizer_iterations=np.asarray([row["optimizer_iterations"] for row in trace]),
        optimizer_evaluations=np.asarray([row["optimizer_evaluations"] for row in trace]),
        normal_leakage=np.asarray([row["normal_leakage"] for row in trace]),
        maximum_displacement=np.asarray([row["maximum_displacement"] for row in trace]),
        certificate_s=certificate_s,
        dense_path_history=paths,
        dense_i_perp=certificate["dense_i_perp"],
        dense_delta_e=certificate["dense_delta_e"],
        dense_barrier_error=certificate["dense_barrier_error"],
        inactive_coordinate_max=inactive,
        optimization_force_calls=np.asarray(oracle.total_calls),
        tangent_energy_evaluations=np.asarray(tangent_energy_evaluations),
        dense_information_enters_optimization=np.asarray(False),
        theory_audit_pass=np.asarray(theory_pass),
        model_path=np.asarray(str(model_path)),
        stop_reason=np.asarray(stop_reason),
    )
    summary = {
        "configuration": asdict(config),
        "configuration_id": cfg_id,
        "method": (
            "random-feature-last-layer-LBFGS"
            if config.random_feature_last_layer_only
            else "full-MLP-LBFGS"
        ),
        "output": str(output.relative_to(root)),
        "model": str(model_path.relative_to(root)),
        "optimization_force_calls": oracle.total_calls,
        "tangent_energy_evaluations": tangent_energy_evaluations,
        "accepted_updates": accepted_updates,
        "stop_reason": stop_reason,
        "final_dense_i_perp": float(certificate["dense_i_perp"][-1]),
        "minimum_dense_i_perp": float(np.min(certificate["dense_i_perp"])),
        "final_dense_barrier_error": float(certificate["dense_barrier_error"][-1]),
        "minimum_dense_barrier_error": float(np.min(certificate["dense_barrier_error"])),
        "maximum_inactive_coordinate": float(np.max(inactive)),
        "fixed_alpha": True,
        "unconstrained_delta": True,
        "lagrangian": False,
        "improved_tangent": True,
        "force_count_definition": "direct sum of the 20 sample-stream counters",
        "theory_audit_pass": theory_pass,
    }
    output.with_suffix(".audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
