"""Continuous neural-path utilities for the LEPS comparison.

The neural representation fixes the two endpoints by construction,

    phi(s; theta) = phi_bar(s) + s (s - 1) delta(s; theta),

and obtains the path tangent from torch autograd.  Equal-arclength parameter
values are used only to choose collocation/diagnostic samples; the neural path
itself is never reparameterized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

from leps_core import (
    PRODUCT,
    REACTANT,
    flat_to_physical,
    leps_energy_gradient,
)


TORCH_DTYPE = torch.float64


class FixedEndpointPathNet(nn.Module):
    """Small tanh MLP with endpoints imposed analytically."""

    def __init__(
        self,
        width: int = 64,
        depth: int = 3,
        fourier_order: int = 4,
        seed: int = 20260725,
    ):
        super().__init__()
        if width < 2:
            raise ValueError("width must be at least 2")
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if fourier_order < 0:
            raise ValueError("fourier_order must be non-negative")
        self.width = int(width)
        self.depth = int(depth)
        self.fourier_order = int(fourier_order)
        self.seed = int(seed)

        layers: list[nn.Module] = []
        input_width = 1 + 2 * self.fourier_order
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            for _ in range(self.depth):
                linear = nn.Linear(input_width, self.width, dtype=TORCH_DTYPE)
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                layers.extend((linear, nn.Tanh()))
                input_width = self.width
            final = nn.Linear(input_width, 2, dtype=TORCH_DTYPE)
            # delta == 0 gives the exact common straight-line initial path.
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            layers.append(final)
        self.delta_network = nn.Sequential(*layers)
        self.register_buffer(
            "reactant",
            torch.as_tensor(REACTANT, dtype=TORCH_DTYPE).reshape(1, 2),
        )
        self.register_buffer(
            "product",
            torch.as_tensor(PRODUCT, dtype=TORCH_DTYPE).reshape(1, 2),
        )

    def input_features(self, s: torch.Tensor) -> torch.Tensor:
        s = torch.as_tensor(s, dtype=TORCH_DTYPE, device=self.reactant.device)
        s = s.reshape(-1, 1)
        normalized_s = 2.0 * s - 1.0
        features = [normalized_s]
        for frequency in range(1, self.fourier_order + 1):
            phase = float(frequency) * np.pi * normalized_s
            features.extend((torch.sin(phase), torch.cos(phase)))
        return torch.cat(features, dim=1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = torch.as_tensor(s, dtype=TORCH_DTYPE, device=self.reactant.device)
        s = s.reshape(-1, 1)
        straight = (1.0 - s) * self.reactant + s * self.product
        correction = (
            s * (s - 1.0) * self.delta_network(self.input_features(s))
        )
        return straight + correction


class ScalarEndpointUpdateNet(nn.Module):
    """Scalar neural displacement that vanishes at both path endpoints."""

    def __init__(
        self,
        width: int = 32,
        depth: int = 2,
        fourier_order: int = 6,
        seed: int = 20260725,
        max_amplitude: float = 1.0,
    ):
        super().__init__()
        if width < 2:
            raise ValueError("width must be at least 2")
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if fourier_order < 0:
            raise ValueError("fourier_order must be non-negative")
        self.width = int(width)
        self.depth = int(depth)
        self.fourier_order = int(fourier_order)
        self.seed = int(seed)
        self.max_amplitude = float(max_amplitude)
        if not np.isfinite(self.max_amplitude) or self.max_amplitude <= 0.0:
            raise ValueError("max_amplitude must be finite and positive")
        layers: list[nn.Module] = []
        input_width = 1 + 2 * self.fourier_order
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            for _ in range(self.depth):
                linear = nn.Linear(input_width, self.width, dtype=TORCH_DTYPE)
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
                layers.extend((linear, nn.Tanh()))
                input_width = self.width
            final = nn.Linear(input_width, 1, dtype=TORCH_DTYPE)
            # Every neural IVM subproblem starts exactly at phi_k.
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            layers.append(final)
        self.network = nn.Sequential(*layers)

    def input_features(self, s: torch.Tensor) -> torch.Tensor:
        s = torch.as_tensor(s, dtype=TORCH_DTYPE)
        s = s.reshape(-1, 1)
        normalized_s = 2.0 * s - 1.0
        features = [normalized_s]
        for frequency in range(1, self.fourier_order + 1):
            phase = float(frequency) * np.pi * normalized_s
            features.extend((torch.sin(phase), torch.cos(phase)))
        return torch.cat(features, dim=1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = torch.as_tensor(s, dtype=TORCH_DTYPE).reshape(-1, 1)
        envelope = 4.0 * s * (1.0 - s)
        return (
            self.max_amplitude
            * envelope
            * torch.tanh(self.network(self.input_features(s)))
        )


class ProjectedUpdatePath(nn.Module):
    """Continuous IVM update phi_k + P_perp,k delta_theta.

    The scalar trainable field is multiplied by the unit normal of the frozen
    base path.  Therefore every subproblem trial, not merely its collocation
    points, is an exactly normal path update with fixed endpoints.
    """

    def __init__(self, base_path: nn.Module, update: ScalarEndpointUpdateNet):
        super().__init__()
        self.base_path = base_path
        self.update = update
        for parameter in self.base_path.parameters():
            parameter.requires_grad_(False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s_input = torch.as_tensor(s, dtype=TORCH_DTYPE).reshape(-1)
        inherited_grad = s_input.requires_grad
        caller_grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            s_work = (
                s_input
                if inherited_grad
                else s_input.detach().clone().requires_grad_(True)
            )
            base = self.base_path(s_work)
            derivatives = []
            for component in range(2):
                derivative = torch.autograd.grad(
                    base[:, component].sum(),
                    s_work,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                derivatives.append(derivative)
            first = torch.stack(derivatives, dim=1)
            speed = torch.linalg.norm(first, dim=1)
            if torch.any(speed <= 1e-12):
                raise RuntimeError(
                    "Base neural path has a near-zero autograd tangent"
                )
            tangent = first / speed[:, None]
            normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=1)
            candidate = base + normal * self.update(s_work)
        return (
            candidate
            if inherited_grad or caller_grad_enabled
            else candidate.detach()
        )


def parameter_vector(model: nn.Module) -> np.ndarray:
    vector = torch.nn.utils.parameters_to_vector(model.parameters())
    return vector.detach().cpu().numpy().astype(np.float64, copy=True)


def load_parameter_vector(model: nn.Module, vector: np.ndarray) -> None:
    # Use an owning copy: vector_to_parameters installs views into this tensor,
    # so sharing the NumPy checkpoint would let later optimizer steps mutate
    # the supposedly immutable rollback state.
    tensor = torch.tensor(
        np.asarray(vector, dtype=np.float64).copy(),
        dtype=TORCH_DTYPE,
        device=next(model.parameters()).device,
    )
    torch.nn.utils.vector_to_parameters(tensor, model.parameters())


def model_coordinates(
    model: FixedEndpointPathNet,
    s_values: Iterable[float],
) -> np.ndarray:
    s_tensor = torch.as_tensor(
        np.asarray(s_values, dtype=np.float64),
        dtype=TORCH_DTYPE,
    )
    with torch.no_grad():
        return model(s_tensor).detach().cpu().numpy()


def coordinates_and_derivatives(
    model: FixedEndpointPathNet,
    s_values: Iterable[float],
    *,
    create_graph: bool = False,
    second: bool = False,
):
    """Return coordinates, dphi/ds, and optionally d2phi/ds2."""
    s_tensor = torch.as_tensor(
        np.asarray(s_values, dtype=np.float64),
        dtype=TORCH_DTYPE,
    ).reshape(-1)
    s_tensor = s_tensor.clone().detach().requires_grad_(True)
    coordinates = model(s_tensor)
    first_components = []
    for component in range(2):
        derivative = torch.autograd.grad(
            coordinates[:, component].sum(),
            s_tensor,
            create_graph=create_graph or second,
            retain_graph=True,
        )[0]
        first_components.append(derivative)
    first = torch.stack(first_components, dim=1)
    if not second:
        return s_tensor, coordinates, first

    second_components = []
    for component in range(2):
        derivative = torch.autograd.grad(
            first[:, component].sum(),
            s_tensor,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        second_components.append(derivative)
    second_derivative = torch.stack(second_components, dim=1)
    return s_tensor, coordinates, first, second_derivative


def normalized_autograd_tangent(
    model: FixedEndpointPathNet,
    s_values: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, coordinates, first = coordinates_and_derivatives(model, s_values)
    speed = torch.linalg.norm(first, dim=1)
    if torch.any(speed <= 1e-12):
        raise RuntimeError("Neural path has a near-zero autograd tangent")
    tangent = first / speed[:, None]
    return (
        coordinates.detach().cpu().numpy(),
        tangent.detach().cpu().numpy(),
        speed.detach().cpu().numpy(),
    )


def equal_arclength_parameters(
    model: FixedEndpointPathNet,
    count: int,
    *,
    dense_count: int = 4097,
    include_endpoints: bool = False,
) -> np.ndarray:
    """Choose parameter values whose coordinates are uniform in arclength."""
    if count < 1:
        raise ValueError("count must be positive")
    if dense_count < max(257, 8 * count):
        raise ValueError("dense_count is too small for arclength inversion")
    dense_s = np.linspace(0.0, 1.0, int(dense_count))
    dense_path = model_coordinates(model, dense_s)
    increments = np.linalg.norm(np.diff(dense_path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 1e-12:
        raise RuntimeError("Neural path has invalid arclength")
    cumulative /= cumulative[-1]
    targets = (
        np.linspace(0.0, 1.0, int(count))
        if include_endpoints
        else np.linspace(0.0, 1.0, int(count) + 2)[1:-1]
    )
    return np.interp(targets, cumulative, dense_s)


def arclength_parameters_for_fractions(
    model: FixedEndpointPathNet,
    fractions: Iterable[float],
    *,
    dense_count: int = 4097,
) -> np.ndarray:
    """Invert the neural path's arclength map at requested fractions."""
    fractions = np.asarray(fractions, dtype=np.float64).reshape(-1)
    if fractions.size == 0:
        raise ValueError("fractions must not be empty")
    if np.any(~np.isfinite(fractions)):
        raise ValueError("fractions must be finite")
    if np.any(fractions < 0.0) or np.any(fractions > 1.0):
        raise ValueError("fractions must lie in [0, 1]")
    if np.any(np.diff(fractions) < 0.0):
        raise ValueError("fractions must be sorted")
    if dense_count < max(257, 8 * fractions.size):
        raise ValueError("dense_count is too small for arclength inversion")
    dense_s = np.linspace(0.0, 1.0, int(dense_count))
    dense_path = model_coordinates(model, dense_s)
    increments = np.linalg.norm(np.diff(dense_path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 1e-12:
        raise RuntimeError("Neural path has invalid arclength")
    cumulative /= cumulative[-1]
    return np.interp(fractions, cumulative, dense_s)


def stratified_jitter_fractions(
    count: int,
    iteration: int,
    *,
    jitter: float,
    seed: int,
) -> np.ndarray:
    """Return reproducible small perturbations of equal-arclength strata.

    The base fractions are j/(count+1).  Each point moves by at most
    ``jitter/(count+1)`` and therefore remains ordered for jitter < 0.5.
    """
    count = int(count)
    iteration = int(iteration)
    jitter = float(jitter)
    if count < 1:
        raise ValueError("count must be positive")
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    if not 0.0 <= jitter < 0.5:
        raise ValueError("jitter must be in [0, 0.5)")
    base = np.arange(1, count + 1, dtype=np.float64) / float(count + 1)
    if jitter == 0.0 or iteration == 0:
        return base
    sequence = np.random.SeedSequence([int(seed), iteration, count])
    generator = np.random.default_rng(sequence)
    perturbation = generator.uniform(-jitter, jitter, size=count)
    return base + perturbation / float(count + 1)


def equal_arclength_sample(
    model: FixedEndpointPathNet,
    image_count: int = 22,
    *,
    dense_count: int = 4097,
) -> tuple[np.ndarray, np.ndarray]:
    s_values = equal_arclength_parameters(
        model,
        image_count,
        dense_count=dense_count,
        include_endpoints=True,
    )
    path = model_coordinates(model, s_values)
    path[0] = REACTANT
    path[-1] = PRODUCT
    return s_values, path


def resample_polyline_equal_arclength(
    path: np.ndarray,
    count: int | None = None,
) -> np.ndarray:
    """Resample a geometric polyline at equal fractions of its arclength."""
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 2:
        raise ValueError("path must have shape (n >= 2, 2)")
    output_count = path.shape[0] if count is None else int(count)
    if output_count < 2:
        raise ValueError("count must be at least 2")
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 1e-14:
        raise ValueError("path has negligible arclength")
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-14))
    cumulative = cumulative[keep] / cumulative[-1]
    unique_path = path[keep]
    targets = np.linspace(0.0, 1.0, output_count)
    result = np.column_stack(
        [
            np.interp(targets, cumulative, unique_path[:, coordinate])
            for coordinate in range(2)
        ]
    )
    result[0] = path[0]
    result[-1] = path[-1]
    return result


@dataclass
class FitResult:
    rms_error: float
    max_error: float
    loss: float
    adam_steps: int
    lbfgs_steps: int


def _fit_loss(
    model: FixedEndpointPathNet,
    sample_s: torch.Tensor,
    targets: torch.Tensor,
    target_tangents: torch.Tensor | None,
    tangent_weight: float,
    smoothness_weight: float,
    smooth_s: np.ndarray,
) -> torch.Tensor:
    column_s = sample_s.reshape(-1, 1)
    straight = (
        (1.0 - column_s) * model.reactant
        + column_s * model.product
    )
    boundary_factor = column_s * (column_s - 1.0)
    target_delta = (targets - straight) / boundary_factor
    predicted_delta = model.delta_network(model.input_features(column_s))
    # Fitting the latent endpoint-compatible correction prevents the small
    # s(1-s) factor from underweighting the samples nearest the endpoints.
    data_loss = torch.mean(
        torch.sum((predicted_delta - target_delta) ** 2, dim=1)
    )
    tangent_loss = torch.zeros((), dtype=TORCH_DTYPE)
    if target_tangents is not None and tangent_weight > 0.0:
        _, _, first = coordinates_and_derivatives(
            model,
            sample_s.detach().cpu().numpy(),
            create_graph=True,
        )
        unit_tangent = first / torch.clamp(
            torch.linalg.norm(first, dim=1, keepdim=True),
            min=1e-12,
        )
        tangent_loss = torch.mean(
            torch.sum((unit_tangent - target_tangents) ** 2, dim=1)
        )
    if smoothness_weight <= 0.0:
        return data_loss + float(tangent_weight) * tangent_loss
    _, _, _, second = coordinates_and_derivatives(
        model,
        smooth_s,
        create_graph=True,
        second=True,
    )
    curvature_loss = torch.mean(torch.sum(second**2, dim=1))
    return (
        data_loss
        + float(tangent_weight) * tangent_loss
        + float(smoothness_weight) * curvature_loss
    )


def fit_path_targets(
    model: FixedEndpointPathNet,
    s_values: np.ndarray,
    targets: np.ndarray,
    *,
    target_tangents: np.ndarray | None = None,
    tangent_weight: float = 0.0,
    smoothness_weight: float = 1e-9,
    adam_steps: int = 80,
    adam_lr: float = 2e-2,
    lbfgs_steps: int = 80,
    polish_lbfgs_steps: int = 0,
    polish_tangent_weight: float = 0.0,
) -> FitResult:
    """Fit a new neural path to one IVM target band without oracle calls."""
    s_values = np.asarray(s_values, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    if targets.shape[0] != s_values.size:
        raise ValueError("s_values and targets have inconsistent lengths")
    sample_s = torch.as_tensor(s_values, dtype=TORCH_DTYPE)
    target_tensor = torch.as_tensor(targets, dtype=TORCH_DTYPE)
    tangent_tensor = None
    if target_tangents is not None:
        target_tangents = np.asarray(
            target_tangents,
            dtype=np.float64,
        ).reshape(-1, 2)
        if target_tangents.shape != targets.shape:
            raise ValueError(
                "target_tangents and targets have inconsistent shapes"
            )
        tangent_norm = np.linalg.norm(target_tangents, axis=1)
        if np.any(tangent_norm <= 1e-12):
            raise ValueError("target_tangents contain a near-zero vector")
        target_tangents = target_tangents / tangent_norm[:, None]
        tangent_tensor = torch.as_tensor(
            target_tangents,
            dtype=TORCH_DTYPE,
        )
    smooth_s = np.linspace(0.0, 1.0, max(65, 3 * s_values.size + 2))

    if adam_steps > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(adam_lr))
        for _ in range(int(adam_steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = _fit_loss(
                model,
                sample_s,
                target_tensor,
                tangent_tensor,
                tangent_weight,
                smoothness_weight,
                smooth_s,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite neural path fitting loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()

    lbfgs_iterations = 0
    if lbfgs_steps > 0:
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=int(lbfgs_steps),
            max_eval=max(2 * int(lbfgs_steps), int(lbfgs_steps) + 5),
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
            history_size=30,
            line_search_fn="strong_wolfe",
        )

        def closure():
            nonlocal lbfgs_iterations
            optimizer.zero_grad(set_to_none=True)
            loss = _fit_loss(
                model,
                sample_s,
                target_tensor,
                tangent_tensor,
                tangent_weight,
                smoothness_weight,
                smooth_s,
            )
            loss.backward()
            lbfgs_iterations += 1
            return loss

        optimizer.step(closure)

    if polish_lbfgs_steps > 0:
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=int(polish_lbfgs_steps),
            max_eval=max(
                2 * int(polish_lbfgs_steps),
                int(polish_lbfgs_steps) + 5,
            ),
            tolerance_grad=1e-13,
            tolerance_change=1e-15,
            history_size=30,
            line_search_fn="strong_wolfe",
        )

        def polish_closure():
            nonlocal lbfgs_iterations
            optimizer.zero_grad(set_to_none=True)
            loss = _fit_loss(
                model,
                sample_s,
                target_tensor,
                tangent_tensor,
                polish_tangent_weight,
                smoothness_weight,
                smooth_s,
            )
            loss.backward()
            lbfgs_iterations += 1
            return loss

        optimizer.step(polish_closure)

    with torch.no_grad():
        error = model(sample_s) - target_tensor
        point_errors = torch.linalg.norm(error, dim=1)
        rms = torch.sqrt(torch.mean(point_errors**2))
        maximum = torch.max(point_errors)
    final_loss = float(
        _fit_loss(
            model,
            sample_s,
            target_tensor,
            tangent_tensor,
            tangent_weight,
            smoothness_weight,
            smooth_s,
        ).detach()
    )
    return FitResult(
        rms_error=float(rms),
        max_error=float(maximum),
        loss=final_loss,
        adam_steps=int(adam_steps),
        lbfgs_steps=int(lbfgs_iterations),
    )


def spacing_from_sample(path: np.ndarray) -> dict:
    path = np.asarray(path, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    mean = float(np.mean(lengths))
    return {
        "spring_force_max_unit": float(np.max(np.abs(np.diff(lengths)))),
        "segment_length_cv": (
            float(np.std(lengths) / mean) if mean > 0.0 else np.inf
        ),
        "segment_length_min": float(np.min(lengths)),
        "segment_length_mean": mean,
        "segment_length_max": float(np.max(lengths)),
    }


def dense_neural_metrics(
    model: FixedEndpointPathNet,
    *,
    dense_count: int = 4097,
    sample_images: int = 22,
) -> dict:
    """Continuous metrics using the autograd tangent of the neural path."""
    s_values = np.linspace(0.0, 1.0, int(dense_count))
    coordinates, tangent, speed = normalized_autograd_tangent(model, s_values)
    energies, gradient = leps_energy_gradient(
        coordinates[:, 0],
        coordinates[:, 1],
    )
    forces = -np.asarray(gradient, dtype=np.float64)
    force_normal = forces - np.sum(
        forces * tangent,
        axis=1,
        keepdims=True,
    ) * tangent
    normal_norm = np.linalg.norm(force_normal, axis=1)
    i_perp = float(np.trapz(normal_norm**2 * speed, s_values))
    path_length = float(np.trapz(speed, s_values))
    sample_s, sampled_path = equal_arclength_sample(
        model,
        image_count=sample_images,
        dense_count=dense_count,
    )
    metrics = {
        "dense_i_perp": i_perp,
        "dense_fmax": float(np.max(normal_norm[1:-1])),
        "dense_barrier": float(np.max(energies) - energies[0]),
        "path_length": path_length,
        "minimum_speed": float(np.min(speed)),
        "diagnostic_points": int(dense_count),
        "sample_s": sample_s,
        "sample_path": sampled_path,
    }
    metrics.update(spacing_from_sample(sampled_path))
    return metrics


def dense_linear_metrics(
    path: np.ndarray,
    *,
    subdivisions: int = 200,
) -> dict:
    """Dense metrics for a piecewise-linear discrete path.

    Each open segment uses its exact line tangent.  Vertices have measure zero
    in the continuous integral and are included only for the barrier maximum.
    """
    physical = flat_to_physical(path)
    total_i_perp = 0.0
    maximum_force = 0.0
    maximum_energy = -np.inf
    initial_energy = None
    for left, right in zip(physical[:-1], physical[1:]):
        displacement = right - left
        length = float(np.linalg.norm(displacement))
        if length <= 1e-14:
            raise ValueError("Piecewise-linear path contains a zero segment")
        tangent = displacement / length
        fractions = np.linspace(0.0, 1.0, int(subdivisions) + 1)
        points = left[None, :] + fractions[:, None] * displacement[None, :]
        energies, gradient = leps_energy_gradient(points[:, 0], points[:, 1])
        forces = -np.asarray(gradient, dtype=np.float64)
        normal = forces - np.sum(
            forces * tangent[None, :],
            axis=1,
            keepdims=True,
        ) * tangent[None, :]
        normal_norm = np.linalg.norm(normal, axis=1)
        total_i_perp += float(
            np.trapz(normal_norm**2, dx=length / float(subdivisions))
        )
        maximum_force = max(maximum_force, float(np.max(normal_norm)))
        maximum_energy = max(maximum_energy, float(np.max(energies)))
        if initial_energy is None:
            initial_energy = float(energies[0])
    metrics = {
        "dense_i_perp": float(total_i_perp),
        "dense_fmax": float(maximum_force),
        "dense_barrier": float(maximum_energy - initial_energy),
        "path_length": float(
            np.sum(np.linalg.norm(np.diff(physical, axis=0), axis=1))
        ),
        "minimum_speed": np.nan,
        "diagnostic_points": int(
            (physical.shape[0] - 1) * int(subdivisions) + 1
        ),
        "sample_s": np.linspace(0.0, 1.0, physical.shape[0]),
        "sample_path": physical,
    }
    metrics.update(spacing_from_sample(physical))
    return metrics
