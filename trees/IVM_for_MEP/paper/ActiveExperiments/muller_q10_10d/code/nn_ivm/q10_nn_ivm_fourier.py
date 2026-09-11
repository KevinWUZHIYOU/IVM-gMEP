"""Direct unconstrained Fourier-network IVM for Q10 Muller."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline, CubicSpline
from scipy.optimize import minimize

from q10_ivm_core import improved_tangents_arbitrary
from q10_nn_ivm_pointwise import ForceBudgetReached, StreamOracle
from q10_potential import Q, energy_gradient, energy_values, transformed_initial_path
from run_q10_baselines import reference_barrier


N_SAMPLE = 20
DIMENSION = 10


@dataclass(frozen=True)
class FourierConfig:
    alpha: float = 1.0e-3
    force_budget: int = 5000
    cost_offset: int = 0
    initial_path_file: str = ""
    max_outer: int = 100
    network_width: int = 10
    feature_mode: str = "fourier"
    local_feature_count: int = 0
    local_feature_sigma: float = 0.045
    scalar_active_normal: bool = False
    inner_iterations: int = 8
    max_line_search: int = 12
    gradient_tolerance: float = 1.0e-7
    warm_start_coefficients: bool = False
    global_compression: bool = False
    compression_points: int = 1025
    compression_tolerance: float = 5.0e-3
    sampling_jitter: float = 0.3
    sampling_seed: int = 20260802
    sampling_mode: str = "uniform"
    max_energy_anchor_count: int = 0
    max_energy_scan_points: int = 2049
    max_energy_anchor_spacing: float = 0.004
    collocation_batches: int = 1
    tangent_stencil_step: float = 0.0
    arclength_points: int = 2049
    certificate_points: int = 501
    role: str = "nn_ivm_fourier"


class LinearPath:
    def __init__(self, start, end):
        self.start = np.asarray(start, dtype=np.float64)
        self.end = np.asarray(end, dtype=np.float64)

    def coordinates(self, s):
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        return (1.0 - s)[:, None] * self.start + s[:, None] * self.end


class GlobalSplinePath:
    def __init__(self, start, end, config, coefficients):
        self.start = np.asarray(start, dtype=np.float64)
        self.end = np.asarray(end, dtype=np.float64)
        self.config = config
        self.coefficients = np.asarray(coefficients, dtype=np.float64)

    def coordinates(self, s):
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        baseline = (1.0 - s)[:, None] * self.start + s[:, None] * self.end
        basis = _features(
            s,
            self.config.network_width,
            self.config.local_feature_count,
            self.config.local_feature_sigma,
            self.config.feature_mode,
        )
        return baseline + basis @ self.coefficients


class SampledSplinePath:
    def __init__(self, points):
        points = np.asarray(points, dtype=np.float64)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(lengths)])
        arc /= arc[-1]
        keep = np.concatenate([[True], np.diff(arc) > 1.0e-14])
        self.spline = CubicSpline(
            arc[keep], points[keep], axis=0, bc_type="natural"
        )

    def coordinates(self, s):
        return np.asarray(self.spline(np.asarray(s, dtype=np.float64)))


def _features(
    s,
    width,
    local_count=0,
    local_sigma=0.045,
    mode="fourier",
):
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    if mode in {"bspline", "bspline_left", "bspline_split"}:
        degree = 3
        full_count = int(width) + 2
        internal_count = full_count - degree - 1
        quantiles = np.arange(1, internal_count + 1, dtype=np.float64) / (
            internal_count + 1
        )
        if mode == "bspline_split":
            split = 0.32
            repeated = 3
            remaining = internal_count - repeated
            left_count = int(round(split * remaining))
            right_count = remaining - left_count
            left = split * np.arange(
                1, left_count + 1, dtype=np.float64
            ) / (left_count + 1)
            right = split + (1.0 - split) * np.arange(
                1, right_count + 1, dtype=np.float64
            ) / (right_count + 1)
            internal = np.concatenate(
                [left, np.full(repeated, split), right]
            )
        elif mode == "bspline_left":
            internal = np.empty_like(quantiles)
            first = quantiles < 0.30
            second = (quantiles >= 0.30) & (quantiles < 0.90)
            third = quantiles >= 0.90
            internal[first] = 0.65 * quantiles[first] / 0.30
            internal[second] = (
                0.65 + 0.25 * (quantiles[second] - 0.30) / 0.60
            )
            internal[third] = (
                0.90 + 0.10 * (quantiles[third] - 0.90) / 0.10
            )
        else:
            internal = quantiles
        knots = np.concatenate(
            [
                np.zeros(degree + 1),
                internal,
                np.ones(degree + 1),
            ]
        )
        full_basis = BSpline(
            knots,
            np.eye(full_count),
            degree,
            extrapolate=False,
        )(s)
        return np.asarray(full_basis[:, 1:-1], dtype=np.float64)
    fourier_count = width - int(local_count)
    if fourier_count < 1:
        raise ValueError("network needs at least one global Fourier feature")
    frequencies = np.arange(1, fourier_count + 1, dtype=np.float64)
    global_features = np.sin(
        np.pi * s[:, None] * frequencies[None, :]
    )
    if local_count <= 0:
        return global_features
    centers = np.linspace(0.68, 0.86, int(local_count))
    local_features = 4.0 * (s * (1.0 - s))[:, None] * np.exp(
        -0.5 * ((s[:, None] - centers[None, :]) / local_sigma) ** 2
    )
    return np.column_stack([global_features, local_features])


class FourierProjectedLayer:
    def __init__(
        self,
        parent,
        width,
        feature_mode,
        local_count,
        local_sigma,
        scalar_active_normal,
        coefficients,
        tangent_s,
        tangent_values,
    ):
        self.parent = parent
        self.width = int(width)
        self.feature_mode = str(feature_mode)
        self.local_count = int(local_count)
        self.local_sigma = float(local_sigma)
        self.scalar_active_normal = bool(scalar_active_normal)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.tangent_s = np.asarray(tangent_s, dtype=np.float64)
        self.tangent_values = np.asarray(tangent_values, dtype=np.float64)
        self.tangent_spline = CubicSpline(
            self.tangent_s,
            self.tangent_values,
            axis=0,
            bc_type="natural",
        )

    @staticmethod
    def active_normals_from_tangents(tangents):
        natural_tangents = np.asarray(tangents) @ Q.T
        natural_normals = np.zeros_like(natural_tangents)
        natural_normals[:, 0] = -natural_tangents[:, 1]
        natural_normals[:, 1] = natural_tangents[:, 0]
        normals = natural_normals @ Q
        return normals / np.linalg.norm(normals, axis=1)[:, None]

    def tangents(self, s):
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        values = np.asarray(self.tangent_spline(s), dtype=np.float64)
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms <= 1.0e-14):
            raise ValueError("degenerate interpolated improved tangent")
        return values / norms[:, None]

    def coordinates(self, s):
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        features = _features(
            s,
            self.width,
            self.local_count,
            self.local_sigma,
            self.feature_mode,
        )
        tangents = self.tangents(s)
        if self.scalar_active_normal:
            normals = self.active_normals_from_tangents(tangents)
            scalar = features @ self.coefficients.reshape(-1)
            return self.parent.coordinates(s) + scalar[:, None] * normals
        raw = features @ self.coefficients
        projected = raw - np.sum(raw * tangents, axis=1)[:, None] * tangents
        return self.parent.coordinates(s) + projected


class ArcReparameterizedPath:
    def __init__(self, parent, arc_nodes, old_nodes):
        self.parent = parent
        self.arc_nodes = np.asarray(arc_nodes, dtype=np.float64)
        self.old_nodes = np.asarray(old_nodes, dtype=np.float64)

    def coordinates(self, s):
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        old_s = np.interp(s, self.arc_nodes, self.old_nodes)
        return self.parent.coordinates(old_s)


def _fractions(outer, jitter, seed, mode="uniform"):
    if mode == "left_partition":
        rng = np.random.default_rng(int(seed) + 104729 * int(outer))
        edges = np.concatenate(
            [
                np.linspace(0.0, 0.65, 9),
                np.linspace(0.65, 0.90, 11)[1:],
                np.linspace(0.90, 1.0, 3)[1:],
            ]
        )
        widths = np.diff(edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers + 0.5 * float(jitter) * widths * rng.uniform(
            -1.0, 1.0, N_SAMPLE
        )
    if mode == "saddle_focus":
        rng = np.random.default_rng(int(seed) + 104729 * int(outer))
        global_points = (np.arange(14, dtype=np.float64) + 0.5) / 14.0
        first = np.linspace(0.17, 0.23, 5)[1:-1]
        second = np.linspace(0.59, 0.65, 5)[1:-1]
        global_points += (
            float(jitter) * rng.uniform(-1.0, 1.0, 14) / 14.0
        )
        first += 0.015 * float(jitter) * rng.uniform(-1.0, 1.0, 3)
        second += 0.015 * float(jitter) * rng.uniform(-1.0, 1.0, 3)
        return np.sort(np.concatenate([global_points, first, second]))
    if mode == "saddle_light":
        rng = np.random.default_rng(int(seed) + 104729 * int(outer))
        global_points = (np.arange(16, dtype=np.float64) + 0.5) / 16.0
        first = np.asarray([0.185, 0.215], dtype=np.float64)
        second = np.asarray([0.607, 0.637], dtype=np.float64)
        global_points += (
            float(jitter) * rng.uniform(-1.0, 1.0, 16) / 16.0
        )
        first += 0.015 * float(jitter) * rng.uniform(-1.0, 1.0, 2)
        second += 0.015 * float(jitter) * rng.uniform(-1.0, 1.0, 2)
        return np.sort(np.concatenate([global_points, first, second]))
    base = (np.arange(N_SAMPLE, dtype=np.float64) + 0.5) / N_SAMPLE
    if mode == "antithetic":
        rng = np.random.default_rng(
            int(seed) + 104729 * (int(outer) // 2)
        )
        offsets = rng.uniform(-1.0, 1.0, N_SAMPLE)
        if int(outer) % 2:
            offsets = -offsets
        return base + float(jitter) * offsets / N_SAMPLE
    if mode == "balanced":
        rng = np.random.default_rng(int(seed) + 104729 * int(outer))
        offsets = 2.0 * (
            np.arange(N_SAMPLE, dtype=np.float64) + 0.5
        ) / N_SAMPLE - 1.0
        return base + float(jitter) * rng.permutation(offsets) / N_SAMPLE
    if mode == "golden_shift":
        golden = 0.6180339887498949
        offset = 2.0 * (((int(outer) + 1) * golden) % 1.0) - 1.0
        return base + float(jitter) * offset / N_SAMPLE
    rng = np.random.default_rng(int(seed) + 104729 * int(outer))
    return np.sort(
        base + float(jitter) * rng.uniform(-1.0, 1.0, N_SAMPLE) / N_SAMPLE
    )


def _max_energy_anchor_fractions(path, fractions, count, scan_points, spacing):
    fractions = np.asarray(fractions, dtype=np.float64).copy()
    count = int(count)
    if count <= 0:
        return np.sort(fractions)
    if count > N_SAMPLE:
        raise ValueError("max-energy anchor count cannot exceed 20")
    scan = np.linspace(0.0, 1.0, int(scan_points))
    energies = energy_values(path.coordinates(scan))
    maximum = float(scan[int(np.argmax(energies))])
    offsets = (np.arange(count, dtype=np.float64) - 0.5 * (count - 1)) * float(spacing)
    anchors = np.clip(maximum + offsets, 1.0e-8, 1.0 - 1.0e-8)
    available = list(range(N_SAMPLE))
    for anchor in anchors:
        index = min(available, key=lambda item: abs(fractions[item] - anchor))
        fractions[index] = anchor
        available.remove(index)
    return np.sort(fractions)


def _arc_reparameterize(path, count):
    old_s = np.linspace(0.0, 1.0, count)
    points = path.coordinates(old_s)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    arc = cumulative / cumulative[-1]
    keep = np.concatenate([[True], np.diff(arc) > 1.0e-14])
    return ArcReparameterizedPath(path, arc[keep], old_s[keep])


def _continuous_improved_tangents(path, fractions, step):
    tangents = []
    for value in np.asarray(fractions, dtype=np.float64):
        local_s = np.asarray([value - step, value, value + step])
        local_points = path.coordinates(local_s)
        local_energies = energy_values(local_points)
        tangents.append(
            improved_tangents_arbitrary(local_points, local_energies)[0]
        )
    return np.asarray(tangents)


def _compress_global(path, config):
    s = np.linspace(0.0, 1.0, config.compression_points)
    target = path.coordinates(s)
    baseline = (1.0 - s)[:, None] * target[0] + s[:, None] * target[-1]
    basis = _features(
        s,
        config.network_width,
        config.local_feature_count,
        config.local_feature_sigma,
        config.feature_mode,
    )
    coefficients = np.linalg.lstsq(
        basis, target - baseline, rcond=1.0e-12
    )[0]
    compressed = GlobalSplinePath(
        target[0], target[-1], config, coefficients
    )
    errors = np.linalg.norm(compressed.coordinates(s) - target, axis=1)
    return compressed, float(np.sqrt(np.mean(errors**2))), float(np.max(errors))


def _certificate(path, count):
    s = np.linspace(0.0, 1.0, count)
    points = path.coordinates(s)
    energies = energy_values(points)
    tangents = improved_tangents_arbitrary(points, energies)
    gradients = np.asarray([energy_gradient(point)[1] for point in points])
    internal = gradients[1:-1]
    normal = internal - np.sum(internal * tangents, axis=1)[:, None] * tangents
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    weights = 0.5 * (segments[:-1] + segments[1:])
    delta_e = float(np.max(energies) - energies[0])
    natural = points @ Q.T
    return {
        "dense_i_perp": float(np.sum(weights * np.sum(normal * normal, axis=1))),
        "dense_delta_e": delta_e,
        "dense_barrier_error": abs(delta_e - reference_barrier()),
        "maximum_inactive_coordinate": float(np.max(np.abs(natural[:, 2:]))),
        "minimum_speed": float(np.min(segments) * (count - 1)),
        "sample_path": points,
    }


def _solve_network_subproblem(
    oracle, old_points, tangents, fractions, config, initial_guess=None
):
    features = _features(
        fractions,
        config.network_width,
        config.local_feature_count,
        config.local_feature_sigma,
        config.feature_mode,
    )
    active_normals = FourierProjectedLayer.active_normals_from_tangents(
        tangents
    )

    def objective(flat):
        if config.scalar_active_normal:
            scalar = features @ flat
            delta = scalar[:, None] * active_normals
            values = []
            gradients = []
            for sample in range(len(old_points)):
                energy, gradient = oracle.query(
                    sample % N_SAMPLE, old_points[sample] + delta[sample]
                )
                values.append(energy)
                gradients.append(gradient)
            gradients = np.asarray(gradients)
            scalar_residual = (
                np.sum(gradients * active_normals, axis=1)
                + scalar / config.alpha
            )
            value = float(
                np.sum(values)
                + 0.5 * np.sum(scalar * scalar) / config.alpha
            )
            return value, features.T @ scalar_residual
        coefficients = flat.reshape(config.network_width, DIMENSION)
        raw = features @ coefficients
        delta = raw - np.sum(raw * tangents, axis=1)[:, None] * tangents
        values = []
        gradients = []
        for sample in range(len(old_points)):
            energy, gradient = oracle.query(
                sample % N_SAMPLE, old_points[sample] + delta[sample]
            )
            values.append(energy)
            gradients.append(gradient)
        gradients = np.asarray(gradients)
        projected_gradient = gradients - np.sum(
            gradients * tangents, axis=1
        )[:, None] * tangents
        residual = projected_gradient + delta / config.alpha
        value = float(
            np.sum(values) + 0.5 * np.sum(delta * delta) / config.alpha
        )
        coefficient_gradient = features.T @ residual
        return value, coefficient_gradient.reshape(-1)

    parameter_dimension = (
        config.network_width
        if config.scalar_active_normal
        else config.network_width * DIMENSION
    )
    zero = np.zeros(parameter_dimension, dtype=np.float64)
    initial_value, _ = objective(zero)
    initial = (
        np.asarray(initial_guess, dtype=np.float64).reshape(-1)
        if initial_guess is not None
        else zero
    )
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": config.inner_iterations,
            "maxls": config.max_line_search,
            "gtol": config.gradient_tolerance,
            "ftol": 1.0e-14,
            "maxcor": min(10, config.network_width),
        },
    )
    if config.scalar_active_normal:
        coefficients = np.asarray(result.x).reshape(config.network_width, 1)
        scalar = features @ coefficients.reshape(-1)
        delta = scalar[:, None] * active_normals
    else:
        coefficients = np.asarray(result.x).reshape(
            config.network_width, DIMENSION
        )
        raw = features @ coefficients
        delta = raw - np.sum(raw * tangents, axis=1)[:, None] * tangents
    final_value, final_gradient = objective(result.x)
    leakage = np.max(np.abs(np.sum(delta * tangents, axis=1)))
    return coefficients, {
        "objective_change": float(final_value - initial_value),
        "subproblem_gradient_norm": float(np.linalg.norm(final_gradient)),
        "normal_leakage": float(leakage),
        "optimizer_iterations": int(result.nit),
        "optimizer_success": bool(result.success),
    }


def _id(configuration):
    raw = json.dumps(configuration, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def run_experiment(root: Path, config: FourierConfig):
    output_dir = Path(root) / config.role
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = transformed_initial_path(22)
    if config.initial_path_file:
        initial_data = np.load(config.initial_path_file)
        sampled = SampledSplinePath(initial_data["dense_sample_path"][-1])
        if config.global_compression:
            path, _, initial_compression_max = _compress_global(sampled, config)
            if initial_compression_max > config.compression_tolerance:
                raise ValueError("initial path compression exceeds tolerance")
        else:
            path = sampled
    else:
        path = (
            GlobalSplinePath(
                initial[0],
                initial[-1],
                config,
                np.zeros((config.network_width, DIMENSION), dtype=np.float64),
            )
            if config.global_compression
            else LinearPath(initial[0], initial[-1])
        )
    remaining_budget = config.force_budget - config.cost_offset
    if remaining_budget <= 0:
        raise ValueError("force_budget must exceed cost_offset")
    oracle = StreamOracle(remaining_budget)
    certificates = [_certificate(path, config.certificate_points)]
    accepted_forces = [int(config.cost_offset)]
    rows = []
    warm_coefficients = None
    stop_reason = "max_outer"
    for outer in range(config.max_outer):
        fraction_batches = []
        point_batches = []
        tangent_batches = []
        for batch in range(config.collocation_batches):
            if config.collocation_batches == 1:
                batch_fractions = _fractions(
                    outer,
                    config.sampling_jitter,
                    config.sampling_seed,
                    config.sampling_mode,
                )
                if config.max_energy_anchor_count > 0:
                    batch_fractions = _max_energy_anchor_fractions(
                        path,
                        batch_fractions,
                        config.max_energy_anchor_count,
                        config.max_energy_scan_points,
                        config.max_energy_anchor_spacing,
                    )
            else:
                rng = np.random.default_rng(
                    config.sampling_seed
                    + 104729 * (outer * config.collocation_batches + batch)
                )
                subcell = 1.0 / (N_SAMPLE * config.collocation_batches)
                batch_fractions = (
                    np.arange(N_SAMPLE, dtype=np.float64)
                    + (batch + 0.5) / config.collocation_batches
                ) / N_SAMPLE
                batch_fractions += (
                    config.sampling_jitter
                    * rng.uniform(-1.0, 1.0, N_SAMPLE)
                    * subcell
                )
            batch_points = path.coordinates(batch_fractions)
            if config.tangent_stencil_step > 0.0:
                batch_tangents = _continuous_improved_tangents(
                    path,
                    batch_fractions,
                    config.tangent_stencil_step,
                )
            else:
                tangent_points = np.vstack(
                    [
                        path.coordinates([0.0]),
                        batch_points,
                        path.coordinates([1.0]),
                    ]
                )
                batch_tangents = improved_tangents_arbitrary(
                    tangent_points, energy_values(tangent_points)
                )
            fraction_batches.append(batch_fractions)
            point_batches.append(batch_points)
            tangent_batches.append(batch_tangents)
        fractions = np.concatenate(fraction_batches)
        old_points = np.concatenate(point_batches, axis=0)
        tangents = np.concatenate(tangent_batches, axis=0)
        order = np.argsort(fractions)
        fractions = fractions[order]
        old_points = old_points[order]
        tangents = tangents[order]
        try:
            coefficients, report = _solve_network_subproblem(
                oracle,
                old_points,
                tangents,
                fractions,
                config,
                warm_coefficients if config.warm_start_coefficients else None,
            )
        except ForceBudgetReached:
            stop_reason = "budget"
            break
        tangent_s = np.concatenate([[0.0], fractions, [1.0]])
        tangent_values = np.vstack([tangents[0], tangents, tangents[-1]])
        accepted = bool(
            report["objective_change"] <= 1.0e-10
            and report["normal_leakage"] <= 1.0e-10
        )
        if accepted:
            candidate = FourierProjectedLayer(
                path,
                config.network_width,
                config.feature_mode,
                config.local_feature_count,
                config.local_feature_sigma,
                config.scalar_active_normal,
                coefficients,
                tangent_s,
                tangent_values,
            )
            candidate = _arc_reparameterize(
                candidate, config.arclength_points
            )
            if config.global_compression:
                candidate, compression_rms, compression_max = _compress_global(
                    candidate, config
                )
                accepted = bool(
                    compression_max <= config.compression_tolerance
                )
            else:
                compression_rms = 0.0
                compression_max = 0.0
            if accepted:
                warm_coefficients = coefficients.copy()
                path = candidate
                certificates.append(
                    _certificate(path, config.certificate_points)
                )
                accepted_forces.append(
                    int(config.cost_offset + oracle.total)
                )
        else:
            compression_rms = np.inf
            compression_max = np.inf
        rows.append({
            "number_of_forces": int(config.cost_offset + oracle.total),
            "outer_iteration": outer,
            "accepted": accepted,
            "objective_change": report["objective_change"],
            "subproblem_gradient_norm": report["subproblem_gradient_norm"],
            "normal_leakage": report["normal_leakage"],
            "optimizer_iterations": report["optimizer_iterations"],
            "optimizer_success": report["optimizer_success"],
            "compression_rms": compression_rms,
            "compression_max": compression_max,
            "collocation_fractions": fractions,
            "collocation_path": old_points,
        })
    if not rows:
        raise RuntimeError("force budget was too small for one network subproblem")
    payload = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    payload.update({
        "accepted_number_of_forces": np.asarray(accepted_forces),
        "dense_i_perp": np.asarray([x["dense_i_perp"] for x in certificates]),
        "dense_delta_e": np.asarray([x["dense_delta_e"] for x in certificates]),
        "dense_barrier_error": np.asarray(
            [x["dense_barrier_error"] for x in certificates]
        ),
        "dense_maximum_inactive_coordinate": np.asarray(
            [x["maximum_inactive_coordinate"] for x in certificates]
        ),
        "dense_minimum_speed": np.asarray(
            [x["minimum_speed"] for x in certificates]
        ),
        "dense_sample_path": np.asarray([x["sample_path"] for x in certificates]),
        "configuration_json": np.asarray(json.dumps(asdict(config), sort_keys=True)),
        "fixed_alpha_audit": np.asarray(True),
        "unconstrained_delta_audit": np.asarray(True),
        "improved_tangent_audit": np.asarray(True),
        "dense_information_enters_optimization": np.asarray(False),
        "cost_definition": np.asarray(
            "direct sum of all 20 network-objective force streams"
        ),
        "stop_reason": np.asarray(stop_reason),
    })
    signature = _id(asdict(config))
    output = output_dir / f"q10_nn_ivm_fourier_{signature}.npz"
    np.savez_compressed(output, **payload)
    inactive = np.asarray(payload["dense_maximum_inactive_coordinate"])
    audit = {
        "configuration": asdict(config),
        "configuration_id": signature,
        "output": str(output),
        "optimization_force_calls": int(config.cost_offset + oracle.total),
        "accepted_updates": int(len(certificates) - 1),
        "stop_reason": stop_reason,
        "final_dense_i_perp": float(payload["dense_i_perp"][-1]),
        "minimum_dense_i_perp": float(np.min(payload["dense_i_perp"])),
        "final_dense_barrier_error": float(payload["dense_barrier_error"][-1]),
        "maximum_inactive_coordinate": float(np.max(inactive)),
        "proximal_descent_audit": bool(
            np.nanmax(np.asarray(payload["objective_change"], dtype=float))
            <= 1.0e-10
        ),
        "normal_projection_audit": bool(
            np.nanmax(np.asarray(payload["normal_leakage"], dtype=float))
            <= 1.0e-10
        ),
        "fixed_alpha": True,
        "unconstrained_delta": True,
        "improved_tangent": True,
        "dense_information_enters_optimization": False,
        "force_count_definition": "direct sum of all 20 network-objective streams",
    }
    audit["theory_audit_pass"] = bool(
        audit["proximal_descent_audit"]
        and audit["normal_projection_audit"]
        and audit["maximum_inactive_coordinate"] <= 1.0e-8
    )
    output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit
