"""Audited pointwise-proximal NN-IVM for the fixed-Q 10D Muller case.

Twenty moving samples solve the genuine frozen-normal IVM proximal
subproblems.  A fixed-capacity endpoint-exact radial-basis neural network is
then fitted, without physical oracle calls, to the resulting continuous path.
Dense diagnostics are computed only after optimization has finished.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PchipInterpolator
from scipy.stats import qmc

from q10_ivm_core import REFERENCE_BARRIER, improved_tangents, improved_tangents_arbitrary
from q10_potential import DIMENSION, Q, energy_gradient, energy_values, transformed_initial_path


N_SAMPLE = 20


def _hash(configuration: dict) -> str:
    raw = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


class ForceBudgetReached(RuntimeError):
    pass


class StreamOracle:
    """Twenty independent cached streams with direct-sum force accounting."""

    def __init__(self, budget: int):
        self.budget = int(budget)
        self.calls = np.zeros(N_SAMPLE, dtype=np.int64)
        self.hits = np.zeros(N_SAMPLE, dtype=np.int64)
        self.cache: list[dict[bytes, tuple[float, np.ndarray]]] = [dict() for _ in range(N_SAMPLE)]

    @property
    def total(self) -> int:
        return int(np.sum(self.calls))

    def query(self, stream: int, point: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.ascontiguousarray(point, dtype=np.float64)
        key = x.tobytes()
        if key in self.cache[stream]:
            self.hits[stream] += 1
            energy, gradient = self.cache[stream][key]
            return energy, gradient.copy()
        if self.total >= self.budget:
            raise ForceBudgetReached
        energy, gradient = energy_gradient(x)
        self.calls[stream] += 1
        self.cache[stream][key] = (float(energy), np.asarray(gradient, dtype=np.float64).copy())
        return float(energy), np.asarray(gradient, dtype=np.float64)


class EndpointRBFPath:
    """Endpoint-exact one-hidden-layer Gaussian RBF neural path."""

    def __init__(self, start: np.ndarray, end: np.ndarray, centers: np.ndarray, sigma: float, weights: np.ndarray):
        self.start = np.asarray(start, dtype=np.float64)
        self.end = np.asarray(end, dtype=np.float64)
        self.centers = np.asarray(centers, dtype=np.float64)
        self.sigma = float(sigma)
        self.weights = np.asarray(weights, dtype=np.float64)

    @classmethod
    def linear(cls, start: np.ndarray, end: np.ndarray, width: int, sigma_scale: float):
        centers = np.linspace(0.0, 1.0, int(width))
        sigma = float(sigma_scale) / max(int(width) - 1, 1)
        return cls(start, end, centers, sigma, np.zeros((int(width), DIMENSION)))

    def _features(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(s, dtype=np.float64).reshape(-1)
        delta = values[:, None] - self.centers[None, :]
        radial = np.exp(-0.5 * (delta / self.sigma) ** 2)
        envelope = 4.0 * values * (1.0 - values)
        derivative_envelope = 4.0 - 8.0 * values
        features = envelope[:, None] * radial
        derivative = derivative_envelope[:, None] * radial - features * delta / (self.sigma**2)
        return features, derivative

    def coordinates(self, s: np.ndarray) -> np.ndarray:
        values = np.asarray(s, dtype=np.float64).reshape(-1)
        features, _ = self._features(values)
        baseline = self.start[None, :] + values[:, None] * (self.end - self.start)[None, :]
        return baseline + features @ self.weights

    def derivative(self, s: np.ndarray) -> np.ndarray:
        values = np.asarray(s, dtype=np.float64).reshape(-1)
        _, derivative = self._features(values)
        return (self.end - self.start)[None, :] + derivative @ self.weights

    def state(self) -> dict:
        return {
            "start": self.start.copy(),
            "end": self.end.copy(),
            "centers": self.centers.copy(),
            "sigma": self.sigma,
            "weights": self.weights.copy(),
        }


def _van_der_corput(index: int) -> float:
    value = 0.0
    denominator = 1.0
    integer = int(index)
    while integer:
        integer, remainder = divmod(integer, 2)
        denominator *= 2.0
        value += remainder / denominator
    return value


def _stratified_fractions(outer: int, jitter: float, seed: int, mode: str = "random") -> np.ndarray:
    base = (np.arange(N_SAMPLE, dtype=np.float64) + 0.5) / N_SAMPLE
    if mode == "left_curvature":
        rng = np.random.default_rng(int(seed) + 104729 * int(outer))
        global_points = (np.arange(10, dtype=np.float64) + 0.5) / 10.0
        local_points = 0.65 + 0.25 * (
            np.arange(10, dtype=np.float64) + 0.5
        ) / 10.0
        global_points += float(jitter) * rng.uniform(-1.0, 1.0, 10) / 10.0
        local_points += (
            0.25 * float(jitter) * rng.uniform(-1.0, 1.0, 10) / 10.0
        )
        return np.sort(
            np.clip(
                np.concatenate([global_points, local_points]),
                1.0e-8,
                1.0 - 1.0e-8,
            )
        )
    if mode == "gauss_legendre":
        nodes, _ = np.polynomial.legendre.leggauss(N_SAMPLE)
        return 0.5 * (nodes + 1.0)
    if jitter == 0.0 and not (
        mode.startswith("golden_focus_") or mode.startswith("golden_asym_")
    ):
        return base
    if mode == "van_der_corput_shift":
        shift = float(jitter) * (2.0 * _van_der_corput(outer) - 1.0) / N_SAMPLE
        return base + shift
    rng = np.random.default_rng(int(seed) + 104729 * int(outer))
    if mode == "antithetic_random":
        pair_rng = np.random.default_rng(int(seed) + 104729 * (int(outer) // 2))
        offsets = pair_rng.uniform(-1.0, 1.0, N_SAMPLE)
        if int(outer) % 2:
            offsets = -offsets
        return base + float(jitter) * offsets / N_SAMPLE
    if mode == "balanced_random":
        offsets = (2.0 * (np.arange(N_SAMPLE) + 0.5) / N_SAMPLE - 1.0)
        return base + float(jitter) * rng.permutation(offsets) / N_SAMPLE
    if mode == "common_random":
        offsets = np.full(N_SAMPLE, rng.uniform(-1.0, 1.0), dtype=float)
    elif mode == "common_golden":
        golden = 0.6180339887498949
        offset = 2.0 * (((outer + 1) * golden) % 1.0) - 1.0
        offsets = np.full(N_SAMPLE, offset, dtype=float)
    elif mode == "sobol":
        exponent = int(np.ceil(np.log2(int(outer) + 1)))
        samples = qmc.Sobol(d=N_SAMPLE, scramble=False).random_base2(exponent)
        offsets = 2.0 * samples[int(outer)] - 1.0
        return base + float(jitter) * offsets / N_SAMPLE
    if mode in {"common_random", "common_golden"}:
        return base + float(jitter) * offsets / N_SAMPLE
    if mode.startswith("golden_focus_") or mode.startswith("golden_asym_"):
        golden = 0.6180339887498949
        offset = 2.0 * (((outer + 1) * golden) % 1.0) - 1.0
        uniform = base + float(jitter) * offset / N_SAMPLE
        pieces = mode.split("_")
        center = float(pieces[2])
        left_power = float(pieces[3])
        right_power = left_power if pieces[1] == "focus" else float(pieces[4])
        if not (
            0.0 < center < 1.0
            and 0.0 < left_power <= 1.0
            and 0.0 < right_power <= 1.0
        ):
            raise ValueError(f"invalid focused golden mode: {mode}")
        focused = np.empty_like(uniform)
        left = uniform <= center
        focused[left] = center * (uniform[left] / center) ** left_power
        focused[~left] = center + (1.0 - center) * (
            1.0
            - ((1.0 - uniform[~left]) / (1.0 - center)) ** right_power
        )
        return focused
    if (
        mode.startswith("halton_peaks_")
        or mode.startswith("golden_peaks_")
        or mode.startswith("antithetic_peaks_")
    ):
        _, _, first_text, second_text, strength_text, ratio_text, sigma_text = mode.split("_")
        first = float(first_text)
        second = float(second_text)
        strength = float(strength_text)
        ratio = float(ratio_text)
        sigma = float(sigma_text)
        if not (
            0.0 < first < second < 1.0
            and strength >= 0.0
            and ratio >= 0.0
            and sigma > 0.0
        ):
            raise ValueError(f"invalid peaked-Halton mode: {mode}")
        if mode.startswith("halton_peaks_"):
            engine = qmc.Halton(d=N_SAMPLE, scramble=False)
            engine.fast_forward(int(outer))
            offsets = 2.0 * engine.random(1)[0] - 1.0
        elif mode.startswith("antithetic_peaks_"):
            pair = (int(outer) - 1) // 2
            engine = qmc.Halton(d=N_SAMPLE, scramble=False)
            engine.fast_forward(pair)
            offsets = 2.0 * engine.random(1)[0] - 1.0
            if int(outer) % 2 == 0:
                offsets = -offsets
        else:
            golden = 0.6180339887498949
            offset = 2.0 * (((outer + 1) * golden) % 1.0) - 1.0
            offsets = np.full(N_SAMPLE, offset, dtype=np.float64)
        uniform = base + float(jitter) * offsets / N_SAMPLE
        grid = np.linspace(0.0, 1.0, 4097)
        density = 1.0 + strength * (
            np.exp(-0.5 * ((grid - first) / sigma) ** 2)
            + ratio * np.exp(-0.5 * ((grid - second) / sigma) ** 2)
        )
        cumulative = np.zeros_like(grid)
        cumulative[1:] = np.cumsum(
            0.5 * (density[:-1] + density[1:]) * np.diff(grid)
        )
        return np.interp(uniform * cumulative[-1], cumulative, grid)
    if mode.startswith("halton_asym_"):
        _, _, center_text, left_power_text, right_power_text = mode.split("_")
        center = float(center_text)
        left_power = float(left_power_text)
        right_power = float(right_power_text)
        if not (
            0.0 < center < 1.0
            and 0.0 < left_power <= 1.0
            and 0.0 < right_power <= 1.0
        ):
            raise ValueError(f"invalid asymmetric-Halton mode: {mode}")
        engine = qmc.Halton(d=N_SAMPLE, scramble=False)
        engine.fast_forward(int(outer))
        offsets = 2.0 * engine.random(1)[0] - 1.0
        uniform = base + float(jitter) * offsets / N_SAMPLE
        focused = np.empty_like(uniform)
        left = uniform <= center
        focused[left] = center * (uniform[left] / center) ** left_power
        focused[~left] = center + (1.0 - center) * (
            1.0
            - ((1.0 - uniform[~left]) / (1.0 - center)) ** right_power
        )
        return focused
    if mode.startswith("antithetic_tripeaks_"):
        parts = mode.split("_")
        if len(parts) != 11:
            raise ValueError(f"invalid antithetic tripeak mode: {mode}")
        values = [float(value) for value in parts[2:]]
        c1, w1, a1, c2, w2, a2, c3, w3, a3 = values
        pair = int(outer) // 2
        engine = qmc.Halton(d=N_SAMPLE, scramble=False)
        engine.fast_forward(pair)
        offsets = 2.0 * engine.random(1)[0] - 1.0
        if int(outer) % 2:
            offsets = -offsets
        uniform = base + float(jitter) * offsets / N_SAMPLE
        grid = np.linspace(0.0, 1.0, 4097)
        density = 1.0
        for center, width, amplitude in (
            (c1, w1, a1), (c2, w2, a2), (c3, w3, a3)
        ):
            density = density + amplitude * np.exp(
                -0.5 * ((grid - center) / width) ** 2
            )
        cumulative = np.zeros_like(grid)
        cumulative[1:] = np.cumsum(
            0.5 * (density[:-1] + density[1:]) * np.diff(grid)
        )
        return np.interp(uniform * cumulative[-1], cumulative, grid)
    if mode.startswith("hybrid_focus_antipeaks_"):
        _, _, _, switch_text, cleanup_jitter_text = mode.split("_")
        switch = int(switch_text)
        if int(outer) < switch:
            engine = qmc.Halton(d=N_SAMPLE, scramble=False)
            engine.fast_forward(int(outer))
            offsets = 2.0 * engine.random(1)[0] - 1.0
            uniform = base + float(jitter) * offsets / N_SAMPLE
            center, power = 0.68, 0.75
            focused = np.empty_like(uniform)
            left = uniform <= center
            focused[left] = center * (uniform[left] / center) ** power
            focused[~left] = center + (1.0 - center) * (
                1.0
                - ((1.0 - uniform[~left]) / (1.0 - center)) ** power
            )
            return focused
        cleanup_outer = int(outer) - switch
        pair = cleanup_outer // 2
        engine = qmc.Halton(d=N_SAMPLE, scramble=False)
        engine.fast_forward(pair)
        offsets = 2.0 * engine.random(1)[0] - 1.0
        if cleanup_outer % 2:
            offsets = -offsets
        cleanup_jitter = float(cleanup_jitter_text)
        uniform = base + cleanup_jitter * offsets / N_SAMPLE
        grid = np.linspace(0.0, 1.0, 4097)
        density = 1.0 + 0.6 * (
            np.exp(-0.5 * ((grid - 0.62) / 0.04) ** 2)
            + 0.85 * np.exp(-0.5 * ((grid - 0.76) / 0.04) ** 2)
        )
        cumulative = np.zeros_like(grid)
        cumulative[1:] = np.cumsum(
            0.5 * (density[:-1] + density[1:]) * np.diff(grid)
        )
        return np.interp(uniform * cumulative[-1], cumulative, grid)
    if mode.startswith("halton_two_"):
        _, _, c1_text, p1_text, switch_text, c2_text, p2_text = mode.split("_")
        switch = int(switch_text)
        if int(outer) < switch:
            center, power = float(c1_text), float(p1_text)
        else:
            center, power = float(c2_text), float(p2_text)
        if not (0.0 < center < 1.0 and 0.0 < power <= 1.0):
            raise ValueError(f"invalid two-stage Halton mode: {mode}")
        engine = qmc.Halton(d=N_SAMPLE, scramble=False)
        engine.fast_forward(int(outer))
        offsets = 2.0 * engine.random(1)[0] - 1.0
        uniform = base + float(jitter) * offsets / N_SAMPLE
        focused = np.empty_like(uniform)
        left = uniform <= center
        focused[left] = center * (uniform[left] / center) ** power
        focused[~left] = center + (1.0 - center) * (
            1.0 - ((1.0 - uniform[~left]) / (1.0 - center)) ** power
        )
        return focused
    if mode.startswith("halton_focus_"):
        _, _, center_text, power_text = mode.split("_")
        center = float(center_text)
        power = float(power_text)
        if not (0.0 < center < 1.0 and 0.0 < power <= 1.0):
            raise ValueError(f"invalid focused-Halton mode: {mode}")
        engine = qmc.Halton(d=N_SAMPLE, scramble=False)
        engine.fast_forward(int(outer))
        offsets = 2.0 * engine.random(1)[0] - 1.0
        uniform = base + float(jitter) * offsets / N_SAMPLE
        focused = np.empty_like(uniform)
        left = uniform <= center
        focused[left] = center * (uniform[left] / center) ** power
        focused[~left] = center + (1.0 - center) * (
            1.0 - ((1.0 - uniform[~left]) / (1.0 - center)) ** power
        )
        return focused
    if mode == "halton":
        engine = qmc.Halton(d=N_SAMPLE, scramble=False)
        engine.fast_forward(int(outer))
        offsets = 2.0 * engine.random(1)[0] - 1.0
        return base + float(jitter) * offsets / N_SAMPLE
    if mode != "random":
        raise ValueError(f"unknown sampling mode: {mode}")
    return base + float(jitter) * rng.uniform(-1.0, 1.0, N_SAMPLE) / N_SAMPLE


def _arc_parameters(path: EndpointRBFPath, fractions: np.ndarray, count: int) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, int(count))
    derivative = path.derivative(grid)
    speed = np.linalg.norm(derivative, axis=1)
    arc = np.zeros_like(grid)
    arc[1:] = np.cumsum(0.5 * (speed[:-1] + speed[1:]) * np.diff(grid))
    if arc[-1] <= 0.0:
        return np.asarray(fractions, dtype=np.float64)
    return np.interp(np.asarray(fractions) * arc[-1], arc, grid)


def _adaptive_residual_fractions(
    outer: int,
    jitter: float,
    seed: int,
    previous_fractions: np.ndarray,
    previous_scores: np.ndarray,
    strength: float,
    uniform_floor: float,
) -> np.ndarray:
    """Warp twenty deterministic strata using previously counted residuals."""
    uniform = _stratified_fractions(outer, jitter, seed, "halton")
    grid = np.linspace(0.0, 1.0, 4097)
    scores = np.maximum(np.asarray(previous_scores, dtype=np.float64), 0.0)
    signal = np.interp(
        grid,
        np.asarray(previous_fractions, dtype=np.float64),
        scores,
        left=scores[0],
        right=scores[-1],
    )
    kernel = np.hanning(129)
    kernel /= np.sum(kernel)
    signal = np.convolve(signal, kernel, mode="same")
    scale = max(float(np.quantile(signal, 0.9)), 1.0e-14)
    density = float(uniform_floor) + float(strength) * np.minimum(
        signal / scale, 2.0
    )
    cumulative = np.zeros_like(grid)
    cumulative[1:] = np.cumsum(
        0.5 * (density[:-1] + density[1:]) * np.diff(grid)
    )
    return np.interp(uniform * cumulative[-1], cumulative, grid)


@lru_cache(maxsize=None)
def _equal_arc_rbf_factor(width: int, sigma_scale: float, fit_points: int, ridge: float):
    zero = np.zeros(DIMENSION, dtype=np.float64)
    template = EndpointRBFPath.linear(zero, zero, int(width), float(sigma_scale))
    fit_s = np.linspace(0.0, 1.0, int(fit_points))
    features, _ = template._features(fit_s)
    gram = features.T @ features + float(ridge) * np.eye(features.shape[1])
    return cho_factor(gram, lower=True, check_finite=False)


def _fit_rbf(start: np.ndarray, end: np.ndarray, sample_s: np.ndarray, targets: np.ndarray, *, width: int, sigma_scale: float, ridge: float, equal_arc: bool, fit_points: int, target_tangents: np.ndarray | None = None, tangent_weight: float = 0.0, hermite_tangents: np.ndarray | None = None) -> tuple[EndpointRBFPath, dict]:
    knots = np.concatenate(([0.0], np.asarray(sample_s), [1.0]))
    values = np.vstack((start, np.asarray(targets), end))
    if hermite_tangents is None:
        teacher = CubicSpline(knots, values, axis=0, bc_type="natural")
    else:
        derivatives = np.empty_like(values)
        derivatives[0] = (values[1] - values[0]) / (knots[1] - knots[0])
        derivatives[-1] = (values[-1] - values[-2]) / (knots[-1] - knots[-2])
        left_speed = np.linalg.norm(np.diff(values, axis=0), axis=1) / np.diff(knots)
        local_speed = 0.5 * (left_speed[:-1] + left_speed[1:])
        derivatives[1:-1] = local_speed[:, None] * np.asarray(hermite_tangents)
        teacher = CubicHermiteSpline(knots, values, derivatives, axis=0)
    dense_s = np.linspace(0.0, 1.0, max(4097, 8 * int(fit_points) + 1))
    dense_values = teacher(dense_s)
    if equal_arc:
        derivative = teacher(dense_s, 1)
        speed = np.linalg.norm(derivative, axis=1)
        arc = np.zeros_like(dense_s)
        arc[1:] = np.cumsum(0.5 * (speed[:-1] + speed[1:]) * np.diff(dense_s))
        uniform = np.linspace(0.0, 1.0, int(fit_points))
        source_s = np.interp(uniform * arc[-1], arc, dense_s)
        fit_s = uniform
        fit_targets = PchipInterpolator(dense_s, dense_values, axis=0)(source_s)
    else:
        fit_s = np.linspace(0.0, 1.0, int(fit_points))
        fit_targets = teacher(fit_s)
    model = EndpointRBFPath.linear(start, end, int(width), float(sigma_scale))
    features, _ = model._features(fit_s)
    baseline = start[None, :] + fit_s[:, None] * (end - start)[None, :]
    rhs = fit_targets - baseline
    design = features
    response = rhs
    if target_tangents is not None and float(tangent_weight) > 0.0:
        if equal_arc:
            tangent_s = np.interp(sample_s, dense_s, arc / arc[-1])
            target_derivative = arc[-1] * np.asarray(target_tangents)
        else:
            tangent_s = np.asarray(sample_s)
            target_derivative = np.linalg.norm(teacher(tangent_s, 1), axis=1)[:, None] * np.asarray(target_tangents)
        _, derivative_features = model._features(tangent_s)
        derivative_rhs = target_derivative - (end - start)[None, :]
        weight = np.sqrt(float(tangent_weight))
        design = np.vstack((design, weight * derivative_features))
        response = np.vstack((response, weight * derivative_rhs))
    right_hand_side = design.T @ response
    if design is features:
        factor = _equal_arc_rbf_factor(width, sigma_scale, fit_points, ridge)
        model.weights = cho_solve(factor, right_hand_side, check_finite=False)
    else:
        gram = design.T @ design + float(ridge) * np.eye(features.shape[1])
        model.weights = np.linalg.solve(gram, right_hand_side)
    active_projector = Q[:2, :].T @ Q[:2, :]
    model.weights = model.weights @ active_projector
    validation_s = np.linspace(0.0, 1.0, 2049)
    if equal_arc:
        validation_source = np.interp(validation_s * arc[-1], arc, dense_s)
        truth = PchipInterpolator(dense_s, dense_values, axis=0)(validation_source)
    else:
        truth = teacher(validation_s)
    error = model.coordinates(validation_s) - truth
    return model, {
        "coordinate_rms": float(np.sqrt(np.mean(error * error))),
        "coordinate_max": float(np.max(np.linalg.norm(error, axis=1))),
    }


def _increment_rbf(current: EndpointRBFPath, sample_s: np.ndarray, targets: np.ndarray, *, ridge: float, update_cap: float) -> tuple[EndpointRBFPath, dict]:
    features, _ = current._features(sample_s)
    rhs = np.asarray(targets) - current.coordinates(sample_s)
    dual = features @ features.T + float(ridge) * np.eye(features.shape[0])
    delta_weights = features.T @ np.linalg.solve(dual, rhs)
    active_projector = Q[:2, :].T @ Q[:2, :]
    delta_weights = delta_weights @ active_projector
    trial = EndpointRBFPath(
        current.start,
        current.end,
        current.centers,
        current.sigma,
        current.weights + delta_weights,
    )
    validation_s = np.linspace(0.0, 1.0, 2049)
    update = trial.coordinates(validation_s) - current.coordinates(validation_s)
    raw_max = float(np.max(np.linalg.norm(update, axis=1)))
    scale = min(1.0, float(update_cap) / max(raw_max, 1.0e-15))
    model = EndpointRBFPath(
        current.start,
        current.end,
        current.centers,
        current.sigma,
        current.weights + scale * delta_weights,
    )
    scaled_targets = current.coordinates(sample_s) + scale * rhs
    error = model.coordinates(sample_s) - scaled_targets
    return model, {
        "coordinate_rms": float(np.sqrt(np.mean(error * error))),
        "coordinate_max": float(np.max(np.linalg.norm(error, axis=1))),
        "update_max": float(scale * raw_max),
        "update_scale": float(scale),
    }


def _fit_nodes_rbf(start: np.ndarray, end: np.ndarray, targets: np.ndarray, *, width: int, sigma_scale: float, ridge: float) -> tuple[EndpointRBFPath, dict]:
    values = np.vstack((start, np.asarray(targets), end))
    segments = np.linalg.norm(np.diff(values, axis=0), axis=1)
    parameters = np.concatenate(([0.0], np.cumsum(segments)))
    parameters /= parameters[-1]
    internal_s = parameters[1:-1]
    model = EndpointRBFPath.linear(start, end, int(width), float(sigma_scale))
    features, _ = model._features(internal_s)
    baseline = start[None, :] + internal_s[:, None] * (end - start)[None, :]
    rhs = values[1:-1] - baseline
    gram = features.T @ features + float(ridge) * np.eye(features.shape[1])
    model.weights = np.linalg.solve(gram, features.T @ rhs)
    active_projector = Q[:2, :].T @ Q[:2, :]
    model.weights = model.weights @ active_projector
    error = model.coordinates(internal_s) - values[1:-1]
    return model, {
        "coordinate_rms": float(np.sqrt(np.mean(error * error))),
        "coordinate_max": float(np.max(np.linalg.norm(error, axis=1))),
    }


def _fit_nodes_preserving_reference(current: EndpointRBFPath, start: np.ndarray, end: np.ndarray, targets: np.ndarray, *, width: int, sigma_scale: float, ridge: float, preserve_weight: float, fit_points: int, arclength_points: int) -> tuple[EndpointRBFPath, dict]:
    values = np.vstack((start, np.asarray(targets), end))
    segments = np.linalg.norm(np.diff(values, axis=0), axis=1)
    parameters = np.concatenate(([0.0], np.cumsum(segments)))
    parameters /= parameters[-1]
    node_s = parameters[1:-1]
    model = EndpointRBFPath.linear(start, end, int(width), float(sigma_scale))
    node_features, _ = model._features(node_s)
    node_baseline = start[None, :] + node_s[:, None] * (end - start)[None, :]
    node_rhs = values[1:-1] - node_baseline
    preserve_s = np.linspace(0.0, 1.0, int(fit_points))
    current_s = _arc_parameters(current, preserve_s, int(arclength_points))
    preserve_targets = current.coordinates(current_s)
    preserve_features, _ = model._features(preserve_s)
    preserve_baseline = start[None, :] + preserve_s[:, None] * (end - start)[None, :]
    preserve_rhs = preserve_targets - preserve_baseline
    hessian = (
        float(preserve_weight) * preserve_features.T @ preserve_features
        + float(ridge) * np.eye(int(width))
    )
    linear = float(preserve_weight) * preserve_features.T @ preserve_rhs
    constraint_count = node_features.shape[0]
    kkt = np.block([
        [hessian, node_features.T],
        [node_features, np.zeros((constraint_count, constraint_count))],
    ])
    right = np.vstack((linear, node_rhs))
    solution = np.linalg.lstsq(kkt, right, rcond=1.0e-12)[0]
    model.weights = solution[: int(width)] @ (Q[:2, :].T @ Q[:2, :])
    node_error = model.coordinates(node_s) - values[1:-1]
    preserve_error = model.coordinates(preserve_s) - preserve_targets
    return model, {
        "coordinate_rms": float(np.sqrt(np.mean(node_error * node_error))),
        "coordinate_max": float(np.max(np.linalg.norm(node_error, axis=1))),
        "preserve_rms": float(np.sqrt(np.mean(preserve_error * preserve_error))),
    }


def _proximal_scalar(oracle: StreamOracle, stream: int, reference: np.ndarray, tangent: np.ndarray, alpha: float, inner_calls: int, trust_radius: float, residual_rtol: float, warm_potential_curvature: float = 0.0) -> dict:
    energy0, gradient0 = oracle.query(stream, reference)
    normal_gradient = gradient0 - tangent * np.dot(gradient0, tangent)
    norm = float(np.linalg.norm(normal_gradient))
    if norm <= 1.0e-15:
        return {"point": reference.copy(), "energy": energy0, "gradient": gradient0, "objective_change": 0.0, "calls": 0, "residual": 0.0, "normal_leakage": 0.0, "potential_curvature": float(warm_potential_curvature)}
    direction = normal_gradient / norm
    q_previous = 0.0
    residual_previous = norm
    initial_total_curvature = 1.0 / alpha + float(warm_potential_curvature)
    if not np.isfinite(initial_total_curvature) or initial_total_curvature <= 0.0:
        initial_total_curvature = 1.0 / alpha
    q = float(np.clip(-norm / initial_total_curvature, -trust_radius, trust_radius))
    best = (0.0, energy0, gradient0, energy0, norm)
    observed_potential_curvature = float(warm_potential_curvature)
    calls_before = oracle.total
    for _ in range(int(inner_calls)):
        point = reference + q * direction
        energy, gradient = oracle.query(stream, point)
        objective = float(energy + q * q / (2.0 * alpha))
        residual = float(np.dot(gradient, direction) + q / alpha)
        if objective < best[3]:
            best = (q, energy, gradient, objective, abs(residual))
        if abs(residual) <= float(residual_rtol) * max(norm, 1.0):
            break
        denominator = q - q_previous
        curvature = (residual - residual_previous) / denominator if abs(denominator) > 1.0e-15 else np.nan
        if np.isfinite(curvature):
            observed_potential_curvature = float(curvature - 1.0 / alpha)
        q_new = q - residual / curvature if np.isfinite(curvature) and curvature > 0.0 else 0.5 * q
        q_new = float(np.clip(q_new, -trust_radius, trust_radius))
        if abs(q_new - q) <= 1.0e-14:
            break
        q_previous, residual_previous, q = q, residual, q_new
    q_best, energy_best, gradient_best, objective_best, residual_best = best
    displacement = q_best * direction
    return {
        "point": reference + displacement,
        "energy": float(energy_best),
        "gradient": np.asarray(gradient_best),
        "objective_change": float(objective_best - energy0),
        "calls": int(oracle.total - calls_before),
        "potential_curvature": observed_potential_curvature,
        "residual": float(residual_best),
        "normal_leakage": float(abs(np.dot(displacement, tangent))),
    }


def _proximal_scalar_unconstrained(oracle: StreamOracle, stream: int, reference: np.ndarray, tangent: np.ndarray, alpha: float, inner_calls: int, trust_radius: float, residual_rtol: float, warm_potential_curvature: float = 0.0) -> dict:
    """Unbounded one-dimensional L-BFGS solve of the frozen-normal objective."""
    from scipy.optimize import minimize

    energy0, gradient0 = oracle.query(stream, reference)
    normal_gradient = gradient0 - tangent * np.dot(gradient0, tangent)
    norm = float(np.linalg.norm(normal_gradient))
    if norm <= 1.0e-15:
        return {"point": reference.copy(), "energy": energy0, "gradient": gradient0, "objective_change": 0.0, "calls": 0, "residual": 0.0, "normal_leakage": 0.0, "potential_curvature": 0.0}
    direction = normal_gradient / norm
    best = {"q": 0.0, "energy": energy0, "gradient": gradient0, "objective": energy0, "residual": norm}
    calls_before = oracle.total

    def objective_gradient(value):
        q = float(np.asarray(value).reshape(-1)[0])
        energy, gradient = oracle.query(stream, reference + q * direction)
        objective = float(energy + q * q / (2.0 * alpha))
        residual = float(np.dot(gradient, direction) + q / alpha)
        if np.isfinite(objective) and objective < best["objective"]:
            best.update(q=q, energy=energy, gradient=gradient, objective=objective, residual=abs(residual))
        return objective, np.asarray([residual], dtype=np.float64)

    result = minimize(
        objective_gradient,
        np.zeros(1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(inner_calls),
            "maxfun": int(inner_calls),
            "maxls": min(10, int(inner_calls)),
            "maxcor": min(10, int(inner_calls)),
            "ftol": 1.0e-14,
            "gtol": float(residual_rtol) * max(norm, 1.0),
        },
    )
    displacement = float(best["q"]) * direction
    return {
        "point": reference + displacement,
        "energy": float(best["energy"]),
        "gradient": np.asarray(best["gradient"]),
        "objective_change": float(best["objective"] - energy0),
        "calls": int(oracle.total - calls_before),
        "potential_curvature": 0.0,
        "residual": float(best["residual"]),
        "normal_leakage": float(abs(np.dot(displacement, tangent))),
        "optimizer_success": bool(result.success),
    }


def _dense_certificate(path: EndpointRBFPath, count: int) -> dict:
    s = np.linspace(0.0, 1.0, int(count))
    points = path.coordinates(s)
    derivative = path.derivative(s)
    speed = np.linalg.norm(derivative, axis=1)
    energies = energy_values(points)
    tangent = improved_tangents_arbitrary(points, energies)
    gradients = np.asarray([energy_gradient(point)[1] for point in points[1:-1]])
    normal = gradients - tangent * np.sum(gradients * tangent, axis=1, keepdims=True)
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    weights = 0.5 * (segments[:-1] + segments[1:])
    i_perp = float(np.sum(weights * np.sum(normal * normal, axis=1)))
    delta_e = float(np.max(energies) - energies[0])
    z = points @ Q.T
    return {
        "dense_i_perp": i_perp,
        "dense_delta_e": delta_e,
        "dense_barrier_error": float(abs(delta_e - REFERENCE_BARRIER)),
        "maximum_inactive_coordinate": float(np.max(np.abs(z[:, 2:]))),
        "minimum_speed": float(np.min(speed)),
        "sample_s": s,
        "sample_path": points,
    }


@dataclass(frozen=True)
class PointwiseNNIVMConfig:
    alpha: float = 1.0e-3
    force_budget: int = 5000
    cost_offset: int = 0
    initial_path_file: str = ""
    max_outer: int = 100
    inner_calls: int = 1
    unconstrained_lbfgs: bool = False
    sampling_jitter: float = 0.0
    sampling_seed: int = 20260802
    sampling_mode: str = "random"
    adaptive_sampling: bool = False
    adaptive_strength: float = 4.0
    adaptive_uniform_floor: float = 0.25
    equal_arc_compression: bool = False
    tangent_fit_weight: float = 0.0
    force_aligned_hermite: bool = False
    improved_tangent_hermite: bool = False
    direct_node_fit: bool = False
    reference_preserving_fit: bool = False
    preserve_weight: float = 1.0
    incremental_compression: bool = False
    incremental_start_outer: int = 0
    incremental_update_cap: float = 0.03
    rbf_width: int = 48
    rbf_sigma_scale: float = 2.0
    fit_points: int = 513
    ridge: float = 1.0e-11
    arclength_points: int = 2049
    certificate_points: int = 2001
    trust_radius: float = 0.15
    residual_rtol: float = 1.0e-3
    late_subproblem_start: int = 0
    late_inner_calls: int = 0
    late_residual_rtol: float = 0.0
    secant_warm_start: bool = False
    secant_memory: float = 0.5
    secant_curvature_clip: float = 500.0
    role: str = "nn_ivm_pointwise_clean"


def run_experiment(root: Path, config: PointwiseNNIVMConfig) -> dict:
    if config.alpha <= 0.0 or config.inner_calls < 1 or not 0.0 <= config.sampling_jitter < 0.5:
        raise ValueError("invalid alpha, inner_calls, or jitter")
    root = Path(root)
    run_dir = root / config.role
    run_dir.mkdir(parents=True, exist_ok=True)
    initial = transformed_initial_path(22)
    start, end = initial[0], initial[-1]
    if config.initial_path_file:
        resume = np.load(config.initial_path_file, allow_pickle=False)
        resume_path = np.asarray(resume["dense_sample_path"][-1], dtype=np.float64)
        resume_s = np.linspace(0.0, 1.0, resume_path.shape[0])
        current, _ = _fit_rbf(
            resume_path[0],
            resume_path[-1],
            resume_s[1:-1],
            resume_path[1:-1],
            width=config.rbf_width,
            sigma_scale=config.rbf_sigma_scale,
            ridge=config.ridge,
            equal_arc=False,
            fit_points=config.fit_points,
        )
        start, end = resume_path[0], resume_path[-1]
    else:
        current = EndpointRBFPath.linear(start, end, config.rbf_width, config.rbf_sigma_scale)
    oracle = StreamOracle(config.force_budget - config.cost_offset)
    states = [current.state()]
    history = {
        key: [] for key in (
            "number_of_forces", "outer_iteration", "accepted", "objective_change_max",
            "subproblem_residual_max", "normal_leakage_max", "compression_coordinate_rms",
            "compression_coordinate_max", "collocation_fractions", "collocation_s",
            "collocation_path", "stream_force_counts", "stream_cache_hits"
        )
    }

    def record(outer: int, accepted: bool, fractions=None, sample_s=None, sample_path=None, report=None, fit=None):
        report = report or []
        fit = fit or {"coordinate_rms": np.nan, "coordinate_max": np.nan}
        values = {
            "number_of_forces": config.cost_offset + oracle.total,
            "outer_iteration": outer,
            "accepted": accepted,
            "objective_change_max": max((item["objective_change"] for item in report), default=np.nan),
            "subproblem_residual_max": max((item["residual"] for item in report), default=np.nan),
            "normal_leakage_max": max((item["normal_leakage"] for item in report), default=np.nan),
            "compression_coordinate_rms": fit["coordinate_rms"],
            "compression_coordinate_max": fit["coordinate_max"],
            "collocation_fractions": np.full(N_SAMPLE, np.nan) if fractions is None else fractions,
            "collocation_s": np.full(N_SAMPLE, np.nan) if sample_s is None else sample_s,
            "collocation_path": np.full((N_SAMPLE, DIMENSION), np.nan) if sample_path is None else sample_path,
            "stream_force_counts": oracle.calls.copy(),
            "stream_cache_hits": oracle.hits.copy(),
        }
        for key, value in values.items():
            history[key].append(value)

    record(0, True)
    stop_reason = "outer_limit"
    warm_curvatures = np.zeros(N_SAMPLE, dtype=np.float64)
    previous_fractions = None
    previous_scores = None
    for outer in range(1, config.max_outer + 1):
        use_late_subproblem = bool(
            config.late_subproblem_start > 0
            and outer >= config.late_subproblem_start
        )
        effective_inner_calls = (
            int(config.late_inner_calls)
            if use_late_subproblem
            else int(config.inner_calls)
        )
        effective_residual_rtol = (
            float(config.late_residual_rtol)
            if use_late_subproblem
            else float(config.residual_rtol)
        )
        use_incremental = bool(
            config.incremental_compression
            and outer > int(config.incremental_start_outer)
        )
        required_calls = N_SAMPLE * (
            1 + effective_inner_calls + int(use_incremental)
        )
        if config.force_budget - config.cost_offset - oracle.total < required_calls:
            stop_reason = "budget"
            break
        if config.adaptive_sampling and previous_fractions is not None:
            fractions = _adaptive_residual_fractions(
                outer,
                config.sampling_jitter,
                config.sampling_seed,
                previous_fractions,
                previous_scores,
                config.adaptive_strength,
                config.adaptive_uniform_floor,
            )
            if config.secant_warm_start:
                warm_curvatures = np.interp(
                    fractions,
                    previous_fractions,
                    warm_curvatures,
                )
        else:
            fractions = _stratified_fractions(
                outer,
                config.sampling_jitter,
                config.sampling_seed,
                config.sampling_mode,
            )
        sample_s = _arc_parameters(current, fractions, config.arclength_points)
        old_points = current.coordinates(sample_s)
        endpoint_energies = energy_values(np.vstack((start, end)))
        old_energies = np.empty(N_SAMPLE)
        old_gradients = np.empty_like(old_points)
        for stream, point in enumerate(old_points):
            old_energies[stream], old_gradients[stream] = oracle.query(stream, point)
        full_path = np.vstack((start, old_points, end))
        full_energies = np.concatenate(([endpoint_energies[0]], old_energies, [endpoint_energies[1]]))
        tangents = improved_tangents(full_path, full_energies)
        proximal_solver = (
            _proximal_scalar_unconstrained
            if config.unconstrained_lbfgs
            else _proximal_scalar
        )
        reports = [
            proximal_solver(
                oracle, stream, old_points[stream], tangents[stream], config.alpha,
                effective_inner_calls,
                config.trust_radius,
                effective_residual_rtol,
                warm_curvatures[stream] if config.secant_warm_start else 0.0,
            )
            for stream in range(N_SAMPLE)
        ]
        if config.secant_warm_start:
            observed_curvatures = np.asarray(
                [item["potential_curvature"] for item in reports],
                dtype=np.float64,
            )
            observed_curvatures = np.clip(
                observed_curvatures,
                -float(config.secant_curvature_clip),
                float(config.secant_curvature_clip),
            )
            memory = float(config.secant_memory)
            warm_curvatures = (
                memory * warm_curvatures
                + (1.0 - memory) * observed_curvatures
            )
        targets = np.asarray([item["point"] for item in reports])
        target_energies = np.asarray([item["energy"] for item in reports])
        target_full_path = np.vstack((start, targets, end))
        target_full_energies = np.concatenate(
            ([endpoint_energies[0]], target_energies, [endpoint_energies[1]])
        )
        target_tangents = improved_tangents(target_full_path, target_full_energies)
        target_gradients = np.asarray([item["gradient"] for item in reports])
        if config.adaptive_sampling:
            projected = target_gradients - target_tangents * np.sum(
                target_gradients * target_tangents, axis=1, keepdims=True
            )
            previous_fractions = fractions.copy()
            previous_scores = np.linalg.norm(projected, axis=1)
        gradient_norms = np.linalg.norm(target_gradients, axis=1)
        force_aligned_tangents = target_tangents.copy()
        usable = gradient_norms > 1.0e-12
        force_aligned_tangents[usable] = target_gradients[usable] / gradient_norms[usable, None]
        signs = np.sign(np.sum(force_aligned_tangents * target_tangents, axis=1))
        signs[signs == 0.0] = 1.0
        force_aligned_tangents *= signs[:, None]
        if use_incremental:
            candidate, fit = _increment_rbf(
                current, sample_s, targets, ridge=config.ridge,
                update_cap=config.incremental_update_cap,
            )
            accepted_points = candidate.coordinates(sample_s)
            audited_reports = []
            for stream, point in enumerate(accepted_points):
                accepted_energy, accepted_gradient = oracle.query(stream, point)
                displacement = point - old_points[stream]
                audited_reports.append({
                    "point": point,
                    "energy": accepted_energy,
                    "gradient": accepted_gradient,
                    "objective_change": float(
                        accepted_energy
                        + np.dot(displacement, displacement) / (2.0 * config.alpha)
                        - old_energies[stream]
                    ),
                    "calls": 1,
                    "residual": float(np.linalg.norm(
                        accepted_gradient
                        - tangents[stream] * np.dot(accepted_gradient, tangents[stream])
                        + displacement / config.alpha
                    )),
                    "normal_leakage": float(abs(np.dot(displacement, tangents[stream]))),
                })
            reports = audited_reports
        elif config.reference_preserving_fit:
            candidate, fit = _fit_nodes_preserving_reference(
                current, start, end, targets, width=config.rbf_width,
                sigma_scale=config.rbf_sigma_scale, ridge=config.ridge,
                preserve_weight=config.preserve_weight,
                fit_points=config.fit_points,
                arclength_points=config.arclength_points,
            )
        elif config.direct_node_fit:
            candidate, fit = _fit_nodes_rbf(
                start, end, targets, width=config.rbf_width,
                sigma_scale=config.rbf_sigma_scale, ridge=config.ridge,
            )
        else:
            candidate, fit = _fit_rbf(
                start, end, sample_s, targets, width=config.rbf_width,
                sigma_scale=config.rbf_sigma_scale, ridge=config.ridge,
                equal_arc=config.equal_arc_compression, fit_points=config.fit_points,
                target_tangents=target_tangents,
                tangent_weight=config.tangent_fit_weight,
                hermite_tangents=(
                    target_tangents
                    if config.improved_tangent_hermite
                    else (
                        force_aligned_tangents
                        if config.force_aligned_hermite
                        else None
                    )
                ),
            )
        objective_ok = bool(max(item["objective_change"] for item in reports) <= 1.0e-10)
        compression_ok = bool(fit["coordinate_max"] <= 5.0e-3)
        accepted = objective_ok and compression_ok
        if accepted:
            current = candidate
            states.append(current.state())
        record(outer, accepted, fractions, sample_s, targets, reports, fit)

    certificates = []
    for state in states:
        model = EndpointRBFPath(state["start"], state["end"], state["centers"], state["sigma"], state["weights"])
        certificates.append(_dense_certificate(model, config.certificate_points))
    accepted_mask = np.asarray(history["accepted"], dtype=bool)
    accepted_forces = np.asarray(history["number_of_forces"], dtype=np.int64)[accepted_mask]
    dense_i = np.asarray([item["dense_i_perp"] for item in certificates])
    dense_delta = np.asarray([item["dense_delta_e"] for item in certificates])
    dense_error = np.asarray([item["dense_barrier_error"] for item in certificates])
    dense_inactive = np.asarray([item["maximum_inactive_coordinate"] for item in certificates])
    dense_speed = np.asarray([item["minimum_speed"] for item in certificates])
    payload = {key: np.asarray(value) for key, value in history.items()}
    payload.update({
        "accepted_number_of_forces": accepted_forces,
        "dense_i_perp": dense_i,
        "dense_delta_e": dense_delta,
        "dense_barrier_error": dense_error,
        "dense_maximum_inactive_coordinate": dense_inactive,
        "dense_minimum_speed": dense_speed,
        "dense_sample_s": np.asarray([item["sample_s"] for item in certificates]),
        "dense_sample_path": np.asarray([item["sample_path"] for item in certificates]),
        "configuration_json": np.asarray(json.dumps(asdict(config), sort_keys=True)),
        "cost_definition": np.asarray("direct sum of 20 independent moving-sample force streams; rejected trials included; cache hits excluded"),
        "dense_information_enters_optimization": np.asarray(False),
        "fixed_alpha_audit": np.asarray(True),
        "stop_reason": np.asarray(stop_reason),
    })
    signature = _hash(asdict(config))
    output = run_dir / f"q10_nn_ivm_pointwise_{signature}.npz"
    np.savez_compressed(output, **payload)
    audit = {
        "configuration": asdict(config),
        "configuration_id": signature,
        "output": str(output),
        "optimization_force_calls": oracle.total,
        "accepted_updates": int(len(states) - 1),
        "stop_reason": stop_reason,
        "final_dense_i_perp": float(dense_i[-1]),
        "minimum_dense_i_perp": float(np.min(dense_i)),
        "final_dense_delta_e": float(dense_delta[-1]),
        "final_dense_barrier_error": float(dense_error[-1]),
        "minimum_dense_barrier_error": float(np.min(dense_error)),
        "maximum_inactive_coordinate": float(np.max(dense_inactive)),
        "maximum_compression_error": float(np.nanmax(np.asarray(history["compression_coordinate_max"], dtype=np.float64))),
        "proximal_descent_audit": bool(np.nanmax(np.asarray(history["objective_change_max"], dtype=np.float64)) <= 1.0e-10),
        "normal_projection_audit": bool(np.nanmax(np.asarray(history["normal_leakage_max"], dtype=np.float64)) <= 1.0e-10),
        "fixed_alpha": True,
        "unconstrained_delta_audit": bool(config.unconstrained_lbfgs),
        "dense_information_enters_optimization": False,
        "force_count_definition": "direct sum of 20 independent moving-sample force streams",
    }
    audit["theory_audit_pass"] = bool(
        audit["proximal_descent_audit"]
        and audit["normal_projection_audit"]
        and audit["maximum_inactive_coordinate"] <= 1.0e-8
        and audit["unconstrained_delta_audit"]
    )
    audit_path = run_dir / f"q10_nn_ivm_pointwise_{signature}.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit
