"""Riemannian path geometry for the alanine-dipeptide 2D CV MFEP.

The public metrics in this module intentionally reproduce the definitions in
the plotting notebook behind ``MFEP_IVM_vs_FENEB_bestpair_threepanel.pdf``.
The only extension is explicit shortest-branch handling for the two periodic
dihedral coordinates.  On the historical C5 -> C7eq path this extension is
numerically inactive, but it prevents an accidental 2*pi jump in new runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


Array = np.ndarray
TWO_PI = 2.0 * np.pi
REPARAMETERIZATION_RELATIVE_CHORD_TOLERANCE = 1.0e-10
REPARAMETERIZATION_MAX_ITERATIONS = 50
REPARAMETERIZATION_ZERO_SEGMENT_THRESHOLD = 1.0e-14


class GeometryError(RuntimeError):
    """Raised when a path or sampled metric fails a fail-closed check."""


def _period_array(periods: Optional[Sequence[float]], dimension: int) -> Array:
    if periods is None:
        return np.full(dimension, np.inf, dtype=np.float64)
    result = np.asarray(periods, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(dimension, float(result), dtype=np.float64)
    return result.reshape(dimension)


def periodic_difference(
    q1: Array,
    q0: Array,
    periods: Optional[Sequence[float]] = (TWO_PI, TWO_PI),
) -> Array:
    """Return the shortest periodic displacement ``q1-q0``."""

    difference = np.asarray(q1, dtype=np.float64) - np.asarray(
        q0, dtype=np.float64
    )
    period_array = _period_array(periods, difference.shape[-1])
    result = np.array(difference, copy=True)
    for coordinate, period in enumerate(period_array):
        if np.isfinite(period) and period > 0.0:
            result[..., coordinate] -= period * np.round(
                result[..., coordinate] / period
            )
    return result


def wrap_periodic(
    q: Array,
    periods: Optional[Sequence[float]] = (TWO_PI, TWO_PI),
) -> Array:
    """Wrap periodic coordinates to the centered principal interval."""

    q = np.asarray(q, dtype=np.float64)
    result = np.array(q, copy=True)
    period_array = _period_array(periods, q.shape[-1])
    for coordinate, period in enumerate(period_array):
        if np.isfinite(period) and period > 0.0:
            result[..., coordinate] = (
                (result[..., coordinate] + 0.5 * period) % period
                - 0.5 * period
            )
    return result


def unwrap_path(
    path: Array,
    periods: Optional[Sequence[float]] = (TWO_PI, TWO_PI),
) -> Array:
    """Choose a continuous branch of a periodic polyline."""

    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2:
        raise ValueError("path must have shape (n>=2, dimension)")
    result = np.empty_like(path)
    result[0] = path[0]
    for index in range(1, path.shape[0]):
        result[index] = result[index - 1] + periodic_difference(
            path[index], result[index - 1], periods
        )
    return result


def centered_tangent(
    full_path: Array,
    periods: Optional[Sequence[float]] = (TWO_PI, TWO_PI),
) -> Array:
    """Centered chord tangent used by the article's 2D MFEP example."""

    path = unwrap_path(full_path, periods)
    tangent = path[2:] - path[:-2]
    if not np.all(np.isfinite(tangent)):
        raise GeometryError("non-finite centered tangent")
    if np.any(np.linalg.norm(tangent, axis=1) <= 1.0e-12):
        raise GeometryError("near-zero centered tangent")
    return tangent


@dataclass(frozen=True)
class MetricGeometry:
    """Frozen metric, projectors, and one-dimensional G-normal bases."""

    M: Array
    G: Array
    tangent: Array
    P_parallel: Array
    P_perp: Array
    normal_basis: Array
    metric_eigenvalues: Array


def build_metric_geometry(
    full_path: Array,
    M_all: Array,
    periods: Optional[Sequence[float]] = (TWO_PI, TWO_PI),
    min_eigenvalue: float = 1.0e-10,
) -> MetricGeometry:
    """Build the article projector and a G-orthonormal normal basis.

    In two dimensions the G-normal space is one dimensional.  If ``t`` is the
    path tangent, a Euclidean vector perpendicular to ``G t`` lies in that
    space.  It is then normalized so that ``B.T @ G @ B == 1``.
    """

    full_path = np.asarray(full_path, dtype=np.float64)
    M_all = np.asarray(M_all, dtype=np.float64)
    tangent = centered_tangent(full_path, periods)
    n_internal, dimension = tangent.shape
    if dimension != 2:
        raise ValueError("the alanine CV geometry must be two dimensional")
    if M_all.shape != (n_internal, dimension, dimension):
        raise ValueError(
            f"M shape {M_all.shape} is incompatible with path {full_path.shape}"
        )
    if not np.all(np.isfinite(M_all)):
        raise GeometryError("sampled metric contains non-finite values")

    symmetric_M = 0.5 * (M_all + np.swapaxes(M_all, 1, 2))
    eigenvalues = np.linalg.eigvalsh(symmetric_M)
    if np.any(eigenvalues <= float(min_eigenvalue)):
        index = np.argwhere(eigenvalues <= float(min_eigenvalue))[0]
        raise GeometryError(
            "sampled metric is not safely positive definite: "
            f"image={int(index[0])}, eigenvalue={eigenvalues[tuple(index)]:.6e}"
        )
    G_all = np.linalg.inv(symmetric_M)
    identity = np.eye(dimension, dtype=np.float64)
    P_parallel = np.empty((n_internal, dimension, dimension), dtype=np.float64)
    P_perp = np.empty_like(P_parallel)
    normal_basis = np.empty((n_internal, dimension), dtype=np.float64)

    for image_index, (tangent_i, G_i) in enumerate(zip(tangent, G_all)):
        denominator = float(tangent_i @ G_i @ tangent_i)
        if not np.isfinite(denominator) or denominator <= 1.0e-30:
            raise GeometryError(
                f"invalid G-tangent norm at image {image_index}: {denominator}"
            )
        P_parallel_i = np.outer(tangent_i, G_i @ tangent_i) / denominator
        P_parallel[image_index] = P_parallel_i
        P_perp[image_index] = identity - P_parallel_i

        G_tangent = G_i @ tangent_i
        candidate = np.array([-G_tangent[1], G_tangent[0]], dtype=np.float64)
        candidate_norm = float(np.sqrt(candidate @ G_i @ candidate))
        if not np.isfinite(candidate_norm) or candidate_norm <= 1.0e-30:
            raise GeometryError(
                f"invalid G-normal basis at image {image_index}"
            )
        basis_i = candidate / candidate_norm
        # Fix the otherwise arbitrary sign for deterministic trajectories.
        pivot = int(np.argmax(np.abs(basis_i)))
        if basis_i[pivot] < 0.0:
            basis_i = -basis_i
        normal_basis[image_index] = basis_i

    normal_norm = np.einsum(
        "ni,nij,nj->n", normal_basis, G_all, normal_basis
    )
    normal_tangent = np.einsum(
        "ni,nij,nj->n", normal_basis, G_all, tangent
    )
    if not np.allclose(normal_norm, 1.0, rtol=1.0e-10, atol=1.0e-10):
        raise GeometryError("constructed normal basis is not G-normalized")
    if not np.allclose(normal_tangent, 0.0, rtol=0.0, atol=1.0e-10):
        raise GeometryError("constructed normal basis is not G-orthogonal")

    return MetricGeometry(
        M=symmetric_M,
        G=G_all,
        tangent=tangent,
        P_parallel=P_parallel,
        P_perp=P_perp,
        normal_basis=normal_basis,
        metric_eigenvalues=eigenvalues,
    )


def equal_euclidean_arclength_reparameterize(
    full_path: Array,
    periods: Optional[Sequence[float]] = (TWO_PI, TWO_PI),
    tolerance: float = REPARAMETERIZATION_RELATIVE_CHORD_TOLERANCE,
    max_iterations: int = REPARAMETERIZATION_MAX_ITERATIONS,
    zero_segment_threshold: float = REPARAMETERIZATION_ZERO_SEGMENT_THRESHOLD,
) -> Array:
    """Deterministically relax a periodic chain to equal Euclidean chords.

    One arclength resampling of a bent polyline can connect points on opposite
    sides of an old vertex, so the new chord lengths are not exactly equal.
    Repeating the force-free resampling converges rapidly to a discrete chain
    whose *actual* adjacent chord lengths satisfy the registered tolerance.
    The operation preserves both endpoints and uses no mean-force evaluation.
    """

    original = np.asarray(full_path, dtype=np.float64)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("reparameterization tolerance must be positive")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("reparameterization max_iterations must be a positive integer")
    if not np.isfinite(zero_segment_threshold) or zero_segment_threshold <= 0.0:
        raise ValueError("zero_segment_threshold must be positive")
    result = unwrap_path(original, periods)
    endpoint_start = result[0].copy()
    endpoint_end = result[-1].copy()

    for _ in range(max_iterations):
        segments = np.diff(result, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        if np.any(~np.isfinite(lengths)) or np.any(
            lengths <= float(zero_segment_threshold)
        ):
            raise GeometryError("cannot reparameterize a path with a zero segment")
        mean_length = float(np.mean(lengths))
        relative_deviation = float(
            np.max(np.abs(lengths / mean_length - 1.0))
        )
        if relative_deviation <= float(tolerance):
            result[0] = endpoint_start
            result[-1] = endpoint_end
            return result

        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        targets = np.linspace(0.0, float(cumulative[-1]), result.shape[0])
        result = np.column_stack(
            [
                np.interp(targets, cumulative, result[:, axis])
                for axis in range(result.shape[1])
            ]
        )
        result[0] = endpoint_start
        result[-1] = endpoint_end

    lengths = np.linalg.norm(np.diff(result, axis=0), axis=1)
    relative_deviation = float(
        np.max(np.abs(lengths / float(np.mean(lengths)) - 1.0))
    )
    if relative_deviation <= float(tolerance):
        return result
    raise GeometryError(
        "equal-chord reparameterization did not reach the registered "
        f"relative tolerance {tolerance:.3e} in {max_iterations} iterations; "
        f"last deviation={relative_deviation:.3e}"
    )


def normal_coordinates(
    candidate_path: Array,
    reference_path: Array,
    geometry: MetricGeometry,
    periods: Optional[Sequence[float]] = (TWO_PI, TWO_PI),
) -> Array:
    """Project candidate-reference displacement onto each frozen G-normal."""

    candidate = unwrap_path(candidate_path, periods)
    reference = unwrap_path(reference_path, periods)
    displacement = candidate[1:-1] - reference[1:-1]
    return np.einsum(
        "ni,nij,nj->n", geometry.normal_basis, geometry.G, displacement
    )


def proximal_normal_residual(
    grad_F: Array,
    normal_coordinate: Array,
    normal_basis: Array,
    alpha: float,
) -> Array:
    """Scalar gradient of each frozen Riemannian normal subproblem."""

    grad_F = np.asarray(grad_F, dtype=np.float64)
    normal_coordinate = np.asarray(normal_coordinate, dtype=np.float64)
    normal_basis = np.asarray(normal_basis, dtype=np.float64)
    return np.einsum("ni,ni->n", normal_basis, grad_F) + normal_coordinate / float(
        alpha
    )


def recover_free_energy_profile(full_path: Array, grad_F: Array) -> Dict[str, Array]:
    """Reproduce the article's segment-wise mean-force line integral."""

    path = unwrap_path(full_path)
    grad_F = np.asarray(grad_F, dtype=np.float64)
    n_internal = path.shape[0] - 2
    if grad_F.shape != (n_internal, 2):
        raise ValueError("grad_F has the wrong shape")
    segment_gradients = np.zeros((n_internal + 1, 2), dtype=np.float64)
    segment_gradients[0] = grad_F[0]
    segment_gradients[-1] = grad_F[-1]
    if n_internal > 1:
        segment_gradients[1:-1] = 0.5 * (grad_F[:-1] + grad_F[1:])
    increments = np.einsum("ni,ni->n", segment_gradients, np.diff(path, axis=0))
    profile = np.zeros(path.shape[0], dtype=np.float64)
    profile[1:] = np.cumsum(increments)
    barrier = float(np.max(profile) - profile[0])
    return {
        "profile": profile,
        "increments": increments,
        "barrier": np.asarray(barrier),
        "max_index": np.asarray(int(np.argmax(profile))),
        "final_delta_F": np.asarray(float(profile[-1] - profile[0])),
    }


def arclength_residual(full_path: Array) -> float:
    """Article equi-speed residual I_arc on uniform path parameters."""

    path = unwrap_path(full_path)
    ds = 1.0 / float(path.shape[0] - 1)
    phi_prime = (path[2:] - path[:-2]) / (2.0 * ds)
    phi_second = (path[2:] - 2.0 * path[1:-1] + path[:-2]) / (ds * ds)
    inner = np.einsum("ni,ni->n", phi_prime, phi_second)
    speed = np.linalg.norm(phi_prime, axis=1)
    return float(np.sum(inner * inner * speed) * ds)


def compute_article_metrics(full_path: Array, M_all: Array, grad_F: Array) -> Dict[str, Array]:
    """Compute all three public panels plus diagnostic arrays."""

    path = unwrap_path(full_path)
    geometry = build_metric_geometry(path, M_all)
    grad_F = np.asarray(grad_F, dtype=np.float64)
    if grad_F.shape != (path.shape[0] - 2, 2):
        raise ValueError("grad_F has the wrong shape")
    Mg = np.einsum("nij,nj->ni", geometry.M, grad_F)
    residual = np.einsum("nij,nj->ni", geometry.P_perp, Mg)
    residual_G_sq = np.maximum(
        np.einsum("ni,nij,nj->n", residual, geometry.G, residual), 0.0
    )
    internal = path[1:-1]
    minus = internal - path[:-2]
    plus = path[2:] - internal
    weight_minus = np.sqrt(
        np.maximum(np.einsum("ni,nij,nj->n", minus, geometry.G, minus), 0.0)
    )
    weight_plus = np.sqrt(
        np.maximum(np.einsum("ni,nij,nj->n", plus, geometry.G, plus), 0.0)
    )
    weights_G = 0.5 * (weight_minus + weight_plus)
    euclidean_segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    segment_length_relative_deviation = float(
        np.max(
            np.abs(
                euclidean_segment_lengths
                / float(np.mean(euclidean_segment_lengths))
                - 1.0
            )
        )
    )
    free_energy = recover_free_energy_profile(path, grad_F)
    return {
        "I_perp_G": np.asarray(float(np.sum(residual_G_sq * weights_G))),
        "I_arc": np.asarray(arclength_residual(path)),
        "Delta_F_barrier": free_energy["barrier"],
        "Delta_F_final": free_energy["final_delta_F"],
        "free_energy_profile": free_energy["profile"],
        "perp_G_Mg": residual,
        "weights_G": weights_G,
        "euclidean_segment_lengths": euclidean_segment_lengths,
        "segment_length_relative_deviation": np.asarray(
            segment_length_relative_deviation
        ),
        "maximum_euclidean_segment_length": np.asarray(
            float(np.max(euclidean_segment_lengths))
        ),
        "metric_eigenvalues": geometry.metric_eigenvalues,
    }


def periodic_path_rms_distance(path_a: Array, path_b: Array) -> float:
    """Pointwise RMS angular distance for two equally sampled paths."""

    a = np.asarray(path_a, dtype=np.float64)
    b = np.asarray(path_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("path shapes differ")
    displacement = periodic_difference(a, b)
    return float(np.sqrt(np.mean(np.sum(displacement * displacement, axis=1))))
