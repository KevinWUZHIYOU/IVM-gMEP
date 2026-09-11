"""Development V14 core for noisy Riemannian Ala2D IR-IVM.

This module is deliberately a *development* implementation.  It does not
instantiate an oracle and, in particular, never instantiates
``RestrainedMDOracle``.  A caller supplies an object with the same
``evaluate_band`` protocol as :class:`ala2d_oracles.OracleBand`.

The solver is band synchronous.  One outer subproblem freezes the reference
path, article centered tangent, sampled ``M``, ``G=M^-1``, and the
one-dimensional ``G``-normal, ``G``-orthonormal basis at all twenty internal
images.  Every paid, reparameterized path is represented in that frozen frame
by

    u_i = B_i.T @ G_i @ (x_i - x_ref_i)

and its proximal residual is

    z_i = B_i.T @ grad_F_i + u_i / alpha.

Block means give a standard error for ``z``.  The direction uses the fixed
smooth shrinkage ``z_eff = z*z**2/(z**2 + SE**2)``.  Historical paid bands
provide a local diagonal H0 regression before takeover, while each frozen
frame reconstructs a joint seven-pair L-BFGS model from the local distinct-path
suffix.  The 66D one-way selector is global: after the first strict L-BFGS win,
H0/cold recovery is permanently forbidden.  A rebuilt frame with fewer than
two valid pairs uses only the persisted bounded warm scalar safeguard.

Both genuine candidates are passed through the unchanged article spring
reparameterizer before selection.  Only the selected complete band is sent
to the supplied oracle.  Its *actual reparameterized coordinates* are then
projected back into the frozen frame, so the reparameterization mapping is
included in residuals and learned secants.

Persistence follows the V13 public-state-first/checkpoint-second protocol.
Every paid public state costs exactly twenty image-wise mean-force
evaluations.  If interruption occurs between those two atomic publications,
the one-state-ahead checkpoint is reconstructed by deterministic replay.
Completed per-image oracle artifacts remain the oracle's responsibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ala2d_article_neb_reparam import (
    ArticleReparameterizationError,
    ArticleReparameterizationResult,
    REPARAMETERIZER_ID,
    reparameterize_article_neb,
)
from ala2d_oracles import OracleBand
from riemannian_torus_geometry import (
    build_metric_geometry,
    compute_article_metrics,
    periodic_difference,
    unwrap_path,
    wrap_periodic,
)


Array = np.ndarray
IMPLEMENTATION_ID = (
    "ala2d-riemannian-v14-development-frozen-noise-aware-joint-lbfgs"
)
DEVELOPMENT_ONLY = True
STATE_KIND = "reparameterized_frozen_inner_trial"
HOLD_STATE_KIND = "exact_center_hold_resample"
CHECKPOINT_PHASE = "v14_development_paid_state_ready"
N_INTERNAL = 20
N_PATH = N_INTERNAL + 2
DIMENSION = 2
JOINT_HISTORY_SIZE = 7
MAX_PAID_OBSERVATIONS = 16


class IRIVMError(RuntimeError):
    """Fail-closed V14 algorithm, oracle, or persistence error."""


@dataclass(frozen=True)
class IRIVMControls:
    """Three development-grid controls plus fixed, preregistrable safeguards.

    Only ``alpha``, ``r`` and ``m_max`` are intended grid dimensions.  The
    remaining fields make the development hypothesis executable and auditable;
    validation fixes them so they cannot silently become extra tuned knobs.
    """

    alpha: float
    r: float
    m_max: int
    history_size: int = JOINT_HISTORY_SIZE
    observation_history_size: int = MAX_PAID_OBSERVATIONS
    kkt_floor: float = 1.0e-14
    secant_norm_floor: float = 1.0e-14
    secant_curvature_cosine_minimum: float = 1.0e-4
    secant_noise_z: float = 2.0
    residual_bound_kappa: float = 2.262
    regression_minimum_points: int = 3
    regression_minimum_effective_points: float = 2.0
    regression_noise_floor: float = 1.0e-8
    regression_age_half_life_bands: float = 16.0
    regression_inverse_scale_minimum_factor: float = 0.25
    regression_inverse_scale_maximum_factor: float = 4.0
    trust_radius_fraction: float = 2.0
    trust_radius_maximum_rad: float = 1.0
    reparameterization_k_s: float = 1.0
    reparameterization_dtau: float = 0.5
    reparameterization_tolerance: float = 1.0e-6
    reparameterization_max_iterations: int = 20000

    def validate(self) -> "IRIVMControls":
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be positive and finite")
        if not np.isfinite(self.r) or not 0.0 < self.r < 0.8:
            raise ValueError("V14 requires 0 < r < 0.8")
        if type(self.m_max) is not int or self.m_max <= 3:
            raise ValueError("V14 requires integer m_max > 3")
        fixed: Dict[str, object] = {
            "history_size": JOINT_HISTORY_SIZE,
            "observation_history_size": MAX_PAID_OBSERVATIONS,
            "kkt_floor": 1.0e-14,
            "secant_norm_floor": 1.0e-14,
            "secant_curvature_cosine_minimum": 1.0e-4,
            "secant_noise_z": 2.0,
            "residual_bound_kappa": 2.262,
            "regression_minimum_points": 3,
            "regression_minimum_effective_points": 2.0,
            "regression_noise_floor": 1.0e-8,
            "regression_age_half_life_bands": 16.0,
            "regression_inverse_scale_minimum_factor": 0.25,
            "regression_inverse_scale_maximum_factor": 4.0,
            "trust_radius_fraction": 2.0,
            "trust_radius_maximum_rad": 1.0,
            "reparameterization_k_s": 1.0,
            "reparameterization_dtau": 0.5,
            "reparameterization_tolerance": 1.0e-6,
            "reparameterization_max_iterations": 20000,
        }
        for name, expected in fixed.items():
            observed = getattr(self, name)
            if isinstance(expected, int):
                if type(observed) is not int or observed != expected:
                    raise ValueError(f"V14 development fixes {name}={expected}")
            elif not np.isfinite(observed) or float(observed) != float(expected):
                raise ValueError(f"V14 development fixes {name}={expected}")
        return self


# Explicit V14 name for callers that prefer versioned imports.
IRIVMControlsV14 = IRIVMControls


@dataclass(frozen=True)
class ResidualStatistics:
    normal_coordinate: Array
    projected_gradient: Array
    raw_residual: Array
    residual_block_means: Array
    standard_error: Array
    standard_error_available: Array
    shrinkage_factor: Array
    effective_residual: Array
    raw_rms: float
    effective_rms: float
    lower_norm: float
    upper_norm: float
    bounds_available: bool


@dataclass(frozen=True)
class H0Regression:
    inverse_scale: Array
    raw_gradient_slope: Array
    proximal_curvature: Array
    local_point_count: Array
    effective_point_count: Array
    clipped: Array
    valid: Array
    reason: Array


@dataclass(frozen=True)
class Candidate:
    label: str
    valid: bool
    error: str
    raw_direction: Array
    capped_direction: Array
    common_cap_scale: float
    common_cap_truncated: bool
    raw_path: Array
    reparameterization: Optional[ArticleReparameterizationResult]
    actual_normal_coordinate: Array
    actual_normal_step: Array
    residual_linear_decrease: float
    proximal_quadratic_penalty: float
    admissibility_score: float


@dataclass(frozen=True)
class PreparedUpdate:
    source_checkpoint_sha256: str
    proposal_sha256: str
    source_statistics: ResidualStatistics
    source_segment_lengths: Array
    global_trust_radius: float
    endpoint_trust_radii: Array
    h0_regression: H0Regression
    h0_candidate: Candidate
    lbfgs_candidate: Candidate
    selected_label: str
    selected_path: Array
    latch_before: bool
    latch_after: bool
    strict_lbfgs_win: bool
    forced_lbfgs_after_takeover: bool
    h0_invalid_fallback: bool
    hold_selected: bool
    hold_reason: str
    global_signal_to_noise: float
    admissibility_difference_standard_error: float
    initial_no_se_special_case: bool
    selected_admissibility_standard_error: float
    selected_admissibility_lower_bound: float


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _array_sha256(value: Array) -> str:
    """Hash an ndarray without erasing its dtype or Boolean/integer identity."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("object arrays are not admissible in a V14 digest")
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and not np.little_endian
    ):
        array = array.byteswap().newbyteorder("<")
    array = np.ascontiguousarray(array)
    header = _canonical_json(
        {
            "schema": "BinaryArray/ala2d-irivm-v14-development",
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        }
    )
    return hashlib.sha256(header + b"\x00" + array.tobytes()).hexdigest()


def _array_equal(left: Array, right: Array) -> bool:
    left_value = np.asarray(left)
    right_value = np.asarray(right)
    if left_value.dtype.kind in "fc" or right_value.dtype.kind in "fc":
        return bool(np.array_equal(left_value, right_value, equal_nan=True))
    return bool(np.array_equal(left_value, right_value))


def _atomic_save_npz(path: Path, **arrays: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_publish_npz_once(path: Path, **arrays: Array) -> None:
    """Publish one immutable public state without replacing an existing one."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise IRIVMError(f"refusing to replace public state {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_npz_record(path: Path) -> Dict[str, Array]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.array(loaded[key], copy=True) for key in loaded.files}


def _state_path(run_directory: Path, state_index: int) -> Path:
    return run_directory / "states" / f"state_{int(state_index):06d}.npz"


def noise_shrink_residual(raw_residual: Array, standard_error: Array) -> Tuple[Array, Array]:
    """Apply the fixed V14 smooth residual shrinkage component by component."""

    raw = np.asarray(raw_residual, dtype=np.float64)
    se = np.asarray(standard_error, dtype=np.float64)
    if raw.shape != se.shape or np.any(~np.isfinite(raw)) or np.any(~np.isfinite(se)):
        raise ValueError("raw residual and SE must be finite arrays of one shape")
    if np.any(se < 0.0):
        raise ValueError("residual SE cannot be negative")
    raw_sq = raw * raw
    denominator = raw_sq + se * se
    factor = np.divide(
        raw_sq,
        denominator,
        out=np.zeros_like(raw_sq),
        where=denominator > 0.0,
    )
    effective = raw * factor
    return effective, factor


def _validate_path_metric_gradient(path: Array, M: Array, grad: Array) -> Tuple[Array, Array, Array]:
    path_value = unwrap_path(np.asarray(path, dtype=np.float64))
    M_value = np.asarray(M, dtype=np.float64)
    grad_value = np.asarray(grad, dtype=np.float64)
    if path_value.shape != (N_PATH, DIMENSION):
        raise IRIVMError(f"Ala2D path must have shape {(N_PATH, DIMENSION)}")
    if M_value.shape != (N_INTERNAL, DIMENSION, DIMENSION):
        raise IRIVMError("Ala2D metric has the wrong shape")
    if grad_value.shape != (N_INTERNAL, DIMENSION):
        raise IRIVMError("Ala2D mean-force gradient has the wrong shape")
    if not (
        np.all(np.isfinite(path_value))
        and np.all(np.isfinite(M_value))
        and np.all(np.isfinite(grad_value))
    ):
        raise IRIVMError("Ala2D state contains non-finite values")
    build_metric_geometry(path_value, M_value)
    return path_value, M_value, grad_value


def _validate_gradient_blocks(blocks: Array) -> Array:
    value = np.asarray(blocks, dtype=np.float64)
    if (
        value.ndim != 3
        or value.shape[0] != N_INTERNAL
        or value.shape[2] != DIMENSION
        or value.shape[1] < 1
        or np.any(~np.isfinite(value))
    ):
        raise IRIVMError("gradient block means must have finite shape (20,B,2), B>=1")
    return value


def _validate_metric_blocks(blocks: Array, block_count: Optional[int] = None) -> Array:
    value = np.asarray(blocks, dtype=np.float64)
    if (
        value.ndim != 4
        or value.shape[0] != N_INTERNAL
        or value.shape[2:] != (DIMENSION, DIMENSION)
        or value.shape[1] < 1
        or (block_count is not None and value.shape[1] != int(block_count))
        or np.any(~np.isfinite(value))
    ):
        raise IRIVMError("metric block means must have finite shape (20,B,2,2)")
    symmetric = 0.5 * (value + np.swapaxes(value, 2, 3))
    if np.any(np.linalg.eigvalsh(symmetric) <= 1.0e-10):
        raise IRIVMError("a metric block mean is not safely positive definite")
    return symmetric


def _block_frame_counts(frame_counts: Array, block_count: int) -> Array:
    """Reconstruct ``np.array_split`` retained-frame counts per image/block."""

    totals = np.asarray(frame_counts)
    if (
        totals.shape != (N_INTERNAL,)
        or totals.dtype.kind not in "iu"
        or np.any(totals < int(block_count))
        or type(block_count) is not int
        or block_count < 1
    ):
        raise IRIVMError("retained frame counts cannot define the requested blocks")
    quotient = totals.astype(np.int64) // int(block_count)
    remainder = totals.astype(np.int64) % int(block_count)
    slots = np.arange(int(block_count), dtype=np.int64)[None, :]
    counts = quotient[:, None] + (slots < remainder[:, None]).astype(np.int64)
    if not np.array_equal(np.sum(counts, axis=1), totals.astype(np.int64)):
        raise IRIVMError("reconstructed block frame counts do not sum to totals")
    return counts


def _validate_block_frame_counts(
    counts: Array, block_count: int, *, allow_all_zero: bool = False
) -> Array:
    value = np.asarray(counts)
    if value.shape != (N_INTERNAL, int(block_count)) or value.dtype.kind not in "iu":
        raise IRIVMError("block frame counts must have integer shape (20,B)")
    value = value.astype(np.int64, copy=False)
    if allow_all_zero and np.all(value == 0):
        return value
    if np.any(value <= 0):
        raise IRIVMError("every observed block must contain retained frames")
    return value


def _weighted_block_mean(blocks: Array, block_frame_counts: Array) -> Array:
    value = np.asarray(blocks, dtype=np.float64)
    counts = _validate_block_frame_counts(block_frame_counts, value.shape[1])
    totals = np.sum(counts, axis=1, dtype=np.int64)
    expand = (None,) * (value.ndim - 2)
    weights = counts[(slice(None), slice(None)) + expand]
    return np.sum(value * weights, axis=1) / totals[(slice(None),) + expand]


def _gradient_covariance_of_mean(
    blocks: Array, block_frame_counts: Optional[Array] = None
) -> Tuple[Array, Array]:
    value = _validate_gradient_blocks(blocks)
    count = value.shape[1]
    covariance = np.zeros((N_INTERNAL, DIMENSION, DIMENSION), dtype=np.float64)
    available = np.zeros(N_INTERNAL, dtype=bool)
    if block_frame_counts is None:
        counts = np.ones((N_INTERNAL, count), dtype=np.int64)
    else:
        counts = _validate_block_frame_counts(
            block_frame_counts, count, allow_all_zero=True
        )
    observed = np.all(counts > 0, axis=1)
    if count >= 2 and np.any(observed):
        totals = np.sum(counts, axis=1, dtype=np.int64)
        means = np.zeros((N_INTERNAL, DIMENSION), dtype=np.float64)
        means[observed] = (
            np.einsum("nk,nki->ni", counts[observed], value[observed])
            / totals[observed, None]
        )
        centered = value - means[:, None, :]
        numerator = np.einsum(
            "nk,nki,nkj->nij", counts.astype(np.float64), centered, centered
        )
        covariance[observed] = numerator[observed] / (
            (count - 1) * totals[observed, None, None]
        )
        available[observed] = True
    return covariance, available


def _frozen_frame(
    path: Array,
    M: Array,
    grad: Array,
    previous_basis: Optional[Array] = None,
) -> Dict[str, Array]:
    path_value, M_value, grad_value = _validate_path_metric_gradient(path, M, grad)
    geometry = build_metric_geometry(path_value, M_value)
    basis = np.asarray(geometry.normal_basis, dtype=np.float64).copy()
    gauge_flipped = np.zeros(N_INTERNAL, dtype=bool)
    if previous_basis is not None:
        previous = np.asarray(previous_basis, dtype=np.float64)
        if previous.shape != basis.shape or np.any(~np.isfinite(previous)):
            raise IRIVMError("previous normal basis is invalid for gauge continuation")
        orientation = np.einsum("ni,nij,nj->n", basis, geometry.G, previous)
        gauge_flipped = orientation < 0.0
        basis[gauge_flipped] *= -1.0
    gram = np.einsum("ni,nij,nj->n", basis, geometry.G, basis)
    orthogonality = np.einsum("ni,nij,nj->n", basis, geometry.G, geometry.tangent)
    if not np.allclose(gram, 1.0, rtol=1.0e-10, atol=1.0e-10):
        raise IRIVMError("frozen normal basis is not G-orthonormal")
    if not np.allclose(orthogonality, 0.0, rtol=0.0, atol=1.0e-10):
        raise IRIVMError("frozen normal basis is not G-normal")
    return {
        "path": path_value.copy(),
        "M": np.asarray(geometry.M, dtype=np.float64).copy(),
        "grad": grad_value.copy(),
        "tangent": np.asarray(geometry.tangent, dtype=np.float64).copy(),
        "G": np.asarray(geometry.G, dtype=np.float64).copy(),
        "basis": basis.copy(),
        "gauge_flipped": gauge_flipped,
    }


def _normal_coordinate(path: Array, frozen_path: Array, basis: Array, G: Array) -> Array:
    displacement = periodic_difference(
        np.asarray(path, dtype=np.float64)[1:-1],
        np.asarray(frozen_path, dtype=np.float64)[1:-1],
    )
    coordinate = np.einsum("ni,nij,nj->n", basis, G, displacement)
    if coordinate.shape != (N_INTERNAL,) or np.any(~np.isfinite(coordinate)):
        raise IRIVMError("frozen normal coordinate is invalid")
    return coordinate


def _residual_statistics(
    *,
    path: Array,
    grad: Array,
    grad_blocks: Array,
    frozen_path: Array,
    frozen_basis: Array,
    frozen_G: Array,
    alpha: float,
    bound_kappa: float = 2.262,
    single_block_exact: bool = False,
    block_frame_counts: Optional[Array] = None,
) -> ResidualStatistics:
    blocks = _validate_gradient_blocks(grad_blocks)
    gradient = np.asarray(grad, dtype=np.float64)
    u = _normal_coordinate(path, frozen_path, frozen_basis, frozen_G)
    projected = np.einsum("ni,ni->n", frozen_basis, gradient)
    raw = projected + u / float(alpha)
    projected_blocks = np.einsum("ni,nki->nk", frozen_basis, blocks)
    residual_blocks = projected_blocks + u[:, None] / float(alpha)
    block_count = blocks.shape[1]
    if block_frame_counts is None:
        counts = np.ones((N_INTERNAL, block_count), dtype=np.int64)
    else:
        counts = _validate_block_frame_counts(
            block_frame_counts, block_count, allow_all_zero=True
        )
    observed = np.all(counts > 0, axis=1)
    available = np.full(N_INTERNAL, bool(single_block_exact), dtype=bool)
    if block_count >= 2 and np.any(observed):
        totals = np.sum(counts, axis=1, dtype=np.int64)
        weighted_mean = np.zeros(N_INTERNAL, dtype=np.float64)
        weighted_mean[observed] = (
            np.einsum("nk,nk->n", counts[observed], residual_blocks[observed])
            / totals[observed]
        )
        numerator = np.sum(
            counts.astype(np.float64)
            * (residual_blocks - weighted_mean[:, None]) ** 2,
            axis=1,
        )
        standard_error = np.full(N_INTERNAL, np.nan, dtype=np.float64)
        standard_error[observed] = np.sqrt(
            numerator[observed]
            / ((block_count - 1) * totals[observed].astype(np.float64))
        )
        available[observed] = True
    elif single_block_exact:
        standard_error = np.zeros(N_INTERNAL, dtype=np.float64)
    else:
        # The historical state-0 band has no block information.  NaN plus the
        # availability mask prevents it from masquerading as a precise zero-SE
        # estimate; the one-time initial rule nevertheless uses its raw P.
        standard_error = np.full(N_INTERNAL, np.nan, dtype=np.float64)
    if np.all(available):
        effective, factor = noise_shrink_residual(raw, standard_error)
        lower_norm = float(
            np.linalg.norm(
                np.maximum(np.abs(raw) - float(bound_kappa) * standard_error, 0.0)
            )
        )
        upper_norm = float(
            np.linalg.norm(np.abs(raw) + float(bound_kappa) * standard_error)
        )
        bounds_available = True
    else:
        effective = raw.copy()
        factor = np.ones_like(raw)
        lower_norm = math.nan
        upper_norm = math.nan
        bounds_available = False
    return ResidualStatistics(
        normal_coordinate=u,
        projected_gradient=projected,
        raw_residual=raw,
        residual_block_means=residual_blocks,
        standard_error=standard_error,
        standard_error_available=available,
        shrinkage_factor=factor,
        effective_residual=effective,
        raw_rms=float(np.sqrt(np.mean(raw * raw))),
        effective_rms=float(np.sqrt(np.mean(effective * effective))),
        lower_norm=lower_norm,
        upper_norm=upper_norm,
        bounds_available=bounds_available,
    )


def _trust_radii(frozen_path: Array, controls: IRIVMControls) -> Tuple[Array, float, Array]:
    segments = periodic_difference(frozen_path[1:], frozen_path[:-1])
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= controls.secant_norm_floor):
        raise IRIVMError("frozen reference contains a zero or non-finite segment")
    rho = min(
        controls.trust_radius_fraction * float(np.median(lengths)),
        controls.trust_radius_maximum_rad,
    )
    left = np.cumsum(lengths)[:-1]
    right = np.cumsum(lengths[::-1])[::-1][1:]
    radii = np.minimum(np.minimum(left, right), rho)
    if radii.shape != (N_INTERNAL,) or np.any(radii <= 0.0):
        raise IRIVMError("endpoint-aware frozen trust radii are invalid")
    return lengths, float(rho), np.asarray(radii, dtype=np.float64)


def _whole_band_common_cap(
    direction: Array, basis: Array, radii: Array
) -> Tuple[Array, float, bool]:
    """Apply one scalar physical cap to the complete twenty-image direction."""

    reduced = np.asarray(direction, dtype=np.float64)
    if reduced.shape != (N_INTERNAL,) or np.any(~np.isfinite(reduced)):
        raise IRIVMError("joint direction is invalid")
    physical = np.asarray(basis, dtype=np.float64) * reduced[:, None]
    physical_norms = np.linalg.norm(physical, axis=1)
    ratios = physical_norms / np.asarray(radii, dtype=np.float64)
    maximum_ratio = float(np.max(ratios))
    if not np.isfinite(maximum_ratio):
        raise IRIVMError("whole-band physical cap ratio is non-finite")
    scale = 1.0 if maximum_ratio <= 1.0 else 1.0 / maximum_ratio
    capped = reduced * scale
    # Round toward zero once if binary64 roundoff exceeded a radius.
    capped_physical = np.linalg.norm(basis * capped[:, None], axis=1)
    if np.any(capped_physical > radii):
        scale = float(np.nextafter(scale, 0.0))
        capped = reduced * scale
        capped_physical = np.linalg.norm(basis * capped[:, None], axis=1)
    if np.any(capped_physical > radii):
        raise IRIVMError("whole-band cap failed to enforce every image radius")
    return capped, float(scale), bool(scale < 1.0)


def _fit_h0_regression(
    controller: Dict[str, Any], statistics: ResidualStatistics, controls: IRIVMControls
) -> H0Regression:
    """Fit twenty local gradient slopes from prior complete paid bands."""

    count = int(controller["observation_count"])
    paths = np.asarray(controller["observation_paths"][:count], dtype=np.float64)
    gradients = np.asarray(controller["observation_gradients"][:count], dtype=np.float64)
    covariances = np.asarray(
        controller["observation_gradient_covariances"][:count], dtype=np.float64
    )
    covariance_available = np.asarray(
        controller["observation_covariance_available"][:count], dtype=bool
    )
    state_indices = np.asarray(
        controller["observation_state_indices"][:count], dtype=np.int64
    )
    basis = np.asarray(controller["frozen_basis"], dtype=np.float64)
    G = np.asarray(controller["frozen_G"], dtype=np.float64)
    reference = np.asarray(controller["frozen_path"], dtype=np.float64)
    current = np.asarray(controller["current_path"], dtype=np.float64)
    _, _, local_radii = _trust_radii(reference, controls)

    inverse = np.full(N_INTERNAL, controls.alpha, dtype=np.float64)
    slope = np.full(N_INTERNAL, np.nan, dtype=np.float64)
    proximal = np.full(N_INTERNAL, np.nan, dtype=np.float64)
    point_count = np.zeros(N_INTERNAL, dtype=np.int64)
    effective_count = np.zeros(N_INTERNAL, dtype=np.float64)
    clipped = np.zeros(N_INTERNAL, dtype=bool)
    valid = np.zeros(N_INTERNAL, dtype=bool)
    reason = np.full(N_INTERNAL, "fallback_alpha", dtype="U64")

    age = int(controller["state_index"]) - state_indices
    age_weight = np.exp2(-age / controls.regression_age_half_life_bands)
    for image in range(N_INTERNAL):
        displacement_from_reference = periodic_difference(
            paths[:, image + 1], reference[image + 1]
        )
        x = np.einsum(
            "i,ij,nj->n", basis[image], G[image], displacement_from_reference
        )
        q = np.einsum("i,ni->n", basis[image], gradients[:, image])
        physical_locality = np.linalg.norm(
            periodic_difference(paths[:, image + 1], current[image + 1]), axis=1
        )
        admitted = physical_locality <= float(local_radii[image])
        indices = np.flatnonzero(admitted)
        point_count[image] = indices.size
        if indices.size < controls.regression_minimum_points:
            reason[image] = "too_few_local_paid_points"
            continue
        x_local = x[indices]
        q_local = q[indices]
        variance = np.zeros(indices.size, dtype=np.float64)
        for local_slot, observation in enumerate(indices):
            if covariance_available[observation, image]:
                variance[local_slot] = float(
                    basis[image]
                    @ covariances[observation, image]
                    @ basis[image]
                )
        variance = np.maximum(variance, controls.regression_noise_floor**2)
        spatial = np.exp(
            -0.5
            * (physical_locality[indices] / float(local_radii[image])) ** 2
        )
        weights = spatial * age_weight[indices] / variance
        maximum_weight = float(np.max(weights))
        if not np.isfinite(maximum_weight) or maximum_weight <= 0.0:
            reason[image] = "invalid_regression_weights"
            continue
        weights /= maximum_weight
        weight_sum = float(np.sum(weights))
        weight_square_sum = float(np.dot(weights, weights))
        n_effective = weight_sum * weight_sum / weight_square_sum
        effective_count[image] = n_effective
        if n_effective < controls.regression_minimum_effective_points:
            reason[image] = "too_few_effective_local_points"
            continue
        x_centered = x_local - float(np.dot(weights, x_local) / weight_sum)
        q_centered = q_local - float(np.dot(weights, q_local) / weight_sum)
        denominator = float(np.dot(weights, x_centered * x_centered))
        if (
            not np.isfinite(denominator)
            or denominator <= controls.secant_norm_floor**2
        ):
            reason[image] = "insufficient_local_coordinate_spread"
            continue
        fitted_slope = float(np.dot(weights, x_centered * q_centered) / denominator)
        total_curvature = fitted_slope + 1.0 / controls.alpha
        slope[image] = fitted_slope
        proximal[image] = total_curvature
        if not np.isfinite(total_curvature) or total_curvature <= 0.0:
            reason[image] = "nonpositive_proximal_curvature"
            continue
        raw_inverse = 1.0 / total_curvature
        lower = controls.regression_inverse_scale_minimum_factor * controls.alpha
        upper = controls.regression_inverse_scale_maximum_factor * controls.alpha
        inverse[image] = float(np.clip(raw_inverse, lower, upper))
        clipped[image] = bool(inverse[image] != raw_inverse)
        valid[image] = True
        reason[image] = "local_curvature_regression_clipped" if clipped[image] else "local_curvature_regression"

    return H0Regression(
        inverse_scale=inverse,
        raw_gradient_slope=slope,
        proximal_curvature=proximal,
        local_point_count=point_count,
        effective_point_count=effective_count,
        clipped=clipped,
        valid=valid,
        reason=reason,
    )


def joint_lbfgs_direction(
    gradient: Array, history_s: Array, history_y: Array, diagonal_h0: Array
) -> Array:
    """Return the 66D-style joint L-BFGS direction with diagonal H0."""

    q = np.asarray(gradient, dtype=np.float64).copy()
    s_pairs = np.asarray(history_s, dtype=np.float64)
    y_pairs = np.asarray(history_y, dtype=np.float64)
    diagonal = np.asarray(diagonal_h0, dtype=np.float64)
    if q.shape != (N_INTERNAL,) or diagonal.shape != q.shape:
        raise IRIVMError("joint L-BFGS vector or H0 has the wrong shape")
    if s_pairs.ndim != 2 or s_pairs.shape[1:] != q.shape or s_pairs.shape != y_pairs.shape:
        raise IRIVMError("joint L-BFGS pair arrays have incompatible shapes")
    if np.any(~np.isfinite(q)) or np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise IRIVMError("joint L-BFGS input is invalid")
    coefficients: List[Tuple[float, float, Array, Array]] = []
    for s_value, y_value in zip(s_pairs[::-1], y_pairs[::-1]):
        curvature = float(np.dot(s_value, y_value))
        if not np.isfinite(curvature) or curvature <= 0.0:
            raise IRIVMError("stored joint L-BFGS curvature is not positive")
        rho = 1.0 / curvature
        coefficient = rho * float(np.dot(s_value, q))
        coefficients.append((coefficient, rho, s_value, y_value))
        q -= coefficient * y_value
    result = diagonal * q
    for coefficient, rho, s_value, y_value in coefficients[::-1]:
        beta = rho * float(np.dot(y_value, result))
        result += (coefficient - beta) * s_value
    direction = -result
    if direction.shape != (N_INTERNAL,) or np.any(~np.isfinite(direction)):
        raise IRIVMError("joint L-BFGS returned an invalid direction")
    return direction


def _proximal_admissibility_terms(
    effective_residual: Array, actual_normal_step: Array, alpha: float
) -> Tuple[float, float, float]:
    """Return the frozen-root linear gain, proximal penalty, and score.

    The post-reparameterization candidate is admissible only when

        Psi = -P_eff dot delta - ||delta||^2 / (2 alpha)

    is positive.  The quadratic term is deterministic conditional on the
    candidate path; omitting it would systematically favor large capped steps.
    """

    residual = np.asarray(effective_residual, dtype=np.float64)
    step = np.asarray(actual_normal_step, dtype=np.float64)
    if residual.shape != (N_INTERNAL,) or step.shape != residual.shape:
        raise IRIVMError("admissibility residual/step has the wrong shape")
    if np.any(~np.isfinite(residual)) or np.any(~np.isfinite(step)):
        raise IRIVMError("admissibility residual/step is non-finite")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise IRIVMError("admissibility alpha must be positive and finite")
    linear = -float(np.dot(residual, step))
    quadratic = float(np.dot(step, step)) / (2.0 * float(alpha))
    score = linear - quadratic
    return linear, quadratic, score


def _invalid_candidate(label: str, error: str, direction: Optional[Array] = None) -> Candidate:
    vector = (
        np.full(N_INTERNAL, np.nan, dtype=np.float64)
        if direction is None
        else np.asarray(direction, dtype=np.float64).copy()
    )
    return Candidate(
        label=label,
        valid=False,
        error=error,
        raw_direction=vector,
        capped_direction=np.full(N_INTERNAL, np.nan, dtype=np.float64),
        common_cap_scale=0.0,
        common_cap_truncated=False,
        raw_path=np.full((N_PATH, DIMENSION), np.nan, dtype=np.float64),
        reparameterization=None,
        actual_normal_coordinate=np.full(N_INTERNAL, np.nan, dtype=np.float64),
        actual_normal_step=np.full(N_INTERNAL, np.nan, dtype=np.float64),
        residual_linear_decrease=0.0,
        proximal_quadratic_penalty=0.0,
        admissibility_score=0.0,
    )


def _build_candidate(
    *,
    label: str,
    direction: Array,
    controller: Dict[str, Any],
    statistics: ResidualStatistics,
    radii: Array,
    controls: IRIVMControls,
) -> Candidate:
    """Cap, article-reparameterize, and screen one zero-NoF candidate."""

    try:
        raw_direction = np.asarray(direction, dtype=np.float64)
        if raw_direction.shape != (N_INTERNAL,) or np.any(~np.isfinite(raw_direction)):
            raise IRIVMError("raw direction is non-finite or has the wrong shape")
        raw_decrease = -float(np.dot(statistics.effective_residual, raw_direction))
        if not raw_decrease > controls.kkt_floor:
            raise IRIVMError("raw direction is not effective-residual descent")
        capped, scale, truncated = _whole_band_common_cap(
            raw_direction, controller["frozen_basis"], radii
        )
        raw_path = np.asarray(controller["current_path"], dtype=np.float64).copy()
        raw_path[1:-1] += controller["frozen_basis"] * capped[:, None]
        raw_path = unwrap_path(raw_path)
        reparameterized = reparameterize_article_neb(
            raw_path,
            k_s=controls.reparameterization_k_s,
            dtau=controls.reparameterization_dtau,
            tolerance=controls.reparameterization_tolerance,
            max_iterations=controls.reparameterization_max_iterations,
        )
        actual_u = _normal_coordinate(
            reparameterized.path,
            controller["frozen_path"],
            controller["frozen_basis"],
            controller["frozen_G"],
        )
        actual_step = actual_u - statistics.normal_coordinate
        actual_physical_distance = np.linalg.norm(
            periodic_difference(
                reparameterized.path[1:-1], controller["frozen_path"][1:-1]
            ),
            axis=1,
        )
        if np.any(actual_physical_distance > np.asarray(radii)):
            return Candidate(
                label=label,
                valid=False,
                error="post_reparameterization_path_exceeds_frozen_anchor_trust_radii",
                raw_direction=raw_direction.copy(),
                capped_direction=capped,
                common_cap_scale=scale,
                common_cap_truncated=truncated,
                raw_path=raw_path,
                reparameterization=reparameterized,
                actual_normal_coordinate=actual_u,
                actual_normal_step=actual_step,
                residual_linear_decrease=math.nan,
                proximal_quadratic_penalty=math.nan,
                admissibility_score=math.nan,
            )
        linear, quadratic, score = _proximal_admissibility_terms(
            statistics.effective_residual, actual_step, controls.alpha
        )
        if not np.isfinite(score) or not score > controls.kkt_floor:
            return Candidate(
                label=label,
                valid=False,
                error="post_reparameterization_admissibility_score_nonpositive",
                raw_direction=raw_direction.copy(),
                capped_direction=capped,
                common_cap_scale=scale,
                common_cap_truncated=truncated,
                raw_path=raw_path,
                reparameterization=reparameterized,
                actual_normal_coordinate=actual_u,
                actual_normal_step=actual_step,
                residual_linear_decrease=linear,
                proximal_quadratic_penalty=quadratic,
                admissibility_score=score,
            )
        return Candidate(
            label=label,
            valid=True,
            error="none",
            raw_direction=raw_direction.copy(),
            capped_direction=capped,
            common_cap_scale=scale,
            common_cap_truncated=truncated,
            raw_path=raw_path,
            reparameterization=reparameterized,
            actual_normal_coordinate=actual_u,
            actual_normal_step=actual_step,
            residual_linear_decrease=linear,
            proximal_quadratic_penalty=quadratic,
            admissibility_score=score,
        )
    except (ArticleReparameterizationError, ArithmeticError, FloatingPointError, ValueError, IRIVMError) as error:
        return _invalid_candidate(
            label,
            f"{type(error).__name__}:{error}",
            np.asarray(direction, dtype=np.float64),
        )


def _candidate_digest(candidate: Candidate) -> Dict[str, object]:
    result: Dict[str, object] = {
        "label": candidate.label,
        "valid": bool(candidate.valid),
        "error": candidate.error,
        "raw_direction": _array_sha256(candidate.raw_direction),
        "capped_direction": _array_sha256(candidate.capped_direction),
        "common_cap_scale_binary64": float(candidate.common_cap_scale).hex(),
        "raw_path": _array_sha256(candidate.raw_path),
        "actual_u": _array_sha256(candidate.actual_normal_coordinate),
        "actual_step": _array_sha256(candidate.actual_normal_step),
        "residual_linear_decrease_binary64": float(
            candidate.residual_linear_decrease
        ).hex(),
        "proximal_quadratic_penalty_binary64": float(
            candidate.proximal_quadratic_penalty
        ).hex(),
        "admissibility_score_binary64": float(candidate.admissibility_score).hex(),
    }
    result["reparameterization_audit_sha256"] = (
        "none"
        if candidate.reparameterization is None
        else candidate.reparameterization.audit_sha256
    )
    return result


def _checkpoint_digest(controller: Dict[str, Any]) -> str:
    scalar_keys = (
        "state_index",
        "nof",
        "outer_index",
        "inner_step",
        "observation_count",
        "lbfgs_count",
        "warm_safeguard_source_state_index",
        "lbfgs_latched",
        "single_block_exact_after_initial",
        "outer_anchor_bounds_ready",
    )
    array_keys = (
        "current_path",
        "current_M",
        "current_grad",
        "current_grad_blocks",
        "current_M_blocks",
        "current_block_frame_counts",
        "frozen_path",
        "frozen_M",
        "frozen_grad",
        "frozen_tangent",
        "frozen_G",
        "frozen_basis",
        "outer_initial_effective_rms",
        "outer_anchor_lower_norm",
        "outer_anchor_upper_norm",
        "observation_paths",
        "observation_gradients",
        "observation_gradient_covariances",
        "observation_covariance_available",
        "observation_state_indices",
        "lbfgs_s",
        "lbfgs_y",
        "lbfgs_anchor_u",
        "lbfgs_anchor_raw_residual",
        "lbfgs_anchor_standard_error",
        "warm_safeguard_inverse_scale",
    )
    payload: Dict[str, object] = {
        "schema": "Ala2DIRIVMControllerCheckpoint/v14-development",
        "implementation_id": IMPLEMENTATION_ID,
        "implementation_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    for key in scalar_keys:
        value = controller[key]
        payload[key] = (
            bool(value)
            if key
            in (
                "lbfgs_latched",
                "single_block_exact_after_initial",
                "outer_anchor_bounds_ready",
            )
            else int(value)
        )
    for key in array_keys:
        payload[key] = _array_sha256(np.asarray(controller[key]))
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _append_observation(
    controller: Dict[str, Any],
    path: Array,
    grad: Array,
    blocks: Array,
    block_frame_counts: Array,
    state_index: int,
) -> Dict[str, Array]:
    paths = np.asarray(controller["observation_paths"], dtype=np.float64).copy()
    gradients = np.asarray(controller["observation_gradients"], dtype=np.float64).copy()
    covariances = np.asarray(
        controller["observation_gradient_covariances"], dtype=np.float64
    ).copy()
    available = np.asarray(
        controller["observation_covariance_available"], dtype=bool
    ).copy()
    indices = np.asarray(controller["observation_state_indices"], dtype=np.int64).copy()
    count = int(controller["observation_count"])
    covariance, covariance_available = _gradient_covariance_of_mean(
        blocks, block_frame_counts
    )
    if np.array_equal(paths[count - 1], path):
        # Exact-center resampling refines the existing distinct observation;
        # it cannot manufacture a zero-displacement history point.
        slot = count - 1
    elif count == MAX_PAID_OBSERVATIONS:
        paths[:-1] = paths[1:]
        gradients[:-1] = gradients[1:]
        covariances[:-1] = covariances[1:]
        available[:-1] = available[1:]
        indices[:-1] = indices[1:]
        slot = count - 1
    else:
        slot = count
        count += 1
    paths[slot] = path
    gradients[slot] = grad
    covariances[slot] = covariance
    available[slot] = covariance_available
    indices[slot] = int(state_index)
    return {
        "observation_paths": paths,
        "observation_gradients": gradients,
        "observation_gradient_covariances": covariances,
        "observation_covariance_available": available,
        "observation_state_indices": indices,
        "observation_count": np.asarray(count, dtype=np.int64),
    }


def _new_controller(
    path: Array,
    M: Array,
    grad: Array,
    grad_blocks: Array,
    controls: IRIVMControls,
    M_blocks: Optional[Array] = None,
    single_block_exact_after_initial: bool = False,
) -> Dict[str, Any]:
    path_value, M_value, grad_value = _validate_path_metric_gradient(path, M, grad)
    blocks = _validate_gradient_blocks(grad_blocks)
    metric_blocks = _validate_metric_blocks(
        M_value[:, None, :, :] if M_blocks is None else M_blocks,
        blocks.shape[1],
    )
    initial_block_frame_counts = np.zeros(
        (N_INTERNAL, blocks.shape[1]), dtype=np.int64
    )
    frame = _frozen_frame(path_value, M_value, grad_value)
    statistics = _residual_statistics(
        path=path_value,
        grad=grad_value,
        grad_blocks=blocks,
        frozen_path=frame["path"],
        frozen_basis=frame["basis"],
        frozen_G=frame["G"],
        alpha=controls.alpha,
        block_frame_counts=initial_block_frame_counts,
    )
    covariance, covariance_available = _gradient_covariance_of_mean(
        blocks, initial_block_frame_counts
    )
    observation_paths = np.zeros(
        (MAX_PAID_OBSERVATIONS, N_PATH, DIMENSION), dtype=np.float64
    )
    observation_gradients = np.zeros(
        (MAX_PAID_OBSERVATIONS, N_INTERNAL, DIMENSION), dtype=np.float64
    )
    observation_covariances = np.zeros(
        (MAX_PAID_OBSERVATIONS, N_INTERNAL, DIMENSION, DIMENSION),
        dtype=np.float64,
    )
    observation_available = np.zeros(
        (MAX_PAID_OBSERVATIONS, N_INTERNAL), dtype=bool
    )
    observation_indices = np.zeros(MAX_PAID_OBSERVATIONS, dtype=np.int64)
    observation_paths[0] = path_value
    observation_gradients[0] = grad_value
    observation_covariances[0] = covariance
    observation_available[0] = covariance_available
    controller: Dict[str, Any] = {
        "state_index": 0,
        "nof": 0,
        "outer_index": 0,
        "inner_step": 0,
        "current_path": path_value.copy(),
        "current_M": M_value.copy(),
        "current_grad": grad_value.copy(),
        "current_grad_blocks": blocks.copy(),
        "current_M_blocks": metric_blocks.copy(),
        "current_block_frame_counts": initial_block_frame_counts,
        "frozen_path": frame["path"],
        "frozen_M": frame["M"],
        "frozen_grad": frame["grad"],
        "frozen_tangent": frame["tangent"],
        "frozen_G": frame["G"],
        "frozen_basis": frame["basis"],
        "outer_initial_effective_rms": np.asarray(
            statistics.effective_rms, dtype=np.float64
        ),
        "outer_anchor_lower_norm": np.asarray(math.nan, dtype=np.float64),
        "outer_anchor_upper_norm": np.asarray(math.nan, dtype=np.float64),
        "outer_anchor_bounds_ready": False,
        "observation_paths": observation_paths,
        "observation_gradients": observation_gradients,
        "observation_gradient_covariances": observation_covariances,
        "observation_covariance_available": observation_available,
        "observation_state_indices": observation_indices,
        "observation_count": 1,
        "lbfgs_s": np.zeros((JOINT_HISTORY_SIZE, N_INTERNAL), dtype=np.float64),
        "lbfgs_y": np.zeros((JOINT_HISTORY_SIZE, N_INTERNAL), dtype=np.float64),
        "lbfgs_count": 0,
        "lbfgs_latched": False,
        "warm_safeguard_inverse_scale": np.asarray(
            controls.alpha, dtype=np.float64
        ),
        "warm_safeguard_source_state_index": -1,
        "single_block_exact_after_initial": bool(single_block_exact_after_initial),
        "lbfgs_anchor_u": statistics.normal_coordinate.copy(),
        "lbfgs_anchor_raw_residual": statistics.raw_residual.copy(),
        "lbfgs_anchor_standard_error": statistics.standard_error.copy(),
        "controls": controls,
    }
    _validate_controller(controller, controls)
    return controller


def _single_block_is_exact(controller: Dict[str, Any], state_index: Optional[int] = None) -> bool:
    index = int(controller["state_index"] if state_index is None else state_index)
    return bool(controller.get("single_block_exact_after_initial", False)) and index >= 1


def _validate_controller(controller: Dict[str, Any], controls: IRIVMControls) -> None:
    controls.validate()
    path, M, grad = _validate_path_metric_gradient(
        controller["current_path"], controller["current_M"], controller["current_grad"]
    )
    blocks = _validate_gradient_blocks(controller["current_grad_blocks"])
    _validate_metric_blocks(controller["current_M_blocks"], blocks.shape[1])
    block_frame_counts = _validate_block_frame_counts(
        controller["current_block_frame_counts"],
        blocks.shape[1],
        allow_all_zero=int(controller["state_index"]) == 0,
    )
    if int(controller["state_index"]) > 0 and not (
        np.allclose(
            grad,
            _weighted_block_mean(blocks, block_frame_counts),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        and np.allclose(
            M,
            _weighted_block_mean(controller["current_M_blocks"], block_frame_counts),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    ):
        raise IRIVMError("checkpoint aggregate is not block-frame-count weighted")
    frozen = _frozen_frame(
        controller["frozen_path"], controller["frozen_M"], controller["frozen_grad"]
    )
    for key in ("frozen_tangent", "frozen_G"):
        observed = np.asarray(controller[key], dtype=np.float64)
        expected = frozen[key.split("frozen_")[-1]]
        if observed.shape != expected.shape or not np.allclose(
            observed, expected, rtol=1.0e-12, atol=1.0e-12
        ):
            raise IRIVMError(f"checkpoint {key} differs from its frozen reference")
    observed_basis = np.asarray(controller["frozen_basis"], dtype=np.float64)
    if observed_basis.shape != (N_INTERNAL, DIMENSION) or np.any(
        ~np.isfinite(observed_basis)
    ):
        raise IRIVMError("checkpoint frozen basis is invalid")
    basis_gram = np.einsum(
        "ni,nij,nj->n", observed_basis, controller["frozen_G"], observed_basis
    )
    basis_normal = np.einsum(
        "ni,nij,nj->n",
        observed_basis,
        controller["frozen_G"],
        controller["frozen_tangent"],
    )
    if not np.allclose(basis_gram, 1.0, rtol=1.0e-10, atol=1.0e-10):
        raise IRIVMError("checkpoint frozen basis is not G-orthonormal")
    if not np.allclose(basis_normal, 0.0, rtol=0.0, atol=1.0e-10):
        raise IRIVMError("checkpoint frozen basis is not G-normal")
    if not np.array_equal(path[[0, -1]], np.asarray(controller["frozen_path"])[[0, -1]]):
        raise IRIVMError("current and frozen paths do not share fixed endpoints")
    state_index = int(controller["state_index"])
    nof = int(controller["nof"])
    outer_index = int(controller["outer_index"])
    inner_step = int(controller["inner_step"])
    if state_index < 0 or nof != N_INTERNAL * state_index:
        raise IRIVMError("checkpoint state index and NoF ledger are inconsistent")
    if outer_index < 0 or not 0 <= inner_step < controls.m_max:
        raise IRIVMError("checkpoint outer/inner counters are invalid")
    lbfgs_count = int(controller["lbfgs_count"])
    if not 0 <= lbfgs_count <= JOINT_HISTORY_SIZE:
        raise IRIVMError("checkpoint joint L-BFGS count is invalid")
    safeguard = float(controller["warm_safeguard_inverse_scale"])
    safeguard_lower = controls.regression_inverse_scale_minimum_factor * controls.alpha
    safeguard_upper = controls.regression_inverse_scale_maximum_factor * controls.alpha
    if not np.isfinite(safeguard) or not safeguard_lower <= safeguard <= safeguard_upper:
        raise IRIVMError("warm scalar safeguard is outside its fixed alpha bounds")
    if bool(controller["lbfgs_latched"]) and int(
        controller["warm_safeguard_source_state_index"]
    ) < 0:
        raise IRIVMError("global takeover lacks a valid warm safeguard source")
    for key in ("lbfgs_s", "lbfgs_y"):
        value = np.asarray(controller[key], dtype=np.float64)
        if value.shape != (JOINT_HISTORY_SIZE, N_INTERNAL) or np.any(~np.isfinite(value)):
            raise IRIVMError(f"checkpoint {key} is invalid")
    for s_value, y_value in zip(
        controller["lbfgs_s"][:lbfgs_count], controller["lbfgs_y"][:lbfgs_count]
    ):
        if not float(np.dot(s_value, y_value)) > 0.0:
            raise IRIVMError("checkpoint contains a nonpositive L-BFGS pair")
    statistics = _residual_statistics(
        path=path,
        grad=grad,
        grad_blocks=blocks,
        frozen_path=controller["frozen_path"],
        frozen_basis=controller["frozen_basis"],
        frozen_G=controller["frozen_G"],
        alpha=controls.alpha,
        bound_kappa=controls.residual_bound_kappa,
        single_block_exact=_single_block_is_exact(controller),
        block_frame_counts=block_frame_counts,
    )
    anchors = {
        "lbfgs_anchor_u": statistics.normal_coordinate,
        "lbfgs_anchor_raw_residual": statistics.raw_residual,
        "lbfgs_anchor_standard_error": statistics.standard_error,
    }
    for key, expected in anchors.items():
        if not np.allclose(
            controller[key],
            expected,
            rtol=1.0e-13,
            atol=1.0e-13,
            equal_nan=True,
        ):
            raise IRIVMError(f"checkpoint {key} is not the current paid anchor")
    initial_rms = float(np.asarray(controller["outer_initial_effective_rms"]).item())
    if not np.isfinite(initial_rms) or initial_rms < 0.0:
        raise IRIVMError("checkpoint outer initial residual RMS is invalid")
    anchor_lower = float(np.asarray(controller["outer_anchor_lower_norm"]).item())
    anchor_upper = float(np.asarray(controller["outer_anchor_upper_norm"]).item())
    bounds_ready = bool(controller["outer_anchor_bounds_ready"])
    if bounds_ready:
        if not (
            np.isfinite(anchor_lower)
            and np.isfinite(anchor_upper)
            and anchor_lower > 0.0
            and anchor_upper >= anchor_lower
        ):
            raise IRIVMError("ready outer anchor has invalid residual bounds")
    elif not (np.isnan(anchor_lower) and np.isnan(anchor_upper)):
        raise IRIVMError("unready outer anchor must carry unavailable NaN bounds")
    if inner_step > 0 and not bounds_ready:
        raise IRIVMError("an outer with completed movements must have anchor bounds")
    count = int(controller["observation_count"])
    if not 1 <= count <= MAX_PAID_OBSERVATIONS:
        raise IRIVMError("checkpoint paid-observation count is invalid")
    expected_shapes = {
        "observation_paths": (MAX_PAID_OBSERVATIONS, N_PATH, DIMENSION),
        "observation_gradients": (MAX_PAID_OBSERVATIONS, N_INTERNAL, DIMENSION),
        "observation_gradient_covariances": (
            MAX_PAID_OBSERVATIONS,
            N_INTERNAL,
            DIMENSION,
            DIMENSION,
        ),
        "observation_covariance_available": (MAX_PAID_OBSERVATIONS, N_INTERNAL),
        "observation_state_indices": (MAX_PAID_OBSERVATIONS,),
    }
    for key, shape in expected_shapes.items():
        value = np.asarray(controller[key])
        if value.shape != shape:
            raise IRIVMError(f"checkpoint {key} has shape {value.shape}, expected {shape}")
        if value.dtype.kind not in "bui" and np.any(~np.isfinite(value)):
            raise IRIVMError(f"checkpoint {key} contains non-finite values")
    last = count - 1
    if not (
        np.array_equal(controller["observation_paths"][last], path)
        and np.array_equal(controller["observation_gradients"][last], grad)
        and int(controller["observation_state_indices"][last]) == state_index
    ):
        raise IRIVMError("paid-observation bank does not end at the current state")
    used_indices = np.asarray(controller["observation_state_indices"][:count])
    if count > 1 and np.any(np.diff(used_indices) <= 0):
        raise IRIVMError("distinct paid-observation state indices are not increasing")


def _save_checkpoint(run_directory: Path, controller: Dict[str, Any]) -> None:
    _validate_controller(controller, controller["controls"])
    keys = (
        "current_path",
        "current_M",
        "current_grad",
        "current_grad_blocks",
        "current_M_blocks",
        "current_block_frame_counts",
        "frozen_path",
        "frozen_M",
        "frozen_grad",
        "frozen_tangent",
        "frozen_G",
        "frozen_basis",
        "outer_initial_effective_rms",
        "outer_anchor_lower_norm",
        "outer_anchor_upper_norm",
        "observation_paths",
        "observation_gradients",
        "observation_gradient_covariances",
        "observation_covariance_available",
        "observation_state_indices",
        "lbfgs_s",
        "lbfgs_y",
        "lbfgs_anchor_u",
        "lbfgs_anchor_raw_residual",
        "lbfgs_anchor_standard_error",
        "warm_safeguard_inverse_scale",
    )
    arrays = {key: np.asarray(controller[key]) for key in keys}
    arrays.update(
        implementation_id=np.asarray(IMPLEMENTATION_ID),
        phase=np.asarray(CHECKPOINT_PHASE),
        checkpoint_sha256=np.asarray(_checkpoint_digest(controller)),
        state_index=np.asarray(controller["state_index"], dtype=np.int64),
        nof=np.asarray(controller["nof"], dtype=np.int64),
        outer_index=np.asarray(controller["outer_index"], dtype=np.int64),
        inner_step=np.asarray(controller["inner_step"], dtype=np.int64),
        observation_count=np.asarray(controller["observation_count"], dtype=np.int64),
        lbfgs_count=np.asarray(controller["lbfgs_count"], dtype=np.int64),
        warm_safeguard_source_state_index=np.asarray(
            controller["warm_safeguard_source_state_index"], dtype=np.int64
        ),
        lbfgs_latched=np.asarray(controller["lbfgs_latched"], dtype=bool),
        single_block_exact_after_initial=np.asarray(
            controller["single_block_exact_after_initial"], dtype=bool
        ),
        outer_anchor_bounds_ready=np.asarray(
            controller["outer_anchor_bounds_ready"], dtype=bool
        ),
    )
    _atomic_save_npz(run_directory / "controller_checkpoint.npz", **arrays)


def _controller_from_checkpoint(record: Dict[str, Array], controls: IRIVMControls) -> Dict[str, Any]:
    if str(record.get("implementation_id", np.asarray("")).item()) != IMPLEMENTATION_ID:
        raise IRIVMError("checkpoint implementation identity differs")
    if str(record.get("phase", np.asarray("")).item()) != CHECKPOINT_PHASE:
        raise IRIVMError("checkpoint phase is not V14 paid-state-ready")
    scalar_int = (
        "state_index",
        "nof",
        "outer_index",
        "inner_step",
        "observation_count",
        "lbfgs_count",
        "warm_safeguard_source_state_index",
    )
    controller: Dict[str, Any] = {key: int(record[key]) for key in scalar_int}
    controller["lbfgs_latched"] = bool(record["lbfgs_latched"])
    controller["single_block_exact_after_initial"] = bool(
        record["single_block_exact_after_initial"]
    )
    controller["outer_anchor_bounds_ready"] = bool(
        record["outer_anchor_bounds_ready"]
    )
    for key in (
        "current_path",
        "current_M",
        "current_grad",
        "current_grad_blocks",
        "current_M_blocks",
        "current_block_frame_counts",
        "frozen_path",
        "frozen_M",
        "frozen_grad",
        "frozen_tangent",
        "frozen_G",
        "frozen_basis",
        "outer_initial_effective_rms",
        "outer_anchor_lower_norm",
        "outer_anchor_upper_norm",
        "observation_paths",
        "observation_gradients",
        "observation_gradient_covariances",
        "observation_covariance_available",
        "observation_state_indices",
        "lbfgs_s",
        "lbfgs_y",
        "lbfgs_anchor_u",
        "lbfgs_anchor_raw_residual",
        "lbfgs_anchor_standard_error",
        "warm_safeguard_inverse_scale",
    ):
        if key not in record:
            raise IRIVMError(f"checkpoint is missing {key}")
        controller[key] = record[key]
    controller["controls"] = controls
    _validate_controller(controller, controls)
    if str(record["checkpoint_sha256"].item()) != _checkpoint_digest(controller):
        raise IRIVMError("checkpoint digest differs from its contents")
    return controller


def _prepare_update(
    controller: Dict[str, Any], controls: IRIVMControls
) -> Tuple[Optional[PreparedUpdate], Dict[str, object]]:
    statistics = _residual_statistics(
        path=controller["current_path"],
        grad=controller["current_grad"],
        grad_blocks=controller["current_grad_blocks"],
        frozen_path=controller["frozen_path"],
        frozen_basis=controller["frozen_basis"],
        frozen_G=controller["frozen_G"],
        alpha=controls.alpha,
        bound_kappa=controls.residual_bound_kappa,
        single_block_exact=_single_block_is_exact(controller),
        block_frame_counts=controller["current_block_frame_counts"],
    )
    initial_no_se_special_case = bool(
        int(controller["state_index"]) == 0
        and not statistics.standard_error_available.any()
    )
    if int(controller["state_index"]) >= 1 and not np.all(
        statistics.standard_error_available
    ):
        raise IRIVMError(
            "state >= 1 requires real block means for every noise gate; "
            "only state 0 may use the one-time no-SE cold special case"
        )
    if statistics.raw_rms <= controls.kkt_floor:
        return None, {
            "reason": "raw_global_residual_at_numerical_zero",
            "raw_residual_rms": statistics.raw_rms,
            "effective_residual_rms": statistics.effective_rms,
        }
    lengths, rho, radii = _trust_radii(controller["frozen_path"], controls)
    h0 = _fit_h0_regression(controller, statistics, controls)
    resolved_se = statistics.standard_error_available & (
        statistics.standard_error > 0.0
    )
    if np.any(resolved_se):
        noise_norm = float(np.linalg.norm(statistics.standard_error[resolved_se]))
        global_snr = float(
            np.linalg.norm(statistics.raw_residual[resolved_se]) / noise_norm
        )
    else:
        # Deterministic/surrogate one-block evaluations take the exact branch.
        global_snr = math.inf
    bounds_ready = bool(controller["outer_anchor_bounds_ready"])
    if not initial_no_se_special_case and not statistics.bounds_available:
        raise IRIVMError("noninitial V14 decision lacks residual uncertainty bounds")
    anchor_low_signal = bool(
        not initial_no_se_special_case
        and not bounds_ready
        and statistics.lower_norm <= controls.kkt_floor
    )
    residual_ambiguous = False
    if (
        not initial_no_se_special_case
        and bounds_ready
        and int(controller["inner_step"]) > 0
    ):
        anchor_lower = float(controller["outer_anchor_lower_norm"])
        anchor_upper = float(controller["outer_anchor_upper_norm"])
        clear_stop = statistics.upper_norm <= controls.r * anchor_lower
        clear_continue = statistics.lower_norm > controls.r * anchor_upper
        if clear_stop:
            raise IRIVMError("clear residual stop was not committed at previous paid state")
        residual_ambiguous = not clear_continue

    pre_motion_hold_reason = "none"
    if anchor_low_signal:
        pre_motion_hold_reason = "anchor_lower_bound_zero_exact_center_resample"
    elif residual_ambiguous:
        pre_motion_hold_reason = "relative_residual_bounds_ambiguous_exact_center_resample"

    # A HOLD is an exact byte-for-byte center resample.  In the low-SNR branch
    # no article map is called at all.  This prevents noisy directions from
    # being converted into irreversible path motion merely to collect data.
    if pre_motion_hold_reason != "none":
        h0_candidate = _invalid_candidate(
            "H0", "suppressed_by_noise_protected_hold"
        )
        lbfgs_candidate = _invalid_candidate(
            "LBFGS", "suppressed_by_noise_protected_hold"
        )
        selected = None
        hold_selected = True
        hold_reason = pre_motion_hold_reason
        difference_se = math.inf
        latch_before = bool(controller["lbfgs_latched"])
        latch_after = latch_before
        strict_win = False
        forced_lbfgs = False
        h0_fallback = False
        selected_score_se = math.nan
        selected_lcb = math.nan
    else:
        latch_before = bool(controller["lbfgs_latched"])
        if latch_before:
            h0_candidate = _invalid_candidate(
                "H0", "cold_forbidden_after_global_one_way_takeover"
            )
        else:
            h0_direction = -h0.inverse_scale * statistics.effective_residual
            h0_candidate = _build_candidate(
                label="H0",
                direction=h0_direction,
                controller=controller,
                statistics=statistics,
                radii=radii,
                controls=controls,
            )
        pair_count = int(controller["lbfgs_count"])
        # Two independently noise-qualified, nondegenerate composite secants
        # are the minimum evidence for a warm candidate or takeover.
        if pair_count >= 2:
            try:
                lbfgs_direction = joint_lbfgs_direction(
                    statistics.effective_residual,
                    controller["lbfgs_s"][:pair_count],
                    controller["lbfgs_y"][:pair_count],
                    (
                        np.full(
                            N_INTERNAL,
                            float(controller["warm_safeguard_inverse_scale"]),
                        )
                        if latch_before
                        else h0.inverse_scale
                    ),
                )
                lbfgs_candidate = _build_candidate(
                    label="LBFGS",
                    direction=lbfgs_direction,
                    controller=controller,
                    statistics=statistics,
                    radii=radii,
                    controls=controls,
                )
            except (ArithmeticError, FloatingPointError, ValueError, IRIVMError) as error:
                lbfgs_candidate = _invalid_candidate(
                    "LBFGS", f"{type(error).__name__}:{error}"
                )
        elif latch_before:
            safeguard_direction = -float(
                controller["warm_safeguard_inverse_scale"]
            ) * statistics.effective_residual
            lbfgs_candidate = _build_candidate(
                label="WARM_SCALAR_SAFEGUARD",
                direction=safeguard_direction,
                controller=controller,
                statistics=statistics,
                radii=radii,
                controls=controls,
            )
        else:
            lbfgs_candidate = _invalid_candidate(
                "LBFGS", "fewer_than_two_noise_qualified_post_R_secants"
            )

        latch_after = latch_before
        strict_win = False
        forced_lbfgs = False
        h0_fallback = False
        hold_selected = False
        hold_reason = "none"
        selected: Optional[Candidate] = None
        difference_se = math.inf
        if not latch_before and h0_candidate.valid and lbfgs_candidate.valid:
            # Both scores use the same source-band residual estimate.  Their
            # difference therefore has the shared-noise coefficient
            # delta_LBFGS-delta_H0; treating the scores as independent would
            # discard that covariance.
            difference_se = float(
                np.linalg.norm(
                    statistics.standard_error
                    * (
                        lbfgs_candidate.actual_normal_step
                        - h0_candidate.actual_normal_step
                    )
                )
            )
            score_difference = (
                lbfgs_candidate.admissibility_score
                - h0_candidate.admissibility_score
            )
            ambiguous = bool(
                np.any(statistics.standard_error > 0.0)
                and difference_se > 0.0
                and abs(score_difference)
                <= controls.secant_noise_z * difference_se
            )
        else:
            score_difference = math.nan
            ambiguous = False

        if latch_before:
            # Global takeover is irreversible across outer refreshes.  There
            # is no cold candidate and therefore no cold-vs-warm ambiguity
            # contest in a latched state.
            if lbfgs_candidate.valid:
                selected = lbfgs_candidate
                forced_lbfgs = True
        elif ambiguous:
            # H0/L-BFGS geometry candidates have already passed through R, but
            # the selected HOLD itself reuses current_path exactly and never
            # calls the article map on that exact center.
            hold_selected = True
            hold_reason = "candidate_admissibility_score_ambiguous_with_block_noise"
            latch_after = latch_before
        else:
            if h0_candidate.valid and lbfgs_candidate.valid:
                strict_win = bool(
                    lbfgs_candidate.admissibility_score
                    > h0_candidate.admissibility_score
                )
                selected = lbfgs_candidate if strict_win else h0_candidate
                if strict_win:
                    latch_after = True
            elif h0_candidate.valid:
                selected = h0_candidate
            elif lbfgs_candidate.valid:
                selected = lbfgs_candidate
        if selected is None and not hold_selected:
            reducible_uncertainty = bool(
                not initial_no_se_special_case
                and np.any(statistics.standard_error > 0.0)
            )
            if not reducible_uncertainty:
                return None, {
                    "reason": "no_valid_motion_candidate_terminal",
                    "phase": "TERMINAL",
                    "initial_no_se_special_case": initial_no_se_special_case,
                    "globally_latched": latch_before,
                    "h0_error": h0_candidate.error,
                    "warm_error": lbfgs_candidate.error,
                }
            hold_selected = True
            hold_reason = "no_valid_motion_candidate_with_reducible_noise"
            latch_after = latch_before

        selected_score_se = math.nan
        selected_lcb = math.nan
        if selected is not None and not hold_selected:
            if initial_no_se_special_case:
                # Fair state-0 exception: uncertainty is unavailable, so no
                # noise-resolved claim is made, but one cold proposal is allowed.
                selected_score_se = math.nan
                selected_lcb = math.nan
            else:
                selected_score_se = float(
                    np.linalg.norm(
                        statistics.standard_error * selected.actual_normal_step
                    )
                )
                selected_lcb = float(
                    selected.admissibility_score
                    - controls.residual_bound_kappa * selected_score_se
                )
                if selected_lcb <= 0.0 and selected_score_se > 0.0:
                    hold_selected = True
                    hold_reason = "selected_admissibility_score_LCB_nonpositive"
                    latch_after = latch_before
                    strict_win = False
                    forced_lbfgs = False
                    h0_fallback = False

    if hold_selected:
        selected_label = "HOLD"
        selected_path = np.asarray(controller["current_path"], dtype=np.float64).copy()
        if not np.array_equal(selected_path, controller["current_path"]):
            raise IRIVMError("HOLD failed exact path-byte reuse")
    else:
        if selected is None or selected.reparameterization is None:
            raise IRIVMError("internal selector did not produce motion or HOLD")
        selected_label = selected.label
        selected_path = selected.reparameterization.path.copy()

    source_digest = _checkpoint_digest(controller)
    proposal_payload = {
        "schema": "Ala2DPreparedUpdate/v14-development",
        "implementation_id": IMPLEMENTATION_ID,
        "source_checkpoint_sha256": source_digest,
        "source_effective_residual": _array_sha256(statistics.effective_residual),
        "h0_inverse_scale": _array_sha256(h0.inverse_scale),
        "h0_candidate": _candidate_digest(h0_candidate),
        "lbfgs_candidate": _candidate_digest(lbfgs_candidate),
        "selected_label": selected_label,
        "selected_path": _array_sha256(selected_path),
        "latch_before": latch_before,
        "latch_after": latch_after,
        "strict_lbfgs_win": strict_win,
        "hold_selected": hold_selected,
        "hold_reason": hold_reason,
        "global_signal_to_noise_binary64": float(global_snr).hex(),
        "admissibility_difference_standard_error_binary64": float(difference_se).hex(),
        "initial_no_se_special_case": initial_no_se_special_case,
        "selected_admissibility_standard_error_binary64": float(
            selected_score_se
        ).hex(),
        "selected_admissibility_lower_bound_binary64": float(selected_lcb).hex(),
    }
    proposal_sha = hashlib.sha256(_canonical_json(proposal_payload)).hexdigest()
    return PreparedUpdate(
        source_checkpoint_sha256=source_digest,
        proposal_sha256=proposal_sha,
        source_statistics=statistics,
        source_segment_lengths=lengths,
        global_trust_radius=rho,
        endpoint_trust_radii=radii,
        h0_regression=h0,
        h0_candidate=h0_candidate,
        lbfgs_candidate=lbfgs_candidate,
        selected_label=selected_label,
        selected_path=selected_path,
        latch_before=latch_before,
        latch_after=latch_after,
        strict_lbfgs_win=strict_win,
        forced_lbfgs_after_takeover=forced_lbfgs,
        h0_invalid_fallback=h0_fallback,
        hold_selected=hold_selected,
        hold_reason=hold_reason,
        global_signal_to_noise=global_snr,
        admissibility_difference_standard_error=difference_se,
        initial_no_se_special_case=initial_no_se_special_case,
        selected_admissibility_standard_error=selected_score_se,
        selected_admissibility_lower_bound=selected_lcb,
    ), {}


def _secant_gate(
    s_value: Array,
    y_value: Array,
    old_se: Array,
    new_se: Array,
    controls: IRIVMControls,
) -> Dict[str, object]:
    s = np.asarray(s_value, dtype=np.float64)
    y = np.asarray(y_value, dtype=np.float64)
    sigma_y = np.hypot(old_se, new_se)
    s_norm = float(np.linalg.norm(s))
    y_norm = float(np.linalg.norm(y))
    curvature = float(np.dot(s, y))
    cosine_threshold = (
        controls.secant_curvature_cosine_minimum * s_norm * y_norm
    )
    curvature_se = float(np.linalg.norm(s * sigma_y))
    noise_threshold = controls.secant_noise_z * curvature_se
    finite = bool(
        np.all(np.isfinite(s))
        and np.all(np.isfinite(y))
        and np.all(np.isfinite(sigma_y))
    )
    accepted = bool(
        finite
        and s_norm > controls.secant_norm_floor
        and y_norm > controls.secant_norm_floor
        and curvature > cosine_threshold
        and curvature > noise_threshold
    )
    if not finite:
        reason = "nonfinite_secant"
    elif s_norm <= controls.secant_norm_floor:
        reason = "degenerate_composite_step"
    elif y_norm <= controls.secant_norm_floor:
        reason = "degenerate_residual_change"
    elif not curvature > cosine_threshold:
        reason = "positive_curvature_cosine_gate_failed"
    elif not curvature > noise_threshold:
        reason = "curvature_not_resolved_above_block_noise"
    else:
        reason = "accepted_noise_aware_composite_secant"
    return {
        "accepted": accepted,
        "reason": reason,
        "s": s.copy(),
        "y": y.copy(),
        "sigma_y": sigma_y,
        "s_norm": s_norm,
        "y_norm": y_norm,
        "curvature": curvature,
        "cosine_threshold": cosine_threshold,
        "curvature_standard_error": curvature_se,
        "noise_threshold": noise_threshold,
    }


def _append_joint_pair(
    old_s: Array, old_y: Array, count: int, s_value: Array, y_value: Array
) -> Tuple[Array, Array, int]:
    s_pairs = np.asarray(old_s, dtype=np.float64).copy()
    y_pairs = np.asarray(old_y, dtype=np.float64).copy()
    count_value = int(count)
    if count_value == JOINT_HISTORY_SIZE:
        s_pairs[:-1] = s_pairs[1:]
        y_pairs[:-1] = y_pairs[1:]
        slot = count_value - 1
    else:
        slot = count_value
        count_value += 1
    s_pairs[slot] = s_value
    y_pairs[slot] = y_value
    return s_pairs, y_pairs, count_value


def _reproject_observation_history(
    observations: Dict[str, Array],
    frame: Dict[str, Array],
    controls: IRIVMControls,
    prior_safeguard: float,
    prior_safeguard_source_state: int,
) -> Dict[str, object]:
    """Rebuild current-frame secants from the local distinct-path suffix.

    Normal-coordinate retention is a structural, parameter-free test here:
    both endpoints must lie inside the existing per-image trust radii and the
    reprojected ``||s||`` must exceed ``secant_norm_floor``.  Accepted pairs
    must additionally pass the existing curvature/cosine/noise gate.
    """

    count = int(observations["observation_count"])
    paths = np.asarray(observations["observation_paths"][:count])
    gradients = np.asarray(observations["observation_gradients"][:count])
    covariances = np.asarray(
        observations["observation_gradient_covariances"][:count]
    )
    available = np.asarray(
        observations["observation_covariance_available"][:count]
    )
    state_indices = np.asarray(observations["observation_state_indices"][:count])
    _, _, radii = _trust_radii(frame["path"], controls)
    physical = np.linalg.norm(
        periodic_difference(paths[:, 1:-1], frame["path"][None, 1:-1]), axis=2
    )
    local = np.all(physical <= radii[None, :], axis=1)
    suffix_start = count - 1
    while suffix_start > 0 and local[suffix_start - 1]:
        suffix_start -= 1
    s_pairs = np.zeros((JOINT_HISTORY_SIZE, N_INTERNAL), dtype=np.float64)
    y_pairs = np.zeros_like(s_pairs)
    pair_count = 0
    safeguard = float(prior_safeguard)
    safeguard_source = int(prior_safeguard_source_state)
    accepted_source_pairs: List[Tuple[int, int]] = []
    u_values: List[Array] = []
    p_values: List[Array] = []
    se_values: List[Array] = []
    for slot in range(suffix_start, count):
        u = _normal_coordinate(
            paths[slot], frame["path"], frame["basis"], frame["G"]
        )
        q = np.einsum("ni,ni->n", frame["basis"], gradients[slot])
        u_values.append(u)
        p_values.append(q + u / controls.alpha)
        if np.all(available[slot]):
            variance = np.einsum(
                "ni,nij,nj->n",
                frame["basis"],
                covariances[slot],
                frame["basis"],
            )
            se_values.append(np.sqrt(np.maximum(variance, 0.0)))
        else:
            se_values.append(np.full(N_INTERNAL, np.nan))
    for relative in range(1, len(u_values)):
        left = suffix_start + relative - 1
        right = suffix_start + relative
        if not (local[left] and local[right]):
            continue
        s_value = u_values[relative] - u_values[relative - 1]
        y_value = p_values[relative] - p_values[relative - 1]
        gate = _secant_gate(
            s_value,
            y_value,
            se_values[relative - 1],
            se_values[relative],
            controls,
        )
        if not bool(gate["accepted"]):
            continue
        s_pairs, y_pairs, pair_count = _append_joint_pair(
            s_pairs, y_pairs, pair_count, s_value, y_value
        )
        curvature = float(np.dot(s_value, y_value))
        y_norm_squared = float(np.dot(y_value, y_value))
        raw_scale = curvature / y_norm_squared
        safeguard = float(
            np.clip(
                raw_scale,
                controls.regression_inverse_scale_minimum_factor * controls.alpha,
                controls.regression_inverse_scale_maximum_factor * controls.alpha,
            )
        )
        safeguard_source = int(state_indices[right])
        accepted_source_pairs.append((int(state_indices[left]), int(state_indices[right])))
    return {
        "lbfgs_s": s_pairs,
        "lbfgs_y": y_pairs,
        "lbfgs_count": pair_count,
        "warm_safeguard_inverse_scale": safeguard,
        "warm_safeguard_source_state_index": safeguard_source,
        "reprojected_suffix_start_state_index": int(state_indices[suffix_start]),
        "reprojected_pair_source_state_indices": np.asarray(
            accepted_source_pairs, dtype=np.int64
        ).reshape((-1, 2)),
    }


def _advance_controller(
    controller: Dict[str, Any],
    prepared: PreparedUpdate,
    *,
    M: Array,
    grad: Array,
    grad_blocks: Array,
    M_blocks: Array,
    frame_counts: Array,
    discarded_frame_counts: Array,
    controls: IRIVMControls,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if prepared.source_checkpoint_sha256 != _checkpoint_digest(controller):
        raise IRIVMError("prepared update belongs to a different checkpoint")
    path, raw_M, raw_grad = _validate_path_metric_gradient(
        prepared.selected_path, M, grad
    )
    raw_grad_blocks = _validate_gradient_blocks(grad_blocks)
    raw_M_blocks = _validate_metric_blocks(M_blocks, raw_grad_blocks.shape[1])
    retained_frame_counts = np.asarray(frame_counts) - np.asarray(
        discarded_frame_counts
    )
    raw_block_frame_counts = _block_frame_counts(
        retained_frame_counts, raw_grad_blocks.shape[1]
    )
    weighted_raw_grad = _weighted_block_mean(
        raw_grad_blocks, raw_block_frame_counts
    )
    weighted_raw_M = _weighted_block_mean(raw_M_blocks, raw_block_frame_counts)
    weighted_raw_M = 0.5 * (weighted_raw_M + np.swapaxes(weighted_raw_M, 1, 2))
    if not (
        np.allclose(raw_grad, weighted_raw_grad, rtol=1.0e-12, atol=1.0e-12)
        and np.allclose(raw_M, weighted_raw_M, rtol=1.0e-12, atol=1.0e-12)
    ):
        raise IRIVMError("oracle aggregate is not the retained-frame weighted block mean")
    hold_resample = prepared.selected_label == "HOLD"
    if hold_resample:
        if not np.array_equal(path, controller["current_path"]):
            raise IRIVMError("HOLD/resample did not reuse exact current path bytes")
        blocks = np.concatenate(
            (controller["current_grad_blocks"], raw_grad_blocks), axis=1
        )
        metric_blocks = np.concatenate(
            (controller["current_M_blocks"], raw_M_blocks), axis=1
        )
        block_frame_counts = np.concatenate(
            (controller["current_block_frame_counts"], raw_block_frame_counts), axis=1
        )
        grad_value = _weighted_block_mean(blocks, block_frame_counts)
        M_value = _weighted_block_mean(metric_blocks, block_frame_counts)
        M_value = 0.5 * (M_value + np.swapaxes(M_value, 1, 2))
        path, M_value, grad_value = _validate_path_metric_gradient(
            path, M_value, grad_value
        )
    else:
        blocks = raw_grad_blocks.copy()
        metric_blocks = raw_M_blocks.copy()
        block_frame_counts = raw_block_frame_counts.copy()
        M_value = weighted_raw_M.copy()
        grad_value = weighted_raw_grad.copy()
    source_bounds_ready = bool(controller["outer_anchor_bounds_ready"])
    provisional_anchor_frame: Optional[Dict[str, Array]] = None
    if hold_resample and not source_bounds_ready:
        # ANCHORING is provisional: every exact-center pooled estimate updates
        # M/g and therefore rebuilds B/G before deciding whether READY exists.
        provisional_anchor_frame = _frozen_frame(
            path, M_value, grad_value, previous_basis=controller["frozen_basis"]
        )
        statistics_frame = provisional_anchor_frame
    else:
        statistics_frame = {
            "path": controller["frozen_path"],
            "basis": controller["frozen_basis"],
            "G": controller["frozen_G"],
        }
    new_statistics_old_frame = _residual_statistics(
        path=path,
        grad=grad_value,
        grad_blocks=blocks,
        frozen_path=statistics_frame["path"],
        frozen_basis=statistics_frame["basis"],
        frozen_G=statistics_frame["G"],
        alpha=controls.alpha,
        bound_kappa=controls.residual_bound_kappa,
        single_block_exact=_single_block_is_exact(controller, int(controller["state_index"]) + 1),
        block_frame_counts=block_frame_counts,
    )
    if not np.allclose(
        prepared.source_statistics.normal_coordinate,
        controller["lbfgs_anchor_u"],
        rtol=1.0e-13,
        atol=1.0e-13,
    ):
        raise IRIVMError("prepared source is not the same-outer L-BFGS anchor")
    secant = _secant_gate(
        new_statistics_old_frame.normal_coordinate - controller["lbfgs_anchor_u"],
        new_statistics_old_frame.raw_residual
        - controller["lbfgs_anchor_raw_residual"],
        controller["lbfgs_anchor_standard_error"],
        new_statistics_old_frame.standard_error,
        controls,
    )
    next_state = int(controller["state_index"]) + 1
    observations = _append_observation(
        controller, path, grad_value, blocks, block_frame_counts, next_state
    )
    movement = not hold_resample
    completed_inner = int(controller["inner_step"]) + int(movement)
    initial_rms = float(controller["outer_initial_effective_rms"])
    anchor_lower = float(controller["outer_anchor_lower_norm"])
    anchor_upper = float(controller["outer_anchor_upper_norm"])
    if source_bounds_ready:
        clear_r_stop = bool(
            new_statistics_old_frame.upper_norm <= controls.r * anchor_lower
        )
        clear_continue = bool(
            new_statistics_old_frame.lower_norm > controls.r * anchor_upper
        )
        residual_ambiguous = not clear_r_stop and not clear_continue
        residual_ratio = float(
            new_statistics_old_frame.upper_norm
            / max(anchor_lower, controls.kkt_floor)
        )
    else:
        clear_r_stop = False
        clear_continue = False
        residual_ambiguous = False
        residual_ratio = math.nan

    initial_special_completed = bool(prepared.initial_no_se_special_case)
    cap_stop = bool(movement and completed_inner >= controls.m_max)
    if initial_special_completed:
        stopped = True
        r_stop = False
        stop_reason = "initial_no_se_special_case_completed_rebuild_outer"
    elif source_bounds_ready and clear_r_stop:
        stopped = True
        r_stop = True
        stop_reason = "global_r_interval_stop"
    elif cap_stop:
        stopped = True
        r_stop = False
        stop_reason = "m_max_movement_stop"
    else:
        stopped = False
        r_stop = False
        if hold_resample and not source_bounds_ready:
            stop_reason = (
                "anchor_resolved_ready"
                if new_statistics_old_frame.lower_norm > controls.kkt_floor
                else "anchor_still_unresolved_hold_again"
            )
        elif hold_resample and residual_ambiguous:
            stop_reason = "relative_residual_still_ambiguous_hold_again"
        elif hold_resample:
            stop_reason = "hold_resample_resolved_continue"
        elif residual_ambiguous:
            stop_reason = "movement_result_ambiguous_resample_next"
        else:
            stop_reason = "continue_same_frozen_outer"

    if stopped:
        next_frame = _frozen_frame(
            path,
            M_value,
            grad_value,
            previous_basis=controller["frozen_basis"],
        )
        next_statistics = _residual_statistics(
            path=path,
            grad=grad_value,
            grad_blocks=blocks,
            frozen_path=next_frame["path"],
            frozen_basis=next_frame["basis"],
            frozen_G=next_frame["G"],
            alpha=controls.alpha,
            bound_kappa=controls.residual_bound_kappa,
            single_block_exact=_single_block_is_exact(controller, int(controller["state_index"]) + 1),
            block_frame_counts=block_frame_counts,
        )
        rebuilt = _reproject_observation_history(
            observations,
            next_frame,
            controls,
            float(controller["warm_safeguard_inverse_scale"]),
            int(controller["warm_safeguard_source_state_index"]),
        )
        lbfgs_s = rebuilt["lbfgs_s"]
        lbfgs_y = rebuilt["lbfgs_y"]
        lbfgs_count = int(rebuilt["lbfgs_count"])
        lbfgs_latched = bool(controller["lbfgs_latched"] or prepared.latch_after)
        safeguard_scale = float(rebuilt["warm_safeguard_inverse_scale"])
        safeguard_source = int(rebuilt["warm_safeguard_source_state_index"])
        next_outer = int(controller["outer_index"]) + 1
        next_inner = 0
        next_initial_rms = next_statistics.effective_rms
        if (
            next_statistics.bounds_available
            and next_statistics.lower_norm > controls.kkt_floor
        ):
            next_anchor_lower = next_statistics.lower_norm
            next_anchor_upper = next_statistics.upper_norm
            next_bounds_ready = True
        else:
            next_anchor_lower = math.nan
            next_anchor_upper = math.nan
            next_bounds_ready = False
        pair_stored = bool(movement and secant["accepted"])
    else:
        next_frame = provisional_anchor_frame or {
            "path": np.asarray(controller["frozen_path"], dtype=np.float64).copy(),
            "M": np.asarray(controller["frozen_M"], dtype=np.float64).copy(),
            "grad": np.asarray(controller["frozen_grad"], dtype=np.float64).copy(),
            "tangent": np.asarray(controller["frozen_tangent"], dtype=np.float64).copy(),
            "G": np.asarray(controller["frozen_G"], dtype=np.float64).copy(),
            "basis": np.asarray(controller["frozen_basis"], dtype=np.float64).copy(),
            "gauge_flipped": np.zeros(N_INTERNAL, dtype=bool),
        }
        if provisional_anchor_frame is not None:
            rebuilt = _reproject_observation_history(
                observations,
                next_frame,
                controls,
                float(controller["warm_safeguard_inverse_scale"]),
                int(controller["warm_safeguard_source_state_index"]),
            )
            lbfgs_s = rebuilt["lbfgs_s"]
            lbfgs_y = rebuilt["lbfgs_y"]
            lbfgs_count = int(rebuilt["lbfgs_count"])
            safeguard_scale = float(rebuilt["warm_safeguard_inverse_scale"])
            safeguard_source = int(rebuilt["warm_safeguard_source_state_index"])
            pair_stored = False
        elif movement and bool(secant["accepted"]):
            lbfgs_s, lbfgs_y, lbfgs_count = _append_joint_pair(
                controller["lbfgs_s"],
                controller["lbfgs_y"],
                int(controller["lbfgs_count"]),
                secant["s"],
                secant["y"],
            )
            raw_scale = float(np.dot(secant["s"], secant["y"])) / float(
                np.dot(secant["y"], secant["y"])
            )
            safeguard_scale = float(
                np.clip(
                    raw_scale,
                    controls.regression_inverse_scale_minimum_factor * controls.alpha,
                    controls.regression_inverse_scale_maximum_factor * controls.alpha,
                )
            )
            safeguard_source = next_state
            pair_stored = True
        else:
            lbfgs_s = np.asarray(controller["lbfgs_s"], dtype=np.float64).copy()
            lbfgs_y = np.asarray(controller["lbfgs_y"], dtype=np.float64).copy()
            lbfgs_count = int(controller["lbfgs_count"])
            safeguard_scale = float(controller["warm_safeguard_inverse_scale"])
            safeguard_source = int(controller["warm_safeguard_source_state_index"])
            pair_stored = False
        lbfgs_latched = bool(controller["lbfgs_latched"] or prepared.latch_after)
        next_outer = int(controller["outer_index"])
        next_inner = completed_inner
        if source_bounds_ready:
            next_initial_rms = initial_rms
            next_anchor_lower = anchor_lower
            next_anchor_upper = anchor_upper
            next_bounds_ready = True
        elif new_statistics_old_frame.lower_norm > controls.kkt_floor:
            next_initial_rms = new_statistics_old_frame.effective_rms
            next_anchor_lower = new_statistics_old_frame.lower_norm
            next_anchor_upper = new_statistics_old_frame.upper_norm
            next_bounds_ready = True
        else:
            next_initial_rms = new_statistics_old_frame.effective_rms
            next_anchor_lower = math.nan
            next_anchor_upper = math.nan
            next_bounds_ready = False
        next_statistics = new_statistics_old_frame

    new_controller: Dict[str, Any] = {
        "state_index": next_state,
        "nof": int(controller["nof"]) + N_INTERNAL,
        "outer_index": next_outer,
        "inner_step": next_inner,
        "current_path": path.copy(),
        "current_M": M_value.copy(),
        "current_grad": grad_value.copy(),
        "current_grad_blocks": blocks.copy(),
        "current_M_blocks": metric_blocks.copy(),
        "current_block_frame_counts": block_frame_counts.copy(),
        "frozen_path": next_frame["path"],
        "frozen_M": next_frame["M"],
        "frozen_grad": next_frame["grad"],
        "frozen_tangent": next_frame["tangent"],
        "frozen_G": next_frame["G"],
        "frozen_basis": next_frame["basis"],
        "outer_initial_effective_rms": np.asarray(next_initial_rms, dtype=np.float64),
        "outer_anchor_lower_norm": np.asarray(next_anchor_lower, dtype=np.float64),
        "outer_anchor_upper_norm": np.asarray(next_anchor_upper, dtype=np.float64),
        "outer_anchor_bounds_ready": bool(next_bounds_ready),
        "observation_paths": observations["observation_paths"],
        "observation_gradients": observations["observation_gradients"],
        "observation_gradient_covariances": observations[
            "observation_gradient_covariances"
        ],
        "observation_covariance_available": observations[
            "observation_covariance_available"
        ],
        "observation_state_indices": observations["observation_state_indices"],
        "observation_count": int(observations["observation_count"]),
        "lbfgs_s": lbfgs_s,
        "lbfgs_y": lbfgs_y,
        "lbfgs_count": lbfgs_count,
        "lbfgs_latched": lbfgs_latched,
        "warm_safeguard_inverse_scale": np.asarray(
            safeguard_scale, dtype=np.float64
        ),
        "warm_safeguard_source_state_index": safeguard_source,
        "single_block_exact_after_initial": bool(
            controller["single_block_exact_after_initial"]
        ),
        "lbfgs_anchor_u": next_statistics.normal_coordinate.copy(),
        "lbfgs_anchor_raw_residual": next_statistics.raw_residual.copy(),
        "lbfgs_anchor_standard_error": next_statistics.standard_error.copy(),
        "controls": controls,
    }
    _validate_controller(new_controller, controls)
    audit: Dict[str, Any] = {
        "source_outer_index": int(controller["outer_index"]),
        "completed_inner_step": completed_inner,
        "new_statistics_in_source_frozen_frame": new_statistics_old_frame,
        "secant": secant,
        "secant_stored_after_transition": pair_stored,
        "residual_ratio": float(residual_ratio),
        "r_stop": r_stop,
        "m_max_stop": cap_stop,
        "stopped_inner": stopped,
        "inner_stop_reason": stop_reason,
        "frame_rebuilt": bool(stopped or provisional_anchor_frame is not None),
        "next_statistics": next_statistics,
        "hold_resample": hold_resample,
        "oracle_raw_M": raw_M,
        "oracle_raw_grad": raw_grad,
        "oracle_raw_grad_blocks": raw_grad_blocks,
        "oracle_raw_M_blocks": raw_M_blocks,
        "oracle_raw_block_frame_counts": raw_block_frame_counts,
        "movement_committed": movement,
        "source_anchor_bounds_ready": source_bounds_ready,
        "source_anchor_lower_norm": anchor_lower,
        "source_anchor_upper_norm": anchor_upper,
        "clear_continue": clear_continue,
        "residual_decision_ambiguous": residual_ambiguous,
        "initial_no_se_special_case_completed": initial_special_completed,
        "next_frozen_basis_gauge_flipped": next_frame["gauge_flipped"],
        "warm_safeguard_inverse_scale_next": safeguard_scale,
        "warm_safeguard_source_state_index_next": safeguard_source,
        "global_takeover_persisted_across_outer": bool(
            controller["lbfgs_latched"] and lbfgs_latched
        ),
        "current_frame_lbfgs_memory_available_next": bool(lbfgs_count >= 2),
    }
    return new_controller, audit


def _validate_complete_band(band: OracleBand, candidate_path: Array) -> None:
    if int(band.nof) != N_INTERNAL:
        raise IRIVMError(
            f"one complete Ala2D band must cost exactly {N_INTERNAL} NoF, got {band.nof}"
        )
    expected_shapes = {
        "centers": (N_INTERNAL, DIMENSION),
        "mean_coordinates": (N_INTERNAL, DIMENSION),
        "grad_F": (N_INTERNAL, DIMENSION),
        "M": (N_INTERNAL, DIMENSION, DIMENSION),
        "seeds": (N_INTERNAL,),
        "thermostat_seeds": (N_INTERNAL,),
        "frame_counts": (N_INTERNAL,),
        "discarded_frame_counts": (N_INTERNAL,),
        "wall_seconds": (N_INTERNAL,),
    }
    for name, shape in expected_shapes.items():
        value = np.asarray(getattr(band, name))
        if value.shape != shape:
            raise IRIVMError(f"oracle {name} has shape {value.shape}, expected {shape}")
        if value.dtype.kind not in "bui" and np.any(~np.isfinite(value)):
            raise IRIVMError(f"oracle {name} contains non-finite values")
    if not np.array_equal(np.asarray(band.centers), np.asarray(candidate_path)[1:-1]):
        raise IRIVMError("oracle centers do not exactly match the selected path")
    grad_blocks = _validate_gradient_blocks(band.grad_F_block_means)
    metric_blocks = np.asarray(band.M_block_means, dtype=np.float64)
    if (
        metric_blocks.ndim != 4
        or metric_blocks.shape[0] != N_INTERNAL
        or metric_blocks.shape[1] != grad_blocks.shape[1]
        or metric_blocks.shape[2:] != (DIMENSION, DIMENSION)
        or np.any(~np.isfinite(metric_blocks))
    ):
        raise IRIVMError("metric block means do not match the complete gradient band")
    retained_frame_counts = np.asarray(band.frame_counts) - np.asarray(
        band.discarded_frame_counts
    )
    block_frame_counts = _block_frame_counts(
        retained_frame_counts, grad_blocks.shape[1]
    )
    weighted_grad = _weighted_block_mean(grad_blocks, block_frame_counts)
    weighted_M = _weighted_block_mean(metric_blocks, block_frame_counts)
    if not (
        np.allclose(band.grad_F, weighted_grad, rtol=1.0e-12, atol=1.0e-12)
        and np.allclose(band.M, weighted_M, rtol=1.0e-12, atol=1.0e-12)
    ):
        raise IRIVMError("oracle band aggregate is not frame-count weighted")
    _validate_path_metric_gradient(candidate_path, band.M, band.grad_F)
    try:
        json.dumps(band.metadata, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise IRIVMError("oracle metadata is not finite JSON") from error


def _metrics_arrays(metrics: Dict[str, Array]) -> Dict[str, Array]:
    return {
        "I_perp_G": np.asarray(metrics["I_perp_G"], dtype=np.float64),
        "I_arc": np.asarray(metrics["I_arc"], dtype=np.float64),
        "Delta_F_barrier": np.asarray(metrics["Delta_F_barrier"], dtype=np.float64),
        "Delta_F_final": np.asarray(metrics["Delta_F_final"], dtype=np.float64),
        "free_energy_profile": np.asarray(metrics["free_energy_profile"], dtype=np.float64),
        "metric_eigenvalues": np.asarray(metrics["metric_eigenvalues"], dtype=np.float64),
        "euclidean_segment_lengths": np.asarray(
            metrics["euclidean_segment_lengths"], dtype=np.float64
        ),
        "segment_length_relative_deviation": np.asarray(
            metrics["segment_length_relative_deviation"], dtype=np.float64
        ),
        "maximum_euclidean_segment_length": np.asarray(
            metrics["maximum_euclidean_segment_length"], dtype=np.float64
        ),
    }


def _residual_arrays(prefix: str, statistics: ResidualStatistics) -> Dict[str, Array]:
    return {
        f"{prefix}normal_coordinate": statistics.normal_coordinate,
        f"{prefix}projected_gradient": statistics.projected_gradient,
        f"{prefix}raw_proximal_residual": statistics.raw_residual,
        f"{prefix}residual_block_means": statistics.residual_block_means,
        f"{prefix}residual_standard_error": statistics.standard_error,
        f"{prefix}residual_standard_error_available": statistics.standard_error_available,
        f"{prefix}noise_shrinkage_factor": statistics.shrinkage_factor,
        f"{prefix}effective_proximal_residual": statistics.effective_residual,
        f"{prefix}raw_residual_rms": np.asarray(statistics.raw_rms),
        f"{prefix}effective_residual_rms": np.asarray(statistics.effective_rms),
        f"{prefix}residual_lower_norm": np.asarray(statistics.lower_norm),
        f"{prefix}residual_upper_norm": np.asarray(statistics.upper_norm),
        f"{prefix}residual_bounds_available": np.asarray(
            statistics.bounds_available, dtype=bool
        ),
    }


def _candidate_arrays(prefix: str, candidate: Candidate) -> Dict[str, Array]:
    reparam = candidate.reparameterization
    if reparam is None:
        reparam_path = np.full((N_PATH, DIMENSION), np.nan, dtype=np.float64)
        audit_sha = "none"
        iterations = -1
        pre_residual = math.nan
        output_residual = math.nan
        chord_deviation = math.nan
        i_arc = math.nan
    else:
        reparam_path = reparam.path
        audit_sha = reparam.audit_sha256
        iterations = reparam.iteration_count
        pre_residual = reparam.stopping_pre_update_spring_residual
        output_residual = reparam.output_spring_residual
        chord_deviation = reparam.actual_chord_relative_deviation
        i_arc = reparam.actual_I_arc
    return {
        f"{prefix}candidate_valid": np.asarray(candidate.valid, dtype=bool),
        f"{prefix}candidate_error": np.asarray(candidate.error),
        f"{prefix}raw_direction": candidate.raw_direction,
        f"{prefix}capped_direction": candidate.capped_direction,
        f"{prefix}whole_band_common_cap_scale": np.asarray(candidate.common_cap_scale),
        f"{prefix}whole_band_common_cap_truncated": np.asarray(
            candidate.common_cap_truncated, dtype=bool
        ),
        f"{prefix}raw_path_unwrapped": candidate.raw_path,
        f"{prefix}reparameterized_path_unwrapped": reparam_path,
        f"{prefix}reparameterization_audit_sha256": np.asarray(audit_sha),
        f"{prefix}reparameterization_iterations": np.asarray(iterations, dtype=np.int64),
        f"{prefix}reparameterization_pre_update_residual": np.asarray(pre_residual),
        f"{prefix}reparameterization_output_residual": np.asarray(output_residual),
        f"{prefix}reparameterization_chord_relative_deviation": np.asarray(chord_deviation),
        f"{prefix}reparameterization_I_arc": np.asarray(i_arc),
        f"{prefix}actual_normal_coordinate_after_reparameterization": (
            candidate.actual_normal_coordinate
        ),
        f"{prefix}actual_normal_step_after_reparameterization": candidate.actual_normal_step,
        f"{prefix}post_reparameterization_residual_linear_decrease": np.asarray(
            candidate.residual_linear_decrease
        ),
        f"{prefix}post_reparameterization_proximal_quadratic_penalty": np.asarray(
            candidate.proximal_quadratic_penalty
        ),
        f"{prefix}post_reparameterization_admissibility_score": np.asarray(
            candidate.admissibility_score
        ),
    }


def _save_initial_state(
    run_directory: Path,
    controller: Dict[str, Any],
    initial_M_blocks: Array,
) -> None:
    statistics = _residual_statistics(
        path=controller["current_path"],
        grad=controller["current_grad"],
        grad_blocks=controller["current_grad_blocks"],
        frozen_path=controller["frozen_path"],
        frozen_basis=controller["frozen_basis"],
        frozen_G=controller["frozen_G"],
        alpha=controller["controls"].alpha,
        bound_kappa=controller["controls"].residual_bound_kappa,
        single_block_exact=False,
        block_frame_counts=controller["current_block_frame_counts"],
    )
    metrics = _metrics_arrays(
        compute_article_metrics(
            controller["current_path"], controller["current_M"], controller["current_grad"]
        )
    )
    h0 = _fit_h0_regression(controller, statistics, controller["controls"])
    arrays: Dict[str, Array] = {
        "implementation_id": np.asarray(IMPLEMENTATION_ID),
        "development_only": np.asarray(True, dtype=bool),
        "state_index": np.asarray(0, dtype=np.int64),
        "nof_mean_force_incremental": np.asarray(0, dtype=np.int64),
        "nof_mean_force_absolute_including_shared_initial": np.asarray(
            N_INTERNAL, dtype=np.int64
        ),
        "outer_index": np.asarray(0, dtype=np.int64),
        "inner_step": np.asarray(0, dtype=np.int64),
        "state_kind": np.asarray("common_initial"),
        "full_path_unwrapped": controller["current_path"],
        "full_path_wrapped": wrap_periodic(controller["current_path"]),
        "centers": controller["current_path"][1:-1],
        "mean_coordinates": controller["current_path"][1:-1],
        "M": controller["current_M"],
        "grad_F": controller["current_grad"],
        "grad_F_block_means": controller["current_grad_blocks"],
        "M_block_means": controller["current_M_blocks"],
        "block_frame_counts": controller["current_block_frame_counts"],
        "md_seeds": np.full(N_INTERNAL, -1, dtype=np.int64),
        "md_thermostat_seeds": np.full(N_INTERNAL, -1, dtype=np.int64),
        "frame_counts": np.zeros(N_INTERNAL, dtype=np.int64),
        "discarded_frame_counts": np.zeros(N_INTERNAL, dtype=np.int64),
        "image_wall_seconds": np.zeros(N_INTERNAL, dtype=np.float64),
        "oracle_metadata_json": np.asarray("{}"),
        "frozen_reference_path_unwrapped": controller["frozen_path"],
        "frozen_reference_M": controller["frozen_M"],
        "frozen_reference_grad_F": controller["frozen_grad"],
        "frozen_reference_centered_tangent": controller["frozen_tangent"],
        "frozen_reference_G": controller["frozen_G"],
        "frozen_reference_normal_basis": controller["frozen_basis"],
        "outer_initial_effective_residual_rms": controller[
            "outer_initial_effective_rms"
        ],
        "outer_anchor_residual_lower_norm": controller["outer_anchor_lower_norm"],
        "outer_anchor_residual_upper_norm": controller["outer_anchor_upper_norm"],
        "outer_anchor_bounds_ready": np.asarray(False, dtype=bool),
        "controller_phase": np.asarray("INITIAL_NO_SE_SPECIAL_CASE"),
        "h0_inverse_diagonal_scale": h0.inverse_scale,
        "h0_regression_valid": h0.valid,
        "h0_regression_reason": h0.reason,
        "joint_lbfgs_pair_count": np.asarray(0, dtype=np.int64),
        "one_way_takeover_latched": np.asarray(False, dtype=bool),
        "warm_safeguard_inverse_scale": np.asarray(
            controller["warm_safeguard_inverse_scale"], dtype=np.float64
        ),
        "warm_safeguard_source_state_index": np.asarray(-1, dtype=np.int64),
        "stopped_inner": np.asarray(False, dtype=bool),
        "inner_stop_reason": np.asarray("common_initial"),
        "reparameterization_applied_before_evaluation": np.asarray(False, dtype=bool),
        "normal_mapping_recomputed_after_reparameterization": np.asarray(False, dtype=bool),
        "selected_reparameterization_audit_sha256": np.asarray(
            "common-initial-no-reparameterization"
        ),
        "selected_reparameterization_iterations": np.asarray(0, dtype=np.int64),
        "selected_reparameterization_output_residual": np.asarray(math.nan),
        "hold_reused_source_reparameterization_certificate": np.asarray(
            False, dtype=bool
        ),
        "complete_band": np.asarray(True, dtype=bool),
        "common_initial_state": np.asarray(True, dtype=bool),
        "mean_force_vector_dimension": np.asarray(DIMENSION, dtype=np.int64),
        "block_count": np.asarray(controller["current_grad_blocks"].shape[1], dtype=np.int64),
        "noise_shrinkage_formula": np.asarray("z_eff=z*z^2/(z^2+SE^2)"),
        "initial_no_se_special_case_available": np.asarray(
            not statistics.standard_error_available.any(), dtype=bool
        ),
        "initial_no_se_special_case_used": np.asarray(False, dtype=bool),
    }
    arrays.update(_residual_arrays("", statistics))
    arrays.update(metrics)
    _atomic_publish_npz_once(_state_path(run_directory, 0), **arrays)


def _save_public_state(
    run_directory: Path,
    before: Dict[str, Any],
    after: Dict[str, Any],
    prepared: PreparedUpdate,
    band: OracleBand,
    advance: Dict[str, Any],
    metrics: Dict[str, Array],
) -> None:
    state_index = int(after["state_index"])
    statistics = advance["new_statistics_in_source_frozen_frame"]
    secant = advance["secant"]
    selected: Optional[Candidate]
    if prepared.selected_label == "H0":
        selected = prepared.h0_candidate
    elif prepared.selected_label == prepared.lbfgs_candidate.label:
        selected = prepared.lbfgs_candidate
    else:
        selected = None
    if prepared.hold_selected:
        if prepared.selected_label != "HOLD" or not np.array_equal(
            prepared.selected_path, before["current_path"]
        ):
            raise IRIVMError("HOLD public state did not reuse exact source path")
        source_row = _load_npz_record(
            _state_path(run_directory, int(before["state_index"]))
        )
        if not np.array_equal(source_row["full_path_unwrapped"], before["current_path"]):
            raise IRIVMError("HOLD source public path differs from controller path")
        selected_audit_sha = str(
            source_row["selected_reparameterization_audit_sha256"].item()
        )
        selected_iterations = int(
            source_row["selected_reparameterization_iterations"].item()
        )
        selected_output_residual = float(
            source_row["selected_reparameterization_output_residual"].item()
        )
    else:
        if selected is None or selected.reparameterization is None:
            raise IRIVMError("selected public motion lacks article reparameterization")
        selected_audit_sha = selected.reparameterization.audit_sha256
        selected_iterations = selected.reparameterization.iteration_count
        selected_output_residual = selected.reparameterization.output_spring_residual
    if prepared.initial_no_se_special_case:
        phase_before = "INITIAL_NO_SE_SPECIAL_CASE"
    elif prepared.hold_selected and not bool(before["outer_anchor_bounds_ready"]):
        phase_before = "ANCHORING"
    elif prepared.hold_selected:
        phase_before = "HOLD"
    else:
        phase_before = "MOVE_PREPARED"
    phase_after = (
        "READY" if bool(after["outer_anchor_bounds_ready"]) else "ANCHORING"
    )
    current_geometry = build_metric_geometry(after["current_path"], after["current_M"])
    arrays: Dict[str, Array] = {
        "implementation_id": np.asarray(IMPLEMENTATION_ID),
        "development_only": np.asarray(True, dtype=bool),
        "state_index": np.asarray(state_index, dtype=np.int64),
        "nof_mean_force_incremental": np.asarray(after["nof"], dtype=np.int64),
        "nof_mean_force_absolute_including_shared_initial": np.asarray(
            after["nof"] + N_INTERNAL, dtype=np.int64
        ),
        "outer_index": np.asarray(advance["source_outer_index"], dtype=np.int64),
        "inner_step": np.asarray(advance["completed_inner_step"], dtype=np.int64),
        "state_kind": np.asarray(
            HOLD_STATE_KIND if prepared.hold_selected else STATE_KIND
        ),
        "proposal_sha256": np.asarray(prepared.proposal_sha256),
        "source_checkpoint_sha256": np.asarray(prepared.source_checkpoint_sha256),
        "selected_candidate_kind": np.asarray(prepared.selected_label),
        "controller_phase_before_paid_band": np.asarray(phase_before),
        "controller_phase_after_paid_band": np.asarray(phase_after),
        "full_path_unwrapped": after["current_path"],
        "full_path_wrapped": wrap_periodic(after["current_path"]),
        "source_paid_path_unwrapped": before["current_path"],
        "frozen_reference_path_unwrapped": before["frozen_path"],
        "frozen_reference_M": before["frozen_M"],
        "frozen_reference_grad_F": before["frozen_grad"],
        "frozen_reference_centered_tangent": before["frozen_tangent"],
        "frozen_reference_G": before["frozen_G"],
        "frozen_reference_normal_basis": before["frozen_basis"],
        "source_segment_lengths": prepared.source_segment_lengths,
        "global_trust_radius": np.asarray(prepared.global_trust_radius),
        "endpoint_trust_radii": prepared.endpoint_trust_radii,
        "whole_band_common_physical_cap": np.asarray(True, dtype=bool),
        "h0_inverse_diagonal_scale": prepared.h0_regression.inverse_scale,
        "h0_raw_gradient_slope": prepared.h0_regression.raw_gradient_slope,
        "h0_proximal_curvature": prepared.h0_regression.proximal_curvature,
        "h0_local_paid_point_count": prepared.h0_regression.local_point_count,
        "h0_effective_paid_point_count": prepared.h0_regression.effective_point_count,
        "h0_inverse_scale_clipped": prepared.h0_regression.clipped,
        "h0_regression_valid": prepared.h0_regression.valid,
        "h0_regression_reason": prepared.h0_regression.reason,
        "joint_lbfgs_pair_count_before": np.asarray(before["lbfgs_count"], dtype=np.int64),
        "joint_lbfgs_history_size_fixed": np.asarray(JOINT_HISTORY_SIZE, dtype=np.int64),
        "one_way_takeover_latched_before": np.asarray(prepared.latch_before, dtype=bool),
        "one_way_takeover_latched_after_selection": np.asarray(prepared.latch_after, dtype=bool),
        "strict_lbfgs_win_trigger": np.asarray(prepared.strict_lbfgs_win, dtype=bool),
        "forced_lbfgs_after_takeover": np.asarray(
            prepared.forced_lbfgs_after_takeover, dtype=bool
        ),
        "h0_invalid_fallback_after_takeover": np.asarray(
            prepared.h0_invalid_fallback, dtype=bool
        ),
        "hold_resample_selected": np.asarray(prepared.hold_selected, dtype=bool),
        "hold_resample_reason": np.asarray(prepared.hold_reason),
        "hold_reused_exact_path_bytes": np.asarray(
            prepared.hold_selected
            and np.array_equal(prepared.selected_path, before["current_path"]),
            dtype=bool,
        ),
        "hold_source_path_sha256": np.asarray(_array_sha256(before["current_path"])),
        "hold_reused_source_reparameterization_certificate": np.asarray(
            prepared.hold_selected, dtype=bool
        ),
        "global_residual_signal_to_noise": np.asarray(
            prepared.global_signal_to_noise
        ),
        "initial_no_se_special_case_used": np.asarray(
            prepared.initial_no_se_special_case, dtype=bool
        ),
        "candidate_admissibility_difference_standard_error": np.asarray(
            prepared.admissibility_difference_standard_error
        ),
        "selected_admissibility_score_standard_error": np.asarray(
            prepared.selected_admissibility_standard_error
        ),
        "selected_admissibility_score_lower_bound": np.asarray(
            prepared.selected_admissibility_lower_bound
        ),
        "normal_mapping_recomputed_after_reparameterization": np.asarray(
            not prepared.hold_selected, dtype=bool
        ),
        "candidate_reparameterization_calls": np.asarray(
            int(prepared.h0_candidate.reparameterization is not None)
            + int(prepared.lbfgs_candidate.reparameterization is not None),
            dtype=np.int64,
        ),
        "evaluated_candidate_count": np.asarray(1, dtype=np.int64),
        "preselection_mean_force_evaluations": np.asarray(0, dtype=np.int64),
        "only_selected_complete_band_evaluated": np.asarray(True, dtype=bool),
        "reparameterizer_id": np.asarray(REPARAMETERIZER_ID),
        "selected_reparameterization_audit_sha256": np.asarray(
            selected_audit_sha
        ),
        "selected_reparameterization_iterations": np.asarray(
            selected_iterations, dtype=np.int64
        ),
        "selected_reparameterization_output_residual": np.asarray(
            selected_output_residual
        ),
        "secant_s_composite_actual_u": secant["s"],
        "secant_y_raw_proximal_residual": secant["y"],
        "secant_y_standard_error": secant["sigma_y"],
        "secant_s_norm": np.asarray(secant["s_norm"]),
        "secant_y_norm": np.asarray(secant["y_norm"]),
        "secant_curvature": np.asarray(secant["curvature"]),
        "secant_cosine_threshold": np.asarray(secant["cosine_threshold"]),
        "secant_curvature_standard_error": np.asarray(
            secant["curvature_standard_error"]
        ),
        "secant_noise_threshold": np.asarray(secant["noise_threshold"]),
        "secant_gate_accepted": np.asarray(secant["accepted"], dtype=bool),
        "secant_gate_reason": np.asarray(secant["reason"]),
        "secant_stored_after_transition": np.asarray(
            advance["secant_stored_after_transition"], dtype=bool
        ),
        "outer_initial_effective_residual_rms": before[
            "outer_initial_effective_rms"
        ],
        "source_outer_anchor_residual_lower_norm": before[
            "outer_anchor_lower_norm"
        ],
        "source_outer_anchor_residual_upper_norm": before[
            "outer_anchor_upper_norm"
        ],
        "source_outer_anchor_bounds_ready": np.asarray(
            before["outer_anchor_bounds_ready"], dtype=bool
        ),
        "paid_action_is_movement": np.asarray(
            advance["movement_committed"], dtype=bool
        ),
        "hold_does_not_increment_movement_count": np.asarray(
            prepared.hold_selected, dtype=bool
        ),
        "residual_interval_clear_continue": np.asarray(
            advance["clear_continue"], dtype=bool
        ),
        "residual_interval_ambiguous": np.asarray(
            advance["residual_decision_ambiguous"], dtype=bool
        ),
        "global_effective_residual_ratio": np.asarray(advance["residual_ratio"]),
        "global_r_threshold": np.asarray(before["controls"].r),
        "r_global_stop": np.asarray(advance["r_stop"], dtype=bool),
        "m_max_stop": np.asarray(advance["m_max_stop"], dtype=bool),
        "r_stop_has_precedence": np.asarray(True, dtype=bool),
        "stopped_inner": np.asarray(advance["stopped_inner"], dtype=bool),
        "inner_stop_reason": np.asarray(advance["inner_stop_reason"]),
        "frozen_frame_rebuilt_after_stop": np.asarray(
            advance["frame_rebuilt"], dtype=bool
        ),
        "next_outer_index": np.asarray(after["outer_index"], dtype=np.int64),
        "next_inner_step": np.asarray(after["inner_step"], dtype=np.int64),
        "next_outer_anchor_residual_lower_norm": after["outer_anchor_lower_norm"],
        "next_outer_anchor_residual_upper_norm": after["outer_anchor_upper_norm"],
        "next_outer_anchor_bounds_ready": np.asarray(
            after["outer_anchor_bounds_ready"], dtype=bool
        ),
        "next_frozen_reference_path_unwrapped": after["frozen_path"],
        "next_frozen_reference_M": after["frozen_M"],
        "next_frozen_reference_centered_tangent": after["frozen_tangent"],
        "next_frozen_reference_G": after["frozen_G"],
        "next_frozen_reference_normal_basis": after["frozen_basis"],
        "next_frozen_basis_gauge_flipped": np.asarray(
            advance["next_frozen_basis_gauge_flipped"], dtype=bool
        ),
        "joint_lbfgs_pair_count_next": np.asarray(after["lbfgs_count"], dtype=np.int64),
        "one_way_takeover_latched_next": np.asarray(after["lbfgs_latched"], dtype=bool),
        "warm_safeguard_inverse_scale_next": np.asarray(
            after["warm_safeguard_inverse_scale"], dtype=np.float64
        ),
        "warm_safeguard_source_state_index_next": np.asarray(
            after["warm_safeguard_source_state_index"], dtype=np.int64
        ),
        "current_frame_lbfgs_memory_available_next": np.asarray(
            int(after["lbfgs_count"]) >= 2, dtype=bool
        ),
        "cold_recovery_forbidden_after_global_takeover": np.asarray(
            bool(after["lbfgs_latched"]), dtype=bool
        ),
        "current_centered_tangent": np.asarray(current_geometry.tangent),
        "current_G": np.asarray(current_geometry.G),
        "current_normal_basis": np.asarray(current_geometry.normal_basis),
        "centers": np.asarray(band.centers, dtype=np.float64),
        "mean_coordinates": np.asarray(band.mean_coordinates, dtype=np.float64),
        "grad_F": np.asarray(after["current_grad"], dtype=np.float64),
        "M": np.asarray(after["current_M"], dtype=np.float64),
        "grad_F_block_means": np.asarray(
            after["current_grad_blocks"], dtype=np.float64
        ),
        "M_block_means": np.asarray(after["current_M_blocks"], dtype=np.float64),
        "block_frame_counts": np.asarray(
            after["current_block_frame_counts"], dtype=np.int64
        ),
        "oracle_raw_grad_F": np.asarray(band.grad_F, dtype=np.float64),
        "oracle_raw_M": np.asarray(band.M, dtype=np.float64),
        "oracle_raw_grad_F_block_means": np.asarray(
            band.grad_F_block_means, dtype=np.float64
        ),
        "oracle_raw_M_block_means": np.asarray(
            band.M_block_means, dtype=np.float64
        ),
        "oracle_raw_block_frame_counts": np.asarray(
            advance["oracle_raw_block_frame_counts"], dtype=np.int64
        ),
        "md_seeds": np.asarray(band.seeds, dtype=np.int64),
        "md_thermostat_seeds": np.asarray(band.thermostat_seeds, dtype=np.int64),
        "frame_counts": np.asarray(band.frame_counts, dtype=np.int64),
        "discarded_frame_counts": np.asarray(
            band.discarded_frame_counts, dtype=np.int64
        ),
        "image_wall_seconds": np.asarray(band.wall_seconds, dtype=np.float64),
        "oracle_metadata_json": np.asarray(
            json.dumps(band.metadata, sort_keys=True, separators=(",", ":"))
        ),
        "block_count": np.asarray(
            np.asarray(after["current_grad_blocks"]).shape[1], dtype=np.int64
        ),
        "oracle_raw_block_count": np.asarray(
            np.asarray(band.grad_F_block_means).shape[1], dtype=np.int64
        ),
        "hold_estimator_pooled_all_exact_center_blocks": np.asarray(
            prepared.hold_selected, dtype=bool
        ),
        "nof_this_complete_band": np.asarray(N_INTERNAL, dtype=np.int64),
        "mean_force_vector_dimension": np.asarray(DIMENSION, dtype=np.int64),
        "reparameterization_applied_before_evaluation": np.asarray(
            not prepared.hold_selected, dtype=bool
        ),
        "complete_band": np.asarray(True, dtype=bool),
        "common_initial_state": np.asarray(False, dtype=bool),
        "noise_shrinkage_formula": np.asarray("z_eff=z*z^2/(z^2+SE^2)"),
    }
    arrays.update(_residual_arrays("source_", prepared.source_statistics))
    arrays.update(_residual_arrays("", statistics))
    arrays.update(_candidate_arrays("h0_", prepared.h0_candidate))
    arrays.update(_candidate_arrays("lbfgs_", prepared.lbfgs_candidate))
    arrays.update(_metrics_arrays(metrics))
    _atomic_publish_npz_once(_state_path(run_directory, state_index), **arrays)


def _manifest(
    oracle: object,
    controls: IRIVMControls,
    nof_budget: int,
    initial_path: Array,
    initial_M: Array,
    initial_grad: Array,
    initial_grad_blocks: Array,
    initial_M_blocks: Array,
) -> Dict[str, object]:
    provenance_method = getattr(oracle, "provenance", None)
    provenance = (
        provenance_method()
        if callable(provenance_method)
        else {
            "oracle_id": str(getattr(oracle, "oracle_id", type(oracle).__name__)),
            "formal_evidence": bool(getattr(oracle, "formal_evidence", False)),
        }
    )
    return {
        "implementation_id": IMPLEMENTATION_ID,
        "implementation_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "development_only": True,
        "controls": asdict(controls),
        "causal_initial_path_sha256": _array_sha256(initial_path),
        "causal_initial_M_sha256": _array_sha256(initial_M),
        "causal_initial_grad_F_sha256": _array_sha256(initial_grad),
        "causal_initial_grad_F_block_means_sha256": _array_sha256(
            initial_grad_blocks
        ),
        "causal_initial_M_block_means_sha256": _array_sha256(initial_M_blocks),
        "causal_initial_block_frame_counts_sha256": _array_sha256(
            np.zeros((N_INTERNAL, initial_grad_blocks.shape[1]), dtype=np.int64)
        ),
        "nof_budget_mean_force_incremental": int(nof_budget),
        "nof_definition": (
            "one estimate of the complete two-component mean-force vector at "
            "one internal collective-variable point equals one NoF"
        ),
        "complete_20_image_band_cost_nof": N_INTERNAL,
        "common_initial_band_excluded_from_incremental_axis": True,
        "absolute_nof_additive_constant": N_INTERNAL,
        "oracle_id": str(getattr(oracle, "oracle_id", type(oracle).__name__)),
        "oracle_formal_evidence": bool(getattr(oracle, "formal_evidence", False)),
        "oracle_provenance": provenance,
        "single_block_exact_after_initial": bool(
            getattr(oracle, "single_block_standard_error_exact", False)
            or str(getattr(oracle, "oracle_id", ""))
            == "fes-spline-plus-historical-M-knn-prescreen-v1"
        ),
        "algorithm_variant": "v14_development_frozen_noise_aware_joint_lbfgs",
        "internal_image_count": N_INTERNAL,
        "outer_frozen_objects": [
            "reference_path",
            "article_centered_tangent",
            "M",
            "G=M^-1",
            "G_normal_G_orthonormal_basis",
        ],
        "normal_coordinate": "u=B_ref.T*G_ref*periodic(x_actual-x_ref)",
        "proximal_residual": "z=B_ref.T*grad_F+u/alpha",
        "residual_standard_error": "sample_SE_from_oracle_gradient_block_means",
        "initial_no_se_special_case": (
            "state 0 only: mark SE unavailable and permit one cold frozen-P "
            "proposal before the first paid 20-NoF band; never bootstrap-resample"
        ),
        "noise_shrinkage": "z_eff=z*z^2/(z^2+SE^2)",
        "noise_residual_bounds": (
            "L=norm((abs(P)-2.262*SE)_+), U=norm(abs(P)+2.262*SE)"
        ),
        "H0": (
            "per-image diagonal inverse scale from historical local weighted "
            "gradient-curvature regression; invalid entries equal alpha"
        ),
        "joint_lbfgs_history_size": JOINT_HISTORY_SIZE,
        "secant": "actual_post_reparameterization_delta_u, delta_raw_proximal_residual",
        "secant_noise_gate": "sTy > 2*sqrt(sum((s*SE_delta_y)^2)) plus cosine gate",
        "selector": (
            "66D global one-way strict-LBFGS takeover, eligible only after at "
            "least two current-frame noise-qualified distinct post-R secants; "
            "cold recovery is permanently forbidden after takeover"
        ),
        "post_takeover_fallback": (
            "current-frame L-BFGS when at least two pairs exist, otherwise the "
            "last sTy/yTy scalar clipped to [0.25 alpha,4 alpha]; never H0"
        ),
        "physical_cap": "one common scalar for the complete 20-image band",
        "reparameterizer_id": REPARAMETERIZER_ID,
        "candidate_reparameterization_before_selection": True,
        "post_reparameterization_actual_u_recomputed": True,
        "candidate_admissibility_score": (
            "Psi=-P_eff dot delta_u-||delta_u||^2/(2*alpha), using actual "
            "post-reparameterization delta_u"
        ),
        "low_snr_policy": (
            "exact-center HOLD/resample; byte-identical path; no article "
            "reparameterization of the HOLD"
        ),
        "ambiguous_candidate_policy": (
            "HOLD/resample when the H0-versus-LBFGS admissibility-score "
            "difference is within two shared-source block-standard-errors"
        ),
        "hold_resample_estimator": (
            "pool all gradient and metric block means observed at that exact center"
        ),
        "only_selected_candidate_is_evaluated": True,
        "reparameterization_mean_force_cost": 0,
        "paid_band_mean_force_cost": N_INTERNAL,
        "global_r_stop_rebuilds_frozen_outer": True,
        "m_max_stop_rebuilds_frozen_outer": True,
        "m_max_counts_movements_not_exact_center_resamples": True,
        "relative_residual_stop": "U(P_j) <= r*L(P_anchor)",
        "relative_residual_continue": "L(P_j) > r*U(P_anchor)",
        "relative_residual_ambiguous_action": "exact-center HOLD/resample",
        "r_stop_precedes_m_max_stop": True,
        "performance_metrics_visible_to_optimizer": False,
        "next_action_is_independent_of_declared_budget": True,
        "atomic_immutable_public_states": True,
        "atomic_replaceable_checkpoint": True,
        "one_state_ahead_deterministic_recovery": True,
        "jobs": 1,
    }


def _assert_metrics(row: Dict[str, Array], state_index: int) -> None:
    expected = _metrics_arrays(
        compute_article_metrics(row["full_path_unwrapped"], row["M"], row["grad_F"])
    )
    for key, value in expected.items():
        if key not in row or not np.array_equal(row[key], value):
            raise IRIVMError(f"public state {state_index} has inconsistent {key}")


def _replay_one_state(
    controller: Dict[str, Any], row: Dict[str, Array], controls: IRIVMControls
) -> Dict[str, Any]:
    expected_index = int(controller["state_index"]) + 1
    if int(row["state_index"]) != expected_index:
        raise IRIVMError("public replay state index is not the next state")
    prepared, stop = _prepare_update(controller, controls)
    if prepared is None:
        raise IRIVMError(
            f"public state exists after deterministic zero-NoF stop: {stop}"
        )
    exact = {
        "proposal_sha256": np.asarray(prepared.proposal_sha256),
        "source_checkpoint_sha256": np.asarray(prepared.source_checkpoint_sha256),
        "selected_candidate_kind": np.asarray(prepared.selected_label),
        "full_path_unwrapped": prepared.selected_path,
        "frozen_reference_path_unwrapped": controller["frozen_path"],
        "frozen_reference_M": controller["frozen_M"],
        "frozen_reference_centered_tangent": controller["frozen_tangent"],
        "frozen_reference_G": controller["frozen_G"],
        "frozen_reference_normal_basis": controller["frozen_basis"],
        "h0_inverse_diagonal_scale": prepared.h0_regression.inverse_scale,
    }
    for key, value in exact.items():
        if key not in row or not np.array_equal(row[key], value):
            raise IRIVMError(
                f"public state {expected_index} differs from deterministic V14 proposal: {key}"
            )
    if not bool(row["complete_band"]):
        raise IRIVMError("public paid state is not a complete band")
    expected_reparameterization = not prepared.hold_selected
    if bool(row["reparameterization_applied_before_evaluation"]) != expected_reparameterization:
        raise IRIVMError("public paid state has inconsistent reparameterization/HOLD semantics")
    if prepared.hold_selected and not (
        bool(row["hold_reused_exact_path_bytes"])
        and np.array_equal(row["full_path_unwrapped"], controller["current_path"])
    ):
        raise IRIVMError("public HOLD state changed an exact center")
    if not np.array_equal(row["centers"], prepared.selected_path[1:-1]):
        raise IRIVMError("public paid centers differ from selected path")
    if int(row["nof_this_complete_band"]) != N_INTERNAL:
        raise IRIVMError("public paid state has a non-20 NoF band")

    try:
        oracle_metadata = json.loads(str(row["oracle_metadata_json"].item()))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise IRIVMError(
            f"public state {expected_index} has invalid oracle metadata"
        ) from error
    replayed_band = OracleBand(
        centers=row["centers"],
        mean_coordinates=row["mean_coordinates"],
        grad_F=row["oracle_raw_grad_F"],
        M=row["oracle_raw_M"],
        grad_F_block_means=row["oracle_raw_grad_F_block_means"],
        M_block_means=row["oracle_raw_M_block_means"],
        seeds=row["md_seeds"],
        thermostat_seeds=row["md_thermostat_seeds"],
        frame_counts=row["frame_counts"],
        discarded_frame_counts=row["discarded_frame_counts"],
        wall_seconds=row["image_wall_seconds"],
        nof=int(row["nof_this_complete_band"]),
        metadata=oracle_metadata,
    )
    _validate_complete_band(replayed_band, prepared.selected_path)
    next_controller, advance = _advance_controller(
        controller,
        prepared,
        M=replayed_band.M,
        grad=replayed_band.grad_F,
        grad_blocks=replayed_band.grad_F_block_means,
        M_blocks=replayed_band.M_block_means,
        frame_counts=replayed_band.frame_counts,
        discarded_frame_counts=replayed_band.discarded_frame_counts,
        controls=controls,
    )
    phase_before = (
        "INITIAL_NO_SE_SPECIAL_CASE"
        if prepared.initial_no_se_special_case
        else (
            "ANCHORING"
            if prepared.hold_selected
            and not bool(controller["outer_anchor_bounds_ready"])
            else "HOLD" if prepared.hold_selected else "MOVE_PREPARED"
        )
    )
    phase_after = (
        "READY" if bool(next_controller["outer_anchor_bounds_ready"]) else "ANCHORING"
    )
    current_geometry = build_metric_geometry(
        next_controller["current_path"], next_controller["current_M"]
    )
    secant = advance["secant"]
    expected_advance = {
        "nof_mean_force_incremental": np.asarray(next_controller["nof"]),
        "nof_mean_force_absolute_including_shared_initial": np.asarray(
            next_controller["nof"] + N_INTERNAL
        ),
        "outer_index": np.asarray(advance["source_outer_index"]),
        "inner_step": np.asarray(advance["completed_inner_step"]),
        "state_kind": np.asarray(
            HOLD_STATE_KIND if prepared.hold_selected else STATE_KIND
        ),
        "source_paid_path_unwrapped": controller["current_path"],
        "frozen_reference_grad_F": controller["frozen_grad"],
        "source_segment_lengths": prepared.source_segment_lengths,
        "global_trust_radius": np.asarray(prepared.global_trust_radius),
        "endpoint_trust_radii": prepared.endpoint_trust_radii,
        "whole_band_common_physical_cap": np.asarray(True),
        "controller_phase_before_paid_band": np.asarray(phase_before),
        "controller_phase_after_paid_band": np.asarray(phase_after),
        "joint_lbfgs_pair_count_before": np.asarray(controller["lbfgs_count"]),
        "joint_lbfgs_history_size_fixed": np.asarray(JOINT_HISTORY_SIZE),
        "one_way_takeover_latched_before": np.asarray(prepared.latch_before),
        "one_way_takeover_latched_after_selection": np.asarray(
            prepared.latch_after
        ),
        "strict_lbfgs_win_trigger": np.asarray(prepared.strict_lbfgs_win),
        "forced_lbfgs_after_takeover": np.asarray(
            prepared.forced_lbfgs_after_takeover
        ),
        "h0_invalid_fallback_after_takeover": np.asarray(
            prepared.h0_invalid_fallback
        ),
        "hold_resample_selected": np.asarray(prepared.hold_selected),
        "hold_resample_reason": np.asarray(prepared.hold_reason),
        "hold_source_path_sha256": np.asarray(
            _array_sha256(controller["current_path"])
        ),
        "hold_estimator_pooled_all_exact_center_blocks": np.asarray(
            prepared.hold_selected
        ),
        "paid_action_is_movement": np.asarray(advance["movement_committed"]),
        "hold_does_not_increment_movement_count": np.asarray(
            prepared.hold_selected
        ),
        "initial_no_se_special_case_used": np.asarray(
            prepared.initial_no_se_special_case
        ),
        "candidate_reparameterization_calls": np.asarray(
            int(prepared.h0_candidate.reparameterization is not None)
            + int(prepared.lbfgs_candidate.reparameterization is not None)
        ),
        "evaluated_candidate_count": np.asarray(1),
        "preselection_mean_force_evaluations": np.asarray(0),
        "only_selected_complete_band_evaluated": np.asarray(True),
        "reparameterization_applied_before_evaluation": np.asarray(
            not prepared.hold_selected
        ),
        "normal_mapping_recomputed_after_reparameterization": np.asarray(
            not prepared.hold_selected
        ),
        "full_path_unwrapped": next_controller["current_path"],
        "M": next_controller["current_M"],
        "grad_F": next_controller["current_grad"],
        "grad_F_block_means": next_controller["current_grad_blocks"],
        "M_block_means": next_controller["current_M_blocks"],
        "block_frame_counts": next_controller["current_block_frame_counts"],
        "oracle_raw_block_frame_counts": advance[
            "oracle_raw_block_frame_counts"
        ],
        "block_count": np.asarray(
            next_controller["current_grad_blocks"].shape[1]
        ),
        "oracle_raw_block_count": np.asarray(
            replayed_band.grad_F_block_means.shape[1]
        ),
        "secant_s_composite_actual_u": secant["s"],
        "secant_y_raw_proximal_residual": secant["y"],
        "secant_y_standard_error": secant["sigma_y"],
        "secant_s_norm": np.asarray(secant["s_norm"]),
        "secant_y_norm": np.asarray(secant["y_norm"]),
        "secant_curvature": np.asarray(secant["curvature"]),
        "secant_cosine_threshold": np.asarray(secant["cosine_threshold"]),
        "secant_curvature_standard_error": np.asarray(
            secant["curvature_standard_error"]
        ),
        "secant_noise_threshold": np.asarray(secant["noise_threshold"]),
        "secant_gate_accepted": np.asarray(secant["accepted"]),
        "secant_gate_reason": np.asarray(secant["reason"]),
        "secant_stored_after_transition": np.asarray(
            advance["secant_stored_after_transition"]
        ),
        "source_outer_anchor_bounds_ready": np.asarray(
            controller["outer_anchor_bounds_ready"]
        ),
        "source_outer_anchor_residual_lower_norm": controller[
            "outer_anchor_lower_norm"
        ],
        "source_outer_anchor_residual_upper_norm": controller[
            "outer_anchor_upper_norm"
        ],
        "outer_initial_effective_residual_rms": controller[
            "outer_initial_effective_rms"
        ],
        "residual_interval_clear_continue": np.asarray(advance["clear_continue"]),
        "residual_interval_ambiguous": np.asarray(
            advance["residual_decision_ambiguous"]
        ),
        "global_effective_residual_ratio": np.asarray(advance["residual_ratio"]),
        "global_r_threshold": np.asarray(controls.r),
        "r_global_stop": np.asarray(advance["r_stop"]),
        "m_max_stop": np.asarray(advance["m_max_stop"]),
        "r_stop_has_precedence": np.asarray(True),
        "stopped_inner": np.asarray(advance["stopped_inner"]),
        "inner_stop_reason": np.asarray(advance["inner_stop_reason"]),
        "next_outer_index": np.asarray(next_controller["outer_index"]),
        "next_inner_step": np.asarray(next_controller["inner_step"]),
        "next_outer_anchor_residual_lower_norm": next_controller[
            "outer_anchor_lower_norm"
        ],
        "next_outer_anchor_residual_upper_norm": next_controller[
            "outer_anchor_upper_norm"
        ],
        "next_outer_anchor_bounds_ready": np.asarray(
            next_controller["outer_anchor_bounds_ready"]
        ),
        "next_frozen_reference_path_unwrapped": next_controller["frozen_path"],
        "next_frozen_reference_M": next_controller["frozen_M"],
        "next_frozen_reference_centered_tangent": next_controller[
            "frozen_tangent"
        ],
        "next_frozen_reference_G": next_controller["frozen_G"],
        "next_frozen_reference_normal_basis": next_controller["frozen_basis"],
        "next_frozen_basis_gauge_flipped": advance[
            "next_frozen_basis_gauge_flipped"
        ],
        "joint_lbfgs_pair_count_next": np.asarray(next_controller["lbfgs_count"]),
        "one_way_takeover_latched_next": np.asarray(next_controller["lbfgs_latched"]),
        "warm_safeguard_inverse_scale_next": np.asarray(
            next_controller["warm_safeguard_inverse_scale"]
        ),
        "warm_safeguard_source_state_index_next": np.asarray(
            next_controller["warm_safeguard_source_state_index"]
        ),
        "current_frame_lbfgs_memory_available_next": np.asarray(
            int(next_controller["lbfgs_count"]) >= 2
        ),
        "cold_recovery_forbidden_after_global_takeover": np.asarray(
            next_controller["lbfgs_latched"]
        ),
        "current_centered_tangent": np.asarray(current_geometry.tangent),
        "current_G": np.asarray(current_geometry.G),
        "current_normal_basis": np.asarray(current_geometry.normal_basis),
        "h0_raw_gradient_slope": prepared.h0_regression.raw_gradient_slope,
        "h0_proximal_curvature": prepared.h0_regression.proximal_curvature,
        "h0_local_paid_point_count": prepared.h0_regression.local_point_count,
        "h0_effective_paid_point_count": prepared.h0_regression.effective_point_count,
        "h0_inverse_scale_clipped": prepared.h0_regression.clipped,
        "h0_regression_valid": prepared.h0_regression.valid,
        "h0_regression_reason": prepared.h0_regression.reason,
        "global_residual_signal_to_noise": np.asarray(
            prepared.global_signal_to_noise
        ),
        "candidate_admissibility_difference_standard_error": np.asarray(
            prepared.admissibility_difference_standard_error
        ),
        "selected_admissibility_score_standard_error": np.asarray(
            prepared.selected_admissibility_standard_error
        ),
        "selected_admissibility_score_lower_bound": np.asarray(
            prepared.selected_admissibility_lower_bound
        ),
        "reparameterizer_id": np.asarray(REPARAMETERIZER_ID),
        "complete_band": np.asarray(True),
        "common_initial_state": np.asarray(False),
        "mean_force_vector_dimension": np.asarray(DIMENSION),
        "noise_shrinkage_formula": np.asarray("z_eff=z*z^2/(z^2+SE^2)"),
    }
    if not prepared.hold_selected:
        selected_candidate = (
            prepared.h0_candidate
            if prepared.selected_label == "H0"
            else prepared.lbfgs_candidate
        )
        if selected_candidate.reparameterization is None:
            raise IRIVMError("replayed movement lacks its article certificate")
        expected_advance.update(
            {
                "selected_reparameterization_audit_sha256": np.asarray(
                    selected_candidate.reparameterization.audit_sha256
                ),
                "selected_reparameterization_iterations": np.asarray(
                    selected_candidate.reparameterization.iteration_count
                ),
                "selected_reparameterization_output_residual": np.asarray(
                    selected_candidate.reparameterization.output_spring_residual
                ),
            }
        )
    expected_advance.update(_residual_arrays("source_", prepared.source_statistics))
    expected_advance.update(
        _residual_arrays("", advance["new_statistics_in_source_frozen_frame"])
    )
    expected_advance.update(_candidate_arrays("h0_", prepared.h0_candidate))
    expected_advance.update(_candidate_arrays("lbfgs_", prepared.lbfgs_candidate))
    for key, value in expected_advance.items():
        if key not in row or not _array_equal(row[key], value):
            raise IRIVMError(f"public state {expected_index} has inconsistent {key}")
    _assert_metrics(row, expected_index)
    return next_controller


def _controllers_equal(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    keys = (
        "state_index",
        "nof",
        "outer_index",
        "inner_step",
        "current_path",
        "current_M",
        "current_grad",
        "current_grad_blocks",
        "current_M_blocks",
        "current_block_frame_counts",
        "frozen_path",
        "frozen_M",
        "frozen_grad",
        "frozen_tangent",
        "frozen_G",
        "frozen_basis",
        "outer_initial_effective_rms",
        "outer_anchor_lower_norm",
        "outer_anchor_upper_norm",
        "outer_anchor_bounds_ready",
        "observation_paths",
        "observation_gradients",
        "observation_gradient_covariances",
        "observation_covariance_available",
        "observation_state_indices",
        "observation_count",
        "lbfgs_s",
        "lbfgs_y",
        "lbfgs_count",
        "lbfgs_latched",
        "warm_safeguard_inverse_scale",
        "warm_safeguard_source_state_index",
        "single_block_exact_after_initial",
        "lbfgs_anchor_u",
        "lbfgs_anchor_raw_residual",
        "lbfgs_anchor_standard_error",
    )
    return all(_array_equal(left[key], right[key]) for key in keys)


def load_public_trajectory(run_directory: Path) -> Dict[str, Any]:
    """Validate and deterministically replay every durable complete V14 band."""

    run_directory = Path(run_directory)
    state_paths = sorted((run_directory / "states").glob("state_*.npz"))
    if not state_paths:
        raise FileNotFoundError("run has no public V14 states")
    manifest_path = run_directory / "run_manifest.json"
    if not manifest_path.is_file():
        raise IRIVMError("public V14 states exist without a run manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("implementation_id") != IMPLEMENTATION_ID:
        raise IRIVMError("manifest implementation identity differs")
    implementation_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if manifest.get("implementation_source_sha256") != implementation_source_sha256:
        raise IRIVMError("manifest implementation source hash differs")
    controls = IRIVMControls(**manifest["controls"]).validate()
    rows = [_load_npz_record(path) for path in state_paths]
    indices = np.asarray([int(row["state_index"]) for row in rows], dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(rows))):
        raise IRIVMError("public V14 state indices are not contiguous")
    nof = np.asarray(
        [int(row["nof_mean_force_incremental"]) for row in rows], dtype=np.int64
    )
    if not np.array_equal(nof, indices * N_INTERNAL):
        raise IRIVMError("public V14 NoF ledger is not exactly 20 per paid band")
    for index, row in enumerate(rows):
        if str(row["implementation_id"].item()) != IMPLEMENTATION_ID:
            raise IRIVMError(f"public state {index} implementation identity differs")
        if not bool(row["complete_band"]):
            raise IRIVMError(f"public state {index} is not a complete band")
        _assert_metrics(row, index)
        if index == 0:
            if bool(row["hold_reused_source_reparameterization_certificate"]):
                raise IRIVMError("common initial state cannot reuse a HOLD certificate")
        elif bool(row["hold_resample_selected"]):
            previous = rows[index - 1]
            if not bool(row["hold_reused_source_reparameterization_certificate"]):
                raise IRIVMError(
                    f"public HOLD state {index} did not mark certificate reuse"
                )
            for key in (
                "selected_reparameterization_audit_sha256",
                "selected_reparameterization_iterations",
                "selected_reparameterization_output_residual",
            ):
                if not _array_equal(row[key], previous[key]):
                    raise IRIVMError(
                        f"public HOLD state {index} changed source certificate field {key}"
                    )
        elif bool(row["hold_reused_source_reparameterization_certificate"]):
            raise IRIVMError(
                f"public movement state {index} falsely marked HOLD certificate reuse"
            )
    initial = rows[0]
    initial_hashes = {
        "causal_initial_path_sha256": _array_sha256(initial["full_path_unwrapped"]),
        "causal_initial_M_sha256": _array_sha256(initial["M"]),
        "causal_initial_grad_F_sha256": _array_sha256(initial["grad_F"]),
        "causal_initial_grad_F_block_means_sha256": _array_sha256(
            initial["grad_F_block_means"]
        ),
        "causal_initial_M_block_means_sha256": _array_sha256(initial["M_block_means"]),
        "causal_initial_block_frame_counts_sha256": _array_sha256(
            initial["block_frame_counts"]
        ),
    }
    for key, value in initial_hashes.items():
        if manifest.get(key) != value:
            raise IRIVMError(f"common initial data differ from manifest hash {key}")
    controller = _new_controller(
        initial["full_path_unwrapped"],
        initial["M"],
        initial["grad_F"],
        initial["grad_F_block_means"],
        controls,
        initial["M_block_means"],
        bool(manifest.get("single_block_exact_after_initial", False)),
    )
    for row in rows[1:]:
        controller = _replay_one_state(controller, row, controls)
    checkpoint_path = run_directory / "controller_checkpoint.npz"
    if checkpoint_path.is_file():
        checkpoint = _controller_from_checkpoint(
            _load_npz_record(checkpoint_path), controls
        )
        if int(checkpoint["state_index"]) == int(controller["state_index"]) and not _controllers_equal(
            checkpoint, controller
        ):
            raise IRIVMError("checkpoint differs from replayed public trajectory")
    result: Dict[str, Any] = {
        "state_index": indices,
        "nof_mean_force_incremental": nof,
        "state_kind": np.asarray([str(row["state_kind"].item()) for row in rows]),
        "full_path_unwrapped": np.stack([row["full_path_unwrapped"] for row in rows]),
        "M": np.stack([row["M"] for row in rows]),
        "grad_F": np.stack([row["grad_F"] for row in rows]),
        # Exact-center HOLD pooling makes these ledgers genuinely ragged across
        # public states.  Preserve each state array rather than padding it or
        # silently discarding the retained-frame weights.
        "grad_F_block_means": tuple(
            np.asarray(row["grad_F_block_means"]).copy() for row in rows
        ),
        "M_block_means": tuple(
            np.asarray(row["M_block_means"]).copy() for row in rows
        ),
        "block_frame_counts": tuple(
            np.asarray(row["block_frame_counts"]).copy() for row in rows
        ),
        "controller_phase": np.asarray(
            [
                str(row["controller_phase"].item())
                if index == 0
                else str(row["controller_phase_after_paid_band"].item())
                for index, row in enumerate(rows)
            ]
        ),
        "selected_candidate_kind": np.asarray(
            [
                "COMMON_INITIAL"
                if index == 0
                else str(row["selected_candidate_kind"].item())
                for index, row in enumerate(rows)
            ]
        ),
        "hold_resample_selected": np.asarray(
            [False]
            + [bool(row["hold_resample_selected"]) for row in rows[1:]],
            dtype=bool,
        ),
        "one_way_takeover_latched": np.asarray(
            [False]
            + [bool(row["one_way_takeover_latched_next"]) for row in rows[1:]],
            dtype=bool,
        ),
        "joint_lbfgs_pair_count": np.asarray(
            [0]
            + [int(row["joint_lbfgs_pair_count_next"]) for row in rows[1:]],
            dtype=np.int64,
        ),
        "warm_safeguard_inverse_scale": np.asarray(
            [float(initial["warm_safeguard_inverse_scale"])]
            + [
                float(row["warm_safeguard_inverse_scale_next"])
                for row in rows[1:]
            ],
            dtype=np.float64,
        ),
        "warm_safeguard_source_state_index": np.asarray(
            [int(initial["warm_safeguard_source_state_index"])]
            + [
                int(row["warm_safeguard_source_state_index_next"])
                for row in rows[1:]
            ],
            dtype=np.int64,
        ),
    }
    for key in (
        "outer_index",
        "inner_step",
        "I_perp_G",
        "I_arc",
        "Delta_F_barrier",
        "segment_length_relative_deviation",
        "selected_reparameterization_output_residual",
        "effective_residual_rms",
        "stopped_inner",
        "inner_stop_reason",
    ):
        result[key] = np.asarray([row[key].item() for row in rows])
    return result


def _assert_checkpoint_matches_state(controller: Dict[str, Any], state: Dict[str, Array]) -> None:
    expected = {
        "state_index": np.asarray(controller["state_index"]),
        "nof_mean_force_incremental": np.asarray(controller["nof"]),
        "full_path_unwrapped": controller["current_path"],
        "M": controller["current_M"],
        "grad_F": controller["current_grad"],
        "grad_F_block_means": controller["current_grad_blocks"],
        "M_block_means": controller["current_M_blocks"],
        "block_frame_counts": controller["current_block_frame_counts"],
    }
    for key, value in expected.items():
        if key not in state or not np.array_equal(state[key], value):
            raise IRIVMError(f"checkpoint/public-state mismatch for {key}")


def run_irivm(
    *,
    oracle: object,
    controls: IRIVMControls,
    initial_path: Array,
    initial_M: Array,
    initial_grad_F: Array,
    nof_budget: int,
    run_directory: Path,
    initial_grad_F_block_means: Optional[Array] = None,
    initial_M_block_means: Optional[Array] = None,
    resume: bool = True,
    stop_after_nof: Optional[int] = None,
) -> Dict[str, object]:
    """Run or exactly resume one supplied-oracle V14 development trajectory."""

    controls = controls.validate()
    if type(nof_budget) is not int or nof_budget < N_INTERNAL or nof_budget % N_INTERNAL:
        raise ValueError("nof_budget must be a positive multiple of one 20-image band")
    if stop_after_nof is None:
        stop_after_nof = nof_budget
    if (
        type(stop_after_nof) is not int
        or stop_after_nof < 0
        or stop_after_nof > nof_budget
        or stop_after_nof % N_INTERNAL
    ):
        raise ValueError("stop_after_nof must be a multiple of 20 within the budget")
    path, M, grad = _validate_path_metric_gradient(initial_path, initial_M, initial_grad_F)
    grad_blocks = _validate_gradient_blocks(
        grad[:, None, :]
        if initial_grad_F_block_means is None
        else initial_grad_F_block_means
    )
    if grad_blocks.shape[1] != 1:
        raise ValueError(
            "historical state 0 must contain exactly one aggregate block with SE unavailable"
        )
    if initial_M_block_means is None:
        M_blocks = M[:, None, :, :]
    else:
        M_blocks = np.asarray(initial_M_block_means, dtype=np.float64)
    if (
        M_blocks.ndim != 4
        or M_blocks.shape[0] != N_INTERNAL
        or M_blocks.shape[1] != grad_blocks.shape[1]
        or M_blocks.shape[2:] != (DIMENSION, DIMENSION)
        or np.any(~np.isfinite(M_blocks))
    ):
        raise ValueError("initial M block means must match initial gradient blocks")

    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = run_directory / "run_manifest.json"
    checkpoint_path = run_directory / "controller_checkpoint.npz"
    requested_manifest = _manifest(
        oracle, controls, nof_budget, path, M, grad, grad_blocks, M_blocks
    )
    single_block_exact_after_initial = bool(
        requested_manifest["single_block_exact_after_initial"]
    )
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != requested_manifest:
            raise IRIVMError("existing V14 run manifest differs from request")
    else:
        _atomic_save_json(manifest_path, requested_manifest)

    completion_path = run_directory / "completion.json"
    if resume and completion_path.is_file():
        prior_completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if (
            prior_completion.get("phase") == "TERMINAL"
            and prior_completion.get("algorithmic_zero_nof_stop") is not None
        ):
            load_public_trajectory(run_directory)
            return prior_completion

    state_paths = sorted((run_directory / "states").glob("state_*.npz"))
    if resume and checkpoint_path.exists():
        if not state_paths:
            raise IRIVMError("checkpoint exists without a public state")
        controller = _controller_from_checkpoint(
            _load_npz_record(checkpoint_path), controls
        )
        latest = _load_npz_record(state_paths[-1])
        checkpoint_index = int(controller["state_index"])
        latest_index = int(latest["state_index"])
        if latest_index == checkpoint_index:
            _assert_checkpoint_matches_state(controller, latest)
        elif latest_index == checkpoint_index + 1:
            controller = _replay_one_state(controller, latest, controls)
            _save_checkpoint(run_directory, controller)
        else:
            raise IRIVMError("checkpoint/public-state indices differ by more than one")
        load_public_trajectory(run_directory)
    elif resume and state_paths:
        if len(state_paths) != 1:
            raise IRIVMError("multiple public states exist without checkpoint")
        initial = _load_npz_record(state_paths[0])
        if not (
            int(initial["state_index"]) == 0
            and int(initial["nof_mean_force_incremental"]) == 0
            and np.array_equal(initial["full_path_unwrapped"], path)
            and np.array_equal(initial["M"], M)
            and np.array_equal(initial["grad_F"], grad)
            and np.array_equal(initial["grad_F_block_means"], grad_blocks)
        ):
            raise IRIVMError("orphan common initial differs from requested input")
        controller = _new_controller(
            path,
            M,
            grad,
            grad_blocks,
            controls,
            M_blocks,
            single_block_exact_after_initial,
        )
        _save_checkpoint(run_directory, controller)
        load_public_trajectory(run_directory)
    else:
        if checkpoint_path.exists() or state_paths:
            raise IRIVMError("V14 run directory is nonempty and resume=False")
        controller = _new_controller(
            path,
            M,
            grad,
            grad_blocks,
            controls,
            M_blocks,
            single_block_exact_after_initial,
        )
        _save_initial_state(run_directory, controller, M_blocks)
        _save_checkpoint(run_directory, controller)

    if int(controller["nof"]) > int(stop_after_nof):
        raise IRIVMError("requested stop lies behind durable V14 checkpoint")
    algorithmic_stop: Optional[Dict[str, object]] = None
    start_time = time.perf_counter()
    while int(controller["nof"]) + N_INTERNAL <= int(stop_after_nof):
        prepared, zero_stop = _prepare_update(controller, controls)
        if prepared is None:
            algorithmic_stop = zero_stop
            _atomic_save_json(
                run_directory / "status.json",
                {
                    "implementation_id": IMPLEMENTATION_ID,
                    "development_only": True,
                    "phase": "TERMINAL",
                    "state_index": int(controller["state_index"]),
                    "nof_mean_force_incremental": int(controller["nof"]),
                    "algorithmic_zero_nof_stop": zero_stop,
                    "updated_unix": time.time(),
                },
            )
            break
        if not (
            np.array_equal(prepared.selected_path[0], path[0])
            and np.array_equal(prepared.selected_path[-1], path[-1])
        ):
            raise IRIVMError("fixed endpoints changed in V14 proposal")
        next_state_index = int(controller["state_index"]) + 1
        prepared_phase = (
            "RESAMPLE_PREPARED" if prepared.hold_selected else "MOVE_PREPARED"
        )
        _atomic_save_json(
            run_directory / "status.json",
            {
                "implementation_id": IMPLEMENTATION_ID,
                "development_only": True,
                "phase": prepared_phase,
                "state_index": int(controller["state_index"]),
                "nof_mean_force_incremental": int(controller["nof"]),
                "proposal_sha256": prepared.proposal_sha256,
                "selected_candidate_kind": prepared.selected_label,
                "updated_unix": time.time(),
            },
        )
        try:
            band = oracle.evaluate_band(
                prepared.selected_path,
                band_id=next_state_index,
                band_directory=run_directory / "bands" / f"band_{next_state_index:06d}",
            )
        except BaseException as error:
            _atomic_save_json(
                run_directory / "status.json",
                {
                    "implementation_id": IMPLEMENTATION_ID,
                    "development_only": True,
                    "phase": "FAILED",
                    "state_index": int(controller["state_index"]),
                    "nof_mean_force_incremental": int(controller["nof"]),
                    "proposal_sha256": prepared.proposal_sha256,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "updated_unix": time.time(),
                },
            )
            raise
        try:
            _validate_complete_band(band, prepared.selected_path)
            next_controller, advance = _advance_controller(
                controller,
                prepared,
                M=band.M,
                grad=band.grad_F,
                grad_blocks=band.grad_F_block_means,
                M_blocks=band.M_block_means,
                frame_counts=band.frame_counts,
                discarded_frame_counts=band.discarded_frame_counts,
                controls=controls,
            )
            metrics = compute_article_metrics(
                next_controller["current_path"],
                next_controller["current_M"],
                next_controller["current_grad"],
            )
            _save_public_state(
                run_directory,
                controller,
                next_controller,
                prepared,
                band,
                advance,
                metrics,
            )
        except BaseException as error:
            _atomic_save_json(
                run_directory / "status.json",
                {
                    "implementation_id": IMPLEMENTATION_ID,
                    "development_only": True,
                    "phase": "FAILED",
                    "state_index": int(controller["state_index"]),
                    "nof_mean_force_incremental": int(controller["nof"]),
                    "proposal_sha256": prepared.proposal_sha256,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "updated_unix": time.time(),
                },
            )
            raise
        controller = next_controller
        _save_checkpoint(run_directory, controller)
        _atomic_save_json(
            run_directory / "status.json",
            {
                "implementation_id": IMPLEMENTATION_ID,
                "development_only": True,
                "phase": (
                    "READY"
                    if bool(controller["outer_anchor_bounds_ready"])
                    else "ANCHORING"
                ),
                "state_index": int(controller["state_index"]),
                "nof_mean_force_incremental": int(controller["nof"]),
                "nof_budget": int(nof_budget),
                "outer_index_next": int(controller["outer_index"]),
                "inner_step_next": int(controller["inner_step"]),
                "joint_lbfgs_pair_count_next": int(controller["lbfgs_count"]),
                "one_way_takeover_latched_next": bool(controller["lbfgs_latched"]),
                "selected_candidate_kind": prepared.selected_label,
                "I_perp_G": float(metrics["I_perp_G"]),
                "I_arc": float(metrics["I_arc"]),
                "Delta_F_barrier": float(metrics["Delta_F_barrier"]),
                "effective_residual_ratio": (
                    float(advance["residual_ratio"])
                    if np.isfinite(advance["residual_ratio"])
                    else None
                ),
                "inner_stop_reason": advance["inner_stop_reason"],
                "elapsed_seconds_this_invocation": time.perf_counter() - start_time,
                "updated_unix": time.time(),
            },
        )

    trajectory = load_public_trajectory(run_directory)
    final_status = {
        "phase": "TERMINAL" if algorithmic_stop is not None else (
            "READY" if bool(controller["outer_anchor_bounds_ready"]) else "ANCHORING"
        ),
        "completed": bool(int(controller["nof"]) == nof_budget),
        "paused_at_requested_stop": bool(
            int(controller["nof"]) == stop_after_nof
            and int(controller["nof"]) < nof_budget
            and algorithmic_stop is None
        ),
        "algorithmic_zero_nof_stop": algorithmic_stop,
        "nof_mean_force_incremental": int(controller["nof"]),
        "nof_budget": int(nof_budget),
        "state_count_including_common_initial": int(trajectory["state_index"].size),
        "outer_index_next": int(controller["outer_index"]),
        "inner_step_next": int(controller["inner_step"]),
        "implementation_id": IMPLEMENTATION_ID,
        "development_only": True,
        "run_directory": str(run_directory.resolve()),
    }
    _atomic_save_json(completion_path, final_status)
    return final_status


__all__ = [
    "Array",
    "CHECKPOINT_PHASE",
    "DEVELOPMENT_ONLY",
    "IMPLEMENTATION_ID",
    "IRIVMControls",
    "IRIVMControlsV14",
    "IRIVMError",
    "JOINT_HISTORY_SIZE",
    "N_INTERNAL",
    "STATE_KIND",
    "joint_lbfgs_direction",
    "load_public_trajectory",
    "noise_shrink_residual",
    "run_irivm",
]
