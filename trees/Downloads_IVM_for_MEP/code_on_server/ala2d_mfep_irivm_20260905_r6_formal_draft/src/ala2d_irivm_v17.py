"""V17 core for the candidate-agnostic noisy Riemannian Ala2D formal draft.

This module never instantiates ``RestrainedMDOracle``.  A separately sealed
formal-campaign wrapper must supply an authorized object with the same
``evaluate_band`` protocol as :class:`ala2d_oracles.OracleBand`.  The r6
registration is deliberately a non-executable draft, so these bytes alone
grant no restrained-MD authority.

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
suffix.  The 66D one-way selector is global: L-BFGS takes over only after its
score advantage over H0 is resolved above the fixed block-noise margin.  Until
then, an admissible H0 is used without buying a band only to arbitrate the
candidates.  After takeover, H0/cold recovery is permanently forbidden.  A
rebuilt frame with fewer than two valid pairs uses only the persisted bounded
warm scalar safeguard.

The relative-residual parameter ``r`` is a one-sided confident stop.  A frozen
outer frame is rebuilt only when ``U_current <= r*L_anchor`` or after
``m_max`` movements.  Failure to prove the stop is a continuation decision,
not an exact-center resampling decision.

Both genuine candidates are passed through the unchanged article spring
reparameterizer before selection.  A single anchor-aware scale first keeps
the whole raw band inside the frozen trust balls, and fixed one-half
backtracking then requires the actual article-reparameterized band to remain
inside them as well.  Exhaustion of the currently required candidate permits
one zero-NoF outer-frame refresh at an unchanged paid path; the next paid
state, or a rederived terminal record, carries that event.  Only the selected
complete band is sent to the supplied oracle.  Its *actual reparameterized
coordinates* are then mapped into frozen normal coordinates, so the
reparameterization mapping is included in residuals and learned secants.

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
    REPARAMETERIZATION_ZERO_SEGMENT_THRESHOLD,
    REPARAMETERIZER_ID,
    reparameterize_article_neb,
)
from ala2d_oracles import OracleBand
from riemannian_torus_geometry import (
    arclength_residual,
    build_metric_geometry,
    compute_article_metrics,
    periodic_difference,
    unwrap_path,
    wrap_periodic,
)


Array = np.ndarray
IMPLEMENTATION_ID = (
    "ala2d-riemannian-v17-formal-independent-warm-validation-restart"
)
DEVELOPMENT_ONLY = False
STATE_KIND = "reparameterized_frozen_inner_trial"
HOLD_STATE_KIND = "exact_center_hold_resample"
WARM_VALIDATION_STATE_KIND = "independent_exact_center_warm_validation"
CHECKPOINT_PHASE = "v17_formal_paid_state_ready"
N_INTERNAL = 20
N_PATH = N_INTERNAL + 2
DIMENSION = 2
JOINT_HISTORY_SIZE = 7
MAX_PAID_OBSERVATIONS = 16
ANCHOR_CAP_BRACKET_ITERATIONS = 64
POST_R_BACKTRACK_LIMIT = 32
WARM_VALIDATION_PENDING_NONE = "none"
WARM_VALIDATION_PENDING_COMMIT = "validated_positive_pending_commit"
WARM_VALIDATION_CLASSIFICATIONS = frozenset(
    {"not_applicable", "validated_positive", "confirmed_bad", "unresolved"}
)
ANCHOR_CAP_STRUCTURAL_ERRORS = frozenset(
    {
        "current_paid_path_outside_frozen_anchor_trust_radii",
        "anchor_aware_common_cap_nonfinite_quadratic",
        "anchor_aware_common_cap_invalid_positive_root",
        "anchor_aware_common_cap_invalid_theoretical_scale",
        "anchor_aware_common_cap_zero_scale_not_feasible",
    }
)


class IRIVMError(RuntimeError):
    """Fail-closed V17 algorithm, oracle, or persistence error."""


@dataclass(frozen=True)
class IRIVMControls:
    """Three candidate controls plus fixed, preregistrable safeguards.

    Only ``alpha``, ``r`` and ``m_max`` may come from the separately sealed
    development selection.  The remaining fields make the formal hypothesis
    executable and auditable; validation fixes them so they cannot silently
    become extra tuned knobs.
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
            raise ValueError("V17 requires 0 < r < 0.8")
        if type(self.m_max) is not int or self.m_max <= 3:
            raise ValueError("V17 requires integer m_max > 3")
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
                    raise ValueError(f"V17 formal protocol fixes {name}={expected}")
            elif not np.isfinite(observed) or float(observed) != float(expected):
                raise ValueError(f"V17 formal protocol fixes {name}={expected}")
        return self


# Explicit V17 name for callers that prefer versioned imports.
IRIVMControlsV17 = IRIVMControls


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
    # Geometry/construction validity only; never a merit or confidence test.
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
    proposal_effective_linear_term: float
    proposal_quadratic_penalty: float
    proposal_effective_proximal_value: float
    anchor_cap_theoretical_scale: float
    anchor_cap_binary_bracket_used: bool
    anchor_cap_binary_bracket_iterations: int
    anchor_cap_source_distances: Array
    anchor_cap_raw_path_distances: Array
    post_r_backtrack_count: int
    post_r_backtrack_factor: float
    total_direction_scale: float
    post_r_attempt_count: int
    post_r_attempt_scales: Array
    post_r_attempt_status: Array
    post_r_attempt_raw_path_sha256: Array
    post_r_attempt_reparameterization_audit_sha256: Array
    post_r_attempt_max_anchor_distance: Array
    trust_globalization_exhausted: bool


@dataclass(frozen=True)
class AnchorAwareCap:
    valid: bool
    error: str
    capped_direction: Array
    scale: float
    truncated: bool
    theoretical_scale: float
    binary_bracket_used: bool
    binary_bracket_iterations: int
    source_distances: Array
    raw_path_distances: Array


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
    h0_raw_improvement: Dict[str, float]
    lbfgs_raw_improvement: Dict[str, float]
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
    lbfgs_raw_improvement_advantage: float
    raw_improvement_difference_standard_error: float
    lbfgs_raw_advantage_resolution_threshold: float
    lbfgs_raw_advantage_resolved_positive: bool
    initial_no_se_special_case: bool
    selected_raw_improvement_standard_error: float
    selected_raw_improvement_lower_bound: float
    warm_validation_selected: bool
    frozen_warm_commit: bool
    frozen_warm_candidate: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class PreparationOutcome:
    """One deterministic preparation, including an optional zero-NoF refresh.

    ``proposal_controller`` is the exact controller whose frozen frame produced
    ``prepared``.  It may differ from the durable paid-state checkpoint only by
    one deterministic trust refresh.  That same-index refresh is never
    checkpointed on its own; the next paid state records ``zero_nof_event`` and
    replay derives it again from the preceding durable controller.
    """

    prepared: Optional[PreparedUpdate]
    proposal_controller: Dict[str, Any]
    terminal: Dict[str, object]
    zero_nof_event: Dict[str, object]

    def __iter__(self):
        """Preserve the former private two-value test helper interface."""

        yield self.prepared
        yield self.terminal


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
        raise TypeError("object arrays are not admissible in a V17 digest")
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and not np.little_endian
    ):
        array = array.byteswap().newbyteorder("<")
    array = np.ascontiguousarray(array)
    header = _canonical_json(
        {
            "schema": "BinaryArray/ala2d-irivm-v17-formal",
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
    """Apply the fixed V17 smooth residual shrinkage component by component."""

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


def _anchor_trust_distances(path: Array, frozen_path: Array) -> Array:
    value = np.asarray(path, dtype=np.float64)
    anchor = np.asarray(frozen_path, dtype=np.float64)
    if value.shape != (N_PATH, DIMENSION) or anchor.shape != value.shape:
        raise IRIVMError("anchor trust path has the wrong shape")
    if np.any(~np.isfinite(value)) or np.any(~np.isfinite(anchor)):
        raise IRIVMError("anchor trust path is non-finite")
    displacement = periodic_difference(value[1:-1], anchor[1:-1])
    distances = np.hypot(displacement[:, 0], displacement[:, 1])
    if distances.shape != (N_INTERNAL,) or np.any(~np.isfinite(distances)):
        raise IRIVMError("anchor trust distances are invalid")
    return distances


def _raw_path_on_direction_ray(
    current_path: Array, basis: Array, direction: Array, scale: float
) -> Array:
    if not np.isfinite(scale) or scale < 0.0:
        raise IRIVMError("direction-ray scale is invalid")
    current = np.asarray(current_path, dtype=np.float64)
    normal = np.asarray(basis, dtype=np.float64)
    reduced = np.asarray(direction, dtype=np.float64)
    if (
        current.shape != (N_PATH, DIMENSION)
        or normal.shape != (N_INTERNAL, DIMENSION)
        or reduced.shape != (N_INTERNAL,)
        or np.any(~np.isfinite(current))
        or np.any(~np.isfinite(normal))
        or np.any(~np.isfinite(reduced))
    ):
        raise IRIVMError("direction-ray input is invalid")
    raw = current.copy()
    raw[1:-1] += normal * (reduced * float(scale))[:, None]
    if np.any(~np.isfinite(raw)):
        raise IRIVMError("direction-ray path is non-finite")
    raw = unwrap_path(raw)
    if not np.array_equal(raw[[0, -1]], current[[0, -1]]):
        raise IRIVMError("direction-ray construction moved an endpoint")
    return raw


def _whole_band_common_cap(
    direction: Array,
    basis: Array,
    radii: Array,
    *,
    current_path: Array,
    frozen_path: Array,
    secant_norm_floor: float,
) -> AnchorAwareCap:
    """Apply one anchor-aware ray/ball scale to all twenty images.

    V15 limited only the new direction length.  V17 instead accounts for the
    displacement already accumulated from the frozen outer anchor.  For image
    ``i`` it solves ``||d_i + lambda v_i|| <= rho_i`` and uses one common
    ``lambda``.  The analytic binary64 value is always checked on the actual
    raw path.  If roundoff makes it infeasible, a fixed 64-step bracket returns
    the largest explicitly verified lower endpoint.
    """

    reduced = np.asarray(direction, dtype=np.float64)
    normal = np.asarray(basis, dtype=np.float64)
    radius = np.asarray(radii, dtype=np.float64)
    if reduced.shape != (N_INTERNAL,) or np.any(~np.isfinite(reduced)):
        raise IRIVMError("joint direction is invalid")
    if normal.shape != (N_INTERNAL, DIMENSION) or np.any(~np.isfinite(normal)):
        raise IRIVMError("frozen normal basis is invalid for the common cap")
    if radius.shape != (N_INTERNAL,) or np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise IRIVMError("anchor trust radii are invalid for the common cap")
    if not np.isfinite(secant_norm_floor) or secant_norm_floor <= 0.0:
        raise IRIVMError("secant norm floor is invalid for the common cap")

    current = np.asarray(current_path, dtype=np.float64)
    anchor = np.asarray(frozen_path, dtype=np.float64)
    displacement = periodic_difference(current[1:-1], anchor[1:-1])
    source_distances = _anchor_trust_distances(current, anchor)
    if np.any(source_distances > radius):
        return AnchorAwareCap(
            valid=False,
            error="current_paid_path_outside_frozen_anchor_trust_radii",
            capped_direction=np.full(N_INTERNAL, np.nan),
            scale=0.0,
            truncated=False,
            theoretical_scale=math.nan,
            binary_bracket_used=False,
            binary_bracket_iterations=0,
            source_distances=source_distances,
            raw_path_distances=np.full(N_INTERNAL, np.nan),
        )

    physical = normal * reduced[:, None]
    roots = np.full(N_INTERNAL, math.inf, dtype=np.float64)
    for image in range(N_INTERNAL):
        velocity = physical[image]
        a = float(np.dot(velocity, velocity))
        if not np.isfinite(a):
            return AnchorAwareCap(
                False,
                "anchor_aware_common_cap_nonfinite_quadratic",
                np.full(N_INTERNAL, np.nan),
                0.0,
                False,
                math.nan,
                False,
                0,
                source_distances,
                np.full(N_INTERNAL, np.nan),
            )
        if a <= np.finfo(np.float64).tiny:
            continue
        b = float(np.dot(displacement[image], velocity))
        c = float(np.dot(displacement[image], displacement[image]) - radius[image] ** 2)
        discriminant = b * b - a * c
        if not (np.isfinite(b) and np.isfinite(c) and np.isfinite(discriminant)):
            return AnchorAwareCap(
                False,
                "anchor_aware_common_cap_nonfinite_quadratic",
                np.full(N_INTERNAL, np.nan),
                0.0,
                False,
                math.nan,
                False,
                0,
                source_distances,
                np.full(N_INTERNAL, np.nan),
            )
        root = (-b + math.sqrt(max(0.0, discriminant))) / a
        if not np.isfinite(root) or root < 0.0:
            return AnchorAwareCap(
                False,
                "anchor_aware_common_cap_invalid_positive_root",
                np.full(N_INTERNAL, np.nan),
                0.0,
                False,
                math.nan,
                False,
                0,
                source_distances,
                np.full(N_INTERNAL, np.nan),
            )
        roots[image] = root

    theoretical = float(min(1.0, float(np.min(roots))))
    if not np.isfinite(theoretical) or theoretical < 0.0:
        return AnchorAwareCap(
            False,
            "anchor_aware_common_cap_invalid_theoretical_scale",
            np.full(N_INTERNAL, np.nan),
            0.0,
            False,
            theoretical,
            False,
            0,
            source_distances,
            np.full(N_INTERNAL, np.nan),
        )

    def checked(scale: float) -> Tuple[bool, Array]:
        try:
            trial = _raw_path_on_direction_ray(current, normal, reduced, scale)
            distances = _anchor_trust_distances(trial, anchor)
        except (ArithmeticError, FloatingPointError, ValueError, IRIVMError):
            return False, np.full(N_INTERNAL, np.nan)
        return bool(np.all(distances <= radius)), distances

    feasible, distances = checked(theoretical)
    bracket_used = False
    bracket_iterations = 0
    scale = theoretical
    if not feasible:
        bracket_used = True
        lo = 0.0
        hi = theoretical
        lo_feasible, lo_distances = checked(lo)
        if not lo_feasible:
            return AnchorAwareCap(
                False,
                "anchor_aware_common_cap_zero_scale_not_feasible",
                np.full(N_INTERNAL, np.nan),
                0.0,
                False,
                theoretical,
                True,
                0,
                source_distances,
                lo_distances,
            )
        for _ in range(ANCHOR_CAP_BRACKET_ITERATIONS):
            midpoint = float(lo + (hi - lo) * 0.5)
            midpoint_feasible, midpoint_distances = checked(midpoint)
            if midpoint_feasible:
                lo = midpoint
                lo_distances = midpoint_distances
            else:
                hi = midpoint
            bracket_iterations += 1
        scale = lo
        distances = lo_distances

    capped = reduced * scale
    if (
        not np.isfinite(scale)
        or scale <= 0.0
        or np.any(~np.isfinite(capped))
        or float(np.linalg.norm(capped)) <= float(secant_norm_floor)
    ):
        return AnchorAwareCap(
            False,
            "anchor_aware_common_cap_exhausted_below_secant_norm_floor",
            capped,
            scale,
            bool(scale < 1.0),
            theoretical,
            bracket_used,
            bracket_iterations,
            source_distances,
            distances,
        )
    final_raw = _raw_path_on_direction_ray(current, normal, reduced, scale)
    final_distances = _anchor_trust_distances(final_raw, anchor)
    if np.any(final_distances > radius):
        raise IRIVMError("verified anchor-aware common cap became infeasible")
    return AnchorAwareCap(
        True,
        "none",
        capped,
        scale,
        bool(scale < 1.0),
        theoretical,
        bracket_used,
        bracket_iterations,
        source_distances,
        final_distances,
    )


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


def _proximal_value_terms(
    residual_vector: Array, actual_normal_step: Array, alpha: float
) -> Tuple[float, float, float]:
    """Return linear, quadratic, and net values for a named residual vector.

    With ``P_eff`` this is a noninferential direction diagnostic.  With
    ``P_raw`` it is the sole improvement target used by decisions.  Keeping
    the residual argument explicit prevents a raw standard error from ever
    being attached to an effective-centered value.
    """

    residual = np.asarray(residual_vector, dtype=np.float64)
    step = np.asarray(actual_normal_step, dtype=np.float64)
    if residual.shape != (N_INTERNAL,) or step.shape != residual.shape:
        raise IRIVMError("proximal-value residual/step has the wrong shape")
    if np.any(~np.isfinite(residual)) or np.any(~np.isfinite(step)):
        raise IRIVMError("proximal-value residual/step is non-finite")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise IRIVMError("proximal-value alpha must be positive and finite")
    linear = -float(np.dot(residual, step))
    quadratic = float(np.dot(step, step)) / (2.0 * float(alpha))
    score = linear - quadratic
    return linear, quadratic, score


def _candidate_raw_improvement_evidence(
    candidate: Candidate,
    statistics: ResidualStatistics,
    controls: IRIVMControls,
) -> Dict[str, float]:
    """Return the only improvement statistic allowed to make decisions.

    ``effective_residual`` may generate a direction and its diagnostic proposal
    score, but it is never paired with the raw block standard error.  All
    selection, HOLD, validation, and takeover bounds use ``P_raw`` here.
    """

    if candidate.reparameterization is None or not np.all(
        np.isfinite(candidate.actual_normal_step)
    ):
        return {
            "linear": math.nan,
            "quadratic": math.nan,
            "score": math.nan,
            "standard_error": math.nan,
            "lower_bound": math.nan,
            "upper_bound": math.nan,
        }
    linear, quadratic, score = _proximal_value_terms(
        statistics.raw_residual,
        candidate.actual_normal_step,
        controls.alpha,
    )
    if not statistics.bounds_available:
        score_se = math.nan
        lower = math.nan
        upper = math.nan
    else:
        score_se = float(
            np.linalg.norm(
                statistics.standard_error * candidate.actual_normal_step
            )
        )
        _, lower, upper = _classify_raw_improvement_interval(
            score, score_se, controls.residual_bound_kappa
        )
    return {
        "linear": linear,
        "quadratic": quadratic,
        "score": score,
        "standard_error": score_se,
        "lower_bound": lower,
        "upper_bound": upper,
    }


def _validate_article_reparameterization_attempt(
    raw_path: Array,
    result: ArticleReparameterizationResult,
    controls: IRIVMControls,
) -> None:
    """Strictly validate one zero-NoF article-map result and its certificate."""

    if not isinstance(result, ArticleReparameterizationResult):
        raise IRIVMError("article reparameterization returned the wrong result type")
    source = np.asarray(raw_path, dtype=np.float64)
    output = np.asarray(result.path, dtype=np.float64)
    if output.shape != (N_PATH, DIMENSION) or np.any(~np.isfinite(output)):
        raise IRIVMError("article reparameterization output path is invalid")
    if not np.array_equal(output[[0, -1]], source[[0, -1]]):
        raise IRIVMError("article reparameterization output moved a fixed endpoint")
    if type(result.iteration_count) is not int or not (
        1 <= result.iteration_count <= controls.reparameterization_max_iterations
    ):
        raise IRIVMError("article reparameterization iteration count is invalid")
    diagnostics = (
        float(result.stopping_pre_update_spring_residual),
        float(result.output_spring_residual),
        float(result.maximum_neighbor_length_difference),
        float(result.actual_chord_relative_deviation),
        float(result.actual_I_arc),
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in diagnostics):
        raise IRIVMError("article reparameterization certificate is non-finite")
    if not (
        diagnostics[0] < controls.reparameterization_tolerance
        and diagnostics[1] < controls.reparameterization_tolerance
    ):
        raise IRIVMError("article reparameterization certificate missed tolerance")
    lengths = np.linalg.norm(
        periodic_difference(output[1:], output[:-1]), axis=1
    )
    recorded_lengths = np.asarray(result.segment_lengths, dtype=np.float64)
    if (
        recorded_lengths.shape != (N_PATH - 1,)
        or np.any(~np.isfinite(recorded_lengths))
        or np.any(
            recorded_lengths <= REPARAMETERIZATION_ZERO_SEGMENT_THRESHOLD
        )
        or not np.array_equal(recorded_lengths, lengths)
    ):
        raise IRIVMError("article reparameterization segment ledger is invalid")
    expected_neighbor = float(np.max(np.abs(np.diff(lengths))))
    expected_chord = float(
        np.max(np.abs(lengths / float(np.mean(lengths)) - 1.0))
    )
    expected_i_arc = float(arclength_residual(output))
    if not (
        diagnostics[2] == expected_neighbor
        and diagnostics[3] == expected_chord
        and diagnostics[4] == expected_i_arc
    ):
        raise IRIVMError("article reparameterization diagnostics are inconsistent")
    expected_audit = hashlib.sha256(
        _canonical_json(
            result.audit_payload(
                source_path=source,
                k_s=controls.reparameterization_k_s,
                dtau=controls.reparameterization_dtau,
                tolerance=controls.reparameterization_tolerance,
                max_iterations=controls.reparameterization_max_iterations,
            )
        )
    ).hexdigest()
    if result.audit_sha256 != expected_audit:
        raise IRIVMError("article reparameterization audit digest is invalid")


def _invalid_candidate(
    label: str,
    error: str,
    direction: Optional[Array] = None,
    *,
    anchor_cap: Optional[AnchorAwareCap] = None,
    trust_globalization_exhausted: bool = False,
    post_r_attempt_scales: Optional[Array] = None,
    post_r_attempt_status: Optional[Array] = None,
    post_r_attempt_raw_path_sha256: Optional[Array] = None,
    post_r_attempt_reparameterization_audit_sha256: Optional[Array] = None,
    post_r_attempt_max_anchor_distance: Optional[Array] = None,
) -> Candidate:
    vector = (
        np.full(N_INTERNAL, np.nan, dtype=np.float64)
        if direction is None
        else np.asarray(direction, dtype=np.float64).copy()
    )
    cap = anchor_cap
    attempt_scales = np.asarray(
        [] if post_r_attempt_scales is None else post_r_attempt_scales,
        dtype=np.float64,
    )
    attempt_status = np.asarray(
        [] if post_r_attempt_status is None else post_r_attempt_status,
        dtype="U256",
    )
    attempt_raw_hashes = np.asarray(
        []
        if post_r_attempt_raw_path_sha256 is None
        else post_r_attempt_raw_path_sha256,
        dtype="U64",
    )
    attempt_audit_hashes = np.asarray(
        []
        if post_r_attempt_reparameterization_audit_sha256 is None
        else post_r_attempt_reparameterization_audit_sha256,
        dtype="U64",
    )
    attempt_maximum_distances = np.asarray(
        []
        if post_r_attempt_max_anchor_distance is None
        else post_r_attempt_max_anchor_distance,
        dtype=np.float64,
    )
    attempt_count = int(attempt_scales.size)
    if not (
        attempt_status.size
        == attempt_raw_hashes.size
        == attempt_audit_hashes.size
        == attempt_maximum_distances.size
        == attempt_count
    ):
        raise IRIVMError("invalid candidate trust-attempt ledgers differ in length")
    return Candidate(
        label=label,
        valid=False,
        error=error,
        raw_direction=vector,
        capped_direction=(
            np.full(N_INTERNAL, np.nan, dtype=np.float64)
            if cap is None
            else np.asarray(cap.capped_direction, dtype=np.float64).copy()
        ),
        common_cap_scale=0.0 if cap is None else float(cap.scale),
        common_cap_truncated=False if cap is None else bool(cap.truncated),
        raw_path=np.full((N_PATH, DIMENSION), np.nan, dtype=np.float64),
        reparameterization=None,
        actual_normal_coordinate=np.full(N_INTERNAL, np.nan, dtype=np.float64),
        actual_normal_step=np.full(N_INTERNAL, np.nan, dtype=np.float64),
        proposal_effective_linear_term=0.0,
        proposal_quadratic_penalty=0.0,
        proposal_effective_proximal_value=0.0,
        anchor_cap_theoretical_scale=(
            math.nan if cap is None else float(cap.theoretical_scale)
        ),
        anchor_cap_binary_bracket_used=(
            False if cap is None else bool(cap.binary_bracket_used)
        ),
        anchor_cap_binary_bracket_iterations=(
            0 if cap is None else int(cap.binary_bracket_iterations)
        ),
        anchor_cap_source_distances=(
            np.full(N_INTERNAL, np.nan)
            if cap is None
            else np.asarray(cap.source_distances, dtype=np.float64).copy()
        ),
        anchor_cap_raw_path_distances=(
            np.full(N_INTERNAL, np.nan)
            if cap is None
            else np.asarray(cap.raw_path_distances, dtype=np.float64).copy()
        ),
        post_r_backtrack_count=-1,
        post_r_backtrack_factor=math.nan,
        total_direction_scale=math.nan,
        post_r_attempt_count=attempt_count,
        post_r_attempt_scales=attempt_scales,
        post_r_attempt_status=attempt_status,
        post_r_attempt_raw_path_sha256=attempt_raw_hashes,
        post_r_attempt_reparameterization_audit_sha256=attempt_audit_hashes,
        post_r_attempt_max_anchor_distance=attempt_maximum_distances,
        trust_globalization_exhausted=bool(trust_globalization_exhausted),
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
    """Anchor-cap, article-reparameterize, and screen one zero-NoF candidate.

    The complete-band direction is first capped on the ray from the *current*
    paid path inside the frozen-anchor product of balls.  Article
    reparameterization can move images away from that raw ray, so V17 then
    retries the same direction at the fixed scales ``1, 1/2, ..., 1/2**32``.
    Each retry reconstructs and strictly verifies both raw and post-map paths;
    no projection and no tolerance-expanded radius are permitted.
    """

    try:
        raw_direction = np.asarray(direction, dtype=np.float64)
        if raw_direction.shape != (N_INTERNAL,) or np.any(~np.isfinite(raw_direction)):
            raise IRIVMError("raw direction is non-finite or has the wrong shape")
        raw_decrease = -float(np.dot(statistics.effective_residual, raw_direction))
        if not raw_decrease > controls.kkt_floor:
            raise IRIVMError("raw direction is not effective-residual descent")
        cap = _whole_band_common_cap(
            raw_direction,
            controller["frozen_basis"],
            radii,
            current_path=controller["current_path"],
            frozen_path=controller["frozen_path"],
            secant_norm_floor=controls.secant_norm_floor,
        )
        if not cap.valid:
            return _invalid_candidate(
                label,
                cap.error,
                raw_direction,
                anchor_cap=cap,
                trust_globalization_exhausted=bool(
                    cap.error
                    == "anchor_aware_common_cap_exhausted_below_secant_norm_floor"
                ),
            )

        attempt_scales: List[float] = []
        attempt_status: List[str] = []
        attempt_raw_hashes: List[str] = []
        attempt_audit_hashes: List[str] = []
        attempt_maximum_distances: List[float] = []
        accepted_backtrack = -1
        accepted_raw_path: Optional[Array] = None
        accepted_reparameterization: Optional[ArticleReparameterizationResult] = None
        for backtrack in range(POST_R_BACKTRACK_LIMIT + 1):
            factor = math.ldexp(1.0, -backtrack)
            total_scale = float(cap.scale * factor)
            attempted_direction = raw_direction * total_scale
            attempt_scales.append(total_scale)
            if (
                not np.isfinite(total_scale)
                or total_scale <= 0.0
                or np.any(~np.isfinite(attempted_direction))
                or float(np.linalg.norm(attempted_direction))
                <= controls.secant_norm_floor
            ):
                attempt_status.append("direction_below_secant_norm_floor")
                attempt_raw_hashes.append("none")
                attempt_audit_hashes.append("none")
                attempt_maximum_distances.append(math.nan)
                break
            try:
                raw_path = _raw_path_on_direction_ray(
                    controller["current_path"],
                    controller["frozen_basis"],
                    raw_direction,
                    total_scale,
                )
                raw_distances = _anchor_trust_distances(
                    raw_path, controller["frozen_path"]
                )
                attempt_raw_hashes.append(_array_sha256(raw_path))
                if np.any(raw_distances > np.asarray(radii, dtype=np.float64)):
                    attempt_status.append("raw_path_outside_anchor_trust")
                    attempt_audit_hashes.append("none")
                    attempt_maximum_distances.append(float(np.max(raw_distances)))
                    continue
                reparameterized = reparameterize_article_neb(
                    raw_path,
                    k_s=controls.reparameterization_k_s,
                    dtau=controls.reparameterization_dtau,
                    tolerance=controls.reparameterization_tolerance,
                    max_iterations=controls.reparameterization_max_iterations,
                )
                _validate_article_reparameterization_attempt(
                    raw_path, reparameterized, controls
                )
                post_distances = _anchor_trust_distances(
                    reparameterized.path, controller["frozen_path"]
                )
                attempt_audit_hashes.append(reparameterized.audit_sha256)
                attempt_maximum_distances.append(float(np.max(post_distances)))
                if np.any(post_distances > np.asarray(radii, dtype=np.float64)):
                    attempt_status.append("post_r_path_outside_anchor_trust")
                    continue
                attempt_status.append("accepted_inside_anchor_trust")
                accepted_backtrack = backtrack
                accepted_raw_path = raw_path
                accepted_reparameterization = reparameterized
                break
            except (
                ArticleReparameterizationError,
                ArithmeticError,
                FloatingPointError,
                ValueError,
                IRIVMError,
            ) as error:
                if len(attempt_raw_hashes) < len(attempt_scales):
                    attempt_raw_hashes.append("none")
                attempt_audit_hashes.append("none")
                attempt_maximum_distances.append(math.nan)
                attempt_status.append(
                    f"{type(error).__name__}:{error}"
                )

        attempt_arrays = {
            "post_r_attempt_scales": np.asarray(attempt_scales, dtype=np.float64),
            "post_r_attempt_status": np.asarray(attempt_status, dtype="U256"),
            "post_r_attempt_raw_path_sha256": np.asarray(
                attempt_raw_hashes, dtype="U64"
            ),
            "post_r_attempt_reparameterization_audit_sha256": np.asarray(
                attempt_audit_hashes, dtype="U64"
            ),
            "post_r_attempt_max_anchor_distance": np.asarray(
                attempt_maximum_distances, dtype=np.float64
            ),
        }
        if accepted_reparameterization is None or accepted_raw_path is None:
            return _invalid_candidate(
                label,
                "post_reparameterization_trust_globalization_exhausted",
                raw_direction,
                anchor_cap=cap,
                trust_globalization_exhausted=True,
                **attempt_arrays,
            )

        reparameterized = accepted_reparameterization
        raw_path = accepted_raw_path
        factor = math.ldexp(1.0, -accepted_backtrack)
        total_scale = float(cap.scale * factor)
        capped = raw_direction * total_scale
        actual_u = _normal_coordinate(
            reparameterized.path,
            controller["frozen_path"],
            controller["frozen_basis"],
            controller["frozen_G"],
        )
        actual_step = actual_u - statistics.normal_coordinate
        actual_physical_distance = _anchor_trust_distances(
            reparameterized.path, controller["frozen_path"]
        )
        if np.any(actual_physical_distance > np.asarray(radii)):
            raise IRIVMError(
                "accepted post-reparameterization path escaped anchor trust"
            )
        linear, quadratic, score = _proximal_value_terms(
            statistics.effective_residual, actual_step, controls.alpha
        )
        construction_valid = bool(
            np.isfinite(score)
            and float(np.linalg.norm(actual_step)) > controls.secant_norm_floor
        )
        candidate = Candidate(
            label=label,
            valid=construction_valid,
            error=(
                "none"
                if construction_valid
                else "post_reparameterization_actual_step_degenerate_or_nonfinite"
            ),
            raw_direction=raw_direction.copy(),
            capped_direction=capped,
            common_cap_scale=cap.scale,
            common_cap_truncated=cap.truncated,
            raw_path=raw_path,
            reparameterization=reparameterized,
            actual_normal_coordinate=actual_u,
            actual_normal_step=actual_step,
            proposal_effective_linear_term=linear,
            proposal_quadratic_penalty=quadratic,
            proposal_effective_proximal_value=score,
            anchor_cap_theoretical_scale=cap.theoretical_scale,
            anchor_cap_binary_bracket_used=cap.binary_bracket_used,
            anchor_cap_binary_bracket_iterations=cap.binary_bracket_iterations,
            anchor_cap_source_distances=cap.source_distances,
            anchor_cap_raw_path_distances=cap.raw_path_distances,
            post_r_backtrack_count=accepted_backtrack,
            post_r_backtrack_factor=factor,
            total_direction_scale=total_scale,
            post_r_attempt_count=len(attempt_scales),
            post_r_attempt_scales=attempt_arrays["post_r_attempt_scales"],
            post_r_attempt_status=attempt_arrays["post_r_attempt_status"],
            post_r_attempt_raw_path_sha256=attempt_arrays[
                "post_r_attempt_raw_path_sha256"
            ],
            post_r_attempt_reparameterization_audit_sha256=attempt_arrays[
                "post_r_attempt_reparameterization_audit_sha256"
            ],
            post_r_attempt_max_anchor_distance=attempt_arrays[
                "post_r_attempt_max_anchor_distance"
            ],
            trust_globalization_exhausted=False,
        )
        return candidate
    except (ArticleReparameterizationError, ArithmeticError, FloatingPointError, ValueError, IRIVMError) as error:
        return _invalid_candidate(
            label,
            f"{type(error).__name__}:{error}",
            np.asarray(direction, dtype=np.float64),
        )


def _candidate_digest(candidate: Candidate) -> Dict[str, object]:
    result: Dict[str, object] = {
        "label": candidate.label,
        "structural_valid": bool(candidate.valid),
        "error": candidate.error,
        "raw_direction": _array_sha256(candidate.raw_direction),
        "capped_direction": _array_sha256(candidate.capped_direction),
        "common_cap_scale_binary64": float(candidate.common_cap_scale).hex(),
        "anchor_cap_theoretical_scale_binary64": float(
            candidate.anchor_cap_theoretical_scale
        ).hex(),
        "anchor_cap_binary_bracket_used": bool(
            candidate.anchor_cap_binary_bracket_used
        ),
        "anchor_cap_binary_bracket_iterations": int(
            candidate.anchor_cap_binary_bracket_iterations
        ),
        "anchor_cap_source_distances": _array_sha256(
            candidate.anchor_cap_source_distances
        ),
        "anchor_cap_raw_path_distances": _array_sha256(
            candidate.anchor_cap_raw_path_distances
        ),
        "post_r_backtrack_count": int(candidate.post_r_backtrack_count),
        "post_r_backtrack_factor_binary64": float(
            candidate.post_r_backtrack_factor
        ).hex(),
        "total_direction_scale_binary64": float(
            candidate.total_direction_scale
        ).hex(),
        "post_r_attempt_count": int(candidate.post_r_attempt_count),
        "post_r_attempt_scales": _array_sha256(candidate.post_r_attempt_scales),
        "post_r_attempt_status": _array_sha256(candidate.post_r_attempt_status),
        "post_r_attempt_raw_path_sha256": _array_sha256(
            candidate.post_r_attempt_raw_path_sha256
        ),
        "post_r_attempt_reparameterization_audit_sha256": _array_sha256(
            candidate.post_r_attempt_reparameterization_audit_sha256
        ),
        "post_r_attempt_max_anchor_distance": _array_sha256(
            candidate.post_r_attempt_max_anchor_distance
        ),
        "trust_globalization_exhausted": bool(
            candidate.trust_globalization_exhausted
        ),
        "raw_path": _array_sha256(candidate.raw_path),
        "actual_u": _array_sha256(candidate.actual_normal_coordinate),
        "actual_step": _array_sha256(candidate.actual_normal_step),
        "proposal_effective_linear_term_binary64": float(
            candidate.proposal_effective_linear_term
        ).hex(),
        "proposal_quadratic_penalty_binary64": float(
            candidate.proposal_quadratic_penalty
        ).hex(),
        "proposal_effective_proximal_value_binary64": float(
            candidate.proposal_effective_proximal_value
        ).hex(),
    }
    result["reparameterization_audit_sha256"] = (
        "none"
        if candidate.reparameterization is None
        else candidate.reparameterization.audit_sha256
    )
    return result


def _frozen_frame_sha256(controller: Dict[str, Any], controls: IRIVMControls) -> str:
    payload = {
        "schema": "Ala2DFrozenWarmValidationFrame/v17-formal",
        "alpha_binary64": float(controls.alpha).hex(),
        "frozen_path": _array_sha256(controller["frozen_path"]),
        "frozen_M": _array_sha256(controller["frozen_M"]),
        "frozen_grad": _array_sha256(controller["frozen_grad"]),
        "frozen_tangent": _array_sha256(controller["frozen_tangent"]),
        "frozen_G": _array_sha256(controller["frozen_G"]),
        "frozen_basis": _array_sha256(controller["frozen_basis"]),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _freeze_warm_candidate(
    controller: Dict[str, Any],
    candidate: Candidate,
    *,
    source_score: float,
    source_score_standard_error: float,
    source_score_lower_bound: float,
    controls: IRIVMControls,
) -> Dict[str, Any]:
    """Freeze one post-takeover candidate before its sole fresh validation.

    The full candidate is present in the validation public state.  This compact
    immutable record contains every byte required to submit that *same*
    article-reparameterized path after a positive validation, plus independent
    hashes of the source frame and the original complete candidate digest.
    """

    reparam = candidate.reparameterization
    if (
        reparam is None
        or not candidate.valid
        or candidate.trust_globalization_exhausted
    ):
        raise IRIVMError("only a structurally valid warm candidate can be frozen")
    path = np.asarray(reparam.path, dtype=np.float64)
    delta = np.asarray(candidate.actual_normal_step, dtype=np.float64)
    actual_u = np.asarray(candidate.actual_normal_coordinate, dtype=np.float64)
    if (
        path.shape != (N_PATH, DIMENSION)
        or delta.shape != (N_INTERNAL,)
        or actual_u.shape != (N_INTERNAL,)
        or np.any(~np.isfinite(path))
        or np.any(~np.isfinite(delta))
        or np.any(~np.isfinite(actual_u))
    ):
        raise IRIVMError("the post-takeover warm candidate is not freezeable")
    if not np.array_equal(path[[0, -1]], controller["current_path"][[0, -1]]):
        raise IRIVMError("the frozen warm candidate moved a fixed endpoint")
    candidate_sha = hashlib.sha256(
        _canonical_json(_candidate_digest(candidate))
    ).hexdigest()
    frame_sha = _frozen_frame_sha256(controller, controls)
    source_checkpoint_sha = _checkpoint_digest(controller)
    request_payload = {
        "schema": "Ala2DFrozenWarmValidationRequest/v17-formal",
        "implementation_id": IMPLEMENTATION_ID,
        "source_checkpoint_sha256": source_checkpoint_sha,
        "source_state_index": int(controller["state_index"]),
        "source_nof": int(controller["nof"]),
        "source_path_sha256": _array_sha256(controller["current_path"]),
        "candidate_sha256": candidate_sha,
        "candidate_path_sha256": _array_sha256(path),
        "candidate_delta_sha256": _array_sha256(delta),
        "frozen_frame_sha256": frame_sha,
        "source_score_binary64": float(source_score).hex(),
        "source_score_standard_error_binary64": float(
            source_score_standard_error
        ).hex(),
        "source_score_lower_bound_binary64": float(
            source_score_lower_bound
        ).hex(),
    }
    request_sha = hashlib.sha256(_canonical_json(request_payload)).hexdigest()
    return {
        "label": str(candidate.label),
        "candidate_structural_valid": bool(candidate.valid),
        "candidate_error": str(candidate.error),
        "candidate_sha256": candidate_sha,
        "frame_sha256": frame_sha,
        "request_sha256": request_sha,
        "source_checkpoint_sha256": source_checkpoint_sha,
        "source_state_index": int(controller["state_index"]),
        "source_nof": int(controller["nof"]),
        "source_outer_index": int(controller["outer_index"]),
        "source_inner_step": int(controller["inner_step"]),
        "path": path.copy(),
        "raw_path": np.asarray(candidate.raw_path, dtype=np.float64).copy(),
        "source_path": np.asarray(controller["current_path"], dtype=np.float64).copy(),
        "frozen_path": np.asarray(controller["frozen_path"], dtype=np.float64).copy(),
        "frozen_M": np.asarray(controller["frozen_M"], dtype=np.float64).copy(),
        "frozen_grad": np.asarray(controller["frozen_grad"], dtype=np.float64).copy(),
        "frozen_tangent": np.asarray(
            controller["frozen_tangent"], dtype=np.float64
        ).copy(),
        "frozen_G": np.asarray(controller["frozen_G"], dtype=np.float64).copy(),
        "frozen_basis": np.asarray(
            controller["frozen_basis"], dtype=np.float64
        ).copy(),
        "actual_normal_coordinate": actual_u.copy(),
        "actual_normal_step": delta.copy(),
        "candidate_raw_direction": np.asarray(
            candidate.raw_direction, dtype=np.float64
        ).copy(),
        "candidate_capped_direction": np.asarray(
            candidate.capped_direction, dtype=np.float64
        ).copy(),
        "candidate_common_cap_scale": float(candidate.common_cap_scale),
        "candidate_common_cap_truncated": bool(candidate.common_cap_truncated),
        "candidate_anchor_cap_theoretical_scale": float(
            candidate.anchor_cap_theoretical_scale
        ),
        "candidate_anchor_cap_binary_bracket_used": bool(
            candidate.anchor_cap_binary_bracket_used
        ),
        "candidate_anchor_cap_binary_bracket_iterations": int(
            candidate.anchor_cap_binary_bracket_iterations
        ),
        "candidate_anchor_cap_source_distances": np.asarray(
            candidate.anchor_cap_source_distances, dtype=np.float64
        ).copy(),
        "candidate_anchor_cap_raw_path_distances": np.asarray(
            candidate.anchor_cap_raw_path_distances, dtype=np.float64
        ).copy(),
        "candidate_post_r_backtrack_count": int(candidate.post_r_backtrack_count),
        "candidate_post_r_backtrack_factor": float(
            candidate.post_r_backtrack_factor
        ),
        "candidate_total_direction_scale": float(candidate.total_direction_scale),
        "candidate_post_r_attempt_count": int(candidate.post_r_attempt_count),
        "candidate_post_r_attempt_scales": np.asarray(
            candidate.post_r_attempt_scales, dtype=np.float64
        ).copy(),
        "candidate_post_r_attempt_status": np.asarray(
            candidate.post_r_attempt_status
        ).copy(),
        "candidate_post_r_attempt_raw_path_sha256": np.asarray(
            candidate.post_r_attempt_raw_path_sha256
        ).copy(),
        "candidate_post_r_attempt_reparameterization_audit_sha256": np.asarray(
            candidate.post_r_attempt_reparameterization_audit_sha256
        ).copy(),
        "candidate_post_r_attempt_max_anchor_distance": np.asarray(
            candidate.post_r_attempt_max_anchor_distance, dtype=np.float64
        ).copy(),
        "candidate_trust_globalization_exhausted": bool(
            candidate.trust_globalization_exhausted
        ),
        "candidate_proposal_effective_linear_term": float(
            candidate.proposal_effective_linear_term
        ),
        "candidate_proposal_quadratic_penalty": float(
            candidate.proposal_quadratic_penalty
        ),
        "candidate_proposal_effective_proximal_value": float(
            candidate.proposal_effective_proximal_value
        ),
        "source_score": float(source_score),
        "source_score_standard_error": float(source_score_standard_error),
        "source_score_lower_bound": float(source_score_lower_bound),
        "reparameterization_iteration_count": int(reparam.iteration_count),
        "reparameterization_pre_update_residual": float(
            reparam.stopping_pre_update_spring_residual
        ),
        "reparameterization_output_residual": float(
            reparam.output_spring_residual
        ),
        "reparameterization_segment_lengths": np.asarray(
            reparam.segment_lengths, dtype=np.float64
        ).copy(),
        "reparameterization_maximum_neighbor_length_difference": float(
            reparam.maximum_neighbor_length_difference
        ),
        "reparameterization_actual_chord_relative_deviation": float(
            reparam.actual_chord_relative_deviation
        ),
        "reparameterization_actual_I_arc": float(reparam.actual_I_arc),
        "reparameterization_audit_sha256": str(reparam.audit_sha256),
    }


def _empty_pending_warm_fields() -> Dict[str, Any]:
    """Return the canonical no-pending-candidate checkpoint representation."""

    return {
        "pending_warm_status": WARM_VALIDATION_PENDING_NONE,
        "pending_warm_label": "none",
        "pending_warm_candidate_error": "none",
        "pending_warm_candidate_sha256": "none",
        "pending_warm_frame_sha256": "none",
        "pending_warm_request_sha256": "none",
        "pending_warm_source_checkpoint_sha256": "none",
        "pending_warm_source_state_index": -1,
        "pending_warm_source_nof": -1,
        "pending_warm_source_outer_index": -1,
        "pending_warm_source_inner_step": -1,
        "pending_warm_path": np.full((N_PATH, DIMENSION), np.nan),
        "pending_warm_raw_path": np.full((N_PATH, DIMENSION), np.nan),
        "pending_warm_source_path": np.full((N_PATH, DIMENSION), np.nan),
        "pending_warm_frozen_path": np.full((N_PATH, DIMENSION), np.nan),
        "pending_warm_frozen_M": np.full(
            (N_INTERNAL, DIMENSION, DIMENSION), np.nan
        ),
        "pending_warm_frozen_grad": np.full((N_INTERNAL, DIMENSION), np.nan),
        "pending_warm_frozen_tangent": np.full((N_INTERNAL, DIMENSION), np.nan),
        "pending_warm_frozen_G": np.full(
            (N_INTERNAL, DIMENSION, DIMENSION), np.nan
        ),
        "pending_warm_frozen_basis": np.full((N_INTERNAL, DIMENSION), np.nan),
        "pending_warm_actual_normal_coordinate": np.full(N_INTERNAL, np.nan),
        "pending_warm_actual_normal_step": np.full(N_INTERNAL, np.nan),
        "pending_warm_candidate_structural_valid": np.asarray(False),
        "pending_warm_candidate_raw_direction": np.full(N_INTERNAL, np.nan),
        "pending_warm_candidate_capped_direction": np.full(N_INTERNAL, np.nan),
        "pending_warm_candidate_common_cap_scale": np.asarray(np.nan),
        "pending_warm_candidate_common_cap_truncated": np.asarray(False),
        "pending_warm_candidate_anchor_cap_theoretical_scale": np.asarray(np.nan),
        "pending_warm_candidate_anchor_cap_binary_bracket_used": np.asarray(False),
        "pending_warm_candidate_anchor_cap_binary_bracket_iterations": -1,
        "pending_warm_candidate_anchor_cap_source_distances": np.full(
            N_INTERNAL, np.nan
        ),
        "pending_warm_candidate_anchor_cap_raw_path_distances": np.full(
            N_INTERNAL, np.nan
        ),
        "pending_warm_candidate_post_r_backtrack_count": -1,
        "pending_warm_candidate_post_r_backtrack_factor": np.asarray(np.nan),
        "pending_warm_candidate_total_direction_scale": np.asarray(np.nan),
        "pending_warm_candidate_post_r_attempt_count": -1,
        "pending_warm_candidate_post_r_attempt_scales": np.asarray(
            [], dtype=np.float64
        ),
        "pending_warm_candidate_post_r_attempt_status": np.asarray([], dtype="U256"),
        "pending_warm_candidate_post_r_attempt_raw_path_sha256": np.asarray(
            [], dtype="U64"
        ),
        "pending_warm_candidate_post_r_attempt_reparameterization_audit_sha256": np.asarray(
            [], dtype="U64"
        ),
        "pending_warm_candidate_post_r_attempt_max_anchor_distance": np.asarray(
            [], dtype=np.float64
        ),
        "pending_warm_candidate_trust_globalization_exhausted": np.asarray(False),
        "pending_warm_candidate_proposal_effective_linear_term": np.asarray(np.nan),
        "pending_warm_candidate_proposal_quadratic_penalty": np.asarray(np.nan),
        "pending_warm_candidate_proposal_effective_proximal_value": np.asarray(np.nan),
        "pending_warm_source_score": np.asarray(np.nan),
        "pending_warm_source_score_standard_error": np.asarray(np.nan),
        "pending_warm_source_score_lower_bound": np.asarray(np.nan),
        "pending_warm_reparameterization_iteration_count": -1,
        "pending_warm_reparameterization_pre_update_residual": np.asarray(np.nan),
        "pending_warm_reparameterization_output_residual": np.asarray(np.nan),
        "pending_warm_reparameterization_segment_lengths": np.full(
            N_PATH - 1, np.nan
        ),
        "pending_warm_reparameterization_maximum_neighbor_length_difference": np.asarray(
            np.nan
        ),
        "pending_warm_reparameterization_actual_chord_relative_deviation": np.asarray(
            np.nan
        ),
        "pending_warm_reparameterization_actual_I_arc": np.asarray(np.nan),
        "pending_warm_reparameterization_audit_sha256": "none",
        "pending_warm_validation_state_index": -1,
        "pending_warm_validation_score_mean": np.asarray(np.nan),
        "pending_warm_validation_score_standard_error": np.asarray(np.nan),
        "pending_warm_validation_score_lower_bound": np.asarray(np.nan),
        "pending_warm_validation_score_upper_bound": np.asarray(np.nan),
    }


def _pending_warm_fields_from_validated(
    frozen: Dict[str, Any], validation: Dict[str, object]
) -> Dict[str, Any]:
    if validation.get("classification") != "validated_positive":
        raise IRIVMError("only a positive independent validation may pend a commit")
    fields = {
        "pending_warm_status": WARM_VALIDATION_PENDING_COMMIT,
        "pending_warm_label": str(frozen["label"]),
        "pending_warm_candidate_error": str(frozen["candidate_error"]),
        "pending_warm_candidate_sha256": str(frozen["candidate_sha256"]),
        "pending_warm_frame_sha256": str(frozen["frame_sha256"]),
        "pending_warm_request_sha256": str(frozen["request_sha256"]),
        "pending_warm_source_checkpoint_sha256": str(
            frozen["source_checkpoint_sha256"]
        ),
        "pending_warm_source_state_index": int(frozen["source_state_index"]),
        "pending_warm_source_nof": int(frozen["source_nof"]),
        "pending_warm_source_outer_index": int(frozen["source_outer_index"]),
        "pending_warm_source_inner_step": int(frozen["source_inner_step"]),
        "pending_warm_path": np.asarray(frozen["path"], dtype=np.float64).copy(),
        "pending_warm_raw_path": np.asarray(
            frozen["raw_path"], dtype=np.float64
        ).copy(),
        "pending_warm_source_path": np.asarray(
            frozen["source_path"], dtype=np.float64
        ).copy(),
        "pending_warm_frozen_path": np.asarray(
            frozen["frozen_path"], dtype=np.float64
        ).copy(),
        "pending_warm_frozen_M": np.asarray(
            frozen["frozen_M"], dtype=np.float64
        ).copy(),
        "pending_warm_frozen_grad": np.asarray(
            frozen["frozen_grad"], dtype=np.float64
        ).copy(),
        "pending_warm_frozen_tangent": np.asarray(
            frozen["frozen_tangent"], dtype=np.float64
        ).copy(),
        "pending_warm_frozen_G": np.asarray(
            frozen["frozen_G"], dtype=np.float64
        ).copy(),
        "pending_warm_frozen_basis": np.asarray(
            frozen["frozen_basis"], dtype=np.float64
        ).copy(),
        "pending_warm_actual_normal_coordinate": np.asarray(
            frozen["actual_normal_coordinate"], dtype=np.float64
        ).copy(),
        "pending_warm_actual_normal_step": np.asarray(
            frozen["actual_normal_step"], dtype=np.float64
        ).copy(),
        "pending_warm_candidate_structural_valid": np.asarray(
            frozen["candidate_structural_valid"], dtype=bool
        ),
        "pending_warm_candidate_raw_direction": np.asarray(
            frozen["candidate_raw_direction"], dtype=np.float64
        ).copy(),
        "pending_warm_candidate_capped_direction": np.asarray(
            frozen["candidate_capped_direction"], dtype=np.float64
        ).copy(),
        "pending_warm_candidate_common_cap_scale": np.asarray(
            frozen["candidate_common_cap_scale"], dtype=np.float64
        ),
        "pending_warm_candidate_common_cap_truncated": np.asarray(
            frozen["candidate_common_cap_truncated"], dtype=bool
        ),
        "pending_warm_candidate_anchor_cap_theoretical_scale": np.asarray(
            frozen["candidate_anchor_cap_theoretical_scale"], dtype=np.float64
        ),
        "pending_warm_candidate_anchor_cap_binary_bracket_used": np.asarray(
            frozen["candidate_anchor_cap_binary_bracket_used"], dtype=bool
        ),
        "pending_warm_candidate_anchor_cap_binary_bracket_iterations": int(
            frozen["candidate_anchor_cap_binary_bracket_iterations"]
        ),
        "pending_warm_candidate_anchor_cap_source_distances": np.asarray(
            frozen["candidate_anchor_cap_source_distances"], dtype=np.float64
        ).copy(),
        "pending_warm_candidate_anchor_cap_raw_path_distances": np.asarray(
            frozen["candidate_anchor_cap_raw_path_distances"], dtype=np.float64
        ).copy(),
        "pending_warm_candidate_post_r_backtrack_count": int(
            frozen["candidate_post_r_backtrack_count"]
        ),
        "pending_warm_candidate_post_r_backtrack_factor": np.asarray(
            frozen["candidate_post_r_backtrack_factor"], dtype=np.float64
        ),
        "pending_warm_candidate_total_direction_scale": np.asarray(
            frozen["candidate_total_direction_scale"], dtype=np.float64
        ),
        "pending_warm_candidate_post_r_attempt_count": int(
            frozen["candidate_post_r_attempt_count"]
        ),
        "pending_warm_candidate_post_r_attempt_scales": np.asarray(
            frozen["candidate_post_r_attempt_scales"], dtype=np.float64
        ).copy(),
        "pending_warm_candidate_post_r_attempt_status": np.asarray(
            frozen["candidate_post_r_attempt_status"]
        ).copy(),
        "pending_warm_candidate_post_r_attempt_raw_path_sha256": np.asarray(
            frozen["candidate_post_r_attempt_raw_path_sha256"]
        ).copy(),
        "pending_warm_candidate_post_r_attempt_reparameterization_audit_sha256": np.asarray(
            frozen["candidate_post_r_attempt_reparameterization_audit_sha256"]
        ).copy(),
        "pending_warm_candidate_post_r_attempt_max_anchor_distance": np.asarray(
            frozen["candidate_post_r_attempt_max_anchor_distance"], dtype=np.float64
        ).copy(),
        "pending_warm_candidate_trust_globalization_exhausted": np.asarray(
            frozen["candidate_trust_globalization_exhausted"], dtype=bool
        ),
        "pending_warm_candidate_proposal_effective_linear_term": np.asarray(
            frozen["candidate_proposal_effective_linear_term"], dtype=np.float64
        ),
        "pending_warm_candidate_proposal_quadratic_penalty": np.asarray(
            frozen["candidate_proposal_quadratic_penalty"], dtype=np.float64
        ),
        "pending_warm_candidate_proposal_effective_proximal_value": np.asarray(
            frozen["candidate_proposal_effective_proximal_value"], dtype=np.float64
        ),
        "pending_warm_source_score": np.asarray(
            frozen["source_score"], dtype=np.float64
        ),
        "pending_warm_source_score_standard_error": np.asarray(
            frozen["source_score_standard_error"], dtype=np.float64
        ),
        "pending_warm_source_score_lower_bound": np.asarray(
            frozen["source_score_lower_bound"], dtype=np.float64
        ),
        "pending_warm_reparameterization_iteration_count": int(
            frozen["reparameterization_iteration_count"]
        ),
        "pending_warm_reparameterization_pre_update_residual": np.asarray(
            frozen["reparameterization_pre_update_residual"], dtype=np.float64
        ),
        "pending_warm_reparameterization_output_residual": np.asarray(
            frozen["reparameterization_output_residual"], dtype=np.float64
        ),
        "pending_warm_reparameterization_segment_lengths": np.asarray(
            frozen["reparameterization_segment_lengths"], dtype=np.float64
        ).copy(),
        "pending_warm_reparameterization_maximum_neighbor_length_difference": np.asarray(
            frozen["reparameterization_maximum_neighbor_length_difference"],
            dtype=np.float64,
        ),
        "pending_warm_reparameterization_actual_chord_relative_deviation": np.asarray(
            frozen["reparameterization_actual_chord_relative_deviation"],
            dtype=np.float64,
        ),
        "pending_warm_reparameterization_actual_I_arc": np.asarray(
            frozen["reparameterization_actual_I_arc"], dtype=np.float64
        ),
        "pending_warm_reparameterization_audit_sha256": str(
            frozen["reparameterization_audit_sha256"]
        ),
        "pending_warm_validation_state_index": int(validation["state_index"]),
        "pending_warm_validation_score_mean": np.asarray(
            validation["score_mean"], dtype=np.float64
        ),
        "pending_warm_validation_score_standard_error": np.asarray(
            validation["score_standard_error"], dtype=np.float64
        ),
        "pending_warm_validation_score_lower_bound": np.asarray(
            validation["score_lower_bound"], dtype=np.float64
        ),
        "pending_warm_validation_score_upper_bound": np.asarray(
            validation["score_upper_bound"], dtype=np.float64
        ),
    }
    return fields


PENDING_WARM_STRING_KEYS = (
    "pending_warm_status",
    "pending_warm_label",
    "pending_warm_candidate_error",
    "pending_warm_candidate_sha256",
    "pending_warm_frame_sha256",
    "pending_warm_request_sha256",
    "pending_warm_source_checkpoint_sha256",
    "pending_warm_reparameterization_audit_sha256",
)
PENDING_WARM_INTEGER_KEYS = (
    "pending_warm_source_state_index",
    "pending_warm_source_nof",
    "pending_warm_source_outer_index",
    "pending_warm_source_inner_step",
    "pending_warm_reparameterization_iteration_count",
    "pending_warm_validation_state_index",
    "pending_warm_candidate_anchor_cap_binary_bracket_iterations",
    "pending_warm_candidate_post_r_backtrack_count",
    "pending_warm_candidate_post_r_attempt_count",
)
PENDING_WARM_ARRAY_KEYS = (
    "pending_warm_path",
    "pending_warm_raw_path",
    "pending_warm_source_path",
    "pending_warm_frozen_path",
    "pending_warm_frozen_M",
    "pending_warm_frozen_grad",
    "pending_warm_frozen_tangent",
    "pending_warm_frozen_G",
    "pending_warm_frozen_basis",
    "pending_warm_actual_normal_coordinate",
    "pending_warm_actual_normal_step",
    "pending_warm_candidate_structural_valid",
    "pending_warm_candidate_raw_direction",
    "pending_warm_candidate_capped_direction",
    "pending_warm_candidate_common_cap_scale",
    "pending_warm_candidate_common_cap_truncated",
    "pending_warm_candidate_anchor_cap_theoretical_scale",
    "pending_warm_candidate_anchor_cap_binary_bracket_used",
    "pending_warm_candidate_anchor_cap_source_distances",
    "pending_warm_candidate_anchor_cap_raw_path_distances",
    "pending_warm_candidate_post_r_backtrack_factor",
    "pending_warm_candidate_total_direction_scale",
    "pending_warm_candidate_post_r_attempt_scales",
    "pending_warm_candidate_post_r_attempt_status",
    "pending_warm_candidate_post_r_attempt_raw_path_sha256",
    "pending_warm_candidate_post_r_attempt_reparameterization_audit_sha256",
    "pending_warm_candidate_post_r_attempt_max_anchor_distance",
    "pending_warm_candidate_trust_globalization_exhausted",
    "pending_warm_candidate_proposal_effective_linear_term",
    "pending_warm_candidate_proposal_quadratic_penalty",
    "pending_warm_candidate_proposal_effective_proximal_value",
    "pending_warm_source_score",
    "pending_warm_source_score_standard_error",
    "pending_warm_source_score_lower_bound",
    "pending_warm_reparameterization_pre_update_residual",
    "pending_warm_reparameterization_output_residual",
    "pending_warm_reparameterization_segment_lengths",
    "pending_warm_reparameterization_maximum_neighbor_length_difference",
    "pending_warm_reparameterization_actual_chord_relative_deviation",
    "pending_warm_reparameterization_actual_I_arc",
    "pending_warm_validation_score_mean",
    "pending_warm_validation_score_standard_error",
    "pending_warm_validation_score_lower_bound",
    "pending_warm_validation_score_upper_bound",
)


def _pending_frame_sha256(controller: Dict[str, Any], controls: IRIVMControls) -> str:
    frame_view = {
        "frozen_path": controller["pending_warm_frozen_path"],
        "frozen_M": controller["pending_warm_frozen_M"],
        "frozen_grad": controller["pending_warm_frozen_grad"],
        "frozen_tangent": controller["pending_warm_frozen_tangent"],
        "frozen_G": controller["pending_warm_frozen_G"],
        "frozen_basis": controller["pending_warm_frozen_basis"],
    }
    return _frozen_frame_sha256(frame_view, controls)


def _validate_pending_warm_fields(
    controller: Dict[str, Any], controls: IRIVMControls
) -> None:
    status = str(controller["pending_warm_status"])
    if status == WARM_VALIDATION_PENDING_NONE:
        expected = _empty_pending_warm_fields()
        for key, value in expected.items():
            if not _array_equal(controller[key], value):
                raise IRIVMError(
                    f"cleared pending warm record has noncanonical field {key}"
                )
        return
    if status != WARM_VALIDATION_PENDING_COMMIT:
        raise IRIVMError("checkpoint has an unknown pending warm status")
    if not bool(controller["lbfgs_latched"]):
        raise IRIVMError("a pending warm commit requires the global takeover latch")
    if int(controller["pending_warm_validation_state_index"]) != int(
        controller["state_index"]
    ):
        raise IRIVMError("pending warm validation is not bound to the current state")
    if int(controller["pending_warm_source_state_index"]) + 1 != int(
        controller["state_index"]
    ):
        raise IRIVMError("pending warm source and validation states are not adjacent")
    if int(controller["pending_warm_source_nof"]) + N_INTERNAL != int(
        controller["nof"]
    ):
        raise IRIVMError("pending warm source and validation NoF differ by one band")
    if (
        int(controller["pending_warm_source_outer_index"])
        != int(controller["outer_index"])
        or int(controller["pending_warm_source_inner_step"])
        != int(controller["inner_step"])
    ):
        raise IRIVMError(
            "pending warm validation changed outer or movement indices"
        )
    if not np.array_equal(
        controller["pending_warm_source_path"], controller["current_path"]
    ):
        raise IRIVMError("pending warm validation did not keep the exact source path")
    if np.array_equal(controller["pending_warm_path"], controller["current_path"]):
        raise IRIVMError("pending warm movement has zero exact path displacement")
    for pending_key, current_key in (
        ("pending_warm_frozen_path", "frozen_path"),
        ("pending_warm_frozen_M", "frozen_M"),
        ("pending_warm_frozen_grad", "frozen_grad"),
        ("pending_warm_frozen_tangent", "frozen_tangent"),
        ("pending_warm_frozen_G", "frozen_G"),
        ("pending_warm_frozen_basis", "frozen_basis"),
    ):
        if not np.array_equal(controller[pending_key], controller[current_key]):
            raise IRIVMError(
                "pending warm candidate frame changed after positive validation"
            )
    if str(controller["pending_warm_frame_sha256"]) != _pending_frame_sha256(
        controller, controls
    ):
        raise IRIVMError("pending warm frozen-frame hash differs")
    if not (
        np.isfinite(float(controller["pending_warm_source_score"]))
        and np.isfinite(
            float(controller["pending_warm_source_score_standard_error"])
        )
        and float(controller["pending_warm_source_score_standard_error"]) >= 0.0
        and np.isfinite(float(controller["pending_warm_source_score_lower_bound"]))
        and float(controller["pending_warm_source_score_lower_bound"])
        == float(controller["pending_warm_source_score"])
        - controls.residual_bound_kappa
        * float(controller["pending_warm_source_score_standard_error"])
        and float(controller["pending_warm_source_score_lower_bound"]) <= 0.0
    ):
        raise IRIVMError("pending warm source evidence is not the registered trigger")
    validation_mean = float(controller["pending_warm_validation_score_mean"])
    validation_se = float(
        controller["pending_warm_validation_score_standard_error"]
    )
    validation_lower = float(
        controller["pending_warm_validation_score_lower_bound"]
    )
    validation_upper = float(
        controller["pending_warm_validation_score_upper_bound"]
    )
    if not (
        np.isfinite(validation_mean)
        and np.isfinite(validation_se)
        and validation_se >= 0.0
        and validation_lower == validation_mean - controls.residual_bound_kappa * validation_se
        and validation_upper == validation_mean + controls.residual_bound_kappa * validation_se
        and validation_lower > 0.0
    ):
        raise IRIVMError("pending warm candidate lacks a positive fresh validation")
    if str(controller["pending_warm_candidate_sha256"]) == "none":
        raise IRIVMError("pending warm candidate lacks its original digest")
    request_payload = {
        "schema": "Ala2DFrozenWarmValidationRequest/v17-formal",
        "implementation_id": IMPLEMENTATION_ID,
        "source_checkpoint_sha256": str(
            controller["pending_warm_source_checkpoint_sha256"]
        ),
        "source_state_index": int(controller["pending_warm_source_state_index"]),
        "source_nof": int(controller["pending_warm_source_nof"]),
        "source_path_sha256": _array_sha256(
            controller["pending_warm_source_path"]
        ),
        "candidate_sha256": str(controller["pending_warm_candidate_sha256"]),
        "candidate_path_sha256": _array_sha256(controller["pending_warm_path"]),
        "candidate_delta_sha256": _array_sha256(
            controller["pending_warm_actual_normal_step"]
        ),
        "frozen_frame_sha256": str(controller["pending_warm_frame_sha256"]),
        "source_score_binary64": float(
            controller["pending_warm_source_score"]
        ).hex(),
        "source_score_standard_error_binary64": float(
            controller["pending_warm_source_score_standard_error"]
        ).hex(),
        "source_score_lower_bound_binary64": float(
            controller["pending_warm_source_score_lower_bound"]
        ).hex(),
    }
    expected_request_sha = hashlib.sha256(
        _canonical_json(request_payload)
    ).hexdigest()
    if str(controller["pending_warm_request_sha256"]) != expected_request_sha:
        raise IRIVMError("pending warm validation request digest differs")
    reparam = ArticleReparameterizationResult(
        path=np.asarray(controller["pending_warm_path"], dtype=np.float64),
        iteration_count=int(
            controller["pending_warm_reparameterization_iteration_count"]
        ),
        stopping_pre_update_spring_residual=float(
            controller["pending_warm_reparameterization_pre_update_residual"]
        ),
        output_spring_residual=float(
            controller["pending_warm_reparameterization_output_residual"]
        ),
        segment_lengths=np.asarray(
            controller["pending_warm_reparameterization_segment_lengths"],
            dtype=np.float64,
        ),
        maximum_neighbor_length_difference=float(
            controller[
                "pending_warm_reparameterization_maximum_neighbor_length_difference"
            ]
        ),
        actual_chord_relative_deviation=float(
            controller[
                "pending_warm_reparameterization_actual_chord_relative_deviation"
            ]
        ),
        actual_I_arc=float(controller["pending_warm_reparameterization_actual_I_arc"]),
        audit_sha256=str(
            controller["pending_warm_reparameterization_audit_sha256"]
        ),
    )
    _validate_article_reparameterization_attempt(
        controller["pending_warm_raw_path"], reparam, controls
    )
    recomputed_u = _normal_coordinate(
        controller["pending_warm_path"],
        controller["pending_warm_frozen_path"],
        controller["pending_warm_frozen_basis"],
        controller["pending_warm_frozen_G"],
    )
    source_u = _normal_coordinate(
        controller["pending_warm_source_path"],
        controller["pending_warm_frozen_path"],
        controller["pending_warm_frozen_basis"],
        controller["pending_warm_frozen_G"],
    )
    if not (
        np.array_equal(recomputed_u, controller["pending_warm_actual_normal_coordinate"])
        and np.array_equal(
            recomputed_u - source_u,
            controller["pending_warm_actual_normal_step"],
        )
    ):
        raise IRIVMError("pending warm normal displacement differs from frozen paths")


def _candidate_from_pending_warm(
    controller: Dict[str, Any], controls: IRIVMControls
) -> Candidate:
    _validate_pending_warm_fields(controller, controls)
    delta = np.asarray(
        controller["pending_warm_actual_normal_step"], dtype=np.float64
    ).copy()
    reparam = ArticleReparameterizationResult(
        path=np.asarray(controller["pending_warm_path"], dtype=np.float64).copy(),
        iteration_count=int(
            controller["pending_warm_reparameterization_iteration_count"]
        ),
        stopping_pre_update_spring_residual=float(
            controller["pending_warm_reparameterization_pre_update_residual"]
        ),
        output_spring_residual=float(
            controller["pending_warm_reparameterization_output_residual"]
        ),
        segment_lengths=np.asarray(
            controller["pending_warm_reparameterization_segment_lengths"],
            dtype=np.float64,
        ).copy(),
        maximum_neighbor_length_difference=float(
            controller[
                "pending_warm_reparameterization_maximum_neighbor_length_difference"
            ]
        ),
        actual_chord_relative_deviation=float(
            controller[
                "pending_warm_reparameterization_actual_chord_relative_deviation"
            ]
        ),
        actual_I_arc=float(controller["pending_warm_reparameterization_actual_I_arc"]),
        audit_sha256=str(
            controller["pending_warm_reparameterization_audit_sha256"]
        ),
    )
    candidate = Candidate(
        label=str(controller["pending_warm_label"]),
        valid=bool(controller["pending_warm_candidate_structural_valid"]),
        error=str(controller["pending_warm_candidate_error"]),
        raw_direction=np.asarray(
            controller["pending_warm_candidate_raw_direction"], dtype=np.float64
        ).copy(),
        capped_direction=np.asarray(
            controller["pending_warm_candidate_capped_direction"], dtype=np.float64
        ).copy(),
        common_cap_scale=float(
            controller["pending_warm_candidate_common_cap_scale"]
        ),
        common_cap_truncated=bool(
            controller["pending_warm_candidate_common_cap_truncated"]
        ),
        raw_path=np.asarray(controller["pending_warm_raw_path"], dtype=np.float64).copy(),
        reparameterization=reparam,
        actual_normal_coordinate=np.asarray(
            controller["pending_warm_actual_normal_coordinate"], dtype=np.float64
        ).copy(),
        actual_normal_step=delta,
        proposal_effective_linear_term=float(
            controller["pending_warm_candidate_proposal_effective_linear_term"]
        ),
        proposal_quadratic_penalty=float(
            controller["pending_warm_candidate_proposal_quadratic_penalty"]
        ),
        proposal_effective_proximal_value=float(
            controller["pending_warm_candidate_proposal_effective_proximal_value"]
        ),
        anchor_cap_theoretical_scale=float(
            controller["pending_warm_candidate_anchor_cap_theoretical_scale"]
        ),
        anchor_cap_binary_bracket_used=bool(
            controller["pending_warm_candidate_anchor_cap_binary_bracket_used"]
        ),
        anchor_cap_binary_bracket_iterations=int(
            controller["pending_warm_candidate_anchor_cap_binary_bracket_iterations"]
        ),
        anchor_cap_source_distances=np.asarray(
            controller["pending_warm_candidate_anchor_cap_source_distances"],
            dtype=np.float64,
        ).copy(),
        anchor_cap_raw_path_distances=np.asarray(
            controller["pending_warm_candidate_anchor_cap_raw_path_distances"],
            dtype=np.float64,
        ).copy(),
        post_r_backtrack_count=int(
            controller["pending_warm_candidate_post_r_backtrack_count"]
        ),
        post_r_backtrack_factor=float(
            controller["pending_warm_candidate_post_r_backtrack_factor"]
        ),
        total_direction_scale=float(
            controller["pending_warm_candidate_total_direction_scale"]
        ),
        post_r_attempt_count=int(
            controller["pending_warm_candidate_post_r_attempt_count"]
        ),
        post_r_attempt_scales=np.asarray(
            controller["pending_warm_candidate_post_r_attempt_scales"],
            dtype=np.float64,
        ).copy(),
        post_r_attempt_status=np.asarray(
            controller["pending_warm_candidate_post_r_attempt_status"]
        ).copy(),
        post_r_attempt_raw_path_sha256=np.asarray(
            controller["pending_warm_candidate_post_r_attempt_raw_path_sha256"]
        ).copy(),
        post_r_attempt_reparameterization_audit_sha256=np.asarray(
            controller[
                "pending_warm_candidate_post_r_attempt_reparameterization_audit_sha256"
            ]
        ).copy(),
        post_r_attempt_max_anchor_distance=np.asarray(
            controller["pending_warm_candidate_post_r_attempt_max_anchor_distance"],
            dtype=np.float64,
        ).copy(),
        trust_globalization_exhausted=bool(
            controller["pending_warm_candidate_trust_globalization_exhausted"]
        ),
    )
    candidate_sha = hashlib.sha256(
        _canonical_json(_candidate_digest(candidate))
    ).hexdigest()
    if candidate_sha != str(controller["pending_warm_candidate_sha256"]):
        raise IRIVMError("pending warm candidate bytes differ from its frozen digest")
    return candidate


def _fresh_raw_warm_validation_statistics(
    frozen: Dict[str, Any],
    *,
    raw_grad: Array,
    raw_grad_blocks: Array,
    raw_block_frame_counts: Array,
    controls: IRIVMControls,
    single_block_exact: bool,
    validation_state_index: int,
) -> Dict[str, object]:
    """Score a frozen displacement from only one newly purchased raw band."""

    statistics = _residual_statistics(
        path=frozen["source_path"],
        grad=raw_grad,
        grad_blocks=raw_grad_blocks,
        frozen_path=frozen["frozen_path"],
        frozen_basis=frozen["frozen_basis"],
        frozen_G=frozen["frozen_G"],
        alpha=controls.alpha,
        bound_kappa=controls.residual_bound_kappa,
        single_block_exact=single_block_exact,
        block_frame_counts=raw_block_frame_counts,
    )
    if not statistics.bounds_available:
        raise IRIVMError("fresh warm validation lacks raw-block uncertainty")
    delta = np.asarray(frozen["actual_normal_step"], dtype=np.float64)
    linear, quadratic, score = _proximal_value_terms(
        statistics.raw_residual, delta, controls.alpha
    )
    score_se = float(np.linalg.norm(statistics.standard_error * delta))
    classification, lower, upper = _classify_raw_improvement_interval(
        score, score_se, controls.residual_bound_kappa
    )
    return {
        "classification": classification,
        "state_index": int(validation_state_index),
        "score_mean": score,
        "score_standard_error": score_se,
        "score_lower_bound": lower,
        "score_upper_bound": upper,
        "raw_linear_decrease": linear,
        "proximal_quadratic_penalty": quadratic,
        "fresh_raw_residual": statistics.raw_residual.copy(),
        "fresh_raw_residual_standard_error": statistics.standard_error.copy(),
        "fresh_raw_block_count": int(np.asarray(raw_grad_blocks).shape[1]),
        "used_pooled_source_estimate": False,
        "used_shrinkage_centered_residual": False,
    }


def _classify_raw_improvement_interval(
    mean: float, standard_error: float, registered_margin_multiplier: float
) -> Tuple[str, float, float]:
    """Apply the registered three-way evidence rule, including boundaries."""

    values = (mean, standard_error, registered_margin_multiplier)
    if not all(np.isfinite(value) for value in values):
        raise IRIVMError("raw improvement evidence interval is non-finite")
    if standard_error < 0.0 or registered_margin_multiplier <= 0.0:
        raise IRIVMError("raw improvement evidence interval has invalid scales")
    lower = float(mean - registered_margin_multiplier * standard_error)
    upper = float(mean + registered_margin_multiplier * standard_error)
    if lower > 0.0:
        classification = "validated_positive"
    elif upper <= 0.0:
        classification = "confirmed_bad"
    else:
        classification = "unresolved"
    return classification, lower, upper


def _checkpoint_digest(controller: Dict[str, Any]) -> str:
    scalar_keys = (
        "state_index",
        "nof",
        "outer_index",
        "inner_step",
        "observation_count",
        "lbfgs_count",
        "warm_safeguard_source_state_index",
        "curvature_epoch_floor_state_index",
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
    ) + PENDING_WARM_ARRAY_KEYS
    payload: Dict[str, object] = {
        "schema": "Ala2DIRIVMControllerCheckpoint/v17-formal",
        "implementation_id": IMPLEMENTATION_ID,
        "implementation_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "trust_refresh_used_path_sha256": str(
            controller["trust_refresh_used_path_sha256"]
        ),
        "warm_restart_used_path_sha256": str(
            controller["warm_restart_used_path_sha256"]
        ),
    }
    for key in PENDING_WARM_STRING_KEYS:
        payload[key] = str(controller[key])
    for key in PENDING_WARM_INTEGER_KEYS:
        payload[key] = int(controller[key])
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
        "curvature_epoch_floor_state_index": 0,
        "single_block_exact_after_initial": bool(single_block_exact_after_initial),
        "trust_refresh_used_path_sha256": "none",
        "warm_restart_used_path_sha256": "none",
        "lbfgs_anchor_u": statistics.normal_coordinate.copy(),
        "lbfgs_anchor_raw_residual": statistics.raw_residual.copy(),
        "lbfgs_anchor_standard_error": statistics.standard_error.copy(),
        "controls": controls,
    }
    controller.update(_empty_pending_warm_fields())
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
    curvature_floor = int(controller["curvature_epoch_floor_state_index"])
    if not 0 <= curvature_floor <= state_index:
        raise IRIVMError("curvature epoch floor is outside the paid trajectory")
    refresh_path_sha = str(controller.get("trust_refresh_used_path_sha256", ""))
    if refresh_path_sha not in ("none", _array_sha256(path)):
        raise IRIVMError(
            "trust-refresh path marker is neither clear nor the current paid path"
        )
    warm_restart_path_sha = str(
        controller.get("warm_restart_used_path_sha256", "")
    )
    if warm_restart_path_sha not in ("none", _array_sha256(path)):
        raise IRIVMError(
            "warm-restart path marker is neither clear nor the current paid path"
        )
    _validate_pending_warm_fields(controller, controls)
    if warm_restart_path_sha != "none":
        if not bool(controller["lbfgs_latched"]):
            raise IRIVMError("warm restart cannot clear the global takeover latch")
        if int(controller["lbfgs_count"]) != 0:
            raise IRIVMError("same-path warm restart retained curvature pairs")
        if float(controller["warm_safeguard_inverse_scale"]) != float(
            controls.alpha
        ):
            raise IRIVMError("warm restart scalar safeguard is not exactly alpha")
        if str(controller["pending_warm_status"]) != WARM_VALIDATION_PENDING_NONE:
            raise IRIVMError("warm restart and pending frozen commit cannot coexist")
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
        "trust_refresh_used_path_sha256",
        "warm_restart_used_path_sha256",
    ) + PENDING_WARM_STRING_KEYS + PENDING_WARM_INTEGER_KEYS + PENDING_WARM_ARRAY_KEYS
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
        curvature_epoch_floor_state_index=np.asarray(
            controller["curvature_epoch_floor_state_index"], dtype=np.int64
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
        raise IRIVMError("checkpoint phase is not V17 paid-state-ready")
    scalar_int = (
        "state_index",
        "nof",
        "outer_index",
        "inner_step",
        "observation_count",
        "lbfgs_count",
        "warm_safeguard_source_state_index",
        "curvature_epoch_floor_state_index",
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
        "trust_refresh_used_path_sha256",
        "warm_restart_used_path_sha256",
    ) + PENDING_WARM_STRING_KEYS + PENDING_WARM_INTEGER_KEYS + PENDING_WARM_ARRAY_KEYS:
        if key not in record:
            raise IRIVMError(f"checkpoint is missing {key}")
        controller[key] = record[key]
    controller["trust_refresh_used_path_sha256"] = str(
        np.asarray(controller["trust_refresh_used_path_sha256"]).item()
    )
    controller["warm_restart_used_path_sha256"] = str(
        np.asarray(controller["warm_restart_used_path_sha256"]).item()
    )
    for key in PENDING_WARM_STRING_KEYS:
        controller[key] = str(np.asarray(controller[key]).item())
    for key in PENDING_WARM_INTEGER_KEYS:
        controller[key] = int(np.asarray(controller[key]).item())
    controller["controls"] = controls
    _validate_controller(controller, controls)
    if str(record["checkpoint_sha256"].item()) != _checkpoint_digest(controller):
        raise IRIVMError("checkpoint digest differs from its contents")
    return controller


def _prepare_pending_warm_commit(
    controller: Dict[str, Any], controls: IRIVMControls
) -> PreparedUpdate:
    """Submit exactly the candidate frozen before the preceding validation."""

    _validate_pending_warm_fields(controller, controls)
    if str(controller["pending_warm_status"]) != WARM_VALIDATION_PENDING_COMMIT:
        raise IRIVMError("frozen warm commit preparation lacks a pending candidate")
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
    lengths, rho, radii = _trust_radii(controller["frozen_path"], controls)
    candidate = _candidate_from_pending_warm(controller, controls)
    if np.any(
        _anchor_trust_distances(candidate.reparameterization.path, controller["frozen_path"])
        > radii
    ):
        raise IRIVMError("validated frozen warm candidate left its frozen trust region")
    h0 = _fit_h0_regression(controller, statistics, controls)
    h0_candidate = _invalid_candidate(
        "H0", "cold_forbidden_after_global_one_way_takeover"
    )
    validation_score = float(controller["pending_warm_validation_score_mean"])
    validation_se = float(
        controller["pending_warm_validation_score_standard_error"]
    )
    validation_lower = float(
        controller["pending_warm_validation_score_lower_bound"]
    )
    validation_upper = float(
        controller["pending_warm_validation_score_upper_bound"]
    )
    quadratic = float(np.dot(candidate.actual_normal_step, candidate.actual_normal_step)) / (
        2.0 * controls.alpha
    )
    raw_evidence = {
        "linear": validation_score + quadratic,
        "quadratic": quadratic,
        "score": validation_score,
        "standard_error": validation_se,
        "lower_bound": validation_lower,
        "upper_bound": validation_upper,
    }
    h0_raw = {
        key: math.nan
        for key in (
            "linear",
            "quadratic",
            "score",
            "standard_error",
            "lower_bound",
            "upper_bound",
        )
    }
    source_digest = _checkpoint_digest(controller)
    proposal_payload = {
        "schema": "Ala2DPreparedUpdate/v17-formal",
        "implementation_id": IMPLEMENTATION_ID,
        "action": "commit_validated_frozen_warm_candidate",
        "source_checkpoint_sha256": source_digest,
        "pending_warm_request_sha256": str(
            controller["pending_warm_request_sha256"]
        ),
        "pending_warm_candidate_sha256": str(
            controller["pending_warm_candidate_sha256"]
        ),
        "pending_warm_frame_sha256": str(controller["pending_warm_frame_sha256"]),
        "selected_path": _array_sha256(candidate.reparameterization.path),
        "validation_score_mean_binary64": validation_score.hex(),
        "validation_score_standard_error_binary64": validation_se.hex(),
        "validation_score_lower_bound_binary64": validation_lower.hex(),
        "validation_score_upper_bound_binary64": validation_upper.hex(),
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
        lbfgs_candidate=candidate,
        h0_raw_improvement=h0_raw,
        lbfgs_raw_improvement=raw_evidence,
        selected_label=candidate.label,
        selected_path=candidate.reparameterization.path.copy(),
        latch_before=True,
        latch_after=True,
        strict_lbfgs_win=False,
        forced_lbfgs_after_takeover=True,
        h0_invalid_fallback=False,
        hold_selected=False,
        hold_reason="none",
        global_signal_to_noise=(
            float(np.linalg.norm(statistics.raw_residual) / np.linalg.norm(statistics.standard_error))
            if np.linalg.norm(statistics.standard_error) > 0.0
            else math.inf
        ),
        lbfgs_raw_improvement_advantage=math.nan,
        raw_improvement_difference_standard_error=math.nan,
        lbfgs_raw_advantage_resolution_threshold=math.nan,
        lbfgs_raw_advantage_resolved_positive=False,
        initial_no_se_special_case=False,
        selected_raw_improvement_standard_error=validation_se,
        selected_raw_improvement_lower_bound=validation_lower,
        warm_validation_selected=False,
        frozen_warm_commit=True,
        frozen_warm_candidate=None,
    )


def _prepare_update_once(
    controller: Dict[str, Any], controls: IRIVMControls
) -> Tuple[Optional[PreparedUpdate], Dict[str, object]]:
    if str(controller["pending_warm_status"]) == WARM_VALIDATION_PENDING_COMMIT:
        return _prepare_pending_warm_commit(controller, controls), {}
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
        raise IRIVMError("noninitial V17 decision lacks residual uncertainty bounds")
    current_path_sha256 = _array_sha256(controller["current_path"])
    warm_restart_at_current_path = bool(
        controller["lbfgs_latched"]
        and str(controller["warm_restart_used_path_sha256"])
        == current_path_sha256
    )
    anchor_low_signal = bool(
        not initial_no_se_special_case
        and not bounds_ready
        and statistics.lower_norm <= controls.kkt_floor
        and not warm_restart_at_current_path
    )
    if (
        not initial_no_se_special_case
        and bounds_ready
        and int(controller["inner_step"]) > 0
    ):
        anchor_lower = float(controller["outer_anchor_lower_norm"])
        clear_stop = statistics.upper_norm <= controls.r * anchor_lower
        if clear_stop:
            raise IRIVMError("clear residual stop was not committed at previous paid state")

    pre_motion_hold_reason = "none"
    if anchor_low_signal:
        pre_motion_hold_reason = "anchor_lower_bound_zero_exact_center_resample"

    # A HOLD is an exact byte-for-byte center resample.  It remains available
    # while an outer anchor has genuinely unresolved signal.  A bounded warm
    # restart is the one exception: it already consumed the single independent
    # validation at these exact path bytes, so another generic anchor HOLD
    # would bypass the same-path terminal latch indefinitely.  It must prepare
    # the new-frame scalar-warm proposal (or terminate) without another band.
    # Once the anchor
    # is ready, uncertainty in the r interval is deliberately *not* a HOLD:
    # V17 keeps moving in the same frozen frame until the one-sided stop is
    # proved or the movement count reaches m_max.
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
        score_difference = math.nan
        difference_se = math.inf
        difference_threshold = math.inf
        resolved_lbfgs_advantage = False
        latch_before = bool(controller["lbfgs_latched"])
        latch_after = latch_before
        strict_win = False
        forced_lbfgs = False
        h0_fallback = False
        selected_score_se = math.nan
        selected_lcb = math.nan
        h0_raw = _candidate_raw_improvement_evidence(
            h0_candidate, statistics, controls
        )
        lbfgs_raw = _candidate_raw_improvement_evidence(
            lbfgs_candidate, statistics, controls
        )
        warm_validation_selected = False
        frozen_warm_candidate = None
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

        required_candidate = lbfgs_candidate if latch_before else h0_candidate
        if required_candidate.trust_globalization_exhausted:
            return None, {
                "phase": "TRUST_REFRESH_REQUIRED",
                "reason": "required_candidate_trust_globalization_exhausted",
                "required_candidate_label": required_candidate.label,
                "required_candidate_error": required_candidate.error,
                "required_candidate_digest": _candidate_digest(required_candidate),
                "globally_latched": latch_before,
            }
        if required_candidate.error in ANCHOR_CAP_STRUCTURAL_ERRORS:
            return None, {
                "phase": "TERMINAL",
                "reason": "required_candidate_anchor_cap_structural_failure",
                "required_candidate_label": required_candidate.label,
                "required_candidate_error": required_candidate.error,
                "required_candidate_digest": _candidate_digest(required_candidate),
                "globally_latched": latch_before,
            }

        # P_eff generated both directions.  Every inferential decision below
        # is instead centered on the same raw proximal-improvement target.
        h0_raw = _candidate_raw_improvement_evidence(
            h0_candidate, statistics, controls
        )
        lbfgs_raw = _candidate_raw_improvement_evidence(
            lbfgs_candidate, statistics, controls
        )
        latch_after = latch_before
        strict_win = False
        forced_lbfgs = False
        h0_fallback = False
        hold_selected = False
        hold_reason = "none"
        selected: Optional[Candidate] = None
        score_difference = math.nan
        difference_se = math.inf
        difference_threshold = math.inf
        resolved_lbfgs_advantage = False
        warm_validation_selected = False
        frozen_warm_candidate: Optional[Dict[str, Any]] = None
        if not latch_before and h0_candidate.valid and lbfgs_candidate.valid:
            # Both raw scores use the same source band, hence the shared-noise
            # coefficient is delta_LBFGS-delta_H0 rather than two independent
            # standard errors.
            difference_se = float(
                np.linalg.norm(
                    statistics.standard_error
                    * (
                        lbfgs_candidate.actual_normal_step
                        - h0_candidate.actual_normal_step
                    )
                )
            )
            score_difference = float(
                lbfgs_raw["score"] - h0_raw["score"]
            )
            difference_threshold = float(controls.secant_noise_z * difference_se)
            resolved_lbfgs_advantage = bool(
                np.isfinite(score_difference)
                and np.isfinite(difference_threshold)
                and score_difference > difference_threshold
            )

        if latch_before:
            if not lbfgs_candidate.valid:
                return None, {
                    "reason": "no_freezeable_warm_motion_candidate_terminal",
                    "phase": "TERMINAL",
                    "initial_no_se_special_case": False,
                    "globally_latched": True,
                    "h0_error": h0_candidate.error,
                    "warm_error": lbfgs_candidate.error,
                }
            selected_score_se = float(lbfgs_raw["standard_error"])
            selected_lcb = float(lbfgs_raw["lower_bound"])
            if selected_lcb > 0.0:
                selected = lbfgs_candidate
                forced_lbfgs = True
            else:
                current_path_hash = _array_sha256(controller["current_path"])
                if str(controller["warm_restart_used_path_sha256"]) == current_path_hash:
                    return None, {
                        "reason": "warm_candidate_failed_after_one_same_path_restart",
                        "phase": "TERMINAL",
                        "globally_latched": True,
                        "current_paid_path_sha256": current_path_hash,
                        "source_raw_improvement_lower_bound": selected_lcb,
                        "same_path_warm_restart_previously_used": True,
                    }
                frozen_warm_candidate = _freeze_warm_candidate(
                    controller,
                    lbfgs_candidate,
                    source_score=float(lbfgs_raw["score"]),
                    source_score_standard_error=selected_score_se,
                    source_score_lower_bound=selected_lcb,
                    controls=controls,
                )
                warm_validation_selected = True
                hold_selected = True
                hold_reason = "post_takeover_raw_improvement_lower_bound_nonpositive_single_validation"
        else:
            selected_score_se = math.nan
            selected_lcb = math.nan
            if initial_no_se_special_case:
                if h0_candidate.valid:
                    selected = h0_candidate
            elif (
                resolved_lbfgs_advantage
                and lbfgs_candidate.valid
                and float(lbfgs_raw["lower_bound"]) > 0.0
            ):
                selected = lbfgs_candidate
                selected_score_se = float(lbfgs_raw["standard_error"])
                selected_lcb = float(lbfgs_raw["lower_bound"])
                strict_win = True
                latch_after = True
            elif h0_candidate.valid and float(h0_raw["lower_bound"]) > 0.0:
                selected = h0_candidate
                selected_score_se = float(h0_raw["standard_error"])
                selected_lcb = float(h0_raw["lower_bound"])
            elif h0_candidate.valid and np.any(statistics.standard_error > 0.0):
                hold_selected = True
                hold_reason = "pre_takeover_h0_raw_improvement_lower_bound_nonpositive"
            else:
                return None, {
                    "reason": "no_valid_motion_candidate_terminal",
                    "phase": "TERMINAL",
                    "initial_no_se_special_case": initial_no_se_special_case,
                    "globally_latched": False,
                    "h0_error": h0_candidate.error,
                    "warm_error": lbfgs_candidate.error,
                    "h0_raw_improvement_lower_bound": h0_raw["lower_bound"],
                }

    if hold_selected:
        selected_label = (
            "WARM_VALIDATION_HOLD" if warm_validation_selected else "HOLD"
        )
        selected_path = np.asarray(controller["current_path"], dtype=np.float64).copy()
        if not np.array_equal(selected_path, controller["current_path"]):
            raise IRIVMError("HOLD failed exact path-byte reuse")
    else:
        if selected is None or selected.reparameterization is None:
            return None, {
                "reason": "no_structurally_valid_motion_candidate_terminal",
                "phase": "TERMINAL",
                "initial_no_se_special_case": initial_no_se_special_case,
                "globally_latched": bool(controller["lbfgs_latched"]),
                "h0_error": h0_candidate.error,
                "warm_error": lbfgs_candidate.error,
            }
        selected_label = selected.label
        selected_path = selected.reparameterization.path.copy()

    source_digest = _checkpoint_digest(controller)
    proposal_payload = {
        "schema": "Ala2DPreparedUpdate/v17-formal",
        "implementation_id": IMPLEMENTATION_ID,
        "source_checkpoint_sha256": source_digest,
        "source_effective_residual": _array_sha256(statistics.effective_residual),
        "source_raw_residual": _array_sha256(statistics.raw_residual),
        "h0_inverse_scale": _array_sha256(h0.inverse_scale),
        "h0_candidate": _candidate_digest(h0_candidate),
        "lbfgs_candidate": _candidate_digest(lbfgs_candidate),
        "h0_raw_improvement": {
            key: float(value).hex() for key, value in h0_raw.items()
        },
        "lbfgs_raw_improvement": {
            key: float(value).hex() for key, value in lbfgs_raw.items()
        },
        "selected_label": selected_label,
        "selected_path": _array_sha256(selected_path),
        "latch_before": latch_before,
        "latch_after": latch_after,
        "strict_lbfgs_win": strict_win,
        "hold_selected": hold_selected,
        "hold_reason": hold_reason,
        "warm_validation_selected": warm_validation_selected,
        "frozen_warm_validation_request_sha256": (
            "none"
            if frozen_warm_candidate is None
            else str(frozen_warm_candidate["request_sha256"])
        ),
        "global_signal_to_noise_binary64": float(global_snr).hex(),
        "lbfgs_raw_improvement_advantage_binary64": float(
            score_difference
        ).hex(),
        "raw_improvement_difference_standard_error_binary64": float(difference_se).hex(),
        "lbfgs_raw_advantage_resolution_threshold_binary64": float(
            difference_threshold
        ).hex(),
        "lbfgs_raw_advantage_resolved_positive": resolved_lbfgs_advantage,
        "initial_no_se_special_case": initial_no_se_special_case,
        "selected_raw_improvement_standard_error_binary64": float(
            selected_score_se
        ).hex(),
        "selected_raw_improvement_lower_bound_binary64": float(selected_lcb).hex(),
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
        h0_raw_improvement=h0_raw,
        lbfgs_raw_improvement=lbfgs_raw,
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
        lbfgs_raw_improvement_advantage=score_difference,
        raw_improvement_difference_standard_error=difference_se,
        lbfgs_raw_advantage_resolution_threshold=difference_threshold,
        lbfgs_raw_advantage_resolved_positive=resolved_lbfgs_advantage,
        initial_no_se_special_case=initial_no_se_special_case,
        selected_raw_improvement_standard_error=selected_score_se,
        selected_raw_improvement_lower_bound=selected_lcb,
        warm_validation_selected=warm_validation_selected,
        frozen_warm_commit=False,
        frozen_warm_candidate=frozen_warm_candidate,
    ), {}


def _rebuilt_outer_fields_at_current_paid(
    controller: Dict[str, Any], controls: IRIVMControls
) -> Dict[str, Any]:
    """Derive every frame-local field rebuilt by a zero-NoF outer refresh."""

    frame = _frozen_frame(
        controller["current_path"],
        controller["current_M"],
        controller["current_grad"],
        previous_basis=controller["frozen_basis"],
    )
    statistics = _residual_statistics(
        path=controller["current_path"],
        grad=controller["current_grad"],
        grad_blocks=controller["current_grad_blocks"],
        frozen_path=frame["path"],
        frozen_basis=frame["basis"],
        frozen_G=frame["G"],
        alpha=controls.alpha,
        bound_kappa=controls.residual_bound_kappa,
        single_block_exact=_single_block_is_exact(controller),
        block_frame_counts=controller["current_block_frame_counts"],
    )
    observations = {
        key: controller[key]
        for key in (
            "observation_paths",
            "observation_gradients",
            "observation_gradient_covariances",
            "observation_covariance_available",
            "observation_state_indices",
            "observation_count",
        )
    }
    rebuilt = _reproject_observation_history(
        observations,
        frame,
        controls,
        float(controller["warm_safeguard_inverse_scale"]),
        int(controller["warm_safeguard_source_state_index"]),
        int(controller["curvature_epoch_floor_state_index"]),
    )
    if statistics.bounds_available and statistics.lower_norm > controls.kkt_floor:
        anchor_lower = statistics.lower_norm
        anchor_upper = statistics.upper_norm
        bounds_ready = True
    else:
        anchor_lower = math.nan
        anchor_upper = math.nan
        bounds_ready = False

    return {
        "frozen_path": frame["path"],
        "frozen_M": frame["M"],
        "frozen_grad": frame["grad"],
        "frozen_tangent": frame["tangent"],
        "frozen_G": frame["G"],
        "frozen_basis": frame["basis"],
        "outer_initial_effective_rms": np.asarray(
            statistics.effective_rms, dtype=np.float64
        ),
        "outer_anchor_lower_norm": np.asarray(anchor_lower, dtype=np.float64),
        "outer_anchor_upper_norm": np.asarray(anchor_upper, dtype=np.float64),
        "outer_anchor_bounds_ready": bool(bounds_ready),
        "lbfgs_s": rebuilt["lbfgs_s"],
        "lbfgs_y": rebuilt["lbfgs_y"],
        "lbfgs_count": int(rebuilt["lbfgs_count"]),
        "warm_safeguard_inverse_scale": np.asarray(
            rebuilt["warm_safeguard_inverse_scale"], dtype=np.float64
        ),
        "warm_safeguard_source_state_index": int(
            rebuilt["warm_safeguard_source_state_index"]
        ),
        "lbfgs_anchor_u": statistics.normal_coordinate.copy(),
        "lbfgs_anchor_raw_residual": statistics.raw_residual.copy(),
        "lbfgs_anchor_standard_error": statistics.standard_error.copy(),
    }


def _zero_nof_outer_trust_refresh(
    controller: Dict[str, Any],
    controls: IRIVMControls,
    trigger: Dict[str, object],
) -> Tuple[Dict[str, Any], Dict[str, object]]:
    """Rebuild one outer frame at the current paid path without an oracle call."""

    _validate_controller(controller, controls)
    if trigger.get("phase") != "TRUST_REFRESH_REQUIRED":
        raise IRIVMError("zero-NoF trust refresh lacks the typed trigger")
    current_path_hash = _array_sha256(controller["current_path"])
    if str(controller["trust_refresh_used_path_sha256"]) == current_path_hash:
        raise IRIVMError("a second zero-NoF trust refresh at one path is forbidden")

    source_digest = _checkpoint_digest(controller)
    rebuilt_fields = _rebuilt_outer_fields_at_current_paid(controller, controls)

    refreshed = dict(controller)
    refreshed.update(
        rebuilt_fields,
        outer_index=int(controller["outer_index"]) + 1,
        inner_step=0,
        trust_refresh_used_path_sha256=current_path_hash,
    )
    _validate_controller(refreshed, controls)
    proposal_digest = _checkpoint_digest(refreshed)
    trigger_digest = hashlib.sha256(_canonical_json(trigger)).hexdigest()
    event: Dict[str, object] = {
        "schema": "Ala2DZeroNoFTrustRefresh/v17-formal",
        "implementation_id": IMPLEMENTATION_ID,
        "action": "refresh_frozen_outer_at_current_paid_path",
        "reason": "required_candidate_trust_globalization_exhausted",
        "required_candidate_label": str(trigger["required_candidate_label"]),
        "required_candidate_error": str(trigger["required_candidate_error"]),
        "required_candidate_digest": trigger["required_candidate_digest"],
        "trigger_sha256": trigger_digest,
        "source_checkpoint_sha256": source_digest,
        "proposal_controller_checkpoint_sha256": proposal_digest,
        "current_paid_path_sha256": current_path_hash,
        "state_index_before_and_after": int(controller["state_index"]),
        "nof_before_and_after": int(controller["nof"]),
        "outer_index_before": int(controller["outer_index"]),
        "outer_index_after": int(refreshed["outer_index"]),
        "inner_step_before": int(controller["inner_step"]),
        "inner_step_after": 0,
        "global_takeover_latched_before": bool(controller["lbfgs_latched"]),
        "global_takeover_latched_after": bool(refreshed["lbfgs_latched"]),
        "current_paid_arrays_byte_identical": all(
            _array_equal(controller[key], refreshed[key])
            for key in (
                "current_path",
                "current_M",
                "current_grad",
                "current_grad_blocks",
                "current_M_blocks",
                "current_block_frame_counts",
            )
        ),
        "observation_bank_byte_identical": all(
            _array_equal(controller[key], refreshed[key])
            for key in (
                "observation_paths",
                "observation_gradients",
                "observation_gradient_covariances",
                "observation_covariance_available",
                "observation_state_indices",
            )
        )
        and int(controller["observation_count"])
        == int(refreshed["observation_count"]),
        "mean_force_evaluations": 0,
        "same_path_refresh_limit": 1,
    }
    if not (
        event["current_paid_arrays_byte_identical"]
        and event["observation_bank_byte_identical"]
        and event["global_takeover_latched_before"]
        == event["global_takeover_latched_after"]
    ):
        raise IRIVMError("zero-NoF trust refresh changed protected paid state")
    return refreshed, event


def _frozen_outer_is_current_paid_frame(controller: Dict[str, Any]) -> bool:
    """Return whether an outer refresh would reproduce all frame-local state.

    Exact equality is intentional.  In particular, a same-path paid resample
    may leave aggregate M/g unchanged while changing block uncertainty or the
    observation bank; those changes can alter bounds or reconstructed history
    and therefore must still permit the one useful refresh.
    """

    controls = controller["controls"]
    rebuilt = _rebuilt_outer_fields_at_current_paid(controller, controls)
    return bool(
        int(controller["inner_step"]) == 0
        and all(_array_equal(controller[key], value) for key, value in rebuilt.items())
    )


def _prepare_update(
    controller: Dict[str, Any], controls: IRIVMControls
) -> PreparationOutcome:
    """Prepare a movement/HOLD, with at most one deterministic trust refresh."""

    _validate_controller(controller, controls)
    prepared, terminal = _prepare_update_once(controller, controls)
    if prepared is not None or terminal.get("phase") != "TRUST_REFRESH_REQUIRED":
        return PreparationOutcome(prepared, controller, terminal, {})

    current_path_hash = _array_sha256(controller["current_path"])
    if str(controller["trust_refresh_used_path_sha256"]) == current_path_hash:
        exhausted = {
            "phase": "TERMINAL",
            "reason": (
                "required_candidate_trust_globalization_exhausted_after_"
                "one_outer_refresh"
            ),
            "required_candidate_label": terminal["required_candidate_label"],
            "required_candidate_error": terminal["required_candidate_error"],
            "required_candidate_digest": terminal["required_candidate_digest"],
            "globally_latched": bool(controller["lbfgs_latched"]),
            "current_paid_path_sha256": current_path_hash,
            "zero_nof_outer_refresh_applied_this_preparation": False,
            "same_path_refresh_previously_used": True,
        }
        return PreparationOutcome(None, controller, exhausted, {})

    if _frozen_outer_is_current_paid_frame(controller):
        exhausted = {
            "phase": "TERMINAL",
            "reason": (
                "required_candidate_trust_globalization_exhausted_in_"
                "already_fresh_outer"
            ),
            "required_candidate_label": terminal["required_candidate_label"],
            "required_candidate_error": terminal["required_candidate_error"],
            "required_candidate_digest": terminal["required_candidate_digest"],
            "globally_latched": bool(controller["lbfgs_latched"]),
            "current_paid_path_sha256": current_path_hash,
            "zero_nof_outer_refresh_applied_this_preparation": False,
            "same_path_refresh_previously_used": False,
            "outer_was_already_fresh": True,
        }
        return PreparationOutcome(None, controller, exhausted, {})

    refreshed, event = _zero_nof_outer_trust_refresh(
        controller, controls, terminal
    )
    prepared_after, terminal_after = _prepare_update_once(refreshed, controls)
    if (
        prepared_after is None
        and terminal_after.get("phase") == "TRUST_REFRESH_REQUIRED"
    ):
        terminal_after = {
            "phase": "TERMINAL",
            "reason": (
                "required_candidate_trust_globalization_exhausted_after_"
                "one_outer_refresh"
            ),
            "required_candidate_label": terminal_after[
                "required_candidate_label"
            ],
            "required_candidate_error": terminal_after[
                "required_candidate_error"
            ],
            "required_candidate_digest": terminal_after[
                "required_candidate_digest"
            ],
            "globally_latched": bool(refreshed["lbfgs_latched"]),
            "current_paid_path_sha256": current_path_hash,
            "zero_nof_outer_refresh_applied_this_preparation": True,
            "same_path_refresh_previously_used": False,
            "zero_nof_event_sha256": hashlib.sha256(
                _canonical_json(event)
            ).hexdigest(),
        }
    elif prepared_after is None:
        terminal_after = dict(terminal_after)
        terminal_after["zero_nof_outer_refresh_applied_this_preparation"] = True
        terminal_after["zero_nof_event_sha256"] = hashlib.sha256(
            _canonical_json(event)
        ).hexdigest()
    return PreparationOutcome(
        prepared_after, refreshed, terminal_after, event
    )


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
    curvature_epoch_floor_state: int,
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
    if type(curvature_epoch_floor_state) is not int or curvature_epoch_floor_state < 0:
        raise IRIVMError("curvature epoch floor is invalid")
    _, _, radii = _trust_radii(frame["path"], controls)
    physical = np.linalg.norm(
        periodic_difference(paths[:, 1:-1], frame["path"][None, 1:-1]), axis=2
    )
    local = np.all(physical <= radii[None, :], axis=1)
    epoch_slots = np.flatnonzero(state_indices >= curvature_epoch_floor_state)
    if epoch_slots.size == 0 or int(epoch_slots[-1]) != count - 1:
        raise IRIVMError("current observation is outside the curvature epoch")
    epoch_start = int(epoch_slots[0])
    suffix_start = count - 1
    while suffix_start > epoch_start and local[suffix_start - 1]:
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
        "curvature_epoch_floor_state_index": int(curvature_epoch_floor_state),
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
    hold_resample = bool(prepared.hold_selected)
    warm_validation = bool(prepared.warm_validation_selected)
    if warm_validation:
        if (
            prepared.selected_label != "WARM_VALIDATION_HOLD"
            or not hold_resample
            or prepared.frozen_warm_candidate is None
        ):
            raise IRIVMError("warm validation preparation is internally inconsistent")
        validation = _fresh_raw_warm_validation_statistics(
            prepared.frozen_warm_candidate,
            raw_grad=weighted_raw_grad,
            raw_grad_blocks=raw_grad_blocks,
            raw_block_frame_counts=raw_block_frame_counts,
            controls=controls,
            single_block_exact=_single_block_is_exact(
                controller, int(controller["state_index"]) + 1
            ),
            validation_state_index=int(controller["state_index"]) + 1,
        )
    else:
        if prepared.selected_label == "WARM_VALIDATION_HOLD":
            raise IRIVMError("warm validation label lacks its frozen request")
        validation = {
            "classification": "not_applicable",
            "state_index": -1,
        }
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
    if movement and np.array_equal(path, controller["current_path"]):
        raise IRIVMError("a paid movement did not change the current path bytes")
    completed_inner = int(controller["inner_step"]) + int(movement)
    initial_rms = float(controller["outer_initial_effective_rms"])
    anchor_lower = float(controller["outer_anchor_lower_norm"])
    anchor_upper = float(controller["outer_anchor_upper_norm"])
    r_stop_decision_eligible = bool(
        source_bounds_ready and completed_inner > 0
    )
    if r_stop_decision_eligible:
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
    elif r_stop_decision_eligible and clear_r_stop:
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
        elif hold_resample:
            stop_reason = "movement_evidence_hold_resample_continue"
        elif residual_ambiguous:
            stop_reason = "continue_same_frozen_outer_r_stop_not_resolved"
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
            int(controller["curvature_epoch_floor_state_index"]),
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
                int(controller["curvature_epoch_floor_state_index"]),
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

    # A positive independent validation is checkpointed before its target is
    # evaluated, so a crash cannot regenerate or alter the adaptive candidate.
    # Bad and unresolved outcomes instead perform one zero-NoF warm restart at
    # the now-pooled source path.  The r-stop above has precedence over either
    # action because the fresh exact-center band first becomes paid source data.
    pending_warm_fields = _empty_pending_warm_fields()
    curvature_epoch_floor_next = int(
        controller["curvature_epoch_floor_state_index"]
    )
    warm_restart_path_sha_next = (
        "none"
        if movement
        else str(controller["warm_restart_used_path_sha256"])
    )
    warm_validation_disposition = "not_applicable"
    warm_restart_event: Dict[str, object] = {}
    if warm_validation:
        classification = str(validation["classification"])
        if classification not in WARM_VALIDATION_CLASSIFICATIONS - {"not_applicable"}:
            raise IRIVMError("fresh warm validation has an unknown classification")
        if r_stop:
            warm_validation_disposition = "discarded_by_pooled_r_stop"
        elif classification == "validated_positive":
            if stopped:
                raise IRIVMError(
                    "positive warm validation stopped without the pooled r criterion"
                )
            pending_warm_fields = _pending_warm_fields_from_validated(
                prepared.frozen_warm_candidate, validation
            )
            warm_validation_disposition = "frozen_candidate_pending_commit"
            stop_reason = "positive_warm_validation_pending_frozen_commit"
        else:
            current_path_sha = _array_sha256(path)
            if str(controller["warm_restart_used_path_sha256"]) == current_path_sha:
                raise IRIVMError("a second zero-NoF warm restart at one path is forbidden")
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
                single_block_exact=_single_block_is_exact(controller, next_state),
                block_frame_counts=block_frame_counts,
            )
            lbfgs_s = np.zeros(
                (JOINT_HISTORY_SIZE, N_INTERNAL), dtype=np.float64
            )
            lbfgs_y = np.zeros_like(lbfgs_s)
            lbfgs_count = 0
            lbfgs_latched = True
            safeguard_scale = float(controls.alpha)
            safeguard_source = next_state
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
            pair_stored = False
            stopped = True
            r_stop = False
            stop_reason = (
                "confirmed_bad_warm_zero_nof_restart"
                if classification == "confirmed_bad"
                else "unresolved_warm_after_one_validation_zero_nof_restart"
            )
            curvature_epoch_floor_next = next_state
            warm_restart_path_sha_next = current_path_sha
            warm_validation_disposition = "zero_nof_warm_restart"
            warm_restart_event = {
                "schema": "Ala2DZeroNoFWarmRestart/v17-formal",
                "implementation_id": IMPLEMENTATION_ID,
                "action": "reset_warm_memory_at_current_pooled_path",
                "validation_classification": classification,
                "validation_request_sha256": str(
                    prepared.frozen_warm_candidate["request_sha256"]
                ),
                "frozen_candidate_sha256": str(
                    prepared.frozen_warm_candidate["candidate_sha256"]
                ),
                "current_paid_path_sha256": current_path_sha,
                "state_index_before": int(controller["state_index"]),
                "state_index_after": next_state,
                "nof_before": int(controller["nof"]),
                "nof_after": int(controller["nof"]) + N_INTERNAL,
                "outer_index_before": int(controller["outer_index"]),
                "outer_index_after": next_outer,
                "curvature_epoch_floor_state_index": next_state,
                "warm_safeguard_inverse_scale_binary64": float(
                    controls.alpha
                ).hex(),
                "global_takeover_latched_after": True,
                "old_pairs_retained": 0,
                "additional_mean_force_evaluations": 0,
                "same_path_restart_limit": 1,
            }

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
        "curvature_epoch_floor_state_index": curvature_epoch_floor_next,
        "single_block_exact_after_initial": bool(
            controller["single_block_exact_after_initial"]
        ),
        "trust_refresh_used_path_sha256": (
            "none"
            if movement
            else str(controller["trust_refresh_used_path_sha256"])
        ),
        "warm_restart_used_path_sha256": warm_restart_path_sha_next,
        "lbfgs_anchor_u": next_statistics.normal_coordinate.copy(),
        "lbfgs_anchor_raw_residual": next_statistics.raw_residual.copy(),
        "lbfgs_anchor_standard_error": next_statistics.standard_error.copy(),
        "controls": controls,
    }
    new_controller.update(pending_warm_fields)
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
        "warm_validation_selected": warm_validation,
        "warm_validation": validation,
        "warm_validation_disposition": warm_validation_disposition,
        "warm_restart_event": warm_restart_event,
        "pending_warm_status_next": str(new_controller["pending_warm_status"]),
        "curvature_epoch_floor_state_index_next": curvature_epoch_floor_next,
        "warm_restart_used_path_sha256_next": warm_restart_path_sha_next,
        "oracle_raw_M": raw_M,
        "oracle_raw_grad": raw_grad,
        "oracle_raw_grad_blocks": raw_grad_blocks,
        "oracle_raw_M_blocks": raw_M_blocks,
        "oracle_raw_block_frame_counts": raw_block_frame_counts,
        "movement_committed": movement,
        "source_anchor_bounds_ready": source_bounds_ready,
        "r_stop_decision_eligible_after_at_least_one_movement": (
            r_stop_decision_eligible
        ),
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
        f"{prefix}candidate_structural_valid": np.asarray(
            candidate.valid, dtype=bool
        ),
        f"{prefix}candidate_error": np.asarray(candidate.error),
        f"{prefix}raw_direction": candidate.raw_direction,
        f"{prefix}capped_direction": candidate.capped_direction,
        f"{prefix}whole_band_common_cap_scale": np.asarray(candidate.common_cap_scale),
        f"{prefix}whole_band_common_cap_truncated": np.asarray(
            candidate.common_cap_truncated, dtype=bool
        ),
        f"{prefix}anchor_aware_common_cap": np.asarray(True, dtype=bool),
        f"{prefix}anchor_cap_theoretical_scale": np.asarray(
            candidate.anchor_cap_theoretical_scale
        ),
        f"{prefix}anchor_cap_binary_bracket_used": np.asarray(
            candidate.anchor_cap_binary_bracket_used, dtype=bool
        ),
        f"{prefix}anchor_cap_binary_bracket_iterations": np.asarray(
            candidate.anchor_cap_binary_bracket_iterations, dtype=np.int64
        ),
        f"{prefix}anchor_cap_source_distances": candidate.anchor_cap_source_distances,
        f"{prefix}anchor_cap_raw_path_distances": candidate.anchor_cap_raw_path_distances,
        f"{prefix}post_r_backtrack_count": np.asarray(
            candidate.post_r_backtrack_count, dtype=np.int64
        ),
        f"{prefix}post_r_backtrack_factor": np.asarray(
            candidate.post_r_backtrack_factor
        ),
        f"{prefix}total_direction_scale": np.asarray(
            candidate.total_direction_scale
        ),
        f"{prefix}post_r_attempt_count": np.asarray(
            candidate.post_r_attempt_count, dtype=np.int64
        ),
        f"{prefix}post_r_attempt_scales": candidate.post_r_attempt_scales,
        f"{prefix}post_r_attempt_status": candidate.post_r_attempt_status,
        f"{prefix}post_r_attempt_raw_path_sha256": (
            candidate.post_r_attempt_raw_path_sha256
        ),
        f"{prefix}post_r_attempt_reparameterization_audit_sha256": (
            candidate.post_r_attempt_reparameterization_audit_sha256
        ),
        f"{prefix}post_r_attempt_max_anchor_distance": (
            candidate.post_r_attempt_max_anchor_distance
        ),
        f"{prefix}trust_globalization_exhausted": np.asarray(
            candidate.trust_globalization_exhausted, dtype=bool
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
        f"{prefix}proposal_diagnostic_effective_linear_term": np.asarray(
            candidate.proposal_effective_linear_term
        ),
        f"{prefix}proposal_diagnostic_quadratic_penalty": np.asarray(
            candidate.proposal_quadratic_penalty
        ),
        f"{prefix}proposal_diagnostic_effective_proximal_value": np.asarray(
            candidate.proposal_effective_proximal_value
        ),
    }


def _paid_state_kind(prepared: PreparedUpdate) -> str:
    if prepared.warm_validation_selected:
        return WARM_VALIDATION_STATE_KIND
    return HOLD_STATE_KIND if prepared.hold_selected else STATE_KIND


def _raw_improvement_arrays(
    prefix: str, evidence: Dict[str, float]
) -> Dict[str, Array]:
    """Serialize the sole inferential proximal-improvement target.

    These values are always centered on ``P_raw``.  Candidate fields centered
    on ``P_eff`` are direction-construction diagnostics only.
    """

    return {
        f"{prefix}raw_proximal_linear_term": np.asarray(evidence["linear"]),
        f"{prefix}raw_proximal_quadratic_penalty": np.asarray(
            evidence["quadratic"]
        ),
        f"{prefix}raw_proximal_improvement_mean": np.asarray(evidence["score"]),
        f"{prefix}raw_proximal_improvement_standard_error": np.asarray(
            evidence["standard_error"]
        ),
        f"{prefix}raw_proximal_improvement_lower_bound": np.asarray(
            evidence["lower_bound"]
        ),
        f"{prefix}raw_proximal_improvement_upper_bound": np.asarray(
            evidence["upper_bound"]
        ),
    }


def _warm_validation_arrays(
    before: Dict[str, Any],
    after: Dict[str, Any],
    prepared: PreparedUpdate,
    advance: Dict[str, Any],
) -> Dict[str, Array]:
    selected = bool(prepared.warm_validation_selected)
    validation = advance["warm_validation"]
    frozen = prepared.frozen_warm_candidate if selected else None
    if selected and frozen is None:
        raise IRIVMError("warm validation public state lacks its frozen candidate")
    fresh_vector = (
        np.asarray(validation["fresh_raw_residual"], dtype=np.float64)
        if selected
        else np.full(N_INTERNAL, np.nan)
    )
    fresh_se = (
        np.asarray(
            validation["fresh_raw_residual_standard_error"], dtype=np.float64
        )
        if selected
        else np.full(N_INTERNAL, np.nan)
    )
    if selected:
        candidate_sha = str(frozen["candidate_sha256"])
        frame_sha = str(frozen["frame_sha256"])
        request_sha = str(frozen["request_sha256"])
        source_checkpoint_sha = str(frozen["source_checkpoint_sha256"])
        frozen_path = np.asarray(frozen["path"], dtype=np.float64)
        frozen_delta = np.asarray(frozen["actual_normal_step"], dtype=np.float64)
        source_score = float(frozen["source_score"])
        source_se = float(frozen["source_score_standard_error"])
        source_lower = float(frozen["source_score_lower_bound"])
    elif prepared.frozen_warm_commit:
        candidate_sha = str(before["pending_warm_candidate_sha256"])
        frame_sha = str(before["pending_warm_frame_sha256"])
        request_sha = str(before["pending_warm_request_sha256"])
        source_checkpoint_sha = str(
            before["pending_warm_source_checkpoint_sha256"]
        )
        frozen_path = np.asarray(before["pending_warm_path"], dtype=np.float64)
        frozen_delta = np.asarray(
            before["pending_warm_actual_normal_step"], dtype=np.float64
        )
        source_score = float(before["pending_warm_source_score"])
        source_se = float(before["pending_warm_source_score_standard_error"])
        source_lower = float(before["pending_warm_source_score_lower_bound"])
    else:
        candidate_sha = frame_sha = request_sha = source_checkpoint_sha = "none"
        frozen_path = np.full((N_PATH, DIMENSION), np.nan)
        frozen_delta = np.full(N_INTERNAL, np.nan)
        source_score = source_se = source_lower = math.nan
    restart_event = advance["warm_restart_event"]
    return {
        "warm_validation_selected": np.asarray(selected, dtype=bool),
        "warm_validation_formula": np.asarray(
            "Psi_raw(delta)=-P_raw.T*delta-||delta||^2/(2*alpha)"
        ),
        "warm_validation_classification": np.asarray(
            str(validation["classification"])
        ),
        "warm_validation_disposition": np.asarray(
            str(advance["warm_validation_disposition"])
        ),
        "warm_validation_frozen_candidate_sha256": np.asarray(candidate_sha),
        "warm_validation_frozen_frame_sha256": np.asarray(frame_sha),
        "warm_validation_request_sha256": np.asarray(request_sha),
        "warm_validation_source_checkpoint_sha256": np.asarray(
            source_checkpoint_sha
        ),
        "warm_validation_frozen_candidate_path_unwrapped": frozen_path,
        "warm_validation_frozen_actual_normal_step": frozen_delta,
        "warm_validation_source_raw_improvement_mean": np.asarray(source_score),
        "warm_validation_source_raw_improvement_standard_error": np.asarray(
            source_se
        ),
        "warm_validation_source_raw_improvement_lower_bound": np.asarray(
            source_lower
        ),
        "warm_validation_fresh_raw_improvement_mean": np.asarray(
            validation.get("score_mean", math.nan)
        ),
        "warm_validation_fresh_raw_improvement_standard_error": np.asarray(
            validation.get("score_standard_error", math.nan)
        ),
        "warm_validation_fresh_raw_improvement_lower_bound": np.asarray(
            validation.get("score_lower_bound", math.nan)
        ),
        "warm_validation_fresh_raw_improvement_upper_bound": np.asarray(
            validation.get("score_upper_bound", math.nan)
        ),
        "warm_validation_fresh_raw_residual": fresh_vector,
        "warm_validation_fresh_raw_residual_standard_error": fresh_se,
        "warm_validation_fresh_raw_block_count": np.asarray(
            validation.get("fresh_raw_block_count", 0), dtype=np.int64
        ),
        "warm_validation_used_pooled_source_estimate": np.asarray(
            validation.get("used_pooled_source_estimate", False), dtype=bool
        ),
        "warm_validation_used_shrinkage_centered_residual": np.asarray(
            validation.get("used_shrinkage_centered_residual", False), dtype=bool
        ),
        "warm_validation_band_mean_force_cost": np.asarray(
            N_INTERNAL if selected else 0, dtype=np.int64
        ),
        "warm_validation_incremented_movement_count": np.asarray(False, dtype=bool),
        "warm_validation_pooled_r_stop_precedence": np.asarray(
            selected and bool(advance["r_stop"]), dtype=bool
        ),
        "frozen_warm_commit_selected": np.asarray(
            prepared.frozen_warm_commit, dtype=bool
        ),
        "pending_warm_status_before": np.asarray(
            str(before["pending_warm_status"])
        ),
        "pending_warm_status_next": np.asarray(str(after["pending_warm_status"])),
        "pending_warm_candidate_sha256_before": np.asarray(
            str(before["pending_warm_candidate_sha256"])
        ),
        "pending_warm_candidate_sha256_next": np.asarray(
            str(after["pending_warm_candidate_sha256"])
        ),
        "zero_nof_warm_restart_applied": np.asarray(
            bool(restart_event), dtype=bool
        ),
        "zero_nof_warm_restart_event_json": np.asarray(
            _canonical_json(restart_event).decode("ascii")
        ),
        "curvature_epoch_floor_state_index_before": np.asarray(
            before["curvature_epoch_floor_state_index"], dtype=np.int64
        ),
        "curvature_epoch_floor_state_index_next": np.asarray(
            after["curvature_epoch_floor_state_index"], dtype=np.int64
        ),
        "warm_restart_used_path_sha256_before": np.asarray(
            str(before["warm_restart_used_path_sha256"])
        ),
        "warm_restart_used_path_sha256_next": np.asarray(
            str(after["warm_restart_used_path_sha256"])
        ),
    }


def _initial_state_arrays(controller: Dict[str, Any]) -> Dict[str, Array]:
    """Reconstruct the complete deterministic common-initial public record."""

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
        "development_only": np.asarray(False, dtype=bool),
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
        "trust_refresh_used_path_sha256": np.asarray("none"),
        "warm_restart_used_path_sha256": np.asarray("none"),
        "curvature_epoch_floor_state_index": np.asarray(0, dtype=np.int64),
        "pending_warm_status": np.asarray(WARM_VALIDATION_PENDING_NONE),
        "pending_warm_candidate_sha256": np.asarray("none"),
        "zero_nof_trust_refresh_applied": np.asarray(False, dtype=bool),
        "zero_nof_trust_refresh_event_json": np.asarray("{}"),
        "anchor_cap_bracket_iterations_fixed": np.asarray(
            ANCHOR_CAP_BRACKET_ITERATIONS, dtype=np.int64
        ),
        "post_r_backtrack_limit_fixed": np.asarray(
            POST_R_BACKTRACK_LIMIT, dtype=np.int64
        ),
    }
    arrays.update(_residual_arrays("", statistics))
    arrays.update(metrics)
    return arrays


def _save_initial_state(
    run_directory: Path,
    controller: Dict[str, Any],
    initial_M_blocks: Array,
) -> None:
    if not _array_equal(initial_M_blocks, controller["current_M_blocks"]):
        raise IRIVMError("initial metric blocks differ from the controller")
    arrays = _initial_state_arrays(controller)
    _atomic_publish_npz_once(_state_path(run_directory, 0), **arrays)


def _save_public_state(
    run_directory: Path,
    before: Dict[str, Any],
    after: Dict[str, Any],
    prepared: PreparedUpdate,
    zero_nof_event: Dict[str, object],
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
        expected_hold_label = (
            "WARM_VALIDATION_HOLD"
            if prepared.warm_validation_selected
            else "HOLD"
        )
        if prepared.selected_label != expected_hold_label or not np.array_equal(
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
    elif prepared.warm_validation_selected:
        phase_before = "INDEPENDENT_WARM_VALIDATION"
    elif prepared.hold_selected and not bool(before["outer_anchor_bounds_ready"]):
        phase_before = "ANCHORING"
    elif prepared.hold_selected:
        phase_before = "HOLD"
    else:
        phase_before = "MOVE_PREPARED"
    if str(after["pending_warm_status"]) == WARM_VALIDATION_PENDING_COMMIT:
        phase_after = "VALIDATED_FROZEN_WARM_PENDING_COMMIT"
    elif bool(advance["warm_restart_event"]):
        phase_after = "WARM_RESTART_READY"
    else:
        phase_after = (
            "READY" if bool(after["outer_anchor_bounds_ready"]) else "ANCHORING"
        )
    current_geometry = build_metric_geometry(after["current_path"], after["current_M"])
    arrays: Dict[str, Array] = {
        "implementation_id": np.asarray(IMPLEMENTATION_ID),
        "development_only": np.asarray(False, dtype=bool),
        "state_index": np.asarray(state_index, dtype=np.int64),
        "nof_mean_force_incremental": np.asarray(after["nof"], dtype=np.int64),
        "nof_mean_force_absolute_including_shared_initial": np.asarray(
            after["nof"] + N_INTERNAL, dtype=np.int64
        ),
        "outer_index": np.asarray(advance["source_outer_index"], dtype=np.int64),
        "inner_step": np.asarray(advance["completed_inner_step"], dtype=np.int64),
        "state_kind": np.asarray(_paid_state_kind(prepared)),
        "proposal_sha256": np.asarray(prepared.proposal_sha256),
        "source_checkpoint_sha256": np.asarray(prepared.source_checkpoint_sha256),
        "proposal_controller_checkpoint_sha256": np.asarray(
            _checkpoint_digest(before)
        ),
        "zero_nof_trust_refresh_applied": np.asarray(
            bool(zero_nof_event), dtype=bool
        ),
        "zero_nof_trust_refresh_event_json": np.asarray(
            _canonical_json(zero_nof_event).decode("ascii")
        ),
        "zero_nof_trust_refresh_source_checkpoint_sha256": np.asarray(
            str(zero_nof_event.get("source_checkpoint_sha256", "none"))
        ),
        "zero_nof_trust_refresh_path_sha256": np.asarray(
            str(zero_nof_event.get("current_paid_path_sha256", "none"))
        ),
        "trust_refresh_used_path_sha256_before": np.asarray(
            str(before["trust_refresh_used_path_sha256"])
        ),
        "trust_refresh_used_path_sha256_next": np.asarray(
            str(after["trust_refresh_used_path_sha256"])
        ),
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
        "lbfgs_raw_proximal_improvement_advantage": np.asarray(
            prepared.lbfgs_raw_improvement_advantage
        ),
        "initial_no_se_special_case_used": np.asarray(
            prepared.initial_no_se_special_case, dtype=bool
        ),
        "raw_proximal_improvement_difference_standard_error": np.asarray(
            prepared.raw_improvement_difference_standard_error
        ),
        "lbfgs_raw_advantage_resolution_threshold": np.asarray(
            prepared.lbfgs_raw_advantage_resolution_threshold
        ),
        "lbfgs_raw_advantage_resolved_positive": np.asarray(
            prepared.lbfgs_raw_advantage_resolved_positive, dtype=bool
        ),
        "selected_raw_proximal_improvement_standard_error": np.asarray(
            prepared.selected_raw_improvement_standard_error
        ),
        "selected_raw_proximal_improvement_lower_bound": np.asarray(
            prepared.selected_raw_improvement_lower_bound
        ),
        "normal_mapping_recomputed_after_reparameterization": np.asarray(
            not prepared.hold_selected, dtype=bool
        ),
        "candidate_reparameterization_calls": np.asarray(
            prepared.h0_candidate.post_r_attempt_count
            + prepared.lbfgs_candidate.post_r_attempt_count,
            dtype=np.int64,
        ),
        "evaluated_candidate_count": np.asarray(1, dtype=np.int64),
        "preselection_mean_force_evaluations": np.asarray(0, dtype=np.int64),
        "only_selected_complete_band_evaluated": np.asarray(True, dtype=bool),
        "reparameterizer_id": np.asarray(REPARAMETERIZER_ID),
        "anchor_cap_bracket_iterations_fixed": np.asarray(
            ANCHOR_CAP_BRACKET_ITERATIONS, dtype=np.int64
        ),
        "post_r_backtrack_limit_fixed": np.asarray(
            POST_R_BACKTRACK_LIMIT, dtype=np.int64
        ),
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
        "r_stop_decision_eligible_after_at_least_one_movement": np.asarray(
            advance["r_stop_decision_eligible_after_at_least_one_movement"],
            dtype=bool,
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
        "r_stop_is_one_sided_confident_stop": np.asarray(True, dtype=bool),
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
    arrays.update(_raw_improvement_arrays("h0_", prepared.h0_raw_improvement))
    arrays.update(
        _raw_improvement_arrays("lbfgs_", prepared.lbfgs_raw_improvement)
    )
    arrays.update(_warm_validation_arrays(before, after, prepared, advance))
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
        "development_only": False,
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
        "algorithm_variant": (
            "v17_formal_raw_consistent_single_validation_bounded_warm_restart"
        ),
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
            "registered evidence interval L=norm((abs(P)-2.262*SE)_+), "
            "U=norm(abs(P)+2.262*SE)"
        ),
        "registered_evidence_margin": controls.residual_bound_kappa,
        "registered_evidence_margin_is_confidence_coverage_claim": False,
        "H0": (
            "per-image diagonal inverse scale from historical local weighted "
            "gradient-curvature regression; invalid entries equal alpha"
        ),
        "joint_lbfgs_history_size": JOINT_HISTORY_SIZE,
        "secant": "actual_post_reparameterization_delta_u, delta_raw_proximal_residual",
        "secant_noise_gate": "sTy > 2*sqrt(sum((s*SE_delta_y)^2)) plus cosine gate",
        "selector": (
            "66D global one-way L-BFGS takeover only when D=Psi_LBFGS-Psi_H0 "
            "is greater than secant_noise_z*SE_D, eligible only after at least "
            "two current-frame noise-qualified distinct post-R secants; H0 is "
            "used before an unresolved takeover and cold recovery is permanently "
            "forbidden after takeover"
        ),
        "post_takeover_fallback": (
            "current-frame L-BFGS when at least two pairs exist, otherwise the "
            "last sTy/yTy scalar clipped to [0.25 alpha,4 alpha]; never H0; "
            "a geometrically invalid warm candidate terminates; a freezeable "
            "candidate with nonpositive raw lower bound receives exactly one "
            "independent exact-center validation"
        ),
        "physical_cap": (
            "one anchor-aware common ray/ball scale for all 20 images, "
            "accounting for current-to-frozen-anchor accumulated displacement"
        ),
        "anchor_cap_binary_bracket_iterations": ANCHOR_CAP_BRACKET_ITERATIONS,
        "anchor_cap_binary_bracket_policy": (
            "strict actual-raw-path check, then exactly 64 binary64 bracket "
            "iterations on [0,theoretical_scale] if needed"
        ),
        "anchor_cap_failure_typing": (
            "a verified capped direction at or below secant_norm_floor is trust "
            "exhaustion; an outside paid source or nonfinite/invalid quadratic "
            "is a structural terminal and cannot refresh or HOLD"
        ),
        "post_article_reparameterization_backtrack_factor": 0.5,
        "post_article_reparameterization_backtrack_limit": POST_R_BACKTRACK_LIMIT,
        "post_article_reparameterization_backtrack_policy": (
            "try total scales scale/2**j for j=0..32; reconstruct and strictly "
            "verify raw and article-reparameterized paths at every attempt; "
            "no projection and no radius tolerance"
        ),
        "required_candidate_trust_exhaustion_policy": (
            "required H0 before takeover or required warm candidate after takeover "
            "may trigger exactly one zero-NoF outer refresh at a nonfresh unchanged "
            "paid path; nonrequired pre-takeover warm exhaustion cannot trigger "
            "refresh; an already-fresh outer or a second same-path exhaustion "
            "terminates durably"
        ),
        "zero_nof_trust_refresh_policy": (
            "preserve state_index, NoF, paid arrays, observations and global latch; "
            "increment outer_index, reset inner_step, rebuild frozen frame, bounds, "
            "history and warm anchor; persist the exact current-path marker until "
            "a genuine movement"
        ),
        "reparameterizer_id": REPARAMETERIZER_ID,
        "candidate_reparameterization_before_selection": True,
        "post_reparameterization_actual_u_recomputed": True,
        "proposal_direction_diagnostic": (
            "P_eff may generate a direction and is recorded only as the "
            "noninferential value -P_eff dot delta_u-||delta_u||^2/(2*alpha)"
        ),
        "decision_proximal_improvement": (
            "every source, comparison, validation and movement decision uses "
            "the common raw target Psi_raw=-P_raw dot delta_u-"
            "||delta_u||^2/(2*alpha), with actual post-reparameterization delta_u"
        ),
        "independent_warm_validation": (
            "after global takeover, source L_raw<=0 freezes path, delta, outer "
            "frame and hashes; exactly one fresh exact-center 20-NoF band is "
            "scored from only its raw blocks against the same frozen target"
        ),
        "warm_validation_three_way_rule": (
            "fresh L_raw>0 commits the original frozen candidate after the "
            "validation checkpoint; fresh U_raw<=0 is confirmed_bad; otherwise "
            "unresolved; pooled r-stop has precedence"
        ),
        "bounded_warm_restart": (
            "confirmed_bad or unresolved clears history, advances a persistent "
            "curvature epoch, refreshes the current paid frame, fixes the warm "
            "scalar to alpha, preserves takeover, and is allowed once per exact path"
        ),
        "low_snr_policy": (
            "exact-center HOLD/resample only for an unresolved anchor or "
            "nonpositive pre-takeover H0 raw movement evidence; after takeover, "
            "a byte-identical exact-center band is a single independent frozen-"
            "candidate validation, never an indefinite HOLD"
        ),
        "ambiguous_candidate_policy": (
            "choose valid H0 when the positive L-BFGS score advantage is not "
            "resolved above secant_noise_z shared-source standard errors; never "
            "HOLD solely to arbitrate H0 versus L-BFGS"
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
        "relative_residual_continue": (
            "whenever the one-sided stop U(P_j) <= r*L(P_anchor) is not proved"
        ),
        "relative_residual_ambiguous_action": "continue_movement_same_frozen_outer",
        "r_stop_is_one_sided_confident_stop": True,
        "r_stop_precedes_m_max_stop": True,
        "performance_metrics_visible_to_optimizer": False,
        "next_action_is_independent_of_declared_budget": True,
        "atomic_immutable_public_states": True,
        "atomic_replaceable_checkpoint": True,
        "one_state_ahead_deterministic_recovery": True,
        "same_index_zero_nof_refresh_checkpoint_published": False,
        "zero_nof_refresh_rederived_on_replay_and_terminal_resume": True,
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
    source_controller = controller
    outcome = _prepare_update(source_controller, controls)
    prepared = outcome.prepared
    if prepared is None:
        raise IRIVMError(
            "public state exists after deterministic zero-NoF stop: "
            f"{outcome.terminal}"
        )
    controller = outcome.proposal_controller
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
        "proposal_controller_checkpoint_sha256": np.asarray(
            _checkpoint_digest(controller)
        ),
        "zero_nof_trust_refresh_applied": np.asarray(
            bool(outcome.zero_nof_event)
        ),
        "zero_nof_trust_refresh_event_json": np.asarray(
            _canonical_json(outcome.zero_nof_event).decode("ascii")
        ),
        "zero_nof_trust_refresh_source_checkpoint_sha256": np.asarray(
            str(
                outcome.zero_nof_event.get(
                    "source_checkpoint_sha256", "none"
                )
            )
        ),
        "zero_nof_trust_refresh_path_sha256": np.asarray(
            str(
                outcome.zero_nof_event.get(
                    "current_paid_path_sha256", "none"
                )
            )
        ),
    }
    for key, value in exact.items():
        if key not in row or not np.array_equal(row[key], value):
            raise IRIVMError(
                f"public state {expected_index} differs from deterministic V17 proposal: {key}"
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
            "INDEPENDENT_WARM_VALIDATION"
            if prepared.warm_validation_selected
            else (
            "ANCHORING"
            if prepared.hold_selected
            and not bool(controller["outer_anchor_bounds_ready"])
            else "HOLD" if prepared.hold_selected else "MOVE_PREPARED"
            )
        )
    )
    if str(next_controller["pending_warm_status"]) == WARM_VALIDATION_PENDING_COMMIT:
        phase_after = "VALIDATED_FROZEN_WARM_PENDING_COMMIT"
    elif bool(advance["warm_restart_event"]):
        phase_after = "WARM_RESTART_READY"
    else:
        phase_after = (
            "READY"
            if bool(next_controller["outer_anchor_bounds_ready"])
            else "ANCHORING"
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
        "state_kind": np.asarray(_paid_state_kind(prepared)),
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
            prepared.h0_candidate.post_r_attempt_count
            + prepared.lbfgs_candidate.post_r_attempt_count
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
        "r_stop_decision_eligible_after_at_least_one_movement": np.asarray(
            advance["r_stop_decision_eligible_after_at_least_one_movement"]
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
        "r_stop_is_one_sided_confident_stop": np.asarray(True),
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
        "lbfgs_raw_proximal_improvement_advantage": np.asarray(
            prepared.lbfgs_raw_improvement_advantage
        ),
        "raw_proximal_improvement_difference_standard_error": np.asarray(
            prepared.raw_improvement_difference_standard_error
        ),
        "lbfgs_raw_advantage_resolution_threshold": np.asarray(
            prepared.lbfgs_raw_advantage_resolution_threshold
        ),
        "lbfgs_raw_advantage_resolved_positive": np.asarray(
            prepared.lbfgs_raw_advantage_resolved_positive
        ),
        "selected_raw_proximal_improvement_standard_error": np.asarray(
            prepared.selected_raw_improvement_standard_error
        ),
        "selected_raw_proximal_improvement_lower_bound": np.asarray(
            prepared.selected_raw_improvement_lower_bound
        ),
        "reparameterizer_id": np.asarray(REPARAMETERIZER_ID),
        "anchor_cap_bracket_iterations_fixed": np.asarray(
            ANCHOR_CAP_BRACKET_ITERATIONS
        ),
        "post_r_backtrack_limit_fixed": np.asarray(POST_R_BACKTRACK_LIMIT),
        "trust_refresh_used_path_sha256_before": np.asarray(
            str(controller["trust_refresh_used_path_sha256"])
        ),
        "trust_refresh_used_path_sha256_next": np.asarray(
            str(next_controller["trust_refresh_used_path_sha256"])
        ),
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
    expected_advance.update(
        _raw_improvement_arrays("h0_", prepared.h0_raw_improvement)
    )
    expected_advance.update(
        _raw_improvement_arrays("lbfgs_", prepared.lbfgs_raw_improvement)
    )
    expected_advance.update(
        _warm_validation_arrays(controller, next_controller, prepared, advance)
    )
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
        "curvature_epoch_floor_state_index",
        "trust_refresh_used_path_sha256",
        "warm_restart_used_path_sha256",
        "single_block_exact_after_initial",
        "lbfgs_anchor_u",
        "lbfgs_anchor_raw_residual",
        "lbfgs_anchor_standard_error",
    ) + PENDING_WARM_STRING_KEYS + PENDING_WARM_INTEGER_KEYS + PENDING_WARM_ARRAY_KEYS
    return all(_array_equal(left[key], right[key]) for key in keys)


def load_public_trajectory(run_directory: Path) -> Dict[str, Any]:
    """Validate and deterministically replay every durable complete V17 band."""

    run_directory = Path(run_directory)
    state_paths = sorted((run_directory / "states").glob("state_*.npz"))
    if not state_paths:
        raise FileNotFoundError("run has no public V17 states")
    manifest_path = run_directory / "run_manifest.json"
    if not manifest_path.is_file():
        raise IRIVMError("public V17 states exist without a run manifest")
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
        raise IRIVMError("public V17 state indices are not contiguous")
    nof = np.asarray(
        [int(row["nof_mean_force_incremental"]) for row in rows], dtype=np.int64
    )
    if not np.array_equal(nof, indices * N_INTERNAL):
        raise IRIVMError("public V17 NoF ledger is not exactly 20 per paid band")
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
    expected_initial = _initial_state_arrays(controller)
    if set(initial) != set(expected_initial):
        missing = sorted(set(expected_initial) - set(initial))
        unexpected = sorted(set(initial) - set(expected_initial))
        raise IRIVMError(
            "common initial public-state schema differs from deterministic V17 "
            f"record: missing={missing}, unexpected={unexpected}"
        )
    for key, expected in expected_initial.items():
        if not _array_equal(initial[key], expected):
            raise IRIVMError(
                "common initial public state differs from deterministic V17 "
                f"record: {key}"
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
        "implementation_id": np.asarray(
            [str(row["implementation_id"].item()) for row in rows]
        ),
        "development_only": np.asarray(
            [bool(row["development_only"].item()) for row in rows], dtype=bool
        ),
        "state_index": indices,
        "nof_mean_force_incremental": nof,
        "nof_mean_force_absolute_including_shared_initial": np.asarray(
            [
                int(row["nof_mean_force_absolute_including_shared_initial"].item())
                for row in rows
            ],
            dtype=np.int64,
        ),
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
        "warm_validation_selected": np.asarray(
            [False]
            + [bool(row["warm_validation_selected"]) for row in rows[1:]],
            dtype=bool,
        ),
        "warm_validation_band_mean_force_cost": np.asarray(
            [0]
            + [int(row["warm_validation_band_mean_force_cost"]) for row in rows[1:]],
            dtype=np.int64,
        ),
        "warm_validation_fresh_raw_block_count": np.asarray(
            [0]
            + [int(row["warm_validation_fresh_raw_block_count"]) for row in rows[1:]],
            dtype=np.int64,
        ),
        "warm_validation_used_pooled_source_estimate": np.asarray(
            [False]
            + [
                bool(row["warm_validation_used_pooled_source_estimate"])
                for row in rows[1:]
            ],
            dtype=bool,
        ),
        "warm_validation_used_shrinkage_centered_residual": np.asarray(
            [False]
            + [
                bool(row["warm_validation_used_shrinkage_centered_residual"])
                for row in rows[1:]
            ],
            dtype=bool,
        ),
        "warm_validation_incremented_movement_count": np.asarray(
            [False]
            + [
                bool(row["warm_validation_incremented_movement_count"])
                for row in rows[1:]
            ],
            dtype=bool,
        ),
        "paid_action_is_movement": np.asarray(
            [False]
            + [bool(row["paid_action_is_movement"]) for row in rows[1:]],
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
        "lbfgs_raw_proximal_improvement_advantage": np.asarray(
            [math.nan]
            + [
                float(row["lbfgs_raw_proximal_improvement_advantage"])
                for row in rows[1:]
            ],
            dtype=np.float64,
        ),
        "lbfgs_raw_advantage_resolution_threshold": np.asarray(
            [math.nan]
            + [
                float(row["lbfgs_raw_advantage_resolution_threshold"])
                for row in rows[1:]
            ],
            dtype=np.float64,
        ),
        "lbfgs_raw_advantage_resolved_positive": np.asarray(
            [False]
            + [
                bool(row["lbfgs_raw_advantage_resolved_positive"])
                for row in rows[1:]
            ],
            dtype=bool,
        ),
        "r_stop_is_one_sided_confident_stop": np.asarray(
            [False]
            + [
                bool(row["r_stop_is_one_sided_confident_stop"])
                for row in rows[1:]
            ],
            dtype=bool,
        ),
        "r_stop_decision_eligible_after_at_least_one_movement": np.asarray(
            [False]
            + [
                bool(
                    row[
                        "r_stop_decision_eligible_after_at_least_one_movement"
                    ]
                )
                for row in rows[1:]
            ],
            dtype=bool,
        ),
        "zero_nof_trust_refresh_applied": np.asarray(
            [False]
            + [
                bool(row["zero_nof_trust_refresh_applied"])
                for row in rows[1:]
            ],
            dtype=bool,
        ),
        "zero_nof_trust_refresh_event_json": np.asarray(
            ["{}"]
            + [
                str(row["zero_nof_trust_refresh_event_json"].item())
                for row in rows[1:]
            ]
        ),
        "trust_refresh_used_path_sha256": np.asarray(
            [str(initial["trust_refresh_used_path_sha256"].item())]
            + [
                str(row["trust_refresh_used_path_sha256_next"].item())
                for row in rows[1:]
            ]
        ),
    }
    result["movement_count"] = np.cumsum(
        result["paid_action_is_movement"].astype(np.int64)
    )
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
    marker_key = (
        "trust_refresh_used_path_sha256"
        if int(controller["state_index"]) == 0
        else "trust_refresh_used_path_sha256_next"
    )
    if marker_key not in state or str(state[marker_key].item()) != str(
        controller["trust_refresh_used_path_sha256"]
    ):
        raise IRIVMError("checkpoint/public-state trust-refresh marker differs")
    if int(controller["state_index"]) == 0:
        warm_marker_key = "warm_restart_used_path_sha256"
        floor_key = "curvature_epoch_floor_state_index"
        pending_status_key = "pending_warm_status"
        pending_candidate_key = "pending_warm_candidate_sha256"
    else:
        warm_marker_key = "warm_restart_used_path_sha256_next"
        floor_key = "curvature_epoch_floor_state_index_next"
        pending_status_key = "pending_warm_status_next"
        pending_candidate_key = "pending_warm_candidate_sha256_next"
    expected_scalars = {
        warm_marker_key: str(controller["warm_restart_used_path_sha256"]),
        floor_key: int(controller["curvature_epoch_floor_state_index"]),
        pending_status_key: str(controller["pending_warm_status"]),
        pending_candidate_key: str(controller["pending_warm_candidate_sha256"]),
    }
    for key, expected_value in expected_scalars.items():
        if key not in state or state[key].item() != expected_value:
            raise IRIVMError(
                f"checkpoint/public-state warm-validation field differs: {key}"
            )


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
    """Run or exactly resume one supplied-oracle V17 formal trajectory."""

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
            raise IRIVMError("existing V17 run manifest differs from request")
    else:
        _atomic_save_json(manifest_path, requested_manifest)

    completion_path = run_directory / "completion.json"
    prior_completion: Optional[Dict[str, object]] = None
    if resume and completion_path.is_file():
        prior_completion = json.loads(completion_path.read_text(encoding="utf-8"))

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
            raise IRIVMError("V17 run directory is nonempty and resume=False")
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
        raise IRIVMError("requested stop lies behind durable V17 checkpoint")
    algorithmic_stop: Optional[Dict[str, object]] = None
    terminal_proposal_controller = controller
    terminal_zero_nof_event: Dict[str, object] = {}
    terminal_rederived_on_resume = False
    if (
        prior_completion is not None
        and prior_completion.get("phase") == "TERMINAL"
        and prior_completion.get("algorithmic_zero_nof_stop") is not None
    ):
        derived = _prepare_update(controller, controls)
        if derived.prepared is not None or not derived.terminal:
            raise IRIVMError(
                "durable terminal completion no longer rederives from its paid state"
            )
        if prior_completion.get("algorithmic_zero_nof_stop") != derived.terminal:
            raise IRIVMError(
                "durable terminal completion differs from deterministic rederivation"
            )
        expected_terminal_digest = _checkpoint_digest(
            derived.proposal_controller
        )
        if prior_completion.get(
            "terminal_proposal_controller_checkpoint_sha256"
        ) != expected_terminal_digest:
            raise IRIVMError(
                "durable terminal proposal-controller digest differs on rederivation"
            )
        if prior_completion.get("zero_nof_trust_refresh_event", {}) != (
            derived.zero_nof_event
        ):
            raise IRIVMError(
                "durable terminal zero-NoF event differs on rederivation"
            )
        algorithmic_stop = derived.terminal
        terminal_proposal_controller = derived.proposal_controller
        terminal_zero_nof_event = derived.zero_nof_event
        terminal_rederived_on_resume = True
    start_time = time.perf_counter()
    while (
        algorithmic_stop is None
        and int(controller["nof"]) + N_INTERNAL <= int(stop_after_nof)
    ):
        outcome = _prepare_update(controller, controls)
        prepared = outcome.prepared
        if prepared is None:
            if not outcome.terminal:
                raise IRIVMError("preparation returned neither a proposal nor terminal")
            algorithmic_stop = outcome.terminal
            terminal_proposal_controller = outcome.proposal_controller
            terminal_zero_nof_event = outcome.zero_nof_event
            _atomic_save_json(
                run_directory / "status.json",
                {
                    "implementation_id": IMPLEMENTATION_ID,
                    "development_only": False,
                    "phase": "TERMINAL",
                    "state_index": int(controller["state_index"]),
                    "nof_mean_force_incremental": int(controller["nof"]),
                    "algorithmic_zero_nof_stop": algorithmic_stop,
                    "durable_paid_controller_checkpoint_sha256": (
                        _checkpoint_digest(controller)
                    ),
                    "terminal_proposal_controller_checkpoint_sha256": (
                        _checkpoint_digest(terminal_proposal_controller)
                    ),
                    "zero_nof_trust_refresh_event": terminal_zero_nof_event,
                    "same_index_zero_nof_refresh_checkpoint_published": False,
                    "updated_unix": time.time(),
                },
            )
            break
        proposal_controller = outcome.proposal_controller
        if not (
            np.array_equal(prepared.selected_path[0], path[0])
            and np.array_equal(prepared.selected_path[-1], path[-1])
        ):
            raise IRIVMError("fixed endpoints changed in V17 proposal")
        next_state_index = int(proposal_controller["state_index"]) + 1
        prepared_phase = (
            "INDEPENDENT_WARM_VALIDATION_PREPARED"
            if prepared.warm_validation_selected
            else "RESAMPLE_PREPARED"
            if prepared.hold_selected
            else "VALIDATED_FROZEN_WARM_COMMIT_PREPARED"
            if prepared.frozen_warm_commit
            else "MOVE_PREPARED"
        )
        _atomic_save_json(
            run_directory / "status.json",
            {
                "implementation_id": IMPLEMENTATION_ID,
                "development_only": False,
                "phase": prepared_phase,
                "state_index": int(proposal_controller["state_index"]),
                "nof_mean_force_incremental": int(proposal_controller["nof"]),
                "proposal_sha256": prepared.proposal_sha256,
                "proposal_controller_checkpoint_sha256": _checkpoint_digest(
                    proposal_controller
                ),
                "zero_nof_trust_refresh_event": outcome.zero_nof_event,
                "same_index_zero_nof_refresh_checkpoint_published": False,
                "selected_candidate_kind": prepared.selected_label,
                "warm_validation_request_sha256": (
                    str(prepared.frozen_warm_candidate["request_sha256"])
                    if prepared.warm_validation_selected
                    and prepared.frozen_warm_candidate is not None
                    else "none"
                ),
                "frozen_warm_commit": bool(prepared.frozen_warm_commit),
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
                    "development_only": False,
                    "phase": "FAILED",
                    "state_index": int(proposal_controller["state_index"]),
                    "nof_mean_force_incremental": int(proposal_controller["nof"]),
                    "proposal_sha256": prepared.proposal_sha256,
                    "zero_nof_trust_refresh_event": outcome.zero_nof_event,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "updated_unix": time.time(),
                },
            )
            raise
        try:
            _validate_complete_band(band, prepared.selected_path)
            next_controller, advance = _advance_controller(
                proposal_controller,
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
                proposal_controller,
                next_controller,
                prepared,
                outcome.zero_nof_event,
                band,
                advance,
                metrics,
            )
        except BaseException as error:
            _atomic_save_json(
                run_directory / "status.json",
                {
                    "implementation_id": IMPLEMENTATION_ID,
                    "development_only": False,
                    "phase": "FAILED",
                    "state_index": int(proposal_controller["state_index"]),
                    "nof_mean_force_incremental": int(proposal_controller["nof"]),
                    "proposal_sha256": prepared.proposal_sha256,
                    "zero_nof_trust_refresh_event": outcome.zero_nof_event,
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
                "development_only": False,
                "phase": (
                    "VALIDATED_FROZEN_WARM_PENDING_COMMIT"
                    if str(controller["pending_warm_status"])
                    == WARM_VALIDATION_PENDING_COMMIT
                    else "WARM_RESTART_READY"
                    if bool(advance["warm_restart_event"])
                    else "READY"
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
                "zero_nof_trust_refresh_applied_before_paid_band": bool(
                    outcome.zero_nof_event
                ),
                "trust_refresh_used_path_sha256_next": str(
                    controller["trust_refresh_used_path_sha256"]
                ),
                "warm_restart_used_path_sha256_next": str(
                    controller["warm_restart_used_path_sha256"]
                ),
                "curvature_epoch_floor_state_index_next": int(
                    controller["curvature_epoch_floor_state_index"]
                ),
                "pending_warm_status_next": str(
                    controller["pending_warm_status"]
                ),
                "warm_validation_classification": str(
                    advance["warm_validation"]["classification"]
                ),
                "warm_validation_disposition": str(
                    advance["warm_validation_disposition"]
                ),
                "zero_nof_warm_restart_event": advance["warm_restart_event"],
                "selected_candidate_kind": prepared.selected_label,
                "lbfgs_raw_advantage_resolved_positive": bool(
                    prepared.lbfgs_raw_advantage_resolved_positive
                ),
                "I_perp_G": float(metrics["I_perp_G"]),
                "I_arc": float(metrics["I_arc"]),
                "Delta_F_barrier": float(metrics["Delta_F_barrier"]),
                "effective_residual_ratio": (
                    float(advance["residual_ratio"])
                    if np.isfinite(advance["residual_ratio"])
                    else None
                ),
                "inner_stop_reason": advance["inner_stop_reason"],
                "r_global_stop": bool(advance["r_stop"]),
                "r_stop_is_one_sided_confident_stop": True,
                "r_stop_decision_eligible_after_at_least_one_movement": bool(
                    advance[
                        "r_stop_decision_eligible_after_at_least_one_movement"
                    ]
                ),
                "residual_interval_ambiguous": bool(
                    advance["residual_decision_ambiguous"]
                ),
                "elapsed_seconds_this_invocation": time.perf_counter() - start_time,
                "updated_unix": time.time(),
            },
        )

    trajectory = load_public_trajectory(run_directory)
    final_status = {
        "phase": "TERMINAL" if algorithmic_stop is not None else (
            "VALIDATED_FROZEN_WARM_PENDING_COMMIT"
            if str(controller["pending_warm_status"])
            == WARM_VALIDATION_PENDING_COMMIT
            else "READY"
            if bool(controller["outer_anchor_bounds_ready"])
            else "ANCHORING"
        ),
        # Completing the registered budget prefix is orthogonal to whether the
        # controller already knows its first action beyond that prefix.  In
        # particular, a validation HOLD at the exact horizon is a complete,
        # scoreable public prefix even when its frozen movement remains pending
        # for a future, separately budgeted 20-NoF band.
        "completed": bool(int(controller["nof"]) == nof_budget),
        "budget_prefix_complete": bool(int(controller["nof"]) == nof_budget),
        "requested_prefix_complete": bool(
            int(controller["nof"]) == stop_after_nof
        ),
        "budget_fully_spent": bool(int(controller["nof"]) == nof_budget),
        "pending_next_action": bool(
            str(controller["pending_warm_status"])
            != WARM_VALIDATION_PENDING_NONE
        ),
        "pending_validated_commit_requires_another_band": bool(
            str(controller["pending_warm_status"])
            == WARM_VALIDATION_PENDING_COMMIT
        ),
        "paused_at_requested_stop": bool(
            int(controller["nof"]) == stop_after_nof
            and int(controller["nof"]) < nof_budget
            and algorithmic_stop is None
        ),
        "algorithmic_zero_nof_stop": algorithmic_stop,
        "nof_mean_force_incremental": int(controller["nof"]),
        "nof_budget": int(nof_budget),
        "state_count_including_common_initial": int(trajectory["state_index"].size),
        "outer_index_next": int(
            terminal_proposal_controller["outer_index"]
            if algorithmic_stop is not None
            else controller["outer_index"]
        ),
        "inner_step_next": int(
            terminal_proposal_controller["inner_step"]
            if algorithmic_stop is not None
            else controller["inner_step"]
        ),
        "durable_paid_controller_checkpoint_sha256": _checkpoint_digest(controller),
        "terminal_proposal_controller_checkpoint_sha256": (
            _checkpoint_digest(terminal_proposal_controller)
            if algorithmic_stop is not None
            else None
        ),
        "zero_nof_trust_refresh_event": terminal_zero_nof_event,
        "same_index_zero_nof_refresh_checkpoint_published": False,
        "pending_warm_status": str(controller["pending_warm_status"]),
        "warm_restart_used_path_sha256": str(
            controller["warm_restart_used_path_sha256"]
        ),
        "curvature_epoch_floor_state_index": int(
            controller["curvature_epoch_floor_state_index"]
        ),
        "terminal_rederived_on_resume": terminal_rederived_on_resume,
        "implementation_id": IMPLEMENTATION_ID,
        "development_only": False,
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
    "IRIVMControlsV17",
    "IRIVMError",
    "JOINT_HISTORY_SIZE",
    "HOLD_STATE_KIND",
    "N_INTERNAL",
    "STATE_KIND",
    "WARM_VALIDATION_STATE_KIND",
    "joint_lbfgs_direction",
    "load_public_trajectory",
    "noise_shrink_residual",
    "run_irivm",
]
