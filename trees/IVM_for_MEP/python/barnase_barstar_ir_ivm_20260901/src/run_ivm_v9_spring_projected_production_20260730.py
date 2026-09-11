#!/usr/bin/env python3
"""Run the frozen Chignolin central-tangent IR-IVM on Barnase--Barstar."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from bb_ffprecon import install_call_free_fast_builder

from bb_core import (
    EV_TO_KCAL,
    EvaluationBudgetReached,
    OpenMMBackend,
    TANGENT_DEFINITION,
    array_sha256,
    atomic_save_npz,
    audit_counter_consistency,
    central_tangents,
    clip_max_atomic_displacement,
    counter_arrays,
    equidistance_diagnostics,
    evaluate_path,
    load_common_path,
    make_evaluators,
    path_diagnostics,
    preload_frozen_endpoint_cache,
    project_normal,
    spring_reparameterize,
    write_json,
)
from frozen_discrete_b040 import (
    CANONICAL_STOP_MODE as DISCRETE_CANONICAL_STOP_MODE,
    IMPLEMENTATION_ID as DISCRETE_B040_IMPLEMENTATION_ID,
    RESIDUAL_STOP_MODES as DISCRETE_RESIDUAL_STOP_MODES,
    VALID_STOP_MODES as DISCRETE_VALID_STOP_MODES,
    solve_frozen_discrete_b040_band,
)
from storage_policy import MODES as PATH_STORAGE_MODES, should_store_path

SPRING_PROJECTION_METHOD = (
    "identical nearest-neighbour spring flow with normalized central "
    "difference tangent"
)
SPRING_PROJECTION_DTAU = 0.5
SPRING_PROJECTION_MAX_ITER = 20000
# The projected-state trigger is segment_length_cv <= 1e-8.  At 43 or more
# images, a 1e-9 spring-force residual can still leave CV around 2e-8 and
# spuriously disable the equal-spring solver on the next outer iteration.
# Tightening only this call-free geometry solve preserves the exact same IVM
# subproblem and gives the trigger a genuine numerical safety margin across
# the declared path resolutions.
SPRING_PROJECTION_TOLERANCE = 1e-10
ALA_DSEL_IMPLEMENTATION_ID = (
    "ala-legacy-ivm-fixedalpha-triggered-ffrbk-vs-h0-lbfgs-"
    "postgeometry-dsel-v3"
)


def scheduled_value(initial: float, schedule: str, calls: int) -> float:
    value = float(initial)
    if schedule:
        for item in schedule.split(","):
            start, new_value = item.split(":", 1)
            if calls >= int(start):
                value = float(new_value)
    return value


def immediate_relock_fallback_budget(
    remaining_after_direct: int,
    transactional_relock_budget: bool,
    reparam_strategy: str,
) -> tuple[int, int]:
    """Return raw-block budget and reserved relock evaluations/image."""
    reserve = int(
        transactional_relock_budget
        and reparam_strategy
        in ("spring_relock", "cv_relock", "polyline_relock")
    )
    return max(0, int(remaining_after_direct) - reserve), reserve


def largest_normal_force_block_mask(per_image_l2) -> np.ndarray:
    """Select one frozen-normal block by a state-only, alpha-free rule."""
    per_image_l2 = np.asarray(per_image_l2, dtype=float)
    if per_image_l2.ndim != 1 or per_image_l2.size == 0:
        raise ValueError("per_image_l2 must be a nonempty vector")
    if not np.all(np.isfinite(per_image_l2)):
        raise ValueError("per_image_l2 must contain only finite values")
    selected = np.zeros(per_image_l2.size, dtype=bool)
    selected[int(np.argmax(per_image_l2))] = True
    return selected


def residual_precision_cap(alpha: float, physical_fmax: float, policy: str):
    """Apply state-triggered alpha caps, formatted as ``fmax:cap,...``."""
    value = float(alpha)
    active_cap = np.inf
    if policy:
        for item in policy.split(","):
            threshold, cap = item.split(":", 1)
            if physical_fmax <= float(threshold):
                active_cap = min(active_cap, float(cap))
    if np.isfinite(active_cap):
        value = min(value, active_cap)
    return value, float(active_cap)


def relative_iperp_precision_cap(
    alpha: float, relative_i_perp: float, policy: str
):
    """Apply dimensionless I_perp/I_perp(0) caps ``ratio:cap,...``."""
    value = float(alpha)
    active_cap = np.inf
    if policy:
        for item in policy.split(","):
            threshold, cap = item.split(":", 1)
            if relative_i_perp <= float(threshold):
                active_cap = min(active_cap, float(cap))
    if np.isfinite(active_cap):
        value = min(value, active_cap)
    return value, float(active_cap)


def history_warm_start_is_active(enabled, trigger_iperp, current_iperp):
    """Activate accumulated curvature only after the path is locally stable."""
    if not enabled:
        return False
    return (
        float(trigger_iperp) <= 0.0
        or float(current_iperp) <= float(trigger_iperp)
    )


def state_outer_history_is_active(enabled, trigger_fmax, current_fmax):
    """Reuse cross-subproblem curvature only in a stable residual regime."""
    if not enabled:
        return False
    return (
        float(trigger_fmax) <= 0.0
        or float(current_fmax) <= float(trigger_fmax)
    )


def state_threshold_value(initial, policy, current_value):
    """Apply ``threshold:value`` overrides keyed by a decreasing merit."""
    value = float(initial)
    if policy:
        for item in policy.split(","):
            threshold, new_value = item.split(":", 1)
            if float(current_value) <= float(threshold):
                value = float(new_value)
    return value


def ffprecon_update_mask(n_internal, iteration, scheme):
    """Choose simultaneous or red-black central-tangent block updates."""
    if scheme == "all":
        return np.ones(int(n_internal), dtype=bool)
    if scheme == "red_black":
        indices = np.arange(int(n_internal))
        return indices % 2 == int(iteration) % 2
    raise ValueError(f"Unknown FF preconditioner update scheme: {scheme}")


def proximal_model_quality(
    reference_path,
    reference_energies,
    reference_forces,
    candidate_path,
    candidate_energies,
    tangents,
    alpha,
    max_step,
):
    """Return robust actual/predicted proximal decrease and step utilization."""
    displacement = project_normal(
        np.asarray(candidate_path, dtype=float)[1:-1]
        - np.asarray(reference_path, dtype=float)[1:-1],
        tangents,
    )
    gradient = project_normal(
        -np.asarray(reference_forces, dtype=float)[1:-1],
        tangents,
    )
    quadratic = 0.5 * np.sum(displacement**2, axis=1) / float(alpha)
    predicted = -np.sum(gradient * displacement, axis=1) - quadratic
    actual = (
        np.asarray(reference_energies, dtype=float)[1:-1]
        - np.asarray(candidate_energies, dtype=float)[1:-1]
        - quadratic
    )
    moved = np.linalg.norm(displacement, axis=1) > 1e-14
    valid = moved & np.isfinite(predicted) & np.isfinite(actual) & (predicted > 1e-14)
    ratios = actual[valid] / predicted[valid]
    ratios = ratios[np.isfinite(ratios)]
    quality = float(np.median(ratios)) if ratios.size else np.nan
    atom_steps = np.linalg.norm(
        displacement.reshape(displacement.shape[0], -1, 3), axis=2
    )
    largest = float(np.max(atom_steps)) if atom_steps.size else 0.0
    utilization = largest / max(float(max_step), 1e-30)
    return quality, utilization, int(np.count_nonzero(valid))


def adapt_proximal_alpha(
    alpha,
    quality,
    step_utilization,
    accepted_fraction,
    alpha_min,
    alpha_max,
    shrink_threshold=0.25,
    grow_threshold=0.75,
    grow_utilization=0.8,
    shrink_factor=0.5,
    grow_factor=1.5,
):
    """Trust-quality heuristic for the next proximal subproblem.

    Poor agreement between the nonlinear subproblem and its local model
    shrinks alpha.  Strong agreement while the displacement cap is active
    grows alpha.  Otherwise it is held fixed.
    """
    value = float(alpha)
    action = "hold"
    if (
        not np.isfinite(quality)
        or quality < shrink_threshold
        or accepted_fraction < 0.5
    ):
        value *= shrink_factor
        action = "shrink"
    elif (
        quality > grow_threshold
        and step_utilization > grow_utilization
        and accepted_fraction >= 0.8
    ):
        value *= grow_factor
        action = "grow"
    clipped = float(np.clip(value, alpha_min, alpha_max))
    if clipped == float(alpha):
        action = "hold"
    return clipped, action


def geometric_spring_reparameterization(
    path,
    step_scale,
    spring_constant=1.0,
    dtau=0.05,
    max_iter=500,
    tolerance=1e-5,
):
    """Propose and accept reparameterization using coordinates alone."""
    path = np.asarray(path, dtype=float)
    step_scale = np.asarray(step_scale, dtype=float)
    proposed, residual = spring_reparameterize(
        path,
        spring_constant=spring_constant,
        dtau=dtau,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    scaled = path.copy()
    scaled[1:-1] += step_scale[:, None] * (
        proposed[1:-1] - path[1:-1]
    )
    before = equidistance_diagnostics(path)
    after = equidistance_diagnostics(scaled)
    improved = (
        after["equidistance_integral"]
        <= before["equidistance_integral"] + 1e-30
        and after["segment_length_cv"]
        <= before["segment_length_cv"] + 1e-14
    )
    if not improved or np.array_equal(scaled, path):
        return path.copy(), float(residual), False, before, after
    return scaled, float(residual), True, before, after


def euler_spring_reparameterization(
    path,
    step_scale,
    dtau=0.02,
    backtrack_factor=0.5,
    max_backtracks=12,
):
    """Take one geometry-only central-tangent spring Euler microstep.

    Unlike interpolation toward a fully relaxed spring band, this displacement
    is exactly tangential to the *current* central-difference path.  A
    geometry-only backtracking loop accepts only a decrease of both the raw
    requested equidistance integral and segment-length CV.
    """
    path = np.asarray(path, dtype=float)
    step_scale = np.asarray(step_scale, dtype=float)
    segments = path[1:] - path[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    tangents = central_tangents(path)
    spring_force = (lengths[1:] - lengths[:-1])[:, None] * tangents
    residual = float(np.max(np.linalg.norm(spring_force, axis=1)))
    before = equidistance_diagnostics(path)
    local_dtau = float(dtau)
    for _ in range(int(max_backtracks) + 1):
        candidate = path.copy()
        candidate[1:-1] += (
            local_dtau * step_scale[:, None] * spring_force
        )
        after = equidistance_diagnostics(candidate)
        if (
            after["equidistance_integral"]
            < before["equidistance_integral"] - 1e-30
            and after["segment_length_cv"]
            <= before["segment_length_cv"] + 1e-14
        ):
            return (
                candidate,
                residual,
                True,
                before,
                after,
                local_dtau,
            )
        local_dtau *= float(backtrack_factor)
    return path.copy(), residual, False, before, before, 0.0


def central_spring_force_equilibrium(
    path,
    dtau=SPRING_PROJECTION_DTAU,
    max_iter=SPRING_PROJECTION_MAX_ITER,
    tolerance=SPRING_PROJECTION_TOLERANCE,
):
    """Relax identical springs using only the shared central tangent.

    This is the discrete string/NEB spring flow

    ``dx_i/dtau = (|x_{i+1}-x_i|-|x_i-x_{i-1}|) tau_i``

    with ``tau_i`` equal to the normalized central difference.  It contains
    no spline, OpenMM energy, or OpenMM force.  The strict tolerance keeps
    the projected whole-band solver on the same numerically equal-spring
    manifold used by its state trigger.
    """
    result, residual = spring_reparameterize(
        np.asarray(path, dtype=float),
        spring_constant=1.0,
        dtau=float(dtau),
        max_iter=int(max_iter),
        tolerance=float(tolerance),
    )
    return result, float(residual)


def piecewise_linear_equal_arclength(path):
    """Redistribute images by arclength on the represented polyline.

    This is the coordinate-only image-mesh refresh described in the paper:
    it samples the current piecewise-linear interpolant and never consults
    the potential or force.  In contrast to integrating a discrete spring
    flow, every proposed coordinate lies on the current represented path.
    The refreshed coordinates still receive a fully charged OpenMM query
    before any path-level acceptance decision.
    """

    path = np.asarray(path, dtype=float)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError(
            "piecewise-linear arclength refresh requires positive finite "
            "segment lengths"
        )
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, float(cumulative[-1]), path.shape[0])
    output = np.empty_like(path)
    output[0] = path[0]
    output[-1] = path[-1]
    for image_index, target in enumerate(targets[1:-1], start=1):
        segment = int(
            np.searchsorted(cumulative, target, side="right") - 1
        )
        segment = min(segment, path.shape[0] - 2)
        fraction = (target - cumulative[segment]) / lengths[segment]
        output[image_index] = (
            (1.0 - fraction) * path[segment]
            + fraction * path[segment + 1]
        )
    return output


def rms_segment_trust_radius(path, fraction, maximum=np.inf):
    """Scale the atom-wise trust radius to the current image resolution.

    The median Cartesian segment length is divided by ``sqrt(n_atoms)`` to
    obtain an RMS displacement per atom.  ``fraction`` is therefore
    dimensionless and transfers between molecular systems and image counts.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[0] < 3 or path.shape[1] % 3:
        raise ValueError(f"Invalid Cartesian image path: {path.shape}")
    n_atoms = path.shape[1] // 3
    segment_rms = np.linalg.norm(np.diff(path, axis=0), axis=1) / np.sqrt(
        float(n_atoms)
    )
    radius = float(fraction) * float(np.median(segment_rms))
    return min(radius, float(maximum))


def spring_proposal_to_target_cv(
    path,
    target_cv,
    contraction=0.5,
    spring_constant=1.0,
    dtau=0.05,
    max_iter=500,
    tolerance=1e-5,
):
    """Return the smallest pure-spring move reaching a state-based CV target.

    The full spring relaxation is computed without energy or force calls.
    Bisection then selects the smallest interpolation fraction that reduces
    the current segment-length CV by ``contraction`` or reaches ``target_cv``.
    This avoids an arbitrary fixed reparameterization step size.
    """
    path = np.asarray(path, dtype=float)
    before = equidistance_diagnostics(path)
    full, residual = spring_reparameterize(
        path,
        spring_constant=spring_constant,
        dtau=dtau,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    full_diag = equidistance_diagnostics(full)
    desired_cv = max(
        float(target_cv),
        float(contraction) * float(before["segment_length_cv"]),
    )
    if (
        full_diag["segment_length_cv"]
        >= before["segment_length_cv"] - 1e-14
        or np.array_equal(full, path)
    ):
        return path.copy(), float(residual), 0.0, before, full_diag
    if full_diag["segment_length_cv"] > desired_cv:
        return full, float(residual), 1.0, before, full_diag
    low = 0.0
    high = 1.0
    candidate = full.copy()
    candidate_diag = full_diag
    for _ in range(48):
        scale = 0.5 * (low + high)
        trial = path.copy()
        trial[1:-1] += scale * (full[1:-1] - path[1:-1])
        trial_diag = equidistance_diagnostics(trial)
        if trial_diag["segment_length_cv"] <= desired_cv:
            high = scale
            candidate = trial
            candidate_diag = trial_diag
        else:
            low = scale
    return (
        candidate,
        float(residual),
        float(high),
        before,
        candidate_diag,
    )


def spring_proposal_to_target_equidistance(
    path,
    contraction=0.75,
    spring_constant=1.0,
    dtau=0.05,
    max_iter=500,
    tolerance=1e-5,
):
    """Return the smallest spring-flow move reaching a raw-integral target.

    Every internal relaxation step is the standard length-difference spring
    force along the shared central-difference tangent.  The final bisection is
    geometry-only and chooses a fractional contraction of the requested
    equidistance integral, so its scale is independent of force-call count and
    of the absolute size of the molecule.
    """
    if not 0.0 < float(contraction) < 1.0:
        raise ValueError("Equidistance contraction must lie strictly in (0, 1)")
    path = np.asarray(path, dtype=float)
    before = equidistance_diagnostics(path)
    full, residual = spring_reparameterize(
        path,
        spring_constant=spring_constant,
        dtau=dtau,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    full_diag = equidistance_diagnostics(full)
    desired = float(contraction) * float(
        before["equidistance_integral"]
    )
    if (
        full_diag["equidistance_integral"]
        >= before["equidistance_integral"] - 1e-30
        or np.array_equal(full, path)
    ):
        return path.copy(), float(residual), 0.0, before, full_diag
    if full_diag["equidistance_integral"] > desired:
        return full, float(residual), 1.0, before, full_diag
    low = 0.0
    high = 1.0
    candidate = full.copy()
    candidate_diag = full_diag
    for _ in range(48):
        scale = 0.5 * (low + high)
        trial = path.copy()
        trial[1:-1] += scale * (full[1:-1] - path[1:-1])
        trial_diag = equidistance_diagnostics(trial)
        if trial_diag["equidistance_integral"] <= desired:
            high = scale
            candidate = trial
            candidate_diag = trial_diag
        else:
            low = scale
    return candidate, float(residual), float(high), before, candidate_diag


def spring_proposal_to_target_relative_defect(
    path,
    target_relative_defect,
    spring_constant=1.0,
    dtau=0.05,
    max_iter=500,
    tolerance=1e-5,
):
    """Return the smallest geometry-only spring move reaching a tolerance.

    The tolerance is the dimensionless relative squared-speed defect reported
    by :func:`equidistance_diagnostics`.  A full relaxation under the
    identical nearest-neighbour spring force is first computed using only the
    shared central-difference tangent; bisection then selects the smallest
    fraction of that move which reaches the requested tolerance.  No spline,
    OpenMM energy, or OpenMM force enters either operation.
    """
    target = float(target_relative_defect)
    if not np.isfinite(target) or target < 0.0:
        raise ValueError(
            "Target relative equidistance defect must be finite and nonnegative"
        )
    path = np.asarray(path, dtype=float)
    before = equidistance_diagnostics(path)
    if before["equidistance_relative_defect"] <= target:
        return path.copy(), 0.0, 0.0, before, before
    full, residual = spring_reparameterize(
        path,
        spring_constant=spring_constant,
        dtau=dtau,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    full_diag = equidistance_diagnostics(full)
    if (
        full_diag["equidistance_relative_defect"]
        >= before["equidistance_relative_defect"] - 1e-30
        or np.array_equal(full, path)
    ):
        return path.copy(), float(residual), 0.0, before, full_diag
    if full_diag["equidistance_relative_defect"] > target:
        return full, float(residual), 1.0, before, full_diag
    low = 0.0
    high = 1.0
    candidate = full.copy()
    candidate_diag = full_diag
    for _ in range(60):
        scale = 0.5 * (low + high)
        trial = path.copy()
        trial[1:-1] += scale * (full[1:-1] - path[1:-1])
        trial_diag = equidistance_diagnostics(trial)
        if trial_diag["equidistance_relative_defect"] <= target:
            high = scale
            candidate = trial
            candidate_diag = trial_diag
        else:
            low = scale
    return (
        candidate,
        float(residual),
        float(high),
        before,
        candidate_diag,
    )


def state_reparameterization_is_due(
    diagnostic,
    initial_i_perp,
    last_attempt_i_perp,
    ready_relative_i_perp,
    cv_trigger,
    retry_reduction,
):
    """Dimensionless state trigger for optional reparameterization."""
    current_i_perp = float(diagnostic["i_perp"])
    if current_i_perp > float(ready_relative_i_perp) * float(
        initial_i_perp
    ):
        return False
    if float(diagnostic["segment_length_cv"]) <= float(cv_trigger):
        return False
    if np.isfinite(last_attempt_i_perp) and current_i_perp > (
        float(retry_reduction) * float(last_attempt_i_perp)
    ):
        return False
    return True


def reparameterization_acceptance_mask(
    current_path,
    candidate_path,
    current_energies,
    candidate_energies,
    current_forces,
    candidate_forces,
    alpha,
    objective_tolerance_kcal,
    energy_tolerance_kcal,
    force_rtol,
):
    """Audit a reparameterization proposal image by image.

    The IVP objective uses the normal component of the reparameterization
    displacement under the shared central-difference tangent. Energy and
    physical normal-force checks are kept explicit so no one criterion can
    silently substitute for another.
    """
    current_path = np.asarray(current_path, dtype=float)
    candidate_path = np.asarray(candidate_path, dtype=float)
    current_energies = np.asarray(current_energies, dtype=float)
    candidate_energies = np.asarray(candidate_energies, dtype=float)
    current_forces = np.asarray(current_forces, dtype=float)
    candidate_forces = np.asarray(candidate_forces, dtype=float)
    current_tangent = central_tangents(current_path)
    displacement = project_normal(
        candidate_path[1:-1] - current_path[1:-1],
        current_tangent,
    )
    current_objective = current_energies[1:-1]
    candidate_objective = candidate_energies[1:-1] + (
        0.5 * np.sum(displacement**2, axis=1) / float(alpha)
    )
    objective_ok = candidate_objective <= (
        current_objective + objective_tolerance_kcal / EV_TO_KCAL
    )
    energy_ok = candidate_energies[1:-1] <= (
        current_energies[1:-1] + energy_tolerance_kcal / EV_TO_KCAL
    )
    current_diag = path_diagnostics(
        current_path,
        current_energies,
        current_forces[1:-1],
    )
    candidate_diag = path_diagnostics(
        candidate_path,
        candidate_energies,
        candidate_forces[1:-1],
    )
    force_ok = candidate_diag["per_image_l2"] <= (
        current_diag["per_image_l2"] * (1.0 + force_rtol) + 1e-12
    )
    return objective_ok & energy_ok & force_ok, {
        "objective_ok": objective_ok,
        "energy_ok": energy_ok,
        "force_ok": force_ok,
        "current_objective": current_objective,
        "candidate_objective": candidate_objective,
        "current_per_image_l2": current_diag["per_image_l2"],
        "candidate_per_image_l2": candidate_diag["per_image_l2"],
    }


def greedy_global_iperp_acceptance(
    current_path,
    current_energies,
    current_forces,
    candidate_path,
    candidate_energies,
    candidate_forces,
    eligible,
    global_iperp_rtol,
    order="local_ratio",
):
    """Merge already-evaluated image proposals without worsening global I_perp.

    Central-difference tangents couple neighbouring images.  A simultaneous
    Jacobi step can therefore improve the path merit even when replacing
    either neighbour in isolation makes it worse.  ``collective_first``
    explicitly tests the complete locally eligible update before falling
    back to the conservative one-image-at-a-time merge.
    """
    path = np.asarray(current_path, dtype=float).copy()
    energies = np.asarray(current_energies, dtype=float).copy()
    forces = np.asarray(current_forces, dtype=float).copy()
    candidate_path = np.asarray(candidate_path, dtype=float)
    candidate_energies = np.asarray(candidate_energies, dtype=float)
    candidate_forces = np.asarray(candidate_forces, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    diagnostic = path_diagnostics(path, energies, forces[1:-1])
    candidate_diag = path_diagnostics(
        candidate_path,
        candidate_energies,
        candidate_forces[1:-1],
    )
    denominator = np.maximum(diagnostic["per_image_l2"], 1e-30)
    priority = candidate_diag["per_image_l2"] / denominator
    accepted = np.zeros_like(eligible)
    if order in {"collective_first", "collective_block"}:
        collective_path = path.copy()
        collective_energies = energies.copy()
        collective_forces = forces.copy()
        collective_path[1:-1][eligible] = candidate_path[1:-1][eligible]
        collective_energies[1:-1][eligible] = candidate_energies[1:-1][eligible]
        collective_forces[1:-1][eligible] = candidate_forces[1:-1][eligible]
        collective_diag = path_diagnostics(
            collective_path,
            collective_energies,
            collective_forces[1:-1],
        )
        if collective_diag["i_perp"] <= (
            diagnostic["i_perp"] * (1.0 + global_iperp_rtol) + 1e-12
        ):
            accepted[eligible] = True
            return (
                collective_path,
                collective_energies,
                collective_forces,
                collective_diag,
                accepted,
            )
        if order == "collective_block":
            # A central-difference tangent at image i depends on coordinates
            # i-1 and i+1.  Consequently, individually merging otherwise
            # valid subproblem steps can reject a coupled descent direction.
            # Search already-evaluated contiguous blocks (no extra OpenMM
            # calls) and greedily merge the block giving the largest true
            # global I_perp decrease.
            remaining = set(np.flatnonzero(eligible).tolist())
            n_internal = int(eligible.size)
            max_block_size = min(8, n_internal)
            while remaining:
                best_indices = None
                best_i_perp = np.inf
                for block_size in range(1, max_block_size + 1):
                    for start in range(
                        0, n_internal - block_size + 1
                    ):
                        indices = [
                            index
                            for index in range(
                                start, start + block_size
                            )
                            if index in remaining
                        ]
                        if not indices:
                            continue
                        trial_path = path.copy()
                        trial_energies = energies.copy()
                        trial_forces = forces.copy()
                        image_indices = np.asarray(indices) + 1
                        trial_path[image_indices] = candidate_path[
                            image_indices
                        ]
                        trial_energies[image_indices] = candidate_energies[
                            image_indices
                        ]
                        trial_forces[image_indices] = candidate_forces[
                            image_indices
                        ]
                        trial_i_perp = path_diagnostics(
                            trial_path,
                            trial_energies,
                            trial_forces[1:-1],
                        )["i_perp"]
                        if trial_i_perp < best_i_perp:
                            best_i_perp = trial_i_perp
                            best_indices = indices
                if best_i_perp > (
                    diagnostic["i_perp"]
                    * (1.0 + global_iperp_rtol)
                    + 1e-12
                ):
                    break
                image_indices = np.asarray(best_indices) + 1
                path[image_indices] = candidate_path[image_indices]
                energies[image_indices] = candidate_energies[image_indices]
                forces[image_indices] = candidate_forces[image_indices]
                diagnostic = path_diagnostics(
                    path, energies, forces[1:-1]
                )
                accepted[np.asarray(best_indices)] = True
                remaining.difference_update(best_indices)
            return path, energies, forces, diagnostic, accepted
        order = "global_best"
    if order == "local_ratio":
        ordered_indices = list(np.argsort(priority))
    elif order == "global_best":
        ordered_indices = []
        remaining = set(np.flatnonzero(eligible).tolist())
        while remaining:
            best_index = None
            best_i_perp = np.inf
            for internal_index in sorted(remaining):
                image_index = internal_index + 1
                trial_path = path.copy()
                trial_energies = energies.copy()
                trial_forces = forces.copy()
                trial_path[image_index] = candidate_path[image_index]
                trial_energies[image_index] = candidate_energies[image_index]
                trial_forces[image_index] = candidate_forces[image_index]
                trial_i_perp = path_diagnostics(
                    trial_path,
                    trial_energies,
                    trial_forces[1:-1],
                )["i_perp"]
                if trial_i_perp < best_i_perp:
                    best_i_perp = trial_i_perp
                    best_index = internal_index
            if best_i_perp > (
                diagnostic["i_perp"] * (1.0 + global_iperp_rtol) + 1e-12
            ):
                break
            ordered_indices.append(best_index)
            image_index = best_index + 1
            path[image_index] = candidate_path[image_index]
            energies[image_index] = candidate_energies[image_index]
            forces[image_index] = candidate_forces[image_index]
            diagnostic = path_diagnostics(path, energies, forces[1:-1])
            accepted[best_index] = True
            remaining.remove(best_index)
        return path, energies, forces, diagnostic, accepted
    else:
        raise ValueError(f"Unknown global I_perp acceptance order: {order}")
    for internal_index in ordered_indices:
        if not eligible[internal_index]:
            continue
        image_index = internal_index + 1
        trial_path = path.copy()
        trial_energies = energies.copy()
        trial_forces = forces.copy()
        trial_path[image_index] = candidate_path[image_index]
        trial_energies[image_index] = candidate_energies[image_index]
        trial_forces[image_index] = candidate_forces[image_index]
        trial_diag = path_diagnostics(
            trial_path,
            trial_energies,
            trial_forces[1:-1],
        )
        if trial_diag["i_perp"] <= (
            diagnostic["i_perp"] * (1.0 + global_iperp_rtol) + 1e-12
        ):
            path = trial_path
            energies = trial_energies
            forces = trial_forces
            diagnostic = trial_diag
            accepted[internal_index] = True
    return path, energies, forces, diagnostic, accepted


def greedy_bicriteria_recovery_acceptance(
    current_path,
    current_energies,
    current_forces,
    candidate_path,
    candidate_energies,
    candidate_forces,
    eligible,
    equidistance_ceiling,
    global_iperp_rtol=0.0,
    max_block_size=8,
):
    """Merge evaluated IVM recovery iterates under two path-level guards.

    A spring macrostep creates tangential spacing slack and the subsequent
    frozen-normal IVM subproblems remove the physical normal force.  Merging
    all independent image minimizers at once can spend more spacing slack
    than the spring step created.  This routine searches already-evaluated
    contiguous image blocks and commits only a block that

    1. does not increase the refreshed central-tangent ``I_perp``; and
    2. keeps the requested raw equidistance integral below its macrostep
       ceiling.

    No force or energy is evaluated here.  Candidate coordinates, energies,
    and forces must come from genuine frozen-normal IVM subproblem iterates.
    """
    path = np.asarray(current_path, dtype=float).copy()
    energies = np.asarray(current_energies, dtype=float).copy()
    forces = np.asarray(current_forces, dtype=float).copy()
    candidate_path = np.asarray(candidate_path, dtype=float)
    candidate_energies = np.asarray(candidate_energies, dtype=float)
    candidate_forces = np.asarray(candidate_forces, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    ceiling = float(equidistance_ceiling)
    if eligible.shape != (path.shape[0] - 2,):
        raise ValueError("eligible must contain one flag per internal image")
    if not np.isfinite(ceiling) or ceiling < 0.0:
        raise ValueError("equidistance_ceiling must be finite and nonnegative")

    diagnostic = path_diagnostics(path, energies, forces[1:-1])
    accepted = np.zeros_like(eligible)
    remaining = set(np.flatnonzero(eligible).tolist())
    maximum_block = min(int(max_block_size), int(eligible.size))
    if maximum_block < 1:
        return path, energies, forces, diagnostic, accepted

    # Central-difference tangents couple neighbouring image updates.  The
    # synchronized Jacobi iterate can therefore be a valid two-merit descent
    # even when every strict sub-block fails in isolation.  Test that already
    # evaluated collective state first; no OpenMM call is made here.
    if np.any(eligible):
        collective_path = path.copy()
        collective_energies = energies.copy()
        collective_forces = forces.copy()
        collective_images = np.flatnonzero(eligible) + 1
        collective_path[collective_images] = candidate_path[
            collective_images
        ]
        collective_energies[collective_images] = candidate_energies[
            collective_images
        ]
        collective_forces[collective_images] = candidate_forces[
            collective_images
        ]
        collective_diagnostic = path_diagnostics(
            collective_path,
            collective_energies,
            collective_forces[1:-1],
        )
        if (
            collective_diagnostic["equidistance_integral"]
            <= ceiling + 1e-14
            and collective_diagnostic["i_perp"]
            <= (
                diagnostic["i_perp"]
                * (1.0 + float(global_iperp_rtol))
                + 1e-12
            )
        ):
            accepted[eligible] = True
            return (
                collective_path,
                collective_energies,
                collective_forces,
                collective_diagnostic,
                accepted,
            )

    while remaining:
        best = None
        best_i_perp = np.inf
        best_equidistance = np.inf
        for block_size in range(1, maximum_block + 1):
            for start in range(0, eligible.size - block_size + 1):
                indices = [
                    index
                    for index in range(start, start + block_size)
                    if index in remaining
                ]
                if not indices:
                    continue
                trial_path = path.copy()
                trial_energies = energies.copy()
                trial_forces = forces.copy()
                image_indices = np.asarray(indices, dtype=int) + 1
                trial_path[image_indices] = candidate_path[image_indices]
                trial_energies[image_indices] = candidate_energies[
                    image_indices
                ]
                trial_forces[image_indices] = candidate_forces[image_indices]
                trial_diag = path_diagnostics(
                    trial_path, trial_energies, trial_forces[1:-1]
                )
                if trial_diag["equidistance_integral"] > ceiling + 1e-14:
                    continue
                if trial_diag["i_perp"] > (
                    diagnostic["i_perp"]
                    * (1.0 + float(global_iperp_rtol))
                    + 1e-12
                ):
                    continue
                if (
                    trial_diag["i_perp"] < best_i_perp - 1e-18
                    or (
                        abs(trial_diag["i_perp"] - best_i_perp) <= 1e-18
                        and trial_diag["equidistance_integral"]
                        < best_equidistance
                    )
                ):
                    best = (
                        indices,
                        trial_path,
                        trial_energies,
                        trial_forces,
                        trial_diag,
                    )
                    best_i_perp = float(trial_diag["i_perp"])
                    best_equidistance = float(
                        trial_diag["equidistance_integral"]
                    )
        if best is None:
            break
        indices, path, energies, forces, diagnostic = best
        accepted[np.asarray(indices, dtype=int)] = True
        remaining.difference_update(indices)
    return path, energies, forces, diagnostic, accepted


def ffprecon_subproblem_acceptance_mask(
    current_energies,
    candidate_energies,
    displacements,
    alpha,
    energy_rtol,
    mode,
    current_per_image_l2=None,
    candidate_per_image_l2=None,
    force_rtol=0.0,
):
    """Select proposals using a local IVP condition, not a coupled tangent proxy.

    ``I_perp`` and the per-image normal forces depend on neighbouring images
    through the central-difference tangent.  Consequently, a normal-force
    value measured on the all-candidate path is not a valid independent
    acceptance condition when images are merged one at a time.  The
    ``proximal_merit`` mode instead enforces the frozen-tangent IVP
    subproblem objective image by image; the caller then enforces monotone
    global ``I_perp`` on the actually merged path.
    """
    current_energies = np.asarray(current_energies, dtype=float)
    candidate_energies = np.asarray(candidate_energies, dtype=float)
    displacements = np.asarray(displacements, dtype=float)
    if mode == "proximal_merit":
        if alpha <= 0.0:
            raise ValueError("IVP alpha must be positive")
        candidate_objective = candidate_energies + (
            0.5 * np.sum(displacements**2, axis=1) / float(alpha)
        )
        tolerance = float(energy_rtol) * np.maximum(
            1.0, np.abs(current_energies)
        )
        return (
            candidate_objective <= current_energies + tolerance,
            candidate_objective,
        )
    if mode == "force_monotone":
        if current_per_image_l2 is None or candidate_per_image_l2 is None:
            raise ValueError("force_monotone requires per-image force norms")
        current_per_image_l2 = np.asarray(
            current_per_image_l2, dtype=float
        )
        candidate_per_image_l2 = np.asarray(
            candidate_per_image_l2, dtype=float
        )
        eligible = candidate_per_image_l2 <= (
            current_per_image_l2 * (1.0 + float(force_rtol)) + 1e-12
        )
        return eligible, candidate_energies.copy()
    raise ValueError(f"Unknown FF preconditioner acceptance mode: {mode}")


class ForceFieldPreconditionerBank:
    """Bond/angle/torsion block Hessians for the 20 normal subproblems.

    The matrices are assembled only from parameters already present in the
    current OpenMM system.  Building or applying them performs no OpenMM
    energy/force evaluation, so it does not alter the audited force-call
    axis.
    """

    KJ_MOL_PER_EV = 96.4853321233

    def __init__(self, backend, topology_pdb, c_stab=1.0):
        from ase.io import read
        from ase.utils.ff import Angle, Bond, Dihedral
        from openmm.unit import (
            kilojoule_per_mole,
            nanometer,
            radian,
        )

        self.template = read(str(topology_pdb))
        if 3 * len(self.template) != backend.dimension:
            raise ValueError(
                "ASE/OpenMM atom counts differ in FF preconditioner"
            )
        self.c_stab = float(c_stab)
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        for force in backend.system.getForces():
            force_name = type(force).__name__
            if force_name == "HarmonicBondForce":
                for index in range(force.getNumBonds()):
                    atom_i, atom_j, length, stiffness = (
                        force.getBondParameters(index)
                    )
                    self.bonds.append(
                        Bond(
                            int(atom_i),
                            int(atom_j),
                            stiffness.value_in_unit(
                                kilojoule_per_mole / nanometer**2
                            )
                            / self.KJ_MOL_PER_EV
                            / 100.0,
                            length.value_in_unit(nanometer) * 10.0,
                        )
                    )
            elif force_name == "HarmonicAngleForce":
                for index in range(force.getNumAngles()):
                    atom_i, atom_j, atom_k, angle, stiffness = (
                        force.getAngleParameters(index)
                    )
                    self.angles.append(
                        Angle(
                            int(atom_i),
                            int(atom_j),
                            int(atom_k),
                            stiffness.value_in_unit(
                                kilojoule_per_mole / radian**2
                            )
                            / self.KJ_MOL_PER_EV,
                            angle.value_in_unit(radian),
                        )
                    )
            elif force_name == "PeriodicTorsionForce":
                for index in range(force.getNumTorsions()):
                    (
                        atom_i,
                        atom_j,
                        atom_k,
                        atom_l,
                        periodicity,
                        phase,
                        stiffness,
                    ) = force.getTorsionParameters(index)
                    self.dihedrals.append(
                        Dihedral(
                            int(atom_i),
                            int(atom_j),
                            int(atom_k),
                            int(atom_l),
                            stiffness.value_in_unit(kilojoule_per_mole)
                            / self.KJ_MOL_PER_EV,
                            phase.value_in_unit(radian),
                            int(periodicity),
                        )
                    )
        if not self.bonds or not self.angles or not self.dihedrals:
            raise RuntimeError(
                "OpenMM bonded terms required by FF preconditioner are missing"
            )
        self.preconditioners = None
        self._shifted_solvers = {}
        self.rebuilds = 0

    def build(self, path):
        from ase.optimize.precon import FF

        path = np.asarray(path, dtype=float)
        self.preconditioners = []
        for position in path[1:-1]:
            atoms = self.template.copy()
            atoms.set_positions(position.reshape(-1, 3))
            # ASE's reduced force-field Hessian helper inverts the cell even
            # for this nonperiodic molecule.  A large inert cell avoids a
            # singular zero-cell matrix without changing Cartesian geometry
            # or requesting an OpenMM energy/force evaluation.
            if abs(float(np.linalg.det(np.asarray(atoms.cell)))) < 1e-12:
                atoms.set_cell(np.eye(3) * 100.0)
                atoms.set_pbc(False)
            preconditioner = FF(
                c_stab=self.c_stab,
                apply_cell=False,
                hessian="reduced",
                bonds=self.bonds,
                angles=self.angles,
                dihedrals=self.dihedrals,
            )
            preconditioner.make_precon(atoms)
            self.preconditioners.append(preconditioner)
        self._shifted_solvers = {}
        self.rebuilds += 1

    def solve_normal(
        self, image_index, vector, tangent, diagonal_shift=0.0
    ):
        """Apply the tangent-constrained force-field inverse Hessian."""
        if self.preconditioners is None:
            raise RuntimeError("FF preconditioner bank has not been built")
        image_index = int(image_index)
        preconditioner = self.preconditioners[image_index]
        vector = project_normal(
            np.asarray(vector, dtype=float)[None, :],
            np.asarray(tangent, dtype=float)[None, :],
        )[0]
        diagonal_shift = float(diagonal_shift)
        if diagonal_shift > 0.0:
            from scipy.sparse import eye
            from scipy.sparse.linalg import factorized

            key = (image_index, diagonal_shift)
            solver = self._shifted_solvers.get(key)
            if solver is None:
                matrix = preconditioner.P + diagonal_shift * eye(
                    preconditioner.P.shape[0], format="csr"
                )
                solver = factorized(matrix.tocsc())
                self._shifted_solvers[key] = solver
            inverse_vector = solver(vector)
            inverse_tangent = solver(tangent)
        else:
            inverse_vector = preconditioner.solve(vector)
            inverse_tangent = preconditioner.solve(tangent)
        denominator = float(np.dot(tangent, inverse_tangent))
        if not np.isfinite(denominator) or abs(denominator) < 1e-20:
            raise RuntimeError("Singular tangent KKT denominator")
        solution = (
            inverse_vector
            - inverse_tangent
            * float(np.dot(tangent, inverse_vector))
            / denominator
        )
        return project_normal(solution[None, :], tangent[None, :])[0]

    def directions(
        self,
        normal_gradients,
        tangents,
        max_step,
        step_scales,
        curvature_memory=None,
        alpha=1.0,
    ):
        if self.preconditioners is None:
            raise RuntimeError("FF preconditioner bank has not been built")
        directions = []
        for image_index, (gradient, tangent, scale) in enumerate(zip(
            normal_gradients,
            tangents,
            step_scales,
        )):
            if curvature_memory is None:
                direction = -self.solve_normal(
                    image_index, gradient, tangent
                )
            else:
                direction = curvature_memory.direction_with_operator(
                    image_index,
                    gradient,
                    tangent,
                    alpha,
                    lambda vector, index=image_index, tau=tangent: (
                        self.solve_normal(index, vector, tau)
                    ),
                )
            direction = project_normal(
                direction[None, :], tangent[None, :]
            )
            direction = direction[0]
            if (
                not np.all(np.isfinite(direction))
                or float(np.dot(direction, gradient)) >= 0.0
            ):
                direction = -gradient
            direction = clip_max_atomic_displacement(
                direction, float(max_step)
            )
            directions.append(float(scale) * direction)
        return np.asarray(directions)


class ProjectedLBFGSMemory:
    def __init__(
        self,
        n_images: int,
        history_size: int,
        initial_curvature: float = 0.0,
    ):
        self.history_size = int(history_size)
        self.initial_curvature = float(initial_curvature)
        self.pairs = [[] for _ in range(n_images)]
        self.previous = [None for _ in range(n_images)]

    def reset(self):
        self.pairs = [[] for _ in self.pairs]
        self.previous = [None for _ in self.previous]

    def observe(self, positions, gradients, accept_mask=None):
        if accept_mask is None:
            accept_mask = np.ones(len(positions), dtype=bool)
        for i, (position, gradient, accepted) in enumerate(
            zip(positions, gradients, accept_mask)
        ):
            if not accepted:
                continue
            previous = self.previous[i]
            if previous is not None:
                self.pairs[i].append(
                    (
                        position.copy() - previous[0],
                        gradient.copy() - previous[1],
                    )
                )
                if len(self.pairs[i]) > self.history_size:
                    self.pairs[i].pop(0)
            self.previous[i] = (position.copy(), gradient.copy())

    def direction(self, image_index, gradient, tangent, alpha):
        gradient = project_normal(
            np.asarray(gradient)[None, :], np.asarray(tangent)[None, :]
        )[0]
        usable = []
        for raw_s, raw_y in self.pairs[image_index]:
            s = project_normal(raw_s[None, :], tangent[None, :])[0]
            y = (
                project_normal(raw_y[None, :], tangent[None, :])[0]
                + s / alpha
            )
            snorm = np.linalg.norm(s)
            ynorm = np.linalg.norm(y)
            ys = float(np.dot(y, s))
            if (
                snorm > 1e-14
                and ynorm > 1e-14
                and np.isfinite(ys)
                and ys > 1e-4 * snorm * ynorm
            ):
                usable.append((s, y))
        q = gradient.copy()
        coefficients = []
        for s, y in reversed(usable):
            rho = 1.0 / np.dot(y, s)
            coefficient = rho * np.dot(s, q)
            coefficients.append((coefficient, rho, s, y))
            q -= coefficient * y
        if self.initial_curvature > 0.0:
            if usable:
                last_s, last_y = usable[-1]
                gamma = float(
                    np.dot(last_s, last_y) / np.dot(last_y, last_y)
                )
            else:
                gamma = 1.0 / self.initial_curvature
        else:
            # Legacy behavior retained for reproducibility of existing runs.
            gamma = alpha
        result = gamma * q
        for coefficient, rho, s, y in reversed(coefficients):
            beta = rho * np.dot(y, result)
            result += (coefficient - beta) * s
        direction = -result
        direction = project_normal(direction[None, :], tangent[None, :])[0]
        if not np.all(np.isfinite(direction)) or np.dot(direction, gradient) >= 0:
            direction = -gamma * gradient
        return direction

    def direction_with_operator(
        self,
        image_index,
        gradient,
        tangent,
        alpha,
        apply_initial_inverse,
    ):
        """L-BFGS direction seeded by a physical inverse-Hessian operator.

        The run-local ``s,y`` pairs are observed at consecutive *outer*
        subproblem centres.  The proximal gradient is zero at every such
        centre, so their gradient differences contain physical-energy
        curvature only.  Adding ``s/alpha`` here would double-count a
        proximal curvature that is absent from these observations.
        """
        del alpha
        gradient = project_normal(
            np.asarray(gradient)[None, :], np.asarray(tangent)[None, :]
        )[0]
        usable = []
        for raw_s, raw_y in self.pairs[image_index]:
            s = project_normal(raw_s[None, :], tangent[None, :])[0]
            y = project_normal(raw_y[None, :], tangent[None, :])[0]
            snorm = np.linalg.norm(s)
            ynorm = np.linalg.norm(y)
            ys = float(np.dot(y, s))
            if (
                snorm > 1e-14
                and ynorm > 1e-14
                and np.isfinite(ys)
                and ys > 1e-4 * snorm * ynorm
            ):
                usable.append((s, y))
        q = gradient.copy()
        coefficients = []
        for s, y in reversed(usable):
            rho = 1.0 / float(np.dot(y, s))
            coefficient = rho * float(np.dot(s, q))
            coefficients.append((coefficient, rho, s, y))
            q -= coefficient * y
        result = np.asarray(apply_initial_inverse(q), dtype=float)
        for coefficient, rho, s, y in reversed(coefficients):
            beta = rho * float(np.dot(y, result))
            result += (coefficient - beta) * s
        direction = -project_normal(
            result[None, :], tangent[None, :]
        )[0]
        if (
            not np.all(np.isfinite(direction))
            or float(np.dot(direction, gradient)) >= 0.0
        ):
            direction = -np.asarray(
                apply_initial_inverse(gradient), dtype=float
            )
        return project_normal(direction[None, :], tangent[None, :])[0]


def _projected_lbfgs_direction(
    gradient,
    tangent,
    pairs,
    initial_inverse_scale,
):
    """Return an intrinsic L-BFGS direction in a frozen normal space."""
    gradient = project_normal(
        np.asarray(gradient, dtype=float)[None, :],
        np.asarray(tangent, dtype=float)[None, :],
    )[0]
    usable = []
    for raw_s, raw_y in pairs:
        s = project_normal(raw_s[None, :], tangent[None, :])[0]
        y = project_normal(raw_y[None, :], tangent[None, :])[0]
        snorm = np.linalg.norm(s)
        ynorm = np.linalg.norm(y)
        ys = float(np.dot(y, s))
        if (
            snorm > 1e-14
            and ynorm > 1e-14
            and np.isfinite(ys)
            and ys > 1e-6 * snorm * ynorm
        ):
            usable.append((s, y))
    q = gradient.copy()
    coefficients = []
    for s, y in reversed(usable):
        rho = 1.0 / float(np.dot(y, s))
        coefficient = rho * float(np.dot(s, q))
        coefficients.append((coefficient, rho, s, y))
        q -= coefficient * y
    if usable:
        last_s, last_y = usable[-1]
        gamma = float(np.dot(last_s, last_y) / np.dot(last_y, last_y))
    else:
        gamma = float(initial_inverse_scale)
    gamma = max(gamma, 1e-12)
    result = gamma * q
    for coefficient, rho, s, y in reversed(coefficients):
        beta = rho * float(np.dot(y, result))
        result += (coefficient - beta) * s
    direction = project_normal((-result)[None, :], tangent[None, :])[0]
    if (
        not np.all(np.isfinite(direction))
        or float(np.dot(direction, gradient)) >= -1e-14
    ):
        return project_normal((-gradient)[None, :], tangent[None, :])[0]
    return direction


def _projected_lbfgs_direction_with_operator(
    gradient,
    tangent,
    pairs,
    apply_initial_inverse,
):
    """Intrinsic L-BFGS for one frozen subproblem with an operator H0."""
    gradient = project_normal(
        np.asarray(gradient, dtype=float)[None, :],
        np.asarray(tangent, dtype=float)[None, :],
    )[0]
    usable = []
    for raw_s, raw_y in pairs:
        s = project_normal(raw_s[None, :], tangent[None, :])[0]
        y = project_normal(raw_y[None, :], tangent[None, :])[0]
        snorm = np.linalg.norm(s)
        ynorm = np.linalg.norm(y)
        ys = float(np.dot(y, s))
        if (
            snorm > 1e-14
            and ynorm > 1e-14
            and np.isfinite(ys)
            and ys > 1e-6 * snorm * ynorm
        ):
            usable.append((s, y))
    q = gradient.copy()
    coefficients = []
    for s, y in reversed(usable):
        rho = 1.0 / float(np.dot(y, s))
        coefficient = rho * float(np.dot(s, q))
        coefficients.append((coefficient, rho, s, y))
        q -= coefficient * y
    result = np.asarray(apply_initial_inverse(q), dtype=float)
    for coefficient, rho, s, y in reversed(coefficients):
        beta = rho * float(np.dot(y, result))
        result += (coefficient - beta) * s
    direction = project_normal((-result)[None, :], tangent[None, :])[0]
    if (
        not np.all(np.isfinite(direction))
        or float(np.dot(direction, gradient)) >= -1e-18
    ):
        direction = project_normal(
            (-np.asarray(apply_initial_inverse(gradient)))[None, :],
            tangent[None, :],
        )[0]
    return direction


def _screen_ala_dsel_band(label, raw_directions, gradients, tangents, max_step):
    """Apply the legacy per-image Ala cap to one whole-band proposal."""
    raw_directions = np.asarray(raw_directions, dtype=float)
    gradients = np.asarray(gradients, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    if not (
        raw_directions.shape == gradients.shape == tangents.shape
        and raw_directions.ndim == 2
        and np.all(np.isfinite(raw_directions))
    ):
        return {
            "label": str(label),
            "valid": False,
            "error": "shape-or-finite-check",
            "step": None,
            "raw_decrease": 0.0,
            "capped_decrease": 0.0,
        }
    # Preserve the legacy Ala candidate byte path: its direction helpers
    # already project, then the existing cap is applied before one final
    # projection.  Do not insert an extra pre-cap projection here.
    raw = raw_directions.copy()
    raw_decrease = -float(np.sum(gradients * raw))
    if not np.isfinite(raw_decrease) or raw_decrease <= 0.0:
        return {
            "label": str(label),
            "valid": False,
            "error": "raw-direction-not-strict-descent",
            "step": None,
            "raw_decrease": float(raw_decrease),
            "capped_decrease": 0.0,
        }
    capped = np.asarray(
        [
            project_normal(
                clip_max_atomic_displacement(direction, max_step)[None, :],
                tangent[None, :],
            )[0]
            for direction, tangent in zip(raw, tangents)
        ],
        dtype=float,
    )
    capped_decrease = -float(np.sum(gradients * capped))
    if (
        not np.all(np.isfinite(capped))
        or not np.isfinite(capped_decrease)
        or capped_decrease <= 0.0
    ):
        return {
            "label": str(label),
            "valid": False,
            "error": "capped-direction-not-strict-descent",
            "step": None,
            "raw_decrease": float(raw_decrease),
            "capped_decrease": float(capped_decrease),
        }
    return {
        "label": str(label),
        "valid": True,
        "error": "none",
        "step": capped,
        "raw_decrease": float(raw_decrease),
        "capped_decrease": float(capped_decrease),
    }


def _select_ala_dsel_direction(h0_proposal, lbfgs_proposal):
    """Select the larger capped predicted decrease; exact tie chooses H0."""
    if h0_proposal["valid"] and lbfgs_proposal["valid"]:
        if (
            h0_proposal["capped_decrease"]
            >= lbfgs_proposal["capped_decrease"]
        ):
            return h0_proposal
        return lbfgs_proposal
    if h0_proposal["valid"]:
        return h0_proposal
    if lbfgs_proposal["valid"]:
        return lbfgs_proposal
    raise RuntimeError(
        "Both Ala DSEL H0 and legacy-LBFGS proposals are invalid: "
        f"H0={h0_proposal['error']}; LBFGS={lbfgs_proposal['error']}"
    )


def _ala_dsel_legacy_uses_curvature(
    local_pairs,
    curvature_memory,
    internal_index,
    tangent,
):
    """Return whether the legacy branch has one usable secant pair.

    When no usable curvature exists, the legacy direction is the H0 direction
    up to redundant floating-point projections.  Reusing the already computed
    legacy bytes in that case makes the no-history tie exact without changing
    the legacy candidate.
    """
    tangent = np.asarray(tangent, dtype=float)
    if local_pairs:
        pairs = local_pairs
        curvature_tolerance = 1e-6
    elif curvature_memory is not None:
        pairs = curvature_memory.pairs[internal_index]
        curvature_tolerance = 1e-4
    else:
        return False
    for raw_s, raw_y in pairs:
        s = project_normal(
            np.asarray(raw_s, dtype=float)[None, :], tangent[None, :]
        )[0]
        y = project_normal(
            np.asarray(raw_y, dtype=float)[None, :], tangent[None, :]
        )[0]
        snorm = float(np.linalg.norm(s))
        ynorm = float(np.linalg.norm(y))
        sy = float(np.dot(s, y))
        if (
            snorm > 1e-14
            and ynorm > 1e-14
            and np.isfinite(sy)
            and sy > curvature_tolerance * snorm * ynorm
        ):
            return True
    return False


def _score_ala_curve_proposal_after_geometry(
    proposal,
    reference_path,
    current_path,
    displacement,
    residual_gradient,
    tangents,
):
    """Score a capped proposal after Ala's zero-call spring projection.

    B040 compares the actual capped/stored step, not an upstream raw vector.
    The equal-spring mapping is part of Ala's coordinate pipeline and calls no
    physical evaluator, so include it before selecting which proposal receives
    the one paid OpenMM evaluation.
    """
    scored = dict(proposal)
    if not proposal["valid"]:
        scored["post_geometry_decrease"] = 0.0
        scored["post_geometry_path"] = None
        return scored
    proposed_displacement = project_normal(
        np.asarray(displacement, dtype=float)
        + np.asarray(proposal["step"], dtype=float),
        np.asarray(tangents, dtype=float),
    )
    raw_trial = np.asarray(reference_path, dtype=float).copy()
    raw_trial[1:-1] += proposed_displacement
    projected_trial, _ = central_spring_force_equilibrium(raw_trial)
    actual_step = project_normal(
        projected_trial[1:-1]
        - np.asarray(current_path, dtype=float)[1:-1],
        np.asarray(tangents, dtype=float),
    )
    decrease = -float(
        np.sum(np.asarray(residual_gradient, dtype=float) * actual_step)
    )
    if not np.isfinite(decrease) or decrease <= 0.0:
        scored["valid"] = False
        scored["error"] = "post-geometry-step-not-strict-descent"
        scored["post_geometry_decrease"] = float(decrease)
        scored["post_geometry_path"] = projected_trial
        return scored
    scored["post_geometry_decrease"] = float(decrease)
    scored["post_geometry_path"] = projected_trial
    # The selector consumes capped_decrease.  At this point its scientific
    # meaning is the predicted decrease of the actual post-geometry step.
    scored["capped_decrease"] = float(decrease)
    return scored


def solve_projected_proximal_band(
    path,
    energies,
    forces,
    tangents,
    evaluators,
    alpha,
    max_step,
    max_new_calls,
    history_size=7,
    curvature_memory=None,
    armijo=1e-4,
    backtrack_factor=0.5,
    kkt_rtol=0.1,
    kkt_atol=1e-8,
    min_accepted_steps=0,
    preconditioner_bank=None,
    active_images=None,
    direction_selection="legacy",
    max_accepted_steps=None,
):
    """Solve frozen-tangent IVM subproblems without tangential null variables.

    Every iterate, gradient, secant pair, and search direction is projected
    into the frozen central-difference normal space.  Armijo globalization
    is applied to the exact proximal objective and convergence is measured
    by its KKT residual.  If an L-BFGS direction fails, the same proximal
    subproblem is restarted along projected steepest descent; this is a
    safeguard, not a different optimization objective.
    """
    n_internal = path.shape[0] - 2
    if active_images is None:
        active_images = np.ones(n_internal, dtype=bool)
    else:
        active_images = np.asarray(active_images, dtype=bool)
        if active_images.shape != (n_internal,):
            raise ValueError(
                "active_images must contain one flag per internal image"
            )
    candidate_path = path.copy()
    candidate_energies = energies.copy()
    candidate_forces = forces.copy()
    starting_counters = counter_arrays(evaluators)
    position_traces = [[path[i + 1].copy()] for i in range(n_internal)]
    energy_traces = [[float(energies[i + 1])] for i in range(n_internal)]
    force_traces = [[forces[i + 1].copy()] for i in range(n_internal)]
    counter_traces = [
        [
            {
                name: int(starting_counters[name][i + 1])
                for name in starting_counters
            }
        ]
        for i in range(n_internal)
    ]
    used_calls = np.zeros(n_internal, dtype=int)
    initial_kkt = np.full(n_internal, np.nan, dtype=float)
    final_kkt = np.full(n_internal, np.nan, dtype=float)
    converged = np.zeros(n_internal, dtype=bool)
    hit_call_limit = np.zeros(n_internal, dtype=bool)
    accepted_steps = np.zeros(n_internal, dtype=int)
    backtracks = np.zeros(n_internal, dtype=int)
    curvature_restarts = np.zeros(n_internal, dtype=int)
    dsel_selection = []
    dsel_h0_capped_decrease = []
    dsel_lbfgs_capped_decrease = []

    if direction_selection not in {"legacy", "ala_dsel"}:
        raise ValueError("Unknown projected-proximal direction selection")
    if max_accepted_steps is not None and int(max_accepted_steps) < 1:
        raise ValueError("max_accepted_steps must be positive or None")

    for internal_index in range(n_internal):
        if not active_images[internal_index]:
            continue
        image_index = internal_index + 1
        reference = path[image_index].copy()
        tangent = tangents[internal_index].copy()
        displacement = np.zeros_like(reference)
        position = reference.copy()
        energy = float(energies[image_index])
        force = forces[image_index].copy()
        objective = energy
        normal_gradient = project_normal(
            (-force)[None, :], tangent[None, :]
        )[0]
        residual_gradient = normal_gradient.copy()
        initial_kkt[internal_index] = np.linalg.norm(residual_gradient)
        target_kkt = max(
            float(kkt_atol),
            float(kkt_rtol) * initial_kkt[internal_index],
        )
        local_pairs = []
        start_calls = evaluators[image_index].counters.joint_evaluations

        def append_trace_if_new(joint_before):
            if (
                evaluators[image_index].counters.joint_evaluations
                <= joint_before
            ):
                return
            position_traces[internal_index].append(position.copy())
            energy_traces[internal_index].append(float(energy))
            force_traces[internal_index].append(force.copy())
            counter_traces[internal_index].append(
                {
                    name: int(
                        getattr(evaluators[image_index].counters, name)
                    )
                    for name in starting_counters
                }
            )

        while True:
            residual_norm = float(np.linalg.norm(residual_gradient))
            if (
                max_accepted_steps is not None
                and accepted_steps[internal_index]
                >= int(max_accepted_steps)
            ):
                break
            if (
                residual_norm <= target_kkt
                and accepted_steps[internal_index]
                >= int(min_accepted_steps)
            ):
                converged[internal_index] = True
                break
            used = (
                evaluators[image_index].counters.joint_evaluations
                - start_calls
            )
            if used >= max_new_calls:
                hit_call_limit[internal_index] = True
                break

            apply_initial_inverse = None
            if preconditioner_bank is not None:
                apply_initial_inverse = (
                    lambda vector, index=internal_index, tau=tangent: (
                        preconditioner_bank.solve_normal(
                            index,
                            vector,
                            tau,
                            diagonal_shift=1.0 / float(alpha),
                        )
                    )
                )
            if local_pairs and apply_initial_inverse is not None:
                direction = _projected_lbfgs_direction_with_operator(
                    residual_gradient,
                    tangent,
                    local_pairs,
                    apply_initial_inverse,
                )
            elif local_pairs:
                direction = _projected_lbfgs_direction(
                    residual_gradient,
                    tangent,
                    local_pairs,
                    initial_inverse_scale=alpha,
                )
            elif (
                apply_initial_inverse is not None
                and curvature_memory is not None
            ):
                direction = curvature_memory.direction_with_operator(
                    internal_index,
                    residual_gradient,
                    tangent,
                    alpha,
                    apply_initial_inverse,
                )
            elif apply_initial_inverse is not None:
                direction = -np.asarray(
                    apply_initial_inverse(residual_gradient), dtype=float
                )
            elif curvature_memory is not None:
                direction = curvature_memory.direction(
                    internal_index,
                    residual_gradient,
                    tangent,
                    alpha,
                )
            else:
                # The exact steepest-descent fallback for the same proximal
                # subproblem. Backtracking determines its scale.
                direction = -residual_gradient
            if direction_selection == "ala_dsel":
                if apply_initial_inverse is None:
                    raise RuntimeError(
                        "Ala DSEL requires the frozen FFrbk H0 operator"
                    )
                legacy_raw = np.asarray(direction, dtype=float).copy()
                if not _ala_dsel_legacy_uses_curvature(
                    local_pairs,
                    curvature_memory,
                    internal_index,
                    tangent,
                ):
                    h0_raw = legacy_raw.copy()
                else:
                    h0_raw = -np.asarray(
                        apply_initial_inverse(residual_gradient), dtype=float
                    )
                h0_proposal = _screen_ala_dsel_band(
                    "H0",
                    h0_raw[None, :],
                    residual_gradient[None, :],
                    tangent[None, :],
                    max_step,
                )
                lbfgs_proposal = _screen_ala_dsel_band(
                    "LBFGS",
                    legacy_raw[None, :],
                    residual_gradient[None, :],
                    tangent[None, :],
                    max_step,
                )
                if not h0_proposal["valid"] and not lbfgs_proposal["valid"]:
                    # Preserve the legacy solver's own invalid-direction
                    # safeguard instead of making DSEL a new failure mode.
                    direction = legacy_raw
                    selected_label = "SAFEGUARD"
                else:
                    selected_proposal = _select_ala_dsel_direction(
                        h0_proposal, lbfgs_proposal
                    )
                    direction = selected_proposal["step"][0]
                    selected_label = selected_proposal["label"]
                dsel_selection.append(selected_label)
                dsel_h0_capped_decrease.append(
                    h0_proposal["capped_decrease"]
                )
                dsel_lbfgs_capped_decrease.append(
                    lbfgs_proposal["capped_decrease"]
                )
            else:
                direction = clip_max_atomic_displacement(
                    direction, max_step
                )
                direction = project_normal(
                    direction[None, :], tangent[None, :]
                )[0]
            directional_derivative = float(
                np.dot(residual_gradient, direction)
            )
            used_curvature = bool(local_pairs) or curvature_memory is not None
            if (
                not np.all(np.isfinite(direction))
                or directional_derivative >= -1e-18
            ):
                local_pairs.clear()
                fallback_direction = -residual_gradient
                if apply_initial_inverse is not None:
                    fallback_direction = -np.asarray(
                        apply_initial_inverse(residual_gradient), dtype=float
                    )
                direction = clip_max_atomic_displacement(
                    fallback_direction, max_step
                )
                direction = project_normal(
                    direction[None, :], tangent[None, :]
                )[0]
                directional_derivative = float(
                    np.dot(residual_gradient, direction)
                )
                curvature_restarts[internal_index] += 1
                used_curvature = False
            if directional_derivative >= -1e-18:
                break

            accepted = False
            tried_steepest_fallback = not used_curvature
            for _ in range(2):
                step_length = 1.0
                for _ in range(64):
                    used = (
                        evaluators[image_index].counters.joint_evaluations
                        - start_calls
                    )
                    trial_displacement = project_normal(
                        (
                            displacement + step_length * direction
                        )[None, :],
                        tangent[None, :],
                    )[0]
                    trial_position = reference + trial_displacement
                    if (
                        used >= max_new_calls
                        and not evaluators[image_index].has_cached(
                            trial_position
                        )
                    ):
                        hit_call_limit[internal_index] = True
                        break
                    joint_before = (
                        evaluators[
                            image_index
                        ].counters.joint_evaluations
                    )
                    trial_energy, trial_force = evaluators[
                        image_index
                    ].query(trial_position, "both")
                    trial_normal_gradient = project_normal(
                        (-trial_force)[None, :], tangent[None, :]
                    )[0]
                    trial_residual = (
                        trial_normal_gradient
                        + trial_displacement / float(alpha)
                    )
                    trial_objective = float(
                        trial_energy
                        + 0.5
                        * np.dot(trial_displacement, trial_displacement)
                        / float(alpha)
                    )
                    trial_residual_norm = float(
                        np.linalg.norm(trial_residual)
                    )
                    armijo_ok = trial_objective <= (
                        objective
                        + float(armijo)
                        * step_length
                        * directional_derivative
                        + 1e-14 * max(1.0, abs(objective))
                    )
                    if armijo_ok:
                        old_displacement = displacement.copy()
                        old_residual = residual_gradient.copy()
                        displacement = trial_displacement
                        position = trial_position
                        energy = float(trial_energy)
                        force = trial_force.copy()
                        objective = trial_objective
                        normal_gradient = trial_normal_gradient
                        residual_gradient = trial_residual
                        s = displacement - old_displacement
                        y = residual_gradient - old_residual
                        sy = float(np.dot(s, y))
                        if (
                            sy
                            > 1e-6
                            * np.linalg.norm(s)
                            * np.linalg.norm(y)
                        ):
                            local_pairs.append((s.copy(), y.copy()))
                            if len(local_pairs) > int(history_size):
                                local_pairs.pop(0)
                        accepted_steps[internal_index] += 1
                        accepted = True
                        append_trace_if_new(joint_before)
                        break
                    backtracks[internal_index] += 1
                    append_trace_if_new(joint_before)
                    step_length *= float(backtrack_factor)
                    if step_length < 2.0**-40:
                        break
                if accepted or hit_call_limit[internal_index]:
                    break
                if tried_steepest_fallback:
                    break
                # Discard an unhelpful curvature model and guarantee that the
                # projected-force direction is explicitly tried.
                local_pairs.clear()
                direction = clip_max_atomic_displacement(
                    -residual_gradient, max_step
                )
                direction = project_normal(
                    direction[None, :], tangent[None, :]
                )[0]
                directional_derivative = float(
                    np.dot(residual_gradient, direction)
                )
                curvature_restarts[internal_index] += 1
                tried_steepest_fallback = True
                if directional_derivative >= -1e-18:
                    break
            if not accepted:
                break

        used = (
            evaluators[image_index].counters.joint_evaluations - start_calls
        )
        if used > max_new_calls:
            raise RuntimeError(
                "Projected proximal solver exceeded its image-call budget"
            )
        final_kkt[internal_index] = np.linalg.norm(residual_gradient)
        candidate_path[image_index] = position
        candidate_energies[image_index] = energy
        candidate_forces[image_index] = force
        used_calls[internal_index] = used
        # Cache hits can change the accepted state without adding a new trace
        # point. Keep the final accepted state at the existing call label.
        position_traces[internal_index][used] = position.copy()
        energy_traces[internal_index][used] = float(energy)
        force_traces[internal_index][used] = force.copy()

    max_used = int(np.max(used_calls))
    synchronized_paths = []
    synchronized_energies = []
    synchronized_forces = []
    synchronized_counters = []
    final_counters = counter_arrays(evaluators)
    for local_call in range(1, max_used + 1):
        trace_path = path.copy()
        trace_energies = energies.copy()
        trace_forces = forces.copy()
        trace_counters = {
            name: starting_counters[name].copy()
            for name in starting_counters
        }
        for internal_index in range(n_internal):
            image_index = internal_index + 1
            trace_index = min(local_call, int(used_calls[internal_index]))
            trace_path[image_index] = position_traces[internal_index][
                trace_index
            ]
            trace_energies[image_index] = energy_traces[internal_index][
                trace_index
            ]
            trace_forces[image_index] = force_traces[internal_index][
                trace_index
            ]
            if trace_index > 0:
                snapshot = counter_traces[internal_index][trace_index]
                for name in trace_counters:
                    trace_counters[name][image_index] = snapshot[name]
            if local_call >= used_calls[internal_index]:
                for name in trace_counters:
                    trace_counters[name][image_index] = final_counters[name][
                        image_index
                    ]
        synchronized_paths.append(trace_path)
        synchronized_energies.append(trace_energies)
        synchronized_forces.append(trace_forces)
        synchronized_counters.append(trace_counters)
    report = {
        "initial_kkt_l2": initial_kkt,
        "final_kkt_l2": final_kkt,
        "relative_kkt": final_kkt / np.maximum(initial_kkt, 1e-30),
        "converged": converged,
        "hit_call_limit": hit_call_limit,
        "accepted_steps": accepted_steps,
        "backtracks": backtracks,
        "curvature_restarts": curvature_restarts,
        "ala_dsel": direction_selection == "ala_dsel",
        "direction_selection": np.asarray(dsel_selection, dtype="U16"),
        "h0_capped_predicted_decrease": np.asarray(
            dsel_h0_capped_decrease, dtype=float
        ),
        "lbfgs_capped_predicted_decrease": np.asarray(
            dsel_lbfgs_capped_decrease, dtype=float
        ),
    }
    return (
        candidate_path,
        candidate_energies,
        candidate_forces,
        used_calls,
        synchronized_paths,
        synchronized_energies,
        synchronized_forces,
        synchronized_counters,
        report,
    )


def solve_curve_projected_proximal_band(
    path,
    energies,
    forces,
    tangents,
    evaluators,
    alpha,
    max_step,
    max_new_calls,
    history_size=7,
    armijo=1e-4,
    backtrack_factor=0.5,
    kkt_rtol=0.1,
    kkt_atol=1e-8,
    min_accepted_steps=0,
    preconditioner_bank=None,
    curvature_memory=None,
    active_images=None,
    enforce_global_iperp=True,
    line_search_trial_limit=64,
    objective_acceptance="per_image",
    direction_selection="legacy",
    max_accepted_steps=None,
):
    """Solve one frozen-normal IVM problem on the equal-spring gauge.

    A trial is first constructed in the frozen central-difference normal
    space.  It is then mapped, without an OpenMM query, to identical-spring
    equilibrium under the central-tangent spring flow.  Only that final
    coordinate is evaluated.  Hence every published inner iterate has the
    requested parameterization, while the exact proximal objective and KKT
    residual still certify the IVM subproblem.

    The curve projection couples images, so this solver deliberately uses a
    synchronized whole-band line search.  Rejected projected trials remain in
    the evaluator counters, but their coordinates are not published.
    """
    path = np.asarray(path, dtype=float)
    energies = np.asarray(energies, dtype=float)
    forces = np.asarray(forces, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    n_internal = path.shape[0] - 2
    if int(line_search_trial_limit) < 1:
        raise ValueError("line-search trial limit must be at least one")
    if objective_acceptance not in ("per_image", "path_sum"):
        raise ValueError(
            "objective_acceptance must be 'per_image' or 'path_sum'"
        )
    if active_images is not None and not np.all(
        np.asarray(active_images, dtype=bool)
    ):
        raise ValueError(
            "curve-projected IVM requires a synchronized all-image update"
        )

    reference_path = path.copy()
    reference_energies = energies.copy()
    current_path = path.copy()
    current_energies = energies.copy()
    current_forces = forces.copy()
    current_diagnostic = path_diagnostics(
        current_path, current_energies, current_forces[1:-1]
    )
    starting_counters = counter_arrays(evaluators)
    start_calls = starting_counters["joint_evaluations"][1:-1].copy()

    displacement = np.zeros_like(path[1:-1])
    normal_gradient = project_normal(-forces[1:-1], tangents)
    residual_gradient = normal_gradient.copy()
    objective = energies[1:-1].copy()
    initial_kkt = np.linalg.norm(residual_gradient, axis=1)
    target_kkt = np.maximum(
        float(kkt_atol), float(kkt_rtol) * initial_kkt
    )
    local_pairs = [[] for _ in range(n_internal)]
    accepted_steps = np.zeros(n_internal, dtype=int)
    backtracks = np.zeros(n_internal, dtype=int)
    curvature_restarts = np.zeros(n_internal, dtype=int)
    hit_call_limit = np.zeros(n_internal, dtype=bool)
    synchronized_paths = []
    synchronized_energies = []
    synchronized_forces = []
    synchronized_counters = []
    dsel_selection = []
    dsel_h0_capped_decrease = []
    dsel_lbfgs_capped_decrease = []
    if direction_selection not in {"legacy", "ala_dsel"}:
        raise ValueError("Unknown curve-projected direction selection")
    if max_accepted_steps is not None and int(max_accepted_steps) < 1:
        raise ValueError("max_accepted_steps must be positive or None")

    def used_calls():
        return (
            counter_arrays(evaluators)["joint_evaluations"][1:-1]
            - start_calls
        )

    while True:
        residual_norm = np.linalg.norm(residual_gradient, axis=1)
        if (
            max_accepted_steps is not None
            and int(np.min(accepted_steps)) >= int(max_accepted_steps)
        ):
            break
        if (
            np.all(residual_norm <= target_kkt)
            and np.all(accepted_steps >= int(min_accepted_steps))
        ):
            break
        if int(np.max(used_calls())) >= int(max_new_calls):
            hit_call_limit = residual_norm > target_kkt
            break

        directions = np.zeros_like(residual_gradient)
        legacy_raw_directions = np.zeros_like(residual_gradient)
        h0_raw_directions = np.zeros_like(residual_gradient)
        for internal_index in range(n_internal):
            tangent = tangents[internal_index]
            residual = residual_gradient[internal_index]
            apply_initial_inverse = None
            if preconditioner_bank is not None:
                apply_initial_inverse = (
                    lambda vector, index=internal_index, tau=tangent: (
                        preconditioner_bank.solve_normal(
                            index,
                            vector,
                            tau,
                            diagonal_shift=1.0 / float(alpha),
                        )
                    )
                )
            if local_pairs[internal_index] and apply_initial_inverse is not None:
                direction = _projected_lbfgs_direction_with_operator(
                    residual,
                    tangent,
                    local_pairs[internal_index],
                    apply_initial_inverse,
                )
            elif local_pairs[internal_index]:
                direction = _projected_lbfgs_direction(
                    residual,
                    tangent,
                    local_pairs[internal_index],
                    initial_inverse_scale=alpha,
                )
            elif (
                apply_initial_inverse is not None
                and curvature_memory is not None
            ):
                direction = curvature_memory.direction_with_operator(
                    internal_index,
                    residual,
                    tangent,
                    alpha,
                    apply_initial_inverse,
                )
            elif apply_initial_inverse is not None:
                direction = -np.asarray(
                    apply_initial_inverse(residual), dtype=float
                )
            elif curvature_memory is not None:
                direction = curvature_memory.direction(
                    internal_index,
                    residual,
                    tangent,
                    alpha,
                )
            else:
                direction = -residual
            if direction_selection == "ala_dsel":
                if apply_initial_inverse is None:
                    raise RuntimeError(
                        "Ala DSEL requires the frozen FFrbk H0 operator"
                    )
                legacy_raw = np.asarray(direction, dtype=float).copy()
                if not _ala_dsel_legacy_uses_curvature(
                    local_pairs[internal_index],
                    curvature_memory,
                    internal_index,
                    tangent,
                ):
                    h0_raw = legacy_raw.copy()
                else:
                    h0_raw = -np.asarray(
                        apply_initial_inverse(residual), dtype=float
                    )
                legacy_raw_directions[internal_index] = legacy_raw
                h0_raw_directions[internal_index] = h0_raw
                continue
            else:
                direction = clip_max_atomic_displacement(
                    direction, max_step
                )
                direction = project_normal(
                    direction[None, :], tangent[None, :]
                )[0]
            if (
                not np.all(np.isfinite(direction))
                or np.dot(residual, direction) >= -1e-18
            ):
                local_pairs[internal_index].clear()
                fallback = -residual
                if apply_initial_inverse is not None:
                    fallback = -np.asarray(
                        apply_initial_inverse(residual), dtype=float
                    )
                direction = clip_max_atomic_displacement(
                    fallback, max_step
                )
                direction = project_normal(
                    direction[None, :], tangent[None, :]
                )[0]
                curvature_restarts[internal_index] += 1
            directions[internal_index] = direction

        if direction_selection == "ala_dsel":
            h0_proposal = _screen_ala_dsel_band(
                "H0",
                h0_raw_directions,
                residual_gradient,
                tangents,
                max_step,
            )
            lbfgs_proposal = _screen_ala_dsel_band(
                "LBFGS",
                legacy_raw_directions,
                residual_gradient,
                tangents,
                max_step,
            )
            h0_scored = _score_ala_curve_proposal_after_geometry(
                h0_proposal,
                reference_path,
                current_path,
                displacement,
                residual_gradient,
                tangents,
            )
            lbfgs_scored = _score_ala_curve_proposal_after_geometry(
                lbfgs_proposal,
                reference_path,
                current_path,
                displacement,
                residual_gradient,
                tangents,
            )
            if not h0_scored["valid"] and not lbfgs_scored["valid"]:
                # Preserve the original legacy direction and let its existing
                # line search/monotone objective guard decide the paid trial.
                selected_label = "SAFEGUARD"
                if lbfgs_proposal["valid"]:
                    directions = lbfgs_proposal["step"].copy()
                else:
                    directions = legacy_raw_directions.copy()
                    for internal_index in range(n_internal):
                        residual = residual_gradient[internal_index]
                        direction = directions[internal_index]
                        if (
                            not np.all(np.isfinite(direction))
                            or np.dot(residual, direction) >= -1e-18
                        ):
                            local_pairs[internal_index].clear()
                            direction = h0_raw_directions[internal_index]
                            curvature_restarts[internal_index] += 1
                        directions[internal_index] = project_normal(
                            clip_max_atomic_displacement(
                                direction, max_step
                            )[None, :],
                            tangents[internal_index][None, :],
                        )[0]
            else:
                selected_proposal = _select_ala_dsel_direction(
                    h0_scored, lbfgs_scored
                )
                directions = selected_proposal["step"].copy()
                selected_label = selected_proposal["label"]
            dsel_selection.append(selected_label)
            dsel_h0_capped_decrease.append(
                h0_scored.get("post_geometry_decrease", 0.0)
            )
            dsel_lbfgs_capped_decrease.append(
                lbfgs_scored.get("post_geometry_decrease", 0.0)
            )

        if not np.any(np.linalg.norm(directions, axis=1) > 1e-18):
            break

        accepted = False
        scale = 1.0
        for _ in range(int(line_search_trial_limit)):
            proposed_displacement = project_normal(
                displacement + scale * directions, tangents
            )
            raw_trial = reference_path.copy()
            raw_trial[1:-1] += proposed_displacement
            trial_path, _ = central_spring_force_equilibrium(raw_trial)

            current_used = used_calls()
            blocked = np.asarray(
                [
                    current_used[index] >= int(max_new_calls)
                    and not evaluators[index + 1].has_cached(
                        trial_path[index + 1]
                    )
                    for index in range(n_internal)
                ],
                dtype=bool,
            )
            if np.any(blocked):
                hit_call_limit |= blocked
                break

            counters_before = counter_arrays(evaluators)[
                "joint_evaluations"
            ].copy()
            trial_energies, trial_forces = evaluate_path(
                trial_path, evaluators
            )
            counters_after = counter_arrays(evaluators)
            trial_displacement = project_normal(
                trial_path[1:-1] - reference_path[1:-1], tangents
            )
            trial_normal_gradient = project_normal(
                -trial_forces[1:-1], tangents
            )
            trial_residual = (
                trial_normal_gradient
                + trial_displacement / float(alpha)
            )
            trial_objective = (
                trial_energies[1:-1]
                + 0.5
                * np.sum(trial_displacement**2, axis=1)
                / float(alpha)
            )
            # The equal-spring gauge projection is nonlinear and collective,
            # so the unprojected Euclidean search direction is not the
            # derivative of the final projected coordinate.  Globalize with
            # monotone decrease of the *exact* proximal objective instead of
            # attaching an invalid Armijo slope to that projection.
            armijo_ok = trial_objective <= (
                objective
                + 1e-14 * np.maximum(1.0, np.abs(objective))
            )
            trial_diagnostic = path_diagnostics(
                trial_path, trial_energies, trial_forces[1:-1]
            )
            global_ok = (
                not enforce_global_iperp
                or trial_diagnostic["i_perp"]
                <= current_diagnostic["i_perp"] + 1e-12
            )
            if objective_acceptance == "path_sum":
                # The discrete IVM objective approximates one path integral;
                # its variational descent condition is therefore collective.
                # Requiring every quadrature sample to decrease separately is
                # a stronger, non-theoretical restriction that can stall when
                # central-tangent coupling transfers objective between images.
                objective_ok = bool(
                    np.sum(trial_objective)
                    <= np.sum(objective)
                    + 1e-14
                    * max(1.0, abs(float(np.sum(objective))))
                )
            else:
                objective_ok = bool(np.all(armijo_ok))
            accepted = bool(objective_ok and global_ok)

            if accepted:
                for internal_index in range(n_internal):
                    s = (
                        trial_displacement[internal_index]
                        - displacement[internal_index]
                    )
                    y = (
                        trial_residual[internal_index]
                        - residual_gradient[internal_index]
                    )
                    sy = float(np.dot(s, y))
                    if (
                        sy
                        > 1e-6
                        * np.linalg.norm(s)
                        * np.linalg.norm(y)
                    ):
                        local_pairs[internal_index].append(
                            (s.copy(), y.copy())
                        )
                        if (
                            len(local_pairs[internal_index])
                            > int(history_size)
                        ):
                            local_pairs[internal_index].pop(0)
                current_path = trial_path
                current_energies = trial_energies
                current_forces = trial_forces
                current_diagnostic = trial_diagnostic
                displacement = trial_displacement
                normal_gradient = trial_normal_gradient
                residual_gradient = trial_residual
                objective = trial_objective
                accepted_steps += 1
            else:
                backtracks += 1

            new_evaluation = not np.array_equal(
                counters_before,
                counters_after["joint_evaluations"],
            )
            if new_evaluation or accepted:
                synchronized_paths.append(current_path.copy())
                synchronized_energies.append(current_energies.copy())
                synchronized_forces.append(current_forces.copy())
                synchronized_counters.append(
                    {
                        name: values.copy()
                        for name, values in counters_after.items()
                    }
                )
            if not new_evaluation and not accepted:
                break
            if accepted:
                break
            scale *= float(backtrack_factor)
            if scale < 2.0**-40:
                break

        if not accepted:
            break

    final_kkt = np.linalg.norm(residual_gradient, axis=1)
    used = used_calls().astype(int)
    converged = final_kkt <= target_kkt
    hit_call_limit |= (
        (used >= int(max_new_calls)) & (~converged)
    )
    report = {
        "initial_kkt_l2": initial_kkt,
        "final_kkt_l2": final_kkt,
        "relative_kkt": final_kkt / np.maximum(initial_kkt, 1e-30),
        "converged": converged,
        "hit_call_limit": hit_call_limit,
        "accepted_steps": accepted_steps,
        "backtracks": backtracks,
        "curvature_restarts": curvature_restarts,
        "curve_projected": True,
        "objective_acceptance": objective_acceptance,
        "ala_dsel": direction_selection == "ala_dsel",
        "direction_selection": np.asarray(dsel_selection, dtype="U16"),
        "h0_capped_predicted_decrease": np.asarray(
            dsel_h0_capped_decrease, dtype=float
        ),
        "lbfgs_capped_predicted_decrease": np.asarray(
            dsel_lbfgs_capped_decrease, dtype=float
        ),
    }
    return (
        current_path,
        current_energies,
        current_forces,
        used,
        synchronized_paths,
        synchronized_energies,
        synchronized_forces,
        synchronized_counters,
        report,
    )


def solve_contractive_proximal_band(
    path,
    energies,
    forces,
    tangents,
    evaluators,
    alpha,
    max_step,
    max_new_calls,
    contraction_limit=0.5,
    max_alpha_halvings=16,
    kkt_rtol=0.1,
    kkt_atol=1e-8,
):
    """KKT-certified proximal solve with a contraction-guided alpha.

    For a frozen tangent, the proximal first-order condition is

        delta = alpha * F_perp(reference + delta).

    A shared alpha is halved until the fixed-point map is locally
    contractive for every internal image. Damped Picard iterations then
    solve that same condition. This deliberately avoids subtracting nearly
    equal absolute OpenMM energies near convergence; contraction plus KKT
    convergence certifies the local proximal minimizer from force data.
    """
    n_internal = path.shape[0] - 2
    candidate_path = path.copy()
    candidate_energies = energies.copy()
    candidate_forces = forces.copy()
    starting_counters = counter_arrays(evaluators)
    start_calls = starting_counters["joint_evaluations"][1:-1].copy()
    position_traces = [[path[i + 1].copy()] for i in range(n_internal)]
    energy_traces = [[float(energies[i + 1])] for i in range(n_internal)]
    force_traces = [[forces[i + 1].copy()] for i in range(n_internal)]
    counter_traces = [
        [
            {
                name: int(starting_counters[name][i + 1])
                for name in starting_counters
            }
        ]
        for i in range(n_internal)
    ]

    references = path[1:-1].copy()
    initial_energies = energies[1:-1].copy()
    initial_forces = forces[1:-1].copy()
    initial_force_perp = project_normal(initial_forces, tangents)
    initial_kkt = np.linalg.norm(initial_force_perp, axis=1)
    current_positions = references.copy()
    current_displacements = np.zeros_like(references)
    current_energies = initial_energies.copy()
    current_forces = initial_forces.copy()
    current_residuals = -float(alpha) * initial_force_perp
    current_objectives = initial_energies.copy()
    effective_alpha = float(alpha)
    alpha_halvings = 0
    probe_contraction = np.full(n_internal, np.inf, dtype=float)
    accepted_probe = False
    accepted_steps = np.zeros(n_internal, dtype=int)
    backtracks = np.zeros(n_internal, dtype=int)
    hit_call_limit = np.zeros(n_internal, dtype=bool)

    def used_calls_for(internal_index):
        image_index = internal_index + 1
        return int(
            evaluators[image_index].counters.joint_evaluations
            - start_calls[internal_index]
        )

    def append_current_trace(internal_index):
        image_index = internal_index + 1
        position_traces[internal_index].append(
            current_positions[internal_index].copy()
        )
        energy_traces[internal_index].append(
            float(current_energies[internal_index])
        )
        force_traces[internal_index].append(
            current_forces[internal_index].copy()
        )
        counter_traces[internal_index].append(
            {
                name: int(
                    getattr(evaluators[image_index].counters, name)
                )
                for name in starting_counters
            }
        )

    for alpha_attempt in range(int(max_alpha_halvings) + 1):
        probe_displacements = np.asarray(
            [
                clip_max_atomic_displacement(
                    effective_alpha * force_perp, max_step
                )
                for force_perp in initial_force_perp
            ]
        )
        probe_displacements = project_normal(
            probe_displacements, tangents
        )
        probe_positions = references + probe_displacements
        probe_energies = np.empty(n_internal, dtype=float)
        probe_forces = np.empty_like(initial_forces)
        probe_is_new = np.zeros(n_internal, dtype=bool)
        exhausted = False
        for internal_index in range(n_internal):
            image_index = internal_index + 1
            if (
                used_calls_for(internal_index) >= max_new_calls
                and not evaluators[image_index].has_cached(
                    probe_positions[internal_index]
                )
            ):
                hit_call_limit[internal_index] = True
                exhausted = True
                break
            joint_before = evaluators[
                image_index
            ].counters.joint_evaluations
            (
                probe_energies[internal_index],
                probe_forces[internal_index],
            ) = evaluators[image_index].query(
                probe_positions[internal_index], "both"
            )
            probe_is_new[internal_index] = (
                evaluators[image_index].counters.joint_evaluations
                > joint_before
            )
        if exhausted:
            for internal_index in range(n_internal):
                if probe_is_new[internal_index]:
                    append_current_trace(internal_index)
            break
        probe_force_perp = project_normal(probe_forces, tangents)
        probe_residuals = (
            probe_displacements - effective_alpha * probe_force_perp
        )
        denominator = np.maximum(
            effective_alpha * initial_kkt, 1e-30
        )
        probe_contraction = (
            np.linalg.norm(probe_residuals, axis=1) / denominator
        )
        probe_objectives = probe_energies + 0.5 * np.sum(
            probe_displacements**2, axis=1
        ) / effective_alpha
        accepted_probe = bool(
            np.all(np.isfinite(probe_contraction))
            and np.max(probe_contraction) <= float(contraction_limit)
        )
        if accepted_probe:
            current_positions = probe_positions
            current_displacements = probe_displacements
            current_energies = probe_energies
            current_forces = probe_forces
            current_residuals = probe_residuals
            current_objectives = probe_objectives
            accepted_steps[:] = 1
        for internal_index in range(n_internal):
            if probe_is_new[internal_index]:
                append_current_trace(internal_index)
        if accepted_probe:
            break
        effective_alpha *= 0.5
        alpha_halvings += 1

    if accepted_probe:
        target_kkt = np.maximum(
            float(kkt_atol),
            float(kkt_rtol) * initial_kkt,
        )
        for internal_index in range(n_internal):
            image_index = internal_index + 1
            while True:
                residual_norm = float(
                    np.linalg.norm(current_residuals[internal_index])
                    / effective_alpha
                )
                if residual_norm <= target_kkt[internal_index]:
                    break
                if used_calls_for(internal_index) >= max_new_calls:
                    hit_call_limit[internal_index] = True
                    break
                force_perp = project_normal(
                    current_forces[internal_index][None, :],
                    tangents[internal_index][None, :],
                )[0]
                fixed_point = effective_alpha * force_perp
                picard_direction = (
                    fixed_point - current_displacements[internal_index]
                )
                picard_direction = clip_max_atomic_displacement(
                    picard_direction, max_step
                )
                accepted = False
                damping = 1.0
                for _ in range(32):
                    trial_displacement = project_normal(
                        (
                            current_displacements[internal_index]
                            + damping * picard_direction
                        )[None, :],
                        tangents[internal_index][None, :],
                    )[0]
                    trial_position = (
                        references[internal_index] + trial_displacement
                    )
                    if (
                        used_calls_for(internal_index) >= max_new_calls
                        and not evaluators[image_index].has_cached(
                            trial_position
                        )
                    ):
                        hit_call_limit[internal_index] = True
                        break
                    joint_before = evaluators[
                        image_index
                    ].counters.joint_evaluations
                    trial_energy, trial_force = evaluators[
                        image_index
                    ].query(trial_position, "both")
                    trial_force_perp = project_normal(
                        trial_force[None, :],
                        tangents[internal_index][None, :],
                    )[0]
                    trial_residual = (
                        trial_displacement
                        - effective_alpha * trial_force_perp
                    )
                    trial_objective = float(
                        trial_energy
                        + 0.5
                        * np.dot(trial_displacement, trial_displacement)
                        / effective_alpha
                    )
                    residual_ok = np.linalg.norm(trial_residual) < (
                        np.linalg.norm(
                            current_residuals[internal_index]
                        )
                        - 1e-14
                    )
                    is_new = (
                        evaluators[
                            image_index
                        ].counters.joint_evaluations
                        > joint_before
                    )
                    if residual_ok:
                        current_positions[internal_index] = trial_position
                        current_displacements[
                            internal_index
                        ] = trial_displacement
                        current_energies[internal_index] = trial_energy
                        current_forces[internal_index] = trial_force
                        current_residuals[
                            internal_index
                        ] = trial_residual
                        current_objectives[
                            internal_index
                        ] = trial_objective
                        accepted_steps[internal_index] += 1
                        accepted = True
                    else:
                        backtracks[internal_index] += 1
                    if is_new:
                        append_current_trace(internal_index)
                    if accepted or hit_call_limit[internal_index]:
                        break
                    damping *= 0.5
                    if damping < 2.0**-30:
                        break
                if not accepted:
                    break

    used_calls = np.asarray(
        [used_calls_for(i) for i in range(n_internal)], dtype=int
    )
    final_force_perp = project_normal(current_forces, tangents)
    final_kkt_vectors = (
        -final_force_perp + current_displacements / effective_alpha
    )
    final_kkt = np.linalg.norm(final_kkt_vectors, axis=1)
    converged = final_kkt <= np.maximum(
        float(kkt_atol), float(kkt_rtol) * initial_kkt
    )
    candidate_path[1:-1] = current_positions
    candidate_energies[1:-1] = current_energies
    candidate_forces[1:-1] = current_forces
    for internal_index, used in enumerate(used_calls):
        position_traces[internal_index][used] = current_positions[
            internal_index
        ].copy()
        energy_traces[internal_index][used] = float(
            current_energies[internal_index]
        )
        force_traces[internal_index][used] = current_forces[
            internal_index
        ].copy()

    max_used = int(np.max(used_calls))
    synchronized_paths = []
    synchronized_energies = []
    synchronized_forces = []
    synchronized_counters = []
    final_counters = counter_arrays(evaluators)
    for local_call in range(1, max_used + 1):
        trace_path = path.copy()
        trace_energies = energies.copy()
        trace_forces = forces.copy()
        trace_counters = {
            name: starting_counters[name].copy()
            for name in starting_counters
        }
        for internal_index in range(n_internal):
            image_index = internal_index + 1
            trace_index = min(local_call, int(used_calls[internal_index]))
            trace_path[image_index] = position_traces[internal_index][
                trace_index
            ]
            trace_energies[image_index] = energy_traces[internal_index][
                trace_index
            ]
            trace_forces[image_index] = force_traces[internal_index][
                trace_index
            ]
            if trace_index > 0:
                snapshot = counter_traces[internal_index][trace_index]
                for name in trace_counters:
                    trace_counters[name][image_index] = snapshot[name]
            if local_call >= used_calls[internal_index]:
                for name in trace_counters:
                    trace_counters[name][image_index] = final_counters[name][
                        image_index
                    ]
        synchronized_paths.append(trace_path)
        synchronized_energies.append(trace_energies)
        synchronized_forces.append(trace_forces)
        synchronized_counters.append(trace_counters)
    report = {
        "initial_kkt_l2": initial_kkt,
        "final_kkt_l2": final_kkt,
        "relative_kkt": final_kkt / np.maximum(initial_kkt, 1e-30),
        "converged": converged,
        "hit_call_limit": hit_call_limit,
        "accepted_steps": accepted_steps,
        "backtracks": backtracks,
        "curvature_restarts": np.zeros(n_internal, dtype=int),
        "effective_alpha": float(effective_alpha),
        "alpha_halvings": int(alpha_halvings),
        "probe_contraction": probe_contraction,
        "probe_accepted": bool(accepted_probe),
    }
    return (
        candidate_path,
        candidate_energies,
        candidate_forces,
        used_calls,
        synchronized_paths,
        synchronized_energies,
        synchronized_forces,
        synchronized_counters,
        report,
    )


def solve_lbfgsb_band(
    path,
    energies,
    forces,
    tangents,
    evaluators,
    alpha,
    max_step,
    initial_step,
    max_new_calls,
    curvature_memory=None,
):
    from scipy.optimize import Bounds, minimize

    n_internal, dimension = path.shape[0] - 2, path.shape[1]
    candidate_path = path.copy()
    candidate_energies = energies.copy()
    candidate_forces = forces.copy()
    starting_counters = counter_arrays(evaluators)
    position_traces = [[path[i + 1].copy()] for i in range(n_internal)]
    energy_traces = [[float(energies[i + 1])] for i in range(n_internal)]
    force_traces = [[forces[i + 1].copy()] for i in range(n_internal)]
    counter_traces = [
        [
            {
                name: int(starting_counters[name][i + 1])
                for name in starting_counters
            }
        ]
        for i in range(n_internal)
    ]
    used_calls = np.zeros(n_internal, dtype=int)
    coordinate_cap = max_step / np.sqrt(3.0)
    bounds = Bounds(
        np.full(dimension, -coordinate_cap),
        np.full(dimension, coordinate_cap),
    )

    for internal_index in range(n_internal):
        image_index = internal_index + 1
        reference = path[image_index].copy()
        tangent = tangents[internal_index].copy()
        initial_energy = float(energies[image_index])
        initial_force = forces[image_index].copy()
        best = {
            "objective": initial_energy,
            "position": reference.copy(),
            "energy": initial_energy,
            "force": initial_force.copy(),
        }
        start_calls = evaluators[image_index].counters.joint_evaluations

        def objective(displacement):
            displacement = np.asarray(displacement, dtype=float)
            displacement = project_normal(
                displacement[None, :], tangent[None, :]
            )[0]
            position = reference + displacement
            used = (
                evaluators[image_index].counters.joint_evaluations
                - start_calls
            )
            if (
                used >= max_new_calls
                and not evaluators[image_index].has_current(position)
            ):
                raise EvaluationBudgetReached(
                    "Frozen-tangent L-BFGS-B subproblem call limit reached"
                )
            joint_before = (
                evaluators[image_index].counters.joint_evaluations
            )
            energy, force = evaluators[image_index].query(position, "both")
            physical_gradient = -force
            normal_gradient = project_normal(
                physical_gradient[None, :], tangent[None, :]
            )[0]
            value = float(
                energy + 0.5 * np.dot(displacement, displacement) / alpha
            )
            gradient = normal_gradient + displacement / alpha
            if value < best["objective"]:
                best.update(
                    objective=value,
                    position=position.copy(),
                    energy=float(energy),
                    force=force.copy(),
                )
            if (
                evaluators[image_index].counters.joint_evaluations
                > joint_before
            ):
                position_traces[internal_index].append(
                    best["position"].copy()
                )
                energy_traces[internal_index].append(best["energy"])
                force_traces[internal_index].append(best["force"].copy())
                counter_traces[internal_index].append(
                    {
                        name: int(
                            getattr(
                                evaluators[image_index].counters,
                                name,
                            )
                        )
                        for name in starting_counters
                    }
                )
            return value, gradient

        try:
            if initial_step > 0.0:
                initial_gradient = project_normal(
                    (-initial_force)[None, :], tangent[None, :]
                )[0]
                if curvature_memory is not None:
                    initial_direction = curvature_memory.direction(
                        internal_index,
                        initial_gradient,
                        tangent,
                        alpha,
                    )
                else:
                    initial_direction = -initial_gradient
                initial_displacement = clip_max_atomic_displacement(
                    initial_direction, initial_step
                )
                initial_displacement = project_normal(
                    initial_displacement[None, :], tangent[None, :]
                )[0]
                initial_displacement = np.clip(
                    initial_displacement, -coordinate_cap, coordinate_cap
                )
            else:
                initial_displacement = np.zeros(dimension)
            minimize(
                objective,
                initial_displacement,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={
                    "maxiter": max_new_calls,
                    "maxfun": max_new_calls,
                    "maxls": max(1, max_new_calls),
                    "maxcor": 7,
                    "ftol": 1e-12,
                    "gtol": 1e-6,
                },
            )
        except EvaluationBudgetReached:
            pass
        used = (
            evaluators[image_index].counters.joint_evaluations - start_calls
        )
        if used > max_new_calls:
            raise RuntimeError("L-BFGS-B exceeded its declared image-call budget")
        candidate_path[image_index] = best["position"]
        candidate_energies[image_index] = best["energy"]
        candidate_forces[image_index] = best["force"]
        used_calls[internal_index] = used
        # A late exact-history cache hit can change the selected best point
        # without consuming another OpenMM call.  Represent that final state
        # at the existing call label instead of inventing a new evaluation.
        position_traces[internal_index][used] = best["position"].copy()
        energy_traces[internal_index][used] = best["energy"]
        force_traces[internal_index][used] = best["force"].copy()

    max_used = int(np.max(used_calls))
    synchronized_paths = []
    synchronized_energies = []
    synchronized_forces = []
    synchronized_counters = []
    final_counters = counter_arrays(evaluators)
    for local_call in range(1, max_used + 1):
        trace_path = path.copy()
        trace_energies = energies.copy()
        trace_forces = forces.copy()
        trace_counters = {
            name: starting_counters[name].copy()
            for name in starting_counters
        }
        for internal_index in range(n_internal):
            image_index = internal_index + 1
            trace_index = min(local_call, int(used_calls[internal_index]))
            trace_path[image_index] = position_traces[internal_index][
                trace_index
            ]
            trace_energies[image_index] = energy_traces[internal_index][
                trace_index
            ]
            trace_forces[image_index] = force_traces[internal_index][
                trace_index
            ]
            if trace_index > 0:
                snapshot = counter_traces[internal_index][trace_index]
                for name in trace_counters:
                    trace_counters[name][image_index] = snapshot[name]
            if local_call >= used_calls[internal_index]:
                for name in trace_counters:
                    trace_counters[name][image_index] = final_counters[name][
                        image_index
                    ]
        synchronized_paths.append(trace_path)
        synchronized_energies.append(trace_energies)
        synchronized_forces.append(trace_forces)
        synchronized_counters.append(trace_counters)
    return (
        candidate_path,
        candidate_energies,
        candidate_forces,
        used_calls,
        synchronized_paths,
        synchronized_energies,
        synchronized_forces,
        synchronized_counters,
    )


def run(args) -> Path:
    if args.path_storage_mode not in PATH_STORAGE_MODES:
        raise ValueError("invalid path storage mode")
    if args.path_storage_mode == "committed" and args.save_all_paths:
        raise ValueError(
            "committed path storage is explicit and cannot be combined "
            "with --save-all-paths"
        )
    if args.frozen_subproblem in {"discrete_b040", "ala_dsel"}:
        if args.solver != "state_ivm":
            raise ValueError(
                "extended frozen subproblem requires solver=state_ivm"
            )
        if args.discrete_inner_max_steps < 1:
            raise ValueError(
                "discrete inner max steps must be at least one"
            )
        if args.discrete_history < 1:
            raise ValueError("discrete history must be at least one")
        if args.discrete_stop_mode not in DISCRETE_VALID_STOP_MODES:
            raise ValueError(
                "discrete stop mode must be r08c10, r08, or c10"
            )
        if (
            args.frozen_subproblem == "ala_dsel"
            and int(args.discrete_history) != int(args.history)
        ):
            raise ValueError(
                "Ala DSEL must use the legacy --history value; set "
                "--discrete-history equal to --history"
            )
        if not 0.0 < args.discrete_residual_ratio < 1.0:
            raise ValueError(
                "discrete residual ratio must lie in (0, 1)"
            )
        if args.frozen_subproblem == "ala_dsel":
            if not args.ala_dsel_trigger_iperp > 0.0:
                raise ValueError("Ala DSEL trigger I_perp must be positive")
    if args.state_cycle_iperp_window_rtol < 0.0:
        raise ValueError("state-cycle I_perp window must be nonnegative")
    if args.state_curve_direct_trial_calls < 1:
        raise ValueError(
            "state-curve direct trial calls must be at least one"
        )
    if args.state_curve_direct_failure_line_search_trials < 0:
        raise ValueError(
            "state-curve direct failure line-search trials must be "
            "nonnegative"
        )
    if args.state_curve_stall_free_polish_count < 1:
        raise ValueError(
            "state-curve stall count must be at least one"
        )
    if not (
        0.0 < args.state_curve_relock_escape_trust_factor <= 1.0
    ):
        raise ValueError(
            "state-curve relock escape trust factor must lie in (0, 1]"
        )
    if not (
        0.0
        <= args.state_free_polish_relative_defect_target
        < args.state_free_polish_relative_defect_trigger
    ):
        raise ValueError(
            "free-polish relative-defect tolerances must satisfy "
            "0 <= target < trigger"
        )
    if args.plbfgs_min_accepted_steps < 0:
        raise ValueError(
            "plbfgs minimum accepted steps must be nonnegative"
        )
    if args.reparam_spring_microsteps < 1:
        raise ValueError(
            "reparameterization spring microsteps must be at least one"
        )
    if not (
        0.0 < args.state_split_dtau_min
        <= args.reparam_spring_dtau
        <= args.state_split_dtau_max
    ):
        raise ValueError(
            "state-split pseudo-time steps must satisfy "
            "0 < minimum <= initial <= maximum"
        )
    if args.state_split_growth <= 1.0:
        raise ValueError("state-split growth factor must exceed one")
    if not 0.0 < args.state_split_shrink < 1.0:
        raise ValueError("state-split shrink factor must lie in (0, 1)")
    if not (
        0
        <= args.state_split_fast_recovery
        < args.state_split_slow_recovery
    ):
        raise ValueError(
            "state-split recovery thresholds must satisfy "
            "0 <= fast < slow"
        )
    root = Path(args.root).expanduser().resolve()
    initial_path_file = Path(args.initial_path_file).expanduser()
    if not initial_path_file.is_absolute():
        initial_path_file = root / initial_path_file
    initial_path = load_common_path(initial_path_file)
    backend = OpenMMBackend(
        root / "inputs" / "complex_fixed.pdb",
        platform_preference=("CPU",),
        precision="double",
        label="ivm_optimization",
    )
    evaluators = make_evaluators(
        backend,
        count=initial_path.shape[0],
        internal_limit=args.budget,
    )
    endpoint_cache_audit = preload_frozen_endpoint_cache(
        root / "inputs" / "endpoint_joint_ef_cache.npz",
        initial_path,
        evaluators,
    )
    ffprecon_bank = (
        install_call_free_fast_builder(
            ForceFieldPreconditionerBank(
                backend,
                root / "inputs" / "complex_fixed.pdb",
                c_stab=args.ffprecon_c_stab,
            ),
            build_workers=1,
        )
        if args.solver in {"hybrid_ff", "state_ivm"}
        else None
    )
    ffprecon_active = False
    ffprecon_retired = False
    ffprecon_exit_iterations = 0
    ffprecon_exit_globalization_active = False
    ffprecon_exit_polish_active = False
    ffprecon_raw_force_active = False
    ffprecon_late_c_stab_active = False
    ffprecon_collective_active = False
    ffprecon_history_active = False
    ffprecon_iterations = 0
    ffprecon_step_scale = np.ones(initial_path.shape[0] - 2, dtype=float)
    ffprecon_frozen_path = None
    ffprecon_frozen_tangents = None
    ffprecon_accepted_objective = None
    memory = ProjectedLBFGSMemory(
        initial_path.shape[0] - 2,
        args.history,
        initial_curvature=args.lbfgs_initial_curvature,
    )
    step_scale = np.ones(initial_path.shape[0] - 2, dtype=float)
    reparam_step_scale = np.full(
        initial_path.shape[0] - 2,
        args.reparam_initial_step_scale,
        dtype=float,
    )

    path = initial_path.copy()
    energies, forces = evaluate_path(path, evaluators)
    diagnostic = path_diagnostics(path, energies, forces[1:-1])
    initial_i_perp = float(diagnostic["i_perp"])
    cycle_best_i_perp = float(initial_i_perp)
    gradients = -forces[1:-1]
    memory.observe(path[1:-1], gradients)
    adaptive_alpha_value = float(args.alpha)
    contractive_alpha_value = float(args.alpha)
    record_alpha_value = float(args.alpha)
    last_alpha_quality = np.nan
    last_alpha_utilization = 0.0
    last_alpha_action = "initial"
    last_alpha_precision_cap = np.inf
    path_state_stale = False
    pending_reparam_residual = 0.0
    pending_geometric_guard = None
    geometric_reparam_joint_evaluations = 0
    last_state_reparam_attempt_i_perp = np.inf
    last_successful_reparam_cv_before = np.nan
    last_successful_reparam_equidistance_after = np.nan
    state_reparam_attempts = 0
    state_reparam_accepts = 0
    state_reparam_recovery_joint_evaluations = 0
    state_split_dtau = float(args.reparam_spring_dtau)
    state_strict_polish_active = False
    curve_direct_retired = False
    curve_projected_stall_count = 0
    state_curve_relock_count = 0
    state_free_polish_active = (
        args.reparam_mode == "state_relative_defect_split"
    )
    last_subproblem_kkt = np.full(
        initial_path.shape[0] - 2, np.nan, dtype=float
    )
    last_subproblem_relative_kkt = np.full_like(
        last_subproblem_kkt, np.nan
    )
    last_subproblem_unconverged = 0
    last_subproblem_hit_call_limit = 0
    last_subproblem_backtracks = 0
    last_subproblem_curvature_restarts = 0
    last_subproblem_probe_contraction = np.nan
    last_subproblem_alpha_halvings = 0
    active_plbfgs_kkt_rtol = float(args.plbfgs_kkt_rtol)
    discrete_audit_outer = []
    discrete_audit_step = []
    discrete_audit_selection = []
    discrete_audit_h0_raw_decrease = []
    discrete_audit_lbfgs_raw_decrease = []
    discrete_audit_h0_capped_decrease = []
    discrete_audit_lbfgs_capped_decrease = []
    discrete_audit_h0_cap_scale = []
    discrete_audit_lbfgs_cap_scale = []
    discrete_audit_history_count = []
    discrete_audit_curvature_status = []
    discrete_audit_global_relative_residual = []
    discrete_audit_stop_reason = []
    discrete_audit_all_states_evaluated = True
    ala_dsel_audit_outer = []
    ala_dsel_audit_selection = []
    ala_dsel_audit_h0_capped_decrease = []
    ala_dsel_audit_lbfgs_capped_decrease = []
    ala_dsel_active = False
    ala_dsel_first_activation_outer = -1
    ala_dsel_first_activation_calls_per_image = -1
    ala_dsel_first_activation_nf = -1
    ala_dsel_first_activation_iperp = np.nan
    ala_dsel_decisions_before_activation = 0

    def record_ala_dsel_report(report, outer_index):
        nonlocal ala_dsel_decisions_before_activation
        if (
            report is None
            or not bool(report.get("ala_dsel", False))
            or bool(report.get("_ala_dsel_audit_recorded", False))
        ):
            return
        selections = np.asarray(
            report["direction_selection"], dtype="U16"
        )
        h0_decrease = np.asarray(
            report["h0_capped_predicted_decrease"], dtype=float
        )
        lbfgs_decrease = np.asarray(
            report["lbfgs_capped_predicted_decrease"], dtype=float
        )
        if not (
            selections.ndim == 1
            and h0_decrease.shape == selections.shape
            and lbfgs_decrease.shape == selections.shape
        ):
            raise RuntimeError("Ala DSEL audit arrays differ in length")
        if not ala_dsel_active:
            ala_dsel_decisions_before_activation += int(selections.size)
        ala_dsel_audit_outer.extend(
            [int(outer_index)] * int(selections.size)
        )
        ala_dsel_audit_selection.extend(selections.tolist())
        ala_dsel_audit_h0_capped_decrease.extend(h0_decrease.tolist())
        ala_dsel_audit_lbfgs_capped_decrease.extend(
            lbfgs_decrease.tolist()
        )
        report["_ala_dsel_audit_recorded"] = True

    history = {
        "calls_per_image": [],
        "mean_calls_per_internal_image": [],
        "total_internal_joint_evaluations": [],
        "physical_fmax": [],
        "i_perp": [],
        "barrier_kcal": [],
        "equidistance_integral": [],
        "equidistance_relative_defect": [],
        "segment_length_cv": [],
        "segment_length_ratio": [],
        "state_free_polish_active": [],
        "accepted_fraction": [],
        "rejected_images": [],
        "active_alpha": [],
        "alpha_quality_ratio": [],
        "alpha_step_utilization": [],
        "alpha_action": [],
        "alpha_precision_cap": [],
        "active_max_step": [],
        "active_inner_calls": [],
        "active_plbfgs_kkt_rtol": [],
        "active_lbfgsb_initial_step": [],
        "active_iperp_rtol": [],
        "active_global_iperp_rtol": [],
        "active_reparam_every": [],
        "reparam_residual": [],
        "reparam_accepted_fraction": [],
        "reparam_rejected_images": [],
        "reparam_step_scale_mean": [],
        "reparam_direct_joint_evaluations": [],
        "reparam_attempted": [],
        "reparam_committed": [],
        "reparam_proposal_scale": [],
        "cycle_macro_step_scale": [],
        "reparam_cv_before": [],
        "reparam_cv_after": [],
        "reparam_equidistance_before": [],
        "reparam_equidistance_after": [],
        "reparam_recovery_joint_evaluations": [],
        "state_split_dtau": [],
        "state_strict_polish_active": [],
        "state_outer_history_active": [],
        "wall_seconds": [],
        "counter_vectors": [],
        "energy_request_vectors": [],
        "force_request_vectors": [],
        "cache_hit_vectors": [],
        "subproblem_kkt_vectors": [],
        "subproblem_relative_kkt_vectors": [],
        "subproblem_kkt_min": [],
        "subproblem_kkt_mean": [],
        "subproblem_kkt_max": [],
        "subproblem_relative_kkt_mean": [],
        "subproblem_relative_kkt_max": [],
        "subproblem_unconverged_images": [],
        "subproblem_hit_call_limit_images": [],
        "subproblem_line_search_backtracks": [],
        "subproblem_curvature_restarts": [],
        "subproblem_probe_contraction_max": [],
        "subproblem_alpha_halvings": [],
        "ffprecon_active": [],
        "ffprecon_collective_active": [],
        "ffprecon_history_active": [],
        "ffprecon_exit_globalization_active": [],
        "ffprecon_exit_polish_active": [],
        "ffprecon_rebuilds": [],
        "ffprecon_step_scale_min": [],
        "ffprecon_step_scale_mean": [],
        "ffprecon_step_scale_max": [],
    }
    path_history = []
    start_time = time.time()

    def record(
        accepted_fraction=1.0,
        rejected=0,
        reparam_residual=0.0,
        reparam_accepted_fraction=np.nan,
        reparam_rejected=0,
        reparam_direct_joint_evaluations=0,
        reparam_attempted=False,
        reparam_committed=False,
        reparam_proposal_scale=np.nan,
        cycle_macro_step_scale=np.nan,
        reparam_cv_before=np.nan,
        reparam_cv_after=np.nan,
        reparam_equidistance_before=np.nan,
        reparam_equidistance_after=np.nan,
        reparam_recovery_joint_evaluations=0,
        counter_snapshot=None,
    ):
        counters = (
            counter_arrays(evaluators)
            if counter_snapshot is None
            else counter_snapshot
        )
        internal = counters["joint_evaluations"][1:-1]
        call = int(np.max(internal))
        active_alpha = record_alpha_value
        active_step = scheduled_value(
            args.max_step, args.max_step_schedule, call
        )
        if args.solver == "state_ivm":
            trust_fraction = (
                args.state_polish_trust_fraction
                if state_strict_polish_active
                else args.state_trust_fraction
            )
            active_step = rms_segment_trust_radius(
                path,
                trust_fraction,
                args.max_step,
            )
        active_inner_calls = int(
            round(
                scheduled_value(
                    args.inner_calls, args.inner_calls_schedule, call
                )
            )
        )
        if (
            args.reparam_mode == "geometric_euler_guarded"
            and args.reparam_state_inner_calls > 0
            and (
                args.reparam_ready_fmax <= 0.0
                or diagnostic["fmax"] <= args.reparam_ready_fmax
            )
        ):
            active_inner_calls = args.reparam_state_inner_calls
        active_iperp_rtol = scheduled_value(
            args.iperp_rtol, args.iperp_rtol_schedule, call
        )
        active_lbfgsb_initial_step = scheduled_value(
            args.lbfgsb_initial_step,
            args.lbfgsb_initial_step_schedule,
            call,
        )
        active_global_iperp_rtol = scheduled_value(
            args.global_iperp_rtol,
            args.global_iperp_rtol_schedule,
            call,
        )
        active_reparam_every = int(
            round(
                scheduled_value(
                    args.reparam_every,
                    args.reparam_every_schedule,
                    call,
                )
            )
        )
        history["calls_per_image"].append(call)
        history["mean_calls_per_internal_image"].append(float(np.mean(internal)))
        history["total_internal_joint_evaluations"].append(int(np.sum(internal)))
        history["physical_fmax"].append(diagnostic["fmax"])
        history["i_perp"].append(diagnostic["i_perp"])
        history["barrier_kcal"].append(diagnostic["barrier_kcal"])
        history["equidistance_integral"].append(
            diagnostic["equidistance_integral"]
        )
        history["equidistance_relative_defect"].append(
            diagnostic["equidistance_relative_defect"]
        )
        history["segment_length_cv"].append(diagnostic["segment_length_cv"])
        history["segment_length_ratio"].append(
            diagnostic["segment_length_ratio"]
        )
        history["state_free_polish_active"].append(
            bool(state_free_polish_active)
        )
        history["accepted_fraction"].append(float(accepted_fraction))
        history["rejected_images"].append(int(rejected))
        history["active_alpha"].append(active_alpha)
        history["alpha_quality_ratio"].append(last_alpha_quality)
        history["alpha_step_utilization"].append(last_alpha_utilization)
        history["alpha_action"].append(last_alpha_action)
        history["alpha_precision_cap"].append(last_alpha_precision_cap)
        history["active_max_step"].append(active_step)
        history["active_inner_calls"].append(max(1, active_inner_calls))
        history["active_plbfgs_kkt_rtol"].append(
            float(active_plbfgs_kkt_rtol)
        )
        history["active_lbfgsb_initial_step"].append(
            active_lbfgsb_initial_step
        )
        history["active_iperp_rtol"].append(active_iperp_rtol)
        history["active_global_iperp_rtol"].append(
            active_global_iperp_rtol
        )
        history["active_reparam_every"].append(
            max(0, active_reparam_every)
        )
        history["reparam_residual"].append(float(reparam_residual))
        history["reparam_accepted_fraction"].append(
            float(reparam_accepted_fraction)
        )
        history["reparam_rejected_images"].append(int(reparam_rejected))
        history["reparam_step_scale_mean"].append(
            float(np.mean(reparam_step_scale))
        )
        history["reparam_direct_joint_evaluations"].append(
            int(reparam_direct_joint_evaluations)
        )
        history["reparam_attempted"].append(bool(reparam_attempted))
        history["reparam_committed"].append(bool(reparam_committed))
        history["reparam_proposal_scale"].append(
            float(reparam_proposal_scale)
        )
        history["cycle_macro_step_scale"].append(
            float(cycle_macro_step_scale)
        )
        history["reparam_cv_before"].append(float(reparam_cv_before))
        history["reparam_cv_after"].append(float(reparam_cv_after))
        history["reparam_equidistance_before"].append(
            float(reparam_equidistance_before)
        )
        history["reparam_equidistance_after"].append(
            float(reparam_equidistance_after)
        )
        history["reparam_recovery_joint_evaluations"].append(
            int(reparam_recovery_joint_evaluations)
        )
        history["state_split_dtau"].append(float(state_split_dtau))
        history["state_strict_polish_active"].append(
            bool(state_strict_polish_active)
        )
        history["state_outer_history_active"].append(
            state_outer_history_is_active(
                args.state_outer_history,
                args.state_outer_history_trigger_fmax,
                diagnostic["fmax"],
            )
        )
        history["wall_seconds"].append(time.time() - start_time)
        history["counter_vectors"].append(counters["joint_evaluations"].copy())
        history["energy_request_vectors"].append(
            counters["energy_requests"].copy()
        )
        history["force_request_vectors"].append(counters["force_requests"].copy())
        history["cache_hit_vectors"].append(counters["cache_hits"].copy())
        history["subproblem_kkt_vectors"].append(
            last_subproblem_kkt.copy()
        )
        history["subproblem_relative_kkt_vectors"].append(
            last_subproblem_relative_kkt.copy()
        )
        finite_kkt = last_subproblem_kkt[
            np.isfinite(last_subproblem_kkt)
        ]
        finite_relative = last_subproblem_relative_kkt[
            np.isfinite(last_subproblem_relative_kkt)
        ]
        history["subproblem_kkt_min"].append(
            float(np.min(finite_kkt)) if finite_kkt.size else np.nan
        )
        history["subproblem_kkt_mean"].append(
            float(np.mean(finite_kkt)) if finite_kkt.size else np.nan
        )
        history["subproblem_kkt_max"].append(
            float(np.max(finite_kkt)) if finite_kkt.size else np.nan
        )
        history["subproblem_relative_kkt_mean"].append(
            float(np.mean(finite_relative))
            if finite_relative.size
            else np.nan
        )
        history["subproblem_relative_kkt_max"].append(
            float(np.max(finite_relative))
            if finite_relative.size
            else np.nan
        )
        history["subproblem_unconverged_images"].append(
            int(last_subproblem_unconverged)
        )
        history["subproblem_hit_call_limit_images"].append(
            int(last_subproblem_hit_call_limit)
        )
        history["subproblem_line_search_backtracks"].append(
            int(last_subproblem_backtracks)
        )
        history["subproblem_curvature_restarts"].append(
            int(last_subproblem_curvature_restarts)
        )
        history["subproblem_probe_contraction_max"].append(
            float(last_subproblem_probe_contraction)
        )
        history["subproblem_alpha_halvings"].append(
            int(last_subproblem_alpha_halvings)
        )
        history["ffprecon_active"].append(bool(ffprecon_active))
        history["ffprecon_collective_active"].append(
            bool(ffprecon_collective_active)
        )
        history["ffprecon_history_active"].append(
            bool(ffprecon_history_active)
        )
        history["ffprecon_exit_globalization_active"].append(
            bool(ffprecon_exit_globalization_active)
        )
        history["ffprecon_exit_polish_active"].append(
            bool(ffprecon_exit_polish_active)
        )
        history["ffprecon_rebuilds"].append(
            0 if ffprecon_bank is None else int(ffprecon_bank.rebuilds)
        )
        history["ffprecon_step_scale_min"].append(
            float(np.min(ffprecon_step_scale))
        )
        history["ffprecon_step_scale_mean"].append(
            float(np.mean(ffprecon_step_scale))
        )
        history["ffprecon_step_scale_max"].append(
            float(np.max(ffprecon_step_scale))
        )
        if should_store_path(
            mode=args.path_storage_mode,
            save_all_paths=args.save_all_paths,
            call=call,
            is_initial=len(history["calls_per_image"]) == 1,
            reparam_committed=reparam_committed,
        ):
            path_history.append((call, path.copy()))
        print(
            f"[IVM central {args.solver}] calls={call:4d} "
            f"alpha={active_alpha:g} step={active_step:g} "
            f"Iperp={diagnostic['i_perp']:.6e} "
            f"sparse_barrier={diagnostic['barrier_kcal']:.4f} "
            f"fmax={diagnostic['fmax']:.5f} rejected={rejected} "
            f"spacing_CV={diagnostic['segment_length_cv']:.4f} "
            f"reparam_rejected={reparam_rejected}",
            flush=True,
        )

    record()

    def update_alpha_controller(
        reference_path,
        reference_energies,
        reference_forces,
        candidate_path,
        candidate_energies,
        frozen_tangents,
        alpha_used,
        max_step_used,
    ):
        nonlocal adaptive_alpha_value
        nonlocal last_alpha_quality, last_alpha_utilization, last_alpha_action
        if args.solver == "proximal":
            last_alpha_quality = np.nan
            last_alpha_utilization = 0.0
            last_alpha_action = "contraction"
            return
        if args.alpha_mode != "adaptive":
            last_alpha_quality = np.nan
            last_alpha_utilization = 0.0
            last_alpha_action = "fixed"
            return
        quality, utilization, valid_count = proximal_model_quality(
            reference_path,
            reference_energies,
            reference_forces,
            candidate_path,
            candidate_energies,
            frozen_tangents,
            alpha_used,
            max_step_used,
        )
        moved = np.linalg.norm(
            candidate_path[1:-1] - reference_path[1:-1], axis=1
        ) > 1e-14
        accepted_fraction = float(np.mean(moved))
        if valid_count == 0:
            accepted_fraction = 0.0
        adaptive_alpha_value, action = adapt_proximal_alpha(
            alpha_used,
            quality,
            utilization,
            accepted_fraction,
            args.alpha_min,
            args.alpha_max,
            shrink_threshold=args.alpha_shrink_threshold,
            grow_threshold=args.alpha_grow_threshold,
            grow_utilization=args.alpha_grow_utilization,
            shrink_factor=args.alpha_shrink_factor,
            grow_factor=args.alpha_grow_factor,
        )
        last_alpha_quality = quality
        last_alpha_utilization = utilization
        last_alpha_action = action

    def attempt_reparameterization(
        active_alpha, cycle_reference=None, cycle_candidates=None
    ):
        nonlocal path, energies, forces, gradients, diagnostic
        nonlocal path_state_stale, pending_reparam_residual
        nonlocal pending_geometric_guard
        nonlocal geometric_reparam_joint_evaluations
        nonlocal last_state_reparam_attempt_i_perp
        nonlocal last_successful_reparam_cv_before
        nonlocal last_successful_reparam_equidistance_after
        nonlocal state_reparam_attempts, state_reparam_accepts
        nonlocal state_reparam_recovery_joint_evaluations
        nonlocal state_split_dtau
        nonlocal state_free_polish_active
        nonlocal curve_direct_retired
        nonlocal curve_projected_stall_count
        nonlocal state_curve_relock_count
        nonlocal cycle_best_i_perp

        complete_reference_i_perp = float(diagnostic["i_perp"])
        if cycle_reference is not None:
            complete_reference_i_perp = float(
                cycle_reference[3]["i_perp"]
            )
        cycle_best_i_perp = min(
            cycle_best_i_perp, complete_reference_i_perp
        )

        def full_geometry_relock(candidate_path):
            """Apply the declared call-free mesh operator once."""

            if (
                args.state_free_polish_reparam_strategy
                == "polyline_relock"
            ):
                return piecewise_linear_equal_arclength(candidate_path), 0.0
            return central_spring_force_equilibrium(candidate_path)

        def backtrack_transactional_cycle(raw_cycle_path, guard_i_perp):
            """Globalize one fixed-alpha proximal/reparameterization cycle.

            The image-local solver has already produced ``raw_cycle_path``
            by minimizing the declared frozen-normal proximal subproblems.
            If the completed operator-split macrostate raises refreshed
            I_perp, test only scalar contractions of that *same* physical
            displacement.  Each contracted path receives the same optional
            geometry operator and a fully charged OpenMM refresh.  Thus this
            is a conventional macrostep line search: it changes neither
            alpha, the proximal objective, nor the computed direction.
            """

            if cycle_reference is None:
                return None, 0
            reference_path = np.asarray(cycle_reference[0], dtype=float)
            raw_cycle_path = np.asarray(raw_cycle_path, dtype=float)
            counter_before = int(
                np.sum(
                    counter_arrays(evaluators)["joint_evaluations"]
                )
            )
            line_scale = 1.0
            for _ in range(1, args.state_curve_direct_trial_calls):
                if int(
                    np.max(
                        counter_arrays(evaluators)["joint_evaluations"][
                            1:-1
                        ]
                    )
                ) >= int(args.budget):
                    break
                line_scale *= args.reparam_spring_backtrack
                if line_scale <= 1e-12:
                    break
                physical_trial = reference_path + line_scale * (
                    raw_cycle_path - reference_path
                )
                physical_spacing = equidistance_diagnostics(
                    physical_trial
                )
                trial_path = physical_trial
                trial_residual = 0.0
                geometry_scale = 0.0
                if args.state_free_polish_reparam_strategy == "spring_relock":
                    trial_path, trial_residual = full_geometry_relock(
                        physical_trial
                    )
                    geometry_scale = 1.0
                elif (
                    args.state_free_polish_reparam_strategy
                    == "polyline_relock"
                    and physical_spacing["segment_length_cv"]
                    > args.state_free_polish_target_cv
                ):
                    trial_path, trial_residual = full_geometry_relock(
                        physical_trial
                    )
                    geometry_scale = 1.0
                elif (
                    args.state_free_polish_reparam_strategy == "cv_relock"
                    and physical_spacing["segment_length_cv"]
                    > args.state_free_polish_target_cv
                ):
                    equal_path, trial_residual = (
                        central_spring_force_equilibrium(physical_trial)
                    )
                    full_correction = equal_path - physical_trial
                    lower = 0.0
                    upper = 1.0
                    for _ in range(60):
                        midpoint = 0.5 * (lower + upper)
                        midpoint_path = (
                            physical_trial
                            + midpoint * full_correction
                        )
                        midpoint_cv = equidistance_diagnostics(
                            midpoint_path
                        )["segment_length_cv"]
                        if (
                            midpoint_cv
                            <= args.state_free_polish_target_cv
                        ):
                            upper = midpoint
                        else:
                            lower = midpoint
                    geometry_scale = upper
                    trial_path = (
                        physical_trial
                        + geometry_scale * full_correction
                    )
                trial_spacing = equidistance_diagnostics(trial_path)
                trial_energies, trial_forces = evaluate_path(
                    trial_path, evaluators
                )
                trial_diagnostic = path_diagnostics(
                    trial_path,
                    trial_energies,
                    trial_forces[1:-1],
                )
                mesh_ok = bool(
                    (
                        args.state_free_polish_reparam_strategy
                        == "spring_relock"
                        and trial_spacing["segment_length_cv"] <= 1e-8
                    )
                    or (
                        args.state_free_polish_reparam_strategy
                        in ("cv_relock", "polyline_relock")
                        and trial_spacing["segment_length_cv"]
                        <= args.state_free_polish_target_cv + 1e-12
                    )
                )
                if (
                    mesh_ok
                    and trial_diagnostic["i_perp"]
                    <= guard_i_perp + 1e-12
                ):
                    direct_calls = int(
                        np.sum(
                            counter_arrays(evaluators)[
                                "joint_evaluations"
                            ]
                        )
                    ) - counter_before
                    return (
                        trial_path,
                        trial_energies,
                        trial_forces,
                        trial_diagnostic,
                        trial_spacing,
                        trial_residual,
                        geometry_scale,
                        line_scale,
                    ), direct_calls
            direct_calls = int(
                np.sum(
                    counter_arrays(evaluators)["joint_evaluations"]
                )
            ) - counter_before
            return None, direct_calls

        def search_earlier_transactional_cycles(guard_i_perp):
            """Test complete relocked macrostates from the proximal trace.

            A synchronized frozen-normal solve naturally produces a sequence
            of legitimate fixed-alpha proximal iterates.  The former
            transactional implementation kept only the last one, even when
            an earlier iterate plus the declared geometry operator was a
            better complete path.  Search those already-computed physical
            candidates in reverse order and refresh every relocked path with
            OpenMM before acceptance.  This selects a stopping point of the
            same proximal solve; it changes neither alpha nor its objective.
            """

            if cycle_reference is None or not cycle_candidates:
                return None, 0
            reference_path = np.asarray(cycle_reference[0], dtype=float)
            counter_before = int(
                np.sum(counter_arrays(evaluators)["joint_evaluations"])
            )
            # The final raw candidate has already received its full geometry
            # trial above.  Search only the earlier synchronized iterates.
            for raw_candidate in reversed(cycle_candidates[:-1]):
                if int(
                    np.max(
                        counter_arrays(evaluators)["joint_evaluations"][
                            1:-1
                        ]
                    )
                ) >= int(args.budget):
                    break
                physical_trial = np.asarray(raw_candidate, dtype=float)
                if np.array_equal(physical_trial, reference_path):
                    continue
                physical_spacing = equidistance_diagnostics(physical_trial)
                trial_path = physical_trial
                trial_residual = 0.0
                geometry_scale = 0.0
                if args.state_free_polish_reparam_strategy == "spring_relock":
                    trial_path, trial_residual = full_geometry_relock(
                        physical_trial
                    )
                    geometry_scale = 1.0
                elif (
                    args.state_free_polish_reparam_strategy
                    == "polyline_relock"
                    and physical_spacing["segment_length_cv"]
                    > args.state_free_polish_target_cv
                ):
                    trial_path, trial_residual = full_geometry_relock(
                        physical_trial
                    )
                    geometry_scale = 1.0
                elif (
                    args.state_free_polish_reparam_strategy == "cv_relock"
                    and physical_spacing["segment_length_cv"]
                    > args.state_free_polish_target_cv
                ):
                    equal_path, trial_residual = (
                        central_spring_force_equilibrium(physical_trial)
                    )
                    full_correction = equal_path - physical_trial
                    lower = 0.0
                    upper = 1.0
                    for _ in range(60):
                        midpoint = 0.5 * (lower + upper)
                        midpoint_path = (
                            physical_trial + midpoint * full_correction
                        )
                        midpoint_cv = equidistance_diagnostics(midpoint_path)[
                            "segment_length_cv"
                        ]
                        if midpoint_cv <= args.state_free_polish_target_cv:
                            upper = midpoint
                        else:
                            lower = midpoint
                    geometry_scale = upper
                    trial_path = (
                        physical_trial + geometry_scale * full_correction
                    )
                trial_spacing = equidistance_diagnostics(trial_path)
                trial_energies, trial_forces = evaluate_path(
                    trial_path, evaluators
                )
                trial_diagnostic = path_diagnostics(
                    trial_path,
                    trial_energies,
                    trial_forces[1:-1],
                )
                mesh_ok = bool(
                    (
                        args.state_free_polish_reparam_strategy
                        == "spring_relock"
                        and trial_spacing["segment_length_cv"] <= 1e-8
                    )
                    or (
                        args.state_free_polish_reparam_strategy
                        in ("cv_relock", "polyline_relock")
                        and trial_spacing["segment_length_cv"]
                        <= args.state_free_polish_target_cv + 1e-12
                    )
                )
                if (
                    mesh_ok
                    and trial_diagnostic["i_perp"]
                    <= guard_i_perp + 1e-12
                ):
                    direct_calls = int(
                        np.sum(
                            counter_arrays(evaluators)["joint_evaluations"]
                        )
                    ) - counter_before
                    return (
                        trial_path,
                        trial_energies,
                        trial_forces,
                        trial_diagnostic,
                        trial_spacing,
                        trial_residual,
                        geometry_scale,
                        1.0,
                    ), direct_calls
            direct_calls = int(
                np.sum(counter_arrays(evaluators)["joint_evaluations"])
            ) - counter_before
            return None, direct_calls

        if state_free_polish_active:
            transactional_spring_relock = bool(
                args.state_free_polish_reparam_strategy
                in ("spring_relock", "cv_relock", "polyline_relock")
                and cycle_reference is not None
            )
            guard_i_perp = float(diagnostic["i_perp"])
            if transactional_spring_relock:
                guard_i_perp = cycle_best_i_perp * (
                    1.0 + args.state_cycle_iperp_window_rtol
                )
            before = equidistance_diagnostics(path)
            if args.state_free_polish_reparam_strategy == "spring_relock":
                correction_due = (
                    before["segment_length_cv"] > 1e-12
                )
            elif args.state_free_polish_reparam_strategy == "cv_relock":
                correction_due = (
                    before["segment_length_cv"]
                    > args.state_free_polish_target_cv
                )
            elif (
                args.state_free_polish_reparam_strategy
                == "polyline_relock"
            ):
                correction_due = (
                    before["segment_length_cv"]
                    > args.state_free_polish_target_cv
                )
            elif args.state_free_polish_reparam_strategy == "euler":
                correction_due = (
                    before["equidistance_integral"] > 1e-30
                )
            else:
                correction_due = (
                    before["equidistance_relative_defect"]
                    > args.state_free_polish_relative_defect_trigger
                )
            if not correction_due:
                if transactional_spring_relock:
                    # The physical proximal half-step already satisfies the
                    # declared mesh bound, so it is itself a complete
                    # publishable macrostate.  Do not erase that normal-space
                    # progress by projecting back to machine-exact equal
                    # spacing; record the real force ledger at the accepted
                    # path and leave the optional geometry operator idle.
                    reference_path = np.asarray(
                        cycle_reference[0], dtype=float
                    )
                    moved = np.linalg.norm(
                        path[1:-1] - reference_path[1:-1], axis=1
                    ) > 1e-14
                    if (
                        args.state_globalization
                        in (
                            "iperp_monotone",
                            "delayed_iperp_monotone",
                            "cycle_iperp_monotone",
                        )
                        and diagnostic["i_perp"]
                        > guard_i_perp + 1e-12
                    ):
                        raw_cycle_path = path.copy()
                        macro_trial, direct_calls = (
                            backtrack_transactional_cycle(
                                raw_cycle_path, guard_i_perp
                            )
                        )
                        if macro_trial is None:
                            (
                                macro_trial,
                                earlier_direct_calls,
                            ) = search_earlier_transactional_cycles(
                                guard_i_perp
                            )
                            direct_calls += earlier_direct_calls
                        if macro_trial is not None:
                            (
                                path,
                                energies,
                                forces,
                                diagnostic,
                                accepted_spacing,
                                trial_residual,
                                geometry_scale,
                                macro_scale,
                            ) = macro_trial
                            gradients = -forces[1:-1]
                            memory.reset()
                            memory.observe(path[1:-1], gradients)
                            cycle_best_i_perp = min(
                                cycle_best_i_perp,
                                float(diagnostic["i_perp"]),
                            )
                            moved = np.linalg.norm(
                                path[1:-1] - reference_path[1:-1],
                                axis=1,
                            ) > 1e-14
                            record(
                                accepted_fraction=float(np.mean(moved)),
                                rejected=int(np.count_nonzero(~moved)),
                                reparam_residual=trial_residual,
                                reparam_accepted_fraction=1.0,
                                reparam_rejected=0,
                                reparam_direct_joint_evaluations=(
                                    direct_calls
                                ),
                                reparam_attempted=True,
                                reparam_committed=True,
                                reparam_proposal_scale=geometry_scale,
                                cycle_macro_step_scale=macro_scale,
                                reparam_cv_before=before[
                                    "segment_length_cv"
                                ],
                                reparam_cv_after=accepted_spacing[
                                    "segment_length_cv"
                                ],
                                reparam_equidistance_before=before[
                                    "equidistance_integral"
                                ],
                                reparam_equidistance_after=(
                                    accepted_spacing[
                                        "equidistance_integral"
                                    ]
                                ),
                            )
                            return (
                                trial_residual,
                                moved,
                                True,
                                direct_calls,
                            )
                        path = reference_path.copy()
                        energies = np.asarray(
                            cycle_reference[1], dtype=float
                        ).copy()
                        forces = np.asarray(
                            cycle_reference[2], dtype=float
                        ).copy()
                        diagnostic = cycle_reference[3]
                        gradients = -forces[1:-1]
                        memory.reset()
                        memory.observe(path[1:-1], gradients)
                        record(
                            accepted_fraction=0.0,
                            rejected=path.shape[0] - 2,
                            reparam_direct_joint_evaluations=direct_calls,
                            reparam_attempted=True,
                            reparam_committed=False,
                            cycle_macro_step_scale=0.0,
                        )
                        return (
                            0.0,
                            np.zeros(path.shape[0] - 2, dtype=bool),
                            False,
                            direct_calls,
                        )
                    record(
                        accepted_fraction=float(np.mean(moved)),
                        rejected=int(np.count_nonzero(~moved)),
                        reparam_attempted=False,
                        reparam_committed=False,
                        cycle_macro_step_scale=1.0,
                    )
                    cycle_best_i_perp = min(
                        cycle_best_i_perp,
                        float(diagnostic["i_perp"]),
                    )
                return (
                    0.0,
                    np.zeros(path.shape[0] - 2, dtype=bool),
                    False,
                    0,
                )
            state_reparam_attempts += 1
            if (
                args.state_free_polish_reparam_strategy
                in ("spring_relock", "cv_relock", "polyline_relock")
            ):
                working_path, residual = full_geometry_relock(path)
                proposed_spacing = equidistance_diagnostics(
                    working_path
                )
                proposal_scale = 1.0
                if (
                    args.state_free_polish_reparam_strategy
                    == "cv_relock"
                    and before["segment_length_cv"]
                    > args.state_free_polish_target_cv
                ):
                    # Use the smallest geometry-only correction that restores
                    # the declared production mesh bound.  Bisection calls no
                    # potential/force oracle; the selected path is refreshed
                    # and globally tested exactly once below.
                    full_correction = working_path - path
                    lower = 0.0
                    upper = 1.0
                    for _ in range(60):
                        midpoint = 0.5 * (lower + upper)
                        midpoint_path = path + midpoint * full_correction
                        midpoint_cv = equidistance_diagnostics(
                            midpoint_path
                        )["segment_length_cv"]
                        if midpoint_cv <= args.state_free_polish_target_cv:
                            upper = midpoint
                        else:
                            lower = midpoint
                    proposal_scale = upper
                    working_path = path + proposal_scale * full_correction
                    proposed_spacing = equidistance_diagnostics(
                        working_path
                    )
            elif args.state_free_polish_reparam_strategy == "euler":
                working_path = path.copy()
                proposed_spacing = before
                residual = 0.0
                proposal_scale = 0.0
                for _ in range(args.reparam_spring_microsteps):
                    (
                        micro_path,
                        micro_residual,
                        micro_applied,
                        _,
                        micro_spacing,
                        micro_dtau,
                    ) = euler_spring_reparameterization(
                        working_path,
                        np.ones(
                            path.shape[0] - 2, dtype=float
                        ),
                        dtau=args.reparam_spring_dtau,
                        backtrack_factor=(
                            args.reparam_spring_backtrack
                        ),
                    )
                    residual = max(residual, micro_residual)
                    if not micro_applied:
                        break
                    working_path = micro_path
                    proposed_spacing = micro_spacing
                    proposal_scale += micro_dtau
            else:
                (
                    working_path,
                    residual,
                    proposal_scale,
                    before,
                    proposed_spacing,
                ) = spring_proposal_to_target_relative_defect(
                    path,
                    args.state_free_polish_relative_defect_target,
                    dtau=args.reparam_spring_dtau,
                )
            if proposal_scale <= 0.0 or np.array_equal(
                working_path, path
            ):
                return (
                    residual,
                    np.zeros(path.shape[0] - 2, dtype=bool),
                    False,
                    0,
                )
            counter_before = int(
                np.sum(
                    counter_arrays(evaluators)["joint_evaluations"]
                )
            )
            cycle_macro_scale = 1.0
            working_energies, working_forces = evaluate_path(
                working_path, evaluators
            )
            direct_calls = int(
                np.sum(
                    counter_arrays(evaluators)["joint_evaluations"]
                )
            ) - counter_before
            working_diagnostic = path_diagnostics(
                working_path,
                working_energies,
                working_forces[1:-1],
            )
            if (
                args.state_globalization
                in (
                    "iperp_monotone",
                    "delayed_iperp_monotone",
                    "cycle_iperp_monotone",
                )
                and working_diagnostic["i_perp"]
                > guard_i_perp + 1e-12
                and not transactional_spring_relock
            ):
                # Backtrack only the optional geometry correction. The
                # fixed-alpha proximal subproblem is unchanged. Every
                # refreshed trial remains charged to the optimization run.
                full_correction = working_path - path
                original_proposal_scale = proposal_scale
                line_scale = 1.0
                trials = 1
                while (
                    trials < args.state_curve_direct_trial_calls
                    and int(
                        np.max(
                            counter_arrays(evaluators)[
                                "joint_evaluations"
                            ][1:-1]
                        )
                    ) < args.budget
                ):
                    line_scale *= args.reparam_spring_backtrack
                    if line_scale <= 1e-12:
                        break
                    trial_path = path + line_scale * full_correction
                    trial_energies, trial_forces = evaluate_path(
                        trial_path, evaluators
                    )
                    trials += 1
                    direct_calls = int(
                        np.sum(
                            counter_arrays(evaluators)[
                                "joint_evaluations"
                            ]
                        )
                    ) - counter_before
                    trial_diagnostic = path_diagnostics(
                        trial_path,
                        trial_energies,
                        trial_forces[1:-1],
                    )
                    trial_spacing = equidistance_diagnostics(trial_path)
                    if (
                        trial_diagnostic["i_perp"]
                        <= guard_i_perp + 1e-12
                        and trial_spacing[
                            "equidistance_relative_defect"
                        ]
                        < before["equidistance_relative_defect"]
                    ):
                        working_path = trial_path
                        working_energies = trial_energies
                        working_forces = trial_forces
                        working_diagnostic = trial_diagnostic
                        proposed_spacing = trial_spacing
                        proposal_scale = (
                            original_proposal_scale * line_scale
                        )
                        break
            if (
                transactional_spring_relock
                and args.state_globalization
                == "cycle_iperp_monotone"
                and working_diagnostic["i_perp"]
                > guard_i_perp + 1e-12
            ):
                # The lambda=1 operator-split state failed the refreshed
                # path merit.  Contract the already-computed proximal
                # displacement and reapply the same mesh operator.  No new
                # direction, alpha, or proximal subproblem is introduced.
                macro_trial, _ = backtrack_transactional_cycle(
                    path.copy(), guard_i_perp
                )
                if macro_trial is None:
                    macro_trial, _ = search_earlier_transactional_cycles(
                        guard_i_perp
                    )
                direct_calls = int(
                    np.sum(
                        counter_arrays(evaluators)[
                            "joint_evaluations"
                        ]
                    )
                ) - counter_before
                if macro_trial is not None:
                    (
                        working_path,
                        working_energies,
                        working_forces,
                        working_diagnostic,
                        proposed_spacing,
                        residual,
                        proposal_scale,
                        cycle_macro_scale,
                    ) = macro_trial
            if (
                args.state_globalization
                in (
                    "iperp_monotone",
                    "delayed_iperp_monotone",
                    "cycle_iperp_monotone",
                )
                and working_diagnostic["i_perp"]
                > guard_i_perp + 1e-12
            ):
                # The geometry-only proposal is optional. A strict global
                # I_perp policy must reject a spacing correction that raises
                # the refreshed physical merit. Its OpenMM query remains
                # fully charged and visible in the audit.
                if transactional_spring_relock:
                    # Record the publishable macrostate, not the rejected
                    # unconstrained half-step.  The proposal diagnostics
                    # below still expose the exact relock trial, and all of
                    # its OpenMM evaluations remain charged.
                    path = np.asarray(cycle_reference[0], dtype=float).copy()
                    energies = np.asarray(
                        cycle_reference[1], dtype=float
                    ).copy()
                    forces = np.asarray(
                        cycle_reference[2], dtype=float
                    ).copy()
                    diagnostic = cycle_reference[3]
                    gradients = -forces[1:-1]
                    memory.reset()
                    memory.observe(path[1:-1], gradients)
                    state_free_polish_active = bool(
                        args.state_free_polish_reparam_strategy
                        in ("cv_relock", "polyline_relock")
                    )
                    curve_direct_retired = state_free_polish_active
                    curve_projected_stall_count = 0
                record(
                    accepted_fraction=0.0,
                    rejected=path.shape[0] - 2,
                    reparam_residual=residual,
                    reparam_accepted_fraction=0.0,
                    reparam_rejected=path.shape[0] - 2,
                    reparam_direct_joint_evaluations=direct_calls,
                    reparam_attempted=True,
                    reparam_committed=False,
                    reparam_proposal_scale=proposal_scale,
                    cycle_macro_step_scale=0.0,
                    reparam_cv_before=before["segment_length_cv"],
                    reparam_cv_after=proposed_spacing[
                        "segment_length_cv"
                    ],
                    reparam_equidistance_before=before[
                        "equidistance_integral"
                    ],
                    reparam_equidistance_after=proposed_spacing[
                        "equidistance_integral"
                    ],
                )
                return (
                    residual,
                    np.zeros(path.shape[0] - 2, dtype=bool),
                    False,
                    direct_calls,
                )
            # This is an optional parameterization step. Its geometry is
            # selected without OpenMM; the refreshed joint evaluation is
            # fully charged to the run before a permitted commit.
            path = working_path
            energies = working_energies
            forces = working_forces
            diagnostic = working_diagnostic
            gradients = -forces[1:-1]
            memory.reset()
            memory.observe(path[1:-1], gradients)
            state_reparam_accepts += 1
            cycle_best_i_perp = min(
                cycle_best_i_perp, float(diagnostic["i_perp"])
            )
            last_successful_reparam_cv_before = float(
                before["segment_length_cv"]
            )
            last_successful_reparam_equidistance_after = float(
                diagnostic["equidistance_integral"]
            )
            committed = np.ones(
                path.shape[0] - 2, dtype=bool
            )
            record(
                accepted_fraction=1.0,
                rejected=0,
                reparam_residual=residual,
                reparam_accepted_fraction=1.0,
                reparam_rejected=0,
                reparam_direct_joint_evaluations=direct_calls,
                reparam_attempted=True,
                reparam_committed=True,
                reparam_proposal_scale=proposal_scale,
                cycle_macro_step_scale=cycle_macro_scale,
                reparam_cv_before=before["segment_length_cv"],
                reparam_cv_after=proposed_spacing[
                    "segment_length_cv"
                ],
                reparam_equidistance_before=before[
                    "equidistance_integral"
                ],
                reparam_equidistance_after=proposed_spacing[
                    "equidistance_integral"
                ],
            )
            if args.state_free_polish_reparam_strategy == "spring_relock":
                # One unconstrained frozen-normal IVM macrostep has escaped
                # the discrete equal-spring fixed point.  Relock exactly and
                # reactivate the same synchronized equal-spring predictor.
                # This is a state-triggered cycle, not an alpha/call schedule.
                state_free_polish_active = False
                curve_direct_retired = False
                curve_projected_stall_count = 0
                state_curve_relock_count += 1
            elif args.state_free_polish_reparam_strategy in (
                "cv_relock",
                "polyline_relock",
            ):
                # A bounded-CV path is intentionally not the machine-exact
                # equal-spring fixed point. Continue the same operator split
                # so every subsequent physical step is reparameterized back
                # to the production mesh bound before it can be published.
                state_free_polish_active = True
                curve_direct_retired = True
                curve_projected_stall_count = 0
                state_curve_relock_count += 1
            return residual, committed, True, direct_calls
        if args.reparam_mode in {
            "state_delayed",
            "state_metric",
            "state_split",
            "state_curve",
            "state_polyline",
            "state_curve_projected",
        }:
            split_mode = args.reparam_mode == "state_split"
            curve_mode = args.reparam_mode in {
                "state_curve",
                "state_polyline",
                "state_curve_projected",
            }
            projected_curve_mode = (
                args.reparam_mode == "state_curve_projected"
            )
            metric_mode = args.reparam_mode in {
                "state_metric",
                "state_split",
                "state_curve",
                "state_polyline",
                "state_curve_projected",
            }
            cv_trigger = (
                args.reparam_strict_cv_trigger
                if state_strict_polish_active
                else args.reparam_cv_trigger
            )
            if (
                not metric_mode
                and np.isfinite(last_successful_reparam_cv_before)
            ):
                # Hysteresis: after a successful spacing correction, do not
                # re-arm until spacing has deteriorated back to at least the
                # level that justified the previous correction.  This is a
                # state condition, not a call/budget schedule.
                cv_trigger = max(
                    cv_trigger,
                    last_successful_reparam_cv_before,
                )
            due = state_reparameterization_is_due(
                diagnostic,
                initial_i_perp,
                last_state_reparam_attempt_i_perp,
                args.reparam_ready_relative_iperp,
                cv_trigger,
                args.reparam_retry_reduction,
            )
            if (
                metric_mode
                and not (split_mode or curve_mode)
                and np.isfinite(
                    last_successful_reparam_equidistance_after
                )
                and diagnostic["equidistance_integral"]
                < args.reparam_equidistance_rearm_growth
                * last_successful_reparam_equidistance_after
            ):
                # Re-arm only after the requested raw integral has genuinely
                # drifted away from the last committed parameterization.
                due = False
            if (
                args.reparam_ready_fmax > 0.0
                and diagnostic["fmax"] > args.reparam_ready_fmax
            ):
                due = False
            if not due:
                return (
                    0.0,
                    np.zeros(path.shape[0] - 2, dtype=bool),
                    False,
                    0,
                )
            state_reparam_attempts += 1
            last_state_reparam_attempt_i_perp = float(
                diagnostic["i_perp"]
            )
            accepted_path = path.copy()
            accepted_energies = energies.copy()
            accepted_forces = forces.copy()
            accepted_diagnostic = diagnostic
            attempt_start = counter_arrays(evaluators)[
                "joint_evaluations"
            ].copy()
            if curve_mode:
                if args.reparam_mode == "state_polyline":
                    working_path = piecewise_linear_equal_arclength(path)
                    residual = 0.0
                else:
                    working_path, residual = central_spring_force_equilibrium(
                        path
                    )
                proposal_scale = 1.0
                before_spacing = equidistance_diagnostics(path)
            elif split_mode:
                (
                    working_path,
                    residual,
                    applied,
                    before_spacing,
                    _,
                    applied_dtau,
                ) = euler_spring_reparameterization(
                    path,
                    np.ones(path.shape[0] - 2, dtype=float),
                    dtau=state_split_dtau,
                    backtrack_factor=args.reparam_spring_backtrack,
                )
                proposal_scale = float(applied_dtau)
                if not applied:
                    proposal_scale = 0.0
            elif (
                metric_mode
                and args.reparam_proposal_metric == "equidistance"
            ):
                (
                    working_path,
                    residual,
                    proposal_scale,
                    before_spacing,
                    _,
                ) = spring_proposal_to_target_equidistance(
                    path,
                    contraction=args.reparam_equidistance_contraction,
                )
            else:
                (
                    working_path,
                    residual,
                    proposal_scale,
                    before_spacing,
                    _,
                ) = spring_proposal_to_target_cv(
                    path,
                    args.reparam_target_cv,
                    contraction=args.reparam_cv_contraction,
                )
            if proposal_scale <= 0.0 or np.array_equal(
                working_path, path
            ):
                return (
                    residual,
                    np.zeros(path.shape[0] - 2, dtype=bool),
                    False,
                    0,
                )
            counter_before = int(
                np.sum(counter_arrays(evaluators)["joint_evaluations"])
            )
            working_energies, working_forces = evaluate_path(
                working_path, evaluators
            )
            counter_after = int(
                np.sum(counter_arrays(evaluators)["joint_evaluations"])
            )
            direct_calls = counter_after - counter_before
            working_diagnostic = path_diagnostics(
                working_path,
                working_energies,
                working_forces[1:-1],
            )
            if curve_mode:
                # The frozen-curve proposal is the exact zero-residual state
                # of the discrete identical-spring problem.  Its subsequent
                # normal IVM recovery must therefore be a genuine
                # continuation step toward the spring manifold, rather than
                # being allowed to spend almost all of the newly created
                # spacing slack.  The contraction is state based and
                # independent of force-call count.
                recovery_equidistance_ceiling = max(
                    float(working_diagnostic["equidistance_integral"]),
                    float(accepted_diagnostic["equidistance_integral"])
                    * float(args.reparam_equidistance_contraction),
                )
            else:
                recovery_equidistance_ceiling = (
                    float(accepted_diagnostic["equidistance_integral"])
                    * (
                        1.0
                        - float(
                            args.reparam_min_equidistance_improvement
                        )
                    )
                )
            recovery_memory = ProjectedLBFGSMemory(
                path.shape[0] - 2,
                args.history,
                initial_curvature=args.lbfgs_initial_curvature,
            )
            recovery_memory.observe(
                working_path[1:-1], -working_forces[1:-1]
            )
            record(
                accepted_fraction=0.0,
                rejected=0,
                reparam_residual=residual,
                reparam_accepted_fraction=0.0,
                reparam_rejected=path.shape[0] - 2,
                reparam_direct_joint_evaluations=direct_calls,
                reparam_attempted=True,
                reparam_committed=False,
                reparam_proposal_scale=proposal_scale,
                reparam_cv_before=before_spacing["segment_length_cv"],
                reparam_cv_after=working_diagnostic[
                    "segment_length_cv"
                ],
                reparam_equidistance_before=before_spacing[
                    "equidistance_integral"
                ],
                reparam_equidistance_after=working_diagnostic[
                    "equidistance_integral"
                ],
            )

            def recovered():
                return (
                    working_diagnostic["i_perp"]
                    <= accepted_diagnostic["i_perp"]
                    * (1.0 + args.reparam_recovery_iperp_rtol)
                    + 1e-12
                    and (
                        (
                            metric_mode
                            and working_diagnostic[
                                "equidistance_integral"
                            ]
                            <= recovery_equidistance_ceiling + 1e-14
                        )
                        or (
                            not metric_mode
                            and working_diagnostic[
                                "segment_length_cv"
                            ]
                            <= accepted_diagnostic[
                                "segment_length_cv"
                            ]
                            * (1.0 - args.reparam_min_cv_improvement)
                            + 1e-14
                        )
                    )
                )

            while not recovered():
                current_counters = counter_arrays(evaluators)[
                    "joint_evaluations"
                ]
                used_per_image = (
                    current_counters[1:-1] - attempt_start[1:-1]
                )
                recovery_remaining = int(
                    args.reparam_recovery_calls
                    - np.max(used_per_image)
                )
                budget_remaining = int(
                    args.budget - np.max(current_counters[1:-1])
                )
                allowed = min(
                    args.reparam_recovery_inner_calls,
                    recovery_remaining,
                    (
                        budget_remaining - 1
                        if projected_curve_mode
                        else budget_remaining
                    ),
                )
                if allowed <= 0:
                    break
                working_tangents = central_tangents(working_path)
                recovery_step = rms_segment_trust_radius(
                    working_path,
                    (
                        args.state_polish_trust_fraction
                        if state_strict_polish_active
                        else args.state_trust_fraction
                    ),
                    args.max_step,
                )
                if ffprecon_bank is not None:
                    raw_before = int(backend.raw_joint_attempts)
                    ffprecon_bank.build(working_path)
                    if int(backend.raw_joint_attempts) != raw_before:
                        raise RuntimeError(
                            "FF preconditioner construction called OpenMM"
                        )
                (
                    candidate_path,
                    candidate_energies,
                    candidate_forces,
                    used_calls,
                    synchronized_paths,
                    synchronized_energies,
                    synchronized_forces,
                    synchronized_counters,
                    _,
                ) = solve_projected_proximal_band(
                    working_path,
                    working_energies,
                    working_forces,
                    working_tangents,
                    evaluators,
                    active_alpha,
                    recovery_step,
                    allowed,
                    history_size=args.history,
                    curvature_memory=(
                        recovery_memory
                        if state_outer_history_is_active(
                            args.state_outer_history,
                            args.state_outer_history_trigger_fmax,
                            working_diagnostic["fmax"],
                        )
                        else None
                    ),
                    armijo=args.plbfgs_armijo,
                    backtrack_factor=args.plbfgs_backtrack,
                    kkt_rtol=active_plbfgs_kkt_rtol,
                    kkt_atol=args.plbfgs_kkt_atol,
                    min_accepted_steps=args.plbfgs_min_accepted_steps,
                    preconditioner_bank=ffprecon_bank,
                )
                if not np.any(used_calls):
                    break
                if projected_curve_mode:
                    # Riemannian/operator-splitting recovery: the evaluated
                    # frozen-normal IVM trial is projected back onto the
                    # identical-spring manifold by a geometry-only frozen
                    # central-tangent spring solve. OpenMM is queried once at
                    # that final projected coordinate.  The already evaluated
                    # unprojected trial remains in the force-call audit.
                    projected_path, _ = (
                        central_spring_force_equilibrium(candidate_path)
                    )
                    (
                        projected_energies,
                        projected_forces,
                    ) = evaluate_path(projected_path, evaluators)
                    candidate_path = projected_path
                    candidate_energies = projected_energies
                    candidate_forces = projected_forces
                    synchronized_paths = [projected_path.copy()]
                    synchronized_energies = [
                        projected_energies.copy()
                    ]
                    synchronized_forces = [projected_forces.copy()]
                    synchronized_counters.append(
                        {
                            name: values.copy()
                            for name, values in counter_arrays(
                                evaluators
                            ).items()
                        }
                    )
                # Trial points are genuine force calls but not accepted global
                # paths.  Keep the pre-reparameterization metrics visible
                # until a recovered macrostate is ready to commit.
                for trace_counters in synchronized_counters:
                    record(
                        accepted_fraction=0.0,
                        rejected=0,
                        reparam_residual=residual,
                        reparam_accepted_fraction=0.0,
                        reparam_rejected=path.shape[0] - 2,
                        reparam_attempted=True,
                        reparam_committed=False,
                        reparam_proposal_scale=proposal_scale,
                        reparam_cv_before=before_spacing[
                            "segment_length_cv"
                        ],
                        reparam_cv_after=working_diagnostic[
                            "segment_length_cv"
                        ],
                        reparam_equidistance_before=before_spacing[
                            "equidistance_integral"
                        ],
                        reparam_equidistance_after=working_diagnostic[
                            "equidistance_integral"
                        ],
                        counter_snapshot=trace_counters,
                    )
                if (
                    metric_mode
                    and args.reparam_recovery_merge == "bicriteria"
                ):
                    equidistance_ceiling = recovery_equidistance_ceiling
                    for (
                        trace_path,
                        trace_energies,
                        trace_forces,
                    ) in zip(
                        synchronized_paths,
                        synchronized_energies,
                        synchronized_forces,
                    ):
                        trace_eligible = (
                            np.linalg.norm(
                                trace_path[1:-1]
                                - working_path[1:-1],
                                axis=1,
                            )
                            > 1e-14
                        )
                        (
                            working_path,
                            working_energies,
                            working_forces,
                            working_diagnostic,
                            _,
                        ) = greedy_bicriteria_recovery_acceptance(
                            working_path,
                            working_energies,
                            working_forces,
                            trace_path,
                            trace_energies,
                            trace_forces,
                            trace_eligible,
                            equidistance_ceiling,
                            global_iperp_rtol=0.0,
                        )
                else:
                    working_path = candidate_path
                    working_energies = candidate_energies
                    working_forces = candidate_forces
                    working_diagnostic = path_diagnostics(
                        working_path,
                        working_energies,
                        working_forces[1:-1],
                    )
                recovery_memory.observe(
                    working_path[1:-1], -working_forces[1:-1]
                )

            attempt_end = counter_arrays(evaluators)[
                "joint_evaluations"
            ]
            recovery_calls = int(
                np.sum(attempt_end[1:-1] - attempt_start[1:-1])
                - direct_calls
            )
            recovery_max_calls = max(
                0,
                int(
                    np.max(
                        attempt_end[1:-1] - attempt_start[1:-1]
                    )
                )
                - 1,
            )
            state_reparam_recovery_joint_evaluations += recovery_calls
            if recovered():
                if curve_mode:
                    print(
                        "[IVM curve recovery] status=committed "
                        f"Iperp={working_diagnostic['i_perp']:.6e} "
                        "equidistance="
                        f"{working_diagnostic['equidistance_integral']:.6e} "
                        f"recovery_calls={recovery_max_calls}"
                    )
                path = working_path
                energies = working_energies
                forces = working_forces
                diagnostic = working_diagnostic
                gradients = -forces[1:-1]
                memory.reset()
                memory.observe(path[1:-1], gradients)
                state_reparam_accepts += 1
                last_successful_reparam_cv_before = float(
                    before_spacing["segment_length_cv"]
                )
                last_successful_reparam_equidistance_after = float(
                    diagnostic["equidistance_integral"]
                )
                # A successful commit has its own spacing-state hysteresis;
                # the I_perp retry gate is reserved for failed proposals.
                if metric_mode:
                    last_state_reparam_attempt_i_perp = np.inf
                if split_mode:
                    if recovery_max_calls <= args.state_split_fast_recovery:
                        state_split_dtau = min(
                            args.state_split_dtau_max,
                            state_split_dtau * args.state_split_growth,
                        )
                    elif recovery_max_calls >= args.state_split_slow_recovery:
                        state_split_dtau = max(
                            args.state_split_dtau_min,
                            state_split_dtau * args.state_split_shrink,
                        )
                committed = np.ones(
                    path.shape[0] - 2, dtype=bool
                )
                record(
                    accepted_fraction=1.0,
                    rejected=0,
                    reparam_residual=residual,
                    reparam_accepted_fraction=1.0,
                    reparam_rejected=0,
                    reparam_direct_joint_evaluations=direct_calls,
                    reparam_attempted=True,
                    reparam_committed=True,
                    reparam_proposal_scale=proposal_scale,
                    reparam_cv_before=before_spacing[
                        "segment_length_cv"
                    ],
                    reparam_cv_after=diagnostic["segment_length_cv"],
                    reparam_equidistance_before=before_spacing[
                        "equidistance_integral"
                    ],
                    reparam_equidistance_after=diagnostic[
                        "equidistance_integral"
                    ],
                    reparam_recovery_joint_evaluations=recovery_calls,
                )
                return residual, committed, True, direct_calls
            # Rollback is logical only: every rejected evaluation remains in
            # the counters and the run-local coordinate cache.
            if curve_mode:
                print(
                    "[IVM curve recovery] status=rolled_back "
                    f"Iperp={working_diagnostic['i_perp']:.6e} "
                    "equidistance="
                    f"{working_diagnostic['equidistance_integral']:.6e} "
                    f"recovery_calls={recovery_max_calls}"
                )
            path = accepted_path
            energies = accepted_energies
            forces = accepted_forces
            diagnostic = accepted_diagnostic
            gradients = -forces[1:-1]
            if split_mode:
                # The next state-split attempt uses a fresh current-tangent
                # Euler microstep.  A failed trial must not impose the
                # macrostep retry gate intended for large spring proposals.
                last_state_reparam_attempt_i_perp = np.inf
                state_split_dtau = max(
                    args.state_split_dtau_min,
                    state_split_dtau * args.state_split_shrink,
                )
            if curve_mode:
                # A failed curve-continuation stage is retried only after the
                # ordinary state gate observes further I_perp reduction.
                last_state_reparam_attempt_i_perp = float(
                    accepted_diagnostic["i_perp"]
                )
            record(
                accepted_fraction=0.0,
                rejected=0,
                reparam_residual=residual,
                reparam_accepted_fraction=0.0,
                reparam_rejected=path.shape[0] - 2,
                reparam_direct_joint_evaluations=direct_calls,
                reparam_attempted=True,
                reparam_committed=False,
                reparam_proposal_scale=proposal_scale,
                reparam_cv_before=before_spacing["segment_length_cv"],
                reparam_cv_after=accepted_diagnostic[
                    "segment_length_cv"
                ],
                reparam_equidistance_before=before_spacing[
                    "equidistance_integral"
                ],
                reparam_equidistance_after=accepted_diagnostic[
                    "equidistance_integral"
                ],
                reparam_recovery_joint_evaluations=recovery_calls,
            )
            return (
                residual,
                np.zeros(path.shape[0] - 2, dtype=bool),
                True,
                direct_calls,
            )
        if args.reparam_mode == "geometric_euler_guarded":
            if (
                args.reparam_ready_fmax > 0.0
                and diagnostic["fmax"] > args.reparam_ready_fmax
            ):
                return (
                    0.0,
                    np.zeros(path.shape[0] - 2, dtype=bool),
                    False,
                    0,
                )
            counter_before = int(
                np.sum(counter_arrays(evaluators)["joint_evaluations"])
            )
            candidate_path = path.copy()
            spacing_before = equidistance_diagnostics(path)
            spacing_candidate = spacing_before
            residual = 0.0
            applied = False
            applied_dtau = 0.0
            for _ in range(args.reparam_spring_microsteps):
                (
                    micro_path,
                    micro_residual,
                    micro_applied,
                    _,
                    micro_spacing,
                    micro_dtau,
                ) = euler_spring_reparameterization(
                    candidate_path,
                    reparam_step_scale,
                    dtau=args.reparam_spring_dtau,
                    backtrack_factor=args.reparam_spring_backtrack,
                )
                residual = max(residual, micro_residual)
                if not micro_applied:
                    break
                candidate_path = micro_path
                spacing_candidate = micro_spacing
                applied = True
                applied_dtau += micro_dtau
            counter_after = int(
                np.sum(counter_arrays(evaluators)["joint_evaluations"])
            )
            direct_calls = counter_after - counter_before
            geometric_reparam_joint_evaluations += direct_calls
            if direct_calls != 0:
                raise RuntimeError(
                    "Euler spring reparameterization called OpenMM"
                )
            if applied:
                pending_geometric_guard = {
                    "path": path.copy(),
                    "energies": energies.copy(),
                    "forces": forces.copy(),
                    "diagnostic": diagnostic,
                    "spacing_before": spacing_before,
                    "spacing_candidate": spacing_candidate,
                    "proposal_scale": float(
                        applied_dtau * np.mean(reparam_step_scale)
                    ),
                }
                path = candidate_path
                path_state_stale = True
                pending_reparam_residual = residual
                memory.reset()
            return (
                residual,
                np.full(path.shape[0] - 2, applied, dtype=bool),
                applied,
                direct_calls,
            )
        if args.reparam_mode in {"geometric", "geometric_guarded"}:
            counter_before = int(
                np.sum(counter_arrays(evaluators)["joint_evaluations"])
            )
            (
                candidate_path,
                residual,
                applied,
                _,
                _,
            ) = geometric_spring_reparameterization(
                path,
                reparam_step_scale,
            )
            counter_after = int(
                np.sum(counter_arrays(evaluators)["joint_evaluations"])
            )
            direct_calls = counter_after - counter_before
            geometric_reparam_joint_evaluations += direct_calls
            if direct_calls != 0:
                raise RuntimeError(
                    "Geometry-only reparameterization called OpenMM"
                )
            if applied:
                if args.reparam_mode == "geometric_guarded":
                    pending_geometric_guard = {
                        "path": path.copy(),
                        "energies": energies.copy(),
                        "forces": forces.copy(),
                        "diagnostic": diagnostic,
                        "spacing_before": equidistance_diagnostics(path),
                        "spacing_candidate": equidistance_diagnostics(
                            candidate_path
                        ),
                        "proposal_scale": float(
                            np.mean(reparam_step_scale)
                        ),
                    }
                path = candidate_path
                path_state_stale = True
                pending_reparam_residual = residual
                memory.reset()
            return (
                residual,
                np.full(path.shape[0] - 2, applied, dtype=bool),
                applied,
                direct_calls,
            )
        proposed, residual = spring_reparameterize(
            path,
            spring_constant=1.0,
            dtau=0.05,
            max_iter=500,
            tolerance=1e-5,
        )
        scaled = path.copy()
        scaled[1:-1] += reparam_step_scale[:, None] * (
            proposed[1:-1] - path[1:-1]
        )
        if np.array_equal(scaled, path):
            return (
                residual,
                np.ones(path.shape[0] - 2, dtype=bool),
                False,
                0,
            )
        counter_before = int(
            np.sum(counter_arrays(evaluators)["joint_evaluations"])
        )
        candidate_energies, candidate_forces = evaluate_path(
            scaled, evaluators
        )
        direct_calls = int(
            np.sum(counter_arrays(evaluators)["joint_evaluations"])
        ) - counter_before
        accepted, _ = reparameterization_acceptance_mask(
            path,
            scaled,
            energies,
            candidate_energies,
            forces,
            candidate_forces,
            active_alpha,
            args.reparam_objective_tolerance_kcal,
            args.reparam_energy_tolerance_kcal,
            args.reparam_force_rtol,
        )
        reparam_call = int(
            np.max(counter_arrays(evaluators)["joint_evaluations"][1:-1])
        )
        active_global_iperp_rtol = scheduled_value(
            args.global_iperp_rtol,
            args.global_iperp_rtol_schedule,
            reparam_call,
        )
        (
            path,
            energies,
            forces,
            diagnostic,
            globally_accepted,
        ) = greedy_global_iperp_acceptance(
            path,
            energies,
            forces,
            scaled,
            candidate_energies,
            candidate_forces,
            accepted,
            active_global_iperp_rtol,
            order=args.global_iperp_acceptance_order,
        )
        reparam_step_scale[globally_accepted] = np.minimum(
            1.0, reparam_step_scale[globally_accepted] * 1.1
        )
        reparam_step_scale[~globally_accepted] = np.maximum(
            0.01, reparam_step_scale[~globally_accepted] * 0.5
        )
        gradients = -forces[1:-1]
        memory.observe(
            path[1:-1], gradients, accept_mask=globally_accepted
        )
        return residual, globally_accepted, True, direct_calls

    stop_reason = "evaluation_budget"
    outer = 0
    restart_at_calls = {
        int(item)
        for item in args.restart_at_calls.split(",")
        if item.strip()
    }
    completed_scheduled_restarts = set()
    zero_cost_no_progress_outer = 0
    while True:
        if args.stop_iperp > 0.0 and diagnostic["i_perp"] <= args.stop_iperp:
            stop_reason = "i_perp_target"
            break
        if (
            args.stop_complete_iperp > 0.0
            and diagnostic["i_perp"] <= args.stop_complete_iperp
            and history["reparam_committed"]
            and bool(history["reparam_committed"][-1])
        ):
            # An online, state-only stopping rule for operator-split IVM:
            # publish only after the requested residual has been reached at a
            # fully evaluated and accepted reparameterization macrostate.
            # The rule uses neither a dense-barrier diagnostic nor knowledge
            # of any competing method.
            stop_reason = "complete_reparameterized_i_perp_target"
            break
        if (
            args.state_kkt_iperp_tighten_trigger > 0.0
            and args.state_kkt_stall_tighten_target > 0.0
            and active_plbfgs_kkt_rtol
            > args.state_kkt_stall_tighten_target + 1e-14
            and diagnostic["i_perp"]
            <= args.state_kkt_iperp_tighten_trigger
        ):
            # Standard residual-driven inexact-solve forcing.  The trigger
            # depends only on the accepted path state; alpha and the
            # proximal path functional remain unchanged.
            active_plbfgs_kkt_rtol = float(
                args.state_kkt_stall_tighten_target
            )
            memory.reset()
            memory.observe(path[1:-1], -forces[1:-1])
        if (
            (
                args.solver == "state_ivm"
                and not state_strict_polish_active
                and state_reparam_accepts > 0
                and diagnostic["segment_length_cv"]
                <= args.reparam_cv_trigger
            )
            or (
                args.solver == "state_ivm"
                and not state_strict_polish_active
                and args.state_strict_polish_fmax > 0.0
                and diagnostic["fmax"]
                <= args.state_strict_polish_fmax
            )
        ):
            # High physical accuracy, or a successful well-resolved spring
            # macrostep, makes refreshed central tangents suitable for strict
            # residual globalization.  Keep the switch permanent so later
            # spacing drift cannot reintroduce realignment oscillations.
            state_strict_polish_active = True
        if path_state_stale:
            try:
                energies, forces = evaluate_path(path, evaluators)
            except EvaluationBudgetReached:
                if pending_geometric_guard is not None:
                    path = pending_geometric_guard["path"]
                    energies = pending_geometric_guard["energies"]
                    forces = pending_geometric_guard["forces"]
                    diagnostic = pending_geometric_guard["diagnostic"]
                    gradients = -forces[1:-1]
                    pending_geometric_guard = None
                stop_reason = "reparameterization_exhausted_budget"
                break
            diagnostic = path_diagnostics(path, energies, forces[1:-1])
            reparam_attempted = pending_geometric_guard is not None
            reparam_committed = True
            proposal_scale = np.nan
            spacing_before = None
            spacing_candidate = None
            if pending_geometric_guard is not None:
                spacing_before = pending_geometric_guard[
                    "spacing_before"
                ]
                spacing_candidate = pending_geometric_guard[
                    "spacing_candidate"
                ]
                proposal_scale = pending_geometric_guard[
                    "proposal_scale"
                ]
                reference_diagnostic = pending_geometric_guard[
                    "diagnostic"
                ]
                ratio = diagnostic["i_perp"] / max(
                    reference_diagnostic["i_perp"], 1e-300
                )
                reparam_committed = ratio <= (
                    1.0 + args.reparam_guard_iperp_rtol
                )
                if reparam_committed:
                    if ratio <= (
                        1.0 + args.reparam_guard_growth_iperp_rtol
                    ):
                        reparam_step_scale[:] = np.minimum(
                            args.reparam_guard_max_scale,
                            reparam_step_scale
                            * args.reparam_guard_growth,
                        )
                else:
                    # The rejected OpenMM evaluation remains in all counters.
                    # Exact rollback reuses this run's coordinate cache.
                    path = pending_geometric_guard["path"]
                    energies = pending_geometric_guard["energies"]
                    forces = pending_geometric_guard["forces"]
                    diagnostic = reference_diagnostic
                    reparam_step_scale[:] = np.maximum(
                        args.reparam_guard_min_scale,
                        reparam_step_scale
                        * args.reparam_guard_shrink,
                    )
                pending_geometric_guard = None
            gradients = -forces[1:-1]
            memory.observe(path[1:-1], gradients)
            path_state_stale = False
            record(
                reparam_residual=pending_reparam_residual,
                reparam_accepted_fraction=(
                    float(reparam_committed)
                    if reparam_attempted
                    else 1.0
                ),
                reparam_rejected=(
                    0
                    if reparam_committed
                    else path.shape[0] - 2
                ),
                reparam_direct_joint_evaluations=0,
                reparam_attempted=reparam_attempted,
                reparam_committed=(
                    reparam_committed if reparam_attempted else False
                ),
                reparam_proposal_scale=proposal_scale,
                reparam_cv_before=(
                    spacing_before["segment_length_cv"]
                    if spacing_before is not None
                    else np.nan
                ),
                reparam_cv_after=(
                    spacing_candidate["segment_length_cv"]
                    if spacing_candidate is not None
                    else np.nan
                ),
                reparam_equidistance_before=(
                    spacing_before["equidistance_integral"]
                    if spacing_before is not None
                    else np.nan
                ),
                reparam_equidistance_after=(
                    spacing_candidate["equidistance_integral"]
                    if spacing_candidate is not None
                    else np.nan
                ),
            )
            pending_reparam_residual = 0.0
        counters = counter_arrays(evaluators)["joint_evaluations"][1:-1]
        current_call = int(np.max(counters))
        if current_call >= args.budget:
            break
        if (
            args.frozen_subproblem == "ala_dsel"
            and not ala_dsel_active
            and diagnostic["i_perp"]
            <= float(args.ala_dsel_trigger_iperp)
        ):
            # This state-only switch is permanent.  A later I_perp rebound
            # must not make the direction/stop policy oscillate.  The
            # selector does not alter alpha: both the legacy prefix and the
            # activated DSEL suffix use the same existing active-alpha path.
            ala_dsel_active = True
            ala_dsel_first_activation_outer = int(outer)
            ala_dsel_first_activation_calls_per_image = int(current_call)
            ala_dsel_first_activation_nf = int(np.sum(counters))
            ala_dsel_first_activation_iperp = float(
                diagnostic["i_perp"]
            )
        if (
            current_call in restart_at_calls
            and current_call not in completed_scheduled_restarts
        ):
            memory.reset()
            memory.observe(path[1:-1], gradients)
            step_scale[:] = 1.0
            completed_scheduled_restarts.add(current_call)
        if args.solver == "proximal":
            active_alpha = contractive_alpha_value
        elif args.alpha_mode == "adaptive":
            active_alpha = adaptive_alpha_value
        else:
            active_alpha = scheduled_value(
                args.alpha, args.alpha_schedule, current_call
            )
        active_alpha, last_alpha_precision_cap = residual_precision_cap(
            active_alpha,
            diagnostic["fmax"],
            args.alpha_precision_caps,
        )
        active_alpha, relative_cap = relative_iperp_precision_cap(
            active_alpha,
            diagnostic["i_perp"] / initial_i_perp,
            args.alpha_relative_iperp_caps,
        )
        last_alpha_precision_cap = min(
            last_alpha_precision_cap, relative_cap
        )
        if args.alpha_mode == "adaptive":
            adaptive_alpha_value = active_alpha
        if args.solver == "state_ivm" and state_strict_polish_active:
            active_alpha = min(
                active_alpha,
                args.alpha_max * args.state_polish_alpha_factor,
            )
        if ffprecon_retired and args.ffprecon_exit_alpha > 0.0:
            active_alpha = state_threshold_value(
                args.ffprecon_exit_alpha,
                args.ffprecon_exit_alpha_policy,
                diagnostic["i_perp"],
            )
        record_alpha_value = active_alpha
        active_step = scheduled_value(
            args.max_step, args.max_step_schedule, current_call
        )
        if args.solver == "state_ivm":
            trust_fraction = (
                args.state_polish_trust_fraction
                if state_strict_polish_active
                else args.state_trust_fraction
            )
            active_step = rms_segment_trust_radius(
                path,
                trust_fraction,
                args.max_step,
            )
        if (
            ffprecon_retired
            and args.ffprecon_exit_max_step > 0.0
        ):
            active_step = args.ffprecon_exit_max_step
        active_iperp_rtol = scheduled_value(
            args.iperp_rtol,
            args.iperp_rtol_schedule,
            current_call,
        )
        if args.restart_every > 0 and outer > 0 and outer % args.restart_every == 0:
            memory.reset()
            memory.observe(path[1:-1], gradients)
        tangents = central_tangents(path)
        current_diag = diagnostic
        current_path = path.copy()
        current_energies = energies.copy()
        current_forces = forces.copy()
        current_gradients = gradients.copy()

        if (
            ffprecon_active
            and args.ffprecon_exit_iperp > 0.0
            and diagnostic["i_perp"] <= args.ffprecon_exit_iperp
        ):
            ffprecon_active = False
            ffprecon_retired = True
            ffprecon_exit_iterations = 0
            ffprecon_exit_globalization_active = False
            ffprecon_exit_polish_active = False
            if args.ffprecon_exit_alpha > 0.0:
                active_alpha = state_threshold_value(
                    args.ffprecon_exit_alpha,
                    args.ffprecon_exit_alpha_policy,
                    diagnostic["i_perp"],
                )
                record_alpha_value = active_alpha
            if args.ffprecon_exit_max_step > 0.0:
                active_step = args.ffprecon_exit_max_step
        if (
            ffprecon_retired
            and not ffprecon_exit_globalization_active
            and args.ffprecon_exit_globalization
            == "proximal_collective"
            and (
                args.ffprecon_exit_globalization_trigger_iperp <= 0.0
                or diagnostic["i_perp"]
                <= args.ffprecon_exit_globalization_trigger_iperp
            )
        ):
            # The switch is permanent.  A collective IVM update can
            # transiently raise I_perp while the central tangents realign;
            # reverting the policy on that rise would recreate the original
            # false-monotonicity lock.
            ffprecon_exit_globalization_active = True
        if (
            ffprecon_retired
            and not ffprecon_exit_polish_active
            and args.ffprecon_exit_polish_iperp > 0.0
            and diagnostic["i_perp"]
            <= args.ffprecon_exit_polish_iperp
        ):
            # Once the collective phase has realigned the central tangents,
            # return to strict I_perp globalization for a protected finish.
            # This prevents continued large-radius subproblem motion from
            # degrading the densified barrier through sparse path sampling.
            ffprecon_exit_polish_active = True
        if (
            args.solver == "hybrid_ff"
            and not ffprecon_active
            and not ffprecon_retired
            and args.ffprecon_trigger_iperp > 0.0
            and diagnostic["i_perp"] <= args.ffprecon_trigger_iperp
        ):
            ffprecon_active = True
            ffprecon_iterations = 0
            ffprecon_step_scale[:] = 1.0
            ffprecon_raw_force_active = False
            ffprecon_frozen_path = None
            ffprecon_frozen_tangents = None
            ffprecon_accepted_objective = None
            # Early history was accumulated while the path and its normal
            # spaces rotated rapidly.  Start a valley-local history at this
            # state; subsequent accepted FF-preconditioned subproblem steps
            # warm up the non-bonded curvature correction.
            memory.reset()
            memory.observe(path[1:-1], gradients)

        if ffprecon_active:
            if (
                args.ffprecon_lbfgs_history
                and not ffprecon_history_active
                and (
                    args.ffprecon_lbfgs_history_trigger_iperp <= 0.0
                    or diagnostic["i_perp"]
                    <= args.ffprecon_lbfgs_history_trigger_iperp
                )
            ):
                ffprecon_history_active = True
                # History before this state belongs to rapidly rotating
                # normal spaces.  Start a local, run-owned physical-curvature
                # memory and restore proposal scales at the same state.
                memory.reset()
                memory.observe(path[1:-1], gradients)
                ffprecon_step_scale[:] = 1.0
                ffprecon_frozen_path = None
                ffprecon_frozen_tangents = None
                ffprecon_accepted_objective = None
            if (
                not ffprecon_collective_active
                and args.ffprecon_collective_trigger_iperp > 0.0
                and diagnostic["i_perp"]
                <= args.ffprecon_collective_trigger_iperp
            ):
                ffprecon_collective_active = True
                # Earlier one-image merges may have collapsed the step scales
                # of images whose descent requires their neighbours to move.
                # Re-enable those run-local subproblem candidates exactly once
                # when coupled acceptance is activated.
                ffprecon_step_scale[:] = 1.0
                ffprecon_frozen_path = None
                ffprecon_frozen_tangents = None
                ffprecon_accepted_objective = None
            if (
                not ffprecon_late_c_stab_active
                and args.ffprecon_late_c_stab > 0.0
                and args.ffprecon_late_c_stab_trigger_iperp > 0.0
                and diagnostic["i_perp"]
                <= args.ffprecon_late_c_stab_trigger_iperp
            ):
                ffprecon_late_c_stab_active = True
                ffprecon_bank.c_stab = args.ffprecon_late_c_stab
                ffprecon_bank.preconditioners = None
                ffprecon_step_scale[:] = 1.0
                ffprecon_frozen_path = None
                ffprecon_frozen_tangents = None
                ffprecon_accepted_objective = None
            if (
                not ffprecon_raw_force_active
                and args.ffprecon_raw_force_trigger_iperp > 0.0
                and diagnostic["i_perp"]
                <= args.ffprecon_raw_force_trigger_iperp
            ):
                ffprecon_raw_force_active = True
                ffprecon_step_scale[:] = 1.0
                ffprecon_frozen_path = None
                ffprecon_frozen_tangents = None
                ffprecon_accepted_objective = None
            if (
                ffprecon_frozen_path is None
                or ffprecon_iterations
                % max(1, args.ffprecon_inner_calls)
                == 0
            ):
                ffprecon_frozen_path = path.copy()
                ffprecon_frozen_tangents = central_tangents(path)
                ffprecon_accepted_objective = energies[1:-1].copy()
            if (
                ffprecon_bank.preconditioners is None
                or ffprecon_iterations
                % max(1, args.ffprecon_rebuild_every)
                == 0
            ):
                raw_before = int(backend.raw_joint_attempts)
                ffprecon_bank.build(path)
                if int(backend.raw_joint_attempts) != raw_before:
                    raise RuntimeError(
                        "FF preconditioner construction called OpenMM"
                    )
            ffprecon_displacement = project_normal(
                path[1:-1] - ffprecon_frozen_path[1:-1],
                ffprecon_frozen_tangents,
            )
            normal_gradients = (
                project_normal(
                    gradients, ffprecon_frozen_tangents
                )
                + ffprecon_displacement / active_alpha
            )
            if ffprecon_raw_force_active:
                if args.ffprecon_raw_force_learning_rate > 0.0:
                    directions = np.asarray(
                        [
                            -float(scale)
                            * args.ffprecon_raw_force_learning_rate
                            * gradient
                            for gradient, scale in zip(
                                normal_gradients, ffprecon_step_scale
                            )
                        ]
                    )
                    if args.ffprecon_raw_force_max_step > 0.0:
                        directions = np.asarray(
                            [
                                clip_max_atomic_displacement(
                                    direction,
                                    args.ffprecon_raw_force_max_step,
                                )
                                for direction in directions
                            ]
                        )
                else:
                    directions = np.asarray(
                        [
                            float(scale)
                            * clip_max_atomic_displacement(
                                -gradient,
                                args.ffprecon_raw_force_max_step,
                            )
                            for gradient, scale in zip(
                                normal_gradients, ffprecon_step_scale
                            )
                        ]
                    )
            else:
                directions = ffprecon_bank.directions(
                    normal_gradients,
                    ffprecon_frozen_tangents,
                    args.ffprecon_max_step,
                    ffprecon_step_scale,
                    curvature_memory=(
                        memory if ffprecon_history_active else None
                    ),
                    alpha=active_alpha,
                )
            update_mask = ffprecon_update_mask(
                path.shape[0] - 2,
                ffprecon_iterations,
                args.ffprecon_update_scheme,
            )
            directions[~update_mask] = 0.0
            candidate = path.copy()
            candidate[1:-1] += directions
            candidate_displacement = project_normal(
                candidate[1:-1] - ffprecon_frozen_path[1:-1],
                ffprecon_frozen_tangents,
            )
            candidate[1:-1] = (
                ffprecon_frozen_path[1:-1] + candidate_displacement
            )
            previous_max_call = int(
                np.max(
                    counter_arrays(evaluators)[
                        "joint_evaluations"
                    ][1:-1]
                )
            )
            try:
                candidate_energies, candidate_forces = evaluate_path(
                    candidate, evaluators
                )
            except EvaluationBudgetReached:
                if transactional_operator_split:
                    # A path refresh can exhaust the per-image budget after
                    # only a prefix of images has been queried. Roll back the
                    # incomplete macrostep, but snapshot those real counters
                    # so the saved trajectory and backend audit close exactly.
                    path = current_path.copy()
                    energies = current_energies.copy()
                    forces = current_forces.copy()
                    diagnostic = current_diag
                    gradients = -forces[1:-1]
                    record(
                        accepted_fraction=0.0,
                        rejected=path.shape[0] - 2,
                        reparam_attempted=True,
                        reparam_committed=False,
                    )
                stop_reason = "evaluation_budget_during_relock"
                break
            candidate_max_call = int(
                np.max(
                    counter_arrays(evaluators)[
                        "joint_evaluations"
                    ][1:-1]
                )
            )
            if candidate_max_call == previous_max_call:
                stop_reason = "stalled_cached_rejection"
                break
            candidate_diag = path_diagnostics(
                candidate,
                candidate_energies,
                candidate_forces[1:-1],
            )
            if args.ffprecon_acceptance == "proximal_merit":
                candidate_objective = (
                    candidate_energies[1:-1]
                    + 0.5
                    * np.sum(candidate_displacement**2, axis=1)
                    / active_alpha
                )
                objective_tolerance = args.energy_rtol * np.maximum(
                    1.0, np.abs(ffprecon_accepted_objective)
                )
                eligible = candidate_objective <= (
                    ffprecon_accepted_objective + objective_tolerance
                )
            else:
                eligible, candidate_objective = (
                    ffprecon_subproblem_acceptance_mask(
                        energies[1:-1],
                        candidate_energies[1:-1],
                        candidate[1:-1] - path[1:-1],
                        active_alpha,
                        args.energy_rtol,
                        args.ffprecon_acceptance,
                        current_per_image_l2=diagnostic["per_image_l2"],
                        candidate_per_image_l2=(
                            candidate_diag["per_image_l2"]
                        ),
                        force_rtol=args.ffprecon_force_rtol,
                    )
                )
            eligible &= update_mask
            ff_call = int(
                np.max(
                    counter_arrays(evaluators)[
                        "joint_evaluations"
                    ][1:-1]
                )
            )
            active_global_iperp_rtol = (
                args.ffprecon_global_iperp_rtol
            )
            (
                path,
                energies,
                forces,
                diagnostic,
                accepted,
            ) = greedy_global_iperp_acceptance(
                current_path,
                current_energies,
                current_forces,
                candidate,
                candidate_energies,
                candidate_forces,
                eligible,
                active_global_iperp_rtol,
                order=(
                    "collective_first"
                    if ffprecon_raw_force_active
                    else (
                        "collective_block"
                        if ffprecon_collective_active
                        else args.global_iperp_acceptance_order
                    )
                ),
            )
            ffprecon_step_scale[update_mask & accepted] = np.minimum(
                1.0,
                ffprecon_step_scale[update_mask & accepted]
                * args.ffprecon_step_growth,
            )
            rejected_mask = update_mask & ~accepted
            ffprecon_accepted_objective[accepted] = (
                candidate_objective[accepted]
            )
            ffprecon_step_scale[rejected_mask] = np.maximum(
                args.ffprecon_step_min,
                ffprecon_step_scale[rejected_mask]
                * args.ffprecon_step_shrink,
            )
            gradients = -forces[1:-1]
            memory.observe(
                path[1:-1], gradients, accept_mask=accepted
            )
            ffprecon_iterations += 1
            record(
                accepted_fraction=float(
                    np.count_nonzero(accepted)
                    / max(1, np.count_nonzero(update_mask))
                ),
                rejected=int(np.count_nonzero(rejected_mask)),
            )
            reparam_call = int(
                np.max(
                    counter_arrays(evaluators)["joint_evaluations"][1:-1]
                )
            )
            active_reparam_every = int(
                round(
                    scheduled_value(
                        args.reparam_every,
                        args.reparam_every_schedule,
                        reparam_call,
                    )
                )
            )
            if (
                active_reparam_every > 0
                and reparam_call < args.budget
                and (outer + 1) % active_reparam_every == 0
            ):
                try:
                    (
                        residual,
                        reparam_accepted,
                        reparameterized,
                        reparam_direct_calls,
                    ) = attempt_reparameterization(active_alpha)
                except EvaluationBudgetReached:
                    break
                if reparameterized and args.reparam_mode == "audited":
                    record(
                        reparam_residual=residual,
                        reparam_accepted_fraction=float(
                            np.mean(reparam_accepted)
                        ),
                        reparam_rejected=int(
                            np.count_nonzero(~reparam_accepted)
                        ),
                        reparam_direct_joint_evaluations=(
                            reparam_direct_calls
                        ),
                    )
                if reparameterized:
                    ffprecon_frozen_path = None
                    ffprecon_frozen_tangents = None
                    ffprecon_accepted_objective = None
            outer += 1
            continue

        use_projected_lbfgs = args.solver == "lbfgs" or (
            args.solver in {"hybrid", "hybrid_ff"}
            and (
                current_call < args.hybrid_switch_call
                or (
                    args.hybrid_return_call > 0
                    and current_call >= args.hybrid_return_call
                )
            )
        )
        if use_projected_lbfgs:
            active_inner_calls = int(
                round(
                    scheduled_value(
                        args.inner_calls,
                        args.inner_calls_schedule,
                        current_call,
                    )
                )
            )
            active_inner_calls = max(1, active_inner_calls)
            if (
                args.reparam_mode == "geometric_euler_guarded"
                and args.reparam_state_inner_calls > 0
                and (
                    args.reparam_ready_fmax <= 0.0
                    or diagnostic["fmax"] <= args.reparam_ready_fmax
                )
            ):
                active_inner_calls = args.reparam_state_inner_calls
            if (
                ffprecon_retired
                and args.ffprecon_exit_inner_calls > 0
            ):
                active_inner_calls = args.ffprecon_exit_inner_calls
            frozen_path = path.copy()
            frozen_tangents = tangents.copy()
            displacement = np.zeros_like(path[1:-1])
            accepted_positions = path[1:-1].copy()
            accepted_energies = energies[1:-1].copy()
            accepted_forces = forces[1:-1].copy()
            accepted_gradients = gradients.copy()
            accepted_l2 = current_diag["per_image_l2"].copy()
            accepted_objective = accepted_energies.copy()
            last_accepted = np.ones(path.shape[0] - 2, dtype=bool)
            for _ in range(active_inner_calls):
                counters = counter_arrays(evaluators)["joint_evaluations"][1:-1]
                current_call = int(np.max(counters))
                if current_call >= args.budget:
                    break
                normal_gradient = project_normal(
                    accepted_gradients, frozen_tangents
                )
                residual_gradient = (
                    normal_gradient + displacement / active_alpha
                )
                proposal_positions = accepted_positions.copy()
                for i in range(path.shape[0] - 2):
                    direction = memory.direction(
                        i,
                        residual_gradient[i],
                        frozen_tangents[i],
                        active_alpha,
                    )
                    proposal_positions[i] += clip_max_atomic_displacement(
                        direction, active_step * step_scale[i]
                    )
                proposal_displacement = project_normal(
                    proposal_positions - frozen_path[1:-1],
                    frozen_tangents,
                )
                proposal_positions = (
                    frozen_path[1:-1] + proposal_displacement
                )
                candidate = frozen_path.copy()
                candidate[1:-1] = proposal_positions
                try:
                    candidate_energies, candidate_forces = evaluate_path(
                        candidate, evaluators
                    )
                except EvaluationBudgetReached:
                    break
                new_call = int(
                    np.max(
                        counter_arrays(evaluators)["joint_evaluations"][1:-1]
                    )
                )
                if new_call == current_call:
                    stop_reason = "stalled_cached_rejection"
                    break
                candidate_diag = path_diagnostics(
                    candidate,
                    candidate_energies,
                    candidate_forces[1:-1],
                )
                candidate_objective = (
                    candidate_energies[1:-1]
                    + 0.5
                    * np.sum(proposal_displacement**2, axis=1)
                    / active_alpha
                )
                objective_tolerance = args.energy_rtol * np.maximum(
                    1.0, np.abs(accepted_objective)
                )
                objective_ok = (
                    candidate_objective
                    <= accepted_objective + objective_tolerance
                )
                force_ok = candidate_diag["per_image_l2"] <= (
                    accepted_l2 * (1.0 + active_iperp_rtol) + 1e-12
                )
                last_accepted = objective_ok & force_ok
                step_scale[last_accepted] = np.minimum(
                    1.0, step_scale[last_accepted] * 1.1
                )
                step_scale[~last_accepted] = np.maximum(
                    0.01, step_scale[~last_accepted] * 0.5
                )
                accepted_positions[last_accepted] = proposal_positions[last_accepted]
                accepted_energies[last_accepted] = candidate_energies[1:-1][
                    last_accepted
                ]
                accepted_forces[last_accepted] = candidate_forces[1:-1][
                    last_accepted
                ]
                accepted_gradients[last_accepted] = -candidate_forces[1:-1][
                    last_accepted
                ]
                accepted_objective[last_accepted] = candidate_objective[
                    last_accepted
                ]
                displacement = project_normal(
                    accepted_positions - frozen_path[1:-1],
                    frozen_tangents,
                )
                accepted_l2[last_accepted] = candidate_diag["per_image_l2"][
                    last_accepted
                ]
                memory.observe(
                    accepted_positions,
                    accepted_gradients,
                    accept_mask=last_accepted,
                )
                path = frozen_path.copy()
                path[1:-1] = accepted_positions
                energies = current_energies.copy()
                energies[1:-1] = accepted_energies
                forces = current_forces.copy()
                forces[1:-1] = accepted_forces
                gradients = accepted_gradients.copy()
                diagnostic = path_diagnostics(
                    path, energies, forces[1:-1]
                )
                record(
                    accepted_fraction=float(np.mean(last_accepted)),
                    rejected=int(np.count_nonzero(~last_accepted)),
                    reparam_residual=0.0,
                )

            if stop_reason == "stalled_cached_rejection":
                break
            update_alpha_controller(
                current_path,
                current_energies,
                current_forces,
                path,
                energies,
                frozen_tangents,
                active_alpha,
                active_step,
            )
            record(
                accepted_fraction=float(np.mean(last_accepted)),
                rejected=int(np.count_nonzero(~last_accepted)),
            )
            residual = 0.0
            reparameterized_and_evaluated = False
            reparam_direct_calls = 0
            reparam_accepted = np.ones(path.shape[0] - 2, dtype=bool)
            reparam_call = int(
                np.max(
                    counter_arrays(evaluators)["joint_evaluations"][1:-1]
                )
            )
            active_reparam_every = int(
                round(
                    scheduled_value(
                        args.reparam_every,
                        args.reparam_every_schedule,
                        reparam_call,
                    )
                )
            )
            if (
                active_reparam_every > 0
                and reparam_call < args.budget
                and (outer + 1) % active_reparam_every == 0
            ):
                try:
                    (
                        residual,
                        reparam_accepted,
                        reparameterized_and_evaluated,
                        reparam_direct_calls,
                    ) = attempt_reparameterization(active_alpha)
                except EvaluationBudgetReached:
                    break
            if (
                reparameterized_and_evaluated
                and args.reparam_mode == "audited"
            ):
                record(
                    accepted_fraction=float(np.mean(last_accepted)),
                    rejected=int(np.count_nonzero(~last_accepted)),
                    reparam_residual=residual,
                    reparam_accepted_fraction=float(
                        np.mean(reparam_accepted)
                    ),
                    reparam_rejected=int(
                        np.count_nonzero(~reparam_accepted)
                    ),
                    reparam_direct_joint_evaluations=reparam_direct_calls,
                )
            outer += 1
            continue

        try:
            remaining = args.budget - current_call
            if state_free_polish_active:
                # A published operator-split macrostate consists of a
                # physical subproblem step and, when its dimensionless
                # spacing defect crosses the trigger, one refreshed
                # geometry-only spring correction.  Reserve that last query
                # rather than ending on an unparameterized intermediate.
                if remaining <= 1:
                    stop_reason = "complete_operator_split_cycle_budget"
                    break
                solve_remaining = remaining - 1
            else:
                solve_remaining = remaining
            active_lbfgsb_initial_step = scheduled_value(
                args.lbfgsb_initial_step,
                args.lbfgsb_initial_step_schedule,
                current_call,
            )
            active_inner_calls = int(
                round(
                    scheduled_value(
                        args.inner_calls,
                        args.inner_calls_schedule,
                        current_call,
                    )
                )
            )
            active_inner_calls = max(1, active_inner_calls)
            if (
                args.reparam_mode == "geometric_euler_guarded"
                and args.reparam_state_inner_calls > 0
                and (
                    args.reparam_ready_fmax <= 0.0
                    or diagnostic["fmax"] <= args.reparam_ready_fmax
                )
            ):
                active_inner_calls = args.reparam_state_inner_calls
            if (
                ffprecon_retired
                and args.ffprecon_exit_inner_calls > 0
            ):
                active_inner_calls = args.ffprecon_exit_inner_calls
            use_preconditioned_exit_solver = (
                ffprecon_retired
                and args.ffprecon_exit_solver == "plbfgs"
            ) or args.solver == "state_ivm"
            exit_active_images = np.ones(
                path.shape[0] - 2, dtype=bool
            )
            if (
                state_free_polish_active
                and args.state_free_polish_reparam_strategy
                in ("spring_relock", "cv_relock", "polyline_relock")
            ):
                # Escape the discrete equal-spring fixed point by one greedy
                # block subminimization.  Moving only the image with largest
                # current physical normal force limits rotation of adjacent
                # central-difference tangents.  The choice is state based and
                # independent of alpha and force-call count.
                if args.state_free_polish_active_images == "largest":
                    exit_active_images = largest_normal_force_block_mask(
                        diagnostic["per_image_l2"]
                    )
            if (
                use_preconditioned_exit_solver
                and args.ffprecon_exit_update_scheme == "red_black"
                and (
                    args.solver != "state_ivm"
                    or state_free_polish_active
                )
            ):
                exit_active_images = ffprecon_update_mask(
                    path.shape[0] - 2,
                    ffprecon_exit_iterations,
                    "red_black",
                )
            if use_preconditioned_exit_solver:
                raw_before = int(backend.raw_joint_attempts)
                ffprecon_bank.build(path)
                if int(backend.raw_joint_attempts) != raw_before:
                    raise RuntimeError(
                        "FF preconditioner construction called OpenMM"
                    )
            curve_projected_direct = bool(
                args.state_curve_direct_solve
                and args.frozen_subproblem != "discrete_b040"
                and not curve_direct_retired
                and not state_free_polish_active
                and args.solver == "state_ivm"
                and args.reparam_mode == "state_curve_projected"
                and equidistance_diagnostics(path)[
                    "segment_length_cv"
                ]
                <= 1e-8
            )
            discrete_primary_solve = bool(
                args.solver == "state_ivm"
                and args.frozen_subproblem == "discrete_b040"
                and not state_free_polish_active
            )
            if discrete_primary_solve:
                discrete_commit_reserve = int(
                    args.reparam_mode == "state_curve_projected"
                    and equidistance_diagnostics(path)[
                        "segment_length_cv"
                    ]
                    <= 1e-8
                )
                (
                    candidate,
                    candidate_energies,
                    candidate_forces,
                    used_calls,
                    synchronized_paths,
                    synchronized_energies,
                    synchronized_forces,
                    synchronized_counters,
                    subproblem_report,
                ) = solve_frozen_discrete_b040_band(
                    path,
                    energies,
                    forces,
                    tangents,
                    evaluators,
                    active_alpha,
                    active_step,
                    min(
                        int(args.discrete_inner_max_steps),
                        max(
                            0,
                            int(solve_remaining)
                            - discrete_commit_reserve,
                        ),
                    ),
                    inner_max_steps=int(args.discrete_inner_max_steps),
                    stop_mode=args.discrete_stop_mode,
                    relative_residual=args.discrete_residual_ratio,
                    history_size=int(args.discrete_history),
                    kkt_atol=args.plbfgs_kkt_atol,
                    preconditioner_bank=ffprecon_bank,
                    active_images=exit_active_images,
                )
            elif curve_projected_direct:
                (
                    candidate,
                    candidate_energies,
                    candidate_forces,
                    used_calls,
                    synchronized_paths,
                    synchronized_energies,
                    synchronized_forces,
                    synchronized_counters,
                    subproblem_report,
                ) = solve_curve_projected_proximal_band(
                    path,
                    energies,
                    forces,
                    tangents,
                    evaluators,
                    active_alpha,
                    active_step,
                    min(
                        active_inner_calls,
                        args.state_curve_direct_trial_calls,
                        remaining,
                    ),
                    history_size=args.history,
                    armijo=args.plbfgs_armijo,
                    backtrack_factor=args.plbfgs_backtrack,
                    kkt_rtol=(
                        args.discrete_residual_ratio
                        if (
                            ala_dsel_active
                            and args.discrete_stop_mode
                            in DISCRETE_RESIDUAL_STOP_MODES
                        )
                        else 0.0
                        if ala_dsel_active
                        else active_plbfgs_kkt_rtol
                    ),
                    kkt_atol=args.plbfgs_kkt_atol,
                    min_accepted_steps=args.plbfgs_min_accepted_steps,
                    preconditioner_bank=ffprecon_bank,
                    curvature_memory=(
                        memory
                        if state_outer_history_is_active(
                            args.state_outer_history,
                            args.state_outer_history_trigger_fmax,
                            diagnostic["fmax"],
                        )
                        else None
                    ),
                    active_images=exit_active_images,
                    enforce_global_iperp=(
                        args.state_globalization
                        in ("iperp_monotone", "cycle_iperp_monotone")
                    ),
                    line_search_trial_limit=(
                        args.state_curve_direct_failure_line_search_trials
                        if (
                            args.state_curve_immediate_block_relock
                            and args
                            .state_curve_direct_failure_line_search_trials
                            > 0
                        )
                        else 64
                    ),
                    objective_acceptance=(
                        args.state_curve_direct_objective_acceptance
                    ),
                    direction_selection=(
                        "ala_dsel"
                        if ala_dsel_active
                        else "legacy"
                    ),
                    max_accepted_steps=(
                        args.discrete_inner_max_steps
                        if ala_dsel_active
                        else None
                    ),
                )
                record_ala_dsel_report(subproblem_report, outer)
                direct_coordinate_progress = bool(
                    np.any(
                        np.linalg.norm(
                            candidate[1:-1] - path[1:-1], axis=1
                        )
                        > 1e-14
                    )
                )
                if (
                    not np.any(
                        np.asarray(
                            subproblem_report["accepted_steps"], dtype=int
                        )
                    )
                    or not direct_coordinate_progress
                ):
                    # A failed equal-spring line search is a state trigger,
                    # not a call schedule.  Permanently retire the fast
                    # projected predictor and fall back to the original
                    # frozen-normal subproblem followed by a separately
                    # audited spring projection.  Failed predictor queries
                    # remain in the synchronized trace and in every counter.
                    curve_direct_retired = True
                    direct_paths = synchronized_paths
                    direct_energies = synchronized_energies
                    direct_forces = synchronized_forces
                    direct_counters = synchronized_counters
                    direct_used_calls = used_calls.copy()
                    remaining_after_direct = int(
                        args.budget
                        - np.max(
                            counter_arrays(evaluators)[
                                "joint_evaluations"
                            ][1:-1]
                        )
                    )
                    (
                        fallback_budget,
                        _relock_reserve,
                    ) = immediate_relock_fallback_budget(
                        remaining_after_direct,
                        args.state_curve_transactional_relock_budget,
                        args.state_free_polish_reparam_strategy,
                    )
                    if (
                        args.state_curve_immediate_block_relock
                        and fallback_budget > 0
                    ):
                        # The synchronized equal-spring trial has failed its
                        # exact proximal-objective line search.  Immediately
                        # continue the same fixed-tangent IVM problem as one
                        # greedy block subminimization, chosen by the largest
                        # current physical normal force.  This is the original
                        # subminimization followed by optional geometry-only
                        # relocking; it does not weaken or change alpha.
                        state_free_polish_active = True
                        block_active_images = np.ones(
                            path.shape[0] - 2, dtype=bool
                        )
                        if (
                            args.state_free_polish_active_images
                            == "largest"
                        ):
                            block_active_images = (
                                largest_normal_force_block_mask(
                                    diagnostic["per_image_l2"]
                                )
                            )
                        (
                            candidate,
                            candidate_energies,
                            candidate_forces,
                            fallback_used_calls,
                            fallback_paths,
                            fallback_energies,
                            fallback_forces,
                            fallback_counters,
                            subproblem_report,
                        ) = solve_projected_proximal_band(
                            path,
                            energies,
                            forces,
                            tangents,
                            evaluators,
                            active_alpha,
                            active_step
                            * args.state_curve_relock_escape_trust_factor,
                            min(
                                active_inner_calls,
                                fallback_budget,
                            ),
                            history_size=args.history,
                            curvature_memory=None,
                            armijo=args.plbfgs_armijo,
                            backtrack_factor=args.plbfgs_backtrack,
                            kkt_rtol=active_plbfgs_kkt_rtol,
                            kkt_atol=args.plbfgs_kkt_atol,
                            min_accepted_steps=max(
                                1, args.plbfgs_min_accepted_steps
                            ),
                            preconditioner_bank=None,
                            active_images=block_active_images,
                        )
                        synchronized_paths = (
                            direct_paths + fallback_paths
                        )
                        synchronized_energies = (
                            direct_energies + fallback_energies
                        )
                        synchronized_forces = (
                            direct_forces + fallback_forces
                        )
                        synchronized_counters = (
                            direct_counters + fallback_counters
                        )
                        used_calls = (
                            direct_used_calls + fallback_used_calls
                        )
                    elif (
                        args.state_curve_immediate_block_relock
                        and args.state_curve_transactional_relock_budget
                        and remaining_after_direct > 0
                    ):
                        # Failed predictor queries are real and remain
                        # charged, but the last available evaluation is
                        # reserved for a complete spring-force relock. If there
                        # is no room for even one raw block trial plus that
                        # relock, publish the unchanged equal-spring
                        # macrostate and let the next outer budget check stop.
                        # This prevents a budget boundary from exposing a
                        # half-completed operator-split cycle.
                        state_free_polish_active = True
                        candidate = path.copy()
                        candidate_energies = energies.copy()
                        candidate_forces = forces.copy()
                        synchronized_paths = direct_paths
                        synchronized_energies = direct_energies
                        synchronized_forces = direct_forces
                        synchronized_counters = direct_counters
                        used_calls = direct_used_calls
                    elif remaining_after_direct > 0:
                        (
                            candidate,
                            candidate_energies,
                            candidate_forces,
                            fallback_used_calls,
                            fallback_paths,
                            fallback_energies,
                            fallback_forces,
                            fallback_counters,
                            subproblem_report,
                        ) = solve_projected_proximal_band(
                            path,
                            energies,
                            forces,
                            tangents,
                            evaluators,
                            active_alpha,
                            active_step,
                            min(
                                active_inner_calls,
                                remaining_after_direct,
                            ),
                            history_size=args.history,
                            curvature_memory=(
                                memory
                                if state_outer_history_is_active(
                                    args.state_outer_history,
                                    args.state_outer_history_trigger_fmax,
                                    diagnostic["fmax"],
                                )
                                else None
                            ),
                            armijo=args.plbfgs_armijo,
                            backtrack_factor=args.plbfgs_backtrack,
                            kkt_rtol=active_plbfgs_kkt_rtol,
                            kkt_atol=args.plbfgs_kkt_atol,
                            min_accepted_steps=(
                                args.plbfgs_min_accepted_steps
                            ),
                            preconditioner_bank=ffprecon_bank,
                            active_images=exit_active_images,
                        )
                        synchronized_paths = (
                            direct_paths + fallback_paths
                        )
                        synchronized_energies = (
                            direct_energies + fallback_energies
                        )
                        synchronized_forces = (
                            direct_forces + fallback_forces
                        )
                        synchronized_counters = (
                            direct_counters + fallback_counters
                        )
                        used_calls = (
                            direct_used_calls + fallback_used_calls
                        )
                    curve_projected_direct = False
            elif args.solver == "plbfgs" or use_preconditioned_exit_solver:
                (
                    candidate,
                    candidate_energies,
                    candidate_forces,
                    used_calls,
                    synchronized_paths,
                    synchronized_energies,
                    synchronized_forces,
                    synchronized_counters,
                    subproblem_report,
                ) = solve_projected_proximal_band(
                    path,
                    energies,
                    forces,
                    tangents,
                    evaluators,
                    active_alpha,
                    (
                        active_step
                        * args.state_curve_relock_escape_trust_factor
                        if (
                            state_free_polish_active
                            and args.state_free_polish_reparam_strategy
                            in (
                                "spring_relock",
                                "cv_relock",
                                "polyline_relock",
                            )
                        )
                        else active_step
                    ),
                    min(active_inner_calls, solve_remaining),
                    history_size=args.history,
                    curvature_memory=(
                        memory
                        if (
                            args.solver != "state_ivm"
                            or state_outer_history_is_active(
                                args.state_outer_history,
                                args.state_outer_history_trigger_fmax,
                                diagnostic["fmax"],
                            )
                        )
                        else None
                    ),
                    armijo=args.plbfgs_armijo,
                    backtrack_factor=args.plbfgs_backtrack,
                    kkt_rtol=(
                        args.discrete_residual_ratio
                        if (
                            ala_dsel_active
                            and not state_free_polish_active
                            and args.discrete_stop_mode
                            in DISCRETE_RESIDUAL_STOP_MODES
                        )
                        else 0.0
                        if (
                            ala_dsel_active
                            and not state_free_polish_active
                        )
                        else active_plbfgs_kkt_rtol
                    ),
                    kkt_atol=args.plbfgs_kkt_atol,
                    min_accepted_steps=(
                        max(1, args.plbfgs_min_accepted_steps)
                        if (
                            state_free_polish_active
                            and args.state_free_polish_reparam_strategy
                            in (
                                "spring_relock",
                                "cv_relock",
                                "polyline_relock",
                            )
                        )
                        else args.plbfgs_min_accepted_steps
                    ),
                    preconditioner_bank=(
                        None
                        if (
                            state_free_polish_active
                            and args.state_free_polish_reparam_strategy
                            in (
                                "spring_relock",
                                "cv_relock",
                                "polyline_relock",
                            )
                        )
                        else ffprecon_bank
                        if use_preconditioned_exit_solver
                        else None
                    ),
                    active_images=exit_active_images,
                    direction_selection=(
                        "ala_dsel"
                        if (
                            ala_dsel_active
                            and not state_free_polish_active
                        )
                        else "legacy"
                    ),
                    max_accepted_steps=(
                        args.discrete_inner_max_steps
                        if (
                            ala_dsel_active
                            and not state_free_polish_active
                        )
                        else None
                    ),
                )
                record_ala_dsel_report(subproblem_report, outer)
                if (
                    args.solver == "state_ivm"
                    and state_free_polish_active
                    and args.ffprecon_exit_update_scheme == "red_black"
                ):
                    # Complete a red-black IVM sweep before the geometry
                    # relock.  Each parity is one legitimate fixed-alpha
                    # outer iteration: the second half freezes freshly
                    # recomputed central tangents at the accepted first-half
                    # path.  The existing transactional operator split below
                    # publishes only the final two-parity-plus-relock state.
                    first_report = subproblem_report
                    second_active_images = ~np.asarray(
                        exit_active_images, dtype=bool
                    )
                    second_tangents = central_tangents(candidate)
                    (
                        second_candidate,
                        second_energies,
                        second_forces,
                        second_used_calls,
                        second_paths,
                        second_trace_energies,
                        second_trace_forces,
                        second_counters,
                        second_report,
                    ) = solve_projected_proximal_band(
                        candidate,
                        candidate_energies,
                        candidate_forces,
                        second_tangents,
                        evaluators,
                        active_alpha,
                        active_step
                        * args.state_curve_relock_escape_trust_factor,
                        min(active_inner_calls, solve_remaining),
                        history_size=args.history,
                        curvature_memory=None,
                        armijo=args.plbfgs_armijo,
                        backtrack_factor=args.plbfgs_backtrack,
                        kkt_rtol=active_plbfgs_kkt_rtol,
                        kkt_atol=args.plbfgs_kkt_atol,
                        min_accepted_steps=max(
                            1, args.plbfgs_min_accepted_steps
                        ),
                        preconditioner_bank=None,
                        active_images=second_active_images,
                    )
                    candidate = second_candidate
                    candidate_energies = second_energies
                    candidate_forces = second_forces
                    used_calls = used_calls + second_used_calls
                    synchronized_paths.extend(second_paths)
                    synchronized_energies.extend(second_trace_energies)
                    synchronized_forces.extend(second_trace_forces)
                    synchronized_counters.extend(second_counters)
                    merged_report = {}
                    for key, first_value in first_report.items():
                        second_value = second_report[key]
                        first_array = np.asarray(first_value)
                        if first_array.shape == second_active_images.shape:
                            merged_value = first_array.copy()
                            merged_value[second_active_images] = np.asarray(
                                second_value
                            )[second_active_images]
                            merged_report[key] = merged_value
                        else:
                            merged_report[key] = second_value
                    subproblem_report = merged_report
            elif args.solver == "proximal":
                (
                    candidate,
                    candidate_energies,
                    candidate_forces,
                    used_calls,
                    synchronized_paths,
                    synchronized_energies,
                    synchronized_forces,
                    synchronized_counters,
                    subproblem_report,
                ) = solve_contractive_proximal_band(
                    path,
                    energies,
                    forces,
                    tangents,
                    evaluators,
                    active_alpha,
                    active_step,
                    min(active_inner_calls, solve_remaining),
                    contraction_limit=(
                        args.proximal_contraction_limit
                    ),
                    max_alpha_halvings=args.proximal_max_alpha_halvings,
                    kkt_rtol=active_plbfgs_kkt_rtol,
                    kkt_atol=args.plbfgs_kkt_atol,
                )
            else:
                (
                    candidate,
                    candidate_energies,
                    candidate_forces,
                    used_calls,
                    synchronized_paths,
                    synchronized_energies,
                    synchronized_forces,
                    synchronized_counters,
                ) = solve_lbfgsb_band(
                    path,
                    energies,
                    forces,
                    tangents,
                    evaluators,
                    active_alpha,
                    active_step,
                    active_lbfgsb_initial_step,
                    min(active_inner_calls, solve_remaining),
                    curvature_memory=(
                        memory
                        if history_warm_start_is_active(
                            args.lbfgsb_history_warm_start,
                            args.lbfgsb_history_warm_start_trigger_iperp,
                            diagnostic["i_perp"],
                        )
                        else None
                    ),
                )
                subproblem_report = None
        except EvaluationBudgetReached:
            break

        if (
            subproblem_report is not None
            and bool(subproblem_report.get("discrete_b040", False))
        ):
            selections = np.asarray(
                subproblem_report["direction_selection"], dtype="U8"
            )
            step_count = int(subproblem_report["joint_steps"])
            if selections.shape != (step_count,):
                raise RuntimeError(
                    "Discrete direction-selection audit length differs"
                )
            discrete_audit_outer.extend([int(outer)] * step_count)
            discrete_audit_step.extend(range(step_count))
            discrete_audit_selection.extend(selections.tolist())
            discrete_audit_h0_raw_decrease.extend(
                np.asarray(
                    subproblem_report["h0_raw_predicted_decrease"],
                    dtype=float,
                ).tolist()
            )
            discrete_audit_lbfgs_raw_decrease.extend(
                np.asarray(
                    subproblem_report["lbfgs_raw_predicted_decrease"],
                    dtype=float,
                ).tolist()
            )
            discrete_audit_h0_capped_decrease.extend(
                np.asarray(
                    subproblem_report["h0_capped_predicted_decrease"],
                    dtype=float,
                ).tolist()
            )
            discrete_audit_lbfgs_capped_decrease.extend(
                np.asarray(
                    subproblem_report[
                        "lbfgs_capped_predicted_decrease"
                    ],
                    dtype=float,
                ).tolist()
            )
            discrete_audit_h0_cap_scale.extend(
                np.asarray(
                    subproblem_report["h0_cap_scale"], dtype=float
                ).tolist()
            )
            discrete_audit_lbfgs_cap_scale.extend(
                np.asarray(
                    subproblem_report["lbfgs_cap_scale"], dtype=float
                ).tolist()
            )
            discrete_audit_history_count.extend(
                np.asarray(
                    subproblem_report["history_count"], dtype=int
                ).tolist()
            )
            discrete_audit_curvature_status.extend(
                np.asarray(
                    subproblem_report["curvature_status"], dtype="U64"
                ).tolist()
            )
            discrete_audit_global_relative_residual.append(
                float(subproblem_report["global_relative_residual"])
            )
            discrete_audit_stop_reason.append(
                str(subproblem_report["stop_reason"])
            )
            discrete_audit_all_states_evaluated = bool(
                discrete_audit_all_states_evaluated
                and subproblem_report["all_returned_states_evaluated"]
            )
        record_ala_dsel_report(subproblem_report, outer)

        if not synchronized_paths:
            stop_reason = "lbfgsb_no_progress"
            break

        if (
            subproblem_report is not None
            and "effective_alpha" in subproblem_report
        ):
            active_alpha = float(subproblem_report["effective_alpha"])
            record_alpha_value = active_alpha
            probe = np.asarray(
                subproblem_report["probe_contraction"], dtype=float
            )
            finite_probe = probe[np.isfinite(probe)]
            probe_max = (
                float(np.max(finite_probe))
                if finite_probe.size
                else np.inf
            )
            if probe_max < 0.2:
                contractive_alpha_value = active_alpha * 1.25
            elif probe_max > 0.45:
                contractive_alpha_value = active_alpha * 0.8
            else:
                contractive_alpha_value = active_alpha

        accepted_positions = current_path[1:-1].copy()
        accepted_energies = current_energies[1:-1].copy()
        accepted_forces = current_forces[1:-1].copy()
        accepted_objective = accepted_energies.copy()
        accepted_l2 = current_diag["per_image_l2"].copy()
        accepted = np.zeros(path.shape[0] - 2, dtype=bool)
        path = current_path.copy()
        energies = current_energies.copy()
        forces = current_forces.copy()
        diagnostic = current_diag
        projected_outer_mode = (
            args.reparam_mode == "state_curve_projected"
            and not state_free_polish_active
            and equidistance_diagnostics(current_path)[
                "segment_length_cv"
            ]
            <= 1e-8
        )
        if (
            projected_outer_mode
            and synchronized_paths
            and not curve_projected_direct
        ):
            # Constrained IVM step on the identical-spring manifold.  The
            # frozen-normal subproblem trials have already been evaluated and
            # therefore remain in the audit, but they are not published as
            # path iterates.  Project only the final trial geometrically and
            # evaluate that coordinate for the collective path-level test.
            for raw_counter_snapshot in synchronized_counters:
                record(
                    accepted_fraction=0.0,
                    rejected=0,
                    counter_snapshot=raw_counter_snapshot,
                )
            projected_path, _ = central_spring_force_equilibrium(
                synchronized_paths[-1]
            )
            if int(
                np.max(
                    counter_arrays(evaluators)["joint_evaluations"][
                        1:-1
                    ]
                )
            ) >= int(args.budget):
                stop_reason = "evaluation_budget"
                break
            projected_energies, projected_forces = evaluate_path(
                projected_path, evaluators
            )
            synchronized_paths = [projected_path]
            synchronized_energies = [projected_energies]
            synchronized_forces = [projected_forces]
            synchronized_counters = [
                {
                    name: values.copy()
                    for name, values in counter_arrays(evaluators).items()
                }
            ]
        transactional_operator_split = bool(
            state_free_polish_active
            and args.state_free_polish_reparam_strategy
            in ("spring_relock", "cv_relock", "polyline_relock")
        )
        transactional_cycle_candidates = None
        if transactional_operator_split:
            # The frozen-normal escape is only one half of a publishable
            # operator-split macrostep. Charge every trial at the current
            # macrostate, then pass only the final raw candidate to the
            # geometry relock below. This keeps the public convergence trace
            # free of intermediate mesh defects without hiding any oracle
            # calls or rejected work.
            transactional_cycle_candidates = [
                candidate.copy() for candidate in synchronized_paths
            ]
            for raw_counter_snapshot in synchronized_counters[:-1]:
                record(
                    accepted_fraction=0.0,
                    rejected=0,
                    counter_snapshot=raw_counter_snapshot,
                )
            synchronized_paths = [synchronized_paths[-1]]
            synchronized_energies = [synchronized_energies[-1]]
            synchronized_forces = [synchronized_forces[-1]]
            synchronized_counters = [synchronized_counters[-1]]
        for local_call, (
            trace_path,
            trace_energies,
            trace_forces,
            trace_counters,
        ) in enumerate(
            zip(
                synchronized_paths,
                synchronized_energies,
                synchronized_forces,
                synchronized_counters,
            ),
            start=1,
        ):
            trace_diag = path_diagnostics(
                trace_path, trace_energies, trace_forces[1:-1]
            )
            trace_displacement = project_normal(
                trace_path[1:-1] - current_path[1:-1],
                tangents,
            )
            trace_objective = (
                trace_energies[1:-1]
                + 0.5
                * np.sum(trace_displacement**2, axis=1)
                / active_alpha
            )
            objective_tolerance = args.energy_rtol * np.maximum(
                1.0, np.abs(accepted_objective)
            )
            objective_ok = (
                trace_objective
                <= accepted_objective + objective_tolerance
            )
            force_ok = trace_diag["per_image_l2"] <= (
                accepted_l2 * (1.0 + active_iperp_rtol) + 1e-12
            )
            if ffprecon_retired or args.solver == "state_ivm":
                # Once the path is in late exact polishing, the per-image
                # normal-force gate is not a valid local condition because
                # central-difference tangents couple neighbouring images.
                # The frozen-tangent proximal objective is local; the actual
                # merged path is protected by the strict global merit below.
                force_ok = np.ones_like(force_ok, dtype=bool)
            use_proximal_collective = (
                (
                    args.solver == "state_ivm"
                    and (
                        (
                            args.state_globalization == "proximal_only"
                            and diagnostic["fmax"]
                            <= args.state_delayed_commit_fmax
                            and not state_strict_polish_active
                        )
                        or (
                            args.state_globalization
                            == "cycle_iperp_monotone"
                            and state_free_polish_active
                        )
                    )
                )
                or (
                    ffprecon_retired
                    and args.ffprecon_exit_solver == "plbfgs"
                    and ffprecon_exit_globalization_active
                    and not ffprecon_exit_polish_active
                )
            )
            if (
                use_proximal_collective
                and local_call < len(synchronized_paths)
            ):
                # Inner evaluations solve image-local frozen-tangent
                # subproblems; they are not global path iterates.  Keep the
                # accepted band unchanged (while recording their real calls)
                # until all final subproblem minimizers are available.
                record(
                    accepted_fraction=0.0,
                    rejected=0,
                    counter_snapshot=trace_counters,
                )
                continue
            attempted = (
                used_calls > 0
                if use_proximal_collective
                else used_calls >= local_call
            )
            eligible = attempted & objective_ok & force_ok
            trace_call = int(
                np.max(
                    np.asarray(
                        trace_counters["joint_evaluations"],
                        dtype=int,
                    )[1:-1]
                )
            )
            active_global_iperp_rtol = scheduled_value(
                args.global_iperp_rtol,
                args.global_iperp_rtol_schedule,
                trace_call,
            )
            if ffprecon_retired or (
                args.solver == "state_ivm"
                and args.state_globalization
                in (
                    "iperp_monotone",
                    "delayed_iperp_monotone",
                    "cycle_iperp_monotone",
                )
            ):
                active_global_iperp_rtol = 0.0
            outer_equidistance_ceiling = None
            if (
                args.solver == "state_ivm"
                and state_reparam_accepts > 0
                and (
                    args.state_equidistance_growth_rtol >= 0.0
                    or args.state_equidistance_ceiling_factor > 1.0
                )
            ):
                outer_equidistance_ceiling = (
                    last_successful_reparam_equidistance_after
                    * args.state_equidistance_ceiling_factor
                    if (
                        args.state_equidistance_ceiling_factor > 1.0
                        and np.isfinite(
                            last_successful_reparam_equidistance_after
                        )
                    )
                    else current_diag["equidistance_integral"]
                    * (1.0 + args.state_equidistance_growth_rtol)
                )
            if projected_outer_mode:
                moved = (
                    np.linalg.norm(
                        trace_path[1:-1] - current_path[1:-1],
                        axis=1,
                    )
                    > 1e-14
                )
                collective_ok = bool(
                    np.all((~moved) | (objective_ok & force_ok))
                    and (
                        args.state_globalization
                        in ("proximal_only", "cycle_iperp_monotone")
                        or trace_diag["i_perp"]
                        <= diagnostic["i_perp"] + 1e-12
                    )
                )
                if collective_ok:
                    path = trace_path.copy()
                    energies = trace_energies.copy()
                    forces = trace_forces.copy()
                    diagnostic = trace_diag
                    accepted = moved
                else:
                    accepted = np.zeros_like(moved)
            elif outer_equidistance_ceiling is not None:
                # The synchronized frozen-normal subproblem has already paid
                # for every force evaluation in ``trace_*``.  First try that
                # collective state, then use already-evaluated contiguous
                # image blocks when the full Jacobi merge would overspend the
                # spacing slack created by the spring macrostep.  This avoids
                # the former all-or-nothing rollback while preserving both
                # global merits and makes no additional OpenMM query.
                (
                    path,
                    energies,
                    forces,
                    diagnostic,
                    accepted,
                ) = greedy_bicriteria_recovery_acceptance(
                    path,
                    energies,
                    forces,
                    trace_path,
                    trace_energies,
                    trace_forces,
                    eligible,
                    outer_equidistance_ceiling,
                    global_iperp_rtol=active_global_iperp_rtol,
                )
            elif use_proximal_collective:
                image_indices = np.flatnonzero(eligible) + 1
                path[image_indices] = trace_path[image_indices]
                energies[image_indices] = trace_energies[image_indices]
                forces[image_indices] = trace_forces[image_indices]
                accepted = eligible.copy()
                diagnostic = path_diagnostics(
                    path, energies, forces[1:-1]
                )
            else:
                (
                    path,
                    energies,
                    forces,
                    diagnostic,
                    accepted,
                ) = greedy_global_iperp_acceptance(
                    path,
                    energies,
                    forces,
                    trace_path,
                    trace_energies,
                    trace_forces,
                    eligible,
                    active_global_iperp_rtol,
                    order=(
                        "collective_block"
                        if (
                            args.solver == "state_ivm"
                            or (
                                ffprecon_retired
                                and args.ffprecon_exit_solver == "plbfgs"
                            )
                        )
                        else args.global_iperp_acceptance_order
                    ),
                )
            if (
                outer_equidistance_ceiling is not None
                and diagnostic["equidistance_integral"]
                > outer_equidistance_ceiling + 1e-14
            ):
                raise RuntimeError(
                    "Bicriteria outer merge violated equidistance ceiling"
                )
            accepted_positions = path[1:-1].copy()
            accepted_energies = energies[1:-1].copy()
            accepted_forces = forces[1:-1].copy()
            accepted_objective[accepted] = trace_objective[accepted]

            gradients = -forces[1:-1]
            accepted_l2 = diagnostic["per_image_l2"].copy()
            if not transactional_operator_split:
                record(
                    accepted_fraction=float(
                        np.count_nonzero(accepted) / (path.shape[0] - 2)
                    ),
                    rejected=int(np.count_nonzero(attempted & ~accepted)),
                    reparam_residual=0.0,
                    counter_snapshot=trace_counters,
                )

        memory.observe(path[1:-1], gradients, accept_mask=np.ones_like(accepted))
        if subproblem_report is not None:
            last_subproblem_kkt = np.asarray(
                subproblem_report["final_kkt_l2"], dtype=float
            )
            last_subproblem_relative_kkt = np.asarray(
                subproblem_report["relative_kkt"], dtype=float
            )
            last_subproblem_unconverged = int(
                np.count_nonzero(~subproblem_report["converged"])
            )
            last_subproblem_hit_call_limit = int(
                np.count_nonzero(subproblem_report["hit_call_limit"])
            )
            last_subproblem_backtracks = int(
                np.sum(subproblem_report["backtracks"])
            )
            last_subproblem_curvature_restarts = int(
                np.sum(subproblem_report["curvature_restarts"])
            )
            probe = subproblem_report.get("probe_contraction")
            if probe is None:
                last_subproblem_probe_contraction = np.nan
            else:
                finite_probe = np.asarray(probe, dtype=float)
                finite_probe = finite_probe[np.isfinite(finite_probe)]
                last_subproblem_probe_contraction = (
                    float(np.max(finite_probe))
                    if finite_probe.size
                    else np.nan
                )
            last_subproblem_alpha_halvings = int(
                subproblem_report.get("alpha_halvings", 0)
            )
        update_alpha_controller(
            current_path,
            current_energies,
            current_forces,
            path,
            energies,
            tangents,
            active_alpha,
            active_step,
        )
        if use_preconditioned_exit_solver:
            ffprecon_exit_iterations += 1
        moved_after_subproblem = np.linalg.norm(
            path[1:-1] - current_path[1:-1], axis=1
        ) > 1e-14
        if (
            args.state_curve_stall_free_polish
            and args.reparam_mode == "state_curve_projected"
            and projected_outer_mode
            and (
                curve_direct_retired
                or args.state_globalization == "cycle_iperp_monotone"
            )
            and not state_free_polish_active
        ):
            if np.any(moved_after_subproblem):
                curve_projected_stall_count = 0
            else:
                curve_projected_stall_count += 1
                if (
                    curve_projected_stall_count
                    >= args.state_curve_stall_free_polish_count
                ):
                    # The constrained identical-spring path has stopped even
                    # though its frozen-normal IVM subproblem was evaluated.
                    # Switch once, by this state condition, to an operator
                    # split: unconstrained IVM subminimization followed only
                    # when needed by a geometry-only spring correction.
                    state_free_polish_active = True
                    curve_direct_retired = True
                    memory.reset()
                    memory.observe(path[1:-1], gradients)
        if not transactional_operator_split:
            record(
                accepted_fraction=float(np.mean(moved_after_subproblem)),
                rejected=int(np.count_nonzero(~moved_after_subproblem)),
            )

        residual = 0.0
        reparameterized_and_evaluated = False
        reparam_direct_calls = 0
        reparam_accepted = np.ones(path.shape[0] - 2, dtype=bool)
        reparam_call = int(
            np.max(counter_arrays(evaluators)["joint_evaluations"][1:-1])
        )
        active_reparam_every = int(
            round(
                scheduled_value(
                    args.reparam_every,
                    args.reparam_every_schedule,
                    reparam_call,
                )
            )
        )
        if (
            active_reparam_every > 0
            and reparam_call < args.budget
            and (outer + 1) % active_reparam_every == 0
            and not (
                args.state_curve_direct_solve
                and args.reparam_mode == "state_curve_projected"
                and not state_free_polish_active
                and equidistance_diagnostics(path)[
                    "segment_length_cv"
                ]
                <= 1e-8
            )
        ):
            try:
                (
                    residual,
                    reparam_accepted,
                    reparameterized_and_evaluated,
                    reparam_direct_calls,
                ) = attempt_reparameterization(
                    active_alpha,
                    cycle_reference=(
                        current_path,
                        current_energies,
                        current_forces,
                        current_diag,
                    )
                    if (
                        state_free_polish_active
                        and args.state_free_polish_reparam_strategy
                        in (
                            "spring_relock",
                            "cv_relock",
                            "polyline_relock",
                        )
                    )
                    else None,
                    cycle_candidates=transactional_cycle_candidates,
                )
            except EvaluationBudgetReached:
                break

        if (
            reparameterized_and_evaluated
            and args.reparam_mode == "audited"
        ):
            record(
                accepted_fraction=float(np.mean(accepted)),
                rejected=int(np.count_nonzero(~accepted)),
                reparam_residual=residual,
                reparam_accepted_fraction=float(np.mean(reparam_accepted)),
                reparam_rejected=int(
                    np.count_nonzero(~reparam_accepted)
                ),
                reparam_direct_joint_evaluations=reparam_direct_calls,
            )
        ending_call = int(
            np.max(
                counter_arrays(evaluators)["joint_evaluations"][1:-1]
            )
        )
        outer_displacement = float(
            np.max(
                np.linalg.norm(
                    path[1:-1] - current_path[1:-1], axis=1
                )
            )
        )
        if ending_call == current_call and outer_displacement <= 1e-10:
            zero_cost_no_progress_outer += 1
        else:
            zero_cost_no_progress_outer = 0
        if zero_cost_no_progress_outer >= 3 and (
            args.state_kkt_stall_tighten_target > 0.0
            and active_plbfgs_kkt_rtol
            > args.state_kkt_stall_tighten_target + 1e-14
        ):
            # Standard state-based inexact-solve forcing: only after the
            # current tolerance produces a certified cached-coordinate stall,
            # tighten the KKT target without changing alpha or the objective.
            active_plbfgs_kkt_rtol = float(
                args.state_kkt_stall_tighten_target
            )
            zero_cost_no_progress_outer = 0
            memory.reset()
            memory.observe(path[1:-1], -forces[1:-1])
        elif zero_cost_no_progress_outer >= 3:
            # Cached line-search coordinates must not create an unbounded
            # zero-cost outer loop.  Three identical call levels with no
            # material coordinate motion certify a state-based numerical
            # stall; all previously evaluated trials remain charged.
            stop_reason = "cached_coordinate_stall"
            break
        outer += 1

    final_counter_snapshot = counter_arrays(evaluators)
    if (
        not history["counter_vectors"]
        or not np.array_equal(
            np.asarray(history["counter_vectors"][-1], dtype=int),
            np.asarray(
                final_counter_snapshot["joint_evaluations"], dtype=int
            ),
        )
    ):
        # A budget boundary or a no-progress solver exit may occur after a
        # real rejected query but before an ordinary iterate record. Preserve
        # the current publishable path and close the force-call ledger.
        record(
            accepted_fraction=0.0,
            rejected=0,
            counter_snapshot=final_counter_snapshot,
        )

    if not path_history or path_history[-1][0] != history["calls_per_image"][-1]:
        path_history.append((history["calls_per_image"][-1], path.copy()))

    history_calls = np.asarray(history["calls_per_image"], dtype=int)
    _, reverse_history_indices = np.unique(
        history_calls[::-1], return_index=True
    )
    history_keep = len(history_calls) - 1 - reverse_history_indices
    history_keep = history_keep[
        np.argsort(history_calls[history_keep], kind="stable")
    ]
    for name in history:
        history[name] = [history[name][index] for index in history_keep]

    stored_call_labels = np.asarray(
        [item[0] for item in path_history], dtype=int
    )
    _, reverse_path_indices = np.unique(
        stored_call_labels[::-1], return_index=True
    )
    path_keep = len(stored_call_labels) - 1 - reverse_path_indices
    path_keep = path_keep[
        np.argsort(stored_call_labels[path_keep], kind="stable")
    ]
    path_history = [path_history[index] for index in path_keep]

    configuration_signature = repr(
        (
            str(initial_path_file),
            array_sha256(initial_path),
            args.solver,
            args.alpha,
            args.alpha_mode,
            args.alpha_min,
            args.alpha_max,
            args.alpha_shrink_threshold,
            args.alpha_grow_threshold,
            args.alpha_grow_utilization,
            args.alpha_shrink_factor,
            args.alpha_grow_factor,
            args.alpha_precision_caps,
            args.alpha_relative_iperp_caps,
            args.alpha_schedule,
            args.max_step,
            args.max_step_schedule,
            args.state_trust_fraction,
            args.state_delayed_commit_fmax,
            args.state_strict_polish_fmax,
            args.state_polish_trust_fraction,
            args.state_polish_alpha_factor,
            args.state_globalization,
            args.state_cycle_iperp_window_rtol,
            args.state_equidistance_growth_rtol,
            args.state_equidistance_ceiling_factor,
            args.state_outer_history,
            args.state_outer_history_trigger_fmax,
            args.state_curve_direct_solve,
            SPRING_PROJECTION_DTAU,
            SPRING_PROJECTION_MAX_ITER,
            SPRING_PROJECTION_TOLERANCE,
            args.state_curve_direct_trial_calls,
            args.state_curve_direct_objective_acceptance,
            args.state_curve_immediate_block_relock,
            args.state_curve_direct_failure_line_search_trials,
            args.state_curve_transactional_relock_budget,
            args.state_curve_stall_free_polish,
            args.state_curve_stall_free_polish_count,
            args.state_curve_relock_escape_trust_factor,
            args.state_free_polish_reparam_strategy,
            args.state_free_polish_relative_defect_trigger,
            args.state_free_polish_relative_defect_target,
            args.history,
            args.lbfgs_initial_curvature,
            args.reparam_every,
            args.reparam_mode,
            args.reparam_every_schedule,
            args.reparam_objective_tolerance_kcal,
            args.reparam_energy_tolerance_kcal,
            args.reparam_force_rtol,
            args.reparam_initial_step_scale,
            args.reparam_spring_dtau,
            args.reparam_spring_microsteps,
            args.reparam_spring_backtrack,
            args.state_split_dtau_min,
            args.state_split_dtau_max,
            args.state_split_growth,
            args.state_split_shrink,
            args.state_split_fast_recovery,
            args.state_split_slow_recovery,
            args.reparam_guard_iperp_rtol,
            args.reparam_guard_growth_iperp_rtol,
            args.reparam_guard_growth,
            args.reparam_guard_shrink,
            args.reparam_guard_min_scale,
            args.reparam_guard_max_scale,
            args.reparam_ready_relative_iperp,
            args.reparam_ready_fmax,
            args.reparam_state_inner_calls,
            args.reparam_cv_trigger,
            args.reparam_strict_cv_trigger,
            args.reparam_target_cv,
            args.reparam_cv_contraction,
            args.reparam_proposal_metric,
            args.reparam_equidistance_contraction,
            args.reparam_equidistance_rearm_growth,
            args.reparam_retry_reduction,
            args.reparam_recovery_calls,
            args.reparam_recovery_inner_calls,
            args.reparam_recovery_merge,
            args.reparam_recovery_iperp_rtol,
            args.reparam_min_cv_improvement,
            args.reparam_min_equidistance_improvement,
            args.restart_every,
            args.restart_at_calls,
            args.hybrid_switch_call,
            args.hybrid_return_call,
            args.lbfgsb_initial_step,
            args.lbfgsb_initial_step_schedule,
            args.lbfgsb_history_warm_start,
            args.lbfgsb_history_warm_start_trigger_iperp,
            args.plbfgs_armijo,
            args.plbfgs_backtrack,
            args.plbfgs_kkt_rtol,
            args.state_kkt_stall_tighten_target,
            args.state_kkt_iperp_tighten_trigger,
            args.plbfgs_kkt_atol,
            args.plbfgs_min_accepted_steps,
            args.proximal_contraction_limit,
            args.proximal_max_alpha_halvings,
            args.ffprecon_trigger_iperp,
            args.ffprecon_exit_iperp,
            args.ffprecon_exit_solver,
            args.ffprecon_exit_alpha,
            args.ffprecon_exit_alpha_policy,
            args.ffprecon_exit_max_step,
            args.ffprecon_exit_inner_calls,
            args.ffprecon_exit_update_scheme,
            args.ffprecon_exit_globalization,
            args.ffprecon_exit_globalization_trigger_iperp,
            args.ffprecon_exit_polish_iperp,
            args.ffprecon_c_stab,
            args.ffprecon_late_c_stab,
            args.ffprecon_late_c_stab_trigger_iperp,
            args.ffprecon_collective_trigger_iperp,
            args.ffprecon_rebuild_every,
            args.ffprecon_max_step,
            args.ffprecon_inner_calls,
            args.ffprecon_raw_force_trigger_iperp,
            args.ffprecon_raw_force_max_step,
            args.ffprecon_raw_force_learning_rate,
            args.ffprecon_lbfgs_history,
            args.ffprecon_lbfgs_history_trigger_iperp,
            args.ffprecon_update_scheme,
            args.ffprecon_acceptance,
            args.ffprecon_force_rtol,
            args.ffprecon_global_iperp_rtol,
            args.ffprecon_step_growth,
            args.ffprecon_step_shrink,
            args.ffprecon_step_min,
            args.inner_calls,
            args.inner_calls_schedule,
            args.energy_rtol,
            args.iperp_rtol,
            args.iperp_rtol_schedule,
            args.global_iperp_rtol,
            args.global_iperp_rtol_schedule,
            args.global_iperp_acceptance_order,
            args.stop_iperp,
            args.stop_complete_iperp,
            args.budget,
        )
    )
    if args.frozen_subproblem != "legacy":
        extension_implementation_id = (
            ALA_DSEL_IMPLEMENTATION_ID
            if args.frozen_subproblem == "ala_dsel"
            else DISCRETE_B040_IMPLEMENTATION_ID
        )
        configuration_signature += repr(
            (
                "frozen_subproblem_extension_v1",
                args.frozen_subproblem,
                extension_implementation_id,
                args.discrete_inner_max_steps,
                args.discrete_stop_mode,
                args.discrete_residual_ratio,
                args.discrete_history,
            )
        )
        if args.frozen_subproblem == "ala_dsel":
            configuration_signature += repr(
                (
                    "ala_dsel_state_trigger_v1",
                    args.ala_dsel_trigger_iperp,
                    "latched",
                    True,
                    "selector_changes_alpha",
                    False,
                )
            )
    configuration_hash = hashlib.sha256(
        configuration_signature.encode("utf-8")
    ).hexdigest()[:12]
    tag = (
        f"ivm_{args.run_label or 'tuning'}_central_{args.solver}_"
        f"cfg{configuration_hash}_budget{args.budget}_storage{args.path_storage_mode}"
    ).replace(".", "p").replace(":", "to").replace(",", "_")
    output = root / args.output_dir / f"{tag}.npz"
    discrete_output = {}
    if args.frozen_subproblem == "discrete_b040":
        discrete_output = {
            "frozen_subproblem": np.array(args.frozen_subproblem),
            "discrete_implementation_id": np.array(
                DISCRETE_B040_IMPLEMENTATION_ID
            ),
            "discrete_inner_max_steps": np.array(
                args.discrete_inner_max_steps
            ),
            "discrete_stop_mode": np.array(args.discrete_stop_mode),
            "discrete_residual_ratio": np.array(
                args.discrete_residual_ratio
            ),
            "discrete_history": np.array(args.discrete_history),
            "discrete_audit_outer": np.asarray(
                discrete_audit_outer, dtype=int
            ),
            "discrete_audit_step": np.asarray(
                discrete_audit_step, dtype=int
            ),
            "discrete_audit_selection": np.asarray(
                discrete_audit_selection, dtype="U8"
            ),
            "discrete_audit_h0_raw_predicted_decrease": np.asarray(
                discrete_audit_h0_raw_decrease, dtype=float
            ),
            "discrete_audit_lbfgs_raw_predicted_decrease": np.asarray(
                discrete_audit_lbfgs_raw_decrease, dtype=float
            ),
            "discrete_audit_h0_capped_predicted_decrease": np.asarray(
                discrete_audit_h0_capped_decrease, dtype=float
            ),
            "discrete_audit_lbfgs_capped_predicted_decrease": np.asarray(
                discrete_audit_lbfgs_capped_decrease, dtype=float
            ),
            "discrete_audit_h0_cap_scale": np.asarray(
                discrete_audit_h0_cap_scale, dtype=float
            ),
            "discrete_audit_lbfgs_cap_scale": np.asarray(
                discrete_audit_lbfgs_cap_scale, dtype=float
            ),
            "discrete_audit_history_count": np.asarray(
                discrete_audit_history_count, dtype=int
            ),
            "discrete_audit_curvature_status": np.asarray(
                discrete_audit_curvature_status, dtype="U64"
            ),
            "discrete_audit_global_relative_residual": np.asarray(
                discrete_audit_global_relative_residual, dtype=float
            ),
            "discrete_audit_stop_reason": np.asarray(
                discrete_audit_stop_reason, dtype="U64"
            ),
            "discrete_all_returned_states_evaluated": np.array(
                discrete_audit_all_states_evaluated
            ),
            "discrete_h0_selection_count": np.array(
                discrete_audit_selection.count("H0")
            ),
            "discrete_lbfgs_selection_count": np.array(
                discrete_audit_selection.count("LBFGS")
            ),
        }
    elif args.frozen_subproblem == "ala_dsel":
        discrete_output = {
            "frozen_subproblem": np.array(args.frozen_subproblem),
            "ala_dsel_implementation_id": np.array(
                ALA_DSEL_IMPLEMENTATION_ID
            ),
            "discrete_inner_max_steps": np.array(
                args.discrete_inner_max_steps
            ),
            "discrete_stop_mode": np.array(args.discrete_stop_mode),
            "discrete_residual_ratio": np.array(
                args.discrete_residual_ratio
            ),
            "discrete_history": np.array(args.discrete_history),
            "ala_dsel_trigger_iperp": np.array(
                args.ala_dsel_trigger_iperp
            ),
            "ala_dsel_trigger_latched": np.array(True),
            "ala_dsel_selector_changes_alpha": np.array(False),
            "ala_dsel_legacy_direction_before_activation": np.array(True),
            "ala_dsel_legacy_inner_stop_before_activation": np.array(True),
            "ala_dsel_registered_stop_only_after_activation": np.array(True),
            "ala_dsel_active_final": np.array(ala_dsel_active),
            "ala_dsel_first_activation_outer": np.array(
                ala_dsel_first_activation_outer
            ),
            "ala_dsel_first_activation_calls_per_image": np.array(
                ala_dsel_first_activation_calls_per_image
            ),
            "ala_dsel_first_activation_nf": np.array(
                ala_dsel_first_activation_nf
            ),
            "ala_dsel_first_activation_iperp": np.array(
                ala_dsel_first_activation_iperp
            ),
            "ala_dsel_decisions_before_activation": np.array(
                ala_dsel_decisions_before_activation
            ),
            "ala_dsel_audit_outer": np.asarray(
                ala_dsel_audit_outer, dtype=int
            ),
            "ala_dsel_audit_selection": np.asarray(
                ala_dsel_audit_selection, dtype="U16"
            ),
            "ala_dsel_audit_h0_capped_predicted_decrease": np.asarray(
                ala_dsel_audit_h0_capped_decrease, dtype=float
            ),
            "ala_dsel_audit_lbfgs_capped_predicted_decrease": np.asarray(
                ala_dsel_audit_lbfgs_capped_decrease, dtype=float
            ),
            "ala_dsel_h0_selection_count": np.array(
                ala_dsel_audit_selection.count("H0")
            ),
            "ala_dsel_lbfgs_selection_count": np.array(
                ala_dsel_audit_selection.count("LBFGS")
            ),
            "ala_dsel_preserves_legacy_direction_candidate": np.array(True),
            "ala_dsel_preserves_outer_reparameterization": np.array(True),
            "ala_dsel_changes_registered_inner_stop": np.array(True),
            "ala_dsel_curve_selection_score_stage": np.array(
                "after_cap_and_zero_call_spring_projection"
            ),
            "ala_dsel_selector_extra_physical_candidate_evaluations": (
                np.array(0)
            ),
        }
    atomic_save_npz(
        output,
        method=np.array("IVM"),
        run_label=np.array(args.run_label or "tuning"),
        configuration_sha256=np.array(
            hashlib.sha256(
                configuration_signature.encode("utf-8")
            ).hexdigest()
        ),
        tangent_definition=np.array(TANGENT_DEFINITION),
        initial_path=initial_path,
        initial_path_file=np.array(str(initial_path_file)),
        initial_path_sha256=np.array(array_sha256(initial_path)),
        final_path=path,
        stored_path_calls=np.asarray([item[0] for item in path_history], dtype=int),
        stored_paths=np.asarray([item[1] for item in path_history]),
        path_storage_mode=np.array(args.path_storage_mode),
        path_storage_affects_online_algorithm=np.array(False),
        optimization_internal_nf=np.array(
            int(np.sum(final_counter_snapshot["joint_evaluations"][1:-1]))
        ),
        endpoint_joint_ef_cache_misses=np.array(
            int(
                final_counter_snapshot["joint_evaluations"][0]
                + final_counter_snapshot["joint_evaluations"][-1]
            )
        ),
        endpoint_inherited_joint_ef_cache_entries=np.array(
            int(
                final_counter_snapshot["inherited_cache_entries"][0]
                + final_counter_snapshot["inherited_cache_entries"][-1]
            )
        ),
        endpoint_cache_coordinate_sha256=np.array(
            endpoint_cache_audit["coordinate_sha256"]
        ),
        endpoint_cache_energy_sha256=np.array(
            endpoint_cache_audit["energy_sha256"]
        ),
        endpoint_cache_force_sha256=np.array(
            endpoint_cache_audit["force_sha256"]
        ),
        number_of_forces=np.array(int(backend.raw_joint_attempts)),
        successful_joint_ef_evaluations=np.array(
            int(backend.raw_joint_evaluations)
        ),
        solver=np.array(args.solver),
        alpha=np.array(args.alpha),
        alpha_mode=np.array(args.alpha_mode),
        alpha_min=np.array(args.alpha_min),
        alpha_max=np.array(args.alpha_max),
        alpha_shrink_threshold=np.array(args.alpha_shrink_threshold),
        alpha_grow_threshold=np.array(args.alpha_grow_threshold),
        alpha_grow_utilization=np.array(args.alpha_grow_utilization),
        alpha_shrink_factor=np.array(args.alpha_shrink_factor),
        alpha_grow_factor=np.array(args.alpha_grow_factor),
        alpha_precision_caps=np.array(args.alpha_precision_caps),
        alpha_relative_iperp_caps=np.array(
            args.alpha_relative_iperp_caps
        ),
        alpha_schedule=np.array(args.alpha_schedule),
        max_step=np.array(args.max_step),
        max_step_schedule=np.array(args.max_step_schedule),
        state_trust_fraction=np.array(args.state_trust_fraction),
        state_delayed_commit_fmax=np.array(
            args.state_delayed_commit_fmax
        ),
        state_strict_polish_fmax=np.array(
            args.state_strict_polish_fmax
        ),
        state_polish_trust_fraction=np.array(
            args.state_polish_trust_fraction
        ),
        state_polish_alpha_factor=np.array(
            args.state_polish_alpha_factor
        ),
        state_globalization=np.array(args.state_globalization),
        state_cycle_iperp_window_rtol=np.array(
            args.state_cycle_iperp_window_rtol
        ),
        state_equidistance_growth_rtol=np.array(
            args.state_equidistance_growth_rtol
        ),
        state_equidistance_ceiling_factor=np.array(
            args.state_equidistance_ceiling_factor
        ),
        state_outer_history=np.array(args.state_outer_history),
        state_outer_history_trigger_fmax=np.array(
            args.state_outer_history_trigger_fmax
        ),
        state_curve_direct_solve=np.array(
            args.state_curve_direct_solve
        ),
        state_curve_projection_method=np.array(
            SPRING_PROJECTION_METHOD
        ),
        state_curve_projection_uses_spline=np.array(False),
        state_curve_projection_openmm_calls=np.array(0),
        state_curve_projection_dtau=np.array(
            SPRING_PROJECTION_DTAU
        ),
        state_curve_projection_max_iter=np.array(
            SPRING_PROJECTION_MAX_ITER
        ),
        state_curve_projection_tolerance=np.array(
            SPRING_PROJECTION_TOLERANCE
        ),
        state_curve_direct_trial_calls=np.array(
            args.state_curve_direct_trial_calls
        ),
        state_curve_direct_objective_acceptance=np.array(
            args.state_curve_direct_objective_acceptance
        ),
        state_curve_immediate_block_relock=np.array(
            args.state_curve_immediate_block_relock
        ),
        state_curve_direct_failure_line_search_trials=np.array(
            args.state_curve_direct_failure_line_search_trials
        ),
        state_curve_transactional_relock_budget=np.array(
            args.state_curve_transactional_relock_budget
        ),
        state_curve_direct_retired_final=np.array(
            curve_direct_retired
        ),
        state_curve_stall_free_polish=np.array(
            args.state_curve_stall_free_polish
        ),
        state_curve_stall_free_polish_count=np.array(
            args.state_curve_stall_free_polish_count
        ),
        state_curve_relock_escape_trust_factor=np.array(
            args.state_curve_relock_escape_trust_factor
        ),
        state_curve_projected_stall_count_final=np.array(
            curve_projected_stall_count
        ),
        state_free_polish_relative_defect_trigger=np.array(
            args.state_free_polish_relative_defect_trigger
        ),
        state_free_polish_reparam_strategy=np.array(
            args.state_free_polish_reparam_strategy
        ),
        state_free_polish_active_images=np.array(
            args.state_free_polish_active_images
        ),
        state_free_polish_relative_defect_target=np.array(
            args.state_free_polish_relative_defect_target
        ),
        state_free_polish_target_cv=np.array(
            args.state_free_polish_target_cv
        ),
        state_free_polish_active_final=np.array(
            state_free_polish_active
        ),
        state_curve_relock_count=np.array(state_curve_relock_count),
        history_size=np.array(args.history),
        lbfgs_initial_curvature=np.array(args.lbfgs_initial_curvature),
        reparam_every=np.array(args.reparam_every),
        reparam_mode=np.array(args.reparam_mode),
        geometric_reparam_joint_evaluations=np.array(
            geometric_reparam_joint_evaluations
        ),
        reparam_every_schedule=np.array(args.reparam_every_schedule),
        reparam_objective_tolerance_kcal=np.array(
            args.reparam_objective_tolerance_kcal
        ),
        reparam_energy_tolerance_kcal=np.array(
            args.reparam_energy_tolerance_kcal
        ),
        reparam_force_rtol=np.array(args.reparam_force_rtol),
        reparam_initial_step_scale=np.array(
            args.reparam_initial_step_scale
        ),
        reparam_spring_dtau=np.array(args.reparam_spring_dtau),
        reparam_spring_microsteps=np.array(
            args.reparam_spring_microsteps
        ),
        reparam_spring_backtrack=np.array(
            args.reparam_spring_backtrack
        ),
        state_split_dtau_min=np.array(args.state_split_dtau_min),
        state_split_dtau_max=np.array(args.state_split_dtau_max),
        state_split_growth=np.array(args.state_split_growth),
        state_split_shrink=np.array(args.state_split_shrink),
        state_split_fast_recovery=np.array(
            args.state_split_fast_recovery
        ),
        state_split_slow_recovery=np.array(
            args.state_split_slow_recovery
        ),
        reparam_guard_iperp_rtol=np.array(
            args.reparam_guard_iperp_rtol
        ),
        reparam_guard_growth_iperp_rtol=np.array(
            args.reparam_guard_growth_iperp_rtol
        ),
        reparam_guard_growth=np.array(args.reparam_guard_growth),
        reparam_guard_shrink=np.array(args.reparam_guard_shrink),
        reparam_guard_min_scale=np.array(
            args.reparam_guard_min_scale
        ),
        reparam_guard_max_scale=np.array(
            args.reparam_guard_max_scale
        ),
        reparam_ready_relative_iperp=np.array(
            args.reparam_ready_relative_iperp
        ),
        reparam_ready_fmax=np.array(args.reparam_ready_fmax),
        reparam_state_inner_calls=np.array(
            args.reparam_state_inner_calls
        ),
        reparam_cv_trigger=np.array(args.reparam_cv_trigger),
        reparam_strict_cv_trigger=np.array(
            args.reparam_strict_cv_trigger
        ),
        reparam_target_cv=np.array(args.reparam_target_cv),
        reparam_cv_contraction=np.array(
            args.reparam_cv_contraction
        ),
        reparam_proposal_metric=np.array(
            args.reparam_proposal_metric
        ),
        reparam_equidistance_contraction=np.array(
            args.reparam_equidistance_contraction
        ),
        reparam_equidistance_rearm_growth=np.array(
            args.reparam_equidistance_rearm_growth
        ),
        reparam_retry_reduction=np.array(
            args.reparam_retry_reduction
        ),
        reparam_recovery_calls=np.array(
            args.reparam_recovery_calls
        ),
        reparam_recovery_inner_calls=np.array(
            args.reparam_recovery_inner_calls
        ),
        reparam_recovery_merge=np.array(args.reparam_recovery_merge),
        reparam_recovery_iperp_rtol=np.array(
            args.reparam_recovery_iperp_rtol
        ),
        reparam_min_cv_improvement=np.array(
            args.reparam_min_cv_improvement
        ),
        reparam_min_equidistance_improvement=np.array(
            args.reparam_min_equidistance_improvement
        ),
        state_reparam_attempts=np.array(state_reparam_attempts),
        state_reparam_accepts=np.array(state_reparam_accepts),
        state_reparam_recovery_joint_evaluations=np.array(
            state_reparam_recovery_joint_evaluations
        ),
        state_reparam_rearm_cv=np.array(
            last_successful_reparam_cv_before
        ),
        state_reparam_rearm_equidistance=np.array(
            last_successful_reparam_equidistance_after
        ),
        state_strict_polish_active_final=np.array(
            state_strict_polish_active
        ),
        restart_every=np.array(args.restart_every),
        restart_at_calls=np.array(args.restart_at_calls),
        hybrid_switch_call=np.array(args.hybrid_switch_call),
        hybrid_return_call=np.array(args.hybrid_return_call),
        lbfgsb_initial_step=np.array(args.lbfgsb_initial_step),
        lbfgsb_initial_step_schedule=np.array(
            args.lbfgsb_initial_step_schedule
        ),
        lbfgsb_history_warm_start=np.array(
            args.lbfgsb_history_warm_start
        ),
        lbfgsb_history_warm_start_trigger_iperp=np.array(
            args.lbfgsb_history_warm_start_trigger_iperp
        ),
        plbfgs_armijo=np.array(args.plbfgs_armijo),
        plbfgs_backtrack=np.array(args.plbfgs_backtrack),
        plbfgs_kkt_rtol=np.array(args.plbfgs_kkt_rtol),
        state_kkt_stall_tighten_target=np.array(
            args.state_kkt_stall_tighten_target
        ),
        state_kkt_iperp_tighten_trigger=np.array(
            args.state_kkt_iperp_tighten_trigger
        ),
        plbfgs_kkt_atol=np.array(args.plbfgs_kkt_atol),
        plbfgs_min_accepted_steps=np.array(
            args.plbfgs_min_accepted_steps
        ),
        proximal_contraction_limit=np.array(
            args.proximal_contraction_limit
        ),
        proximal_max_alpha_halvings=np.array(
            args.proximal_max_alpha_halvings
        ),
        ffprecon_trigger_iperp=np.array(
            args.ffprecon_trigger_iperp
        ),
        ffprecon_exit_iperp=np.array(args.ffprecon_exit_iperp),
        ffprecon_exit_solver=np.array(args.ffprecon_exit_solver),
        ffprecon_exit_alpha=np.array(args.ffprecon_exit_alpha),
        ffprecon_exit_alpha_policy=np.array(
            args.ffprecon_exit_alpha_policy
        ),
        ffprecon_exit_max_step=np.array(
            args.ffprecon_exit_max_step
        ),
        ffprecon_exit_inner_calls=np.array(
            args.ffprecon_exit_inner_calls
        ),
        ffprecon_exit_update_scheme=np.array(
            args.ffprecon_exit_update_scheme
        ),
        ffprecon_exit_globalization=np.array(
            args.ffprecon_exit_globalization
        ),
        ffprecon_exit_globalization_trigger_iperp=np.array(
            args.ffprecon_exit_globalization_trigger_iperp
        ),
        ffprecon_exit_polish_iperp=np.array(
            args.ffprecon_exit_polish_iperp
        ),
        ffprecon_c_stab=np.array(args.ffprecon_c_stab),
        ffprecon_late_c_stab=np.array(args.ffprecon_late_c_stab),
        ffprecon_late_c_stab_trigger_iperp=np.array(
            args.ffprecon_late_c_stab_trigger_iperp
        ),
        ffprecon_collective_trigger_iperp=np.array(
            args.ffprecon_collective_trigger_iperp
        ),
        ffprecon_rebuild_every=np.array(
            args.ffprecon_rebuild_every
        ),
        ffprecon_max_step=np.array(args.ffprecon_max_step),
        ffprecon_inner_calls=np.array(args.ffprecon_inner_calls),
        ffprecon_raw_force_trigger_iperp=np.array(
            args.ffprecon_raw_force_trigger_iperp
        ),
        ffprecon_raw_force_max_step=np.array(
            args.ffprecon_raw_force_max_step
        ),
        ffprecon_raw_force_learning_rate=np.array(
            args.ffprecon_raw_force_learning_rate
        ),
        ffprecon_lbfgs_history=np.array(
            args.ffprecon_lbfgs_history
        ),
        ffprecon_lbfgs_history_trigger_iperp=np.array(
            args.ffprecon_lbfgs_history_trigger_iperp
        ),
        ffprecon_update_scheme=np.array(args.ffprecon_update_scheme),
        ffprecon_acceptance=np.array(args.ffprecon_acceptance),
        ffprecon_force_rtol=np.array(args.ffprecon_force_rtol),
        ffprecon_global_iperp_rtol=np.array(
            args.ffprecon_global_iperp_rtol
        ),
        ffprecon_step_growth=np.array(args.ffprecon_step_growth),
        ffprecon_step_shrink=np.array(args.ffprecon_step_shrink),
        ffprecon_step_min=np.array(args.ffprecon_step_min),
        inner_calls=np.array(args.inner_calls),
        inner_calls_schedule=np.array(args.inner_calls_schedule),
        energy_rtol=np.array(args.energy_rtol),
        iperp_rtol=np.array(args.iperp_rtol),
        iperp_rtol_schedule=np.array(args.iperp_rtol_schedule),
        global_iperp_rtol=np.array(args.global_iperp_rtol),
        global_iperp_rtol_schedule=np.array(
            args.global_iperp_rtol_schedule
        ),
        global_iperp_acceptance_order=np.array(
            args.global_iperp_acceptance_order
        ),
        stop_iperp=np.array(args.stop_iperp),
        stop_complete_iperp=np.array(args.stop_complete_iperp),
        budget=np.array(args.budget),
        optimization_barrier_is_sparse=np.array(True),
        barrier_metric_for_figures=np.array(
            "adaptive real-OpenMM path densification: 10 then 20 "
            "subdivisions per segment; double to 40/80 when successive "
            "barriers differ by more than 0.05 kcal/mol"
        ),
        stop_reason=np.array(stop_reason),
        **discrete_output,
        **{name: np.asarray(values) for name, values in history.items()},
    )
    audit = audit_counter_consistency(backend, evaluators)
    audit.update(
        {
            "result_file": str(output),
            "endpoint_cache": endpoint_cache_audit,
            "tangent_definition": TANGENT_DEFINITION,
            "stop_reason": stop_reason,
            "reparam_mode": args.reparam_mode,
            "geometric_reparam_joint_evaluations": int(
                geometric_reparam_joint_evaluations
            ),
            "state_reparam_attempts": int(state_reparam_attempts),
            "state_reparam_accepts": int(state_reparam_accepts),
            "state_reparam_recovery_joint_evaluations": int(
                state_reparam_recovery_joint_evaluations
            ),
            "state_curve_stall_free_polish": bool(
                args.state_curve_stall_free_polish
            ),
            "state_curve_projection_method": (
                SPRING_PROJECTION_METHOD
            ),
            "state_curve_projection_uses_spline": False,
            "state_curve_projection_openmm_calls": 0,
            "state_curve_projection_dtau": (
                SPRING_PROJECTION_DTAU
            ),
            "state_curve_projection_max_iter": (
                SPRING_PROJECTION_MAX_ITER
            ),
            "state_curve_projection_tolerance": (
                SPRING_PROJECTION_TOLERANCE
            ),
            "state_curve_direct_objective_acceptance": str(
                args.state_curve_direct_objective_acceptance
            ),
            "state_curve_transactional_relock_budget": bool(
                args.state_curve_transactional_relock_budget
            ),
            "state_curve_relock_escape_trust_factor": float(
                args.state_curve_relock_escape_trust_factor
            ),
            "state_free_polish_active_final": bool(
                state_free_polish_active
            ),
            "state_curve_relock_count": int(
                state_curve_relock_count
            ),
            "state_free_polish_reparam_strategy": str(
                args.state_free_polish_reparam_strategy
            ),
            "state_free_polish_active_images": str(
                args.state_free_polish_active_images
            ),
            "state_free_polish_relative_defect_trigger": float(
                args.state_free_polish_relative_defect_trigger
            ),
            "state_free_polish_relative_defect_target": float(
                args.state_free_polish_relative_defect_target
            ),
            "state_free_polish_target_cv": float(
                args.state_free_polish_target_cv
            ),
            "state_strict_polish_active": bool(
                state_strict_polish_active
            ),
        }
    )
    if args.frozen_subproblem == "discrete_b040":
        audit.update(
            {
                "frozen_subproblem": args.frozen_subproblem,
                "discrete_implementation_id": (
                    DISCRETE_B040_IMPLEMENTATION_ID
                ),
                "discrete_inner_max_steps": int(
                    args.discrete_inner_max_steps
                ),
                "discrete_stop_mode": str(args.discrete_stop_mode),
                "discrete_residual_ratio": float(
                    args.discrete_residual_ratio
                ),
                "discrete_history": int(args.discrete_history),
                "discrete_joint_steps": len(discrete_audit_selection),
                "discrete_h0_selection_count": (
                    discrete_audit_selection.count("H0")
                ),
                "discrete_lbfgs_selection_count": (
                    discrete_audit_selection.count("LBFGS")
                ),
                "discrete_all_returned_states_evaluated": bool(
                    discrete_audit_all_states_evaluated
                ),
                "discrete_reparameterization_inside_subproblem": False,
            }
        )
        if not discrete_audit_all_states_evaluated:
            audit["consistent"] = False
    elif args.frozen_subproblem == "ala_dsel":
        audit.update(
            {
                "frozen_subproblem": args.frozen_subproblem,
                "ala_dsel_implementation_id": (
                    ALA_DSEL_IMPLEMENTATION_ID
                ),
                "discrete_inner_max_steps": int(
                    args.discrete_inner_max_steps
                ),
                "discrete_stop_mode": str(args.discrete_stop_mode),
                "discrete_residual_ratio": float(
                    args.discrete_residual_ratio
                ),
                "discrete_history": int(args.discrete_history),
                "ala_dsel_trigger_iperp": float(
                    args.ala_dsel_trigger_iperp
                ),
                "ala_dsel_trigger_latched": True,
                "ala_dsel_selector_changes_alpha": False,
                "ala_dsel_legacy_direction_before_activation": True,
                "ala_dsel_legacy_inner_stop_before_activation": True,
                "ala_dsel_registered_stop_only_after_activation": True,
                "ala_dsel_active_final": bool(ala_dsel_active),
                "ala_dsel_first_activation_outer": int(
                    ala_dsel_first_activation_outer
                ),
                "ala_dsel_first_activation_calls_per_image": int(
                    ala_dsel_first_activation_calls_per_image
                ),
                "ala_dsel_first_activation_nf": int(
                    ala_dsel_first_activation_nf
                ),
                "ala_dsel_first_activation_iperp": float(
                    ala_dsel_first_activation_iperp
                ),
                "ala_dsel_decisions_before_activation": int(
                    ala_dsel_decisions_before_activation
                ),
                "ala_dsel_direction_decisions": len(
                    ala_dsel_audit_selection
                ),
                "ala_dsel_h0_selection_count": (
                    ala_dsel_audit_selection.count("H0")
                ),
                "ala_dsel_lbfgs_selection_count": (
                    ala_dsel_audit_selection.count("LBFGS")
                ),
                "ala_dsel_preserves_legacy_direction_candidate": True,
                "ala_dsel_preserves_outer_reparameterization": True,
                "ala_dsel_changes_registered_inner_stop": True,
                "ala_dsel_curve_selection_score_stage": (
                    "after_cap_and_zero_call_spring_projection"
                ),
                "ala_dsel_selector_extra_physical_candidate_evaluations": 0,
            }
        )
        if ala_dsel_decisions_before_activation != 0:
            audit["consistent"] = False
    write_json(output.with_suffix(".audit.json"), audit)
    if not audit["consistent"]:
        raise RuntimeError(f"IVM counter audit failed: {audit}")
    print(f"Saved {output}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--initial-path-file",
        default="inputs/common_initial_path.npz",
        help=(
            "Common IDPP-derived path used by both methods. Relative paths "
            "are resolved below --root."
        ),
    )
    parser.add_argument("--run-label", default="")
    parser.add_argument(
        "--solver",
        choices=(
            "lbfgs",
            "lbfgsb",
            "plbfgs",
            "proximal",
            "hybrid",
            "hybrid_ff",
            "state_ivm",
        ),
        default="lbfgs",
    )
    parser.add_argument(
        "--frozen-subproblem",
        choices=("legacy", "discrete_b040", "ala_dsel"),
        default="legacy",
        help=(
            "Primary state_ivm frozen-path solver. discrete_b040 uses a "
            "joint direct-Cartesian FFrbk-H0/L-BFGS direction competition; "
            "Ala reparameterization and recovery remain unchanged."
        ),
    )
    parser.add_argument(
        "--discrete-inner-max-steps",
        type=int,
        default=10,
        help=(
            "Maximum number of fully evaluated joint inner updates. This "
            "is unrelated to the atom-displacement --max-step in angstrom."
        ),
    )
    parser.add_argument(
        "--discrete-stop-mode",
        choices=DISCRETE_VALID_STOP_MODES,
        default=DISCRETE_CANONICAL_STOP_MODE,
        help=(
            "r08c10 is the canonical rule: stop when R/R0 reaches the "
            "relative-residual target or after the evaluated-step cap, "
            "whichever comes first. r08 is its archived alias; c10 retains "
            "the archived cap-only ablation."
        ),
    )
    parser.add_argument(
        "--discrete-residual-ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument("--discrete-history", type=int, default=7)
    parser.add_argument(
        "--ala-dsel-trigger-iperp",
        type=float,
        default=float("inf"),
        help=(
            "Permanently activate Ala DSEL once the current accepted "
            "I_perp is at or below this state threshold. Before activation "
            "the legacy direction and legacy stop rule are used exactly. "
            "The selector never changes --alpha; inf activates at the "
            "first eligible outer iteration."
        ),
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument(
        "--alpha-mode", choices=("fixed", "adaptive"), default="fixed"
    )
    parser.add_argument("--alpha-min", type=float, default=1e-3)
    parser.add_argument("--alpha-max", type=float, default=100.0)
    parser.add_argument("--alpha-shrink-threshold", type=float, default=0.25)
    parser.add_argument("--alpha-grow-threshold", type=float, default=0.75)
    parser.add_argument("--alpha-grow-utilization", type=float, default=0.8)
    parser.add_argument("--alpha-shrink-factor", type=float, default=0.5)
    parser.add_argument("--alpha-grow-factor", type=float, default=1.5)
    parser.add_argument(
        "--alpha-precision-caps",
        default="",
        help=(
            "Optional state-based late-polish policy fmax:alpha_cap,...; "
            "for example 0.02:3,0.01:0.3. This is driven by the current "
            "physical residual, not the force-call index."
        ),
    )
    parser.add_argument(
        "--alpha-relative-iperp-caps",
        default="",
        help=(
            "Optional dimensionless late-polish policy "
            "(I_perp/I_perp_initial):alpha_cap,...; for example "
            "1e-6:0.3."
        ),
    )
    parser.add_argument("--alpha-schedule", default="")
    parser.add_argument("--max-step", type=float, required=True)
    parser.add_argument("--max-step-schedule", default="")
    parser.add_argument(
        "--state-trust-fraction",
        type=float,
        default=0.5,
        help=(
            "For state_ivm, atom-wise trust radius as this dimensionless "
            "fraction of the median RMS displacement between adjacent images; "
            "--max-step is only a safety cap."
        ),
    )
    parser.add_argument(
        "--state-delayed-commit-fmax",
        type=float,
        default=0.05,
        help=(
            "state_ivm uses strict refreshed-I_perp globalization above this "
            "physical normal fmax, then delays publication of inner "
            "frozen-tangent trials until the collective subproblem solution."
        ),
    )
    parser.add_argument(
        "--state-strict-polish-fmax",
        type=float,
        default=0.005,
        help=(
            "Permanently activate strict refreshed-I_perp polishing after "
            "this physical normal-force accuracy is reached."
        ),
    )
    parser.add_argument(
        "--state-polish-trust-fraction",
        type=float,
        default=0.5,
        help=(
            "Dimensionless RMS image-spacing trust fraction used only after "
            "strict high-accuracy polishing activates."
        ),
    )
    parser.add_argument(
        "--state-polish-alpha-factor",
        type=float,
        default=0.5,
        help=(
            "Strict-polish alpha cap as a fraction of --alpha-max."
        ),
    )
    parser.add_argument(
        "--state-globalization",
        choices=(
            "iperp_monotone",
            "delayed_iperp_monotone",
            "cycle_iperp_monotone",
            "proximal_only",
        ),
        default="iperp_monotone",
        help=(
            "Ablation-only outer acceptance policy for state_ivm. "
            "iperp_monotone applies the refreshed-I_perp guard inside the "
            "whole-band subproblem line search; delayed_iperp_monotone lets "
            "the same fixed-alpha proximal subproblem take charged internal "
            "steps but publishes only successively nonincreasing refreshed-"
            "I_perp path states; "
            "cycle_iperp_monotone stages exact proximal-objective descent, "
            "then applies the refreshed-I_perp guard only to the completed "
            "proximal-plus-geometry operator-split macrostate; "
            "proximal_only accepts exact frozen-tangent proximal-objective "
            "descent without imposing I_perp as a per-step Lyapunov merit."
        ),
    )
    parser.add_argument(
        "--state-cycle-iperp-window-rtol",
        type=float,
        default=0.0,
        help=(
            "For cycle_iperp_monotone, allow a completed CV-valid macrostate "
            "to remain within this fixed relative window above the best "
            "completed I_perp observed so far. Zero recovers strict "
            "monotonicity; all raw and relock trials remain charged."
        ),
    )
    parser.add_argument(
        "--state-equidistance-growth-rtol",
        type=float,
        default=-1.0,
        help=(
            "After a committed state spring macrostep, reject an outer IVM "
            "merge if the raw equidistance integral grows by more than this "
            "relative amount. Negative values disable the lexicographic "
            "spacing guard."
        ),
    )
    parser.add_argument(
        "--state-equidistance-ceiling-factor",
        type=float,
        default=-1.0,
        help=(
            "After a committed spring macrostep, optionally replace the "
            "per-merge spacing guard by a global raw-integral ceiling equal "
            "to this factor times the committed value. Values at or below "
            "one disable the global ceiling."
        ),
    )
    parser.add_argument(
        "--state-outer-history",
        action="store_true",
        help=(
            "For state_ivm only, reuse L-BFGS secant pairs across outer "
            "frozen-tangent subproblems. The default keeps L-BFGS history "
            "inside each fixed normal space because the central-difference "
            "normal spaces rotate between outer IVM iterations."
        ),
    )
    parser.add_argument(
        "--state-outer-history-trigger-fmax",
        type=float,
        default=-1.0,
        help=(
            "If positive, reuse cross-subproblem L-BFGS curvature only after "
            "the central-difference physical normal fmax reaches this value. "
            "A nonpositive value preserves immediate activation."
        ),
    )
    parser.add_argument(
        "--state-curve-direct-solve",
        action="store_true",
        help=(
            "For state_ivm on state_curve_projected, solve the frozen-normal "
            "proximal subproblem with a synchronized whole-band line search: "
            "each trial is first projected geometrically to identical-spring "
            "equilibrium, then evaluated once and certified by the exact "
            "proximal objective, KKT residual, and refreshed global I_perp."
        ),
    )
    parser.add_argument(
        "--state-curve-direct-trial-calls",
        type=int,
        default=15,
        help=(
            "Maximum new evaluations per image used by one synchronized "
            "equal-spring predictor. If no projected trial is accepted, the "
            "predictor is retired permanently and state_ivm continues with "
            "the original frozen-normal solve plus audited projection."
        ),
    )
    parser.add_argument(
        "--state-curve-direct-objective-acceptance",
        choices=("per_image", "path_sum"),
        default="per_image",
        help=(
            "Accept an equal-spring direct trial when either every image's "
            "quadrature contribution decreases (legacy per_image) or the "
            "discrete path-summed proximal functional decreases (path_sum). "
            "The refreshed global-I_perp guard is applied independently."
        ),
    )
    parser.add_argument(
        "--state-curve-immediate-block-relock",
        action="store_true",
        help=(
            "When the synchronized equal-spring proximal line search fails, "
            "immediately solve the largest-normal-force image block in the "
            "same frozen-tangent, fixed-alpha proximal problem. Any later "
            "optional reparameterization uses the selected geometry-only "
            "free-polish spring strategy."
        ),
    )
    parser.add_argument(
        "--state-curve-direct-failure-line-search-trials",
        type=int,
        default=0,
        help=(
            "Universal maximum number of rejected line-search trials in one "
            "synchronized projected inner iteration; zero keeps the legacy "
            "64-trial limit. This does not cap accepted inner iterations, "
            "which continue to the KKT condition or the independent "
            "--state-curve-direct-trial-calls evaluation budget."
        ),
    )
    parser.add_argument(
        "--state-curve-transactional-relock-budget",
        action="store_true",
        help=(
            "Reserve one final evaluation per active image whenever a failed "
            "equal-spring predictor enters the raw-block plus spring-relock "
            "cycle. Failed predictor and block evaluations remain charged, "
            "but a force-call boundary cannot publish a half-completed "
            "operator-split macrostate."
        ),
    )
    parser.add_argument(
        "--state-curve-stall-free-polish",
        action="store_true",
        help=(
            "After an evaluated identical-spring projected macrostep makes "
            "no coordinate progress, switch permanently to frozen-normal "
            "IVM subminimization with dimensionless geometry-only spring "
            "corrections. The trigger is path state, not force-call count."
        ),
    )
    parser.add_argument(
        "--state-curve-stall-free-polish-count",
        type=int,
        default=1,
        help=(
            "Consecutive no-progress projected macrostates required before "
            "activating operator-split free polishing."
        ),
    )
    parser.add_argument(
        "--state-curve-relock-escape-trust-factor",
        type=float,
        default=1e-4,
        help=(
            "Fixed fraction of the ordinary geometry-scaled trust radius "
            "used by the single raw frozen-normal escape subproblem before "
            "spring-force relocking. It is state based and independent of alpha "
            "and force-call count."
        ),
    )
    parser.add_argument(
        "--state-free-polish-reparam-strategy",
        choices=(
            "euler",
            "relative_defect",
            "spring_relock",
            "cv_relock",
            "polyline_relock",
        ),
        default="relative_defect",
        help=(
            "Geometry-only correction used after projected-curve stalling. "
            "euler composes a fixed number of current-central-tangent spring "
            "microsteps; relative_defect relaxes to a dimensionless spacing "
            "tolerance; spring_relock makes one free frozen-normal escape "
            "step, restores identical-spring equilibrium under the shared "
            "central-tangent spring flow, and reactivates the synchronized "
            "projected predictor; cv_relock uses the smallest such correction "
            "that restores the declared segment-length CV bound; "
            "polyline_relock instead performs the paper's coordinate-only "
            "equal-arclength sampling on the current piecewise-linear path."
        ),
    )
    parser.add_argument(
        "--state-free-polish-target-cv",
        type=float,
        default=1e-3,
        help=(
            "Maximum segment-length coefficient of variation committed by "
            "the cv_relock production reparameterization."
        ),
    )
    parser.add_argument(
        "--state-free-polish-active-images",
        choices=("largest", "all"),
        default="largest",
        help=(
            "Images included in the frozen-normal escape subproblem before "
            "spring relocking. 'largest' preserves the original greedy block "
            "policy; 'all' performs a synchronized fixed-tangent IVM step."
        ),
    )
    parser.add_argument(
        "--state-free-polish-relative-defect-trigger",
        type=float,
        default=5e-7,
        help=(
            "Dimensionless relative squared-speed defect that triggers a "
            "geometry-only identical-spring correction during free polish."
        ),
    )
    parser.add_argument(
        "--state-free-polish-relative-defect-target",
        type=float,
        default=2.5e-7,
        help=(
            "Dimensionless relative squared-speed defect reached by the "
            "smallest geometry-only spring correction."
        ),
    )
    parser.add_argument("--history", type=int, default=7)
    parser.add_argument(
        "--lbfgs-initial-curvature",
        type=float,
        default=0.0,
        help=(
            "Positive values use the notebook-consistent initial inverse "
            "Hessian 1/value and standard L-BFGS gamma; zero preserves the "
            "legacy alpha-scaled behavior."
        ),
    )
    parser.add_argument("--reparam-every", type=int, default=1)
    parser.add_argument(
        "--reparam-mode",
        choices=(
            "audited",
            "geometric",
            "geometric_guarded",
            "geometric_euler_guarded",
            "state_delayed",
            "state_metric",
            "state_split",
            "state_curve",
            "state_polyline",
            "state_curve_projected",
            "state_relative_defect_split",
        ),
        default="audited",
        help=(
            "geometric uses only central-tangent spring forces for the "
            "reparameterization decision; OpenMM is next queried only by the "
            "following optimization state evaluation. state_split repeatedly "
            "interleaves one current-central-tangent spring Euler microstep "
            "with genuine frozen-normal IVM subproblem recovery. state_curve "
            "relaxes identical springs to equilibrium using the same "
            "central-difference tangent, "
            "then requires genuine IVM recovery before commit. "
            "state_polyline instead resamples the current Cartesian "
            "polyline at equal arclength before the same genuine IVM "
            "recovery, so the geometry-only proposal lies on the old "
            "discrete curve. "
            "state_curve_projected additionally projects every evaluated "
            "normal-recovery trial back to that spring manifold before its "
            "path-level acceptance test. state_relative_defect_split is a "
            "standalone diagnostic form of free IVM subminimization followed "
            "by geometry-only spring correction at a dimensionless defect "
            "tolerance."
        ),
    )
    parser.add_argument("--reparam-every-schedule", default="")
    parser.add_argument(
        "--reparam-objective-tolerance-kcal", type=float, default=1.0
    )
    parser.add_argument(
        "--reparam-energy-tolerance-kcal", type=float, default=1.0
    )
    parser.add_argument("--reparam-force-rtol", type=float, default=0.2)
    parser.add_argument(
        "--reparam-initial-step-scale", type=float, default=1.0
    )
    parser.add_argument(
        "--reparam-spring-dtau",
        type=float,
        default=0.02,
        help=(
            "Dimensionless pseudo-time step for one current-tangent spring "
            "Euler microstep."
        ),
    )
    parser.add_argument(
        "--reparam-spring-microsteps",
        type=int,
        default=1,
        help=(
            "Number of geometry-only current-central-tangent spring Euler "
            "microsteps composed before the next joint OpenMM evaluation."
        ),
    )
    parser.add_argument(
        "--reparam-spring-backtrack", type=float, default=0.5
    )
    parser.add_argument(
        "--state-split-dtau-min",
        type=float,
        default=1e-4,
        help="Minimum current-tangent spring pseudo-time step.",
    )
    parser.add_argument(
        "--state-split-dtau-max",
        type=float,
        default=0.05,
        help="Maximum current-tangent spring pseudo-time step.",
    )
    parser.add_argument(
        "--state-split-growth",
        type=float,
        default=1.5,
        help=(
            "Grow the state-split spring step by this factor after a "
            "cheap successful IVM recovery."
        ),
    )
    parser.add_argument(
        "--state-split-shrink",
        type=float,
        default=0.5,
        help=(
            "Shrink the state-split spring step by this factor after a "
            "failed or expensive IVM recovery."
        ),
    )
    parser.add_argument(
        "--state-split-fast-recovery",
        type=int,
        default=2,
        help=(
            "A successful spring step using no more than this many recovery "
            "calls/image is considered cheap and grows pseudo-time."
        ),
    )
    parser.add_argument(
        "--state-split-slow-recovery",
        type=int,
        default=8,
        help=(
            "A successful spring step using at least this many recovery "
            "calls/image is considered expensive and shrinks pseudo-time."
        ),
    )
    parser.add_argument(
        "--reparam-guard-iperp-rtol",
        type=float,
        default=0.5,
        help=(
            "For geometric_guarded, reject and exactly roll back a spring "
            "candidate whose refreshed I_perp rises by more than this "
            "relative amount."
        ),
    )
    parser.add_argument(
        "--reparam-guard-growth-iperp-rtol",
        type=float,
        default=0.1,
        help=(
            "Grow the guarded spring scale only when its relative I_perp "
            "increase is no larger than this value."
        ),
    )
    parser.add_argument(
        "--reparam-guard-growth", type=float, default=1.25
    )
    parser.add_argument(
        "--reparam-guard-shrink", type=float, default=0.5
    )
    parser.add_argument(
        "--reparam-guard-min-scale", type=float, default=1e-4
    )
    parser.add_argument(
        "--reparam-guard-max-scale", type=float, default=0.2
    )
    parser.add_argument(
        "--reparam-ready-relative-iperp",
        type=float,
        default=1e-6,
        help=(
            "For state_delayed, allow a spring macrostep only after "
            "I_perp/I_perp(initial) falls below this dimensionless value."
        ),
    )
    parser.add_argument(
        "--reparam-ready-fmax",
        type=float,
        default=-1.0,
        help=(
            "If positive, state_delayed additionally requires physical "
            "normal fmax at or below this value before a spring proposal."
        ),
    )
    parser.add_argument(
        "--reparam-state-inner-calls",
        type=int,
        default=0,
        help=(
            "For geometric_euler_guarded, use this frozen-normal subproblem "
            "call cap after the physical --reparam-ready-fmax state is "
            "reached; zero keeps --inner-calls."
        ),
    )
    parser.add_argument(
        "--reparam-cv-trigger",
        type=float,
        default=0.10,
        help="Dimensionless segment-length CV that triggers reparameterization.",
    )
    parser.add_argument(
        "--reparam-strict-cv-trigger",
        type=float,
        default=0.15,
        help=(
            "CV hysteresis threshold for a new spring proposal after strict "
            "polishing has activated."
        ),
    )
    parser.add_argument(
        "--reparam-target-cv",
        type=float,
        default=0.05,
        help="Smallest requested segment-length CV after a spring proposal.",
    )
    parser.add_argument(
        "--reparam-cv-contraction",
        type=float,
        default=0.5,
        help=(
            "Each geometry-only proposal targets this fraction of the current "
            "segment-length CV, subject to --reparam-target-cv."
        ),
    )
    parser.add_argument(
        "--reparam-proposal-metric",
        choices=("cv", "equidistance"),
        default="cv",
        help=(
            "Geometry-only target used to size a state_metric spring move. "
            "Both choices are committed only after the raw integral and "
            "I_perp pass their state checks."
        ),
    )
    parser.add_argument(
        "--reparam-equidistance-contraction",
        type=float,
        default=0.75,
        help=(
            "For state_metric, a spring proposal targets this fractional "
            "contraction of the raw equidistance integral."
        ),
    )
    parser.add_argument(
        "--reparam-equidistance-rearm-growth",
        type=float,
        default=2.0,
        help=(
            "After a committed state_metric proposal, re-arm only when the "
            "raw equidistance integral grows by this factor."
        ),
    )
    parser.add_argument(
        "--reparam-retry-reduction",
        type=float,
        default=0.5,
        help=(
            "After an attempt, retry only after I_perp has fallen by this "
            "factor; this is a state rule rather than a call schedule."
        ),
    )
    parser.add_argument(
        "--reparam-recovery-calls",
        type=int,
        default=40,
        help=(
            "Safety cap on additional calls/image used to solve IVM "
            "subproblems before committing or rolling back a spring proposal."
        ),
    )
    parser.add_argument(
        "--reparam-recovery-inner-calls",
        type=int,
        default=10,
        help="Call cap for each frozen-tangent recovery subproblem.",
    )
    parser.add_argument(
        "--reparam-recovery-merge",
        choices=("collective", "bicriteria"),
        default="collective",
        help=(
            "How state_metric merges already-evaluated frozen-normal recovery "
            "iterates. bicriteria requires refreshed I_perp descent while "
            "preserving the spring macrostep's raw-equidistance ceiling; it "
            "performs no additional OpenMM evaluations."
        ),
    )
    parser.add_argument(
        "--reparam-recovery-iperp-rtol",
        type=float,
        default=0.0,
        help=(
            "A delayed spring proposal is committed only after I_perp returns "
            "within this relative tolerance of the pre-proposal value."
        ),
    )
    parser.add_argument(
        "--reparam-min-cv-improvement",
        type=float,
        default=0.05,
        help="Required relative CV improvement at delayed commit.",
    )
    parser.add_argument(
        "--reparam-min-equidistance-improvement",
        type=float,
        default=0.05,
        help=(
            "Required relative raw-integral improvement when committing a "
            "state_metric spring proposal."
        ),
    )
    parser.add_argument("--restart-every", type=int, default=0)
    parser.add_argument("--restart-at-calls", default="")
    parser.add_argument("--hybrid-switch-call", type=int, default=0)
    parser.add_argument(
        "--hybrid-return-call",
        type=int,
        default=0,
        help=(
            "For the hybrid solver, return from per-image L-BFGS-B to "
            "projected L-BFGS at this max-call label; zero disables."
        ),
    )
    parser.add_argument("--lbfgsb-initial-step", type=float, default=0.0)
    parser.add_argument("--lbfgsb-initial-step-schedule", default="")
    parser.add_argument(
        "--lbfgsb-history-warm-start",
        action="store_true",
        help=(
            "Use run-accumulated projected L-BFGS s,y pairs to precondition "
            "the initial displacement of every frozen-tangent L-BFGS-B "
            "subproblem."
        ),
    )
    parser.add_argument(
        "--lbfgsb-history-warm-start-trigger-iperp",
        type=float,
        default=-1.0,
        help=(
            "If positive, inject run-accumulated L-BFGS curvature only after "
            "raw I_perp falls below this state threshold."
        ),
    )
    parser.add_argument("--plbfgs-armijo", type=float, default=1e-4)
    parser.add_argument("--plbfgs-backtrack", type=float, default=0.5)
    parser.add_argument("--plbfgs-kkt-rtol", type=float, default=0.1)
    parser.add_argument(
        "--state-kkt-stall-tighten-target",
        type=float,
        default=-1.0,
        help=(
            "For state_ivm, tighten the active relative KKT tolerance to "
            "this value only after the cached-coordinate stall detector "
            "fires. Nonpositive values disable state-based tightening."
        ),
    )
    parser.add_argument(
        "--state-kkt-iperp-tighten-trigger",
        type=float,
        default=-1.0,
        help=(
            "If positive, tighten to --state-kkt-stall-tighten-target "
            "when accepted raw I_perp first reaches this state threshold. "
            "Nonpositive values disable residual-driven tightening."
        ),
    )
    parser.add_argument("--plbfgs-kkt-atol", type=float, default=1e-8)
    parser.add_argument(
        "--plbfgs-min-accepted-steps",
        type=int,
        default=0,
        help=(
            "Require at least this many accepted L-BFGS steps before a "
            "frozen proximal subproblem may stop on its KKT tolerance. "
            "The same value is applied to every image and every alpha."
        ),
    )
    parser.add_argument(
        "--proximal-contraction-limit", type=float, default=0.5
    )
    parser.add_argument(
        "--proximal-max-alpha-halvings", type=int, default=16
    )
    parser.add_argument(
        "--stop-iperp",
        type=float,
        default=-1.0,
        help=(
            "Stop after an accepted path reaches this raw I_perp target. "
            "Nonpositive values disable residual-based convergence stopping."
        ),
    )
    parser.add_argument(
        "--stop-complete-iperp",
        type=float,
        default=-1.0,
        help=(
            "Stop only when an accepted, fully evaluated reparameterization "
            "macrostate reaches this raw I_perp target. Nonpositive values "
            "disable this operator-split completion rule."
        ),
    )
    parser.add_argument(
        "--ffprecon-trigger-iperp",
        type=float,
        default=-1.0,
        help=(
            "For hybrid_ff, activate the bonded force-field block "
            "preconditioner when raw I_perp falls below this value."
        ),
    )
    parser.add_argument(
        "--ffprecon-exit-iperp",
        type=float,
        default=-1.0,
        help=(
            "If positive, permanently leave the approximate force-field "
            "preconditioned stage after raw I_perp reaches this state value, "
            "then resume the exact frozen-tangent subproblem solver."
        ),
    )
    parser.add_argument(
        "--ffprecon-exit-solver",
        choices=("lbfgsb", "plbfgs"),
        default="lbfgsb",
        help=(
            "Late solver after --ffprecon-exit-iperp. plbfgs solves each "
            "frozen-tangent proximal subproblem with within-subproblem "
            "L-BFGS and the shifted force-field inverse Hessian."
        ),
    )
    parser.add_argument(
        "--ffprecon-exit-alpha",
        type=float,
        default=-1.0,
        help=(
            "Fixed initial proximal alpha for the late exit solver; "
            "non-positive keeps the main alpha controller."
        ),
    )
    parser.add_argument(
        "--ffprecon-exit-alpha-policy",
        default="",
        help=(
            "State rule threshold:alpha,... keyed by decreasing raw I_perp."
        ),
    )
    parser.add_argument(
        "--ffprecon-exit-max-step",
        type=float,
        default=-1.0,
        help=(
            "Maximum per-atom displacement in each late frozen-tangent "
            "subproblem step."
        ),
    )
    parser.add_argument(
        "--ffprecon-exit-inner-calls",
        type=int,
        default=0,
        help=(
            "Per-image call cap for each late frozen-tangent subproblem; "
            "zero keeps the common inner-calls schedule."
        ),
    )
    parser.add_argument(
        "--ffprecon-exit-update-scheme",
        choices=("all", "red_black"),
        default="all",
        help=(
            "Late exact-subproblem update scheme. red_black solves one "
            "independent parity of internal images, refreshes the central "
            "tangents on the accepted path, then solves the other parity."
        ),
    )
    parser.add_argument(
        "--ffprecon-exit-globalization",
        choices=("iperp_monotone", "proximal_collective"),
        default="iperp_monotone",
        help=(
            "Late exact-subproblem globalization. proximal_collective "
            "accepts every image whose frozen-tangent proximal objective "
            "decreased, and deliberately does not treat I_perp as a "
            "per-iteration Lyapunov function."
        ),
    )
    parser.add_argument(
        "--ffprecon-exit-globalization-trigger-iperp",
        type=float,
        default=-1.0,
        help=(
            "Permanently activate proximal_collective after raw I_perp "
            "reaches this state value; non-positive activates immediately "
            "upon entering the late exact-subproblem stage."
        ),
    )
    parser.add_argument(
        "--ffprecon-exit-polish-iperp",
        type=float,
        default=-1.0,
        help=(
            "Permanently leave proximal_collective and restore strict "
            "global I_perp acceptance after reaching this raw I_perp. "
            "Use with a smaller state-based exit-alpha policy for protected "
            "late polishing."
        ),
    )
    parser.add_argument("--ffprecon-c-stab", type=float, default=1.0)
    parser.add_argument(
        "--ffprecon-late-c-stab",
        type=float,
        default=-1.0,
        help=(
            "Optional stronger FF stability term used after the raw I_perp "
            "state threshold below."
        ),
    )
    parser.add_argument(
        "--ffprecon-late-c-stab-trigger-iperp",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--ffprecon-collective-trigger-iperp",
        type=float,
        default=-1.0,
        help=(
            "Activate central-tangent block acceptance once raw I_perp is "
            "at or below this value; disabled when non-positive"
        ),
    )
    parser.add_argument(
        "--ffprecon-rebuild-every", type=int, default=10
    )
    parser.add_argument("--ffprecon-max-step", type=float, default=0.1)
    parser.add_argument(
        "--ffprecon-inner-calls",
        type=int,
        default=1,
        help=(
            "Number of audited proximal steps taken in one frozen central-"
            "tangent normal subproblem before refreshing its tangent."
        ),
    )
    parser.add_argument(
        "--ffprecon-raw-force-trigger-iperp",
        type=float,
        default=-1.0,
        help=(
            "If positive, switch from the bonded force-field inverse Hessian "
            "to a small direct normal-force step below this raw I_perp."
        ),
    )
    parser.add_argument(
        "--ffprecon-raw-force-max-step",
        type=float,
        default=3e-5,
    )
    parser.add_argument(
        "--ffprecon-raw-force-learning-rate",
        type=float,
        default=0.0,
        help=(
            "If positive, use displacement = learning_rate * F_perp in the "
            "raw-force polishing stage, preserving relative force magnitude."
        ),
    )
    parser.add_argument(
        "--ffprecon-lbfgs-history",
        action="store_true",
        help=(
            "Warm up the force-field inverse Hessian with projected L-BFGS "
            "curvature accumulated after the state-triggered solver switch. "
            "No history is shared between experiments."
        ),
    )
    parser.add_argument(
        "--ffprecon-lbfgs-history-trigger-iperp",
        type=float,
        default=-1.0,
        help=(
            "Begin a fresh run-local physical-curvature L-BFGS history once "
            "raw I_perp is at or below this value; a non-positive value "
            "activates immediately when --ffprecon-lbfgs-history is set."
        ),
    )
    parser.add_argument(
        "--ffprecon-update-scheme",
        choices=("all", "red_black"),
        default="all",
        help=(
            "red_black alternates even/odd internal-image subproblems and "
            "recomputes central-difference tangents between the two colors."
        ),
    )
    parser.add_argument(
        "--ffprecon-acceptance",
        choices=("force_monotone", "proximal_merit"),
        default="force_monotone",
        help=(
            "Accept FF-preconditioned image proposals either with the legacy "
            "per-image normal-force gate or with the frozen-tangent IVP "
            "proximal objective followed by global I_perp merit acceptance."
        ),
    )
    parser.add_argument("--ffprecon-force-rtol", type=float, default=0.0)
    parser.add_argument(
        "--ffprecon-global-iperp-rtol", type=float, default=0.0
    )
    parser.add_argument("--ffprecon-step-growth", type=float, default=1.2)
    parser.add_argument("--ffprecon-step-shrink", type=float, default=0.5)
    parser.add_argument("--ffprecon-step-min", type=float, default=1e-6)
    parser.add_argument("--inner-calls", type=int, default=5)
    parser.add_argument("--inner-calls-schedule", default="")
    parser.add_argument("--energy-rtol", type=float, default=1e-10)
    parser.add_argument("--iperp-rtol", type=float, default=0.02)
    parser.add_argument("--iperp-rtol-schedule", default="")
    parser.add_argument("--global-iperp-rtol", type=float, default=1e9)
    parser.add_argument("--global-iperp-rtol-schedule", default="")
    parser.add_argument(
        "--global-iperp-acceptance-order",
        choices=("local_ratio", "global_best", "collective_first"),
        default="local_ratio",
    )
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        default="runs/ivm",
        help=(
            "Root-relative result directory. This changes only file routing, "
            "not the IVM configuration signature."
        ),
    )
    parser.add_argument("--save-all-paths", action="store_true")
    parser.add_argument(
        "--path-storage-mode",
        choices=PATH_STORAGE_MODES,
        default="legacy",
        help=(
            "legacy preserves the upstream recorder; committed retains only "
            "initial, fully reparameterization-committed, and final paths. "
            "This option changes stored history only."
        ),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
