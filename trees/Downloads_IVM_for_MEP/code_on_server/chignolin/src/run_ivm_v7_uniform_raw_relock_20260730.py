#!/usr/bin/env python3
"""Run audited Chignolin IVM with KKT solve then geometric relocking."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from chignolin_core import (
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
    project_normal,
    spring_reparameterize,
    write_json,
)

KKT_COMPLETION_PUBLISH_POLICY_VERSION = (
    "v7_uniform_raw_kkt_then_unconditional_geometry_relock"
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
        and reparam_strategy == "hermite_relock"
    )
    return max(0, int(remaining_after_direct) - reserve), reserve


def direct_predictor_requires_raw_completion(
    accepted_steps,
    converged,
    require_kkt_completion: bool,
) -> bool:
    """Decide whether an unfinished synchronized predictor may be published."""
    accepted_any = bool(np.any(np.asarray(accepted_steps, dtype=int)))
    kkt_complete = bool(np.all(np.asarray(converged, dtype=bool)))
    return bool(
        not accepted_any
        or (bool(require_kkt_completion) and not kkt_complete)
    )


def kkt_completion_image_mask(
    per_image_l2,
    require_kkt_completion: bool,
) -> np.ndarray:
    """Select the full band for KKT completion, otherwise the worst image."""
    per_image_l2 = np.asarray(per_image_l2, dtype=float)
    if per_image_l2.ndim != 1 or not per_image_l2.size:
        raise ValueError("per_image_l2 must be a nonempty vector")
    if require_kkt_completion:
        return np.ones(per_image_l2.size, dtype=bool)
    output = np.zeros(per_image_l2.size, dtype=bool)
    output[int(np.argmax(per_image_l2))] = True
    return output


def completed_subproblem_is_publishable(
    subproblem_report,
    require_kkt_completion: bool,
    completion_phase_active: bool,
) -> bool:
    """Only a bandwise KKT-complete correction may become an IVM iterate."""
    if not require_kkt_completion or not completion_phase_active:
        return True
    if subproblem_report is None:
        return False
    return bool(
        np.all(np.asarray(subproblem_report["converged"], dtype=bool))
    )


def relocked_macrostate_is_publishable(
    uniform_raw_subproblem: bool,
    moved,
    objective_ok,
    force_ok,
    state_globalization: str,
    trial_i_perp: float,
    current_i_perp: float,
) -> bool:
    """Separate the certified IVM solve from optional mesh relabeling.

    In the V7 uniform mode, the original frozen-normal proximal subproblem
    has already passed its bandwise KKT rule before this function is called.
    Central-Hermite relocking is then a deterministic, energy/force-free
    change of mesh labels on that completed path.  Reapplying the nodewise
    proximal objective to the relocked coordinates conflates these two
    operators and can create a deterministic cached-rejection fixed point.
    Therefore the geometry-only relock is published unconditionally in V7.
    Legacy modes retain their existing objective/force/global-merit checks.
    """
    moved = np.asarray(moved, dtype=bool)
    objective_ok = np.asarray(objective_ok, dtype=bool)
    force_ok = np.asarray(force_ok, dtype=bool)
    if uniform_raw_subproblem:
        return True
    return bool(
        np.all((~moved) | (objective_ok & force_ok))
        and (
            state_globalization == "proximal_only"
            or float(trial_i_perp) <= float(current_i_perp) + 1e-12
        )
    )


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


def central_hermite_spring_equilibrium(
    path,
    dense_points=20001,
):
    """Place images at spring equilibrium on one frozen Hermite curve.

    The curve interpolates the current images.  Its internal derivatives use
    exactly the shared normalized central-difference tangent; endpoint
    derivatives are one-sided because the fixed endpoints have no central
    stencil.  Positive scalar parameter increments are optimized until the
    adjacent Cartesian chord lengths are equal: this is the zero-residual
    state of identical nearest-neighbour springs constrained to the frozen
    curve, and it matches the monitored discrete integral exactly.  This
    construction is geometry-only and calls neither OpenMM energy nor force.
    """
    from scipy.interpolate import CubicHermiteSpline
    from scipy.optimize import least_squares

    path = np.asarray(path, dtype=float)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    parameter = np.concatenate(([0.0], np.cumsum(lengths)))
    derivatives = np.empty_like(path)
    derivatives[1:-1] = central_tangents(path)
    derivatives[0] = (path[1] - path[0]) / lengths[0]
    derivatives[-1] = (path[-1] - path[-2]) / lengths[-1]
    spline = CubicHermiteSpline(
        parameter,
        path,
        derivatives,
        axis=0,
    )
    span = float(parameter[-1] - parameter[0])
    initial_increments = np.diff(parameter)
    initial_logits = np.log(
        initial_increments[:-1] / initial_increments[-1]
    )

    def positions_from_logits(logits):
        logits = np.concatenate((np.asarray(logits, dtype=float), [0.0]))
        logits -= np.max(logits)
        increments = np.exp(logits)
        increments *= span / np.sum(increments)
        node_parameter = parameter[0] + np.concatenate(
            ([0.0], np.cumsum(increments))
        )
        return np.asarray(spline(node_parameter), dtype=float)

    def spring_residual(logits):
        candidate = positions_from_logits(logits)
        candidate_lengths = np.linalg.norm(
            np.diff(candidate, axis=0), axis=1
        )
        return candidate_lengths[1:] - candidate_lengths[:-1]

    solution = least_squares(
        spring_residual,
        initial_logits,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=max(200, int(dense_points) // 10),
    )
    result = positions_from_logits(solution.x)
    result[0] = path[0]
    result[-1] = path[-1]
    residual = float(
        np.max(
            np.abs(
                np.diff(
                    np.linalg.norm(np.diff(result, axis=0), axis=1)
                )
            )
        )
    )
    return result, residual


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
            direction = clip_max_atomic_displacement(direction, max_step)
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
):
    """Solve one frozen-normal IVM problem on the equal-spring gauge.

    A trial is first constructed in the frozen central-difference normal
    space.  It is then mapped, without an OpenMM query, to identical-spring
    equilibrium on the frozen central-Hermite curve.  Only that final
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

    def used_calls():
        return (
            counter_arrays(evaluators)["joint_evaluations"][1:-1]
            - start_calls
        )

    while True:
        residual_norm = np.linalg.norm(residual_gradient, axis=1)
        if (
            np.all(residual_norm <= target_kkt)
            and np.all(accepted_steps >= int(min_accepted_steps))
        ):
            break
        if int(np.max(used_calls())) >= int(max_new_calls):
            hit_call_limit = residual_norm > target_kkt
            break

        directions = np.zeros_like(residual_gradient)
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
            direction = clip_max_atomic_displacement(direction, max_step)
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
            trial_path, _ = central_hermite_spring_equilibrium(raw_trial)

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
            accepted = bool(np.all(armijo_ok) and global_ok)

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
    if args.state_uniform_raw_exact_subproblem:
        if args.solver != "state_ivm":
            raise ValueError(
                "uniform raw exact subproblems require --solver state_ivm"
            )
        if args.reparam_mode != "state_curve_projected":
            raise ValueError(
                "uniform raw exact subproblems require "
                "--reparam-mode state_curve_projected"
            )
        if args.state_curve_direct_solve:
            raise ValueError(
                "uniform raw exact subproblems deliberately exclude the "
                "synchronized equal-spring predictor"
            )
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
        root / "inputs" / "chignolin_clean.pdb",
        platform_preference=("CPU",),
        precision="double",
        label="ivm_optimization",
    )
    evaluators = make_evaluators(backend, internal_limit=args.budget)
    ffprecon_bank = (
        ForceFieldPreconditionerBank(
            backend,
            root / "inputs" / "chignolin_clean.pdb",
            c_stab=args.ffprecon_c_stab,
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
    state_curve_kkt_completion_rollbacks = 0
    state_curve_kkt_completion_incomplete_final = False
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
        if args.save_all_paths or call in {1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500}:
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

    def attempt_reparameterization(active_alpha):
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
        if state_free_polish_active:
            before = equidistance_diagnostics(path)
            if (
                args.state_free_polish_reparam_strategy
                == "hermite_relock"
            ):
                correction_due = (
                    before["segment_length_cv"] > 1e-12
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
                return (
                    0.0,
                    np.zeros(path.shape[0] - 2, dtype=bool),
                    False,
                    0,
                )
            state_reparam_attempts += 1
            if (
                args.state_free_polish_reparam_strategy
                == "hermite_relock"
            ):
                (
                    working_path,
                    residual,
                ) = central_hermite_spring_equilibrium(path)
                proposed_spacing = equidistance_diagnostics(
                    working_path
                )
                proposal_scale = 1.0
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
            # This is an optional parameterization step, not a physical
            # descent step.  Its geometry is selected without OpenMM and is
            # therefore committed without an I_perp acceptance guard; the
            # refreshed joint evaluation is fully charged to the run.
            path = working_path
            energies = working_energies
            forces = working_forces
            diagnostic = working_diagnostic
            gradients = -forces[1:-1]
            memory.reset()
            memory.observe(path[1:-1], gradients)
            state_reparam_accepts += 1
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
            if (
                args.state_free_polish_reparam_strategy
                == "hermite_relock"
            ):
                # One unconstrained frozen-normal IVM macrostep has escaped
                # the discrete equal-spring fixed point.  Relock exactly and
                # reactivate the same synchronized equal-spring predictor.
                # This is a state-triggered cycle, not an alpha/call schedule.
                state_free_polish_active = False
                curve_direct_retired = False
                curve_projected_stall_count = 0
                state_curve_relock_count += 1
            return residual, committed, True, direct_calls
        if args.reparam_mode in {
            "state_delayed",
            "state_metric",
            "state_split",
            "state_curve",
            "state_curve_projected",
        }:
            split_mode = args.reparam_mode == "state_split"
            curve_mode = args.reparam_mode in {
                "state_curve",
                "state_curve_projected",
            }
            projected_curve_mode = (
                args.reparam_mode == "state_curve_projected"
            )
            metric_mode = args.reparam_mode in {
                "state_metric",
                "state_split",
                "state_curve",
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
                working_path, residual = central_hermite_spring_equilibrium(
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
                    raw_before = int(backend.raw_joint_evaluations)
                    ffprecon_bank.build(working_path)
                    if int(backend.raw_joint_evaluations) != raw_before:
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
                    kkt_rtol=args.plbfgs_kkt_rtol,
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
                    # central-Hermite solve.  OpenMM is then queried once at
                    # that final projected coordinate.  The already evaluated
                    # unprojected trial remains in the force-call audit.
                    projected_path, _ = (
                        central_hermite_spring_equilibrium(candidate_path)
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
    while True:
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
                raw_before = int(backend.raw_joint_evaluations)
                ffprecon_bank.build(path)
                if int(backend.raw_joint_evaluations) != raw_before:
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
            if args.state_uniform_raw_exact_subproblem:
                # Every alpha solves the same original frozen-normal
                # proximal problem.  Reserve one evaluation per image for
                # the optional, purely geometric central-Hermite spring
                # relock so a finite budget cannot expose an unrelocked
                # macrostate.
                if remaining <= 1:
                    stop_reason = "complete_uniform_raw_cycle_budget"
                    break
                solve_remaining = remaining - 1
            elif state_free_polish_active:
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
                == "hermite_relock"
            ):
                if args.state_curve_require_kkt_completion:
                    # Complete the original frozen-normal IVM subproblem for
                    # the whole band before the optional geometry-only relock.
                    # This common rule deliberately prevents a difficult
                    # large-alpha subproblem from obtaining extra outer
                    # tangent refreshes merely because an inner solve stopped
                    # early.
                    exit_active_images[:] = True
                else:
                    # Escape the discrete equal-spring fixed point by one
                    # greedy block subminimization.  Moving only the image with
                    # largest current physical normal force limits rotation of
                    # adjacent central-difference tangents.  The choice is
                    # state based and independent of alpha and force-call
                    # count.
                    exit_active_images = kkt_completion_image_mask(
                        diagnostic["per_image_l2"], False
                    )
            if (
                use_preconditioned_exit_solver
                and args.ffprecon_exit_update_scheme == "red_black"
            ):
                exit_active_images = ffprecon_update_mask(
                    path.shape[0] - 2,
                    ffprecon_exit_iterations,
                    "red_black",
                )
            if use_preconditioned_exit_solver and (
                not args.state_uniform_raw_exact_subproblem
                or args.state_uniform_raw_use_preconditioner
            ):
                raw_before = int(backend.raw_joint_evaluations)
                ffprecon_bank.build(path)
                if int(backend.raw_joint_evaluations) != raw_before:
                    raise RuntimeError(
                        "FF preconditioner construction called OpenMM"
                    )
            curve_projected_direct = bool(
                args.state_curve_direct_solve
                and not curve_direct_retired
                and not state_free_polish_active
                and args.solver == "state_ivm"
                and args.reparam_mode == "state_curve_projected"
                and equidistance_diagnostics(path)[
                    "segment_length_cv"
                ]
                <= 1e-8
            )
            if curve_projected_direct:
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
                    kkt_rtol=args.plbfgs_kkt_rtol,
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
                        args.state_globalization == "iperp_monotone"
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
                )
                require_raw_completion = (
                    direct_predictor_requires_raw_completion(
                        subproblem_report["accepted_steps"],
                        subproblem_report["converged"],
                        args.state_curve_require_kkt_completion,
                    )
                )
                if require_raw_completion:
                    # A failed equal-spring line search is a state trigger,
                    # not a call schedule.  Permanently retire the fast
                    # projected predictor and fall back to the original
                    # frozen-normal subproblem followed by a separately
                    # audited spring projection.  Failed predictor queries
                    # remain in the synchronized trace and in every counter.
                    # With --state-curve-require-kkt-completion, an accepted
                    # but KKT-incomplete synchronized proposal is likewise
                    # treated as an unfinished inner solve: it is not used as
                    # the reference of a new proximal problem.
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
                        block_active_images = kkt_completion_image_mask(
                            diagnostic["per_image_l2"],
                            args.state_curve_require_kkt_completion,
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
                            (
                                active_step
                                if args.state_curve_require_kkt_completion
                                else active_step
                                * args
                                .state_curve_relock_escape_trust_factor
                            ),
                            (
                                fallback_budget
                                if args.state_curve_require_kkt_completion
                                else min(
                                    active_inner_calls,
                                    fallback_budget,
                                )
                            ),
                            history_size=args.history,
                            curvature_memory=None,
                            armijo=args.plbfgs_armijo,
                            backtrack_factor=args.plbfgs_backtrack,
                            kkt_rtol=args.plbfgs_kkt_rtol,
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
                        # reserved for a complete Hermite relock.  If there
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
                            kkt_rtol=args.plbfgs_kkt_rtol,
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
                        if args.state_curve_require_kkt_completion
                        else active_step
                        * args.state_curve_relock_escape_trust_factor
                        if (
                            state_free_polish_active
                            and args.state_free_polish_reparam_strategy
                            == "hermite_relock"
                        )
                        else active_step
                    ),
                    (
                        solve_remaining
                        if (
                            args.state_uniform_raw_exact_subproblem
                            or (
                                args.state_curve_require_kkt_completion
                                and state_free_polish_active
                                and args.state_free_polish_reparam_strategy
                                == "hermite_relock"
                            )
                        )
                        else min(active_inner_calls, solve_remaining)
                    ),
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
                    kkt_rtol=args.plbfgs_kkt_rtol,
                    kkt_atol=args.plbfgs_kkt_atol,
                    min_accepted_steps=(
                        max(1, args.plbfgs_min_accepted_steps)
                        if (
                            state_free_polish_active
                            and args.state_free_polish_reparam_strategy
                            == "hermite_relock"
                        )
                        else args.plbfgs_min_accepted_steps
                    ),
                    preconditioner_bank=(
                        None
                        if (
                            (
                                state_free_polish_active
                                and args.state_free_polish_reparam_strategy
                                == "hermite_relock"
                            )
                            or (
                                args.state_uniform_raw_exact_subproblem
                                and not args
                                .state_uniform_raw_use_preconditioner
                            )
                        )
                        else ffprecon_bank
                        if use_preconditioned_exit_solver
                        else None
                    ),
                    active_images=exit_active_images,
                )
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
                    kkt_rtol=args.plbfgs_kkt_rtol,
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

        if not synchronized_paths:
            stop_reason = "lbfgsb_no_progress"
            break

        kkt_completion_publishable = completed_subproblem_is_publishable(
            subproblem_report,
            (
                args.state_curve_require_kkt_completion
                or args.state_uniform_raw_exact_subproblem
            ),
            (
                state_free_polish_active
                or args.state_uniform_raw_exact_subproblem
            ),
        )
        if not kkt_completion_publishable:
            # Every force/energy evaluation of the unfinished subproblem
            # remains in the counters and raw trace.  However, exact IVM
            # changes the public path only after the frozen-normal proximal
            # subproblem has been completed.  At a finite budget boundary,
            # repeat the last complete macrostate at each audited counter
            # snapshot instead of exposing an inner L-BFGS iterate.
            state_curve_kkt_completion_rollbacks += 1
            state_curve_kkt_completion_incomplete_final = True
            synchronized_paths = [
                current_path.copy() for _ in synchronized_counters
            ]
            synchronized_energies = [
                current_energies.copy() for _ in synchronized_counters
            ]
            synchronized_forces = [
                current_forces.copy() for _ in synchronized_counters
            ]
            candidate = current_path.copy()
            candidate_energies = current_energies.copy()
            candidate_forces = current_forces.copy()

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
            and kkt_completion_publishable
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
            projected_path, _ = central_hermite_spring_equilibrium(
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
            reparam_backend_before = int(backend.raw_joint_evaluations)
            projected_energies, projected_forces = evaluate_path(
                projected_path, evaluators
            )
            geometric_reparam_joint_evaluations += int(
                backend.raw_joint_evaluations - reparam_backend_before
            )
            state_curve_relock_count += 1
            synchronized_paths = [projected_path]
            synchronized_energies = [projected_energies]
            synchronized_forces = [projected_forces]
            synchronized_counters = [
                {
                    name: values.copy()
                    for name, values in counter_arrays(evaluators).items()
                }
            ]
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
                    and diagnostic["fmax"]
                    <= args.state_delayed_commit_fmax
                    and not state_strict_polish_active
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
                and args.state_globalization == "iperp_monotone"
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
                collective_ok = relocked_macrostate_is_publishable(
                    args.state_uniform_raw_exact_subproblem,
                    moved,
                    objective_ok,
                    force_ok,
                    args.state_globalization,
                    trace_diag["i_perp"],
                    current_diag["i_perp"],
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
            and curve_direct_retired
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
                    memory.reset()
                    memory.observe(path[1:-1], gradients)
        record(
            accepted_fraction=float(np.mean(moved_after_subproblem)),
            rejected=int(np.count_nonzero(~moved_after_subproblem)),
        )

        if not kkt_completion_publishable:
            # No half-completed proximal solve or geometry relock is carried
            # across the reporting boundary.  The run terminates at the last
            # complete IVM macrostate, with all failed/unfinished work still
            # fully charged on the force-call axis.
            state_free_polish_active = False
            curve_direct_retired = False
            curve_projected_stall_count = 0
            stop_reason = "kkt_completion_incomplete_budget"
            break

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
                (
                    args.state_curve_direct_solve
                    or args.state_uniform_raw_exact_subproblem
                )
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
                ) = attempt_reparameterization(active_alpha)
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
        outer += 1

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
            args.state_equidistance_growth_rtol,
            args.state_equidistance_ceiling_factor,
            args.state_outer_history,
            args.state_outer_history_trigger_fmax,
            args.state_curve_direct_solve,
            args.state_curve_direct_trial_calls,
            args.state_curve_immediate_block_relock,
            args.state_curve_direct_failure_line_search_trials,
            args.state_curve_transactional_relock_budget,
            args.state_curve_require_kkt_completion,
            args.state_uniform_raw_exact_subproblem,
            args.state_uniform_raw_use_preconditioner,
            KKT_COMPLETION_PUBLISH_POLICY_VERSION,
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
            args.budget,
        )
    )
    configuration_hash = hashlib.sha256(
        configuration_signature.encode("utf-8")
    ).hexdigest()[:12]
    tag = (
        f"ivm_{args.run_label or 'tuning'}_central_{args.solver}_"
        f"cfg{configuration_hash}_budget{args.budget}"
    ).replace(".", "p").replace(":", "to").replace(",", "_")
    output = root / "runs" / "ivm" / f"{tag}.npz"
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
        state_curve_direct_trial_calls=np.array(
            args.state_curve_direct_trial_calls
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
        state_curve_require_kkt_completion=np.array(
            args.state_curve_require_kkt_completion
        ),
        state_uniform_raw_exact_subproblem=np.array(
            args.state_uniform_raw_exact_subproblem
        ),
        state_uniform_raw_use_preconditioner=np.array(
            args.state_uniform_raw_use_preconditioner
        ),
        state_uniform_relock_is_unconditional_geometry_only=np.array(
            args.state_uniform_raw_exact_subproblem
        ),
        state_curve_kkt_completion_publish_policy_version=np.array(
            KKT_COMPLETION_PUBLISH_POLICY_VERSION
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
        state_free_polish_relative_defect_target=np.array(
            args.state_free_polish_relative_defect_target
        ),
        state_free_polish_active_final=np.array(
            state_free_polish_active
        ),
        state_curve_relock_count=np.array(state_curve_relock_count),
        state_curve_kkt_completion_rollbacks=np.array(
            state_curve_kkt_completion_rollbacks
        ),
        state_curve_kkt_completion_incomplete_final=np.array(
            state_curve_kkt_completion_incomplete_final
        ),
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
        budget=np.array(args.budget),
        optimization_barrier_is_sparse=np.array(True),
        barrier_metric_for_figures=np.array(
            "adaptive real-OpenMM path densification: 10 then 20 "
            "subdivisions per segment; double to 40/80 when successive "
            "barriers differ by more than 0.05 kcal/mol"
        ),
        stop_reason=np.array(stop_reason),
        **{name: np.asarray(values) for name, values in history.items()},
    )
    audit = audit_counter_consistency(backend, evaluators)
    audit.update(
        {
            "result_file": str(output),
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
            "state_curve_transactional_relock_budget": bool(
                args.state_curve_transactional_relock_budget
            ),
            "state_uniform_raw_exact_subproblem": bool(
                args.state_uniform_raw_exact_subproblem
            ),
            "state_uniform_raw_use_preconditioner": bool(
                args.state_uniform_raw_use_preconditioner
            ),
            "state_uniform_relock_is_unconditional_geometry_only": bool(
                args.state_uniform_raw_exact_subproblem
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
            "state_curve_kkt_completion_rollbacks": int(
                state_curve_kkt_completion_rollbacks
            ),
            "state_curve_kkt_completion_incomplete_final": bool(
                state_curve_kkt_completion_incomplete_final
            ),
            "state_curve_kkt_completion_publish_policy_version": (
                KKT_COMPLETION_PUBLISH_POLICY_VERSION
            ),
            "state_free_polish_reparam_strategy": str(
                args.state_free_polish_reparam_strategy
            ),
            "state_free_polish_relative_defect_trigger": float(
                args.state_free_polish_relative_defect_trigger
            ),
            "state_free_polish_relative_defect_target": float(
                args.state_free_polish_relative_defect_target
            ),
            "state_strict_polish_active": bool(
                state_strict_polish_active
            ),
        }
    )
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
        choices=("iperp_monotone", "proximal_only"),
        default="iperp_monotone",
        help=(
            "Ablation-only outer acceptance policy for state_ivm. "
            "iperp_monotone retains the production refreshed-I_perp guard; "
            "proximal_only accepts exact frozen-tangent proximal-objective "
            "descent without imposing I_perp as a per-step Lyapunov merit."
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
        "--state-curve-immediate-block-relock",
        action="store_true",
        help=(
            "When the synchronized equal-spring proximal line search fails, "
            "immediately solve the largest-normal-force image block in the "
            "same frozen-tangent, fixed-alpha proximal problem, then apply "
            "the optional geometry-only Hermite relock."
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
            "equal-spring predictor enters the raw-block plus Hermite-relock "
            "cycle. Failed predictor and block evaluations remain charged, "
            "but a force-call boundary cannot publish a half-completed "
            "operator-split macrostate."
        ),
    )
    parser.add_argument(
        "--state-curve-require-kkt-completion",
        action="store_true",
        help=(
            "If the synchronized equal-spring predictor stops before every "
            "image satisfies the common proximal KKT tolerance, discard the "
            "unfinished candidate macrostep (while retaining all of its "
            "audited evaluations), solve the same frozen-tangent proximal "
            "problem for the full image band with the remaining evaluation "
            "budget, and only then apply the optional geometry-only relock. "
            "The rule is independent of alpha and force-call index."
        ),
    )
    parser.add_argument(
        "--state-uniform-raw-exact-subproblem",
        action="store_true",
        help=(
            "Use one alpha-independent solver path from the first macrostep: "
            "solve every image's original frozen-central-normal proximal "
            "subproblem to the common KKT rule, publish only a bandwise "
            "complete solve, and then apply one optional geometry-only "
            "central-Hermite equal-spring relock. This disables the "
            "synchronized equal-spring predictor/solver switch."
        ),
    )
    parser.add_argument(
        "--state-uniform-raw-use-preconditioner",
        action="store_true",
        help=(
            "Use the geometry-derived ASE force-field preconditioner inside "
            "the uniform raw proximal solver. It never requests OpenMM "
            "energy, force, or Hessian data; the default clean screen omits "
            "it so every alpha uses projected L-BFGS with the same identity "
            "initial metric."
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
            "Hermite relocking. It is state based and independent of alpha "
            "and force-call count."
        ),
    )
    parser.add_argument(
        "--state-free-polish-reparam-strategy",
        choices=("euler", "relative_defect", "hermite_relock"),
        default="relative_defect",
        help=(
            "Geometry-only correction used after projected-curve stalling. "
            "euler composes a fixed number of current-central-tangent spring "
            "microsteps; relative_defect relaxes to a dimensionless spacing "
            "tolerance; hermite_relock makes one free frozen-normal escape "
            "step, restores exact identical-spring equilibrium on the shared "
            "central-tangent Hermite curve, and reactivates the synchronized "
            "projected predictor."
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
            "places identical springs at equilibrium on one frozen Hermite "
            "curve whose internal derivatives are the same central tangent, "
            "then requires genuine IVM recovery before commit. "
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
    parser.add_argument("--save-all-paths", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
