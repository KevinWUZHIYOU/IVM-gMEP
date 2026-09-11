#!/usr/bin/env python3
"""Recovery-only scale-consistency candidate with literal commit counting.

Online control is identical for every path resolution.  Every actual
reparameterization is the nearest-neighbour spring
``(L[i+1]-L[i])*tau_i`` with ``tau_i`` from the improved tangent.  Normal
momentum and proximal L-BFGS steps are recovery subproblem steps, never
reparameterization.  A generic provisional trust expansion protects early
non-monotone proposals.  At a recovery cap, monotone normal-recovery steps
stream publicly.  After each m_max block, a non-positive regularized
prediction rejects the HJ spring without NoF and preserves recovery memory;
otherwise the spring is evaluated and accepted only if online I_perp does not
increase.  Recovery-side monotone publication is tested exactly, without an
additive rounding envelope.  Before a paid recovery trial, the local proposal
and the existing ``ceil(r*m_max)`` horizon proposal are both constructed with
the same improved-tangent spring.  The ordinary regularized model remains the
default; the horizon is selected only when normalizing each displacement by
its own registered fraction reverses that choice by more than the existing
prediction-resolution floor.  A directly accepted horizon counts as one
recovery commit; the online recovery cap, rather than a synthetic counter
increment, determines whether another paid step is needed.  This choice is
zero-NoF and introduces no new control, resolution branch, NoF-stage branch,
baseline input, or offline metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ALG_SRC = ROOT / (
    "python/q10_irivm_low_r_coherence_secant_v2_20260830/src"
)
sys.path.insert(0, str(ALG_SRC))
import q10_irivm_atomic_tangent_secant_core as alg  # noqa: E402


# Fail-closed implementation guards.  They are deliberately far above every
# selected trajectory and are not optimization parameters or stopping stages.
MAX_OUTER_ITERATIONS = 4096
MAX_CONSECUTIVE_ZERO_NOF_ITERATIONS = 64


@dataclass(frozen=True)
class Controls:
    """The complete, resolution-independent IR-IVM control surface."""

    n_internal: int
    alpha: float
    r: float
    m_max: int

    @property
    def alpha_label(self):
        value = float(self.alpha)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("alpha must be finite and positive")
        exponent = int(np.floor(np.log10(value)))
        mantissa = value / (10.0 ** exponent)
        integer_mantissa = int(np.rint(mantissa))
        tolerance = (
            64.0 * np.finfo(np.float64).eps * max(1.0, abs(mantissa))
        )
        if not (
            1 <= integer_mantissa <= 9
            and abs(mantissa - integer_mantissa) <= tolerance
        ):
            raise ValueError("alpha must have an integer scientific mantissa")
        return f"{integer_mantissa}e{exponent}"

    def validate(self):
        if int(self.n_internal) not in (10, 20):
            raise ValueError("n_internal must be 10 or 20")
        if not (0.0 < float(self.r) < 0.7):
            raise ValueError("r must satisfy 0 < r < 0.7")
        if int(self.m_max) < 4:
            raise ValueError("m_max must be at least 4")
        _ = self.alpha_label

    def as_dict(self):
        return {
            "n_internal": int(self.n_internal),
            "alpha": float(self.alpha),
            "alpha_label": self.alpha_label,
            "r": float(self.r),
            "m_max": int(self.m_max),
        }

    @property
    def run_id(self):
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def hj_nearest_neighbor_spring_reparameterize(core, path, energies):
    """Apply only ``(L[i+1]-L[i])*tau_i`` with HJ improved tangents."""

    result = np.asarray(path, dtype=np.float64).copy()
    fixed_energies = np.asarray(energies, dtype=np.float64)
    residual = np.inf
    iterations = 0
    for iteration in range(alg.SPRING_MAX_ITER):
        lengths = np.linalg.norm(np.diff(result, axis=0), axis=1)
        tangents = core.improved_tangents(result, fixed_energies)
        if tangents.shape != result[1:-1].shape:
            raise RuntimeError("improved-tangent spring shape mismatch")
        length_differences = lengths[1:] - lengths[:-1]
        spring_force = length_differences[:, None] * tangents
        residual = float(np.max(np.linalg.norm(spring_force, axis=1)))
        iterations = iteration + 1
        if residual <= alg.SPRING_TOLERANCE:
            break
        result[1:-1] += alg.SPRING_DTAU * spring_force
    result[0] = path[0]
    result[-1] = path[-1]
    if not (
        np.array_equal(result[0], path[0])
        and np.array_equal(result[-1], path[-1])
    ):
        raise RuntimeError("spring reparameterization moved an endpoint")
    return result, residual, iterations


def append_record(records, nf, phase, state):
    path, energies, gradients = state
    record = {
        "nf": int(nf),
        "phase": str(phase),
        "path": np.asarray(path, dtype=np.float64).copy(),
        "energies": np.asarray(energies, dtype=np.float64).copy(),
        "gradients": np.asarray(gradients, dtype=np.float64).copy(),
    }
    if records and int(records[-1]["nf"]) == int(nf):
        records[-1] = record
    else:
        records.append(record)


def publish_pairs(memories, old_state, new_state, history_size, mode="all"):
    """Publish only curvature-valid physical per-image secant pairs."""

    old_path, _, old_gradients = old_state
    new_path, _, new_gradients = new_state
    candidates = []
    for index in range(1, old_path.shape[0] - 1):
        step = new_path[index] - old_path[index]
        change = new_gradients[index] - old_gradients[index]
        curvature = float(np.dot(step, change))
        threshold = alg.RECOVERY_SECANT_CURVATURE * float(
            np.linalg.norm(step) * np.linalg.norm(change)
        )
        if curvature > threshold:
            normalized = curvature / max(
                np.linalg.norm(step) * np.linalg.norm(change),
                np.finfo(np.float64).tiny,
            )
            candidates.append((
                float(normalized),
                index,
                step.copy(),
                change.copy(),
            ))
    if mode == "top1" and candidates:
        candidates = [max(candidates, key=lambda item: (item[0], -item[1]))]
    elif mode != "all":
        raise ValueError(mode)
    for _, index, step, change in candidates:
        memories[index - 1].append((step, change))
        del memories[index - 1][:-int(history_size)]
    return len(candidates)


def published_i_perp(core, state):
    """Online path-integrated perpendicular-force metric."""

    path, energies, gradients = state
    tangents = core.improved_tangents(path, energies)
    forces = -np.asarray(gradients, dtype=np.float64)[1:-1]
    normal = forces - np.sum(
        forces * tangents, axis=1
    )[:, None] * tangents
    coordinate = np.linspace(0.0, 1.0, path.shape[0])
    speed = np.linalg.norm(
        np.gradient(path, coordinate, axis=0, edge_order=2), axis=1
    )
    ds = 1.0 / float(path.shape[0] - 1)
    return float(
        np.sum(np.sum(normal * normal, axis=1) * speed[1:-1]) * ds
    )

def copy_state(state):
    return tuple(np.asarray(item, dtype=np.float64).copy() for item in state)


def append_public(records, nf, phase, state):
    append_record(records, nf, phase, state)


def agreement_merits(anchor, candidate, alpha):
    displacement = candidate[0][1:-1] - anchor[0][1:-1]
    norm_penalty = float(
        np.sum(displacement * displacement) / (2.0 * float(alpha))
    )
    predicted = float(
        -np.sum(anchor[2][1:-1] * displacement) - norm_penalty
    )
    actual = float(
        np.sum(anchor[1][1:-1] - candidate[1][1:-1]) - norm_penalty
    )
    return predicted, actual


def assess_recovery_candidate(
    core,
    anchor,
    candidate,
    anchor_i,
    public_envelope_i,
    controls,
    *,
    normal_relaxation,
):
    """Apply the shared online acceptance tests to one evaluated proposal."""

    candidate_i = published_i_perp(core, candidate)
    changed = bool(
        np.max(np.linalg.norm(candidate[0] - anchor[0], axis=1))
        > 1.0e-14
    )
    monotone_to_work = bool(
        changed
        and candidate_i
        <= anchor_i + alg._rounding_slack(candidate_i, anchor_i)
    )
    predicted, actual = agreement_merits(anchor, candidate, controls.alpha)
    positive_merit = bool(
        changed and np.isfinite(actual) and actual > 0.0
    )
    raw_accepted = bool(monotone_to_work or positive_merit)
    monotone_to_public = bool(
        changed
        and candidate_i <= public_envelope_i
    )
    trusted = bool(
        positive_merit
        and predicted > 0.0
        and actual >= (1.0 - float(controls.r)) * predicted
    )
    if normal_relaxation:
        changed = bool(not np.array_equal(candidate[0], anchor[0]))
        monotone_to_work = bool(
            changed
            and candidate_i
            <= anchor_i + alg._rounding_slack(candidate_i, anchor_i)
        )
        monotone_to_public = bool(
            changed
            and candidate_i <= public_envelope_i
        )
        positive_merit = False
        raw_accepted = monotone_to_work
        trusted = False
    agreement = float(actual / predicted) if predicted > 0.0 else None
    return {
        "candidate_i": float(candidate_i),
        "changed": changed,
        "monotone_to_work": monotone_to_work,
        "monotone_to_public": monotone_to_public,
        "predicted_actual_step": float(predicted),
        "actual_merit": float(actual),
        "actual_over_predicted": agreement,
        "trusted": trusted,
        "raw_accepted": raw_accepted,
    }


def recovery_proposal_spring(
    core, state, memories, controls, fraction, direction_mode="warm"
):
    """One whole-band normal proposal followed by the formal spring."""

    if direction_mode != "warm":
        raise ValueError("clean recovery has one warm whole-band direction")
    path, energies, gradients = state
    trust = alg.rms_segment_trust_radius(path)
    selected = alg.recovery_direction_candidate(
        core,
        path,
        energies,
        gradients,
        memories,
        float(controls.alpha),
        trust,
        "recovery_first",
    )
    prediction = float(selected["predicted_decrease"])
    warm_pairs = int(selected["warm_pair_count"])
    step = float(selected["step_length"]) * float(fraction)
    if step <= 0.0:
        return None, {"reason": "no_descent_direction", "step": step}
    raw = path + step * np.asarray(selected["directions"], dtype=np.float64)
    raw[0] = path[0]
    raw[-1] = path[-1]
    spring, spring_residual, spring_iterations = (
        hj_nearest_neighbor_spring_reparameterize(core, raw, energies)
    )
    lengths = np.linalg.norm(np.diff(spring, axis=0), axis=1)
    residual = float(np.max(np.abs(lengths[1:] - lengths[:-1])))
    finite = bool(np.all(np.isfinite(spring)) and np.isfinite(residual))
    pre_eq = alg.equidistance_integral(path)
    spring_eq = alg.equidistance_integral(spring)
    ceiling = max(
        spring_eq, alg.RECOVERY_EQUIDISTANCE_CONTRACTION * pre_eq
    )
    audit = {
        "reason": "projected" if finite else "projection_infeasible",
        "step": step,
        "base_step": float(selected["step_length"]),
        "fraction": float(fraction),
        "prediction": prediction,
        "warm_pairs": warm_pairs,
        "selected_direction": str(selected["name"]),
        "warm_prediction": prediction,
        "pre_reset_warm_prediction": prediction,
        "pre_reset_warm_pairs": warm_pairs,
        "prediction_resolution_floor": float(
            64.0
            * np.finfo(np.float64).eps
            * max(1.0, float(np.sum(np.abs(energies[1:-1]))))
        ),
        "memory_reset_at_prediction_floor": False,
        "projection_residual": residual,
        "projection_iterations": 0,
        "projection_backtracks": 0,
        "projection_feasible": finite,
        "pre_equidistance": float(pre_eq),
        "seed_equidistance": float(spring_eq),
        "equidistance_ceiling": float(ceiling),
        "seed": {
            "mode": "improved_tangent_spring",
            "residual": float(spring_residual),
            "iterations": int(spring_iterations),
        },
    }
    return (spring if finite else None), audit


def synchronized_spring_recovery(
    core,
    pre,
    direct,
    evaluators,
    accepted_recovery_memories,
    recovery_backtrack_state,
    controls,
    residual,
    attempt_start_calls,
    *,
    nof_breakdown,
):
    """One atomic full-band normal recovery followed by the formal spring."""

    del residual, attempt_start_calls
    pre_path, _, _ = pre
    path, energies, gradients = (
        np.asarray(item, dtype=np.float64).copy() for item in direct
    )
    pre_i, _ = alg.internal_i_perp(core, *pre)
    direct_i, _ = alg.internal_i_perp(core, *direct)
    pre_eq = alg.equidistance_integral(pre_path)
    direct_eq = alg.equidistance_integral(path)
    ceiling = max(
        direct_eq, alg.RECOVERY_EQUIDISTANCE_CONTRACTION * pre_eq
    )
    transaction = {
        "committed": False,
        "interrupted": False,
        "sync_trial_evaluated": False,
    }
    if alg._i_ok(direct_i, pre_i) and alg._eq_ok(direct_eq, ceiling):
        transaction["committed"] = True
        return (path, energies, gradients), transaction, None

    trust = alg.rms_segment_trust_radius(pre_path)
    if recovery_backtrack_state is None:
        selected = alg.recovery_direction_candidate(
            core,
            path,
            energies,
            gradients,
            accepted_recovery_memories,
            float(controls.alpha),
            trust,
            "accepted_history",
        )
        selected["backtrack_fraction"] = 1.0
        selected["base_step_length"] = float(selected["step_length"])
        selected["base_predicted_decrease"] = float(
            selected["predicted_decrease"]
        )
    else:
        selected = alg.recovery_backtrack_candidate(
            recovery_backtrack_state, float(controls.alpha)
        )
    step = float(selected["step_length"])
    if step <= 0.0 or float(selected["predicted_decrease"]) <= 0.0:
        return pre, transaction, None

    if recovery_backtrack_state is None:
        candidate_displacement = step * np.asarray(
            selected["directions"], dtype=np.float64
        )
    else:
        candidate_displacement = float(
            selected["backtrack_fraction"]
        ) * np.asarray(selected["base_displacement"], dtype=np.float64)
    raw = path + candidate_displacement
    raw[0] = path[0]
    raw[-1] = path[-1]
    if recovery_backtrack_state is None:
        selected["base_displacement"] = raw - path

    spring, _, _ = hj_nearest_neighbor_spring_reparameterize(
        core, raw, energies
    )
    before = int(sum(item.calls for item in evaluators))
    try:
        candidate_energies, candidate_gradients = alg.evaluate_path(
            core, spring, evaluators
        )
    except alg.BudgetReached:
        nof_breakdown["synchronized_recovery_trial"] += (
            int(sum(item.calls for item in evaluators)) - before
        )
        transaction["interrupted"] = True
        return pre, transaction, None
    nof_breakdown["synchronized_recovery_trial"] += (
        int(sum(item.calls for item in evaluators)) - before
    )
    transaction["sync_trial_evaluated"] = True
    candidate = (spring, candidate_energies, candidate_gradients)
    merged, accepted, gate = alg.atomic_recovery_merge(
        core, direct, candidate, ceiling, float(controls.alpha), float(pre_i)
    )
    if bool(gate["accepted_by_atomic_gate"]) and int(np.count_nonzero(accepted)) == int(core.N_INTERNAL):
        alg.publish_accepted_recovery_secants(
            accepted_recovery_memories,
            direct,
            candidate,
            strong_recovery_selected=False,
            stationary_normal_probe=False,
        )
        transaction["committed"] = True
        return merged, transaction, None
    return pre, transaction, selected


def normal_gradient_and_tangents(core, state, n_internal):
    """Compute the improved-tangent normal gradient for one band."""

    path, energies, gradients = state
    tangents = core.improved_tangents(path, energies)
    normal_gradient = np.zeros_like(path)
    for image in range(1, int(n_internal) + 1):
        tangent = tangents[image - 1]
        normal_gradient[image] = (
            gradients[image]
            - np.dot(gradients[image], tangent) * tangent
        )
    return normal_gradient, tangents


def select_normal_recovery_proposal(
    core,
    anchor,
    controls,
    previous_direction,
    lbfgs_pairs,
    severe_spring_retry,
):
    """Choose one opponent-blind normal-recovery direction at zero NoF."""

    anchor_normal_gradient, normal_tangents = normal_gradient_and_tangents(
        core, anchor, controls.n_internal
    )
    base_direction = (
        -float(controls.alpha)
        / float(controls.m_max)
        * anchor_normal_gradient
    )
    momentum_direction = base_direction.copy()
    momentum_used = bool(previous_direction is not None)
    if momentum_used:
        momentum_direction += float(controls.r) * previous_direction
    direction_candidates = [(
        "-(alpha/m_max)*g_perp+r*d_prev",
        momentum_direction,
    )]

    if lbfgs_pairs:
        q = anchor_normal_gradient[1:-1].reshape(-1).copy()
        reverse_coefficients = []
        for secant_s, secant_y in reversed(lbfgs_pairs):
            rho = 1.0 / float(np.dot(secant_s, secant_y))
            coefficient = rho * float(np.dot(secant_s, q))
            reverse_coefficients.append((rho, coefficient))
            q = q - coefficient * secant_y
        inverse_q = (
            float(controls.alpha) / float(controls.m_max) * q
        )
        for pair, values in zip(
            lbfgs_pairs, reversed(reverse_coefficients)
        ):
            secant_s, secant_y = pair
            rho, coefficient = values
            beta = rho * float(np.dot(secant_y, inverse_q))
            inverse_q += secant_s * (coefficient - beta)
        lbfgs_direction = np.zeros_like(anchor[0])
        lbfgs_direction[1:-1] = -inverse_q.reshape(
            anchor[0][1:-1].shape
        )
        for image in range(1, int(controls.n_internal) + 1):
            tangent = normal_tangents[image - 1]
            lbfgs_direction[image] -= (
                np.dot(lbfgs_direction[image], tangent) * tangent
            )
        direction_candidates.append((
            "proximal_whole_band_lbfgs",
            lbfgs_direction,
        ))

    predicted_candidates = []
    for direction_name, candidate_direction in direction_candidates:
        candidate_proposal = (
            np.asarray(anchor[0]).copy() + candidate_direction
        )
        candidate_proposal[0] = anchor[0][0]
        candidate_proposal[-1] = anchor[0][-1]
        displacement = candidate_proposal[1:-1] - anchor[0][1:-1]
        prediction = float(
            -np.sum(anchor[2][1:-1] * displacement)
            - np.sum(displacement * displacement)
            / (2.0 * float(controls.alpha))
        )
        predicted_candidates.append((
            prediction,
            direction_name,
            candidate_direction,
            candidate_proposal,
        ))

    if severe_spring_retry is not None:
        candidate_proposal = np.asarray(
            severe_spring_retry, dtype=np.float64
        ).copy()
        displacement = candidate_proposal[1:-1] - anchor[0][1:-1]
        prediction = float(
            -np.sum(anchor[2][1:-1] * displacement)
            - np.sum(displacement * displacement)
            / (2.0 * float(controls.alpha))
        )
        predicted_candidates.append((
            prediction,
            "severe_rejection_spring_secant",
            candidate_proposal - anchor[0],
            candidate_proposal,
        ))

    (
        normal_prediction,
        normal_direction_name,
        _normal_direction,
        proposal,
    ) = max(predicted_candidates, key=lambda entry: entry[0])
    audit = {
        "reason": "online_normal_relaxation",
        "selected_direction": normal_direction_name,
        "normal_relaxation": True,
        "normal_momentum_beta": float(controls.r),
        "normal_momentum_used": momentum_used,
        "normal_lbfgs_pairs": int(len(lbfgs_pairs)),
        "normal_candidate_predictions": {
            name: float(prediction)
            for prediction, name, _direction, _proposal
            in predicted_candidates
        },
        "normal_prediction": float(normal_prediction),
    }
    return proposal, audit, anchor_normal_gradient


def record_accepted_normal_recovery(
    core,
    anchor,
    candidate,
    anchor_normal_gradient,
    controls,
    lbfgs_pairs,
):
    """Update momentum and proximal L-BFGS memory after one public step."""

    previous_direction = candidate[0] - anchor[0]
    candidate_normal_gradient, _ = normal_gradient_and_tangents(
        core, candidate, controls.n_internal
    )
    secant_s = (
        candidate[0][1:-1] - anchor[0][1:-1]
    ).reshape(-1)
    secant_y = (
        candidate_normal_gradient[1:-1]
        - anchor_normal_gradient[1:-1]
    ).reshape(-1) + secant_s / float(controls.alpha)
    curvature = float(np.dot(secant_s, secant_y))
    curvature_floor = float(
        1.0e-8
        * np.linalg.norm(secant_s)
        * np.linalg.norm(secant_y)
    )
    if curvature > curvature_floor:
        lbfgs_pairs.append((secant_s.copy(), secant_y.copy()))
        lbfgs_pairs[:] = lbfgs_pairs[-int(controls.m_max):]
    return previous_direction


def build_severe_rejection_hj_spring_retry(
    core,
    records,
    anchor,
    rejected_candidate,
    controls,
):
    """Build the online secant retry and apply the only legal HJ spring."""

    previous_path = None
    for previous_record in reversed(records):
        candidate_previous = np.asarray(
            previous_record["path"], dtype=np.float64
        )
        if not np.array_equal(candidate_previous, anchor[0]):
            previous_path = candidate_previous
            break
    if previous_path is None:
        return None, {"severe_retry_secant_available": False}

    current_displacement = rejected_candidate[0] - anchor[0]
    previous_displacement = anchor[0] - previous_path
    secant_displacement = (
        float(alg.BACKTRACK) * current_displacement
        + float(controls.r)
        * (current_displacement - previous_displacement)
    )
    raw_secant = np.asarray(anchor[0]).copy() + secant_displacement
    raw_secant[0] = anchor[0][0]
    raw_secant[-1] = anchor[0][-1]
    proposal, residual, iterations = hj_nearest_neighbor_spring_reparameterize(
        core, raw_secant, anchor[1]
    )
    return proposal, {
        "severe_retry_secant_available": True,
        "severe_retry_secant_residual": float(residual),
        "severe_retry_secant_iterations": int(iterations),
    }


def maybe_start_cap_normal_stream(
    core,
    anchor,
    memories,
    controls,
    proposal,
    proposal_audit,
    recovery_commit_streak,
    expanding_provisional,
):
    """Replace a rejected cap spring model by one warm normal step."""

    anchor_normal_gradient = None
    at_cap = bool(
        proposal is not None
        and not expanding_provisional
        and recovery_commit_streak + 1 >= int(controls.m_max)
    )
    if not at_cap:
        return proposal, proposal_audit, anchor_normal_gradient

    cap_displacement = proposal[1:-1] - anchor[0][1:-1]
    cap_prediction = float(
        -np.sum(anchor[2][1:-1] * cap_displacement)
        - np.sum(cap_displacement * cap_displacement)
        / (2.0 * float(controls.alpha))
    )
    cap_shortened = bool(
        float(proposal_audit.get("step", 1.0))
        < 1.0 - float(controls.r)
    )
    if not (cap_shortened and cap_prediction <= 0.0):
        return proposal, proposal_audit, anchor_normal_gradient

    # This is a normal recovery subproblem direction.  It is deliberately
    # not passed through reparameterization.
    cap_direction = alg.recovery_direction_candidate(
        core,
        *anchor,
        memories,
        float(controls.alpha),
        alg.rms_segment_trust_radius(anchor[0]),
        "ordinary_warm_normal_lbfgs",
    )
    cap_step = float(cap_direction["step_length"])
    cap_warm_prediction = float(cap_direction["predicted_decrease"])
    if cap_step <= 0.0 or cap_warm_prediction <= 0.0:
        return proposal, proposal_audit, anchor_normal_gradient

    proposal = np.asarray(anchor[0]).copy()
    proposal += cap_step * np.asarray(
        cap_direction["directions"], dtype=np.float64
    )
    proposal[0] = anchor[0][0]
    proposal[-1] = anchor[0][-1]
    anchor_normal_gradient, _ = normal_gradient_and_tangents(
        core, anchor, controls.n_internal
    )
    proposal_audit = dict(proposal_audit)
    proposal_audit.update({
        "reason": "cap_normal_stream_start",
        "selected_direction": "ordinary_warm_normal_lbfgs",
        "normal_relaxation": True,
        "normal_prediction": cap_warm_prediction,
        "normal_candidate_predictions": {
            "ordinary_warm_normal_lbfgs": cap_warm_prediction,
        },
        "rejected_cap_spring_prediction": cap_prediction,
        "rejected_cap_spring_shortened": True,
    })
    return proposal, proposal_audit, anchor_normal_gradient


def prepare_normal_stream_exit_hj_spring(
    core,
    anchor,
    controls,
    *,
    outer,
    nf,
):
    """Construct and score the sole reparameterization at an episode cap."""

    spring, residual, iterations = hj_nearest_neighbor_spring_reparameterize(
        core, anchor[0], anchor[1]
    )
    displacement = spring[1:-1] - anchor[0][1:-1]
    prediction = float(
        -np.sum(anchor[2][1:-1] * displacement)
        - np.sum(displacement * displacement)
        / (2.0 * float(controls.alpha))
    )
    audit = {
        "outer": int(outer),
        "nf": int(nf),
        "reason": "normal_stream_exit_hj_spring",
        "spring_prediction": prediction,
        "spring_residual": float(residual),
        "spring_iterations": int(iterations),
        "spring_evaluated": False,
        "spring_accepted": False,
        "normal_memory_preserved": True,
    }
    return spring, prediction, audit


def close_normal_stream_episode(
    core,
    controls,
    public_state,
    public_memories,
    working_state,
    working_memories,
    normal_stage_reference_i,
    previous_normal_recovery_direction,
    normal_lbfgs_pairs,
    records,
    ledger,
    evaluate_band_counted,
    *,
    outer,
):
    """Test the cap HJ spring and preserve or close normal recovery state."""

    exit_anchor = copy_state(public_state)
    exit_spring, exit_prediction, exit_audit = (
        prepare_normal_stream_exit_hj_spring(
            core,
            exit_anchor,
            controls,
            outer=outer,
            nf=ledger.total,
        )
    )
    result = {
        "public_state": public_state,
        "public_memories": public_memories,
        "working_state": working_state,
        "working_memories": working_memories,
        "normal_fallback_active": True,
        "normal_stage_reference_i": normal_stage_reference_i,
        "previous_normal_recovery_direction": (
            previous_normal_recovery_direction
        ),
        "normal_lbfgs_pairs": normal_lbfgs_pairs,
        "recovery_commit_streak": 0,
        "interrupted": False,
        "audit": exit_audit,
    }
    if exit_prediction <= 0.0:
        # A rejected zero-NoF reparameterization cannot erase useful
        # normal-recovery state or consume a force call.
        exit_audit["public_action"] = (
            "zero_nof_reparam_reject_continue_normal"
        )
        return result

    exit_start_nf = int(ledger.total)
    try:
        exit_energies, exit_gradients = evaluate_band_counted(
            "recovery_band_trial", exit_spring
        )
    except alg.BudgetReached:
        exit_audit["spring_interrupted"] = True
        result["interrupted"] = True
        return result

    exit_candidate = (
        np.asarray(exit_spring, dtype=np.float64),
        exit_energies,
        exit_gradients,
    )
    exit_anchor_i = published_i_perp(core, exit_anchor)
    exit_candidate_i = published_i_perp(core, exit_candidate)
    exit_changed = bool(
        not np.array_equal(exit_candidate[0], exit_anchor[0])
    )
    exit_accepted = bool(
        exit_changed
        and exit_candidate_i <= exit_anchor_i
    )
    exit_audit.update({
        "nf": int(ledger.total),
        "new_force_calls": int(ledger.total - exit_start_nf),
        "spring_evaluated": True,
        "anchor_i": float(exit_anchor_i),
        "candidate_i": float(exit_candidate_i),
        "spring_accepted": exit_accepted,
    })
    if not exit_accepted:
        exit_audit["public_action"] = (
            "evaluated_hj_spring_reject_continue_normal"
        )
        append_public(
            records,
            ledger.total,
            "normal_stream_exit_hj_spring_reject",
            public_state,
        )
        return result

    exit_memories = alg.clone_physical_memories(public_memories)
    publish_pairs(
        exit_memories,
        exit_anchor,
        exit_candidate,
        alg.HISTORY_SIZE,
        mode="all",
    )
    public_state = copy_state(exit_candidate)
    public_memories = exit_memories
    working_state = copy_state(public_state)
    working_memories = alg.clone_physical_memories(public_memories)
    exit_audit["normal_memory_preserved"] = False
    exit_audit["public_action"] = (
        "positive_hj_spring_commit_open_spring_episode"
    )
    append_public(
        records,
        ledger.total,
        "normal_stream_exit_hj_spring_commit",
        public_state,
    )
    result.update({
        "public_state": public_state,
        "public_memories": public_memories,
        "working_state": working_state,
        "working_memories": working_memories,
        "normal_fallback_active": False,
        "normal_stage_reference_i": None,
        "previous_normal_recovery_direction": None,
        "normal_lbfgs_pairs": [],
    })
    return result


def run_clean(
    n_internal=20,
    alpha=1.0e-3,
    ratio=0.3,
    m_max=8,
    budget=3000,
):
    controls = Controls(
        int(n_internal), float(alpha), float(ratio), int(m_max)
    )
    controls.validate()
    core, potential = alg._load_backend(int(n_internal))
    ledger = alg.Ledger(limit=int(budget))
    evaluators = alg.make_evaluators(core, potential, ledger)
    nof_breakdown = {
        "initial_band": 0,
        "recovery_band_trial": 0,
        "normal_subproblem": 0,
        "direct_spring_check": 0,
        "synchronized_recovery_trial": 0,
    }

    def evaluate_band_counted(label, path):
        before = int(ledger.total)
        try:
            return alg.evaluate_path(core, path, evaluators)
        finally:
            nof_breakdown[label] += int(ledger.total) - before

    def solve_main_stage_counted(core_, *arguments):
        before = int(ledger.total)
        try:
            return alg.solve_main_stage(core_, *arguments)
        finally:
            nof_breakdown["normal_subproblem"] += (
                int(ledger.total) - before
            )

    initial_path = np.asarray(core.common_path(), dtype=np.float64)
    initial_energies, initial_gradients = evaluate_band_counted(
        "initial_band", initial_path
    )
    public_state = (
        initial_path,
        initial_energies,
        initial_gradients,
    )
    public_memories = [[] for _ in range(int(n_internal))]
    working_state = copy_state(public_state)
    working_memories = alg.clone_physical_memories(public_memories)
    records = []
    append_public(records, ledger.total, "initial", public_state)
    recovery_audits = []
    recovery_states = []
    main_audits = []
    outer = 0
    provisional_depth = 0
    max_provisional_depth = 0
    held_trials = 0
    trusted_nonmonotone_commits = 0
    provisional_rollbacks = 0
    recovery_commit_streak = 0
    expanding_provisional = False
    stationary_exhaustions = 0
    normal_fallback_anchor = None
    normal_fallback_active = False
    severe_rejection_secant_proposal = None
    normal_stage_reference_i = None
    previous_normal_recovery_direction = None
    normal_lbfgs_pairs = []
    bootstrap_active = True
    post_backtrack_state = None
    stop_reason = "budget"
    previous_outer_nof = -1
    zero_nof_iteration_streak = 0

    while int(ledger.total) < int(budget):
        current_outer_nof = int(ledger.total)
        if current_outer_nof == previous_outer_nof:
            zero_nof_iteration_streak += 1
        else:
            zero_nof_iteration_streak = 0
        previous_outer_nof = current_outer_nof
        if zero_nof_iteration_streak >= MAX_CONSECUTIVE_ZERO_NOF_ITERATIONS:
            stop_reason = "zero_nof_safety_limit"
            break
        outer += 1
        if outer > MAX_OUTER_ITERATIONS:
            stop_reason = "outer_safety_limit"
            break
        outer_start = copy_state(public_state)
        recovery_committed = False
        private_continuation = False

        if bootstrap_active and recovery_commit_streak < int(controls.m_max):
            if int(ledger.total) + int(n_internal) > int(budget):
                stop_reason = "budget"
                break
            anchor = copy_state(working_state)
            public_envelope_i = published_i_perp(core, public_state)
            anchor_i = published_i_perp(core, anchor)
            horizon_fraction = float(
                max(
                    1,
                    int(np.ceil(float(controls.r) * int(controls.m_max))),
                )
            )
            if expanding_provisional:
                proposal_fractions = (max(2.0, horizon_fraction),)
            elif horizon_fraction == 1.0:
                proposal_fractions = (1.0,)
            else:
                proposal_fractions = (1.0, horizon_fraction)
            proposal_options = []
            for proposal_fraction in proposal_fractions:
                option, option_audit = recovery_proposal_spring(
                    core,
                    anchor,
                    working_memories,
                    controls,
                    proposal_fraction,
                    "warm",
                )
                if option is None:
                    option_prediction = -np.inf
                    option_scaled_prediction = -np.inf
                else:
                    option_displacement = (
                        option[1:-1] - anchor[0][1:-1]
                    )
                    option_prediction = float(
                        -np.sum(
                            anchor[2][1:-1] * option_displacement
                        )
                        - np.sum(option_displacement * option_displacement)
                        / (2.0 * float(controls.alpha))
                    )
                    scaled_displacement = (
                        option_displacement / float(proposal_fraction)
                    )
                    option_scaled_prediction = float(
                        -np.sum(
                            anchor[2][1:-1] * scaled_displacement
                        )
                        - np.sum(scaled_displacement * scaled_displacement)
                        / (2.0 * float(controls.alpha))
                    )
                proposal_options.append(
                    {
                        "regularized_prediction": option_prediction,
                        "fraction_scaled_prediction": option_scaled_prediction,
                        "fraction": float(proposal_fraction),
                        "proposal": option,
                        "audit": dict(option_audit),
                    }
                )
            standard_option = max(
                proposal_options,
                key=lambda item: (
                    item["regularized_prediction"],
                    -item["fraction"],
                ),
            )
            scaled_option = max(
                proposal_options,
                key=lambda item: (
                    item["fraction_scaled_prediction"],
                    -item["fraction"],
                ),
            )
            prediction_floor = float(
                max(
                    item["audit"].get("prediction_resolution_floor", 0.0)
                    for item in proposal_options
                )
            )
            scale_disagreement = bool(
                not expanding_provisional
                and standard_option["fraction"] == 1.0
                and scaled_option["fraction"] > 1.0
                and scaled_option["regularized_prediction"] > 0.0
                and np.isfinite(scaled_option["fraction_scaled_prediction"])
                and scaled_option["fraction_scaled_prediction"]
                > standard_option["fraction_scaled_prediction"]
                + prediction_floor
            )
            selected_option = scaled_option if scale_disagreement else standard_option
            selected_fraction = selected_option["fraction"]
            proposal = selected_option["proposal"]
            proposal_audit = selected_option["audit"]
            proposal_audit["zero_nof_fraction_choice"] = selected_fraction
            proposal_audit["scale_disagreement_horizon_replaces_expansion"] = (
                scale_disagreement
            )
            proposal_audit["zero_nof_fraction_options"] = [
                {
                    "fraction": item["fraction"],
                    "regularized_prediction": item["regularized_prediction"],
                    "fraction_scaled_prediction": item[
                        "fraction_scaled_prediction"
                    ],
                    "finite_proposal": item["proposal"] is not None,
                }
                for item in proposal_options
            ]
            proposal, proposal_audit, anchor_normal_gradient = (
                maybe_start_cap_normal_stream(
                    core,
                    anchor,
                    working_memories,
                    controls,
                    proposal,
                    proposal_audit,
                    recovery_commit_streak,
                    expanding_provisional,
                )
            )
            normal_fallback_now = bool(
                normal_fallback_active
                or (
                    normal_fallback_anchor is not None
                    and np.array_equal(
                        anchor[0], normal_fallback_anchor[0]
                    )
                )
            )
            if (
                normal_fallback_anchor is not None
                and not normal_fallback_now
            ):
                normal_fallback_anchor = None
            if (
                proposal is not None
                and not expanding_provisional
                and provisional_depth == 0
            ):
                selected_displacement = (
                    proposal[1:-1] - anchor[0][1:-1]
                )
                selected_prediction = float(
                    -np.sum(
                        anchor[2][1:-1] * selected_displacement
                    )
                    - np.sum(
                        selected_displacement * selected_displacement
                    )
                    / (2.0 * float(controls.alpha))
                )
                proposal_audit["post_spring_regularized_prediction"] = (
                    selected_prediction
                )
                if normal_fallback_now:
                    normal_fallback_anchor = None
                    normal_fallback_active = True
                    # This is recovery subproblem continuation, not
                    # reparameterization.
                    proposal, normal_audit, anchor_normal_gradient = (
                        select_normal_recovery_proposal(
                            core,
                            anchor,
                            controls,
                            previous_normal_recovery_direction,
                            normal_lbfgs_pairs,
                            severe_rejection_secant_proposal,
                        )
                    )
                    severe_rejection_secant_proposal = None
                    proposal_audit.update(normal_audit)
            proposal_audit.update(
                {
                    "outer": int(outer),
                    "attempt": 1,
                    "provisional_depth_before": int(provisional_depth),
                    "public_envelope_i": float(public_envelope_i),
                    "working_anchor_i": float(anchor_i),
                }
            )
            if proposal is None:
                proposal_audit.update(
                    {
                        "nf": int(ledger.total),
                        "raw_accepted": False,
                        "public_action": "provisional_rollback_projection_failure",
                    }
                )
                recovery_audits.append(proposal_audit)
                working_state = copy_state(public_state)
                working_memories = alg.clone_physical_memories(public_memories)
                provisional_rollbacks += int(provisional_depth > 0)
                provisional_depth = 0
                bootstrap_active = False
                append_public(
                    records,
                    ledger.total,
                    "recovery_provisional_rollback",
                    public_state,
                )
            else:
                start_nf = int(ledger.total)
                try:
                    candidate_energies, candidate_gradients = evaluate_band_counted(
                        "recovery_band_trial", proposal
                    )
                except alg.BudgetReached:
                    stop_reason = "budget"
                    break
                candidate = (
                    np.asarray(proposal, dtype=np.float64),
                    candidate_energies,
                    candidate_gradients,
                )
                nf = int(ledger.total)
                assessment = assess_recovery_candidate(
                    core,
                    anchor,
                    candidate,
                    anchor_i,
                    public_envelope_i,
                    controls,
                    normal_relaxation=bool(
                        proposal_audit.get("normal_relaxation", False)
                    ),
                )
                candidate_i = assessment["candidate_i"]
                changed = assessment["changed"]
                monotone_to_work = assessment["monotone_to_work"]
                monotone_to_public = assessment["monotone_to_public"]
                raw_accepted = assessment["raw_accepted"]
                trusted = assessment["trusted"]
                proposal_audit.update({
                    "nf": nf,
                    "new_force_calls": int(nf - start_nf),
                    **assessment,
                })
                recovery_states.append(
                    {
                        "nf": nf,
                        "anchor_path": anchor[0].copy(),
                        "candidate_path": candidate[0].copy(),
                        "public_path_before": public_state[0].copy(),
                    }
                )

                if raw_accepted:
                    if expanding_provisional:
                        if monotone_to_public:
                            publish_pairs(
                                working_memories,
                                anchor,
                                candidate,
                                alg.HISTORY_SIZE,
                                mode="all",
                            )
                            public_state = copy_state(candidate)
                            public_memories = alg.clone_physical_memories(
                                working_memories
                            )
                            working_state = copy_state(public_state)
                            working_memories = alg.clone_physical_memories(
                                public_memories
                            )
                            proposal_audit["public_action"] = (
                                "expanded_envelope_atomic_commit"
                            )
                            recovery_commit_streak += 1
                            recovery_committed = True
                        else:
                            # The provisional trusted state remains public; the
                            # extrapolated endpoint is discarded atomically.
                            working_state = copy_state(public_state)
                            working_memories = alg.clone_physical_memories(
                                public_memories
                            )
                            proposal_audit["public_action"] = (
                                "expanded_trial_rollback_to_provisional"
                            )
                            provisional_rollbacks += 1
                            bootstrap_active = False
                        provisional_depth = 0
                        expanding_provisional = False
                        append_public(
                            records,
                            nf,
                            proposal_audit["public_action"],
                            public_state,
                        )
                    elif monotone_to_public:
                        if proposal_audit.get("normal_relaxation", False):
                            previous_normal_recovery_direction = (
                                record_accepted_normal_recovery(
                                    core,
                                    anchor,
                                    candidate,
                                    anchor_normal_gradient,
                                    controls,
                                    normal_lbfgs_pairs,
                                )
                            )
                            proposal_audit["normal_stage_stop"] = (
                                "optimizer rejection, stationarity, or m_max"
                            )
                        publish_pairs(
                            working_memories,
                            anchor,
                            candidate,
                            alg.HISTORY_SIZE,
                            mode="all",
                        )
                        working_state = copy_state(candidate)
                        provisional_depth = 0
                        recovery_commit_increment = 1
                        recovery_commit_streak += recovery_commit_increment
                        proposal_audit["recovery_commit_increment"] = (
                            recovery_commit_increment
                        )
                        recovery_committed = True
                        if proposal_audit.get("reason") == (
                            "cap_normal_stream_start"
                        ):
                            public_state = copy_state(candidate)
                            public_memories = alg.clone_physical_memories(
                                working_memories
                            )
                            working_state = copy_state(public_state)
                            working_memories = alg.clone_physical_memories(
                                public_memories
                            )
                            normal_fallback_active = True
                            normal_stage_reference_i = float(anchor_i)
                            recovery_commit_streak = 1
                            proposal_audit["public_action"] = (
                                "cap_normal_stream_commit"
                            )
                        else:
                            public_state = copy_state(candidate)
                            public_memories = alg.clone_physical_memories(
                                working_memories
                            )
                            working_state = copy_state(public_state)
                            working_memories = alg.clone_physical_memories(
                                public_memories
                            )
                            proposal_audit["public_action"] = (
                                "envelope_atomic_commit"
                            )
                        append_public(
                            records,
                            nf,
                            proposal_audit["public_action"],
                            public_state,
                        )
                    elif trusted:
                        # Publish the evaluated full step, but retain the
                        # original private anchor/memory for one expansion.
                        provisional_memories = alg.clone_physical_memories(
                            working_memories
                        )
                        publish_pairs(
                            provisional_memories,
                            anchor,
                            candidate,
                            alg.HISTORY_SIZE,
                            mode="all",
                        )
                        public_state = copy_state(candidate)
                        public_memories = provisional_memories
                        provisional_depth = 1
                        max_provisional_depth = max(max_provisional_depth, provisional_depth)
                        held_trials += 1
                        trusted_nonmonotone_commits += 1
                        recovery_commit_streak += 1
                        expanding_provisional = True
                        private_continuation = True
                        proposal_audit["public_action"] = (
                            "trusted_provisional_expand"
                        )
                        append_public(
                            records,
                            nf,
                            proposal_audit["public_action"],
                            public_state,
                        )
                    else:
                        publish_pairs(
                            working_memories,
                            anchor,
                            candidate,
                            alg.HISTORY_SIZE,
                            mode="all",
                        )
                        working_state = copy_state(candidate)
                        provisional_depth += 1
                        max_provisional_depth = max(max_provisional_depth, provisional_depth)
                        held_trials += 1
                        proposal_audit["public_action"] = "private_hold"
                        if provisional_depth >= int(controls.m_max):
                            working_state = copy_state(public_state)
                            working_memories = alg.clone_physical_memories(
                                public_memories
                            )
                            provisional_depth = 0
                            provisional_rollbacks += 1
                            bootstrap_active = False
                            proposal_audit["public_action"] = (
                                "provisional_rollback_mmax"
                            )
                            append_public(
                                records,
                                nf,
                                "recovery_provisional_rollback_mmax",
                                public_state,
                            )
                        else:
                            private_continuation = True
                            append_public(
                                records,
                                nf,
                                "recovery_private_hold",
                                public_state,
                            )
                else:
                    proposal_audit["public_action"] = (
                        "provisional_rollback_trial_failure"
                    )
                    working_state = copy_state(public_state)
                    working_memories = alg.clone_physical_memories(
                        public_memories
                    )
                    provisional_rollbacks += int(provisional_depth > 0)
                    provisional_depth = 0
                    expanding_provisional = False
                    if proposal_audit.get("normal_relaxation", False):
                        # A rejected normal fallback closes this recovery
                        # attempt and returns to the unchanged main solve.
                        normal_fallback_anchor = None
                        normal_fallback_active = False
                        normal_stage_reference_i = None
                        bootstrap_active = False
                    else:
                        rejection_envelope = (
                            float(public_envelope_i)
                            / (1.0 - float(controls.r))
                        )
                        severe_rejection = bool(
                            candidate_i > rejection_envelope
                            + alg._rounding_slack(
                                candidate_i, rejection_envelope
                            )
                        )
                        proposal_audit[
                            "normal_fallback_severe_rejection"
                        ] = severe_rejection
                        if severe_rejection:
                            (
                                severe_rejection_secant_proposal,
                                severe_retry_audit,
                            ) = build_severe_rejection_hj_spring_retry(
                                core,
                                records,
                                anchor,
                                candidate,
                                controls,
                            )
                            proposal_audit.update(severe_retry_audit)
                            normal_fallback_anchor = copy_state(public_state)
                            normal_fallback_active = False
                            normal_stage_reference_i = float(
                                public_envelope_i
                            )
                            bootstrap_active = True
                            private_continuation = True
                        else:
                            normal_fallback_anchor = None
                            normal_fallback_active = False
                            normal_stage_reference_i = None
                            bootstrap_active = False
                    append_public(
                        records,
                        nf,
                        "recovery_provisional_rollback_trial_failure",
                        public_state,
                    )
                proposal_audit["provisional_depth_after"] = int(provisional_depth)
                proposal_audit["expanding_provisional_after"] = bool(
                    expanding_provisional
                )
                recovery_audits.append(proposal_audit)
        else:
            recovery_audits.append(
                {
                    "outer": int(outer),
                    "attempt": 0,
                    "reason": "recovery_bootstrap_already_completed",
                    "raw_accepted": False,
                    "public_action": "not_attempted",
                }
            )
            append_public(records, ledger.total, "recovery_not_attempted", public_state)

        if (
            recovery_committed
            and normal_fallback_active
            and recovery_commit_streak >= int(controls.m_max)
        ):
            exit_result = close_normal_stream_episode(
                core,
                controls,
                public_state,
                public_memories,
                working_state,
                working_memories,
                normal_stage_reference_i,
                previous_normal_recovery_direction,
                normal_lbfgs_pairs,
                records,
                ledger,
                evaluate_band_counted,
                outer=outer,
            )
            public_state = exit_result["public_state"]
            public_memories = exit_result["public_memories"]
            working_state = exit_result["working_state"]
            working_memories = exit_result["working_memories"]
            normal_fallback_active = exit_result["normal_fallback_active"]
            normal_stage_reference_i = exit_result[
                "normal_stage_reference_i"
            ]
            previous_normal_recovery_direction = exit_result[
                "previous_normal_recovery_direction"
            ]
            normal_lbfgs_pairs = exit_result["normal_lbfgs_pairs"]
            recovery_commit_streak = exit_result[
                "recovery_commit_streak"
            ]
            recovery_audits.append(exit_result["audit"])
            if exit_result["interrupted"]:
                stop_reason = "budget"
                break

        if recovery_committed or private_continuation:
            continue

        # A failed provisional trust expansion is fully rolled back before the unchanged v2 main.
        if provisional_depth != 0:
            raise RuntimeError("main entered with an open recovery provisional")
        state = copy_state(
            public_state
        )
        memories = alg.clone_physical_memories(
            public_memories
        )
        if int(ledger.total) >= int(budget):
            break
        try:
            pre_main_memories = alg.clone_physical_memories(memories)
            path, energies, gradients, audit = solve_main_stage_counted(
                core, *state, evaluators, memories, controls
            )
            state = (path, energies, gradients)
            audit = dict(audit)
            audit.update({"outer": int(outer), "nf": int(ledger.total)})
            main_audits.append(audit)
            public_state = copy_state(state)
            public_memories = alg.clone_physical_memories(memories)
            append_public(
                records,
                ledger.total,
                "complete_main_stage",
                public_state,
            )
            main_changed = bool(
                not audit["rolled_back_by_i_perp_guard"]
                and not np.array_equal(public_state[0], outer_start[0])
            )
            if main_changed:
                post_backtrack_state = None
                normal_fallback_anchor = None
                normal_fallback_active = False
                severe_rejection_secant_proposal = None
                normal_stage_reference_i = None
                previous_normal_recovery_direction = None
        except alg.BudgetReached:
            stop_reason = "budget"
            break
        except Exception as error:
            stop_reason = type(error).__name__ + ": " + str(error)
            break

        if int(ledger.total) < int(budget):
            post_main_state = copy_state(public_state)
            post_main_i = published_i_perp(core, post_main_state)
            attempt_start_calls = np.asarray(
                [item.calls for item in evaluators], dtype=np.int64
            )
            direct_path, residual, _ = hj_nearest_neighbor_spring_reparameterize(
                core, public_state[0], public_state[1]
            )
            try:
                direct_energies, direct_gradients = evaluate_band_counted(
                    "direct_spring_check", direct_path
                )
            except alg.BudgetReached:
                stop_reason = "budget"
                break
            main_rolled_back = bool(audit["rolled_back_by_i_perp_guard"])
            recovery_memories = (
                pre_main_memories if main_rolled_back else memories
            )
            if post_backtrack_state is not None and not (
                main_rolled_back
                and alg.recovery_anchor_matches(
                    post_backtrack_state,
                    direct_path,
                    direct_energies,
                    direct_gradients,
                )
            ):
                post_backtrack_state = None
            try:
                updated, transaction, rejected_descriptor = (
                    synchronized_spring_recovery(
                        core,
                        post_main_state,
                        (direct_path, direct_energies, direct_gradients),
                        evaluators,
                        recovery_memories,
                        post_backtrack_state,
                        controls,
                        residual,
                        attempt_start_calls,
                        nof_breakdown=nof_breakdown,
                    )
                )
            except alg.BudgetReached:
                stop_reason = "budget"
                break
            updated_i = published_i_perp(core, updated)
            if (
                transaction["committed"]
                and updated_i <= post_main_i
            ):
                public_state = copy_state(updated)
                if main_rolled_back:
                    memories[:] = alg.clone_physical_memories(
                        recovery_memories
                    )
                public_memories = alg.clone_physical_memories(memories)
                post_phase = "post_recovery_committed"
            else:
                public_state = copy_state(post_main_state)
                public_memories = alg.clone_physical_memories(memories)
                post_phase = "post_recovery_held"
            append_public(records, ledger.total, post_phase, public_state)

            if transaction["committed"] or transaction["interrupted"]:
                post_backtrack_state = None
            elif (
                transaction["sync_trial_evaluated"]
                and rejected_descriptor is not None
            ):
                completed = int(
                    post_backtrack_state["completed_attempts"]
                    if post_backtrack_state is not None
                    else 0
                )
                completed, retry = alg.same_anchor_recovery_retry_decision(
                    completed, True, controls.m_max
                )
                if retry:
                    post_backtrack_state = alg.make_recovery_backtrack_state(
                        direct_path,
                        direct_energies,
                        direct_gradients,
                        rejected_descriptor,
                        completed,
                    )
                else:
                    post_backtrack_state = None
            else:
                post_backtrack_state = None

        working_state = copy_state(public_state)
        working_memories = alg.clone_physical_memories(public_memories)
        recovery_commit_streak = 0
        expanding_provisional = False
        bootstrap_active = True
        movement = float(
            np.max(
                np.linalg.norm(
                    public_state[0] - outer_start[0], axis=1
                )
            )
        )
        if movement <= 1.0e-14:
            if post_backtrack_state is not None:
                continue
            if stationary_exhaustions < int(controls.m_max):
                # Retry only while the optimizer itself remains stationary;
                # the existing m_max is the sole retry limit.
                stationary_exhaustions += 1
                bootstrap_active = True
                recovery_commit_streak = 0
                continue
            stop_reason = "stationary"
            break

    if provisional_depth:
        working_state = copy_state(public_state)
        working_memories = alg.clone_physical_memories(public_memories)
        provisional_rollbacks += 1
        provisional_depth = 0
    per_image = np.asarray([item.calls for item in evaluators], dtype=np.int64)
    if int(np.sum(per_image)) != int(ledger.total):
        raise RuntimeError("per-image ledger does not close")
    if int(sum(nof_breakdown.values())) != int(ledger.total):
        raise RuntimeError("NoF category ledger does not close")
    return {
        "controls": controls,
        "q_matrix": np.asarray(potential.Q),
        "records": records,
        "recovery_audits": recovery_audits,
        "recovery_states": recovery_states,
        "main_audits": main_audits,
        "memories": alg.clone_physical_memories(public_memories),
        "terminal_nf": int(ledger.total),
        "stop_reason": stop_reason,
        "per_image_joint_evaluations": per_image,
        "nof_breakdown": dict(nof_breakdown),
        "evaluation_miss_trace": [dict(item) for item in ledger.evaluation_miss_trace],
        "held_recovery_steps": int(held_trials),
        "trusted_nonmonotone_commits": int(trusted_nonmonotone_commits),
        "provisional_rollbacks": int(provisional_rollbacks),
        "max_provisional_depth": int(max_provisional_depth),
        "working_equals_public_at_exit": bool(
            all(
                np.array_equal(left, right)
                for left, right in zip(working_state, public_state)
            )
        ),
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, choices=(10, 20), required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--budget", type=int, default=3000)
    args = parser.parse_args()
    result = run_clean(
        args.n, args.alpha, args.r, args.m, budget=args.budget
    )
    exit_audits = [
        audit for audit in result["recovery_audits"]
        if audit.get("reason") == "normal_stream_exit_hj_spring"
    ]
    payload = {
        "n_internal": int(args.n),
        "alpha": float(args.alpha),
        "r": float(args.r),
        "m_max": int(args.m),
        "terminal_nf": int(result["terminal_nf"]),
        "stop_reason": result["stop_reason"],
        "nof_breakdown": result["nof_breakdown"],
        "per_image_nof_sum": int(np.sum(
            result["per_image_joint_evaluations"]
        )),
        "exit_spring_decisions": int(len(exit_audits)),
        "exit_spring_zero_nof_rejections": int(sum(
            not audit["spring_evaluated"] for audit in exit_audits
        )),
        "exit_spring_evaluations": int(sum(
            audit["spring_evaluated"] for audit in exit_audits
        )),
        "exit_spring_commits": int(sum(
            audit["spring_accepted"] for audit in exit_audits
        )),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
