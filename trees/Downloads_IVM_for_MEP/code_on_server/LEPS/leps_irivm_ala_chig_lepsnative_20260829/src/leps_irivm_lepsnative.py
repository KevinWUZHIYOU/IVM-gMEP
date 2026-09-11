#!/usr/bin/env python3
"""Preregistered LEPS-native Ala/Chignolin-style IR-IVM.

The three exposed controls are ``alpha``, main-subproblem ``r``, and
``m_max``.  Every tangent is the Henkelman--Jonsson improved tangent.  The
main step is the complete-normal proximal Armijo L-BFGS subproblem used by
the current IR-IVM, with the LEPS article's geometry-derived *total* atomic
trust radius in place of the transferred QM10 per-inner-step cap.

Every fifth completed main stage starts the frozen Ala/Chignolin
reparameterization transaction: identical-spring direct proposal, fixed
recovery KKT fraction, synchronized one-step recovery traces, bicriteria
merge, and atomic commit or charged rollback.  Recovery never reads the
tuned main ``r`` and has no QM10 ``r``-routing controls.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np


THIS_FILE = Path(__file__).resolve()
CAMPAIGN_ROOT = THIS_FILE.parents[1]
LEPS_ROOT = THIS_FILE.parents[2]
CURRENT_SOURCE = (
    LEPS_ROOT / "leps_irivm_ala_chig_20260829/src/leps_irivm_ala_chig.py"
)
spec = importlib.util.spec_from_file_location("_leps_current_irivm_primitives", CURRENT_SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(CURRENT_SOURCE)
current = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = current
spec.loader.exec_module(current)
base = current.base

LEPS_SRC = LEPS_ROOT / "src"
if str(LEPS_SRC) not in sys.path:
    sys.path.insert(0, str(LEPS_SRC))
from leps_core import rms_segment_trust_radius  # noqa: E402


EXPERIMENT_ID = "leps-irivm-ala-chig-lepsnative-grid-20260829"
REGISTRATION = CAMPAIGN_ROOT / "registrations/CAMPAIGN_PREREGISTRATION.json"
ARTICLE_NF_BUDGET = 3000

# Clean, Cartesian, preregistered grid.  These are the only tuned controls.
ALPHAS = (10.0, 100.0, 1000.0)
R_VALUES = (0.001, 0.01, 0.1, 0.3)
M_MAX_VALUES = (4, 5, 6, 8)
GRID = tuple(
    (alpha, r, m_max)
    for alpha in ALPHAS
    for r in R_VALUES
    for m_max in M_MAX_VALUES
)

# Fixed method constants, inherited from the article LEPS scale and the
# Ala/Chignolin recovery transaction.  None is searched or metric-dependent.
REPARAMETERIZATION_PERIOD = 5
TRUST_RADIUS_FRACTION = 2.0
TRUST_RADIUS_MAXIMUM = 1.0
SPRING_DTAU = 0.05
SPRING_MAX_ITER = 500
SPRING_TOLERANCE = 1e-5
RECOVERY_KKT_FRACTION = 0.8
RECOVERY_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT = 320
RECOVERY_INNER_CALLS = 15
RECOVERY_EQUIDISTANCE_CONTRACTION = 0.2
RECOVERY_BLOCK_SIZE = 8
RECOVERY_IPERP_ATOL = 1e-12
RECOVERY_EQUIDISTANCE_ATOL = 1e-14

# Point the inherited contract primitives at this immutable formal campaign.
for module in (current, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.CAMPAIGN_ROOT = CAMPAIGN_ROOT
    module.REGISTRATION = REGISTRATION
    module.ARTICLE_NF_BUDGET = ARTICLE_NF_BUDGET
    module.ALPHAS = ALPHAS
    module.R_VALUES = R_VALUES
    module.M_MAX_VALUES = M_MAX_VALUES
    module.GRID = GRID
base.__file__ = str(THIS_FILE)

Controls = base.Controls
ContractError = base.ContractError
BudgetReached = base.BudgetReached
CertificationFailure = base.CertificationFailure
validate_preregistration = base.validate_preregistration


def _rounding_slack(left, right):
    return float(
        base.ROUNDING_FACTOR
        * np.finfo(np.float64).eps
        * max(1.0, abs(float(left)), abs(float(right)))
    )


def _max_atomic_norm(vector):
    return float(
        np.max(np.linalg.norm(np.asarray(vector, dtype=np.float64).reshape(-1, 3), axis=1))
    )


def solve_normal_image(
    evaluator,
    reference,
    tangent,
    controls,
    memory,
    *,
    trust_radius,
    kkt_fraction=None,
    accepted_step_cap=None,
    new_call_cap=None,
):
    """Complete-normal proximal Armijo L-BFGS under one total trust ball."""

    accepted_step_cap = int(
        controls.m_max if accepted_step_cap is None else accepted_step_cap
    )
    new_call_cap = int(
        max(12, 4 * int(controls.m_max)) if new_call_cap is None else new_call_cap
    )
    fraction = float(controls.r if kkt_fraction is None else kkt_fraction)
    trust_radius = float(trust_radius)
    basis = base.normal_basis(tangent)
    displacement = np.zeros(base.DIMENSION - 1, dtype=np.float64)
    reference_energy, _ = evaluator.query(reference)
    value, reduced, energy, gradient, position = base._proximal_query(
        evaluator, reference, basis, controls.alpha, displacement, reference_energy
    )
    zero_value = float(value)
    initial_kkt = float(np.linalg.norm(reduced))
    target = max(base.KKT_ATOL, fraction * initial_kkt)
    start_calls = evaluator.calls
    apply_h0 = base._initial_inverse(basis, controls.alpha)
    warm_pairs = base._usable_pairs(memory, basis, controls.alpha)
    local_pairs = []
    accepted = 0
    backtracks = 0
    hit_boundary = False

    while accepted < accepted_step_cap:
        if np.linalg.norm(reduced) <= target and accepted >= 1:
            break
        if evaluator.calls - start_calls >= new_call_cap:
            break
        pairs = (warm_pairs + local_pairs)[-base.HISTORY_SIZE :]
        direction = base._lbfgs_direction(reduced, pairs, apply_h0)
        directional = float(np.dot(reduced, direction))
        if directional >= 0.0:
            warm_pairs = []
            local_pairs = []
            direction = -np.asarray(apply_h0(reduced))
            directional = float(np.dot(reduced, direction))
        if directional >= 0.0:
            break

        accepted_trial = False
        step_length = 1.0
        for _ in range(64):
            if evaluator.calls - start_calls >= new_call_cap:
                break
            trial_displacement = displacement + step_length * direction
            largest = _max_atomic_norm(basis @ trial_displacement)
            if largest > trust_radius:
                trial_displacement *= trust_radius / largest
                hit_boundary = True
            effective = trial_displacement - displacement
            effective_directional = float(np.dot(reduced, effective))
            if np.linalg.norm(effective) <= 1e-18 or effective_directional >= 0.0:
                step_length *= base.BACKTRACK
                backtracks += 1
                continue
            trial = base._proximal_query(
                evaluator,
                reference,
                basis,
                controls.alpha,
                trial_displacement,
                reference_energy,
            )
            trial_value, trial_reduced, trial_energy, trial_gradient, trial_position = trial
            slack = _rounding_slack(value, trial_value)
            if (
                trial_value <= value + base.ARMIJO * effective_directional + slack
                and trial_value <= zero_value + slack
            ):
                old_displacement = displacement.copy()
                old_reduced = reduced.copy()
                old_position = position.copy()
                old_gradient = gradient.copy()
                displacement = trial_displacement
                value = float(trial_value)
                reduced = np.asarray(trial_reduced)
                energy = float(trial_energy)
                gradient = np.asarray(trial_gradient)
                position = np.asarray(trial_position)
                s = displacement - old_displacement
                y = reduced - old_reduced
                if np.dot(s, y) > 1e-6 * np.linalg.norm(s) * np.linalg.norm(y):
                    local_pairs.append((s.copy(), y.copy()))
                    local_pairs[:] = local_pairs[-base.HISTORY_SIZE :]
                physical_s = position - old_position
                physical_y = gradient - old_gradient
                if np.dot(physical_s, physical_y) > (
                    -0.99 * np.dot(physical_s, physical_s) / controls.alpha
                ):
                    memory.append((physical_s.copy(), physical_y.copy()))
                    memory[:] = memory[-base.HISTORY_SIZE :]
                accepted += 1
                accepted_trial = True
                break
            step_length *= base.BACKTRACK
            backtracks += 1
            if step_length < 2.0**-40:
                break
        if not accepted_trial:
            break

    if value > zero_value + _rounding_slack(zero_value, value):
        raise CertificationFailure("normal proximal objective increased")
    return position, energy, gradient, {
        "initial_kkt": initial_kkt,
        "final_kkt": float(np.linalg.norm(reduced)),
        "accepted_steps": accepted,
        "new_calls": evaluator.calls - start_calls,
        "backtracks": backtracks,
        "target_fraction": fraction,
        "total_atomic_trust_radius": trust_radius,
        "hit_boundary": hit_boundary,
    }


def solve_main_stage(path, energies, gradients, evaluators, memories, controls):
    """Solve one frozen-band main stage and apply Ala state globalization."""

    tangents = base.improved_tangents(path, energies)
    trust_radius = rms_segment_trust_radius(
        path, fraction=TRUST_RADIUS_FRACTION, maximum=TRUST_RADIUS_MAXIMUM
    )
    proposed = np.asarray(path).copy()
    proposed_energies = np.asarray(energies).copy()
    proposed_gradients = np.asarray(gradients).copy()
    audits = []
    for image_index in range(1, base.N_IMAGES - 1):
        position, energy, gradient, audit = solve_normal_image(
            evaluators[image_index],
            path[image_index],
            tangents[image_index - 1],
            controls,
            memories[image_index - 1],
            trust_radius=trust_radius,
        )
        proposed[image_index] = position
        proposed_energies[image_index] = energy
        proposed_gradients[image_index] = gradient
        audits.append(audit)

    reference_i, _ = base.article_i_perp(path, energies, gradients)
    candidate_i, _ = base.article_i_perp(proposed, proposed_energies, proposed_gradients)
    rolled_back = bool(candidate_i > reference_i)
    if rolled_back:
        for memory in memories:
            memory.clear()
        proposed = np.asarray(path).copy()
        proposed_energies = np.asarray(energies).copy()
        proposed_gradients = np.asarray(gradients).copy()
    return proposed, proposed_energies, proposed_gradients, {
        "trust_radius": float(trust_radius),
        "pre_i_perp": float(reference_i),
        "candidate_i_perp": float(candidate_i),
        "rolled_back_by_i_perp_guard": rolled_back,
        "image_audits": audits,
    }


def improved_spring_reparameterize(path, energies):
    """Ala/Chignolin identical-spring flow, using improved tangents only."""

    result = np.asarray(path, dtype=np.float64).copy()
    energies = np.asarray(energies, dtype=np.float64)
    residual = np.inf
    for _ in range(SPRING_MAX_ITER):
        lengths = np.linalg.norm(np.diff(result, axis=0), axis=1)
        tangents = base.improved_tangents(result, energies)
        spring_force = (lengths[1:] - lengths[:-1])[:, None] * tangents
        residual = float(np.max(np.linalg.norm(spring_force, axis=1)))
        if residual <= SPRING_TOLERANCE:
            break
        result[1:-1] += SPRING_DTAU * spring_force
    result[0] = path[0]
    result[-1] = path[-1]
    return result, residual


def _i_ok(value, reference):
    return float(value) <= float(reference) + RECOVERY_IPERP_ATOL


def _eq_ok(value, ceiling):
    return float(value) <= float(ceiling) + RECOVERY_EQUIDISTANCE_ATOL


def bicriteria_merge(current_state, candidate_state, ceiling):
    """Frozen Ala/Chignolin collective-first, then contiguous-block merge."""

    path, energies, gradients = (np.asarray(item).copy() for item in current_state)
    cpath, cenergies, cgradients = candidate_state
    eligible = np.linalg.norm(cpath[1:-1] - path[1:-1], axis=1) > 1e-14
    current_i, _ = base.article_i_perp(path, energies, gradients)
    accepted = np.zeros(base.N_INTERNAL, dtype=bool)
    if np.any(eligible):
        trial = (path.copy(), energies.copy(), gradients.copy())
        images = np.flatnonzero(eligible) + 1
        trial[0][images] = cpath[images]
        trial[1][images] = cenergies[images]
        trial[2][images] = cgradients[images]
        trial_i, _ = base.article_i_perp(*trial)
        if _i_ok(trial_i, current_i) and _eq_ok(
            base.equidistance_integral(trial[0]), ceiling
        ):
            accepted[eligible] = True
            return trial, accepted

    remaining = set(np.flatnonzero(eligible).tolist())
    while remaining:
        best = None
        best_i = np.inf
        best_eq = np.inf
        for size in range(1, min(RECOVERY_BLOCK_SIZE, base.N_INTERNAL) + 1):
            for start in range(0, base.N_INTERNAL - size + 1):
                indices = [index for index in range(start, start + size) if index in remaining]
                if not indices:
                    continue
                trial = (path.copy(), energies.copy(), gradients.copy())
                images = np.asarray(indices, dtype=int) + 1
                trial[0][images] = cpath[images]
                trial[1][images] = cenergies[images]
                trial[2][images] = cgradients[images]
                trial_i, _ = base.article_i_perp(*trial)
                trial_eq = base.equidistance_integral(trial[0])
                if not _i_ok(trial_i, current_i) or not _eq_ok(trial_eq, ceiling):
                    continue
                if trial_i < best_i - 1e-18 or (
                    abs(trial_i - best_i) <= 1e-18 and trial_eq < best_eq
                ):
                    best = (indices, trial)
                    best_i = float(trial_i)
                    best_eq = float(trial_eq)
        if best is None:
            break
        indices, (path, energies, gradients) = best
        current_i = best_i
        accepted[np.asarray(indices, dtype=int)] = True
        remaining.difference_update(indices)
    return (path, energies, gradients), accepted


def ala_chig_recovery(
    pre,
    direct,
    evaluators,
    main_memories,
    controls,
    residual,
    attempt_start_calls,
):
    """Ala/Chignolin recovery with atomic commit/rollback and charged trials."""

    pre_path, _, _ = pre
    path, energies, gradients = (np.asarray(item).copy() for item in direct)
    pre_i, _ = base.article_i_perp(*pre)
    direct_i, _ = base.article_i_perp(*direct)
    pre_eq = base.equidistance_integral(pre_path)
    direct_eq = base.equidistance_integral(path)
    ceiling = max(direct_eq, RECOVERY_EQUIDISTANCE_CONTRACTION * pre_eq)
    attempt_start_calls = np.asarray(attempt_start_calls, dtype=int)
    recovery_memories = [[] for _ in range(base.N_INTERNAL)]
    recovery_trust = rms_segment_trust_radius(
        pre_path, fraction=TRUST_RADIUS_FRACTION, maximum=TRUST_RADIUS_MAXIMUM
    )
    transaction = {
        "mode": "ala_chig_synchronized_bicriteria",
        "spring_residual": float(residual),
        "pre_i_perp": float(pre_i),
        "direct_i_perp": float(direct_i),
        "pre_equidistance": float(pre_eq),
        "direct_equidistance": float(direct_eq),
        "equidistance_ceiling": float(ceiling),
        "main_r_read_by_recovery": False,
        "fixed_recovery_kkt_fraction": RECOVERY_KKT_FRACTION,
        "fixed_recovery_call_cap_per_image_including_direct": (
            RECOVERY_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT
        ),
        "fixed_recovery_inner_calls": RECOVERY_INNER_CALLS,
        "total_atomic_trust_radius": float(recovery_trust),
        "i_perp_absolute_tolerance": RECOVERY_IPERP_ATOL,
        "equidistance_absolute_tolerance": RECOVERY_EQUIDISTANCE_ATOL,
        "cycles": 0,
        "merged_images": 0,
        "committed": False,
        "rolled_back": False,
        "interrupted": False,
    }

    def calls_now():
        return np.asarray([evaluator.calls for evaluator in evaluators], dtype=int)

    direct_vector = calls_now() - attempt_start_calls
    transaction["direct_joint_evaluations"] = int(np.sum(direct_vector))

    def finish():
        total_vector = calls_now() - attempt_start_calls
        transaction["transaction_joint_evaluations"] = int(np.sum(total_vector))
        transaction["recovery_joint_evaluations"] = int(
            transaction["transaction_joint_evaluations"]
            - transaction["direct_joint_evaluations"]
        )
        return transaction

    def recovered():
        value, _ = base.article_i_perp(path, energies, gradients)
        return _i_ok(value, pre_i) and _eq_ok(
            base.equidistance_integral(path), ceiling
        )

    while not recovered():
        used = calls_now() - attempt_start_calls
        remaining = RECOVERY_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT - int(
            np.max(used[1:-1])
        )
        allowed = min(RECOVERY_INNER_CALLS, remaining)
        if allowed <= 0:
            break
        tangents = base.improved_tangents(path, energies)
        candidate_path = path.copy()
        candidate_energies = energies.copy()
        candidate_gradients = gradients.copy()
        any_calls = False
        recovery_controls = SimpleNamespace(
            alpha=float(controls.alpha), r=RECOVERY_KKT_FRACTION, m_max=1
        )
        try:
            for image_index in range(1, base.N_IMAGES - 1):
                before = evaluators[image_index].calls
                position, energy, gradient, _ = solve_normal_image(
                    evaluators[image_index],
                    path[image_index],
                    tangents[image_index - 1],
                    recovery_controls,
                    recovery_memories[image_index - 1],
                    trust_radius=recovery_trust,
                    kkt_fraction=RECOVERY_KKT_FRACTION,
                    accepted_step_cap=1,
                    new_call_cap=allowed,
                )
                any_calls = any_calls or evaluators[image_index].calls > before
                candidate_path[image_index] = position
                candidate_energies[image_index] = energy
                candidate_gradients[image_index] = gradient
        except BudgetReached:
            transaction["interrupted"] = True
            transaction["rolled_back"] = True
            return pre, finish()
        if not any_calls:
            break
        (path, energies, gradients), merged = bicriteria_merge(
            (path, energies, gradients),
            (candidate_path, candidate_energies, candidate_gradients),
            ceiling,
        )
        transaction["cycles"] += 1
        transaction["merged_images"] += int(np.count_nonzero(merged))
        if not np.any(merged):
            break

    if recovered():
        transaction["committed"] = True
        for memory in main_memories:
            memory.clear()
        return (path, energies, gradients), finish()
    transaction["rolled_back"] = True
    return pre, finish()


def append_record(records, nf, outer, phase, path, energies, gradients):
    record = base.state_record(nf, outer, phase, path, energies, gradients)
    if records and int(records[-1]["article_nf"]) == int(nf):
        records[-1] = record
    else:
        records.append(record)


def run(controls: Controls, *, budget=ARTICLE_NF_BUDGET, outer_limit=10000):
    controls.validate()
    validate_preregistration()
    if int(budget) != ARTICLE_NF_BUDGET or int(outer_limit) != 10000:
        raise ContractError("formal budget/outer safeguard drift")
    with np.load(LEPS_ROOT / "inputs/common_initial_path.npz", allow_pickle=False) as saved:
        path = np.asarray(saved["initial_path"], dtype=np.float64)
    ledger = base.Ledger(limit=int(budget))
    evaluators = base.make_evaluators(ledger)
    energies, gradients = base.evaluate_path(path, evaluators)
    memories = [[] for _ in range(base.N_INTERNAL)]
    records = [base.state_record(ledger.total, 0, "initial", path, energies, gradients)]
    transactions = []
    main_stage_audits = []
    direct_paths, direct_energies, direct_gradients, direct_nf = [], [], [], []
    stop_reason, failure_message = "outer_limit", ""

    for outer in range(1, outer_limit + 1):
        start_nf = ledger.total
        old_path = path.copy()
        try:
            path, energies, gradients, stage_audit = solve_main_stage(
                path, energies, gradients, evaluators, memories, controls
            )
            stage_audit["outer"] = outer
            stage_audit["terminal_article_nf"] = int(ledger.total)
            main_stage_audits.append(stage_audit)
            append_record(
                records,
                ledger.total,
                outer,
                "complete_main_stage",
                path,
                energies,
                gradients,
            )

            if outer % REPARAMETERIZATION_PERIOD == 0:
                pre = (path.copy(), energies.copy(), gradients.copy())
                attempt_start_calls = np.asarray(
                    [evaluator.calls for evaluator in evaluators], dtype=int
                )
                direct_path, residual = improved_spring_reparameterize(path, energies)
                direct_energy, direct_gradient = base.evaluate_path(direct_path, evaluators)
                direct_paths.append(direct_path.copy())
                direct_energies.append(direct_energy.copy())
                direct_gradients.append(direct_gradient.copy())
                direct_nf.append(int(ledger.total))
                updated, transaction = ala_chig_recovery(
                    pre,
                    (direct_path, direct_energy, direct_gradient),
                    evaluators,
                    memories,
                    controls,
                    residual,
                    attempt_start_calls,
                )
                transaction["outer"] = outer
                transactions.append(transaction)
                path, energies, gradients = updated
                if transaction["committed"]:
                    append_record(
                        records,
                        ledger.total,
                        outer,
                        "reparameterization_committed",
                        path,
                        energies,
                        gradients,
                    )
                elif transaction["interrupted"]:
                    raise BudgetReached("Ala/Chignolin recovery interrupted")

            movement = float(np.max(np.linalg.norm(path - old_path, axis=1)))
            if movement <= 1e-14:
                stop_reason = "stationary"
                break
        except BudgetReached:
            stop_reason = "budget"
            break
        except (CertificationFailure, FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            stop_reason = "numerical_failure"
            failure_message = f"{type(exc).__name__}: {exc}"
            break
        if ledger.total == start_nf:
            stop_reason = "stationary"
            break

    return {
        "controls": controls,
        "records": records,
        "transactions": transactions,
        "main_stage_audits": main_stage_audits,
        "direct_paths": direct_paths,
        "direct_energies": direct_energies,
        "direct_gradients": direct_gradients,
        "direct_nf": direct_nf,
        "stop_reason": stop_reason,
        "failure_message": failure_message,
        "terminal_article_nf": int(ledger.total),
        "blocked_misses": int(ledger.blocked_misses),
        "per_image_joint_evaluations": np.asarray(
            [evaluator.calls for evaluator in evaluators], dtype=np.int64
        ),
    }


def atomic_save_result(result, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = result["records"]
    controls = result["controls"]
    transactions = result["transactions"]
    per_image = np.asarray(result["per_image_joint_evaluations"], dtype=np.int64)
    if int(np.sum(per_image)) != int(result["terminal_article_nf"]):
        raise ContractError("per-image Number-of-Forces ledger does not close")
    payload = {
        "experiment_id": np.asarray(EXPERIMENT_ID),
        "run_id": np.asarray(controls.run_id),
        "config_json": np.asarray(json.dumps(controls.as_dict(), sort_keys=True)),
        "source_sha256": np.asarray(base._sha256(THIS_FILE)),
        "registration_sha256": np.asarray(base._sha256(REGISTRATION)),
        "tangent_definition": np.asarray("Henkelman-Jonsson improved everywhere"),
        "main_subproblem": np.asarray(
            "complete-normal proximal Armijo L-BFGS; LEPS total atomic trust"
        ),
        "main_state_globalization": np.asarray("I_perp monotone atomic commit"),
        "recovery_definition": np.asarray(
            "Ala/Chignolin fixed synchronized bicriteria recovery"
        ),
        "recovery_reads_main_r": np.asarray(False),
        "reparameterization_period": np.asarray(REPARAMETERIZATION_PERIOD),
        "recovery_call_cap_per_image_including_direct": np.asarray(
            RECOVERY_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT
        ),
        "trust_radius_fraction": np.asarray(TRUST_RADIUS_FRACTION),
        "trust_radius_maximum": np.asarray(TRUST_RADIUS_MAXIMUM),
        "optimizer_reads_baseline": np.asarray(False),
        "optimizer_reads_reference_barrier": np.asarray(False),
        "optimizer_reads_densified_delta_e": np.asarray(False),
        "stop_reason": np.asarray(result["stop_reason"]),
        "failure_message": np.asarray(result["failure_message"]),
        "terminal_article_nf": np.asarray(result["terminal_article_nf"], dtype=np.int64),
        "blocked_misses": np.asarray(result["blocked_misses"], dtype=np.int64),
        "per_image_joint_evaluations": per_image,
        "article_nf": np.asarray([record["article_nf"] for record in records], dtype=np.int64),
        "outer": np.asarray([record["outer"] for record in records], dtype=np.int64),
        "state_phase": np.asarray([record["state_phase"] for record in records]),
        "path_history": np.asarray([record["path"] for record in records]),
        "energy_history": np.asarray([record["energies"] for record in records]),
        "gradient_history": np.asarray([record["gradients"] for record in records]),
        "direct_trial_nf": np.asarray(result["direct_nf"], dtype=np.int64),
        "direct_trial_path_history": np.asarray(
            result["direct_paths"], dtype=np.float64
        ).reshape((-1, base.N_IMAGES, base.DIMENSION)),
        "direct_trial_energy_history": np.asarray(
            result["direct_energies"], dtype=np.float64
        ).reshape((-1, base.N_IMAGES)),
        "direct_trial_gradient_history": np.asarray(
            result["direct_gradients"], dtype=np.float64
        ).reshape((-1, base.N_IMAGES, base.DIMENSION)),
        "transaction_json": np.asarray(
            [json.dumps(item, sort_keys=True) for item in transactions]
        ),
        "main_stage_audit_json": np.asarray(
            [json.dumps(item, sort_keys=True) for item in result["main_stage_audits"]]
        ),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp.npz", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_one(controls):
    destination = CAMPAIGN_ROOT / "runs" / f"{controls.run_id}.npz"
    if destination.exists():
        raise ContractError(f"refusing overwrite: {destination}")
    result = run(controls)
    atomic_save_result(result, destination)
    return destination, result


if __name__ == "__main__":
    raise SystemExit("use run_grid.py")
