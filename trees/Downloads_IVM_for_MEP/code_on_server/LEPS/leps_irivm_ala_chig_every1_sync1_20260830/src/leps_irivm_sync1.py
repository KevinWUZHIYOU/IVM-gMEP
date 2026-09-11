#!/usr/bin/env python3
"""LEPS period-one IR-IVM with one synchronized recovery trial.

The parent period-one method and its full direct spring proposal are retained.
Only recovery changes: direct gradients form one complete-band normal proposal,
a global step is chosen from the proximal regularized prediction without new
force calls, and the complete band is evaluated once before the unchanged
bicriteria commit or atomic rollback.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


THIS_FILE = Path(__file__).resolve()
CAMPAIGN_ROOT = THIS_FILE.parents[1]
LEPS_ROOT = THIS_FILE.parents[2]
PARENT_SOURCE = (
    LEPS_ROOT
    / "leps_irivm_ala_chig_every1_20260830/src/leps_irivm_every1.py"
)
spec = importlib.util.spec_from_file_location(
    "_frozen_leps_ala_chig_every1_for_sync1", PARENT_SOURCE
)
if spec is None or spec.loader is None:
    raise ImportError(PARENT_SOURCE)
period1 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = period1
spec.loader.exec_module(period1)
parent = period1.parent
base = parent.base

EXPERIMENT_ID = "leps-irivm-ala-chig-every1-sync1-grid-20260830"
REGISTRATION = CAMPAIGN_ROOT / "registrations/CAMPAIGN_PREREGISTRATION.json"
ARTICLE_NF_BUDGET = parent.ARTICLE_NF_BUDGET
ALPHAS = parent.ALPHAS
R_VALUES = parent.R_VALUES
M_MAX_VALUES = parent.M_MAX_VALUES
GRID = parent.GRID
REPARAMETERIZATION_PERIOD = 1
TRANSACTION_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT = 2
RECOVERY_JOINT_EVALUATION_CAP = base.N_INTERNAL

parent.THIS_FILE = THIS_FILE
parent.CAMPAIGN_ROOT = CAMPAIGN_ROOT
parent.EXPERIMENT_ID = EXPERIMENT_ID
parent.REGISTRATION = REGISTRATION
parent.REPARAMETERIZATION_PERIOD = REPARAMETERIZATION_PERIOD
for module in (parent.current, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.CAMPAIGN_ROOT = CAMPAIGN_ROOT
    module.REGISTRATION = REGISTRATION
    module.ARTICLE_NF_BUDGET = ARTICLE_NF_BUDGET
    module.ALPHAS = ALPHAS
    module.R_VALUES = R_VALUES
    module.M_MAX_VALUES = M_MAX_VALUES
    module.GRID = GRID
base.__file__ = str(THIS_FILE)

Controls = parent.Controls
ContractError = parent.ContractError
BudgetReached = parent.BudgetReached
validate_preregistration = parent.validate_preregistration


def single_synchronized_recovery(
    pre,
    direct,
    evaluators,
    main_memories,
    controls,
    residual,
    attempt_start_calls,
):
    """Evaluate at most one synchronized complete-band recovery proposal."""

    pre_path, _, _ = pre
    path, energies, gradients = (np.asarray(item).copy() for item in direct)
    pre_i, _ = base.article_i_perp(*pre)
    direct_i, _ = base.article_i_perp(*direct)
    pre_eq = base.equidistance_integral(pre_path)
    direct_eq = base.equidistance_integral(path)
    ceiling = max(
        direct_eq,
        parent.RECOVERY_EQUIDISTANCE_CONTRACTION * pre_eq,
    )
    attempt_start_calls = np.asarray(attempt_start_calls, dtype=np.int64)
    recovery_trust = parent.rms_segment_trust_radius(
        pre_path,
        fraction=parent.TRUST_RADIUS_FRACTION,
        maximum=parent.TRUST_RADIUS_MAXIMUM,
    )
    transaction = {
        "mode": "ala_chig_single_synchronized_regularized_prediction",
        "spring_residual": float(residual),
        "pre_i_perp": float(pre_i),
        "direct_i_perp": float(direct_i),
        "pre_equidistance": float(pre_eq),
        "direct_equidistance": float(direct_eq),
        "equidistance_ceiling": float(ceiling),
        "main_r_read_by_recovery": False,
        "regularized_prediction_uses_main_r": False,
        "regularized_prediction_alpha_eff": float(controls.alpha),
        "transaction_call_cap_per_image_including_direct": (
            TRANSACTION_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT
        ),
        "single_sync_recovery_joint_evaluation_cap": RECOVERY_JOINT_EVALUATION_CAP,
        "total_atomic_trust_radius": float(recovery_trust),
        "i_perp_absolute_tolerance": parent.RECOVERY_IPERP_ATOL,
        "equidistance_absolute_tolerance": parent.RECOVERY_EQUIDISTANCE_ATOL,
        "cycles": 0,
        "merged_images": 0,
        "sync_trial_evaluated": False,
        "regularized_step_length": 0.0,
        "predicted_regularized_decrease": 0.0,
        "committed": False,
        "rolled_back": False,
        "interrupted": False,
    }

    def calls_now():
        return np.asarray([evaluator.calls for evaluator in evaluators], dtype=np.int64)

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

    def recovered(state):
        value, _ = base.article_i_perp(*state)
        return parent._i_ok(value, pre_i) and parent._eq_ok(
            base.equidistance_integral(state[0]), ceiling
        )

    if recovered((path, energies, gradients)):
        transaction["committed"] = True
        for memory in main_memories:
            memory.clear()
        return (path, energies, gradients), finish()

    tangents = base.improved_tangents(path, energies)
    directions = np.zeros_like(path)
    g_dot_d = 0.0
    norm_squared = 0.0
    for image_index in range(1, base.N_IMAGES - 1):
        tangent = tangents[image_index - 1]
        basis = base.normal_basis(tangent)
        reduced_gradient = basis.T @ gradients[image_index]
        apply_h0 = base._initial_inverse(basis, controls.alpha)
        reduced_direction = base._lbfgs_direction(
            reduced_gradient,
            [],
            apply_h0,
        )
        direction = basis @ reduced_direction
        largest = parent._max_atomic_norm(direction)
        if largest > recovery_trust:
            direction *= recovery_trust / largest
        directions[image_index] = direction
        g_dot_d += float(np.dot(gradients[image_index], direction))
        norm_squared += float(np.dot(direction, direction))

    if norm_squared <= np.finfo(np.float64).tiny or g_dot_d >= 0.0:
        transaction["rolled_back"] = True
        return pre, finish()

    alpha_eff = float(controls.alpha)
    step_length = float(np.clip(-alpha_eff * g_dot_d / norm_squared, 0.0, 1.0))
    predicted_decrease = float(
        -step_length * g_dot_d
        - (step_length * step_length * norm_squared) / (2.0 * alpha_eff)
    )
    transaction["regularized_step_length"] = step_length
    transaction["predicted_regularized_decrease"] = predicted_decrease
    if step_length <= 0.0 or predicted_decrease <= 0.0:
        transaction["rolled_back"] = True
        return pre, finish()

    candidate_path = path + step_length * directions
    candidate_path[0] = path[0]
    candidate_path[-1] = path[-1]
    try:
        candidate_energies, candidate_gradients = base.evaluate_path(
            candidate_path,
            evaluators,
        )
    except BudgetReached:
        transaction["interrupted"] = True
        transaction["rolled_back"] = True
        return pre, finish()
    transaction["sync_trial_evaluated"] = True
    transaction["cycles"] = 1
    merged, accepted = parent.bicriteria_merge(
        (path, energies, gradients),
        (candidate_path, candidate_energies, candidate_gradients),
        ceiling,
    )
    transaction["merged_images"] = int(np.count_nonzero(accepted))
    if recovered(merged):
        transaction["committed"] = True
        for memory in main_memories:
            memory.clear()
        return merged, finish()
    transaction["rolled_back"] = True
    return pre, finish()


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
            "Ala/Chignolin single synchronized regularized-prediction recovery"
        ),
        "recovery_reads_main_r": np.asarray(False),
        "reparameterization_period": np.asarray(REPARAMETERIZATION_PERIOD),
        "transaction_call_cap_per_image_including_direct": np.asarray(
            TRANSACTION_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT
        ),
        "single_sync_recovery_joint_evaluation_cap": np.asarray(
            RECOVERY_JOINT_EVALUATION_CAP
        ),
        "trust_radius_fraction": np.asarray(parent.TRUST_RADIUS_FRACTION),
        "trust_radius_maximum": np.asarray(parent.TRUST_RADIUS_MAXIMUM),
        "optimizer_reads_baseline": np.asarray(False),
        "optimizer_reads_reference_barrier": np.asarray(False),
        "optimizer_reads_densified_delta_e": np.asarray(False),
        "stop_reason": np.asarray(result["stop_reason"]),
        "failure_message": np.asarray(result["failure_message"]),
        "terminal_article_nf": np.asarray(result["terminal_article_nf"], dtype=np.int64),
        "blocked_misses": np.asarray(result["blocked_misses"], dtype=np.int64),
        "per_image_joint_evaluations": per_image,
        "article_nf": np.asarray([item["article_nf"] for item in records], dtype=np.int64),
        "outer": np.asarray([item["outer"] for item in records], dtype=np.int64),
        "state_phase": np.asarray([item["state_phase"] for item in records]),
        "path_history": np.asarray([item["path"] for item in records]),
        "energy_history": np.asarray([item["energies"] for item in records]),
        "gradient_history": np.asarray([item["gradients"] for item in records]),
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


parent.ala_chig_recovery = single_synchronized_recovery
parent.atomic_save_result = atomic_save_result
run = parent.run
run_one = parent.run_one


if __name__ == "__main__":
    raise SystemExit("use run_grid.py")
