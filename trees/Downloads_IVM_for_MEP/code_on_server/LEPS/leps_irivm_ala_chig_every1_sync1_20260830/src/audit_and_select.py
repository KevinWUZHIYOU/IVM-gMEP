#!/usr/bin/env python3
"""Independent method, NoF, and full-course audit for sync-one recovery."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


THIS_FILE = Path(__file__).resolve()
CAMPAIGN_ROOT = THIS_FILE.parents[1]
LEPS_ROOT = THIS_FILE.parents[2]
REGISTRATION = CAMPAIGN_ROOT / "registrations/CAMPAIGN_PREREGISTRATION.json"
METRIC_SOURCE = LEPS_ROOT / "leps_irivm_ala_chig_20260829/src/audit_and_select.py"
spec = importlib.util.spec_from_file_location(
    "_independent_leps_sync1_metrics", METRIC_SOURCE
)
if spec is None or spec.loader is None:
    raise ImportError(METRIC_SOURCE)
metric = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = metric
spec.loader.exec_module(metric)
metric.CAMPAIGN_ROOT = CAMPAIGN_ROOT
metric.LEPS_ROOT = LEPS_ROOT
metric.BASELINE = (
    LEPS_ROOT
    / "runs/ivm/ivm_production_alpha30_tight_reparam_cfg29674d42197b_budget500.npz"
)
metric.INITIAL = LEPS_ROOT / "inputs/common_initial_path.npz"
metric.REGISTRATION = REGISTRATION

ALLOWED_PHASES = {"initial", "complete_main_stage", "reparameterization_committed"}
EXPECTED_RECOVERY = "Ala/Chignolin single synchronized regularized-prediction recovery"


def scalar(saved, key):
    return saved[key].item()


def preflight(path, registration):
    with np.load(path, allow_pickle=False) as saved:
        if str(scalar(saved, "experiment_id")) != registration["campaign_id"]:
            raise RuntimeError(f"experiment ID drift: {path}")
        if str(scalar(saved, "source_sha256")) != registration["implementation_sha256"]:
            raise RuntimeError(f"source hash drift: {path}")
        if str(scalar(saved, "registration_sha256")) != metric.sha256(REGISTRATION):
            raise RuntimeError(f"registration hash drift: {path}")
        if str(scalar(saved, "main_subproblem")) != (
            "complete-normal proximal Armijo L-BFGS; LEPS total atomic trust"
        ):
            raise RuntimeError(f"main subproblem drift: {path}")
        if str(scalar(saved, "main_state_globalization")) != (
            "I_perp monotone atomic commit"
        ):
            raise RuntimeError(f"main globalization drift: {path}")
        if str(scalar(saved, "recovery_definition")) != EXPECTED_RECOVERY:
            raise RuntimeError(f"recovery definition drift: {path}")
        if int(scalar(saved, "reparameterization_period")) != 1:
            raise RuntimeError(f"reparameterization period drift: {path}")
        if int(scalar(saved, "transaction_call_cap_per_image_including_direct")) != 2:
            raise RuntimeError(f"transaction per-image cap drift: {path}")
        if int(scalar(saved, "single_sync_recovery_joint_evaluation_cap")) != 20:
            raise RuntimeError(f"single-sync joint cap drift: {path}")
        if bool(scalar(saved, "recovery_reads_main_r")):
            raise RuntimeError(f"recovery read main r: {path}")
        for key in (
            "optimizer_reads_baseline",
            "optimizer_reads_reference_barrier",
            "optimizer_reads_densified_delta_e",
        ):
            if bool(scalar(saved, key)):
                raise RuntimeError(f"forbidden optimizer input {key}: {path}")

        nof = np.asarray(saved["article_nf"], dtype=np.int64)
        phases = [str(value) for value in np.asarray(saved["state_phase"])]
        paths = np.asarray(saved["path_history"], dtype=float)
        energies = np.asarray(saved["energy_history"], dtype=float)
        gradients = np.asarray(saved["gradient_history"], dtype=float)
        per_image = np.asarray(saved["per_image_joint_evaluations"], dtype=np.int64)
        terminal_nf = int(scalar(saved, "terminal_article_nf"))
        direct_nf = np.asarray(saved["direct_trial_nf"], dtype=np.int64)
        transactions = [
            json.loads(str(value)) for value in np.asarray(saved["transaction_json"])
        ]
        main_audits = [
            json.loads(str(value)) for value in np.asarray(saved["main_stage_audit_json"])
        ]

    if np.any(np.diff(nof) <= 0):
        raise RuntimeError(f"published state NoF is not strictly increasing: {path}")
    if phases[0] != "initial" or int(nof[0]) != 20 or set(phases) - ALLOWED_PHASES:
        raise RuntimeError(f"published state contract drift: {path}")
    if int(np.sum(per_image)) != terminal_nf:
        raise RuntimeError(f"Number-of-Forces ledger does not close: {path}")
    if len(transactions) != direct_nf.size or len(transactions) != len(main_audits):
        raise RuntimeError(f"not every completed outer stage attempted recovery: {path}")
    expected_outer = list(range(1, len(main_audits) + 1))
    if [int(item["outer"]) for item in main_audits] != expected_outer:
        raise RuntimeError(f"main-stage sequence drift: {path}")
    if [int(item["outer"]) for item in transactions] != expected_outer:
        raise RuntimeError(f"transaction schedule is not every outer stage: {path}")

    for transaction in transactions:
        if transaction.get("main_r_read_by_recovery") is not False:
            raise RuntimeError(f"transaction recovery-r drift: {path}")
        if transaction.get("regularized_prediction_uses_main_r") is not False:
            raise RuntimeError(f"regularized predictor read main r: {path}")
        direct_calls = int(transaction["direct_joint_evaluations"])
        recovery_calls = int(transaction["recovery_joint_evaluations"])
        total_calls = int(transaction["transaction_joint_evaluations"])
        if total_calls != direct_calls + recovery_calls:
            raise RuntimeError(f"transaction NoF subtotal does not close: {path}")
        if direct_calls < 0 or direct_calls > 20:
            raise RuntimeError(f"direct proposal exceeded one band evaluation: {path}")
        if recovery_calls < 0 or recovery_calls > 20:
            raise RuntimeError(f"recovery exceeded one band evaluation: {path}")
        if total_calls > 40:
            raise RuntimeError(f"transaction exceeded the 40-NoF cap: {path}")
        if int(transaction["cycles"]) not in (0, 1):
            raise RuntimeError(f"more than one synchronized recovery cycle: {path}")
        if bool(transaction["sync_trial_evaluated"]) != (recovery_calls > 0):
            if not bool(transaction["interrupted"]):
                raise RuntimeError(f"sync-trial evaluation/counter mismatch: {path}")

    for index, candidate_path in enumerate(paths):
        checked_energy, checked_gradient = metric.energies_and_gradients(candidate_path)
        if not np.allclose(energies[index], checked_energy, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"stored energy mismatch: {path}")
        if not np.allclose(gradients[index], checked_gradient, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"stored gradient mismatch: {path}")


def main():
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    run_paths = sorted((CAMPAIGN_ROOT / "runs").glob("*.npz"))
    expected_count = int(registration["registered_grid_member_count"])
    if len(run_paths) != expected_count:
        raise RuntimeError(f"expected {expected_count} results, found {len(run_paths)}")
    for path in run_paths:
        preflight(path, registration)

    metric_exit = 0
    try:
        metric.main()
    except SystemExit as error:
        metric_exit = int(error.code or 0)

    manifest_path = CAMPAIGN_ROOT / "reports/acceptance/selection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["method_contract_audit"] = {
        "passed": True,
        "audited_result_count": len(run_paths),
        "published_states": sorted(ALLOWED_PHASES),
        "reparameterization_period_complete_main_stages": 1,
        "every_completed_outer_stage_has_one_transaction": True,
        "maximum_direct_joint_evaluations_per_transaction": 20,
        "maximum_recovery_joint_evaluations_per_transaction": 20,
        "maximum_transaction_joint_evaluations": 40,
        "recovery_reads_main_r": False,
        "optimizer_reads_article_baseline_or_acceptance_metrics": False,
        "stored_energy_and_gradient_independently_recomputed": True,
        "audit_source": str(THIS_FILE.relative_to(LEPS_ROOT)),
        "audit_source_sha256": metric.sha256(THIS_FILE),
        "independent_metric_source": str(METRIC_SOURCE.relative_to(LEPS_ROOT)),
        "independent_metric_source_sha256": metric.sha256(METRIC_SOURCE),
    }
    metric.atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if metric_exit:
        raise SystemExit(metric_exit)


if __name__ == "__main__":
    main()
