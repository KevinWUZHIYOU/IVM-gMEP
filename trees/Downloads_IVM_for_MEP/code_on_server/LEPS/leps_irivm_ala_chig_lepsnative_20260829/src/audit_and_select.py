#!/usr/bin/env python3
"""Independent method-contract, NoF, and full-course two-metric audit."""

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
BASELINE = (
    LEPS_ROOT
    / "runs/ivm/ivm_production_alpha30_tight_reparam_cfg29674d42197b_budget500.npz"
)
INITIAL = LEPS_ROOT / "inputs/common_initial_path.npz"
INDEPENDENT_METRIC_SOURCE = (
    LEPS_ROOT / "leps_irivm_ala_chig_20260829/src/audit_and_select.py"
)

spec = importlib.util.spec_from_file_location(
    "_independent_leps_metric_audit", INDEPENDENT_METRIC_SOURCE
)
if spec is None or spec.loader is None:
    raise ImportError(INDEPENDENT_METRIC_SOURCE)
metric_audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = metric_audit
spec.loader.exec_module(metric_audit)
metric_audit.CAMPAIGN_ROOT = CAMPAIGN_ROOT
metric_audit.LEPS_ROOT = LEPS_ROOT
metric_audit.BASELINE = BASELINE
metric_audit.INITIAL = INITIAL
metric_audit.REGISTRATION = REGISTRATION


EXPECTED_MAIN_SUBPROBLEM = (
    "complete-normal proximal Armijo L-BFGS; LEPS total atomic trust"
)
EXPECTED_MAIN_GLOBALIZATION = "I_perp monotone atomic commit"
EXPECTED_RECOVERY = "Ala/Chignolin fixed synchronized bicriteria recovery"
ALLOWED_PHASES = {
    "initial",
    "complete_main_stage",
    "reparameterization_committed",
}


def scalar(saved, key):
    return saved[key].item()


def preflight_result(path, registration):
    with np.load(path, allow_pickle=False) as saved:
        if str(scalar(saved, "experiment_id")) != registration["campaign_id"]:
            raise RuntimeError(f"experiment ID drift: {path}")
        if str(scalar(saved, "main_subproblem")) != EXPECTED_MAIN_SUBPROBLEM:
            raise RuntimeError(f"main subproblem drift: {path}")
        if str(scalar(saved, "main_state_globalization")) != EXPECTED_MAIN_GLOBALIZATION:
            raise RuntimeError(f"main globalization drift: {path}")
        if str(scalar(saved, "recovery_definition")) != EXPECTED_RECOVERY:
            raise RuntimeError(f"recovery definition drift: {path}")
        if int(scalar(saved, "reparameterization_period")) != 5:
            raise RuntimeError(f"reparameterization period drift: {path}")
        if int(scalar(saved, "recovery_call_cap_per_image_including_direct")) != 320:
            raise RuntimeError(f"recovery cap drift: {path}")
        if float(scalar(saved, "trust_radius_fraction")) != 2.0:
            raise RuntimeError(f"trust fraction drift: {path}")
        if float(scalar(saved, "trust_radius_maximum")) != 1.0:
            raise RuntimeError(f"trust maximum drift: {path}")
        if bool(scalar(saved, "recovery_reads_main_r")):
            raise RuntimeError(f"recovery read the tuned main r: {path}")
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
        direct_nf = np.asarray(saved["direct_trial_nf"], dtype=np.int64)
        transactions = [
            json.loads(str(value)) for value in np.asarray(saved["transaction_json"])
        ]
        main_audits = [
            json.loads(str(value)) for value in np.asarray(saved["main_stage_audit_json"])
        ]

    if np.any(np.diff(nof) <= 0):
        raise RuntimeError(f"published state NoF must be strictly increasing: {path}")
    if set(phases) - ALLOWED_PHASES:
        raise RuntimeError(f"unregistered public state phase: {path}")
    if phases[0] != "initial" or int(nof[0]) != 20:
        raise RuntimeError(f"initial-state contract drift: {path}")
    if len(transactions) != direct_nf.size:
        raise RuntimeError(f"direct/recovery transaction count drift: {path}")
    if any(int(item["outer"]) % 5 != 0 for item in transactions):
        raise RuntimeError(f"recovery occurred outside the fixed five-stage schedule: {path}")
    if any(item.get("main_r_read_by_recovery") is not False for item in transactions):
        raise RuntimeError(f"transaction-level recovery-r drift: {path}")
    if any(
        int(item["transaction_joint_evaluations"])
        != int(item["direct_joint_evaluations"])
        + int(item["recovery_joint_evaluations"])
        for item in transactions
    ):
        raise RuntimeError(f"transaction NoF subtotal does not close: {path}")
    if any(
        int(item["terminal_article_nf"]) < 20
        or int(item["outer"]) != index + 1
        for index, item in enumerate(main_audits)
    ):
        raise RuntimeError(f"main-stage audit sequence drift: {path}")

    for index, candidate_path in enumerate(paths):
        checked_energy, checked_gradient = metric_audit.energies_and_gradients(candidate_path)
        if not np.allclose(energies[index], checked_energy, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"stored energy does not match independent evaluation: {path}")
        if not np.allclose(gradients[index], checked_gradient, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"stored gradient does not match independent evaluation: {path}")


def main():
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    run_paths = sorted((CAMPAIGN_ROOT / "runs").glob("*.npz"))
    expected_count = int(registration["registered_grid_member_count"])
    if len(run_paths) != expected_count:
        raise RuntimeError(f"expected {expected_count} results, found {len(run_paths)}")
    for run_path in run_paths:
        preflight_result(run_path, registration)

    # This source contains no optimizer imports.  The numerical metric and
    # strict same-NoF comparison are delegated to the frozen independent LEPS
    # audit source, whose hash is itself preregistered.
    metric_audit.main()

    manifest_path = CAMPAIGN_ROOT / "reports/acceptance/selection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["method_contract_audit"] = {
        "passed": True,
        "audited_result_count": len(run_paths),
        "published_states": sorted(ALLOWED_PHASES),
        "reparameterization_period_complete_main_stages": 5,
        "recovery_call_cap_per_image_including_direct": 320,
        "recovery_reads_main_r": False,
        "optimizer_reads_article_baseline_or_acceptance_metrics": False,
        "stored_energy_and_gradient_independently_recomputed": True,
        "audit_source": str(THIS_FILE.relative_to(LEPS_ROOT)),
        "audit_source_sha256": metric_audit.sha256(THIS_FILE),
        "independent_metric_source": str(INDEPENDENT_METRIC_SOURCE.relative_to(LEPS_ROOT)),
        "independent_metric_source_sha256": metric_audit.sha256(
            INDEPENDENT_METRIC_SOURCE
        ),
    }
    metric_audit.atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
