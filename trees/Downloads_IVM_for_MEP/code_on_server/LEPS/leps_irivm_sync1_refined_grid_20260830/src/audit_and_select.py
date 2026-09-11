#!/usr/bin/env python3
"""Independent cache-aware method and metric audit for the refined grid."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


THIS_FILE = Path(__file__).resolve()
CAMPAIGN_ROOT = THIS_FILE.parents[1]
LEPS_ROOT = THIS_FILE.parents[2]
REGISTRATION = CAMPAIGN_ROOT / "registrations/CAMPAIGN_PREREGISTRATION.json"
FROZEN_AUDIT = (
    LEPS_ROOT
    / "leps_irivm_ala_chig_every1_sync1_20260830/src/audit_and_select_cacheaware.py"
)
spec = importlib.util.spec_from_file_location("_frozen_sync1_cacheaware_audit", FROZEN_AUDIT)
if spec is None or spec.loader is None:
    raise ImportError(FROZEN_AUDIT)
frozen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = frozen
spec.loader.exec_module(frozen)

# Retain the frozen method preflight and independent LEPS metric definitions;
# only bind their input/output locations to the refined preregistered grid.
frozen.CAMPAIGN_ROOT = CAMPAIGN_ROOT
frozen.REGISTRATION = REGISTRATION
metric = frozen.original.metric
metric.CAMPAIGN_ROOT = CAMPAIGN_ROOT
metric.LEPS_ROOT = LEPS_ROOT
metric.BASELINE = (
    LEPS_ROOT
    / "runs/ivm/ivm_production_alpha30_tight_reparam_cfg29674d42197b_budget500.npz"
)
metric.INITIAL = LEPS_ROOT / "inputs/common_initial_path.npz"
metric.REGISTRATION = REGISTRATION


def main():
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    run_paths = sorted((CAMPAIGN_ROOT / "runs").glob("*.npz"))
    expected = int(registration["registered_grid_member_count"])
    if len(run_paths) != expected:
        raise RuntimeError(f"expected {expected} results, found {len(run_paths)}")
    for path in run_paths:
        frozen.preflight(path, registration)

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
        "every_completed_outer_stage_has_one_direct_reparam_transaction": True,
        "maximum_direct_joint_evaluations_per_transaction": 20,
        "maximum_recovery_joint_evaluations_per_transaction": 20,
        "maximum_transaction_joint_evaluations": 40,
        "fully_cached_sync_trial_may_add_zero_nof": True,
        "recovery_reads_main_r": False,
        "optimizer_reads_article_baseline_or_acceptance_metrics": False,
        "stored_energy_and_gradient_independently_recomputed": True,
        "audit_source": str(THIS_FILE.relative_to(LEPS_ROOT)),
        "audit_source_sha256": metric.sha256(THIS_FILE),
        "frozen_audit_source": str(FROZEN_AUDIT.relative_to(LEPS_ROOT)),
        "frozen_audit_source_sha256": metric.sha256(FROZEN_AUDIT),
    }
    metric.atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if metric_exit:
        raise SystemExit(metric_exit)


if __name__ == "__main__":
    main()
