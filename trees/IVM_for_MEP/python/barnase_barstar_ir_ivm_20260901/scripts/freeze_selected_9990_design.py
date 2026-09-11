#!/usr/bin/env python3
"""Freeze the independently restarted 9990-N_F confirmation design."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    selection_path = args.selection.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    require(not output.exists(), f"refusing to overwrite {output}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    require(
        selection.get("status") == "PASS_SELECTED_SETTING_FOR_INDEPENDENT_9990_N_F_RUN",
        "long-budget selection has not passed",
    )
    setting = selection["selected_setting"]
    require(
        (float(setting["alpha"]), float(setting["r"]), int(setting["m_max"]))
        == (1.0, 0.1, 5),
        "unexpected selected setting",
    )
    require(int(setting["independent_optimization_N_F_upper_bound"]) == 9990, "9990 budget mismatch")
    require(setting.get("fresh_from_common_initial_path") is True, "fresh-start requirement missing")

    paths = {
        "selection": selection_path,
        "campaign_preregistration": root / "registrations" / "CAMPAIGN_PREREGISTRATION.json",
        "staged_search_protocol": root / "registrations" / "STAGED_SEARCH_PROTOCOL.json",
        "migration_contract": root / "contracts" / "MIGRATION_CONTRACT.json",
        "efficient_scan_amendment": root / "registrations" / "USER_EFFICIENT_CONTROLLED_SCAN_AMENDMENT_20260901_R1.json",
        "metric_policy_amendment": root / "registrations" / "USER_METRIC_POLICY_AMENDMENT_20260901_R1.json",
        "budget_atomicity_amendment": root / "registrations" / "USER_BUDGET_ATOMICITY_AMENDMENT_20260901_R1.json",
        "long_budget_design": root / "registrations" / "BB_IRIVM_LONG_BUDGET_DESIGN_20260902_R1.json",
        "prefix_reproducibility_criterion": root / "registrations" / "BB_IRIVM_LONG_PREFIX_REPRODUCIBILITY_CRITERION_20260902_R1.json",
        "candidate_B_4080_verification": root / "reports" / "verifications" / "BB_IRIVM_A1E0_R01_M05_INF4080_R1_VERIFICATION_20260902.json",
        "candidate_B_arc500_audit": root / "runs" / "diagnostics" / "bb_irivm_candidate_b_long_arc500_20260902_r1.audit.json",
        "common_initial_path": root / "inputs" / "common_initial_path32.npz",
        "topology": root / "inputs" / "complex_fixed.pdb",
        "endpoint_cache": root / "inputs" / "endpoint_joint_ef_cache.npz",
    }
    for role, path in paths.items():
        require(path.is_file(), f"missing {role}: {path}")
    campaign = json.loads(paths["campaign_preregistration"].read_text(encoding="utf-8"))
    require(int(campaign["production_optimization_internal_nf_budget"]) == 9990, "campaign production budget mismatch")
    require(int(campaign["disk_free_gate_bytes"]["production"]) == 8 * 1024**3, "production disk gate mismatch")

    source_files = (
        "bb_core.py",
        "bb_ffprecon.py",
        "storage_policy.py",
        "run_ivm_v9_spring_projected_production_20260730.py",
        "frozen_discrete_b040.py",
        "run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py",
        "run_ivm_v9_discrete_selected_eta_shim.py",
        "run_ivm_v9_discrete_ffrbk_takeover_grid_shim.py",
        "run_ivm_v9_total_internal_nf_budget.py",
        "run_barnase_barstar_ir_ivm_member.py",
    )
    source_closure = {
        name: sha256(root / "src" / name) for name in source_files
    }
    payload = {
        "schema": "bb-irivm-selected-9990-design-v1",
        "status": "FROZEN_BEFORE_FIRST_9990_PHYSICAL_CALL",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selected_setting": {
            "alpha": 1.0,
            "r": 0.1,
            "m_max": 5,
            "eta": 1.1,
            "history": 7,
            "physical_max_step_angstrom": 1.0,
            "optimization_Number_of_Forces_upper_bound": 9990,
        },
        "execution": {
            "tier": "production",
            "attempt_tag": "r1",
            "fresh_from_common_initial_path": True,
            "warm_start_forbidden": True,
            "reference_host": setting["reference_host"],
            "jobs": 1,
            "all_thread_caps": 1,
            "one_BB_task_on_host": True,
            "two_host_local_fcntl_locks_required_before_child": True,
            "minimum_free_disk_bytes_before_child": int(
                campaign["disk_free_gate_bytes"]["production"]
            ),
            "fixed_endpoint_optimization_Number_of_Forces": 0,
            "diagnostic_feedback_during_optimization": "forbidden",
        },
        "algorithm_identity": {
            "migration_rule": "the frozen Chignolin IR-IVM with only the molecular system and registered alpha, r, m_max controls changed",
            "reparameterization": "state_curve every outer iteration with per_image direct acceptance and recovery",
            "tangent": "normalized central chord",
            "path_storage": "reparameterization-committed states",
            "forbidden_changes": [
                "globalization rule",
                "reparameterization or recovery rule",
                "eta",
                "history length",
                "physical maximum step",
                "central tangent convention",
                "Number-of-Forces definition",
            ],
        },
        "required_post_run_gates": [
            "zero-OpenMM long-member verification",
            "exact 4080 eligible-prefix reproducibility against the reference-host candidate-B run",
            "same-state discrete I_perp and uniform-arc-length-500 densified Delta E",
            "necessary ODE-NEB comparison-path diagnostics only",
        ],
        "bindings": {
            role: {"path": str(path), "sha256": sha256(path)}
            for role, path in paths.items()
        },
        "source_closure_sha256": source_closure,
        "input_closure_sha256": {
            "complex_fixed.pdb": sha256(paths["topology"]),
            "common_initial_path32.npz": sha256(paths["common_initial_path"]),
            "endpoint_joint_ef_cache.npz": sha256(paths["endpoint_cache"]),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    print(f"Saved {output}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
