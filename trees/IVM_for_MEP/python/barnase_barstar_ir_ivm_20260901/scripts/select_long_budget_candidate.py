#!/usr/bin/env python3
"""Select the compliant long-budget IR-IVM setting from frozen diagnostics."""

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


def baseline_row(audit: dict, optimization_nf: int) -> dict:
    matches = [
        row for row in audit.get("records", [])
        if int(row.get("optimization_N_F", -1)) == optimization_nf
    ]
    require(len(matches) == 1, f"expected one baseline row at N_F={optimization_nf}")
    row = matches[0]
    path_sha = row.get(
        "selected_path_sha256",
        row.get("official_path32_sha256", row.get("official_state_sha256")),
    )
    require(path_sha is not None, "baseline row lacks a path/state hash")
    return {
        "optimization_N_F": optimization_nf,
        "discrete_I_perp": float(row["discrete_I_perp"]),
        "densified_Delta_E_kcal_per_mol": float(
            row["densified_Delta_E_kcal_per_mol"]
        ),
        "path_sha256": str(path_sha),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require(not output.exists(), f"refusing to overwrite {output}")
    paths = {
        "long_budget_design": root / "registrations" / "BB_IRIVM_LONG_BUDGET_DESIGN_20260902_R1.json",
        "candidate_A_failure": root / "reports" / "failures" / "BB_IRIVM_A1E1_R05_M05_INF4080_R2_ALGORITHMIC_EARLY_STOP_20260902.json",
        "candidate_B_verification": root / "reports" / "verifications" / "BB_IRIVM_A1E0_R01_M05_INF4080_R1_VERIFICATION_20260902.json",
        "A8_trigger_decision": root / "reports" / "decisions" / "BB_IRIVM_CACHED_STALL_FOLLOWUP_DECISION_20260902_R1.json",
        "candidate_B_metric_contract": root / "registrations" / "diagnostics" / "BB_IRIVM_CANDIDATE_B_LONG_ARC500_20260902_R1.json",
        "candidate_B_metric_output": root / "runs" / "diagnostics" / "bb_irivm_candidate_b_long_arc500_20260902_r1.npz",
        "candidate_B_metric_audit": root / "runs" / "diagnostics" / "bb_irivm_candidate_b_long_arc500_20260902_r1.audit.json",
        "original_IVM_metric_audit": root / "runs" / "diagnostics" / "bb_original_ivm_arc500_20260901_r1.audit.json",
        "ODE_NEB_metric_audit": root / "runs" / "diagnostics" / "bb_ode_neb_arc500_preview_20260901_r1.audit.json",
        "metric_policy": root / "registrations" / "USER_METRIC_POLICY_AMENDMENT_20260901_R1.json",
        "prefix_criterion": root / "registrations" / "BB_IRIVM_LONG_PREFIX_REPRODUCIBILITY_CRITERION_20260902_R1.json",
    }
    for role, path in paths.items():
        require(path.is_file(), f"missing {role}: {path}")

    design = json.loads(paths["long_budget_design"].read_text(encoding="utf-8"))
    failure = json.loads(paths["candidate_A_failure"].read_text(encoding="utf-8"))
    verification = json.loads(paths["candidate_B_verification"].read_text(encoding="utf-8"))
    trigger = json.loads(paths["A8_trigger_decision"].read_text(encoding="utf-8"))
    audit = json.loads(paths["candidate_B_metric_audit"].read_text(encoding="utf-8"))
    old_audit = json.loads(paths["original_IVM_metric_audit"].read_text(encoding="utf-8"))
    ode_audit = json.loads(paths["ODE_NEB_metric_audit"].read_text(encoding="utf-8"))
    require(design.get("status") == "FROZEN_BEFORE_FIRST_4080_PHYSICAL_CALL", "long design status mismatch")
    require("ALGORITHMIC_EARLY_STOP" in json.dumps(failure, sort_keys=True), "candidate A class mismatch")
    require(verification.get("status") == "PASS", "candidate B verification failed")
    require(trigger.get("status") == "A8_NOT_TRIGGERED", "A8 decision mismatch")
    require(audit.get("status") == "PASS", "candidate B metric audit failed")
    require(int(audit.get("diagnostic_force_requests", -1)) == 0, "metric audit requested force")
    require(int(audit.get("optimization_N_F_contribution", -1)) == 0, "metric audit changed optimization N_F")
    require(audit.get("contract_sha256") == sha256(paths["candidate_B_metric_contract"]), "metric contract binding mismatch")
    require(audit.get("output_sha256") == sha256(paths["candidate_B_metric_output"]), "metric output binding mismatch")

    target_i = float(design["ranking_targets"]["discrete_I_perp"])
    target_e = float(design["ranking_targets"]["densified_Delta_E_kcal_per_mol"])
    rows = []
    for record in audit.get("records", []):
        if record.get("label") == "common_initial_path":
            continue
        i_value = float(record["discrete_I_perp"])
        e_value = float(record["densified_Delta_E_kcal_per_mol"])
        rows.append(
            {
                "optimization_N_F": int(record["optimization_N_F"]),
                "discrete_I_perp": i_value,
                "densified_Delta_E_kcal_per_mol": e_value,
                "joint_target_score": max(i_value / target_i, e_value / target_e),
                "path_sha256": str(record["selected_path_sha256"]),
                "diagnostic_energy_reused": bool(record["diagnostic_energy_reused"]),
            }
        )
    require([row["optimization_N_F"] for row in rows] == [210, 429, 1993], "unexpected B metric rows")
    best = min(
        rows,
        key=lambda row: (
            row["joint_target_score"],
            row["discrete_I_perp"],
            row["densified_Delta_E_kcal_per_mol"],
            row["optimization_N_F"],
        ),
    )
    require(best["optimization_N_F"] == 1993, "unexpected best same-state B path")

    old = baseline_row(old_audit, 9990)
    ode = baseline_row(ode_audit, 9990)
    dominance = {}
    for name, baseline in (("current_BB_IVM", old), ("ODE_NEB", ode)):
        dominates = (
            best["optimization_N_F"] < baseline["optimization_N_F"]
            and best["discrete_I_perp"] < baseline["discrete_I_perp"]
            and best["densified_Delta_E_kcal_per_mol"]
            < baseline["densified_Delta_E_kcal_per_mol"]
        )
        require(dominates, f"B does not dominate the audited {name} N_F=9990 point")
        dominance[name] = {
            "baseline": baseline,
            "strictly_dominated_in_N_F_discrete_I_perp_and_densified_Delta_E": True,
            "baseline_to_IR_IVM_cost_ratio": baseline["optimization_N_F"] / best["optimization_N_F"],
            "baseline_to_IR_IVM_I_perp_ratio": baseline["discrete_I_perp"] / best["discrete_I_perp"],
            "baseline_to_IR_IVM_Delta_E_ratio": baseline["densified_Delta_E_kcal_per_mol"] / best["densified_Delta_E_kcal_per_mol"],
        }

    payload = {
        "schema": "bb-irivm-long-budget-two-metric-selection-v1",
        "status": "PASS_SELECTED_SETTING_FOR_INDEPENDENT_9990_N_F_RUN",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidate_classes": {
            "alpha_10_r_0p5_m_max_5": "ALGORITHMIC_EARLY_STOP_NOT_RANKABLE",
            "alpha_1_r_0p1_m_max_5": "COMPLIANT_LONG_BUDGET_WITH_VALID_TWO_METRIC_DIAGNOSTIC",
        },
        "selected_setting": {
            "alpha": 1.0,
            "r": 0.1,
            "m_max": 5,
            "independent_optimization_N_F_upper_bound": 9990,
            "fresh_from_common_initial_path": True,
            "reference_host": design["reference_host"],
        },
        "same_state_rows": rows,
        "selected_4080_run_state": best,
        "ranking_targets": {
            "discrete_I_perp": target_i,
            "densified_Delta_E_kcal_per_mol": target_e,
        },
        "selection_reason": (
            "The alpha=1, r=0.1, m_max=5 trajectory is the only compliant "
            "long-budget candidate and has a valid same-state two-metric diagnostic."
        ),
        "audited_preview_dominance": dominance,
        "preview_disclosure": (
            "This three-objective dominance uses the best eligible N_F=1993 state "
            "from the 4080-upper-bound run.  The independent 9990 run and exact "
            "4080-prefix reproducibility test remain required for the final claim."
        ),
        "bindings": {
            role: {"path": str(path), "sha256": sha256(path)}
            for role, path in paths.items()
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
