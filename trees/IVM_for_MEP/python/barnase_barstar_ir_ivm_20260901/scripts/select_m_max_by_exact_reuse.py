#!/usr/bin/env python3
"""Select m_max using exact-path reuse of the sealed r-scan diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_discrete_iperp_densified_barrier import sha256_array, sha256_file
from run_barnase_barstar_ir_ivm_batch import validate_result
from run_barnase_barstar_ir_ivm_member import member_label
from select_alpha_from_principal_diagnostics import pareto_nondominated


CAPS = (5, 8, 10, 15, 20, 50)
ALPHA = 10.0
RATIO = 0.1
BUDGET = 480


def unique_result(root: Path, cap: int) -> Path:
    label = member_label(ALPHA, RATIO, cap, BUDGET, "r1")
    matches = sorted((root / "runs" / "screen").glob(f"ivm_{label}_*.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one result for {label}, got {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--diagnostic-audit", type=Path, required=True)
    parser.add_argument("--r-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    diagnostics = args.diagnostics.expanduser().resolve(strict=True)
    diagnostic_audit = args.diagnostic_audit.expanduser().resolve(strict=True)
    r_selection = args.r_selection.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    audit = json.loads(diagnostic_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("output_sha256") != sha256_file(diagnostics):
        raise ValueError("source r diagnostics failed their audit")
    if int(audit.get("diagnostic_force_requests", -1)) != 0 or int(
        audit.get("optimization_N_F_contribution", -1)
    ) != 0:
        raise ValueError("source diagnostics contaminate optimization Number of Forces")
    r_receipt = json.loads(r_selection.read_text(encoding="utf-8"))
    if r_receipt.get("status") != "PASS_R_CONTROLLED_SCAN_TWO_METRIC_SELECTION":
        raise ValueError("r selection has not passed")
    if float(r_receipt["r_star_for_m_max_controlled_scan"]) != RATIO:
        raise ValueError("m_max scan r does not match selected r_star")
    with np.load(diagnostics, allow_pickle=False) as saved:
        labels = np.asarray(saved["label"]).astype(str)
        path_sha = np.asarray(saved["selected_path_sha256"]).astype(str)
        nf = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        i_perp = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
        delta_e = np.asarray(saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64)
    initial = np.flatnonzero(labels == "common_initial_path")
    if initial.size != 1:
        raise ValueError("diagnostics require one common initial path")
    i0 = float(i_perp[int(initial[0])])
    e0 = float(delta_e[int(initial[0])])
    metric_by_path = {}
    for index, digest in enumerate(path_sha):
        row = (int(nf[index]), float(i_perp[index]), float(delta_e[index]))
        previous = metric_by_path.setdefault(str(digest), row)
        if previous != row:
            raise ValueError("one exact path has inconsistent diagnostic values")
    per_state = []
    run_bindings = []
    selected_by_cap = []
    for order, cap in enumerate(CAPS):
        label = member_label(ALPHA, RATIO, cap, BUDGET, "r1")
        result = unique_result(root, cap)
        validate_result(result, expected_label=label, expected_budget=BUDGET)
        result_audit = result.with_suffix(".audit.json")
        registration = root / "registrations" / "screen" / f"{label}.json"
        authorization = root / "registrations" / "execution_authorizations" / f"{label}.json"
        with np.load(result, allow_pickle=False) as saved:
            calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
            stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
            committed = np.asarray(saved["reparam_committed"], dtype=bool)
            counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
            stored_paths = np.asarray(saved["stored_paths"], dtype=np.float64)
            observed_i = np.asarray(saved["i_perp"], dtype=np.float64)
            actual_nf = int(np.asarray(saved["number_of_forces"]).item())
            stop_reasons = np.asarray(saved["discrete_audit_stop_reason"]).astype(str)
        candidates = []
        for stored_index, call in enumerate(stored_calls):
            matches = np.flatnonzero(calls == int(call))
            if matches.size == 0:
                raise ValueError("stored call is absent from history")
            history_index = int(matches[-1])
            if int(call) == 1 or not bool(committed[history_index]):
                continue
            digest = sha256_array(stored_paths[stored_index])
            if digest not in metric_by_path:
                raise ValueError(f"eligible m_max={cap} path lacks exact diagnostic reuse")
            diagnostic_nf, diagnostic_i, diagnostic_e = metric_by_path[digest]
            observed_nf = int(counters[history_index].sum(dtype=np.int64))
            if observed_nf != diagnostic_nf or float(observed_i[history_index]) != diagnostic_i:
                raise ValueError("optimization state and reused diagnostic metadata disagree")
            row = {
                "m_max": cap,
                "label": label,
                "history_index": history_index,
                "calls_per_image": int(call),
                "optimization_N_F": observed_nf,
                "discrete_I_perp": diagnostic_i,
                "densified_Delta_E_kcal_per_mol": diagnostic_e,
                "selected_path_sha256": digest,
                "normalized_discrete_I_perp": diagnostic_i / i0,
                "normalized_densified_Delta_E": diagnostic_e / e0,
                "registered_m_max_order": order,
            }
            row["joint_quantity"] = max(
                row["normalized_discrete_I_perp"], row["normalized_densified_Delta_E"]
            )
            candidates.append(row)
            per_state.append(row)
        if not candidates:
            raise ValueError(f"m_max={cap} has no eligible committed state")
        selected_by_cap.append(
            dict(min(candidates, key=lambda row: (
                row["joint_quantity"], row["normalized_discrete_I_perp"],
                row["normalized_densified_Delta_E"], row["optimization_N_F"],
            )))
        )
        values, counts = np.unique(stop_reasons, return_counts=True)
        run_bindings.append({
            "m_max": cap,
            "label": label,
            "registered_optimization_N_F_upper_bound": BUDGET,
            "actual_optimization_N_F": actual_nf,
            "result": str(result),
            "result_sha256": sha256_file(result),
            "result_audit": str(result_audit),
            "result_audit_sha256": sha256_file(result_audit),
            "member_registration": str(registration),
            "member_registration_sha256": sha256_file(registration),
            "execution_authorization": str(authorization),
            "execution_authorization_sha256": sha256_file(authorization),
            "eligible_committed_state_count": len(candidates),
            "stop_reason_counts": {str(v): int(n) for v, n in zip(values, counts)},
        })
    for row, flag in zip(selected_by_cap, pareto_nondominated(selected_by_cap)):
        row["Pareto_nondominated_among_m_max_values"] = bool(flag)
    ordered = sorted(selected_by_cap, key=lambda row: (
        row["joint_quantity"], row["normalized_discrete_I_perp"],
        row["normalized_densified_Delta_E"], row["optimization_N_F"],
        row["registered_m_max_order"],
    ))
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    payload = {
        "schema": "bb-irivm-m-max-controlled-scan-selection-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "PASS_M_MAX_CONTROLLED_SCAN_TWO_METRIC_SELECTION",
        "fixed_alpha": ALPHA,
        "fixed_r": RATIO,
        "registered_budget_upper_bound": BUDGET,
        "m_star_for_local_interaction": int(ordered[0]["m_max"]),
        "m_2_for_local_interaction": int(ordered[1]["m_max"]),
        "common_initial_normalization": {"discrete_I_perp": i0, "densified_Delta_E_kcal_per_mol": e0},
        "per_state_results": per_state,
        "per_m_max_selected_state": ordered,
        "run_bindings": run_bindings,
        "r_selection": str(r_selection),
        "r_selection_sha256": sha256_file(r_selection),
        "reused_diagnostics": str(diagnostics),
        "reused_diagnostics_sha256": sha256_file(diagnostics),
        "reused_diagnostic_audit": str(diagnostic_audit),
        "reused_diagnostic_audit_sha256": sha256_file(diagnostic_audit),
        "new_diagnostic_energy_attempts": 0,
        "new_diagnostic_force_requests": 0,
        "optimization_N_F_contribution_from_diagnostics": 0,
        "exact_path_reuse_required_and_satisfied": True,
        "same_state_rule_satisfied": True,
        "raw_diagnostics_are_primary": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output.chmod(0o444)
    print(f"Saved {output}")
    print(f"sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
