#!/usr/bin/env python3
"""Freeze the five-setting r-controlled scan before arc-length-500 diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from evaluate_discrete_iperp_densified_barrier import (
    ALPHA_SCOPE_R2_SHA256,
    CONTRACT_SCHEMA,
    COORDINATE_SPACE,
    GRID_FORMULA,
    INTERPOLATION,
    METRIC_AMENDMENT_SHA256,
    N_DENSE_POINTS,
    SAMPLING,
    _history_index_for_call,
    sha256_array,
    sha256_file,
)
from run_barnase_barstar_ir_ivm_batch import validate_result


RUNS = (
    (0.1, "bb_irivm_a1e1_r01_m05_inf480_r1", "local"),
    (0.3, "bb_irivm_a1e1_r03_m05_inf480_r1", "remote"),
    (0.5, "bb_irivm_a1e1_r05_m05_inf480_r1", "reused_local"),
    (0.7, "bb_irivm_a1e1_r07_m05_inf480_r1", "local"),
    (0.8, "bb_irivm_a1e1_r08_m05_inf480_r1", "remote"),
)
BUDGET = 480


def relative(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), start=base.resolve())


def unique_result(root: Path, label: str) -> Path:
    matches = sorted((root / "runs" / "screen").glob(f"ivm_{label}_*.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one result for {label}, got {len(matches)}")
    return matches[0]


def run_record(root: Path, ratio: float, label: str, host: str) -> dict:
    result = unique_result(root, label)
    validate_result(result, expected_label=label, expected_budget=BUDGET)
    audit = result.with_suffix(".audit.json")
    registration = root / "registrations" / "screen" / f"{label}.json"
    authorization = (
        root / "registrations" / "execution_authorizations" / f"{label}.json"
    )
    registered = json.loads(registration.read_text(encoding="utf-8"))
    authorized = json.loads(authorization.read_text(encoding="utf-8"))
    if authorized.get("enabled") is not True or authorized.get("label") != label:
        raise ValueError(f"{label} authorization is not enabled")
    if authorized.get("member_registration_sha256") != sha256_file(registration):
        raise ValueError(f"{label} authorization/registration hash mismatch")
    if (
        float(registered["alpha"]) != 10.0
        or float(registered["residual_ratio_r"]) != ratio
        or int(registered["inner_max_steps_m_max"]) != 5
        or int(registered["internal_nf_budget"]) != BUDGET
    ):
        raise ValueError(f"{label} registered controls mismatch")
    with np.load(result, allow_pickle=False) as saved:
        calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        i_perp = np.asarray(saved["i_perp"], dtype=np.float64)
        counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
        paths = np.asarray(saved["stored_paths"], dtype=np.float64)
        actual_nf = int(np.asarray(saved["number_of_forces"]).item())
    eligible_calls = []
    eligible_states = []
    for stored_index, call in enumerate(stored_calls):
        history_index = _history_index_for_call(calls, int(call))
        if int(call) == 1 or not bool(committed[history_index]):
            continue
        path = np.asarray(paths[stored_index], dtype=np.float64)
        nf = int(counters[history_index].sum(dtype=np.int64))
        eligible_calls.append(int(call))
        eligible_states.append(
            {
                "history_index": history_index,
                "calls_per_image": int(call),
                "optimization_N_F": nf,
                "discrete_I_perp": float(i_perp[history_index]),
                "path_sha256": sha256_array(path),
            }
        )
    if not eligible_states:
        raise RuntimeError(f"{label} has no eligible reparameterized path")
    return {
        "label": label,
        "alpha": 10.0,
        "r": ratio,
        "m_max": 5,
        "registered_optimization_N_F": BUDGET,
        "actual_optimization_N_F": actual_nf,
        "execution_host": host,
        "member_registration": registration,
        "member_registration_sha256": sha256_file(registration),
        "execution_authorization": authorization,
        "execution_authorization_sha256": sha256_file(authorization),
        "source_result": result,
        "source_result_sha256": sha256_file(result),
        "source_audit": audit,
        "source_audit_sha256": sha256_file(audit),
        "eligible_committed_calls": eligible_calls,
        "eligible_states_at_freeze": eligible_states,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    records = [run_record(root, ratio, label, host) for ratio, label, host in RUNS]
    common = root / "inputs" / "common_initial_path32.npz"
    with np.load(common, allow_pickle=False) as saved:
        common_path = np.asarray(saved["initial_path"], dtype=np.float64)
    topology = root / "inputs" / "complex_fixed.pdb"
    bindings = {
        "parent_protocol": root / "registrations" / "STAGED_SEARCH_PROTOCOL.json",
        "metric_policy_amendment": root / "registrations" / "USER_METRIC_POLICY_AMENDMENT_20260901_R1.json",
        "alpha_scope_R2": root / "registrations" / "USER_ALPHA_SCOPE_AMENDMENT_20260901_R2.json",
        "budget_atomicity_amendment": root / "registrations" / "USER_BUDGET_ATOMICITY_AMENDMENT_20260901_R1.json",
        "exact_path_reuse_amendment": root / "registrations" / "USER_EXACT_PATH_DIAGNOSTIC_REUSE_AMENDMENT_20260901_R1.json",
        "efficient_scan_amendment": root / "registrations" / "USER_EFFICIENT_CONTROLLED_SCAN_AMENDMENT_20260901_R1.json",
        "alpha_confirmation_selection": root / "reports" / "selections" / "BB_IRIVM_ALPHA_CONFIRMATION_SELECTION_20260901_R1.json",
        "diagnostic_evaluator": root / "scripts" / "evaluate_discrete_iperp_densified_barrier.py",
    }
    base = output.parent
    payload = {
        "schema": CONTRACT_SCHEMA,
        "contract_id": "bb_irivm_r_controlled_scan_arc500_20260901_r1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "FROZEN_AFTER_ALL_R_SCAN_RESULTS_AND_BEFORE_DENSIFIED_DIAGNOSTICS",
        "scientific_terms": {
            "stationarity_diagnostic": "discrete unnormalized normal residual I_perp",
            "energetic_diagnostic": "densified 0 K forward barrier Delta E",
            "cost_axis": "actual optimization Number of Forces N_F",
        },
        "diagnostic_protocol": {
            "coordinate_space": COORDINATE_SPACE,
            "interpolation": INTERPOLATION,
            "sampling": SAMPLING,
            "number_of_points": N_DENSE_POINTS,
            "include_both_endpoints": True,
            "grid_formula": GRID_FORMULA,
            "force_evaluations_per_path": 0,
            "optimization_N_F_contribution": 0,
            "sparse_barrier_used_for_selection": False,
        },
        "bindings": {
            name: {"path": relative(path, base), "sha256": sha256_file(path)}
            for name, path in bindings.items()
        },
        "common_initial_path": {
            "path": relative(common, base),
            "file_sha256": sha256_file(common),
            "array_sha256": sha256_array(common_path),
        },
        "topology": {
            "path": relative(topology, base),
            "sha256": sha256_file(topology),
        },
        "runs": [
            {
                **record,
                "member_registration": relative(record["member_registration"], base),
                "execution_authorization": relative(record["execution_authorization"], base),
                "source_result": relative(record["source_result"], base),
                "source_audit": relative(record["source_audit"], base),
            }
            for record in records
        ],
        "selection_decisions_made_after_r_scan_results": 0,
        "selection_rule": "apply the frozen same-state two-metric joint order and report the raw Pareto relation",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output.chmod(0o444)
    print(f"Saved {output}")
    print(f"sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
