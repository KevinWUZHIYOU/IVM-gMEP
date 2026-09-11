#!/usr/bin/env python3
"""Freeze the selected long candidate before arc-length-500 diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from evaluate_discrete_iperp_densified_barrier import (
    CONTRACT_SCHEMA,
    COORDINATE_SPACE,
    GRID_FORMULA,
    INTERPOLATION,
    N_DENSE_POINTS,
    SAMPLING,
    _history_index_for_call,
    sha256_array,
    sha256_file,
)


LABEL = "bb_irivm_a1e0_r01_m05_inf4080_r1"
RESULT_NAME = (
    "ivm_bb_irivm_a1e0_r01_m05_inf4080_r1_central_state_ivm_"
    "cfg47c24114c93c_budget4080_storagecommitted.npz"
)


def relative(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), start=base.resolve())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--trigger-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    decision_path = args.trigger_decision.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    require(not output.exists(), f"refusing to overwrite {output}")

    result = root / "runs" / "screen" / RESULT_NAME
    audit = result.with_suffix(".audit.json")
    registration = root / "registrations" / "screen" / f"{LABEL}.json"
    authorization = root / "registrations" / "execution_authorizations" / f"{LABEL}.json"
    verification_path = root / "reports" / "verifications" / "BB_IRIVM_A1E0_R01_M05_INF4080_R1_VERIFICATION_20260902.json"
    common = root / "inputs" / "common_initial_path32.npz"
    topology = root / "inputs" / "complex_fixed.pdb"
    for role, path in {
        "result": result,
        "audit": audit,
        "registration": registration,
        "authorization": authorization,
        "verification": verification_path,
        "trigger decision": decision_path,
        "common path": common,
        "topology": topology,
    }.items():
        require(path.is_file(), f"missing {role}: {path}")

    registered = json.loads(registration.read_text(encoding="utf-8"))
    producer_audit = json.loads(audit.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    require(registered["label"] == LABEL, "registration label mismatch")
    require(float(registered["alpha"]) == 1.0, "alpha mismatch")
    require(float(registered["residual_ratio_r"]) == 0.1, "r mismatch")
    require(int(registered["inner_max_steps_m_max"]) == 5, "m_max mismatch")
    require(int(registered["internal_nf_budget"]) == 4080, "budget mismatch")
    require(producer_audit.get("consistent") is True, "producer audit failed")
    require(verification.get("status") == "PASS", "long-member verification failed")
    require(decision.get("status") == "A8_NOT_TRIGGERED", "A8 trigger decision mismatch")
    require(
        verification["bindings"]["result"]["sha256"] == sha256_file(result),
        "verification/result binding mismatch",
    )
    require(
        verification["bindings"]["audit"]["sha256"] == sha256_file(audit),
        "verification/audit binding mismatch",
    )

    verifier_rows = {
        int(row["call_per_image"]): row
        for row in verification["post_initial_reparam_committed_stored_paths"]
    }
    with np.load(result, allow_pickle=False) as saved:
        calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
        stored_paths = np.asarray(saved["stored_paths"], dtype=np.float64)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
        i_perp = np.asarray(saved["i_perp"], dtype=np.float64)
        actual_nf = int(np.asarray(saved["optimization_internal_nf"]).item())
    require(actual_nf == int(verification["actual_optimization_N_F"]), "actual N_F mismatch")

    states = []
    selected_calls = []
    for stored_index, call_value in enumerate(stored_calls):
        call = int(call_value)
        history_index = _history_index_for_call(calls, call)
        if call == 1 or not bool(committed[history_index]):
            continue
        require(call in verifier_rows, f"call {call} absent from verifier")
        optimization_nf = int(counters[history_index, 1:-1].sum(dtype=np.int64))
        require(
            optimization_nf == int(verifier_rows[call]["actual_optimization_N_F"]),
            f"call {call} verifier N_F mismatch",
        )
        path = np.asarray(stored_paths[stored_index], dtype=np.float64)
        selected_calls.append(call)
        states.append(
            {
                "history_index": history_index,
                "calls_per_image": call,
                "optimization_N_F": optimization_nf,
                "discrete_I_perp": float(i_perp[history_index]),
                "path_sha256": sha256_array(path),
                "verifier_path_sha256": str(verifier_rows[call]["path_sha256"]),
            }
        )
    require(selected_calls == [7, 20, 72], f"unexpected eligible calls {selected_calls}")
    require(set(selected_calls) == set(verifier_rows), "eligible call set/verifier mismatch")

    with np.load(common, allow_pickle=False) as saved:
        common_path = np.asarray(saved["initial_path"], dtype=np.float64)
    base = output.parent
    bindings = {
        "parent_protocol": root / "registrations" / "STAGED_SEARCH_PROTOCOL.json",
        "metric_policy_amendment": root / "registrations" / "USER_METRIC_POLICY_AMENDMENT_20260901_R1.json",
        "alpha_scope_R2": root / "registrations" / "USER_ALPHA_SCOPE_AMENDMENT_20260901_R2.json",
        "budget_atomicity_amendment": root / "registrations" / "USER_BUDGET_ATOMICITY_AMENDMENT_20260901_R1.json",
        "exact_path_reuse_amendment": root / "registrations" / "USER_EXACT_PATH_DIAGNOSTIC_REUSE_AMENDMENT_20260901_R1.json",
        "long_budget_design": root / "registrations" / "BB_IRIVM_LONG_BUDGET_DESIGN_20260902_R1.json",
        "cached_stall_followup_design": root / "registrations" / "BB_IRIVM_CACHED_STALL_TARGETED_FOLLOWUP_DESIGN_20260902_R1.json",
        "cached_stall_trigger_decision": decision_path,
        "long_member_verification": verification_path,
        "diagnostic_evaluator": root / "scripts" / "evaluate_discrete_iperp_densified_barrier.py",
    }
    record = {
        "label": LABEL,
        "alpha": 1.0,
        "r": 0.1,
        "m_max": 5,
        "registered_optimization_N_F": 4080,
        "actual_optimization_N_F": actual_nf,
        "execution_host": "Mac-mini.local / Apple M4 / arm64",
        "member_registration": relative(registration, base),
        "member_registration_sha256": sha256_file(registration),
        "execution_authorization": relative(authorization, base),
        "execution_authorization_sha256": sha256_file(authorization),
        "source_result": relative(result, base),
        "source_result_sha256": sha256_file(result),
        "source_audit": relative(audit, base),
        "source_audit_sha256": sha256_file(audit),
        "terminal_verification": relative(verification_path, base),
        "terminal_verification_sha256": sha256_file(verification_path),
        "eligible_committed_calls": selected_calls,
        "eligible_states_at_freeze": states,
    }
    payload = {
        "schema": CONTRACT_SCHEMA,
        "contract_id": "bb_irivm_candidate_b_long_arc500_20260902_r1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "FROZEN_AFTER_TERMINAL_VERIFICATION_AND_BEFORE_LONG_CANDIDATE_DENSIFIED_DIAGNOSTICS",
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
        "topology": {"path": relative(topology, base), "sha256": sha256_file(topology)},
        "runs": [record],
        "eligible_path_policy": "post-initial reparameterization-committed stored paths only",
        "selection_decisions_made_after_long_candidate_result": 0,
        "exact_path_reuse_policy": (
            "Reuse only hash-identical path/grid/oracle rows from a PASS diagnostic; "
            "every absent path receives 500 fresh energy-only evaluations."
        ),
        "selection_rule": "apply the frozen same-state two-metric joint order and report raw Pareto relations",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    output.chmod(0o444)
    print(f"Saved {output}")
    print(f"sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
