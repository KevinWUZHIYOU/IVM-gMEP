#!/usr/bin/env python3
"""Freeze the three independent 990-N_F alpha confirmations for diagnostics.

This is a zero-OpenMM sealing step.  It requires all three optimization
members and their exact Number-of-Forces audits to exist before it writes an
immutable contract.  The later evaluator then applies the same discrete
``I_perp`` and 500-point equal-arclength ``Delta E`` definitions to every
eligible, fully reparameterized state named here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Dict, List

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
    _validate_terminal_budget_metadata,
    _validate_optimization_audit,
    sha256_array,
    sha256_file,
)


EXPECTED_RUNS = (
    (1.0, "bb_irivm_a1e0_r05_m05_inf990_r1", "local"),
    (0.1, "bb_irivm_a1em1_r05_m05_inf990_r1", "remote"),
    (10.0, "bb_irivm_a1e1_r05_m05_inf990_r3", "local"),
)
EXPECTED_PROMOTED_ALPHAS = (1.0, 0.1, 10.0)
OPTIMIZATION_N_F = 990


def relative(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), start=base.resolve())


def unique_result(root: Path, label: str) -> Path:
    matches = sorted((root / "runs" / "screen").glob(f"ivm_{label}_*.npz"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one sealed result for {label}, got {len(matches)}"
        )
    return matches[0]


def validate_principal_selection(receipt: Dict[str, object]) -> None:
    if receipt.get("status") != "PASS_ALPHA_SELECTION_FROM_TWO_PRINCIPAL_DIAGNOSTICS":
        raise ValueError("the principal alpha-selection receipt has not passed")
    if receipt.get("metric_policy_amendment_sha256") != METRIC_AMENDMENT_SHA256:
        raise ValueError("principal alpha-selection metric-policy binding mismatch")
    if receipt.get("alpha_scope_R2_sha256") != ALPHA_SCOPE_R2_SHA256:
        raise ValueError("principal alpha-selection alpha-scope binding mismatch")
    promoted = tuple(
        float(value)
        for value in receipt.get(
            "promoted_alpha_values_for_independent_990_N_F_runs", []
        )
    )
    if promoted != EXPECTED_PROMOTED_ALPHAS:
        raise ValueError(
            "principal alpha-selection receipt does not name the registered "
            "990-N_F confirmation order"
        )


def extract_run_record(
    root: Path, result: Path, alpha: float, label: str, host: str
) -> Dict[str, object]:
    registration = root / "registrations" / "screen" / f"{label}.json"
    authorization = (
        root / "registrations" / "execution_authorizations" / f"{label}.json"
    )
    if not registration.is_file() or not authorization.is_file():
        raise FileNotFoundError(
            f"{label} is missing its member registration or execution authorization"
        )
    authorization_payload = json.loads(authorization.read_text(encoding="utf-8"))
    if authorization_payload.get("enabled") is not True:
        raise ValueError(f"{label} execution authorization is not enabled")
    if authorization_payload.get("label") != label:
        raise ValueError(f"{label} execution-authorization label mismatch")
    if authorization_payload.get("member_registration_sha256") != sha256_file(
        registration
    ):
        raise ValueError(f"{label} execution authorization/registration mismatch")
    promotion_gate = authorization_payload.get("promotion_gate", {})
    if promotion_gate.get("stage") != (
        "alpha_controlled_scan_independent_990_N_F_confirmation"
    ):
        raise ValueError(f"{label} lacks the alpha-confirmation promotion gate")
    if label.endswith(("_r2", "_r3")):
        retry_gate = authorization_payload.get("infrastructure_retry_gate", {})
        if retry_gate.get("replacement_label") != label:
            raise ValueError(f"{label} lacks the infrastructure-retry gate")
    audit_path = result.with_suffix(".audit.json")
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    actual_nf = _validate_optimization_audit(audit, OPTIMIZATION_N_F)
    with np.load(result, allow_pickle=False) as saved:
        if str(np.asarray(saved["run_label"]).item()) != label:
            raise ValueError(f"{label} run-label mismatch")
        _validate_terminal_budget_metadata(saved, OPTIMIZATION_N_F, actual_nf)
        calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        i_perp = np.asarray(saved["i_perp"], dtype=np.float64)
        counter_vectors = np.asarray(saved["counter_vectors"], dtype=np.int64)
        stored_paths = np.asarray(saved["stored_paths"], dtype=np.float64)

    eligible_calls: List[int] = []
    eligible_states: List[Dict[str, object]] = []
    for stored_index, call in enumerate(stored_calls):
        history_index = _history_index_for_call(calls, int(call))
        if int(call) == 1 or not bool(committed[history_index]):
            continue
        optimization_nf = int(counter_vectors[history_index].sum(dtype=np.int64))
        if optimization_nf > OPTIMIZATION_N_F:
            raise ValueError(f"{label} stored an eligible state beyond its budget")
        path = np.asarray(stored_paths[stored_index], dtype=np.float64)
        eligible_calls.append(int(call))
        eligible_states.append(
            {
                "history_index": history_index,
                "calls_per_image": int(call),
                "optimization_N_F": optimization_nf,
                "discrete_I_perp": float(i_perp[history_index]),
                "path_sha256": sha256_array(path),
            }
        )
    if not eligible_calls:
        raise RuntimeError(f"{label} has no fully evaluated reparameterized state")
    return {
        "label": label,
        "alpha": float(alpha),
        "r": 0.5,
        "m_max": 5,
        "registered_optimization_N_F": OPTIMIZATION_N_F,
        "actual_optimization_N_F": actual_nf,
        "execution_host": host,
        "member_registration": str(registration),
        "member_registration_sha256": sha256_file(registration),
        "execution_authorization": str(authorization),
        "execution_authorization_sha256": sha256_file(authorization),
        "source_result": str(result),
        "source_result_sha256": sha256_file(result),
        "source_audit": str(audit_path),
        "source_audit_sha256": sha256_file(audit_path),
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
        raise FileExistsError(f"refusing to overwrite {output}")

    metric = root / "registrations" / "USER_METRIC_POLICY_AMENDMENT_20260901_R1.json"
    alpha_scope = root / "registrations" / "USER_ALPHA_SCOPE_AMENDMENT_20260901_R2.json"
    principal_selection = (
        root
        / "reports"
        / "selections"
        / "BB_IRIVM_ALPHA_PRINCIPAL_SELECTION_20260901_R1.json"
    )
    if sha256_file(metric) != METRIC_AMENDMENT_SHA256:
        raise ValueError("metric-policy amendment hash mismatch")
    if sha256_file(alpha_scope) != ALPHA_SCOPE_R2_SHA256:
        raise ValueError("alpha-scope R2 hash mismatch")
    principal_payload = json.loads(principal_selection.read_text(encoding="utf-8"))
    validate_principal_selection(principal_payload)

    runs = [
        extract_run_record(root, unique_result(root, label), alpha, label, host)
        for alpha, label, host in EXPECTED_RUNS
    ]
    common_file = root / "inputs" / "common_initial_path32.npz"
    with np.load(common_file, allow_pickle=False) as saved:
        common_path = np.asarray(saved["initial_path"], dtype=np.float64)
    topology = root / "inputs" / "complex_fixed.pdb"
    evaluator = root / "scripts" / "evaluate_discrete_iperp_densified_barrier.py"
    evaluator_test = root / "tests" / "test_discrete_iperp_densified_barrier.py"
    parent_protocol = root / "registrations" / "STAGED_SEARCH_PROTOCOL.json"
    budget_atomicity = (
        root
        / "registrations"
        / "USER_BUDGET_ATOMICITY_AMENDMENT_20260901_R1.json"
    )
    exact_path_reuse = (
        root
        / "registrations"
        / "USER_EXACT_PATH_DIAGNOSTIC_REUSE_AMENDMENT_20260901_R1.json"
    )
    principal_diagnostics = (
        root
        / "runs"
        / "diagnostics"
        / "bb_irivm_alpha_discrete_iperp_densified_delta_e_500_20260901_r1.npz"
    )
    principal_diagnostic_audit = principal_diagnostics.with_suffix(".audit.json")
    alpha10_failures = (
        root
        / "reports"
        / "failures"
        / "BB_IRIVM_ALPHA10_CONFIRMATION_INFRASTRUCTURE_FAILURE_20260901_R1.json",
        root
        / "reports"
        / "failures"
        / "BB_IRIVM_ALPHA10_CONFIRMATION_INFRASTRUCTURE_FAILURE_20260901_R2.json",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    base = output.parent
    contract = {
        "schema": CONTRACT_SCHEMA,
        "contract_id": "bb_irivm_alpha_990_nf_confirmation_diagnostics_20260901_r1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "FROZEN_AFTER_ALL_990_N_F_RESULTS_AND_BEFORE_CONFIRMATION_DIAGNOSTICS",
        "scientific_terms": {
            "stationarity_diagnostic": "discrete unnormalized normal residual I_perp",
            "energetic_diagnostic": "densified 0 K forward barrier Delta E",
            "cost_axis": "optimization Number of Forces N_F",
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
            "parent_protocol": {
                "path": relative(parent_protocol, base),
                "sha256": sha256_file(parent_protocol),
            },
            "metric_policy_amendment": {
                "path": relative(metric, base),
                "sha256": METRIC_AMENDMENT_SHA256,
            },
            "alpha_scope_R2": {
                "path": relative(alpha_scope, base),
                "sha256": ALPHA_SCOPE_R2_SHA256,
            },
            "budget_atomicity_amendment": {
                "path": relative(budget_atomicity, base),
                "sha256": sha256_file(budget_atomicity),
            },
            "exact_path_diagnostic_reuse_amendment": {
                "path": relative(exact_path_reuse, base),
                "sha256": sha256_file(exact_path_reuse),
            },
            "registered_exact_reuse_source": {
                "diagnostics": relative(principal_diagnostics, base),
                "diagnostics_sha256": sha256_file(principal_diagnostics),
                "audit": relative(principal_diagnostic_audit, base),
                "audit_sha256": sha256_file(principal_diagnostic_audit),
            },
            "principal_alpha_selection": {
                "path": relative(principal_selection, base),
                "sha256": sha256_file(principal_selection),
            },
            "alpha10_infrastructure_failures": [
                {
                    "path": relative(path, base),
                    "sha256": sha256_file(path),
                }
                for path in alpha10_failures
            ],
            "diagnostic_evaluator": {
                "path": relative(evaluator, base),
                "sha256": sha256_file(evaluator),
            },
            "zero_OpenMM_tests": {
                "path": relative(evaluator_test, base),
                "sha256": sha256_file(evaluator_test),
                "command": (
                    "PYTHONPATH=src:scripts "
                    "/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python "
                    "tests/test_discrete_iperp_densified_barrier.py"
                ),
                "result": "6 unittest cases passed before freezing",
            },
        },
        "common_initial_path": {
            "path": relative(common_file, base),
            "file_sha256": sha256_file(common_file),
            "array_sha256": sha256_array(common_path),
        },
        "topology": {
            "path": relative(topology, base),
            "sha256": sha256_file(topology),
        },
        "runs": [
            {
                **record,
                "member_registration": relative(
                    Path(record["member_registration"]), base
                ),
                "execution_authorization": relative(
                    Path(record["execution_authorization"]), base
                ),
                "source_result": relative(Path(record["source_result"]), base),
                "source_audit": relative(Path(record["source_audit"]), base),
            }
            for record in runs
        ],
        "selection_decisions_made_after_990_N_F_results": 0,
        "selection_rule": (
            "Apply the frozen user metric-policy amendment to every listed "
            "eligible state after all listed 500-point diagnostic ledgers close; "
            "retain alpha_star and alpha_2 by its deterministic same-state ranking."
        ),
    }
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    output.chmod(0o444)
    print(f"Saved {output}")
    print(f"sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
