#!/usr/bin/env python3
"""Freeze the five-run alpha comparison before densified diagnostics.

The contract includes every fully evaluated equal-arclength reparameterized
state from each retained alpha value.  It performs no OpenMM evaluation and
makes no parameter-selection decision.
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
    _validate_optimization_audit,
    sha256_array,
    sha256_file,
)


EXPECTED_RUNS = (
    (0.1, "bb_irivm_a1em1_r05_m05_inf480_r1", "local"),
    (1.0, "bb_irivm_a1e0_r05_m05_inf480_r1", "remote"),
    (10.0, "bb_irivm_a1e1_r05_m05_inf480_r1", "local"),
    (100.0, "bb_irivm_a1e2_r05_m05_inf480_r1", "remote"),
    (1000.0, "bb_irivm_a1e3_r05_m05_inf480_r1", "local"),
)
CANCELLED_ALPHAS = (10000.0, 100000.0, 1000000.0)


def relative(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), start=base.resolve())


def unique_result(root: Path, label: str) -> Path:
    matches = sorted((root / "runs" / "screen").glob(f"ivm_{label}_*.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one sealed result for {label}, got {len(matches)}")
    return matches[0]


def extract_run_record(root: Path, result: Path, alpha: float, label: str, host: str) -> Dict[str, object]:
    audit_path = result.with_suffix(".audit.json")
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _validate_optimization_audit(audit, 480)
    with np.load(result, allow_pickle=False) as saved:
        if str(np.asarray(saved["run_label"]).item()) != label:
            raise ValueError(f"{label} run-label mismatch")
        if not bool(np.asarray(saved["total_internal_nf_budget_exhausted"]).item()):
            raise ValueError(f"{label} did not exhaust 480 N_F")
        if not bool(np.asarray(saved["total_internal_nf_budget_no_overshoot"]).item()):
            raise ValueError(f"{label} overshot 480 N_F")
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
        "registered_optimization_N_F": 480,
        "execution_host": host,
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
    if sha256_file(metric) != METRIC_AMENDMENT_SHA256:
        raise ValueError("metric-policy amendment hash mismatch")
    if sha256_file(alpha_scope) != ALPHA_SCOPE_R2_SHA256:
        raise ValueError("alpha-scope R2 hash mismatch")
    for alpha in CANCELLED_ALPHAS:
        exponent = int(round(np.log10(alpha)))
        label_fragment = f"a1e{exponent}"
        cancelled = list((root / "runs" / "screen").glob(f"ivm_bb_irivm_{label_fragment}_*.npz"))
        if cancelled:
            raise RuntimeError(f"cancelled alpha={alpha:g} unexpectedly has a result")

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
    output.parent.mkdir(parents=True, exist_ok=True)
    base = output.parent
    contract = {
        "schema": CONTRACT_SCHEMA,
        "contract_id": "bb_irivm_alpha_1em1_to_1e3_principal_diagnostics_20260901_r1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "FROZEN_BEFORE_DENSIFIED_ENERGY_EVALUATIONS_AND_ALPHA_SELECTION",
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
                "result": "5 unittest cases passed before freezing",
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
                "source_result": relative(Path(record["source_result"]), base),
                "source_audit": relative(Path(record["source_audit"]), base),
            }
            for record in runs
        ],
        "cancelled_untested_alpha_values": list(CANCELLED_ALPHAS),
        "selection_decisions_made": 0,
        "selection_rule": "Apply the frozen user metric-policy amendment to all listed eligible states after every listed densified diagnostic is complete.",
    }
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    output.chmod(0o444)
    print(f"Saved {output}")
    print(f"sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
