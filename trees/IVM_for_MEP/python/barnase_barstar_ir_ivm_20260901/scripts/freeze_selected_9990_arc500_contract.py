#!/usr/bin/env python3
"""Freeze the selected 9990 BB IR-IVM arc-length-500 diagnostic contract.

This script is terminal-only and zero-OpenMM: it loads sealed NumPy/JSON
artifacts, enumerates eligible reparameterization-committed paths, and proves
which exact path hashes can reuse an already audited 500-energy vector.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from evaluate_discrete_iperp_densified_barrier import (
    ALPHA_SCOPE_R2_SHA256,
    CONTRACT_SCHEMA,
    COORDINATE_SPACE,
    GRID_FORMULA,
    INTERPOLATION,
    N_DENSE_POINTS,
    SAMPLING,
    load_exact_reuse_rows,
    sha256_array,
    sha256_file,
)


LABEL = "bb_irivm_a1e0_r01_m05_inf9990_r1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def scalar(saved: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def relative(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), start=base.resolve())


def write_exclusive(path: Path, payload: Dict[str, object]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", closefd=False) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def binding_path(receipt: Dict[str, object], role: str) -> Path:
    record = receipt.get("bindings", {}).get(role)
    require(record is not None, f"receipt lacks {role} binding")
    path = Path(str(record["path"])).expanduser().resolve(strict=True)
    require(sha256_file(path) == str(record["sha256"]), f"{role} binding drift")
    return path


def main(argv: Sequence[str] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--selected-result", type=Path, required=True)
    parser.add_argument("--selected-audit", type=Path, required=True)
    parser.add_argument("--selected-verification", type=Path, required=True)
    parser.add_argument("--prefix-verification", type=Path, required=True)
    parser.add_argument("--reuse-diagnostics", type=Path, required=True)
    parser.add_argument("--reuse-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    result = args.selected_result.expanduser().resolve(strict=True)
    audit_path = args.selected_audit.expanduser().resolve(strict=True)
    verification_path = args.selected_verification.expanduser().resolve(strict=True)
    prefix_path = args.prefix_verification.expanduser().resolve(strict=True)
    reuse_path = args.reuse_diagnostics.expanduser().resolve(strict=True)
    reuse_audit_path = args.reuse_audit.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    require(not output.exists(), f"refusing to overwrite {output}")

    registration = root / "registrations" / "production" / f"{LABEL}.json"
    authorization = root / "registrations" / "execution_authorizations" / f"{LABEL}.json"
    common = root / "inputs" / "common_initial_path32.npz"
    topology = root / "inputs" / "complex_fixed.pdb"
    for role, path in {
        "registration": registration,
        "authorization": authorization,
        "common initial path": common,
        "topology": topology,
    }.items():
        require(path.is_file(), f"missing {role}: {path}")

    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    registered = json.loads(registration.read_text(encoding="utf-8"))
    require(verification.get("status") == "PASS", "selected long-member verification failed")
    require(prefix.get("status") == "PASS_EXACT_PREFIX", "exact 4080-prefix verification failed")
    require(audit.get("consistent") is True, "producer NoF audit failed")
    require(registered.get("label") == LABEL, "selected registration label mismatch")
    require(int(registered.get("internal_nf_budget", -1)) == 9990, "selected budget mismatch")
    require(binding_path(verification, "result") == result, "verification/result mismatch")
    require(binding_path(verification, "audit") == audit_path, "verification/audit mismatch")
    require(
        Path(str(prefix["bindings"]["selected_result"]["path"])).resolve() == result,
        "prefix receipt/result mismatch",
    )
    require(
        str(prefix["bindings"]["selected_result"]["sha256"]) == sha256_file(result),
        "prefix receipt/result hash mismatch",
    )

    with np.load(common, allow_pickle=False) as saved:
        require(tuple(saved.files) == ("initial_path",), "common path archive fields changed")
        common_path = np.asarray(saved["initial_path"], dtype=np.float64)

    verifier_rows = {
        int(row["call_per_image"]): row
        for row in verification["post_initial_reparam_committed_stored_paths"]
    }
    with np.load(result, allow_pickle=False) as saved:
        require(str(scalar(saved, "run_label")) == LABEL, "selected result label mismatch")
        calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
        i_perp = np.asarray(saved["i_perp"], dtype=np.float64)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
        stored_paths = np.asarray(saved["stored_paths"], dtype=np.float64)
        actual_nf = int(scalar(saved, "optimization_internal_nf"))
    require(actual_nf == int(verification["actual_optimization_N_F"]), "selected actual N_F mismatch")
    require(counters.shape == (calls.size, 32), "selected counter history shape mismatch")
    require(stored_paths.shape[0] == stored_calls.size, "selected stored path shape mismatch")

    selected_calls: List[int] = []
    states: List[Dict[str, object]] = []
    path_hashes = [sha256_array(common_path)]
    for stored_index, call_value in enumerate(stored_calls):
        call = int(call_value)
        matching = np.flatnonzero(calls == call)
        require(matching.size == 1, f"non-unique stored call {call}")
        history_index = int(matching[0])
        if call == 1 or not bool(committed[history_index]):
            continue
        require(call in verifier_rows, f"committed call {call} absent from terminal verification")
        optimization_nf = int(counters[history_index, 1:-1].sum(dtype=np.int64))
        require(
            optimization_nf == int(verifier_rows[call]["actual_optimization_N_F"]),
            f"committed call {call} N_F mismatch",
        )
        path = np.asarray(stored_paths[stored_index], dtype=np.float64)
        path_hash = sha256_array(path)
        selected_calls.append(call)
        path_hashes.append(path_hash)
        states.append(
            {
                "history_index": history_index,
                "calls_per_image": call,
                "optimization_N_F": optimization_nf,
                "discrete_I_perp": float(i_perp[history_index]),
                "path_sha256": path_hash,
            }
        )
    require(set(selected_calls) == set(verifier_rows), "eligible call set/verifier mismatch")
    require(selected_calls == sorted(selected_calls), "eligible calls are not ordered")

    topology_sha = sha256_file(topology)
    reusable = load_exact_reuse_rows(reuse_path, reuse_audit_path, topology_sha)
    unique_path_hashes = list(dict.fromkeys(path_hashes))
    reusable_hashes = [value for value in unique_path_hashes if value in reusable]
    fresh_hashes = [value for value in unique_path_hashes if value not in reusable]
    # Exact-prefix PASS establishes that the common path and three reference
    # committed paths are byte-identical.  The registered reuse source must
    # therefore cover those four paths under the evaluator hash convention.
    require(
        len(reusable_hashes) >= 4,
        "audited reuse source does not cover the common path and 4080 prefix",
    )

    base = output.parent
    bindings = {
        "parent_protocol": root / "registrations" / "STAGED_SEARCH_PROTOCOL.json",
        "metric_policy_amendment": root / "registrations" / "USER_METRIC_POLICY_AMENDMENT_20260901_R1.json",
        "alpha_scope_R2": root / "registrations" / "USER_ALPHA_SCOPE_AMENDMENT_20260901_R2.json",
        "budget_atomicity_amendment": root / "registrations" / "USER_BUDGET_ATOMICITY_AMENDMENT_20260901_R1.json",
        "exact_path_reuse_amendment": root / "registrations" / "USER_EXACT_PATH_DIAGNOSTIC_REUSE_AMENDMENT_20260901_R1.json",
        "selected_9990_design": root / "registrations" / "BB_IRIVM_SELECTED_9990_DESIGN_20260902_R1.json",
        "prefix_criterion": root / "registrations" / "BB_IRIVM_LONG_PREFIX_REPRODUCIBILITY_CRITERION_20260902_R1.json",
        "selected_long_member_verification": verification_path,
        "exact_prefix_verification": prefix_path,
        "diagnostic_evaluator": root / "scripts" / "evaluate_discrete_iperp_densified_barrier.py",
        "registered_exact_reuse_diagnostics": reuse_path,
        "registered_exact_reuse_audit": reuse_audit_path,
    }
    for role, path in bindings.items():
        require(path.is_file(), f"missing binding {role}: {path}")

    run_record = {
        "label": LABEL,
        "alpha": 1.0,
        "r": 0.1,
        "m_max": 5,
        "registered_optimization_N_F": 9990,
        "actual_optimization_N_F": actual_nf,
        "member_registration": relative(registration, base),
        "member_registration_sha256": sha256_file(registration),
        "execution_authorization": relative(authorization, base),
        "execution_authorization_sha256": sha256_file(authorization),
        "source_result": relative(result, base),
        "source_result_sha256": sha256_file(result),
        "source_audit": relative(audit_path, base),
        "source_audit_sha256": sha256_file(audit_path),
        "terminal_verification": relative(verification_path, base),
        "terminal_verification_sha256": sha256_file(verification_path),
        "exact_prefix_verification": relative(prefix_path, base),
        "exact_prefix_verification_sha256": sha256_file(prefix_path),
        "eligible_committed_calls": selected_calls,
        "eligible_states_at_freeze": states,
    }
    payload = {
        "schema": CONTRACT_SCHEMA,
        "contract_id": "bb_irivm_selected_9990_arc500_20260902_r1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "FROZEN_AFTER_TERMINAL_AND_EXACT_PREFIX_VERIFICATION_BEFORE_ANY_SELECTED_9990_DIAGNOSTIC_CALL",
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
            role: {"path": relative(path, base), "sha256": sha256_file(path)}
            for role, path in bindings.items()
        },
        "common_initial_path": {
            "path": relative(common, base),
            "file_sha256": sha256_file(common),
            "array_sha256": sha256_array(common_path),
        },
        "topology": {"path": relative(topology, base), "sha256": topology_sha},
        "runs": [run_record],
        "eligible_path_policy": "post-initial reparameterization-committed stored paths only",
        "exact_path_reuse_policy": (
            "Reuse only hash-identical path/grid/oracle rows from the registered PASS diagnostic; "
            "each absent unique path receives exactly 500 fresh energy-only evaluations."
        ),
        "exact_reuse_preflight": {
            "source_diagnostics": relative(reuse_path, base),
            "source_diagnostics_sha256": sha256_file(reuse_path),
            "source_audit": relative(reuse_audit_path, base),
            "source_audit_sha256": sha256_file(reuse_audit_path),
            "unique_selected_path_count_including_common_initial": len(unique_path_hashes),
            "exact_reused_unique_path_count": len(reusable_hashes),
            "fresh_unique_path_count": len(fresh_hashes),
            "exact_reused_path_sha256": reusable_hashes,
            "fresh_path_sha256": fresh_hashes,
            "fresh_diagnostic_energy_calls": N_DENSE_POINTS * len(fresh_hashes),
            "diagnostic_force_calls": 0,
            "optimization_N_F_contribution": 0,
        },
        "selection_rule": "same-state two-metric joint order with raw Pareto relations reported",
    }
    require(
        payload["bindings"]["alpha_scope_R2"]["sha256"] == ALPHA_SCOPE_R2_SHA256,
        "alpha-scope binding mismatch",
    )
    write_exclusive(output, payload)
    print(f"Saved {output}")
    print(f"sha256={sha256_file(output)}")
    print(f"exact_reused_unique_paths={len(reusable_hashes)}")
    print(f"fresh_unique_paths={len(fresh_hashes)}")
    print(f"fresh_energy_calls={N_DENSE_POINTS * len(fresh_hashes)}")


if __name__ == "__main__":
    main()
