#!/usr/bin/env python3
"""Freeze four verified IR-IVM full-state paths for joint-E/F arc500."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = ROOT / "scripts" / "evaluate_bb_dense_four_ivm_joint_v2.py"
ENGINE = ROOT / "scripts" / "evaluate_bb_dense_four_paths_joint_v2.py"
BB_CORE = ROOT / "src" / "bb_core.py"
TOPOLOGY = ROOT / "inputs" / "complex_fixed.pdb"
CANONICAL_R4_REGISTRATION = ROOT / "registrations" / "diagnostics" / "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF9990_20260902_R4.json"
CANONICAL_R4_AUTHORIZATION = ROOT / "registrations" / "execution_authorizations" / "bb_irivm_a1e0_r01_m05_inf9990_fullstate_r4.json"
R4_LABEL = "bb_irivm_a1e0_r01_m05_inf9990_fullstate_r4"
CALLS = (2, 5, 10, 50)
EXPECTED_NF = (60, 150, 300, 1329)
N_DENSE = 500
DIMENSION = 9396
N_IMAGES = 32
MINIMUM_FREE_BYTES_RESERVE = 8 * 1024 ** 3
ESTIMATED_SHARD_BYTES = (2 * N_DENSE * DIMENSION + 2 * N_IMAGES * DIMENSION) * 8 + 8 * 1024 ** 2
METRIC_POLICY = {
    "coordinate_space": "unweighted_3N_cartesian", "sampling": "uniform_cartesian_arclength",
    "interpolation": "piecewise_linear", "number_of_points": 500,
    "grid_formula": "s_j=j*L/499,j=0,...,499", "include_both_endpoints": True,
    "dense_physical_call": "one_joint_energy_full_force_getState_per_point",
    "dense_force_use": "stored_only_not_used_by_metrics_or_optimization",
    "discrete_I_perp_implementation": "bb_core.path_diagnostics",
    "tangent": "normalized central difference: x[i+1] - x[i-1]",
    "weight": "half sum of adjacent Cartesian chord lengths",
    "original_fixed_endpoint_diagnostic_force_evaluations": 0,
    "dense_fixed_endpoint_diagnostic_force_evaluations": 2,
    "optimization_N_F_contribution": 0,
    "original_discrete_state": "verified_online_paid_path_energy_full_force_zero_new_calls",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(path: Path):
    path = path.expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-state-verification", type=Path, required=True)
    parser.add_argument("--topology", type=Path, default=TOPOLOGY)
    parser.add_argument("--host-role", choices=("local", "remote"), default="remote")
    parser.add_argument("--global-lock-path", type=Path, required=True)
    parser.add_argument("--peer-global-lock-path", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    verification_path = args.full_state_verification.expanduser().resolve(strict=True)
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("schema") != "bb-irivm-selected-full-state-history-verification-v4" or verification.get("status") != "PASS_EXACT_FULL_STATE_HISTORY_OUTPUT_ONLY_REPLAY":
        raise ValueError("full-state verification is not canonical PASS v4")
    if verification.get("all_discrete_I_perp_values_exactly_reconstructed") is not True or int(verification.get("additional_physical_calls_from_capture", -1)) != 0:
        raise ValueError("full-state verification lacks exact zero-call capture proof")
    closure = verification.get("bindings", {})
    required_roles = {"registration", "authorization", "reference_result", "replay_result", "replay_audit", "capture_manifest", "capture_audit"}
    if set(closure) != required_roles:
        raise ValueError("R4 verification closure roles mismatch")
    closure_paths = {}
    for role, record in closure.items():
        path = Path(record["path"]).expanduser().resolve(strict=True)
        if sha256(path) != str(record["sha256"]):
            raise ValueError(f"R4 verification {role} binding mismatch")
        closure_paths[role] = path
    if closure_paths["registration"] != CANONICAL_R4_REGISTRATION.resolve(strict=True):
        raise ValueError("verification is not bound to the canonical R4 registration")
    if closure_paths["authorization"] != CANONICAL_R4_AUTHORIZATION.resolve(strict=True):
        raise ValueError("verification is not bound to the canonical R4 authorization")
    registration = json.loads(closure_paths["registration"].read_text(encoding="utf-8"))
    authorization = json.loads(closure_paths["authorization"].read_text(encoding="utf-8"))
    replay_audit = json.loads(closure_paths["replay_audit"].read_text(encoding="utf-8"))
    capture_manifest = json.loads(closure_paths["capture_manifest"].read_text(encoding="utf-8"))
    capture_audit = json.loads(closure_paths["capture_audit"].read_text(encoding="utf-8"))
    if registration.get("schema") != "bb-irivm-selected-full-state-history-output-only-replay-v4" or registration.get("status") != "FROZEN_BEFORE_FIRST_R4_FULL_STATE_HISTORY_REPLAY_PHYSICAL_CALL" or registration.get("label") != R4_LABEL:
        raise ValueError("canonical R4 registration identity mismatch")
    if authorization.get("schema") != "bb-irivm-selected-full-state-history-output-only-replay-authorization-v4" or authorization.get("enabled") is not True or authorization.get("label") != R4_LABEL:
        raise ValueError("canonical R4 authorization identity mismatch")
    if replay_audit.get("consistent") is not True:
        raise ValueError("R4 replay runtime audit is inconsistent")
    if capture_manifest.get("schema") != "bb-irivm-full-state-history-manifest-v2" or capture_manifest.get("status") != "COMPLETE_PENDING_EXACT_PREFIX_VERIFICATION" or capture_manifest.get("label") != R4_LABEL:
        raise ValueError("R4 capture manifest identity mismatch")
    if capture_audit.get("schema") != "bb-irivm-full-state-history-audit-v2" or capture_audit.get("status") != "PASS_SCHEMA_AWARE_STORAGE_ONLY_CAPTURE_PENDING_EXACT_PREFIX_VERIFICATION" or capture_audit.get("label") != R4_LABEL or int(capture_audit.get("additional_physical_calls", -1)) != 0 or capture_audit.get("missing_counter_values_guessed_or_backfilled") is not False:
        raise ValueError("R4 capture audit identity mismatch")
    inventory = {int(item["call_per_image"]): item for item in verification["path_inventory"]}
    paths = []
    for call, expected_nf in zip(CALLS, EXPECTED_NF):
        item = inventory.get(call)
        if item is None or item.get("formally_reportable") is not True or item.get("terminal_uncommitted_charged_tail") is not False:
            raise ValueError(f"call {call} is not formally reportable")
        source = Path(item["shard"]).expanduser().resolve(strict=True)
        if sha256(source) != str(item["shard_sha256"]):
            raise ValueError(f"call {call} source shard SHA mismatch")
        if int(item["optimization_N_F"]) != expected_nf:
            raise ValueError(f"call {call} N_F mismatch")
        with np.load(source, allow_pickle=False) as saved:
            history_index = int(np.asarray(saved["history_index"]).item())
            if int(np.asarray(saved["call_per_image"]).item()) != call:
                raise ValueError(f"call {call} shard metadata mismatch")
            source_label = str(np.asarray(saved["label"]).item())
        paths.append({
            "label": f"IVM_call{call}_NF{expected_nf}", "method": "IVM", "host_role": args.host_role,
            "optimization_N_F": expected_nf,
            "path_identifier": {"calls_per_image": call, "history_index": history_index, "fully_evaluated_current_path": True, "terminal_tail": False, "source_replay_label": source_label},
            "online_full_state_source": bind(source), "path_sha256": str(item["path_sha256"]),
            "online_energy_sha256": str(item["energy_sha256"]), "online_force_sha256": str(item["force_sha256"]),
            "online_discrete_I_perp": float(item["discrete_I_perp_eV2_per_A"]),
            "counter_source_kind": str(item["counter_source_kind"]),
            "counter_schema_keys": list(item["counter_schema_keys"]),
            "unavailable_counter_fields": list(item["unavailable_counter_fields"]),
            "original_discrete_policy": "verified_online_paid_path_energy_full_force_zero_new_calls",
            "dense_policy": "fresh_or_complete_validated_shard_500_joint_EF",
        })
    locks = [args.global_lock_path.expanduser().resolve(), args.peer_global_lock_path.expanduser().resolve()]
    if len(set(locks)) != 2:
        raise ValueError("two distinct host-local lock paths required")
    contract = {
        "schema": "bb-dense-four-ivm-joint-ef-contract-v2", "frozen": True,
        "purpose": "retain verified online IVM full states and fresh joint-E/F arc500 diagnostics for calls 2/5/10/50",
        "method": "IVM", "jobs": 1, "evaluator": bind(EVALUATOR), "joint_engine": bind(ENGINE),
        "bb_core": bind(BB_CORE), "topology": bind(args.topology), "full_state_verification": bind(verification_path),
        "R4_full_state_closure": closure,
        "host_local_lock_paths": [str(path) for path in locks], "metric_policy": METRIC_POLICY, "paths": paths,
        "physical_call_policy": {
            "original_online_paid_joint_EF_reuse_per_path": 32, "new_original_discrete_joint_EF_per_path": 0,
            "dense_joint_EF_per_path": 500, "dense_endpoint_joint_EF_per_path": 2,
            "dense_full_force_storage": True, "optimization_N_F_contribution": 0,
            "all_six_thread_caps": 1, "two_host_local_fcntl_locks": True, "jobs": 1,
        },
        "durability_policy": {
            "separate_online_reuse_and_dense_joint_append_only_fsync_WAL": True,
            "atomic_complete_path_shard": True, "shard_contains_dense_coordinates_energy_full_force": True,
            "manifest_binds_every_shard_and_array_hash": True,
            "restart_reuse_unit": "complete_fully_validated_path_shard",
            "reuse_individual_incomplete_WAL_successes": False, "refuse_final_output_overwrite": True,
        },
        "storage_policy": {
            "authoritative_unit": "complete_atomic_path_shard",
            "aggregate_role": "index_metrics_and_plot_compatible_dense_energy_not_dense_coordinates_or_forces",
            "minimum_free_bytes_reserve": MINIMUM_FREE_BYTES_RESERVE,
            "estimated_uncompressed_shard_bytes_per_path": ESTIMATED_SHARD_BYTES,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if destination.exists():
        temporary.unlink()
        raise FileExistsError(f"refusing to overwrite {destination}")
    os.rename(temporary, destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({"status": "FROZEN", "contract": str(destination), "sha256": sha256(destination), "calls": list(CALLS), "optimization_N_F": list(EXPECTED_NF)}, indent=2))


if __name__ == "__main__":
    main()
