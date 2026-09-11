#!/usr/bin/env python3
"""Joint-E/F arc500 evaluator for four verified IR-IVM full-state paths.

The original 32-image energies and full forces are reused only from the
verified output-only full-state replay; this costs zero new physical calls and
reconstructs the article's discrete I_perp. Every arc500 point is then
evaluated once with a fresh joint E/F request and its full force is retained.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENGINE_PATH = HERE / "evaluate_bb_dense_four_paths_joint_v2.py"
SPEC = importlib.util.spec_from_file_location("bb_dense_joint_v2_frozen", ENGINE_PATH)
ENGINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ENGINE)

SCHEMA = "bb-dense-four-ivm-joint-ef-contract-v2"
SHARD_SCHEMA = "bb-dense-ivm-path-joint-ef-shard-v2"
AUDIT_SCHEMA = "bb-dense-four-ivm-joint-ef-audit-v2"
ONLINE_WAL_SCHEMA = "bb-ivm-online-paid-full-state-reuse-wal-v2"
DENSE_WAL_SCHEMA = "bb-ivm-dense-path-joint-ef-wal-v2"
VERIFICATION_SCHEMA = "bb-irivm-selected-full-state-history-verification-v4"
VERIFICATION_STATUS = "PASS_EXACT_FULL_STATE_HISTORY_OUTPUT_ONLY_REPLAY"
R4_LABEL = "bb_irivm_a1e0_r01_m05_inf9990_fullstate_r4"
CANONICAL_R4_REGISTRATION = ROOT / "registrations" / "diagnostics" / "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF9990_20260902_R4.json"
CANONICAL_R4_AUTHORIZATION = ROOT / "registrations" / "execution_authorizations" / "bb_irivm_a1e0_r01_m05_inf9990_fullstate_r4.json"
COMMON_COUNTER_KEYS = (
    "joint_evaluations", "energy_requests", "force_requests", "cache_hits",
    "total_query_events", "both_requests", "budget_blocked_misses",
)
INHERITED_COUNTER_KEY = "inherited_cache_entries"
CALLS = (2, 5, 10, 50)
EXPECTED_NF = (60, 150, 300, 1329)
N_IMAGES = 32
N_INTERNAL = 30
N_ATOMS = 3132
DIMENSION = 9396
N_DENSE = 500
MINIMUM_FREE_BYTES_RESERVE = 8 * 1024 ** 3
ESTIMATED_SHARD_BYTES = (
    2 * N_DENSE * DIMENSION + 2 * N_IMAGES * DIMENSION
) * 8 + 8 * 1024 ** 2
METRIC_POLICY = {
    **ENGINE.METRIC_POLICY,
    "original_discrete_state": "verified_online_paid_path_energy_full_force_zero_new_calls",
}


def require(condition: bool, message: str) -> None:
    ENGINE.require(condition, message)


def _scalar(saved, key: str):
    return np.asarray(saved[key]).item()


def _same_float(left: float, right: float) -> bool:
    return np.asarray(float(left), dtype=np.float64).tobytes() == np.asarray(float(right), dtype=np.float64).tobytes()


def _load_online_source(source: Path, row: dict):
    with np.load(source, allow_pickle=False) as saved:
        required = {
            "schema", "label", "history_index", "call_per_image", "optimization_N_F",
            "path32_A", "energy_eV", "force_eV_per_A", "path_sha256", "energy_sha256", "force_sha256",
            "discrete_I_perp", "fully_evaluated_current_path", "provisional_reportable",
            "terminal_uncommitted_tail", "additional_physical_calls",
            "counter_source_kind", "counter_schema_keys", "unavailable_counter_fields",
            "inherited_cache_entries_present", "joint_evaluation_vector",
            "energy_request_vector", "force_request_vector", "cache_hit_vector",
            "total_query_event_vector", "both_request_vector", "budget_blocked_miss_vector",
        }
        require(required.issubset(saved.files), "online full-state shard fields missing")
        path = np.ascontiguousarray(saved["path32_A"], dtype=np.float64)
        energy = np.ascontiguousarray(saved["energy_eV"], dtype=np.float64)
        force = np.ascontiguousarray(saved["force_eV_per_A"], dtype=np.float64)
        require(path.shape == force.shape == (N_IMAGES, DIMENSION) and energy.shape == (N_IMAGES,), "online full-state shape mismatch")
        require(np.all(np.isfinite(path)) and np.all(np.isfinite(energy)) and np.all(np.isfinite(force)), "online full-state non-finite")
        require(str(_scalar(saved, "schema")) == "bb-irivm-full-state-history-recorder-v2", "online source is not a canonical R4 recorder-v2 shard")
        require(str(_scalar(saved, "label")) == R4_LABEL, "online source is not labeled as the canonical R4 replay")
        source_kind = str(_scalar(saved, "counter_source_kind"))
        schema_keys_raw = np.asarray(saved["counter_schema_keys"])
        unavailable_raw = np.asarray(saved["unavailable_counter_fields"])
        require(schema_keys_raw.ndim == unavailable_raw.ndim == 1 and schema_keys_raw.dtype.kind in "US" and unavailable_raw.dtype.kind in "US", "online R4 counter metadata dtype/shape mismatch")
        schema_keys = schema_keys_raw.astype(str).tolist()
        unavailable = unavailable_raw.astype(str).tolist()
        inherited_present = bool(_scalar(saved, "inherited_cache_entries_present"))
        expected_seven = list(COMMON_COUNTER_KEYS)
        expected_eight = [*COMMON_COUNTER_KEYS, INHERITED_COUNTER_KEY]
        if source_kind == "live_counter_arrays":
            require(schema_keys == expected_eight and inherited_present and unavailable == [], "online R4 live counter schema is not exact-8")
        else:
            require(source_kind == "historical_counter_snapshot", "online R4 counter source kind mismatch")
            if inherited_present:
                require(schema_keys == expected_eight and unavailable == [], "online R4 historical schema-8 metadata mismatch")
            else:
                require(schema_keys == expected_seven and unavailable == [INHERITED_COUNTER_KEY], "online R4 historical schema-7 metadata mismatch")
                require("inherited_cache_entry_vector" not in saved.files, "online R4 schema-7 shard guessed inherited values")
        if inherited_present:
            require("inherited_cache_entry_vector" in saved.files, "online R4 schema-8 shard lacks inherited values")
        require(str(row["counter_source_kind"]) == source_kind, "contract R4 counter source kind mismatch")
        require(list(row["counter_schema_keys"]) == schema_keys, "contract R4 counter schema keys mismatch")
        require(list(row["unavailable_counter_fields"]) == unavailable, "contract R4 unavailable counter fields mismatch")
        require(int(_scalar(saved, "history_index")) == int(row["path_identifier"]["history_index"]), "online history index mismatch")
        require(int(_scalar(saved, "call_per_image")) == int(row["path_identifier"]["calls_per_image"]), "online call mismatch")
        require(int(_scalar(saved, "optimization_N_F")) == int(row["optimization_N_F"]), "online N_F mismatch")
        require(bool(_scalar(saved, "fully_evaluated_current_path")) and bool(_scalar(saved, "provisional_reportable")), "online path is not reportable")
        require(not bool(_scalar(saved, "terminal_uncommitted_tail")), "online source is terminal tail")
        require(int(_scalar(saved, "additional_physical_calls")) == 0, "online capture made physical calls")
        require(ENGINE.sha256_array(path) == str(_scalar(saved, "path_sha256")) == str(row["path_sha256"]), "online path hash mismatch")
        require(ENGINE.sha256_array(energy) == str(_scalar(saved, "energy_sha256")) == str(row["online_energy_sha256"]), "online energy hash mismatch")
        require(ENGINE.sha256_array(force) == str(_scalar(saved, "force_sha256")) == str(row["online_force_sha256"]), "online force hash mismatch")
        diagnostic = ENGINE.path_diagnostics(path, energy, force[1:-1])
        i_perp = float(diagnostic["i_perp"])
        require(_same_float(i_perp, float(_scalar(saved, "discrete_I_perp"))), "online shard I_perp does not reconstruct exactly")
        require(_same_float(i_perp, float(row["online_discrete_I_perp"])), "contract I_perp does not reconstruct exactly")
    return path, energy, force, i_perp


def validate_and_load(contract_path: Path, host_role: str):
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(contract.get("schema") == SCHEMA and contract.get("frozen") is True, "invalid/unfrozen IVM v2 contract")
    require(contract.get("method") == "IVM" and int(contract.get("jobs", -1)) == 1, "IVM method/jobs mismatch")
    require(contract.get("metric_policy") == METRIC_POLICY, "metric policy mismatch")
    adapter = ENGINE.require_file(contract["evaluator"], contract_path, "IVM evaluator")
    require(adapter == Path(__file__).resolve(), "contract binds another IVM evaluator")
    engine = ENGINE.require_file(contract["joint_engine"], contract_path, "joint engine")
    require(engine == ENGINE_PATH.resolve(), "contract binds another joint engine")
    bb_core = ENGINE.require_file(contract["bb_core"], contract_path, "bb_core")
    require(bb_core == (ROOT / "src" / "bb_core.py").resolve(), "contract binds another bb_core")
    topology = ENGINE.require_file(contract["topology"], contract_path, "topology")
    verification_path = ENGINE.require_file(contract["full_state_verification"], contract_path, "full-state verification")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    require(verification.get("schema") == VERIFICATION_SCHEMA and verification.get("status") == VERIFICATION_STATUS, "full-state verification is not PASS")
    require(verification.get("all_discrete_I_perp_values_exactly_reconstructed") is True, "full-state verification did not reconstruct I_perp")
    require(int(verification.get("additional_physical_calls_from_capture", -1)) == 0, "full-state capture made physical calls")
    verified_closure = verification.get("bindings", {})
    contract_closure = contract.get("R4_full_state_closure", {})
    required_roles = {"registration", "authorization", "reference_result", "replay_result", "replay_audit", "capture_manifest", "capture_audit"}
    require(set(contract_closure) == required_roles and contract_closure == verified_closure, "contract does not bind the exact R4 verification closure")
    closure_paths = {role: ENGINE.require_file(record, contract_path, f"R4 {role}") for role, record in contract_closure.items()}
    require(closure_paths["registration"] == CANONICAL_R4_REGISTRATION.resolve(strict=True), "verification is not bound to the canonical R4 registration")
    require(closure_paths["authorization"] == CANONICAL_R4_AUTHORIZATION.resolve(strict=True), "verification is not bound to the canonical R4 authorization")
    registration = json.loads(closure_paths["registration"].read_text(encoding="utf-8"))
    authorization = json.loads(closure_paths["authorization"].read_text(encoding="utf-8"))
    replay_audit = json.loads(closure_paths["replay_audit"].read_text(encoding="utf-8"))
    capture_manifest = json.loads(closure_paths["capture_manifest"].read_text(encoding="utf-8"))
    capture_audit = json.loads(closure_paths["capture_audit"].read_text(encoding="utf-8"))
    require(registration.get("schema") == "bb-irivm-selected-full-state-history-output-only-replay-v4" and registration.get("status") == "FROZEN_BEFORE_FIRST_R4_FULL_STATE_HISTORY_REPLAY_PHYSICAL_CALL" and registration.get("label") == R4_LABEL, "canonical R4 registration identity mismatch")
    require(authorization.get("schema") == "bb-irivm-selected-full-state-history-output-only-replay-authorization-v4" and authorization.get("enabled") is True and authorization.get("label") == R4_LABEL, "canonical R4 authorization identity mismatch")
    require(replay_audit.get("consistent") is True, "R4 replay runtime audit is inconsistent")
    require(capture_manifest.get("schema") == "bb-irivm-full-state-history-manifest-v2" and capture_manifest.get("status") == "COMPLETE_PENDING_EXACT_PREFIX_VERIFICATION" and capture_manifest.get("label") == R4_LABEL, "R4 capture manifest mismatch")
    require(capture_audit.get("schema") == "bb-irivm-full-state-history-audit-v2" and capture_audit.get("status") == "PASS_SCHEMA_AWARE_STORAGE_ONLY_CAPTURE_PENDING_EXACT_PREFIX_VERIFICATION" and capture_audit.get("label") == R4_LABEL and int(capture_audit.get("additional_physical_calls", -1)) == 0 and capture_audit.get("missing_counter_values_guessed_or_backfilled") is False, "R4 capture audit mismatch")
    inventory = {int(item["call_per_image"]): item for item in verification["path_inventory"]}
    rows = contract.get("paths")
    require(isinstance(rows, list) and len(rows) == 4, "IVM v2 contract must bind four paths")
    require(tuple(int(row["path_identifier"]["calls_per_image"]) for row in rows) == CALLS, "IVM calls must be 2/5/10/50")
    require(tuple(int(row["optimization_N_F"]) for row in rows) == EXPECTED_NF, "IVM N_F sequence mismatch")
    loaded = []
    for row in rows:
        call = int(row["path_identifier"]["calls_per_image"])
        item = inventory.get(call)
        require(item is not None and item.get("formally_reportable") is True and item.get("terminal_uncommitted_charged_tail") is False, f"call {call} is not a reportable full state")
        require(row.get("method") == "IVM" and row.get("host_role") == host_role, f"call {call} method/host mismatch")
        require(row.get("original_discrete_policy") == "verified_online_paid_path_energy_full_force_zero_new_calls", f"call {call} original discrete policy mismatch")
        require(row.get("dense_policy") == "fresh_or_complete_validated_shard_500_joint_EF", f"call {call} dense policy mismatch")
        source = ENGINE.require_file(row["online_full_state_source"], contract_path, f"call {call} online shard")
        require(str(source) == str(Path(item["shard"]).expanduser().resolve(strict=True)), f"call {call} source path differs from verification")
        require(str(row["online_full_state_source"]["sha256"]) == str(item["shard_sha256"]), f"call {call} source SHA differs from verification")
        require(int(row["optimization_N_F"]) == int(item["optimization_N_F"]), f"call {call} N_F differs from verification")
        for contract_key, item_key in (("path_sha256", "path_sha256"), ("online_energy_sha256", "energy_sha256"), ("online_force_sha256", "force_sha256")):
            require(str(row[contract_key]) == str(item[item_key]), f"call {call} {contract_key} differs from verification")
        require(_same_float(float(row["online_discrete_I_perp"]), float(item["discrete_I_perp_eV2_per_A"])), f"call {call} I_perp differs from verification")
        require(str(row["counter_source_kind"]) == str(item["counter_source_kind"]), f"call {call} counter source differs from verification")
        require(list(row["counter_schema_keys"]) == list(item["counter_schema_keys"]), f"call {call} counter schema differs from verification")
        require(list(row["unavailable_counter_fields"]) == list(item["unavailable_counter_fields"]), f"call {call} unavailable counters differ from verification")
        path, energy, force, i_perp = _load_online_source(source, row)
        loaded.append({**row, "source_path": source, "path": path, "online_energy": energy, "online_force": force, "i_perp": i_perp})
    locks = [Path(value).expanduser().resolve() for value in contract["host_local_lock_paths"]]
    require(len(locks) == 2 and len(set(locks)) == 2, "two distinct lock paths required")
    physical = contract.get("physical_call_policy", {})
    require(physical.get("original_online_paid_joint_EF_reuse_per_path") == N_IMAGES, "online state reuse policy mismatch")
    require(physical.get("new_original_discrete_joint_EF_per_path") == 0, "IVM original path must make zero new calls")
    require(physical.get("dense_joint_EF_per_path") == N_DENSE and physical.get("dense_full_force_storage") is True, "dense physical/storage policy mismatch")
    require(physical.get("optimization_N_F_contribution") == 0, "diagnostics must not enter optimization N_F")
    storage = contract.get("storage_policy", {})
    require(storage.get("authoritative_unit") == "complete_atomic_path_shard", "shard authority mismatch")
    require(storage.get("aggregate_role") == "index_metrics_and_plot_compatible_dense_energy_not_dense_coordinates_or_forces", "aggregate role mismatch")
    require(int(storage.get("minimum_free_bytes_reserve", -1)) == MINIMUM_FREE_BYTES_RESERVE, "disk reserve mismatch")
    require(int(storage.get("estimated_uncompressed_shard_bytes_per_path", -1)) == ESTIMATED_SHARD_BYTES, "disk estimate mismatch")
    durability = contract.get("durability_policy", {})
    for key in ("separate_online_reuse_and_dense_joint_append_only_fsync_WAL", "atomic_complete_path_shard", "shard_contains_dense_coordinates_energy_full_force", "manifest_binds_every_shard_and_array_hash", "refuse_final_output_overwrite"):
        require(durability.get(key) is True, f"durability policy {key} must be true")
    require(durability.get("restart_reuse_unit") == "complete_fully_validated_path_shard", "restart unit mismatch")
    require(durability.get("reuse_individual_incomplete_WAL_successes") is False, "incomplete WAL reuse must be false")
    return contract, topology, locks, loaded


def shard_path_for(output: Path, call: int) -> Path:
    return output.with_suffix("").with_name(output.stem + ".shards") / f"ivm_call{call:04d}_joint_ef_arc500_v2.npz"


def _validate_shard(path: Path, row: dict, contract_sha: str, topology_sha: str, evaluator_sha: str, engine_sha: str, bb_core_sha: str):
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "schema", "status", "method", "label", "optimization_N_F", "path_identifier_json",
            "contract_sha256", "evaluator_sha256", "joint_engine_sha256", "bb_core_sha256", "topology_sha256",
            "online_full_state_source", "online_full_state_source_sha256", "selected_path", "selected_path_sha256",
            "online_energy32_eV", "online_force32_eV_per_A", "online_energy_sha256", "online_force_sha256", "discrete_I_perp",
            "dense_coordinates_A", "dense_energy_eV", "dense_force_eV_per_A", "dense_coordinate_sha256", "dense_energy_sha256", "dense_force_sha256",
            "densified_Delta_E_kcal_per_mol", "number_of_points", "optimization_N_F_contribution",
        }
        require(required.issubset(saved.files), "incomplete IVM dense shard")
        require(str(_scalar(saved, "schema")) == SHARD_SCHEMA and str(_scalar(saved, "status")) == "PASS" and str(_scalar(saved, "method")) == "IVM", "IVM dense shard identity mismatch")
        require(str(_scalar(saved, "contract_sha256")) == contract_sha and str(_scalar(saved, "evaluator_sha256")) == evaluator_sha and str(_scalar(saved, "joint_engine_sha256")) == engine_sha and str(_scalar(saved, "bb_core_sha256")) == bb_core_sha and str(_scalar(saved, "topology_sha256")) == topology_sha, "IVM dense shard provenance mismatch")
        require(int(_scalar(saved, "optimization_N_F")) == int(row["optimization_N_F"]) and str(_scalar(saved, "label")) == str(row["label"]), "IVM dense shard row mismatch")
        require(json.loads(str(_scalar(saved, "path_identifier_json"))) == row["path_identifier"], "IVM dense shard identifier mismatch")
        require(str(_scalar(saved, "online_full_state_source")) == str(row["source_path"]) and str(_scalar(saved, "online_full_state_source_sha256")) == str(row["online_full_state_source"]["sha256"]), "IVM online source binding mismatch")
        selected = np.asarray(saved["selected_path"], dtype=np.float64)
        online_e = np.asarray(saved["online_energy32_eV"], dtype=np.float64)
        online_f = np.asarray(saved["online_force32_eV_per_A"], dtype=np.float64)
        dense = np.asarray(saved["dense_coordinates_A"], dtype=np.float64)
        dense_e = np.asarray(saved["dense_energy_eV"], dtype=np.float64)
        dense_f = np.asarray(saved["dense_force_eV_per_A"], dtype=np.float64)
        require(selected.shape == online_f.shape == (N_IMAGES, DIMENSION) and online_e.shape == (N_IMAGES,), "IVM online arrays shape mismatch")
        require(dense.shape == dense_f.shape == (N_DENSE, DIMENSION) and dense_e.shape == (N_DENSE,), "IVM dense arrays shape mismatch")
        require(np.all(np.isfinite(selected)) and np.all(np.isfinite(online_e)) and np.all(np.isfinite(online_f)) and np.all(np.isfinite(dense)) and np.all(np.isfinite(dense_e)) and np.all(np.isfinite(dense_f)), "IVM shard non-finite")
        require(ENGINE.sha256_array(selected) == str(_scalar(saved, "selected_path_sha256")) == str(row["path_sha256"]), "IVM selected path hash mismatch")
        require(ENGINE.sha256_array(online_e) == str(_scalar(saved, "online_energy_sha256")) == str(row["online_energy_sha256"]), "IVM online energy hash mismatch")
        require(ENGINE.sha256_array(online_f) == str(_scalar(saved, "online_force_sha256")) == str(row["online_force_sha256"]), "IVM online force hash mismatch")
        require(np.array_equal(selected, row["path"]) and np.array_equal(online_e, row["online_energy"]) and np.array_equal(online_f, row["online_force"]), "IVM copied online arrays differ from source")
        computed_dense, _ = ENGINE.uniform_arc500(selected)
        require(np.array_equal(dense, computed_dense) and ENGINE.sha256_array(dense) == str(_scalar(saved, "dense_coordinate_sha256")), "IVM dense coordinate closure mismatch")
        require(ENGINE.sha256_array(dense_e) == str(_scalar(saved, "dense_energy_sha256")) and ENGINE.sha256_array(dense_f) == str(_scalar(saved, "dense_force_sha256")), "IVM dense E/F hash mismatch")
        i_perp = float(ENGINE.path_diagnostics(selected, online_e, online_f[1:-1])["i_perp"])
        require(_same_float(i_perp, float(_scalar(saved, "discrete_I_perp"))), "IVM dense shard Iperp mismatch")
        delta_e = float((np.max(dense_e) - dense_e[0]) * ENGINE.EV_TO_KCAL)
        require(_same_float(delta_e, float(_scalar(saved, "densified_Delta_E_kcal_per_mol"))), "IVM dense shard Delta E mismatch")
        require(int(_scalar(saved, "number_of_points")) == N_DENSE and int(_scalar(saved, "optimization_N_F_contribution")) == 0, "IVM shard count policy mismatch")
        return {key: np.asarray(saved[key]).copy() for key in saved.files}


def _evaluate_one(row_index: int, row: dict, backend, online_wal, dense_wal, contract_sha: str, topology_sha: str, evaluator_sha: str, engine_sha: str, bb_core_sha: str, shard_path: Path):
    online_wal.append({
        "event": "verified_online_paid_full_state_reused", "utc": ENGINE.utc_now(), "row_index": row_index,
        "label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "path_identifier": row["path_identifier"],
        "path_sha256": row["path_sha256"], "online_energy_sha256": row["online_energy_sha256"], "online_force_sha256": row["online_force_sha256"],
        "source": str(row["source_path"]), "source_sha256": row["online_full_state_source"]["sha256"],
        "new_physical_calls": 0, "optimization_N_F_contribution": 0,
    })
    dense, geometry = ENGINE.uniform_arc500(row["path"])
    dense_e = np.empty(N_DENSE, dtype=np.float64)
    dense_f = np.empty((N_DENSE, DIMENSION), dtype=np.float64)
    dense_wal.append({"event": "path_stage_started", "utc": ENGINE.utc_now(), "stage": "dense_arc500_joint_EF", "row_index": row_index, "label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "path_sha256": row["path_sha256"], "dense_coordinate_sha256": ENGINE.sha256_array(dense), "number_of_points": N_DENSE})
    for index, coordinate in enumerate(dense):
        dense_e[index], dense_f[index] = ENGINE._attempt(dense_wal, backend, row_index, row, "dense_arc500_joint_EF", index, coordinate)
    delta_e = float((np.max(dense_e) - dense_e[0]) * ENGINE.EV_TO_KCAL)
    dense_wal.append({"event": "path_stage_data_complete_uncommitted", "utc": ENGINE.utc_now(), "stage": "dense_arc500_joint_EF", "row_index": row_index, "label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "path_sha256": row["path_sha256"], "joint_attempts": N_DENSE, "joint_successes": N_DENSE, "dense_coordinate_sha256": ENGINE.sha256_array(dense), "dense_energy_sha256": ENGINE.sha256_array(dense_e), "dense_force_sha256": ENGINE.sha256_array(dense_f), "densified_Delta_E_kcal_per_mol": delta_e})
    arrays = {
        "schema": np.asarray(SHARD_SCHEMA), "status": np.asarray("PASS"), "method": np.asarray("IVM"), "label": np.asarray(row["label"]),
        "optimization_N_F": np.asarray(int(row["optimization_N_F"]), dtype=np.int64), "path_identifier_json": np.asarray(json.dumps(row["path_identifier"], sort_keys=True)),
        "contract_sha256": np.asarray(contract_sha), "evaluator_sha256": np.asarray(evaluator_sha), "joint_engine_sha256": np.asarray(engine_sha), "bb_core_sha256": np.asarray(bb_core_sha), "topology_sha256": np.asarray(topology_sha),
        "online_full_state_source": np.asarray(str(row["source_path"])), "online_full_state_source_sha256": np.asarray(row["online_full_state_source"]["sha256"]),
        "selected_path": row["path"], "selected_path_sha256": np.asarray(row["path_sha256"]),
        "online_energy32_eV": row["online_energy"], "online_force32_eV_per_A": row["online_force"], "online_energy_sha256": np.asarray(row["online_energy_sha256"]), "online_force_sha256": np.asarray(row["online_force_sha256"]), "discrete_I_perp": np.asarray(row["i_perp"]),
        "dense_coordinates_A": dense, "dense_energy_eV": dense_e, "dense_force_eV_per_A": dense_f, "dense_coordinate_sha256": np.asarray(ENGINE.sha256_array(dense)), "dense_energy_sha256": np.asarray(ENGINE.sha256_array(dense_e)), "dense_force_sha256": np.asarray(ENGINE.sha256_array(dense_f)),
        "densified_Delta_E_kcal_per_mol": np.asarray(delta_e), "total_arclength_A": np.asarray(geometry["total_arclength_A"]), "dense_spacing_A": np.asarray(geometry["dense_spacing_A"]), "number_of_points": np.asarray(N_DENSE),
        "new_original_discrete_joint_EF_attempts": np.asarray(0), "dense_joint_EF_attempts": np.asarray(N_DENSE), "dense_endpoint_joint_EF_attempts": np.asarray(2), "optimization_N_F_contribution": np.asarray(0),
    }
    ENGINE._atomic_save_npz_no_overwrite(shard_path, arrays)
    return _validate_shard(shard_path, row, contract_sha, topology_sha, evaluator_sha, engine_sha, bb_core_sha)


def _validate_online_wal(rows, events):
    matched = []
    for row_index, row in enumerate(rows):
        candidates = [event for event in events if event.get("event") == "verified_online_paid_full_state_reused" and int(event.get("row_index", -1)) == row_index and event.get("path_sha256") == row["path_sha256"]]
        require(candidates, f"{row['label']} lacks online reuse WAL record")
        event = candidates[-1]
        require(event.get("online_energy_sha256") == row["online_energy_sha256"] and event.get("online_force_sha256") == row["online_force_sha256"], f"{row['label']} online reuse WAL hash mismatch")
        require(event.get("source_sha256") == row["online_full_state_source"]["sha256"] and int(event.get("new_physical_calls", -1)) == 0, f"{row['label']} online reuse WAL provenance mismatch")
        matched.append({"label": row["label"], "status": "PASS", "new_physical_calls": 0})
    return matched


def _validate_aggregate(output: Path, rows, shards, contract_sha: str):
    with np.load(output, allow_pickle=False) as saved:
        required = {
            "schema", "status", "method", "authoritative_data_unit", "labels", "optimization_N_F", "path_identifier_json",
            "selected_path", "selected_path_sha256", "discrete_I_perp", "internal_diagnostic_force_eV_per_A", "internal_diagnostic_force_sha256",
            "internal_joint_energy_eV", "internal_joint_energy_sha256", "energy_eV", "energy_sha256", "dense_coordinate_sha256", "dense_force_sha256",
            "densified_Delta_E_kcal_per_mol", "shard_path", "shard_sha256", "contract_sha256", "optimization_N_F_contribution",
        }
        require(required.issubset(saved.files), "IVM aggregate incomplete")
        require(str(_scalar(saved, "schema")) == "bb-dense-four-ivm-joint-ef-result-v2" and str(_scalar(saved, "status")) == "PASS" and str(_scalar(saved, "method")) == "IVM", "IVM aggregate identity mismatch")
        require(str(_scalar(saved, "authoritative_data_unit")) == "complete_atomic_path_shard" and str(_scalar(saved, "contract_sha256")) == contract_sha and int(_scalar(saved, "optimization_N_F_contribution")) == 0, "IVM aggregate provenance mismatch")
        require("dense_coordinates_A" not in saved.files and "dense_force_eV_per_A" not in saved.files, "IVM aggregate duplicates authoritative dense coordinates/forces")
        expected_nf = np.asarray([row["optimization_N_F"] for row in rows], dtype=np.int64)
        require(np.array_equal(saved["optimization_N_F"], expected_nf), "IVM aggregate N_F mismatch")
        require(np.array_equal(saved["labels"].astype(str), np.asarray([row["label"] for row in rows])), "IVM aggregate labels mismatch")
        expected_ids = np.asarray([json.dumps(row["path_identifier"], sort_keys=True) for row in rows])
        require(np.array_equal(saved["path_identifier_json"].astype(str), expected_ids), "IVM aggregate identifiers mismatch")
        array_expected = {
            "selected_path": np.asarray([row["path"] for row in rows]),
            "internal_diagnostic_force_eV_per_A": np.asarray([row["online_force"][1:-1] for row in rows]),
            "internal_joint_energy_eV": np.asarray([row["online_energy"][1:-1] for row in rows]),
            "energy_eV": np.asarray([shard["dense_energy_eV"] for shard in shards]),
        }
        for key, expected in array_expected.items():
            actual = np.asarray(saved[key])
            require(actual.dtype.str == expected.dtype.str and actual.shape == expected.shape and actual.tobytes(order="C") == expected.tobytes(order="C"), f"IVM aggregate {key} value mismatch")
        scalar_expected = {
            "discrete_I_perp": np.asarray([float(row["i_perp"]) for row in rows], dtype=np.float64),
            "densified_Delta_E_kcal_per_mol": np.asarray([float(_scalar(shard, "densified_Delta_E_kcal_per_mol")) for shard in shards], dtype=np.float64),
        }
        for key, expected in scalar_expected.items():
            require(np.asarray(saved[key], dtype=np.float64).tobytes(order="C") == expected.tobytes(order="C"), f"IVM aggregate {key} value mismatch")
        hash_expected = {
            "selected_path_sha256": [row["path_sha256"] for row in rows],
            "internal_diagnostic_force_sha256": [ENGINE.sha256_array(row["online_force"][1:-1]) for row in rows],
            "internal_joint_energy_sha256": [ENGINE.sha256_array(row["online_energy"][1:-1]) for row in rows],
            "energy_sha256": [str(_scalar(shard, "dense_energy_sha256")) for shard in shards],
            "dense_coordinate_sha256": [str(_scalar(shard, "dense_coordinate_sha256")) for shard in shards],
            "dense_force_sha256": [str(_scalar(shard, "dense_force_sha256")) for shard in shards],
            "shard_sha256": [ENGINE.sha256_file(shard_path_for(output, call)) for call in CALLS],
        }
        for key, expected in hash_expected.items():
            require(np.array_equal(saved[key].astype(str), np.asarray(expected)), f"IVM aggregate {key} mismatch")
        expected_paths = np.asarray([str(shard_path_for(output, call)) for call in CALLS])
        require(np.array_equal(saved["shard_path"].astype(str), expected_paths), "IVM aggregate shard paths mismatch")


def run(contract_path: Path, topology: Path, locks, rows, output: Path) -> None:
    contract_sha = ENGINE.sha256_file(contract_path)
    topology_sha = ENGINE.sha256_file(topology)
    evaluator_sha = ENGINE.sha256_file(Path(__file__))
    engine_sha = ENGINE.sha256_file(ENGINE_PATH)
    bb_core_sha = ENGINE.sha256_file(ROOT / "src" / "bb_core.py")
    online_wal_path = output.with_suffix(".online_paid_state.wal.jsonl")
    dense_wal_path = output.with_suffix(".dense_joint_ef.wal.jsonl")
    manifest_path = output.with_suffix(".manifest.json")
    audit_path = output.with_suffix(".audit.json")
    if audit_path.exists():
        raise FileExistsError("refusing to overwrite completed IVM audit")
    if manifest_path.exists() and not output.exists():
        raise RuntimeError("orphan IVM manifest without aggregate")
    if output.exists() and any(not shard_path_for(output, call).exists() for call in CALLS):
        raise RuntimeError("IVM aggregate exists without all authoritative shards")
    output.parent.mkdir(parents=True, exist_ok=True)
    missing = sum(not shard_path_for(output, int(row["path_identifier"]["calls_per_image"])).exists() for row in rows)
    required_free = MINIMUM_FREE_BYTES_RESERVE + missing * ESTIMATED_SHARD_BYTES
    observed_free = shutil.disk_usage(output.parent).free
    require(observed_free >= required_free, f"insufficient free disk: {observed_free} < {required_free}")
    online_wal = ENGINE.DurableWal(online_wal_path, ONLINE_WAL_SCHEMA, contract_path, contract_sha)
    dense_wal = ENGINE.DurableWal(dense_wal_path, DENSE_WAL_SCHEMA, contract_path, contract_sha)
    shards = []
    fresh, reused = [], []
    backend = None
    try:
        with ENGINE.acquire_two_locks(locks) as held_locks:
            for row_index, row in enumerate(rows):
                call = int(row["path_identifier"]["calls_per_image"])
                path = shard_path_for(output, call)
                if path.exists():
                    shard = _validate_shard(path, row, contract_sha, topology_sha, evaluator_sha, engine_sha, bb_core_sha)
                    reused.append(call)
                    event = {"event": "complete_validated_shard_reused", "utc": ENGINE.utc_now(), "row_index": row_index, "label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "path_sha256": row["path_sha256"], "shard": str(path), "shard_sha256": ENGINE.sha256_file(path), "individual_incomplete_WAL_attempts_reused": 0}
                    online_wal.append(event)
                    dense_wal.append(event)
                else:
                    online_wal.append({"event": "path_restart_from_zero", "utc": ENGINE.utc_now(), "row_index": row_index, "label": row["label"], "path_sha256": row["path_sha256"], "individual_incomplete_WAL_attempts_reused": 0})
                    dense_wal.append({"event": "path_restart_from_zero", "utc": ENGINE.utc_now(), "row_index": row_index, "label": row["label"], "path_sha256": row["path_sha256"], "individual_incomplete_WAL_attempts_reused": 0})
                    if backend is None:
                        backend = ENGINE.OpenMMBackend(topology, platform_preference=("CPU",), precision="double", label="bb_ivm_dense_joint_ef_v2")
                    before = int(backend.raw_joint_attempts)
                    shard = _evaluate_one(row_index, row, backend, online_wal, dense_wal, contract_sha, topology_sha, evaluator_sha, engine_sha, bb_core_sha, path)
                    require(int(backend.raw_joint_attempts) - before == N_DENSE, "IVM dense backend attempt mismatch")
                    require(int(backend.raw_joint_attempts) == int(backend.raw_joint_evaluations), "IVM dense backend success mismatch")
                    fresh.append(call)
                    published = {"event": "complete_shard_published", "utc": ENGINE.utc_now(), "row_index": row_index, "label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "path_sha256": row["path_sha256"], "shard": str(path), "shard_sha256": ENGINE.sha256_file(path)}
                    online_wal.append(published)
                    dense_wal.append(published)
                shards.append(shard)
            online_closure = _validate_online_wal(rows, ENGINE._read_wal_events(online_wal_path))
            dense_events = ENGINE._read_wal_events(dense_wal_path)
            dense_closure = []
            for row_index, (row, shard) in enumerate(zip(rows, shards)):
                dense_closure.append({"label": row["label"], **ENGINE._validate_stage_wal_closure(dense_events, row_index, row, "dense_arc500_joint_EF", np.asarray(shard["dense_coordinates_A"]), np.asarray(shard["dense_energy_eV"]), np.asarray(shard["dense_force_eV_per_A"]))})
            arrays = {
                "schema": np.asarray("bb-dense-four-ivm-joint-ef-result-v2"), "status": np.asarray("PASS"), "method": np.asarray("IVM"), "authoritative_data_unit": np.asarray("complete_atomic_path_shard"),
                "labels": np.asarray([row["label"] for row in rows]), "optimization_N_F": np.asarray([row["optimization_N_F"] for row in rows], dtype=np.int64), "path_identifier_json": np.asarray([json.dumps(row["path_identifier"], sort_keys=True) for row in rows]),
                "selected_path": np.asarray([row["path"] for row in rows]), "selected_path_sha256": np.asarray([row["path_sha256"] for row in rows]), "discrete_I_perp": np.asarray([row["i_perp"] for row in rows]),
                "internal_diagnostic_force_eV_per_A": np.asarray([row["online_force"][1:-1] for row in rows]), "internal_diagnostic_force_sha256": np.asarray([ENGINE.sha256_array(row["online_force"][1:-1]) for row in rows]), "internal_joint_energy_eV": np.asarray([row["online_energy"][1:-1] for row in rows]), "internal_joint_energy_sha256": np.asarray([ENGINE.sha256_array(row["online_energy"][1:-1]) for row in rows]),
                "diagnostic_force_attempts_per_path": np.zeros(4, dtype=np.int64), "fixed_endpoint_diagnostic_force_attempts_per_path": np.zeros(4, dtype=np.int64),
                "energy_eV": np.asarray([shard["dense_energy_eV"] for shard in shards]), "energy_sha256": np.asarray([str(_scalar(shard, "dense_energy_sha256")) for shard in shards]), "dense_coordinate_sha256": np.asarray([str(_scalar(shard, "dense_coordinate_sha256")) for shard in shards]), "dense_force_sha256": np.asarray([str(_scalar(shard, "dense_force_sha256")) for shard in shards]), "densified_Delta_E_kcal_per_mol": np.asarray([float(_scalar(shard, "densified_Delta_E_kcal_per_mol")) for shard in shards]),
                "diagnostic_energy_attempts_lineage_per_path": np.full(4, N_DENSE), "diagnostic_energy_attempts_fresh_per_path": np.full(4, N_DENSE), "dense_joint_EF_attempts_per_path": np.full(4, N_DENSE), "dense_full_force_stored_per_path": np.ones(4, dtype=bool),
                "number_of_points": np.asarray(N_DENSE), "coordinate_space": np.asarray(ENGINE.METRIC_POLICY["coordinate_space"]), "sampling": np.asarray(ENGINE.METRIC_POLICY["sampling"]), "interpolation": np.asarray(ENGINE.METRIC_POLICY["interpolation"]), "grid_formula": np.asarray(ENGINE.METRIC_POLICY["grid_formula"]),
                "shard_path": np.asarray([str(shard_path_for(output, call)) for call in CALLS]), "shard_sha256": np.asarray([ENGINE.sha256_file(shard_path_for(output, call)) for call in CALLS]),
                "contract": np.asarray(str(contract_path)), "contract_sha256": np.asarray(contract_sha), "evaluator_sha256": np.asarray(evaluator_sha), "joint_engine_sha256": np.asarray(engine_sha), "bb_core_sha256": np.asarray(bb_core_sha), "topology": np.asarray(str(topology)), "topology_sha256": np.asarray(topology_sha), "force_wal": np.asarray(str(online_wal_path)), "energy_wal": np.asarray(str(dense_wal_path)), "dense_joint_wal": np.asarray(str(dense_wal_path)), "held_host_locks": np.asarray(held_locks), "optimization_N_F_contribution": np.asarray(0),
            }
            if output.exists():
                _validate_aggregate(output, rows, shards, contract_sha)
                aggregate_event = "preexisting_aggregate_validated"
            else:
                ENGINE._atomic_save_npz_no_overwrite(output, arrays)
                _validate_aggregate(output, rows, shards, contract_sha)
                aggregate_event = "aggregate_published"
            online_wal.append({"event": aggregate_event, "utc": ENGINE.utc_now(), "output": str(output), "output_sha256": ENGINE.sha256_file(output)})
            dense_wal.append({"event": aggregate_event, "utc": ENGINE.utc_now(), "output": str(output), "output_sha256": ENGINE.sha256_file(output)})
    finally:
        online_wal.close()
        dense_wal.close()
    manifest = {
        "schema": "bb-dense-four-ivm-joint-ef-shard-manifest-v2", "status": "PASS", "created_utc": ENGINE.utc_now(),
        "contract": str(contract_path), "contract_sha256": contract_sha, "evaluator_sha256": evaluator_sha, "joint_engine_sha256": engine_sha, "bb_core_sha256": bb_core_sha, "topology_sha256": topology_sha,
        "aggregate": str(output), "aggregate_sha256": ENGINE.sha256_file(output),
        "shards": [{"label": row["label"], "calls_per_image": int(row["path_identifier"]["calls_per_image"]), "optimization_N_F": int(row["optimization_N_F"]), "path_identifier": row["path_identifier"], "path": str(shard_path_for(output, int(row["path_identifier"]["calls_per_image"]))), "sha256": ENGINE.sha256_file(shard_path_for(output, int(row["path_identifier"]["calls_per_image"]))), "selected_path_sha256": str(_scalar(shard, "selected_path_sha256")), "online_energy_sha256": str(_scalar(shard, "online_energy_sha256")), "online_force_sha256": str(_scalar(shard, "online_force_sha256")), "dense_coordinate_sha256": str(_scalar(shard, "dense_coordinate_sha256")), "dense_energy_sha256": str(_scalar(shard, "dense_energy_sha256")), "dense_force_sha256": str(_scalar(shard, "dense_force_sha256"))} for row, shard in zip(rows, shards)],
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("schema", "status", "contract", "contract_sha256", "evaluator_sha256", "joint_engine_sha256", "bb_core_sha256", "topology_sha256", "aggregate", "aggregate_sha256", "shards"):
            require(existing_manifest.get(key) == manifest.get(key), f"preexisting IVM manifest {key} mismatch")
    else:
        ENGINE._atomic_write_json_no_overwrite(manifest_path, manifest)
    audit = {
        "schema": AUDIT_SCHEMA, "status": "PASS", "completed_utc": ENGINE.utc_now(), "method": "IVM", "contract": str(contract_path), "contract_sha256": contract_sha,
        "evaluator_sha256": evaluator_sha, "joint_engine_sha256": engine_sha, "bb_core_sha256": bb_core_sha, "topology_sha256": topology_sha,
        "output": str(output), "output_sha256": ENGINE.sha256_file(output), "shard_manifest": str(manifest_path), "shard_manifest_sha256": ENGINE.sha256_file(manifest_path),
        "online_paid_state_wal": str(online_wal_path), "online_paid_state_wal_sha256": ENGINE.sha256_file(online_wal_path), "dense_joint_ef_wal": str(dense_wal_path), "dense_joint_ef_wal_sha256": ENGINE.sha256_file(dense_wal_path),
        "path_count": 4, "fresh_paths_this_invocation": fresh, "complete_validated_shards_reused": reused, "online_paid_original_full_states_reused": 4 * N_IMAGES, "new_original_discrete_joint_EF_attempts": 0,
        "dense_joint_EF_attempts_lineage": 4 * N_DENSE, "dense_full_force_vectors_stored": 4 * N_DENSE, "dense_endpoint_joint_EF_attempts_lineage": 8,
        "online_source_closure": online_closure, "dense_WAL_shard_pointwise_closure": {"status": "PASS", "lifetime": ENGINE._wal_lifetime_stats(dense_events), "paths": dense_closure},
        "free_disk_gate": {"observed_free_bytes": observed_free, "required_free_bytes": required_free, "minimum_free_bytes_reserve": MINIMUM_FREE_BYTES_RESERVE, "missing_shards_at_start": missing, "status": "PASS"},
        "individual_incomplete_WAL_attempts_reused": 0, "optimization_N_F_contribution": 0,
        "paths": [{
            "label": row["label"], "optimization_N_F": int(row["optimization_N_F"]), "path_identifier": row["path_identifier"],
            "path_sha256": row["path_sha256"], "online_energy_sha256": row["online_energy_sha256"], "online_force_sha256": row["online_force_sha256"],
            "discrete_I_perp": float(row["i_perp"]), "densified_Delta_E_kcal_per_mol": float(_scalar(shard, "densified_Delta_E_kcal_per_mol")),
            "dense_coordinate_sha256": str(_scalar(shard, "dense_coordinate_sha256")), "dense_energy_sha256": str(_scalar(shard, "dense_energy_sha256")), "dense_force_sha256": str(_scalar(shard, "dense_force_sha256")),
            "new_original_discrete_joint_EF_attempts": 0, "dense_joint_EF_attempts_lineage": N_DENSE, "dense_full_force_vectors_stored": N_DENSE,
            "individual_incomplete_WAL_attempts_reused": 0, "optimization_N_F_contribution": 0,
        } for row, shard in zip(rows, shards)],
    }
    ENGINE._atomic_write_json_no_overwrite(audit_path, audit)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host-role", choices=("local", "remote"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    ENGINE.check_thread_caps()
    contract_path = args.contract.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    require(output.suffix == ".npz", "--output must end in .npz")
    contract, topology, locks, rows = validate_and_load(contract_path, args.host_role)
    summary = {
        "status": "ZERO_OPENMM_PREFLIGHT_PASS", "contract": str(contract_path), "contract_sha256": ENGINE.sha256_file(contract_path), "method": "IVM", "host_role": args.host_role,
        "calls_per_image": list(CALLS), "optimization_N_F": list(EXPECTED_NF), "path_sha256": [row["path_sha256"] for row in rows],
        "online_paid_original_full_state_vectors_reused": 4 * N_IMAGES, "new_original_discrete_joint_EF_attempts_planned": 0,
        "dense_joint_EF_attempts_planned": 4 * N_DENSE, "dense_full_force_vectors_planned_for_storage": 4 * N_DENSE, "optimization_N_F_contribution": 0,
    }
    if args.preflight_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    run(contract_path, topology, locks, rows, output)


if __name__ == "__main__":
    main()
