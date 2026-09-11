#!/usr/bin/env python3
"""Evaluate four verified short-replay IVM paths on Cartesian arc500 with joint E/F."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENGINE_PATH = HERE / "evaluate_bb_dense_four_paths_joint_v2.py"
FREEZER_PATH = HERE / "freeze_bb_dense_ivm_short_four_joint_contract.py"
SPEC = importlib.util.spec_from_file_location("bb_joint_engine_short_ivm", ENGINE_PATH)
ENGINE = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(ENGINE)

SCHEMA = "bb-dense-four-ivm-short-prefix-joint-ef-contract-v1"
SHARD_SCHEMA = "bb-dense-ivm-short-prefix-path-joint-ef-shard-v1"
RESULT_SCHEMA = "bb-dense-four-ivm-short-prefix-joint-ef-result-v1"
AUDIT_SCHEMA = "bb-dense-four-ivm-short-prefix-joint-ef-audit-v1"
SOURCE_WAL_SCHEMA = "bb-ivm-short-prefix-source-reuse-wal-v1"
DENSE_WAL_SCHEMA = "bb-ivm-short-prefix-dense-joint-ef-wal-v1"
VERIFICATION_SCHEMA = "bb-irivm-selected-legacy-replay-verification-v1"
VERIFICATION_STATUS = "PASS_EXACT_OUTPUT_ONLY_PREFIX"
CALLS = (2, 5, 10, 50)
EXPECTED_NF = (60, 150, 300, 1329)
N_IMAGES, DIMENSION, N_DENSE = 32, 9396, 500
MINIMUM_FREE_BYTES_RESERVE = 8 * 1024**3
ESTIMATED_SHARD_BYTES = (2 * N_DENSE * DIMENSION + N_IMAGES * DIMENSION) * 8 + 8 * 1024**2
CANONICAL_VERIFICATION = ROOT / "reports/verifications/BB_IRIVM_SELECTED_LEGACY_REPLAY_NF1500_20260902_R1.json"
CANONICAL_VERIFICATION_SHA256 = "dfa12125bd68420e9c196443121b34104f83202fde64113b37856281f7fae322"
CANONICAL_REGISTRATION = ROOT / "registrations/diagnostics/BB_IRIVM_SELECTED_LEGACY_REPLAY_NF1500_20260902_R1.json"
CANONICAL_AUTHORIZATION = ROOT / "registrations/execution_authorizations/bb_irivm_a1e0_r01_m05_inf1500_legacy_replay_r1.json"
SHORT_LABEL = "bb_irivm_a1e0_r01_m05_inf1500_legacy_replay_r1"
CANONICAL_LOCKS = {
    Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock").resolve(),
    Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock").resolve(),
}
METRIC_POLICY = {
    **ENGINE.METRIC_POLICY,
    "original_discrete_state": "verified_short_replay_path_and_sealed_online_Iperp_zero_new_calls",
}


def require(condition: bool, message: str) -> None: ENGINE.require(condition, message)
def scalar(saved, key: str): return np.asarray(saved[key]).item()
def same_float(a, b) -> bool: return np.asarray(float(a), dtype=np.float64).tobytes() == np.asarray(float(b), dtype=np.float64).tobytes()


def load_contract(contract_path: Path, host_role: str):
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(contract.get("schema") == SCHEMA and contract.get("frozen") is True, "invalid short-prefix contract")
    require(contract.get("method") == "IVM" and int(contract.get("jobs", -1)) == 1, "method/jobs mismatch")
    require(contract.get("metric_policy") == METRIC_POLICY, "metric policy mismatch")
    require(ENGINE.require_file(contract["evaluator"], contract_path, "evaluator") == Path(__file__).resolve(), "evaluator binding mismatch")
    require(ENGINE.require_file(contract["contract_freezer"], contract_path, "contract freezer") == FREEZER_PATH.resolve(), "contract freezer binding mismatch")
    require(ENGINE.require_file(contract["joint_engine"], contract_path, "joint engine") == ENGINE_PATH.resolve(), "engine binding mismatch")
    require(ENGINE.require_file(contract["bb_core"], contract_path, "bb_core") == (ROOT / "src/bb_core.py").resolve(), "bb_core binding mismatch")
    topology = ENGINE.require_file(contract["topology"], contract_path, "topology")
    require(topology == (ROOT / "inputs/complex_fixed.pdb").resolve(strict=True) and ENGINE.sha256_file(topology) == ENGINE.OpenMMBackend.EXPECTED_PDB_SHA256, "noncanonical topology")
    verification_path = ENGINE.require_file(contract["short_replay_verification"], contract_path, "verification")
    require(verification_path == CANONICAL_VERIFICATION.resolve(strict=True), "noncanonical short verification")
    require(ENGINE.sha256_file(verification_path) == CANONICAL_VERIFICATION_SHA256, "canonical short verification hash mismatch")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    require(verification.get("schema") == VERIFICATION_SCHEMA and verification.get("status") == VERIFICATION_STATUS, "short verification is not PASS")
    require(int(verification.get("exact_prefix_through_call", -1)) == 50 and int(verification.get("exact_prefix_through_optimization_N_F", -1)) == 1329, "verified prefix boundary mismatch")
    closure = contract.get("short_replay_closure", {})
    require(closure == verification.get("bindings") and set(closure) == {"amendment", "authorization", "reference_result", "registration", "replay_audit", "replay_result"}, "short verification closure mismatch")
    bound = {role: ENGINE.require_file(value, contract_path, role) for role, value in closure.items()}
    require(bound["registration"] == CANONICAL_REGISTRATION.resolve(strict=True) and bound["authorization"] == CANONICAL_AUTHORIZATION.resolve(strict=True), "canonical short registration/authorization mismatch")
    reg = json.loads(bound["registration"].read_text(encoding="utf-8")); auth = json.loads(bound["authorization"].read_text(encoding="utf-8")); audit = json.loads(bound["replay_audit"].read_text(encoding="utf-8"))
    require(reg.get("schema") == "bb-irivm-selected-output-only-legacy-replay-v1" and reg.get("status") == "FROZEN_BEFORE_FIRST_REPLAY_PHYSICAL_CALL" and reg.get("label") == SHORT_LABEL, "short registration identity mismatch")
    require(auth.get("schema") == "bb-irivm-selected-output-only-legacy-replay-authorization-v1" and auth.get("enabled") is True and auth.get("label") == SHORT_LABEL, "short authorization identity mismatch")
    require(audit.get("consistent") is True and int(audit.get("endpoint_joint_ef_cache_misses", -1)) == 0, "short replay NoF audit mismatch")
    rows = contract.get("paths")
    require(isinstance(rows, list) and len(rows) == 4, "contract must contain four paths")
    require(tuple(int(row["path_identifier"]["calls_per_image"]) for row in rows) == CALLS, "call sequence mismatch")
    require(tuple(int(row["optimization_N_F"]) for row in rows) == EXPECTED_NF, "N_F sequence mismatch")
    verified = {int(row["call_per_image"]): row for row in verification["new_reporting_rows"]}
    result = bound["replay_result"]
    with np.load(result, allow_pickle=False) as saved:
        required = {"stored_path_calls", "stored_paths", "calls_per_image", "counter_vectors", "i_perp", "path_storage_mode"}
        require(required.issubset(saved.files), "short result fields missing")
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64); stored_paths = np.asarray(saved["stored_paths"], dtype=np.float64)
        calls = np.asarray(saved["calls_per_image"], dtype=np.int64); counters = np.asarray(saved["counter_vectors"], dtype=np.int64); iperp = np.asarray(saved["i_perp"], dtype=np.float64)
        require(str(scalar(saved, "path_storage_mode")) == "legacy" and stored_paths.shape == (stored_calls.size, N_IMAGES, DIMENSION), "short path storage mismatch")
        require(np.array_equal(calls[:50], np.arange(1, 51)) and counters.shape == (calls.size, N_IMAGES) and iperp.shape == calls.shape, "short history mismatch")
        loaded = []
        for row in rows:
            call = int(row["path_identifier"]["calls_per_image"]); si = np.flatnonzero(stored_calls == call); hi = np.flatnonzero(calls == call)
            require(si.size == hi.size == 1, f"call {call} absent/nonunique")
            path = np.ascontiguousarray(stored_paths[int(si[0])]); index = int(hi[0]); nf = int(np.sum(counters[index], dtype=np.int64)); value = float(iperp[index]); item = verified.get(call)
            require(np.all(np.isfinite(path)) and np.isfinite(value), f"call {call} source has non-finite values")
            require(item is not None and item.get("fully_evaluated_current_path") is True and item.get("terminal_tail") is False, f"call {call} not eligible")
            require(row.get("method") == "IVM" and row.get("host_role") == host_role, f"call {call} host/method mismatch")
            require(row.get("original_discrete_policy") == "verified_short_replay_path_and_sealed_online_Iperp_zero_new_calls", f"call {call} discrete policy mismatch")
            require(row.get("dense_policy") == "fresh_or_complete_validated_shard_500_joint_EF", f"call {call} dense policy mismatch")
            require(index == int(row["path_identifier"]["history_index"]) == int(item["history_index"]), f"call {call} history index mismatch")
            require(nf == int(row["optimization_N_F"]) == int(item["optimization_N_F"]), f"call {call} N_F mismatch")
            require(same_float(value, row["online_discrete_I_perp"]) and same_float(value, item["discrete_I_perp_eV2_per_A"]), f"call {call} Iperp mismatch")
            require(ENGINE.sha256_array(path) == row["path_sha256"] == item["selected_path_sha256"], f"call {call} path hash mismatch")
            require(ENGINE.require_file(row["short_replay_source"], contract_path, f"call {call} source") == result, f"call {call} source mismatch")
            loaded.append({**row, "path": path, "i_perp": value, "source_path": result})
    locks = [Path(value).expanduser().resolve() for value in contract["host_local_lock_paths"]]
    require(len(locks) == len(set(locks)) == 2 and set(locks) == CANONICAL_LOCKS, "canonical two-lock set required")
    policy = contract.get("physical_call_policy", {})
    require(policy.get("original_discrete_new_calls_per_path") == 0 and policy.get("dense_joint_EF_per_path") == N_DENSE and policy.get("dense_full_force_storage") is True and policy.get("optimization_N_F_contribution") == 0, "physical policy mismatch")
    require(policy.get("dense_endpoint_joint_EF_per_path") == 2 and policy.get("jobs") == 1 and policy.get("all_six_thread_caps") == 1 and policy.get("two_host_local_fcntl_locks") is True, "physical execution policy mismatch")
    storage = contract.get("storage_policy", {})
    require(storage.get("authoritative_unit") == "complete_atomic_path_shard", "shard authority mismatch")
    require(storage.get("aggregate_role") == "index_metrics_and_plot_compatible_dense_energy_not_dense_coordinates_or_forces", "aggregate storage role mismatch")
    require(int(storage.get("minimum_free_bytes_reserve", -1)) == MINIMUM_FREE_BYTES_RESERVE, "free disk reserve mismatch")
    require(int(storage.get("estimated_uncompressed_shard_bytes_per_path", -1)) == ESTIMATED_SHARD_BYTES, "shard size estimate mismatch")
    durability = contract.get("durability_policy", {})
    for key in ("separate_source_and_dense_append_only_fsync_WAL", "atomic_complete_path_shard", "shard_contains_dense_coordinates_energy_full_force", "manifest_binds_every_shard_and_array_hash", "source_and_dense_WAL_bind_published_shard", "refuse_final_output_overwrite"):
        require(durability.get(key) is True, f"durability policy {key} must be true")
    require(durability.get("restart_reuse_unit") == "complete_fully_validated_path_shard", "restart reuse policy mismatch")
    require(durability.get("reuse_individual_incomplete_WAL_successes") is False, "incomplete WAL reuse must be false")
    return contract, topology, locks, loaded


def shard_path(output: Path, call: int) -> Path: return output.with_suffix("").with_name(output.stem + ".shards") / f"ivm_call{call:04d}_arc500_joint_ef.npz"


def validate_shard(path: Path, row: dict, contract_sha: str, topology_sha: str):
    with np.load(path, allow_pickle=False) as saved:
        required = {"schema","status","method","label","optimization_N_F","path_identifier_json","contract_sha256","evaluator_sha256","joint_engine_sha256","bb_core_sha256","topology_sha256","short_replay_source","short_replay_source_sha256","selected_path","selected_path_sha256","discrete_I_perp","discrete_I_perp_source","dense_coordinates_A","dense_energy_eV","dense_force_eV_per_A","dense_coordinate_sha256","dense_energy_sha256","dense_force_sha256","densified_Delta_E_kcal_per_mol","number_of_points","dense_joint_EF_attempts","dense_endpoint_joint_EF_attempts","original_discrete_new_calls","optimization_N_F_contribution"}
        require(required.issubset(saved.files), "incomplete shard")
        require(str(scalar(saved,"schema")) == SHARD_SCHEMA and str(scalar(saved,"status")) == "PASS", "shard identity mismatch")
        require(str(scalar(saved,"method")) == "IVM", "shard method mismatch")
        require(str(scalar(saved,"contract_sha256")) == contract_sha and str(scalar(saved,"topology_sha256")) == topology_sha, "shard provenance mismatch")
        require(str(scalar(saved,"evaluator_sha256")) == ENGINE.sha256_file(Path(__file__)) and str(scalar(saved,"joint_engine_sha256")) == ENGINE.sha256_file(ENGINE_PATH) and str(scalar(saved,"bb_core_sha256")) == ENGINE.sha256_file(ROOT / "src/bb_core.py"), "shard code provenance mismatch")
        require(str(scalar(saved,"label")) == row["label"] and int(scalar(saved,"optimization_N_F")) == row["optimization_N_F"], "shard row mismatch")
        require(json.loads(str(scalar(saved,"path_identifier_json"))) == row["path_identifier"], "shard identifier mismatch")
        require(str(scalar(saved,"short_replay_source")) == str(row["source_path"]) and str(scalar(saved,"short_replay_source_sha256")) == row["short_replay_source"]["sha256"], "shard source mismatch")
        selected = np.asarray(saved["selected_path"],dtype=np.float64); dense=np.asarray(saved["dense_coordinates_A"],dtype=np.float64); energy=np.asarray(saved["dense_energy_eV"],dtype=np.float64); force=np.asarray(saved["dense_force_eV_per_A"],dtype=np.float64)
        require(selected.shape==(N_IMAGES,DIMENSION) and dense.shape==force.shape==(N_DENSE,DIMENSION) and energy.shape==(N_DENSE,), "shard shape mismatch")
        require(np.all(np.isfinite(selected)) and np.all(np.isfinite(dense)) and np.all(np.isfinite(energy)) and np.all(np.isfinite(force)), "shard contains non-finite values")
        require(np.array_equal(selected,row["path"]) and ENGINE.sha256_array(selected)==str(scalar(saved,"selected_path_sha256"))==row["path_sha256"], "selected path mismatch")
        expected_dense,_=ENGINE.uniform_arc500(selected); require(np.array_equal(dense,expected_dense) and ENGINE.sha256_array(dense)==str(scalar(saved,"dense_coordinate_sha256")), "dense coordinate mismatch")
        require(ENGINE.sha256_array(energy)==str(scalar(saved,"dense_energy_sha256")) and ENGINE.sha256_array(force)==str(scalar(saved,"dense_force_sha256")), "dense E/F hash mismatch")
        require(str(scalar(saved,"discrete_I_perp_source"))=="sealed_verified_short_replay_online_history" and same_float(scalar(saved,"discrete_I_perp"),row["i_perp"]), "Iperp provenance mismatch")
        delta=float((np.max(energy)-energy[0])*ENGINE.EV_TO_KCAL); require(same_float(delta,scalar(saved,"densified_Delta_E_kcal_per_mol")), "DeltaE mismatch")
        require(int(scalar(saved,"number_of_points")) == N_DENSE and int(scalar(saved,"dense_joint_EF_attempts")) == N_DENSE and int(scalar(saved,"dense_endpoint_joint_EF_attempts")) == 2, "shard dense call accounting mismatch")
        require(int(scalar(saved,"original_discrete_new_calls")) == 0, "shard discrete call accounting mismatch")
        require(int(scalar(saved,"optimization_N_F_contribution"))==0, "diagnostic entered optimization cost")
        return {key:np.asarray(saved[key]).copy() for key in saved.files}


def _strict_wal_lifetime(events: list[dict], role: str) -> dict:
    starts = [event for event in events if event.get("event") == "attempt_started"]
    terminals = [event for event in events if event.get("event") in {"attempt_succeeded", "attempt_failed"}]
    for event in starts + terminals:
        require(set(event) >= {"attempt_id","row_index","stage","point_index"}, f"{role} WAL attempt metadata missing")
    start_ids = [str(event.get("attempt_id")) for event in starts]
    terminal_ids = [str(event.get("attempt_id")) for event in terminals]
    require(None not in [event.get("attempt_id") for event in starts + terminals], f"{role} WAL attempt lacks id")
    require(len(start_ids) == len(set(start_ids)), f"{role} WAL has duplicate attempt start")
    require(len(terminal_ids) == len(set(terminal_ids)), f"{role} WAL has duplicate attempt terminal")
    require(set(terminal_ids).issubset(set(start_ids)), f"{role} WAL has terminal without start")
    successes = [event for event in terminals if event["event"] == "attempt_succeeded"]
    failures = [event for event in terminals if event["event"] == "attempt_failed"]
    require(not ({str(event["attempt_id"]) for event in successes} & {str(event["attempt_id"]) for event in failures}), f"{role} WAL attempt has conflicting terminals")
    logical = {(int(event["row_index"]), str(event["stage"]), int(event["point_index"])) for event in starts}
    return {"attempt_started":len(starts),"attempt_succeeded":len(successes),"attempt_failed":len(failures),"attempt_open":len(set(start_ids)-set(terminal_ids)),"retry_attempts":len(starts)-len(logical)}


def _publication_event(event: dict, row_index: int, row: dict, target: Path, shard: dict, role: str) -> bool:
    return (
        event.get("event") == "complete_shard_published"
        and event.get("wal_role") == role
        and int(event.get("row_index", -1)) == row_index
        and int(event.get("calls_per_image", -1)) == int(row["path_identifier"]["calls_per_image"])
        and event.get("label") == row["label"]
        and event.get("path_sha256") == row["path_sha256"]
        and event.get("shard") == str(target)
        and event.get("shard_sha256") == ENGINE.sha256_file(target)
        and event.get("dense_coordinate_sha256") == str(scalar(shard,"dense_coordinate_sha256"))
        and event.get("dense_energy_sha256") == str(scalar(shard,"dense_energy_sha256"))
        and event.get("dense_force_sha256") == str(scalar(shard,"dense_force_sha256"))
    )


def _validate_source_wal_closure(events: list[dict], row_index: int, row: dict, target: Path, shard: dict) -> dict:
    reuses = [(i,event) for i,event in enumerate(events) if event.get("event") == "verified_short_replay_path_and_Iperp_reused" and int(event.get("row_index",-1)) == row_index]
    publishes = [(i,event) for i,event in enumerate(events) if _publication_event(event,row_index,row,target,shard,"source")]
    require(reuses and publishes, f"{row['label']} source WAL lacks reuse/publish closure")
    publish_index, _ = publishes[-1]
    eligible = [event for index,event in reuses if index < publish_index]
    require(eligible, f"{row['label']} source publish lacks prior reuse")
    event = eligible[-1]
    require(event.get("label") == row["label"] and event.get("path_sha256") == row["path_sha256"], f"{row['label']} source WAL path mismatch")
    require(event.get("source") == str(row["source_path"]) and event.get("source_sha256") == row["short_replay_source"]["sha256"], f"{row['label']} source WAL lineage mismatch")
    require(same_float(event.get("discrete_I_perp"),row["i_perp"]) and int(event.get("new_physical_calls",-1)) == 0 and int(event.get("optimization_N_F_contribution",-1)) == 0, f"{row['label']} source WAL accounting mismatch")
    return {"status":"PASS","reuse_events_lifetime":len(reuses),"publish_events_lifetime":len(publishes),"accepted_reuse_new_physical_calls":0}


def _validate_dense_wal_closure(events: list[dict], row_index: int, row: dict, target: Path, shard: dict) -> dict:
    relevant_markers = [event for event in events if int(event.get("row_index",-1)) == row_index and event.get("event") in {"path_stage_started","path_stage_data_complete_uncommitted"}]
    require(relevant_markers and all(event.get("stage") == "dense_arc500_joint_EF" for event in relevant_markers), f"{row['label']} dense WAL marker stage missing/mismatched")
    relevant_attempts = [event for event in events if int(event.get("row_index",-1)) == row_index and event.get("event") in {"attempt_started","attempt_succeeded","attempt_failed"}]
    require(relevant_attempts and all(event.get("stage") == "dense_arc500_joint_EF" for event in relevant_attempts), f"{row['label']} dense WAL attempt stage missing/mismatched")
    report = ENGINE._validate_stage_wal_closure(events,row_index,row,"dense_arc500_joint_EF",np.asarray(shard["dense_coordinates_A"]),np.asarray(shard["dense_energy_eV"]),np.asarray(shard["dense_force_eV_per_A"]))
    publishes = [(i,event) for i,event in enumerate(events) if _publication_event(event,row_index,row,target,shard,"dense")]
    require(publishes, f"{row['label']} dense WAL lacks shard publish")
    last_complete = max(i for i,event in enumerate(events) if event.get("event") == "path_stage_data_complete_uncommitted" and int(event.get("row_index",-1)) == row_index and event.get("stage") == "dense_arc500_joint_EF")
    require(publishes[-1][0] > last_complete, f"{row['label']} dense shard publish precedes data completion")
    return report


def _validate_aggregate(output: Path, rows: list[dict], shards: list[dict], contract_sha: str) -> None:
    with np.load(output,allow_pickle=False) as saved:
        required={"schema","status","method","authoritative_data_unit","labels","optimization_N_F","selected_path","selected_path_sha256","discrete_I_perp","energy_eV","energy_sha256","dense_coordinate_sha256","dense_force_sha256","densified_Delta_E_kcal_per_mol","shard_path","shard_sha256","contract_sha256","optimization_N_F_contribution"}
        require(required.issubset(saved.files),"aggregate fields missing")
        require(str(scalar(saved,"schema"))==RESULT_SCHEMA and str(scalar(saved,"status"))=="PASS" and str(scalar(saved,"method"))=="IVM", "aggregate identity mismatch")
        require(str(scalar(saved,"authoritative_data_unit"))=="complete_atomic_path_shard" and str(scalar(saved,"contract_sha256"))==contract_sha and int(scalar(saved,"optimization_N_F_contribution"))==0,"aggregate authority mismatch")
        require("dense_coordinates_A" not in saved.files and "dense_force_eV_per_A" not in saved.files,"aggregate duplicates authoritative dense arrays")
        expected_arrays={
            "labels":np.asarray([r["label"] for r in rows]),
            "optimization_N_F":np.asarray([r["optimization_N_F"] for r in rows],dtype=np.int64),
            "selected_path":np.asarray([r["path"] for r in rows]),
            "selected_path_sha256":np.asarray([r["path_sha256"] for r in rows]),
            "discrete_I_perp":np.asarray([r["i_perp"] for r in rows],dtype=np.float64),
            "energy_eV":np.asarray([s["dense_energy_eV"] for s in shards]),
            "energy_sha256":np.asarray([str(scalar(s,"dense_energy_sha256")) for s in shards]),
            "dense_coordinate_sha256":np.asarray([str(scalar(s,"dense_coordinate_sha256")) for s in shards]),
            "dense_force_sha256":np.asarray([str(scalar(s,"dense_force_sha256")) for s in shards]),
            "densified_Delta_E_kcal_per_mol":np.asarray([float(scalar(s,"densified_Delta_E_kcal_per_mol")) for s in shards],dtype=np.float64),
            "shard_path":np.asarray([str(shard_path(output,int(r["path_identifier"]["calls_per_image"]))) for r in rows]),
            "shard_sha256":np.asarray([ENGINE.sha256_file(shard_path(output,int(r["path_identifier"]["calls_per_image"]))) for r in rows]),
        }
        for key,expected in expected_arrays.items():
            actual=np.asarray(saved[key])
            require(actual.dtype.str==expected.dtype.str and actual.shape==expected.shape and actual.tobytes(order="C")==expected.tobytes(order="C"),f"aggregate {key} mismatch")


def _wal_paths(output: Path) -> tuple[Path,Path]:
    return output.with_suffix(".source.wal.jsonl"), output.with_suffix(".dense_joint_ef.wal.jsonl")


def _final_paths(output: Path) -> tuple[Path,Path,Path]:
    return output, output.with_suffix(".manifest.json"), output.with_suffix(".audit.json")


def _read_validated_wal(path: Path, schema: str, contract_path: Path, contract_sha: str) -> list[dict]:
    events=ENGINE._read_wal_events(path)
    require(events and events[0].get("event")=="wal_header" and events[0].get("schema")==schema,"invalid WAL header")
    require(events[0].get("contract_sha256")==contract_sha and Path(events[0].get("contract","")).resolve()==contract_path,"WAL contract binding mismatch")
    return events


def preflight(contract_path: Path, topology: Path, locks: list[Path], rows: list[dict], output: Path) -> dict:
    contract_sha=ENGINE.sha256_file(contract_path); topology_sha=ENGINE.sha256_file(topology)
    require({Path(path).expanduser().resolve() for path in locks}==CANONICAL_LOCKS,"canonical two-lock set required at execution")
    for final in _final_paths(output):
        if final.exists(): raise FileExistsError(f"refusing to overwrite {final}")
    output.parent.mkdir(parents=True,exist_ok=True)
    source_path,dense_path=_wal_paths(output); existing=[]
    require(source_path.exists()==dense_path.exists(),"source and dense WAL must both exist or both be absent")
    source_events=dense_events=None
    if source_path.exists():
        source_events=_read_validated_wal(source_path,SOURCE_WAL_SCHEMA,contract_path,contract_sha)
        dense_events=_read_validated_wal(dense_path,DENSE_WAL_SCHEMA,contract_path,contract_sha)
        _strict_wal_lifetime(dense_events,"dense")
        require(all(event.get("stage")=="dense_arc500_joint_EF" for event in dense_events if event.get("event") in {"attempt_started","attempt_succeeded","attempt_failed","path_stage_started","path_stage_data_complete_uncommitted"}),"dense WAL stage missing/mismatched")
    for row_index,row in enumerate(rows):
        call=int(row["path_identifier"]["calls_per_image"]); target=shard_path(output,call)
        if target.exists():
            require(source_path.exists() and dense_path.exists(),f"call {call} shard lacks WAL files")
            shard=validate_shard(target,row,contract_sha,topology_sha)
            assert source_events is not None and dense_events is not None
            _validate_source_wal_closure(source_events,row_index,row,target,shard)
            _validate_dense_wal_closure(dense_events,row_index,row,target,shard)
            existing.append(call)
    missing=4-len(existing); required_free=MINIMUM_FREE_BYTES_RESERVE+missing*ESTIMATED_SHARD_BYTES; free=shutil.disk_usage(output.parent).free
    require(free>=required_free,f"insufficient free disk: {free} < {required_free}")
    with ENGINE.acquire_two_locks(locks):
        pass
    return {"existing_complete_shards":existing,"fresh_paths_planned":missing,"fresh_dense_joint_EF_attempts_planned":missing*N_DENSE,"observed_free_bytes":free,"required_free_bytes":required_free,"canonical_two_locks_available":True}


def _attempt(wal,backend,row_index,row,stage,index,coordinate):
    coordinate=np.ascontiguousarray(coordinate,dtype=np.float64); coordinate_sha=ENGINE.sha256_array(coordinate)
    attempt_id=f"{wal.session_id}:{row_index}:{stage}:{index}:{int(backend.raw_joint_attempts)+1}"
    started={"event":"attempt_started","utc":ENGINE.utc_now(),"stage":stage,"row_index":row_index,"label":row["label"],"optimization_N_F":int(row["optimization_N_F"]),"path_identifier":row["path_identifier"],"path_sha256":row["path_sha256"],"point_index":index,"coordinate_sha256":coordinate_sha,"backend_attempt_ordinal":int(backend.raw_joint_attempts)+1,"session_id":wal.session_id,"attempt_id":attempt_id}
    wal.append(started)
    try:
        energy,force=backend.evaluate_joint(coordinate); energy=float(energy); force=np.ascontiguousarray(force,dtype=np.float64)
        require(force.shape==(DIMENSION,) and np.all(np.isfinite(force)) and np.isfinite(energy),"bad joint E/F result")
    except Exception as error:
        wal.append({**started,"event":"attempt_failed","utc":ENGINE.utc_now(),"error":repr(error)})
        raise
    wal.append({**started,"event":"attempt_succeeded","utc":ENGINE.utc_now(),"energy_eV":energy,"energy_scalar_sha256":ENGINE.sha256_array(np.asarray(energy,dtype=np.float64)),"force_sha256":ENGINE.sha256_array(force)})
    return energy,force


def run(contract_path: Path, topology: Path, locks, rows, output: Path) -> None:
    contract_sha=ENGINE.sha256_file(contract_path); topology_sha=ENGINE.sha256_file(topology)
    source_wal_path,dense_wal_path=_wal_paths(output); _,manifest_path,audit_path=_final_paths(output)
    pre=preflight(contract_path,topology,locks,rows,output)
    shards=[]; fresh=[]; reused=[]; backend=None
    with ENGINE.acquire_two_locks(locks) as held:
        for final in _final_paths(output):
            if final.exists(): raise FileExistsError(f"refusing to overwrite {final}")
        missing=sum(not shard_path(output,c).exists() for c in CALLS); required_free=MINIMUM_FREE_BYTES_RESERVE+missing*ESTIMATED_SHARD_BYTES; free=shutil.disk_usage(output.parent).free; require(free>=required_free,f"insufficient free disk: {free} < {required_free}")
        source_wal=ENGINE.DurableWal(source_wal_path,SOURCE_WAL_SCHEMA,contract_path,contract_sha); dense_wal=ENGINE.DurableWal(dense_wal_path,DENSE_WAL_SCHEMA,contract_path,contract_sha)
        try:
            for row_index,row in enumerate(rows):
                call=int(row["path_identifier"]["calls_per_image"]); target=shard_path(output,call)
                if target.exists():
                    shard=validate_shard(target,row,contract_sha,topology_sha)
                    source_events=_read_validated_wal(source_wal_path,SOURCE_WAL_SCHEMA,contract_path,contract_sha); dense_events=_read_validated_wal(dense_wal_path,DENSE_WAL_SCHEMA,contract_path,contract_sha)
                    _validate_source_wal_closure(source_events,row_index,row,target,shard); _validate_dense_wal_closure(dense_events,row_index,row,target,shard); reused.append(call)
                else:
                    source_wal.append({"event":"verified_short_replay_path_and_Iperp_reused","utc":ENGINE.utc_now(),"row_index":row_index,"calls_per_image":call,"label":row["label"],"path_sha256":row["path_sha256"],"discrete_I_perp":row["i_perp"],"source":str(row["source_path"]),"source_sha256":row["short_replay_source"]["sha256"],"new_physical_calls":0,"optimization_N_F_contribution":0})
                    dense,_=ENGINE.uniform_arc500(row["path"]); energy=np.empty(N_DENSE); force=np.empty((N_DENSE,DIMENSION)); dense_wal.append({"event":"path_stage_started","utc":ENGINE.utc_now(),"stage":"dense_arc500_joint_EF","row_index":row_index,"label":row["label"],"optimization_N_F":row["optimization_N_F"],"path_sha256":row["path_sha256"],"dense_coordinate_sha256":ENGINE.sha256_array(dense),"number_of_points":N_DENSE})
                    if backend is None: backend=ENGINE.OpenMMBackend(topology,platform_preference=("CPU",),precision="double",label="bb_ivm_short_dense_joint_ef")
                    before=int(backend.raw_joint_attempts)
                    for index,coordinate in enumerate(dense): energy[index],force[index]=_attempt(dense_wal,backend,row_index,row,"dense_arc500_joint_EF",index,coordinate)
                    require(int(backend.raw_joint_attempts)-before==N_DENSE,"backend attempt mismatch")
                    require(np.all(np.isfinite(energy)) and np.all(np.isfinite(force)),"non-finite dense E/F")
                    delta=float((np.max(energy)-energy[0])*ENGINE.EV_TO_KCAL); dense_wal.append({"event":"path_stage_data_complete_uncommitted","utc":ENGINE.utc_now(),"stage":"dense_arc500_joint_EF","row_index":row_index,"label":row["label"],"optimization_N_F":row["optimization_N_F"],"path_sha256":row["path_sha256"],"joint_attempts":N_DENSE,"joint_successes":N_DENSE,"dense_coordinate_sha256":ENGINE.sha256_array(dense),"dense_energy_sha256":ENGINE.sha256_array(energy),"dense_force_sha256":ENGINE.sha256_array(force),"densified_Delta_E_kcal_per_mol":delta})
                    arrays={"schema":np.asarray(SHARD_SCHEMA),"status":np.asarray("PASS"),"method":np.asarray("IVM"),"label":np.asarray(row["label"]),"optimization_N_F":np.asarray(row["optimization_N_F"],dtype=np.int64),"path_identifier_json":np.asarray(json.dumps(row["path_identifier"],sort_keys=True)),"contract_sha256":np.asarray(contract_sha),"evaluator_sha256":np.asarray(ENGINE.sha256_file(Path(__file__))),"joint_engine_sha256":np.asarray(ENGINE.sha256_file(ENGINE_PATH)),"bb_core_sha256":np.asarray(ENGINE.sha256_file(ROOT/"src/bb_core.py")),"topology_sha256":np.asarray(topology_sha),"short_replay_source":np.asarray(str(row["source_path"])),"short_replay_source_sha256":np.asarray(row["short_replay_source"]["sha256"]),"selected_path":row["path"],"selected_path_sha256":np.asarray(row["path_sha256"]),"discrete_I_perp":np.asarray(row["i_perp"]),"discrete_I_perp_source":np.asarray("sealed_verified_short_replay_online_history"),"dense_coordinates_A":dense,"dense_energy_eV":energy,"dense_force_eV_per_A":force,"dense_coordinate_sha256":np.asarray(ENGINE.sha256_array(dense)),"dense_energy_sha256":np.asarray(ENGINE.sha256_array(energy)),"dense_force_sha256":np.asarray(ENGINE.sha256_array(force)),"densified_Delta_E_kcal_per_mol":np.asarray(delta),"number_of_points":np.asarray(N_DENSE,dtype=np.int64),"dense_joint_EF_attempts":np.asarray(N_DENSE,dtype=np.int64),"dense_endpoint_joint_EF_attempts":np.asarray(2,dtype=np.int64),"original_discrete_new_calls":np.asarray(0,dtype=np.int64),"optimization_N_F_contribution":np.asarray(0,dtype=np.int64)}
                    ENGINE._atomic_save_npz_no_overwrite(target,arrays); shard=validate_shard(target,row,contract_sha,topology_sha)
                    publication={"event":"complete_shard_published","utc":ENGINE.utc_now(),"row_index":row_index,"calls_per_image":call,"label":row["label"],"path_sha256":row["path_sha256"],"shard":str(target),"shard_sha256":ENGINE.sha256_file(target),"dense_coordinate_sha256":str(scalar(shard,"dense_coordinate_sha256")),"dense_energy_sha256":str(scalar(shard,"dense_energy_sha256")),"dense_force_sha256":str(scalar(shard,"dense_force_sha256")),"optimization_N_F_contribution":0}
                    dense_wal.append({**publication,"wal_role":"dense"}); source_wal.append({**publication,"wal_role":"source"}); fresh.append(call)
                    source_events=_read_validated_wal(source_wal_path,SOURCE_WAL_SCHEMA,contract_path,contract_sha); dense_events=_read_validated_wal(dense_wal_path,DENSE_WAL_SCHEMA,contract_path,contract_sha)
                    _validate_source_wal_closure(source_events,row_index,row,target,shard); _validate_dense_wal_closure(dense_events,row_index,row,target,shard)
                shards.append(shard)
            source_events=_read_validated_wal(source_wal_path,SOURCE_WAL_SCHEMA,contract_path,contract_sha); dense_events=_read_validated_wal(dense_wal_path,DENSE_WAL_SCHEMA,contract_path,contract_sha)
            dense_lifetime=_strict_wal_lifetime(dense_events,"dense")
            source_closures=[]; dense_closures=[]
            for i,(row,shard) in enumerate(zip(rows,shards)):
                target=shard_path(output,int(row["path_identifier"]["calls_per_image"])); source_closures.append({"label":row["label"],**_validate_source_wal_closure(source_events,i,row,target,shard)}); dense_closures.append({"label":row["label"],**_validate_dense_wal_closure(dense_events,i,row,target,shard)})
            accepted_successes=sum(int(item["accepted_stage_successes"]) for item in dense_closures); require(accepted_successes==4*N_DENSE,"accepted dense lineage must contain exactly 2000 successes")
            result={"schema":np.asarray(RESULT_SCHEMA),"status":np.asarray("PASS"),"method":np.asarray("IVM"),"authoritative_data_unit":np.asarray("complete_atomic_path_shard"),"labels":np.asarray([r["label"] for r in rows]),"optimization_N_F":np.asarray(EXPECTED_NF,dtype=np.int64),"selected_path":np.asarray([r["path"] for r in rows]),"selected_path_sha256":np.asarray([r["path_sha256"] for r in rows]),"discrete_I_perp":np.asarray([r["i_perp"] for r in rows],dtype=np.float64),"energy_eV":np.asarray([s["dense_energy_eV"] for s in shards]),"energy_sha256":np.asarray([str(scalar(s,"dense_energy_sha256")) for s in shards]),"dense_coordinate_sha256":np.asarray([str(scalar(s,"dense_coordinate_sha256")) for s in shards]),"dense_force_sha256":np.asarray([str(scalar(s,"dense_force_sha256")) for s in shards]),"densified_Delta_E_kcal_per_mol":np.asarray([float(scalar(s,"densified_Delta_E_kcal_per_mol")) for s in shards],dtype=np.float64),"shard_path":np.asarray([str(shard_path(output,c)) for c in CALLS]),"shard_sha256":np.asarray([ENGINE.sha256_file(shard_path(output,c)) for c in CALLS]),"contract_sha256":np.asarray(contract_sha),"optimization_N_F_contribution":np.asarray(0,dtype=np.int64)}
            ENGINE._atomic_save_npz_no_overwrite(output,result); _validate_aggregate(output,rows,shards,contract_sha)
            manifest={"schema":"bb-dense-four-ivm-short-prefix-joint-ef-manifest-v1","status":"PASS","contract":str(contract_path),"contract_sha256":contract_sha,"aggregate":str(output),"aggregate_sha256":ENGINE.sha256_file(output),"source_wal":{"path":str(source_wal_path),"sha256":ENGINE.sha256_file(source_wal_path)},"dense_joint_ef_wal":{"path":str(dense_wal_path),"sha256":ENGINE.sha256_file(dense_wal_path)},"shards":[{"call_per_image":c,"path":str(shard_path(output,c)),"sha256":ENGINE.sha256_file(shard_path(output,c)),"selected_path_sha256":r["path_sha256"],"dense_coordinate_sha256":str(scalar(s,"dense_coordinate_sha256")),"dense_energy_sha256":str(scalar(s,"dense_energy_sha256")),"dense_force_sha256":str(scalar(s,"dense_force_sha256"))} for c,r,s in zip(CALLS,rows,shards)]}; ENGINE._atomic_write_json_no_overwrite(manifest_path,manifest)
            audit={"schema":AUDIT_SCHEMA,"status":"PASS","method":"IVM","contract":str(contract_path),"contract_sha256":contract_sha,"output":str(output),"output_sha256":ENGINE.sha256_file(output),"manifest":str(manifest_path),"manifest_sha256":ENGINE.sha256_file(manifest_path),"source_wal":str(source_wal_path),"source_wal_sha256":ENGINE.sha256_file(source_wal_path),"dense_joint_ef_wal":str(dense_wal_path),"dense_joint_ef_wal_sha256":ENGINE.sha256_file(dense_wal_path),"fresh_paths_this_invocation":fresh,"complete_validated_shards_reused":reused,"original_discrete_new_calls":0,"dense_joint_EF_attempts_fresh_this_invocation":len(fresh)*N_DENSE,"dense_joint_EF_attempts_lineage":accepted_successes,"dense_joint_EF_accepted_successes_lineage":accepted_successes,"dense_joint_EF_attempts_lifetime":dense_lifetime,"dense_full_force_vectors_stored":4*N_DENSE,"optimization_N_F_contribution":0,"source_WAL_shard_closure":{"status":"PASS","paths":source_closures},"dense_WAL_shard_pointwise_closure":{"status":"PASS","lifetime":dense_lifetime,"paths":dense_closures},"free_disk_gate":{"status":"PASS","observed_free_bytes":free,"required_free_bytes":required_free},"canonical_two_locks_held":held,"initial_preflight":pre}; ENGINE._atomic_write_json_no_overwrite(audit_path,audit)
        finally:
            source_wal.close(); dense_wal.close()


def main(argv: Iterable[str] | None=None) -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--contract",type=Path,required=True); parser.add_argument("--host-role",choices=("local","remote"),required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--preflight-only",action="store_true"); args=parser.parse_args(list(argv) if argv is not None else None)
    ENGINE.check_thread_caps(); contract_path=args.contract.expanduser().resolve(strict=True); output=args.output.expanduser().resolve(); require(output.suffix==".npz","output must be npz"); contract,topology,locks,rows=load_contract(contract_path,args.host_role)
    if args.preflight_only:
        physical=preflight(contract_path,topology,locks,rows,output)
        summary={"status":"ZERO_OPENMM_PREFLIGHT_PASS","contract":str(contract_path),"method":"IVM","host_role":args.host_role,"calls_per_image":list(CALLS),"optimization_N_F":list(EXPECTED_NF),"path_sha256":[r["path_sha256"] for r in rows],"original_discrete_new_calls_planned":0,"dense_joint_EF_attempts_planned":physical["fresh_dense_joint_EF_attempts_planned"],"dense_full_force_vectors_planned_for_storage":physical["fresh_dense_joint_EF_attempts_planned"],"optimization_N_F_contribution":0,"physical_preflight":physical}
        print(json.dumps(summary,indent=2,sort_keys=True)); return
    run(contract_path,topology,locks,rows,output)


if __name__ == "__main__": main()
