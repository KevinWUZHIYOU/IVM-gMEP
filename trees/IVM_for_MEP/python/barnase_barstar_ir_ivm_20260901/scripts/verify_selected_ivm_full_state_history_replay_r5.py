#!/usr/bin/env python3
"""Zero-OpenMM exact-prefix and stable-last full-state verifier for R5B."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import numpy as np

import run_selected_ivm_full_state_history_replay_r5 as RUNNER
import full_state_history_recorder_r5 as RECORDER
import verify_selected_ivm_full_state_history_replay as BASE


SCHEMA = "bb-irivm-selected-full-state-history-verification-v5b"
STATUS = "PASS_R5B_EXACT_PREFIX318_AND_DYNAMIC_STABLE_LAST_FULL_STATE_REPLAY"
PREFIX_CALLS = RECORDER.OLD_COMPLETE_PREFIX_CALLS
PREFIX_NF = 8982
REQUIRED_PREFIX_KEYS = {
    "calls_per_image",
    "counter_vectors",
    "energy_request_vectors",
    "force_request_vectors",
    "cache_hit_vectors",
    "i_perp",
    "barrier_kcal",
    "reparam_committed",
    "total_internal_joint_evaluations",
}
VECTOR_FIELDS = {
    "joint_evaluations": "joint_evaluation_vector",
    "energy_requests": "energy_request_vector",
    "force_requests": "force_request_vector",
    "cache_hits": "cache_hit_vector",
    "total_query_events": "total_query_event_vector",
    "both_requests": "both_request_vector",
    "budget_blocked_misses": "budget_blocked_miss_vector",
    "inherited_cache_entries": "inherited_cache_entry_vector",
}


def scalar(saved, key: str):
    return BASE.scalar(saved, key)


def strings(saved, key: str) -> list[str]:
    value = np.asarray(saved[key])
    BASE.require(value.ndim == 1 and value.dtype.kind in "US", f"invalid string vector {key}")
    return value.astype(str).tolist()


def exact_prefix(reference, replay, key: str) -> None:
    left, right = np.asarray(reference[key]), np.asarray(replay[key])
    BASE.require(left.ndim >= 1 and right.ndim >= 1, f"prefix key is scalar: {key}")
    BASE.require(left.shape[0] >= PREFIX_CALLS and right.shape[0] >= PREFIX_CALLS, f"short prefix key: {key}")
    BASE.require(BASE.exact(left[:PREFIX_CALLS], right[:PREFIX_CALLS]), f"old-9990 exact prefix mismatch: {key}")


def verify(
    root: Path,
    registration_path: Path,
    authorization_path: Path,
    reference_path: Path,
    replay_path: Path,
    replay_audit_path: Path,
    manifest_path: Path,
    canonical_index_path: Path,
    capture_audit_path: Path,
) -> dict:
    registration = RUNNER.validate_registration(root, registration_path)
    RUNNER.validate_authorization(registration_path, authorization_path, registration)
    replay_audit = BASE.read_json(replay_audit_path)
    BASE.require(replay_audit.get("consistent") is True and Path(str(replay_audit.get("result_file"))).resolve() == replay_path, "runtime result/audit invalid")
    manifest = BASE.read_json(manifest_path)
    canonical_index = BASE.read_json(canonical_index_path)
    capture_audit = BASE.read_json(capture_audit_path)
    BASE.require(manifest.get("schema") == RECORDER.MANIFEST_SCHEMA and manifest.get("status") == "COMPLETE_PENDING_EXACT_PREFIX_VERIFICATION", "R5 manifest invalid")
    BASE.require(canonical_index.get("schema") == RECORDER.CANONICAL_INDEX_SCHEMA and canonical_index.get("status") == "COMPLETE_PENDING_EXACT_RESULT_VERIFICATION", "R5 canonical index invalid")
    BASE.require(capture_audit.get("schema") == RECORDER.AUDIT_SCHEMA and capture_audit.get("status") == RECORDER.AUDIT_STATUS, "R5 capture audit invalid")
    BASE.require(manifest.get("canonical_index") == str(canonical_index_path) and manifest.get("canonical_index_sha256") == BASE.sha256_file(canonical_index_path), "manifest/index binding mismatch")
    BASE.require(capture_audit.get("manifest") == str(manifest_path) and capture_audit.get("manifest_sha256") == BASE.sha256_file(manifest_path), "audit/manifest binding mismatch")
    BASE.require(capture_audit.get("additional_physical_calls") == 0 and capture_audit.get("optimization_N_F_contribution_from_capture") == 0, "capture added physical calls")
    with np.load(reference_path, allow_pickle=False) as reference, np.load(replay_path, allow_pickle=False) as replay:
        for key in REQUIRED_PREFIX_KEYS:
            BASE.require(key in reference.files and key in replay.files, f"missing prefix key: {key}")
            exact_prefix(reference, replay, key)
        calls = np.asarray(replay["calls_per_image"], dtype=np.int64)
        count = int(calls.size)
        BASE.require(0 < count <= RECORDER.MAXIMUM_CANONICAL_CALLS and np.array_equal(calls, np.arange(1, count + 1)), "R5 formal call axis is not continuous")
        nf = np.asarray(replay["total_internal_joint_evaluations"], dtype=np.int64)
        BASE.require(nf.shape == (count,) and int(nf[PREFIX_CALLS - 1]) == PREFIX_NF, "R5 prefix N_F closure failed")
        BASE.require(int(scalar(replay, "total_internal_nf_consumed")) <= RECORDER.OPTIMIZATION_NF_BUDGET, "R5 exceeded 15000 optimization N_F")
        last_complete = int(scalar(replay, "total_internal_nf_last_complete_state_index")) + 1
        BASE.require(PREFIX_CALLS <= last_complete <= count, "R5 last-complete call invalid")
        stored_calls = np.asarray(replay["stored_path_calls"], dtype=np.int64)
        stored_paths = np.asarray(replay["stored_paths"], dtype=np.float64)
        BASE.require(np.array_equal(stored_calls, calls) and stored_paths.shape == (count, 32, 9396), "R5 native all-path result invalid")
        formal = {
            "count": count,
            "nf": nf,
            "i_perp": np.asarray(replay["i_perp"], dtype=np.float64),
            "barrier": np.asarray(replay["barrier_kcal"], dtype=np.float64),
            "counters": np.asarray(replay["counter_vectors"], dtype=np.int64),
            "energy_requests": np.asarray(replay["energy_request_vectors"], dtype=np.int64),
            "force_requests": np.asarray(replay["force_request_vectors"], dtype=np.int64),
            "cache_hits": np.asarray(replay["cache_hit_vectors"], dtype=np.int64),
            "stored_paths": stored_paths,
            "last_complete": last_complete,
            "terminal_nf": int(scalar(replay, "total_internal_nf_consumed")),
        }
        reference_calls = np.asarray(reference["stored_path_calls"], dtype=np.int64)
        reference_paths = np.asarray(reference["stored_paths"], dtype=np.float64)
        for index, call in enumerate(reference_calls):
            if int(call) <= PREFIX_CALLS:
                BASE.require(BASE.exact(reference_paths[index], stored_paths[int(call) - 1]), f"sealed committed prefix path mismatch at call {call}")
    raw = manifest.get("raw_callbacks")
    canonical = manifest.get("canonical_records")
    BASE.require(isinstance(raw, list) and isinstance(canonical, list), "manifest inventories missing")
    BASE.require(len(raw) <= RECORDER.MAXIMUM_RAW_CALLBACKS and len(canonical) == formal["count"], "R5 manifest counts invalid")
    BASE.require(manifest.get("raw_record_count") == len(raw) and manifest.get("canonical_record_count") == len(canonical), "R5 manifest declared counts drift")
    BASE.require(canonical_index.get("records") == canonical and canonical_index.get("raw_record_count") == len(raw) and canonical_index.get("canonical_record_count") == len(canonical), "R5 canonical index inventory drift")
    raw_calls = np.asarray([row["call_per_image"] for row in raw], dtype=np.int64)
    stable = RECORDER.stable_last_raw_ordinals(raw_calls)
    BASE.require(canonical_index.get("source_raw_record_ordinals") == stable.tolist(), "R5 stable-last ordinal index mismatch")
    BASE.require([row["source_raw_record_ordinal"] for row in canonical] == stable.tolist(), "R5 canonical records are not last-per-call")
    wal_path = Path(capture_audit["wal"]).resolve(strict=True)
    BASE.require(BASE.sha256_file(wal_path) == capture_audit["wal_sha256"], "R5 WAL hash drift")
    wal = BASE.read_wal(wal_path)
    raw_events = [row for row in wal if row.get("event") == "raw_callback_pending_state_committed"]
    promotions = [row for row in wal if row.get("event") == "canonical_shard_promoted"]
    BASE.require(raw_events == raw and promotions == canonical, "R5 WAL/manifest inventory mismatch")
    BASE.require(wal[0].get("schema") == RECORDER.SCHEMA and wal[0].get("event") == "header" and wal[0].get("registration_sha256") == BASE.sha256_file(registration_path), "R5 WAL header invalid")
    BASE.require(wal[-1].get("event") == "manifest_committed" and wal[-1].get("manifest_sha256") == BASE.sha256_file(manifest_path), "R5 WAL terminal commit invalid")
    schema_counts = {"live_schema8": 0, "historical_schema7": 0, "historical_schema8": 0}
    inventory = []
    for index, record in enumerate(canonical):
        call = index + 1
        BASE.require(record["history_index"] == index and record["call_per_image"] == call, f"canonical ordering mismatch at call {call}")
        source = raw[record["source_raw_record_ordinal"]]
        BASE.require(source["call_per_image"] == call and source["pending_file_sha256_at_callback"] == record["shard_sha256"], f"canonical source lineage mismatch at call {call}")
        shard_path = Path(record["shard"]).resolve(strict=True)
        BASE.require(shard_path.parent == manifest_path.parent / "shards" and BASE.sha256_file(shard_path) == record["shard_sha256"], f"canonical shard binding mismatch at call {call}")
        with np.load(shard_path, allow_pickle=False) as shard:
            path = np.asarray(shard["path32_A"], dtype=np.float64); energy = np.asarray(shard["energy_eV"], dtype=np.float64); force = np.asarray(shard["force_eV_per_A"], dtype=np.float64)
            BASE.require(path.shape == force.shape == (32, 9396) and energy.shape == (32,) and np.all(np.isfinite(path)) and np.all(np.isfinite(energy)) and np.all(np.isfinite(force)), f"invalid full state at call {call}")
            BASE.require(str(scalar(shard, "schema")) == RECORDER.SCHEMA and int(scalar(shard, "call_per_image")) == call and int(scalar(shard, "raw_record_ordinal")) == record["source_raw_record_ordinal"], f"shard identity mismatch at call {call}")
            for array, field in ((path, "path"), (energy, "energy"), (force, "force")):
                expected = record[f"{field}_sha256"]
                BASE.require(BASE.sha256_array(array) == expected == str(scalar(shard, f"{field}_sha256")), f"{field} hash mismatch at call {call}")
            BASE.require(BASE.exact(path, formal["stored_paths"][index]), f"formal path/shard mismatch at call {call}")
            joint = np.asarray(shard["joint_evaluation_vector"], dtype=np.int64)
            energy_req = np.asarray(shard["energy_request_vector"], dtype=np.int64); force_req = np.asarray(shard["force_request_vector"], dtype=np.int64); hits = np.asarray(shard["cache_hit_vector"], dtype=np.int64)
            BASE.require(BASE.exact(joint, formal["counters"][index]) and BASE.exact(energy_req, formal["energy_requests"][index]) and BASE.exact(force_req, formal["force_requests"][index]) and BASE.exact(hits, formal["cache_hits"][index]), f"formal counter/shard mismatch at call {call}")
            BASE.require(int(joint[1:-1].sum()) == int(formal["nf"][index]) == int(scalar(shard, "optimization_N_F")), f"N_F closure mismatch at call {call}")
            measured_i = BASE.discrete_i_perp(path, force); measured_barrier = float((np.max(energy) - energy[0]) * 23.0605)
            BASE.require(np.asarray(measured_i).tobytes() == np.asarray(formal["i_perp"][index]).tobytes() == np.asarray(scalar(shard, "discrete_I_perp")).tobytes(), f"I_perp mismatch at call {call}")
            BASE.require(np.asarray(measured_barrier).tobytes() == np.asarray(formal["barrier"][index]).tobytes(), f"barrier mismatch at call {call}")
            keys = strings(shard, "counter_schema_keys"); unavailable = strings(shard, "unavailable_counter_fields"); present = bool(scalar(shard, "inherited_cache_entries_present")); kind = str(scalar(shard, "counter_source_kind"))
            if kind == "live_counter_arrays": schema_counts["live_schema8"] += 1
            elif present: schema_counts["historical_schema8"] += 1
            else: schema_counts["historical_schema7"] += 1
            BASE.require(int(scalar(shard, "additional_physical_calls")) == 0 and bool(scalar(shard, "formal_reportability_pending_final_result")), f"capture side effect metadata invalid at call {call}")
        inventory.append({"history_index": index, "call_per_image": call, "optimization_N_F": int(formal["nf"][index]), "path_sha256": record["path_sha256"], "energy_sha256": record["energy_sha256"], "force_sha256": record["force_sha256"], "shard": str(shard_path), "shard_sha256": record["shard_sha256"], "formally_reportable": call <= formal["last_complete"], "terminal_uncommitted_charged_tail": call > formal["last_complete"]})
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tolerance_fallback_used": False,
        "exact_old_9990_complete_prefix_call_range": [1, PREFIX_CALLS],
        "exact_old_9990_complete_prefix_last_N_F": PREFIX_NF,
        "old_9990_terminal_tail_required_equal": False,
        "actual_raw_callback_count": len(raw),
        "actual_canonical_call_count": formal["count"],
        "last_complete_call": formal["last_complete"],
        "terminal_optimization_N_F": formal["terminal_nf"],
        "native_all_paths_equal_canonical_shards": True,
        "canonical_path_energy_full_force_same_callback": True,
        "raw_callbacks_large_arrays_materialized": False,
        "additional_physical_calls_from_capture": 0,
        "optimization_N_F_contribution_from_capture": 0,
        "path_inventory": inventory,
        "bindings": {"registration": BASE.bind(registration_path), "authorization": BASE.bind(authorization_path), "reference_result": BASE.bind(reference_path), "replay_result": BASE.bind(replay_path), "replay_audit": BASE.bind(replay_audit_path), "capture_manifest": BASE.bind(manifest_path), "canonical_index": BASE.bind(canonical_index_path), "capture_audit": BASE.bind(capture_audit_path)},
    }
    return payload


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); parser.add_argument("--registration", type=Path, required=True); parser.add_argument("--authorization", type=Path, required=True); parser.add_argument("--reference-result", type=Path, required=True); parser.add_argument("--replay-result", type=Path, required=True); parser.add_argument("--replay-audit", type=Path, required=True); parser.add_argument("--capture-manifest", type=Path, required=True); parser.add_argument("--canonical-index", type=Path, required=True); parser.add_argument("--capture-audit", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None); root = args.root.expanduser().resolve(strict=True)
    inputs = {name: value.expanduser().resolve(strict=True) for name, value in vars(args).items() if isinstance(value, Path) and name not in {"root", "output"}}
    payload = verify(root, inputs["registration"], inputs["authorization"], inputs["reference_result"], inputs["replay_result"], inputs["replay_audit"], inputs["capture_manifest"], inputs["canonical_index"], inputs["capture_audit"])
    BASE.write_exclusive(args.output, payload); destination = args.output.expanduser().resolve(strict=True); print(destination); print(BASE.sha256_file(destination))


if __name__ == "__main__":
    main()
