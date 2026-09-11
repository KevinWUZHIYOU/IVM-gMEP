#!/usr/bin/env python3
"""Zero-OpenMM verifier for schema-aware selected full-state replay R4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import numpy as np

import verify_selected_ivm_full_state_history_replay as BASE
from full_state_history_recorder_r4 import (
    AUDIT_SCHEMA,
    AUDIT_STATUS,
    COMMON_COUNTER_KEYS,
    INHERITED_COUNTER_KEY,
    LIVE_COUNTER_KEYS,
    MANIFEST_SCHEMA,
    SCHEMA as RECORDER_SCHEMA,
)


SCHEMA = "bb-irivm-selected-full-state-history-verification-v4"
REGISTRATION_SCHEMA = "bb-irivm-selected-full-state-history-output-only-replay-v4"
AUTHORIZATION_SCHEMA = "bb-irivm-selected-full-state-history-output-only-replay-authorization-v4"
REGISTRATION_STATUS = "FROZEN_BEFORE_FIRST_R4_FULL_STATE_HISTORY_REPLAY_PHYSICAL_CALL"
LABEL = "bb_irivm_a1e0_r01_m05_inf9990_fullstate_r4"
ROWS = 352
LAST_COMPLETE_CALL = 318


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


def _strings(saved, key: str) -> list[str]:
    BASE.require(key in saved.files, f"shard lacks {key}")
    value = np.asarray(saved[key])
    BASE.require(value.ndim == 1 and value.dtype.kind in "US", f"{key} is not an explicit Unicode/string vector")
    return value.astype(str).tolist()


def _counter_vector(saved, field: str, call: int) -> np.ndarray:
    key = VECTOR_FIELDS[field]
    BASE.require(key in saved.files, f"shard lacks {field} at call {call}")
    raw = np.asarray(saved[key])
    BASE.require(raw.shape == (32,) and raw.dtype.kind in "iu", f"invalid {field} vector at call {call}")
    BASE.require(np.all(raw >= 0), f"negative {field} vector at call {call}")
    return np.ascontiguousarray(raw, dtype=np.int64)


def verify_shards_r4(
    registration_path: Path,
    manifest_path: Path,
    capture_audit: dict,
    manifest: dict,
    arrays: dict,
):
    records = manifest.get("records")
    BASE.require(isinstance(records, list) and len(records) == ROWS, "manifest must contain 352 records")
    BASE.require(manifest.get("counter_schema_policy") == "exact live-8 or historical-7/8; absent inherited values remain unavailable", "manifest counter policy mismatch")
    wal_path = Path(str(capture_audit.get("wal", ""))).expanduser().resolve(strict=True)
    BASE.require(BASE.sha256_file(wal_path) == capture_audit.get("wal_sha256"), "capture WAL hash drift")
    wal = BASE.read_wal(wal_path)
    BASE.require(len(wal) == ROWS + 2, "WAL must contain header, 352 records, and manifest commit")
    BASE.require(
        wal[0].get("schema") == RECORDER_SCHEMA
        and wal[0].get("event") == "header"
        and wal[0].get("label") == LABEL,
        "WAL header invalid",
    )
    BASE.require(wal[0].get("registration") == str(registration_path), "WAL registration path mismatch")
    BASE.require(wal[0].get("registration_sha256") == BASE.sha256_file(registration_path), "WAL registration hash mismatch")
    BASE.require(wal[0].get("missing_counter_policy") == "explicitly unavailable; never guessed or backfilled", "WAL missing-counter policy mismatch")
    inventory = []
    stored_index = {int(call): index for index, call in enumerate(arrays["stored_calls"])}
    schema_counts = {"live_schema8": 0, "historical_schema7": 0, "historical_schema8": 0}
    for index, record in enumerate(records):
        call = index + 1
        BASE.require(record == wal[index + 1], f"manifest/WAL mismatch at call {call}")
        BASE.require(record.get("event") == "record_committed", f"bad WAL event at call {call}")
        BASE.require(record.get("history_index") == index and record.get("call_per_image") == call, f"record ordering mismatch at call {call}")
        BASE.require(record.get("optimization_N_F") == int(arrays["nf"][index]), f"WAL N_F mismatch at call {call}")
        shard_path = Path(str(record.get("shard", ""))).expanduser().resolve(strict=True)
        BASE.require(shard_path.parent == manifest_path.parent / "shards", f"shard outside authoritative directory at call {call}")
        BASE.require(BASE.sha256_file(shard_path) == record.get("shard_sha256"), f"shard file hash drift at call {call}")
        with np.load(shard_path, allow_pickle=False) as shard:
            path = np.asarray(shard["path32_A"], dtype=np.float64)
            energy = np.asarray(shard["energy_eV"], dtype=np.float64)
            force = np.asarray(shard["force_eV_per_A"], dtype=np.float64)
            BASE.require(path.shape == force.shape == (32, 9396) and energy.shape == (32,), f"full-state shape mismatch at call {call}")
            BASE.require(np.all(np.isfinite(path)) and np.all(np.isfinite(energy)) and np.all(np.isfinite(force)), f"non-finite full state at call {call}")
            BASE.require(str(BASE.scalar(shard, "schema")) == RECORDER_SCHEMA, f"shard schema mismatch at call {call}")
            BASE.require(str(BASE.scalar(shard, "label")) == LABEL, f"shard label mismatch at call {call}")
            BASE.require(int(BASE.scalar(shard, "history_index")) == index and int(BASE.scalar(shard, "call_per_image")) == call, f"shard index mismatch at call {call}")
            BASE.require(int(BASE.scalar(shard, "optimization_N_F")) == int(arrays["nf"][index]), f"shard N_F mismatch at call {call}")
            source = str(BASE.scalar(shard, "counter_source_kind"))
            keys = _strings(shard, "counter_schema_keys")
            unavailable = _strings(shard, "unavailable_counter_fields")
            present = bool(BASE.scalar(shard, "inherited_cache_entries_present"))
            expected_seven = list(COMMON_COUNTER_KEYS)
            expected_eight = [*COMMON_COUNTER_KEYS, INHERITED_COUNTER_KEY]
            BASE.require(record.get("counter_source_kind") == source and record.get("counter_schema_keys") == keys and record.get("unavailable_counter_fields") == unavailable and record.get("inherited_cache_entries_present") is present, f"counter metadata WAL/shard mismatch at call {call}")
            if source == "live_counter_arrays":
                BASE.require(keys == expected_eight and present and unavailable == [], f"live counter schema is not exact-8 at call {call}")
                schema_counts["live_schema8"] += 1
            else:
                BASE.require(source == "historical_counter_snapshot", f"unknown counter source at call {call}")
                if present:
                    BASE.require(keys == expected_eight and unavailable == [], f"historical schema-8 metadata mismatch at call {call}")
                    schema_counts["historical_schema8"] += 1
                else:
                    BASE.require(keys == expected_seven and unavailable == [INHERITED_COUNTER_KEY], f"historical schema-7 metadata mismatch at call {call}")
                    BASE.require("inherited_cache_entry_vector" not in shard.files, f"schema-7 shard guessed inherited values at call {call}")
                    schema_counts["historical_schema7"] += 1
            counter_vectors = {field: _counter_vector(shard, field, call) for field in keys}
            if present:
                BASE.require("inherited_cache_entry_vector" in shard.files, f"schema-8 shard lacks inherited values at call {call}")
                expected_inherited = np.zeros(32, dtype=np.int64)
                expected_inherited[[0, -1]] = 1
                BASE.require(np.array_equal(counter_vectors[INHERITED_COUNTER_KEY], expected_inherited), f"inherited-cache vector drift at call {call}")
            expected_file_fields = {
                "schema", "label", "history_index", "call_per_image", "optimization_N_F",
                "path32_A", "energy_eV", "force_eV_per_A", "path_sha256", "energy_sha256", "force_sha256",
                "counter_source_kind", "counter_schema_keys", "unavailable_counter_fields", "inherited_cache_entries_present",
                "joint_evaluation_vector", "energy_request_vector", "force_request_vector", "cache_hit_vector",
                "total_query_event_vector", "both_request_vector", "budget_blocked_miss_vector",
                "discrete_I_perp", "fully_evaluated_current_path", "provisional_reportable",
                "terminal_uncommitted_tail", "additional_physical_calls"
            }
            if present:
                expected_file_fields.add("inherited_cache_entry_vector")
            BASE.require(set(shard.files) == expected_file_fields, f"unexpected shard field set at call {call}")
            path_sha = BASE.sha256_array(path)
            energy_sha = BASE.sha256_array(energy)
            force_sha = BASE.sha256_array(force)
            BASE.require(path_sha == record.get("path_sha256") == str(BASE.scalar(shard, "path_sha256")), f"path hash mismatch at call {call}")
            BASE.require(energy_sha == record.get("energy_sha256") == str(BASE.scalar(shard, "energy_sha256")), f"energy hash mismatch at call {call}")
            BASE.require(force_sha == record.get("force_sha256") == str(BASE.scalar(shard, "force_sha256")), f"force hash mismatch at call {call}")
            BASE.require(BASE.exact(counter_vectors["joint_evaluations"], arrays["counters"][index]), f"joint counter mismatch at call {call}")
            BASE.require(int(np.sum(counter_vectors["joint_evaluations"][1:-1], dtype=np.int64)) == int(arrays["nf"][index]), f"joint counter/N_F sum mismatch at call {call}")
            BASE.require(BASE.exact(counter_vectors["energy_requests"], arrays["energy_requests"][index]), f"energy request mismatch at call {call}")
            BASE.require(BASE.exact(counter_vectors["force_requests"], arrays["force_requests"][index]), f"force request mismatch at call {call}")
            BASE.require(BASE.exact(counter_vectors["cache_hits"], arrays["cache_hits"][index]), f"cache hit mismatch at call {call}")
            BASE.require(np.array_equal(arrays["energy_requests"][index] + arrays["force_requests"][index] - counter_vectors["both_requests"], counter_vectors["total_query_events"]), f"request identity failed at call {call}")
            BASE.require(np.array_equal(arrays["cache_hits"][index] + arrays["counters"][index] + counter_vectors["budget_blocked_misses"], counter_vectors["total_query_events"]), f"counter outcome identity failed at call {call}")
            measured_i_perp = BASE.discrete_i_perp(path, force)
            BASE.require(np.asarray(measured_i_perp).tobytes() == np.asarray(arrays["i_perp"][index]).tobytes() == np.asarray(BASE.scalar(shard, "discrete_I_perp")).tobytes(), f"discrete I_perp is not exact at call {call}")
            measured_barrier = float((np.max(energy) - energy[0]) * 23.0605)
            BASE.require(np.asarray(measured_barrier).tobytes() == np.asarray(arrays["barrier_kcal"][index]).tobytes(), f"sparse barrier is not exact at call {call}")
            reportable = call <= LAST_COMPLETE_CALL
            BASE.require(bool(BASE.scalar(shard, "fully_evaluated_current_path")), f"path not fully evaluated at call {call}")
            BASE.require(bool(BASE.scalar(shard, "provisional_reportable")) is reportable and bool(BASE.scalar(shard, "terminal_uncommitted_tail")) is (not reportable), f"reportability mask mismatch at call {call}")
            BASE.require(int(BASE.scalar(shard, "additional_physical_calls")) == 0 and record.get("additional_physical_calls") == 0, f"observer physical-call count nonzero at call {call}")
            if call in stored_index:
                BASE.require(BASE.exact(path, arrays["stored_paths"][stored_index[call]]), f"committed result/shard path mismatch at call {call}")
        inventory.append({
            "history_index": index, "call_per_image": call,
            "optimization_N_F": int(arrays["nf"][index]),
            "discrete_I_perp_eV2_per_A": float(arrays["i_perp"][index]),
            "path_sha256": record["path_sha256"], "energy_sha256": record["energy_sha256"], "force_sha256": record["force_sha256"],
            "shard": str(shard_path), "shard_sha256": record["shard_sha256"],
            "counter_source_kind": source, "counter_schema_keys": keys,
            "unavailable_counter_fields": unavailable,
            "fully_evaluated_current_path": True,
            "formally_reportable": reportable,
            "terminal_uncommitted_charged_tail": not reportable,
            "exclusion_reason": None if reportable else "after_last_complete_state_in_terminal_uncommitted_transaction"
        })
    BASE.require(manifest.get("counter_schema_counts") == schema_counts == capture_audit.get("counter_schema_counts"), "counter schema counts mismatch")
    BASE.require(capture_audit.get("missing_counter_values_guessed_or_backfilled") is False, "capture audit permits guessed counters")
    BASE.require(
        wal[-1].get("event") == "manifest_committed"
        and wal[-1].get("record_count") == ROWS
        and wal[-1].get("manifest") == str(manifest_path)
        and wal[-1].get("manifest_sha256") == BASE.sha256_file(manifest_path),
        "WAL manifest commit invalid",
    )
    return inventory, {
        "record_count": ROWS,
        "reportable_record_count": LAST_COMPLETE_CALL,
        "quarantined_terminal_tail_record_count": ROWS - LAST_COMPLETE_CALL,
        "one_authoritative_atomic_shard_per_record": True,
        "duplicate_large_aggregate_created": False,
        "path_energy_full_force_same_record": True,
        "all_discrete_I_perp_values_exactly_reconstructed": True,
        "counter_schema_counts": schema_counts,
        "missing_counter_values_guessed_or_backfilled": False,
        "additional_physical_calls_from_capture": 0,
        "wal": BASE.bind(wal_path)
    }


def main(argv: Iterable[str] | None = None) -> None:
    tokens = list(argv) if argv is not None else None
    identity_parser = argparse.ArgumentParser(add_help=False)
    identity_parser.add_argument("--registration", type=Path, required=True)
    identity, _ = identity_parser.parse_known_args(tokens)
    registration_path = identity.registration.expanduser().resolve(strict=True)
    registration = BASE.read_json(registration_path)
    BASE.require(registration.get("source_sha256", {}).get(Path(__file__).name) == BASE.sha256_file(Path(__file__).resolve(strict=True)), "R4 verifier source drift")
    overrides = {
        "SCHEMA": SCHEMA,
        "REGISTRATION_SCHEMA": REGISTRATION_SCHEMA,
        "AUTHORIZATION_SCHEMA": AUTHORIZATION_SCHEMA,
        "REGISTRATION_STATUS": REGISTRATION_STATUS,
        "MANIFEST_SCHEMA": MANIFEST_SCHEMA,
        "CAPTURE_AUDIT_SCHEMA": AUDIT_SCHEMA,
        "CAPTURE_AUDIT_STATUS": AUDIT_STATUS,
        "RECORDER_SCHEMA": RECORDER_SCHEMA,
        "LABEL": LABEL,
        "verify_shards": verify_shards_r4,
    }
    prior = {name: getattr(BASE, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(BASE, name, value)
        BASE.main(tokens)
    finally:
        for name, value in prior.items():
            setattr(BASE, name, value)


if __name__ == "__main__":
    main()
