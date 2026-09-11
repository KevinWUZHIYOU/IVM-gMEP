#!/usr/bin/env python3
"""Strict zero-OpenMM supplemental counter/lineage verifier for R5B."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

import full_state_history_recorder_r5 as RECORDER
import run_selected_ivm_full_state_history_replay_r5 as RUNNER
import verify_selected_ivm_full_state_history_replay as BASE
import verify_selected_ivm_full_state_history_replay_r5 as PRIMARY


SCHEMA = "bb-irivm-full-state-r5b-supplemental-verification-v1"
STATUS = "PASS_R5B_EXACT_COUNTER_SCHEMA_AND_ARTIFACT_CLOSURE"
FROZEN = {
    "recorder": (
        Path(__file__).resolve().parent / "full_state_history_recorder_r5.py",
        "16faa440598463abed44761d7d34517f35d65bb4794bb92526c7c4aa8997c9aa",
    ),
    "runner": (
        Path(__file__).resolve().parent / "run_selected_ivm_full_state_history_replay_r5.py",
        "f1a0c709d3d98a7d84c3d5b66eb3f34ee4c7c6a7fa51ecab60f6b8ff67edc93d",
    ),
    "primary_verifier": (
        Path(__file__).resolve().parent / "verify_selected_ivm_full_state_history_replay_r5.py",
        "b6d4e7b8258e338f2b689c67be073e164c5440a6d84998bd7d352388935c1d0b",
    ),
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


def require_frozen_sources() -> dict:
    bindings = {}
    for role, (path, expected) in FROZEN.items():
        resolved = path.resolve(strict=True)
        BASE.require(BASE.sha256_file(resolved) == expected, f"frozen R5B source drift: {role}")
        bindings[role] = BASE.bind(resolved)
    return bindings


def _strings(saved, key: str) -> list[str]:
    value = np.asarray(saved[key])
    BASE.require(value.ndim == 1 and value.dtype.kind in "US", f"invalid string vector: {key}")
    return value.astype(str).tolist()


def verify(
    root: Path,
    registration_path: Path,
    authorization_path: Path,
    primary_path: Path,
    manifest_path: Path,
    canonical_path: Path,
    audit_path: Path,
) -> dict:
    source_bindings = require_frozen_sources()
    registration_path = registration_path.resolve(strict=True)
    authorization_path = authorization_path.resolve(strict=True)
    primary_path = primary_path.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    canonical_path = canonical_path.resolve(strict=True)
    audit_path = audit_path.resolve(strict=True)
    registration = RUNNER.validate_registration(root, registration_path)
    RUNNER.validate_authorization(registration_path, authorization_path, registration)
    primary = BASE.read_json(primary_path)
    BASE.require(
        primary.get("schema") == PRIMARY.SCHEMA
        and primary.get("status") == PRIMARY.STATUS,
        "primary R5B verification is not PASS",
    )
    expected_primary_bindings = {
        "registration": BASE.bind(registration_path),
        "authorization": BASE.bind(authorization_path),
        "capture_manifest": BASE.bind(manifest_path),
        "canonical_index": BASE.bind(canonical_path),
        "capture_audit": BASE.bind(audit_path),
    }
    for role, binding in expected_primary_bindings.items():
        BASE.require(primary.get("bindings", {}).get(role) == binding, f"primary binding drift: {role}")

    manifest = BASE.read_json(manifest_path)
    canonical = BASE.read_json(canonical_path)
    audit = BASE.read_json(audit_path)
    BASE.require(manifest.get("schema") == RECORDER.MANIFEST_SCHEMA, "manifest schema drift")
    BASE.require(canonical.get("schema") == RECORDER.CANONICAL_INDEX_SCHEMA, "canonical schema drift")
    BASE.require(audit.get("schema") == RECORDER.AUDIT_SCHEMA, "capture audit schema drift")
    BASE.require(manifest.get("canonical_index") == str(canonical_path) and manifest.get("canonical_index_sha256") == BASE.sha256_file(canonical_path), "manifest/canonical binding drift")
    BASE.require(audit.get("manifest") == str(manifest_path) and audit.get("manifest_sha256") == BASE.sha256_file(manifest_path), "audit/manifest binding drift")
    BASE.require(audit.get("canonical_index") == str(canonical_path) and audit.get("canonical_index_sha256") == BASE.sha256_file(canonical_path), "audit/canonical binding drift")
    for record, role in ((manifest, "manifest"), (audit, "audit")):
        for field, path_field in (("source_result_sha256", "source_result"), ("sealed_reference_result_sha256", "sealed_reference_result")):
            bound = Path(record[path_field]).resolve(strict=True)
            BASE.require(BASE.sha256_file(bound) == record[field], f"{role} {path_field} binding drift")

    raw = manifest.get("raw_callbacks")
    records = manifest.get("canonical_records")
    BASE.require(isinstance(raw, list) and isinstance(records, list), "manifest inventories missing")
    BASE.require(canonical.get("records") == records, "canonical/manifest record inventory drift")
    BASE.require(manifest.get("raw_record_count") == len(raw) and manifest.get("canonical_record_count") == len(records), "manifest count drift")
    BASE.require(canonical.get("raw_record_count") == len(raw) and canonical.get("canonical_record_count") == len(records), "canonical count drift")
    raw_calls = np.asarray([row["call_per_image"] for row in raw], dtype=np.int64)
    stable = RECORDER.stable_last_raw_ordinals(raw_calls)
    BASE.require(canonical.get("source_raw_record_ordinals") == stable.tolist(), "canonical stable-last index drift")
    BASE.require([row["source_raw_record_ordinal"] for row in records] == stable.tolist(), "record stable-last index drift")

    wal_path = Path(audit["wal"]).resolve(strict=True)
    BASE.require(BASE.sha256_file(wal_path) == audit["wal_sha256"], "audit/WAL binding drift")
    wal = BASE.read_wal(wal_path)
    raw_wal = [row for row in wal if row.get("event") == "raw_callback_pending_state_committed"]
    canonical_wal = [row for row in wal if row.get("event") == "canonical_shard_promoted"]
    BASE.require(raw_wal == raw and canonical_wal == records, "WAL inventories differ from manifest")
    BASE.require(wal[0].get("schema") == RECORDER.SCHEMA and wal[0].get("registration_sha256") == BASE.sha256_file(registration_path), "WAL header drift")
    BASE.require(wal[-1].get("event") == "manifest_committed" and wal[-1].get("manifest_sha256") == BASE.sha256_file(manifest_path), "WAL footer drift")

    expected7 = list(RECORDER.COMMON_COUNTER_KEYS)
    expected8 = [*RECORDER.COMMON_COUNTER_KEYS, RECORDER.INHERITED_COUNTER_KEY]
    endpoint = np.zeros(32, dtype=np.int64)
    endpoint[[0, -1]] = 1
    counts = {"live_schema8": 0, "historical_schema7": 0, "historical_schema8": 0}
    inventory = []
    for index, record in enumerate(records):
        call = index + 1
        source = raw[int(record["source_raw_record_ordinal"])]
        for field in (
            "call_per_image", "optimization_N_F", "path_sha256", "energy_sha256",
            "force_sha256", "counter_source_kind", "counter_schema_keys",
            "unavailable_counter_fields", "inherited_cache_entries_present",
            "additional_physical_calls",
        ):
            BASE.require(source.get(field) == record.get(field), f"raw/canonical field drift at call {call}: {field}")
        BASE.require(record["history_index"] == index and record["call_per_image"] == call, f"canonical ordering drift at call {call}")
        BASE.require(source["pending_file_sha256_at_callback"] == record["shard_sha256"], f"pending/shard lineage drift at call {call}")
        shard_path = Path(record["shard"]).resolve(strict=True)
        BASE.require(shard_path.parent == manifest_path.parent / "shards" and BASE.sha256_file(shard_path) == record["shard_sha256"], f"shard binding drift at call {call}")
        with np.load(shard_path, allow_pickle=False) as shard:
            BASE.require(str(PRIMARY.scalar(shard, "schema")) == RECORDER.SCHEMA, f"shard schema drift at call {call}")
            BASE.require(int(PRIMARY.scalar(shard, "call_per_image")) == call and int(PRIMARY.scalar(shard, "raw_record_ordinal")) == record["source_raw_record_ordinal"], f"shard identity drift at call {call}")
            kind = str(PRIMARY.scalar(shard, "counter_source_kind"))
            keys = _strings(shard, "counter_schema_keys")
            unavailable = _strings(shard, "unavailable_counter_fields")
            present = bool(PRIMARY.scalar(shard, "inherited_cache_entries_present"))
            BASE.require(kind == record["counter_source_kind"] and keys == record["counter_schema_keys"] and unavailable == record["unavailable_counter_fields"] and present is record["inherited_cache_entries_present"], f"shard/canonical counter metadata drift at call {call}")
            if kind == "live_counter_arrays":
                BASE.require(present and keys == expected8 and unavailable == [], f"live exact8 violation at call {call}")
                counts["live_schema8"] += 1
            else:
                BASE.require(kind == "historical_counter_snapshot", f"unknown counter source at call {call}")
                if present:
                    BASE.require(keys == expected8 and unavailable == [], f"historical exact8 violation at call {call}")
                    counts["historical_schema8"] += 1
                else:
                    BASE.require(keys == expected7 and unavailable == [RECORDER.INHERITED_COUNTER_KEY], f"historical exact7 violation at call {call}")
                    BASE.require("inherited_cache_entry_vector" not in shard.files, f"historical7 stores unavailable inherited vector at call {call}")
                    counts["historical_schema7"] += 1
            for counter_name in keys:
                field = VECTOR_FIELDS[counter_name]
                vector = np.asarray(shard[field])
                BASE.require(vector.shape == (32,) and vector.dtype.kind in "iu" and np.all(vector >= 0), f"invalid {counter_name} at call {call}")
            if present:
                BASE.require(np.array_equal(np.asarray(shard["inherited_cache_entry_vector"], dtype=np.int64), endpoint), f"inherited endpoint vector drift at call {call}")
            joint = np.asarray(shard["joint_evaluation_vector"], dtype=np.int64)
            BASE.require(int(joint[1:-1].sum(dtype=np.int64)) == record["optimization_N_F"] == int(PRIMARY.scalar(shard, "optimization_N_F")), f"counter/N_F closure drift at call {call}")
            BASE.require(int(PRIMARY.scalar(shard, "additional_physical_calls")) == 0, f"capture call contribution at call {call}")
        inventory.append({"call_per_image": call, "counter_source_kind": kind, "counter_schema_keys": keys, "unavailable_counter_fields": unavailable, "inherited_cache_entries_present": present, "optimization_N_F": record["optimization_N_F"], "shard_sha256": record["shard_sha256"]})

    raw_counts = {"live_schema8": 0, "historical_schema7": 0, "historical_schema8": 0}
    for row in raw:
        kind = row.get("counter_source_kind")
        present = row.get("inherited_cache_entries_present")
        if kind == "live_counter_arrays" and present is True:
            raw_counts["live_schema8"] += 1
        elif kind == "historical_counter_snapshot" and present is True:
            raw_counts["historical_schema8"] += 1
        elif kind == "historical_counter_snapshot" and present is False:
            raw_counts["historical_schema7"] += 1
        else:
            raise RuntimeError("raw callback counter schema classification drift")
    BASE.require(raw_counts == manifest.get("counter_schema_counts") == audit.get("counter_schema_counts"), "raw counter schema counts drift")
    BASE.require(audit.get("missing_counter_values_guessed_or_backfilled") is False, "audit permits counter backfill")
    BASE.require(audit.get("additional_physical_calls") == 0 and audit.get("optimization_N_F_contribution_from_capture") == 0, "capture accounting drift")
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "OpenMM_imported_or_called": False,
        "counter_schema_counts": counts,
        "raw_counter_schema_counts": raw_counts,
        "canonical_record_count": len(records),
        "raw_record_count": len(raw),
        "all_counter_vectors_shape_dtype_nonnegative_checked": True,
        "missing_counter_values_guessed_or_backfilled": False,
        "additional_physical_calls": 0,
        "optimization_N_F_contribution": 0,
        "inventory": inventory,
        "bindings": {
            **source_bindings,
            "registration": BASE.bind(registration_path),
            "authorization": BASE.bind(authorization_path),
            "primary_verification": BASE.bind(primary_path),
            "manifest": BASE.bind(manifest_path),
            "canonical_index": BASE.bind(canonical_path),
            "capture_audit": BASE.bind(audit_path),
            "wal": BASE.bind(wal_path),
        },
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--primary-verification", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--canonical-index", type=Path, required=True)
    parser.add_argument("--capture-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = verify(args.root.resolve(strict=True), args.registration, args.authorization, args.primary_verification, args.manifest, args.canonical_index, args.capture_audit)
    BASE.write_exclusive(args.output, payload)
    output = args.output.resolve(strict=True)
    print(output)
    print(BASE.sha256_file(output))


if __name__ == "__main__":
    main()
