#!/usr/bin/env python3
"""Minimal storage-only full-state recorder for selected BB IR--IVM R5.

Production may call its ``record`` closure more than once at the same
``call_per_image`` value.  R5 writes only small metadata/hash WAL entries for
all raw callbacks.  One atomic pending full-state file is replaced while
the call is unchanged; when the call advances, the previous pending state is
promoted to an immutable shard.  The result is one stable-last shard and one
canonical reference for every call in the actual post-run formal call axis,
matching production's own final-history deduplication without materializing
same-call redundant large states.  The observer
only copies arrays already present in exact hash-bound production frames and
never queries an evaluator.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import shutil

import numpy as np

import full_state_history_recorder_r4 as R4


SCHEMA = "bb-irivm-full-state-history-recorder-v3"
MANIFEST_SCHEMA = "bb-irivm-full-state-history-manifest-v3"
CANONICAL_INDEX_SCHEMA = "bb-irivm-full-state-history-stable-last-index-v1"
AUDIT_SCHEMA = "bb-irivm-full-state-history-audit-v3"
AUDIT_STATUS = (
    "PASS_DYNAMIC_STABLE_LAST_STORAGE_ONLY_CAPTURE_PENDING_EXACT_VERIFICATION"
)
OPTIMIZATION_NF_BUDGET = 15000
MAXIMUM_RAW_CALLBACKS = 4 * OPTIMIZATION_NF_BUDGET + 1
MAXIMUM_CANONICAL_CALLS = OPTIMIZATION_NF_BUDGET
MINIMUM_RUNTIME_FREE_BYTES = 1 * 1024**3
OLD_COMPLETE_PREFIX_CALLS = 318

COMMON_COUNTER_KEYS = R4.COMMON_COUNTER_KEYS
INHERITED_COUNTER_KEY = R4.INHERITED_COUNTER_KEY
HISTORICAL_COUNTER_KEYS = R4.HISTORICAL_COUNTER_KEYS
LIVE_COUNTER_KEYS = R4.LIVE_COUNTER_KEYS
EXPECTED_PRODUCTION_SHA256 = R4.EXPECTED_PRODUCTION_SHA256

require = R4.require
sha256_array = R4.sha256_array
sha256_file = R4.sha256_file
append_fsync_jsonl = R4.append_fsync_jsonl
atomic_npz = R4.atomic_npz
fsync_directory = R4.fsync_directory
freeze_frame_code_contract = R4.freeze_frame_code_contract
_counter_arrays = R4._counter_arrays


def _exclusive_json(path: Path, payload: dict) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    fsync_directory(path.parent)


def stable_last_raw_ordinals(calls) -> np.ndarray:
    values = np.asarray(calls, dtype=np.int64)
    require(
        values.ndim == 1 and 0 < values.size <= MAXIMUM_RAW_CALLBACKS,
        "invalid raw callback count",
    )
    require(np.all(values[1:] >= values[:-1]), "raw call axis is not monotone")
    _, reverse_indices = np.unique(values[::-1], return_index=True)
    keep = values.size - 1 - reverse_indices
    keep = keep[np.argsort(values[keep], kind="stable")]
    require(
        keep.size <= MAXIMUM_CANONICAL_CALLS
        and np.array_equal(values[keep], np.arange(1, keep.size + 1)),
        "stable-last raw callbacks do not form a continuous call axis from 1",
    )
    return np.ascontiguousarray(keep, dtype=np.int64)


class FullStateHistoryRecorder:
    """Persist stable-last full states and raw callback metadata only."""

    def __init__(
        self,
        root: Path,
        label: str,
        registration_path: Path,
        registration_sha256: str,
    ):
        self.root = root.expanduser().resolve()
        require(
            not self.root.exists(),
            f"refusing to overwrite observer directory: {self.root}",
        )
        self.shards = self.root / "shards"
        self.shards.mkdir(parents=True)
        fsync_directory(self.root)
        fsync_directory(self.root.parent)
        self.wal = self.root / "full_state_history.wal.jsonl"
        self.label = str(label)
        self.registration_path = registration_path.expanduser().resolve(strict=True)
        self.registration_sha256 = str(registration_sha256)
        self.raw_records: list[dict] = []
        self.records: list[dict] = []
        self.pending_path = self.root / "pending_current_call.npz"
        self.pending_record: dict | None = None
        append_fsync_jsonl(
            self.wal,
            {
                "schema": SCHEMA,
                "event": "header",
                "label": self.label,
                "registration": str(self.registration_path),
                "registration_sha256": self.registration_sha256,
                "capture": (
                    "stable-last per-call already-current path32, energy32, "
                    "force32; raw callbacks metadata/hash only; zero queries"
                ),
                "record_axis": "independent zero-based raw callback ordinal",
                "canonical_axis": "stable-last raw ordinal for each call_per_image",
                "pending_policy": "atomic replace while call repeats; immutable promote when call advances",
                "missing_counter_policy": (
                    "explicitly unavailable; never guessed or backfilled"
                ),
            },
        )

    def record(
        self,
        call: int,
        path,
        energies,
        forces,
        counters,
        history,
        diagnostic,
        *,
        counter_source_kind: str,
    ) -> None:
        ordinal = len(self.raw_records)
        call = int(call)
        require(ordinal < MAXIMUM_RAW_CALLBACKS, "too many raw record callbacks")
        require(1 <= call <= MAXIMUM_CANONICAL_CALLS, "raw callback call out of range")
        if self.raw_records:
            require(
                call >= int(self.raw_records[-1]["call_per_image"]),
                "raw callback call axis decreased",
            )
            require(
                call <= int(self.raw_records[-1]["call_per_image"]) + 1,
                "raw callback skipped a canonical call",
            )
        require(
            len(history["calls_per_image"]) == ordinal + 1,
            "raw callback ordinal is not aligned with the production history append",
        )
        require(
            int(history["calls_per_image"][-1]) == call,
            "observer is not aligned with the current history append",
        )
        path = np.ascontiguousarray(path, dtype=np.float64)
        energies = np.ascontiguousarray(energies, dtype=np.float64)
        forces = np.ascontiguousarray(forces, dtype=np.float64)
        require(path.shape == forces.shape == (32, 9396), "path/full-force shape mismatch")
        require(energies.shape == (32,), "energy shape mismatch")
        require(
            np.all(np.isfinite(path))
            and np.all(np.isfinite(energies))
            and np.all(np.isfinite(forces)),
            "non-finite full state",
        )
        require(
            shutil.disk_usage(self.root).free >= MINIMUM_RUNTIME_FREE_BYTES,
            "R5 runtime disk reserve exhausted before pending-state write",
        )
        counter_arrays, inherited_present = _counter_arrays(
            counters, counter_source_kind
        )
        joint = counter_arrays["joint_evaluations"]
        optimization_nf = int(joint[1:-1].sum(dtype=np.int64))
        require(
            optimization_nf
            == int(history["total_internal_joint_evaluations"][-1]),
            "observer N_F/history mismatch",
        )
        schema_keys = [
            key
            for key in (*COMMON_COUNTER_KEYS, INHERITED_COUNTER_KEY)
            if key in counter_arrays
        ]
        unavailable = [] if inherited_present else [INHERITED_COUNTER_KEY]
        path_sha = sha256_array(path)
        energy_sha = sha256_array(energies)
        force_sha = sha256_array(forces)
        if self.pending_record is not None and call > int(
            self.pending_record["call_per_image"]
        ):
            self._promote_pending()
        arrays = {
            "schema": np.asarray(SCHEMA),
            "label": np.asarray(self.label),
            "raw_record_ordinal": np.asarray(ordinal, dtype=np.int64),
            "raw_history_append_index": np.asarray(ordinal, dtype=np.int64),
            "call_per_image": np.asarray(call, dtype=np.int64),
            "optimization_N_F": np.asarray(optimization_nf, dtype=np.int64),
            "path32_A": path,
            "energy_eV": energies,
            "force_eV_per_A": forces,
            "path_sha256": np.asarray(path_sha),
            "energy_sha256": np.asarray(energy_sha),
            "force_sha256": np.asarray(force_sha),
            "counter_source_kind": np.asarray(counter_source_kind),
            "counter_schema_keys": np.asarray(schema_keys, dtype="U64"),
            "unavailable_counter_fields": np.asarray(unavailable, dtype="U64"),
            "inherited_cache_entries_present": np.asarray(inherited_present),
            "joint_evaluation_vector": joint,
            "energy_request_vector": counter_arrays["energy_requests"],
            "force_request_vector": counter_arrays["force_requests"],
            "cache_hit_vector": counter_arrays["cache_hits"],
            "total_query_event_vector": counter_arrays["total_query_events"],
            "both_request_vector": counter_arrays["both_requests"],
            "budget_blocked_miss_vector": counter_arrays[
                "budget_blocked_misses"
            ],
            "discrete_I_perp": np.asarray(float(diagnostic["i_perp"])),
            "fully_evaluated_current_path": np.asarray(True),
            "formal_reportability_pending_final_result": np.asarray(True),
            "additional_physical_calls": np.asarray(0, dtype=np.int64),
        }
        if inherited_present:
            expected = np.zeros(32, dtype=np.int64)
            expected[[0, -1]] = 1
            require(
                np.array_equal(counter_arrays[INHERITED_COUNTER_KEY], expected),
                "inherited-cache vector differs from endpoint-only preload",
            )
            arrays["inherited_cache_entry_vector"] = counter_arrays[
                INHERITED_COUNTER_KEY
            ]
        previous_pending_ordinal = (
            None
            if self.pending_record is None
            else int(self.pending_record["raw_record_ordinal"])
        )
        atomic_npz(self.pending_path, **arrays)
        pending_sha = sha256_file(self.pending_path)
        raw_record = {
            "event": "raw_callback_pending_state_committed",
            "raw_record_ordinal": ordinal,
            "raw_history_append_index": ordinal,
            "call_per_image": call,
            "optimization_N_F": optimization_nf,
            "pending_file": str(self.pending_path),
            "pending_file_sha256_at_callback": pending_sha,
            "superseded_same_call_pending_raw_ordinal": (
                previous_pending_ordinal
                if self.pending_record is not None
                and int(self.pending_record["call_per_image"]) == call
                else None
            ),
            "path_sha256": path_sha,
            "energy_sha256": energy_sha,
            "force_sha256": force_sha,
            "counter_source_kind": counter_source_kind,
            "counter_schema_keys": schema_keys,
            "unavailable_counter_fields": unavailable,
            "inherited_cache_entries_present": inherited_present,
            "additional_physical_calls": 0,
            "discrete_I_perp": float(diagnostic["i_perp"]),
            "formal_reportability_pending_final_result": True,
        }
        append_fsync_jsonl(self.wal, raw_record)
        self.raw_records.append(raw_record)
        self.pending_record = raw_record

    def _promote_pending(self) -> None:
        require(self.pending_record is not None, "no pending state to promote")
        record = self.pending_record
        call = int(record["call_per_image"])
        require(call == len(self.records) + 1, "stable-last promotion call drift")
        require(self.pending_path.is_file(), "pending full-state file missing")
        require(
            shutil.disk_usage(self.root).free >= MINIMUM_RUNTIME_FREE_BYTES,
            "R5 runtime disk reserve exhausted before canonical promotion",
        )
        require(
            sha256_file(self.pending_path)
            == record["pending_file_sha256_at_callback"],
            "pending full-state hash drift",
        )
        shard = self.shards / f"call_{call:04d}_stable_last.npz"
        require(not shard.exists(), f"refusing to overwrite canonical shard {shard}")
        os.link(self.pending_path, shard)
        os.chmod(shard, 0o444)
        self.pending_path.unlink()
        fsync_directory(self.shards)
        fsync_directory(self.root)
        canonical = {
            "event": "canonical_shard_promoted",
            "history_index": call - 1,
            "call_per_image": call,
            "source_raw_record_ordinal": record["raw_record_ordinal"],
            "optimization_N_F": record["optimization_N_F"],
            "shard": str(shard),
            "shard_sha256": sha256_file(shard),
            "path_sha256": record["path_sha256"],
            "energy_sha256": record["energy_sha256"],
            "force_sha256": record["force_sha256"],
            "counter_source_kind": record["counter_source_kind"],
            "counter_schema_keys": record["counter_schema_keys"],
            "unavailable_counter_fields": record["unavailable_counter_fields"],
            "inherited_cache_entries_present": record[
                "inherited_cache_entries_present"
            ],
            "additional_physical_calls": 0,
        }
        append_fsync_jsonl(self.wal, canonical)
        self.records.append(canonical)
        self.pending_record = None

    def finalize(
        self, result_path: Path, reference_result_path: Path
    ) -> tuple[Path, Path, Path]:
        result_path = result_path.expanduser().resolve(strict=True)
        reference_result_path = reference_result_path.expanduser().resolve(strict=True)
        require(
            0 < len(self.raw_records) <= MAXIMUM_RAW_CALLBACKS,
            "invalid final raw callback count",
        )
        self._promote_pending()
        require(
            0 < len(self.records) <= MAXIMUM_CANONICAL_CALLS,
            "invalid stable-last canonical shard count",
        )
        calls = np.asarray(
            [record["call_per_image"] for record in self.raw_records], dtype=np.int64
        )
        canonical_ordinals = stable_last_raw_ordinals(calls)
        canonical_records = self.records
        require(
            [row["source_raw_record_ordinal"] for row in canonical_records]
            == canonical_ordinals.tolist(),
            "promoted shards are not stable-last raw callbacks",
        )
        with np.load(result_path, allow_pickle=False) as result, np.load(
            reference_result_path, allow_pickle=False
        ) as reference:
            formal_calls = np.asarray(result["calls_per_image"], dtype=np.int64)
            require(
                np.array_equal(
                    formal_calls, np.arange(1, len(canonical_records) + 1)
                ),
                "formal stable-last call axis differs from captured canonical axis",
            )
            last_complete_call = int(
                np.asarray(result["total_internal_nf_last_complete_state_index"]).item()
            ) + 1
            require(
                1 <= last_complete_call <= len(canonical_records),
                "formal last-complete call is outside canonical axis",
            )
            formal_nf = np.asarray(
                result["total_internal_joint_evaluations"], dtype=np.int64
            )
            for index, canonical in enumerate(canonical_records):
                require(
                    canonical["optimization_N_F"] == int(formal_nf[index]),
                    f"canonical N_F differs from formal history at call {index + 1}",
                )
            stored_calls = np.asarray(result["stored_path_calls"], dtype=np.int64)
            stored_paths = np.asarray(result["stored_paths"], dtype=np.float64)
            reference_calls = np.asarray(
                reference["stored_path_calls"], dtype=np.int64
            )
            reference_paths = np.asarray(
                reference["stored_paths"], dtype=np.float64
            )
            require(
                np.array_equal(
                    stored_calls, np.arange(1, len(canonical_records) + 1)
                )
                and stored_paths.shape
                == (len(canonical_records), 32, 9396),
                "native all-path stable-last storage policy drifted",
            )
            require(
                reference_paths.shape == (reference_calls.size, 32, 9396),
                "sealed committed reference path-storage shape drifted",
            )
            for path_index, call_value in enumerate(stored_calls):
                canonical = canonical_records[int(call_value) - 1]
                require(
                    sha256_array(stored_paths[path_index])
                    == canonical["path_sha256"],
                    f"native stored path differs from canonical raw state at call {call_value}",
                )
            for reference_index, call_value in enumerate(reference_calls):
                if int(call_value) > OLD_COMPLETE_PREFIX_CALLS:
                    continue
                canonical = canonical_records[int(call_value) - 1]
                require(
                    sha256_array(reference_paths[reference_index])
                    == canonical["path_sha256"],
                    f"sealed committed path differs from canonical state at call {call_value}",
                )
        schema_counts = {
            "live_schema8": 0,
            "historical_schema7": 0,
            "historical_schema8": 0,
        }
        for ordinal, record in enumerate(self.raw_records):
            require(
                record["raw_record_ordinal"] == ordinal,
                "raw record ordinal drift",
            )
            if record["counter_source_kind"] == "live_counter_arrays":
                schema_counts["live_schema8"] += 1
            elif record["inherited_cache_entries_present"]:
                schema_counts["historical_schema8"] += 1
            else:
                schema_counts["historical_schema7"] += 1
        canonical_index = self.root / "full_state_history.canonical_index.json"
        canonical_payload = {
            "schema": CANONICAL_INDEX_SCHEMA,
            "status": "COMPLETE_PENDING_EXACT_RESULT_VERIFICATION",
            "label": self.label,
            "selection_rule": (
                "for each call_per_image keep the last raw callback, identical "
                "to production reversed-numpy-unique stable-last finalization"
            ),
            "raw_record_count": len(self.raw_records),
            "canonical_record_count": len(canonical_records),
            "calls": list(range(1, len(canonical_records) + 1)),
            "source_raw_record_ordinals": canonical_ordinals.tolist(),
            "large_arrays_duplicated_for_repeated_raw_callbacks": False,
            "additional_physical_calls": 0,
            "records": canonical_records,
        }
        _exclusive_json(canonical_index, canonical_payload)
        manifest = self.root / "full_state_history.manifest.json"
        manifest_payload = {
            "schema": MANIFEST_SCHEMA,
            "status": "COMPLETE_PENDING_EXACT_PREFIX_VERIFICATION",
            "label": self.label,
            "raw_record_count": len(self.raw_records),
            "canonical_record_count": len(canonical_records),
            "source_result": str(result_path),
            "source_result_sha256": sha256_file(result_path),
            "sealed_reference_result": str(reference_result_path),
            "sealed_reference_result_sha256": sha256_file(reference_result_path),
            "canonical_index": str(canonical_index),
            "canonical_index_sha256": sha256_file(canonical_index),
            "authoritative_physical_data": (
                "one immutable full-state shard per stable-last canonical call; raw callbacks are metadata/hash WAL only"
            ),
            "counter_schema_policy": (
                "exact live-8 or historical-7/8; absent inherited values remain unavailable"
            ),
            "counter_schema_counts": schema_counts,
            "formally_reportable_call_range": [1, last_complete_call],
            "terminal_uncommitted_charged_tail_call_range": [
                last_complete_call + 1,
                len(canonical_records),
            ],
            "additional_physical_calls": 0,
            "raw_callbacks": self.raw_records,
            "canonical_records": canonical_records,
        }
        _exclusive_json(manifest, manifest_payload)
        append_fsync_jsonl(
            self.wal,
            {
                "event": "manifest_committed",
                "raw_record_count": len(self.raw_records),
                "canonical_record_count": len(canonical_records),
                "canonical_index": str(canonical_index),
                "canonical_index_sha256": sha256_file(canonical_index),
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "source_result": str(result_path),
                "source_result_sha256": sha256_file(result_path),
                "additional_physical_calls": 0,
            },
        )
        audit = self.root / "full_state_history.audit.json"
        audit_payload = {
            "schema": AUDIT_SCHEMA,
            "status": AUDIT_STATUS,
            "label": self.label,
            "raw_record_count": len(self.raw_records),
            "canonical_record_count": len(canonical_records),
            "source_result": str(result_path),
            "source_result_sha256": sha256_file(result_path),
            "sealed_reference_result": str(reference_result_path),
            "sealed_reference_result_sha256": sha256_file(reference_result_path),
            "canonical_index": str(canonical_index),
            "canonical_index_sha256": sha256_file(canonical_index),
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "wal": str(self.wal),
            "wal_sha256": sha256_file(self.wal),
            "counter_schema_counts": schema_counts,
            "missing_counter_values_guessed_or_backfilled": False,
            "per_canonical_call_atomic_shards": True,
            "raw_callback_large_arrays_materialized": False,
            "stable_last_index_references_only": True,
            "path_energy_full_force_same_raw_callback": True,
            "additional_physical_calls": 0,
            "optimization_N_F_contribution_from_capture": 0,
        }
        _exclusive_json(audit, audit_payload)
        return manifest, canonical_index, audit


def install_output_observer(
    runner_module,
    recorder: FullStateHistoryRecorder,
    *,
    expected_source_sha256: str = EXPECTED_PRODUCTION_SHA256,
):
    """Wrap only the existing storage-policy call; issue no query."""

    contract = freeze_frame_code_contract(runner_module, expected_source_sha256)
    original = runner_module.should_store_path
    require(
        getattr(original, "__module__", None) == "storage_policy"
        and getattr(original, "__name__", None) == "should_store_path",
        "unexpected original storage-policy callable",
    )

    def observed_should_store_path(
        *, mode, save_all_paths, call, is_initial, reparam_committed
    ):
        current_frame = inspect.currentframe()
        record_frame = current_frame.f_back if current_frame is not None else None
        state_frame = record_frame.f_back if record_frame is not None else None
        try:
            require(
                record_frame is not None
                and record_frame.f_code is contract.record_code,
                "observer was not called by the exact hash-bound record closure",
            )
            require(
                state_frame is not None
                and state_frame.f_code in contract.allowed_state_codes,
                "record was not called by an allowed hash-bound state frame",
            )
            record_local = record_frame.f_locals
            state_local = state_frame.f_locals
            require(
                all(
                    name in record_local
                    for name in (
                        "path",
                        "counters",
                        "counter_snapshot",
                        "history",
                        "diagnostic",
                    )
                ),
                "record frame lacks current counter/history values",
            )
            require(
                all(
                    name in state_local
                    for name in ("path", "energies", "forces", "diagnostic")
                ),
                "state frame lacks current path/energy/full-force values",
            )
            require(
                record_local["path"] is state_local["path"]
                and record_local["diagnostic"] is state_local["diagnostic"],
                "record/state frames do not refer to the same path and diagnostic",
            )
            snapshot = record_local["counter_snapshot"]
            if snapshot is None:
                source_kind = "live_counter_arrays"
            else:
                require(
                    record_local["counters"] is snapshot,
                    "record did not retain the exact historical counter snapshot object",
                )
                source_kind = "historical_counter_snapshot"
            recorder.record(
                call,
                state_local["path"],
                state_local["energies"],
                state_local["forces"],
                record_local["counters"],
                record_local["history"],
                state_local["diagnostic"],
                counter_source_kind=source_kind,
            )
            return original(
                mode=mode,
                save_all_paths=save_all_paths,
                call=call,
                is_initial=is_initial,
                reparam_committed=reparam_committed,
            )
        finally:
            del current_frame, record_frame, state_frame

    runner_module.should_store_path = observed_should_store_path

    def restore() -> None:
        if runner_module.should_store_path is observed_should_store_path:
            runner_module.should_store_path = original

    return restore
