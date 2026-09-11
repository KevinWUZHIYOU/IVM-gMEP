#!/usr/bin/env python3
"""Schema-aware, output-only full-state observer for selected BB IR-IVM R4.

The unchanged production ``record`` closure legitimately receives either the
live eight-field ``bb_core.counter_arrays`` mapping or a seven-field
historical snapshot emitted by ``frozen_discrete_b040``.  R4 stores exactly
what existed at the snapshot time.  In particular, a seven-field snapshot
does not receive a guessed or later-live ``inherited_cache_entries`` value.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
from types import CodeType

import numpy as np

from full_state_history_recorder import (
    append_fsync_jsonl,
    atomic_npz,
    fsync_directory,
    require,
    sha256_array,
    sha256_file,
)


SCHEMA = "bb-irivm-full-state-history-recorder-v2"
MANIFEST_SCHEMA = "bb-irivm-full-state-history-manifest-v2"
AUDIT_SCHEMA = "bb-irivm-full-state-history-audit-v2"
AUDIT_STATUS = "PASS_SCHEMA_AWARE_STORAGE_ONLY_CAPTURE_PENDING_EXACT_PREFIX_VERIFICATION"
EXPECTED_ROWS = 352
LAST_COMPLETE_CALL = 318
EXPECTED_PRODUCTION_SHA256 = (
    "991c11ff1f8eab014a6e54b4bf8ba610b0e52d071b202dc131b03bfabe6143fa"
)
ALLOWED_STATE_FRAME_NAMES = frozenset({"run", "attempt_reparameterization"})
COMMON_COUNTER_KEYS = (
    "joint_evaluations",
    "energy_requests",
    "force_requests",
    "cache_hits",
    "total_query_events",
    "both_requests",
    "budget_blocked_misses",
)
INHERITED_COUNTER_KEY = "inherited_cache_entries"
HISTORICAL_COUNTER_KEYS = frozenset(COMMON_COUNTER_KEYS)
LIVE_COUNTER_KEYS = frozenset((*COMMON_COUNTER_KEYS, INHERITED_COUNTER_KEY))


@dataclass(frozen=True)
class FrameCodeContract:
    source_path: Path
    source_sha256: str
    record_code: CodeType
    allowed_state_codes: frozenset[CodeType]


def nested_codes(code: CodeType):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            yield from nested_codes(constant)


def freeze_frame_code_contract(
    runner_module,
    expected_source_sha256: str = EXPECTED_PRODUCTION_SHA256,
) -> FrameCodeContract:
    source_path = Path(runner_module.__file__).expanduser().resolve(strict=True)
    observed_sha256 = sha256_file(source_path)
    require(
        observed_sha256 == str(expected_source_sha256),
        "production runner source hash differs from the R4 frame contract",
    )
    run_function = getattr(runner_module, "run", None)
    require(
        callable(run_function) and isinstance(run_function.__code__, CodeType),
        "runner lacks run code",
    )
    codes = list(nested_codes(run_function.__code__))
    record_codes = [code for code in codes if code.co_name == "record"]
    require(len(record_codes) == 1, "production runner must contain exactly one record closure")
    state_codes = frozenset(
        code for code in codes if code.co_name in ALLOWED_STATE_FRAME_NAMES
    )
    require(
        {code.co_name for code in state_codes} == ALLOWED_STATE_FRAME_NAMES,
        "production runner state-frame inventory drift",
    )
    require(
        all(
            Path(code.co_filename).expanduser().resolve() == source_path
            for code in (record_codes[0], *state_codes)
        ),
        "frame code does not originate from the hash-bound production source",
    )
    return FrameCodeContract(
        source_path=source_path,
        source_sha256=observed_sha256,
        record_code=record_codes[0],
        allowed_state_codes=state_codes,
    )


def _exclusive_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
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


def _counter_arrays(counters, source_kind: str) -> tuple[dict[str, np.ndarray], bool]:
    require(isinstance(counters, dict), "counter snapshot is not a dictionary")
    observed_keys = frozenset(counters)
    if source_kind == "live_counter_arrays":
        require(observed_keys == LIVE_COUNTER_KEYS, "live counter mapping is not exact schema-8")
    else:
        require(
            source_kind == "historical_counter_snapshot",
            "unknown counter source kind",
        )
        require(
            observed_keys in (HISTORICAL_COUNTER_KEYS, LIVE_COUNTER_KEYS),
            "historical counter mapping is neither exact schema-7 nor exact schema-8",
        )
    arrays: dict[str, np.ndarray] = {}
    for key in observed_keys:
        raw = np.asarray(counters[key])
        require(raw.shape == (32,), f"counter shape mismatch: {key}")
        require(raw.dtype.kind in "iu", f"counter dtype is not integer: {key}")
        require(np.all(raw >= 0), f"negative counter value: {key}")
        if raw.dtype.kind == "u":
            require(
                int(np.max(raw, initial=0)) <= np.iinfo(np.int64).max,
                f"counter exceeds int64: {key}",
            )
        arrays[key] = np.ascontiguousarray(raw, dtype=np.int64)
    return arrays, INHERITED_COUNTER_KEY in observed_keys


class FullStateHistoryRecorder:
    """One durable path/E/F shard for every production record."""

    def __init__(
        self,
        root: Path,
        label: str,
        registration_path: Path,
        registration_sha256: str,
    ):
        self.root = root.expanduser().resolve()
        require(not self.root.exists(), f"refusing to overwrite observer directory: {self.root}")
        self.shards = self.root / "shards"
        self.shards.mkdir(parents=True)
        fsync_directory(self.root)
        fsync_directory(self.root.parent)
        self.wal = self.root / "full_state_history.wal.jsonl"
        self.label = str(label)
        self.registration_path = registration_path.expanduser().resolve(strict=True)
        self.registration_sha256 = str(registration_sha256)
        self.records: list[dict] = []
        append_fsync_jsonl(
            self.wal,
            {
                "schema": SCHEMA,
                "event": "header",
                "label": self.label,
                "registration": str(self.registration_path),
                "registration_sha256": self.registration_sha256,
                "capture": "already-current path32, energy32, force32; exact schema-7/8 counters; zero evaluator calls",
                "missing_counter_policy": "explicitly unavailable; never guessed or backfilled",
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
        index = len(self.records)
        call = int(call)
        require(index < EXPECTED_ROWS, "too many external records")
        require(call == index + 1, f"expected call {index + 1}, got {call}")
        require(
            int(history["calls_per_image"][-1]) == call,
            "observer is not aligned with history append",
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
        counter_arrays, inherited_present = _counter_arrays(counters, counter_source_kind)
        counter_vector = counter_arrays["joint_evaluations"]
        optimization_nf = int(counter_vector[1:-1].sum(dtype=np.int64))
        require(
            optimization_nf
            == int(history["total_internal_joint_evaluations"][-1]),
            "observer N_F/history mismatch",
        )
        schema_keys = [key for key in (*COMMON_COUNTER_KEYS, INHERITED_COUNTER_KEY) if key in counter_arrays]
        unavailable = [] if inherited_present else [INHERITED_COUNTER_KEY]
        path_sha = sha256_array(path)
        energy_sha = sha256_array(energies)
        force_sha = sha256_array(forces)
        shard = self.shards / f"record_{index:04d}_call_{call:04d}.npz"
        arrays = {
            "schema": np.asarray(SCHEMA),
            "label": np.asarray(self.label),
            "history_index": np.asarray(index),
            "call_per_image": np.asarray(call),
            "optimization_N_F": np.asarray(optimization_nf),
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
            "joint_evaluation_vector": counter_vector,
            "energy_request_vector": counter_arrays["energy_requests"],
            "force_request_vector": counter_arrays["force_requests"],
            "cache_hit_vector": counter_arrays["cache_hits"],
            "total_query_event_vector": counter_arrays["total_query_events"],
            "both_request_vector": counter_arrays["both_requests"],
            "budget_blocked_miss_vector": counter_arrays["budget_blocked_misses"],
            "discrete_I_perp": np.asarray(float(diagnostic["i_perp"])),
            "fully_evaluated_current_path": np.asarray(True),
            "provisional_reportable": np.asarray(call <= LAST_COMPLETE_CALL),
            "terminal_uncommitted_tail": np.asarray(call > LAST_COMPLETE_CALL),
            "additional_physical_calls": np.asarray(0),
        }
        if inherited_present:
            expected_inherited = np.zeros(32, dtype=np.int64)
            expected_inherited[[0, -1]] = 1
            require(
                np.array_equal(
                    counter_arrays[INHERITED_COUNTER_KEY], expected_inherited
                ),
                "present inherited-cache vector differs from the frozen endpoint-only preload semantics",
            )
            arrays["inherited_cache_entry_vector"] = counter_arrays[INHERITED_COUNTER_KEY]
        atomic_npz(shard, **arrays)
        record = {
            "event": "record_committed",
            "history_index": index,
            "call_per_image": call,
            "optimization_N_F": optimization_nf,
            "shard": str(shard),
            "shard_sha256": sha256_file(shard),
            "path_sha256": path_sha,
            "energy_sha256": energy_sha,
            "force_sha256": force_sha,
            "counter_source_kind": counter_source_kind,
            "counter_schema_keys": schema_keys,
            "unavailable_counter_fields": unavailable,
            "inherited_cache_entries_present": inherited_present,
            "additional_physical_calls": 0,
            "discrete_I_perp": float(diagnostic["i_perp"]),
            "formally_reportable": call <= LAST_COMPLETE_CALL,
            "terminal_uncommitted_charged_tail": call > LAST_COMPLETE_CALL,
        }
        append_fsync_jsonl(self.wal, record)
        self.records.append(record)

    def finalize(
        self, result_path: Path, reference_result_path: Path
    ) -> tuple[Path, Path]:
        result_path = result_path.expanduser().resolve(strict=True)
        reference_result_path = reference_result_path.expanduser().resolve(strict=True)
        require(len(self.records) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} durable records")
        schema_counts = {
            "live_schema8": 0,
            "historical_schema7": 0,
            "historical_schema8": 0,
        }
        with np.load(result_path, allow_pickle=False) as result, np.load(
            reference_result_path, allow_pickle=False
        ) as reference:
            stored_calls = np.asarray(result["stored_path_calls"], dtype=np.int64)
            stored_paths = np.asarray(result["stored_paths"], dtype=np.float64)
            reference_calls = np.asarray(reference["stored_path_calls"], dtype=np.int64)
            reference_paths = np.asarray(reference["stored_paths"], dtype=np.float64)
            require(stored_paths.shape == (stored_calls.size, 32, 9396), "runtime stored-path shape invalid")
            require(stored_calls.size < EXPECTED_ROWS, "runtime duplicated all paths")
            require(np.array_equal(stored_calls, reference_calls), "committed stored-call policy drift")
            require(reference_paths.shape == stored_paths.shape, "reference stored-path shape mismatch")
            for index, record in enumerate(self.records):
                shard = Path(record["shard"])
                require(sha256_file(shard) == record["shard_sha256"], "shard drift")
                with np.load(shard, allow_pickle=False) as saved:
                    require(str(np.asarray(saved["schema"]).item()) == SCHEMA, "shard schema mismatch")
                    require(int(np.asarray(saved["history_index"]).item()) == index, "shard history index mismatch")
                    require(int(np.asarray(saved["call_per_image"]).item()) == index + 1, "shard call mismatch")
                    source = str(np.asarray(saved["counter_source_kind"]).item())
                    present = bool(np.asarray(saved["inherited_cache_entries_present"]).item())
                    schema_keys_raw = np.asarray(saved["counter_schema_keys"])
                    unavailable_raw = np.asarray(saved["unavailable_counter_fields"])
                    require(
                        schema_keys_raw.ndim == unavailable_raw.ndim == 1
                        and schema_keys_raw.dtype.kind in "US"
                        and unavailable_raw.dtype.kind in "US",
                        "counter schema metadata dtype/shape mismatch",
                    )
                    schema_keys = schema_keys_raw.astype(str).tolist()
                    unavailable = unavailable_raw.astype(str).tolist()
                    expected_seven = list(COMMON_COUNTER_KEYS)
                    expected_eight = [*COMMON_COUNTER_KEYS, INHERITED_COUNTER_KEY]
                    require(
                        record["counter_source_kind"] == source
                        and record["counter_schema_keys"] == schema_keys
                        and record["unavailable_counter_fields"] == unavailable
                        and record["inherited_cache_entries_present"] is present,
                        "WAL/shard counter metadata mismatch",
                    )
                    if source == "live_counter_arrays":
                        require(
                            present and schema_keys == expected_eight and unavailable == [],
                            "live counter record is not exact schema-8",
                        )
                        schema_counts["live_schema8"] += 1
                    else:
                        require(source == "historical_counter_snapshot", "unknown counter source kind")
                        if present:
                            require(schema_keys == expected_eight and unavailable == [], "historical schema-8 metadata mismatch")
                            schema_counts["historical_schema8"] += 1
                        else:
                            require(schema_keys == expected_seven and unavailable == [INHERITED_COUNTER_KEY], "historical schema-7 metadata mismatch")
                            require("inherited_cache_entry_vector" not in saved.files, "schema-7 shard stores unavailable inherited values")
                            schema_counts["historical_schema7"] += 1
                    for field in schema_keys:
                        vector_key = {
                            "joint_evaluations": "joint_evaluation_vector",
                            "energy_requests": "energy_request_vector",
                            "force_requests": "force_request_vector",
                            "cache_hits": "cache_hit_vector",
                            "total_query_events": "total_query_event_vector",
                            "both_requests": "both_request_vector",
                            "budget_blocked_misses": "budget_blocked_miss_vector",
                            "inherited_cache_entries": "inherited_cache_entry_vector",
                        }[field]
                        vector = np.asarray(saved[vector_key])
                        require(
                            vector.shape == (32,)
                            and vector.dtype.kind in "iu"
                            and np.all(vector >= 0),
                            f"invalid stored counter vector: {field}",
                        )
                    joint = np.asarray(saved["joint_evaluation_vector"], dtype=np.int64)
                    stored_nf = int(np.asarray(saved["optimization_N_F"]).item())
                    require(
                        stored_nf == record["optimization_N_F"]
                        == int(np.sum(joint[1:-1], dtype=np.int64)),
                        "stored counter/N_F closure mismatch",
                    )
            for path_index, call_value in enumerate(stored_calls):
                call = int(call_value)
                record = self.records[call - 1]
                require(sha256_array(stored_paths[path_index]) == record["path_sha256"], f"runtime result/shard path mismatch at call {call}")
                require(sha256_array(reference_paths[path_index]) == record["path_sha256"], f"reference result/shard path mismatch at call {call}")
        manifest = self.root / "full_state_history.manifest.json"
        manifest_payload = {
            "schema": MANIFEST_SCHEMA,
            "status": "COMPLETE_PENDING_EXACT_PREFIX_VERIFICATION",
            "label": self.label,
            "record_count": EXPECTED_ROWS,
            "source_result": str(result_path),
            "source_result_sha256": sha256_file(result_path),
            "sealed_reference_result": str(reference_result_path),
            "sealed_reference_result_sha256": sha256_file(reference_result_path),
            "authoritative_physical_data": "one atomic shard per external record; no duplicate aggregate",
            "counter_schema_policy": "exact live-8 or historical-7/8; absent inherited values remain unavailable",
            "counter_schema_counts": schema_counts,
            "formally_reportable_call_range": [1, LAST_COMPLETE_CALL],
            "terminal_uncommitted_charged_tail_call_range": [LAST_COMPLETE_CALL + 1, EXPECTED_ROWS],
            "additional_physical_calls": 0,
            "records": self.records,
        }
        _exclusive_json(manifest, manifest_payload)
        append_fsync_jsonl(
            self.wal,
            {
                "event": "manifest_committed",
                "record_count": EXPECTED_ROWS,
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "source_result": str(result_path),
                "source_result_sha256": sha256_file(result_path),
                "sealed_reference_result": str(reference_result_path),
                "sealed_reference_result_sha256": sha256_file(reference_result_path),
                "additional_physical_calls": 0,
            },
        )
        audit = self.root / "full_state_history.audit.json"
        payload = {
            "schema": AUDIT_SCHEMA,
            "status": AUDIT_STATUS,
            "label": self.label,
            "record_count": EXPECTED_ROWS,
            "reportable_record_count": LAST_COMPLETE_CALL,
            "terminal_tail_record_count": EXPECTED_ROWS - LAST_COMPLETE_CALL,
            "source_result": str(result_path),
            "source_result_sha256": sha256_file(result_path),
            "sealed_reference_result": str(reference_result_path),
            "sealed_reference_result_sha256": sha256_file(reference_result_path),
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "wal": str(self.wal),
            "wal_sha256": sha256_file(self.wal),
            "counter_schema_counts": schema_counts,
            "missing_counter_values_guessed_or_backfilled": False,
            "per_record_atomic_shards": True,
            "all_shards_hash_bound": True,
            "path_energy_full_force_same_record": True,
            "duplicate_large_aggregate_created": False,
            "additional_physical_calls": 0
        }
        _exclusive_json(audit, payload)
        return manifest, audit


def install_output_observer(
    runner_module,
    recorder: FullStateHistoryRecorder,
    *,
    expected_source_sha256: str = EXPECTED_PRODUCTION_SHA256,
):
    """Install the exact frame-bound, zero-query observer."""

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
                record_frame is not None and record_frame.f_code is contract.record_code,
                "observer was not called by the exact hash-bound record closure",
            )
            require(
                state_frame is not None and state_frame.f_code in contract.allowed_state_codes,
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
