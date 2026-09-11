#!/usr/bin/env python3
"""Durable output-only observer for every selected IR-IVM record call.

The observer is installed around the runner's existing ``should_store_path``
output decision.  It copies the already-current path, energy, and full force
arrays from the enclosing ``record`` frame.  It never calls an evaluator and
its return value is always the original storage-policy return value.
"""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "bb-irivm-full-state-history-recorder-v1"
EXPECTED_ROWS = 352
LAST_COMPLETE_CALL = 318


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        # One uncompressed shard keeps peak RAM and CPU overhead bounded; the
        # shard is the sole authoritative copy of this record's physical data.
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def append_fsync_jsonl(path: Path, payload: dict) -> None:
    existed = path.exists()
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    if not existed:
        fsync_directory(path.parent)


class FullStateHistoryRecorder:
    def __init__(self, root: Path, label: str, registration_path: Path, registration_sha256: str):
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
        append_fsync_jsonl(self.wal, {
            "schema": SCHEMA, "event": "header", "label": self.label,
            "registration": str(self.registration_path),
            "registration_sha256": self.registration_sha256,
            "capture": "already-current path32, energy32, force32; zero evaluator calls",
        })

    def record(self, call: int, path, energies, forces, counters, history, diagnostic) -> None:
        index = len(self.records)
        call = int(call)
        require(index < EXPECTED_ROWS, "too many external records")
        require(call == index + 1, f"expected call {index + 1}, got {call}")
        require(int(history["calls_per_image"][-1]) == call, "observer is not aligned with history append")
        path = np.ascontiguousarray(path, dtype=np.float64)
        energies = np.ascontiguousarray(energies, dtype=np.float64)
        forces = np.ascontiguousarray(forces, dtype=np.float64)
        require(path.shape == forces.shape == (32, 9396), "path/full-force shape mismatch")
        require(energies.shape == (32,), "energy shape mismatch")
        require(np.all(np.isfinite(path)) and np.all(np.isfinite(energies)) and np.all(np.isfinite(forces)), "non-finite full state")
        counter_keys = (
            "joint_evaluations", "energy_requests", "force_requests",
            "cache_hits", "total_query_events", "both_requests",
            "budget_blocked_misses", "inherited_cache_entries",
        )
        counter_arrays = {
            key: np.asarray(counters[key], dtype=np.int64) for key in counter_keys
        }
        require(
            all(value.shape == (32,) for value in counter_arrays.values()),
            "counter shape mismatch",
        )
        counter_vector = counter_arrays["joint_evaluations"]
        optimization_nf = int(counter_vector[1:-1].sum(dtype=np.int64))
        require(optimization_nf == int(history["total_internal_joint_evaluations"][-1]), "observer N_F/history mismatch")
        path_sha = sha256_array(path)
        energy_sha = sha256_array(energies)
        force_sha = sha256_array(forces)
        shard = self.shards / f"record_{index:04d}_call_{call:04d}.npz"
        atomic_npz(
            shard,
            schema=np.asarray(SCHEMA), label=np.asarray(self.label),
            history_index=np.asarray(index), call_per_image=np.asarray(call),
            optimization_N_F=np.asarray(optimization_nf),
            path32_A=path, energy_eV=energies, force_eV_per_A=forces,
            path_sha256=np.asarray(path_sha), energy_sha256=np.asarray(energy_sha),
            force_sha256=np.asarray(force_sha),
            joint_evaluation_vector=counter_vector,
            energy_request_vector=counter_arrays["energy_requests"],
            force_request_vector=counter_arrays["force_requests"],
            cache_hit_vector=counter_arrays["cache_hits"],
            total_query_event_vector=counter_arrays["total_query_events"],
            both_request_vector=counter_arrays["both_requests"],
            budget_blocked_miss_vector=counter_arrays["budget_blocked_misses"],
            inherited_cache_entry_vector=counter_arrays["inherited_cache_entries"],
            discrete_I_perp=np.asarray(float(diagnostic["i_perp"])),
            fully_evaluated_current_path=np.asarray(True),
            provisional_reportable=np.asarray(call <= LAST_COMPLETE_CALL),
            terminal_uncommitted_tail=np.asarray(call > LAST_COMPLETE_CALL),
            additional_physical_calls=np.asarray(0),
        )
        record = {
            "event": "record_committed", "history_index": index,
            "call_per_image": call, "optimization_N_F": optimization_nf,
            "shard": str(shard), "shard_sha256": sha256_file(shard),
            "path_sha256": path_sha, "energy_sha256": energy_sha,
            "force_sha256": force_sha, "additional_physical_calls": 0,
            "discrete_I_perp": float(diagnostic["i_perp"]),
            "formally_reportable": call <= LAST_COMPLETE_CALL,
            "terminal_uncommitted_charged_tail": call > LAST_COMPLETE_CALL,
        }
        append_fsync_jsonl(self.wal, record)
        self.records.append(record)

    def finalize(self, result_path: Path, reference_result_path: Path) -> tuple[Path, Path]:
        result_path = result_path.expanduser().resolve(strict=True)
        reference_result_path = reference_result_path.expanduser().resolve(strict=True)
        require(len(self.records) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} durable records")
        with np.load(result_path, allow_pickle=False) as result, np.load(reference_result_path, allow_pickle=False) as reference:
            stored_calls = np.asarray(result["stored_path_calls"], dtype=np.int64)
            stored_paths = np.asarray(result["stored_paths"], dtype=np.float64)
            require(stored_paths.shape == (stored_calls.size, 32, 9396), "runner committed path aggregate invalid")
            require(stored_calls.size < EXPECTED_ROWS, "runner unexpectedly duplicated all 352 paths")
            reference_calls = np.asarray(reference["stored_path_calls"], dtype=np.int64)
            reference_paths = np.asarray(reference["stored_paths"], dtype=np.float64)
            require(
                stored_calls.shape == reference_calls.shape
                and stored_calls.tobytes() == reference_calls.tobytes(),
                "runtime committed stored-call policy drifted from sealed selected run",
            )
            require(reference_paths.shape == stored_paths.shape, "sealed selected stored-path shape mismatch")
            for index, record in enumerate(self.records):
                shard = Path(record["shard"])
                require(sha256_file(shard) == record["shard_sha256"], "shard drift")
                with np.load(shard, allow_pickle=False) as saved:
                    require(int(np.asarray(saved["history_index"]).item()) == index, "shard history index mismatch")
                    require(int(np.asarray(saved["call_per_image"]).item()) == index + 1, "shard call mismatch")
                    require(str(np.asarray(saved["path_sha256"]).item()) == record["path_sha256"], "shard path SHA metadata mismatch")
                    require(str(np.asarray(saved["energy_sha256"]).item()) == record["energy_sha256"], "shard energy SHA metadata mismatch")
                    require(str(np.asarray(saved["force_sha256"]).item()) == record["force_sha256"], "shard force SHA metadata mismatch")
            for path_index, call in enumerate(stored_calls):
                record = self.records[int(call) - 1]
                require(sha256_array(stored_paths[path_index]) == record["path_sha256"], f"committed result/shard path mismatch at call {call}")
                require(
                    sha256_array(reference_paths[path_index]) == record["path_sha256"],
                    f"sealed selected result/shard path mismatch at call {call}",
                )
        manifest = self.root / "full_state_history.manifest.json"
        manifest_payload = {
            "schema": "bb-irivm-full-state-history-manifest-v1",
            "status": "COMPLETE_PENDING_EXACT_PREFIX_VERIFICATION",
            "label": self.label, "record_count": EXPECTED_ROWS,
            "source_result": str(result_path), "source_result_sha256": sha256_file(result_path),
            "sealed_reference_result": str(reference_result_path),
            "sealed_reference_result_sha256": sha256_file(reference_result_path),
            "authoritative_physical_data": "one atomic shard per external record; no duplicate aggregate",
            "formally_reportable_call_range": [1, LAST_COMPLETE_CALL],
            "terminal_uncommitted_charged_tail_call_range": [LAST_COMPLETE_CALL + 1, EXPECTED_ROWS],
            "additional_physical_calls": 0,
            "records": self.records,
        }
        temporary_manifest = manifest.with_name(manifest.name + ".tmp")
        with temporary_manifest.open("x", encoding="utf-8") as stream:
            json.dump(manifest_payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_manifest, manifest)
        fsync_directory(manifest.parent)
        append_fsync_jsonl(self.wal, {
            "event": "manifest_committed", "record_count": EXPECTED_ROWS,
            "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
            "source_result": str(result_path), "source_result_sha256": sha256_file(result_path),
            "sealed_reference_result": str(reference_result_path),
            "sealed_reference_result_sha256": sha256_file(reference_result_path),
            "additional_physical_calls": 0,
        })
        audit = self.root / "full_state_history.audit.json"
        payload = {
            "schema": "bb-irivm-full-state-history-audit-v1",
            "status": "PASS_STORAGE_ONLY_CAPTURE_PENDING_EXACT_PREFIX_VERIFICATION",
            "label": self.label, "record_count": EXPECTED_ROWS,
            "reportable_record_count": LAST_COMPLETE_CALL,
            "terminal_tail_record_count": EXPECTED_ROWS - LAST_COMPLETE_CALL,
            "source_result": str(result_path), "source_result_sha256": sha256_file(result_path),
            "sealed_reference_result": str(reference_result_path),
            "sealed_reference_result_sha256": sha256_file(reference_result_path),
            "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
            "wal": str(self.wal), "wal_sha256": sha256_file(self.wal),
            "per_record_atomic_shards": True, "all_shards_hash_bound": True,
            "path_energy_full_force_same_record": True,
            "duplicate_large_aggregate_created": False,
            "additional_physical_calls": 0,
        }
        temporary = audit.with_name(audit.name + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, audit)
        fsync_directory(audit.parent)
        return manifest, audit


def install_output_observer(runner_module, recorder: FullStateHistoryRecorder):
    original = runner_module.should_store_path

    def observed_should_store_path(*, mode, save_all_paths, call, is_initial, reparam_committed):
        frame = inspect.currentframe().f_back
        require(frame is not None and frame.f_code.co_name == "record", "observer called outside runner record")
        local = frame.f_locals
        recorder.record(call, local["path"], local["energies"], local["forces"], local["counters"], local["history"], local["diagnostic"])
        return original(mode=mode, save_all_paths=save_all_paths, call=call, is_initial=is_initial, reparam_committed=reparam_committed)

    runner_module.should_store_path = observed_should_store_path

    def restore() -> None:
        if runner_module.should_store_path is observed_should_store_path:
            runner_module.should_store_path = original

    return restore
