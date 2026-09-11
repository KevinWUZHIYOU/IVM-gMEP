#!/usr/bin/env python3
"""Fail-closed, zero-OpenMM verifier for the selected full-state replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


SCHEMA = "bb-irivm-selected-full-state-history-verification-v2"
REGISTRATION_SCHEMA = "bb-irivm-selected-full-state-history-output-only-replay-v2"
AUTHORIZATION_SCHEMA = "bb-irivm-selected-full-state-history-output-only-replay-authorization-v2"
REGISTRATION_STATUS = "FROZEN_BEFORE_FIRST_FULL_STATE_HISTORY_REPLAY_PHYSICAL_CALL"
MANIFEST_SCHEMA = "bb-irivm-full-state-history-manifest-v1"
CAPTURE_AUDIT_SCHEMA = "bb-irivm-full-state-history-audit-v1"
CAPTURE_AUDIT_STATUS = "PASS_STORAGE_ONLY_CAPTURE_PENDING_EXACT_PREFIX_VERIFICATION"
RECORDER_SCHEMA = "bb-irivm-full-state-history-recorder-v1"
LABEL = "bb_irivm_a1e0_r01_m05_inf9990_fullstate_r2"
ROWS = 352
LAST_COMPLETE_CALL = 318
LAST_COMPLETE_NF = 8982
TERMINAL_NF = 9985
TAIL_NF = 1003
EXCLUDED_EXACT_RESULT_KEYS = frozenset({"run_label", "wall_seconds"})
REQUIRED_HISTORY_KEYS = frozenset(
    {
        "calls_per_image",
        "counter_vectors",
        "energy_request_vectors",
        "force_request_vectors",
        "i_perp",
        "reparam_committed",
        "total_internal_joint_evaluations",
    }
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.dtype("<f8")))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def exact(left: np.ndarray, right: np.ndarray) -> bool:
    first = np.asarray(left)
    second = np.asarray(right)
    return (
        first.shape == second.shape
        and first.dtype.str == second.dtype.str
        and np.ascontiguousarray(first).tobytes(order="C")
        == np.ascontiguousarray(second).tobytes(order="C")
    )


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def bind(path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def scalar(saved: Mapping[str, np.ndarray], key: str):
    require(key in saved, f"result lacks scalar {key}")
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def discrete_i_perp(path: np.ndarray, full_force: np.ndarray) -> float:
    """Reproduce the article's discrete central-tangent I_perp exactly."""
    path = np.asarray(path, dtype=np.float64)
    full_force = np.asarray(full_force, dtype=np.float64)
    require(path.shape == full_force.shape == (32, 9396), "path/full-force shape mismatch")
    tangent = path[2:] - path[:-2]
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    require(np.all(norms > 1e-30), "zero central tangent")
    tangent = tangent / norms
    internal_force = full_force[1:-1]
    perpendicular = internal_force - np.sum(
        internal_force * tangent, axis=1, keepdims=True
    ) * tangent
    segment_lengths = np.linalg.norm(path[1:] - path[:-1], axis=1)
    weights = 0.5 * (segment_lengths[:-1] + segment_lengths[1:])
    per_image_l2 = np.linalg.norm(perpendicular, axis=1)
    return float(np.sum(weights * per_image_l2**2))


def write_exclusive(path: Path, payload: dict) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_contracts(
    registration_path: Path,
    authorization_path: Path,
    reference_result_path: Path,
    replay_result_path: Path,
    replay_audit_path: Path,
    manifest_path: Path,
    capture_audit_path: Path,
) -> tuple[dict, dict, dict, dict]:
    registration = read_json(registration_path)
    authorization = read_json(authorization_path)
    replay_audit = read_json(replay_audit_path)
    manifest = read_json(manifest_path)
    capture_audit = read_json(capture_audit_path)
    require(
        registration.get("schema") == REGISTRATION_SCHEMA
        and registration.get("status") == REGISTRATION_STATUS,
        "registration identity/status mismatch",
    )
    require(registration.get("label") == LABEL, "registration label mismatch")
    require(
        authorization.get("schema") == AUTHORIZATION_SCHEMA
        and authorization.get("enabled") is True,
        "authorization invalid",
    )
    require(authorization.get("label") == LABEL, "authorization label mismatch")
    require(
        authorization.get("registration") == bind(registration_path),
        "authorization/registration binding mismatch",
    )
    runtime_hash = hashlib.sha256(
        json.dumps(registration["runtime_argv"], separators=(",", ":")).encode()
    ).hexdigest()
    require(
        authorization.get("runtime_argv_sha256") == runtime_hash,
        "authorization runtime binding mismatch",
    )
    require(
        registration["bindings"]["selected_terminal_result"]
        == bind(reference_result_path),
        "registration/reference result binding mismatch",
    )
    require(
        registration["source_sha256"][Path(__file__).name]
        == sha256_file(Path(__file__).resolve(strict=True)),
        "verifier source drift",
    )
    require(replay_audit.get("consistent") is True, "physical replay audit inconsistent")
    require(
        Path(str(replay_audit.get("result_file", ""))).expanduser().resolve()
        == replay_result_path,
        "physical replay audit/result mismatch",
    )
    require(
        manifest.get("schema") == MANIFEST_SCHEMA
        and manifest.get("status") == "COMPLETE_PENDING_EXACT_PREFIX_VERIFICATION",
        "capture manifest invalid",
    )
    require(
        capture_audit.get("schema") == CAPTURE_AUDIT_SCHEMA
        and capture_audit.get("status") == CAPTURE_AUDIT_STATUS,
        "capture audit invalid",
    )
    require(manifest.get("label") == capture_audit.get("label") == LABEL, "capture label mismatch")
    require(
        manifest.get("source_result") == str(replay_result_path)
        and manifest.get("source_result_sha256") == sha256_file(replay_result_path),
        "manifest/replay result binding mismatch",
    )
    require(
        manifest.get("sealed_reference_result") == str(reference_result_path)
        and manifest.get("sealed_reference_result_sha256") == sha256_file(reference_result_path),
        "manifest/reference result binding mismatch",
    )
    require(
        capture_audit.get("manifest") == str(manifest_path)
        and capture_audit.get("manifest_sha256") == sha256_file(manifest_path),
        "capture audit/manifest binding mismatch",
    )
    require(
        capture_audit.get("source_result") == str(replay_result_path)
        and capture_audit.get("source_result_sha256") == sha256_file(replay_result_path),
        "capture audit/replay result binding mismatch",
    )
    require(
        capture_audit.get("sealed_reference_result") == str(reference_result_path)
        and capture_audit.get("sealed_reference_result_sha256") == sha256_file(reference_result_path),
        "capture audit/reference result binding mismatch",
    )
    require(capture_audit.get("additional_physical_calls") == 0, "capture claims physical calls")
    return registration, replay_audit, manifest, capture_audit


def verify_results(reference_path: Path, replay_path: Path) -> tuple[dict, dict]:
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        replay_path, allow_pickle=False
    ) as replay:
        require(set(reference.files) == set(replay.files), "replay result key set drift")
        exact_keys: list[str] = []
        for key in reference.files:
            if key in EXCLUDED_EXACT_RESULT_KEYS:
                continue
            require(exact(reference[key], replay[key]), f"exact selected replay mismatch: {key}")
            exact_keys.append(key)
        require(not REQUIRED_HISTORY_KEYS.difference(exact_keys), "required exact histories absent")
        calls = np.asarray(replay["calls_per_image"], dtype=np.int64)
        nf = np.asarray(replay["total_internal_joint_evaluations"], dtype=np.int64)
        i_perp = np.asarray(replay["i_perp"], dtype=np.float64)
        barrier_kcal = np.asarray(replay["barrier_kcal"], dtype=np.float64)
        committed = np.asarray(replay["reparam_committed"], dtype=bool)
        counters = np.asarray(replay["counter_vectors"], dtype=np.int64)
        energy_requests = np.asarray(replay["energy_request_vectors"], dtype=np.int64)
        force_requests = np.asarray(replay["force_request_vectors"], dtype=np.int64)
        cache_hits = np.asarray(replay["cache_hit_vectors"], dtype=np.int64)
        require(np.array_equal(calls, np.arange(1, ROWS + 1)), "calls are not 1..352")
        require(nf.shape == i_perp.shape == barrier_kcal.shape == committed.shape == (ROWS,), "history row shape mismatch")
        require(counters.shape == energy_requests.shape == force_requests.shape == cache_hits.shape == (ROWS, 32), "per-image history shape mismatch")
        require(str(scalar(replay, "path_storage_mode")) == "committed", "path storage policy changed")
        require(scalar(replay, "path_storage_affects_online_algorithm") is False, "storage affected algorithm")
        require(int(scalar(replay, "total_internal_nf_last_complete_state_index")) + 1 == LAST_COMPLETE_CALL, "last complete call drift")
        require(int(scalar(replay, "total_internal_nf_last_complete_state_nf")) == LAST_COMPLETE_NF, "last complete N_F drift")
        require(int(scalar(replay, "total_internal_nf_consumed")) == TERMINAL_NF, "terminal N_F drift")
        require(int(scalar(replay, "total_internal_nf_terminal_uncommitted_charged_tail_nf")) == TAIL_NF, "tail N_F drift")
        require(bool(scalar(replay, "total_internal_nf_terminal_partial_tail_present")), "terminal tail absent")
        require(bool(scalar(replay, "total_internal_nf_terminal_partial_tail_quarantined")), "terminal tail not quarantined")
        stored_calls = np.asarray(replay["stored_path_calls"], dtype=np.int64)
        stored_paths = np.asarray(replay["stored_paths"], dtype=np.float64)
        require(np.array_equal(stored_calls, np.asarray([1, 7, 20, 72, 318, 352])), "committed stored-call policy drift")
        require(stored_paths.shape == (6, 32, 9396), "committed stored paths shape drift")
        arrays = {
            "calls": calls,
            "nf": nf,
            "i_perp": i_perp,
            "barrier_kcal": barrier_kcal,
            "committed": committed,
            "counters": counters,
            "energy_requests": energy_requests,
            "force_requests": force_requests,
            "cache_hits": cache_hits,
            "stored_calls": stored_calls,
            "stored_paths": stored_paths,
        }
        summary = {
            "exact_result_array_key_count": len(exact_keys),
            "excluded_nondeterministic_or_output_identity_keys": sorted(EXCLUDED_EXACT_RESULT_KEYS),
            "configuration_sha256": str(scalar(replay, "configuration_sha256")),
            "stored_committed_calls": stored_calls.tolist(),
            "last_complete_call": LAST_COMPLETE_CALL,
            "last_complete_optimization_N_F": LAST_COMPLETE_NF,
            "terminal_call": ROWS,
            "terminal_optimization_N_F": TERMINAL_NF,
            "terminal_uncommitted_charged_tail_N_F": TAIL_NF,
        }
    return summary, arrays


def read_wal(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            require(line.endswith("\n"), f"WAL line {line_number} is torn")
            value = json.loads(line)
            require(isinstance(value, dict), f"WAL line {line_number} is not an object")
            rows.append(value)
    return rows


def verify_shards(
    registration_path: Path,
    manifest_path: Path,
    capture_audit: dict,
    manifest: dict,
    arrays: dict,
) -> tuple[list[dict], dict]:
    records = manifest.get("records")
    require(isinstance(records, list) and len(records) == ROWS, "manifest must contain 352 records")
    wal_path = Path(str(capture_audit.get("wal", ""))).expanduser().resolve(strict=True)
    require(sha256_file(wal_path) == capture_audit.get("wal_sha256"), "capture WAL hash drift")
    wal = read_wal(wal_path)
    require(len(wal) == ROWS + 2, "WAL must contain header, 352 records, and manifest commit")
    require(
        wal[0].get("schema") == RECORDER_SCHEMA
        and wal[0].get("event") == "header"
        and wal[0].get("label") == LABEL,
        "WAL header invalid",
    )
    require(wal[0].get("registration") == str(registration_path), "WAL registration path mismatch")
    require(wal[0].get("registration_sha256") == sha256_file(registration_path), "WAL registration hash mismatch")
    inventory: list[dict] = []
    stored_index = {int(call): index for index, call in enumerate(arrays["stored_calls"])}
    for index, record in enumerate(records):
        call = index + 1
        require(record == wal[index + 1], f"manifest/WAL mismatch at call {call}")
        require(record.get("event") == "record_committed", f"bad WAL event at call {call}")
        require(record.get("history_index") == index and record.get("call_per_image") == call, f"record ordering mismatch at call {call}")
        shard_path = Path(str(record.get("shard", ""))).expanduser().resolve(strict=True)
        require(shard_path.parent == manifest_path.parent / "shards", f"shard outside authoritative directory at call {call}")
        require(sha256_file(shard_path) == record.get("shard_sha256"), f"shard file hash drift at call {call}")
        with np.load(shard_path, allow_pickle=False) as shard:
            path = np.asarray(shard["path32_A"], dtype=np.float64)
            energy = np.asarray(shard["energy_eV"], dtype=np.float64)
            force = np.asarray(shard["force_eV_per_A"], dtype=np.float64)
            require(path.shape == force.shape == (32, 9396) and energy.shape == (32,), f"full-state shape mismatch at call {call}")
            require(np.all(np.isfinite(path)) and np.all(np.isfinite(energy)) and np.all(np.isfinite(force)), f"non-finite full state at call {call}")
            require(str(scalar(shard, "schema")) == RECORDER_SCHEMA, f"shard schema mismatch at call {call}")
            require(str(scalar(shard, "label")) == LABEL, f"shard label mismatch at call {call}")
            require(int(scalar(shard, "history_index")) == index and int(scalar(shard, "call_per_image")) == call, f"shard index mismatch at call {call}")
            require(int(scalar(shard, "optimization_N_F")) == int(arrays["nf"][index]), f"shard N_F mismatch at call {call}")
            path_sha = sha256_array(path)
            energy_sha = sha256_array(energy)
            force_sha = sha256_array(force)
            require(path_sha == record.get("path_sha256") == str(scalar(shard, "path_sha256")), f"path hash mismatch at call {call}")
            require(energy_sha == record.get("energy_sha256") == str(scalar(shard, "energy_sha256")), f"energy hash mismatch at call {call}")
            require(force_sha == record.get("force_sha256") == str(scalar(shard, "force_sha256")), f"force hash mismatch at call {call}")
            require(exact(shard["joint_evaluation_vector"], arrays["counters"][index]), f"joint counter mismatch at call {call}")
            require(exact(shard["energy_request_vector"], arrays["energy_requests"][index]), f"energy request mismatch at call {call}")
            require(exact(shard["force_request_vector"], arrays["force_requests"][index]), f"force request mismatch at call {call}")
            require(exact(shard["cache_hit_vector"], arrays["cache_hits"][index]), f"cache hit mismatch at call {call}")
            total_query = np.asarray(shard["total_query_event_vector"], dtype=np.int64)
            both = np.asarray(shard["both_request_vector"], dtype=np.int64)
            blocked = np.asarray(shard["budget_blocked_miss_vector"], dtype=np.int64)
            require(np.array_equal(arrays["energy_requests"][index] + arrays["force_requests"][index] - both, total_query), f"request identity failed at call {call}")
            require(np.array_equal(arrays["cache_hits"][index] + arrays["counters"][index] + blocked, total_query), f"counter outcome identity failed at call {call}")
            measured_i_perp = discrete_i_perp(path, force)
            require(np.asarray(measured_i_perp).tobytes() == np.asarray(arrays["i_perp"][index]).tobytes(), f"discrete I_perp is not exact at call {call}")
            require(np.asarray(measured_i_perp).tobytes() == np.asarray(scalar(shard, "discrete_I_perp")).tobytes(), f"shard I_perp metadata mismatch at call {call}")
            measured_barrier = float((np.max(energy) - energy[0]) * 23.0605)
            require(np.asarray(measured_barrier).tobytes() == np.asarray(arrays["barrier_kcal"][index]).tobytes(), f"sparse barrier is not exact at call {call}")
            reportable = call <= LAST_COMPLETE_CALL
            require(bool(scalar(shard, "fully_evaluated_current_path")), f"path not fully evaluated at call {call}")
            require(bool(scalar(shard, "provisional_reportable")) is reportable, f"reportability mask mismatch at call {call}")
            require(bool(scalar(shard, "terminal_uncommitted_tail")) is (not reportable), f"tail mask mismatch at call {call}")
            require(int(scalar(shard, "additional_physical_calls")) == 0 and record.get("additional_physical_calls") == 0, f"observer physical-call count nonzero at call {call}")
            if call in stored_index:
                require(exact(path, arrays["stored_paths"][stored_index[call]]), f"committed result/shard path mismatch at call {call}")
        inventory.append(
            {
                "history_index": index,
                "call_per_image": call,
                "optimization_N_F": int(arrays["nf"][index]),
                "discrete_I_perp_eV2_per_A": float(arrays["i_perp"][index]),
                "path_sha256": record["path_sha256"],
                "energy_sha256": record["energy_sha256"],
                "force_sha256": record["force_sha256"],
                "shard": str(shard_path),
                "shard_sha256": record["shard_sha256"],
                "fully_evaluated_current_path": True,
                "formally_reportable": call <= LAST_COMPLETE_CALL,
                "terminal_uncommitted_charged_tail": call > LAST_COMPLETE_CALL,
                "exclusion_reason": None if call <= LAST_COMPLETE_CALL else "after_last_complete_state_in_terminal_uncommitted_transaction",
            }
        )
    require(
        wal[-1].get("event") == "manifest_committed"
        and wal[-1].get("record_count") == ROWS
        and wal[-1].get("manifest") == str(manifest_path)
        and wal[-1].get("manifest_sha256") == sha256_file(manifest_path),
        "WAL manifest commit invalid",
    )
    summary = {
        "record_count": ROWS,
        "reportable_record_count": LAST_COMPLETE_CALL,
        "quarantined_terminal_tail_record_count": ROWS - LAST_COMPLETE_CALL,
        "one_authoritative_atomic_shard_per_record": True,
        "duplicate_large_aggregate_created": False,
        "path_energy_full_force_same_record": True,
        "all_discrete_I_perp_values_exactly_reconstructed": True,
        "additional_physical_calls_from_capture": 0,
        "wal": bind(wal_path),
    }
    return inventory, summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--replay-result", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--capture-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    paths = {
        name: value.expanduser().resolve(strict=True)
        for name, value in vars(args).items()
        if name != "output"
    }
    _, _, manifest, capture_audit = validate_contracts(
        paths["registration"], paths["authorization"],
        paths["reference_result"], paths["replay_result"],
        paths["replay_audit"], paths["capture_manifest"],
        paths["capture_audit"],
    )
    result_summary, arrays = verify_results(
        paths["reference_result"], paths["replay_result"]
    )
    inventory, capture_summary = verify_shards(
        paths["registration"], paths["capture_manifest"],
        capture_audit, manifest, arrays,
    )
    payload = {
        "schema": SCHEMA,
        "status": "PASS_EXACT_FULL_STATE_HISTORY_OUTPUT_ONLY_REPLAY",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scientific_scope": "all 352 fully evaluated current paths with their already-current per-image energies and full forces are retained; calls 1..318 are reportable and calls 319..352 remain a quarantined charged terminal tail",
        "tolerance_fallback_used": False,
        **result_summary,
        **capture_summary,
        "path_inventory": inventory,
        "bindings": {name: bind(path) for name, path in paths.items()},
    }
    write_exclusive(args.output, payload)
    destination = args.output.expanduser().resolve(strict=True)
    print(destination)
    print(sha256_file(destination))


if __name__ == "__main__":
    main()
