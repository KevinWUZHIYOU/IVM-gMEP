#!/usr/bin/env python3
"""Freeze, authorize, preflight, or execute schema-aware full-state replay R4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable

import run_selected_ivm_full_history_replay as PATH_ONLY
from full_state_history_recorder_r4 import (
    COMMON_COUNTER_KEYS,
    HISTORICAL_COUNTER_KEYS,
    INHERITED_COUNTER_KEY,
    LIVE_COUNTER_KEYS,
    FullStateHistoryRecorder,
    install_output_observer,
)


SCHEMA = "bb-irivm-selected-full-state-history-output-only-replay-v4"
AUTH_SCHEMA = "bb-irivm-selected-full-state-history-output-only-replay-authorization-v4"
STATUS = "FROZEN_BEFORE_FIRST_R4_FULL_STATE_HISTORY_REPLAY_PHYSICAL_CALL"
LABEL = "bb_irivm_a1e0_r01_m05_inf9990_fullstate_r4"
OUTPUT_SUBDIR = "runs/full_state_history_replay_r4"
OBSERVER_NAME = LABEL + "_observer"
MINIMUM_FREE_BYTES = 6 * 1024**3
R3_FAILURE_RECEIPT = (
    "registrations/diagnostics/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_R3_FAILURE_SUPERSESSION_20260902.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def replay_argv(reference: dict) -> list[str]:
    argv = [str(value) for value in reference["command"]]
    PATH_ONLY.require(PATH_ONLY.option_value(argv, "--run-label") == PATH_ONLY.REFERENCE_LABEL, "reference label drift")
    PATH_ONLY.require(PATH_ONLY.option_value(argv, "--output-dir") == "runs/production", "reference output directory drift")
    PATH_ONLY.require(PATH_ONLY.option_value(argv, "--budget") == "9990", "reference safety budget drift")
    PATH_ONLY.require(PATH_ONLY.option_value(argv, "--total-internal-nf-budget") == "9990", "reference total budget drift")
    PATH_ONLY.require(PATH_ONLY.option_value(argv, "--path-storage-mode") == "committed", "reference storage drift")
    PATH_ONLY.require("--save-all-paths" not in argv, "reference unexpectedly saves all paths")
    PATH_ONLY.replace_option(argv, "--run-label", LABEL)
    PATH_ONLY.replace_option(argv, "--output-dir", OUTPUT_SUBDIR)
    return argv


def validate_runtime_delta(reference: list[str], replay: list[str]) -> None:
    PATH_ONLY.require(reference.count("--save-all-paths") == replay.count("--save-all-paths") == 0, "save-all must remain disabled")
    PATH_ONLY.require(len(reference) == len(replay), "runtime length drift")
    expected = {reference.index(flag) + 1 for flag in ("--run-label", "--output-dir")}
    observed = {index for index, pair in enumerate(zip(reference, replay)) if pair[0] != pair[1]}
    PATH_ONLY.require(observed == expected, f"non-output runtime drift: {observed}")
    PATH_ONLY.require(PATH_ONLY.option_value(replay, "--run-label") == LABEL, "label mismatch")
    PATH_ONLY.require(PATH_ONLY.option_value(replay, "--output-dir") == OUTPUT_SUBDIR, "output mismatch")
    for flag, expected_value in (
        ("--path-storage-mode", "committed"),
        ("--budget", "9990"),
        ("--total-internal-nf-budget", "9990"),
        ("--alpha", "1"),
        ("--discrete-residual-ratio", "0.1"),
        ("--discrete-inner-max-steps", "5"),
        ("--discrete-step-multiplier-eta", "1.1"),
        ("--max-step", "1"),
        ("--reparam-mode", "state_curve"),
        ("--reparam-every", "1"),
    ):
        PATH_ONLY.require(PATH_ONLY.option_value(replay, flag) == expected_value, f"algorithm drift: {flag}")


def source_paths(root: Path) -> dict[str, Path]:
    result = {name: root / "src" / name for name in PATH_ONLY.ALGORITHM_SOURCES}
    for name in (
        Path(__file__).name,
        "run_selected_ivm_full_history_replay.py",
        "full_state_history_recorder.py",
        "full_state_history_recorder_r4.py",
        "verify_selected_ivm_full_state_history_replay.py",
        "verify_selected_ivm_full_state_history_replay_r4.py",
    ):
        result[name] = root / "scripts" / name
    return result


def freeze(root: Path, output: Path) -> None:
    paths = PATH_ONLY.reference_paths(root)
    paths["R3_failure_supersession_receipt"] = root / R3_FAILURE_RECEIPT
    reference = PATH_ONLY.read_json(paths["selected_member_registration"])
    PATH_ONLY.require(set(reference.get("source_sha256", {})) == set(PATH_ONLY.ALGORITHM_SOURCES), "selected registration source inventory drift")
    for name in PATH_ONLY.ALGORITHM_SOURCES:
        PATH_ONLY.require(PATH_ONLY.sha256_file(root / "src" / name) == reference["source_sha256"][name], f"current algorithm source differs from selected registration: {name}")
    original = [str(value) for value in reference["command"]]
    runtime = replay_argv(reference)
    validate_runtime_delta(original, runtime)
    for role, path in {**paths, **source_paths(root)}.items():
        PATH_ONLY.require(path.is_file(), f"missing frozen source/input {role}: {path}")
    failure = PATH_ONLY.read_json(paths["R3_failure_supersession_receipt"])
    PATH_ONLY.require(
        failure.get("status") == "R3_FAILED_AT_FIRST_DISCRETE_TRIAL_RECORD_SUPERSEDED_NO_REUSE"
        and int(failure["failure"]["lifetime_optimization_internal_joint_EF_attempts"]) == 150,
        "R3 failure receipt invalid",
    )
    history = PATH_ONLY.sealed_history_facts(paths["selected_terminal_result"])
    history["authoritative_shard_calls_required"] = history.pop("stored_calls_required")
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "created_utc": utc_now(),
        "execution_authorized_by_this_file": False,
        "purpose": "one fresh exact selected IR-IVM replay retaining same-record path, energy, force, and every actually available counter field",
        "label": LABEL,
        "reference_label": PATH_ONLY.REFERENCE_LABEL,
        "R2_and_R3_reuse_forbidden": True,
        "runtime_argv": runtime,
        "only_permitted_runtime_argv_differences": {"--run-label": LABEL, "--output-dir": OUTPUT_SUBDIR},
        "algorithm_identity": {
            "optimization_N_F_budget": 9990,
            "all_numerical_controls_unchanged": True,
            "all_algorithm_source_hashes_exactly_match_selected_registration": True,
            "no_algorithm_or_solver_snapshot_source_modified": True,
            "alpha": 1.0,
            "r": 0.1,
            "m_max": 5,
            "eta": 1.1,
            "reparameterization": "unchanged state_curve direct/recovery transaction",
            "fresh_from_hash_bound_common_initial_path": True,
            "warm_start": False
        },
        "schema_aware_output_only_capture": {
            "R3_failure_fixed": "accept exactly the two genuine production counter schemas without inventing the unavailable eighth historical field",
            "live_counter_arrays_exact_keys": list((*COMMON_COUNTER_KEYS, INHERITED_COUNTER_KEY)),
            "historical_counter_snapshot_permitted_exact_key_sets": [list(COMMON_COUNTER_KEYS), list((*COMMON_COUNTER_KEYS, INHERITED_COUNTER_KEY))],
            "schema7_missing_field": INHERITED_COUNTER_KEY,
            "schema7_storage": "numeric inherited field absent; presence=false; unavailable_counter_fields records the omission",
            "guess_or_later_live_backfill": "forbidden",
            "optimization_N_F_source": "the actual historical or live joint_evaluations vector stored in that shard",
            "every_record_physical_state": ["path32_A", "energy_eV32", "force_eV_per_A32"],
            "new_energy_or_force_evaluator_calls": 0,
            "atomic_per_record_uncompressed_npz_shard": True,
            "fsync_shard_directory_before_fsync_append_only_WAL_commit": True,
            "terminal_aggregate": "small hash-bound manifest/index only; no duplicate physical-array aggregate",
            "expected_records": 352,
            **history,
            "calls_319_through_352_stored_but_formally_quarantined": True,
            "postrun_exact_prefix_and_array_verification_required": True
        },
        "resource_estimate": {
            "reference_M4_wall_seconds": 11286.506678819656,
            "expected_M4_wall_time": "3.3--3.8 hours including durable per-record shards",
            "exact_raw_path_energy_force_shard_payload_bytes": 1693474816,
            "peak_incremental_disk_GiB": "1.7--2.0",
            "estimated_incremental_peak_RAM_GiB": "<0.10",
            "minimum_free_disk_bytes_before_child": MINIMUM_FREE_BYTES
        },
        "concurrency": {
            "one_BB_task_on_host": True,
            "jobs": 1,
            "all_thread_caps": 1,
            "two_host_local_fcntl_locks_required_before_child": True,
            "minimum_free_disk_bytes_before_child": MINIMUM_FREE_BYTES,
            "dedicated_Mutagen_session_must_be_connected_and_flushed": "bb-irivm-20260901"
        },
        "bindings": {role: PATH_ONLY.bind(path) for role, path in paths.items()},
        "source_sha256": {name: PATH_ONLY.sha256_file(path) for name, path in source_paths(root).items()}
    }
    PATH_ONLY.write_exclusive(output, payload)
    resolved = output.expanduser().resolve(strict=True)
    print(resolved)
    print(PATH_ONLY.sha256_file(resolved))


def validate_registration(root: Path, registration_path: Path) -> dict:
    registration = PATH_ONLY.read_json(registration_path)
    PATH_ONLY.require(registration.get("schema") == SCHEMA and registration.get("status") == STATUS, "registration invalid")
    PATH_ONLY.require(registration.get("label") == LABEL, "registration label mismatch")
    PATH_ONLY.require(registration.get("R2_and_R3_reuse_forbidden") is True, "old replay reuse not forbidden")
    for role, record in registration["bindings"].items():
        PATH_ONLY.require(PATH_ONLY.bind(Path(record["path"])) == record, f"binding drift: {role}")
    sources = source_paths(root)
    PATH_ONLY.require(set(registration["source_sha256"]) == set(sources), "R4 source inventory mismatch")
    for name, expected in registration["source_sha256"].items():
        PATH_ONLY.require(PATH_ONLY.sha256_file(sources[name]) == expected, f"source drift: {name}")
    reference = PATH_ONLY.read_json(Path(registration["bindings"]["selected_member_registration"]["path"]))
    for name in PATH_ONLY.ALGORITHM_SOURCES:
        PATH_ONLY.require(registration["source_sha256"][name] == reference["source_sha256"][name], f"selected algorithm source identity drift: {name}")
    validate_runtime_delta([str(value) for value in reference["command"]], [str(value) for value in registration["runtime_argv"]])
    return registration


def authorize(root: Path, registration_path: Path, output: Path) -> None:
    registration_path = registration_path.expanduser().resolve(strict=True)
    registration = validate_registration(root, registration_path)
    payload = {
        "schema": AUTH_SCHEMA,
        "enabled": True,
        "issued_utc": utc_now(),
        "authorization_scope": "one fresh schema-aware exact selected 9990-N_F R4 full-state-history output-only replay",
        "label": LABEL,
        "registration": PATH_ONLY.bind(registration_path),
        "runtime_argv_sha256": hashlib.sha256(json.dumps(registration["runtime_argv"], separators=(",", ":")).encode()).hexdigest(),
        "required_thread_caps": {name: "1" for name in PATH_ONLY.THREAD_CAPS},
        "required_lock_paths": [str(path) for path in PATH_ONLY.LOCKS],
        "minimum_free_disk_bytes_before_child": MINIMUM_FREE_BYTES,
        "required_Mutagen_session": "bb-irivm-20260901",
        "new_diagnostic_or_optimization_calls_from_capture": 0,
        "R2_and_R3_authorization_reuse_forbidden": True
    }
    PATH_ONLY.write_exclusive(output, payload)
    resolved = output.expanduser().resolve(strict=True)
    print(resolved)
    print(PATH_ONLY.sha256_file(resolved))


def validate_authorization(registration_path: Path, authorization_path: Path, registration: dict) -> dict:
    authorization = PATH_ONLY.read_json(authorization_path)
    PATH_ONLY.require(authorization.get("schema") == AUTH_SCHEMA and authorization.get("enabled") is True, "authorization invalid")
    PATH_ONLY.require(authorization.get("label") == LABEL and authorization.get("registration") == PATH_ONLY.bind(registration_path), "authorization binding mismatch")
    runtime_hash = hashlib.sha256(json.dumps(registration["runtime_argv"], separators=(",", ":")).encode()).hexdigest()
    PATH_ONLY.require(authorization.get("runtime_argv_sha256") == runtime_hash, "authorization runtime hash mismatch")
    PATH_ONLY.require(authorization.get("required_thread_caps") == {name: "1" for name in PATH_ONLY.THREAD_CAPS}, "authorization thread-cap drift")
    PATH_ONLY.require(authorization.get("required_lock_paths") == [str(path) for path in PATH_ONLY.LOCKS], "authorization lock drift")
    PATH_ONLY.require(authorization.get("minimum_free_disk_bytes_before_child") == MINIMUM_FREE_BYTES, "authorization disk gate drift")
    PATH_ONLY.require(authorization.get("required_Mutagen_session") == "bb-irivm-20260901", "authorization Mutagen drift")
    PATH_ONLY.require(authorization.get("new_diagnostic_or_optimization_calls_from_capture") == 0, "authorization permits capture calls")
    PATH_ONLY.require(authorization.get("R2_and_R3_authorization_reuse_forbidden") is True, "old authorization reuse not forbidden")
    return authorization


def preflight(root: Path, registration_path: Path, authorization_path: Path) -> None:
    registration_path = registration_path.expanduser().resolve(strict=True)
    authorization_path = authorization_path.expanduser().resolve(strict=True)
    registration = validate_registration(root, registration_path)
    validate_authorization(registration_path, authorization_path, registration)
    invalid_caps = {name: os.environ.get(name) for name in PATH_ONLY.THREAD_CAPS if os.environ.get(name) != "1"}
    PATH_ONLY.require(not invalid_caps, f"all six thread caps must equal 1: {invalid_caps}")
    free = shutil.disk_usage(root).free
    PATH_ONLY.require(free >= MINIMUM_FREE_BYTES, f"disk gate failed: {free} < {MINIMUM_FREE_BYTES}")
    output_dir = root / OUTPUT_SUBDIR
    PATH_ONLY.require(not list(output_dir.glob(f"ivm_{LABEL}_*")), "refusing to overwrite R4 runtime output")
    PATH_ONLY.require(not (output_dir / OBSERVER_NAME).exists(), "refusing to overwrite R4 observer history")
    print(json.dumps({
        "status": "PASS_R4_ZERO_OPENMM_PREFLIGHT",
        "label": LABEL,
        "registration": PATH_ONLY.bind(registration_path),
        "authorization": PATH_ONLY.bind(authorization_path),
        "R2_or_R3_reused": False,
        "runtime_argv_difference_count": 2,
        "runtime_argv_differences": ["--run-label", "--output-dir"],
        "path_storage_mode": "committed",
        "counter_schemas": {"live": 8, "historical": [7, 8], "guess_or_backfill": False},
        "expected_records": 352,
        "minimum_free_disk_bytes": MINIMUM_FREE_BYTES,
        "observed_free_disk_bytes": free,
        "OpenMM_imported_or_called": False
    }, indent=2, sort_keys=True))


def execute(root: Path, registration_path: Path, authorization_path: Path) -> None:
    registration_path = registration_path.expanduser().resolve(strict=True)
    authorization_path = authorization_path.expanduser().resolve(strict=True)
    preflight(root, registration_path, authorization_path)
    registration = validate_registration(root, registration_path)
    PATH_ONLY.require_parent_lock_holder()
    output_dir = root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    recorder = FullStateHistoryRecorder(output_dir / OBSERVER_NAME, LABEL, registration_path, PATH_ONLY.sha256_file(registration_path))
    sys.path.insert(0, str(root / "src"))
    import run_ivm_v9_spring_projected_production_20260730 as production  # pylint: disable=import-outside-toplevel
    import run_ivm_v9_discrete_ffrbk_takeover_grid_shim as runtime  # pylint: disable=import-outside-toplevel
    restore = install_output_observer(production, recorder)
    try:
        runtime.main([str(value) for value in registration["runtime_argv"]])
    finally:
        restore()
    results = list(output_dir.glob(f"ivm_{LABEL}_*.npz"))
    audits = list(output_dir.glob(f"ivm_{LABEL}_*.audit.json"))
    PATH_ONLY.require(len(results) == len(audits) == 1, "R4 runtime did not produce one result/audit pair")
    reference_result = Path(registration["bindings"]["selected_terminal_result"]["path"])
    manifest, capture_audit = recorder.finalize(results[0], reference_result)
    print(results[0]); print(audits[0]); print(manifest); print(capture_audit)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    modes = parser.add_subparsers(dest="mode", required=True)
    freeze_mode = modes.add_parser("freeze"); freeze_mode.add_argument("--output", type=Path, required=True)
    auth_mode = modes.add_parser("authorize"); auth_mode.add_argument("--registration", type=Path, required=True); auth_mode.add_argument("--output", type=Path, required=True)
    preflight_mode = modes.add_parser("preflight"); preflight_mode.add_argument("--registration", type=Path, required=True); preflight_mode.add_argument("--authorization", type=Path, required=True)
    run_mode = modes.add_parser("execute"); run_mode.add_argument("--registration", type=Path, required=True); run_mode.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root.expanduser().resolve(strict=True)
    if args.mode == "freeze": freeze(root, args.output)
    elif args.mode == "authorize": authorize(root, args.registration, args.output)
    elif args.mode == "preflight": preflight(root, args.registration, args.authorization)
    else: execute(root, args.registration, args.authorization)


if __name__ == "__main__":
    main()
