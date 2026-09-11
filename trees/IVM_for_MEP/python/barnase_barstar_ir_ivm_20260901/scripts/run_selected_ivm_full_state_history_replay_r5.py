#!/usr/bin/env python3
"""Freeze and run the minimal storage-only BB IR--IVM full-state R5 replay."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from typing import Iterable

import run_selected_ivm_full_history_replay as BASE
from full_state_history_recorder_r5 import (
    MAXIMUM_CANONICAL_CALLS,
    MAXIMUM_RAW_CALLBACKS,
    OLD_COMPLETE_PREFIX_CALLS,
    OPTIMIZATION_NF_BUDGET,
    MINIMUM_RUNTIME_FREE_BYTES,
    FullStateHistoryRecorder,
    install_output_observer,
)


SCHEMA = "bb-irivm-selected-full-state-history-output-only-replay-v5b"
AUTH_SCHEMA = "bb-irivm-selected-full-state-history-output-only-replay-authorization-v5b"
MUTAGEN_GATE_SCHEMA = "bb-irivm-full-state-r5b-mutagen-gate-v1"
STATUS = "FROZEN_BEFORE_FIRST_R5B_FULL_STATE_HISTORY_REPLAY_PHYSICAL_CALL"
LABEL = "bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b"
OUTPUT_SUBDIR = "runs/full_state_history_replay_r5b"
OBSERVER_NAME = LABEL + "_observer"
MINIMUM_FREE_BYTES = 8 * 1024**3
REMOTE_HOSTNAME = "Mac-mini.local"
REMOTE_ADDRESS = "100.74.85.3"
REMOTE_MACHINE = "arm64"
MUTAGEN = Path("/Users/wuzhiyou/.local/bin/mutagen")
MUTAGEN_NAME = "bb-irivm-20260901"
MUTAGEN_IDENTIFIER = "sync_EaEvvA9594NRGtMQSN0jneepyUq2tCef27jljwXXPc9"
R2_REGISTRATION = Path(
    "registrations/diagnostics/BB_IRIVM_SELECTED_ALL_PATHS_REPLAY_NF9990_20260902_R2.json"
)
R2_TERMINATION = Path(
    "registrations/diagnostics/BB_IRIVM_SELECTED_ALL_PATHS_REPLAY_R2_USER_TERMINATION_SUPERSESSION_20260902.json"
)
R5_MUTAGEN_FAILURE = Path(
    "registrations/diagnostics/BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_R5_MUTAGEN_GATE_FAILURE_SUPERSESSION_20260902.json"
)
ROOT = Path(__file__).resolve().parent.parent
CANONICAL_REGISTRATION = ROOT / "registrations/diagnostics/BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF15000_20260902_R5B.json"
CANONICAL_AUTHORIZATION = ROOT / "registrations/execution_authorizations/bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b.json"
CANONICAL_MUTAGEN_GATE = ROOT / "registrations/diagnostics/BB_IRIVM_SELECTED_FULL_STATE_HISTORY_MUTAGEN_GATE_NF15000_20260902_R5B.json"
# An actual newline is required here.  A literal backslash-n caused the
# superseded R5 prelaunch Mutagen parser to fail closed before any physics.
MUTAGEN_LIST_TEMPLATE = "{{ range . }}{{ .Name }}|{{ .Identifier }}|{{ .Status }}|{{ .Alpha.Connected }}|{{ .Beta.Connected }}\n{{ end }}"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def replay_argv(r2_registration: dict) -> list[str]:
    BASE.require(
        r2_registration.get("schema")
        == "bb-irivm-selected-all-paths-output-only-replay-v2"
        and r2_registration.get("status")
        == "FROZEN_BEFORE_FIRST_ALL_PATHS_REPLAY_PHYSICAL_CALL",
        "R2 all-path baseline registration invalid",
    )
    argv = [str(value) for value in r2_registration["runtime_argv"]]
    BASE.require(
        BASE.option_value(argv, "--run-label")
        == "bb_irivm_a1e0_r01_m05_inf9990_allpaths_r2",
        "R2 baseline label drift",
    )
    BASE.require(
        BASE.option_value(argv, "--output-dir") == "runs/all_paths_replay_r2",
        "R2 baseline output drift",
    )
    BASE.require(
        BASE.option_value(argv, "--path-storage-mode") == "legacy"
        and argv.count("--save-all-paths") == 1,
        "R2 baseline is not native all-path storage",
    )
    BASE.replace_option(argv, "--run-label", LABEL)
    BASE.replace_option(argv, "--output-dir", OUTPUT_SUBDIR)
    BASE.replace_option(argv, "--budget", str(OPTIMIZATION_NF_BUDGET))
    BASE.replace_option(
        argv, "--total-internal-nf-budget", str(OPTIMIZATION_NF_BUDGET)
    )
    return argv


def validate_runtime_delta(reference: list[str], replay: list[str]) -> None:
    BASE.require(len(reference) == len(replay), "R5B runtime argv length drift")
    expected = {
        reference.index("--run-label") + 1,
        reference.index("--output-dir") + 1,
        reference.index("--budget") + 1,
        reference.index("--total-internal-nf-budget") + 1,
    }
    observed = {
        index
        for index, pair in enumerate(zip(reference, replay))
        if pair[0] != pair[1]
    }
    BASE.require(observed == expected, f"R5B unauthorized argv drift: {observed}")
    BASE.require(
        BASE.option_value(replay, "--path-storage-mode") == "legacy"
        and replay.count("--save-all-paths") == 1,
        "R5B changed native all-path storage",
    )
    for flag, expected_value in (
        ("--budget", "15000"),
        ("--total-internal-nf-budget", "15000"),
        ("--alpha", "1"),
        ("--discrete-residual-ratio", "0.1"),
        ("--discrete-inner-max-steps", "5"),
        ("--discrete-step-multiplier-eta", "1.1"),
        ("--max-step", "1"),
        ("--reparam-mode", "state_curve"),
        ("--reparam-every", "1"),
    ):
        BASE.require(
            BASE.option_value(replay, flag) == expected_value,
            f"R5B algorithm drift: {flag}",
        )


def source_paths(root: Path) -> dict[str, Path]:
    result = {name: root / "src" / name for name in BASE.ALGORITHM_SOURCES}
    for name in (
        "run_selected_ivm_full_history_replay.py",
        "full_state_history_recorder.py",
        "full_state_history_recorder_r4.py",
        "full_state_history_recorder_r5.py",
        Path(__file__).name,
        "verify_selected_ivm_full_state_history_replay.py",
        "verify_selected_ivm_full_state_history_replay_r5.py",
    ):
        result[name] = root / "scripts" / name
    return result


def freeze(root: Path, output: Path) -> None:
    root = root.expanduser().resolve(strict=True)
    r2_path = (root / R2_REGISTRATION).resolve(strict=True)
    termination_path = (root / R2_TERMINATION).resolve(strict=True)
    r5_failure_path = (root / R5_MUTAGEN_FAILURE).resolve(strict=True)
    r2 = BASE.read_json(r2_path)
    termination = BASE.read_json(termination_path)
    r5_failure = BASE.read_json(r5_failure_path)
    BASE.require(
        termination.get("status")
        == "USER_TERMINATED_OUTPUT_ONLY_REPLAY_SUPERSEDED_NOT_A_SCIENTIFIC_FAILURE"
        and termination.get("supersession", {}).get(
            "replacement_requires_new_registration_authorization_label_and_output"
        )
        is True,
        "R2 user-termination receipt invalid",
    )
    BASE.require(
        r5_failure.get("status")
        == "R5_PRELAUNCH_GATE_FAILED_CLOSED_SUPERSEDED_BY_R5B_NO_PHYSICAL_CALLS"
        and r5_failure.get("physical_execution", {}).get(
            "OpenMM_imported_or_called"
        )
        is False
        and r5_failure.get("supersession", {}).get(
            "old_registration_or_authorization_may_be_reused"
        )
        is False,
        "R5 zero-physical Mutagen failure receipt invalid",
    )
    reference = [str(value) for value in r2["runtime_argv"]]
    runtime = replay_argv(r2)
    validate_runtime_delta(reference, runtime)
    sources = source_paths(root)
    for role, path in sources.items():
        BASE.require(path.is_file(), f"missing R5B source {role}: {path}")
    selected_registration = Path(
        r2["bindings"]["selected_member_registration"]["path"]
    ).resolve(strict=True)
    selected = BASE.read_json(selected_registration)
    for name in BASE.ALGORITHM_SOURCES:
        BASE.require(
            BASE.sha256_file(root / "src" / name)
            == selected["source_sha256"][name],
            f"selected algorithm source drift: {name}",
        )
    reference_result = Path(
        r2["bindings"]["selected_terminal_result"]["path"]
    ).resolve(strict=True)
    sealed_9990 = BASE.sealed_history_facts(reference_result)
    expected_calls_estimate = 520
    full_state_bytes = expected_calls_estimate * (2 * 32 * 9396 + 32) * 8
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "created_utc": utc_now(),
        "execution_authorized_by_this_file": False,
        "purpose": (
            "fresh exact R2-all-path replay plus storage-only capture of the "
            "already-current stable-last path32, energy32, and full-force32"
        ),
        "label": LABEL,
        "baseline_label": r2["label"],
        "prospective_outputs": {
            "registration": str(CANONICAL_REGISTRATION),
            "authorization": str(CANONICAL_AUTHORIZATION),
            "Mutagen_gate": str(CANONICAL_MUTAGEN_GATE),
            "runtime_output_directory": str((ROOT / OUTPUT_SUBDIR).resolve()),
        },
        "Mutagen_gate_contract": {
            "canonical_output": str(CANONICAL_MUTAGEN_GATE),
            "generator": BASE.bind(Path(__file__).resolve()),
            "schema": MUTAGEN_GATE_SCHEMA,
            "required_session_name": MUTAGEN_NAME,
            "required_session_identifier": MUTAGEN_IDENTIFIER,
            "list_template_contains_actual_newline": "\n" in MUTAGEN_LIST_TEMPLATE,
            "list_template_contains_literal_backslash_n": "\\n" in MUTAGEN_LIST_TEMPLATE,
        },
        "runtime_argv": runtime,
        "only_runtime_argv_differences_from_interrupted_R2": {
            "--run-label": LABEL,
            "--output-dir": OUTPUT_SUBDIR,
            "--budget": OPTIMIZATION_NF_BUDGET,
            "--total-internal-nf-budget": OPTIMIZATION_NF_BUDGET,
        },
        "algorithm_identity": {
            "R2_runtime_argv_other_than_identity_and_user_extended_budget_byte_for_byte_equal": True,
            "all_numerical_controls_except_user_extended_budget_unchanged": True,
            "user_authorized_budget_extension_9990_to_15000": True,
            "native_save_all_paths_retained": True,
            "path_storage_mode": "legacy",
            "optimization_N_F_budget": OPTIMIZATION_NF_BUDGET,
            "fresh_from_common_initial_path": True,
            "warm_start": False,
        },
        "storage_only_capture": {
            "raw_callback_count": "measured_at_runtime",
            "maximum_raw_callback_fail_closed_cap": MAXIMUM_RAW_CALLBACKS,
            "raw_callbacks_store_only_small_metadata_hash_WAL": True,
            "atomic_pending_same_call_replacement": True,
            "immutable_stable_last_full_state_shards": "actual formal call count K",
            "canonical_index_references": "K",
            "maximum_canonical_call_fail_closed_cap": MAXIMUM_CANONICAL_CALLS,
            "stable_last_rule": "same reversed-numpy-unique rule as production finalization",
            "each_canonical_shard": ["path32_A", "energy_eV32", "force_eV_per_A32"],
            "new_energy_or_force_queries": 0,
            "optimization_N_F_contribution": 0,
            "observer_return_is_exact_original_storage_policy_return": True,
            "old_9990_complete_prefix_verification_minimum": {
                "call_range": [1, OLD_COMPLETE_PREFIX_CALLS],
                "last_optimization_N_F": 8982,
                "source_facts": sealed_9990,
                "old_terminal_tail_319_352_required_equal": False,
            },
        },
        "latest_user_instruction_supersedes_R4_raw455_full_array_requirement": {
            "R4_failure_receipt_preserved": True,
            "raw455_full_arrays": False,
            "raw455_metadata_hash_WAL": True,
            "stable_last_K_full_arrays": True,
        },
        "resource_estimate": {
            "expected_stable_last_calls_extrapolated_from_9990": expected_calls_estimate,
            "expected_stable_last_array_payload_bytes": full_state_bytes,
            "expected_stable_last_array_payload_GiB": full_state_bytes / 1024**3,
            "expected_total_capture_disk_GiB": "2.5--3.2",
            "native_stored_paths_estimated_GiB_at_K520": (520 * 32 * 9396 * 8) / 1024**3,
            "single_atomic_pending_full_state_MiB": ((2 * 32 * 9396 + 32) * 8) / 1024**2,
            "peak_upper_estimate_including_native_result_atomic_publication_GiB": "5.0--5.8",
            "minimum_free_disk_bytes_before_child": MINIMUM_FREE_BYTES,
            "minimum_free_disk_bytes_checked_before_every_pending_write_and_promotion": MINIMUM_RUNTIME_FREE_BYTES,
        },
        "concurrency": {
            "host_role": "remote Mac mini",
            "hostname": REMOTE_HOSTNAME,
            "address": REMOTE_ADDRESS,
            "machine": REMOTE_MACHINE,
            "jobs": 1,
            "all_six_thread_caps": 1,
            "two_host_local_fcntl_locks_required_before_child": True,
            "dedicated_Mutagen_session": MUTAGEN_NAME,
            "dedicated_Mutagen_identifier": MUTAGEN_IDENTIFIER,
        },
        "bindings": {
            "R2_registration": BASE.bind(r2_path),
            "R2_user_termination_supersession": BASE.bind(termination_path),
            "R5_zero_physical_Mutagen_failure_supersession": BASE.bind(r5_failure_path),
            **r2["bindings"],
        },
        "source_sha256": {
            name: BASE.sha256_file(path) for name, path in sources.items()
        },
    }
    BASE.write_exclusive(output, payload)
    resolved = output.expanduser().resolve(strict=True)
    print(resolved)
    print(BASE.sha256_file(resolved))


def validate_registration(root: Path, path: Path) -> dict:
    path = path.expanduser().resolve(strict=True)
    BASE.require(path == CANONICAL_REGISTRATION.resolve(), "noncanonical R5B registration path")
    value = BASE.read_json(path)
    BASE.require(
        value.get("schema") == SCHEMA
        and value.get("status") == STATUS
        and value.get("label") == LABEL,
        "R5B registration invalid",
    )
    for role, binding in value["bindings"].items():
        BASE.require(BASE.bind(Path(binding["path"])) == binding, f"binding drift: {role}")
    sources = source_paths(root)
    BASE.require(set(value["source_sha256"]) == set(sources), "R5B source inventory drift")
    for name, source in sources.items():
        BASE.require(
            BASE.sha256_file(source) == value["source_sha256"][name],
            f"R5B source drift: {name}",
        )
    r2 = BASE.read_json(Path(value["bindings"]["R2_registration"]["path"]))
    BASE.require(value.get("prospective_outputs") == {
        "registration": str(CANONICAL_REGISTRATION),
        "authorization": str(CANONICAL_AUTHORIZATION),
        "Mutagen_gate": str(CANONICAL_MUTAGEN_GATE),
        "runtime_output_directory": str((ROOT / OUTPUT_SUBDIR).resolve()),
    }, "R5B canonical prospective outputs drift")
    BASE.require(value.get("Mutagen_gate_contract") == {
        "canonical_output": str(CANONICAL_MUTAGEN_GATE),
        "generator": BASE.bind(Path(__file__).resolve()),
        "schema": MUTAGEN_GATE_SCHEMA,
        "required_session_name": MUTAGEN_NAME,
        "required_session_identifier": MUTAGEN_IDENTIFIER,
        "list_template_contains_actual_newline": True,
        "list_template_contains_literal_backslash_n": False,
    }, "R5B Mutagen gate contract drift")
    validate_runtime_delta(
        [str(item) for item in r2["runtime_argv"]],
        [str(item) for item in value["runtime_argv"]],
    )
    return value


def authorize(root: Path, registration_path: Path, output: Path) -> None:
    registration_path = registration_path.expanduser().resolve(strict=True)
    registration = validate_registration(root, registration_path)
    payload = {
        "schema": AUTH_SCHEMA,
        "enabled": True,
        "issued_utc": utc_now(),
        "authorization_scope": "one fresh remote R5B storage-only full-state replay",
        "label": LABEL,
        "registration": BASE.bind(registration_path),
        "runtime_argv_sha256": hashlib.sha256(
            json.dumps(registration["runtime_argv"], separators=(",", ":")).encode()
        ).hexdigest(),
        "required_thread_caps": {name: "1" for name in BASE.THREAD_CAPS},
        "required_lock_paths": [str(path) for path in BASE.LOCKS],
        "minimum_free_disk_bytes_before_child": MINIMUM_FREE_BYTES,
        "required_Mutagen_session": MUTAGEN_NAME,
        "new_capture_queries": 0,
        "old_R2_R3_R4_authorizations_reused": False,
    }
    BASE.write_exclusive(output, payload)
    print(output.expanduser().resolve(strict=True))
    print(BASE.sha256_file(output.expanduser().resolve(strict=True)))


def validate_authorization(registration_path: Path, path: Path, registration: dict) -> dict:
    path = path.expanduser().resolve(strict=True)
    BASE.require(path == CANONICAL_AUTHORIZATION.resolve(), "noncanonical R5B authorization path")
    value = BASE.read_json(path)
    BASE.require(value.get("schema") == AUTH_SCHEMA and value.get("enabled") is True and value.get("label") == LABEL, "R5B authorization invalid")
    BASE.require(value.get("registration") == BASE.bind(registration_path), "authorization registration drift")
    digest = hashlib.sha256(json.dumps(registration["runtime_argv"], separators=(",", ":")).encode()).hexdigest()
    BASE.require(value.get("runtime_argv_sha256") == digest, "authorization argv drift")
    BASE.require(value.get("required_thread_caps") == {name: "1" for name in BASE.THREAD_CAPS}, "authorization thread caps drift")
    BASE.require(value.get("required_lock_paths") == [str(path) for path in BASE.LOCKS], "authorization locks drift")
    BASE.require(value.get("minimum_free_disk_bytes_before_child") == MINIMUM_FREE_BYTES, "authorization disk gate drift")
    BASE.require(value.get("required_Mutagen_session") == MUTAGEN_NAME, "authorization Mutagen session drift")
    BASE.require(value.get("new_capture_queries") == 0, "authorization permits capture queries")
    BASE.require(value.get("old_R2_R3_R4_authorizations_reused") is False, "old authorization reuse permitted")
    return value


def flush_mutagen(registration_path: Path, authorization_path: Path, output: Path) -> None:
    registration_path = registration_path.expanduser().resolve(strict=True)
    authorization_path = authorization_path.expanduser().resolve(strict=True)
    registration = validate_registration(ROOT, registration_path)
    validate_authorization(registration_path, authorization_path, registration)
    BASE.require(output.expanduser().resolve() == CANONICAL_MUTAGEN_GATE.resolve(), "noncanonical R5B Mutagen gate output")
    executable = MUTAGEN.expanduser().resolve(strict=True)
    flush = subprocess.run([str(executable), "sync", "flush", MUTAGEN_NAME], check=False, text=True, capture_output=True)
    BASE.require(flush.returncode == 0, f"Mutagen flush failed: {flush.stderr.strip()}")
    listing = subprocess.run([str(executable), "sync", "list", "--template", MUTAGEN_LIST_TEMPLATE], check=True, text=True, capture_output=True)
    matches = [line.split("|") for line in listing.stdout.splitlines() if line.split("|")[0] == MUTAGEN_NAME]
    BASE.require(len(matches) == 1 and len(matches[0]) == 5, "Mutagen session not uniquely listed")
    name, identifier, status, alpha, beta = matches[0]
    raw = subprocess.run([str(executable), "sync", "list", MUTAGEN_NAME], check=True, text=True, capture_output=True).stdout.lower()
    payload = {
        "schema": MUTAGEN_GATE_SCHEMA,
        "status": "PASS_CONNECTED_FLUSHED",
        "created_utc": utc_now(),
        "registration": BASE.bind(registration_path),
        "authorization": BASE.bind(authorization_path),
        "label": LABEL,
        "gate_generator": BASE.bind(Path(__file__).resolve()),
        "session_name": name,
        "session_identifier": identifier,
        "session_status": status,
        "alpha_connected": alpha.lower() == "true",
        "beta_connected": beta.lower() == "true",
        "transition_problems": int("problems:" in raw or "conflicts:" in raw or "error:" in raw),
        "flush_exit_code": flush.returncode,
    }
    BASE.require(name == MUTAGEN_NAME and identifier == MUTAGEN_IDENTIFIER, "Mutagen identity drift")
    BASE.require(status in {"Watching", "Watching for changes"} and payload["alpha_connected"] and payload["beta_connected"] and payload["transition_problems"] == 0, "Mutagen connectivity gate failed")
    BASE.write_exclusive(output, payload)
    print(output.expanduser().resolve(strict=True)); print(BASE.sha256_file(output.expanduser().resolve(strict=True)))


def validate_mutagen(registration_path: Path, authorization_path: Path, path: Path) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    BASE.require(resolved == CANONICAL_MUTAGEN_GATE.resolve(), "noncanonical R5B Mutagen gate path")
    value = BASE.read_json(resolved)
    BASE.require(value.get("schema") == MUTAGEN_GATE_SCHEMA and value.get("status") == "PASS_CONNECTED_FLUSHED" and value.get("label") == LABEL, "Mutagen gate invalid")
    BASE.require(value.get("registration") == BASE.bind(registration_path) and value.get("authorization") == BASE.bind(authorization_path), "Mutagen gate contract drift")
    BASE.require(value.get("session_name") == MUTAGEN_NAME and value.get("session_identifier") == MUTAGEN_IDENTIFIER, "Mutagen gate identity drift")
    BASE.require(value.get("gate_generator") == BASE.bind(Path(__file__).resolve()), "Mutagen gate generator drift")
    BASE.require(value.get("alpha_connected") is True and value.get("beta_connected") is True and value.get("transition_problems") == 0 and value.get("flush_exit_code") == 0, "Mutagen gate failed")
    return value


@contextmanager
def nonblocking_lock_probe(paths):
    streams = []
    try:
        for path in sorted((Path(value).resolve() for value in paths), key=str):
            stream = path.open("a+")
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            streams.append(stream)
        yield
    finally:
        for stream in reversed(streams):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN); stream.close()


def preflight(root: Path, registration_path: Path, authorization_path: Path, mutagen_path: Path, *, probe_host=True, probe_locks=True) -> dict:
    registration_path = registration_path.expanduser().resolve(strict=True)
    authorization_path = authorization_path.expanduser().resolve(strict=True)
    registration = validate_registration(root, registration_path)
    validate_authorization(registration_path, authorization_path, registration)
    validate_mutagen(registration_path, authorization_path, mutagen_path)
    bad = {name: os.environ.get(name) for name in BASE.THREAD_CAPS if os.environ.get(name) != "1"}
    BASE.require(not bad, f"all six thread caps must equal 1: {bad}")
    observed_host = {"hostname": socket.gethostname(), "platform": platform.system(), "machine": platform.machine()}
    if probe_host:
        BASE.require(observed_host == {"hostname": REMOTE_HOSTNAME, "platform": "Darwin", "machine": REMOTE_MACHINE}, f"wrong R5B host: {observed_host}")
    free = shutil.disk_usage(root).free
    BASE.require(free >= MINIMUM_FREE_BYTES, f"R5B disk gate failed: {free}")
    output_dir = root / OUTPUT_SUBDIR
    BASE.require(not list(output_dir.glob(f"ivm_{LABEL}_*")), "refusing to overwrite R5B runtime output")
    BASE.require(not (output_dir / OBSERVER_NAME).exists(), "refusing to overwrite R5B observer output")
    if probe_locks:
        with nonblocking_lock_probe(BASE.LOCKS):
            pass
    report = {
        "status": "PASS_R5B_ZERO_OPENMM_REMOTE_PREFLIGHT",
        "runtime_argv_difference_count_from_interrupted_R2": 4,
        "runtime_argv_differences": ["--run-label", "--output-dir", "--budget", "--total-internal-nf-budget"],
        "native_save_all_paths": True,
        "path_storage_mode": "legacy",
        "optimization_N_F_budget": OPTIMIZATION_NF_BUDGET,
        "raw_callback_fail_closed_cap": MAXIMUM_RAW_CALLBACKS,
        "stable_last_call_fail_closed_cap": MAXIMUM_CANONICAL_CALLS,
        "new_capture_queries": 0,
        "OpenMM_imported_or_called": False,
        "host": observed_host if probe_host else "SKIPPED_FOR_TEST",
        "lock_probe": "PASS_NONBLOCKING" if probe_locks else "HELD_BY_PARENT_OR_SKIPPED_FOR_TEST",
        "observed_free_disk_bytes": free,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def execute(root: Path, registration_path: Path, authorization_path: Path, mutagen_path: Path) -> None:
    preflight(root, registration_path, authorization_path, mutagen_path, probe_host=True, probe_locks=False)
    BASE.require_parent_lock_holder()
    registration = validate_registration(root, registration_path)
    output_dir = root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    recorder = FullStateHistoryRecorder(output_dir / OBSERVER_NAME, LABEL, registration_path, BASE.sha256_file(registration_path))
    sys.path.insert(0, str(root / "src"))
    import run_ivm_v9_spring_projected_production_20260730 as production  # pylint: disable=import-outside-toplevel
    import run_ivm_v9_discrete_ffrbk_takeover_grid_shim as runtime  # pylint: disable=import-outside-toplevel
    restore = install_output_observer(production, recorder)
    try:
        runtime.main([str(value) for value in registration["runtime_argv"]])
    finally:
        restore()
    results = list(output_dir.glob(f"ivm_{LABEL}_*.npz")); audits = list(output_dir.glob(f"ivm_{LABEL}_*.audit.json"))
    BASE.require(len(results) == len(audits) == 1, "R5B runtime did not produce one result/audit pair")
    reference = Path(registration["bindings"]["selected_terminal_result"]["path"])
    manifest, canonical, capture_audit = recorder.finalize(results[0], reference)
    print(results[0]); print(audits[0]); print(manifest); print(canonical); print(capture_audit)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    modes = parser.add_subparsers(dest="mode", required=True)
    freeze_mode = modes.add_parser("freeze"); freeze_mode.add_argument("--output", type=Path, required=True)
    auth_mode = modes.add_parser("authorize"); auth_mode.add_argument("--registration", type=Path, required=True); auth_mode.add_argument("--output", type=Path, required=True)
    gate = modes.add_parser("mutagen-gate"); gate.add_argument("--registration", type=Path, required=True); gate.add_argument("--authorization", type=Path, required=True); gate.add_argument("--output", type=Path, required=True)
    for name in ("preflight", "execute"):
        mode = modes.add_parser(name); mode.add_argument("--registration", type=Path, required=True); mode.add_argument("--authorization", type=Path, required=True); mode.add_argument("--mutagen-gate", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None); root = args.root.expanduser().resolve(strict=True)
    if args.mode == "freeze":
        BASE.require(args.output.expanduser().resolve() == CANONICAL_REGISTRATION.resolve(), "noncanonical R5B registration output")
        freeze(root, args.output)
    elif args.mode == "authorize":
        BASE.require(args.registration.expanduser().resolve() == CANONICAL_REGISTRATION.resolve() and args.output.expanduser().resolve() == CANONICAL_AUTHORIZATION.resolve(), "noncanonical R5B authorization inputs/output")
        authorize(root, args.registration, args.output)
    elif args.mode == "mutagen-gate": flush_mutagen(args.registration, args.authorization, args.output)
    elif args.mode == "preflight": preflight(root, args.registration, args.authorization, args.mutagen_gate)
    else: execute(root, args.registration, args.authorization, args.mutagen_gate)


if __name__ == "__main__":
    main()
