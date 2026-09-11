#!/usr/bin/env python3
"""Run the selected BB IR--IVM to N_F <= 30000 with full state capture.

This R6 runner is deliberately a storage/budget-only extension of the frozen
R5B N_F=15000 replay.  The scientific command differs in exactly four argv
values: run label, output directory, per-image safety budget, and shared
internal N_F budget.  The numerical method and every other control remain
byte-for-byte identical.  Because R5B did not serialize the live optimizer
history, an exact experiment extension must deterministically replay from the
same registered initial path; warm-starting from the terminal path would be a
different algorithm.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
from typing import Iterable

import full_state_history_recorder_r5 as RECORDER
import run_selected_ivm_full_state_history_replay_r5 as R5


ROOT = Path(__file__).resolve().parent.parent
BUDGET = 30000
LABEL = "bb_irivm_a1e0_r01_m05_inf30000_fullstate_r6b"
EXTERNAL_OUTPUT = Path(
    "/Volumes/GM7_1TB/IVM_for_MEP/active_runs/"
    "bb_irivm_a1e0_r01_m05_inf30000_fullstate_r6b"
)
OBSERVER_NAME = LABEL + "_observer"
SCHEMA = "bb-irivm-selected-full-state-history-budget-extension-v6b"
AUTH_SCHEMA = "bb-irivm-selected-full-state-history-budget-extension-authorization-v6b"
MUTAGEN_GATE_SCHEMA = "bb-irivm-full-state-r6b-nf30000-mutagen-gate-v1"
STATUS = "FROZEN_BEFORE_FIRST_R6B_NF30000_PHYSICAL_CALL"
VERIFY_SCHEMA = "bb-irivm-selected-full-state-history-verification-v6b"
VERIFY_STATUS = "PASS_R6B_NF30000_EXACT_PREFIX318_AND_FULL_STATE_CAPTURE"
MUTAGEN_NAME = "bb-irivm-nf30000-20260904"
MUTAGEN_IDENTIFIER = "sync_E3qb26wEIMCi85xMPMQBLwdKzhlUiYTUSQnB6KgAM7b"

R5B_REGISTRATION = ROOT / (
    "registrations/diagnostics/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF15000_20260902_R5B.json"
)
R5B_AUTHORIZATION = ROOT / (
    "registrations/execution_authorizations/"
    "bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b.json"
)
R5B_RESULT = ROOT / (
    "runs/full_state_history_replay_r5b/"
    "ivm_bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b_"
    "central_state_ivm_cfgf5e79e830f27_budget15000_storagelegacy.npz"
)
R5B_AUDIT = R5B_RESULT.with_suffix(".audit.json")
R5B_OBSERVER = ROOT / (
    "runs/full_state_history_replay_r5b/"
    "bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b_observer"
)
R5B_VERIFICATION = ROOT / (
    "reports/verifications/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF15000_20260902_R5B_PRIMARY.json"
)
CANONICAL_REGISTRATION = ROOT / (
    "registrations/diagnostics/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF30000_20260904_R6B.json"
)
CANONICAL_AUTHORIZATION = ROOT / (
    "registrations/execution_authorizations/"
    "bb_irivm_a1e0_r01_m05_inf30000_fullstate_r6b.json"
)
CANONICAL_MUTAGEN_GATE = ROOT / (
    "registrations/diagnostics/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_MUTAGEN_GATE_NF30000_20260904_R6B.json"
)
CANONICAL_VERIFICATION = ROOT / (
    "reports/verifications/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF30000_20260904_R6B_PRIMARY.json"
)
LAUNCH_RECEIPT = ROOT / (
    "reports/campaign_status/"
    "bb-irivm-fullstate-nf30000-remote-r6b-lock-receipt.json"
)
R6_FAILURE_SUPERSESSION = ROOT / (
    "registrations/diagnostics/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF30000_R6_"
    "PRELAUNCH_GATE_FAILURE_SUPERSESSION_20260904.json"
)
MINIMUM_EXTERNAL_FREE_BYTES = 100 * 1024**3

_R5_SOURCE_PATHS = R5.source_paths


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def source_paths(root: Path) -> dict[str, Path]:
    paths = _R5_SOURCE_PATHS(root)
    paths[Path(__file__).name] = Path(__file__).resolve()
    return paths


def runtime_argv(reference: dict) -> list[str]:
    argv = [str(value) for value in reference["runtime_argv"]]
    R5.BASE.replace_option(argv, "--run-label", LABEL)
    R5.BASE.replace_option(argv, "--output-dir", str(EXTERNAL_OUTPUT))
    R5.BASE.replace_option(argv, "--budget", str(BUDGET))
    R5.BASE.replace_option(argv, "--total-internal-nf-budget", str(BUDGET))
    validate_runtime_delta(reference["runtime_argv"], argv)
    return argv


def validate_runtime_delta(reference_values, replay_values) -> None:
    reference = [str(value) for value in reference_values]
    replay = [str(value) for value in replay_values]
    R5.BASE.require(len(reference) == len(replay), "R6 argv length drift")
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
    R5.BASE.require(observed == expected, f"R6 unauthorized argv drift: {observed}")
    required = {
        "--run-label": LABEL,
        "--output-dir": str(EXTERNAL_OUTPUT),
        "--budget": str(BUDGET),
        "--total-internal-nf-budget": str(BUDGET),
        "--alpha": "1",
        "--alpha-mode": "fixed",
        "--max-step": "1",
        "--history": "7",
        "--inner-calls": "15",
        "--reparam-mode": "state_curve",
        "--reparam-every": "1",
        "--discrete-inner-max-steps": "5",
        "--discrete-stop-mode": "r08c10",
        "--discrete-residual-ratio": "0.1",
        "--discrete-history": "7",
        "--discrete-step-multiplier-eta": "1.1",
        "--path-storage-mode": "legacy",
    }
    for flag, expected_value in required.items():
        R5.BASE.require(
            R5.BASE.option_value(replay, flag) == expected_value,
            f"R6 algorithm drift: {flag}",
        )
    R5.BASE.require(replay.count("--save-all-paths") == 1, "R6 lost all-path storage")


def bind_existing(path: Path) -> dict:
    return R5.BASE.bind(path.expanduser().resolve(strict=True))


def freeze(output: Path) -> None:
    reference = R5.BASE.read_json(R5B_REGISTRATION.resolve(strict=True))
    R5.BASE.require(
        reference.get("schema")
        == "bb-irivm-selected-full-state-history-output-only-replay-v5b",
        "R5B reference registration invalid",
    )
    argv = runtime_argv(reference)
    paths = source_paths(ROOT)
    for role, path in paths.items():
        R5.BASE.require(path.is_file(), f"missing frozen R6 source {role}: {path}")
    bindings = {
        "R5B_registration": bind_existing(R5B_REGISTRATION),
        "R5B_authorization": bind_existing(R5B_AUTHORIZATION),
        "R5B_result": bind_existing(R5B_RESULT),
        "R5B_runtime_audit": bind_existing(R5B_AUDIT),
        "R5B_capture_manifest": bind_existing(
            R5B_OBSERVER / "full_state_history.manifest.json"
        ),
        "R5B_capture_audit": bind_existing(
            R5B_OBSERVER / "full_state_history.audit.json"
        ),
        "R5B_primary_verification": bind_existing(R5B_VERIFICATION),
        "R6_prelaunch_gate_failure_supersession": bind_existing(R6_FAILURE_SUPERSESSION),
        # R5.execute/finalize deliberately compares against this sealed 9990 result.
        "selected_terminal_result": reference["bindings"]["selected_terminal_result"],
        "common_initial_path": reference["bindings"]["common_initial_path"],
        "endpoint_cache": reference["bindings"]["endpoint_cache"],
        "selected_member_registration": reference["bindings"]["selected_member_registration"],
        "topology": reference["bindings"]["topology"],
        "metric_policy": reference["bindings"]["metric_policy"],
    }
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "created_utc": utc_now(),
        "execution_authorized_by_this_file": False,
        "purpose": (
            "extend the selected Barnase--Barstar IR-IVM experiment to N_F <= 30000 "
            "while retaining every canonical path, its 32 energies, and full 32-image forces"
        ),
        "label": LABEL,
        "reference_label": reference["label"],
        "runtime_argv": argv,
        "only_runtime_argv_differences_from_R5B": {
            "--run-label": LABEL,
            "--output-dir": str(EXTERNAL_OUTPUT),
            "--budget": BUDGET,
            "--total-internal-nf-budget": BUDGET,
        },
        "algorithm_identity": {
            "all_numerical_controls_except_budget_byte_for_byte_equal_to_R5B": True,
            "user_authorized_budget_extension_15000_to_30000": True,
            "same_registered_common_initial_path": True,
            "same_32_images": True,
            "same_fixed_endpoint_N_F_policy": True,
            "same_tangent_and_reparameterization": True,
            "native_save_all_paths_retained": True,
            "warm_start": False,
            "reason_fresh_deterministic_replay_required": (
                "R5B stores paths, energies and forces but not the live L-BFGS correction "
                "pairs and outer control state; terminal-path warm start would change the algorithm"
            ),
        },
        "storage_only_capture": {
            "one_stable_last_shard_per_canonical_call": True,
            "each_shard": ["path32_A", "energy_eV32", "force_eV_per_A32"],
            "native_stored_paths": True,
            "additional_physical_calls": 0,
            "optimization_N_F_contribution": 0,
            "maximum_raw_callbacks": RECORDER.MAXIMUM_RAW_CALLBACKS,
            "maximum_canonical_calls": RECORDER.MAXIMUM_CANONICAL_CALLS,
        },
        "prospective_outputs": {
            "registration": str(CANONICAL_REGISTRATION),
            "authorization": str(CANONICAL_AUTHORIZATION),
            "Mutagen_gate": str(CANONICAL_MUTAGEN_GATE),
            "runtime_output_directory": str(EXTERNAL_OUTPUT),
            "primary_verification": str(CANONICAL_VERIFICATION),
            "lock_receipt": str(LAUNCH_RECEIPT),
        },
        "Mutagen_gate_contract": {
            "canonical_output": str(CANONICAL_MUTAGEN_GATE),
            "generator": R5.BASE.bind(Path(R5.__file__).resolve()),
            "schema": MUTAGEN_GATE_SCHEMA,
            "required_session_name": MUTAGEN_NAME,
            "required_session_identifier": MUTAGEN_IDENTIFIER,
        },
        "resource_policy": {
            "host": "remote Mac mini",
            "jobs": 1,
            "all_six_thread_caps": 1,
            "host_local_BB_locks": [str(path) for path in R5.BASE.LOCKS],
            "runtime_output_on_external_disk": str(EXTERNAL_OUTPUT),
            "minimum_external_free_bytes": MINIMUM_EXTERNAL_FREE_BYTES,
        },
        "bindings": bindings,
        "source_sha256": {
            name: R5.BASE.sha256_file(path) for name, path in paths.items()
        },
    }
    R5.BASE.write_exclusive(output, payload)
    resolved = output.expanduser().resolve(strict=True)
    print(resolved)
    print(R5.BASE.sha256_file(resolved))


def validate_registration(root: Path, path: Path) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    R5.BASE.require(resolved == CANONICAL_REGISTRATION.resolve(), "noncanonical R6 registration")
    value = R5.BASE.read_json(resolved)
    R5.BASE.require(
        value.get("schema") == SCHEMA
        and value.get("status") == STATUS
        and value.get("label") == LABEL,
        "R6 registration invalid",
    )
    reference = R5.BASE.read_json(Path(value["bindings"]["R5B_registration"]["path"]))
    validate_runtime_delta(reference["runtime_argv"], value["runtime_argv"])
    paths = source_paths(root)
    R5.BASE.require(set(value["source_sha256"]) == set(paths), "R6 source inventory drift")
    for name, source in paths.items():
        R5.BASE.require(
            R5.BASE.sha256_file(source) == value["source_sha256"][name],
            f"R6 source drift: {name}",
        )
    for role, binding in value["bindings"].items():
        R5.BASE.require(R5.BASE.bind(Path(binding["path"])) == binding, f"R6 binding drift: {role}")
    expected_outputs = value.get("prospective_outputs", {})
    R5.BASE.require(
        expected_outputs.get("runtime_output_directory") == str(EXTERNAL_OUTPUT)
        and expected_outputs.get("registration") == str(CANONICAL_REGISTRATION)
        and expected_outputs.get("authorization") == str(CANONICAL_AUTHORIZATION),
        "R6 prospective output drift",
    )
    return value


def authorize(registration_path: Path, output: Path) -> None:
    registration = validate_registration(ROOT, registration_path)
    payload = {
        "schema": AUTH_SCHEMA,
        "enabled": True,
        "issued_utc": utc_now(),
        "authorization_scope": "one fresh exact remote R6 N_F=30000 replay",
        "label": LABEL,
        "registration": R5.BASE.bind(registration_path.resolve(strict=True)),
        "runtime_argv_sha256": hashlib.sha256(
            json.dumps(registration["runtime_argv"], separators=(",", ":")).encode()
        ).hexdigest(),
        "required_thread_caps": {name: "1" for name in R5.BASE.THREAD_CAPS},
        "required_lock_paths": [str(path) for path in R5.BASE.LOCKS],
        "required_Mutagen_session": MUTAGEN_NAME,
        "runtime_output_directory": str(EXTERNAL_OUTPUT),
        "optimization_N_F_budget": BUDGET,
        "new_capture_queries": 0,
    }
    R5.BASE.write_exclusive(output, payload)
    print(output.resolve(strict=True))
    print(R5.BASE.sha256_file(output.resolve(strict=True)))


def validate_authorization(registration_path: Path, path: Path, registration: dict) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    R5.BASE.require(resolved == CANONICAL_AUTHORIZATION.resolve(), "noncanonical R6 authorization")
    value = R5.BASE.read_json(resolved)
    digest = hashlib.sha256(
        json.dumps(registration["runtime_argv"], separators=(",", ":")).encode()
    ).hexdigest()
    R5.BASE.require(
        value.get("schema") == AUTH_SCHEMA
        and value.get("enabled") is True
        and value.get("label") == LABEL
        and value.get("registration") == R5.BASE.bind(registration_path.resolve(strict=True))
        and value.get("runtime_argv_sha256") == digest
        and value.get("required_thread_caps") == {name: "1" for name in R5.BASE.THREAD_CAPS}
        and value.get("required_lock_paths") == [str(item) for item in R5.BASE.LOCKS]
        and value.get("required_Mutagen_session") == MUTAGEN_NAME
        and value.get("runtime_output_directory") == str(EXTERNAL_OUTPUT)
        and value.get("optimization_N_F_budget") == BUDGET
        and value.get("new_capture_queries") == 0,
        "R6 authorization invalid",
    )
    return value


def preflight(
    root: Path,
    registration_path: Path,
    authorization_path: Path,
    mutagen_path: Path,
    *,
    probe_host: bool = True,
    probe_locks: bool = True,
) -> dict:
    registration = validate_registration(root, registration_path)
    validate_authorization(registration_path, authorization_path, registration)
    R5.validate_mutagen(registration_path, authorization_path, mutagen_path)
    bad = {name: os.environ.get(name) for name in R5.BASE.THREAD_CAPS if os.environ.get(name) != "1"}
    R5.BASE.require(not bad, f"all six thread caps must equal 1: {bad}")
    host = {"hostname": socket.gethostname(), "platform": platform.system(), "machine": platform.machine()}
    if probe_host:
        R5.BASE.require(
            host == {"hostname": R5.REMOTE_HOSTNAME, "platform": "Darwin", "machine": R5.REMOTE_MACHINE},
            f"wrong R6 host: {host}",
        )
    R5.BASE.require(EXTERNAL_OUTPUT.parent.is_dir(), "external active-runs parent missing")
    external_free = shutil.disk_usage(EXTERNAL_OUTPUT.parent).free
    R5.BASE.require(external_free >= MINIMUM_EXTERNAL_FREE_BYTES, "external disk free-space gate failed")
    R5.BASE.require(not EXTERNAL_OUTPUT.exists(), "refusing to overwrite R6 output directory")
    if probe_locks:
        with R5.nonblocking_lock_probe(R5.BASE.LOCKS):
            pass
    report = {
        "status": "PASS_R6_ZERO_OPENMM_REMOTE_PREFLIGHT",
        "host": host if probe_host else "SKIPPED_FOR_TEST",
        "lock_probe": "PASS_NONBLOCKING" if probe_locks else "HELD_BY_PARENT_OR_SKIPPED_FOR_TEST",
        "optimization_N_F_budget": BUDGET,
        "runtime_argv_difference_count_from_R5B": 4,
        "runtime_output_directory": str(EXTERNAL_OUTPUT),
        "external_free_bytes": external_free,
        "native_save_all_paths": True,
        "full_path_energy_force_capture": True,
        "new_capture_queries": 0,
        "OpenMM_imported_or_called": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def configure_reused_components() -> None:
    RECORDER.OPTIMIZATION_NF_BUDGET = BUDGET
    RECORDER.MAXIMUM_RAW_CALLBACKS = 4 * BUDGET + 1
    RECORDER.MAXIMUM_CANONICAL_CALLS = BUDGET
    R5.OPTIMIZATION_NF_BUDGET = BUDGET
    R5.MAXIMUM_RAW_CALLBACKS = RECORDER.MAXIMUM_RAW_CALLBACKS
    R5.MAXIMUM_CANONICAL_CALLS = RECORDER.MAXIMUM_CANONICAL_CALLS
    R5.LABEL = LABEL
    R5.OUTPUT_SUBDIR = EXTERNAL_OUTPUT
    R5.OBSERVER_NAME = OBSERVER_NAME
    R5.SCHEMA = SCHEMA
    R5.AUTH_SCHEMA = AUTH_SCHEMA
    R5.MUTAGEN_GATE_SCHEMA = MUTAGEN_GATE_SCHEMA
    R5.MUTAGEN_NAME = MUTAGEN_NAME
    R5.MUTAGEN_IDENTIFIER = MUTAGEN_IDENTIFIER
    R5.STATUS = STATUS
    R5.CANONICAL_REGISTRATION = CANONICAL_REGISTRATION
    R5.CANONICAL_AUTHORIZATION = CANONICAL_AUTHORIZATION
    R5.CANONICAL_MUTAGEN_GATE = CANONICAL_MUTAGEN_GATE
    R5.source_paths = source_paths
    R5.validate_runtime_delta = validate_runtime_delta
    R5.validate_registration = validate_registration
    R5.validate_authorization = validate_authorization
    R5.preflight = preflight


def verify_mode(args) -> None:
    import verify_selected_ivm_full_state_history_replay_r5 as VERIFY

    VERIFY.RUNNER = R5
    VERIFY.RECORDER = RECORDER
    VERIFY.SCHEMA = VERIFY_SCHEMA
    VERIFY.STATUS = VERIFY_STATUS
    inputs = {
        name: value.expanduser().resolve(strict=True)
        for name, value in vars(args).items()
        if isinstance(value, Path) and name not in {"root", "output"}
    }
    payload = VERIFY.verify(
        args.root,
        inputs["registration"],
        inputs["authorization"],
        inputs["reference_result"],
        inputs["replay_result"],
        inputs["replay_audit"],
        inputs["capture_manifest"],
        inputs["canonical_index"],
        inputs["capture_audit"],
    )
    R5.BASE.write_exclusive(args.output, payload)
    destination = args.output.resolve(strict=True)
    print(destination)
    print(R5.BASE.sha256_file(destination))


def main(argv: Iterable[str] | None = None) -> None:
    configure_reused_components()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    modes = parser.add_subparsers(dest="mode", required=True)
    freeze_parser = modes.add_parser("freeze")
    freeze_parser.add_argument("--output", type=Path, required=True)
    auth_parser = modes.add_parser("authorize")
    auth_parser.add_argument("--registration", type=Path, required=True)
    auth_parser.add_argument("--output", type=Path, required=True)
    gate_parser = modes.add_parser("mutagen-gate")
    gate_parser.add_argument("--registration", type=Path, required=True)
    gate_parser.add_argument("--authorization", type=Path, required=True)
    gate_parser.add_argument("--output", type=Path, required=True)
    for name in ("preflight", "execute"):
        mode = modes.add_parser(name)
        mode.add_argument("--registration", type=Path, required=True)
        mode.add_argument("--authorization", type=Path, required=True)
        mode.add_argument("--mutagen-gate", type=Path, required=True)
    verify_parser = modes.add_parser("verify")
    for name in (
        "registration", "authorization", "reference_result", "replay_result",
        "replay_audit", "capture_manifest", "canonical_index", "capture_audit",
    ):
        verify_parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.root = args.root.expanduser().resolve(strict=True)
    if args.mode == "freeze":
        R5.BASE.require(args.output.expanduser().resolve() == CANONICAL_REGISTRATION.resolve(), "noncanonical R6 registration output")
        freeze(args.output)
    elif args.mode == "authorize":
        R5.BASE.require(args.output.expanduser().resolve() == CANONICAL_AUTHORIZATION.resolve(), "noncanonical R6 authorization output")
        authorize(args.registration, args.output)
    elif args.mode == "mutagen-gate":
        R5.flush_mutagen(args.registration, args.authorization, args.output)
    elif args.mode == "preflight":
        preflight(args.root, args.registration, args.authorization, args.mutagen_gate)
    elif args.mode == "execute":
        R5.execute(args.root, args.registration, args.authorization, args.mutagen_gate)
    else:
        verify_mode(args)


if __name__ == "__main__":
    main()
