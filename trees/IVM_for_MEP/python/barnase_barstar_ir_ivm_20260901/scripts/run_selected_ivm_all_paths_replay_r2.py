#!/usr/bin/env python3
"""Prepare or run the selected BB IR-IVM with native all-path retention.

This is a thin, hash-bound specialization of the audited path-only replay.
It lowers only the conservative disk preflight threshold to 6 GiB; the child
argv still differs from the selected run only in label/output storage fields
and the native ``--save-all-paths`` flag.  No observer or evaluator hook is
installed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

import run_selected_ivm_full_history_replay as BASE


SCHEMA = "bb-irivm-selected-all-paths-output-only-replay-v2"
AUTH_SCHEMA = "bb-irivm-selected-all-paths-output-only-replay-authorization-v2"
STATUS = "FROZEN_BEFORE_FIRST_ALL_PATHS_REPLAY_PHYSICAL_CALL"
LABEL = "bb_irivm_a1e0_r01_m05_inf9990_allpaths_r2"
OUTPUT_SUBDIR = "runs/all_paths_replay_r2"
MINIMUM_FREE_BYTES = 6 * 1024**3
VERIFIER_NAME = "verify_selected_ivm_all_paths_replay_r2.py"
R4_FAILURE_RECEIPT = (
    "registrations/diagnostics/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_R4_FAILURE_SUPERSESSION_20260902.json"
)


@contextmanager
def _specialize_base():
    replacements = {
        "SCHEMA": SCHEMA,
        "AUTH_SCHEMA": AUTH_SCHEMA,
        "STATUS": STATUS,
        "LABEL": LABEL,
        "OUTPUT_SUBDIR": OUTPUT_SUBDIR,
        "MINIMUM_FREE_BYTES": MINIMUM_FREE_BYTES,
    }
    previous = {name: getattr(BASE, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(BASE, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(BASE, name, value)


def replay_argv(reference: dict) -> list[str]:
    with _specialize_base():
        return BASE.replay_argv(reference)


def validate_runtime_delta(reference: list[str], replay: list[str]) -> None:
    with _specialize_base():
        BASE.validate_runtime_delta(reference, replay)


def freeze(root: Path, output: Path) -> None:
    root = root.expanduser().resolve(strict=True)
    verifier = root / "scripts" / VERIFIER_NAME
    BASE.require(verifier.is_file(), f"missing R2 verifier: {verifier}")
    failure = root / R4_FAILURE_RECEIPT
    BASE.require(failure.is_file(), f"missing R4 failure receipt: {failure}")

    original_writer = BASE.write_exclusive

    emitted = []

    def augment_and_write(destination: Path, payload: dict) -> None:
        BASE.require(not emitted, "base freeze emitted more than one payload")
        payload["purpose"] = (
            "exact selected 9990-N_F IR-IVM replay using the production native "
            "--save-all-paths output policy to retain the stable-last path at calls 1..352"
        )
        payload["full_state_observer_installed"] = False
        payload["failed_full_state_outputs_reused"] = False
        payload["native_storage_only_proof"] = {
            "storage_policy": "legacy plus --save-all-paths",
            "decision_inputs": ["call", "is_initial", "reparam_committed"],
            "decision_effect": "append an already-current path.copy() to path_history",
            "online_path_energy_force_or_counter_mutation": False,
            "additional_OpenMM_calls": 0,
            "production_finalization": (
                "stable-last de-duplication by call_per_image at source lines 7673--7694"
            ),
            "expected_unique_calls": 352,
            "expected_stored_path_shape": [352, 32, 9396],
        }
        payload["resource_estimate"].update(
            {
                "minimum_free_disk_bytes_before_child": MINIMUM_FREE_BYTES,
                "raw_stored_path_bytes": 352 * 32 * 9396 * 8,
                "raw_stored_path_GiB": (352 * 32 * 9396 * 8) / 1024**3,
                "expected_result_disk_GiB": "0.8--1.1",
                "estimated_incremental_peak_RAM_GiB_from_full_history": "1.6--2.4",
                "why_6_GiB_is_fail_closed": (
                    "more than 5x the 1.1-GiB expected result plus room for the "
                    "in-memory 0.789-GiB path tensor and atomic result publication"
                ),
            }
        )
        payload["concurrency"]["minimum_free_disk_bytes_before_child"] = (
            MINIMUM_FREE_BYTES
        )
        payload["bindings"]["R4_failure_supersession_receipt"] = BASE.bind(
            failure
        )
        payload["source_sha256"][Path(__file__).name] = BASE.sha256_file(
            Path(__file__).resolve(strict=True)
        )
        payload["source_sha256"][VERIFIER_NAME] = BASE.sha256_file(verifier)
        original_writer(destination, payload)
        emitted.append(True)

    with _specialize_base():
        BASE.write_exclusive = augment_and_write
        try:
            BASE.freeze(root, output)
        finally:
            BASE.write_exclusive = original_writer
    BASE.require(emitted == [True], "base freeze did not emit one payload")


def validate_registration(root: Path, registration_path: Path) -> dict:
    with _specialize_base():
        registration = BASE.validate_registration(root, registration_path)
    BASE.require(
        registration.get("full_state_observer_installed") is False
        and registration.get("failed_full_state_outputs_reused") is False,
        "registration permits a failed observer or partial-output reuse",
    )
    proof = registration.get("native_storage_only_proof", {})
    BASE.require(
        proof.get("additional_OpenMM_calls") == 0
        and proof.get("online_path_energy_force_or_counter_mutation") is False
        and proof.get("expected_unique_calls") == 352,
        "native output-only storage proof is incomplete",
    )
    return registration


def authorize(root: Path, registration_path: Path, output: Path) -> None:
    with _specialize_base():
        BASE.authorize(root, registration_path, output)


def validate_authorization(registration_path: Path, authorization_path: Path) -> dict:
    authorization = BASE.read_json(authorization_path)
    BASE.require(
        authorization.get("schema") == AUTH_SCHEMA
        and authorization.get("enabled") is True,
        "authorization disabled or invalid",
    )
    BASE.require(
        authorization.get("label") == LABEL
        and authorization.get("registration") == BASE.bind(registration_path),
        "authorization registration binding mismatch",
    )
    registration = BASE.read_json(registration_path)
    runtime_digest = hashlib.sha256(
        json.dumps(
            registration["runtime_argv"], separators=(",", ":")
        ).encode()
    ).hexdigest()
    BASE.require(
        authorization.get("runtime_argv_sha256") == runtime_digest,
        "authorization runtime argv hash mismatch",
    )
    BASE.require(
        authorization.get("minimum_free_disk_bytes_before_child")
        == MINIMUM_FREE_BYTES,
        "authorization disk gate mismatch",
    )
    return authorization


def preflight(
    root: Path, registration_path: Path, authorization_path: Path
) -> None:
    registration_path = registration_path.expanduser().resolve(strict=True)
    authorization_path = authorization_path.expanduser().resolve(strict=True)
    registration = validate_registration(root, registration_path)
    authorization = validate_authorization(registration_path, authorization_path)
    invalid_caps = {
        name: os.environ.get(name)
        for name in BASE.THREAD_CAPS
        if os.environ.get(name) != "1"
    }
    BASE.require(not invalid_caps, f"all six thread caps must equal 1: {invalid_caps}")
    free = shutil.disk_usage(root).free
    BASE.require(free >= MINIMUM_FREE_BYTES, f"disk gate failed: {free}")
    output_dir = root / OUTPUT_SUBDIR
    BASE.require(
        not list(output_dir.glob(f"ivm_{LABEL}_*")),
        "refusing to overwrite an R2 all-path replay artifact",
    )
    print(
        json.dumps(
            {
                "status": "PASS_R2_ALL_PATHS_ZERO_OPENMM_PREFLIGHT",
                "label": LABEL,
                "registration": BASE.bind(registration_path),
                "authorization": BASE.bind(authorization_path),
                "runtime_argv": registration["runtime_argv"],
                "runtime_argv_difference_count": 4,
                "runtime_argv_differences": [
                    "--run-label",
                    "--output-dir",
                    "--path-storage-mode committed->legacy",
                    "--save-all-paths enabled",
                ],
                "optimization_N_F_budget": 9990,
                "expected_stored_paths": 352,
                "expected_stored_path_shape": [352, 32, 9396],
                "minimum_free_disk_bytes": MINIMUM_FREE_BYTES,
                "observed_free_disk_bytes": free,
                "full_state_observer_installed": False,
                "OpenMM_imported_or_called": False,
                "authorization_enabled": authorization["enabled"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def execute(
    root: Path, registration_path: Path, authorization_path: Path
) -> None:
    preflight(root, registration_path, authorization_path)
    with _specialize_base():
        BASE.execute(root, registration_path, authorization_path)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    freeze_mode = modes.add_parser("freeze")
    freeze_mode.add_argument("--output", type=Path, required=True)
    auth_mode = modes.add_parser("authorize")
    auth_mode.add_argument("--registration", type=Path, required=True)
    auth_mode.add_argument("--output", type=Path, required=True)
    for name in ("preflight", "execute"):
        mode = modes.add_parser(name)
        mode.add_argument("--registration", type=Path, required=True)
        mode.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root.expanduser().resolve(strict=True)
    if args.mode == "freeze":
        freeze(root, args.output)
    elif args.mode == "authorize":
        authorize(root, args.registration, args.output)
    elif args.mode == "preflight":
        preflight(root, args.registration, args.authorization)
    else:
        execute(root, args.registration, args.authorization)


if __name__ == "__main__":
    main()
