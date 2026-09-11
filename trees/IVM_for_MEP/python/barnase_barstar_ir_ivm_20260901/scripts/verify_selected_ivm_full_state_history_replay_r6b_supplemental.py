#!/usr/bin/env python3
"""Zero-OpenMM terminal accounting supplement for the R6B N_F=30000 replay.

The primary R6B verifier proves exact-prefix reproduction and binds every
canonical path/energy/force shard to the producer result.  This supplement
adds the terminal budget-atomicity, fixed-endpoint NoF, launch-lock, and exact
counter-schema closure needed for final acceptance.  It only reads artifacts
already written by the completed run and never imports or calls OpenMM.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

import run_selected_ivm_full_state_history_replay_r6_nf30000 as RUNNER


import verify_long_budget_member as BUDGET  # noqa: E402
import verify_selected_ivm_full_state_history_replay_r5 as PRIMARY  # noqa: E402
import verify_selected_ivm_full_state_history_replay_r5b_supplemental as BASE  # noqa: E402


SCHEMA = "bb-irivm-full-state-r6b-supplemental-verification-v1"
STATUS = "PASS_R6B_NF30000_COUNTER_BUDGET_LOCK_AND_ARTIFACT_CLOSURE"


def scalar(saved, key: str):
    return PRIMARY.scalar(saved, key)


def expected_launch_command(root: Path) -> list[str]:
    return [
        "/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python",
        "scripts/run_selected_ivm_full_state_history_replay_r6_nf30000.py",
        "execute",
        "--registration",
        str(RUNNER.CANONICAL_REGISTRATION.relative_to(root)),
        "--authorization",
        str(RUNNER.CANONICAL_AUTHORIZATION.relative_to(root)),
        "--mutagen-gate",
        str(RUNNER.CANONICAL_MUTAGEN_GATE.relative_to(root)),
    ]


def validate_terminal_lock_receipt(root: Path, lock_receipt: dict) -> None:
    BASE.BASE.require(
        lock_receipt.get("schema") == "bb-host-local-lock-launch-receipt-v1"
        and lock_receipt.get("status") == "CHILD_EXITED_LOCKS_RELEASED"
        and lock_receipt.get("returncode") == 0,
        "R6B locked child lacks a clean terminal receipt",
    )
    BASE.BASE.require(
        lock_receipt.get("command") == expected_launch_command(root)
        and lock_receipt.get("working_directory") == str(root)
        and lock_receipt.get("lock_paths")
        == [str(path) for path in RUNNER.R5.BASE.LOCKS]
        and lock_receipt.get("host", {}).get("hostname") == RUNNER.R5.REMOTE_HOSTNAME
        and lock_receipt.get("host", {}).get("architecture")
        == RUNNER.R5.REMOTE_MACHINE,
        "R6B lock receipt command, host, working directory, or lock set drifted",
    )


def validate_runtime_counter_and_endpoint_accounting(
    runtime_audit: dict, result_path: Path
) -> tuple[int, np.ndarray]:
    BASE.BASE.require(
        runtime_audit.get("consistent") is True
        and Path(str(runtime_audit.get("result_file"))).resolve() == result_path,
        "R6B runtime result/audit pair is inconsistent",
    )
    actual = int(runtime_audit["optimization_internal_nf"])
    for key in (
        "backend_raw_joint_attempts",
        "backend_raw_joint_evaluations",
        "sum_image_joint_evaluations",
        "number_of_forces",
    ):
        BASE.BASE.require(
            int(runtime_audit[key]) == actual,
            f"R6B runtime counter mismatch: {key}",
        )
    per_image = np.asarray(
        runtime_audit["per_image_joint_evaluations"], dtype=np.int64
    )
    BASE.BASE.require(
        per_image.shape == (32,)
        and int(per_image[0]) == 0
        and int(per_image[-1]) == 0
        and int(per_image[1:-1].sum(dtype=np.int64)) == actual,
        "R6B fixed-endpoint or internal-image NoF closure failed",
    )
    BASE.BASE.require(
        int(runtime_audit["endpoint_getstate_attempts"]) == 0
        and int(runtime_audit["endpoint_joint_ef_cache_misses"]) == 0
        and int(runtime_audit["endpoint_inherited_joint_ef_cache_entries"]) == 2,
        "R6B endpoint-cache policy drifted",
    )
    BASE.BASE.require(
        all(runtime_audit["per_image_request_identity_ok"])
        and all(runtime_audit["per_image_query_outcome_identity_ok"])
        and runtime_audit["discrete_all_returned_states_evaluated"] is True,
        "R6B request identity or evaluated-state audit failed",
    )
    return actual, per_image


def validate_shard_fixed_endpoints(joint, call: int) -> None:
    vector = np.asarray(joint, dtype=np.int64)
    BASE.BASE.require(
        vector.shape == (32,)
        and int(vector[0]) == 0
        and int(vector[-1]) == 0,
        f"R6B shard charges a fixed endpoint at call {call}",
    )


def verify(
    root: Path,
    registration_path: Path,
    authorization_path: Path,
    primary_path: Path,
    manifest_path: Path,
    canonical_path: Path,
    capture_audit_path: Path,
    result_path: Path,
    runtime_audit_path: Path,
    lock_receipt_path: Path,
) -> dict:
    # Reuse the exhaustive R5B shard/WAL/counter-schema audit under R6B's
    # registered validator and primary PASS identity.
    RUNNER.configure_reused_components()
    BASE.RUNNER = RUNNER
    BASE.RECORDER = RUNNER.RECORDER
    BASE.SCHEMA = SCHEMA
    BASE.STATUS = STATUS
    BASE.PRIMARY.SCHEMA = RUNNER.VERIFY_SCHEMA
    BASE.PRIMARY.STATUS = RUNNER.VERIFY_STATUS
    payload = BASE.verify(
        root,
        registration_path,
        authorization_path,
        primary_path,
        manifest_path,
        canonical_path,
        capture_audit_path,
    )

    registration = RUNNER.validate_registration(root, registration_path)
    RUNNER.validate_authorization(registration_path, authorization_path, registration)
    runtime_audit = BASE.BASE.read_json(runtime_audit_path)
    lock_receipt = BASE.BASE.read_json(lock_receipt_path)
    primary = BASE.BASE.read_json(primary_path)
    canonical = BASE.BASE.read_json(canonical_path)
    atomicity_path = (
        root / "registrations/USER_BUDGET_ATOMICITY_AMENDMENT_20260901_R1.json"
    ).resolve(strict=True)
    atomicity = BASE.BASE.read_json(atomicity_path)

    validate_terminal_lock_receipt(root, lock_receipt)
    BASE.BASE.require(
        atomicity.get("schema") == "bb-irivm-budget-atomicity-amendment-v1"
        and atomicity.get("status") == "OPERATIVE_ACCOUNTING_CORRECTION",
        "R6B budget-atomicity policy is not operative",
    )
    actual, _ = validate_runtime_counter_and_endpoint_accounting(
        runtime_audit, result_path
    )

    with np.load(result_path, allow_pickle=False) as saved:
        BASE.BASE.require(
            str(scalar(saved, "run_label")) == RUNNER.LABEL,
            "R6B result label mismatch",
        )
        for key in (
            "number_of_forces",
            "successful_joint_ef_evaluations",
            "optimization_internal_nf",
            "total_internal_nf_consumed",
        ):
            BASE.BASE.require(
                int(scalar(saved, key)) == actual,
                f"R6B result counter mismatch: {key}",
            )
        BASE.BASE.require(
            int(scalar(saved, "total_internal_nf_budget")) == RUNNER.BUDGET
            and bool(scalar(saved, "total_internal_nf_budget_no_overshoot"))
            and int(scalar(saved, "total_internal_nf_remaining"))
            == RUNNER.BUDGET - actual,
            "R6B budget accounting drifted",
        )
        BASE.BASE.require(
            bool(scalar(saved, "total_internal_nf_endpoint_evaluations_excluded"))
            and int(
                scalar(saved, "total_internal_nf_endpoint_getstate_attempts_expected")
            )
            == 0,
            "R6B result does not exclude fixed endpoints from NoF",
        )
        partial, quarantined, atomicity_case = BUDGET.validate_budget_atomicity(
            saved, actual=actual, budget=RUNNER.BUDGET
        )

    records = canonical.get("records", [])
    BASE.BASE.require(records, "R6B canonical record inventory is empty")
    for record in records:
        with np.load(Path(record["shard"]), allow_pickle=False) as shard:
            validate_shard_fixed_endpoints(
                shard["joint_evaluation_vector"], int(record["call_per_image"])
            )

    BASE.BASE.require(
        int(primary["terminal_optimization_N_F"]) == actual
        and int(primary["actual_canonical_call_count"]) == len(records),
        "R6B primary terminal counters differ from producer/canonical artifacts",
    )
    payload.update(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "registered_optimization_N_F_upper_bound": RUNNER.BUDGET,
            "actual_optimization_N_F": actual,
            "remaining_registered_N_F": RUNNER.BUDGET - actual,
            "budget_atomicity_case": atomicity_case,
            "terminal_partial_tail_present": partial,
            "terminal_partial_tail_quarantined": quarantined,
            "fixed_endpoint_optimization_N_F": 0,
            "all_canonical_shard_endpoint_counters_checked": True,
            "lock_receipt_terminal_and_returncode_zero": True,
            "OpenMM_imported_or_called": False,
        }
    )
    payload["bindings"].update(
        {
            "R6B_runner": BASE.BASE.bind(Path(RUNNER.__file__).resolve(strict=True)),
            "R6B_result": BASE.BASE.bind(result_path),
            "R6B_runtime_audit": BASE.BASE.bind(runtime_audit_path),
            "R6B_lock_receipt": BASE.BASE.bind(lock_receipt_path),
            "budget_atomicity_amendment": BASE.BASE.bind(
                atomicity_path
            ),
            "supplemental_verifier": BASE.BASE.bind(Path(__file__).resolve(strict=True)),
        }
    )
    return payload


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--primary-verification", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--canonical-index", type=Path, required=True)
    parser.add_argument("--capture-audit", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument("--lock-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root.expanduser().resolve(strict=True)
    inputs = {
        name: value.expanduser().resolve(strict=True)
        for name, value in vars(args).items()
        if isinstance(value, Path) and name not in {"root", "output"}
    }
    payload = verify(
        root,
        inputs["registration"],
        inputs["authorization"],
        inputs["primary_verification"],
        inputs["manifest"],
        inputs["canonical_index"],
        inputs["capture_audit"],
        inputs["result"],
        inputs["runtime_audit"],
        inputs["lock_receipt"],
    )
    BASE.BASE.write_exclusive(args.output, payload)
    output = args.output.expanduser().resolve(strict=True)
    print(output)
    print(BASE.BASE.sha256_file(output))


if __name__ == "__main__":
    main()
