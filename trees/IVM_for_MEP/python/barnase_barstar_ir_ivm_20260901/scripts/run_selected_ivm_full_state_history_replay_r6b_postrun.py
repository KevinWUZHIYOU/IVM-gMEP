#!/usr/bin/env python3
"""Discover and verify the completed R6B artifacts without physical queries.

``status`` is safe while the optimizer is running and reports readiness only.
``verify`` is fail-closed until the locked child has exited cleanly and the
producer has atomically published its result, audit, manifest, canonical index,
and capture audit.  Verification imports no simulation backend and performs no
energy or force call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import run_selected_ivm_full_state_history_replay_r6_nf30000 as RUNNER


ROOT = Path(__file__).resolve().parents[1]
OUT = RUNNER.EXTERNAL_OUTPUT
OBSERVER = OUT / RUNNER.OBSERVER_NAME
REGISTRATION = RUNNER.CANONICAL_REGISTRATION
AUTHORIZATION = RUNNER.CANONICAL_AUTHORIZATION
MUTAGEN_GATE = RUNNER.CANONICAL_MUTAGEN_GATE
LOCK_RECEIPT = RUNNER.LAUNCH_RECEIPT
PRIMARY = RUNNER.CANONICAL_VERIFICATION
SUPPLEMENTAL = ROOT / (
    "reports/verifications/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF30000_20260904_R6B_SUPPLEMENTAL.json"
)


def discover() -> dict:
    RUNNER.configure_reused_components()
    registration = RUNNER.validate_registration(ROOT, REGISTRATION)
    RUNNER.validate_authorization(REGISTRATION, AUTHORIZATION, registration)
    RUNNER.R5.validate_mutagen(REGISTRATION, AUTHORIZATION, MUTAGEN_GATE)
    receipt = RUNNER.R5.BASE.read_json(LOCK_RECEIPT)
    results = sorted(OUT.glob(f"ivm_{RUNNER.LABEL}_*.npz")) if OUT.is_dir() else []
    audits = (
        sorted(OUT.glob(f"ivm_{RUNNER.LABEL}_*.audit.json"))
        if OUT.is_dir()
        else []
    )
    terminal = {
        "manifest": OBSERVER / "full_state_history.manifest.json",
        "canonical_index": OBSERVER / "full_state_history.canonical_index.json",
        "capture_audit": OBSERVER / "full_state_history.audit.json",
    }
    child_clean = (
        receipt.get("status") == "CHILD_EXITED_LOCKS_RELEASED"
        and receipt.get("returncode") == 0
    )
    producer_complete = (
        len(results) == 1
        and len(audits) == 1
        and all(path.is_file() for path in terminal.values())
    )
    return {
        "schema": "bb-irivm-r6b-postrun-readiness-v1",
        "status": "READY_FOR_ZERO_QUERY_VERIFICATION"
        if child_clean and producer_complete
        else "RUNNING_OR_PRODUCER_FINALIZATION_INCOMPLETE",
        "OpenMM_imported_or_called": False,
        "child_receipt_status": receipt.get("status"),
        "child_returncode": receipt.get("returncode"),
        "runtime_results": [str(path) for path in results],
        "runtime_audits": [str(path) for path in audits],
        "capture_terminal_files": {
            name: {"path": str(path), "exists": path.is_file()}
            for name, path in terminal.items()
        },
        "primary_verification": {
            "path": str(PRIMARY),
            "exists": PRIMARY.is_file(),
        },
        "supplemental_verification": {
            "path": str(SUPPLEMENTAL),
            "exists": SUPPLEMENTAL.is_file(),
        },
        "ready": child_clean and producer_complete,
    }


def run_verification(readiness: dict) -> None:
    RUNNER.R5.BASE.require(readiness["ready"], "R6B producer is not terminal-ready")
    results = [Path(value) for value in readiness["runtime_results"]]
    audits = [Path(value) for value in readiness["runtime_audits"]]
    terminal = {
        name: Path(value["path"])
        for name, value in readiness["capture_terminal_files"].items()
    }
    registration = RUNNER.R5.BASE.read_json(REGISTRATION)
    reference = Path(registration["bindings"]["selected_terminal_result"]["path"])
    if not PRIMARY.exists():
        subprocess.run(
            [
                sys.executable,
                str(Path(RUNNER.__file__).resolve()),
                "verify",
                "--root",
                str(ROOT),
                "--registration",
                str(REGISTRATION),
                "--authorization",
                str(AUTHORIZATION),
                "--reference-result",
                str(reference),
                "--replay-result",
                str(results[0]),
                "--replay-audit",
                str(audits[0]),
                "--capture-manifest",
                str(terminal["manifest"]),
                "--canonical-index",
                str(terminal["canonical_index"]),
                "--capture-audit",
                str(terminal["capture_audit"]),
                "--output",
                str(PRIMARY),
            ],
            check=True,
        )
    if not SUPPLEMENTAL.exists():
        subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "scripts/verify_selected_ivm_full_state_history_replay_r6b_supplemental.py"
                ),
                "--root",
                str(ROOT),
                "--registration",
                str(REGISTRATION),
                "--authorization",
                str(AUTHORIZATION),
                "--primary-verification",
                str(PRIMARY),
                "--manifest",
                str(terminal["manifest"]),
                "--canonical-index",
                str(terminal["canonical_index"]),
                "--capture-audit",
                str(terminal["capture_audit"]),
                "--result",
                str(results[0]),
                "--runtime-audit",
                str(audits[0]),
                "--lock-receipt",
                str(LOCK_RECEIPT),
                "--output",
                str(SUPPLEMENTAL),
            ],
            check=True,
        )
    primary = RUNNER.R5.BASE.read_json(PRIMARY)
    supplemental = RUNNER.R5.BASE.read_json(SUPPLEMENTAL)
    RUNNER.R5.BASE.require(
        primary.get("schema") == RUNNER.VERIFY_SCHEMA
        and primary.get("status") == RUNNER.VERIFY_STATUS,
        "R6B primary verification is not the frozen PASS identity",
    )
    RUNNER.R5.BASE.require(
        supplemental.get("schema")
        == "bb-irivm-full-state-r6b-supplemental-verification-v1"
        and supplemental.get("status")
        == "PASS_R6B_NF30000_COUNTER_BUDGET_LOCK_AND_ARTIFACT_CLOSURE",
        "R6B supplemental verification is not PASS",
    )
    print(
        json.dumps(
            {
                "status": "PASS_R6B_TERMINAL_ACCEPTANCE",
                "primary": RUNNER.R5.BASE.bind(PRIMARY),
                "supplemental": RUNNER.R5.BASE.bind(SUPPLEMENTAL),
                "actual_optimization_N_F": supplemental["actual_optimization_N_F"],
                "remaining_registered_N_F": supplemental[
                    "remaining_registered_N_F"
                ],
                "canonical_record_count": supplemental[
                    "canonical_record_count"
                ],
                "additional_physical_calls": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("status", "verify"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    readiness = discover()
    if args.mode == "status":
        print(json.dumps(readiness, indent=2, sort_keys=True))
        return
    run_verification(readiness)


if __name__ == "__main__":
    main()
