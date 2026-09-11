#!/usr/bin/env python3
"""Freeze the pre-result package or post-selection IR late-six contract.

``preregister-package`` binds every source, input, remote-host gate, Mutagen
session, storage rule, and physical-attempt cap before the all-path result is
available.  Later, ``freeze`` flushes and verifies the dedicated Mutagen
session, validates the already-frozen blind selection and exact replay receipt,
loads the six selected paths, then writes the immutable execution contract.
Neither mode constructs OpenMM or acquires a heavy-task lock.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable


HERE = Path(__file__).resolve().parent
EVALUATOR_PATH = HERE / "evaluate_bb_irivm_late_six_joint_ef.py"
MUTAGEN = Path("/Users/wuzhiyou/.local/bin/mutagen")


def _load():
    spec = importlib.util.spec_from_file_location(
        "bb_ir_late_six_evaluator_for_freeze", EVALUATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load()


def write_exclusive(path: Path, payload: dict) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, destination)
        os.chmod(destination, 0o444)
    finally:
        temporary.unlink()


def flush_and_query_mutagen() -> dict:
    executable = MUTAGEN.expanduser().resolve(strict=True)
    flush = subprocess.run(
        [str(executable), "sync", "flush", EVALUATOR.MUTAGEN_SESSION_NAME],
        check=False,
        text=True,
        capture_output=True,
    )
    if flush.returncode != 0:
        raise RuntimeError(
            f"dedicated Mutagen flush failed: {flush.stderr.strip()}"
        )
    template = (
        "{{ range . }}{{ .Name }}|{{ .Identifier }}|{{ .Status }}|"
        "{{ .Alpha.Connected }}|{{ .Beta.Connected }}\\n{{ end }}"
    )
    listing = subprocess.run(
        [str(executable), "sync", "list", "--template", template],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [line.split("|") for line in listing.stdout.splitlines()]
    matches = [row for row in rows if row and row[0] == EVALUATOR.MUTAGEN_SESSION_NAME]
    if len(matches) != 1 or len(matches[0]) != 5:
        raise RuntimeError("dedicated Mutagen session not uniquely listed")
    name, identifier, status, alpha, beta = matches[0]
    raw = subprocess.run(
        [str(executable), "sync", "list", EVALUATOR.MUTAGEN_SESSION_NAME],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    lower = raw.lower()
    transition_problems = int(
        "problems:" in lower or "conflicts:" in lower or "error:" in lower
    )
    snapshot = {
        "session_name": name,
        "session_identifier": identifier,
        "flush_exit_code": int(flush.returncode),
        "status": status,
        "alpha_connected": alpha.lower() == "true",
        "beta_connected": beta.lower() == "true",
        "transition_problems": transition_problems,
        "mutagen_executable": str(executable),
        "mutagen_executable_sha256": EVALUATOR.ENGINE.sha256_file(executable),
    }
    EVALUATOR.validate_mutagen_snapshot(snapshot)
    return snapshot


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    preregister = modes.add_parser("preregister-package")
    preregister.add_argument("--output", type=Path, required=True)
    preregister.add_argument("--global-lock-path", type=Path, required=True)
    preregister.add_argument("--peer-global-lock-path", type=Path, required=True)
    check = modes.add_parser("package-preflight")
    check.add_argument("--package", type=Path, required=True)
    freeze = modes.add_parser("freeze")
    freeze.add_argument("--package", type=Path, required=True)
    freeze.add_argument("--selection", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.mode == "preregister-package":
        output = args.output.expanduser().resolve()
        if output != EVALUATOR.CANONICAL_PACKAGE_PREREGISTRATION:
            raise ValueError("package preregistration output must be canonical")
        payload = EVALUATOR.build_package_preregistration(
            [args.global_lock_path, args.peer_global_lock_path]
        )
        write_exclusive(output, payload)
        print(
            json.dumps(
                {
                    "status": "FROZEN_ZERO_PHYSICAL_PACKAGE_PREREGISTRATION",
                    "path": str(output),
                    "sha256": EVALUATOR.ENGINE.sha256_file(output),
                    "result_or_selection_read": False,
                    "OpenMM_constructed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.mode == "package-preflight":
        package = args.package.expanduser().resolve(strict=True)
        payload = EVALUATOR.validate_package_preregistration(package)
        print(
            json.dumps(
                {
                    "status": "PASS_ZERO_OPENMM_PACKAGE_PREFLIGHT",
                    "package": EVALUATOR.bind(package),
                    "host_role": payload["host_role"],
                    "remote_identity": payload["remote_identity"],
                    "required_Mutagen_session": payload[
                        "required_Mutagen_session"
                    ],
                    "stored_joint_EF_values_future": EVALUATOR.TOTAL_STORED_JOINT_EF,
                    "maximum_fresh_attempts_future": EVALUATOR.MAXIMUM_FRESH_ATTEMPT_STARTED,
                    "result_or_selection_read": False,
                    "OpenMM_constructed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        package = args.package.expanduser().resolve(strict=True)
        selection = args.selection.expanduser().resolve(strict=True)
        output = args.output.expanduser().resolve()
        if package != EVALUATOR.CANONICAL_PACKAGE_PREREGISTRATION:
            raise ValueError("final freeze requires the canonical package preregistration")
        if selection != EVALUATOR.CANONICAL_SELECTION:
            raise ValueError("final freeze requires the canonical blind selection")
        if output != EVALUATOR.CANONICAL_CONTRACT:
            raise ValueError("final contract output must be canonical")
        snapshot = flush_and_query_mutagen()
        payload = EVALUATOR.build_final_contract(package, selection, snapshot)
        write_exclusive(output, payload)
        # The first flush proves that the selected replay inputs are stable
        # locally.  This second flush publishes the newly frozen contract to
        # the remote endpoint before its host-gated preflight is attempted.
        post_publish_snapshot = flush_and_query_mutagen()
        print(
            json.dumps(
                {
                    "status": "FROZEN_AFTER_BLIND_SELECTION_ZERO_OPENMM",
                    "path": str(output),
                    "sha256": EVALUATOR.ENGINE.sha256_file(output),
                    "selected_optimization_N_F": [
                        row["optimization_N_F"] for row in payload["paths"]
                    ],
                    "stored_joint_EF_values": EVALUATOR.TOTAL_STORED_JOINT_EF,
                    "fresh_joint_EF_attempt_cap": EVALUATOR.MAXIMUM_FRESH_ATTEMPT_STARTED,
                    "post_contract_publish_Mutagen_gate": post_publish_snapshot,
                    "OpenMM_constructed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
