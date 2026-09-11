#!/usr/bin/env python3
"""Create the canonical R4 execution authorization after bound zero-OpenMM preflight."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = ROOT / "scripts/evaluate_bb_irivm_nf8982_0k_section_energy_r4.py"
CONTRACT = ROOT / "registrations/diagnostics/BB_IRIVM_NF8982_0K_SECTION_ENERGY_CONTRACT_R4_20260902.json"
PREAUTH = ROOT / "registrations/diagnostics/BB_IRIVM_NF8982_0K_SECTION_ENERGY_PREAUTH_PREFLIGHT_R4_20260902.json"
AUTH = ROOT / "registrations/diagnostics/BB_IRIVM_NF8982_0K_SECTION_ENERGY_EXECUTION_AUTHORIZATION_R4_20260902.json"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("bb_nf8982_energy_r4_authorized", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--preflight", type=Path, default=PREAUTH)
    parser.add_argument("--output", type=Path, default=AUTH)
    args = parser.parse_args(argv)
    evaluator = load_evaluator()
    evaluator.check_thread_caps()
    contract_path = args.contract.expanduser().resolve(strict=True)
    preauth_path = args.preflight.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    evaluator.require(contract_path == CONTRACT.resolve(), "noncanonical R4 contract")
    evaluator.require(preauth_path == PREAUTH.resolve(), "noncanonical R4 preflight")
    evaluator.require(output == AUTH.resolve(), "noncanonical R4 authorization")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    preauth = json.loads(preauth_path.read_text(encoding="utf-8"))
    evaluator.require(contract.get("schema") == evaluator.CONTRACT_SCHEMA and contract.get("status") == evaluator.CONTRACT_STATUS, "R4 contract identity mismatch")
    evaluator.require(contract.get("evaluator") == evaluator.file_binding(EVALUATOR), "R4 evaluator binding mismatch")
    evaluator.require(contract.get("execution_authorizer") == evaluator.file_binding(Path(__file__).resolve()), "R4 authorizer binding mismatch")
    evaluator.require(preauth.get("schema") == evaluator.PREAUTH_SCHEMA and preauth.get("status") == evaluator.PREAUTH_STATUS, "R4 preflight identity mismatch")
    evaluator.require(preauth.get("contract") == evaluator.file_binding(contract_path), "R4 preflight contract mismatch")
    evaluator.require(preauth.get("evaluator") == evaluator.file_binding(EVALUATOR), "R4 preflight evaluator mismatch")
    expected_locks = {str(path) for path in evaluator.CANONICAL_LOCKS}
    evaluator.require(preauth.get("canonical_two_lock_nonblocking_probe", {}).get("status") == "PASS" and set(preauth["canonical_two_lock_nonblocking_probe"].get("locks", [])) == expected_locks, "R4 preflight lock set mismatch")
    evaluator.require(preauth.get("thread_policy") == {name: "1" for name in evaluator.THREAD_VARS}, "R4 preflight thread policy mismatch")
    evaluator.require(int(preauth.get("physical_calls_started", -1)) == 0 and int(preauth.get("openmm_contexts_constructed", -1)) == 0, "R4 preflight was not zero-OpenMM")
    evaluator.require(int(preauth.get("path_stage_fresh_attempt_maximum", -1)) == evaluator.PATH_INTERNAL_CALLS, "R4 path bound mismatch")
    evaluator.require(int(preauth.get("surface_stage_fresh_attempt_maximum", -1)) == evaluator.SURFACE_FRESH_CALLS, "R4 surface bound mismatch")
    evaluator.require(int(preauth.get("total_fresh_attempt_maximum", -1)) == evaluator.TOTAL_FRESH_CALLS, "R4 total bound mismatch")
    surface_output = Path(contract["result_output"]).resolve()
    path_output = Path(contract["original_path32_result_output"]).resolve()
    for result in (path_output, surface_output):
        for artifact in (result, result.with_suffix(".energy.wal.jsonl"), result.with_suffix(".audit.json")):
            evaluator.require(not artifact.exists(), f"physical artifact predates R4 authorization: {artifact}")
    locks = [Path(value).resolve() for value in contract["host_local_lock_paths"]]
    with evaluator.acquire_two_locks(locks) as held:
        second_probe = list(held)
    payload = {
        "schema": evaluator.AUTH_SCHEMA,
        "status": evaluator.AUTH_STATUS,
        "enabled": True,
        "authorized_utc": evaluator.utc_now(),
        "host_role": "local",
        "jobs": 1,
        "contract": evaluator.file_binding(contract_path),
        "evaluator": evaluator.file_binding(EVALUATOR),
        "authorizer": evaluator.file_binding(Path(__file__).resolve()),
        "zero_openmm_preflight": evaluator.file_binding(preauth_path),
        "authorized_stages": [evaluator.PATH_STAGE, evaluator.SURFACE_STAGE],
        "original_path32_result_output": str(path_output),
        "result_output": str(surface_output),
        "maximum_energy_only_attempts": evaluator.TOTAL_FRESH_CALLS,
        "maximum_surface_energy_only_attempts": evaluator.SURFACE_FRESH_CALLS,
        "maximum_original_internal_energy_only_attempts": evaluator.PATH_INTERNAL_CALLS,
        "maximum_force_calls": 0,
        "optimization_N_F_contribution": 0,
        "thread_policy": {name: "1" for name in evaluator.THREAD_VARS},
        "canonical_two_lock_second_probe": {"status": "PASS", "locks": second_probe},
        "physical_calls_started_before_authorization": 0,
        "failed_or_open_attempt_policy": "fail closed; require a superseding authorization and new output",
        "successful_prefix_resume_policy": "reuse only exact coordinate-hash-bound successful WAL entries",
    }
    atomic_json(output, payload)
    print(json.dumps({"status": evaluator.AUTH_STATUS, "authorization": str(output), "sha256": evaluator.sha256_file(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
