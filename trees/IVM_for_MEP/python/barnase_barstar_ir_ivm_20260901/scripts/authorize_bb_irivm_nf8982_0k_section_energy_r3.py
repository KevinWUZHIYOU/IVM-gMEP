#!/usr/bin/env python3
"""Authorize the frozen R3 diagnostic energy stages after zero-OpenMM preflight."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EVALUATOR_PATH = ROOT / "scripts/evaluate_bb_irivm_nf8982_0k_section_energy.py"
CANONICAL_CONTRACT = ROOT / "registrations/diagnostics/BB_IRIVM_NF8982_0K_SECTION_ENERGY_CONTRACT_R3_20260902.json"
CANONICAL_PREAUTH = ROOT / "registrations/diagnostics/BB_IRIVM_NF8982_0K_SECTION_ENERGY_PREAUTH_PREFLIGHT_R3_20260902.json"
CANONICAL_AUTHORIZATION = ROOT / "registrations/diagnostics/BB_IRIVM_NF8982_0K_SECTION_ENERGY_EXECUTION_AUTHORIZATION_R3_20260902.json"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("bb_nf8982_energy_authorizer", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def atomic_json_no_overwrite(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale temporary file: {temporary}")
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CANONICAL_CONTRACT)
    parser.add_argument("--preflight", type=Path, default=CANONICAL_PREAUTH)
    parser.add_argument("--output", type=Path, default=CANONICAL_AUTHORIZATION)
    args = parser.parse_args(argv)
    evaluator = load_evaluator()
    evaluator.check_thread_caps()
    contract_path = args.contract.expanduser().resolve(strict=True)
    preflight_path = args.preflight.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if contract_path != CANONICAL_CONTRACT.resolve() or preflight_path != CANONICAL_PREAUTH.resolve():
        raise ValueError("noncanonical R3 contract or preflight")
    if output != CANONICAL_AUTHORIZATION.resolve():
        raise ValueError("noncanonical R3 authorization destination")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != evaluator.CONTRACT_SCHEMA or contract.get("status") != evaluator.CONTRACT_STATUS:
        raise ValueError("contract is not frozen")
    if contract.get("evaluator") != evaluator.file_binding(EVALUATOR_PATH):
        raise ValueError("contract evaluator binding mismatch")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("schema") != evaluator.PREAUTH_SCHEMA or preflight.get("status") != evaluator.PREAUTH_STATUS:
        raise ValueError("preauthorization preflight did not pass")
    if preflight.get("contract") != evaluator.file_binding(contract_path):
        raise ValueError("preflight contract binding mismatch")
    if preflight.get("evaluator") != evaluator.file_binding(EVALUATOR_PATH):
        raise ValueError("preflight evaluator binding mismatch")
    if preflight.get("canonical_two_lock_nonblocking_probe", {}).get("status") != "PASS":
        raise ValueError("canonical lock probe did not pass")
    if int(preflight.get("physical_calls_started", -1)) != 0 or int(preflight.get("openmm_contexts_constructed", -1)) != 0:
        raise ValueError("preflight was not zero-OpenMM")
    surface_output = Path(contract["result_output"]).resolve()
    path_output = Path(contract["original_path32_result_output"]).resolve()
    for result in (path_output, surface_output):
        for artifact in (result, result.with_suffix(".energy.wal.jsonl"), result.with_suffix(".audit.json")):
            if artifact.exists():
                raise FileExistsError(f"physical artifact predates authorization: {artifact}")
    locks = [Path(value).resolve() for value in contract["host_local_lock_paths"]]
    with evaluator.acquire_two_locks(locks) as probed_locks:
        lock_probe = list(probed_locks)
    payload = {
        "schema": evaluator.AUTH_SCHEMA,
        "status": evaluator.AUTH_STATUS,
        "enabled": True,
        "authorized_utc": evaluator.utc_now(),
        "host_role": "local",
        "jobs": 1,
        "contract": evaluator.file_binding(contract_path),
        "evaluator": evaluator.file_binding(EVALUATOR_PATH),
        "zero_openmm_preflight": evaluator.file_binding(preflight_path),
        "authorized_stages": [evaluator.PATH_STAGE, evaluator.SURFACE_STAGE],
        "original_path32_result_output": str(path_output),
        "result_output": str(surface_output),
        "maximum_energy_only_attempts": evaluator.TOTAL_FRESH_CALLS,
        "maximum_surface_energy_only_attempts": evaluator.SURFACE_FRESH_CALLS,
        "maximum_original_internal_energy_only_attempts": evaluator.PATH_INTERNAL_CALLS,
        "maximum_force_calls": 0,
        "optimization_N_F_contribution": 0,
        "failed_or_open_attempt_policy": "fail closed; require a superseding authorization and new output",
        "successful_prefix_resume_policy": "reuse only exact coordinate-hash-bound successful WAL entries",
        "thread_policy": {name: "1" for name in evaluator.THREAD_VARS},
        "canonical_two_lock_second_probe": {"status": "PASS", "locks": lock_probe},
        "physical_calls_started_before_authorization": 0,
    }
    atomic_json_no_overwrite(output, payload)
    print(json.dumps({"status": evaluator.AUTH_STATUS, "authorization": str(output), "sha256": evaluator.sha256_file(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
