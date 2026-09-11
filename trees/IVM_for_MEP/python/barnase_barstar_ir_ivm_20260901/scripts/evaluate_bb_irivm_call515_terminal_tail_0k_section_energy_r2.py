#!/usr/bin/env python3
"""Evaluate the call515-centered diagnostic 59x28 surface, energy only.

This is a versioned wrapper around the tested NF8982 evaluator machinery.  It
changes the bound coordinates/state semantics and removes the separate path
stage.  Exactly 1,650 new surface energies are authorized; the two fixed
endpoint energies are inherited.  No Force is requested and optimization NoF
is unchanged.  The returned center remains explicitly uncommitted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np

import freeze_bb_irivm_call515_terminal_tail_0k_section_r2 as geometry


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
BASE = ROOT / "scripts/evaluate_bb_irivm_nf8982_0k_section_energy.py"
BASE_SHA256 = "af17cda66bee788d23b3948f6a0cb0ebad9ab8d061fac6cbf62a96e3a0000c9c"
SCHEMA_STEM = "bb-irivm-call515-terminal-tail-section-energy-r2-v1"
CONTRACT_SCHEMA = SCHEMA_STEM + "-contract"
CONTRACT_STATUS = "FROZEN_BEFORE_FIRST_CALL515_SECTION_ENERGY_CALL"
AUTH_SCHEMA = SCHEMA_STEM + "-authorization"
AUTH_STATUS = "AUTHORIZED_AFTER_ZERO_OPENMM_PREFLIGHT"
PREAUTH_SCHEMA = SCHEMA_STEM + "-preauthorization-preflight"
PREAUTH_STATUS = "PASS_ZERO_OPENMM_BEFORE_AUTHORIZATION"
CALL_POLICY_SCHEMA = SCHEMA_STEM + "-call-policy"
CALL_POLICY_STATUS = "FROZEN_BEFORE_FIRST_CALL515_SECTION_ENERGY_CALL"
RESULT_SCHEMA = SCHEMA_STEM + "-result"
AUDIT_SCHEMA = SCHEMA_STEM + "-audit"
WAL_SCHEMA = SCHEMA_STEM + "-wal"

REGISTRATION_DIR = ROOT / "registrations/diagnostics"
OUTPUT_DIR = ROOT / "runs/diagnostics"
CALL_POLICY = REGISTRATION_DIR / (
    "BB_IRIVM_CALL515_TERMINAL_UNCOMMITTED_TAIL_0K_SECTION_ENERGY_"
    "CALL_POLICY_R2B_20260903.json"
)
CONTRACT = REGISTRATION_DIR / (
    "BB_IRIVM_CALL515_TERMINAL_UNCOMMITTED_TAIL_0K_SECTION_ENERGY_"
    "CONTRACT_R2B_20260903.json"
)
PREAUTH = REGISTRATION_DIR / (
    "BB_IRIVM_CALL515_TERMINAL_UNCOMMITTED_TAIL_0K_SECTION_ENERGY_"
    "PREAUTH_R2B_20260903.json"
)
AUTHORIZATION = REGISTRATION_DIR / (
    "BB_IRIVM_CALL515_TERMINAL_UNCOMMITTED_TAIL_0K_SECTION_ENERGY_"
    "AUTHORIZATION_R2B_20260903.json"
)
OUTPUT = OUTPUT_DIR / (
    "bb_irivm_call515_nf14983_terminal_uncommitted_tail_0k_section_"
    "energy_59x28_20260903_r2b.npz"
)
UNUSED_PATH_OUTPUT = OUTPUT_DIR / (
    "bb_irivm_call515_nf14983_terminal_uncommitted_tail_unused_path_stage_r2b.npz"
)
THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENMM_CPU_THREADS",
)
LOCKS = geometry.LOCKS
SURFACE_VALUES = geometry.POINTS * geometry.NORMAL_POINTS
FRESH_CALLS = SURFACE_VALUES - 2


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size}


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def probe_locks() -> list[str]:
    streams = []
    try:
        for path in LOCKS:
            stream = path.open("a+")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                stream.close()
                raise RuntimeError(f"another BB task owns {path}") from error
            streams.append(stream)
        return [str(path) for path in LOCKS]
    finally:
        for stream in reversed(streams):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


def prepare_chain() -> None:
    require(sha256_file(BASE) == BASE_SHA256, "tested base evaluator changed")
    coordinate = geometry.OUTPUT.resolve(strict=True)
    coordinate_audit = geometry.AUDIT.resolve(strict=True)
    geometry.validate_coordinate_manifest_for_energy(coordinate, sha256_file(coordinate))
    audit = json.loads(coordinate_audit.read_text(encoding="utf-8"))
    require(audit.get("status") == geometry.AUDIT_STATUS
            and audit.get("coordinate_manifest") == binding(coordinate),
            "coordinate audit mismatch")
    outputs = (CALL_POLICY, CONTRACT, PREAUTH, AUTHORIZATION, OUTPUT,
               OUTPUT.with_suffix(".audit.json"), OUTPUT.with_suffix(".energy.wal.jsonl"))
    require(not any(path.exists() for path in outputs), "R2 energy chain output already exists")
    locks = probe_locks()
    free = shutil.disk_usage(OUTPUT.parent).free
    require(free >= 8 * 1024**3 + 320 * 1024**2, "insufficient free disk")
    thread_policy = {name: "1" for name in THREAD_VARS}
    call_policy_payload = {
        "schema": CALL_POLICY_SCHEMA, "status": CALL_POLICY_STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state_role": "returned terminal state; terminal uncommitted charged tail; not committed IVM",
        "coordinate_manifest": binding(coordinate),
        "coordinate_audit": binding(coordinate_audit),
        "surface_energy_values": SURFACE_VALUES,
        "fresh_energy_only_attempts": FRESH_CALLS,
        "inherited_fixed_endpoint_energy_values": 2,
        "physical_energy_or_force_calls_started": 0,
        "force_calls": 0, "optimization_N_F_contribution": 0,
    }
    atomic_json(CALL_POLICY, call_policy_payload)
    contract_payload = {
        "schema": CONTRACT_SCHEMA, "status": CONTRACT_STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host_role": "local", "jobs": 1,
        "evaluator": binding(THIS_FILE),
        "bb_core": binding(ROOT / "src/bb_core.py"),
        "topology": binding(ROOT / "inputs/complex_fixed.pdb"),
        "coordinate_manifest": binding(coordinate),
        "coordinate_audit": binding(coordinate_audit),
        "coordinate_freeze_contract": binding(geometry.CONTRACT),
        "energy_call_policy": binding(CALL_POLICY),
        "endpoint_cache": binding(ROOT / "inputs/endpoint_joint_ef_cache.npz"),
        "host_local_lock_paths": [str(path) for path in LOCKS],
        "thread_policy": thread_policy,
        "result_output": str(OUTPUT.resolve()),
        "original_path32_result_output": str(UNUSED_PATH_OUTPUT.resolve()),
        "zero_openmm_preflight_output": str(PREAUTH.resolve()),
        "execution_authorization_output": str(AUTHORIZATION.resolve()),
        "physical_call_policy": {
            "surface_energy_values_stored": SURFACE_VALUES,
            "surface_fresh_energy_only_attempts": FRESH_CALLS,
            "surface_inherited_fixed_endpoint_values": 2,
            "exact_original_path_internal_fresh_energy_only_attempts": 0,
            "exact_original_path_inherited_fixed_endpoint_values": 2,
            "total_fresh_energy_only_attempts": FRESH_CALLS,
            "force_calls": 0, "getState_getEnergy": True,
            "getState_getForces": False, "optimization_N_F_contribution": 0,
        },
    }
    atomic_json(CONTRACT, contract_payload)
    preauth_payload = {
        "schema": PREAUTH_SCHEMA, "status": PREAUTH_STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": binding(CONTRACT), "evaluator": binding(THIS_FILE),
        "coordinate_manifest": binding(coordinate),
        "coordinate_audit": binding(coordinate_audit),
        "original_path32_result_output": str(UNUSED_PATH_OUTPUT.resolve()),
        "surface_result_output": str(OUTPUT.resolve()),
        "canonical_two_lock_nonblocking_probe": {"status": "PASS", "locks": locks},
        "thread_policy": thread_policy,
        "path_stage_fresh_attempt_maximum": 0,
        "surface_stage_fresh_attempt_maximum": FRESH_CALLS,
        "total_fresh_attempt_maximum": FRESH_CALLS,
        "observed_free_bytes": free,
        "physical_calls_started": 0, "openmm_contexts_constructed": 0,
        "force_calls": 0, "optimization_N_F_contribution": 0,
    }
    atomic_json(PREAUTH, preauth_payload)
    authorization_payload = {
        "schema": AUTH_SCHEMA, "status": AUTH_STATUS, "enabled": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host_role": "local", "jobs": 1, "thread_policy": thread_policy,
        "contract": binding(CONTRACT), "evaluator": binding(THIS_FILE),
        "zero_openmm_preflight": binding(PREAUTH),
        "coordinate_manifest": binding(coordinate),
        "coordinate_audit": binding(coordinate_audit),
        "result_output": str(OUTPUT.resolve()),
        "original_path32_result_output": str(UNUSED_PATH_OUTPUT.resolve()),
        "authorized_stages": ["original_path32", "surface"],
        "maximum_energy_only_attempts": FRESH_CALLS,
        "maximum_surface_energy_only_attempts": FRESH_CALLS,
        "maximum_original_internal_energy_only_attempts": 0,
        "maximum_force_calls": 0, "optimization_N_F_contribution": 0,
        "physical_calls_started_before_authorization": 0,
        "canonical_two_lock_second_probe": {"status": "PASS", "locks": locks},
        "state_role": "returned terminal state; terminal uncommitted charged tail; not committed IVM",
    }
    atomic_json(AUTHORIZATION, authorization_payload)
    print(json.dumps({"status": AUTH_STATUS, "call_policy": binding(CALL_POLICY),
                      "contract": binding(CONTRACT), "preflight": binding(PREAUTH),
                      "authorization": binding(AUTHORIZATION),
                      "physical_calls_started": 0, "locks_available": True,
                      "fresh_energy_only_attempts_authorized": FRESH_CALLS}, indent=2))


def load_base():
    require(sha256_file(BASE) == BASE_SHA256, "tested base evaluator changed")
    spec = importlib.util.spec_from_file_location("bb_call515_energy_base", BASE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.__file__ = str(THIS_FILE)
    module.CONTRACT_SCHEMA = CONTRACT_SCHEMA
    module.CONTRACT_STATUS = CONTRACT_STATUS
    module.AUTH_SCHEMA = AUTH_SCHEMA
    module.AUTH_STATUS = AUTH_STATUS
    module.PREAUTH_SCHEMA = PREAUTH_SCHEMA
    module.PREAUTH_STATUS = PREAUTH_STATUS
    module.COORDINATE_SCHEMA = geometry.MANIFEST_SCHEMA
    module.COORDINATE_STATUS = geometry.MANIFEST_STATUS
    module.RESULT_SCHEMA = RESULT_SCHEMA
    module.AUDIT_SCHEMA = AUDIT_SCHEMA
    module.WAL_SCHEMA = WAL_SCHEMA
    module.CALL_POLICY_SCHEMA = CALL_POLICY_SCHEMA
    module.CALL_POLICY_STATUS = CALL_POLICY_STATUS
    module.PATH_INTERNAL_CALLS = 0
    module.TOTAL_FRESH_CALLS = FRESH_CALLS
    module.validate_coordinate_manifest = geometry.validate_coordinate_manifest_for_energy
    # R2B registrations use the stronger path/SHA/byte binding emitted by this
    # wrapper.  The frozen base used path/SHA only; changing the helper keeps
    # all prospective bindings exact without weakening validation.
    module.file_binding = lambda path: binding(Path(path))

    original_load_contract = module.load_contract

    def load_contract(contract_path, authorization_path, host_role, output, stage):
        state = original_load_contract(contract_path, authorization_path,
                                       host_role, output, stage)
        require(stage == module.SURFACE_STAGE, "R2 authorizes surface stage only")
        coordinate_audit = module.resolve_binding(
            contract_path, state["contract"]["coordinate_audit"], "coordinate audit")
        payload = json.loads(coordinate_audit.read_text(encoding="utf-8"))
        require(payload.get("status") == geometry.AUDIT_STATUS
                and payload.get("center_source_call_per_image") == geometry.CALL
                and payload.get("center_source_optimization_N_F") == geometry.NF,
                "coordinate audit state mismatch")
        state["coordinate_audit"] = coordinate_audit
        return state

    def preflight(state, stage):
        require(stage == module.SURFACE_STAGE, "R2 preflight is surface-only")
        output, audit, wal = module.stage_artifacts(state, stage)
        output.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(output.parent).free
        required_free = module.MINIMUM_FREE_BYTES_RESERVE + (
            0 if output.exists() else module.ESTIMATED_FINAL_BYTES)
        require(free >= required_free, f"insufficient free disk: {free} < {required_free}")
        existing = module._inspect_existing_stage_wal(state, stage)
        with module.acquire_two_locks(state["locks"]) as probed:
            lock_probe = list(probed)
        return {
            "status": "ZERO_OPENMM_PREFLIGHT_PASS", "stage": stage,
            "stage_energy_values": SURFACE_VALUES,
            "stage_fresh_energy_only_attempts_maximum": FRESH_CALLS,
            "total_fresh_energy_only_attempts_authorized": FRESH_CALLS,
            "surface_inherited_fixed_endpoint_values": 2,
            "path_inherited_fixed_endpoint_values": 0,
            "force_calls": 0, "optimization_N_F_contribution": 0,
            "canonical_two_lock_nonblocking_probe": {"status": "PASS", "locks": lock_probe},
            "result_exists_pending_audit": output.exists(),
            "audit_already_exists": audit.exists(), "resume_wal_exists": wal.exists(),
            "stage_existing_attempt_started": 0 if existing is None else
                existing["lifetime"]["attempt_started"],
            "authorization_existing_attempt_started_across_both_stages": 0 if existing is None else
                existing["lifetime"]["attempt_started"],
            "observed_free_bytes": free, "required_free_bytes": required_free,
        }

    original_common = module._common_result_payload

    def common_payload(state, stage, held_locks):
        payload = original_common(state, stage, held_locks)
        payload["method"] = np.asarray("IR-IVM returned terminal diagnostic")
        payload["center_source_optimization_N_F"] = np.asarray(geometry.NF, dtype=np.int64)
        payload["center_source_call_per_image"] = np.asarray(geometry.CALL, dtype=np.int64)
        payload["returned_terminal_state"] = np.asarray(True)
        payload["formally_reportable"] = np.asarray(False)
        payload["terminal_uncommitted_charged_tail"] = np.asarray(True)
        payload["coordinate_audit"] = np.asarray(str(state["coordinate_audit"]))
        payload["coordinate_audit_sha256"] = np.asarray(sha256_file(state["coordinate_audit"]))
        return payload

    def build_result_payload(state, stage, wal_state, held_locks):
        require(stage == module.SURFACE_STAGE, "R2 result is surface-only")
        module.validate_stage_wal(state, stage, wal_state, complete=True)
        payload = common_payload(state, stage, held_locks)
        successes = wal_state["successes"]
        energy = np.empty((geometry.POINTS, geometry.NORMAL_POINTS), dtype=np.float64)
        fresh_mask = np.ones_like(energy, dtype=np.bool_)
        fresh_mask[0, geometry.ZERO_INDEX] = False
        fresh_mask[-1, geometry.ZERO_INDEX] = False
        inherited_mask = ~fresh_mask
        energy[0, geometry.ZERO_INDEX] = state["endpoint"]["energies"][0]
        energy[-1, geometry.ZERO_INDEX] = state["endpoint"]["energies"][1]
        for item in module._logical_coordinates(state["data"], stage):
            i, j = int(item["center_index"]), int(item["normal_index"])
            energy[i, j] = float(successes[(stage, int(item["logical_index"]))]["energy_eV"])
        require(np.all(np.isfinite(energy)), "non-finite surface energy")
        lineage = np.full(energy.shape, "fresh_unrelaxed_cartesian_surface", dtype="<U48")
        lineage[0, geometry.ZERO_INDEX] = "inherited_canonical_endpoint_cache"
        lineage[-1, geometry.ZERO_INDEX] = "inherited_canonical_endpoint_cache"
        data = state["data"]
        payload.update({
            "surface_coordinates_A": data["surface_coordinates"],
            "surface_coordinate_sha256": data["cell_hashes"],
            "surface_energy_eV": energy,
            "surface_energy_sha256": np.asarray(module.sha256_array(energy)),
            "center_path59_A": data["center"], "center_s": data["center_s"],
            "normal_directions": data["normal_directions"],
            "common_initial_path59_A": data["common_initial_path59_A"],
            "normal_coordinate_rms_A": data["n_grid"],
            "zero_normal_index": np.asarray(geometry.ZERO_INDEX, dtype=np.int64),
            "fresh_evaluated_mask": fresh_mask, "inherited_endpoint_mask": inherited_mask,
            "energy_value_lineage": lineage,
            "stored_energy_values": np.asarray(SURFACE_VALUES, dtype=np.int64),
            "fresh_energy_only_attempts": np.asarray(FRESH_CALLS, dtype=np.int64),
            "inherited_endpoint_energy_values": np.asarray(2, dtype=np.int64),
        })
        return payload

    def run(state, stage, backend_class=None):
        if backend_class is None:
            backend_class = module.EnergyOnlyBackend
        preflight_report = preflight(state, stage)
        output, audit_path, wal_path = module.stage_artifacts(state, stage)
        fresh = reused = 0
        backend = None
        with module.acquire_two_locks(state["locks"]) as held_locks:
            wal = module.DurableWal(wal_path, module.expected_wal_header(state, stage))
            try:
                _, wal_state = module.read_wal(wal_path, module.expected_wal_header(state, stage))
                module.validate_stage_wal(state, stage, wal_state, complete=False)
                if audit_path.exists():
                    module.validate_publication(wal_state, output, stage, required=True)
                    module.validate_result(output, state, stage, wal_state)
                    return module.validate_audit(audit_path, state, stage, wal_state)
                if not output.exists():
                    require(not wal_state["publications"], "publication exists without result")
                    for item in module._logical_coordinates(state["data"], stage):
                        key = (stage, int(item["logical_index"]))
                        existing = wal_state["successes"].get(key)
                        if existing is not None:
                            require(existing["coordinate_sha256"] == item["coordinate_sha256"],
                                    "reused coordinate mismatch")
                            reused += 1
                            continue
                        if backend is None:
                            backend = backend_class(state["topology"])
                        module._attempt_energy(wal, backend, item, 0)
                        fresh += 1
                    _, wal_state = module.read_wal(wal_path, module.expected_wal_header(state, stage))
                    module.validate_stage_wal(state, stage, wal_state, complete=True)
                    module.atomic_npz_no_overwrite(
                        output, build_result_payload(state, stage, wal_state, held_locks))
                else:
                    module.validate_stage_wal(state, stage, wal_state, complete=True)
                module.validate_result(output, state, stage, wal_state)
                if not wal_state["publications"]:
                    wal.append(module.publication_record(output, stage))
                    _, wal_state = module.read_wal(wal_path, module.expected_wal_header(state, stage))
                module.validate_publication(wal_state, output, stage, required=True)
                module.validate_result(output, state, stage, wal_state)
                if backend is not None:
                    require(backend.raw_energy_attempts == fresh
                            and backend.raw_energy_successes == fresh,
                            "backend energy ledger mismatch")
                    require(backend.backend.raw_joint_attempts == 0
                            and backend.backend.raw_joint_evaluations == 0,
                            "energy-only run touched Force ledger")
            finally:
                wal.close()
            audit = module.build_audit(state, stage, wal_state, held_locks,
                                       preflight_report, fresh, reused)
            audit["coordinate_audit"] = module.file_binding(state["coordinate_audit"])
            audit["center_source_call_per_image"] = geometry.CALL
            audit["center_source_optimization_N_F"] = geometry.NF
            audit["state_role"] = (
                "returned terminal state; terminal uncommitted charged tail; not committed IVM"
            )
            audit["authorization_lifetime_attempt_started_across_both_stages"] = \
                wal_state["lifetime"]["attempt_started"]
            module.atomic_json_no_overwrite(audit_path, audit)
            return audit

    module.load_contract = load_contract
    module.preflight = preflight
    module._common_result_payload = common_payload
    module.build_result_payload = build_result_payload
    module.run = run
    return module


def execute(argv: list[str]) -> None:
    module = load_base()
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    module.check_thread_caps()
    state = module.load_contract(args.contract.resolve(strict=True),
                                 args.authorization.resolve(strict=True),
                                 "local", args.output.resolve(), module.SURFACE_STAGE)
    if args.preflight_only:
        print(json.dumps(module.preflight(state, module.SURFACE_STAGE), indent=2,
                         sort_keys=True))
    else:
        print(json.dumps(module.run(state, module.SURFACE_STAGE), indent=2,
                         sort_keys=True))


def main() -> None:
    if "--prepare-chain" in sys.argv[1:]:
        require(sys.argv[1:] == ["--prepare-chain"],
                "--prepare-chain accepts no additional arguments")
        prepare_chain()
        return
    execute(sys.argv[1:])


if __name__ == "__main__":
    main()
