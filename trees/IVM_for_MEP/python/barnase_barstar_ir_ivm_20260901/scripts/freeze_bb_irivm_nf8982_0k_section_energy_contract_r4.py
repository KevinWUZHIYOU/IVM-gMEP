#!/usr/bin/env python3
"""Freeze R4 BB IR-IVM section energy contract and zero-OpenMM preflight receipt."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = ROOT / "scripts/evaluate_bb_irivm_nf8982_0k_section_energy_r4.py"
AUTHORIZER = ROOT / "scripts/authorize_bb_irivm_nf8982_0k_section_energy_r4.py"
BASE_EVALUATOR = ROOT / "scripts/evaluate_bb_irivm_nf8982_0k_section_energy.py"
COORDINATE = ROOT / "runs/diagnostics/bb_irivm_nf8982_0k_section_coordinates_59x28_20260902_r2.npz"
COORDINATE_SHA256 = "9259354b5fce0fb57e040436e7800cd1fc67cd7c1cc71305663fd915057cc02e"
COORDINATE_CONTRACT = ROOT / "registrations/diagnostics/BB_IRIVM_NF8982_0K_SECTION_FREEZE_CONTRACT_20260902_R2.json"
COORDINATE_BUILDER = ROOT / "scripts/freeze_bb_irivm_nf8982_0k_section_r2.py"
CALL_POLICY = ROOT / "registrations/diagnostics/BB_IRIVM_NF8982_0K_SECTION_ENERGY_CALL_POLICY_R3_20260902.json"
ENDPOINT_CACHE = ROOT / "inputs/endpoint_joint_ef_cache.npz"
R3_SUPERSESSION = ROOT / "registrations/diagnostics/BB_IRIVM_NF8982_0K_SECTION_ENERGY_R3_SUPERSESSION_20260902.json"
TOPOLOGY = ROOT / "inputs/complex_fixed.pdb"
BB_CORE = ROOT / "src/bb_core.py"
CONTRACT = ROOT / "registrations/diagnostics/BB_IRIVM_NF8982_0K_SECTION_ENERGY_CONTRACT_R4_20260902.json"
PREAUTH = ROOT / "registrations/diagnostics/BB_IRIVM_NF8982_0K_SECTION_ENERGY_PREAUTH_PREFLIGHT_R4_20260902.json"
AUTH = ROOT / "registrations/diagnostics/BB_IRIVM_NF8982_0K_SECTION_ENERGY_EXECUTION_AUTHORIZATION_R4_20260902.json"
SURFACE_RESULT = ROOT / "runs/diagnostics/bb_irivm_nf8982_0k_section_energy_59x28_20260902_r4.npz"
PATH_RESULT = ROOT / "runs/diagnostics/bb_irivm_nf8982_original_path32_energy_20260902_r4.npz"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("bb_nf8982_energy_r4_freezer", EVALUATOR)
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
    parser.add_argument("--output", type=Path, default=CONTRACT)
    parser.add_argument("--preflight-output", type=Path, default=PREAUTH)
    parser.add_argument("--authorization-output", type=Path, default=AUTH)
    parser.add_argument("--result-output", type=Path, default=SURFACE_RESULT)
    parser.add_argument("--original-path32-result-output", type=Path, default=PATH_RESULT)
    parser.add_argument("--host-role", choices=("local",), default="local")
    parser.add_argument("--global-lock-path", type=Path, required=True)
    parser.add_argument("--peer-global-lock-path", type=Path, required=True)
    args = parser.parse_args(argv)
    evaluator = load_evaluator()
    evaluator.check_thread_caps()
    destinations = {
        "contract": args.output.expanduser().resolve(),
        "preflight": args.preflight_output.expanduser().resolve(),
        "authorization": args.authorization_output.expanduser().resolve(),
        "surface": args.result_output.expanduser().resolve(),
        "path": args.original_path32_result_output.expanduser().resolve(),
    }
    evaluator.require(destinations == {"contract": CONTRACT.resolve(), "preflight": PREAUTH.resolve(), "authorization": AUTH.resolve(), "surface": SURFACE_RESULT.resolve(), "path": PATH_RESULT.resolve()}, "noncanonical R4 destination")
    locks = [args.global_lock_path.expanduser().resolve(), args.peer_global_lock_path.expanduser().resolve()]
    evaluator.require(len(set(locks)) == 2 and set(locks) == evaluator.CANONICAL_LOCKS, "canonical two-lock set required")
    evaluator.require(evaluator.sha256_file(COORDINATE) == COORDINATE_SHA256, "coordinate SHA-256 mismatch")
    evaluator.require(evaluator.sha256_file(BASE_EVALUATOR) == evaluator.BASE_SHA256, "R3 base evaluator SHA-256 mismatch")
    data = evaluator.validate_coordinate_manifest(COORDINATE, COORDINATE_SHA256)
    endpoint = evaluator.load_endpoint_cache(ENDPOINT_CACHE, data, evaluator.ENDPOINT_CACHE_SHA256)
    for result in (destinations["path"], destinations["surface"]):
        result.parent.mkdir(parents=True, exist_ok=True)
        for artifact in (result, result.with_suffix(".energy.wal.jsonl"), result.with_suffix(".audit.json")):
            evaluator.require(not artifact.exists(), f"prospective R4 physical artifact already exists: {artifact}")
    free = shutil.disk_usage(destinations["surface"].parent).free
    required = evaluator.MINIMUM_FREE_BYTES_RESERVE + evaluator.ESTIMATED_FINAL_BYTES
    evaluator.require(free >= required, f"insufficient free disk: {free} < {required}")
    with evaluator.acquire_two_locks(locks) as held:
        lock_probe = list(held)
    contract = {
        "schema": evaluator.CONTRACT_SCHEMA,
        "status": evaluator.CONTRACT_STATUS,
        "label": "bb_irivm_nf8982_0k_section_energy_20260902_r4",
        "host_role": args.host_role,
        "jobs": 1,
        "scientific_identity": "unrelaxed Cartesian 0 K section and exact source path centered on selected IR-IVM NF8982",
        "evaluator": evaluator.file_binding(EVALUATOR),
        "r3_base_evaluator": evaluator.file_binding(BASE_EVALUATOR),
        "contract_freezer": evaluator.file_binding(Path(__file__).resolve()),
        "execution_authorizer": evaluator.file_binding(AUTHORIZER),
        "bb_core": evaluator.file_binding(BB_CORE),
        "topology": evaluator.file_binding(TOPOLOGY),
        "coordinate_manifest": evaluator.file_binding(COORDINATE),
        "coordinate_freeze_contract": evaluator.file_binding(COORDINATE_CONTRACT),
        "coordinate_builder": evaluator.file_binding(COORDINATE_BUILDER),
        "energy_call_policy": evaluator.file_binding(CALL_POLICY),
        "endpoint_cache": evaluator.file_binding(ENDPOINT_CACHE),
        "r3_supersession_receipt": evaluator.file_binding(R3_SUPERSESSION),
        "selected_result": evaluator.file_binding(data["source_result"]),
        "source_path": {"optimization_N_F": 8982, "call_per_image": 318, "shape": [32, 3132, 3], "canonical_le_float64_sha256": data["original_path32_sha256"], "terminal_partial_tail_excluded": True},
        "surface_geometry": {"shape": [59, 28, 3132, 3], "normal_grid_A": [float(x) for x in evaluator.EXPECTED_N_GRID], "zero_normal_index": evaluator.ZERO_INDEX, "all_coordinates_stored_and_cellwise_hashed": True},
        "physical_call_policy": {
            "surface_energy_values_stored": evaluator.SURFACE_CALLS,
            "surface_fresh_energy_only_attempts": evaluator.SURFACE_FRESH_CALLS,
            "surface_inherited_fixed_endpoint_values": 2,
            "exact_original_path_internal_fresh_energy_only_attempts": evaluator.PATH_INTERNAL_CALLS,
            "exact_original_path_inherited_fixed_endpoint_values": 2,
            "total_fresh_energy_only_attempts": evaluator.TOTAL_FRESH_CALLS,
            "force_calls": 0,
            "getState_getEnergy": True,
            "getState_getForces": False,
            "optimization_N_F_contribution": 0,
        },
        "original_path_energy_policy": {"run_and_publish_before_surface": True, "fresh_internal_images": 30, "inherited_fixed_endpoints": 2, "never_approximate_from_center59_or_dense500": True},
        "storage_and_durability_policy": {"independent_WAL_result_audit_per_stage": True, "all_coordinates_and_raw_finite_energies_stored": True, "fresh_and_inherited_masks_and_lineage_stored": True, "append_only_fsync_WAL": True, "unique_result_publication_footer": True, "atomic_no_overwrite_outputs": True, "failed_or_open_attempt_requires_new_authorization_and_output": True, "minimum_free_bytes_reserve": evaluator.MINIMUM_FREE_BYTES_RESERVE, "estimated_final_bytes": evaluator.ESTIMATED_FINAL_BYTES},
        "thread_policy": {name: "1" for name in evaluator.THREAD_VARS},
        "host_local_lock_paths": [str(path) for path in sorted(locks, key=str)],
        "result_output": str(destinations["surface"]),
        "original_path32_result_output": str(destinations["path"]),
        "energy_WAL_output": str(destinations["surface"].with_suffix(".energy.wal.jsonl")),
        "audit_output": str(destinations["surface"].with_suffix(".audit.json")),
        "original_path32_energy_WAL_output": str(destinations["path"].with_suffix(".energy.wal.jsonl")),
        "original_path32_audit_output": str(destinations["path"].with_suffix(".audit.json")),
        "zero_openmm_preflight_output": str(destinations["preflight"]),
        "execution_authorization_output": str(destinations["authorization"]),
        "physical_execution_requires_separate_authorization": True,
    }
    atomic_json(destinations["contract"], contract)
    preauth = {
        "schema": evaluator.PREAUTH_SCHEMA,
        "status": evaluator.PREAUTH_STATUS,
        "completed_utc": evaluator.utc_now(),
        "contract": evaluator.file_binding(destinations["contract"]),
        "evaluator": evaluator.file_binding(EVALUATOR),
        "coordinate_manifest": evaluator.file_binding(COORDINATE),
        "energy_call_policy": evaluator.file_binding(CALL_POLICY),
        "endpoint_cache": evaluator.file_binding(ENDPOINT_CACHE),
        "endpoint_coordinate_pair_sha256": endpoint["coordinate_sha256"],
        "endpoint_energy_pair_sha256": endpoint["energy_sha256"],
        "original_path32_result_output": str(destinations["path"]),
        "surface_result_output": str(destinations["surface"]),
        "canonical_two_lock_nonblocking_probe": {"status": "PASS", "locks": lock_probe},
        "thread_policy": {name: "1" for name in evaluator.THREAD_VARS},
        "observed_free_bytes": free,
        "required_free_bytes": required,
        "path_stage_fresh_attempt_maximum": evaluator.PATH_INTERNAL_CALLS,
        "surface_stage_fresh_attempt_maximum": evaluator.SURFACE_FRESH_CALLS,
        "total_fresh_attempt_maximum": evaluator.TOTAL_FRESH_CALLS,
        "physical_calls_started": 0,
        "openmm_contexts_constructed": 0,
    }
    atomic_json(destinations["preflight"], preauth)
    print(json.dumps({"status": "FROZEN_R4_ZERO_OPENMM", "contract": evaluator.file_binding(destinations["contract"]), "preflight": evaluator.file_binding(destinations["preflight"]), "physical_calls_started": 0}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
