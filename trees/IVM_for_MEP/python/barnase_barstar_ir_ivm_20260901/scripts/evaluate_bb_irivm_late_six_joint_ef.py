#!/usr/bin/env python3
"""Remote-only late-six IR--IVM arc500 joint energy/full-force evaluator.

The six paths are frozen by a metric-blind, cost-only selector after an exact
all-path replay verification.  Their discrete ``I_perp`` values are the sealed
online values computed from the 30 internal-image forces on those exact
32-image paths; no new discrete diagnostic call is made.  Each path stores 500
uniform-Cartesian-arc-length coordinates, energies, and full force vectors.
The two fixed endpoint E/F values are inherited from the immutable endpoint
cache, so a complete path has 500 joint E/F values but only 498 fresh calls.

The evaluator is remote-host gated, requires both canonical nonblocking fcntl
locks, uses append-only fsync WALs, and publishes atomic complete-path shards.
Only a fully validated shard is restartable.  Any failed/open/retried or
otherwise incomplete physical attempt fails closed and requires a superseding
contract.  Contract loading and preflight do not construct OpenMM.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import socket
from typing import Iterable, Sequence
import uuid

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENGINE_PATH = HERE / "evaluate_bb_dense_four_paths_joint_v2.py"
FREEZER_PATH = HERE / "freeze_bb_irivm_late_six_joint_ef_contract.py"
SELECTOR_PATH = HERE / "select_bb_irivm_late_milestones_blind.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ENGINE = _load("bb_ir_late_six_joint_engine", ENGINE_PATH)
SELECTOR = _load("bb_ir_late_six_blind_selector", SELECTOR_PATH)

PACKAGE_SCHEMA = "bb-irivm-late-six-joint-ef-package-preregistration-v1"
PACKAGE_STATUS = "FROZEN_BEFORE_ALL_PATH_RESULT_AND_BEFORE_LATE_METRIC_INSPECTION"
CONTRACT_SCHEMA = "bb-irivm-late-six-arc500-joint-ef-contract-v1"
CONTRACT_STATUS = "FROZEN_AFTER_BLIND_SELECTION_BEFORE_LATE_SIX_PHYSICAL_CALLS"
SHARD_SCHEMA = "bb-irivm-late-path-arc500-joint-ef-shard-v1"
RESULT_SCHEMA = "bb-irivm-late-six-arc500-joint-ef-result-v1"
MANIFEST_SCHEMA = "bb-irivm-late-six-arc500-joint-ef-manifest-v1"
AUDIT_SCHEMA = "bb-irivm-late-six-arc500-joint-ef-audit-v1"
SOURCE_WAL_SCHEMA = "bb-irivm-late-six-sealed-source-wal-v1"
DENSE_WAL_SCHEMA = "bb-irivm-late-six-arc500-joint-ef-wal-v1"

N_PATHS = 6
N_IMAGES = 32
N_INTERNAL = 30
DIMENSION = 9396
N_DENSE = 500
N_FRESH_PER_PATH = N_DENSE - 2
TOTAL_STORED_JOINT_EF = N_PATHS * N_DENSE
MAXIMUM_FRESH_ATTEMPT_STARTED = N_PATHS * N_FRESH_PER_PATH
EXPECTED_HISTORY_ROWS = 352
MAXIMUM_OPTIMIZATION_NF = 9990
EV_TO_KCAL = 23.0605
MINIMUM_FREE_BYTES_RESERVE = 6 * 1024**3
ESTIMATED_SHARD_BYTES = (
    (2 * N_DENSE * DIMENSION + N_IMAGES * DIMENSION) * 8 + 8 * 1024**2
)

REMOTE_HOSTNAME = "Mac-mini.local"
REMOTE_ADDRESS = "100.74.85.3"
REMOTE_MACHINE = "arm64"
MUTAGEN_SESSION_NAME = "bb-irivm-20260901"
MUTAGEN_SESSION_IDENTIFIER = "sync_EaEvvA9594NRGtMQSN0jneepyUq2tCef27jljwXXPc9"

BLIND_PREREGISTRATION = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_LATE_MILESTONE_SELECTION_PREREG_20260902_R1.json"
).resolve()
CANONICAL_PACKAGE_PREREGISTRATION = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_LATE_SIX_JOINT_EF_PACKAGE_PREREG_20260902_R1.json"
).resolve()
CANONICAL_SELECTION = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_LATE_MILESTONE_SELECTION_20260902_R1.json"
).resolve()
CANONICAL_CONTRACT = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_LATE_SIX_JOINT_EF_ARC500_20260902_R1.json"
).resolve()
CANONICAL_RESULT = (
    ROOT
    / "runs/diagnostics/bb_irivm_late_six_arc500_joint_ef_20260902_r1.npz"
).resolve()
COMMON_INITIAL_PATH = (ROOT / "inputs/common_initial_path32.npz").resolve()
ENDPOINT_CACHE = (ROOT / "inputs/endpoint_joint_ef_cache.npz").resolve()
TOPOLOGY = (ROOT / "inputs/complex_fixed.pdb").resolve()
CANONICAL_LOCKS = {
    Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock").resolve(),
    Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock").resolve(),
}

THREAD_CAPS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENMM_CPU_THREADS",
)
METRIC_POLICY = {
    "coordinate_space": "unweighted_3N_cartesian",
    "sampling": "uniform_cartesian_arclength",
    "interpolation": "piecewise_linear",
    "number_of_points": 500,
    "grid_formula": "s_j=j*L/499,j=0,...,499",
    "include_both_endpoints": True,
    "dense_joint_EF_values_stored_per_path": 500,
    "fresh_dense_joint_EF_calls_per_path": 498,
    "inherited_endpoint_joint_EF_values_per_path": 2,
    "dense_energy_and_force_same_call": True,
    "dense_force_storage": "complete_full_force_vector_for_every_point",
    "discrete_I_perp_source": (
        "exact_verified_online_history_computed_from_30_internal_forces_on_"
        "the_same_selected_32_image_path"
    ),
    "discrete_I_perp_new_calls": 0,
    "optimization_N_F_contribution": 0,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bind(path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": ENGINE.sha256_file(resolved)}


def scalar(saved, key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} must be scalar")
    return value.item()


def same_float(left, right) -> bool:
    return np.asarray(float(left), dtype="<f8").tobytes() == np.asarray(
        float(right), dtype="<f8"
    ).tobytes()


def verified_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _load_common_endpoints() -> np.ndarray:
    with np.load(COMMON_INITIAL_PATH, allow_pickle=False) as saved:
        initial = np.asarray(saved["initial_path"], dtype=np.float64)
    require(
        initial.shape == (N_IMAGES, DIMENSION) and np.all(np.isfinite(initial)),
        "common initial path is invalid",
    )
    return np.ascontiguousarray(initial[[0, -1]])


def _load_endpoint_cache() -> dict:
    with np.load(ENDPOINT_CACHE, allow_pickle=False) as saved:
        coordinates = np.asarray(saved["coordinates_A"], dtype=np.float64)
        energies = np.asarray(saved["energies_eV"], dtype=np.float64)
        forces = np.asarray(saved["forces_eV_per_A"], dtype=np.float64)
    require(
        coordinates.shape == forces.shape == (2, DIMENSION)
        and energies.shape == (2,),
        "endpoint cache shape mismatch",
    )
    require(
        np.all(np.isfinite(coordinates))
        and np.all(np.isfinite(energies))
        and np.all(np.isfinite(forces)),
        "endpoint cache contains non-finite values",
    )
    require(
        np.array_equal(coordinates, _load_common_endpoints()),
        "endpoint cache differs from common initial path",
    )
    return {
        "coordinates": np.ascontiguousarray(coordinates),
        "energies": np.ascontiguousarray(energies),
        "forces": np.ascontiguousarray(forces),
    }


def build_package_preregistration(locks: Sequence[Path]) -> dict:
    resolved_locks = [Path(path).expanduser().resolve() for path in locks]
    require(
        len(resolved_locks) == len(set(resolved_locks)) == 2
        and set(resolved_locks) == CANONICAL_LOCKS,
        "exactly the two canonical BB locks are required",
    )
    blind = SELECTOR.validate_preregistration(BLIND_PREREGISTRATION)
    require(
        blind["future_joint_EF_protocol"]["number_of_points_per_selected_path"]
        == N_DENSE,
        "blind preregistration dense point count drifted",
    )
    require(
        blind["future_joint_EF_protocol"]["store_all_full_force_vectors"] is True,
        "blind preregistration does not require full-force storage",
    )
    _load_endpoint_cache()
    return {
        "schema": PACKAGE_SCHEMA,
        "status": PACKAGE_STATUS,
        "frozen": True,
        "created_utc": utc_now(),
        "purpose": (
            "pre-bind the complete remote IR--IVM late-six evaluator/freezer "
            "before the all-path result and any late diagnostic metric arrive"
        ),
        "result_or_selection_bound_now": False,
        "host_role": "remote",
        "remote_identity": {
            "hostname": REMOTE_HOSTNAME,
            "address": REMOTE_ADDRESS,
            "platform": "Darwin",
            "machine": REMOTE_MACHINE,
        },
        "required_Mutagen_session": {
            "name": MUTAGEN_SESSION_NAME,
            "identifier": MUTAGEN_SESSION_IDENTIFIER,
            "input_flush_must_succeed_before_final_contract_freeze": True,
            "contract_publish_flush_must_succeed_after_final_contract_freeze": True,
            "alpha_connected": True,
            "beta_connected": True,
            "transition_problems": 0,
        },
        "host_local_lock_paths": [str(path) for path in resolved_locks],
        "jobs": 1,
        "result_output": str(CANONICAL_RESULT),
        "final_contract_output": str(CANONICAL_CONTRACT),
        "metric_policy": METRIC_POLICY,
        "physical_call_policy": {
            "selected_paths": N_PATHS,
            "stored_joint_EF_values_per_path": N_DENSE,
            "stored_joint_EF_values_total": TOTAL_STORED_JOINT_EF,
            "fresh_joint_EF_attempts_per_path": N_FRESH_PER_PATH,
            "maximum_fresh_attempt_started_records": MAXIMUM_FRESH_ATTEMPT_STARTED,
            "inherited_endpoint_joint_EF_values_total": 2 * N_PATHS,
            "original_discrete_new_calls": 0,
            "optimization_N_F_contribution": 0,
            "jobs": 1,
            "all_six_thread_caps": 1,
            "two_host_local_fcntl_locks": True,
        },
        "durability_policy": {
            "atomic_complete_path_shard": True,
            "append_only_fsync_WAL": True,
            "complete_shard_only_restart": True,
            "individual_partial_success_reuse": False,
            "open_failed_or_retried_attempt_requires_superseding_contract": True,
            "no_overwrite": True,
        },
        "post_arrival_pipeline_templates": {
            "working_directory": str(ROOT),
            "python": "/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python",
            "step_1_selector_argv": [
                str(SELECTOR_PATH),
                "select",
                "--preregistration",
                str(BLIND_PREREGISTRATION),
                "--verification",
                "ABSOLUTE_PASS_R2_ALL_PATHS_VERIFICATION_JSON",
                "--output",
                str(CANONICAL_SELECTION),
            ],
            "step_2_contract_freeze_argv": [
                str(FREEZER_PATH),
                "freeze",
                "--package",
                str(CANONICAL_PACKAGE_PREREGISTRATION),
                "--selection",
                str(CANONICAL_SELECTION),
                "--output",
                str(CANONICAL_CONTRACT),
            ],
            "step_2_includes_dedicated_Mutagen_flush_before_inputs_and_after_contract_publish": True,
            "remote_environment": {name: "1" for name in THREAD_CAPS},
            "step_3_remote_preflight_argv": [
                str(Path(__file__).resolve()),
                "--contract",
                str(CANONICAL_CONTRACT),
                "--host-role",
                "remote",
                "--output",
                str(CANONICAL_RESULT),
                "--preflight-only",
            ],
            "step_4_remote_run_argv": [
                str(Path(__file__).resolve()),
                "--contract",
                str(CANONICAL_CONTRACT),
                "--host-role",
                "remote",
                "--output",
                str(CANONICAL_RESULT),
            ],
        },
        "bindings": {
            "blind_selection_preregistration": bind(BLIND_PREREGISTRATION),
            "blind_selector": bind(SELECTOR_PATH),
            "evaluator": bind(Path(__file__)),
            "contract_freezer": bind(FREEZER_PATH),
            "joint_engine": bind(ENGINE_PATH),
            "bb_core": bind(ROOT / "src/bb_core.py"),
            "topology": bind(TOPOLOGY),
            "common_initial_path": bind(COMMON_INITIAL_PATH),
            "endpoint_joint_EF_cache": bind(ENDPOINT_CACHE),
        },
    }


def validate_package_preregistration(path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    package = _read_json(resolved)
    require(
        package.get("schema") == PACKAGE_SCHEMA
        and package.get("status") == PACKAGE_STATUS
        and package.get("frozen") is True,
        "invalid late-six package preregistration",
    )
    require(package.get("host_role") == "remote", "package is not remote-only")
    require(package.get("remote_identity") == {"hostname": REMOTE_HOSTNAME, "address": REMOTE_ADDRESS, "platform": "Darwin", "machine": REMOTE_MACHINE}, "remote identity drifted")
    require(
        package.get("required_Mutagen_session")
        == {
            "name": MUTAGEN_SESSION_NAME,
            "identifier": MUTAGEN_SESSION_IDENTIFIER,
            "input_flush_must_succeed_before_final_contract_freeze": True,
            "contract_publish_flush_must_succeed_after_final_contract_freeze": True,
            "alpha_connected": True,
            "beta_connected": True,
            "transition_problems": 0,
        },
        "package Mutagen policy drifted",
    )
    require(Path(package.get("result_output", "")).expanduser().resolve() == CANONICAL_RESULT, "package result output drifted")
    require(Path(package.get("final_contract_output", "")).expanduser().resolve() == CANONICAL_CONTRACT, "package final contract output drifted")
    for role, record in package.get("bindings", {}).items():
        require(bind(Path(record["path"])) == record, f"package binding drift: {role}")
    expected = {
        "blind_selection_preregistration": BLIND_PREREGISTRATION,
        "blind_selector": SELECTOR_PATH,
        "evaluator": Path(__file__).resolve(),
        "contract_freezer": FREEZER_PATH,
        "joint_engine": ENGINE_PATH,
        "bb_core": ROOT / "src/bb_core.py",
        "topology": TOPOLOGY,
        "common_initial_path": COMMON_INITIAL_PATH,
        "endpoint_joint_EF_cache": ENDPOINT_CACHE,
    }
    for role, expected_path in expected.items():
        require(Path(package["bindings"][role]["path"]).resolve() == expected_path.resolve(), f"package {role} path drift")
    locks = [Path(value).expanduser().resolve() for value in package["host_local_lock_paths"]]
    require(len(locks) == len(set(locks)) == 2 and set(locks) == CANONICAL_LOCKS, "package lock set drifted")
    require(package.get("metric_policy") == METRIC_POLICY, "package metric policy drifted")
    physical = package.get("physical_call_policy", {})
    require(physical.get("stored_joint_EF_values_total") == TOTAL_STORED_JOINT_EF, "stored E/F count drifted")
    require(physical.get("maximum_fresh_attempt_started_records") == MAXIMUM_FRESH_ATTEMPT_STARTED, "fresh attempt cap drifted")
    require(physical.get("original_discrete_new_calls") == 0 and physical.get("optimization_N_F_contribution") == 0, "NoF policy drifted")
    durability = package.get("durability_policy", {})
    require(durability.get("atomic_complete_path_shard") is True and durability.get("append_only_fsync_WAL") is True, "durability publication policy drifted")
    require(durability.get("complete_shard_only_restart") is True and durability.get("individual_partial_success_reuse") is False, "durability restart policy drifted")
    require(durability.get("open_failed_or_retried_attempt_requires_superseding_contract") is True and durability.get("no_overwrite") is True, "durability fail-close policy drifted")
    pipeline = package.get("post_arrival_pipeline_templates", {})
    require(pipeline.get("working_directory") == str(ROOT), "pipeline working directory drifted")
    require(pipeline.get("step_2_includes_dedicated_Mutagen_flush_before_inputs_and_after_contract_publish") is True, "pipeline Mutagen publication gate drifted")
    require(pipeline.get("remote_environment") == {name: "1" for name in THREAD_CAPS}, "pipeline thread caps drifted")
    require(pipeline.get("step_3_remote_preflight_argv", [])[-1:] == ["--preflight-only"], "pipeline remote preflight drifted")
    require("--preflight-only" not in pipeline.get("step_4_remote_run_argv", []), "pipeline run is still preflight-only")
    return package


def _validate_selection_and_load_rows(selection_path: Path) -> tuple[list[dict], dict]:
    selection_path = Path(selection_path).expanduser().resolve(strict=True)
    selection = _read_json(selection_path)
    require(
        selection.get("schema") == SELECTOR.SELECTION_SCHEMA
        and selection.get("status") == SELECTOR.SELECTION_STATUS,
        "late milestone selection is not frozen PASS",
    )
    require(selection.get("selection_metric_values_consulted") is False, "selection inspected metrics")
    require(selection.get("terminal_partial_tail_selected") is False, "selection includes terminal partial tail")
    require(selection.get("selected_path_count") == N_PATHS, "selection does not contain six paths")
    require(selection["bindings"]["preregistration"] == bind(BLIND_PREREGISTRATION), "selection does not bind frozen blind preregistration")
    require(selection["bindings"]["selector"] == bind(SELECTOR_PATH), "selection selector binding drifted")
    verification_path = Path(selection["bindings"]["exact_all_paths_verification"]["path"]).resolve(strict=True)
    result_path = Path(selection["bindings"]["all_paths_replay_result"]["path"]).resolve(strict=True)
    audit_path = Path(selection["bindings"]["all_paths_replay_audit"]["path"]).resolve(strict=True)
    for role, path in (("exact_all_paths_verification", verification_path), ("all_paths_replay_result", result_path), ("all_paths_replay_audit", audit_path)):
        require(selection["bindings"][role] == bind(path), f"selection {role} binding drifted")
    verification = _read_json(verification_path)
    require(
        verification.get("schema") == SELECTOR.VERIFICATION_SCHEMA
        and verification.get("status") == SELECTOR.VERIFICATION_STATUS,
        "all-path verification is not exact PASS",
    )
    require(verification["bindings"]["replay_result"] == bind(result_path), "verification/result mismatch")
    require(verification["bindings"]["replay_audit"] == bind(audit_path), "verification/audit mismatch")
    endpoint = _load_endpoint_cache()
    with np.load(result_path, allow_pickle=False) as saved:
        required = {
            "calls_per_image",
            "total_internal_joint_evaluations",
            "i_perp",
            "reparam_committed",
            "counter_vectors",
            "force_request_vectors",
            "stored_path_calls",
            "stored_paths",
        }
        require(required.issubset(saved.files), "all-path result lacks required histories")
        calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        nf = np.asarray(saved["total_internal_joint_evaluations"], dtype=np.int64)
        i_perp = np.asarray(saved["i_perp"], dtype=np.float64)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
        force_requests = np.asarray(saved["force_request_vectors"], dtype=np.int64)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
        stored_paths = np.asarray(saved["stored_paths"], dtype=np.float64)
    require(calls.shape == nf.shape == i_perp.shape == committed.shape == (EXPECTED_HISTORY_ROWS,), "all-path scalar history shape mismatch")
    require(counters.shape == force_requests.shape == (EXPECTED_HISTORY_ROWS, N_IMAGES), "all-path counter history shape mismatch")
    require(stored_calls.shape == (EXPECTED_HISTORY_ROWS,) and stored_paths.shape == (EXPECTED_HISTORY_ROWS, N_IMAGES, DIMENSION), "all-path stored path shape mismatch")
    require(np.all(np.isfinite(i_perp)) and np.all(np.isfinite(stored_paths)), "all-path history contains non-finite values")
    require(np.array_equal(stored_calls, calls), "stored paths are not aligned with all history calls")
    selected_rows = selection.get("selected_paths")
    require(isinstance(selected_rows, list) and len(selected_rows) == N_PATHS, "invalid selected path list")
    result = []
    seen_hashes = set()
    for selected in selected_rows:
        index = int(selected["history_index"])
        call = int(selected["call_per_image"])
        optimization_nf = int(selected["optimization_N_F"])
        require(0 <= index < EXPECTED_HISTORY_ROWS, "selected history index out of range")
        require(int(calls[index]) == call and int(nf[index]) == optimization_nf, "selected call/NF differs from replay history")
        require(bool(committed[index]) and selected.get("reparameterization_committed") is True, "selected path was not reparameterization-committed")
        require(selected.get("fully_evaluated_current_path") is True and selected.get("formally_reportable") is True, "selected path is not fully evaluated/reportable")
        require(selected.get("terminal_uncommitted_charged_tail") is False, "selected terminal partial tail")
        require(optimization_nf <= MAXIMUM_OPTIMIZATION_NF, "selected path exceeds optimization budget")
        path = np.ascontiguousarray(stored_paths[index])
        path_sha = verified_array_sha256(path)
        require(path_sha == str(selected["selected_path_sha256"]), "selected path hash differs from exact verifier")
        require(path_sha not in seen_hashes, "selection contains duplicate coordinate paths")
        seen_hashes.add(path_sha)
        require(np.array_equal(path[[0, -1]], endpoint["coordinates"]), "selected path fixed endpoints drifted")
        require(np.all(force_requests[index, 1:-1] > 0), "selected path lacks internal force-request lineage")
        value = float(i_perp[index])
        require(np.isfinite(value) and value >= 0.0, "selected discrete Iperp is invalid")
        result.append(
            {
                "row_index": len(result),
                "label": f"IVM_call{call}_NF{optimization_nf}",
                "method": "IVM",
                "host_role": "remote",
                "selection_role": str(selected["selection_role"]),
                "optimization_N_F": optimization_nf,
                "path_identifier": {
                    "history_index": index,
                    "calls_per_image": call,
                    "optimization_N_F": optimization_nf,
                    "fully_evaluated_current_path": True,
                    "reparameterization_committed": True,
                    "formally_reportable": True,
                    "terminal_uncommitted_charged_tail": False,
                },
                "path_sha256": path_sha,
                "discrete_I_perp": value,
                "source_counter_vector_sha256": ENGINE.sha256_array(counters[index]),
                "source_force_request_vector_sha256": ENGINE.sha256_array(force_requests[index]),
                "path": path,
                "source_counter_vector": np.ascontiguousarray(counters[index]),
                "source_force_request_vector": np.ascontiguousarray(force_requests[index]),
                "result_path": result_path,
                "verification_path": verification_path,
                "audit_path": audit_path,
            }
        )
    require(len(result) == len(seen_hashes) == N_PATHS, "did not load six distinct selected paths")
    require(all(result[i]["optimization_N_F"] <= result[i + 1]["optimization_N_F"] for i in range(N_PATHS - 1)), "selection paths are not ordered by optimization N_F")
    evidence = {
        "selection": bind(selection_path),
        "all_paths_result": bind(result_path),
        "all_paths_audit": bind(audit_path),
        "exact_all_paths_verification": bind(verification_path),
    }
    return result, evidence


def frozen_row(row: dict) -> dict:
    return {
        "label": row["label"],
        "method": "IVM",
        "host_role": "remote",
        "selection_role": row["selection_role"],
        "optimization_N_F": row["optimization_N_F"],
        "path_identifier": row["path_identifier"],
        "path_sha256": row["path_sha256"],
        "discrete_I_perp": row["discrete_I_perp"],
        "discrete_I_perp_source": (
            "exact_verified_online_internal_forces_on_same_selected_path"
        ),
        "source_counter_vector_sha256": row["source_counter_vector_sha256"],
        "source_force_request_vector_sha256": row[
            "source_force_request_vector_sha256"
        ],
        "original_discrete_new_calls": 0,
        "dense_policy": (
            "uniform_arc500_store_coordinates_energy_full_force_with_two_"
            "inherited_fixed_endpoints_and_498_fresh_joint_calls"
        ),
    }


def validate_mutagen_snapshot(snapshot: dict) -> None:
    require(snapshot.get("session_name") == MUTAGEN_SESSION_NAME, "Mutagen session name mismatch")
    require(snapshot.get("session_identifier") == MUTAGEN_SESSION_IDENTIFIER, "Mutagen session identifier mismatch")
    require(snapshot.get("flush_exit_code") == 0, "Mutagen flush did not pass")
    require(snapshot.get("status") in {"Watching", "Watching for changes"}, "Mutagen session is not watching")
    require(snapshot.get("alpha_connected") is True and snapshot.get("beta_connected") is True, "Mutagen endpoint disconnected")
    require(snapshot.get("transition_problems") == 0, "Mutagen transition problem present")


def build_final_contract(
    package_path: Path, selection_path: Path, mutagen_snapshot: dict
) -> dict:
    package_path = Path(package_path).expanduser().resolve(strict=True)
    selection_path = Path(selection_path).expanduser().resolve(strict=True)
    package = validate_package_preregistration(package_path)
    validate_mutagen_snapshot(mutagen_snapshot)
    rows, evidence = _validate_selection_and_load_rows(selection_path)
    endpoint = _load_endpoint_cache()
    return {
        "schema": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "frozen": True,
        "created_utc": utc_now(),
        "purpose": "remote late-six IR--IVM arc500 joint E/F after blind cost-only selection",
        "host_role": "remote",
        "remote_identity": package["remote_identity"],
        "jobs": 1,
        "result_output": package["result_output"],
        "host_local_lock_paths": package["host_local_lock_paths"],
        "metric_policy": METRIC_POLICY,
        "mutagen_gate_at_freeze": mutagen_snapshot,
        "package_preregistration": bind(package_path),
        "selection_and_replay_evidence": evidence,
        "endpoint_cache_arrays": {
            "coordinate_sha256": ENGINE.sha256_array(endpoint["coordinates"]),
            "energy_sha256": ENGINE.sha256_array(endpoint["energies"]),
            "force_sha256": ENGINE.sha256_array(endpoint["forces"]),
        },
        "paths": [frozen_row(row) for row in rows],
        "physical_call_policy": package["physical_call_policy"],
        "durability_policy": package["durability_policy"],
        "bindings": package["bindings"],
        "optimization_N_F_contribution": 0,
    }


def load_contract(contract_path: Path, host_role: str) -> dict:
    contract_path = Path(contract_path).expanduser().resolve(strict=True)
    contract = _read_json(contract_path)
    require(
        contract.get("schema") == CONTRACT_SCHEMA
        and contract.get("status") == CONTRACT_STATUS
        and contract.get("frozen") is True,
        "invalid IR late-six contract",
    )
    require(host_role == contract.get("host_role") == "remote", "IR late-six contract is remote-only")
    package_path = Path(contract["package_preregistration"]["path"]).resolve(strict=True)
    require(contract["package_preregistration"] == bind(package_path), "package preregistration binding drifted")
    package = validate_package_preregistration(package_path)
    validate_mutagen_snapshot(contract["mutagen_gate_at_freeze"])
    require(contract.get("remote_identity") == package["remote_identity"], "remote identity drifted")
    require(contract.get("metric_policy") == METRIC_POLICY, "metric policy drifted")
    selection_path = Path(contract["selection_and_replay_evidence"]["selection"]["path"]).resolve(strict=True)
    rows, evidence = _validate_selection_and_load_rows(selection_path)
    require(contract.get("selection_and_replay_evidence") == evidence, "selection/replay evidence drifted")
    require(contract.get("paths") == [frozen_row(row) for row in rows], "frozen selected rows drifted")
    require(contract.get("bindings") == package["bindings"], "source bindings drifted")
    for role, record in contract["bindings"].items():
        require(bind(Path(record["path"])) == record, f"contract source drift: {role}")
    endpoint = _load_endpoint_cache()
    expected_endpoint_hashes = {
        "coordinate_sha256": ENGINE.sha256_array(endpoint["coordinates"]),
        "energy_sha256": ENGINE.sha256_array(endpoint["energies"]),
        "force_sha256": ENGINE.sha256_array(endpoint["forces"]),
    }
    require(contract.get("endpoint_cache_arrays") == expected_endpoint_hashes, "endpoint cache array hashes drifted")
    physical = contract.get("physical_call_policy", {})
    require(physical.get("stored_joint_EF_values_total") == TOTAL_STORED_JOINT_EF, "stored E/F total drifted")
    require(physical.get("maximum_fresh_attempt_started_records") == MAXIMUM_FRESH_ATTEMPT_STARTED, "attempt cap drifted")
    require(physical.get("original_discrete_new_calls") == 0 and physical.get("optimization_N_F_contribution") == 0, "NoF accounting drifted")
    locks = [Path(value).expanduser().resolve() for value in contract["host_local_lock_paths"]]
    require(len(locks) == len(set(locks)) == 2 and set(locks) == CANONICAL_LOCKS, "canonical remote lock set drifted")
    result_output = Path(contract["result_output"]).expanduser().resolve()
    require(result_output == CANONICAL_RESULT, "canonical result path drifted")
    return {
        "contract": contract,
        "contract_path": contract_path,
        "contract_sha256": ENGINE.sha256_file(contract_path),
        "package": package,
        "selection_path": selection_path,
        "rows": rows,
        "endpoint": endpoint,
        "topology": Path(contract["bindings"]["topology"]["path"]).resolve(strict=True),
        "locks": locks,
        "result_output": result_output,
    }


def _shard_path(output: Path, row: dict) -> Path:
    call = int(row["path_identifier"]["calls_per_image"])
    nf = int(row["optimization_N_F"])
    return (
        output.with_suffix("")
        .with_name(output.stem + ".shards")
        / f"ivm_call{call:04d}_nf{nf:05d}_arc500_joint_ef_late_r1.npz"
    )


def _wal_paths(output: Path) -> tuple[Path, Path]:
    return output.with_suffix(".sealed_source.wal.jsonl"), output.with_suffix(
        ".dense_joint_ef.wal.jsonl"
    )


def _final_paths(output: Path) -> tuple[Path, Path, Path]:
    return output, output.with_suffix(".manifest.json"), output.with_suffix(".audit.json")


def _read_wal(path: Path, schema: str, state: dict) -> list[dict]:
    if not path.exists():
        return []
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require(events, f"empty WAL: {path}")
    header = events[0]
    require(header.get("event") == "wal_header" and header.get("schema") == schema, f"invalid WAL header: {path}")
    require(header.get("contract_sha256") == state["contract_sha256"], "WAL contract hash mismatch")
    require(Path(header["contract"]).resolve() == state["contract_path"], "WAL contract path mismatch")
    return events


def _lifetime(events: list[dict]) -> dict:
    starts = [event for event in events if event.get("event") == "attempt_started"]
    successes = [event for event in events if event.get("event") == "attempt_succeeded"]
    failures = [event for event in events if event.get("event") == "attempt_failed"]
    start_ids = [str(event["attempt_id"]) for event in starts]
    require(len(start_ids) == len(set(start_ids)), "duplicate attempt id")
    success_ids = {str(event["attempt_id"]) for event in successes}
    failure_ids = {str(event["attempt_id"]) for event in failures}
    require(success_ids.isdisjoint(failure_ids), "attempt has conflicting terminals")
    require(success_ids | failure_ids <= set(start_ids), "attempt terminal lacks start")
    logical = {(int(event["row_index"]), int(event["point_index"])) for event in starts}
    return {
        "attempt_started": len(starts),
        "attempt_succeeded": len(successes),
        "attempt_failed": len(failures),
        "attempt_open": len(set(start_ids) - success_ids - failure_ids),
        "retry_attempts": len(starts) - len(logical),
    }


def _validate_shard(path: Path, row: dict, state: dict) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "schema", "status", "method", "label", "selection_role",
            "optimization_N_F", "path_identifier_json", "contract_sha256",
            "evaluator_sha256", "joint_engine_sha256", "bb_core_sha256",
            "topology_sha256", "selection_sha256", "all_paths_result_sha256",
            "exact_verification_sha256", "selected_path", "selected_path_sha256",
            "discrete_I_perp", "discrete_I_perp_source",
            "source_counter_vector", "source_force_request_vector",
            "source_counter_vector_sha256", "source_force_request_vector_sha256",
            "dense_coordinates_A", "dense_energy_eV", "dense_force_eV_per_A",
            "dense_coordinate_sha256", "dense_energy_sha256", "dense_force_sha256",
            "densified_Delta_E_kcal_per_mol", "number_of_points",
            "stored_joint_EF_values", "fresh_dense_joint_EF_attempts",
            "inherited_endpoint_joint_EF_values", "original_discrete_new_calls",
            "optimization_N_F_contribution",
        }
        require(required.issubset(saved.files), f"incomplete shard {path}")
        require(str(scalar(saved, "schema")) == SHARD_SCHEMA and str(scalar(saved, "status")) == "PASS", "shard identity mismatch")
        require(str(scalar(saved, "method")) == "IVM", "shard method mismatch")
        require(str(scalar(saved, "label")) == row["label"] and str(scalar(saved, "selection_role")) == row["selection_role"], "shard label/role mismatch")
        require(int(scalar(saved, "optimization_N_F")) == row["optimization_N_F"], "shard N_F mismatch")
        require(json.loads(str(scalar(saved, "path_identifier_json"))) == row["path_identifier"], "shard identifier mismatch")
        bindings = {
            "contract_sha256": state["contract_sha256"],
            "evaluator_sha256": ENGINE.sha256_file(Path(__file__)),
            "joint_engine_sha256": ENGINE.sha256_file(ENGINE_PATH),
            "bb_core_sha256": ENGINE.sha256_file(ROOT / "src/bb_core.py"),
            "topology_sha256": ENGINE.sha256_file(state["topology"]),
            "selection_sha256": ENGINE.sha256_file(state["selection_path"]),
            "all_paths_result_sha256": ENGINE.sha256_file(row["result_path"]),
            "exact_verification_sha256": ENGINE.sha256_file(row["verification_path"]),
        }
        for key, expected in bindings.items():
            require(str(scalar(saved, key)) == expected, f"shard {key} mismatch")
        selected = np.asarray(saved["selected_path"], dtype=np.float64)
        counter = np.asarray(saved["source_counter_vector"], dtype=np.int64)
        force_requests = np.asarray(saved["source_force_request_vector"], dtype=np.int64)
        dense = np.asarray(saved["dense_coordinates_A"], dtype=np.float64)
        energy = np.asarray(saved["dense_energy_eV"], dtype=np.float64)
        force = np.asarray(saved["dense_force_eV_per_A"], dtype=np.float64)
        require(selected.shape == (N_IMAGES, DIMENSION), "shard path shape mismatch")
        require(counter.shape == force_requests.shape == (N_IMAGES,), "shard source counter shape mismatch")
        require(dense.shape == force.shape == (N_DENSE, DIMENSION) and energy.shape == (N_DENSE,), "shard dense shape mismatch")
        require(all(np.all(np.isfinite(value)) for value in (selected, dense, energy, force)), "shard contains non-finite arrays")
        require(verified_array_sha256(selected) == row["path_sha256"] == str(scalar(saved, "selected_path_sha256")), "shard path hash mismatch")
        require(ENGINE.sha256_array(counter) == row["source_counter_vector_sha256"] == str(scalar(saved, "source_counter_vector_sha256")), "shard counter proof mismatch")
        require(ENGINE.sha256_array(force_requests) == row["source_force_request_vector_sha256"] == str(scalar(saved, "source_force_request_vector_sha256")), "shard force-request proof mismatch")
        require(np.array_equal(counter, row["source_counter_vector"]) and np.array_equal(force_requests, row["source_force_request_vector"]), "shard source proof arrays drifted")
        require(same_float(scalar(saved, "discrete_I_perp"), row["discrete_I_perp"]), "shard discrete Iperp mismatch")
        require(str(scalar(saved, "discrete_I_perp_source")) == "exact_verified_online_internal_forces_on_same_selected_path", "shard Iperp source mismatch")
        expected_dense, _ = ENGINE.uniform_arc500(selected)
        require(np.array_equal(dense, expected_dense), "shard coordinates are not deterministic arc500")
        require(np.array_equal(dense[[0, -1]], state["endpoint"]["coordinates"]), "shard endpoints drifted")
        require(np.array_equal(energy[[0, -1]], state["endpoint"]["energies"]), "shard endpoint energies are not inherited")
        require(np.array_equal(force[[0, -1]], state["endpoint"]["forces"]), "shard endpoint forces are not inherited")
        require(ENGINE.sha256_array(dense) == str(scalar(saved, "dense_coordinate_sha256")), "shard dense coordinate hash mismatch")
        require(ENGINE.sha256_array(energy) == str(scalar(saved, "dense_energy_sha256")), "shard dense energy hash mismatch")
        require(ENGINE.sha256_array(force) == str(scalar(saved, "dense_force_sha256")), "shard dense force hash mismatch")
        delta = float((np.max(energy) - energy[0]) * EV_TO_KCAL)
        require(same_float(delta, scalar(saved, "densified_Delta_E_kcal_per_mol")), "shard Delta E mismatch")
        require(int(scalar(saved, "number_of_points")) == N_DENSE and int(scalar(saved, "stored_joint_EF_values")) == N_DENSE, "shard stored point count mismatch")
        require(int(scalar(saved, "fresh_dense_joint_EF_attempts")) == N_FRESH_PER_PATH and int(scalar(saved, "inherited_endpoint_joint_EF_values")) == 2, "shard fresh/inherited count mismatch")
        require(int(scalar(saved, "original_discrete_new_calls")) == 0 and int(scalar(saved, "optimization_N_F_contribution")) == 0, "shard NoF accounting mismatch")
        return {key: np.asarray(saved[key]).copy() for key in saved.files}


def _validate_dense_closure(events: list[dict], row: dict, shard: dict, path: Path) -> None:
    row_events = [event for event in events if int(event.get("row_index", -1)) == row["row_index"]]
    starts = [event for event in row_events if event.get("event") == "attempt_started"]
    successes = [event for event in row_events if event.get("event") == "attempt_succeeded"]
    failures = [event for event in row_events if event.get("event") == "attempt_failed"]
    require(len(starts) == len(successes) == N_FRESH_PER_PATH and not failures, f"{row['label']} dense WAL incomplete")
    start_by_id = {str(event["attempt_id"]): event for event in starts}
    success_by_point = {}
    for event in successes:
        attempt_id = str(event["attempt_id"])
        require(attempt_id in start_by_id, "dense success lacks start")
        started = start_by_id[attempt_id]
        for key in ("row_index", "label", "optimization_N_F", "path_sha256", "point_index", "coordinate_sha256", "session_id"):
            require(event.get(key) == started.get(key), f"dense start/success {key} mismatch")
        point = int(event["point_index"])
        require(point not in success_by_point, "duplicate dense success")
        success_by_point[point] = event
    require(sorted(success_by_point) == list(range(1, N_DENSE - 1)), "dense interior coverage mismatch")
    dense = np.asarray(shard["dense_coordinates_A"])
    energy = np.asarray(shard["dense_energy_eV"])
    force = np.asarray(shard["dense_force_eV_per_A"])
    for point in range(1, N_DENSE - 1):
        event = success_by_point[point]
        require(event["coordinate_sha256"] == ENGINE.sha256_array(dense[point]), "dense coordinate/WAL mismatch")
        require(same_float(event["energy_eV"], energy[point]), "dense energy/WAL mismatch")
        require(event["force_sha256"] == ENGINE.sha256_array(force[point]), "dense force/WAL mismatch")
    publications = [event for event in row_events if event.get("event") == "complete_shard_published"]
    require(any(event.get("shard") == str(path) and event.get("shard_sha256") == ENGINE.sha256_file(path) for event in publications), "dense WAL lacks shard publication")


def _validate_source_closure(events: list[dict], row: dict, path: Path) -> None:
    proofs = [event for event in events if event.get("event") == "sealed_online_discrete_proof_reused" and int(event.get("row_index", -1)) == row["row_index"]]
    require(any(event.get("path_sha256") == row["path_sha256"] and same_float(event.get("discrete_I_perp"), row["discrete_I_perp"]) and event.get("new_physical_calls") == 0 for event in proofs), "sealed discrete proof closure missing")
    endpoints = [event for event in events if event.get("event") == "endpoint_joint_EF_cache_reused" and int(event.get("row_index", -1)) == row["row_index"]]
    require(any(event.get("inherited_values") == 2 and event.get("new_physical_calls") == 0 for event in endpoints), "endpoint cache closure missing")
    publications = [event for event in events if event.get("event") == "complete_shard_published" and int(event.get("row_index", -1)) == row["row_index"]]
    require(any(event.get("shard") == str(path) and event.get("shard_sha256") == ENGINE.sha256_file(path) for event in publications), "source WAL lacks shard publication")


def check_remote_host(expected: dict) -> dict:
    observed = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "machine": platform.machine(),
    }
    require(observed["hostname"] == expected["hostname"], f"wrong execution host: {observed['hostname']}")
    require(observed["platform"] == expected["platform"], f"wrong execution platform: {observed['platform']}")
    require(observed["machine"] == expected["machine"], f"wrong execution machine: {observed['machine']}")
    return observed


def preflight(
    state: dict,
    output: Path,
    *,
    probe_locks: bool = True,
    probe_remote_host: bool = True,
) -> dict:
    output = Path(output).expanduser().resolve()
    require(output == state["result_output"] or state.get("allow_test_output") is True, "output differs from frozen result path")
    for final in _final_paths(output):
        if final.exists():
            raise FileExistsError(f"refusing to overwrite {final}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path, dense_path = _wal_paths(output)
    source_events = _read_wal(source_path, SOURCE_WAL_SCHEMA, state)
    dense_events = _read_wal(dense_path, DENSE_WAL_SCHEMA, state)
    lifetime = _lifetime(dense_events)
    require(lifetime["attempt_started"] <= MAXIMUM_FRESH_ATTEMPT_STARTED, "physical-attempt cap exceeded")
    require(lifetime["attempt_failed"] == lifetime["attempt_open"] == lifetime["retry_attempts"] == 0, "open/failed/retried attempt requires a superseding contract")
    complete = []
    for row in state["rows"]:
        target = _shard_path(output, row)
        row_starts = [event for event in dense_events if event.get("event") == "attempt_started" and int(event.get("row_index", -1)) == row["row_index"]]
        if target.exists():
            shard = _validate_shard(target, row, state)
            _validate_dense_closure(dense_events, row, shard, target)
            _validate_source_closure(source_events, row, target)
            complete.append(row["optimization_N_F"])
        else:
            require(not row_starts, f"{row['label']} has incomplete attempts; a superseding contract is required")
    require(lifetime["attempt_started"] == len(complete) * N_FRESH_PER_PATH, "WAL attempts are not explained by complete shards")
    missing = N_PATHS - len(complete)
    required_free = MINIMUM_FREE_BYTES_RESERVE + missing * ESTIMATED_SHARD_BYTES
    observed_free = shutil.disk_usage(output.parent).free
    require(observed_free >= required_free, f"insufficient free disk: {observed_free} < {required_free}")
    observed_host = check_remote_host(state["contract"]["remote_identity"]) if probe_remote_host else None
    if probe_locks:
        with ENGINE.acquire_two_locks(state["locks"]):
            pass
    return {
        "status": "PASS_ZERO_OPENMM_REMOTE_PREFLIGHT",
        "remote_host_probe": observed_host if probe_remote_host else "SKIPPED_FOR_TEST",
        "mutagen_freeze_gate": state["contract"]["mutagen_gate_at_freeze"],
        "selected_path_count": N_PATHS,
        "selected_optimization_N_F": [row["optimization_N_F"] for row in state["rows"]],
        "terminal_partial_tail_selected": False,
        "original_discrete_new_calls": 0,
        "stored_joint_EF_values_planned": missing * N_DENSE,
        "fresh_joint_EF_attempts_planned": missing * N_FRESH_PER_PATH,
        "inherited_endpoint_joint_EF_values_planned": missing * 2,
        "full_force_vectors_planned": missing * N_DENSE,
        "complete_shards_reusable": complete,
        "lifetime": lifetime,
        "maximum_fresh_attempt_started_records": MAXIMUM_FRESH_ATTEMPT_STARTED,
        "observed_free_bytes": observed_free,
        "required_free_bytes": required_free,
        "lock_probe": "PASS_NONBLOCKING" if probe_locks else "SKIPPED_FOR_TEST",
        "optimization_N_F_contribution": 0,
        "OpenMM_backend_constructed": False,
    }


def _attempt(wal, backend, row: dict, point: int, coordinate: np.ndarray):
    if not hasattr(wal, "late_attempt_started_count"):
        wal.late_attempt_started_count = _lifetime(
            [json.loads(line) for line in wal.path.read_text().splitlines()]
        )["attempt_started"]
    ordinal = int(wal.late_attempt_started_count) + 1
    require(ordinal <= MAXIMUM_FRESH_ATTEMPT_STARTED, "fresh physical-attempt authorization exhausted")
    coordinate = np.ascontiguousarray(coordinate, dtype=np.float64)
    attempt_id = f"{wal.session_id}:{row['row_index']}:{point}:{ordinal}:{uuid.uuid4().hex}"
    started = {
        "event": "attempt_started", "utc": utc_now(),
        "stage": "dense_arc500_interior_joint_EF",
        "row_index": row["row_index"], "label": row["label"],
        "optimization_N_F": row["optimization_N_F"],
        "path_sha256": row["path_sha256"], "point_index": point,
        "coordinate_sha256": ENGINE.sha256_array(coordinate),
        "attempt_ordinal": ordinal, "session_id": wal.session_id,
        "attempt_id": attempt_id,
    }
    wal.append(started)
    wal.late_attempt_started_count = ordinal
    try:
        energy, force = backend.evaluate_joint(coordinate)
        energy = float(energy)
        force = np.ascontiguousarray(force, dtype=np.float64)
        require(force.shape == (DIMENSION,), "backend returned wrong force shape")
        require(np.isfinite(energy) and np.all(np.isfinite(force)), "backend returned non-finite E/F")
    except Exception as error:
        wal.append({**started, "event": "attempt_failed", "utc": utc_now(), "error": repr(error)})
        raise
    wal.append({**started, "event": "attempt_succeeded", "utc": utc_now(), "energy_eV": energy, "energy_scalar_sha256": ENGINE.sha256_array(np.asarray(energy, dtype=np.float64)), "force_sha256": ENGINE.sha256_array(force)})
    return energy, force


def _evaluate_one(state: dict, row: dict, output: Path, backend, source_wal, dense_wal) -> tuple[dict, Path]:
    target = _shard_path(output, row)
    source_wal.append({
        "event": "sealed_online_discrete_proof_reused", "utc": utc_now(),
        "row_index": row["row_index"], "label": row["label"],
        "optimization_N_F": row["optimization_N_F"],
        "path_sha256": row["path_sha256"],
        "discrete_I_perp": row["discrete_I_perp"],
        "source_counter_vector_sha256": row["source_counter_vector_sha256"],
        "source_force_request_vector_sha256": row["source_force_request_vector_sha256"],
        "internal_force_lineage": "exact_verified_online_forces_on_same_selected_path",
        "new_physical_calls": 0,
    })
    source_wal.append({
        "event": "endpoint_joint_EF_cache_reused", "utc": utc_now(),
        "row_index": row["row_index"], "label": row["label"],
        "optimization_N_F": row["optimization_N_F"],
        "path_sha256": row["path_sha256"],
        "endpoint_cache_sha256": state["contract"]["bindings"]["endpoint_joint_EF_cache"]["sha256"],
        "inherited_values": 2, "new_physical_calls": 0,
    })
    dense, geometry = ENGINE.uniform_arc500(row["path"])
    require(np.array_equal(dense[[0, -1]], state["endpoint"]["coordinates"]), "arc500 endpoints differ from cache")
    energy = np.empty(N_DENSE, dtype=np.float64)
    force = np.empty((N_DENSE, DIMENSION), dtype=np.float64)
    energy[[0, -1]] = state["endpoint"]["energies"]
    force[[0, -1]] = state["endpoint"]["forces"]
    dense_wal.append({
        "event": "path_stage_started", "utc": utc_now(),
        "stage": "dense_arc500_interior_joint_EF", "row_index": row["row_index"],
        "label": row["label"], "optimization_N_F": row["optimization_N_F"],
        "path_sha256": row["path_sha256"],
        "dense_coordinate_sha256": ENGINE.sha256_array(dense),
        "number_of_points": N_DENSE, "fresh_points": N_FRESH_PER_PATH,
        "inherited_endpoint_points": 2,
    })
    for point in range(1, N_DENSE - 1):
        energy[point], force[point] = _attempt(dense_wal, backend, row, point, dense[point])
    delta = float((np.max(energy) - energy[0]) * EV_TO_KCAL)
    dense_wal.append({
        "event": "path_stage_data_complete_uncommitted", "utc": utc_now(),
        "stage": "dense_arc500_interior_joint_EF", "row_index": row["row_index"],
        "label": row["label"], "optimization_N_F": row["optimization_N_F"],
        "path_sha256": row["path_sha256"], "joint_attempts": N_FRESH_PER_PATH,
        "joint_successes": N_FRESH_PER_PATH, "stored_joint_EF_values": N_DENSE,
        "dense_coordinate_sha256": ENGINE.sha256_array(dense),
        "dense_energy_sha256": ENGINE.sha256_array(energy),
        "dense_force_sha256": ENGINE.sha256_array(force),
        "densified_Delta_E_kcal_per_mol": delta,
    })
    evidence = state["contract"]["selection_and_replay_evidence"]
    arrays = {
        "schema": np.asarray(SHARD_SCHEMA), "status": np.asarray("PASS"),
        "method": np.asarray("IVM"), "label": np.asarray(row["label"]),
        "selection_role": np.asarray(row["selection_role"]),
        "optimization_N_F": np.asarray(row["optimization_N_F"], dtype=np.int64),
        "path_identifier_json": np.asarray(json.dumps(row["path_identifier"], sort_keys=True)),
        "contract_sha256": np.asarray(state["contract_sha256"]),
        "evaluator_sha256": np.asarray(ENGINE.sha256_file(Path(__file__))),
        "joint_engine_sha256": np.asarray(ENGINE.sha256_file(ENGINE_PATH)),
        "bb_core_sha256": np.asarray(ENGINE.sha256_file(ROOT / "src/bb_core.py")),
        "topology_sha256": np.asarray(ENGINE.sha256_file(state["topology"])),
        "selection_sha256": np.asarray(evidence["selection"]["sha256"]),
        "all_paths_result_sha256": np.asarray(evidence["all_paths_result"]["sha256"]),
        "exact_verification_sha256": np.asarray(evidence["exact_all_paths_verification"]["sha256"]),
        "selected_path": row["path"], "selected_path_sha256": np.asarray(row["path_sha256"]),
        "discrete_I_perp": np.asarray(row["discrete_I_perp"], dtype=np.float64),
        "discrete_I_perp_source": np.asarray("exact_verified_online_internal_forces_on_same_selected_path"),
        "source_counter_vector": row["source_counter_vector"],
        "source_force_request_vector": row["source_force_request_vector"],
        "source_counter_vector_sha256": np.asarray(row["source_counter_vector_sha256"]),
        "source_force_request_vector_sha256": np.asarray(row["source_force_request_vector_sha256"]),
        "dense_coordinates_A": dense, "dense_energy_eV": energy,
        "dense_force_eV_per_A": force,
        "dense_coordinate_sha256": np.asarray(ENGINE.sha256_array(dense)),
        "dense_energy_sha256": np.asarray(ENGINE.sha256_array(energy)),
        "dense_force_sha256": np.asarray(ENGINE.sha256_array(force)),
        "densified_Delta_E_kcal_per_mol": np.asarray(delta, dtype=np.float64),
        "total_arclength_A": np.asarray(geometry["total_arclength_A"]),
        "dense_spacing_A": np.asarray(geometry["dense_spacing_A"]),
        "number_of_points": np.asarray(N_DENSE, dtype=np.int64),
        "stored_joint_EF_values": np.asarray(N_DENSE, dtype=np.int64),
        "fresh_dense_joint_EF_attempts": np.asarray(N_FRESH_PER_PATH, dtype=np.int64),
        "inherited_endpoint_joint_EF_values": np.asarray(2, dtype=np.int64),
        "original_discrete_new_calls": np.asarray(0, dtype=np.int64),
        "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
    }
    ENGINE._atomic_save_npz_no_overwrite(target, arrays)
    shard = _validate_shard(target, row, state)
    published = {
        "event": "complete_shard_published", "utc": utc_now(),
        "row_index": row["row_index"], "label": row["label"],
        "optimization_N_F": row["optimization_N_F"],
        "path_sha256": row["path_sha256"], "shard": str(target),
        "shard_sha256": ENGINE.sha256_file(target),
    }
    dense_wal.append(published)
    source_wal.append(published)
    return shard, target


def _publish_final(state: dict, output: Path, shards: list[dict], paths: list[Path], source_wal_path: Path, dense_wal_path: Path, held_locks: list[str], fresh: list[int], reused: list[int]) -> None:
    arrays = {
        "schema": np.asarray(RESULT_SCHEMA), "status": np.asarray("PASS"),
        "method": np.asarray("IVM"),
        "authoritative_data_unit": np.asarray("complete_atomic_path_shard"),
        "labels": np.asarray([str(scalar(shard, "label")) for shard in shards]),
        "selection_role": np.asarray([str(scalar(shard, "selection_role")) for shard in shards]),
        "optimization_N_F": np.asarray([int(scalar(shard, "optimization_N_F")) for shard in shards], dtype=np.int64),
        "selected_path_sha256": np.asarray([str(scalar(shard, "selected_path_sha256")) for shard in shards]),
        "discrete_I_perp": np.asarray([float(scalar(shard, "discrete_I_perp")) for shard in shards]),
        "energy_eV": np.asarray([shard["dense_energy_eV"] for shard in shards]),
        "energy_sha256": np.asarray([str(scalar(shard, "dense_energy_sha256")) for shard in shards]),
        "dense_coordinate_sha256": np.asarray([str(scalar(shard, "dense_coordinate_sha256")) for shard in shards]),
        "dense_force_sha256": np.asarray([str(scalar(shard, "dense_force_sha256")) for shard in shards]),
        "densified_Delta_E_kcal_per_mol": np.asarray([float(scalar(shard, "densified_Delta_E_kcal_per_mol")) for shard in shards]),
        "shard_path": np.asarray([str(path) for path in paths]),
        "shard_sha256": np.asarray([ENGINE.sha256_file(path) for path in paths]),
        "number_of_points": np.asarray(N_DENSE, dtype=np.int64),
        "stored_joint_EF_values": np.asarray(TOTAL_STORED_JOINT_EF, dtype=np.int64),
        "fresh_joint_EF_attempts": np.asarray(MAXIMUM_FRESH_ATTEMPT_STARTED, dtype=np.int64),
        "inherited_endpoint_joint_EF_values": np.asarray(2 * N_PATHS, dtype=np.int64),
        "full_force_vectors_stored": np.asarray(TOTAL_STORED_JOINT_EF, dtype=np.int64),
        "original_discrete_new_calls": np.asarray(0, dtype=np.int64),
        "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
        "contract": np.asarray(str(state["contract_path"])),
        "contract_sha256": np.asarray(state["contract_sha256"]),
    }
    ENGINE._atomic_save_npz_no_overwrite(output, arrays)
    dense_events = _read_wal(dense_wal_path, DENSE_WAL_SCHEMA, state)
    source_events = _read_wal(source_wal_path, SOURCE_WAL_SCHEMA, state)
    lifetime = _lifetime(dense_events)
    require(lifetime == {"attempt_started": MAXIMUM_FRESH_ATTEMPT_STARTED, "attempt_succeeded": MAXIMUM_FRESH_ATTEMPT_STARTED, "attempt_failed": 0, "attempt_open": 0, "retry_attempts": 0}, "final dense WAL accounting mismatch")
    for row, shard, path in zip(state["rows"], shards, paths):
        _validate_dense_closure(dense_events, row, shard, path)
        _validate_source_closure(source_events, row, path)
    manifest = {
        "schema": MANIFEST_SCHEMA, "status": "PASS", "created_utc": utc_now(),
        "contract": bind(state["contract_path"]), "aggregate": bind(output),
        "shards": [
            {
                "optimization_N_F": row["optimization_N_F"],
                "calls_per_image": row["path_identifier"]["calls_per_image"],
                "path": str(path), "sha256": ENGINE.sha256_file(path),
                "selected_path_sha256": row["path_sha256"],
                "dense_coordinate_sha256": str(scalar(shard, "dense_coordinate_sha256")),
                "dense_energy_sha256": str(scalar(shard, "dense_energy_sha256")),
                "dense_force_sha256": str(scalar(shard, "dense_force_sha256")),
            }
            for row, shard, path in zip(state["rows"], shards, paths)
        ],
    }
    ENGINE._atomic_write_json_no_overwrite(output.with_suffix(".manifest.json"), manifest)
    audit = {
        "schema": AUDIT_SCHEMA, "status": "PASS", "completed_utc": utc_now(),
        "contract": bind(state["contract_path"]), "output": bind(output),
        "manifest": bind(output.with_suffix(".manifest.json")),
        "sealed_source_wal": bind(source_wal_path),
        "dense_joint_ef_wal": bind(dense_wal_path),
        "held_host_locks": held_locks,
        "remote_host": check_remote_host(state["contract"]["remote_identity"]),
        "mutagen_freeze_gate": state["contract"]["mutagen_gate_at_freeze"],
        "fresh_paths_this_invocation": fresh,
        "complete_validated_shards_reused": reused,
        "selected_path_count": N_PATHS,
        "terminal_partial_tail_selected": False,
        "stored_joint_EF_values": TOTAL_STORED_JOINT_EF,
        "fresh_joint_EF_attempt_lifetime": lifetime,
        "inherited_endpoint_joint_EF_values": 2 * N_PATHS,
        "full_force_vectors_stored": TOTAL_STORED_JOINT_EF,
        "original_discrete_new_calls": 0,
        "optimization_N_F_contribution": 0,
    }
    ENGINE._atomic_write_json_no_overwrite(output.with_suffix(".audit.json"), audit)


def run(state: dict, output: Path) -> None:
    report = preflight(state, output, probe_locks=False, probe_remote_host=True)
    source_wal_path, dense_wal_path = _wal_paths(output)
    source_wal = ENGINE.DurableWal(source_wal_path, SOURCE_WAL_SCHEMA, state["contract_path"], state["contract_sha256"])
    dense_wal = ENGINE.DurableWal(dense_wal_path, DENSE_WAL_SCHEMA, state["contract_path"], state["contract_sha256"])
    shards, paths, fresh, reused = [], [], [], []
    backend = None
    try:
        with ENGINE.acquire_two_locks(state["locks"]) as held_locks:
            for row in state["rows"]:
                target = _shard_path(output, row)
                if target.exists():
                    shard = _validate_shard(target, row, state)
                    reused.append(row["optimization_N_F"])
                else:
                    if backend is None:
                        backend = ENGINE.OpenMMBackend(
                            state["topology"], platform_preference=("CPU",),
                            precision="double", label="bb_irivm_late_six_arc500_joint_ef_r1",
                        )
                    before = int(backend.raw_joint_attempts)
                    shard, target = _evaluate_one(state, row, output, backend, source_wal, dense_wal)
                    require(int(backend.raw_joint_attempts) - before == N_FRESH_PER_PATH, "backend attempt ledger mismatch")
                    fresh.append(row["optimization_N_F"])
                shards.append(shard)
                paths.append(target)
            _publish_final(state, output, shards, paths, source_wal_path, dense_wal_path, held_locks, fresh, reused)
    finally:
        source_wal.close()
        dense_wal.close()
    require(report["fresh_joint_EF_attempts_planned"] == len(fresh) * N_FRESH_PER_PATH, "preflight/run fresh count mismatch")


def check_thread_caps() -> None:
    wrong = {name: os.environ.get(name) for name in THREAD_CAPS if os.environ.get(name) != "1"}
    require(not wrong, f"all six thread caps must equal 1: {wrong}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host-role", choices=("remote",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-lock-probe-for-test", action="store_true")
    parser.add_argument("--skip-remote-host-probe-for-test", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    check_thread_caps()
    state = load_contract(args.contract, args.host_role)
    output = args.output.expanduser().resolve()
    require(output.suffix == ".npz", "--output must end in .npz")
    if args.preflight_only:
        report = preflight(
            state, output,
            probe_locks=not args.skip_lock_probe_for_test,
            probe_remote_host=not args.skip_remote_host_probe_for_test,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    require(not args.skip_lock_probe_for_test and not args.skip_remote_host_probe_for_test, "test-only gate bypass cannot execute")
    run(state, output)


if __name__ == "__main__":
    main()
