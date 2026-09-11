#!/usr/bin/env python3
"""Superseding R2 late-six IR--IVM original32 + arc500 joint-E/F evaluator.

For every blind-selected path, R2 freshly evaluates and stores joint energy
and full force at all 30 original internal images, then independently computes
the article's discrete I_perp from those stored forces.  It also stores 500
uniform-Cartesian-arc-length coordinates, energies, and full forces; the two
fixed endpoints inherit the immutable endpoint cache.

An original internal call may supply an arc500 interior value only when the
two float64 coordinate vectors are byte-identical (and array-equal).  The
post-selection contract freezes every such mapping before physical work.  In
the ordinary no-overlap case the fresh upper bound is 30 + 498 = 528 calls per
path, 3168 over six paths.  Offline diagnostics contribute zero optimization
N_F.

R2 retains the R1 package as immutable superseded history.  It remains
remote-host gated, dual-lock serial, append-only/fync audited, atomic-sharded,
and fail-closed: complete shards may resume, but any open/failed/retried or
otherwise incomplete attempted path requires a new superseding contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
from typing import Iterable, Sequence
import uuid

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R1_EVALUATOR_PATH = HERE / "evaluate_bb_irivm_late_six_joint_ef.py"
R1_FREEZER_PATH = HERE / "freeze_bb_irivm_late_six_joint_ef_contract.py"
FREEZER_PATH = HERE / "freeze_bb_irivm_late_six_joint_ef_r2_contract.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


R1 = _load("bb_ir_late_six_r1_for_r2", R1_EVALUATOR_PATH)
ENGINE = R1.ENGINE
SELECTOR = R1.SELECTOR

PACKAGE_SCHEMA = "bb-irivm-late-six-joint-ef-package-preregistration-v2"
PACKAGE_STATUS = "R2_FROZEN_BEFORE_ALL_PATH_RESULT_AND_BEFORE_LATE_METRIC_INSPECTION"
CONTRACT_SCHEMA = "bb-irivm-late-six-original30-arc500-joint-ef-contract-v2"
CONTRACT_STATUS = "R2_FROZEN_AFTER_BLIND_SELECTION_BEFORE_ANY_R2_PHYSICAL_CALL"
SHARD_SCHEMA = "bb-irivm-late-path-original30-arc500-joint-ef-shard-v2"
RESULT_SCHEMA = "bb-irivm-late-six-original30-arc500-joint-ef-result-v2"
MANIFEST_SCHEMA = "bb-irivm-late-six-original30-arc500-joint-ef-manifest-v2"
AUDIT_SCHEMA = "bb-irivm-late-six-original30-arc500-joint-ef-audit-v2"
SOURCE_WAL_SCHEMA = "bb-irivm-late-six-original30-source-wal-v2"
PHYSICAL_WAL_SCHEMA = "bb-irivm-late-six-original30-arc500-physical-wal-v2"

N_PATHS = 6
N_IMAGES = 32
N_INTERNAL = 30
DIMENSION = 9396
N_DENSE = 500
N_DENSE_FRESH_CANDIDATES = N_DENSE - 2
MAXIMUM_FRESH_PER_PATH = N_INTERNAL + N_DENSE_FRESH_CANDIDATES
MAXIMUM_FRESH_ATTEMPT_STARTED = N_PATHS * MAXIMUM_FRESH_PER_PATH
TOTAL_DENSE_STORED_JOINT_EF = N_PATHS * N_DENSE
TOTAL_ORIGINAL_INTERNAL_STORED_JOINT_EF = N_PATHS * N_INTERNAL
MINIMUM_FREE_BYTES_RESERVE = 6 * 1024**3
ESTIMATED_SHARD_BYTES = (
    (2 * N_DENSE * DIMENSION + 2 * N_INTERNAL * DIMENSION + N_IMAGES * DIMENSION)
    * 8
    + 12 * 1024**2
)
IPERP_CONSISTENCY_RTOL = 1.0e-8
IPERP_CONSISTENCY_ATOL = 1.0e-10

R1_PACKAGE = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_LATE_SIX_JOINT_EF_PACKAGE_PREREG_20260902_R1.json"
).resolve()
R1_PREPARATION_AUDIT = (
    ROOT
    / "reports/preflights/"
    "BB_IRIVM_LATE_SIX_JOINT_EF_FUTURE_PACKAGE_PREPARATION_20260902_R1.json"
).resolve()
CANONICAL_PACKAGE_PREREGISTRATION = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_LATE_SIX_JOINT_EF_PACKAGE_PREREG_20260902_R2.json"
).resolve()
CANONICAL_CONTRACT = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_LATE_SIX_JOINT_EF_ARC500_20260902_R2.json"
).resolve()
CANONICAL_RESULT = (
    ROOT
    / "runs/diagnostics/bb_irivm_late_six_arc500_joint_ef_20260902_r2.npz"
).resolve()

METRIC_POLICY = {
    **R1.METRIC_POLICY,
    "fresh_original_internal_joint_EF_calls_per_path": 30,
    "store_original_internal_energy": True,
    "store_original_internal_full_force": True,
    "discrete_I_perp_source": (
        "independently_recomputed_from_stored_fresh_joint_full_forces_at_"
        "the_30_internal_images_of_the_same_selected_original32_path"
    ),
    "discrete_I_perp_new_calls": 30,
    "original_to_dense_deduplication": (
        "only_byte_identical_float64_coordinates_with_array_equal_confirmation"
    ),
    "maximum_fresh_joint_EF_calls_per_path": 528,
    "maximum_fresh_joint_EF_calls_total": 3168,
    "optimization_N_F_contribution": 0,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bind(path: Path) -> dict:
    return R1.bind(path)


def scalar(saved, key: str):
    return R1.scalar(saved, key)


def same_float(left, right) -> bool:
    return R1.same_float(left, right)


def _read_json(path: Path) -> dict:
    return R1._read_json(path)


def exact_overlap_plan(path: np.ndarray) -> dict:
    """Freeze same-path original-to-dense reuse using exact float64 identity."""

    selected = np.ascontiguousarray(np.asarray(path, dtype=np.float64))
    require(selected.shape == (N_IMAGES, DIMENSION), "wrong selected path shape")
    dense, _ = ENGINE.uniform_arc500(selected)
    mapping = [-1] * N_INTERNAL
    used_dense = set()
    for original_offset, coordinate in enumerate(selected[1:-1]):
        original_image = original_offset + 1
        original_bytes = coordinate.tobytes(order="C")
        matches = []
        for dense_point in range(1, N_DENSE - 1):
            candidate = dense[dense_point]
            if (
                candidate.tobytes(order="C") == original_bytes
                and np.array_equal(candidate, coordinate)
            ):
                matches.append(dense_point)
        require(len(matches) <= 1, f"original image {original_image} maps to multiple dense points")
        if matches:
            dense_point = matches[0]
            require(dense_point not in used_dense, "two original images map to one dense point")
            used_dense.add(dense_point)
            mapping[original_offset] = dense_point
    overlap = len(used_dense)
    return {
        "original_internal_image_to_dense_point": mapping,
        "exact_overlap_count": overlap,
        "original_internal_fresh_calls": N_INTERNAL,
        "dense_interior_fresh_calls": N_DENSE_FRESH_CANDIDATES - overlap,
        "fresh_joint_EF_calls": MAXIMUM_FRESH_PER_PATH - overlap,
        "maximum_without_deduplication": MAXIMUM_FRESH_PER_PATH,
        "identity_requirement": (
            "float64_C_order_bytes_equal_and_numpy_array_equal"
        ),
    }


def build_package_preregistration(locks: Sequence[Path]) -> dict:
    resolved_locks = [Path(path).expanduser().resolve() for path in locks]
    require(
        len(resolved_locks) == len(set(resolved_locks)) == 2
        and set(resolved_locks) == R1.CANONICAL_LOCKS,
        "exactly the two canonical BB locks are required",
    )
    R1.validate_package_preregistration(R1_PACKAGE)
    SELECTOR.validate_preregistration(R1.BLIND_PREREGISTRATION)
    R1._load_endpoint_cache()
    return {
        "schema": PACKAGE_SCHEMA,
        "status": PACKAGE_STATUS,
        "frozen": True,
        "created_utc": utc_now(),
        "supersedes": {
            "R1_package": bind(R1_PACKAGE),
            "R1_preparation_audit": bind(R1_PREPARATION_AUDIT),
            "reason": (
                "R1 retained only sealed online scalar/counter proof; R2 also "
                "stores fresh original30 joint energy/full-force arrays and "
                "independently recomputes discrete I_perp"
            ),
            "R1_artifacts_deleted_or_modified": False,
            "R1_physical_results_reused": False,
        },
        "result_or_selection_bound_now": False,
        "host_role": "remote",
        "remote_identity": {
            "hostname": R1.REMOTE_HOSTNAME,
            "address": R1.REMOTE_ADDRESS,
            "platform": "Darwin",
            "machine": R1.REMOTE_MACHINE,
        },
        "required_Mutagen_session": {
            "name": R1.MUTAGEN_SESSION_NAME,
            "identifier": R1.MUTAGEN_SESSION_IDENTIFIER,
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
            "fresh_original_internal_joint_EF_per_path": N_INTERNAL,
            "dense_joint_EF_values_stored_per_path": N_DENSE,
            "dense_endpoint_joint_EF_values_inherited_per_path": 2,
            "dense_interior_fresh_candidates_per_path": N_DENSE_FRESH_CANDIDATES,
            "maximum_fresh_joint_EF_per_path": MAXIMUM_FRESH_PER_PATH,
            "maximum_fresh_attempt_started_records": MAXIMUM_FRESH_ATTEMPT_STARTED,
            "exact_post_selection_deduplication_may_reduce_calls": True,
            "deduplication_requires_exact_coordinate_identity": True,
            "optimization_N_F_contribution": 0,
            "jobs": 1,
            "all_six_thread_caps": 1,
            "two_host_local_fcntl_locks": True,
        },
        "storage_policy": {
            "authoritative_unit": "atomic_complete_path_shard",
            "each_shard_stores_selected_original32_path": True,
            "each_shard_stores_original30_energy_and_full_force": True,
            "each_shard_stores_arc500_coordinates_energy_and_full_force": True,
            "aggregate_does_not_duplicate_large_force_or_coordinate_arrays": True,
        },
        "durability_policy": {
            "atomic_complete_path_shard": True,
            "separate_append_only_fsync_source_and_physical_WAL": True,
            "complete_shard_only_restart": True,
            "individual_partial_success_reuse": False,
            "open_failed_retried_or_incomplete_attempt_requires_superseding_contract": True,
            "no_overwrite": True,
        },
        "post_arrival_pipeline_templates": {
            "working_directory": str(ROOT),
            "python": "/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python",
            "step_1_selector_argv": R1.build_package_preregistration(
                resolved_locks
            )["post_arrival_pipeline_templates"]["step_1_selector_argv"],
            "step_2_contract_freeze_argv": [
                str(FREEZER_PATH),
                "freeze",
                "--package",
                str(CANONICAL_PACKAGE_PREREGISTRATION),
                "--selection",
                str(R1.CANONICAL_SELECTION),
                "--output",
                str(CANONICAL_CONTRACT),
            ],
            "step_2_includes_dedicated_Mutagen_flush_before_inputs_and_after_contract_publish": True,
            "remote_environment": {name: "1" for name in R1.THREAD_CAPS},
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
            "blind_selection_preregistration": bind(R1.BLIND_PREREGISTRATION),
            "blind_selector": bind(R1.SELECTOR_PATH),
            "R1_evaluator_superseded": bind(R1_EVALUATOR_PATH),
            "R1_freezer_superseded": bind(R1_FREEZER_PATH),
            "evaluator": bind(Path(__file__)),
            "contract_freezer": bind(FREEZER_PATH),
            "joint_engine": bind(R1.ENGINE_PATH),
            "bb_core": bind(ROOT / "src/bb_core.py"),
            "topology": bind(R1.TOPOLOGY),
            "common_initial_path": bind(R1.COMMON_INITIAL_PATH),
            "endpoint_joint_EF_cache": bind(R1.ENDPOINT_CACHE),
        },
    }


def validate_package_preregistration(path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    package = _read_json(resolved)
    require(
        package.get("schema") == PACKAGE_SCHEMA
        and package.get("status") == PACKAGE_STATUS
        and package.get("frozen") is True,
        "invalid R2 package preregistration",
    )
    require(package.get("supersedes", {}).get("R1_package") == bind(R1_PACKAGE), "R2/R1 package binding drifted")
    require(package.get("supersedes", {}).get("R1_preparation_audit") == bind(R1_PREPARATION_AUDIT), "R2/R1 audit binding drifted")
    require(package.get("supersedes", {}).get("R1_artifacts_deleted_or_modified") is False and package.get("supersedes", {}).get("R1_physical_results_reused") is False, "R1 history policy drifted")
    require(package.get("result_or_selection_bound_now") is False, "R2 package prematurely binds result")
    require(package.get("host_role") == "remote" and package.get("remote_identity") == {"hostname": R1.REMOTE_HOSTNAME, "address": R1.REMOTE_ADDRESS, "platform": "Darwin", "machine": R1.REMOTE_MACHINE}, "R2 remote identity drifted")
    expected_mutagen = {
        "name": R1.MUTAGEN_SESSION_NAME,
        "identifier": R1.MUTAGEN_SESSION_IDENTIFIER,
        "input_flush_must_succeed_before_final_contract_freeze": True,
        "contract_publish_flush_must_succeed_after_final_contract_freeze": True,
        "alpha_connected": True,
        "beta_connected": True,
        "transition_problems": 0,
    }
    require(package.get("required_Mutagen_session") == expected_mutagen, "R2 Mutagen policy drifted")
    require(Path(package.get("result_output", "")).resolve() == CANONICAL_RESULT and Path(package.get("final_contract_output", "")).resolve() == CANONICAL_CONTRACT, "R2 output path drifted")
    require(package.get("metric_policy") == METRIC_POLICY, "R2 metric policy drifted")
    for role, record in package.get("bindings", {}).items():
        require(bind(Path(record["path"])) == record, f"R2 package binding drift: {role}")
    expected_paths = {
        "blind_selection_preregistration": R1.BLIND_PREREGISTRATION,
        "blind_selector": R1.SELECTOR_PATH,
        "R1_evaluator_superseded": R1_EVALUATOR_PATH,
        "R1_freezer_superseded": R1_FREEZER_PATH,
        "evaluator": Path(__file__).resolve(),
        "contract_freezer": FREEZER_PATH,
        "joint_engine": R1.ENGINE_PATH,
        "bb_core": ROOT / "src/bb_core.py",
        "topology": R1.TOPOLOGY,
        "common_initial_path": R1.COMMON_INITIAL_PATH,
        "endpoint_joint_EF_cache": R1.ENDPOINT_CACHE,
    }
    for role, expected in expected_paths.items():
        require(Path(package["bindings"][role]["path"]).resolve() == expected.resolve(), f"R2 package {role} path drift")
    locks = [Path(value).expanduser().resolve() for value in package["host_local_lock_paths"]]
    require(len(locks) == len(set(locks)) == 2 and set(locks) == R1.CANONICAL_LOCKS, "R2 lock set drifted")
    physical = package.get("physical_call_policy", {})
    require(physical.get("fresh_original_internal_joint_EF_per_path") == N_INTERNAL, "R2 original call count drifted")
    require(physical.get("maximum_fresh_joint_EF_per_path") == MAXIMUM_FRESH_PER_PATH and physical.get("maximum_fresh_attempt_started_records") == MAXIMUM_FRESH_ATTEMPT_STARTED, "R2 attempt cap drifted")
    require(physical.get("deduplication_requires_exact_coordinate_identity") is True and physical.get("optimization_N_F_contribution") == 0, "R2 dedup/NoF policy drifted")
    storage = package.get("storage_policy", {})
    require(storage.get("each_shard_stores_original30_energy_and_full_force") is True and storage.get("each_shard_stores_arc500_coordinates_energy_and_full_force") is True, "R2 storage policy drifted")
    durability = package.get("durability_policy", {})
    require(durability.get("atomic_complete_path_shard") is True and durability.get("complete_shard_only_restart") is True and durability.get("individual_partial_success_reuse") is False, "R2 restart policy drifted")
    require(durability.get("open_failed_retried_or_incomplete_attempt_requires_superseding_contract") is True and durability.get("no_overwrite") is True, "R2 fail-close policy drifted")
    return package


def frozen_row(row: dict) -> dict:
    plan = exact_overlap_plan(row["path"])
    return {
        **R1.frozen_row(row),
        "discrete_I_perp_online_reference": row["discrete_I_perp"],
        "discrete_I_perp_authoritative_source": (
            "R2_stored_original_internal_force_eV_per_A_recomputed_with_bb_core.path_diagnostics"
        ),
        "original_internal_policy": "fresh_30_joint_energy_full_force_calls_and_store_arrays",
        "exact_original_to_dense_overlap_plan": plan,
        "R2_fresh_joint_EF_calls": plan["fresh_joint_EF_calls"],
        "optimization_N_F_contribution": 0,
    }


def build_final_contract(package_path: Path, selection_path: Path, mutagen_snapshot: dict) -> dict:
    package_path = Path(package_path).expanduser().resolve(strict=True)
    selection_path = Path(selection_path).expanduser().resolve(strict=True)
    package = validate_package_preregistration(package_path)
    R1.validate_mutagen_snapshot(mutagen_snapshot)
    rows, evidence = R1._validate_selection_and_load_rows(selection_path)
    frozen = [frozen_row(row) for row in rows]
    exact_total = sum(row["R2_fresh_joint_EF_calls"] for row in frozen)
    require(exact_total <= MAXIMUM_FRESH_ATTEMPT_STARTED, "R2 exact plan exceeds maximum")
    return {
        "schema": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "frozen": True,
        "created_utc": utc_now(),
        "purpose": "superseding R2 remote late-six original30 plus arc500 full joint E/F",
        "host_role": "remote",
        "remote_identity": package["remote_identity"],
        "jobs": 1,
        "result_output": package["result_output"],
        "host_local_lock_paths": package["host_local_lock_paths"],
        "metric_policy": METRIC_POLICY,
        "mutagen_gate_at_freeze": mutagen_snapshot,
        "package_preregistration": bind(package_path),
        "selection_and_replay_evidence": evidence,
        "paths": frozen,
        "exact_post_selection_call_plan": {
            "fresh_joint_EF_calls_total": exact_total,
            "maximum_fresh_attempt_started_records": MAXIMUM_FRESH_ATTEMPT_STARTED,
            "exact_overlap_count_total": sum(row["exact_original_to_dense_overlap_plan"]["exact_overlap_count"] for row in frozen),
            "per_path": [
                {
                    "label": row["label"],
                    "optimization_N_F": row["optimization_N_F"],
                    **row["exact_original_to_dense_overlap_plan"],
                }
                for row in frozen
            ],
        },
        "physical_call_policy": package["physical_call_policy"],
        "storage_policy": package["storage_policy"],
        "durability_policy": package["durability_policy"],
        "bindings": package["bindings"],
        "optimization_N_F_contribution": 0,
    }


def load_contract(contract_path: Path, host_role: str) -> dict:
    contract_path = Path(contract_path).expanduser().resolve(strict=True)
    contract = _read_json(contract_path)
    require(contract.get("schema") == CONTRACT_SCHEMA and contract.get("status") == CONTRACT_STATUS and contract.get("frozen") is True, "invalid R2 final contract")
    require(host_role == contract.get("host_role") == "remote", "R2 late-six is remote-only")
    package_path = Path(contract["package_preregistration"]["path"]).resolve(strict=True)
    require(contract["package_preregistration"] == bind(package_path), "R2 package binding drifted")
    package = validate_package_preregistration(package_path)
    R1.validate_mutagen_snapshot(contract["mutagen_gate_at_freeze"])
    selection_path = Path(contract["selection_and_replay_evidence"]["selection"]["path"]).resolve(strict=True)
    rows, evidence = R1._validate_selection_and_load_rows(selection_path)
    require(contract.get("selection_and_replay_evidence") == evidence, "R2 selection/replay evidence drifted")
    frozen = [frozen_row(row) for row in rows]
    require(contract.get("paths") == frozen, "R2 frozen rows/dedup plan drifted")
    exact_total = sum(row["R2_fresh_joint_EF_calls"] for row in frozen)
    plan = contract.get("exact_post_selection_call_plan", {})
    require(plan.get("fresh_joint_EF_calls_total") == exact_total and plan.get("maximum_fresh_attempt_started_records") == MAXIMUM_FRESH_ATTEMPT_STARTED, "R2 exact call plan drifted")
    require(plan.get("exact_overlap_count_total") == N_PATHS * MAXIMUM_FRESH_PER_PATH - exact_total, "R2 overlap total drifted")
    require(contract.get("remote_identity") == package["remote_identity"] and contract.get("metric_policy") == METRIC_POLICY, "R2 host/metric policy drifted")
    require(contract.get("bindings") == package["bindings"], "R2 source bindings drifted")
    result_output = Path(contract["result_output"]).expanduser().resolve()
    require(result_output == CANONICAL_RESULT, "R2 result path drifted")
    endpoint = R1._load_endpoint_cache()
    loaded_rows = []
    for row, frozen_item in zip(rows, frozen):
        loaded_rows.append({**row, "R2_frozen": frozen_item, "overlap_plan": frozen_item["exact_original_to_dense_overlap_plan"], "R2_fresh_joint_EF_calls": frozen_item["R2_fresh_joint_EF_calls"]})
    locks = [Path(value).expanduser().resolve() for value in contract["host_local_lock_paths"]]
    require(len(locks) == len(set(locks)) == 2 and set(locks) == R1.CANONICAL_LOCKS, "R2 canonical locks drifted")
    return {
        "contract": contract,
        "contract_path": contract_path,
        "contract_sha256": ENGINE.sha256_file(contract_path),
        "package": package,
        "selection_path": selection_path,
        "rows": loaded_rows,
        "endpoint": endpoint,
        "topology": Path(contract["bindings"]["topology"]["path"]).resolve(strict=True),
        "locks": locks,
        "result_output": result_output,
        "exact_fresh_calls": exact_total,
    }


def _shard_path(output: Path, row: dict) -> Path:
    call = int(row["path_identifier"]["calls_per_image"])
    nf = int(row["optimization_N_F"])
    return output.with_suffix("").with_name(output.stem + ".shards") / f"ivm_call{call:04d}_nf{nf:05d}_original30_arc500_joint_ef_r2.npz"


def _wal_paths(output: Path) -> tuple[Path, Path]:
    return output.with_suffix(".sealed_source.wal.jsonl"), output.with_suffix(".physical_joint_ef.wal.jsonl")


def _final_paths(output: Path) -> tuple[Path, Path, Path]:
    return output, output.with_suffix(".manifest.json"), output.with_suffix(".audit.json")


def _read_wal(path: Path, schema: str, state: dict) -> list[dict]:
    return R1._read_wal(path, schema, state)


def _lifetime(events: list[dict]) -> dict:
    starts = [event for event in events if event.get("event") == "attempt_started"]
    successes = [event for event in events if event.get("event") == "attempt_succeeded"]
    failures = [event for event in events if event.get("event") == "attempt_failed"]
    start_ids = [str(event["attempt_id"]) for event in starts]
    require(len(start_ids) == len(set(start_ids)), "duplicate R2 attempt id")
    success_ids = {str(event["attempt_id"]) for event in successes}
    failure_ids = {str(event["attempt_id"]) for event in failures}
    require(success_ids.isdisjoint(failure_ids), "R2 attempt has conflicting terminals")
    require(success_ids | failure_ids <= set(start_ids), "R2 terminal lacks attempt start")
    logical = {(int(event["row_index"]), str(event["stage"]), int(event["point_index"])) for event in starts}
    return {
        "attempt_started": len(starts),
        "attempt_succeeded": len(successes),
        "attempt_failed": len(failures),
        "attempt_open": len(set(start_ids) - success_ids - failure_ids),
        "retry_attempts": len(starts) - len(logical),
    }


def _attempt(wal, backend, state: dict, row: dict, stage: str, point: int, coordinate: np.ndarray):
    if not hasattr(wal, "R2_attempt_started_count"):
        wal.R2_attempt_started_count = _lifetime([json.loads(line) for line in wal.path.read_text().splitlines()])["attempt_started"]
    ordinal = int(wal.R2_attempt_started_count) + 1
    require(ordinal <= MAXIMUM_FRESH_ATTEMPT_STARTED, "R2 maximum physical-attempt authorization exhausted")
    coordinate = np.ascontiguousarray(coordinate, dtype=np.float64)
    attempt_id = f"{wal.session_id}:{row['row_index']}:{stage}:{point}:{ordinal}:{uuid.uuid4().hex}"
    started = {
        "event": "attempt_started", "utc": utc_now(), "stage": stage,
        "row_index": row["row_index"], "label": row["label"],
        "optimization_N_F": row["optimization_N_F"], "path_sha256": row["path_sha256"],
        "point_index": point, "coordinate_sha256": ENGINE.sha256_array(coordinate),
        "attempt_ordinal": ordinal, "session_id": wal.session_id, "attempt_id": attempt_id,
    }
    wal.append(started)
    wal.R2_attempt_started_count = ordinal
    try:
        energy, force = backend.evaluate_joint(coordinate)
        energy = float(energy)
        force = np.ascontiguousarray(force, dtype=np.float64)
        require(force.shape == (DIMENSION,), "R2 backend returned wrong force shape")
        require(np.isfinite(energy) and np.all(np.isfinite(force)), "R2 backend returned non-finite E/F")
    except Exception as error:
        wal.append({**started, "event": "attempt_failed", "utc": utc_now(), "error": repr(error)})
        raise
    wal.append({**started, "event": "attempt_succeeded", "utc": utc_now(), "energy_eV": energy, "energy_scalar_sha256": ENGINE.sha256_array(np.asarray(energy, dtype=np.float64)), "force_sha256": ENGINE.sha256_array(force)})
    return energy, force


def _expected_stage_points(row: dict) -> tuple[list[int], list[int], dict[int, int]]:
    mapping = row["overlap_plan"]["original_internal_image_to_dense_point"]
    inverse = {int(dense): original_offset for original_offset, dense in enumerate(mapping) if int(dense) >= 0}
    original_points = list(range(1, N_INTERNAL + 1))
    dense_fresh = [point for point in range(1, N_DENSE - 1) if point not in inverse]
    return original_points, dense_fresh, inverse


def _validate_shard(path: Path, row: dict, state: dict) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "schema", "status", "method", "label", "selection_role", "optimization_N_F",
            "path_identifier_json", "contract_sha256", "evaluator_sha256", "bb_core_sha256",
            "topology_sha256", "selection_sha256", "all_paths_result_sha256", "exact_verification_sha256",
            "selected_path", "selected_path_sha256", "online_discrete_I_perp_reference",
            "original_internal_coordinates_A", "original_internal_energy_eV", "original_internal_force_eV_per_A",
            "original_internal_coordinate_sha256", "original_internal_energy_sha256", "original_internal_force_sha256",
            "discrete_I_perp", "discrete_I_perp_source", "online_recomputed_Iperp_relative_difference",
            "dense_coordinates_A", "dense_energy_eV", "dense_force_eV_per_A",
            "dense_coordinate_sha256", "dense_energy_sha256", "dense_force_sha256", "densified_Delta_E_kcal_per_mol",
            "exact_original_to_dense_overlap_plan_json", "fresh_joint_EF_attempts", "maximum_fresh_joint_EF_attempts",
            "original_internal_joint_EF_attempts", "dense_interior_fresh_joint_EF_attempts",
            "dense_endpoint_joint_EF_values_inherited", "stored_dense_joint_EF_values", "optimization_N_F_contribution",
        }
        require(required.issubset(saved.files), f"incomplete R2 shard {path}")
        require(str(scalar(saved, "schema")) == SHARD_SCHEMA and str(scalar(saved, "status")) == "PASS", "R2 shard identity mismatch")
        require(str(scalar(saved, "method")) == "IVM" and str(scalar(saved, "label")) == row["label"], "R2 shard method/label mismatch")
        require(str(scalar(saved, "selection_role")) == row["selection_role"] and int(scalar(saved, "optimization_N_F")) == row["optimization_N_F"], "R2 shard role/NF mismatch")
        require(json.loads(str(scalar(saved, "path_identifier_json"))) == row["path_identifier"], "R2 shard path identifier mismatch")
        evidence = state["contract"]["selection_and_replay_evidence"]
        bindings = {
            "contract_sha256": state["contract_sha256"],
            "evaluator_sha256": ENGINE.sha256_file(Path(__file__)),
            "bb_core_sha256": ENGINE.sha256_file(ROOT / "src/bb_core.py"),
            "topology_sha256": ENGINE.sha256_file(state["topology"]),
            "selection_sha256": evidence["selection"]["sha256"],
            "all_paths_result_sha256": evidence["all_paths_result"]["sha256"],
            "exact_verification_sha256": evidence["exact_all_paths_verification"]["sha256"],
        }
        for key, expected in bindings.items():
            require(str(scalar(saved, key)) == expected, f"R2 shard {key} mismatch")
        selected = np.asarray(saved["selected_path"], dtype=np.float64)
        original_c = np.asarray(saved["original_internal_coordinates_A"], dtype=np.float64)
        original_e = np.asarray(saved["original_internal_energy_eV"], dtype=np.float64)
        original_f = np.asarray(saved["original_internal_force_eV_per_A"], dtype=np.float64)
        dense = np.asarray(saved["dense_coordinates_A"], dtype=np.float64)
        dense_e = np.asarray(saved["dense_energy_eV"], dtype=np.float64)
        dense_f = np.asarray(saved["dense_force_eV_per_A"], dtype=np.float64)
        require(selected.shape == (N_IMAGES, DIMENSION), "R2 shard selected path shape mismatch")
        require(original_c.shape == original_f.shape == (N_INTERNAL, DIMENSION) and original_e.shape == (N_INTERNAL,), "R2 shard original internal shape mismatch")
        require(dense.shape == dense_f.shape == (N_DENSE, DIMENSION) and dense_e.shape == (N_DENSE,), "R2 shard dense shape mismatch")
        require(all(np.all(np.isfinite(value)) for value in (selected, original_c, original_e, original_f, dense, dense_e, dense_f)), "R2 shard contains non-finite arrays")
        require(R1.verified_array_sha256(selected) == row["path_sha256"] == str(scalar(saved, "selected_path_sha256")), "R2 shard selected path hash mismatch")
        require(np.array_equal(original_c, selected[1:-1]), "R2 original internal coordinates differ from selected path")
        require(ENGINE.sha256_array(original_c) == str(scalar(saved, "original_internal_coordinate_sha256")), "R2 original coordinate hash mismatch")
        require(ENGINE.sha256_array(original_e) == str(scalar(saved, "original_internal_energy_sha256")), "R2 original energy hash mismatch")
        require(ENGINE.sha256_array(original_f) == str(scalar(saved, "original_internal_force_sha256")), "R2 original force hash mismatch")
        recomputed = float(ENGINE.path_diagnostics(selected, np.zeros(N_IMAGES), original_f)["i_perp"])
        require(same_float(recomputed, scalar(saved, "discrete_I_perp")), "R2 stored discrete Iperp is not independently recomputed")
        require(str(scalar(saved, "discrete_I_perp_source")) == "stored_original_internal_force_eV_per_A_recomputed_by_bb_core.path_diagnostics", "R2 discrete source mismatch")
        online = float(scalar(saved, "online_discrete_I_perp_reference"))
        require(same_float(online, row["discrete_I_perp"]), "R2 online Iperp reference drifted")
        relative = abs(recomputed - online) / max(abs(online), 1.0e-300)
        require(same_float(relative, scalar(saved, "online_recomputed_Iperp_relative_difference")), "R2 Iperp difference audit mismatch")
        require(np.isclose(recomputed, online, rtol=IPERP_CONSISTENCY_RTOL, atol=IPERP_CONSISTENCY_ATOL), "R2 recomputed/online Iperp consistency gate failed")
        expected_dense, _ = ENGINE.uniform_arc500(selected)
        require(np.array_equal(dense, expected_dense), "R2 shard dense coordinates differ from arc500")
        require(np.array_equal(dense[[0, -1]], state["endpoint"]["coordinates"]), "R2 dense endpoints drifted")
        require(np.array_equal(dense_e[[0, -1]], state["endpoint"]["energies"]) and np.array_equal(dense_f[[0, -1]], state["endpoint"]["forces"]), "R2 endpoint E/F not inherited")
        plan = json.loads(str(scalar(saved, "exact_original_to_dense_overlap_plan_json")))
        require(plan == row["overlap_plan"] == exact_overlap_plan(selected), "R2 overlap plan drifted")
        _, _, inverse = _expected_stage_points(row)
        for dense_point, original_offset in inverse.items():
            require(np.array_equal(dense[dense_point], original_c[original_offset]), "R2 dedup coordinate not exact")
            require(same_float(dense_e[dense_point], original_e[original_offset]) and np.array_equal(dense_f[dense_point], original_f[original_offset]), "R2 dedup E/F was not exactly reused")
        require(ENGINE.sha256_array(dense) == str(scalar(saved, "dense_coordinate_sha256")) and ENGINE.sha256_array(dense_e) == str(scalar(saved, "dense_energy_sha256")) and ENGINE.sha256_array(dense_f) == str(scalar(saved, "dense_force_sha256")), "R2 dense array hash mismatch")
        delta = float((np.max(dense_e) - dense_e[0]) * R1.EV_TO_KCAL)
        require(same_float(delta, scalar(saved, "densified_Delta_E_kcal_per_mol")), "R2 Delta E mismatch")
        require(int(scalar(saved, "fresh_joint_EF_attempts")) == row["R2_fresh_joint_EF_calls"] and int(scalar(saved, "maximum_fresh_joint_EF_attempts")) == MAXIMUM_FRESH_PER_PATH, "R2 shard physical count mismatch")
        require(int(scalar(saved, "original_internal_joint_EF_attempts")) == N_INTERNAL and int(scalar(saved, "dense_interior_fresh_joint_EF_attempts")) == row["overlap_plan"]["dense_interior_fresh_calls"], "R2 shard stage call count mismatch")
        require(int(scalar(saved, "dense_endpoint_joint_EF_values_inherited")) == 2 and int(scalar(saved, "stored_dense_joint_EF_values")) == N_DENSE and int(scalar(saved, "optimization_N_F_contribution")) == 0, "R2 shard storage/NoF count mismatch")
        return {key: np.asarray(saved[key]).copy() for key in saved.files}


def _validate_stage(events: list[dict], row: dict, stage: str, expected_points: list[int], coordinates: np.ndarray, energies: np.ndarray, forces: np.ndarray) -> None:
    row_events = [event for event in events if int(event.get("row_index", -1)) == row["row_index"] and event.get("stage") == stage]
    starts = [event for event in row_events if event.get("event") == "attempt_started"]
    successes = [event for event in row_events if event.get("event") == "attempt_succeeded"]
    failures = [event for event in row_events if event.get("event") == "attempt_failed"]
    require(len(starts) == len(successes) == len(expected_points) and not failures, f"{row['label']} {stage} WAL incomplete")
    start_by_id = {str(event["attempt_id"]): event for event in starts}
    success_by_point = {}
    for event in successes:
        attempt_id = str(event["attempt_id"])
        require(attempt_id in start_by_id, "R2 success lacks start")
        started = start_by_id[attempt_id]
        for key in ("stage", "row_index", "label", "optimization_N_F", "path_sha256", "point_index", "coordinate_sha256", "session_id"):
            require(event.get(key) == started.get(key), f"R2 start/success {key} mismatch")
        point = int(event["point_index"])
        require(point not in success_by_point, "R2 duplicate point success")
        success_by_point[point] = event
    require(sorted(success_by_point) == expected_points, f"{row['label']} {stage} coverage mismatch")
    for array_index, point in enumerate(expected_points):
        event = success_by_point[point]
        require(event["coordinate_sha256"] == ENGINE.sha256_array(coordinates[array_index]), "R2 coordinate/WAL mismatch")
        require(same_float(event["energy_eV"], energies[array_index]), "R2 energy/WAL mismatch")
        require(event["force_sha256"] == ENGINE.sha256_array(forces[array_index]), "R2 force/WAL mismatch")


def _validate_wal_closure(events: list[dict], source_events: list[dict], row: dict, shard: dict, path: Path) -> None:
    original_points, dense_fresh, inverse = _expected_stage_points(row)
    _validate_stage(events, row, "original_discrete_internal_joint_EF", original_points, np.asarray(shard["original_internal_coordinates_A"]), np.asarray(shard["original_internal_energy_eV"]), np.asarray(shard["original_internal_force_eV_per_A"]))
    dense = np.asarray(shard["dense_coordinates_A"])
    dense_e = np.asarray(shard["dense_energy_eV"])
    dense_f = np.asarray(shard["dense_force_eV_per_A"])
    _validate_stage(events, row, "dense_arc500_interior_joint_EF", dense_fresh, dense[dense_fresh], dense_e[dense_fresh], dense_f[dense_fresh])
    dedups = [event for event in events if event.get("event") == "exact_coordinate_stage_dedup_reused" and int(event.get("row_index", -1)) == row["row_index"]]
    require(len(dedups) == len(inverse), "R2 dedup WAL count mismatch")
    by_dense = {int(event["dense_point"]): event for event in dedups}
    require(set(by_dense) == set(inverse), "R2 dedup WAL coverage mismatch")
    for dense_point, original_offset in inverse.items():
        event = by_dense[dense_point]
        require(int(event["original_image"]) == original_offset + 1 and event.get("coordinate_sha256") == ENGINE.sha256_array(dense[dense_point]) and event.get("new_physical_calls") == 0, "R2 dedup WAL identity mismatch")
    publications = [event for event in events if event.get("event") == "complete_shard_published" and int(event.get("row_index", -1)) == row["row_index"]]
    require(any(event.get("shard") == str(path) and event.get("shard_sha256") == ENGINE.sha256_file(path) for event in publications), "R2 physical WAL lacks shard publication")
    proof = [event for event in source_events if event.get("event") == "sealed_online_reference_bound" and int(event.get("row_index", -1)) == row["row_index"]]
    require(any(event.get("path_sha256") == row["path_sha256"] and same_float(event.get("online_discrete_I_perp"), row["discrete_I_perp"]) for event in proof), "R2 sealed online proof missing")
    endpoints = [event for event in source_events if event.get("event") == "endpoint_joint_EF_cache_reused" and int(event.get("row_index", -1)) == row["row_index"]]
    require(any(event.get("inherited_values") == 2 and event.get("new_physical_calls") == 0 for event in endpoints), "R2 endpoint source closure missing")
    source_publications = [event for event in source_events if event.get("event") == "complete_shard_published" and int(event.get("row_index", -1)) == row["row_index"]]
    require(any(event.get("shard") == str(path) and event.get("shard_sha256") == ENGINE.sha256_file(path) for event in source_publications), "R2 source WAL lacks shard publication")


def preflight(state: dict, output: Path, *, probe_locks: bool = True, probe_remote_host: bool = True) -> dict:
    output = Path(output).expanduser().resolve()
    require(output == state["result_output"] or state.get("allow_test_output") is True, "R2 output differs from contract")
    for final in _final_paths(output):
        if final.exists():
            raise FileExistsError(f"refusing to overwrite {final}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path, physical_path = _wal_paths(output)
    source_events = _read_wal(source_path, SOURCE_WAL_SCHEMA, state)
    physical_events = _read_wal(physical_path, PHYSICAL_WAL_SCHEMA, state)
    lifetime = _lifetime(physical_events)
    require(lifetime["attempt_started"] <= MAXIMUM_FRESH_ATTEMPT_STARTED, "R2 attempt cap exceeded")
    require(lifetime["attempt_failed"] == lifetime["attempt_open"] == lifetime["retry_attempts"] == 0, "R2 open/failed/retried attempt requires a superseding contract")
    complete = []
    explained = 0
    for row in state["rows"]:
        target = _shard_path(output, row)
        starts = [event for event in physical_events if event.get("event") == "attempt_started" and int(event.get("row_index", -1)) == row["row_index"]]
        if target.exists():
            shard = _validate_shard(target, row, state)
            _validate_wal_closure(physical_events, source_events, row, shard, target)
            complete.append(row["optimization_N_F"])
            explained += row["R2_fresh_joint_EF_calls"]
        else:
            require(not starts, f"{row['label']} has incomplete R2 attempts; a superseding contract is required")
    require(lifetime["attempt_started"] == explained, "R2 WAL attempts are not explained by complete shards")
    missing_rows = [row for row in state["rows"] if row["optimization_N_F"] not in complete]
    planned = sum(row["R2_fresh_joint_EF_calls"] for row in missing_rows)
    required_free = MINIMUM_FREE_BYTES_RESERVE + len(missing_rows) * ESTIMATED_SHARD_BYTES
    observed_free = shutil.disk_usage(output.parent).free
    require(observed_free >= required_free, f"insufficient R2 free disk: {observed_free} < {required_free}")
    observed_host = R1.check_remote_host(state["contract"]["remote_identity"]) if probe_remote_host else None
    if probe_locks:
        with ENGINE.acquire_two_locks(state["locks"]):
            pass
    return {
        "status": "PASS_R2_ZERO_OPENMM_REMOTE_PREFLIGHT",
        "remote_host_probe": observed_host if probe_remote_host else "SKIPPED_FOR_TEST",
        "selected_path_count": N_PATHS,
        "terminal_partial_tail_selected": False,
        "complete_shards_reusable": complete,
        "missing_paths": len(missing_rows),
        "fresh_original_internal_joint_EF_planned": len(missing_rows) * N_INTERNAL,
        "fresh_dense_interior_joint_EF_planned_after_exact_dedup": planned - len(missing_rows) * N_INTERNAL,
        "fresh_joint_EF_attempts_planned": planned,
        "maximum_fresh_attempt_started_records": MAXIMUM_FRESH_ATTEMPT_STARTED,
        "stored_original_internal_full_force_vectors_planned": len(missing_rows) * N_INTERNAL,
        "stored_dense_full_force_vectors_planned": len(missing_rows) * N_DENSE,
        "inherited_endpoint_joint_EF_values_planned": len(missing_rows) * 2,
        "lifetime": lifetime,
        "observed_free_bytes": observed_free,
        "required_free_bytes": required_free,
        "lock_probe": "PASS_NONBLOCKING" if probe_locks else "SKIPPED_FOR_TEST",
        "optimization_N_F_contribution": 0,
        "OpenMM_backend_constructed": False,
    }


def _evaluate_one(state: dict, row: dict, output: Path, backend, source_wal, physical_wal) -> tuple[dict, Path]:
    target = _shard_path(output, row)
    source_wal.append({
        "event": "sealed_online_reference_bound", "utc": utc_now(),
        "row_index": row["row_index"], "label": row["label"],
        "optimization_N_F": row["optimization_N_F"], "path_sha256": row["path_sha256"],
        "online_discrete_I_perp": row["discrete_I_perp"],
        "all_paths_result_sha256": state["contract"]["selection_and_replay_evidence"]["all_paths_result"]["sha256"],
        "exact_verification_sha256": state["contract"]["selection_and_replay_evidence"]["exact_all_paths_verification"]["sha256"],
        "new_physical_calls": 0,
    })
    source_wal.append({
        "event": "endpoint_joint_EF_cache_reused", "utc": utc_now(),
        "row_index": row["row_index"], "label": row["label"],
        "optimization_N_F": row["optimization_N_F"], "path_sha256": row["path_sha256"],
        "endpoint_cache_sha256": state["contract"]["bindings"]["endpoint_joint_EF_cache"]["sha256"],
        "inherited_values": 2, "new_physical_calls": 0,
    })
    original_c = np.ascontiguousarray(row["path"][1:-1])
    original_e = np.empty(N_INTERNAL, dtype=np.float64)
    original_f = np.empty((N_INTERNAL, DIMENSION), dtype=np.float64)
    physical_wal.append({
        "event": "path_stage_started", "utc": utc_now(), "stage": "original_discrete_internal_joint_EF",
        "row_index": row["row_index"], "label": row["label"], "optimization_N_F": row["optimization_N_F"],
        "path_sha256": row["path_sha256"], "points": N_INTERNAL,
        "original_internal_coordinate_sha256": ENGINE.sha256_array(original_c),
    })
    for offset, coordinate in enumerate(original_c):
        image = offset + 1
        original_e[offset], original_f[offset] = _attempt(physical_wal, backend, state, row, "original_discrete_internal_joint_EF", image, coordinate)
    recomputed_i = float(ENGINE.path_diagnostics(row["path"], np.zeros(N_IMAGES), original_f)["i_perp"])
    online_i = float(row["discrete_I_perp"])
    relative = abs(recomputed_i - online_i) / max(abs(online_i), 1.0e-300)
    require(np.isclose(recomputed_i, online_i, rtol=IPERP_CONSISTENCY_RTOL, atol=IPERP_CONSISTENCY_ATOL), f"{row['label']} recomputed Iperp differs from sealed online reference")
    physical_wal.append({
        "event": "path_stage_data_complete_uncommitted", "utc": utc_now(), "stage": "original_discrete_internal_joint_EF",
        "row_index": row["row_index"], "label": row["label"], "optimization_N_F": row["optimization_N_F"],
        "path_sha256": row["path_sha256"], "joint_attempts": N_INTERNAL, "joint_successes": N_INTERNAL,
        "original_internal_coordinate_sha256": ENGINE.sha256_array(original_c),
        "original_internal_energy_sha256": ENGINE.sha256_array(original_e),
        "original_internal_force_sha256": ENGINE.sha256_array(original_f),
        "recomputed_discrete_I_perp": recomputed_i,
        "online_reference_discrete_I_perp": online_i,
        "relative_difference": relative,
    })

    dense, geometry = ENGINE.uniform_arc500(row["path"])
    dense_e = np.empty(N_DENSE, dtype=np.float64)
    dense_f = np.empty((N_DENSE, DIMENSION), dtype=np.float64)
    dense_e[[0, -1]] = state["endpoint"]["energies"]
    dense_f[[0, -1]] = state["endpoint"]["forces"]
    original_points, dense_fresh, inverse = _expected_stage_points(row)
    require(original_points == list(range(1, N_INTERNAL + 1)), "R2 original point plan drifted")
    physical_wal.append({
        "event": "path_stage_started", "utc": utc_now(), "stage": "dense_arc500_interior_joint_EF",
        "row_index": row["row_index"], "label": row["label"], "optimization_N_F": row["optimization_N_F"],
        "path_sha256": row["path_sha256"], "stored_points": N_DENSE,
        "fresh_points": len(dense_fresh), "exact_reused_points": len(inverse),
        "inherited_endpoint_points": 2, "dense_coordinate_sha256": ENGINE.sha256_array(dense),
    })
    for dense_point in range(1, N_DENSE - 1):
        if dense_point in inverse:
            original_offset = inverse[dense_point]
            require(np.array_equal(dense[dense_point], original_c[original_offset]), "R2 planned dedup coordinate drifted")
            dense_e[dense_point] = original_e[original_offset]
            dense_f[dense_point] = original_f[original_offset]
            physical_wal.append({
                "event": "exact_coordinate_stage_dedup_reused", "utc": utc_now(),
                "stage": "dense_arc500_interior_joint_EF", "row_index": row["row_index"],
                "label": row["label"], "optimization_N_F": row["optimization_N_F"],
                "path_sha256": row["path_sha256"], "dense_point": dense_point,
                "original_image": original_offset + 1,
                "coordinate_sha256": ENGINE.sha256_array(dense[dense_point]),
                "identity_proof": "float64_C_order_bytes_equal_and_numpy_array_equal",
                "new_physical_calls": 0,
            })
        else:
            dense_e[dense_point], dense_f[dense_point] = _attempt(physical_wal, backend, state, row, "dense_arc500_interior_joint_EF", dense_point, dense[dense_point])
    delta = float((np.max(dense_e) - dense_e[0]) * R1.EV_TO_KCAL)
    physical_wal.append({
        "event": "path_stage_data_complete_uncommitted", "utc": utc_now(), "stage": "dense_arc500_interior_joint_EF",
        "row_index": row["row_index"], "label": row["label"], "optimization_N_F": row["optimization_N_F"],
        "path_sha256": row["path_sha256"], "joint_attempts": len(dense_fresh), "joint_successes": len(dense_fresh),
        "exact_reused_original_calls": len(inverse), "stored_joint_EF_values": N_DENSE,
        "dense_coordinate_sha256": ENGINE.sha256_array(dense), "dense_energy_sha256": ENGINE.sha256_array(dense_e),
        "dense_force_sha256": ENGINE.sha256_array(dense_f), "densified_Delta_E_kcal_per_mol": delta,
    })
    evidence = state["contract"]["selection_and_replay_evidence"]
    arrays = {
        "schema": np.asarray(SHARD_SCHEMA), "status": np.asarray("PASS"), "method": np.asarray("IVM"),
        "label": np.asarray(row["label"]), "selection_role": np.asarray(row["selection_role"]),
        "optimization_N_F": np.asarray(row["optimization_N_F"], dtype=np.int64),
        "path_identifier_json": np.asarray(json.dumps(row["path_identifier"], sort_keys=True)),
        "contract_sha256": np.asarray(state["contract_sha256"]), "evaluator_sha256": np.asarray(ENGINE.sha256_file(Path(__file__))),
        "bb_core_sha256": np.asarray(ENGINE.sha256_file(ROOT / "src/bb_core.py")), "topology_sha256": np.asarray(ENGINE.sha256_file(state["topology"])),
        "selection_sha256": np.asarray(evidence["selection"]["sha256"]), "all_paths_result_sha256": np.asarray(evidence["all_paths_result"]["sha256"]),
        "exact_verification_sha256": np.asarray(evidence["exact_all_paths_verification"]["sha256"]),
        "selected_path": row["path"], "selected_path_sha256": np.asarray(row["path_sha256"]),
        "online_discrete_I_perp_reference": np.asarray(online_i),
        "original_internal_coordinates_A": original_c, "original_internal_energy_eV": original_e,
        "original_internal_force_eV_per_A": original_f,
        "original_internal_coordinate_sha256": np.asarray(ENGINE.sha256_array(original_c)),
        "original_internal_energy_sha256": np.asarray(ENGINE.sha256_array(original_e)),
        "original_internal_force_sha256": np.asarray(ENGINE.sha256_array(original_f)),
        "discrete_I_perp": np.asarray(recomputed_i),
        "discrete_I_perp_source": np.asarray("stored_original_internal_force_eV_per_A_recomputed_by_bb_core.path_diagnostics"),
        "online_recomputed_Iperp_relative_difference": np.asarray(relative),
        "dense_coordinates_A": dense, "dense_energy_eV": dense_e, "dense_force_eV_per_A": dense_f,
        "dense_coordinate_sha256": np.asarray(ENGINE.sha256_array(dense)), "dense_energy_sha256": np.asarray(ENGINE.sha256_array(dense_e)),
        "dense_force_sha256": np.asarray(ENGINE.sha256_array(dense_f)), "densified_Delta_E_kcal_per_mol": np.asarray(delta),
        "total_arclength_A": np.asarray(geometry["total_arclength_A"]), "dense_spacing_A": np.asarray(geometry["dense_spacing_A"]),
        "exact_original_to_dense_overlap_plan_json": np.asarray(json.dumps(row["overlap_plan"], sort_keys=True)),
        "fresh_joint_EF_attempts": np.asarray(row["R2_fresh_joint_EF_calls"], dtype=np.int64),
        "maximum_fresh_joint_EF_attempts": np.asarray(MAXIMUM_FRESH_PER_PATH, dtype=np.int64),
        "original_internal_joint_EF_attempts": np.asarray(N_INTERNAL, dtype=np.int64),
        "dense_interior_fresh_joint_EF_attempts": np.asarray(len(dense_fresh), dtype=np.int64),
        "dense_endpoint_joint_EF_values_inherited": np.asarray(2, dtype=np.int64),
        "stored_dense_joint_EF_values": np.asarray(N_DENSE, dtype=np.int64),
        "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
    }
    ENGINE._atomic_save_npz_no_overwrite(target, arrays)
    shard = _validate_shard(target, row, state)
    published = {"event": "complete_shard_published", "utc": utc_now(), "row_index": row["row_index"], "label": row["label"], "optimization_N_F": row["optimization_N_F"], "path_sha256": row["path_sha256"], "shard": str(target), "shard_sha256": ENGINE.sha256_file(target)}
    physical_wal.append(published)
    source_wal.append(published)
    return shard, target


def _publish_final(state: dict, output: Path, shards: list[dict], paths: list[Path], source_wal_path: Path, physical_wal_path: Path, held_locks: list[str], fresh: list[int], reused: list[int]) -> None:
    exact_total = state["exact_fresh_calls"]
    arrays = {
        "schema": np.asarray(RESULT_SCHEMA), "status": np.asarray("PASS"), "method": np.asarray("IVM"),
        "authoritative_data_unit": np.asarray("atomic_complete_path_shard"),
        "labels": np.asarray([str(scalar(shard, "label")) for shard in shards]),
        "selection_role": np.asarray([str(scalar(shard, "selection_role")) for shard in shards]),
        "optimization_N_F": np.asarray([int(scalar(shard, "optimization_N_F")) for shard in shards], dtype=np.int64),
        "selected_path_sha256": np.asarray([str(scalar(shard, "selected_path_sha256")) for shard in shards]),
        "discrete_I_perp": np.asarray([float(scalar(shard, "discrete_I_perp")) for shard in shards]),
        "online_discrete_I_perp_reference": np.asarray([float(scalar(shard, "online_discrete_I_perp_reference")) for shard in shards]),
        "original_internal_energy_sha256": np.asarray([str(scalar(shard, "original_internal_energy_sha256")) for shard in shards]),
        "original_internal_force_sha256": np.asarray([str(scalar(shard, "original_internal_force_sha256")) for shard in shards]),
        "energy_eV": np.asarray([shard["dense_energy_eV"] for shard in shards]),
        "energy_sha256": np.asarray([str(scalar(shard, "dense_energy_sha256")) for shard in shards]),
        "dense_coordinate_sha256": np.asarray([str(scalar(shard, "dense_coordinate_sha256")) for shard in shards]),
        "dense_force_sha256": np.asarray([str(scalar(shard, "dense_force_sha256")) for shard in shards]),
        "densified_Delta_E_kcal_per_mol": np.asarray([float(scalar(shard, "densified_Delta_E_kcal_per_mol")) for shard in shards]),
        "shard_path": np.asarray([str(path) for path in paths]), "shard_sha256": np.asarray([ENGINE.sha256_file(path) for path in paths]),
        "original_internal_joint_EF_values_stored": np.asarray(TOTAL_ORIGINAL_INTERNAL_STORED_JOINT_EF, dtype=np.int64),
        "dense_joint_EF_values_stored": np.asarray(TOTAL_DENSE_STORED_JOINT_EF, dtype=np.int64),
        "fresh_joint_EF_attempts": np.asarray(exact_total, dtype=np.int64),
        "maximum_fresh_attempt_started_records": np.asarray(MAXIMUM_FRESH_ATTEMPT_STARTED, dtype=np.int64),
        "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
        "contract": np.asarray(str(state["contract_path"])), "contract_sha256": np.asarray(state["contract_sha256"]),
    }
    ENGINE._atomic_save_npz_no_overwrite(output, arrays)
    physical_events = _read_wal(physical_wal_path, PHYSICAL_WAL_SCHEMA, state)
    source_events = _read_wal(source_wal_path, SOURCE_WAL_SCHEMA, state)
    lifetime = _lifetime(physical_events)
    require(lifetime == {"attempt_started": exact_total, "attempt_succeeded": exact_total, "attempt_failed": 0, "attempt_open": 0, "retry_attempts": 0}, "R2 final WAL accounting mismatch")
    for row, shard, path in zip(state["rows"], shards, paths):
        _validate_wal_closure(physical_events, source_events, row, shard, path)
    manifest = {
        "schema": MANIFEST_SCHEMA, "status": "PASS", "created_utc": utc_now(),
        "contract": bind(state["contract_path"]), "aggregate": bind(output),
        "shards": [
            {
                "optimization_N_F": row["optimization_N_F"], "calls_per_image": row["path_identifier"]["calls_per_image"],
                "path": str(path), "sha256": ENGINE.sha256_file(path), "selected_path_sha256": row["path_sha256"],
                "original_internal_energy_sha256": str(scalar(shard, "original_internal_energy_sha256")),
                "original_internal_force_sha256": str(scalar(shard, "original_internal_force_sha256")),
                "dense_coordinate_sha256": str(scalar(shard, "dense_coordinate_sha256")),
                "dense_energy_sha256": str(scalar(shard, "dense_energy_sha256")), "dense_force_sha256": str(scalar(shard, "dense_force_sha256")),
                "fresh_joint_EF_attempts": row["R2_fresh_joint_EF_calls"], "exact_overlap_count": row["overlap_plan"]["exact_overlap_count"],
            }
            for row, shard, path in zip(state["rows"], shards, paths)
        ],
    }
    ENGINE._atomic_write_json_no_overwrite(output.with_suffix(".manifest.json"), manifest)
    audit = {
        "schema": AUDIT_SCHEMA, "status": "PASS", "completed_utc": utc_now(),
        "contract": bind(state["contract_path"]), "output": bind(output), "manifest": bind(output.with_suffix(".manifest.json")),
        "sealed_source_wal": bind(source_wal_path), "physical_joint_ef_wal": bind(physical_wal_path),
        "held_host_locks": held_locks, "remote_host": R1.check_remote_host(state["contract"]["remote_identity"]),
        "mutagen_freeze_gate": state["contract"]["mutagen_gate_at_freeze"],
        "fresh_paths_this_invocation": fresh, "complete_validated_shards_reused": reused,
        "selected_path_count": N_PATHS, "terminal_partial_tail_selected": False,
        "original_internal_joint_EF_values_stored": TOTAL_ORIGINAL_INTERNAL_STORED_JOINT_EF,
        "dense_joint_EF_values_stored": TOTAL_DENSE_STORED_JOINT_EF,
        "physical_attempt_lifetime": lifetime, "maximum_fresh_attempt_started_records": MAXIMUM_FRESH_ATTEMPT_STARTED,
        "optimization_N_F_contribution": 0,
    }
    ENGINE._atomic_write_json_no_overwrite(output.with_suffix(".audit.json"), audit)


def run(state: dict, output: Path) -> None:
    report = preflight(state, output, probe_locks=False, probe_remote_host=True)
    source_wal_path, physical_wal_path = _wal_paths(output)
    source_wal = ENGINE.DurableWal(source_wal_path, SOURCE_WAL_SCHEMA, state["contract_path"], state["contract_sha256"])
    physical_wal = ENGINE.DurableWal(physical_wal_path, PHYSICAL_WAL_SCHEMA, state["contract_path"], state["contract_sha256"])
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
                        backend = ENGINE.OpenMMBackend(state["topology"], platform_preference=("CPU",), precision="double", label="bb_irivm_late_six_original30_arc500_r2")
                    before = int(backend.raw_joint_attempts)
                    shard, target = _evaluate_one(state, row, output, backend, source_wal, physical_wal)
                    require(int(backend.raw_joint_attempts) - before == row["R2_fresh_joint_EF_calls"], "R2 backend attempt ledger mismatch")
                    fresh.append(row["optimization_N_F"])
                shards.append(shard)
                paths.append(target)
            _publish_final(state, output, shards, paths, source_wal_path, physical_wal_path, held_locks, fresh, reused)
    finally:
        source_wal.close()
        physical_wal.close()
    require(report["fresh_joint_EF_attempts_planned"] == sum(row["R2_fresh_joint_EF_calls"] for row in state["rows"] if row["optimization_N_F"] in fresh), "R2 preflight/run fresh count mismatch")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host-role", choices=("remote",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-lock-probe-for-test", action="store_true")
    parser.add_argument("--skip-remote-host-probe-for-test", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    R1.check_thread_caps()
    state = load_contract(args.contract, args.host_role)
    output = args.output.expanduser().resolve()
    require(output.suffix == ".npz", "R2 --output must end in .npz")
    if args.preflight_only:
        print(json.dumps(preflight(state, output, probe_locks=not args.skip_lock_probe_for_test, probe_remote_host=not args.skip_remote_host_probe_for_test), indent=2, sort_keys=True))
        return
    require(not args.skip_lock_probe_for_test and not args.skip_remote_host_probe_for_test, "R2 test-only gate bypass cannot execute")
    run(state, output)


if __name__ == "__main__":
    main()
