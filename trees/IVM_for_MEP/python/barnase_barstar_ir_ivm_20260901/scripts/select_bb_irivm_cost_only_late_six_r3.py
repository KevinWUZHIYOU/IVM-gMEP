#!/usr/bin/env python3
"""R5B-bound metric-blind late-six selector and dual-host work-plan freezer.

The preregistration and host split are frozen while R5B is still running.
Selection is impossible until both R5B verifiers are immutable PASS artifacts
and bind the same final manifest and stable-last canonical index.

Only call identity, optimization N_F, formal-reportability, terminal-tail,
reparameterization-commit, and path-identity fields may influence selection.
I_perp, Delta E, energy, force, barrier, and every other metric value are
deliberately outside the selection projection.

This module never imports OpenMM and never acquires a heavy-task lock.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

PREREG_SCHEMA = "bb-irivm-r5b-cost-only-late-six-preregistration-v3"
PREREG_STATUS = "FROZEN_DURING_R5B_BEFORE_FINAL_RESULT_AND_BEFORE_SELECTION"
TEMPLATE_SCHEMA = "bb-irivm-r5b-cost-only-late-six-dual-host-template-v3"
TEMPLATE_STATUS = "FROZEN_ZERO_PHYSICAL_REMOTE4_LOCAL2_PLAN"
SELECTION_SCHEMA = "bb-irivm-r5b-cost-only-late-six-selection-v3"
SELECTION_STATUS = "PASS_SELECTED_FROM_DUAL_R5B_VERIFICATION_WITH_COST_ONLY"

PRIMARY_SCHEMA = "bb-irivm-selected-full-state-history-verification-v5b"
PRIMARY_STATUS = "PASS_R5B_EXACT_PREFIX318_AND_DYNAMIC_STABLE_LAST_FULL_STATE_REPLAY"
SUPPLEMENTAL_SCHEMA = "bb-irivm-full-state-r5b-supplemental-verification-v1"
SUPPLEMENTAL_STATUS = "PASS_R5B_EXACT_COUNTER_SCHEMA_AND_ARTIFACT_CLOSURE"
MANIFEST_SCHEMA = "bb-irivm-full-state-history-manifest-v3"
INDEX_SCHEMA = "bb-irivm-full-state-history-stable-last-index-v1"

EARLY_FOUR_NF = (60, 150, 300, 1329)
ANCHORS_NF = (2000, 4000, 6000, 8000, 8982)
MAXIMUM_NF = 15000
N_PATHS = 6
N_IMAGES = 32
N_INTERNAL = 30
N_DENSE = 500
FRESH_DENSE_PER_PATH = 498
TOTAL_FRESH_DENSE = N_PATHS * FRESH_DENSE_PER_PATH
TOTAL_STORED_DENSE = N_PATHS * N_DENSE

HIGHEST_ROLE = "highest_legal_NF_not_above_15000"
ANCHOR_ROLES = tuple("nearest_unused_to_NF{}".format(value) for value in ANCHORS_NF)
ROLE_ORDER = (HIGHEST_ROLE,) + ANCHOR_ROLES
REMOTE_ROLES = (
    HIGHEST_ROLE,
    "nearest_unused_to_NF8982",
    "nearest_unused_to_NF8000",
    "nearest_unused_to_NF6000",
)
LOCAL_ROLES = (
    "nearest_unused_to_NF2000",
    "nearest_unused_to_NF4000",
)

ALLOWED_PRIMARY_FIELDS = {
    "history_index",
    "call_per_image",
    "optimization_N_F",
    "path_sha256",
    "shard",
    "shard_sha256",
    "formally_reportable",
    "terminal_uncommitted_charged_tail",
}
REPLAY_ELIGIBILITY_KEYS = (
    "calls_per_image",
    "total_internal_joint_evaluations",
    "reparam_committed",
    "total_internal_nf_last_complete_state_index",
)
ELIGIBILITY_ROW_FIELDS = {
    "history_index",
    "call_per_image",
    "optimization_N_F",
    "selected_path_sha256",
    "source_full_state_shard",
    "source_full_state_shard_sha256",
    "reparameterization_committed",
    "fully_evaluated_current_path",
    "stable_last",
    "formally_reportable",
    "terminal_uncommitted_charged_tail",
}
FORBIDDEN_SELECTION_FIELDS = {
    "I_perp",
    "i_perp",
    "Delta_E",
    "densified_Delta_E",
    "energy",
    "energies",
    "energy_eV",
    "force",
    "forces",
    "force_eV_per_A",
    "barrier",
    "action",
    "geometric_action",
}

R5B_REGISTRATION = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF15000_20260902_R5B.json"
).resolve()
R5B_AUTHORIZATION = (
    ROOT
    / "registrations/execution_authorizations/"
    "bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b.json"
).resolve()
R5B_MUTAGEN_GATE = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_MUTAGEN_GATE_NF15000_20260902_R5B.json"
).resolve()
PRIMARY_VERIFIER = (
    HERE / "verify_selected_ivm_full_state_history_replay_r5.py"
).resolve()
SUPPLEMENTAL_VERIFIER = (
    HERE / "verify_selected_ivm_full_state_history_replay_r5b_supplemental.py"
).resolve()
JOINT_ENGINE = (HERE / "evaluate_bb_dense_four_paths_joint_v2.py").resolve()
TEST_PATH = (
    ROOT / "tests/test_select_bb_irivm_cost_only_late_six_r3.py"
).resolve()
ENDPOINT_CACHE = (ROOT / "inputs/endpoint_joint_ef_cache.npz").resolve()
TOPOLOGY = (ROOT / "inputs/complex_fixed.pdb").resolve()

OBSERVER_ROOT = (
    ROOT
    / "runs/full_state_history_replay_r5b/"
    "bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b_observer"
).resolve()
CANONICAL_MANIFEST = (OBSERVER_ROOT / "full_state_history.manifest.json").resolve()
CANONICAL_INDEX = (
    OBSERVER_ROOT / "full_state_history.canonical_index.json"
).resolve()
CANONICAL_PRIMARY = (
    ROOT
    / "reports/verifications/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF15000_20260902_R5B_PRIMARY.json"
).resolve()
CANONICAL_SUPPLEMENTAL = (
    ROOT
    / "reports/verifications/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF15000_20260902_R5B_SUPPLEMENTAL.json"
).resolve()
CANONICAL_PREREGISTRATION = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_COST_ONLY_LATE_SIX_PREREG_NF15000_20260902_R3.json"
).resolve()
CANONICAL_SPLIT_TEMPLATE = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_COST_ONLY_LATE_SIX_DUAL_HOST_TEMPLATE_NF15000_20260902_R3.json"
).resolve()
CANONICAL_SELECTION = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_COST_ONLY_LATE_SIX_SELECTION_NF15000_20260902_R3.json"
).resolve()

OLD_SELECTION_PREREG = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_LATE_MILESTONE_SELECTION_PREREG_20260902_R1.json"
).resolve()
OLD_PACKAGE_PREREG = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_LATE_SIX_JOINT_EF_PACKAGE_PREREG_20260902_R1.json"
).resolve()

THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OPENMM_CPU_THREADS": "1",
}
LOCKS = [
    "/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock",
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock",
]
MUTAGEN = {
    "name": "bb-irivm-20260901",
    "identifier": "sync_EaEvvA9594NRGtMQSN0jneepyUq2tCef27jljwXXPc9",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "{} is not a JSON object".format(path))
    return value


def write_exclusive(path: Path, payload: dict) -> None:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("refusing to overwrite {}".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def option_value(argv: Sequence[str], flag: str) -> str:
    require(argv.count(flag) == 1, "runtime argv flag drift: {}".format(flag))
    index = list(argv).index(flag)
    require(index + 1 < len(argv), "runtime argv value missing: {}".format(flag))
    return str(argv[index + 1])


def validate_frozen_r5b_inputs() -> dict:
    registration = read_json(R5B_REGISTRATION)
    authorization = read_json(R5B_AUTHORIZATION)
    gate = read_json(R5B_MUTAGEN_GATE)
    require(
        registration.get("schema")
        == "bb-irivm-selected-full-state-history-output-only-replay-v5b"
        and registration.get("status")
        == "FROZEN_BEFORE_FIRST_R5B_FULL_STATE_HISTORY_REPLAY_PHYSICAL_CALL",
        "R5B registration is not the frozen NF15000 registration",
    )
    runtime = [str(value) for value in registration.get("runtime_argv", [])]
    require(
        option_value(runtime, "--budget") == "15000"
        and option_value(runtime, "--total-internal-nf-budget") == "15000",
        "R5B budget is not exactly 15000",
    )
    require(
        authorization.get("schema")
        == "bb-irivm-selected-full-state-history-output-only-replay-authorization-v5b"
        and authorization.get("enabled") is True
        and authorization.get("registration") == bind(R5B_REGISTRATION),
        "R5B authorization closure failed",
    )
    require(
        gate.get("schema") == "bb-irivm-full-state-r5b-mutagen-gate-v1"
        and gate.get("status") == "PASS_CONNECTED_FLUSHED"
        and gate.get("registration") == bind(R5B_REGISTRATION)
        and gate.get("authorization") == bind(R5B_AUTHORIZATION),
        "R5B Mutagen gate closure failed",
    )
    return {
        "registration": bind(R5B_REGISTRATION),
        "authorization": bind(R5B_AUTHORIZATION),
        "Mutagen_gate": bind(R5B_MUTAGEN_GATE),
    }


def validate_inherited_anchor_source() -> dict:
    old = read_json(OLD_SELECTION_PREREG)
    deterministic = old.get("deterministic_selection", {})
    require(
        old.get("schema")
        == "bb-irivm-late-milestone-blind-selection-preregistration-v1"
        and old.get("status")
        == "FROZEN_BEFORE_REMOTE_RESULT_AND_BEFORE_ANY_NEW_METRIC_INSPECTION"
        and old.get("early_four_already_covered_optimization_N_F")
        == list(EARLY_FOUR_NF)
        and deterministic.get("anchor_targets_optimization_N_F")
        == list(ANCHORS_NF)
        and deterministic.get("metric_values_consulted") is False,
        "old blind preregistration does not prove the inherited anchors",
    )
    return old


def future_paths() -> dict:
    return {
        "primary_verification": str(CANONICAL_PRIMARY),
        "supplemental_verification": str(CANONICAL_SUPPLEMENTAL),
        "capture_manifest": str(CANONICAL_MANIFEST),
        "canonical_index": str(CANONICAL_INDEX),
        "selection": str(CANONICAL_SELECTION),
    }


def build_preregistration() -> dict:
    r5b = validate_frozen_r5b_inputs()
    validate_inherited_anchor_source()
    return {
        "schema": PREREG_SCHEMA,
        "status": PREREG_STATUS,
        "frozen": True,
        "created_utc": utc_now(),
        "execution_authorized_by_this_file": False,
        "R5B_final_artifacts_read": False,
        "metric_values_read_by_selector_or_this_freezer": False,
        "prior_observation_disclosure": (
            "metric values may already have been observed outside this selector; "
            "they are not inputs and cannot affect the frozen projection or ranking"
        ),
        "purpose": (
            "after dual R5B PASS, choose six late cost milestones without "
            "reading I_perp, Delta E, energy, force, barrier, or action"
        ),
        "supersedes": {
            "old_selection_preregistration": bind(OLD_SELECTION_PREREG),
            "old_late_six_package_preregistration": bind(OLD_PACKAGE_PREREG),
            "reason": (
                "the old protocol was capped at NF9990 and assigned all six "
                "paths to the remote host; R3 retains its five anchors, "
                "mechanically extends only the highest cap with the authorized "
                "NF15000 budget, and freezes a balanced remote4/local2 split"
            ),
            "old_files_modified_or_deleted": False,
        },
        "early_four_optimization_N_F": list(EARLY_FOUR_NF),
        "selection_policy": {
            "late_optimization_N_F_strictly_greater_than": max(EARLY_FOUR_NF),
            "maximum_legal_optimization_N_F": MAXIMUM_NF,
            "number_of_distinct_paths": N_PATHS,
            "role_order": list(ROLE_ORDER),
            "reserve_highest_first": True,
            "highest_rule": "maximum (optimization_N_F, call_per_image)",
            "anchor_targets_N_F": list(ANCHORS_NF),
            "anchor_source": bind(OLD_SELECTION_PREREG),
            "anchors_changed_from_old_preregistration": False,
            "highest_cap_change_reason": "mechanical user-authorized budget extension from 9990 to 15000",
            "anchor_rule": (
                "minimum (absolute_NF_distance, optimization_N_F, "
                "call_per_image) among unused paths"
            ),
            "must_be_fully_evaluated": True,
            "must_be_reparameterization_committed": True,
            "must_be_formally_reportable": True,
            "must_be_stable_last": True,
            "terminal_uncommitted_charged_tail_allowed": False,
            "distinctness_key": "path_sha256",
            "allowed_primary_inventory_fields": sorted(ALLOWED_PRIMARY_FIELDS),
            "allowed_replay_array_keys": list(REPLAY_ELIGIBILITY_KEYS),
            "forbidden_selection_fields": sorted(FORBIDDEN_SELECTION_FIELDS),
            "metric_values_consulted": False,
        },
        "R5B_closure_required_before_selection": {
            "future_canonical_paths": future_paths(),
            "primary_schema": PRIMARY_SCHEMA,
            "primary_status": PRIMARY_STATUS,
            "supplemental_schema": SUPPLEMENTAL_SCHEMA,
            "supplemental_status": SUPPLEMENTAL_STATUS,
            "manifest_schema": MANIFEST_SCHEMA,
            "canonical_index_schema": INDEX_SCHEMA,
            "both_verifiers_must_bind_same_manifest_and_index": True,
            "primary_inventory_must_equal_stable_last_index": True,
        },
        "metric_evaluation_policy_after_selection": {
            "original_images": N_IMAGES,
            "original_internal_joint_EF_reused_from_R5B_per_path": N_INTERNAL,
            "new_original_path_calls": 0,
            "dense_sampling": "500 uniform unweighted-Cartesian-arc-length points",
            "dense_coordinates_energy_full_force_all_stored": True,
            "inherited_fixed_endpoint_joint_EF_per_path": 2,
            "expected_fresh_dense_interior_joint_EF_per_path": FRESH_DENSE_PER_PATH,
            "expected_fresh_dense_interior_joint_EF_total": TOTAL_FRESH_DENSE,
            "optimization_N_F_contribution": 0,
        },
        "frozen_dual_host_assignment": {
            "remote_roles": list(REMOTE_ROLES),
            "local_roles": list(LOCAL_ROLES),
            "remote_path_count": 4,
            "local_path_count": 2,
            "remote_relative_throughput": 2.0,
            "local_relative_throughput": 1.0,
            "normalized_work_remote": 4 * FRESH_DENSE_PER_PATH / 2.0,
            "normalized_work_local": 2 * FRESH_DENSE_PER_PATH,
            "assignment_uses_metric_values": False,
        },
        "bindings": {
            **r5b,
            "primary_verifier": bind(PRIMARY_VERIFIER),
            "supplemental_verifier": bind(SUPPLEMENTAL_VERIFIER),
            "joint_EF_engine": bind(JOINT_ENGINE),
            "selector_and_plan_freezer": bind(Path(__file__).resolve()),
            "zero_physical_tests": bind(TEST_PATH),
            "endpoint_joint_EF_cache": bind(ENDPOINT_CACHE),
            "topology": bind(TOPOLOGY),
        },
    }


def validate_preregistration(path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    require(
        resolved == CANONICAL_PREREGISTRATION,
        "noncanonical R3 preregistration path",
    )
    payload = read_json(resolved)
    validate_inherited_anchor_source()
    require(
        payload.get("schema") == PREREG_SCHEMA
        and payload.get("status") == PREREG_STATUS
        and payload.get("frozen") is True
        and payload.get("execution_authorized_by_this_file") is False
        and payload.get("R5B_final_artifacts_read") is False
        and payload.get("metric_values_read_by_selector_or_this_freezer") is False,
        "invalid R3 preregistration",
    )
    require(
        payload.get("early_four_optimization_N_F") == list(EARLY_FOUR_NF),
        "early-four identity drift",
    )
    policy = payload.get("selection_policy", {})
    require(
        policy.get("role_order") == list(ROLE_ORDER)
        and policy.get("anchor_targets_N_F") == list(ANCHORS_NF)
        and policy.get("anchor_source") == bind(OLD_SELECTION_PREREG)
        and policy.get("anchors_changed_from_old_preregistration") is False
        and policy.get("maximum_legal_optimization_N_F") == MAXIMUM_NF
        and policy.get("metric_values_consulted") is False,
        "selection policy drift",
    )
    require(
        payload.get("R5B_closure_required_before_selection", {}).get(
            "future_canonical_paths"
        )
        == future_paths(),
        "future R5B path closure drift",
    )
    expected_bindings = {
        **validate_frozen_r5b_inputs(),
        "primary_verifier": bind(PRIMARY_VERIFIER),
        "supplemental_verifier": bind(SUPPLEMENTAL_VERIFIER),
        "joint_EF_engine": bind(JOINT_ENGINE),
        "selector_and_plan_freezer": bind(Path(__file__).resolve()),
        "zero_physical_tests": bind(TEST_PATH),
        "endpoint_joint_EF_cache": bind(ENDPOINT_CACHE),
        "topology": bind(TOPOLOGY),
    }
    require(payload.get("bindings") == expected_bindings, "preregistration binding drift")
    return payload


def build_split_template(preregistration_path: Path) -> dict:
    preregistration = validate_preregistration(preregistration_path)
    host_common = {
        "jobs": 1,
        "thread_caps": THREAD_CAPS,
        "dual_host_local_fcntl_locks": LOCKS,
        "atomic_complete_path_shards": True,
        "append_only_fsync_WAL": True,
        "no_overwrite": True,
        "open_failed_or_incomplete_path_fails_closed": True,
        "later_execution_authorization_required": True,
    }
    return {
        "schema": TEMPLATE_SCHEMA,
        "status": TEMPLATE_STATUS,
        "frozen": True,
        "created_utc": utc_now(),
        "execution_authorized_by_this_file": False,
        "selection_or_R5B_final_artifacts_read": False,
        "wall_clock_reasoning": (
            "each path has 498 expected fresh calls; four paths at 2x "
            "remote throughput balance two paths at 1x local throughput"
        ),
        "workload": {
            "selected_paths": N_PATHS,
            "stored_arc500_joint_EF_values_total": TOTAL_STORED_DENSE,
            "expected_fresh_calls_per_path": FRESH_DENSE_PER_PATH,
            "expected_fresh_calls_remote": 4 * FRESH_DENSE_PER_PATH,
            "expected_fresh_calls_local": 2 * FRESH_DENSE_PER_PATH,
            "expected_fresh_calls_total": TOTAL_FRESH_DENSE,
            "optimization_N_F_contribution": 0,
        },
        "remote_four": {
            **host_common,
            "host_role": "remote",
            "identity": {
                "hostname": "Mac-mini.local",
                "address": "100.74.85.3",
                "platform": "Darwin",
                "machine": "arm64",
            },
            "selection_roles": list(REMOTE_ROLES),
            "Mutagen_session": MUTAGEN,
            "future_contract": str(
                ROOT
                / "registrations/diagnostics/"
                "BB_IRIVM_COST_LATE_SIX_REMOTE4_ARC500_CONTRACT_20260902_R3.json"
            ),
            "future_authorization": str(
                ROOT
                / "registrations/execution_authorizations/"
                "bb_irivm_cost_late_six_remote4_arc500_20260902_r3.json"
            ),
            "future_result": str(
                ROOT
                / "runs/diagnostics/"
                "bb_irivm_cost_late_six_remote4_arc500_joint_ef_20260902_r3.npz"
            ),
        },
        "local_two": {
            **host_common,
            "host_role": "local",
            "identity": {
                "hostname": "wuzhiyoudeMacBook-Air-206.local",
                "platform": "Darwin",
                "machine": "arm64",
            },
            "selection_roles": list(LOCAL_ROLES),
            "Mutagen_session": MUTAGEN,
            "future_contract": str(
                ROOT
                / "registrations/diagnostics/"
                "BB_IRIVM_COST_LATE_SIX_LOCAL2_ARC500_CONTRACT_20260902_R3.json"
            ),
            "future_authorization": str(
                ROOT
                / "registrations/execution_authorizations/"
                "bb_irivm_cost_late_six_local2_arc500_20260902_r3.json"
            ),
            "future_result": str(
                ROOT
                / "runs/diagnostics/"
                "bb_irivm_cost_late_six_local2_arc500_joint_ef_20260902_r3.npz"
            ),
        },
        "future_contract_requirements": {
            "bind_selection_and_both_R5B_verifiers": True,
            "bind_R5B_manifest_index_and_selected_full_state_shards": True,
            "remote_local_role_sets_disjoint_and_union_all_six": True,
            "reuse_original32_joint_EF": True,
            "reuse_two_fixed_endpoints": True,
            "arc500_coordinates_energy_full_force_materialized_per_path": True,
            "one_fresh_joint_energy_full_force_call_per_nonendpoint_dense_point": True,
            "cross_host_shard_outputs_must_not_overlap": True,
            "future_freezer_and_evaluator_must_be_hash_bound_before_authorization": True,
        },
        "future_pipeline_order": [
            "R5B primary verifier PASS",
            "R5B supplemental verifier PASS",
            "cost-only selector freezes six paths",
            "Mutagen flush and exact file-hash equality on both hosts",
            "freeze disjoint remote4 and local2 contracts",
            "issue separate host-specific authorizations after zero-OpenMM preflights",
            "launch remote4 and local2 concurrently, one BB job per host",
            "join six complete shards and validate aggregate with zero OpenMM",
        ],
        "bindings": {
            "preregistration": bind(Path(preregistration_path)),
            "selector_and_plan_freezer": preregistration["bindings"][
                "selector_and_plan_freezer"
            ],
            "joint_EF_engine": preregistration["bindings"]["joint_EF_engine"],
            "endpoint_joint_EF_cache": preregistration["bindings"][
                "endpoint_joint_EF_cache"
            ],
            "topology": preregistration["bindings"]["topology"],
        },
    }


def validate_split_template(path: Path, preregistration_path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    require(resolved == CANONICAL_SPLIT_TEMPLATE, "noncanonical split template path")
    preregistration = validate_preregistration(preregistration_path)
    payload = read_json(resolved)
    require(
        payload.get("schema") == TEMPLATE_SCHEMA
        and payload.get("status") == TEMPLATE_STATUS
        and payload.get("frozen") is True
        and payload.get("execution_authorized_by_this_file") is False
        and payload.get("selection_or_R5B_final_artifacts_read") is False,
        "invalid R3 dual-host template",
    )
    remote = payload.get("remote_four", {})
    local = payload.get("local_two", {})
    require(
        remote.get("selection_roles") == list(REMOTE_ROLES)
        and local.get("selection_roles") == list(LOCAL_ROLES)
        and set(remote["selection_roles"]).isdisjoint(local["selection_roles"])
        and set(remote["selection_roles"]) | set(local["selection_roles"])
        == set(ROLE_ORDER),
        "remote4/local2 role partition drift",
    )
    require(
        payload.get("workload", {}).get("expected_fresh_calls_remote") == 1992
        and payload["workload"]["expected_fresh_calls_local"] == 996
        and payload["workload"]["expected_fresh_calls_total"] == 2988
        and payload["workload"]["optimization_N_F_contribution"] == 0,
        "dual-host workload drift",
    )
    require(
        payload.get("bindings", {}).get("preregistration")
        == bind(Path(preregistration_path))
        and payload["bindings"]["selector_and_plan_freezer"]
        == preregistration["bindings"]["selector_and_plan_freezer"],
        "split-template binding drift",
    )
    return payload


def _project_primary_row(raw: Mapping) -> dict:
    require(isinstance(raw, Mapping), "primary inventory row is not a mapping")
    missing = ALLOWED_PRIMARY_FIELDS - set(raw)
    require(not missing, "primary row lacks {}".format(sorted(missing)))
    return {name: raw[name] for name in ALLOWED_PRIMARY_FIELDS}


def _stable_inventory(
    primary: dict, manifest: dict, canonical_index: dict, replay_state: dict
) -> list:
    rows = primary.get("path_inventory")
    records = canonical_index.get("records")
    require(isinstance(rows, list) and isinstance(records, list), "inventories missing")
    require(
        manifest.get("canonical_records") == records
        and len(rows) == len(records)
        and len(rows) == len(replay_state["calls"]),
        "primary/manifest/index/result inventory length drift",
    )
    result = []
    for offset, (raw, record) in enumerate(zip(rows, records)):
        row = _project_primary_row(raw)
        call = offset + 1
        require(
            int(row["history_index"]) == offset
            and int(row["call_per_image"]) == call
            and int(record["history_index"]) == offset
            and int(record["call_per_image"]) == call
            and int(replay_state["calls"][offset]) == call,
            "stable-last call axis drift at {}".format(call),
        )
        require(
            int(row["optimization_N_F"]) == int(record["optimization_N_F"])
            == int(replay_state["nf"][offset]),
            "N_F drift at call {}".format(call),
        )
        require(
            str(row["path_sha256"]) == str(record["path_sha256"])
            and str(row["shard"]) == str(record["shard"])
            and str(row["shard_sha256"]) == str(record["shard_sha256"]),
            "stable-last shard identity drift at call {}".format(call),
        )
        result.append(
            {
                "history_index": offset,
                "call_per_image": call,
                "optimization_N_F": int(row["optimization_N_F"]),
                "selected_path_sha256": str(row["path_sha256"]),
                "source_full_state_shard": str(row["shard"]),
                "source_full_state_shard_sha256": str(row["shard_sha256"]),
                "reparameterization_committed": bool(
                    replay_state["reparam_committed"][offset]
                ),
                "fully_evaluated_current_path": True,
                "stable_last": True,
                "formally_reportable": bool(row["formally_reportable"]),
                "terminal_uncommitted_charged_tail": bool(
                    row["terminal_uncommitted_charged_tail"]
                ),
            }
        )
    return result


def eligible_rows(inventory: Sequence[Mapping]) -> list:
    eligible = []
    for raw in inventory:
        require(isinstance(raw, Mapping), "eligibility row is not a mapping")
        missing = ELIGIBILITY_ROW_FIELDS - set(raw)
        require(not missing, "eligibility row lacks {}".format(sorted(missing)))
        row = {name: raw[name] for name in ELIGIBILITY_ROW_FIELDS}
        nf = int(row["optimization_N_F"])
        if not (
            nf > max(EARLY_FOUR_NF)
            and nf <= MAXIMUM_NF
            and row.get("reparameterization_committed") is True
            and row.get("fully_evaluated_current_path") is True
            and row.get("stable_last") is True
            and row.get("formally_reportable") is True
            and row.get("terminal_uncommitted_charged_tail") is False
        ):
            continue
        path_hash = str(row["selected_path_sha256"])
        require(len(path_hash) == 64, "eligible path identity is not SHA-256")
        eligible.append(row)
    require(eligible, "no eligible late committed stable-last path")
    return sorted(
        eligible,
        key=lambda row: (
            int(row["optimization_N_F"]),
            int(row["call_per_image"]),
            str(row["selected_path_sha256"]),
        ),
    )


def select_rows(inventory: Sequence[Mapping]) -> list:
    candidates = eligible_rows(inventory)
    require(
        len({row["selected_path_sha256"] for row in candidates}) >= N_PATHS,
        "fewer than six distinct eligible coordinate paths",
    )
    highest = max(
        candidates,
        key=lambda row: (int(row["optimization_N_F"]), int(row["call_per_image"])),
    )
    selected = [{**highest, "selection_role": HIGHEST_ROLE}]
    used = {highest["selected_path_sha256"]}
    for target, role in zip(ANCHORS_NF, ANCHOR_ROLES):
        remaining = [
            row for row in candidates if row["selected_path_sha256"] not in used
        ]
        require(remaining, "no unused path remains for NF{}".format(target))
        chosen = min(
            remaining,
            key=lambda row: (
                abs(int(row["optimization_N_F"]) - target),
                int(row["optimization_N_F"]),
                int(row["call_per_image"]),
            ),
        )
        selected.append({**chosen, "selection_role": role})
        used.add(chosen["selected_path_sha256"])
    require(
        len(selected) == len(used) == N_PATHS,
        "selector did not produce six distinct paths",
    )
    return selected


def load_replay_eligibility(path: Path) -> dict:
    with np.load(Path(path).resolve(strict=True), allow_pickle=False) as saved:
        require(
            all(key in saved.files for key in REPLAY_ELIGIBILITY_KEYS),
            "R5B result lacks an allowed eligibility key",
        )
        calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        nf = np.asarray(
            saved["total_internal_joint_evaluations"], dtype=np.int64
        )
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        last_complete = int(
            np.asarray(
                saved["total_internal_nf_last_complete_state_index"]
            ).item()
        ) + 1
    require(
        calls.ndim == nf.ndim == committed.ndim == 1
        and calls.shape == nf.shape == committed.shape
        and np.array_equal(calls, np.arange(1, calls.size + 1))
        and 1 <= last_complete <= calls.size,
        "invalid R5B eligibility arrays",
    )
    return {
        "calls": calls,
        "nf": nf,
        "reparam_committed": committed,
        "last_complete_call": last_complete,
    }


def validate_final_closure(
    preregistration_path: Path,
    split_template_path: Path,
    primary_path: Path,
    supplemental_path: Path,
    manifest_path: Path,
    index_path: Path,
    *,
    enforce_canonical: bool = True,
) -> tuple:
    preregistration = validate_preregistration(preregistration_path)
    split_template = validate_split_template(
        split_template_path, preregistration_path
    )
    resolved = {
        "primary": Path(primary_path).expanduser().resolve(strict=True),
        "supplemental": Path(supplemental_path).expanduser().resolve(strict=True),
        "manifest": Path(manifest_path).expanduser().resolve(strict=True),
        "index": Path(index_path).expanduser().resolve(strict=True),
    }
    if enforce_canonical:
        require(
            resolved
            == {
                "primary": CANONICAL_PRIMARY,
                "supplemental": CANONICAL_SUPPLEMENTAL,
                "manifest": CANONICAL_MANIFEST,
                "index": CANONICAL_INDEX,
            },
            "noncanonical R5B final closure path",
        )
    primary = read_json(resolved["primary"])
    supplemental = read_json(resolved["supplemental"])
    manifest = read_json(resolved["manifest"])
    canonical_index = read_json(resolved["index"])
    require(
        primary.get("schema") == PRIMARY_SCHEMA
        and primary.get("status") == PRIMARY_STATUS,
        "primary R5B verification is not PASS",
    )
    require(
        supplemental.get("schema") == SUPPLEMENTAL_SCHEMA
        and supplemental.get("status") == SUPPLEMENTAL_STATUS,
        "supplemental R5B verification is not PASS",
    )
    require(
        manifest.get("schema") == MANIFEST_SCHEMA
        and manifest.get("status") == "COMPLETE_PENDING_EXACT_PREFIX_VERIFICATION"
        and canonical_index.get("schema") == INDEX_SCHEMA
        and canonical_index.get("status")
        == "COMPLETE_PENDING_EXACT_RESULT_VERIFICATION",
        "manifest/index schema drift",
    )
    expected = {
        "registration": preregistration["bindings"]["registration"],
        "authorization": preregistration["bindings"]["authorization"],
        "capture_manifest": bind(resolved["manifest"]),
        "canonical_index": bind(resolved["index"]),
    }
    for role, record in expected.items():
        require(
            primary.get("bindings", {}).get(role) == record,
            "primary binding drift: {}".format(role),
        )
    require(
        supplemental.get("bindings", {}).get("primary_verification")
        == bind(resolved["primary"])
        and supplemental.get("bindings", {}).get("manifest")
        == bind(resolved["manifest"])
        and supplemental.get("bindings", {}).get("canonical_index")
        == bind(resolved["index"])
        and supplemental.get("bindings", {}).get("registration")
        == expected["registration"]
        and supplemental.get("bindings", {}).get("authorization")
        == expected["authorization"],
        "supplemental closure differs from primary closure",
    )
    primary_capture_audit = primary.get("bindings", {}).get("capture_audit")
    supplemental_capture_audit = supplemental.get("bindings", {}).get(
        "capture_audit"
    )
    supplemental_wal = supplemental.get("bindings", {}).get("wal")
    for role, record in (
        ("primary capture audit", primary_capture_audit),
        ("supplemental capture audit", supplemental_capture_audit),
        ("supplemental WAL", supplemental_wal),
    ):
        require(
            isinstance(record, dict)
            and bind(Path(record["path"])) == record,
            "{} binding drift".format(role),
        )
    require(
        primary_capture_audit == supplemental_capture_audit,
        "primary/supplemental capture-audit binding mismatch",
    )
    require(
        manifest.get("canonical_index") == str(resolved["index"])
        and manifest.get("canonical_index_sha256") == sha256_file(resolved["index"])
        and canonical_index.get("records") == manifest.get("canonical_records"),
        "manifest/index cross-binding drift",
    )
    replay_binding = primary.get("bindings", {}).get("replay_result")
    require(
        isinstance(replay_binding, dict)
        and bind(Path(replay_binding["path"])) == replay_binding,
        "primary replay-result binding drift",
    )
    replay_state = load_replay_eligibility(Path(replay_binding["path"]))
    require(
        replay_state["last_complete_call"] == int(primary["last_complete_call"])
        and manifest.get("formally_reportable_call_range")
        == [1, replay_state["last_complete_call"]],
        "last-complete/reportable range drift",
    )
    inventory = _stable_inventory(
        primary, manifest, canonical_index, replay_state
    )
    count = len(inventory)
    require(
        int(primary.get("actual_canonical_call_count", -1)) == count
        and int(supplemental.get("canonical_record_count", -1)) == count
        and int(manifest.get("canonical_record_count", -1)) == count
        and int(canonical_index.get("canonical_record_count", -1)) == count,
        "dual-verifier canonical count drift",
    )
    return (
        preregistration,
        split_template,
        primary,
        supplemental,
        manifest,
        canonical_index,
        replay_binding,
        inventory,
        resolved,
    )


def build_selection(
    preregistration_path: Path,
    split_template_path: Path,
    primary_path: Path,
    supplemental_path: Path,
    manifest_path: Path,
    index_path: Path,
    *,
    enforce_canonical: bool = True,
) -> dict:
    (
        preregistration,
        split_template,
        primary,
        _supplemental,
        _manifest,
        _index,
        replay_binding,
        inventory,
        resolved,
    ) = validate_final_closure(
        preregistration_path,
        split_template_path,
        primary_path,
        supplemental_path,
        manifest_path,
        index_path,
        enforce_canonical=enforce_canonical,
    )
    selected = select_rows(inventory)
    by_role = {row["selection_role"]: row for row in selected}
    require(set(by_role) == set(ROLE_ORDER), "selected role set drift")
    host_assignments = {
        "remote": [by_role[role] for role in REMOTE_ROLES],
        "local": [by_role[role] for role in LOCAL_ROLES],
    }
    return {
        "schema": SELECTION_SCHEMA,
        "status": SELECTION_STATUS,
        "created_utc": utc_now(),
        "selection_metric_values_consulted": False,
        "prior_metric_values_may_have_been_observed_outside_selector": True,
        "selection_array_keys_read": list(REPLAY_ELIGIBILITY_KEYS),
        "selection_primary_fields_read": sorted(ALLOWED_PRIMARY_FIELDS),
        "forbidden_selection_fields_read": [],
        "early_four_optimization_N_F": list(EARLY_FOUR_NF),
        "selected_path_count": N_PATHS,
        "selected_paths_in_preregistered_role_order": selected,
        "terminal_partial_tail_selected": False,
        "all_selected_reparameterization_committed": True,
        "all_selected_fully_evaluated_stable_last": True,
        "host_assignments": host_assignments,
        "expected_fresh_dense_joint_EF_calls": {
            "per_path": FRESH_DENSE_PER_PATH,
            "remote_four": 1992,
            "local_two": 996,
            "total": TOTAL_FRESH_DENSE,
        },
        "optimization_N_F_contribution": 0,
        "bindings": {
            "preregistration": bind(Path(preregistration_path)),
            "dual_host_template": bind(Path(split_template_path)),
            "primary_R5B_verification": bind(resolved["primary"]),
            "supplemental_R5B_verification": bind(resolved["supplemental"]),
            "R5B_capture_manifest": bind(resolved["manifest"]),
            "R5B_canonical_index": bind(resolved["index"]),
            "R5B_capture_audit": primary["bindings"]["capture_audit"],
            "R5B_capture_WAL": _supplemental["bindings"]["wal"],
            "R5B_replay_result": replay_binding,
            "selector": preregistration["bindings"]["selector_and_plan_freezer"],
            "endpoint_joint_EF_cache": split_template["bindings"][
                "endpoint_joint_EF_cache"
            ],
        },
    }


def preflight(preregistration_path: Path, split_template_path: Path) -> dict:
    preregistration = validate_preregistration(preregistration_path)
    split = validate_split_template(split_template_path, preregistration_path)
    return {
        "status": "PASS_ZERO_PHYSICAL_R5B_COST_ONLY_LATE_SIX_PREFLIGHT",
        "preregistration": bind(Path(preregistration_path)),
        "dual_host_template": bind(Path(split_template_path)),
        "R5B_final_artifacts_read": False,
        "selection_frozen": False,
        "metric_values_consulted": False,
        "early_four_optimization_N_F": list(EARLY_FOUR_NF),
        "late_roles": list(ROLE_ORDER),
        "remote_paths": len(split["remote_four"]["selection_roles"]),
        "local_paths": len(split["local_two"]["selection_roles"]),
        "expected_fresh_calls_total": TOTAL_FRESH_DENSE,
        "optimization_N_F_contribution": 0,
        "OpenMM_imported_or_called": False,
        "validated_bindings": sorted(preregistration["bindings"]),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    prereg = modes.add_parser("preregister")
    prereg.add_argument("--output", type=Path, required=True)
    template = modes.add_parser("freeze-split-template")
    template.add_argument("--preregistration", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    check = modes.add_parser("preflight")
    check.add_argument("--preregistration", type=Path, required=True)
    check.add_argument("--split-template", type=Path, required=True)
    select = modes.add_parser("select")
    select.add_argument("--preregistration", type=Path, required=True)
    select.add_argument("--split-template", type=Path, required=True)
    select.add_argument("--primary-verification", type=Path, required=True)
    select.add_argument("--supplemental-verification", type=Path, required=True)
    select.add_argument("--manifest", type=Path, required=True)
    select.add_argument("--canonical-index", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.mode == "preregister":
        require(
            args.output.expanduser().resolve() == CANONICAL_PREREGISTRATION,
            "noncanonical preregistration output",
        )
        write_exclusive(args.output, build_preregistration())
        output = args.output.resolve(strict=True)
        print(output)
        print(sha256_file(output))
    elif args.mode == "freeze-split-template":
        require(
            args.output.expanduser().resolve() == CANONICAL_SPLIT_TEMPLATE,
            "noncanonical split-template output",
        )
        write_exclusive(
            args.output, build_split_template(args.preregistration)
        )
        output = args.output.resolve(strict=True)
        print(output)
        print(sha256_file(output))
    elif args.mode == "preflight":
        print(
            json.dumps(
                preflight(args.preregistration, args.split_template),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        require(
            args.output.expanduser().resolve() == CANONICAL_SELECTION,
            "noncanonical selection output",
        )
        payload = build_selection(
            args.preregistration,
            args.split_template,
            args.primary_verification,
            args.supplemental_verification,
            args.manifest,
            args.canonical_index,
        )
        write_exclusive(args.output, payload)
        output = args.output.resolve(strict=True)
        print(output)
        print(sha256_file(output))


if __name__ == "__main__":
    main()
