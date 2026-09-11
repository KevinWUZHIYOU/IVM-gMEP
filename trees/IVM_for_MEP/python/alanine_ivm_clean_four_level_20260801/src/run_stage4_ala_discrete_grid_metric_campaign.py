#!/usr/bin/env python3
"""Audited metric campaign for the 248-cell Ala discrete grid.

The optimizer campaign and this diagnostic campaign have deliberately separate
accounting domains.  Optimizer ``Number of Forces`` values are read verbatim
from frozen result files.  Dense-barrier energy queries are content-addressed,
deduplicated by exact path bytes, and reported only as diagnostic joint
evaluations; they are never added to the 5000-NF optimizer budget.

This module has no top-level OpenMM import.  Preregistration is fail-closed on
validated completion of both upstream families (240 inexact/reparameterized
cells and 8 exact/no-reparameterization cells).  A preregistration therefore
must not be made until the successor exact aggregate is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ElementTree

import numpy as np

import run_stage4_ala_discrete_grid_campaign as infrastructure
import run_stage4_ala_discrete_grid_campaign_successor_v2 as upstream


PROTOCOL_ID = "stage4_ala_discrete_grid_metric_campaign_v2"
CANONICAL_ROOT_NAME = infrastructure.CANONICAL_ROOT_NAME
PRIMARY_FAMILY = upstream.PRIMARY_FAMILY
SECONDARY_FAMILY = upstream.SECONDARY_FAMILY
FAMILIES = upstream.FAMILIES
HOSTS = upstream.HOSTS
EXPECTED_CELL_COUNTS = {PRIMARY_FAMILY: 240, SECONDARY_FAMILY: 8}
INCUMBENT_FAMILY = "incumbent_one_way_alpha15_eta1p1"
INCUMBENT_CELL_ID = "incumbent_one_way_alpha15_eta1p1"

DEFAULT_LEVELS = (20, 40, 80, 160)
DEFAULT_CONVERGENCE_TOLERANCE_KCAL = 0.001
EV_TO_KCAL = 23.0605
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
TOTAL_INTERNAL_NF_BUDGET = 5000
N_IMAGES = 22
PATH_DOF = 66
SENTINEL_ABSOLUTE_TOLERANCE_KCAL = 1e-10

REFERENCE_BARRIER_KCAL = 2.4560825131253083
I_PERP_TARGET = 2.1e-4
BARRIER_ERROR_TARGET_KCAL = 0.017

DEFAULT_GLOBAL_LOCK = upstream.DEFAULT_GLOBAL_LOCK
THREAD_ENVIRONMENT = dict(infrastructure.THREAD_ENVIRONMENT)

SOURCE_PATHS = {
    "metric_campaign": "src/run_stage4_ala_discrete_grid_metric_campaign.py",
    "highalpha_analyzer": "src/analyze_stage4_ala_discrete_grid_metrics.py",
    "portable_decision_snapshot": (
        "inputs/stage4_ala_discrete_grid_metric_decision_snapshot_v2.json"
    ),
    "superseded_v1_retirement": (
        "inputs/stage4_ala_discrete_grid_metric_v1_retirement.json"
    ),
    **{
        f"successor_{name}": relative
        for name, relative in upstream.SOURCE_PATHS.items()
    },
}
TOPOLOGY_SOURCE_KEY = "successor_topology"
ALA_CORE_SOURCE_KEY = "successor_ala_core"
COMMON_INITIAL_SOURCE_KEY = "successor_common_initial_path"

HIGHALPHA_ANALYZER_RELATIVE = SOURCE_PATHS["highalpha_analyzer"]
SUPERSEDED_V1_RETIREMENT_RELATIVE = SOURCE_PATHS[
    "superseded_v1_retirement"
]
SUPERSEDED_V1_PREREGISTRATIONS = (
    {
        "relative_path": (
            "reports/stage4_ala_discrete_grid_metric_campaign_v1_"
            "1787607586217444000_preregistered.json"
        ),
        "sha256": (
            "53e8c18ab9b407044cffc407bcc68fb36d78b06c8872d05bd722cd0aa70bf4c4"
        ),
    },
    {
        "relative_path": (
            "reports/stage4_ala_discrete_grid_metric_campaign_v1_"
            "1787607772331640000_preregistered.json"
        ),
        "sha256": (
            "c2785fefb54448f9aeffcc050e9e481e66c9b18448768a4b8626687607791ab3"
        ),
    },
)

INCUMBENT_INPUTS = {
    "campaign_preregistration": {
        "relative": (
            "reports/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_"
            "v1_N254_r1_1787262084320785000_preregistered.json"
        ),
        "sha256": "67dd26b93f216a2d7f429a1b31710bdf1b933c24e10f17779e2499621705fa4e",
    },
    "campaign_report": {
        "relative": (
            "reports/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_"
            "v1_N254_r1_1787262084320785000.json"
        ),
        "sha256": "c64c471d612a22161ff1308bd969f83588b5de4a2aa67e62b4e4b848e96514ad",
    },
    "result": {
        "relative": (
            "runs/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1/"
            "members/ivm_stage4_ala_discrete_single_ffrbk_takeover_matched_"
            "kkt0_v1_one_way_takeover_eta1p1_r08c10_kkt0_a15_N254_"
            "discrete_b040_r1_central_state_ivm_cfg1804ab58ab75_budget254.npz"
        ),
        "sha256": "1c81ad6fe876a0eb94f2fc55d5d2567737face95ff52b6fe1bdfffc536d16787",
    },
    "member_audit": {
        "relative": (
            "runs/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1/"
            "audits/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_"
            "v1_one_way_takeover_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_"
            "r1.json"
        ),
        "sha256": "bd1caa6b744ede59e95b849eef8bf92a80ca321e849754c28d5f726e1147969f",
    },
    "runner_counter_audit": {
        "relative": (
            "runs/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1/"
            "members/ivm_stage4_ala_discrete_single_ffrbk_takeover_matched_"
            "kkt0_v1_one_way_takeover_eta1p1_r08c10_kkt0_a15_N254_"
            "discrete_b040_r1_central_state_ivm_cfg1804ab58ab75_budget254."
            "audit.json"
        ),
        "sha256": "d71bf5d14e7f2d070a3da4e70e85c9276bceb43920a9598de81b789ff2b63534",
    },
    "formal_iperp_ctp_audit": {
        "relative": (
            "reports/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_"
            "v1_N254_r1_1787262084320785000_iperp_ctp_readonly_v1_"
            "1787262442236424000.json"
        ),
        "sha256": "e9f5008381617fa98b04fabeb05cbc8b347e809dce9cdbd0901b7a0fc5a3e80d",
    },
    "postrun_mechanism_audit": {
        "relative": (
            "reports/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_"
            "v1_N254_r1_1787262084320785000_postrun_mechanism_provenance_"
            "readonly_v1.json"
        ),
        "sha256": "28b3c12692ba6ac21576dd193677916b77355370a41a009ba9e33b0a615ed538",
    },
}

DECISION_INPUT_PATHS = {
    "reference_barrier_formal_audit": (
        "figures/clean_four_level_audit/alanine_clean_four_level_audit.json"
    ),
    "portable_threshold_snapshot": (
        "inputs/stage4_ala_discrete_grid_metric_decision_snapshot_v2.json"
    ),
}

EXPECTED_PORTABLE_DECISION_SNAPSHOT = {
    "snapshot_id": "stage4_ala_discrete_grid_metric_decision_snapshot_v2",
    "scope": "portable_frozen_threshold_provenance_for_the_ala_metric_campaign",
    "source_inputs": {
        "active_experiment_figure_readme": {
            "repository_relative_path": (
                "paper/ActiveExperiments/alanine_66d/figures/"
                "active_candidates/README.md"
            ),
            "source_sha256": (
                "91b38392d4260b07b88b14439f670e0b0845be445d1e5a43df0c4d1a0ef8b974"
            ),
            "source_size_bytes": 8265,
            "evidence_line": 60,
            "evidence_text": (
                "`I_perp <= 2.1e-4`.  Panel b analogously compares first "
                "passage to the tightest"
            ),
            "evidence_line_sha256_without_newline": (
                "4c049d43b0d011eefe9a60590a955cf4586332d3187c6a3009053f622d71b8fb"
            ),
        },
        "stage4_v4_acceptance_summary": {
            "repository_relative_path": (
                "paper/ActiveExperiments/alanine_66d/data/audits/"
                "STAGE4_V4_ACCEPTANCE_SUMMARY.md"
            ),
            "source_sha256": (
                "1d6467198149b2f3a80e836e6f3b3c282dc636d7712f2a2aa403a601793f3841"
            ),
            "source_size_bytes": 2394,
            "evidence_line": 39,
            "evidence_text": (
                "The barrier-threshold cost advantage at `0.017 kcal/mol` is "
                "`4.0x`; the"
            ),
            "evidence_line_sha256_without_newline": (
                "1673c855f7ec541bb365d513ac23705a1d94be3c9bb4cf4a90b9e98bc344ab11"
            ),
        },
    },
    "frozen_thresholds": {
        "i_perp_target": I_PERP_TARGET,
        "barrier_error_target_kcal_per_mol": BARRIER_ERROR_TARGET_KCAL,
    },
    "portability_policy": (
        "runtime_validation_uses_only_this_study_local_snapshot_and_the_"
        "study_local_formal_barrier_audit;_repository_relative_origins_and_"
        "hashes_are_traceability_metadata"
    ),
}

DENSE_POLICY_ID = "nested_cartesian_linear_fixed_full_levels_v1"
ENERGY_PROTOCOL = {
    "backend": "OpenMMBackend",
    "platform_preference": ["CPU"],
    "precision": "double",
    "forcefield_files": ["amber14-all.xml"],
    "energy_unit_returned_by_backend": "eV",
    "barrier_unit": "kcal/mol",
    "barrier_definition": "(max(dense_energy)-first_dense_energy)*23.0605",
    "interpolation": "piecewise-linear Cartesian coordinates",
    "cross_path_coordinate_cache": False,
    "per_path_nested_grid_cache": True,
}

DECISION_RULES = {
    "reference_barrier_kcal_per_mol": REFERENCE_BARRIER_KCAL,
    "i_perp_target": I_PERP_TARGET,
    "barrier_error_target_kcal_per_mol": BARRIER_ERROR_TARGET_KCAL,
    "metric_state_inputs": (
        "formal_complete_and_mesh_eligible_states_only_raw_i_perp"
    ),
    "barrier_metric": (
        "absolute_difference_between_selected_dense_barrier_and_reference"
    ),
    "cost_to_precision": "minimum_observed_optimization_nf_no_interpolation",
    "unreachable_cost": "positive_infinity",
    "dense_nonconvergence": (
        "retain_level160_diagnostic_but_any_nonconverged_path_in_the_formal_"
        "comparison_domain_makes_formal_sets_and_winner_not_determined"
    ),
    "reported_sets": [
        "grid_only_nondominated",
        "incumbent_inclusive_nondominated",
    ],
    "unique_winner_scope": "incumbent_inclusive_only",
    "unique_winner_gate_policy_id": (
        "unique_grid_cell_that_full_curve_dominates_every_other_grid_cell_and_"
        "incumbent_on_raw_i_perp_and_dense_barrier_error_reaches_same_state_"
        "joint_target_and_all_comparison_domain_paths_are_dense_converged_v1"
    ),
    "pairwise_curve_strictness": (
        "both_metric_curves_weakly_nonworse_and_at_least_one_strict_point_"
        "across_the_two_curves_per_comparator"
    ),
    "unique_winner_failure_status": "no_unique_winner",
    "nonconvergence_status": "not_determined_due_to_nonconverged_paths",
    "provisional_output": (
        "provisional_nondominated_among_fully_converged_cells_is_engineering_"
        "only_and_not_a_formal_claim"
    ),
}

ELIGIBILITY_POLICY = {
    "common_mesh_gate": {
        "segment_length_cv_max_inclusive": MESH_CV_MAX,
        "segment_length_ratio_max_inclusive": MESH_RATIO_MAX,
    },
    PRIMARY_FAMILY: {
        "formal_complete": "trajectory_index_zero_or_reparam_committed",
        "terminal_partial_state_excluded": True,
    },
    SECONDARY_FAMILY: {
        "formal_complete": "exact_no_reparam_complete_state_mask",
        "terminal_charged_noncomplete_state_excluded": True,
    },
    "i_perp": "raw_result_field_no_recompute_no_cummin_no_interpolation",
}

ACCOUNTING_POLICY = {
    "optimization_nf_source": "result.total_internal_joint_evaluations",
    "optimization_nf_budget_per_grid_cell": TOTAL_INTERNAL_NF_BUDGET,
    "incumbent_frozen_final_optimization_nf": 3389,
    "diagnostic_nf_definition": "OpenMM backend raw joint energy evaluations",
    "diagnostic_nf_is_separate": True,
    "diagnostic_nf_added_to_optimization_nf": False,
    "cost_to_precision_interpolation": False,
}

CONCURRENCY_POLICY = {
    "scope": "per_physical_host",
    "jobs_fixed_per_host": 1,
    "ala_max_concurrent": 1,
    "bb_and_ala_may_overlap": True,
    "host_local_lock": DEFAULT_GLOBAL_LOCK,
    "lock_held_for_complete_shard": True,
    "dynamic_claims_used": False,
}


class MetricCampaignSafetyError(RuntimeError):
    """Fail-closed metric protocol, artifact, or provenance error."""


class PartialMetricTaskError(MetricCampaignSafetyError):
    """A formal task artifact exists but cannot be fully validated."""


def _root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root.name != CANONICAL_ROOT_NAME:
        raise MetricCampaignSafetyError(
            f"metric root must end in {CANONICAL_ROOT_NAME}: {root}"
        )
    return root


def file_sha256(path: Path) -> str:
    return infrastructure.file_sha256(path)


def canonical_json_bytes(payload: Any) -> bytes:
    return infrastructure.canonical_json_bytes(payload)


def payload_sha256(payload: Any) -> str:
    return infrastructure.payload_sha256(payload)


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MetricCampaignSafetyError(f"cannot read JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise MetricCampaignSafetyError(f"expected JSON object: {path}")
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def normalize_levels(levels: Sequence[int]) -> Tuple[int, ...]:
    observed = tuple(int(value) for value in levels)
    if len(observed) < 2 or observed[0] < 1:
        raise MetricCampaignSafetyError("dense policy needs at least two positive levels")
    if any(next_value != 2 * value for value, next_value in zip(observed, observed[1:])):
        raise MetricCampaignSafetyError("dense levels must be strictly doubling")
    if len(set(observed)) != len(observed):
        raise MetricCampaignSafetyError("dense levels contain duplicates")
    return observed


def dense_policy(
    levels: Sequence[int] = DEFAULT_LEVELS,
    tolerance_kcal: float = DEFAULT_CONVERGENCE_TOLERANCE_KCAL,
    energy_model_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    normalized = normalize_levels(levels)
    tolerance = float(tolerance_kcal)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise MetricCampaignSafetyError("dense convergence tolerance must be positive")
    policy = {
        "policy_id": DENSE_POLICY_ID,
        "subdivision_levels": list(normalized),
        "execution_rule": "evaluate_every_preregistered_level_no_early_stop",
        "selection_rule": "always_select_maximum_preregistered_level",
        "convergence_rule": (
            "absolute_barrier_difference_at_two_finest_levels_leq_tolerance"
        ),
        "convergence_tolerance_kcal_per_mol": tolerance,
        "evaluated_level_count": len(normalized),
        "maximum_subdivisions": int(normalized[-1]),
        "energy_protocol": ENERGY_PROTOCOL,
    }
    if energy_model_sha256 is not None:
        value = str(energy_model_sha256)
        if len(value) != 64:
            raise MetricCampaignSafetyError("energy-model SHA256 is malformed")
        policy["energy_model_sha256"] = value
    return policy


def _canonical_path(path: np.ndarray, *, formal_shape: bool = True) -> np.ndarray:
    result = np.asarray(path, dtype=np.float64)
    if result.ndim != 2:
        raise MetricCampaignSafetyError("path must be a two-dimensional array")
    if formal_shape and result.shape != (N_IMAGES, PATH_DOF):
        raise MetricCampaignSafetyError(
            f"formal Ala path shape must be {(N_IMAGES, PATH_DOF)}, got {result.shape}"
        )
    if result.shape[0] < 2 or result.shape[1] < 1 or not np.all(np.isfinite(result)):
        raise MetricCampaignSafetyError("path coordinates are invalid")
    return np.ascontiguousarray(result)


def path_content_sha256(path: np.ndarray, *, formal_shape: bool = True) -> str:
    return array_sha256(_canonical_path(path, formal_shape=formal_shape))


def path_task_spec(
    path_sha256: str,
    path_shape: Sequence[int],
    policy: Mapping[str, Any],
    topology_sha256: str,
) -> Dict[str, Any]:
    if len(path_sha256) != 64 or len(topology_sha256) != 64:
        raise MetricCampaignSafetyError("path/topology SHA256 is malformed")
    return {
        "protocol_id": PROTOCOL_ID,
        "path_sha256": path_sha256,
        "path_shape": [int(value) for value in path_shape],
        "dense_policy": dict(policy),
        "topology_sha256": topology_sha256,
    }


def path_task_id(
    path_sha256: str,
    path_shape: Sequence[int],
    policy: Mapping[str, Any],
    topology_sha256: str,
) -> str:
    return payload_sha256(
        path_task_spec(path_sha256, path_shape, policy, topology_sha256)
    )


def expected_nested_coordinate_accounting(
    path: np.ndarray, levels: Sequence[int]
) -> Dict[str, int]:
    """Return preregisterable logical-query and coordinate-byte miss counts."""

    path = _canonical_path(path)
    normalized = normalize_levels(levels)
    rational_seen: set[Tuple[int, int, int]] = set()
    coordinate_seen: set[bytes] = set()
    segments = path.shape[0] - 1
    for subdivisions in normalized:
        for segment in range(segments):
            for numerator in range(subdivisions):
                key = _rational_key(segment, numerator, subdivisions)
                if key in rational_seen:
                    continue
                rational_seen.add(key)
                fraction = numerator / subdivisions
                point = np.ascontiguousarray(
                    (1.0 - fraction) * path[segment]
                    + fraction * path[segment + 1],
                    dtype=np.float64,
                )
                coordinate_seen.add(point.tobytes())
    final_key = (segments, 0, 1)
    if final_key not in rational_seen:
        rational_seen.add(final_key)
        coordinate_seen.add(np.ascontiguousarray(path[-1]).tobytes())
    return {
        "expected_logical_energy_query_requests": len(rational_seen),
        "expected_diagnostic_raw_joint_evaluations": len(coordinate_seen),
        "expected_diagnostic_cache_hits": len(rational_seen) - len(coordinate_seen),
    }


def assigned_host(task_id: str) -> str:
    if len(task_id) != 64:
        raise MetricCampaignSafetyError("task ID is not a SHA256 digest")
    try:
        parity = int(task_id, 16) % len(HOSTS)
    except ValueError as error:
        raise MetricCampaignSafetyError("task ID is not hexadecimal") from error
    return HOSTS[parity]


def _require_bool_vector(saved: Any, key: str, size: int) -> np.ndarray:
    raw = np.asarray(saved[key])
    if raw.dtype.kind != "b" or raw.shape != (size,):
        raise MetricCampaignSafetyError(f"{key} must be a boolean trajectory vector")
    return raw.astype(bool, copy=False)


def _state_id(
    result_sha256: str,
    trajectory_index: int,
    optimization_nf: int,
    path_sha256: str,
) -> str:
    return payload_sha256(
        {
            "source_result_sha256": result_sha256,
            "trajectory_index": int(trajectory_index),
            "optimization_nf": int(optimization_nf),
            "path_sha256": path_sha256,
        }
    )


def extract_cell_states(
    result_path: Path,
    family: str,
    cell_id: str,
    expected_result_sha256: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Extract only formal-complete, mesh-eligible states from one frozen result."""

    if family not in FAMILIES:
        raise MetricCampaignSafetyError(f"unknown grid family: {family}")
    result_path = result_path.expanduser().resolve()
    if not result_path.is_file():
        raise MetricCampaignSafetyError(f"missing source result: {result_path}")
    result_sha = file_sha256(result_path)
    if expected_result_sha256 is not None and result_sha != expected_result_sha256:
        raise MetricCampaignSafetyError(f"source result hash drifted: {result_path}")

    required = {
        "calls_per_image",
        "stored_path_calls",
        "stored_paths",
        "total_internal_joint_evaluations",
        "i_perp",
        "segment_length_cv",
        "segment_length_ratio",
    }
    required.add(
        "reparam_committed"
        if family == PRIMARY_FAMILY
        else "exact_no_reparam_complete_state_mask"
    )
    if family == PRIMARY_FAMILY:
        required.add("discrete_audit_stop_reason")
    with np.load(result_path, allow_pickle=False) as saved:
        missing = sorted(required.difference(saved.files))
        if missing:
            raise MetricCampaignSafetyError(
                f"metric source result is missing {', '.join(missing)}: {result_path}"
            )
        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=np.int64)
        calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
        paths = np.asarray(saved["stored_paths"], dtype=np.float64)
        i_perp = np.asarray(saved["i_perp"], dtype=np.float64)
        cv = np.asarray(saved["segment_length_cv"], dtype=np.float64)
        ratio = np.asarray(saved["segment_length_ratio"], dtype=np.float64)
        size = int(cost.size)
        vectors = (cost, calls, stored_calls, i_perp, cv, ratio)
        if size < 1 or any(vector.shape != (size,) for vector in vectors):
            raise MetricCampaignSafetyError("metric source trajectory is misaligned")
        if paths.shape != (size, N_IMAGES, PATH_DOF):
            raise MetricCampaignSafetyError("metric source stored paths are misaligned")
        if not np.array_equal(calls, stored_calls):
            raise MetricCampaignSafetyError("stored path labels differ from call labels")
        if cost[0] < 0 or np.any(cost > TOTAL_INTERNAL_NF_BUDGET):
            raise MetricCampaignSafetyError("optimization NF is outside the frozen budget")
        if family == PRIMARY_FAMILY:
            if np.any(np.diff(cost) <= 0):
                raise MetricCampaignSafetyError("inexact optimization NF is not strict")
            complete = _require_bool_vector(saved, "reparam_committed", size).copy()
            complete[0] = True
            complete_definition = "initial_or_reparam_committed"
            stop_reasons = np.asarray(
                saved["discrete_audit_stop_reason"], dtype="U64"
            )
            if stop_reasons.ndim != 1:
                raise MetricCampaignSafetyError(
                    "primary discrete stop-reason audit is not one-dimensional"
                )
            inner_cap_token = "evaluated-inner-max-steps"
            inner_cap_stop_count = int(
                np.count_nonzero(stop_reasons == inner_cap_token)
            )
            inner_cap_evidence: Optional[Dict[str, Any]] = {
                "source_field": "result.discrete_audit_stop_reason",
                "inner_cap_stop_token": inner_cap_token,
                "inner_cap_stop_count": inner_cap_stop_count,
                "inner_cap_binding": inner_cap_stop_count > 0,
                "stop_record_count": int(stop_reasons.size),
            }
        else:
            if np.any(np.diff(cost) < 0):
                raise MetricCampaignSafetyError("exact optimization NF decreased")
            complete = _require_bool_vector(
                saved, "exact_no_reparam_complete_state_mask", size
            )
            complete_definition = "exact_no_reparam_complete_state_mask"
            inner_cap_evidence = None
        if not np.any(complete):
            raise MetricCampaignSafetyError("source result has no formal complete state")
        if not (
            np.all(np.isfinite(i_perp))
            and np.all(i_perp >= 0.0)
            and np.all(np.isfinite(cv))
            and np.all(cv >= 0.0)
            and np.all(np.isfinite(ratio))
            and np.all(ratio >= 1.0)
        ):
            raise MetricCampaignSafetyError("raw I_perp or mesh diagnostics are invalid")

        mesh = (cv <= MESH_CV_MAX) & (ratio <= MESH_RATIO_MAX)
        eligible_indices = np.flatnonzero(complete & mesh)
        if not eligible_indices.size:
            raise MetricCampaignSafetyError("cell has no complete mesh-eligible state")
        states: List[Dict[str, Any]] = []
        unique_paths: Dict[str, np.ndarray] = {}
        for index_value in eligible_indices:
            index = int(index_value)
            path = _canonical_path(paths[index])
            path_hash = array_sha256(path)
            previous = unique_paths.get(path_hash)
            if previous is not None and not np.array_equal(previous, path):
                raise MetricCampaignSafetyError("path SHA256 collision inside source cell")
            unique_paths[path_hash] = path.copy()
            nf = int(cost[index])
            state = {
                "state_id": _state_id(result_sha, index, nf, path_hash),
                "trajectory_index": index,
                "optimization_nf": nf,
                "calls_per_image": int(calls[index]),
                "raw_i_perp": float(i_perp[index]),
                "segment_length_cv": float(cv[index]),
                "segment_length_ratio": float(ratio[index]),
                "formal_complete": True,
                "mesh_eligible": True,
                "path_sha256": path_hash,
            }
            states.append(state)

    cell = {
        "family": family,
        "cell_id": str(cell_id),
        "source_result": str(result_path),
        "source_result_sha256": result_sha,
        "source_trajectory_record_count": size,
        "formal_complete_state_count": int(np.count_nonzero(complete)),
        "complete_mesh_eligible_state_count": len(states),
        "complete_state_definition": complete_definition,
        "raw_i_perp_field": "i_perp",
        "inner_cap_stop_evidence": inner_cap_evidence,
        "states": states,
    }
    return cell, unique_paths


def _task_artifact_path(root: Path, campaign_stem: str, task_id: str) -> Path:
    return (
        root
        / "runs"
        / "diagnostics"
        / campaign_stem
        / "tasks"
        / task_id[:2]
        / f"{task_id}.json"
    ).resolve()


def _task_attempts_path(root: Path, campaign_stem: str, task_id: str) -> Path:
    return (
        root
        / "runs"
        / "diagnostics"
        / campaign_stem
        / "attempts"
        / task_id[:2]
        / task_id
    ).resolve()


def build_unique_path_tasks(
    root: Path,
    campaign_stem: str,
    cells: Sequence[Dict[str, Any]],
    path_arrays: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
    topology_sha256: str,
) -> List[Dict[str, Any]]:
    """Attach task IDs to states and return exact-path-deduplicated tasks."""

    references: Dict[str, List[Dict[str, Any]]] = {}
    task_path_hashes: Dict[str, str] = {}
    observed_state_ids: set[str] = set()
    for cell in cells:
        for state in cell["states"]:
            state_id = str(state["state_id"])
            if state_id in observed_state_ids:
                raise MetricCampaignSafetyError("duplicate state identity")
            observed_state_ids.add(state_id)
            path_hash = str(state["path_sha256"])
            path = path_arrays.get(path_hash)
            if path is None or path_content_sha256(path) != path_hash:
                raise MetricCampaignSafetyError("eligible path bytes/hash are unavailable")
            task_id = path_task_id(path_hash, path.shape, policy, topology_sha256)
            previous_task_hash = task_path_hashes.setdefault(task_id, path_hash)
            if previous_task_hash != path_hash:
                raise MetricCampaignSafetyError("task ID collision across path hashes")
            state["path_task_id"] = task_id
            reference = {
                "state_id": state_id,
                "family": cell["family"],
                "cell_id": cell["cell_id"],
                "source_result": cell["source_result"],
                "source_result_sha256": cell["source_result_sha256"],
                "trajectory_index": state["trajectory_index"],
                "optimization_nf": state["optimization_nf"],
                "raw_i_perp": state["raw_i_perp"],
                "path_sha256": path_hash,
            }
            references.setdefault(task_id, []).append(reference)

    tasks: List[Dict[str, Any]] = []
    for task_id in sorted(references):
        refs = sorted(
            references[task_id],
            key=lambda item: (
                item["family"],
                item["cell_id"],
                item["trajectory_index"],
            ),
        )
        representative = refs[0]
        path_hashes = {str(ref["path_sha256"]) for ref in refs}
        if len(path_hashes) != 1:
            raise MetricCampaignSafetyError("one task unexpectedly references two paths")
        path_hash = next(iter(path_hashes))
        path = _canonical_path(path_arrays[path_hash])
        expected_task = path_task_id(path_hash, path.shape, policy, topology_sha256)
        if expected_task != task_id:
            raise MetricCampaignSafetyError("content-addressed task ID drifted")
        task = {
            "task_id": task_id,
            "task_role": "eligible_unique_path",
            "path_sha256": path_hash,
            "path_shape": list(path.shape),
            "assigned_host": assigned_host(task_id),
            "source_locator": {
                "result": representative["source_result"],
                "result_sha256": representative["source_result_sha256"],
                "trajectory_index": representative["trajectory_index"],
                "state_id": representative["state_id"],
            },
            "reference_count": len(refs),
            "references": refs,
            "artifact": str(_task_artifact_path(root, campaign_stem, task_id)),
            "attempts_directory": str(
                _task_attempts_path(root, campaign_stem, task_id)
            ),
            **expected_nested_coordinate_accounting(
                path, policy["subdivision_levels"]
            ),
        }
        tasks.append(task)
    if not tasks:
        raise MetricCampaignSafetyError("metric manifest contains no path tasks")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise MetricCampaignSafetyError("metric task IDs are not unique")
    return tasks


def build_cross_host_sentinel_tasks(
    root: Path,
    campaign_stem: str,
    policy: Mapping[str, Any],
    topology_sha256: str,
) -> Dict[str, Dict[str, Any]]:
    source = (root / SOURCE_PATHS[COMMON_INITIAL_SOURCE_KEY]).resolve()
    if not source.is_file():
        raise MetricCampaignSafetyError(f"common-initial sentinel source is missing: {source}")
    with np.load(source, allow_pickle=False) as saved:
        if "initial_path" not in saved.files:
            raise MetricCampaignSafetyError("common-initial sentinel lacks initial_path")
        path = _canonical_path(np.asarray(saved["initial_path"], dtype=np.float64))
    path_hash = array_sha256(path)
    task_id = path_task_id(path_hash, path.shape, policy, topology_sha256)
    tasks: Dict[str, Dict[str, Any]] = {}
    for host in HOSTS:
        artifact = (
            root
            / "runs"
            / "diagnostics"
            / campaign_stem
            / "sentinels"
            / host
            / f"{task_id}.json"
        ).resolve()
        attempts = (
            root
            / "runs"
            / "diagnostics"
            / campaign_stem
            / "sentinel_attempts"
            / host
            / task_id
        ).resolve()
        tasks[host] = {
            "task_id": task_id,
            "task_role": "cross_host_common_initial_sentinel",
            "path_sha256": path_hash,
            "path_shape": list(path.shape),
            "assigned_host": host,
            "source_locator": {
                "result": str(source),
                "result_sha256": file_sha256(source),
                "array_key": "initial_path",
                "trajectory_index": None,
                "state_id": "common_initial_sentinel",
            },
            "reference_count": 0,
            "references": [],
            "artifact": str(artifact),
            "attempts_directory": str(attempts),
            **expected_nested_coordinate_accounting(
                path, policy["subdivision_levels"]
            ),
        }
    return tasks


def forcefield_runtime_manifest(
    root_forcefield_files: Sequence[str] = ("amber14-all.xml",),
) -> Dict[str, Any]:
    """Hash the OpenMM XML bundle, following every ``Include`` recursively."""

    try:
        distribution = importlib.metadata.distribution("OpenMM")
    except importlib.metadata.PackageNotFoundError as error:
        raise MetricCampaignSafetyError("OpenMM distribution is unavailable") from error
    data_root = Path(
        distribution.locate_file("openmm/app/data")
    ).resolve()
    if not data_root.is_dir():
        raise MetricCampaignSafetyError(f"OpenMM forcefield data root is missing: {data_root}")
    pending = [str(item) for item in root_forcefield_files]
    records: Dict[str, Dict[str, Any]] = {}
    while pending:
        relative = pending.pop(0)
        if relative in records:
            continue
        candidate = (data_root / relative).resolve()
        try:
            candidate.relative_to(data_root)
        except ValueError as error:
            raise MetricCampaignSafetyError(
                f"forcefield include escapes OpenMM data root: {relative}"
            ) from error
        if not candidate.is_file():
            raise MetricCampaignSafetyError(
                f"OpenMM forcefield include is missing: {candidate}"
            )
        try:
            tree = ElementTree.parse(candidate)
        except ElementTree.ParseError as error:
            raise MetricCampaignSafetyError(
                f"OpenMM forcefield XML cannot be parsed: {candidate}"
            ) from error
        includes = sorted(
            {
                str(element.attrib["file"])
                for element in tree.getroot().iter()
                if element.tag.rsplit("}", 1)[-1] == "Include"
                and "file" in element.attrib
            }
        )
        records[relative] = {
            "path": str(candidate),
            "sha256": file_sha256(candidate),
            "size_bytes": int(candidate.stat().st_size),
            "includes": includes,
        }
        pending.extend(item for item in includes if item not in records)
    payload: Dict[str, Any] = {
        "distribution": "OpenMM",
        "distribution_version": str(distribution.version),
        "data_root": str(data_root),
        "root_forcefield_files": [str(item) for item in root_forcefield_files],
        "files": {name: records[name] for name in sorted(records)},
    }
    payload["bundle_sha256"] = payload_sha256(payload)
    return payload


def _repository_source_manifest(root: Path) -> Dict[str, Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    for name, relative in SOURCE_PATHS.items():
        path = (root / relative).resolve()
        if not path.is_file():
            raise MetricCampaignSafetyError(f"missing metric source/input: {path}")
        records[name] = {"path": str(path), "sha256": file_sha256(path)}
    return records


def _source_manifest(root: Path) -> Dict[str, Any]:
    return {
        "repository_files": _repository_source_manifest(root),
        "openmm_forcefield_bundle": forcefield_runtime_manifest(),
    }


def _route_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _superseded_v1_provenance(root: Path) -> Dict[str, Any]:
    """Revalidate the immutable record that both nonportable v1s never ran."""

    retirement_path = (root / SUPERSEDED_V1_RETIREMENT_RELATIVE).resolve()
    retirement = _read_json(retirement_path)
    exact_header = {
        "retirement_id": (
            "stage4_ala_discrete_grid_metric_v1_superseded_unlaunched"
        ),
        "superseded_protocol_id": (
            "stage4_ala_discrete_grid_metric_campaign_v1"
        ),
        "successor_protocol_id": PROTOCOL_ID,
        "reason": (
            "v1_referenced_paper_decision_inputs_outside_the_portable_ala_"
            "study_sync_tree_and_was_never_launched"
        ),
        "retirement_policy": (
            "v2_loader_rehashes_both_v1_preregistrations_and_requires_every_"
            "registered_v1_artifact_attempt_report_aggregate_and_analysis_"
            "route_to_remain_absent"
        ),
    }
    for key, expected in exact_header.items():
        if retirement.get(key) != expected:
            raise MetricCampaignSafetyError(
                f"superseded-v1 retirement drifted at {key}"
            )
    entries = retirement.get("preregistrations")
    if not isinstance(entries, list) or len(entries) != len(
        SUPERSEDED_V1_PREREGISTRATIONS
    ):
        raise MetricCampaignSafetyError("superseded-v1 retirement coverage drifted")
    validated_entries: List[Dict[str, Any]] = []
    for entry, frozen in zip(entries, SUPERSEDED_V1_PREREGISTRATIONS):
        relative = str(frozen["relative_path"])
        expected_sha = str(frozen["sha256"])
        if (
            not isinstance(entry, dict)
            or entry.get("relative_path") != relative
            or entry.get("sha256") != expected_sha
        ):
            raise MetricCampaignSafetyError(
                "superseded-v1 preregistration identity drifted"
            )
        preregistration_path = (root / relative).resolve()
        if (
            not preregistration_path.is_file()
            or file_sha256(preregistration_path) != expected_sha
        ):
            raise MetricCampaignSafetyError(
                f"superseded-v1 preregistration hash drifted: {preregistration_path}"
            )
        old = _read_json(preregistration_path)
        campaign_stem = str(entry.get("campaign_stem"))
        if (
            old.get("status") != "preregistered"
            or old.get("protocol_id")
            != "stage4_ala_discrete_grid_metric_campaign_v1"
            or old.get("campaign_stem") != campaign_stem
            or Path(str(old.get("preregistration"))).resolve()
            != preregistration_path
            or old.get("source_cell_count") != entry.get("source_cell_count")
            or old.get("eligible_state_reference_count")
            != entry.get("eligible_state_reference_count")
            or old.get("unique_path_task_count")
            != entry.get("unique_path_task_count")
        ):
            raise MetricCampaignSafetyError(
                "superseded-v1 preregistration contents drifted"
            )
        tasks = old.get("tasks")
        sentinels = old.get("cross_host_sentinel_tasks")
        if (
            not isinstance(tasks, list)
            or len(tasks) != int(entry["unique_path_task_count"])
            or not isinstance(sentinels, dict)
            or len(sentinels)
            != int(entry["cross_host_sentinel_task_count"])
        ):
            raise MetricCampaignSafetyError(
                "superseded-v1 registered task coverage drifted"
            )
        planned_outputs = old.get("planned_outputs")
        if planned_outputs != _planned_outputs(root, campaign_stem):
            raise MetricCampaignSafetyError(
                "superseded-v1 planned-output routes drifted"
            )
        observed_zero_launch = {
            "metric_task_artifacts_present": sum(
                _route_exists(Path(str(task["artifact"])).resolve())
                for task in tasks
            ),
            "metric_task_attempt_directories_present": sum(
                _route_exists(Path(str(task["attempts_directory"])).resolve())
                for task in tasks
            ),
            "sentinel_artifacts_present": sum(
                _route_exists(Path(str(task["artifact"])).resolve())
                for task in sentinels.values()
            ),
            "sentinel_attempt_directories_present": sum(
                _route_exists(Path(str(task["attempts_directory"])).resolve())
                for task in sentinels.values()
            ),
            "host_reports_present": sum(
                _route_exists(Path(str(route)).resolve())
                for route in planned_outputs["host_reports"].values()
            ),
            "aggregate_present": _route_exists(
                Path(str(planned_outputs["aggregate"])).resolve()
            ),
            "analysis_present": _route_exists(
                Path(str(planned_outputs["analysis"])).resolve()
            ),
        }
        if observed_zero_launch != entry.get("frozen_zero_launch_observation"):
            raise MetricCampaignSafetyError(
                "superseded-v1 is no longer in its frozen zero-launch state"
            )
        validated_entries.append(
            {
                "path": str(preregistration_path),
                "sha256": expected_sha,
                "campaign_stem": campaign_stem,
                "zero_launch_observation": observed_zero_launch,
            }
        )
    return {
        "retirement_record": {
            "path": str(retirement_path),
            "sha256": file_sha256(retirement_path),
        },
        "reason": exact_header["reason"],
        "validated_superseded_preregistrations": validated_entries,
        "all_superseded_v1_outputs_absent": True,
    }


def _decision_manifest(root: Path) -> Dict[str, Any]:
    inputs: Dict[str, Dict[str, str]] = {}
    for name, relative in DECISION_INPUT_PATHS.items():
        path = (root / relative).resolve()
        if not path.is_file():
            raise MetricCampaignSafetyError(f"missing metric decision input: {path}")
        inputs[name] = {"path": str(path), "sha256": file_sha256(path)}
    formal_audit = _read_json(
        Path(inputs["reference_barrier_formal_audit"]["path"])
    )
    observed_reference = float(
        formal_audit.get("reference_barrier_kcal_per_mol", float("nan"))
    )
    if observed_reference != REFERENCE_BARRIER_KCAL:
        raise MetricCampaignSafetyError("reference barrier decision input drifted")
    portable_snapshot = _read_json(
        Path(inputs["portable_threshold_snapshot"]["path"])
    )
    if portable_snapshot != EXPECTED_PORTABLE_DECISION_SNAPSHOT:
        raise MetricCampaignSafetyError(
            "portable comparison-threshold provenance drifted"
        )
    repository_sources = _repository_source_manifest(root)
    payload: Dict[str, Any] = {
        "inputs": inputs,
        "portable_threshold_snapshot": portable_snapshot,
        "incumbent_comparator_inputs": _incumbent_input_manifest(root),
        "extracted_values": {
            "reference_barrier_kcal_per_mol": observed_reference,
            "i_perp_target": I_PERP_TARGET,
            "barrier_error_target_kcal_per_mol": BARRIER_ERROR_TARGET_KCAL,
        },
        "analysis_rules": DECISION_RULES,
        "analysis_rules_sha256": payload_sha256(DECISION_RULES),
        "analysis_implementation": repository_sources["highalpha_analyzer"],
        "metric_orchestration_implementation": repository_sources[
            "metric_campaign"
        ],
    }
    payload["decision_manifest_sha256"] = payload_sha256(payload)
    return payload


def _incumbent_input_manifest(root: Path) -> Dict[str, Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    for name, frozen in INCUMBENT_INPUTS.items():
        path = (root / str(frozen["relative"])).resolve()
        expected = str(frozen["sha256"])
        if not path.is_file() or file_sha256(path) != expected:
            raise MetricCampaignSafetyError(
                f"frozen incumbent {name} drifted: {path}"
            )
        records[name] = {"path": str(path), "sha256": expected}
    return records


def extract_incumbent_comparator(
    root: Path,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Dict[str, str]]]:
    manifest = _incumbent_input_manifest(root)
    cell, paths = extract_cell_states(
        Path(manifest["result"]["path"]),
        PRIMARY_FAMILY,
        INCUMBENT_CELL_ID,
        manifest["result"]["sha256"],
    )
    cell["family"] = INCUMBENT_FAMILY
    cell["comparison_role"] = "frozen_incumbent_non_grid_comparator"
    cell["controls"] = {
        "alpha": 15,
        "eta": 1.1,
        "residual_ratio": 0.8,
        "inner_max_steps": 10,
        "final_optimization_nf": 3389,
    }
    if int(cell["states"][-1]["optimization_nf"]) > 3389:
        raise MetricCampaignSafetyError("incumbent eligible state exceeds frozen final NF")
    return cell, paths, manifest


def _upstream_input_manifest(
    upstream_preregistration_path: Path,
    upstream_preregistration: Mapping[str, Any],
) -> Dict[str, Dict[str, str]]:
    records = {
        "optimizer_preregistration": {
            "path": str(upstream_preregistration_path.resolve()),
            "sha256": file_sha256(upstream_preregistration_path),
        }
    }
    for family, key in (
        (PRIMARY_FAMILY, "primary_aggregate"),
        (SECONDARY_FAMILY, "secondary_aggregate"),
    ):
        path = Path(str(upstream_preregistration["planned_outputs"][key])).resolve()
        records[f"{family}_aggregate"] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    return records


def _upstream_scientific_runtime_sha256(
    upstream_preregistration: Mapping[str, Any],
) -> str:
    observed: List[str] = []
    for key in ("primary_aggregate", "secondary_aggregate"):
        aggregate = _read_json(
            Path(str(upstream_preregistration["planned_outputs"][key])).resolve()
        )
        value = str(aggregate.get("scientific_runtime_sha256", ""))
        if len(value) != 64:
            raise MetricCampaignSafetyError("upstream aggregate lacks runtime identity")
        observed.append(value)
    if len(set(observed)) != 1:
        raise MetricCampaignSafetyError("upstream family scientific runtimes differ")
    return observed[0]


def collect_validated_upstream_cells(
    root: Path,
    upstream_preregistration_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, Any]]:
    """Validate both upstream aggregates before reading any metric input state."""

    upstream_preregistration_path = upstream_preregistration_path.expanduser().resolve()
    registered = upstream.load_and_validate_preregistration(
        root, upstream_preregistration_path
    )
    # These calls are the hard successor-exact barrier.  Missing, stale, or
    # incomplete aggregates abort preregistration rather than yielding a
    # provisional metric input list.
    with upstream._successor_context():
        infrastructure._validate_aggregate_file(
            root,
            upstream_preregistration_path,
            registered,
            PRIMARY_FAMILY,
        )
        infrastructure._validate_aggregate_file(
            root,
            upstream_preregistration_path,
            registered,
            SECONDARY_FAMILY,
        )

    cells: List[Dict[str, Any]] = []
    paths: Dict[str, np.ndarray] = {}
    for family in FAMILIES:
        registered_cells = list(registered["cells"][family])
        if len(registered_cells) != EXPECTED_CELL_COUNTS[family]:
            raise MetricCampaignSafetyError(f"unexpected upstream {family} cell count")
        for registered_cell in registered_cells:
            with upstream._successor_context():
                state, artifact = infrastructure.inspect_cell(
                    root, registered_cell
                )
            if state != "verified" or artifact is None:
                raise MetricCampaignSafetyError(
                    f"upstream cell is not freshly verified: {registered_cell['cell_id']}"
                )
            cell, cell_paths = extract_cell_states(
                Path(str(artifact["result"])),
                family,
                str(registered_cell["cell_id"]),
                str(artifact["result_sha256"]),
            )
            cell["controls"] = dict(registered_cell["controls"])
            for path_hash, path in cell_paths.items():
                previous = paths.get(path_hash)
                if previous is not None and not np.array_equal(previous, path):
                    raise MetricCampaignSafetyError("path SHA256 collision across cells")
                paths[path_hash] = path
            cells.append(cell)
    return cells, paths, registered


def _planned_outputs(root: Path, campaign_stem: str) -> Dict[str, Any]:
    reports = (root / "reports").resolve()
    return {
        "host_reports": {
            host: str(reports / f"{campaign_stem}_{host}_host_report.json")
            for host in HOSTS
        },
        "aggregate": str(reports / f"{campaign_stem}_aggregate.json"),
        "analysis": str(reports / f"{campaign_stem}_metric_analysis.json"),
    }


def build_preregistration(
    root: Path,
    preregistration_path: Path,
    upstream_preregistration_path: Path,
    levels: Sequence[int] = DEFAULT_LEVELS,
    convergence_tolerance_kcal: float = DEFAULT_CONVERGENCE_TOLERANCE_KCAL,
    global_lock_path: Path = Path(DEFAULT_GLOBAL_LOCK),
) -> Dict[str, Any]:
    root = _root(root)
    preregistration_path = preregistration_path.expanduser().resolve()
    if global_lock_path.expanduser().resolve() != Path(DEFAULT_GLOBAL_LOCK).resolve():
        raise MetricCampaignSafetyError("formal metric campaign requires the Ala lock")
    suffix = "_preregistered.json"
    if not preregistration_path.name.endswith(suffix):
        raise MetricCampaignSafetyError("metric preregistration filename is invalid")
    campaign_stem = preregistration_path.name[: -len(suffix)]
    source_manifest = _source_manifest(root)
    topology_sha = source_manifest["repository_files"][TOPOLOGY_SOURCE_KEY][
        "sha256"
    ]
    energy_model_sha = payload_sha256(
        {
            "topology_sha256": topology_sha,
            "ala_core_sha256": source_manifest["repository_files"][
                ALA_CORE_SOURCE_KEY
            ]["sha256"],
            "openmm_forcefield_bundle_sha256": source_manifest[
                "openmm_forcefield_bundle"
            ]["bundle_sha256"],
            "energy_protocol": ENERGY_PROTOCOL,
        }
    )
    policy = dense_policy(
        levels, convergence_tolerance_kcal, energy_model_sha
    )
    cells, path_arrays, upstream_registered = collect_validated_upstream_cells(
        root, upstream_preregistration_path
    )
    incumbent, incumbent_paths, incumbent_inputs = extract_incumbent_comparator(
        root
    )
    for path_hash, incumbent_path in incumbent_paths.items():
        previous = path_arrays.get(path_hash)
        if previous is not None and not np.array_equal(previous, incumbent_path):
            raise MetricCampaignSafetyError("incumbent/grid path SHA256 collision")
        path_arrays[path_hash] = incumbent_path
    all_metric_cells = [*cells, incumbent]
    tasks = build_unique_path_tasks(
        root, campaign_stem, all_metric_cells, path_arrays, policy, topology_sha
    )
    sentinels = build_cross_host_sentinel_tasks(
        root, campaign_stem, policy, topology_sha
    )
    family_counts = {
        family: sum(cell["family"] == family for cell in cells) for family in FAMILIES
    }
    if family_counts != EXPECTED_CELL_COUNTS:
        raise MetricCampaignSafetyError("metric source does not contain all 248 cells")
    grid_state_count = sum(len(cell["states"]) for cell in cells)
    incumbent_state_count = len(incumbent["states"])
    state_count = grid_state_count + incumbent_state_count
    expected_unique_diagnostic_nf = sum(
        int(task["expected_diagnostic_raw_joint_evaluations"])
        for task in tasks
    )
    expected_sentinel_diagnostic_nf = sum(
        int(task["expected_diagnostic_raw_joint_evaluations"])
        for task in sentinels.values()
    )
    return {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "campaign_stem": campaign_stem,
        "canonical_root": str(root),
        "preregistration": str(preregistration_path),
        "created_unix": time.time(),
        "upstream_protocol_id": upstream_registered["protocol_id"],
        "upstream_inputs": _upstream_input_manifest(
            upstream_preregistration_path, upstream_registered
        ),
        "predecessor_provenance": _superseded_v1_provenance(root),
        "optimizer_scientific_runtime_sha256": (
            _upstream_scientific_runtime_sha256(upstream_registered)
        ),
        "incumbent_inputs": incumbent_inputs,
        "source_manifest": source_manifest,
        "decision_manifest": _decision_manifest(root),
        "dense_policy": policy,
        "eligibility_policy": ELIGIBILITY_POLICY,
        "accounting_policy": ACCOUNTING_POLICY,
        "concurrency_policy": CONCURRENCY_POLICY,
        "host_assignment_formula": "HOSTS[int(task_id,16) % 2]",
        "hosts": list(HOSTS),
        "jobs_fixed_per_host": 1,
        "thread_environment": THREAD_ENVIRONMENT,
        "host_local_global_lock": DEFAULT_GLOBAL_LOCK,
        "overwrite_allowed": False,
        "restart_policy": "validated_skip_or_new_unique_attempt",
        "input_manifest_frozen_only_after_both_upstream_aggregates": True,
        "family_cell_counts": family_counts,
        "source_cell_count": len(cells),
        "non_grid_comparator_count": 1,
        "grid_eligible_state_reference_count": grid_state_count,
        "incumbent_eligible_state_reference_count": incumbent_state_count,
        "eligible_state_reference_count": state_count,
        "unique_path_task_count": len(tasks),
        "deduplicated_reference_count": state_count - len(tasks),
        "expected_unique_path_diagnostic_raw_joint_evaluations": (
            expected_unique_diagnostic_nf
        ),
        "expected_sentinel_diagnostic_raw_joint_evaluations": (
            expected_sentinel_diagnostic_nf
        ),
        "expected_total_diagnostic_raw_joint_evaluations": (
            expected_unique_diagnostic_nf + expected_sentinel_diagnostic_nf
        ),
        "static_host_shards": {
            host: {
                "unique_path_task_count": sum(
                    task["assigned_host"] == host for task in tasks
                ),
                "expected_unique_path_diagnostic_raw_joint_evaluations": sum(
                    int(task["expected_diagnostic_raw_joint_evaluations"])
                    for task in tasks
                    if task["assigned_host"] == host
                ),
                "expected_sentinel_diagnostic_raw_joint_evaluations": int(
                    sentinels[host][
                        "expected_diagnostic_raw_joint_evaluations"
                    ]
                ),
            }
            for host in HOSTS
        },
        "cells": cells,
        "incumbent_comparator": incumbent,
        "tasks": tasks,
        "cross_host_sentinel_tasks": sentinels,
        "planned_outputs": _planned_outputs(root, campaign_stem),
        "formal_comparison": DECISION_RULES,
    }


def preregister(
    root: Path,
    upstream_preregistration_path: Path,
    levels: Sequence[int] = DEFAULT_LEVELS,
    convergence_tolerance_kcal: float = DEFAULT_CONVERGENCE_TOLERANCE_KCAL,
) -> Path:
    root = _root(root)
    stamp = time.time_ns()
    stem = f"{PROTOCOL_ID}_{stamp}"
    path = (root / "reports" / f"{stem}_preregistered.json").resolve()
    payload = build_preregistration(
        root,
        path,
        upstream_preregistration_path,
        levels,
        convergence_tolerance_kcal,
    )
    _write_json_exclusive(path, payload)
    return path


def _validate_file_record(record: Any, subject: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise MetricCampaignSafetyError(f"invalid {subject} file record")
    path = Path(str(record["path"])).expanduser().resolve()
    if not path.is_file() or file_sha256(path) != str(record["sha256"]):
        raise MetricCampaignSafetyError(f"{subject} file hash drifted: {path}")
    return path


def load_and_validate_preregistration(root: Path, path: Path) -> Dict[str, Any]:
    root = _root(root)
    path = path.expanduser().resolve()
    payload = _read_json(path)
    exact = {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "upstream_protocol_id": upstream.PROTOCOL_ID,
        "canonical_root": str(root),
        "preregistration": str(path),
        "eligibility_policy": ELIGIBILITY_POLICY,
        "accounting_policy": ACCOUNTING_POLICY,
        "concurrency_policy": CONCURRENCY_POLICY,
        "host_assignment_formula": "HOSTS[int(task_id,16) % 2]",
        "hosts": list(HOSTS),
        "jobs_fixed_per_host": 1,
        "thread_environment": THREAD_ENVIRONMENT,
        "host_local_global_lock": DEFAULT_GLOBAL_LOCK,
        "overwrite_allowed": False,
        "restart_policy": "validated_skip_or_new_unique_attempt",
        "input_manifest_frozen_only_after_both_upstream_aggregates": True,
        "family_cell_counts": EXPECTED_CELL_COUNTS,
        "source_cell_count": sum(EXPECTED_CELL_COUNTS.values()),
        "non_grid_comparator_count": 1,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise MetricCampaignSafetyError(f"metric preregistration drifted at {key}")
    suffix = "_preregistered.json"
    if not path.name.endswith(suffix):
        raise MetricCampaignSafetyError("metric preregistration filename is invalid")
    stem = path.name[: -len(suffix)]
    if payload.get("campaign_stem") != stem:
        raise MetricCampaignSafetyError("metric campaign stem drifted")
    source_manifest = payload.get("source_manifest")
    if not isinstance(source_manifest, dict) or set(source_manifest) != {
        "repository_files",
        "openmm_forcefield_bundle",
    }:
        raise MetricCampaignSafetyError("metric source manifest keys drifted")
    expected_source = _source_manifest(root)
    if source_manifest != expected_source:
        raise MetricCampaignSafetyError("metric source manifest drifted")
    repository_files = source_manifest["repository_files"]
    if set(repository_files) != set(SOURCE_PATHS):
        raise MetricCampaignSafetyError("metric repository source keys drifted")
    topology_sha = repository_files[TOPOLOGY_SOURCE_KEY]["sha256"]
    energy_model_sha = payload_sha256(
        {
            "topology_sha256": topology_sha,
            "ala_core_sha256": repository_files[ALA_CORE_SOURCE_KEY]["sha256"],
            "openmm_forcefield_bundle_sha256": source_manifest[
                "openmm_forcefield_bundle"
            ]["bundle_sha256"],
            "energy_protocol": ENERGY_PROTOCOL,
        }
    )
    policy = dense_policy(
        payload.get("dense_policy", {}).get("subdivision_levels", ()),
        payload.get("dense_policy", {}).get(
            "convergence_tolerance_kcal_per_mol", float("nan")
        ),
        energy_model_sha,
    )
    if payload.get("dense_policy") != policy:
        raise MetricCampaignSafetyError("dense metric policy drifted")
    if payload.get("decision_manifest") != _decision_manifest(root):
        raise MetricCampaignSafetyError("metric decision manifest drifted")
    if payload.get("formal_comparison") != DECISION_RULES:
        raise MetricCampaignSafetyError("formal metric comparison rules drifted")
    if payload.get("predecessor_provenance") != _superseded_v1_provenance(root):
        raise MetricCampaignSafetyError(
            "superseded-v1 predecessor provenance drifted"
        )
    upstream_inputs = payload.get("upstream_inputs")
    if not isinstance(upstream_inputs, dict) or set(upstream_inputs) != {
        "optimizer_preregistration",
        f"{PRIMARY_FAMILY}_aggregate",
        f"{SECONDARY_FAMILY}_aggregate",
    }:
        raise MetricCampaignSafetyError("upstream metric input manifest drifted")
    for name, record in upstream_inputs.items():
        _validate_file_record(record, name)
    upstream_preregistration_path = Path(
        str(upstream_inputs["optimizer_preregistration"]["path"])
    ).resolve()
    upstream_registered = upstream.load_and_validate_preregistration(
        root, upstream_preregistration_path
    )
    expected_optimizer_runtime = _upstream_scientific_runtime_sha256(
        upstream_registered
    )
    if payload.get("optimizer_scientific_runtime_sha256") != expected_optimizer_runtime:
        raise MetricCampaignSafetyError("optimizer scientific runtime identity drifted")
    if payload.get("incumbent_inputs") != _incumbent_input_manifest(root):
        raise MetricCampaignSafetyError("frozen incumbent provenance drifted")

    cells = payload.get("cells")
    incumbent = payload.get("incumbent_comparator")
    tasks = payload.get("tasks")
    sentinels = payload.get("cross_host_sentinel_tasks")
    if (
        not isinstance(cells, list)
        or not isinstance(incumbent, dict)
        or not isinstance(tasks, list)
        or not isinstance(sentinels, dict)
    ):
        raise MetricCampaignSafetyError("metric cells/comparator/tasks are missing")
    if len(cells) != sum(EXPECTED_CELL_COUNTS.values()):
        raise MetricCampaignSafetyError("metric cell coverage is not 248")
    counts = {family: 0 for family in FAMILIES}
    state_to_task: Dict[str, str] = {}
    result_records: Dict[str, str] = {}
    all_metric_cells = [*cells, incumbent]
    for cell in all_metric_cells:
        family = cell.get("family")
        if family not in (*FAMILIES, INCUMBENT_FAMILY):
            raise MetricCampaignSafetyError("metric cell has unknown family")
        if family in FAMILIES:
            counts[family] += 1
        elif cell.get("cell_id") != INCUMBENT_CELL_ID:
            raise MetricCampaignSafetyError("metric incumbent identity drifted")
        result_path = str(Path(str(cell.get("source_result"))).resolve())
        result_sha = str(cell.get("source_result_sha256"))
        existing = result_records.setdefault(result_path, result_sha)
        if existing != result_sha:
            raise MetricCampaignSafetyError("one result path has two frozen hashes")
        if not Path(result_path).is_file() or file_sha256(Path(result_path)) != result_sha:
            raise MetricCampaignSafetyError(f"frozen source result drifted: {result_path}")
        states = cell.get("states")
        if not isinstance(states, list) or not states:
            raise MetricCampaignSafetyError("metric cell has no eligible states")
        for state in states:
            if state.get("formal_complete") is not True or state.get("mesh_eligible") is not True:
                raise MetricCampaignSafetyError("ineligible state entered metric manifest")
            if float(state.get("segment_length_cv", float("inf"))) > MESH_CV_MAX:
                raise MetricCampaignSafetyError("state violates CV mesh gate")
            if float(state.get("segment_length_ratio", float("inf"))) > MESH_RATIO_MAX:
                raise MetricCampaignSafetyError("state violates ratio mesh gate")
            if not math.isfinite(float(state.get("raw_i_perp", float("nan")))):
                raise MetricCampaignSafetyError("state raw I_perp is invalid")
            state_id = str(state.get("state_id"))
            task_id = str(state.get("path_task_id"))
            if state_id in state_to_task:
                raise MetricCampaignSafetyError("duplicate metric state ID")
            state_to_task[state_id] = task_id
    if counts != EXPECTED_CELL_COUNTS:
        raise MetricCampaignSafetyError("metric family cell counts drifted")
    if int(payload.get("grid_eligible_state_reference_count", -1)) != sum(
        len(cell["states"]) for cell in cells
    ):
        raise MetricCampaignSafetyError("grid eligible-state count drifted")
    if int(payload.get("incumbent_eligible_state_reference_count", -1)) != len(
        incumbent["states"]
    ):
        raise MetricCampaignSafetyError("incumbent eligible-state count drifted")

    task_ids: set[str] = set()
    task_reference_ids: set[str] = set()
    for task in tasks:
        task_id = str(task.get("task_id"))
        if task_id in task_ids:
            raise MetricCampaignSafetyError("duplicate content-addressed task")
        task_ids.add(task_id)
        expected_id = path_task_id(
            str(task.get("path_sha256")),
            task.get("path_shape", ()),
            policy,
            topology_sha,
        )
        if task_id != expected_id or task.get("assigned_host") != assigned_host(task_id):
            raise MetricCampaignSafetyError("metric task content/routing drifted")
        if task.get("artifact") != str(_task_artifact_path(root, stem, task_id)):
            raise MetricCampaignSafetyError("metric task artifact route drifted")
        if task.get("attempts_directory") != str(
            _task_attempts_path(root, stem, task_id)
        ):
            raise MetricCampaignSafetyError("metric task attempts route drifted")
        refs = task.get("references")
        if not isinstance(refs, list) or len(refs) != int(task.get("reference_count", -1)):
            raise MetricCampaignSafetyError("metric task references are invalid")
        for ref in refs:
            state_id = str(ref.get("state_id"))
            if state_id in task_reference_ids:
                raise MetricCampaignSafetyError("state is referenced by two metric tasks")
            task_reference_ids.add(state_id)
            if state_to_task.get(state_id) != task_id:
                raise MetricCampaignSafetyError("state/task fanout mapping drifted")
    if set(state_to_task.values()) != task_ids or task_reference_ids != set(state_to_task):
        raise MetricCampaignSafetyError("metric task/state coverage is incomplete")
    if payload.get("eligible_state_reference_count") != len(state_to_task):
        raise MetricCampaignSafetyError("eligible state count drifted")
    if payload.get("unique_path_task_count") != len(tasks):
        raise MetricCampaignSafetyError("unique path count drifted")
    if payload.get("deduplicated_reference_count") != len(state_to_task) - len(tasks):
        raise MetricCampaignSafetyError("deduplicated reference count drifted")
    if payload.get("planned_outputs") != _planned_outputs(root, stem):
        raise MetricCampaignSafetyError("metric planned outputs drifted")

    if set(sentinels) != set(HOSTS):
        raise MetricCampaignSafetyError("cross-host sentinel coverage drifted")
    expected_sentinels = build_cross_host_sentinel_tasks(
        root, stem, policy, topology_sha
    )
    if sentinels != expected_sentinels:
        raise MetricCampaignSafetyError("cross-host sentinel task identity drifted")
    expected_unique_diagnostic_nf = sum(
        int(task["expected_diagnostic_raw_joint_evaluations"])
        for task in tasks
    )
    expected_sentinel_diagnostic_nf = sum(
        int(task["expected_diagnostic_raw_joint_evaluations"])
        for task in sentinels.values()
    )
    diagnostic_exact = {
        "expected_unique_path_diagnostic_raw_joint_evaluations": (
            expected_unique_diagnostic_nf
        ),
        "expected_sentinel_diagnostic_raw_joint_evaluations": (
            expected_sentinel_diagnostic_nf
        ),
        "expected_total_diagnostic_raw_joint_evaluations": (
            expected_unique_diagnostic_nf + expected_sentinel_diagnostic_nf
        ),
        "static_host_shards": {
            host: {
                "unique_path_task_count": sum(
                    task["assigned_host"] == host for task in tasks
                ),
                "expected_unique_path_diagnostic_raw_joint_evaluations": sum(
                    int(task["expected_diagnostic_raw_joint_evaluations"])
                    for task in tasks
                    if task["assigned_host"] == host
                ),
                "expected_sentinel_diagnostic_raw_joint_evaluations": int(
                    sentinels[host][
                        "expected_diagnostic_raw_joint_evaluations"
                    ]
                ),
            }
            for host in HOSTS
        },
    }
    for key, expected in diagnostic_exact.items():
        if payload.get(key) != expected:
            raise MetricCampaignSafetyError(
                f"metric diagnostic execution plan drifted at {key}"
            )

    # Re-derive every eligible state and content-addressed task from the frozen
    # successor and incumbent sources.  Structural self-consistency alone is
    # insufficient because a tampered preregistration could otherwise alter
    # raw I_perp/NF/mesh/path identities while retaining valid file hashes.
    rebuilt_cells, rebuilt_paths, rebuilt_upstream = collect_validated_upstream_cells(
        root, upstream_preregistration_path
    )
    rebuilt_incumbent, rebuilt_incumbent_paths, rebuilt_incumbent_inputs = (
        extract_incumbent_comparator(root)
    )
    for path_hash, incumbent_path in rebuilt_incumbent_paths.items():
        previous = rebuilt_paths.get(path_hash)
        if previous is not None and not np.array_equal(previous, incumbent_path):
            raise MetricCampaignSafetyError("rebuilt incumbent/grid path collision")
        rebuilt_paths[path_hash] = incumbent_path
    rebuilt_tasks = build_unique_path_tasks(
        root,
        stem,
        [*rebuilt_cells, rebuilt_incumbent],
        rebuilt_paths,
        policy,
        topology_sha,
    )
    if (
        rebuilt_upstream["protocol_id"] != upstream.PROTOCOL_ID
        or rebuilt_incumbent_inputs != payload["incumbent_inputs"]
        or rebuilt_cells != cells
        or rebuilt_incumbent != incumbent
        or rebuilt_tasks != tasks
    ):
        raise MetricCampaignSafetyError(
            "metric preregistration differs from re-derived frozen inputs"
        )
    return payload


def _rational_key(segment: int, numerator: int, denominator: int) -> Tuple[int, int, int]:
    divisor = math.gcd(numerator, denominator)
    return segment, numerator // divisor, denominator // divisor


def evaluate_nested_dense_barrier(
    path: np.ndarray,
    energy_query: Callable[[np.ndarray], float],
    levels: Sequence[int] = DEFAULT_LEVELS,
    convergence_tolerance_kcal: float = DEFAULT_CONVERGENCE_TOLERANCE_KCAL,
    *,
    formal_shape: bool = True,
) -> Dict[str, Any]:
    """Evaluate nested dense grids, querying every rational path point once."""

    path = _canonical_path(path, formal_shape=formal_shape)
    normalized = normalize_levels(levels)
    tolerance = float(convergence_tolerance_kcal)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise MetricCampaignSafetyError("dense convergence tolerance must be positive")
    cache: Dict[Tuple[int, int, int], float] = {}
    barriers: List[float] = []
    rows: List[Dict[str, Any]] = []
    previous: Optional[float] = None
    final_difference: Optional[float] = None
    logical_queries_before = 0
    segments = path.shape[0] - 1
    for subdivisions in normalized:
        energies: List[float] = []
        for segment in range(segments):
            for numerator in range(subdivisions):
                key = _rational_key(segment, numerator, subdivisions)
                if key not in cache:
                    fraction = numerator / subdivisions
                    point = (
                        (1.0 - fraction) * path[segment]
                        + fraction * path[segment + 1]
                    )
                    energy = float(energy_query(point))
                    if not math.isfinite(energy):
                        raise MetricCampaignSafetyError("diagnostic energy is not finite")
                    cache[key] = energy
                energies.append(cache[key])
        final_key = (segments, 0, 1)
        if final_key not in cache:
            final_energy = float(energy_query(path[-1].copy()))
            if not math.isfinite(final_energy):
                raise MetricCampaignSafetyError("diagnostic endpoint energy is not finite")
            cache[final_key] = final_energy
        energies.append(cache[final_key])
        energy_array = np.asarray(energies, dtype=np.float64)
        expected_size = segments * subdivisions + 1
        if energy_array.size != expected_size:
            raise MetricCampaignSafetyError("dense energy trace has wrong size")
        barrier = float((np.max(energy_array) - energy_array[0]) * EV_TO_KCAL)
        logical_queries = len(cache)
        difference = None if previous is None else abs(barrier - previous)
        rows.append(
            {
                "subdivisions": int(subdivisions),
                "dense_point_count": int(expected_size),
                "new_logical_energy_queries": int(
                    logical_queries - logical_queries_before
                ),
                "cumulative_logical_energy_queries": int(logical_queries),
                "barrier_kcal_per_mol": barrier,
                "difference_from_previous_kcal_per_mol": difference,
                "energy_trace_sha256": array_sha256(energy_array),
            }
        )
        logical_queries_before = logical_queries
        barriers.append(barrier)
        final_difference = difference
        previous = barrier
    if len(rows) != len(normalized):
        raise MetricCampaignSafetyError("dense protocol did not evaluate every level")
    selected = rows[-1]
    converged = bool(
        final_difference is not None and final_difference <= tolerance
    )
    return {
        "dense_policy_id": DENSE_POLICY_ID,
        "evaluated_levels": [row["subdivisions"] for row in rows],
        "level_records": rows,
        "selected_subdivisions": selected["subdivisions"],
        "selected_barrier_kcal_per_mol": selected["barrier_kcal_per_mol"],
        "final_level_difference_kcal_per_mol": final_difference,
        "convergence_tolerance_kcal_per_mol": tolerance,
        "converged": converged,
        "logical_energy_query_requests": int(len(cache)),
        "optimization_nf_charged": 0,
    }


def load_task_path(task: Mapping[str, Any]) -> np.ndarray:
    locator = task.get("source_locator")
    if not isinstance(locator, dict):
        raise MetricCampaignSafetyError("metric task source locator is missing")
    result = Path(str(locator.get("result"))).expanduser().resolve()
    if not result.is_file() or file_sha256(result) != str(locator.get("result_sha256")):
        raise MetricCampaignSafetyError("metric task source result drifted")
    with np.load(result, allow_pickle=False) as saved:
        array_key = str(locator.get("array_key", "stored_paths"))
        if array_key not in saved.files:
            raise MetricCampaignSafetyError(
                f"metric task source lacks {array_key}"
            )
        source_array = saved[array_key]
        if array_key == "initial_path":
            if locator.get("trajectory_index") is not None:
                raise MetricCampaignSafetyError("sentinel source index must be null")
            path = _canonical_path(np.asarray(source_array, dtype=np.float64))
        else:
            index = int(locator.get("trajectory_index", -1))
            if index < 0 or index >= int(source_array.shape[0]):
                raise MetricCampaignSafetyError("metric task source index is invalid")
            path = _canonical_path(
                np.asarray(source_array[index], dtype=np.float64)
            )
    if array_sha256(path) != str(task.get("path_sha256")):
        raise MetricCampaignSafetyError("metric task source path hash drifted")
    return path


class ProductionDenseExecutor:
    """One backend per serial shard, one coordinate cache per unique path."""

    def __init__(self, root: Path, policy: Mapping[str, Any]) -> None:
        # Lazy import keeps construction/validation/unit tests OpenMM-free.
        from ala_core import ImageEvaluator, OpenMMBackend, TOPOLOGY_FILENAME

        self._image_evaluator_type = ImageEvaluator
        self._backend = OpenMMBackend(
            root / "inputs" / TOPOLOGY_FILENAME,
            platform_preference=("CPU",),
            precision="double",
            label="grid_metric_dense_barrier",
        )
        self._policy = dict(policy)

    def __call__(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        path = load_task_path(task)
        evaluator = self._image_evaluator_type(self._backend, image_index=0)
        before = int(self._backend.raw_joint_evaluations)
        result = evaluate_nested_dense_barrier(
            path,
            lambda point: evaluator.query(point, "energy"),
            self._policy["subdivision_levels"],
            self._policy["convergence_tolerance_kcal_per_mol"],
        )
        raw = int(self._backend.raw_joint_evaluations) - before
        evaluator_joint = int(evaluator.counters.joint_evaluations)
        if raw != evaluator_joint:
            raise MetricCampaignSafetyError("diagnostic backend/evaluator NF mismatch")
        if raw > int(result["logical_energy_query_requests"]):
            raise MetricCampaignSafetyError("diagnostic cache misses exceed query requests")
        result.update(
            {
                "diagnostic_raw_joint_evaluations": raw,
                "diagnostic_evaluator_joint_evaluations": evaluator_joint,
                "diagnostic_energy_requests": int(evaluator.counters.energy_requests),
                "diagnostic_force_requests": int(evaluator.counters.force_requests),
                "diagnostic_cache_hits": int(evaluator.counters.cache_hits),
            }
        )
        if result["diagnostic_force_requests"] != 0:
            raise MetricCampaignSafetyError("dense barrier unexpectedly requested forces")
        return result


def _validate_metric_payload(
    payload: Any, task: Mapping[str, Any], policy: Mapping[str, Any]
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise PartialMetricTaskError("metric payload is not a JSON object")
    expected_payload_keys = {
        "dense_policy_id",
        "evaluated_levels",
        "level_records",
        "selected_subdivisions",
        "selected_barrier_kcal_per_mol",
        "final_level_difference_kcal_per_mol",
        "convergence_tolerance_kcal_per_mol",
        "converged",
        "logical_energy_query_requests",
        "optimization_nf_charged",
        "diagnostic_raw_joint_evaluations",
        "diagnostic_evaluator_joint_evaluations",
        "diagnostic_energy_requests",
        "diagnostic_force_requests",
        "diagnostic_cache_hits",
    }
    if set(payload) != expected_payload_keys:
        raise PartialMetricTaskError("metric payload schema drifted")
    exact = {
        "dense_policy_id": DENSE_POLICY_ID,
        "convergence_tolerance_kcal_per_mol": policy[
            "convergence_tolerance_kcal_per_mol"
        ],
        "optimization_nf_charged": 0,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise PartialMetricTaskError(f"metric payload drifted at {key}")
    evaluated = payload.get("evaluated_levels")
    allowed = list(policy["subdivision_levels"])
    if evaluated != allowed:
        raise PartialMetricTaskError("metric payload did not evaluate every dense level")
    records = payload.get("level_records")
    if not isinstance(records, list) or len(records) != len(evaluated):
        raise PartialMetricTaskError("metric level records are misaligned")
    previous_cumulative = 0
    previous_barrier: Optional[float] = None
    expected_record_keys = {
        "subdivisions",
        "dense_point_count",
        "new_logical_energy_queries",
        "cumulative_logical_energy_queries",
        "barrier_kcal_per_mol",
        "difference_from_previous_kcal_per_mol",
        "energy_trace_sha256",
    }
    for subdivisions, record in zip(evaluated, records):
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise PartialMetricTaskError("metric dense-level record schema drifted")
        expected_points = (N_IMAGES - 1) * subdivisions + 1
        if record.get("subdivisions") != subdivisions:
            raise PartialMetricTaskError("metric record subdivision drifted")
        if int(record.get("dense_point_count", -1)) != expected_points:
            raise PartialMetricTaskError("metric record point count drifted")
        cumulative = int(record.get("cumulative_logical_energy_queries", -1))
        increment = int(record.get("new_logical_energy_queries", -1))
        if cumulative != expected_points or increment != cumulative - previous_cumulative:
            raise PartialMetricTaskError("nested diagnostic query accounting drifted")
        barrier = float(record.get("barrier_kcal_per_mol", float("nan")))
        if not math.isfinite(barrier) or barrier < 0.0:
            raise PartialMetricTaskError("dense barrier is not finite")
        difference = record.get("difference_from_previous_kcal_per_mol")
        expected_difference = (
            None if previous_barrier is None else abs(barrier - previous_barrier)
        )
        if difference != expected_difference:
            raise PartialMetricTaskError("successive dense barrier difference drifted")
        trace_hash = str(record.get("energy_trace_sha256"))
        try:
            valid_trace_hash = len(trace_hash) == 64 and int(trace_hash, 16) >= 0
        except ValueError:
            valid_trace_hash = False
        if not valid_trace_hash:
            raise PartialMetricTaskError("energy trace hash is malformed")
        previous_barrier = barrier
        previous_cumulative = cumulative
    tolerance = float(policy["convergence_tolerance_kcal_per_mol"])
    expected_converged = bool(records[-1]["difference_from_previous_kcal_per_mol"] <= tolerance)
    if bool(payload.get("converged")) != expected_converged:
        raise PartialMetricTaskError("dense convergence flag drifted")
    if int(payload.get("selected_subdivisions", -1)) != evaluated[-1]:
        raise PartialMetricTaskError("selected dense level drifted")
    if payload.get("selected_barrier_kcal_per_mol") != records[-1][
        "barrier_kcal_per_mol"
    ]:
        raise PartialMetricTaskError("selected dense barrier drifted")
    if payload.get("final_level_difference_kcal_per_mol") != records[-1][
        "difference_from_previous_kcal_per_mol"
    ]:
        raise PartialMetricTaskError("final dense difference drifted")
    if int(payload.get("logical_energy_query_requests", -1)) != previous_cumulative:
        raise PartialMetricTaskError("logical dense query total drifted")
    raw = int(payload.get("diagnostic_raw_joint_evaluations", -1))
    evaluator_joint = int(payload.get("diagnostic_evaluator_joint_evaluations", -1))
    energy_requests = int(payload.get("diagnostic_energy_requests", -1))
    force_requests = int(payload.get("diagnostic_force_requests", -1))
    cache_hits = int(payload.get("diagnostic_cache_hits", -1))
    expected_logical = int(task.get("expected_logical_energy_query_requests", -1))
    expected_raw = int(task.get("expected_diagnostic_raw_joint_evaluations", -1))
    expected_cache_hits = int(task.get("expected_diagnostic_cache_hits", -1))
    if previous_cumulative != expected_logical:
        raise PartialMetricTaskError("logical query count differs from preregistration")
    if raw != expected_raw or evaluator_joint != expected_raw or force_requests != 0:
        raise PartialMetricTaskError("diagnostic physical NF audit drifted")
    if (
        energy_requests != expected_logical
        or cache_hits != expected_cache_hits
        or raw + cache_hits != energy_requests
    ):
        raise PartialMetricTaskError("diagnostic request/cache accounting drifted")
    return dict(payload)


def _formal_task_artifact(
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
    task: Mapping[str, Any],
    metric_payload: Mapping[str, Any],
    producer_runtime_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "preregistration": str(preregistration_path.resolve()),
        "preregistration_sha256": file_sha256(preregistration_path),
        "task_id": task["task_id"],
        "task_role": task["task_role"],
        "assigned_host": task["assigned_host"],
        "path_sha256": task["path_sha256"],
        "source_locator": task["source_locator"],
        "reference_count": task["reference_count"],
        "producer_runtime_identity": dict(producer_runtime_identity),
        "metric": dict(metric_payload),
    }


def inspect_task(
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
    task: Mapping[str, Any],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    artifact_path = Path(str(task["artifact"])).resolve()
    if not artifact_path.exists():
        return "absent", None
    if not artifact_path.is_file():
        raise PartialMetricTaskError(f"metric task artifact is not a file: {artifact_path}")
    try:
        artifact = _read_json(artifact_path)
        exact = {
            "status": "complete",
            "protocol_id": PROTOCOL_ID,
            "preregistration": str(preregistration_path.resolve()),
            "preregistration_sha256": file_sha256(preregistration_path),
            "task_id": task["task_id"],
            "task_role": task["task_role"],
            "assigned_host": task["assigned_host"],
            "path_sha256": task["path_sha256"],
            "source_locator": task["source_locator"],
            "reference_count": task["reference_count"],
        }
        expected_artifact_keys = set(exact) | {
            "producer_runtime_identity",
            "metric",
        }
        if set(artifact) != expected_artifact_keys:
            raise PartialMetricTaskError("metric task artifact schema drifted")
        for key, expected in exact.items():
            if artifact.get(key) != expected:
                raise PartialMetricTaskError(f"metric task artifact drifted at {key}")
        _validate_metric_payload(
            artifact.get("metric"), task, preregistration["dense_policy"]
        )
        producer_identity = infrastructure._validate_worker_runtime_identity(
            artifact.get("producer_runtime_identity"), task["assigned_host"]
        )
        producer_runtime_sha = infrastructure._scientific_runtime_sha256(
            producer_identity
        )
        if producer_runtime_sha != preregistration[
            "optimizer_scientific_runtime_sha256"
        ]:
            raise PartialMetricTaskError(
                "metric artifact runtime differs from optimizer runtime"
            )
    except PartialMetricTaskError:
        raise
    except Exception as error:
        raise PartialMetricTaskError(
            f"metric task has unvalidated existing artifact: {task['task_id']}"
        ) from error
    return "verified", {
        "task_id": task["task_id"],
        "assigned_host": task["assigned_host"],
        "artifact": str(artifact_path),
        "artifact_sha256": file_sha256(artifact_path),
        "reference_count": task["reference_count"],
        "diagnostic_raw_joint_evaluations": int(
            artifact["metric"]["diagnostic_raw_joint_evaluations"]
        ),
        "selected_subdivisions": int(
            artifact["metric"]["selected_subdivisions"]
        ),
        "converged": bool(artifact["metric"]["converged"]),
        "producer_provenance": "artifact_attested_machine",
        "producer_machine_fingerprint": producer_identity[
            "machine_fingerprint"
        ],
        "producer_runtime_identity_sha256": producer_identity["identity_sha256"],
        "producer_scientific_runtime_sha256": producer_runtime_sha,
    }


def execute_task_sequence(
    tasks: Sequence[Mapping[str, Any]],
    inspector: Callable[[Mapping[str, Any]], Tuple[str, Optional[Dict[str, Any]]]],
    launcher: Callable[[Mapping[str, Any]], None],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        state, artifact = inspector(task)
        if state == "verified":
            if artifact is None:
                raise MetricCampaignSafetyError("verified task returned no artifact")
            rows.append({"action": "validated_skip", **artifact})
            continue
        if state != "absent" or artifact is not None:
            raise MetricCampaignSafetyError("metric task inspector returned invalid state")
        launcher(task)
        state, artifact = inspector(task)
        if state != "verified" or artifact is None:
            raise MetricCampaignSafetyError(
                f"launched metric task did not validate: {task['task_id']}"
            )
        rows.append({"action": "executed", **artifact})
    return rows


def _tasks_for_host(preregistration: Mapping[str, Any], host: str) -> List[Dict[str, Any]]:
    if host not in HOSTS:
        raise MetricCampaignSafetyError(f"unknown metric host: {host}")
    return [task for task in preregistration["tasks"] if task["assigned_host"] == host]


def validate_jobs(jobs: int) -> int:
    if int(jobs) != 1:
        raise MetricCampaignSafetyError("each physical host is fixed to --jobs 1")
    return 1


def validate_thread_environment(
    environment: Mapping[str, str] = os.environ,
) -> Dict[str, str]:
    observed = {key: environment.get(key) for key in THREAD_ENVIRONMENT}
    if observed != THREAD_ENVIRONMENT:
        raise MetricCampaignSafetyError(
            "metric worker requires every preregistered thread variable to equal '1'"
        )
    return dict(THREAD_ENVIRONMENT)


def _launch_task(
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
    task: Mapping[str, Any],
    executor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    worker_identity: Mapping[str, Any],
) -> None:
    attempts = Path(str(task["attempts_directory"])).resolve()
    attempt = attempts / f"attempt_{time.time_ns()}_{uuid.uuid4().hex}"
    attempt.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        attempt / "start.json",
        {
            "task_id": task["task_id"],
            "assigned_host": task["assigned_host"],
            "preregistration": str(preregistration_path.resolve()),
            "preregistration_sha256": file_sha256(preregistration_path),
            "started_unix": time.time(),
        },
    )
    try:
        metric_payload = dict(executor(task))
        _validate_metric_payload(
            metric_payload, task, preregistration["dense_policy"]
        )
        formal = _formal_task_artifact(
            preregistration_path,
            preregistration,
            task,
            metric_payload,
            worker_identity,
        )
        _write_json_exclusive(Path(str(task["artifact"])).resolve(), formal)
    except Exception as error:
        _write_json_exclusive(
            attempt / "failure.json",
            {
                "failed_unix": time.time(),
                "error_type": type(error).__name__,
                "error": str(error),
                "formal_artifact_published": False,
            },
        )
        raise
    _write_json_exclusive(
        attempt / "success.json",
        {
            "finished_unix": time.time(),
            "formal_artifact": task["artifact"],
            "formal_artifact_sha256": file_sha256(Path(str(task["artifact"]))),
        },
    )


def _host_report_path(preregistration: Mapping[str, Any], host: str) -> Path:
    return Path(str(preregistration["planned_outputs"]["host_reports"][host])).resolve()


def _stable_task_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key != "action"
    }


def validate_host_report(
    root: Path,
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
    host: str,
) -> Dict[str, Any]:
    report_path = _host_report_path(preregistration, host)
    report = _read_json(report_path)
    expected_tasks = _tasks_for_host(preregistration, host)
    exact = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "host_shard": host,
        "preregistration": str(preregistration_path.resolve()),
        "preregistration_sha256": file_sha256(preregistration_path),
        "jobs": 1,
        "expected_task_count": len(expected_tasks),
        "host_local_global_lock": preregistration["host_local_global_lock"],
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise MetricCampaignSafetyError(f"metric {host} report drifted at {key}")
    expected_report_keys = set(exact) | {
        "worker_runtime_identity",
        "cross_host_sentinel",
        "rows",
        "unique_path_diagnostic_raw_joint_evaluations",
        "sentinel_diagnostic_raw_joint_evaluations",
    }
    if set(report) != expected_report_keys:
        raise MetricCampaignSafetyError("metric host report schema drifted")
    identity = infrastructure._validate_worker_runtime_identity(
        report.get("worker_runtime_identity"), host
    )
    report_runtime_sha = infrastructure._scientific_runtime_sha256(identity)
    if report_runtime_sha != preregistration["optimizer_scientific_runtime_sha256"]:
        raise MetricCampaignSafetyError(
            "metric report runtime differs from optimizer runtime"
        )
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_tasks):
        raise MetricCampaignSafetyError("metric host report coverage is invalid")
    if [row.get("task_id") for row in rows] != [
        task["task_id"] for task in expected_tasks
    ]:
        raise MetricCampaignSafetyError("metric host report order/coverage drifted")
    for task, row in zip(expected_tasks, rows):
        action = row.get("action")
        if action not in ("executed", "validated_skip"):
            raise MetricCampaignSafetyError("metric report action is invalid")
        if row.get("producer_provenance") != "artifact_attested_machine":
            raise MetricCampaignSafetyError("metric report producer provenance drifted")
        state, fresh = inspect_task(preregistration_path, preregistration, task)
        if state != "verified" or fresh is None or _stable_task_row(row) != fresh:
            raise MetricCampaignSafetyError("metric report artifact record drifted")
        if (
            action == "executed"
            and row.get("producer_runtime_identity_sha256")
            != identity["identity_sha256"]
        ):
            raise MetricCampaignSafetyError(
                "executed metric artifact producer differs from report worker"
            )
    sentinel_task = preregistration["cross_host_sentinel_tasks"][host]
    sentinel_row = report.get("cross_host_sentinel")
    if not isinstance(sentinel_row, dict) or sentinel_row.get("action") not in (
        "executed",
        "validated_skip",
    ):
        raise MetricCampaignSafetyError("metric host sentinel row is invalid")
    sentinel_state, fresh_sentinel = inspect_task(
        preregistration_path, preregistration, sentinel_task
    )
    if (
        sentinel_state != "verified"
        or fresh_sentinel is None
        or _stable_task_row(sentinel_row) != fresh_sentinel
    ):
        raise MetricCampaignSafetyError("metric host sentinel artifact drifted")
    if (
        sentinel_row["action"] == "executed"
        and sentinel_row.get("producer_runtime_identity_sha256")
        != identity["identity_sha256"]
    ):
        raise MetricCampaignSafetyError(
            "executed sentinel producer differs from report worker"
        )
    expected_unique_nf = sum(
        int(task["expected_diagnostic_raw_joint_evaluations"])
        for task in expected_tasks
    )
    expected_sentinel_nf = int(
        sentinel_task["expected_diagnostic_raw_joint_evaluations"]
    )
    if int(report.get("unique_path_diagnostic_raw_joint_evaluations", -1)) != expected_unique_nf:
        raise MetricCampaignSafetyError("metric host unique-path NF total drifted")
    if int(report.get("sentinel_diagnostic_raw_joint_evaluations", -1)) != expected_sentinel_nf:
        raise MetricCampaignSafetyError("metric host sentinel NF total drifted")
    return report


def run_shard(
    root: Path,
    preregistration_path: Path,
    host: str,
    jobs: int = 1,
    wait_for_lock: bool = False,
    executor_factory: Optional[
        Callable[[Path, Mapping[str, Any]], Callable[[Mapping[str, Any]], Mapping[str, Any]]]
    ] = None,
) -> Path:
    root = _root(root)
    validate_jobs(jobs)
    validate_thread_environment()
    preregistration = load_and_validate_preregistration(root, preregistration_path)
    report_path = _host_report_path(preregistration, host)
    if report_path.exists():
        validate_host_report(root, preregistration_path, preregistration, host)
        return report_path
    tasks = _tasks_for_host(preregistration, host)
    sentinel = preregistration["cross_host_sentinel_tasks"][host]
    executor: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None
    identity: Optional[Dict[str, Any]] = None

    def inspect(task: Mapping[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        return inspect_task(preregistration_path, preregistration, task)

    def launch(task: Mapping[str, Any]) -> None:
        nonlocal executor, identity
        if identity is None:
            raise MetricCampaignSafetyError("worker runtime identity is unavailable")
        if executor is None:
            factory = executor_factory or (
                lambda campaign_root, policy: ProductionDenseExecutor(
                    campaign_root, policy
                )
            )
            executor = factory(root, preregistration["dense_policy"])
        _launch_task(
            preregistration_path,
            preregistration,
            task,
            executor,
            identity,
        )

    lock_path = Path(str(preregistration["host_local_global_lock"])).resolve()
    with infrastructure.HostLocalFcntlLock(lock_path, wait=wait_for_lock):
        if report_path.exists():
            validate_host_report(root, preregistration_path, preregistration, host)
            return report_path
        identity = infrastructure.worker_runtime_identity(host)
        sentinel_rows = execute_task_sequence([sentinel], inspect, launch)
        if len(sentinel_rows) != 1:
            raise MetricCampaignSafetyError("metric sentinel execution coverage drifted")
        sentinel_row = sentinel_rows[0]
        rows = execute_task_sequence(tasks, inspect, launch)
        payload = {
            "status": "complete",
            "protocol_id": PROTOCOL_ID,
            "host_shard": host,
            "preregistration": str(preregistration_path.resolve()),
            "preregistration_sha256": file_sha256(preregistration_path),
            "jobs": 1,
            "expected_task_count": len(tasks),
            "host_local_global_lock": str(lock_path),
            "worker_runtime_identity": identity,
            "cross_host_sentinel": sentinel_row,
            "rows": rows,
            "unique_path_diagnostic_raw_joint_evaluations": sum(
                int(row["diagnostic_raw_joint_evaluations"]) for row in rows
            ),
            "sentinel_diagnostic_raw_joint_evaluations": int(
                sentinel_row["diagnostic_raw_joint_evaluations"]
            ),
        }
        _write_json_exclusive(report_path, payload)
    validate_host_report(root, preregistration_path, preregistration, host)
    return report_path


def _rebuild_aggregate_payload(
    root: Path,
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
) -> Dict[str, Any]:
    reports: Dict[str, Dict[str, str]] = {}
    identities: Dict[str, Mapping[str, Any]] = {}
    unique_path_diagnostic_nf = 0
    sentinel_diagnostic_nf = 0
    total_tasks = 0
    nonconverged = 0
    sentinel_metrics: Dict[str, Dict[str, Any]] = {}
    for host in HOSTS:
        report_path = _host_report_path(preregistration, host)
        report = validate_host_report(
            root, preregistration_path, preregistration, host
        )
        reports[host] = {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
        }
        identities[host] = report["worker_runtime_identity"]
        unique_path_diagnostic_nf += int(
            report["unique_path_diagnostic_raw_joint_evaluations"]
        )
        sentinel_diagnostic_nf += int(
            report["sentinel_diagnostic_raw_joint_evaluations"]
        )
        total_tasks += len(report["rows"])
        nonconverged += sum(not bool(row["converged"]) for row in report["rows"])
        sentinel_task = preregistration["cross_host_sentinel_tasks"][host]
        sentinel_artifact = _read_json(Path(str(sentinel_task["artifact"])))
        sentinel_metrics[host] = dict(sentinel_artifact["metric"])
    fingerprints, runtime_sha = infrastructure._runtime_set_attestation(identities)
    if runtime_sha != preregistration["optimizer_scientific_runtime_sha256"]:
        raise MetricCampaignSafetyError(
            "metric and optimizer scientific runtime identities differ"
        )
    if total_tasks != int(preregistration["unique_path_task_count"]):
        raise MetricCampaignSafetyError("metric aggregate task coverage drifted")
    sentinel_comparison = compare_cross_host_sentinel_metrics(
        sentinel_metrics, SENTINEL_ABSOLUTE_TOLERANCE_KCAL
    )
    expected_unique_nf = sum(
        int(task["expected_diagnostic_raw_joint_evaluations"])
        for task in preregistration["tasks"]
    )
    expected_sentinel_nf = sum(
        int(task["expected_diagnostic_raw_joint_evaluations"])
        for task in preregistration["cross_host_sentinel_tasks"].values()
    )
    if unique_path_diagnostic_nf != expected_unique_nf:
        raise MetricCampaignSafetyError("aggregate unique-path diagnostic NF drifted")
    if sentinel_diagnostic_nf != expected_sentinel_nf:
        raise MetricCampaignSafetyError("aggregate sentinel diagnostic NF drifted")
    return {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "preregistration": str(preregistration_path.resolve()),
        "preregistration_sha256": file_sha256(preregistration_path),
        "host_reports": reports,
        "host_machine_fingerprints": fingerprints,
        "scientific_runtime_sha256": runtime_sha,
        "optimizer_scientific_runtime_sha256": preregistration[
            "optimizer_scientific_runtime_sha256"
        ],
        "metric_runtime_matches_optimizer_runtime": True,
        "unique_path_task_count": total_tasks,
        "eligible_state_reference_count": preregistration[
            "eligible_state_reference_count"
        ],
        "unique_path_diagnostic_raw_joint_evaluations": unique_path_diagnostic_nf,
        "sentinel_diagnostic_raw_joint_evaluations": sentinel_diagnostic_nf,
        "total_diagnostic_raw_joint_evaluations": (
            unique_path_diagnostic_nf + sentinel_diagnostic_nf
        ),
        "expected_unique_path_diagnostic_raw_joint_evaluations": expected_unique_nf,
        "expected_sentinel_diagnostic_raw_joint_evaluations": expected_sentinel_nf,
        "optimization_nf_charged": 0,
        "nonconverged_path_task_count": nonconverged,
        "all_path_tasks_dense_converged": nonconverged == 0,
        "cross_host_sentinel": {
            "path_sha256": preregistration["cross_host_sentinel_tasks"][
                HOSTS[0]
            ]["path_sha256"],
            **sentinel_comparison,
        },
    }


def compare_cross_host_sentinel_metrics(
    sentinel_metrics: Mapping[str, Mapping[str, Any]],
    tolerance_kcal: float = SENTINEL_ABSOLUTE_TOLERANCE_KCAL,
) -> Dict[str, Any]:
    if set(sentinel_metrics) != set(HOSTS):
        raise MetricCampaignSafetyError("sentinel metric host coverage drifted")
    tolerance = float(tolerance_kcal)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise MetricCampaignSafetyError("sentinel tolerance is invalid")
    sentinel_differences: Dict[str, float] = {}
    left_records = sentinel_metrics[HOSTS[0]]["level_records"]
    right_records = sentinel_metrics[HOSTS[1]]["level_records"]
    if len(left_records) != len(right_records):
        raise MetricCampaignSafetyError("sentinel dense-level coverage differs")
    for left, right in zip(left_records, right_records):
        if left["subdivisions"] != right["subdivisions"]:
            raise MetricCampaignSafetyError("sentinel dense levels differ")
        difference = abs(
            float(left["barrier_kcal_per_mol"])
            - float(right["barrier_kcal_per_mol"])
        )
        sentinel_differences[str(left["subdivisions"])] = difference
        if difference > tolerance:
            raise MetricCampaignSafetyError(
                "cross-host sentinel barriers exceed numerical tolerance"
            )
    return {
        "absolute_tolerance_kcal_per_mol": tolerance,
        "barrier_absolute_differences_by_subdivision": sentinel_differences,
        "all_levels_within_tolerance": True,
        "both_dense_converged": all(
            bool(metric["converged"]) for metric in sentinel_metrics.values()
        ),
        "sentinel_convergence_is_not_a_comparison_domain_gate": True,
    }


def validate_aggregate(
    root: Path,
    preregistration_path: Path,
    preregistration: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    root = _root(root)
    registered = (
        load_and_validate_preregistration(root, preregistration_path)
        if preregistration is None
        else dict(preregistration)
    )
    aggregate_path = Path(str(registered["planned_outputs"]["aggregate"])).resolve()
    observed = _read_json(aggregate_path)
    expected = _rebuild_aggregate_payload(
        root, preregistration_path, registered
    )
    if observed != expected:
        raise MetricCampaignSafetyError("metric aggregate payload drifted")
    return observed


def create_aggregate(root: Path, preregistration_path: Path) -> Path:
    root = _root(root)
    preregistration = load_and_validate_preregistration(root, preregistration_path)
    aggregate_path = Path(str(preregistration["planned_outputs"]["aggregate"])).resolve()
    if aggregate_path.exists():
        validate_aggregate(root, preregistration_path, preregistration)
        return aggregate_path
    payload = _rebuild_aggregate_payload(
        root, preregistration_path, preregistration
    )
    _write_json_exclusive(aggregate_path, payload)
    validate_aggregate(root, preregistration_path, preregistration)
    return aggregate_path


def cost_to_precision(points: Sequence[Tuple[int, float]], threshold: float) -> float:
    eligible = [int(nf) for nf, metric in points if float(metric) <= float(threshold)]
    return float(min(eligible)) if eligible else math.inf


def full_curve_dominates(
    candidate: Sequence[Tuple[int, float]],
    comparator: Sequence[Tuple[int, float]],
) -> Tuple[bool, bool]:
    thresholds = sorted({float(metric) for _, metric in candidate} | {float(metric) for _, metric in comparator})
    if not thresholds:
        return False, False
    nonworse = True
    strict = False
    for threshold in thresholds:
        left = cost_to_precision(candidate, threshold)
        right = cost_to_precision(comparator, threshold)
        if left > right:
            nonworse = False
            break
        if left < right:
            strict = True
    return nonworse, strict


def joint_target_nf(
    states: Sequence[Mapping[str, Any]],
    i_perp_target: float = I_PERP_TARGET,
    barrier_error_target: float = BARRIER_ERROR_TARGET_KCAL,
) -> float:
    eligible = [
        int(state["optimization_nf"])
        for state in states
        if float(state["raw_i_perp"]) <= i_perp_target
        and float(state["barrier_error_kcal_per_mol"]) <= barrier_error_target
    ]
    return float(min(eligible)) if eligible else math.inf


def compare_candidate_metric_states(
    candidate_states: Sequence[Mapping[str, Any]],
    comparator_states: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    candidate_i = [
        (int(state["optimization_nf"]), float(state["raw_i_perp"]))
        for state in candidate_states
    ]
    comparator_i = [
        (int(state["optimization_nf"]), float(state["raw_i_perp"]))
        for state in comparator_states
    ]
    candidate_e = [
        (
            int(state["optimization_nf"]),
            float(state["barrier_error_kcal_per_mol"]),
        )
        for state in candidate_states
    ]
    comparator_e = [
        (
            int(state["optimization_nf"]),
            float(state["barrier_error_kcal_per_mol"]),
        )
        for state in comparator_states
    ]
    i_nonworse, i_strict = full_curve_dominates(candidate_i, comparator_i)
    e_nonworse, e_strict = full_curve_dominates(candidate_e, comparator_e)
    candidate_joint = joint_target_nf(candidate_states)
    comparator_joint = joint_target_nf(comparator_states)
    return {
        "i_perp_curve_nonworse": i_nonworse,
        "i_perp_curve_strict_somewhere": i_strict,
        "barrier_curve_nonworse": e_nonworse,
        "barrier_curve_strict_somewhere": e_strict,
        "candidate_joint_target_nf": candidate_joint,
        "comparator_joint_target_nf": comparator_joint,
        "candidate_reaches_joint_target": math.isfinite(candidate_joint),
        "full_two_metric_dominance": (
            i_nonworse and e_nonworse and (i_strict or e_strict)
        ),
    }


def analyze_materialized_metric_cells(
    grid_cells: Sequence[Mapping[str, Any]],
    incumbent: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply the frozen grid-only and incumbent-inclusive decision rules."""

    grid_by_id = {str(cell["cell_id"]): cell for cell in grid_cells}
    if len(grid_by_id) != len(grid_cells):
        raise MetricCampaignSafetyError("analysis grid cell IDs are not unique")
    incumbent_id = str(incumbent["cell_id"])
    all_by_id = {**grid_by_id, incumbent_id: incumbent}
    nonconverged_by_cell = {
        cell_id: sorted(
            {
                str(state["path_task_id"])
                for state in cell["states"]
                if not bool(state["dense_converged"])
            }
        )
        for cell_id, cell in all_by_id.items()
    }
    invalid_cells = sorted(
        cell_id for cell_id, tasks in nonconverged_by_cell.items() if tasks
    )

    def dominates(left_id: str, right_id: str) -> bool:
        result = compare_candidate_metric_states(
            all_by_id[left_id]["states"], all_by_id[right_id]["states"]
        )
        return bool(result["full_two_metric_dominance"])

    def nondominated(identities: Sequence[str]) -> List[str]:
        return sorted(
            candidate
            for candidate in identities
            if not any(
                other != candidate and dominates(other, candidate)
                for other in identities
            )
        )

    grid_ids = sorted(grid_by_id)
    inclusive_ids = [*grid_ids, incumbent_id]
    if invalid_cells:
        valid_grid_ids = [
            cell_id for cell_id in grid_ids if cell_id not in invalid_cells
        ]
        valid_inclusive_ids = [
            cell_id for cell_id in inclusive_ids if cell_id not in invalid_cells
        ]
        return {
            "qualifier_status": "not_determined_due_to_nonconverged_paths",
            "invalid_cell_ids": invalid_cells,
            "nonconverged_path_tasks_by_cell": nonconverged_by_cell,
            "grid_only_nondominated": None,
            "incumbent_inclusive_nondominated": None,
            "provisional_nondominated_among_fully_converged_grid_cells": (
                nondominated(valid_grid_ids) if valid_grid_ids else []
            ),
            "provisional_nondominated_among_fully_converged_inclusive_cells": (
                nondominated(valid_inclusive_ids) if valid_inclusive_ids else []
            ),
            "unique_winner_status": "not_determined_due_to_nonconverged_paths",
            "unique_winner_cell_id": None,
            "winner_candidates_before_global_nonconvergence_gate": [],
        }

    grid_only = nondominated(grid_ids)
    inclusive = nondominated(inclusive_ids)
    winner_candidates: List[str] = []
    for candidate in grid_ids:
        candidate_states = all_by_id[candidate]["states"]
        if not math.isfinite(joint_target_nf(candidate_states)):
            continue
        comparators = [
            comparator for comparator in inclusive_ids if comparator != candidate
        ]
        if all(dominates(candidate, comparator) for comparator in comparators):
            winner_candidates.append(candidate)
    unique = winner_candidates[0] if len(winner_candidates) == 1 else None
    return {
        "qualifier_status": "all_formal_eligible_paths_dense_converged",
        "invalid_cell_ids": [],
        "nonconverged_path_tasks_by_cell": nonconverged_by_cell,
        "grid_only_nondominated": grid_only,
        "incumbent_inclusive_nondominated": inclusive,
        "unique_winner_status": (
            "unique_winner" if unique is not None else "no_unique_winner"
        ),
        "unique_winner_cell_id": unique,
        "winner_candidates_before_global_nonconvergence_gate": winner_candidates,
        "provisional_nondominated_among_fully_converged_grid_cells": grid_only,
        "provisional_nondominated_among_fully_converged_inclusive_cells": inclusive,
    }


def _materialize_cell_metrics(
    cell: Mapping[str, Any], task_metrics: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    states: List[Dict[str, Any]] = []
    for source_state in cell["states"]:
        task_id = str(source_state["path_task_id"])
        metric = task_metrics.get(task_id)
        if metric is None:
            raise MetricCampaignSafetyError(
                f"analysis lacks dense metric task {task_id}"
            )
        barrier = float(metric["selected_barrier_kcal_per_mol"])
        states.append(
            {
                "state_id": source_state["state_id"],
                "path_task_id": task_id,
                "trajectory_index": source_state["trajectory_index"],
                "optimization_nf": source_state["optimization_nf"],
                "raw_i_perp": source_state["raw_i_perp"],
                "dense_barrier_kcal_per_mol": barrier,
                "barrier_error_kcal_per_mol": abs(
                    barrier - REFERENCE_BARRIER_KCAL
                ),
                "dense_selected_subdivisions": metric["selected_subdivisions"],
                "dense_final_level_difference_kcal_per_mol": metric[
                    "final_level_difference_kcal_per_mol"
                ],
                "dense_converged": metric["converged"],
            }
        )
    joint_nf = joint_target_nf(states)
    return {
        "family": cell["family"],
        "cell_id": cell["cell_id"],
        "controls": cell.get("controls", {}),
        "comparison_role": cell.get("comparison_role", "grid_candidate"),
        "states": states,
        "joint_target_nf": int(joint_nf) if math.isfinite(joint_nf) else None,
    }


def _load_frozen_highalpha_analyzer(
    root: Path, preregistration: Mapping[str, Any]
) -> Any:
    """Load and verify the preregistered formal analysis implementation."""

    # Lazy loading avoids a top-level cycle because the analyzer imports this
    # campaign to reuse its formal artifact validators.
    import analyze_stage4_ala_discrete_grid_metrics as analyzer

    record = preregistration["source_manifest"]["repository_files"][
        "highalpha_analyzer"
    ]
    expected_path = (root / HIGHALPHA_ANALYZER_RELATIVE).resolve()
    module_path = Path(str(analyzer.__file__)).resolve()
    if module_path.suffix == ".pyc":
        module_path = module_path.with_suffix(".py")
    if (
        module_path != expected_path
        or Path(str(record["path"])).resolve() != expected_path
    ):
        raise MetricCampaignSafetyError("formal high-alpha analyzer route drifted")
    if file_sha256(expected_path) != str(record["sha256"]):
        raise MetricCampaignSafetyError("formal high-alpha analyzer hash drifted")
    if (
        str(analyzer.SUPPORTED_UNIQUE_WINNER_RULE)
        != str(DECISION_RULES["unique_winner_gate_policy_id"])
    ):
        raise MetricCampaignSafetyError("formal high-alpha winner rule drifted")
    return analyzer


def _rebuild_analysis_payload(
    root: Path,
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    created_unix: float,
) -> Dict[str, Any]:
    analyzer = _load_frozen_highalpha_analyzer(root, preregistration)
    aggregate_path = Path(
        str(preregistration["planned_outputs"]["aggregate"])
    ).resolve()
    grid_cells, incumbent, loaded_preregistration, provenance = (
        analyzer.load_formal_metric_cells(root, aggregate_path)
    )
    if loaded_preregistration != preregistration:
        raise MetricCampaignSafetyError(
            "formal high-alpha analyzer loaded a different preregistration"
        )
    decision = analyzer.analyze_cells(
        grid_cells, preregistration["formal_comparison"], incumbent
    )
    formal_sets_determined = (
        decision["grid_only_nondominated"]["status"] == "determined"
        and decision["incumbent_inclusive_nondominated"]["status"]
        == "determined"
    )
    if bool(aggregate["all_path_tasks_dense_converged"]) != (
        formal_sets_determined
    ):
        raise MetricCampaignSafetyError(
            "aggregate/analyzer dense convergence gates disagree"
        )
    analyzer_record = preregistration["source_manifest"]["repository_files"][
        "highalpha_analyzer"
    ]
    created = float(created_unix)
    if not math.isfinite(created) or created <= 0.0:
        raise MetricCampaignSafetyError("formal analysis creation time is invalid")
    payload = dict(decision)
    payload.update({
        "created_unix": created,
        "formal_input_provenance": provenance,
        "dense_metric_quality": {
            "all_path_tasks_dense_converged": bool(
                aggregate["all_path_tasks_dense_converged"]
            ),
            "nonconverged_path_task_count": int(
                aggregate["nonconverged_path_task_count"]
            ),
        },
        "analysis_script": str(Path(str(analyzer_record["path"])).resolve()),
        "analysis_script_sha256": str(analyzer_record["sha256"]),
    })
    return payload


def validate_analysis(
    root: Path,
    preregistration_path: Path,
    preregistration: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    root = _root(root)
    registered = (
        load_and_validate_preregistration(root, preregistration_path)
        if preregistration is None
        else dict(preregistration)
    )
    aggregate = validate_aggregate(root, preregistration_path, registered)
    analysis_path = Path(str(registered["planned_outputs"]["analysis"])).resolve()
    observed = _read_json(analysis_path)
    try:
        created_unix = float(observed["created_unix"])
    except (KeyError, TypeError, ValueError) as error:
        raise MetricCampaignSafetyError(
            "metric analysis creation time is missing or invalid"
        ) from error
    expected = _rebuild_analysis_payload(
        root, preregistration_path, registered, aggregate, created_unix
    )
    if observed != expected:
        raise MetricCampaignSafetyError("metric analysis payload drifted")
    return observed


def create_analysis(root: Path, preregistration_path: Path) -> Path:
    root = _root(root)
    preregistration = load_and_validate_preregistration(root, preregistration_path)
    aggregate = validate_aggregate(root, preregistration_path, preregistration)
    analysis_path = Path(str(preregistration["planned_outputs"]["analysis"])).resolve()
    if analysis_path.exists():
        validate_analysis(root, preregistration_path, preregistration)
        return analysis_path
    payload = _rebuild_analysis_payload(
        root, preregistration_path, preregistration, aggregate, time.time()
    )
    _write_json_exclusive(analysis_path, payload)
    validate_analysis(root, preregistration_path, preregistration)
    return analysis_path


def _parse_levels(value: str) -> Tuple[int, ...]:
    try:
        return normalize_levels(tuple(int(item) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("levels must be comma-separated integers") from error


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preregister_parser = subparsers.add_parser("preregister")
    preregister_parser.add_argument("--root", required=True, type=Path)
    preregister_parser.add_argument("--upstream-preregistration", required=True, type=Path)
    preregister_parser.add_argument("--levels", type=_parse_levels, default=DEFAULT_LEVELS)
    preregister_parser.add_argument(
        "--convergence-tolerance-kcal",
        type=float,
        default=DEFAULT_CONVERGENCE_TOLERANCE_KCAL,
    )
    worker_parser = subparsers.add_parser("run-shard")
    worker_parser.add_argument("--root", required=True, type=Path)
    worker_parser.add_argument("--preregistration", required=True, type=Path)
    worker_parser.add_argument("--host", required=True, choices=HOSTS)
    worker_parser.add_argument("--jobs", type=int, default=1)
    worker_parser.add_argument("--wait-for-lock", action="store_true")
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--root", required=True, type=Path)
    aggregate_parser.add_argument("--preregistration", required=True, type=Path)
    analysis_parser = subparsers.add_parser("analyze")
    analysis_parser.add_argument("--root", required=True, type=Path)
    analysis_parser.add_argument("--preregistration", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "preregister":
        path = preregister(
            args.root,
            args.upstream_preregistration,
            args.levels,
            args.convergence_tolerance_kcal,
        )
    elif args.command == "run-shard":
        path = run_shard(
            args.root,
            args.preregistration,
            args.host,
            args.jobs,
            args.wait_for_lock,
        )
    elif args.command == "aggregate":
        path = create_aggregate(args.root, args.preregistration)
    else:
        path = create_analysis(args.root, args.preregistration)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
