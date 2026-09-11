#!/usr/bin/env python3
"""Audited densified metrics for the eight Ala r=0.1, c=5000 alpha runs.

This campaign is deliberately downstream of, and isolated from, the optimizer
campaign.  It never changes an optimizer result and never charges a diagnostic
energy query to optimizer Number of Forces.  Eligible states are exactly the
initial state or a committed reparameterized state that also satisfies the
frozen mesh-quality gates.

Paths already present in the completed metric-v2 campaign are reused by exact
``path_sha256`` identity.  Their validated metric payload and source artifact
hash are copied into a new, exclusive pointer artifact; OpenMM is invoked only
for a path absent from that content-addressed catalogue.  All new evaluations
are local, serial, and protected by the host-local Ala fcntl lock.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import run_stage4_ala_discrete_grid_campaign as infrastructure
import run_stage4_ala_discrete_grid_metric_campaign as metric_v2
import run_stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign as optimizer
import run_stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_member as member


PROTOCOL_ID = "stage4_ala_discrete_alpha_r01_c5000_metric_campaign_v1"
CANONICAL_ROOT_NAME = infrastructure.CANONICAL_ROOT_NAME
HOST = "local"
EXPECTED_CELL_COUNT = 8
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6)
RESIDUAL_RATIO = 0.1
INNER_MAX_STEPS = 5000
TOTAL_INTERNAL_NF_BUDGET = 5000
DEFAULT_LEVELS = (20, 40, 80, 160)
CONVERGENCE_TOLERANCE_KCAL = 0.001
SELECTED_SUBDIVISIONS = 160
EV_TO_KCAL = 23.0605
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
DEFAULT_GLOBAL_LOCK = infrastructure.DEFAULT_GLOBAL_LOCK
THREAD_ENVIRONMENT = dict(infrastructure.THREAD_ENVIRONMENT)

FROZEN_METRIC_V2_PREREGISTRATION_RELATIVE = (
    "reports/stage4_ala_discrete_grid_metric_campaign_v2_"
    "1787608620070816000_preregistered.json"
)
FROZEN_METRIC_V2_PREREGISTRATION_SHA256 = (
    "5fcfad3228e72c267ea1d768dd6f677528ae1fc8e8c6f180ea0d3498503b98a3"
)

SOURCE_PATHS = {
    "metric_campaign": (
        "src/run_stage4_ala_discrete_alpha_r01_c5000_metric_campaign.py"
    ),
    "metric_v2": "src/run_stage4_ala_discrete_grid_metric_campaign.py",
    "optimizer_campaign": (
        "src/run_stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign.py"
    ),
    "optimizer_member": (
        "src/run_stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_member.py"
    ),
    "optimizer_runtime": (
        "src/run_ivm_v9_discrete_ffrbk_takeover_alpha_r01_c5000_shim.py"
    ),
    "campaign_infrastructure": "src/run_stage4_ala_discrete_grid_campaign.py",
    "ala_core": "src/ala_core.py",
    "topology": "inputs/ala2_fixed.pdb",
}

ELIGIBILITY_POLICY = {
    "formal_complete": "trajectory_index_zero_or_reparam_committed",
    "terminal_partial_state_excluded": True,
    "mesh_gate": {
        "segment_length_cv_max_inclusive": MESH_CV_MAX,
        "segment_length_ratio_max_inclusive": MESH_RATIO_MAX,
    },
    "i_perp": "raw_result_field_no_recompute_no_cummin_no_interpolation",
}

DENSE_METRIC_POLICY = {
    "interpolation": "piecewise_linear_cartesian_coordinates",
    "subdivisions_per_original_segment": list(DEFAULT_LEVELS),
    "execution": "all_four_levels_required_no_early_stop",
    "convergence": (
        "absolute_delta_E_difference_between_levels_80_and_160_leq_"
        "0.001_kcal_per_mol"
    ),
    "selected_level": SELECTED_SUBDIVISIONS,
    "delta_E_definition": "(max(dense_energy_eV)-first_dense_energy_eV)*23.0605",
    "eV_to_kcal_per_mol": EV_TO_KCAL,
}

ACCOUNTING_POLICY = {
    "optimization_nf_source": "result.total_internal_joint_evaluations",
    "optimization_nf_budget_per_run": TOTAL_INTERNAL_NF_BUDGET,
    "diagnostic_nf_definition": "OpenMM backend raw joint energy evaluations",
    "optimization_and_diagnostic_nf_are_separate": True,
    "diagnostic_nf_added_to_optimization_nf": False,
    "optimization_nf_charged_by_this_campaign": 0,
    "reused_metric_v2_diagnostic_nf_is_historical_not_reexecuted": True,
}

CONCURRENCY_POLICY = {
    "physical_host": HOST,
    "jobs": 1,
    "ala_max_concurrent": 1,
    "host_local_fcntl_lock": DEFAULT_GLOBAL_LOCK,
    "lock_held_for_complete_run": True,
    "remote_execution": False,
}


class MetricCampaignSafetyError(RuntimeError):
    """Fail-closed protocol, provenance, or artifact validation error."""


class PartialMetricArtifactError(MetricCampaignSafetyError):
    """A registered output exists but cannot be validated completely."""


def _root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root.name != CANONICAL_ROOT_NAME:
        raise MetricCampaignSafetyError(
            f"metric root must end in {CANONICAL_ROOT_NAME}: {root}"
        )
    return root


def file_sha256(path: Path) -> str:
    return infrastructure.file_sha256(path)


def payload_sha256(payload: Any) -> str:
    return infrastructure.payload_sha256(payload)


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


def _source_manifest(root: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for name, relative in SOURCE_PATHS.items():
        path = (root / relative).resolve()
        if not path.is_file():
            raise MetricCampaignSafetyError(f"missing registered source: {path}")
        records[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "size_bytes": int(path.stat().st_size),
        }
    return records


def _file_record(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise MetricCampaignSafetyError(f"missing frozen input: {path}")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _validate_file_record(record: Any, subject: str) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise MetricCampaignSafetyError(f"invalid {subject} file record")
    path = Path(str(record["path"])).expanduser().resolve()
    if (
        not path.is_file()
        or int(path.stat().st_size) != int(record["size_bytes"])
        or file_sha256(path) != str(record["sha256"])
    ):
        raise MetricCampaignSafetyError(f"{subject} hash/size drifted: {path}")
    return path


def _frozen_metric_v2_path(root: Path) -> Path:
    path = (root / FROZEN_METRIC_V2_PREREGISTRATION_RELATIVE).resolve()
    if (
        not path.is_file()
        or file_sha256(path) != FROZEN_METRIC_V2_PREREGISTRATION_SHA256
    ):
        raise MetricCampaignSafetyError(
            f"frozen metric-v2 preregistration drifted: {path}"
        )
    return path


def _load_frozen_metric_v2(root: Path) -> Tuple[Path, Dict[str, Any]]:
    path = _frozen_metric_v2_path(root)
    registered = metric_v2.load_and_validate_preregistration(root, path)
    policy = registered.get("dense_policy")
    if not isinstance(policy, dict):
        raise MetricCampaignSafetyError("metric-v2 dense policy is missing")
    exact = {
        "subdivision_levels": list(DEFAULT_LEVELS),
        "execution_rule": "evaluate_every_preregistered_level_no_early_stop",
        "selection_rule": "always_select_maximum_preregistered_level",
        "convergence_tolerance_kcal_per_mol": CONVERGENCE_TOLERANCE_KCAL,
        "maximum_subdivisions": SELECTED_SUBDIVISIONS,
    }
    for key, expected in exact.items():
        if policy.get(key) != expected:
            raise MetricCampaignSafetyError(
                f"metric-v2 dense policy differs at {key}"
            )
    energy = policy.get("energy_protocol")
    if (
        not isinstance(energy, dict)
        or energy.get("barrier_definition")
        != "(max(dense_energy)-first_dense_energy)*23.0605"
    ):
        raise MetricCampaignSafetyError("metric-v2 Delta E definition drifted")
    return path, registered


def _optimizer_aggregate_path(registered: Mapping[str, Any]) -> Path:
    planned = registered.get("planned_outputs")
    if not isinstance(planned, dict) or "aggregate" not in planned:
        raise MetricCampaignSafetyError("optimizer preregistration lacks aggregate route")
    return Path(str(planned["aggregate"])).expanduser().resolve()


def _call_optimizer_aggregate_validator(
    root: Path,
    preregistration_path: Path,
    registered: Mapping[str, Any],
) -> Dict[str, Any]:
    """Require and validate the already-complete optimizer aggregate."""

    aggregate_path = _optimizer_aggregate_path(registered)
    if not aggregate_path.is_file():
        raise MetricCampaignSafetyError(
            "optimizer aggregate is absent; finish optimizer campaign first"
        )
    validated_path = Path(
        optimizer.validate_campaign(root, preregistration_path)
    ).resolve()
    if validated_path != aggregate_path:
        raise MetricCampaignSafetyError("optimizer validator returned another aggregate")
    return _read_json(aggregate_path)


def _load_optimizer_inputs(
    root: Path, preregistration_path: Path
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, Any]]:
    preregistration_path = preregistration_path.expanduser().resolve()
    registered = optimizer.load_preregistration(root, preregistration_path)
    if not isinstance(registered, dict):
        raise MetricCampaignSafetyError("optimizer preregistration validator returned no manifest")
    aggregate = _call_optimizer_aggregate_validator(
        root, preregistration_path, registered
    )
    aggregate_path = _optimizer_aggregate_path(registered)
    if not aggregate_path.is_file():
        raise MetricCampaignSafetyError("optimizer aggregate is absent")
    if aggregate.get("status") != "complete":
        raise MetricCampaignSafetyError("optimizer aggregate is not complete")

    registered_cells = registered.get("cells")
    if isinstance(registered_cells, dict):
        if len(registered_cells) != 1:
            raise MetricCampaignSafetyError("optimizer preregistration has unexpected families")
        registered_cells = next(iter(registered_cells.values()))
    if not isinstance(registered_cells, list) or len(registered_cells) != EXPECTED_CELL_COUNT:
        raise MetricCampaignSafetyError("optimizer preregistration must contain eight runs")

    cells: List[Dict[str, Any]] = []
    paths: Dict[str, np.ndarray] = {}
    observed_alphas: List[float] = []
    for registered_cell in registered_cells:
        if not isinstance(registered_cell, dict):
            raise MetricCampaignSafetyError("optimizer run registration is malformed")
        controls = registered_cell.get("controls")
        if not isinstance(controls, dict):
            controls = registered_cell
        alpha = float(controls.get("alpha", float("nan")))
        ratio = float(controls.get("residual_ratio", float("nan")))
        cap = int(controls.get("inner_max_steps", -1))
        budget = int(
            controls.get(
                "total_internal_nf_budget",
                controls.get("total_internal_NF_budget", -1),
            )
        )
        if (
            not math.isfinite(alpha)
            or alpha not in ALPHAS
            or ratio != RESIDUAL_RATIO
            or cap != INNER_MAX_STEPS
            or budget != TOTAL_INTERNAL_NF_BUDGET
        ):
            raise MetricCampaignSafetyError("optimizer run controls drifted")
        validated = member.validate_existing(root, alpha)
        result_path = Path(str(validated["result"])).resolve()
        result_sha = str(validated["result_sha256"])
        cell_id = str(
            registered_cell.get(
                "cell_id", payload_sha256({"protocol_id": PROTOCOL_ID, "alpha": alpha})[:16]
            )
        )
        cell, cell_paths = metric_v2.extract_cell_states(
            result_path,
            metric_v2.PRIMARY_FAMILY,
            cell_id,
            result_sha,
        )
        evidence = cell.get("inner_cap_stop_evidence")
        if (
            not isinstance(evidence, dict)
            or int(evidence.get("inner_cap_stop_count", -1)) != 0
            or evidence.get("inner_cap_binding") is not False
        ):
            raise MetricCampaignSafetyError("optimizer safety cap was not empirically nonbinding")
        cell["controls"] = {
            "alpha": alpha,
            "residual_ratio": ratio,
            "inner_max_steps": cap,
            "total_internal_nf_budget": budget,
        }
        cell["member_audit"] = str(validated["member_audit"])
        cell["member_audit_sha256"] = str(validated["member_audit_sha256"])
        cells.append(cell)
        observed_alphas.append(alpha)
        for path_hash, path in cell_paths.items():
            previous = paths.get(path_hash)
            if previous is not None and not np.array_equal(previous, path):
                raise MetricCampaignSafetyError("path SHA256 collision across optimizer runs")
            paths[path_hash] = path
    if sorted(observed_alphas) != sorted(ALPHAS) or len(set(observed_alphas)) != len(ALPHAS):
        raise MetricCampaignSafetyError("optimizer alpha coverage is not exact")
    cells.sort(key=lambda item: float(item["controls"]["alpha"]))
    return cells, paths, {
        "preregistration": _file_record(preregistration_path),
        "aggregate": _file_record(aggregate_path),
        "protocol_id": registered.get("protocol_id"),
    }


def _task_id(path_sha256: str, policy: Mapping[str, Any], topology_sha256: str) -> str:
    return payload_sha256(
        {
            "protocol_id": PROTOCOL_ID,
            "path_sha256": path_sha256,
            "path_shape": [metric_v2.N_IMAGES, metric_v2.PATH_DOF],
            "dense_policy": dict(policy),
            "topology_sha256": topology_sha256,
        }
    )


def _artifact_path(root: Path, campaign_stem: str, task_id: str) -> Path:
    return (
        root
        / "runs"
        / "diagnostics"
        / campaign_stem
        / "tasks"
        / task_id[:2]
        / f"{task_id}.json"
    ).resolve()


def _attempts_path(root: Path, campaign_stem: str, task_id: str) -> Path:
    return (
        root
        / "runs"
        / "diagnostics"
        / campaign_stem
        / "attempts"
        / task_id[:2]
        / task_id
    ).resolve()


def _planned_outputs(root: Path, campaign_stem: str) -> Dict[str, str]:
    reports = (root / "reports").resolve()
    return {
        "run_report": str(reports / f"{campaign_stem}_local_run_report.json"),
        "aggregate": str(reports / f"{campaign_stem}_aggregate.json"),
    }


def _validated_reuse_record(
    old_preregistration_path: Path,
    old_registered: Mapping[str, Any],
    old_task: Mapping[str, Any],
) -> Dict[str, Any]:
    state, summary = metric_v2.inspect_task(
        old_preregistration_path, old_registered, old_task
    )
    if state != "verified" or summary is None:
        raise MetricCampaignSafetyError("matching metric-v2 artifact is not verified")
    artifact_path = Path(str(old_task["artifact"])).resolve()
    artifact = _read_json(artifact_path)
    metric = metric_v2._validate_metric_payload(
        artifact.get("metric"), old_task, old_registered["dense_policy"]
    )
    return {
        "source_protocol_id": metric_v2.PROTOCOL_ID,
        "source_preregistration": str(old_preregistration_path),
        "source_preregistration_sha256": file_sha256(old_preregistration_path),
        "source_task_id": str(old_task["task_id"]),
        "source_artifact": str(artifact_path),
        "source_artifact_sha256": file_sha256(artifact_path),
        "source_metric_payload_sha256": payload_sha256(metric),
        "source_selected_delta_E_kcal_per_mol": float(
            metric["selected_barrier_kcal_per_mol"]
        ),
        "source_diagnostic_raw_joint_evaluations": int(
            metric["diagnostic_raw_joint_evaluations"]
        ),
    }


def _build_tasks(
    root: Path,
    campaign_stem: str,
    cells: Sequence[Dict[str, Any]],
    path_arrays: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
    topology_sha256: str,
    old_preregistration_path: Path,
    old_registered: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    old_by_path: Dict[str, Mapping[str, Any]] = {}
    for old_task in old_registered.get("tasks", []):
        path_hash = str(old_task.get("path_sha256", ""))
        if path_hash in old_by_path:
            raise MetricCampaignSafetyError("metric-v2 has duplicate path tasks")
        old_by_path[path_hash] = old_task

    references: Dict[str, List[Dict[str, Any]]] = {}
    locator_by_path: Dict[str, Dict[str, Any]] = {}
    seen_state_ids: set[str] = set()
    for cell in cells:
        for state in cell["states"]:
            state_id = str(state["state_id"])
            if state_id in seen_state_ids:
                raise MetricCampaignSafetyError("duplicate eligible state ID")
            seen_state_ids.add(state_id)
            path_hash = str(state["path_sha256"])
            path = path_arrays.get(path_hash)
            if path is None or metric_v2.path_content_sha256(path) != path_hash:
                raise MetricCampaignSafetyError("eligible path bytes/hash are unavailable")
            task_id = _task_id(path_hash, policy, topology_sha256)
            state["path_task_id"] = task_id
            references.setdefault(path_hash, []).append(
                {
                    "state_id": state_id,
                    "cell_id": cell["cell_id"],
                    "alpha": float(cell["controls"]["alpha"]),
                    "source_result": cell["source_result"],
                    "source_result_sha256": cell["source_result_sha256"],
                    "trajectory_index": int(state["trajectory_index"]),
                    "optimization_nf": int(state["optimization_nf"]),
                    "raw_i_perp": float(state["raw_i_perp"]),
                    "path_sha256": path_hash,
                }
            )
            locator_by_path.setdefault(
                path_hash,
                {
                    "result": cell["source_result"],
                    "result_sha256": cell["source_result_sha256"],
                    "trajectory_index": int(state["trajectory_index"]),
                    "state_id": state_id,
                },
            )

    tasks: List[Dict[str, Any]] = []
    for path_hash in sorted(references):
        path = metric_v2._canonical_path(path_arrays[path_hash])
        task_id = _task_id(path_hash, policy, topology_sha256)
        refs = sorted(
            references[path_hash],
            key=lambda item: (float(item["alpha"]), int(item["trajectory_index"])),
        )
        old_task = old_by_path.get(path_hash)
        reuse = (
            _validated_reuse_record(
                old_preregistration_path, old_registered, old_task
            )
            if old_task is not None
            else None
        )
        task = {
            "task_id": task_id,
            "task_role": "eligible_unique_path",
            "path_sha256": path_hash,
            "path_shape": list(path.shape),
            "assigned_host": HOST,
            "execution_mode": (
                "reuse_verified_metric_v2" if reuse is not None else "local_openmm"
            ),
            "source_locator": locator_by_path[path_hash],
            "reference_count": len(refs),
            "references": refs,
            "artifact": str(_artifact_path(root, campaign_stem, task_id)),
            "attempts_directory": str(_attempts_path(root, campaign_stem, task_id)),
            "reuse_source": reuse,
            **metric_v2.expected_nested_coordinate_accounting(path, DEFAULT_LEVELS),
        }
        tasks.append(task)
    if not tasks:
        raise MetricCampaignSafetyError("metric campaign contains no eligible paths")
    if len({str(task["task_id"]) for task in tasks}) != len(tasks):
        raise MetricCampaignSafetyError("metric task IDs are not unique")
    return tasks


def build_preregistration(
    root: Path,
    preregistration_path: Path,
    optimizer_preregistration_path: Path,
) -> Dict[str, Any]:
    root = _root(root)
    preregistration_path = preregistration_path.expanduser().resolve()
    suffix = "_preregistered.json"
    if not preregistration_path.name.endswith(suffix):
        raise MetricCampaignSafetyError("metric preregistration filename is invalid")
    campaign_stem = preregistration_path.name[: -len(suffix)]
    source_manifest = _source_manifest(root)
    old_path, old_registered = _load_frozen_metric_v2(root)
    policy = dict(old_registered["dense_policy"])
    cells, path_arrays, optimizer_inputs = _load_optimizer_inputs(
        root, optimizer_preregistration_path
    )
    topology_sha = source_manifest["topology"]["sha256"]
    if topology_sha != str(
        old_registered["source_manifest"]["repository_files"][
            metric_v2.TOPOLOGY_SOURCE_KEY
        ]["sha256"]
    ):
        raise MetricCampaignSafetyError("new and metric-v2 topology identities differ")
    tasks = _build_tasks(
        root,
        campaign_stem,
        cells,
        path_arrays,
        policy,
        topology_sha,
        old_path,
        old_registered,
    )
    eligible_state_count = sum(len(cell["states"]) for cell in cells)
    reused_tasks = sum(task["execution_mode"] == "reuse_verified_metric_v2" for task in tasks)
    new_tasks = len(tasks) - reused_tasks
    return {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "campaign_stem": campaign_stem,
        "canonical_root": str(root),
        "preregistration": str(preregistration_path),
        "created_unix": time.time(),
        "source_manifest": source_manifest,
        "optimizer_inputs": optimizer_inputs,
        "frozen_metric_v2_input": _file_record(old_path),
        "dense_policy": policy,
        "eligibility_policy": ELIGIBILITY_POLICY,
        "dense_metric_policy": DENSE_METRIC_POLICY,
        "accounting_policy": ACCOUNTING_POLICY,
        "concurrency_policy": CONCURRENCY_POLICY,
        "host_local_global_lock": DEFAULT_GLOBAL_LOCK,
        "jobs_fixed": 1,
        "thread_environment": THREAD_ENVIRONMENT,
        "overwrite_allowed": False,
        "restart_policy": "validated_skip_or_new_unique_attempt",
        "source_cell_count": len(cells),
        "eligible_state_reference_count": eligible_state_count,
        "unique_path_task_count": len(tasks),
        "reused_metric_v2_path_task_count": reused_tasks,
        "new_local_openmm_path_task_count": new_tasks,
        "cells": cells,
        "tasks": tasks,
        "planned_outputs": _planned_outputs(root, campaign_stem),
    }


def preregister(root: Path, optimizer_preregistration_path: Path) -> Path:
    root = _root(root)
    stem = f"{PROTOCOL_ID}_{time.time_ns()}"
    path = (root / "reports" / f"{stem}_preregistered.json").resolve()
    payload = build_preregistration(
        root, path, optimizer_preregistration_path
    )
    _write_json_exclusive(path, payload)
    return path


def _strip_created(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "created_unix"}


def load_and_validate_preregistration(root: Path, path: Path) -> Dict[str, Any]:
    root = _root(root)
    path = path.expanduser().resolve()
    payload = _read_json(path)
    exact = {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "canonical_root": str(root),
        "preregistration": str(path),
        "eligibility_policy": ELIGIBILITY_POLICY,
        "dense_metric_policy": DENSE_METRIC_POLICY,
        "accounting_policy": ACCOUNTING_POLICY,
        "concurrency_policy": CONCURRENCY_POLICY,
        "host_local_global_lock": DEFAULT_GLOBAL_LOCK,
        "jobs_fixed": 1,
        "thread_environment": THREAD_ENVIRONMENT,
        "overwrite_allowed": False,
        "restart_policy": "validated_skip_or_new_unique_attempt",
        "source_cell_count": EXPECTED_CELL_COUNT,
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
    if payload.get("source_manifest") != _source_manifest(root):
        raise MetricCampaignSafetyError("metric source manifest drifted")
    old_path = _validate_file_record(
        payload.get("frozen_metric_v2_input"), "metric-v2 preregistration"
    )
    if (
        old_path != _frozen_metric_v2_path(root)
        or str(payload["frozen_metric_v2_input"]["sha256"])
        != FROZEN_METRIC_V2_PREREGISTRATION_SHA256
    ):
        raise MetricCampaignSafetyError("metric-v2 identity drifted")
    optimizer_inputs = payload.get("optimizer_inputs")
    if not isinstance(optimizer_inputs, dict):
        raise MetricCampaignSafetyError("optimizer input manifest is missing")
    optimizer_preregistration = _validate_file_record(
        optimizer_inputs.get("preregistration"), "optimizer preregistration"
    )
    _validate_file_record(optimizer_inputs.get("aggregate"), "optimizer aggregate")
    rebuilt = build_preregistration(root, path, optimizer_preregistration)
    if _strip_created(rebuilt) != _strip_created(payload):
        raise MetricCampaignSafetyError(
            "metric preregistration differs from re-derived frozen inputs"
        )
    return payload


def _task_metric_from_reuse(task: Mapping[str, Any]) -> Dict[str, Any]:
    reuse = task.get("reuse_source")
    if task.get("execution_mode") != "reuse_verified_metric_v2" or not isinstance(reuse, dict):
        raise MetricCampaignSafetyError("task has no registered reuse source")
    source = Path(str(reuse.get("source_artifact"))).resolve()
    if not source.is_file() or file_sha256(source) != str(reuse.get("source_artifact_sha256")):
        raise MetricCampaignSafetyError("metric-v2 reuse artifact hash drifted")
    artifact = _read_json(source)
    metric = artifact.get("metric")
    if not isinstance(metric, dict) or payload_sha256(metric) != str(
        reuse.get("source_metric_payload_sha256")
    ):
        raise MetricCampaignSafetyError("metric-v2 reuse payload hash drifted")
    if float(metric.get("selected_barrier_kcal_per_mol", float("nan"))) != float(
        reuse.get("source_selected_delta_E_kcal_per_mol", float("nan"))
    ):
        raise MetricCampaignSafetyError("metric-v2 reused Delta E drifted")
    return dict(metric)


def _formal_artifact(
    preregistration_path: Path,
    task: Mapping[str, Any],
    metric: Mapping[str, Any],
    worker_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    mode = str(task["execution_mode"])
    historical_nf = int(metric["diagnostic_raw_joint_evaluations"])
    return {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "preregistration": str(preregistration_path.resolve()),
        "preregistration_sha256": file_sha256(preregistration_path),
        "task_id": task["task_id"],
        "path_sha256": task["path_sha256"],
        "execution_mode": mode,
        "source_locator": task["source_locator"],
        "reference_count": task["reference_count"],
        "reuse_source": task["reuse_source"],
        "worker_runtime_identity": dict(worker_identity),
        "metric": dict(metric),
        "metric_payload_sha256": payload_sha256(metric),
        "densified_delta_E_kcal_per_mol": float(
            metric["selected_barrier_kcal_per_mol"]
        ),
        "accounting": {
            "optimization_nf_charged": 0,
            "new_diagnostic_raw_joint_evaluations": (
                0 if mode == "reuse_verified_metric_v2" else historical_nf
            ),
            "reused_historical_diagnostic_raw_joint_evaluations": (
                historical_nf if mode == "reuse_verified_metric_v2" else 0
            ),
        },
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
        raise PartialMetricArtifactError(f"task artifact is not a file: {artifact_path}")
    try:
        artifact = _read_json(artifact_path)
        exact = {
            "status": "complete",
            "protocol_id": PROTOCOL_ID,
            "preregistration": str(preregistration_path.resolve()),
            "preregistration_sha256": file_sha256(preregistration_path),
            "task_id": task["task_id"],
            "path_sha256": task["path_sha256"],
            "execution_mode": task["execution_mode"],
            "source_locator": task["source_locator"],
            "reference_count": task["reference_count"],
            "reuse_source": task["reuse_source"],
        }
        expected_keys = set(exact) | {
            "worker_runtime_identity",
            "metric",
            "metric_payload_sha256",
            "densified_delta_E_kcal_per_mol",
            "accounting",
        }
        if set(artifact) != expected_keys:
            raise PartialMetricArtifactError("task artifact schema drifted")
        for key, expected in exact.items():
            if artifact.get(key) != expected:
                raise PartialMetricArtifactError(f"task artifact drifted at {key}")
        identity = infrastructure._validate_worker_runtime_identity(
            artifact.get("worker_runtime_identity"), HOST
        )
        metric = metric_v2._validate_metric_payload(
            artifact.get("metric"), task, preregistration["dense_policy"]
        )
        if payload_sha256(metric) != artifact.get("metric_payload_sha256"):
            raise PartialMetricArtifactError("metric payload hash drifted")
        selected = float(metric["selected_barrier_kcal_per_mol"])
        if selected != float(artifact.get("densified_delta_E_kcal_per_mol", float("nan"))):
            raise PartialMetricArtifactError("densified Delta E alias drifted")
        mode = str(task["execution_mode"])
        historical_nf = int(metric["diagnostic_raw_joint_evaluations"])
        expected_accounting = {
            "optimization_nf_charged": 0,
            "new_diagnostic_raw_joint_evaluations": (
                0 if mode == "reuse_verified_metric_v2" else historical_nf
            ),
            "reused_historical_diagnostic_raw_joint_evaluations": (
                historical_nf if mode == "reuse_verified_metric_v2" else 0
            ),
        }
        if artifact.get("accounting") != expected_accounting:
            raise PartialMetricArtifactError("task accounting drifted")
        if mode == "reuse_verified_metric_v2":
            reused = _task_metric_from_reuse(task)
            if reused != metric or payload_sha256(reused) != artifact["metric_payload_sha256"]:
                raise PartialMetricArtifactError("reused metric-v2 values/hash drifted")
        elif mode != "local_openmm":
            raise PartialMetricArtifactError("task execution mode is invalid")
        metric_v2.load_task_path(task)
    except PartialMetricArtifactError:
        raise
    except Exception as error:
        raise PartialMetricArtifactError(
            f"unvalidated existing task artifact: {task['task_id']}"
        ) from error
    accounting = artifact["accounting"]
    return "verified", {
        "task_id": task["task_id"],
        "path_sha256": task["path_sha256"],
        "execution_mode": task["execution_mode"],
        "artifact": str(artifact_path),
        "artifact_sha256": file_sha256(artifact_path),
        "reference_count": int(task["reference_count"]),
        "selected_subdivisions": int(metric["selected_subdivisions"]),
        "densified_delta_E_kcal_per_mol": float(
            metric["selected_barrier_kcal_per_mol"]
        ),
        "dense_converged": bool(metric["converged"]),
        "optimization_nf_charged": 0,
        "new_diagnostic_raw_joint_evaluations": int(
            accounting["new_diagnostic_raw_joint_evaluations"]
        ),
        "reused_historical_diagnostic_raw_joint_evaluations": int(
            accounting["reused_historical_diagnostic_raw_joint_evaluations"]
        ),
        "worker_runtime_identity_sha256": identity["identity_sha256"],
    }


def _launch_task(
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
    task: Mapping[str, Any],
    worker_identity: Mapping[str, Any],
    executor_getter: Callable[[], Callable[[Mapping[str, Any]], Mapping[str, Any]]],
) -> None:
    attempts = Path(str(task["attempts_directory"])).resolve()
    attempt = attempts / f"attempt_{time.time_ns()}_{uuid.uuid4().hex}"
    attempt.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        attempt / "start.json",
        {
            "task_id": task["task_id"],
            "execution_mode": task["execution_mode"],
            "preregistration": str(preregistration_path.resolve()),
            "preregistration_sha256": file_sha256(preregistration_path),
            "started_unix": time.time(),
        },
    )
    try:
        if task["execution_mode"] == "reuse_verified_metric_v2":
            metric = _task_metric_from_reuse(task)
        elif task["execution_mode"] == "local_openmm":
            metric = dict(executor_getter()(task))
        else:
            raise MetricCampaignSafetyError("unregistered task execution mode")
        metric_v2._validate_metric_payload(
            metric, task, preregistration["dense_policy"]
        )
        artifact = _formal_artifact(
            preregistration_path, task, metric, worker_identity
        )
        _write_json_exclusive(Path(str(task["artifact"])).resolve(), artifact)
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


def _stable_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if key != "action"}


def validate_run_report(
    root: Path,
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
) -> Dict[str, Any]:
    report_path = Path(str(preregistration["planned_outputs"]["run_report"])).resolve()
    report = _read_json(report_path)
    exact = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "host": HOST,
        "preregistration": str(preregistration_path.resolve()),
        "preregistration_sha256": file_sha256(preregistration_path),
        "jobs": 1,
        "host_local_global_lock": DEFAULT_GLOBAL_LOCK,
        "expected_task_count": len(preregistration["tasks"]),
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise MetricCampaignSafetyError(f"run report drifted at {key}")
    expected_keys = set(exact) | {
        "worker_runtime_identity",
        "rows",
        "optimization_nf_charged",
        "new_diagnostic_raw_joint_evaluations",
        "reused_historical_diagnostic_raw_joint_evaluations",
    }
    if set(report) != expected_keys:
        raise MetricCampaignSafetyError("run report schema drifted")
    identity = infrastructure._validate_worker_runtime_identity(
        report.get("worker_runtime_identity"), HOST
    )
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(preregistration["tasks"]):
        raise MetricCampaignSafetyError("run report task coverage drifted")
    for task, row in zip(preregistration["tasks"], rows):
        if row.get("action") not in ("executed", "validated_skip"):
            raise MetricCampaignSafetyError("run report action is invalid")
        state, fresh = inspect_task(preregistration_path, preregistration, task)
        if state != "verified" or fresh is None or _stable_row(row) != fresh:
            raise MetricCampaignSafetyError("run report task record drifted")
        if row["action"] == "executed" and row["worker_runtime_identity_sha256"] != identity["identity_sha256"]:
            raise MetricCampaignSafetyError("executed task worker identity drifted")
    new_nf = sum(int(row["new_diagnostic_raw_joint_evaluations"]) for row in rows)
    reused_nf = sum(
        int(row["reused_historical_diagnostic_raw_joint_evaluations"])
        for row in rows
    )
    if (
        report.get("optimization_nf_charged") != 0
        or int(report.get("new_diagnostic_raw_joint_evaluations", -1)) != new_nf
        or int(report.get("reused_historical_diagnostic_raw_joint_evaluations", -1)) != reused_nf
    ):
        raise MetricCampaignSafetyError("run report accounting drifted")
    return report


def run(
    root: Path,
    preregistration_path: Path,
    jobs: int = 1,
    wait_for_lock: bool = False,
    executor_factory: Optional[
        Callable[[Path, Mapping[str, Any]], Callable[[Mapping[str, Any]], Mapping[str, Any]]]
    ] = None,
) -> Path:
    root = _root(root)
    if int(jobs) != 1:
        raise MetricCampaignSafetyError("formal metric execution requires --jobs 1")
    metric_v2.validate_thread_environment(os.environ)
    preregistration = load_and_validate_preregistration(root, preregistration_path)
    report_path = Path(str(preregistration["planned_outputs"]["run_report"])).resolve()
    if report_path.exists():
        validate_run_report(root, preregistration_path, preregistration)
        return report_path

    executor: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None

    def get_executor() -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
        nonlocal executor
        if executor is None:
            factory = executor_factory or (
                lambda campaign_root, policy: metric_v2.ProductionDenseExecutor(
                    campaign_root, policy
                )
            )
            executor = factory(root, preregistration["dense_policy"])
        return executor

    lock_path = Path(DEFAULT_GLOBAL_LOCK).resolve()
    with infrastructure.HostLocalFcntlLock(lock_path, wait=wait_for_lock):
        if report_path.exists():
            validate_run_report(root, preregistration_path, preregistration)
            return report_path
        identity = infrastructure.worker_runtime_identity(HOST)
        rows: List[Dict[str, Any]] = []
        for task in preregistration["tasks"]:
            state, summary = inspect_task(
                preregistration_path, preregistration, task
            )
            if state == "verified" and summary is not None:
                rows.append({"action": "validated_skip", **summary})
                continue
            if state != "absent" or summary is not None:
                raise MetricCampaignSafetyError("task inspector returned invalid state")
            _launch_task(
                preregistration_path,
                preregistration,
                task,
                identity,
                get_executor,
            )
            state, summary = inspect_task(
                preregistration_path, preregistration, task
            )
            if state != "verified" or summary is None:
                raise MetricCampaignSafetyError("launched task did not validate")
            rows.append({"action": "executed", **summary})
        report = {
            "status": "complete",
            "protocol_id": PROTOCOL_ID,
            "host": HOST,
            "preregistration": str(preregistration_path.resolve()),
            "preregistration_sha256": file_sha256(preregistration_path),
            "jobs": 1,
            "host_local_global_lock": str(lock_path),
            "expected_task_count": len(preregistration["tasks"]),
            "worker_runtime_identity": identity,
            "rows": rows,
            "optimization_nf_charged": 0,
            "new_diagnostic_raw_joint_evaluations": sum(
                int(row["new_diagnostic_raw_joint_evaluations"]) for row in rows
            ),
            "reused_historical_diagnostic_raw_joint_evaluations": sum(
                int(row["reused_historical_diagnostic_raw_joint_evaluations"])
                for row in rows
            ),
        }
        _write_json_exclusive(report_path, report)
    validate_run_report(root, preregistration_path, preregistration)
    return report_path


def _rebuild_aggregate(
    root: Path,
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
) -> Dict[str, Any]:
    report_path = Path(str(preregistration["planned_outputs"]["run_report"])).resolve()
    report = validate_run_report(root, preregistration_path, preregistration)
    metric_by_task: Dict[str, Dict[str, Any]] = {}
    artifact_by_task: Dict[str, Dict[str, str]] = {}
    for task in preregistration["tasks"]:
        state, summary = inspect_task(preregistration_path, preregistration, task)
        if state != "verified" or summary is None:
            raise MetricCampaignSafetyError("aggregate input task is not verified")
        artifact_path = Path(str(task["artifact"])).resolve()
        artifact = _read_json(artifact_path)
        metric_by_task[str(task["task_id"])] = dict(artifact["metric"])
        artifact_by_task[str(task["task_id"])] = {
            "path": str(artifact_path),
            "sha256": file_sha256(artifact_path),
        }

    state_metrics: List[Dict[str, Any]] = []
    for cell in preregistration["cells"]:
        alpha = float(cell["controls"]["alpha"])
        for state in cell["states"]:
            task_id = str(state["path_task_id"])
            metric = metric_by_task[task_id]
            task = next(
                item for item in preregistration["tasks"] if item["task_id"] == task_id
            )
            state_metrics.append(
                {
                    "cell_id": cell["cell_id"],
                    "alpha": alpha,
                    "trajectory_index": int(state["trajectory_index"]),
                    "optimization_nf": int(state["optimization_nf"]),
                    "raw_i_perp": float(state["raw_i_perp"]),
                    "segment_length_cv": float(state["segment_length_cv"]),
                    "segment_length_ratio": float(state["segment_length_ratio"]),
                    "path_sha256": state["path_sha256"],
                    "path_task_id": task_id,
                    "metric_execution_mode": task["execution_mode"],
                    "metric_artifact": artifact_by_task[task_id],
                    "densified_delta_E_kcal_per_mol": float(
                        metric["selected_barrier_kcal_per_mol"]
                    ),
                    "selected_subdivisions": int(metric["selected_subdivisions"]),
                    "dense_converged": bool(metric["converged"]),
                    "final_level_difference_kcal_per_mol": float(
                        metric["final_level_difference_kcal_per_mol"]
                    ),
                }
            )
    state_metrics.sort(
        key=lambda item: (
            float(item["alpha"]),
            int(item["optimization_nf"]),
            int(item["trajectory_index"]),
        )
    )
    alpha_summaries = []
    for alpha in ALPHAS:
        rows = [item for item in state_metrics if float(item["alpha"]) == alpha]
        if not rows:
            raise MetricCampaignSafetyError("aggregate alpha has no metric states")
        alpha_summaries.append(
            {
                "alpha": alpha,
                "eligible_state_count": len(rows),
                "dense_converged_state_count": sum(
                    bool(item["dense_converged"]) for item in rows
                ),
                "final_eligible_state": rows[-1],
            }
        )
    return {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "preregistration": str(preregistration_path.resolve()),
        "preregistration_sha256": file_sha256(preregistration_path),
        "run_report": {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
        },
        "source_cell_count": EXPECTED_CELL_COUNT,
        "eligible_state_reference_count": len(state_metrics),
        "unique_path_task_count": len(preregistration["tasks"]),
        "reused_metric_v2_path_task_count": int(
            preregistration["reused_metric_v2_path_task_count"]
        ),
        "new_local_openmm_path_task_count": int(
            preregistration["new_local_openmm_path_task_count"]
        ),
        "optimization_nf_charged": 0,
        "new_diagnostic_raw_joint_evaluations": int(
            report["new_diagnostic_raw_joint_evaluations"]
        ),
        "reused_historical_diagnostic_raw_joint_evaluations": int(
            report["reused_historical_diagnostic_raw_joint_evaluations"]
        ),
        "all_four_dense_levels_required": True,
        "selected_subdivisions": SELECTED_SUBDIVISIONS,
        "dense_convergence_tolerance_kcal_per_mol": CONVERGENCE_TOLERANCE_KCAL,
        "delta_E_definition": DENSE_METRIC_POLICY["delta_E_definition"],
        "all_eligible_states_dense_converged": all(
            bool(item["dense_converged"]) for item in state_metrics
        ),
        "state_metrics": state_metrics,
        "alpha_summaries": alpha_summaries,
    }


def validate_aggregate(
    root: Path,
    preregistration_path: Path,
    preregistration: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    root = _root(root)
    registered = (
        dict(preregistration)
        if preregistration is not None
        else load_and_validate_preregistration(root, preregistration_path)
    )
    path = Path(str(registered["planned_outputs"]["aggregate"])).resolve()
    observed = _read_json(path)
    expected = _rebuild_aggregate(root, preregistration_path, registered)
    if observed != expected:
        raise MetricCampaignSafetyError("metric aggregate differs from verified tasks")
    return observed


def create_aggregate(root: Path, preregistration_path: Path) -> Path:
    root = _root(root)
    registered = load_and_validate_preregistration(root, preregistration_path)
    path = Path(str(registered["planned_outputs"]["aggregate"])).resolve()
    if path.exists():
        validate_aggregate(root, preregistration_path, registered)
        return path
    payload = _rebuild_aggregate(root, preregistration_path, registered)
    _write_json_exclusive(path, payload)
    validate_aggregate(root, preregistration_path, registered)
    return path


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    preregister_parser = subparsers.add_parser("preregister")
    preregister_parser.add_argument(
        "--optimizer-preregistration", type=Path, required=True
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--preregistration", type=Path, required=True)
    run_parser.add_argument("--jobs", type=int, default=1)
    run_parser.add_argument("--wait-for-lock", action="store_true")
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--preregistration", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--preregistration", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.action == "preregister":
        path = preregister(args.root, args.optimizer_preregistration)
    elif args.action == "run":
        path = run(
            args.root,
            args.preregistration,
            jobs=args.jobs,
            wait_for_lock=args.wait_for_lock,
        )
    elif args.action == "aggregate":
        path = create_aggregate(args.root, args.preregistration)
    elif args.action == "validate":
        registered = load_and_validate_preregistration(
            args.root, args.preregistration
        )
        aggregate_path = Path(str(registered["planned_outputs"]["aggregate"]))
        if aggregate_path.exists():
            validate_aggregate(args.root, args.preregistration, registered)
        path = args.preregistration.resolve()
    else:
        raise AssertionError("unreachable action")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
