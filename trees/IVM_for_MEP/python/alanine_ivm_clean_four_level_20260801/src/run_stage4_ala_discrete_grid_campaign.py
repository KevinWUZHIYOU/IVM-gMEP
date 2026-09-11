#!/usr/bin/env python3
"""Preregister, execute, and aggregate the two-tier Ala discrete grid.

This module is deliberately an execution *orchestrator*.  It does not import
OpenMM or implement an optimizer.  The primary 240-cell scientific commands
come from :mod:`run_stage4_ala_discrete_inexact_reparam_grid_member`; the
secondary 8-cell commands come from
:mod:`run_stage4_ala_discrete_exact_noreparam_alpha_grid_member`.

The default execution plan has one serial worker on each physical host.  A
cell is assigned before execution, never dynamically claimed, so delayed file
synchronization cannot dispatch the same scientific cell twice.  Each worker
holds an Ala-specific host-local advisory lock for its whole shard, allowing
one independent BB task to overlap on the same machine.  Secondary workers
are fail-closed behind a fully validated primary aggregate.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


PROTOCOL_ID = "stage4_ala_discrete_grid_campaign_v1"
CANONICAL_ROOT_NAME = "alanine_ivm_clean_four_level_20260801"
PRIMARY_FAMILY = "inexact_reparam"
SECONDARY_FAMILY = "exact_no_reparam"
FAMILIES = (PRIMARY_FAMILY, SECONDARY_FAMILY)
HOSTS = ("local", "remote")
FIXED_PYTHON = "/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python"
DEFAULT_GLOBAL_LOCK = (
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/.ala_global_heavy_task.lock"
)
TOTAL_INTERNAL_NF_BUDGET = 5000
PRIMARY_CELL_COUNT = 240
SECONDARY_CELL_COUNT = 8

THREAD_ENVIRONMENT = {
    "OPENMM_CPU_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

SCIENTIFIC_DISTRIBUTIONS = ("numpy", "scipy", "OpenMM")
ASSIGNED_HOST_SEMANTICS = "static_worker_route_not_artifact_producer"
PROVENANCE_POLICY = {
    "assigned_host_semantics": ASSIGNED_HOST_SEMANTICS,
    "executed_row_producer": "attested_worker_machine",
    "validated_skip_row_producer": "unknown_preexisting_artifact",
    "host_identity_security": "hash_chained_runtime_attestation_not_signature",
}
CONCURRENCY_POLICY = {
    "scope": "per_physical_host",
    "bb_max_concurrent": 1,
    "ala_max_concurrent": 1,
    "bb_and_ala_may_overlap": True,
    "ala_lock_held_for_complete_shard": True,
    "bb_lock_consulted_by_ala_worker": False,
}
KNOWN_AUDIT_LIMITATIONS = {
    "inexact_terminal_partial_band_phase_decomposition": {
        "scope": (
            "the 5000-total-internal-NF hard budget may interrupt an inexact "
            "direct/recovery full-band evaluation"
        ),
        "accounting": "all terminal cache misses remain charged in total NF",
        "state_publication": (
            "the interrupted terminal path is unpublished and marked incomplete"
        ),
        "limitation": (
            "legacy per-phase counters may not distinguish a partial direct band "
            "from a partial recovery band"
        ),
        "impact": (
            "total NF and complete-state cost-to-precision remain valid because "
            "the terminal tail is excluded from complete-state CTP"
        ),
    }
}
REPORT_ROW_ONLY_KEYS = frozenset(
    ("action", "producer_provenance", "producer_machine_fingerprint")
)

SOURCE_PATHS = {
    "campaign": "src/run_stage4_ala_discrete_grid_campaign.py",
    "primary_member": "src/run_stage4_ala_discrete_inexact_reparam_grid_member.py",
    "secondary_member": "src/run_stage4_ala_discrete_exact_noreparam_alpha_grid_member.py",
    "parametric_selector_runtime": "src/run_ivm_v9_discrete_ffrbk_takeover_grid_shim.py",
    "exact_no_reparam_runtime": "src/run_ivm_v9_discrete_ffrbk_takeover_exact_noreparam_shim.py",
    "exact_candidate_budget_solver": "src/run_ivm_v9_discrete_exact_candidate_budget_solver.py",
    "total_internal_nf_runtime": "src/run_ivm_v9_total_internal_nf_budget.py",
    "frozen_discrete_solver": "src/frozen_discrete_b040.py",
    "fixed_selector_runtime": "src/run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py",
    "eta_runtime": "src/run_ivm_v9_discrete_selected_eta_shim.py",
    "production_runner": "src/run_ivm_v9_spring_projected_production_20260730.py",
    "command_helper": "src/run_stage4_uniform_optimized.py",
    "exact_frozen_current_command": (
        "src/run_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0.py"
    ),
    "exact_command_helper": (
        "src/run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0.py"
    ),
    "ala_core": "src/ala_core.py",
    "common_initial_path": "inputs/common_initial_path_equidistant.npz",
    "topology": "inputs/ala2_fixed.pdb",
}


class CampaignSafetyError(RuntimeError):
    """Fail-closed campaign identity, coverage, or artifact error."""


class HostLockHeldError(CampaignSafetyError):
    """Another Ala worker already holds the physical host's Ala lock."""


class PartialCellError(CampaignSafetyError):
    """A cell has artifacts but cannot be validated as complete."""


def _members() -> Tuple[Any, Any]:
    primary = importlib.import_module(
        "run_stage4_ala_discrete_inexact_reparam_grid_member"
    )
    secondary = importlib.import_module(
        "run_stage4_ala_discrete_exact_noreparam_alpha_grid_member"
    )
    return primary, secondary


def _root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root.name != CANONICAL_ROOT_NAME:
        raise CampaignSafetyError(
            f"campaign root must end in {CANONICAL_ROOT_NAME}: {root}"
        )
    return root


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def worker_runtime_identity(host_label: str) -> Dict[str, Any]:
    """Record the actual worker process/machine without importing OpenMM."""

    if host_label not in HOSTS:
        raise ValueError("worker host label must be local or remote")
    expected_python = Path(FIXED_PYTHON).expanduser().resolve()
    actual_python = Path(sys.executable).expanduser().resolve()
    if actual_python != expected_python:
        raise CampaignSafetyError(
            f"campaign worker must run under {expected_python}, got {actual_python}"
        )
    distributions: Dict[str, str] = {}
    for name in SCIENTIFIC_DISTRIBUTIONS:
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise CampaignSafetyError(
                f"campaign worker is missing scientific distribution {name}"
            ) from error
    machine_material = {
        "hostname": socket.gethostname(),
        "platform_node": platform.node(),
        "machine": platform.machine(),
        "node_id_sha256": hashlib.sha256(
            f"{uuid.getnode():012x}".encode("ascii")
        ).hexdigest(),
    }
    identity: Dict[str, Any] = {
        "worker_host_label": host_label,
        "machine": machine_material,
        "machine_fingerprint": payload_sha256(machine_material),
        "system": platform.system(),
        "release": platform.release(),
        "python_version": platform.python_version(),
        "python_executable": str(actual_python),
        "python_executable_sha256": file_sha256(actual_python),
        "scientific_distributions": distributions,
    }
    identity["identity_sha256"] = payload_sha256(identity)
    return identity


def _validate_worker_runtime_identity(
    identity: Any, expected_host: str
) -> Dict[str, Any]:
    if not isinstance(identity, dict):
        raise CampaignSafetyError("worker runtime identity is missing")
    required = {
        "worker_host_label",
        "machine",
        "machine_fingerprint",
        "system",
        "release",
        "python_version",
        "python_executable",
        "python_executable_sha256",
        "scientific_distributions",
        "identity_sha256",
    }
    if set(identity) != required or identity.get("worker_host_label") != expected_host:
        raise CampaignSafetyError("worker runtime identity schema/host drifted")
    machine = identity.get("machine")
    if not isinstance(machine, dict) or payload_sha256(machine) != identity.get(
        "machine_fingerprint"
    ):
        raise CampaignSafetyError("worker machine fingerprint drifted")
    versions = identity.get("scientific_distributions")
    if not isinstance(versions, dict) or set(versions) != set(SCIENTIFIC_DISTRIBUTIONS):
        raise CampaignSafetyError("worker scientific distribution inventory drifted")
    without_hash = {
        key: value for key, value in identity.items() if key != "identity_sha256"
    }
    if payload_sha256(without_hash) != identity.get("identity_sha256"):
        raise CampaignSafetyError("worker runtime identity hash drifted")
    return identity


def _scientific_runtime_sha256(identity: Mapping[str, Any]) -> str:
    return payload_sha256(
        {
            "system": identity["system"],
            "machine": identity["machine"]["machine"],
            "python_version": identity["python_version"],
            "python_executable_sha256": identity["python_executable_sha256"],
            "scientific_distributions": identity["scientific_distributions"],
        }
    )


def _runtime_set_attestation(
    identities: Mapping[str, Mapping[str, Any]]
) -> Tuple[Dict[str, str], str]:
    if not identities:
        raise CampaignSafetyError("aggregate has no worker runtime identities")
    machine_fingerprints: Dict[str, str] = {}
    runtime_hashes: Dict[str, str] = {}
    for host, unvalidated in identities.items():
        identity = _validate_worker_runtime_identity(unvalidated, host)
        machine_fingerprints[host] = str(identity["machine_fingerprint"])
        runtime_hashes[host] = _scientific_runtime_sha256(identity)
    if len(identities) > 1 and len(set(machine_fingerprints.values())) != len(
        identities
    ):
        raise CampaignSafetyError("dual-host reports do not attest distinct machines")
    if len(set(runtime_hashes.values())) != 1:
        raise CampaignSafetyError("host scientific runtime inventories differ")
    return machine_fingerprints, next(iter(runtime_hashes.values()))


def _require_runtime_matches_primary(
    primary_aggregate: Mapping[str, Any],
    observed_runtime_sha256: str,
    subject: str,
) -> None:
    if primary_aggregate.get("scientific_runtime_sha256") != observed_runtime_sha256:
        raise CampaignSafetyError(
            f"{subject} scientific runtime differs from primary campaign"
        )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampaignSafetyError(f"expected a JSON object: {path}")
    return payload


def _source_manifest(root: Path) -> Dict[str, Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    for name, relative in SOURCE_PATHS.items():
        path = (root / relative).resolve()
        if not path.is_file():
            raise CampaignSafetyError(f"missing campaign source/input: {path}")
        records[name] = {"path": str(path), "sha256": file_sha256(path)}
    return records


def _validate_source_manifest(root: Path, manifest: Any) -> None:
    if not isinstance(manifest, dict) or set(manifest) != set(SOURCE_PATHS):
        raise CampaignSafetyError("campaign source manifest keys drifted")
    for name, record in manifest.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise CampaignSafetyError(f"invalid source record: {name}")
        path = Path(str(record["path"])).resolve()
        expected_path = (root / SOURCE_PATHS[name]).resolve()
        if path != expected_path:
            raise CampaignSafetyError(f"campaign source path drifted: {name}")
        if not path.is_file() or file_sha256(path) != str(record["sha256"]):
            raise CampaignSafetyError(f"campaign source hash drifted: {name}")


def _primary_scientific_spec(
    alpha: float, ratio: float, inner_max_steps: int
) -> Dict[str, Any]:
    return {
        "algorithm": ("discrete_b040+one_way_takeover+ffrbk_operator_lbfgs+eta1.1"),
        "alpha_hex": float(alpha).hex(),
        "family": PRIMARY_FAMILY,
        "history": 7,
        "inner_stop": "relative_residual_or_evaluated_joint_step_cap",
        "kkt_atol_hex": float(0.0).hex(),
        "maxstep": int(inner_max_steps),
        "physical_max_step_angstrom_hex": float(1.0).hex(),
        "reparameterization": "state_curve_per_image_from_first_state",
        "residual_ratio_hex": float(ratio).hex(),
        "total_internal_nf_budget": TOTAL_INTERNAL_NF_BUDGET,
    }


def _primary_cell_id(alpha: float, ratio: float, cap: int) -> str:
    return payload_sha256(_primary_scientific_spec(alpha, ratio, cap))[:16]


def primary_assigned_host(
    alpha_index: int,
    ratio_index: int,
    maxstep_index: int,
    single_host_override: Optional[str] = None,
) -> str:
    """Balanced default assignment with an explicit preregistered override."""

    if single_host_override is not None:
        if single_host_override not in HOSTS:
            raise ValueError("single-host override must be local or remote")
        return single_host_override
    return HOSTS[(alpha_index + ratio_index + maxstep_index) % len(HOSTS)]


def _worker_cell_log(
    root: Path, campaign_stem: str, family: str, host: str, cell_id: str
) -> Path:
    return root / "logs" / PROTOCOL_ID / campaign_stem / family / host / cell_id


def build_primary_cells(
    root: Path,
    campaign_stem: str,
    single_host_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    root = _root(root)
    primary, _ = _members()
    cells: List[Dict[str, Any]] = []
    for alpha_index, alpha in enumerate(primary.ALPHAS):
        for ratio_index, ratio in enumerate(primary.RESIDUAL_RATIOS):
            for cap_index, cap in enumerate(primary.INNER_MAX_STEPS):
                host = primary_assigned_host(
                    alpha_index,
                    ratio_index,
                    cap_index,
                    single_host_override,
                )
                command, label = primary.single_run_command(root, alpha, ratio, cap)
                paths = primary.member_paths(root, alpha, ratio, cap)
                cell_id = _primary_cell_id(alpha, ratio, cap)
                wrapper_command = [
                    FIXED_PYTHON,
                    str((root / SOURCE_PATHS["primary_member"]).resolve()),
                    "--root",
                    str(root),
                    "--alpha",
                    f"{float(alpha):g}",
                    "--residual-ratio",
                    f"{float(ratio):g}",
                    "--inner-max-steps",
                    str(int(cap)),
                ]
                result_glob = f"ivm_{label}_central_state_ivm_cfg*_budget5000.npz"
                cells.append(
                    {
                        "ordinal": len(cells),
                        "priority": "primary",
                        "family": PRIMARY_FAMILY,
                        "cell_id": cell_id,
                        "assigned_host": host,
                        "grid_indices": {
                            "alpha": alpha_index,
                            "residual_ratio": ratio_index,
                            "maxstep": cap_index,
                        },
                        "controls": {
                            "alpha": float(alpha),
                            "residual_ratio": float(ratio),
                            "inner_max_steps": int(cap),
                            "total_internal_nf_budget": TOTAL_INTERNAL_NF_BUDGET,
                        },
                        "canonical_scientific_spec": _primary_scientific_spec(
                            float(alpha), float(ratio), int(cap)
                        ),
                        "member_policy_version": primary.MEMBER_POLICY_VERSION,
                        "member_label": label,
                        "scientific_command": command,
                        "scientific_command_sha256": command_sha256(command),
                        "wrapper_command": wrapper_command,
                        "wrapper_command_sha256": command_sha256(wrapper_command),
                        "paths": {
                            "result_dir": str(Path(paths["result_dir"]).resolve()),
                            "result_glob": result_glob,
                            "companion_glob": result_glob.replace(
                                ".npz", ".audit.json"
                            ),
                            "audit": str(Path(paths["audit"]).resolve()),
                            "member_log": str(Path(paths["member_log"]).resolve()),
                            "worker_log": str(
                                _worker_cell_log(
                                    root,
                                    campaign_stem,
                                    PRIMARY_FAMILY,
                                    host,
                                    cell_id,
                                ).resolve()
                            ),
                        },
                    }
                )
    _assert_primary_grid(cells, single_host_override)
    return cells


def _assert_primary_grid(
    cells: Sequence[Mapping[str, Any]], single_host_override: Optional[str]
) -> None:
    if len(cells) != PRIMARY_CELL_COUNT:
        raise CampaignSafetyError("primary grid is not exactly 240 cells")
    ids = [str(cell["cell_id"]) for cell in cells]
    if len(set(ids)) != len(ids):
        raise CampaignSafetyError("primary cell IDs are not unique")
    if single_host_override is not None:
        if {cell["assigned_host"] for cell in cells} != {single_host_override}:
            raise CampaignSafetyError("single-host primary override was not total")
        return

    totals = {host: 0 for host in HOSTS}
    alpha_counts: Dict[Tuple[int, str], int] = {}
    ratio_counts: Dict[Tuple[int, str], int] = {}
    cap_counts: Dict[Tuple[int, str], int] = {}
    pair_counts: Dict[Tuple[int, int, str], int] = {}
    for cell in cells:
        host = str(cell["assigned_host"])
        indices = cell["grid_indices"]
        ai = int(indices["alpha"])
        ri = int(indices["residual_ratio"])
        ci = int(indices["maxstep"])
        totals[host] += 1
        alpha_counts[(ai, host)] = alpha_counts.get((ai, host), 0) + 1
        ratio_counts[(ri, host)] = ratio_counts.get((ri, host), 0) + 1
        cap_counts[(ci, host)] = cap_counts.get((ci, host), 0) + 1
        pair_counts[(ai, ri, host)] = pair_counts.get((ai, ri, host), 0) + 1
    if totals != {"local": 120, "remote": 120}:
        raise CampaignSafetyError("primary host shards are not 120/120")
    if set(alpha_counts.values()) != {15}:
        raise CampaignSafetyError("primary alpha margins are not balanced")
    if set(ratio_counts.values()) != {24}:
        raise CampaignSafetyError("primary residual margins are not balanced")
    if set(cap_counts.values()) != {20}:
        raise CampaignSafetyError("primary maxstep margins are not balanced")
    if set(pair_counts.values()) != {3}:
        raise CampaignSafetyError("primary alpha/residual strata are not 3/3")


def build_secondary_cells(
    root: Path,
    campaign_stem: str,
    single_host_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    root = _root(root)
    _, secondary = _members()
    if single_host_override is not None and single_host_override not in HOSTS:
        raise ValueError("secondary single-host override must be local or remote")
    cells: List[Dict[str, Any]] = []
    for alpha_index, alpha in enumerate(secondary.ALPHAS):
        host = secondary.assigned_host(alpha, single_host_override)
        command, label = secondary.single_run_command(root, alpha, host)
        paths = secondary.member_paths(root, alpha, host)
        cell_id = secondary.cell_id(alpha, host)
        wrapper_command = [
            FIXED_PYTHON,
            str((root / SOURCE_PATHS["secondary_member"]).resolve()),
            "--root",
            str(root),
            "--alpha",
            secondary.alpha_text(alpha),
        ]
        if single_host_override is not None:
            wrapper_command.extend(("--execution-host", host))
        result_glob = secondary.expected_result_glob(alpha, host)
        cells.append(
            {
                "ordinal": alpha_index,
                "priority": "secondary_after_primary_aggregate",
                "family": SECONDARY_FAMILY,
                "cell_id": cell_id,
                "assigned_host": host,
                "execution_host_override": single_host_override,
                "grid_indices": {"alpha": alpha_index},
                "controls": {
                    "alpha": float(alpha),
                    "alpha_text": secondary.alpha_text(alpha),
                    "residual_ratio": float(secondary.RESIDUAL_RATIO),
                    "inner_max_steps": int(secondary.INNER_MAX_STEPS),
                    "scientific_maxstep": "unbounded_subject_to_budget",
                    "total_internal_nf_budget": int(secondary.TOTAL_INTERNAL_NF_BUDGET),
                },
                "canonical_scientific_spec": secondary.canonical_cell_spec(alpha, host),
                "member_policy_version": secondary.MEMBER_POLICY_VERSION,
                "member_label": label,
                "scientific_command": command,
                "scientific_command_sha256": command_sha256(command),
                "wrapper_command": wrapper_command,
                "wrapper_command_sha256": command_sha256(wrapper_command),
                "paths": {
                    "result_dir": str(Path(paths["result_dir"]).resolve()),
                    "result_glob": result_glob,
                    "companion_glob": result_glob.replace(".npz", ".audit.json"),
                    "audit": str(Path(paths["audit"]).resolve()),
                    "member_log": str(Path(paths["member_log"]).resolve()),
                    "worker_log": str(
                        _worker_cell_log(
                            root,
                            campaign_stem,
                            SECONDARY_FAMILY,
                            host,
                            cell_id,
                        ).resolve()
                    ),
                },
            }
        )
    if len(cells) != SECONDARY_CELL_COUNT:
        raise CampaignSafetyError("secondary grid is not exactly eight cells")
    if len({cell["cell_id"] for cell in cells}) != len(cells):
        raise CampaignSafetyError("secondary cell IDs are not unique")
    observed_hosts = {
        host: sum(cell["assigned_host"] == host for cell in cells) for host in HOSTS
    }
    if single_host_override is not None:
        if observed_hosts[single_host_override] != SECONDARY_CELL_COUNT:
            raise CampaignSafetyError("secondary single-host override was not total")
    elif observed_hosts != {"local": 4, "remote": 4}:
        raise CampaignSafetyError("secondary alpha grid is not split 4/4")
    return cells


def _planned_outputs(
    root: Path,
    stem: str,
    primary_required_hosts: Sequence[str],
    secondary_required_hosts: Sequence[str],
) -> Dict[str, Any]:
    reports = root / "reports"
    return {
        "host_reports": {
            PRIMARY_FAMILY: {
                host: str(
                    (reports / f"{stem}_{PRIMARY_FAMILY}_{host}_worker.json").resolve()
                )
                for host in primary_required_hosts
            },
            SECONDARY_FAMILY: {
                host: str(
                    (
                        reports / f"{stem}_{SECONDARY_FAMILY}_{host}_worker.json"
                    ).resolve()
                )
                for host in secondary_required_hosts
            },
        },
        "primary_aggregate": str(
            (reports / f"{stem}_{PRIMARY_FAMILY}_aggregate.json").resolve()
        ),
        "secondary_aggregate": str(
            (reports / f"{stem}_{SECONDARY_FAMILY}_aggregate.json").resolve()
        ),
    }


def build_preregistration(
    root: Path,
    preregistration_path: Path,
    primary_single_host_override: Optional[str] = None,
    secondary_single_host_override: Optional[str] = None,
    global_lock_path: Path = Path(DEFAULT_GLOBAL_LOCK),
) -> Dict[str, Any]:
    root = _root(root)
    preregistration_path = preregistration_path.expanduser().resolve()
    stem = preregistration_path.name
    suffix = "_preregistered.json"
    if not stem.endswith(suffix):
        raise CampaignSafetyError(
            "preregistration filename must end in _preregistered.json"
        )
    campaign_stem = stem[: -len(suffix)]
    if (
        primary_single_host_override is not None
        and primary_single_host_override not in HOSTS
    ):
        raise ValueError("primary single-host override must be local or remote")
    if (
        secondary_single_host_override is not None
        and secondary_single_host_override not in HOSTS
    ):
        raise ValueError("secondary single-host override must be local or remote")
    primary_required_hosts = (
        list(HOSTS)
        if primary_single_host_override is None
        else [primary_single_host_override]
    )
    secondary_required_hosts = (
        list(HOSTS)
        if secondary_single_host_override is None
        else [secondary_single_host_override]
    )
    primary_cells = build_primary_cells(
        root, campaign_stem, primary_single_host_override
    )
    secondary_cells = build_secondary_cells(
        root, campaign_stem, secondary_single_host_override
    )
    return {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "created_unix": time.time(),
        "canonical_root": str(root),
        "preregistration": str(preregistration_path),
        "campaign_stem": campaign_stem,
        "priority_order": [PRIMARY_FAMILY, SECONDARY_FAMILY],
        "secondary_requires_complete_primary_aggregate": True,
        "total_internal_nf_budget_per_cell": TOTAL_INTERNAL_NF_BUDGET,
        "endpoint_evaluations_excluded_from_optimization_nf": True,
        "jobs_fixed_per_host": 1,
        "host_local_global_lock": str(global_lock_path.expanduser().resolve()),
        "thread_environment": dict(THREAD_ENVIRONMENT),
        "provenance_policy": dict(PROVENANCE_POLICY),
        "concurrency_policy": dict(CONCURRENCY_POLICY),
        "known_audit_limitations": dict(KNOWN_AUDIT_LIMITATIONS),
        "primary_execution": {
            "mode": (
                "dual_host_balanced_static"
                if primary_single_host_override is None
                else "single_host_override_preregistered"
            ),
            "assignment_formula": (
                "HOSTS[(alpha_index + residual_index + maxstep_index) % 2]"
                if primary_single_host_override is None
                else None
            ),
            "single_host_override": primary_single_host_override,
            "required_hosts": primary_required_hosts,
            "cell_count": PRIMARY_CELL_COUNT,
        },
        "secondary_execution": {
            "mode": (
                "dual_host_static_alpha_parity"
                if secondary_single_host_override is None
                else "single_host_override_preregistered"
            ),
            "assignment_formula": (
                "HOSTS[alpha_index % 2]"
                if secondary_single_host_override is None
                else None
            ),
            "single_host_override": secondary_single_host_override,
            "required_hosts": secondary_required_hosts,
            "cell_count": SECONDARY_CELL_COUNT,
            "primary_barrier": "validated_primary_aggregate",
        },
        "source_manifest": _source_manifest(root),
        "cells": {
            PRIMARY_FAMILY: primary_cells,
            SECONDARY_FAMILY: secondary_cells,
        },
        "planned_outputs": _planned_outputs(
            root,
            campaign_stem,
            primary_required_hosts,
            secondary_required_hosts,
        ),
        "overwrite_allowed": False,
        "dynamic_claims_used": False,
        "completed_cell_restart_policy": "validated_skip_only",
        "partial_cell_restart_policy": "fail_closed_no_overwrite",
    }


def preregister(
    root: Path,
    primary_single_host_override: Optional[str] = None,
    secondary_single_host_override: Optional[str] = None,
    global_lock_path: Path = Path(DEFAULT_GLOBAL_LOCK),
) -> Path:
    root = _root(root)
    if global_lock_path.expanduser().resolve() != Path(DEFAULT_GLOBAL_LOCK).resolve():
        raise CampaignSafetyError(
            "formal preregistration requires the host-local Ala heavy-task lock"
        )
    stamp = time.time_ns()
    stem = f"{PROTOCOL_ID}_{stamp}"
    path = (root / "reports" / f"{stem}_preregistered.json").resolve()
    payload = build_preregistration(
        root,
        path,
        primary_single_host_override=primary_single_host_override,
        secondary_single_host_override=secondary_single_host_override,
        global_lock_path=global_lock_path,
    )
    _write_json_exclusive(path, payload)
    return path


def load_and_validate_preregistration(root: Path, path: Path) -> Dict[str, Any]:
    root = _root(root)
    path = path.expanduser().resolve()
    payload = _read_json(path)
    exact = {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "canonical_root": str(root),
        "preregistration": str(path),
        "priority_order": [PRIMARY_FAMILY, SECONDARY_FAMILY],
        "secondary_requires_complete_primary_aggregate": True,
        "total_internal_nf_budget_per_cell": TOTAL_INTERNAL_NF_BUDGET,
        "endpoint_evaluations_excluded_from_optimization_nf": True,
        "jobs_fixed_per_host": 1,
        "thread_environment": THREAD_ENVIRONMENT,
        "provenance_policy": PROVENANCE_POLICY,
        "concurrency_policy": CONCURRENCY_POLICY,
        "known_audit_limitations": KNOWN_AUDIT_LIMITATIONS,
        "overwrite_allowed": False,
        "dynamic_claims_used": False,
        "completed_cell_restart_policy": "validated_skip_only",
        "partial_cell_restart_policy": "fail_closed_no_overwrite",
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise CampaignSafetyError(f"preregistration drifted at {key}")
    suffix = "_preregistered.json"
    if not path.name.endswith(suffix):
        raise CampaignSafetyError("preregistration filename is invalid")
    expected_stem = path.name[: -len(suffix)]
    if payload.get("campaign_stem") != expected_stem:
        raise CampaignSafetyError("preregistration campaign stem drifted")
    _validate_source_manifest(root, payload.get("source_manifest"))
    primary_execution = payload.get("primary_execution")
    secondary_execution = payload.get("secondary_execution")
    if not isinstance(primary_execution, dict) or not isinstance(
        secondary_execution, dict
    ):
        raise CampaignSafetyError("execution plan is missing")
    primary_override = primary_execution.get("single_host_override")
    if primary_override is not None and primary_override not in HOSTS:
        raise CampaignSafetyError("invalid preregistered single-host override")
    secondary_override = secondary_execution.get("single_host_override")
    if secondary_override is not None and secondary_override not in HOSTS:
        raise CampaignSafetyError("invalid secondary single-host override")
    expected_primary_execution = {
        "mode": (
            "dual_host_balanced_static"
            if primary_override is None
            else "single_host_override_preregistered"
        ),
        "assignment_formula": (
            "HOSTS[(alpha_index + residual_index + maxstep_index) % 2]"
            if primary_override is None
            else None
        ),
        "single_host_override": primary_override,
        "required_hosts": (
            list(HOSTS) if primary_override is None else [primary_override]
        ),
        "cell_count": PRIMARY_CELL_COUNT,
    }
    expected_secondary_execution = {
        "mode": (
            "dual_host_static_alpha_parity"
            if secondary_override is None
            else "single_host_override_preregistered"
        ),
        "assignment_formula": (
            "HOSTS[alpha_index % 2]" if secondary_override is None else None
        ),
        "single_host_override": secondary_override,
        "required_hosts": (
            list(HOSTS) if secondary_override is None else [secondary_override]
        ),
        "cell_count": SECONDARY_CELL_COUNT,
        "primary_barrier": "validated_primary_aggregate",
    }
    if primary_execution != expected_primary_execution:
        raise CampaignSafetyError("primary execution plan drifted")
    if secondary_execution != expected_secondary_execution:
        raise CampaignSafetyError("secondary execution plan drifted")
    expected_primary = build_primary_cells(
        root, str(payload.get("campaign_stem")), primary_override
    )
    expected_secondary = build_secondary_cells(
        root, str(payload.get("campaign_stem")), secondary_override
    )
    cells = payload.get("cells")
    if not isinstance(cells, dict):
        raise CampaignSafetyError("preregistration cells are missing")
    if cells.get(PRIMARY_FAMILY) != expected_primary:
        raise CampaignSafetyError("registered primary cells drifted")
    if cells.get(SECONDARY_FAMILY) != expected_secondary:
        raise CampaignSafetyError("registered secondary cells drifted")
    expected_outputs = _planned_outputs(
        root,
        str(payload["campaign_stem"]),
        list(primary_execution["required_hosts"]),
        list(secondary_execution["required_hosts"]),
    )
    if payload.get("planned_outputs") != expected_outputs:
        raise CampaignSafetyError("planned output routing drifted")
    lock = Path(str(payload.get("host_local_global_lock", ""))).expanduser()
    if not lock.is_absolute() or lock.resolve() != Path(DEFAULT_GLOBAL_LOCK).resolve():
        raise CampaignSafetyError(
            "preregistered host-local lock is not the canonical Ala heavy-task lock"
        )
    return payload


class HostLocalFcntlLock:
    """Ala-only advisory lock held for the complete lifetime of one shard."""

    def __init__(self, path: Path, wait: bool = False) -> None:
        self.path = path.expanduser().resolve()
        self.wait = bool(wait)
        self.descriptor = -1
        self.inode: Optional[int] = None
        self.acquired_unix: Optional[float] = None

    def __enter__(self) -> "HostLocalFcntlLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        flags = fcntl.LOCK_EX
        if not self.wait:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(descriptor, flags)
        except BlockingIOError as error:
            os.close(descriptor)
            raise HostLockHeldError(
                f"host-local Ala heavy-task lock is held: {self.path}"
            ) from error
        self.descriptor = descriptor
        self.inode = int(os.fstat(descriptor).st_ino)
        self.acquired_unix = time.time()
        return self

    def __exit__(self, *unused: object) -> None:
        if self.descriptor >= 0:
            try:
                fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self.descriptor)
                self.descriptor = -1


def validate_jobs(jobs: int) -> int:
    if int(jobs) != 1:
        raise CampaignSafetyError("each physical host is fixed to --jobs 1")
    return 1


def _cells_for_host(
    preregistration: Mapping[str, Any], family: str, host: str
) -> List[Dict[str, Any]]:
    if family not in FAMILIES or host not in HOSTS:
        raise ValueError("unknown family or host")
    cells = preregistration["cells"][family]
    selected = [cell for cell in cells if cell["assigned_host"] == host]
    expected_hosts = preregistration[
        "primary_execution" if family == PRIMARY_FAMILY else "secondary_execution"
    ]["required_hosts"]
    if host not in expected_hosts:
        raise CampaignSafetyError(
            f"host {host} is not active for preregistered {family}"
        )
    if not selected:
        raise CampaignSafetyError(f"host {host} has no assigned {family} cells")
    return selected


def _existing_paths(cell: Mapping[str, Any]) -> Tuple[List[Path], Path, Path, Path]:
    paths = cell["paths"]
    result_dir = Path(str(paths["result_dir"]))
    results = sorted(result_dir.glob(str(paths["result_glob"])))
    companions = sorted(result_dir.glob(str(paths["companion_glob"])))
    audit = Path(str(paths["audit"]))
    member_log = Path(str(paths["member_log"]))
    worker_log = Path(str(paths["worker_log"]))
    return results + companions, audit, member_log, worker_log


def _artifact_presence(cell: Mapping[str, Any]) -> bool:
    outputs, audit, member_log, worker_log = _existing_paths(cell)
    return bool(outputs or audit.exists() or member_log.exists() or worker_log.exists())


def _validate_primary_artifact(root: Path, cell: Mapping[str, Any]) -> Dict[str, Any]:
    primary, _ = _members()
    controls = cell["controls"]
    paths = cell["paths"]
    result_dir = Path(str(paths["result_dir"]))
    results = sorted(result_dir.glob(str(paths["result_glob"])))
    companions = sorted(result_dir.glob(str(paths["companion_glob"])))
    audit_path = Path(str(paths["audit"]))
    member_log = Path(str(paths["member_log"]))
    if (
        len(results) != 1
        or len(companions) != 1
        or not audit_path.is_file()
        or not member_log.is_dir()
    ):
        raise PartialCellError(f"primary cell is incomplete: {cell['cell_id']}")
    result = results[0].resolve()
    audit = _read_json(audit_path)
    if Path(str(audit.get("result", ""))).resolve() != result:
        raise PartialCellError("primary audit points to a different result")
    if file_sha256(result) != str(audit.get("result_sha256")):
        raise PartialCellError("primary result hash differs from member audit")
    if audit.get("command") != cell["scientific_command"]:
        raise PartialCellError("primary member command differs from preregistration")
    if str(audit.get("scientific_command_sha256")) != str(
        cell["scientific_command_sha256"]
    ):
        raise PartialCellError("primary command hash differs from preregistration")
    for key, expected in (
        ("alpha", float(controls["alpha"])),
        ("residual_ratio", float(controls["residual_ratio"])),
        ("inner_max_steps", int(controls["inner_max_steps"])),
        ("total_internal_nf_budget", TOTAL_INTERNAL_NF_BUDGET),
    ):
        if audit.get(key) != expected:
            raise PartialCellError(f"primary audit drifted at {key}")
    grid_controls = primary.controls(
        controls["alpha"], controls["residual_ratio"], controls["inner_max_steps"]
    )
    summary = primary.validate_result(result, grid_controls)
    companion = primary.validate_companion(
        result,
        int(summary["final_optimization_NF"]),
        int(summary["total_budget_blocked_miss_count"]),
    )
    return {
        "cell_id": cell["cell_id"],
        "assigned_host": cell["assigned_host"],
        "assigned_host_semantics": ASSIGNED_HOST_SEMANTICS,
        "result": str(result),
        "result_sha256": file_sha256(result),
        "member_audit": str(audit_path.resolve()),
        "member_audit_sha256": file_sha256(audit_path),
        "companion_audit": companion["companion_audit"],
        "companion_audit_sha256": companion["companion_audit_sha256"],
        "configuration_sha256": summary["configuration_sha256"],
        "final_optimization_NF": int(summary["final_optimization_NF"]),
        "last_complete_state_NF": int(summary["last_complete_state_NF"]),
        "trajectory_record_count": int(summary["trajectory_record_count"]),
        "scientific_command_sha256": cell["scientific_command_sha256"],
    }


def _validate_secondary_artifact(root: Path, cell: Mapping[str, Any]) -> Dict[str, Any]:
    _, secondary = _members()
    controls = cell["controls"]
    paths = cell["paths"]
    result_dir = Path(str(paths["result_dir"]))
    results = sorted(result_dir.glob(str(paths["result_glob"])))
    companions = sorted(result_dir.glob(str(paths["companion_glob"])))
    audit_path = Path(str(paths["audit"]))
    member_log = Path(str(paths["member_log"]))
    if (
        len(results) != 1
        or len(companions) != 1
        or not audit_path.is_file()
        or not member_log.is_dir()
    ):
        raise PartialCellError(f"secondary cell is incomplete: {cell['cell_id']}")
    result = results[0].resolve()
    audit = _read_json(audit_path)
    if audit.get("status") != "complete":
        raise PartialCellError("secondary member audit is not complete")
    if Path(str(audit.get("result", ""))).resolve() != result:
        raise PartialCellError("secondary audit points to a different result")
    if file_sha256(result) != str(audit.get("result_sha256")):
        raise PartialCellError("secondary result hash differs from member audit")
    if audit.get("command") != cell["scientific_command"]:
        raise PartialCellError("secondary command differs from preregistration")
    if str(audit.get("scientific_command_sha256")) != str(
        cell["scientific_command_sha256"]
    ):
        raise PartialCellError("secondary command hash differs from preregistration")
    if audit.get("assigned_host") != cell["assigned_host"]:
        raise PartialCellError("secondary assigned-host identity drifted")
    if audit.get("cell_id") != cell["cell_id"]:
        raise PartialCellError("secondary cell ID drifted")
    expected_default_host = secondary.assigned_host(controls["alpha"])
    expected_override = cell.get("execution_host_override")
    if audit.get("default_assigned_host") != expected_default_host:
        raise PartialCellError("secondary default host routing drifted")
    if audit.get("execution_host_override") != expected_override:
        raise PartialCellError("secondary host override audit drifted")
    summary = secondary.validate_result(
        result, controls["alpha"], cell["assigned_host"]
    )
    return {
        "cell_id": cell["cell_id"],
        "assigned_host": cell["assigned_host"],
        "assigned_host_semantics": ASSIGNED_HOST_SEMANTICS,
        "result": str(result),
        "result_sha256": file_sha256(result),
        "member_audit": str(audit_path.resolve()),
        "member_audit_sha256": file_sha256(audit_path),
        "companion_audit": summary["runner_counter_audit"],
        "companion_audit_sha256": summary["runner_counter_audit_sha256"],
        "configuration_sha256": summary["configuration_sha256"],
        "final_optimization_NF": int(summary["final_optimization_NF"]),
        "last_complete_state_NF": int(summary["last_exact_complete_NF"]),
        "trajectory_record_count": int(summary["trajectory_record_count"]),
        "scientific_command_sha256": cell["scientific_command_sha256"],
    }


def inspect_cell(
    root: Path, cell: Mapping[str, Any]
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return absent or a freshly revalidated complete artifact."""

    if not _artifact_presence(cell):
        return "absent", None
    try:
        if cell["family"] == PRIMARY_FAMILY:
            artifact = _validate_primary_artifact(root, cell)
        elif cell["family"] == SECONDARY_FAMILY:
            artifact = _validate_secondary_artifact(root, cell)
        else:
            raise CampaignSafetyError("unknown cell family")
    except Exception as error:
        if isinstance(error, PartialCellError):
            raise
        raise PartialCellError(
            f"cell {cell['cell_id']} has unvalidated existing artifacts"
        ) from error
    return "verified", artifact


def _launch_cell(
    root: Path,
    cell: Mapping[str, Any],
    worker_identity: Mapping[str, Any],
) -> None:
    worker_log = Path(str(cell["paths"]["worker_log"]))
    if worker_log.exists():
        raise PartialCellError(f"worker log already exists: {worker_log}")
    worker_log.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        worker_log / "command.json",
        {
            "wrapper_command": cell["wrapper_command"],
            "wrapper_command_sha256": cell["wrapper_command_sha256"],
            "scientific_command_sha256": cell["scientific_command_sha256"],
            "cell_id": cell["cell_id"],
            "assigned_host": cell["assigned_host"],
            "assigned_host_semantics": ASSIGNED_HOST_SEMANTICS,
            "worker_runtime_identity": dict(worker_identity),
        },
    )
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    started = time.time()
    with (worker_log / "stdout.txt").open("x", encoding="utf-8") as stdout, (
        worker_log / "stderr.txt"
    ).open("x", encoding="utf-8") as stderr:
        completed = subprocess.run(
            list(cell["wrapper_command"]),
            cwd=root,
            env=environment,
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    _write_json_exclusive(
        worker_log / "exit.json",
        {
            "returncode": completed.returncode,
            "started_unix": started,
            "finished_unix": time.time(),
        },
    )
    if completed.returncode != 0:
        raise CampaignSafetyError(
            f"cell {cell['cell_id']} failed with return code {completed.returncode}"
        )


def execute_cell_sequence(
    cells: Sequence[Mapping[str, Any]],
    inspector: Callable[[Mapping[str, Any]], Tuple[str, Optional[Dict[str, Any]]]],
    launcher: Callable[[Mapping[str, Any]], None],
) -> List[Dict[str, Any]]:
    """Sequential verified-skip executor; dependency-injected for unit tests."""

    rows: List[Dict[str, Any]] = []
    for cell in cells:
        state, artifact = inspector(cell)
        if state == "verified":
            if artifact is None:
                raise CampaignSafetyError("verified inspection returned no artifact")
            rows.append({"action": "validated_skip", **artifact})
            continue
        if state != "absent" or artifact is not None:
            raise CampaignSafetyError("cell inspector returned an invalid state")
        launcher(cell)
        state, artifact = inspector(cell)
        if state != "verified" or artifact is None:
            raise CampaignSafetyError(
                f"launched cell did not become verified: {cell['cell_id']}"
            )
        rows.append({"action": "executed", **artifact})
    return rows


def _host_report_path(
    preregistration: Mapping[str, Any], family: str, host: str
) -> Path:
    try:
        return Path(
            str(preregistration["planned_outputs"]["host_reports"][family][host])
        ).resolve()
    except KeyError as error:
        raise CampaignSafetyError(
            f"no preregistered {family}/{host} host report"
        ) from error


def _stable_artifact(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if key not in REPORT_ROW_ONLY_KEYS}


def _assert_report_coverage(
    expected_cells: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> None:
    expected = [str(cell["cell_id"]) for cell in expected_cells]
    observed = [str(row.get("cell_id")) for row in rows]
    if len(observed) != len(set(observed)):
        raise CampaignSafetyError("host report contains duplicate cell IDs")
    if observed != expected:
        raise CampaignSafetyError("host report does not preserve exact shard order")


def validate_host_report(
    root: Path,
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
    family: str,
    host: str,
) -> Dict[str, Any]:
    report_path = _host_report_path(preregistration, family, host)
    report = _read_json(report_path)
    expected_cells = _cells_for_host(preregistration, family, host)
    exact = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "family": family,
        "host_shard": host,
        "preregistration": str(preregistration_path.resolve()),
        "preregistration_sha256": file_sha256(preregistration_path),
        "jobs": 1,
        "expected_cell_count": len(expected_cells),
        "host_local_global_lock": preregistration["host_local_global_lock"],
        "provenance_policy": PROVENANCE_POLICY,
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise CampaignSafetyError(f"{family}/{host} host report drifted at {key}")
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise CampaignSafetyError("host report rows are missing")
    identity = _validate_worker_runtime_identity(
        report.get("worker_runtime_identity"), host
    )
    _assert_report_coverage(expected_cells, rows)
    for cell, stored in zip(expected_cells, rows):
        if stored.get("action") not in ("executed", "validated_skip"):
            raise CampaignSafetyError("host report has an invalid cell action")
        if stored["action"] == "executed":
            expected_provenance = "attested_worker_machine"
            expected_fingerprint = identity["machine_fingerprint"]
        else:
            expected_provenance = "unknown_preexisting_artifact"
            expected_fingerprint = None
        if (
            stored.get("producer_provenance") != expected_provenance
            or stored.get("producer_machine_fingerprint") != expected_fingerprint
        ):
            raise CampaignSafetyError("host report row producer provenance drifted")
        state, fresh = inspect_cell(root, cell)
        if state != "verified" or fresh is None:
            raise CampaignSafetyError("host report cell is no longer verified")
        if _stable_artifact(stored) != fresh:
            raise CampaignSafetyError("host report artifact record drifted")
    return report


def _aggregate_path(preregistration: Mapping[str, Any], family: str) -> Path:
    key = "primary_aggregate" if family == PRIMARY_FAMILY else "secondary_aggregate"
    return Path(str(preregistration["planned_outputs"][key])).resolve()


def _required_hosts(preregistration: Mapping[str, Any], family: str) -> List[str]:
    plan = preregistration[
        "primary_execution" if family == PRIMARY_FAMILY else "secondary_execution"
    ]
    hosts = list(plan["required_hosts"])
    if not hosts or any(host not in HOSTS for host in hosts):
        raise CampaignSafetyError("aggregate has invalid required hosts")
    return hosts


def _validate_aggregate_file(
    root: Path,
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
    family: str,
) -> Dict[str, Any]:
    path = _aggregate_path(preregistration, family)
    aggregate = _read_json(path)
    hosts = _required_hosts(preregistration, family)
    expected_cells = list(preregistration["cells"][family])
    exact = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "family": family,
        "preregistration": str(preregistration_path.resolve()),
        "preregistration_sha256": file_sha256(preregistration_path),
        "required_hosts": hosts,
        "expected_cell_count": len(expected_cells),
    }
    for key, expected in exact.items():
        if aggregate.get(key) != expected:
            raise CampaignSafetyError(f"{family} aggregate drifted at {key}")
    reports = aggregate.get("host_reports")
    if not isinstance(reports, dict) or set(reports) != set(hosts):
        raise CampaignSafetyError("aggregate host report set drifted")
    rows: List[Mapping[str, Any]] = []
    identities: Dict[str, Mapping[str, Any]] = {}
    for host in hosts:
        report_path = _host_report_path(preregistration, family, host)
        report = validate_host_report(
            root, preregistration_path, preregistration, family, host
        )
        record = reports[host]
        if record != {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
        }:
            raise CampaignSafetyError("aggregate host report hash drifted")
        identity = _validate_worker_runtime_identity(
            report.get("worker_runtime_identity"), host
        )
        identities[host] = identity
        rows.extend(report["rows"])
    machine_fingerprints, runtime_sha256 = _runtime_set_attestation(identities)
    if aggregate.get("host_machine_fingerprints") != machine_fingerprints:
        raise CampaignSafetyError("aggregate machine fingerprints drifted")
    if aggregate.get("scientific_runtime_sha256") != runtime_sha256:
        raise CampaignSafetyError("aggregate scientific runtime hash drifted")
    observed = {str(row["cell_id"]) for row in rows}
    expected = {str(cell["cell_id"]) for cell in expected_cells}
    if len(rows) != len(expected_cells) or observed != expected:
        raise CampaignSafetyError("aggregate cell coverage is incomplete or duplicate")
    if family == SECONDARY_FAMILY:
        primary_path = _aggregate_path(preregistration, PRIMARY_FAMILY)
        primary = aggregate.get("primary_barrier")
        if primary != {
            "path": str(primary_path),
            "sha256": file_sha256(primary_path),
        }:
            raise CampaignSafetyError("secondary aggregate primary barrier drifted")
        primary_aggregate = _validate_aggregate_file(
            root,
            preregistration_path,
            preregistration,
            PRIMARY_FAMILY,
        )
        _require_runtime_matches_primary(
            primary_aggregate, runtime_sha256, "secondary aggregate"
        )
    return aggregate


def require_primary_barrier(
    root: Path,
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
) -> Dict[str, Any]:
    path = _aggregate_path(preregistration, PRIMARY_FAMILY)
    if not path.is_file():
        mode = preregistration["primary_execution"]["mode"]
        raise CampaignSafetyError(
            "secondary is blocked until the preregistered primary aggregate "
            f"is complete (primary mode={mode})"
        )
    return _validate_aggregate_file(
        root,
        preregistration_path,
        preregistration,
        PRIMARY_FAMILY,
    )


def run_host_worker(
    root: Path,
    preregistration_path: Path,
    family: str,
    host: str,
    jobs: int = 1,
    wait_for_lock: bool = False,
) -> Path:
    root = _root(root)
    validate_jobs(jobs)
    preregistration_path = preregistration_path.expanduser().resolve()
    protocol = load_and_validate_preregistration(root, preregistration_path)
    primary_barrier: Optional[Mapping[str, Any]] = None
    if family == SECONDARY_FAMILY:
        primary_barrier = require_primary_barrier(root, preregistration_path, protocol)
    cells = _cells_for_host(protocol, family, host)
    report_path = _host_report_path(protocol, family, host)
    if report_path.exists():
        report = validate_host_report(
            root, preregistration_path, protocol, family, host
        )
        if primary_barrier is not None:
            _require_runtime_matches_primary(
                primary_barrier,
                _scientific_runtime_sha256(
                    _validate_worker_runtime_identity(
                        report.get("worker_runtime_identity"), host
                    )
                ),
                "secondary worker report",
            )
        return report_path

    lock_path = Path(str(protocol["host_local_global_lock"]))
    started = time.time()
    with HostLocalFcntlLock(lock_path, wait=wait_for_lock) as held:
        if report_path.exists():
            report = validate_host_report(
                root, preregistration_path, protocol, family, host
            )
            if primary_barrier is not None:
                _require_runtime_matches_primary(
                    primary_barrier,
                    _scientific_runtime_sha256(
                        _validate_worker_runtime_identity(
                            report.get("worker_runtime_identity"), host
                        )
                    ),
                    "secondary worker report",
                )
            return report_path
        worker_identity = worker_runtime_identity(host)
        if primary_barrier is not None:
            _require_runtime_matches_primary(
                primary_barrier,
                _scientific_runtime_sha256(worker_identity),
                "secondary worker",
            )
        rows = execute_cell_sequence(
            cells,
            lambda cell: inspect_cell(root, cell),
            lambda cell: _launch_cell(root, cell, worker_identity),
        )
        for row in rows:
            if row["action"] == "executed":
                row["producer_provenance"] = "attested_worker_machine"
                row["producer_machine_fingerprint"] = worker_identity[
                    "machine_fingerprint"
                ]
            else:
                row["producer_provenance"] = "unknown_preexisting_artifact"
                row["producer_machine_fingerprint"] = None
        payload = {
            "status": "complete",
            "protocol_id": PROTOCOL_ID,
            "family": family,
            "host_shard": host,
            "preregistration": str(preregistration_path),
            "preregistration_sha256": file_sha256(preregistration_path),
            "jobs": 1,
            "expected_cell_count": len(cells),
            "executed_cell_count": sum(row["action"] == "executed" for row in rows),
            "validated_skip_cell_count": sum(
                row["action"] == "validated_skip" for row in rows
            ),
            "host_local_global_lock": str(lock_path.resolve()),
            "provenance_policy": dict(PROVENANCE_POLICY),
            "worker_runtime_identity": worker_identity,
            "lock_inode": held.inode,
            "lock_acquired_unix": held.acquired_unix,
            "waited_for_lock": bool(wait_for_lock),
            "started_unix": started,
            "finished_unix": time.time(),
            "thread_environment": dict(THREAD_ENVIRONMENT),
            "rows": rows,
        }
        _write_json_exclusive(report_path, payload)
    validate_host_report(root, preregistration_path, protocol, family, host)
    return report_path


def aggregate_family(root: Path, preregistration_path: Path, family: str) -> Path:
    root = _root(root)
    preregistration_path = preregistration_path.expanduser().resolve()
    protocol = load_and_validate_preregistration(root, preregistration_path)
    primary_barrier: Optional[Mapping[str, Any]] = None
    if family == SECONDARY_FAMILY:
        primary_barrier = require_primary_barrier(root, preregistration_path, protocol)
    hosts = _required_hosts(protocol, family)
    expected_cells = protocol["cells"][family]
    host_records: Dict[str, Dict[str, str]] = {}
    all_ids: List[str] = []
    identities: Dict[str, Mapping[str, Any]] = {}
    for host in hosts:
        report_path = _host_report_path(protocol, family, host)
        report = validate_host_report(
            root, preregistration_path, protocol, family, host
        )
        host_records[host] = {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
        }
        identity = _validate_worker_runtime_identity(
            report.get("worker_runtime_identity"), host
        )
        identities[host] = identity
        all_ids.extend(str(row["cell_id"]) for row in report["rows"])
    machine_fingerprints, runtime_sha256 = _runtime_set_attestation(identities)
    if primary_barrier is not None:
        _require_runtime_matches_primary(
            primary_barrier, runtime_sha256, "secondary aggregate"
        )
    expected_ids = {str(cell["cell_id"]) for cell in expected_cells}
    if len(all_ids) != len(expected_cells) or set(all_ids) != expected_ids:
        raise CampaignSafetyError("cannot aggregate incomplete/duplicate cell coverage")
    path = _aggregate_path(protocol, family)
    if path.exists():
        _validate_aggregate_file(root, preregistration_path, protocol, family)
        return path
    payload: Dict[str, Any] = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "family": family,
        "preregistration": str(preregistration_path),
        "preregistration_sha256": file_sha256(preregistration_path),
        "required_hosts": hosts,
        "expected_cell_count": len(expected_cells),
        "unique_cell_count": len(set(all_ids)),
        "host_reports": host_records,
        "host_machine_fingerprints": machine_fingerprints,
        "scientific_runtime_sha256": runtime_sha256,
        "completed_unix": time.time(),
    }
    if family == SECONDARY_FAMILY:
        primary_path = _aggregate_path(protocol, PRIMARY_FAMILY)
        payload["primary_barrier"] = {
            "path": str(primary_path),
            "sha256": file_sha256(primary_path),
        }
    _write_json_exclusive(path, payload)
    _validate_aggregate_file(root, preregistration_path, protocol, family)
    return path


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    prereg = subparsers.add_parser("preregister")
    prereg.add_argument("--root", type=Path, required=True)
    prereg.add_argument("--primary-single-host-override", choices=HOSTS, default=None)
    prereg.add_argument("--secondary-single-host-override", choices=HOSTS, default=None)

    worker = subparsers.add_parser("run-shard")
    worker.add_argument("--root", type=Path, required=True)
    worker.add_argument("--preregistration", type=Path, required=True)
    worker.add_argument("--family", choices=FAMILIES, required=True)
    worker.add_argument("--host", choices=HOSTS, required=True)
    worker.add_argument("--jobs", type=int, default=1)
    worker.add_argument("--wait-for-lock", action="store_true")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--root", type=Path, required=True)
    aggregate.add_argument("--preregistration", type=Path, required=True)
    aggregate.add_argument("--family", choices=FAMILIES, required=True)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    if args.action == "preregister":
        path = preregister(
            args.root,
            primary_single_host_override=args.primary_single_host_override,
            secondary_single_host_override=args.secondary_single_host_override,
        )
    elif args.action == "run-shard":
        path = run_host_worker(
            args.root,
            args.preregistration,
            args.family,
            args.host,
            jobs=args.jobs,
            wait_for_lock=args.wait_for_lock,
        )
    elif args.action == "aggregate":
        path = aggregate_family(args.root, args.preregistration, args.family)
    else:
        raise AssertionError("unreachable action")
    print(path)


if __name__ == "__main__":
    main()
