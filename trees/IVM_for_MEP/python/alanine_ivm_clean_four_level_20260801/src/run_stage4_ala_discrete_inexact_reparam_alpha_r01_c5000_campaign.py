#!/usr/bin/env python3
"""Preregister, run, and validate the local serial Ala 8-alpha campaign.

This orchestrator never imports OpenMM.  It preregisters eight decade alphas,
pins every source/input hash, requires one local serial worker under the
host-local Ala fcntl lock, and accepts only complete exclusive member outputs
whose c5000 implementation safety cap never triggered.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import run_stage4_ala_discrete_grid_campaign as frozen_campaign
import run_stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_member as member


PROTOCOL_ID = "stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign_v1"
CANONICAL_ROOT_NAME = frozen_campaign.CANONICAL_ROOT_NAME
FIXED_PYTHON = frozen_campaign.FIXED_PYTHON
DEFAULT_GLOBAL_LOCK = frozen_campaign.DEFAULT_GLOBAL_LOCK
ALPHAS = member.ALPHAS
TOTAL_INTERNAL_NF_BUDGET = 5000
RESIDUAL_RATIO = 0.1
INNER_MAX_STEPS = 5000
CELL_COUNT = 8
HOST = "local"
JOBS = 1
THREAD_ENVIRONMENT = dict(frozen_campaign.THREAD_ENVIRONMENT)

SOURCE_PATHS = {
    "campaign": "src/run_stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign.py",
    "member": "src/run_stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_member.py",
    "runtime_wrapper": "src/run_ivm_v9_discrete_ffrbk_takeover_alpha_r01_c5000_shim.py",
    "frozen_grid_campaign_helpers": "src/run_stage4_ala_discrete_grid_campaign.py",
    "frozen_grid_member_validator": "src/run_stage4_ala_discrete_inexact_reparam_grid_member.py",
    "frozen_grid_runtime": "src/run_ivm_v9_discrete_ffrbk_takeover_grid_shim.py",
    "total_internal_nf_runtime": "src/run_ivm_v9_total_internal_nf_budget.py",
    "frozen_selector_runtime": "src/run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py",
    "eta_runtime": "src/run_ivm_v9_discrete_selected_eta_shim.py",
    "frozen_discrete_solver": "src/frozen_discrete_b040.py",
    "production_runner": "src/run_ivm_v9_spring_projected_production_20260730.py",
    "command_helper": "src/run_stage4_uniform_optimized.py",
    "ala_core": "src/ala_core.py",
    "canonical_command": member.CANONICAL_COMMAND,
    "common_initial_path": "inputs/common_initial_path_equidistant.npz",
    "topology": "inputs/ala2_fixed.pdb",
}


class CampaignSafetyError(RuntimeError):
    """Fail-closed preregistration, execution, or artifact error."""


def root_path(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root.name != CANONICAL_ROOT_NAME:
        raise CampaignSafetyError(
            f"campaign root must end in {CANONICAL_ROOT_NAME}: {root}"
        )
    return root


def file_sha256(path: Path) -> str:
    return frozen_campaign.file_sha256(path)


def payload_sha256(payload: Any) -> str:
    return frozen_campaign.payload_sha256(payload)


def command_sha256(command: Sequence[str]) -> str:
    return frozen_campaign.command_sha256(command)


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    frozen_campaign._write_json_exclusive(path, payload)


def read_json(path: Path) -> Dict[str, Any]:
    return frozen_campaign._read_json(path)


def alpha_text(alpha: float) -> str:
    exponent = int(round(math.log10(float(alpha))))
    return "1e" + str(exponent)


def source_manifest(root: Path) -> Dict[str, Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    for name, relative in SOURCE_PATHS.items():
        path = (root / relative).resolve()
        if not path.is_file():
            raise CampaignSafetyError(f"missing source/input: {path}")
        records[name] = {"path": str(path), "sha256": file_sha256(path)}
    return records


def campaign_stem(preregistration_path: Path) -> str:
    suffix = "_preregistered.json"
    if not preregistration_path.name.endswith(suffix):
        raise CampaignSafetyError("preregistration filename has the wrong suffix")
    return preregistration_path.name[: -len(suffix)]


def report_path(root: Path, stem: str) -> Path:
    return (root / "reports" / f"{stem}_local_report.json").resolve()


def aggregate_path(root: Path, stem: str) -> Path:
    return (root / "reports" / f"{stem}_aggregate.json").resolve()


def build_cells(root: Path, stem: str) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    member_script = (
        root / "src/run_stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_member.py"
    ).resolve()
    for ordinal, alpha in enumerate(ALPHAS):
        scientific_command, label = member.single_run_command(root, alpha)
        paths = member.member_paths(root, alpha)
        identity = {
            "protocol_id": PROTOCOL_ID,
            "family": "inexact_reparam",
            "alpha": float(alpha),
            "residual_ratio": RESIDUAL_RATIO,
            "inner_max_steps": INNER_MAX_STEPS,
            "total_internal_nf_budget": TOTAL_INTERNAL_NF_BUDGET,
            "replicate": member.CANONICAL_REPLICATE,
        }
        cell_id = payload_sha256(identity)[:16]
        member_command = [
            FIXED_PYTHON,
            str(member_script),
            "--root",
            str(root),
            "--alpha",
            f"{alpha:g}",
        ]
        cells.append(
            {
                "ordinal": ordinal,
                "cell_id": cell_id,
                "identity": identity,
                "alpha_text": alpha_text(alpha),
                "controls": {
                    "alpha": float(alpha),
                    "residual_ratio": RESIDUAL_RATIO,
                    "inner_max_steps": INNER_MAX_STEPS,
                    "total_internal_nf_budget": TOTAL_INTERNAL_NF_BUDGET,
                },
                "assigned_host": HOST,
                "execution_order": ordinal,
                "run_label": label,
                "member_command": member_command,
                "member_command_sha256": command_sha256(member_command),
                "scientific_command": scientific_command,
                "scientific_command_sha256": command_sha256(scientific_command),
                "paths": {
                    "result_dir": str(paths["result_dir"].resolve()),
                    "result_glob": paths["result_glob"],
                    "companion_glob": paths["companion_glob"],
                    "audit": str(paths["audit"].resolve()),
                    "member_log": str(paths["member_log"].resolve()),
                    "worker_log": str(
                        (
                            root
                            / "logs"
                            / PROTOCOL_ID
                            / stem
                            / HOST
                            / cell_id
                        ).resolve()
                    ),
                },
            }
        )
    if len(cells) != CELL_COUNT or len({cell["cell_id"] for cell in cells}) != CELL_COUNT:
        raise CampaignSafetyError("alpha cell coverage is not exactly eight unique cells")
    return cells


def build_preregistration(root: Path, path: Path) -> Dict[str, Any]:
    root = root_path(root)
    path = path.expanduser().resolve()
    stem = campaign_stem(path)
    sources = source_manifest(root)
    scientific_sources = {
        key: value
        for key, value in sources.items()
        if key not in {"campaign", "frozen_grid_campaign_helpers"}
    }
    return {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "campaign_stem": stem,
        "preregistration": str(path),
        "created_unix": time.time(),
        "canonical_root": str(root),
        "family": "inexact_reparam",
        "alpha_grid": [alpha_text(alpha) for alpha in ALPHAS],
        "cell_count": CELL_COUNT,
        "fixed_controls": {
            "residual_ratio": RESIDUAL_RATIO,
            "inner_max_steps": INNER_MAX_STEPS,
            "inner_max_steps_role": "implementation_safety_cap",
            "inner_cap_acceptance_gate": "inner_cap_stop_count == 0 for every member",
            "total_internal_nf_budget": TOTAL_INTERNAL_NF_BUDGET,
            "eta": 1.1,
            "physical_max_step_angstrom": 1.0,
            "kkt_atol": 0.0,
            "lbfgs_history": 7,
            "selector": "one_way_takeover FFrbk/L-BFGS",
            "reparameterization": "state_curve/per_image every outer proposal",
            "complete_state_definition": "initial_or_reparam_committed",
        },
        "cap_policy": {
            "reason_for_c5000": (
                "reuse the formal exact runtime implementation safety cap; "
                "do not assume a fixed 20-new-NF cost per joint step under recovery/cache"
            ),
            "post_run_requirement": "zero evaluated-inner-max-steps stop records",
            "failure_policy": "any cap-stop makes the member and campaign invalid",
        },
        "execution": {
            "host": HOST,
            "jobs": JOBS,
            "scheduling": "single local worker, fixed alpha order, serial",
            "host_local_fcntl_lock": str(Path(DEFAULT_GLOBAL_LOCK).resolve()),
            "lock_held_for_entire_campaign": True,
            "thread_environment": THREAD_ENVIRONMENT,
            "remote_execution": False,
        },
        "artifact_policy": {
            "overwrite_allowed": False,
            "creation": "exclusive",
            "restart": "validated complete members may be skipped; partial members fail closed",
            "optimization_nf_field": "result.total_internal_joint_evaluations",
            "diagnostic_metrics_charged_to_optimization_nf": False,
        },
        "source_manifest": sources,
        "scientific_source_manifest_sha256": payload_sha256(scientific_sources),
        "cells": build_cells(root, stem),
        "planned_outputs": {
            "local_report": str(report_path(root, stem)),
            "aggregate": str(aggregate_path(root, stem)),
        },
    }


def validate_preregistration_payload(
    root: Path, path: Path, payload: Mapping[str, Any]
) -> Dict[str, Any]:
    root = root_path(root)
    path = path.expanduser().resolve()
    stem = campaign_stem(path)
    if payload.get("status") != "preregistered" or payload.get("protocol_id") != PROTOCOL_ID:
        raise CampaignSafetyError("preregistration identity drifted")
    if payload.get("campaign_stem") != stem or payload.get("preregistration") != str(path):
        raise CampaignSafetyError("preregistration path/stem drifted")
    if payload.get("canonical_root") != str(root):
        raise CampaignSafetyError("canonical root drifted")
    if payload.get("alpha_grid") != [alpha_text(alpha) for alpha in ALPHAS]:
        raise CampaignSafetyError("alpha grid drifted")
    if int(payload.get("cell_count", -1)) != CELL_COUNT:
        raise CampaignSafetyError("cell count drifted")
    controls = payload.get("fixed_controls")
    if not isinstance(controls, dict) or (
        controls.get("residual_ratio") != RESIDUAL_RATIO
        or controls.get("inner_max_steps") != INNER_MAX_STEPS
        or controls.get("total_internal_nf_budget") != TOTAL_INTERNAL_NF_BUDGET
        or controls.get("reparameterization")
        != "state_curve/per_image every outer proposal"
        or controls.get("inner_cap_acceptance_gate")
        != "inner_cap_stop_count == 0 for every member"
    ):
        raise CampaignSafetyError("fixed scientific controls drifted")
    execution = payload.get("execution")
    if not isinstance(execution, dict) or execution.get("host") != HOST or execution.get("jobs") != 1:
        raise CampaignSafetyError("local serial execution policy drifted")
    if Path(str(execution.get("host_local_fcntl_lock", ""))).resolve() != Path(DEFAULT_GLOBAL_LOCK).resolve():
        raise CampaignSafetyError("host-local Ala lock path drifted")
    observed_sources = payload.get("source_manifest")
    expected_sources = source_manifest(root)
    if observed_sources != expected_sources:
        raise CampaignSafetyError("registered source/input hashes drifted")
    scientific_sources = {
        key: value
        for key, value in expected_sources.items()
        if key not in {"campaign", "frozen_grid_campaign_helpers"}
    }
    if payload.get("scientific_source_manifest_sha256") != payload_sha256(scientific_sources):
        raise CampaignSafetyError("scientific source-manifest hash drifted")
    if payload.get("cells") != build_cells(root, stem):
        raise CampaignSafetyError("registered cells or commands drifted")
    expected_outputs = {
        "local_report": str(report_path(root, stem)),
        "aggregate": str(aggregate_path(root, stem)),
    }
    if payload.get("planned_outputs") != expected_outputs:
        raise CampaignSafetyError("planned output routing drifted")
    return dict(payload)


def load_preregistration(root: Path, path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise CampaignSafetyError(f"missing preregistration: {path}")
    return validate_preregistration_payload(root, path, read_json(path))


def preregister(root: Path, dry_run: bool = False) -> Path:
    root = root_path(root)
    stem = f"{PROTOCOL_ID}_{time.time_ns()}"
    path = (root / "reports" / f"{stem}_preregistered.json").resolve()
    payload = build_preregistration(root, path)
    validate_preregistration_payload(root, path, payload)
    if dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_preregistration_valid",
                    "protocol_id": PROTOCOL_ID,
                    "would_write": str(path),
                    "cell_count": CELL_COUNT,
                    "alphas": payload["alpha_grid"],
                    "openmm_started": False,
                },
                indent=2,
            )
        )
        return path
    write_json_exclusive(path, payload)
    return path


def cell_has_any_artifact(cell: Mapping[str, Any]) -> bool:
    paths = cell["paths"]
    result_dir = Path(str(paths["result_dir"]))
    return bool(
        list(result_dir.glob(str(paths["result_glob"])))
        or list(result_dir.glob(str(paths["companion_glob"])))
        or Path(str(paths["audit"])).exists()
        or Path(str(paths["member_log"])).exists()
        or Path(str(paths["worker_log"])).exists()
    )


def validate_cell(root: Path, cell: Mapping[str, Any]) -> Dict[str, Any]:
    controls = cell["controls"]
    record = member.validate_existing(root, controls["alpha"])
    audit = read_json(Path(str(record["member_audit"])))
    if audit.get("command") != cell["scientific_command"]:
        raise CampaignSafetyError("member command differs from preregistration")
    if audit.get("scientific_command_sha256") != cell["scientific_command_sha256"]:
        raise CampaignSafetyError("member command hash differs from preregistration")
    if record.get("inner_cap_stop_count") != 0:
        raise CampaignSafetyError("c5000 bound a registered member")
    return {
        "cell_id": cell["cell_id"],
        "alpha": float(controls["alpha"]),
        "result": record["result"],
        "result_sha256": record["result_sha256"],
        "member_audit": record["member_audit"],
        "member_audit_sha256": record["member_audit_sha256"],
        "companion_audit": record["companion_audit"],
        "companion_audit_sha256": record["companion_audit_sha256"],
        "configuration_sha256": record["configuration_sha256"],
        "final_optimization_NF": record["final_optimization_NF"],
        "last_complete_state_NF": record["last_complete_state_NF"],
        "trajectory_record_count": record["trajectory_record_count"],
        "inner_cap_stop_count": 0,
        "scientific_command_sha256": cell["scientific_command_sha256"],
    }


def validate_report(
    root: Path, preregistration_path: Path, protocol: Mapping[str, Any]
) -> Dict[str, Any]:
    root = root_path(root)
    path = Path(str(protocol["planned_outputs"]["local_report"]))
    if not path.is_file():
        raise CampaignSafetyError(f"local report does not exist: {path}")
    payload = read_json(path)
    if (
        payload.get("status") != "complete"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("preregistration") != str(preregistration_path.resolve())
        or payload.get("preregistration_sha256") != file_sha256(preregistration_path)
        or payload.get("jobs") != 1
        or payload.get("host") != HOST
    ):
        raise CampaignSafetyError("local report identity drifted")
    rows = payload.get("rows")
    cells = protocol["cells"]
    if not isinstance(rows, list) or len(rows) != CELL_COUNT:
        raise CampaignSafetyError("local report does not cover eight cells")
    expected_by_id = {cell["cell_id"]: cell for cell in cells}
    if {row.get("cell_id") for row in rows} != set(expected_by_id):
        raise CampaignSafetyError("local report cell coverage drifted")
    for row in rows:
        expected = validate_cell(root, expected_by_id[row["cell_id"]])
        action = row.get("action")
        observed = {key: value for key, value in row.items() if key != "action"}
        if action not in {"executed", "validated_skip"} or observed != expected:
            raise CampaignSafetyError("local report row drifted")
    return payload


def write_aggregate(
    root: Path, preregistration_path: Path, protocol: Mapping[str, Any]
) -> Path:
    root = root_path(root)
    report = validate_report(root, preregistration_path, protocol)
    path = Path(str(protocol["planned_outputs"]["aggregate"]))
    if path.exists():
        payload = read_json(path)
        if payload.get("local_report_sha256") != file_sha256(
            Path(str(protocol["planned_outputs"]["local_report"]))
        ):
            raise CampaignSafetyError("existing aggregate report hash drifted")
        return path
    payload = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "preregistration": str(preregistration_path.resolve()),
        "preregistration_sha256": file_sha256(preregistration_path),
        "cell_count": CELL_COUNT,
        "unique_cell_count": len({row["cell_id"] for row in report["rows"]}),
        "all_inner_caps_nonbinding": all(
            row["inner_cap_stop_count"] == 0 for row in report["rows"]
        ),
        "local_report": str(Path(str(protocol["planned_outputs"]["local_report"]))),
        "local_report_sha256": file_sha256(
            Path(str(protocol["planned_outputs"]["local_report"]))
        ),
        "completed_unix": time.time(),
    }
    write_json_exclusive(path, payload)
    return path


def run_campaign(
    root: Path,
    preregistration_path: Path,
    jobs: int = 1,
    wait_for_lock: bool = False,
) -> Path:
    root = root_path(root)
    if int(jobs) != JOBS:
        raise CampaignSafetyError("this campaign requires --jobs 1")
    expected_python = Path(FIXED_PYTHON).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise CampaignSafetyError(f"campaign must run under {expected_python}")
    preregistration_path = preregistration_path.expanduser().resolve()
    protocol = load_preregistration(root, preregistration_path)
    local_report = Path(str(protocol["planned_outputs"]["local_report"]))
    if local_report.exists():
        validate_report(root, preregistration_path, protocol)
        return write_aggregate(root, preregistration_path, protocol)
    rows: List[Dict[str, Any]] = []
    started = time.time()
    lock_path = Path(str(protocol["execution"]["host_local_fcntl_lock"]))
    with frozen_campaign.HostLocalFcntlLock(lock_path, wait=wait_for_lock) as held:
        worker_identity = frozen_campaign.worker_runtime_identity(HOST)
        for cell in protocol["cells"]:
            try:
                record = validate_cell(root, cell)
                rows.append({"action": "validated_skip", **record})
                continue
            except (FileNotFoundError, member.MemberSafetyError):
                if cell_has_any_artifact(cell):
                    raise CampaignSafetyError(
                        f"partial or invalid member blocks restart: {cell['cell_id']}"
                    )
            worker_log = Path(str(cell["paths"]["worker_log"]))
            worker_log.mkdir(parents=True, exist_ok=False)
            write_json_exclusive(
                worker_log / "command.json",
                {"command": cell["member_command"], "sha256": cell["member_command_sha256"]},
            )
            environment = os.environ.copy()
            environment.update(THREAD_ENVIRONMENT)
            with (worker_log / "stdout.txt").open("x", encoding="utf-8") as stdout, (
                worker_log / "stderr.txt"
            ).open("x", encoding="utf-8") as stderr:
                completed = subprocess.run(
                    cell["member_command"],
                    cwd=root,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            write_json_exclusive(
                worker_log / "exit.json",
                {"returncode": int(completed.returncode), "finished_unix": time.time()},
            )
            if completed.returncode != 0:
                raise CampaignSafetyError(
                    f"member failed with exit {completed.returncode}: {cell['cell_id']}"
                )
            rows.append({"action": "executed", **validate_cell(root, cell)})
        report = {
            "status": "complete",
            "protocol_id": PROTOCOL_ID,
            "host": HOST,
            "jobs": JOBS,
            "preregistration": str(preregistration_path),
            "preregistration_sha256": file_sha256(preregistration_path),
            "host_local_fcntl_lock": str(lock_path.resolve()),
            "lock_inode": held.inode,
            "lock_acquired_unix": held.acquired_unix,
            "waited_for_lock": bool(wait_for_lock),
            "worker_runtime_identity": worker_identity,
            "started_unix": started,
            "finished_unix": time.time(),
            "rows": rows,
        }
        write_json_exclusive(local_report, report)
    validate_report(root, preregistration_path, protocol)
    return write_aggregate(root, preregistration_path, protocol)


def validate_campaign(root: Path, preregistration_path: Path) -> Path:
    root = root_path(root)
    preregistration_path = preregistration_path.expanduser().resolve()
    protocol = load_preregistration(root, preregistration_path)
    validate_report(root, preregistration_path, protocol)
    path = write_aggregate(root, preregistration_path, protocol)
    aggregate = read_json(path)
    if (
        aggregate.get("status") != "complete"
        or aggregate.get("cell_count") != CELL_COUNT
        or aggregate.get("unique_cell_count") != CELL_COUNT
        or aggregate.get("all_inner_caps_nonbinding") is not True
    ):
        raise CampaignSafetyError("aggregate validation failed")
    return path


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    prereg = subparsers.add_parser("preregister")
    prereg.add_argument("--root", type=Path, required=True)
    prereg.add_argument("--dry-run", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--preregistration", type=Path, required=True)
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--wait-for-lock", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--preregistration", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = argument_parser().parse_args(argv)
    if args.action == "preregister":
        path = preregister(args.root, dry_run=args.dry_run)
    elif args.action == "run":
        path = run_campaign(
            args.root,
            args.preregistration,
            jobs=args.jobs,
            wait_for_lock=args.wait_for_lock,
        )
    elif args.action == "validate":
        path = validate_campaign(args.root, args.preregistration)
    else:
        raise AssertionError("unreachable action")
    print(path)


if __name__ == "__main__":
    main()
