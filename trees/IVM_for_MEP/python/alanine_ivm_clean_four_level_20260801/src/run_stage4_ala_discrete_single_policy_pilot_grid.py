#!/usr/bin/env python3
"""Preregister and run the six-cell alpha20/N30 single-policy pilot."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_stage4_ala_discrete_single_policy_pilot as pilot
import run_stage4_uniform_optimized as legacy


PROTOCOL_ID = "stage4_ala_discrete_single_policy_pilot_grid_v1"


def _run_process(command, root, stdout_path, stderr_path):
    environment = os.environ.copy()
    environment.update(
        {
            "OPENMM_CPU_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return completed.returncode, time.time() - started, completed.stderr


def _run_cell(root, policy, solver_role, replicate):
    label = pilot.pilot_label(policy, solver_role, replicate)
    audit_path = (
        root
        / "runs"
        / pilot.POLICY_VERSION
        / "audits"
        / f"{label}.json"
    )
    result_dir = root / "runs" / pilot.POLICY_VERSION / "members"
    existing = sorted(
        result_dir.glob(
            f"ivm_{label}_central_state_ivm_cfg*_budget{pilot.PILOT_BUDGET}.npz"
        )
    )
    if audit_path.exists() or existing:
        target = existing[0] if existing else audit_path
        raise FileExistsError(f"Refusing to overwrite {target}")
    log_dir = root / "logs" / pilot.POLICY_VERSION / "grid" / label
    log_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(root / "src/run_stage4_ala_discrete_single_policy_pilot.py"),
        "--root", str(root),
        "--policy", policy,
        "--solver-role", solver_role,
        "--replicate", replicate,
    ]
    (log_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    returncode, wall_seconds, stderr = _run_process(
        command,
        root,
        log_dir / "stdout.txt",
        log_dir / "stderr.txt",
    )
    row = {
        "policy": policy,
        "solver_role": solver_role,
        "alpha": pilot.PILOT_ALPHA,
        "budget_per_image": pilot.PILOT_BUDGET,
        "returncode": returncode,
        "wall_seconds": wall_seconds,
    }
    if returncode:
        row["error"] = stderr[-4000:]
        return row
    if not audit_path.is_file():
        raise RuntimeError("Pilot member completed without immutable audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    row.update(
        {
            "audit": str(audit_path),
            "audit_sha256": legacy.file_sha256(audit_path),
            "result": audit["result"],
            "result_sha256": audit["result_sha256"],
            "complete_states": audit["complete_states"],
            "mesh_eligible_complete_state_count": audit[
                "mesh_eligible_complete_state_count"
            ],
            "best_mesh_eligible_complete_i_perp": audit[
                "best_mesh_eligible_complete_i_perp"
            ],
            "best_mesh_eligible_complete_NF": audit[
                "best_mesh_eligible_complete_NF"
            ],
            "terminal_i_perp_diagnostic_only": audit[
                "terminal_i_perp_diagnostic_only"
            ],
        }
    )
    return row


def _write_json_exclusive(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--replicate", default="r1")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    if not args.replicate.startswith("r") or not args.replicate[1:].isdigit():
        raise ValueError("replicate must be r followed by digits")

    registered_commands = {}
    for policy in pilot.POLICIES:
        commands = {}
        for role in pilot.SOLVER_ROLES:
            command, _ = pilot.pilot_command(root, policy, role, args.replicate)
            commands[role] = command
        pilot.assert_policy_triple(commands, policy, pilot.PILOT_BUDGET)
        registered_commands[policy] = commands

    source_paths = {
        "core": root / "src/run_ivm_v9_spring_projected_production_20260730.py",
        "ala_core": root / "src/ala_core.py",
        "frozen_discrete_b040": root / "src/frozen_discrete_b040.py",
        "legacy_command_helper": root / "src/run_stage4_uniform_optimized.py",
        "single_protocol_helper": root / "src/run_stage4_ala_dsel_single_fixedalpha.py",
        "pilot_wrapper": root / "src/run_stage4_ala_discrete_single_policy_pilot.py",
        "pilot_launcher": Path(__file__).resolve(),
    }
    common = root / "inputs/common_initial_path_equidistant.npz"
    for path in (*source_paths.values(), common):
        if not path.is_file():
            raise FileNotFoundError(path)

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    unique = time.time_ns()
    stem = f"{PROTOCOL_ID}_a20_N30_{args.replicate}_{unique}"
    preregistration = reports / f"{stem}_preregistered.json"
    final_report = reports / f"{stem}.json"
    protocol = {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "pilot_only_not_formal_alpha_grid": True,
        "alpha": pilot.PILOT_ALPHA,
        "alpha_mode": "fixed",
        "budget_per_image": pilot.PILOT_BUDGET,
        "policies": list(pilot.POLICIES),
        "solver_roles": list(pilot.SOLVER_ROLES),
        "primary_requested_solver": "discrete_b040",
        "ala_dsel_role": "ablation_only",
        "one_same_policy_legacy_control_per_solver_policy": True,
        "matched_estimand": (
            "whole solver plus R08C10 package versus same-policy legacy"
        ),
        "command_audit": (
            "within each policy, commands are exact after removing run label "
            "and registered solver-package options"
        ),
        "complete_state_definition_by_policy": {
            "precision_state_curve": "initial_plus_reparam_committed",
            "projected_pathsum": "all_published_projected_states",
        },
        "mesh_gate": {
            "segment_length_cv_max": pilot.MESH_CV_MAX,
            "segment_length_ratio_max": pilot.MESH_RATIO_MAX,
        },
        "comparison_rule": (
            "complete-state I_perp cost-to-precision under mesh gate; never "
            "raw terminal I_perp"
        ),
        "sparse_barrier_role": "diagnostic_only_not_selection",
        "registered_commands": registered_commands,
        "common_initial_path": str(common),
        "common_initial_path_sha256": legacy.file_sha256(common),
        "source_sha256": {
            name: legacy.file_sha256(path)
            for name, path in source_paths.items()
        },
        "jobs": args.jobs,
        "replicate": args.replicate,
        "created_unix": time.time(),
    }
    _write_json_exclusive(preregistration, protocol)

    cells = [
        (policy, role)
        for policy in pilot.POLICIES
        for role in pilot.SOLVER_ROLES
    ]
    rows = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(_run_cell, root, policy, role, args.replicate): (
                policy,
                role,
            )
            for policy, role in cells
        }
        for future in concurrent.futures.as_completed(futures):
            policy, role = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "policy": policy,
                    "solver_role": role,
                    "alpha": pilot.PILOT_ALPHA,
                    "budget_per_image": pilot.PILOT_BUDGET,
                    "returncode": -1,
                    "error": repr(exc),
                }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    policy_order = {name: i for i, name in enumerate(pilot.POLICIES)}
    role_order = {name: i for i, name in enumerate(pilot.SOLVER_ROLES)}
    rows.sort(
        key=lambda row: (
            policy_order[row["policy"]], role_order[row["solver_role"]]
        )
    )
    successful = [row for row in rows if row.get("returncode") == 0]
    report = {
        **protocol,
        "status": "complete" if len(successful) == len(rows) else "incomplete",
        "started_unix": started,
        "finished_unix": time.time(),
        "preregistration": str(preregistration),
        "preregistration_sha256": legacy.file_sha256(preregistration),
        "formal_policy_choice_deferred": True,
        "results": rows,
    }
    _write_json_exclusive(final_report, report)
    print(final_report)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
