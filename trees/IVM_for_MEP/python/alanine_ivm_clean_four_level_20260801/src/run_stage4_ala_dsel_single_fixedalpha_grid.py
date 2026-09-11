#!/usr/bin/env python3
"""Preregistered standalone discrete-coordinate Ala matched-control grid."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import run_stage4_ala_discrete_single_fixedalpha as single
import run_stage4_uniform_optimized as legacy


PROTOCOL_ID = "stage4_ala_discrete_single_fixedalpha_matched_v2"
CANONICAL_ALPHAS = (
    0.01,
    0.1,
    1.0,
    10.0,
    20.0,
    30.0,
    100.0,
    1000.0,
    10000.0,
    math.inf,
)
CANONICAL_BUDGET = 254
MEMBER_ROLES = ("candidate", "matched_control")
REPORTING_ANCHORS = (
    1.0,
    0.1,
    0.05,
    0.02,
    0.01,
    0.005,
    0.002,
    0.001,
    5e-4,
    2e-4,
    1.5e-4,
)
OFFICIAL_STAGE4_V4 = (
    "runs/pareto_archive/"
    "pareto_clean_stage4_v4_a20_phaseA34_phaseB220.npz"
)


def _alpha_text(alpha: float) -> str:
    return "inf" if math.isinf(alpha) else f"{alpha:g}"


def parse_alphas(text: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        value = math.inf if item in {"inf", "infinity"} else float(item)
        if not value > 0.0:
            raise ValueError("Every alpha must be positive or inf")
        values.append(value)
    if not values:
        raise ValueError("The alpha grid cannot be empty")
    if len(set(values)) != len(values):
        raise ValueError("The alpha grid contains duplicates")
    return tuple(values)


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _run_process(
    command: list[str],
    root: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, float, str]:
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


def _run_cell(
    root: Path,
    alpha: float,
    budget: int,
    replicate: str,
    member_role: str,
) -> dict[str, object]:
    label = single.run_label(alpha, budget, replicate, member_role)
    member_audit = (
        root
        / "runs"
        / single.POLICY_VERSION
        / "audits"
        / f"{label}.json"
    )
    result_dir = root / "runs" / single.POLICY_VERSION / "members"
    existing = sorted(
        result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget{budget}.npz")
    )
    if member_audit.exists() or existing:
        target = existing[0] if existing else member_audit
        raise FileExistsError(f"Refusing to overwrite {target}")

    grid_log = (
        root
        / "logs"
        / single.POLICY_VERSION
        / "grid"
        / label
    )
    grid_log.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(root / "src/run_stage4_ala_discrete_single_fixedalpha.py"),
        "--root", str(root),
        "--alpha", _alpha_text(alpha),
        "--budget", str(budget),
        "--member-role", member_role,
        "--replicate", replicate,
    ]
    (grid_log / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    returncode, wall_seconds, stderr = _run_process(
        command,
        root,
        grid_log / "stdout.txt",
        grid_log / "stderr.txt",
    )
    row: dict[str, object] = {
        "alpha": "inf" if math.isinf(alpha) else alpha,
        "member_role": member_role,
        "budget_per_image": budget,
        "returncode": returncode,
        "wall_seconds": wall_seconds,
        "replicate": replicate,
    }
    if returncode:
        row["error"] = stderr[-4000:]
        return row
    if not member_audit.is_file():
        raise RuntimeError("Member completed without its immutable audit")
    audit = json.loads(member_audit.read_text(encoding="utf-8"))
    if audit["member_role"] != member_role:
        raise RuntimeError("Member audit role mismatch")
    if float(audit["alpha"]) != float(alpha):
        raise RuntimeError("Member audit alpha mismatch")
    if int(audit["configured_budget_per_image"]) != budget:
        raise RuntimeError("Member audit budget mismatch")
    if audit["alpha_mode"] != "fixed" or not audit["fixed_alpha_throughout"]:
        raise RuntimeError("Member failed the fixed-alpha contract")
    row.update(
        {
            "member_audit": str(member_audit),
            "member_audit_sha256": legacy.file_sha256(member_audit),
            "result": audit["result"],
            "result_sha256": audit["result_sha256"],
            "terminal_NF": audit["terminal_NF"],
            "terminal_i_perp": audit["terminal_i_perp"],
            "minimum_i_perp": audit["minimum_i_perp"],
            "terminal_sparse_barrier_kcal": audit[
                "terminal_sparse_barrier_kcal"
            ],
            "terminal_physical_fmax": audit["terminal_physical_fmax"],
            "terminal_segment_length_cv": audit[
                "terminal_segment_length_cv"
            ],
            "terminal_segment_length_ratio": audit[
                "terminal_segment_length_ratio"
            ],
            "h0_selection_count": audit["h0_selection_count"],
            "lbfgs_selection_count": audit["lbfgs_selection_count"],
            "safeguard_selection_count": audit["safeguard_selection_count"],
            "complete_candidate_state_count": audit[
                "complete_candidate_state_count"
            ],
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "alpha",
        "member_role",
        "budget_per_image",
        "terminal_NF",
        "terminal_i_perp",
        "minimum_i_perp",
        "terminal_sparse_barrier_kcal",
        "terminal_physical_fmax",
        "terminal_segment_length_cv",
        "terminal_segment_length_ratio",
        "h0_selection_count",
        "lbfgs_selection_count",
        "safeguard_selection_count",
        "complete_candidate_state_count",
        "returncode",
        "wall_seconds",
        "result",
        "result_sha256",
    )
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--alphas",
        default=",".join(_alpha_text(value) for value in CANONICAL_ALPHAS),
    )
    parser.add_argument("--budget", type=int, default=CANONICAL_BUDGET)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--replicate", default="r1")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    alphas = parse_alphas(args.alphas)
    if alphas != CANONICAL_ALPHAS:
        raise ValueError(
            "Formal grid requires canonical alpha order: "
            + ",".join(_alpha_text(value) for value in CANONICAL_ALPHAS)
        )
    if args.budget != CANONICAL_BUDGET:
        raise ValueError(f"Formal grid budget must equal {CANONICAL_BUDGET}")
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    if not args.replicate.startswith("r") or not args.replicate[1:].isdigit():
        raise ValueError("replicate must be r followed by digits")

    common = root / single.COMMON_INITIAL
    core = root / "src/run_ivm_v9_spring_projected_production_20260730.py"
    ala_core = root / "src/ala_core.py"
    frozen_solver = root / "src/frozen_discrete_b040.py"
    legacy_command_helper = root / "src/run_stage4_uniform_optimized.py"
    complete_state_reference = root / "src/build_pareto_archive.py"
    official_stage4_v4 = root / OFFICIAL_STAGE4_V4
    wrapper = root / "src/run_stage4_ala_discrete_single_fixedalpha.py"
    wrapper_implementation = root / "src/run_stage4_ala_dsel_single_fixedalpha.py"
    launcher_alias = root / "src/run_stage4_ala_discrete_single_fixedalpha_grid.py"
    launcher = Path(__file__).resolve()
    for path in (
        common,
        core,
        ala_core,
        frozen_solver,
        legacy_command_helper,
        complete_state_reference,
        wrapper,
        wrapper_implementation,
        launcher_alias,
        launcher,
        official_stage4_v4,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    for alpha in alphas:
        candidate_command, _ = single.single_run_command(
            root, alpha, args.budget, args.replicate, "candidate"
        )
        control_command, _ = single.single_run_command(
            root, alpha, args.budget, args.replicate, "matched_control"
        )
        single.assert_matched_pair_commands(
            candidate_command, control_command, alpha
        )

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    unique = time.time_ns()
    stem = f"{PROTOCOL_ID}_N{args.budget}_{args.replicate}_{unique}"
    preregistration = reports / f"{stem}_preregistered.json"
    final_report = reports / f"{stem}.json"
    csv_report = reports / f"{stem}.csv"
    protocol: dict[str, object] = {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "single_run_from_canonical_common_path": True,
        "fixed_alpha_throughout_each_member": True,
        "alpha_mode": "fixed",
        "canonical_alpha_grid": [
            "inf" if math.isinf(value) else value for value in alphas
        ],
        "grid_is_preregistered_and_not_data_adaptive": True,
        "refinement_rule": "only refine the locked winner's alpha neighborhood",
        "budget_per_image": args.budget,
        "candidate": {
            "subproblem": "discrete_b040",
            "parameterization": "joint_direct_discrete_cartesian_coordinates",
            "activation": "standalone_from_first_state",
            "stop_rule": "R/R0 <= 0.8 OR evaluated accepted steps == 10",
        },
        "matched_control": {
            "subproblem": "legacy",
            "stop_rule": "legacy state-IV inner stopping",
            "same_alpha_same_initial_path_same_outer_policy": True,
        },
        "one_candidate_and_one_control_per_alpha": True,
        "matched_pair_estimand": (
            "whole standalone discrete-coordinate DSEL plus R08C10 package "
            "versus legacy at identical alpha; not direction selection alone"
        ),
        "matched_pair_command_audit": (
            "exact after removing run label and registered DSEL package options"
        ),
        "tested_stop_variants": False,
        "reparameterization_policy": "state_curve per_image from first state",
        "reparameterization_choice_reason": (
            "existing production precision policy; a single policy avoids an "
            "outer-algorithm switch and protects the low-I_perp mesh"
        ),
        "confirmatory_alpha": 20.0,
        "confirmatory_claim_scope": (
            "alpha=20 matched candidate/control cost-to-precision only"
        ),
        "other_alpha_role": "exploratory preregistered tuning",
        "eligible_record_definition": (
            "single-run adaptation of protocol-native complete_candidate_calls: "
            "the initial evaluated state plus reparam_committed states only"
        ),
        "formal_selection_rule": (
            "I_perp cost-to-precision Pareto comparison on eligible complete "
            "projected states, then require segment_length_cv <= 0.0065 and "
            "segment_length_ratio <= 1.03"
        ),
        "cost_to_precision_definition": (
            "C(tau) is the minimum actual optimization NF among eligible "
            "records with I_perp <= tau; no interpolation; unreachable is infinity"
        ),
        "reporting_anchors": list(REPORTING_ANCHORS),
        "full_curve_dominance_test": {
            "comparators": (
                "same-alpha matched legacy control and, separately, official "
                "Stage-IV v4"
            ),
            "thresholds": (
                "every eligible Pareto-record I_perp threshold of the comparator"
            ),
            "weak_condition": "candidate C(tau) <= comparator C(tau) at every threshold",
            "strict_condition": "strict inequality at at least one threshold",
            "tightest_precision_condition": (
                "candidate must reach I_perp no larger than comparator's tightest "
                "eligible reachable precision"
            ),
        },
        "coarse_grid_failure_rule": (
            "if no coarse alpha dominates, refine alpha only around the best "
            "I_perp cost-to-precision near-miss; every other flag remains frozen"
        ),
        "terminal_point_ranking_allowed": False,
        "raw_terminal_iperp_ranking_allowed": False,
        "sparse_barrier_role": "blind_diagnostic_only_not_selection_or_ranking",
        "densified_energy_barrier_role": (
            "embargoed until alpha is locked, then post-lock validation of the "
            "selected finalist only"
        ),
        "official_stage4_v4_comparator": str(official_stage4_v4),
        "official_stage4_v4_comparator_sha256": legacy.file_sha256(
            official_stage4_v4
        ),
        "common_initial_path": str(common),
        "common_initial_path_sha256": legacy.file_sha256(common),
        "source_sha256": {
            "core": legacy.file_sha256(core),
            "ala_core": legacy.file_sha256(ala_core),
            "frozen_discrete_b040": legacy.file_sha256(frozen_solver),
            "legacy_command_helper": legacy.file_sha256(
                legacy_command_helper
            ),
            "complete_state_protocol_reference": legacy.file_sha256(
                complete_state_reference
            ),
            "wrapper_alias": legacy.file_sha256(wrapper),
            "wrapper_implementation": legacy.file_sha256(
                wrapper_implementation
            ),
            "launcher_alias": legacy.file_sha256(launcher_alias),
            "launcher": legacy.file_sha256(launcher),
        },
        "jobs": args.jobs,
        "replicate": args.replicate,
        "created_unix": time.time(),
    }
    _write_json_exclusive(preregistration, protocol)

    cells = [
        (alpha, member_role)
        for alpha in alphas
        for member_role in MEMBER_ROLES
    ]
    rows: list[dict[str, object]] = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                _run_cell,
                root,
                alpha,
                args.budget,
                args.replicate,
                member_role,
            ): (alpha, member_role)
            for alpha, member_role in cells
        }
        for future in concurrent.futures.as_completed(futures):
            alpha, member_role = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "alpha": "inf" if math.isinf(alpha) else alpha,
                    "member_role": member_role,
                    "budget_per_image": args.budget,
                    "replicate": args.replicate,
                    "returncode": -1,
                    "error": repr(exc),
                }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    role_order = {role: index for index, role in enumerate(MEMBER_ROLES)}
    alpha_order = {value: index for index, value in enumerate(alphas)}
    rows.sort(
        key=lambda row: (
            alpha_order[
                math.inf if row["alpha"] == "inf" else float(row["alpha"])
            ],
            role_order[str(row["member_role"])],
        )
    )
    successful = [row for row in rows if row.get("returncode") == 0]
    pair_completeness = {
        _alpha_text(alpha): all(
            any(
                row.get("returncode") == 0
                and (math.inf if row["alpha"] == "inf" else float(row["alpha"]))
                == alpha
                and row["member_role"] == role
                for row in rows
            )
            for role in MEMBER_ROLES
        )
        for alpha in alphas
    }
    report = {
        **protocol,
        "status": "complete" if len(successful) == len(rows) else "incomplete",
        "started_unix": started,
        "finished_unix": time.time(),
        "preregistration": str(preregistration),
        "preregistration_sha256": legacy.file_sha256(preregistration),
        "matched_pair_completeness": pair_completeness,
        "formal_ranking_deferred_to_cost_to_precision_analysis": True,
        "results": rows,
    }
    _write_json_exclusive(final_report, report)
    _write_csv(csv_report, rows)
    print(final_report)
    print(csv_report)
    if report["status"] != "complete" or not all(pair_completeness.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
