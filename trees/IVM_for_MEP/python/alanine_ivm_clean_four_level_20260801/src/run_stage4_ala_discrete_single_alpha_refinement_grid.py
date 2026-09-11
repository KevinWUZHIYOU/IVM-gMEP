#!/usr/bin/env python3
"""Preregister and later launch the isolated Ala alpha-only refinement.

The preregistration step is deliberately separate from execution.  It freezes
four new alpha values and hashes the completed coarse preregistration/report,
the read-only coarse I_perp CTP report, every source used by a refinement
member, and the canonical input/comparator.  Execution requires that immutable
preregistration and fails closed on any hash or command-policy drift.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import run_stage4_ala_discrete_single_alpha_refinement as refinement


PROTOCOL_ID = "stage4_ala_discrete_single_alpha_refinement_matched_v1"
COARSE_PROTOCOL_ID = "stage4_ala_discrete_single_fixedalpha_matched_v2"
COARSE_CTP_ANALYSIS_ID = "stage4_ala_discrete_single_iperp_ctp_readonly_v1"
REFINEMENT_ALPHAS = refinement.REFINEMENT_ALPHAS
CANONICAL_BUDGET = refinement.CANONICAL_BUDGET
CANONICAL_REPLICATE = refinement.CANONICAL_REPLICATE
MEMBER_ROLES = refinement.MEMBER_ROLES
COARSE_BRACKET = (1.0, 20.0)
LOCKED_CENTER_ALPHA = 10.0
UPPER_BOUNDARY_ALPHA = 20.0
NEIGHBORHOOD_MIN_ALPHA = 5.0
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
COARSE_PREREGISTRATION = (
    "reports/"
    "stage4_ala_discrete_single_fixedalpha_matched_v2_N254_r1_"
    "1787251461968963000_preregistered.json"
)
COARSE_REPORT = (
    "reports/"
    "stage4_ala_discrete_single_fixedalpha_matched_v2_N254_r1_"
    "1787251461968963000.json"
)
COARSE_CTP_REPORT = (
    "reports/"
    "stage4_ala_discrete_single_fixedalpha_matched_v2_N254_r1_"
    "1787251461968963000_iperp_ctp_readonly_v1_1787252176364307000.json"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _hashed_path(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _alpha_value(value: object) -> float:
    if value == "inf":
        return math.inf
    return float(value)


def _source_manifest(root: Path) -> dict[str, dict[str, str]]:
    sources = {
        "refinement_launcher": Path(__file__).resolve(),
        "refinement_member_wrapper": (
            root / "src/run_stage4_ala_discrete_single_alpha_refinement.py"
        ),
        "frozen_coarse_wrapper_alias": (
            root / "src/run_stage4_ala_discrete_single_fixedalpha.py"
        ),
        "frozen_coarse_wrapper_implementation": (
            root / "src/run_stage4_ala_dsel_single_fixedalpha.py"
        ),
        "legacy_command_helper": root / "src/run_stage4_uniform_optimized.py",
        "core_runner": (
            root / "src/run_ivm_v9_spring_projected_production_20260730.py"
        ),
        "ala_core": root / "src/ala_core.py",
        "frozen_discrete_b040": root / "src/frozen_discrete_b040.py",
    }
    return {name: _hashed_path(path) for name, path in sources.items()}


def _decision_source_manifest(root: Path) -> dict[str, dict[str, str]]:
    sources = {
        "complete_state_protocol_reference": root / "src/build_pareto_archive.py",
        "coarse_ctp_analyzer": (
            root / "src/analyze_stage4_ala_discrete_single_iperp_ctp.py"
        ),
    }
    return {name: _hashed_path(path) for name, path in sources.items()}


def _verify_frozen_coarse_sources(
    root: Path, coarse_prereg: dict[str, object]
) -> dict[str, dict[str, str]]:
    paths = {
        "core": root / "src/run_ivm_v9_spring_projected_production_20260730.py",
        "ala_core": root / "src/ala_core.py",
        "frozen_discrete_b040": root / "src/frozen_discrete_b040.py",
        "legacy_command_helper": root / "src/run_stage4_uniform_optimized.py",
        "complete_state_protocol_reference": root / "src/build_pareto_archive.py",
        "wrapper_alias": root / "src/run_stage4_ala_discrete_single_fixedalpha.py",
        "wrapper_implementation": root / "src/run_stage4_ala_dsel_single_fixedalpha.py",
        "launcher_alias": root / "src/run_stage4_ala_discrete_single_fixedalpha_grid.py",
        "launcher": root / "src/run_stage4_ala_dsel_single_fixedalpha_grid.py",
    }
    expected = coarse_prereg.get("source_sha256")
    if not isinstance(expected, dict) or set(expected) != set(paths):
        raise RuntimeError("Coarse source hash manifest is incomplete")
    records = {}
    for name, path in paths.items():
        actual = file_sha256(path)
        if actual != str(expected[name]):
            raise RuntimeError(f"Frozen coarse source changed: {name}")
        records[name] = {"path": str(path.resolve()), "sha256": actual}
    return records


def _extract_reference_cells(
    coarse: dict[str, object], root: Path
) -> dict[str, dict[str, dict[str, object]]]:
    results = coarse.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Coarse report has no result rows")
    references: dict[str, dict[str, dict[str, object]]] = {}
    for alpha in (LOCKED_CENTER_ALPHA, UPPER_BOUNDARY_ALPHA):
        pair: dict[str, dict[str, object]] = {}
        for member_role in MEMBER_ROLES:
            matches = [
                row
                for row in results
                if isinstance(row, dict)
                and _alpha_value(row.get("alpha")) == alpha
                and row.get("member_role") == member_role
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Coarse alpha={alpha:g} {member_role} is not unique"
                )
            row = matches[0]
            if int(row.get("returncode", -1)) != 0:
                raise RuntimeError("A coarse reference member failed")
            result = Path(str(row["result"])).resolve()
            audit = Path(str(row["member_audit"])).resolve()
            if file_sha256(result) != str(row["result_sha256"]):
                raise RuntimeError("Coarse reference result hash mismatch")
            if file_sha256(audit) != str(row["member_audit_sha256"]):
                raise RuntimeError("Coarse reference audit hash mismatch")
            pair[member_role] = {
                "result": str(result),
                "result_sha256": file_sha256(result),
                "member_audit": str(audit),
                "member_audit_sha256": file_sha256(audit),
            }
        references[f"alpha{alpha:g}"] = pair
    return references


def _decision_inputs(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    coarse_prereg_path = (root / COARSE_PREREGISTRATION).resolve()
    coarse_report_path = (root / COARSE_REPORT).resolve()
    coarse_ctp_path = (root / COARSE_CTP_REPORT).resolve()
    coarse_prereg = _read_json(coarse_prereg_path)
    coarse_report = _read_json(coarse_report_path)
    coarse_ctp = _read_json(coarse_ctp_path)

    if coarse_prereg.get("status") != "preregistered":
        raise RuntimeError("Coarse preregistration is not immutable preregistered input")
    if coarse_prereg.get("protocol_id") != COARSE_PROTOCOL_ID:
        raise RuntimeError("Unexpected coarse preregistration protocol")
    if coarse_report.get("status") != "complete":
        raise RuntimeError("Coarse report is not complete")
    if coarse_report.get("protocol_id") != COARSE_PROTOCOL_ID:
        raise RuntimeError("Unexpected coarse report protocol")
    if Path(str(coarse_report["preregistration"])).resolve() != coarse_prereg_path:
        raise RuntimeError("Coarse report points to another preregistration")
    if str(coarse_report["preregistration_sha256"]) != file_sha256(
        coarse_prereg_path
    ):
        raise RuntimeError("Coarse preregistration hash mismatch")

    if coarse_ctp.get("analysis_id") != COARSE_CTP_ANALYSIS_ID:
        raise RuntimeError("Unexpected coarse CTP analysis")
    if not bool(coarse_ctp.get("read_only_formal_input_audit")):
        raise RuntimeError("Coarse CTP analysis is not read-only")
    if coarse_ctp.get("barrier_data_accessed") is not False:
        raise RuntimeError("Coarse CTP analysis accessed barrier data")
    if coarse_ctp.get("barrier_computation_or_densification_performed") is not False:
        raise RuntimeError("Coarse CTP analysis broke the barrier embargo")
    if Path(str(coarse_ctp["formal_grid_report"])).resolve() != coarse_report_path:
        raise RuntimeError("Coarse CTP analysis points to another grid report")
    if str(coarse_ctp["formal_grid_report_sha256"]) != file_sha256(
        coarse_report_path
    ):
        raise RuntimeError("Coarse CTP report's grid hash mismatch")
    if Path(str(coarse_ctp["preregistration"])).resolve() != coarse_prereg_path:
        raise RuntimeError("Coarse CTP analysis points to another preregistration")
    if coarse_ctp.get("joint_full_curve_dominators") != []:
        raise RuntimeError("Alpha-only refinement requires no coarse joint dominator")
    analyzer = (root / "src/analyze_stage4_ala_discrete_single_iperp_ctp.py").resolve()
    if Path(str(coarse_ctp["analysis_script"])).resolve() != analyzer:
        raise RuntimeError("Coarse CTP output names another analyzer")
    if str(coarse_ctp["analysis_script_sha256"]) != file_sha256(analyzer):
        raise RuntimeError("Coarse CTP analyzer changed after its output")
    frontier = [str(value) for value in coarse_ctp["near_miss_deficit_pareto_alphas"]]
    if "10" not in frontier:
        raise RuntimeError("The locked alpha=10 center is not a coarse near-miss")

    official = Path(str(coarse_prereg["official_stage4_v4_comparator"])).resolve()
    common = Path(str(coarse_prereg["common_initial_path"])).resolve()
    if file_sha256(official) != str(
        coarse_prereg["official_stage4_v4_comparator_sha256"]
    ):
        raise RuntimeError("Official comparator changed since coarse preregistration")
    if file_sha256(common) != str(coarse_prereg["common_initial_path_sha256"]):
        raise RuntimeError("Common initial path changed since coarse preregistration")

    inputs = {
        "coarse_preregistration": _hashed_path(coarse_prereg_path),
        "coarse_complete_report": _hashed_path(coarse_report_path),
        "coarse_readonly_iperp_ctp_report": _hashed_path(coarse_ctp_path),
        "common_initial_path": _hashed_path(common),
        "official_stage4_v4_comparator": _hashed_path(official),
    }
    findings = {
        "coarse_joint_full_curve_dominators": [],
        "coarse_near_miss_deficit_pareto_alphas": frontier,
        "locked_center_alpha": LOCKED_CENTER_ALPHA,
        "upper_boundary_alpha": UPPER_BOUNDARY_ALPHA,
        "reference_cells_reused_not_rerun": _extract_reference_cells(
            coarse_report, root
        ),
        "frozen_coarse_source_manifest_verified": _verify_frozen_coarse_sources(
            root, coarse_prereg
        ),
    }
    return inputs, findings


def _assert_no_member_outputs(root: Path) -> None:
    for alpha in REFINEMENT_ALPHAS:
        for member_role in MEMBER_ROLES:
            paths = refinement.member_paths(root, alpha, member_role)
            result_dir = paths["result_dir"]
            audit = paths["audit"]
            member_log = paths["member_log"]
            launcher_log = paths["launcher_log"]
            label = str(paths["label"])
            assert isinstance(result_dir, Path)
            assert isinstance(audit, Path)
            assert isinstance(member_log, Path)
            assert isinstance(launcher_log, Path)
            existing = sorted(
                result_dir.glob(
                    f"ivm_{label}_central_state_ivm_cfg*_budget{CANONICAL_BUDGET}.npz"
                )
            )
            if existing or audit.exists() or member_log.exists() or launcher_log.exists():
                target = existing[0] if existing else audit
                raise FileExistsError(f"Refusing pre-existing refinement data: {target}")


def _audit_registered_commands(root: Path) -> dict[str, object]:
    command_hashes: dict[str, object] = {}
    for alpha in REFINEMENT_ALPHAS:
        candidate, _ = refinement.single_run_command(
            root,
            alpha,
            CANONICAL_BUDGET,
            CANONICAL_REPLICATE,
            "candidate",
        )
        control, _ = refinement.single_run_command(
            root,
            alpha,
            CANONICAL_BUDGET,
            CANONICAL_REPLICATE,
            "matched_control",
        )
        refinement.assert_matched_pair_commands(candidate, control, alpha)
        command_hashes[f"alpha{alpha:g}"] = {
            "candidate_command_sha256": hashlib.sha256(
                json.dumps(candidate, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "matched_control_command_sha256": hashlib.sha256(
                json.dumps(control, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
    return command_hashes


def build_preregistration(
    root: Path, planned_report: Path
) -> dict[str, object]:
    if tuple(REFINEMENT_ALPHAS) != (5.0, 7.5, 12.5, 15.0):
        raise RuntimeError("Refinement alpha grid drifted")
    if not all(
        COARSE_BRACKET[0] < alpha < COARSE_BRACKET[1]
        and NEIGHBORHOOD_MIN_ALPHA <= alpha < UPPER_BOUNDARY_ALPHA
        for alpha in REFINEMENT_ALPHAS
    ):
        raise RuntimeError("Refinement alpha lies outside the locked neighborhood")
    if LOCKED_CENTER_ALPHA in REFINEMENT_ALPHAS or UPPER_BOUNDARY_ALPHA in REFINEMENT_ALPHAS:
        raise RuntimeError("Coarse center/boundary must be reused rather than rerun")
    _assert_no_member_outputs(root)
    inputs, findings = _decision_inputs(root)
    command_hashes = _audit_registered_commands(root)
    return {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "created_unix": time.time(),
        "planned_report": str(planned_report.resolve()),
        "alpha_only_refinement": True,
        "new_alpha_grid": list(REFINEMENT_ALPHAS),
        "coarse_bracket": {
            "lower_alpha": COARSE_BRACKET[0],
            "upper_alpha": COARSE_BRACKET[1],
            "new_points_strictly_inside": True,
        },
        "locked_neighborhood_definition": (
            "inside coarse bracket (1,20), with 5 <= alpha < 20"
        ),
        "locked_center_alpha_reused_not_rerun": LOCKED_CENTER_ALPHA,
        "upper_boundary_alpha_reused_not_rerun": UPPER_BOUNDARY_ALPHA,
        "one_candidate_and_one_same_alpha_legacy_control_per_new_alpha": True,
        "budget_per_image": CANONICAL_BUDGET,
        "replicate": CANONICAL_REPLICATE,
        "candidate": {
            "subproblem": "discrete_b040",
            "fixed_alpha_throughout": True,
            "stop_rule": "R/R0 <= 0.8 OR evaluated accepted steps == 10",
        },
        "matched_control": {
            "subproblem": "legacy",
            "same_alpha_same_initial_path_same_outer_policy": True,
        },
        "scientific_command_contract": (
            "exact frozen coarse-v2 command at each new alpha after removing "
            "only output-dir and run-label routing"
        ),
        "all_non_alpha_scientific_flags_frozen": True,
        "tested_stop_variants": False,
        "reparameterization_policy": "state_curve/per_image from first state",
        "single_run_from_canonical_common_path": True,
        "eligible_record_definition": "initial plus reparam_committed states",
        "mesh_gate": {
            "segment_length_cv_max": refinement.MESH_CV_MAX,
            "segment_length_ratio_max": refinement.MESH_RATIO_MAX,
        },
        "cost_to_precision_definition": (
            "minimum actual optimization NF at I_perp <= threshold; no interpolation"
        ),
        "reporting_anchors": list(REPORTING_ANCHORS),
        "formal_selection_rule": (
            "full-curve Iperp CTP on complete mesh-eligible states versus the "
            "same-alpha control and separately official Stage-IV v4; no terminal rank"
        ),
        "primary_success_criterion": (
            "candidate must simultaneously full-curve dominate its same-alpha "
            "legacy control and the frozen official Stage-IV v4 post-activation tail"
        ),
        "alpha_lock_requires_primary_success": True,
        "no_primary_dominator_rule": {
            "winner_claim_allowed": False,
            "barrier_unembargo_allowed": False,
            "only_allowed_followup": (
                "choose the center alpha for a separately preregistered, "
                "single-variable stop-rule ablation"
            ),
            "eligible_center_alphas": [5.0, 7.5, 10.0, 12.5, 15.0, 20.0],
            "official_tail_deficit_order": [
                (
                    "fewest unreachable frozen official-v4 Pareto thresholds "
                    "in its I_perp<=0.007 domain"
                ),
                "smallest tightest eligible complete-mesh I_perp",
                (
                    "lexicographically smallest candidate-minus-official NF "
                    "vector at official Pareto thresholds ordered tight-to-loose; "
                    "unreachable is positive infinity"
                ),
                "smaller numeric alpha only as an exact deterministic tie-break",
            ],
            "selection_is_not_an_alpha_winner_claim": True,
        },
        "terminal_or_raw_iperp_ranking_allowed": False,
        "iperp_only_scientific_response_during_refinement": True,
        "sparse_barrier_role": "embargoed_not_read_not_reported_not_ranked",
        "densified_energy_barrier_role": "embargoed_until_alpha_is_locked",
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "decision_inputs": inputs,
        "coarse_decision_findings": findings,
        "run_source_manifest": _source_manifest(root),
        "decision_source_manifest": _decision_source_manifest(root),
        "registered_command_hashes": command_hashes,
        "member_policy_version": refinement.MEMBER_POLICY_VERSION,
        "independent_member_paths_and_logs": True,
    }


def _verify_hash_manifest(manifest: object, name: str) -> None:
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError(f"Invalid {name}")
    for key, record in manifest.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"Invalid {name} record: {key}")
        path = Path(str(record["path"])).resolve()
        if file_sha256(path) != str(record["sha256"]):
            raise RuntimeError(f"Hash drift in {name}: {key}")


def validate_preregistration(protocol: dict[str, object], root: Path) -> None:
    if protocol.get("status") != "preregistered":
        raise RuntimeError("Refinement input is not a preregistration")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Unexpected refinement protocol")
    if tuple(float(value) for value in protocol["new_alpha_grid"]) != REFINEMENT_ALPHAS:
        raise RuntimeError("Preregistered refinement alpha grid drifted")
    if int(protocol["budget_per_image"]) != CANONICAL_BUDGET:
        raise RuntimeError("Preregistered budget drifted")
    if protocol["replicate"] != CANONICAL_REPLICATE:
        raise RuntimeError("Preregistered replicate drifted")
    if protocol.get("all_non_alpha_scientific_flags_frozen") is not True:
        raise RuntimeError("Non-alpha flags were not frozen")
    if protocol.get("barrier_fields_accessed_by_wrapper_or_launcher") is not False:
        raise RuntimeError("Preregistration does not preserve the barrier embargo")
    planned_report = Path(str(protocol["planned_report"])).resolve()
    reports = (root / "reports").resolve()
    if planned_report.parent != reports or not planned_report.name.startswith(
        f"{PROTOCOL_ID}_N{CANONICAL_BUDGET}_{CANONICAL_REPLICATE}_"
    ):
        raise RuntimeError("Planned report is outside the immutable protocol namespace")
    _verify_hash_manifest(protocol["decision_inputs"], "decision inputs")
    _verify_hash_manifest(protocol["run_source_manifest"], "run sources")
    _verify_hash_manifest(protocol["decision_source_manifest"], "decision sources")
    if _audit_registered_commands(root) != protocol["registered_command_hashes"]:
        raise RuntimeError("Registered refinement commands changed")
    _assert_no_member_outputs(root)


def _run_process(
    command: list[str], root: Path, stdout_path: Path, stderr_path: Path
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


def _run_cell(root: Path, alpha: float, member_role: str) -> dict[str, object]:
    paths = refinement.member_paths(root, alpha, member_role)
    launcher_log = paths["launcher_log"]
    audit_path = paths["audit"]
    assert isinstance(launcher_log, Path)
    assert isinstance(audit_path, Path)
    launcher_log.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(root / "src/run_stage4_ala_discrete_single_alpha_refinement.py"),
        "--root",
        str(root),
        "--alpha",
        f"{alpha:g}",
        "--budget",
        str(CANONICAL_BUDGET),
        "--member-role",
        member_role,
        "--replicate",
        CANONICAL_REPLICATE,
    ]
    (launcher_log / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    returncode, wall_seconds, stderr = _run_process(
        command,
        root,
        launcher_log / "stdout.txt",
        launcher_log / "stderr.txt",
    )
    row: dict[str, object] = {
        "alpha": alpha,
        "member_role": member_role,
        "budget_per_image": CANONICAL_BUDGET,
        "replicate": CANONICAL_REPLICATE,
        "returncode": returncode,
        "wall_seconds": wall_seconds,
    }
    if returncode:
        row["error"] = stderr[-4000:]
        return row
    if not audit_path.is_file():
        raise RuntimeError("Refinement member completed without immutable audit")
    audit = _read_json(audit_path)
    if audit.get("barrier_fields_accessed") is not False:
        raise RuntimeError("Refinement member audit broke the barrier embargo")
    if audit.get("formal_ranking_deferred_to_complete_mesh_iperp_ctp") is not True:
        raise RuntimeError("Refinement member attempted premature ranking")
    if float(audit["alpha"]) != alpha or audit["member_role"] != member_role:
        raise RuntimeError("Refinement member audit identity mismatch")
    result = Path(str(audit["result"])).resolve()
    if file_sha256(result) != str(audit["result_sha256"]):
        raise RuntimeError("Refinement member result hash mismatch")
    row.update(
        {
            "member_audit": str(audit_path),
            "member_audit_sha256": file_sha256(audit_path),
            "result": str(result),
            "result_sha256": file_sha256(result),
            "final_optimization_NF": int(audit["final_optimization_NF"]),
            "complete_state_count": len(audit["complete_state_NF"]),
            "mesh_eligible_complete_state_count": int(
                audit["mesh_eligible_complete_state_count"]
            ),
        }
    )
    return row


def preregister(root: Path) -> Path:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{PROTOCOL_ID}_N{CANONICAL_BUDGET}_{CANONICAL_REPLICATE}_"
        f"{time.time_ns()}"
    )
    preregistration = reports / f"{stem}_preregistered.json"
    planned_report = reports / f"{stem}.json"
    protocol = build_preregistration(root, planned_report)
    _write_json_exclusive(preregistration, protocol)
    return preregistration


def launch(root: Path, preregistration: Path, jobs: int) -> Path:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    preregistration = preregistration.expanduser().resolve()
    protocol = _read_json(preregistration)
    validate_preregistration(protocol, root)
    final_report = Path(str(protocol["planned_report"])).resolve()
    if final_report.exists():
        raise FileExistsError(f"Refusing to overwrite {final_report}")
    cells = [
        (alpha, member_role)
        for alpha in REFINEMENT_ALPHAS
        for member_role in MEMBER_ROLES
    ]
    rows: list[dict[str, object]] = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_run_cell, root, alpha, member_role): (alpha, member_role)
            for alpha, member_role in cells
        }
        for future in concurrent.futures.as_completed(futures):
            alpha, member_role = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "alpha": alpha,
                    "member_role": member_role,
                    "budget_per_image": CANONICAL_BUDGET,
                    "replicate": CANONICAL_REPLICATE,
                    "returncode": -1,
                    "error": repr(exc),
                }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    alpha_order = {value: index for index, value in enumerate(REFINEMENT_ALPHAS)}
    role_order = {value: index for index, value in enumerate(MEMBER_ROLES)}
    rows.sort(
        key=lambda row: (
            alpha_order[float(row["alpha"])],
            role_order[str(row["member_role"])],
        )
    )
    complete = all(int(row.get("returncode", -1)) == 0 for row in rows)
    report = {
        **protocol,
        "status": "complete" if complete else "incomplete",
        "preregistration": str(preregistration),
        "preregistration_sha256": file_sha256(preregistration),
        "execution_jobs_operational_only": jobs,
        "started_unix": started,
        "finished_unix": time.time(),
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
        "results": rows,
    }
    _write_json_exclusive(final_report, report)
    if not complete:
        raise SystemExit(1)
    return final_report


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preregister-only", action="store_true")
    action.add_argument("--preregistration", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    root = args.root.expanduser().resolve()
    if args.preregister_only:
        if args.jobs != 1:
            raise ValueError("--jobs applies only to execution")
        print(preregister(root))
        return
    print(launch(root, args.preregistration, args.jobs))


if __name__ == "__main__":
    main()
