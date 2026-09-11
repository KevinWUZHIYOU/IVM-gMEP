#!/usr/bin/env python3
"""Preregister, then separately launch the matched Current0/A/B grid.

Preregistration and execution are deliberately separate actions.  This file
does not rank results; the preregistered complete+mesh I_perp CTP decision is
performed later by an independent, barrier-blind read-only analyzer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0 as member


PROTOCOL_ID = member.MEMBER_POLICY_VERSION
CANONICAL_ROOT_NAME = "alanine_ivm_clean_four_level_20260801"
CELL_ORDER = member.ARMS
REPORTING_ANCHORS = (
    1.0,
    0.1,
    0.05,
    0.02,
    0.01,
    0.005,
    0.002,
    0.001,
    5.0e-4,
    2.0e-4,
    1.5e-4,
)
COMMON_INITIAL_PATH = "inputs/common_initial_path_equidistant.npz"
COMMON_INITIAL_FILE_SHA256 = (
    "4a08066e1f0a89ad1d0ee6ecc442d7255d511217576d1ff65e9cb5e2164d891b"
)
COMMON_INITIAL_ARRAY_SHA256 = member.COMMON_INITIAL_ARRAY_SHA256
TOPOLOGY_PATH = "inputs/ala2_fixed.pdb"
TOPOLOGY_SHA256 = (
    "7ffaf924e345b6fbbba06d0388a67d18f04c07729a2ab7d65ede9a61cd27eda2"
)

OLD_CURRENT_PREREGISTRATION = (
    "reports/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_"
    "N254_r1_1787257808883811000_preregistered.json"
)
OLD_CURRENT_PREREGISTRATION_SHA256 = (
    "dbffae0850f39128c313f2819c28b7679d5e80ef0a6f9fe59246dc43a176df1d"
)
OLD_CURRENT_REPORT = (
    "reports/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_"
    "N254_r1_1787257808883811000.json"
)
OLD_CURRENT_REPORT_SHA256 = (
    "fa0bf64922d91d15a48b0b6b6949d8ddb3ff109661ff011b351f8f9caee79212"
)
OLD_CURRENT_CTP = (
    "reports/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_"
    "N254_r1_1787257808883811000_iperp_ctp_readonly_v1_"
    "1787258244070171000.json"
)
OLD_CURRENT_CTP_SHA256 = (
    "0ef85ac737d57c5a5b6b7a2e725eef8684e66e225934b9e8858c167fa887d9ae"
)
OLD_CURRENT_POSTRUN = (
    "reports/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_"
    "N254_r1_1787257808883811000_postrun_mechanism_provenance_readonly_v1.json"
)
OLD_CURRENT_POSTRUN_SHA256 = (
    "7dbd94c4627db5938d189ec24aa2fa8f5790a62122b506410b9b34c86c1ea95a"
)
OLD_CURRENT_AUDIT = (
    "runs/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1/audits/"
    "stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_eta1p1_"
    "r08c10_a15_N254_discrete_b040_r1.json"
)
OLD_CURRENT_AUDIT_SHA256 = (
    "058d591e5fccc4a0e7a4dee0854c4f372b7b2648d2673dd25a9bd58363eed91a"
)
LEGACY_RESULT = (
    "runs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/members/"
    "ivm_stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "legacy_stop_a15_N254_legacy_r1_central_state_ivm_"
    "cfg75672fcfca59_budget254.npz"
)
LEGACY_RESULT_SHA256 = (
    "2ffa7aeb8122cbf4cdfecf9d4a372da8774c6485ec82f9b6693c3047c2d247dc"
)
LEGACY_AUDIT = (
    "runs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/audits/"
    "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "legacy_stop_a15_N254_legacy_r1.json"
)
LEGACY_AUDIT_SHA256 = (
    "fb26c9fcb09ba1fbff7058c367965c694a573ecd5ed6110ce81366cbdef3fe83"
)
LEGACY_COMMAND = (
    "logs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/members/"
    "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "legacy_stop_a15_N254_legacy_r1/command.json"
)
LEGACY_COMMAND_SHA256 = (
    "9842cbe8e47971a49e4b90203192d59d45245b19baadf06e28b36ef049d4db16"
)
OFFICIAL_STAGE4_V4 = (
    "runs/pareto_archive/pareto_clean_stage4_v4_a20_phaseA34_phaseB220.npz"
)
OFFICIAL_STAGE4_V4_SHA256 = (
    "09ccbf7db731c96c0beda9d5879c9dd46cae2b428db8b71820aad1c56cb574fc"
)
ALPHA15_FORMAL_CTP = (
    "reports/stage4_ala_discrete_single_alpha_refinement_matched_v1_"
    "N254_r1_1787252714040428000_iperp_ctp_readonly_v1_"
    "1787253163640931000.json"
)
ALPHA15_FORMAL_CTP_SHA256 = (
    "8ff9d9824eb2d3719d28f73f33b92b62e115ff9f7449bf60be054298d7ba5212"
)

SOURCE_PATHS = {
    "grid": "src/run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_grid.py",
    "member": "src/run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0.py",
    "member_test": "tests/test_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0.py",
    "scalar_runtime": "src/run_ivm_v9_discrete_scalar_lbfgs_shim.py",
    "scalar_runtime_test": "tests/test_run_ivm_v9_discrete_scalar_lbfgs_shim.py",
    "eta_runtime": "src/run_ivm_v9_discrete_selected_eta_shim.py",
    "eta_runtime_test": "tests/test_run_ivm_v9_discrete_selected_eta_shim.py",
    "frozen_discrete": "src/frozen_discrete_b040.py",
    "frozen_discrete_test": "tests/test_frozen_discrete_b040.py",
    "production_runner": "src/run_ivm_v9_spring_projected_production_20260730.py",
    "command_helper": "src/run_stage4_uniform_optimized.py",
    "ala_core": "src/ala_core.py",
    "old_current_member": "src/run_stage4_ala_discrete_single_eta1p1_alpha15_followup.py",
    "old_current_grid": "src/run_stage4_ala_discrete_single_eta1p1_alpha15_followup_grid.py",
    "old_current_ctp": "src/analyze_stage4_ala_discrete_single_eta1p1_alpha15_iperp_ctp.py",
    "base_ctp": "src/analyze_stage4_ala_discrete_single_alpha_refinement_iperp_ctp.py",
    "eta_ctp": "src/analyze_stage4_ala_discrete_single_selected_eta_iperp_ctp.py",
}
LOCKED_SOURCE_SHA256 = {
    "scalar_runtime": "1d20bff8dd1d8c202f9b18d7a5f948219066c482bf898bfce9246764782000ae",
    "scalar_runtime_test": "8e0d3840a8b169ce3d5e6f06d39c4fcb66e6872b016b2a85278a43ba7d05ea37",
    "eta_runtime": "a2f071ea787a008ed001fd806b8952f84d8db90335b544308478f3f6ae2facf9",
    "eta_runtime_test": "35909b6453110f64b6998551e4d7ff1d84c3d3d17c440bc54d33464889572fed",
    "frozen_discrete": "8384a84666af463991df2e8d8a344779ae675feaddc73c12bcc82ff6ba324e22",
    "frozen_discrete_test": "3cdaa3a6bcbd043b9a15de5b5af90b6fc49f3505bf891ec4892a8193ab365aa9",
    "production_runner": "9abd6e96bb9e6cead380a28c849fd11c15c5fe9a8854d77fe41a991240b0510f",
    "command_helper": "990be5520093af674bc44061c0c38af9169a1e207f06c33f7bea9d53ac2d7e02",
    "ala_core": "e3d9b2d472d8b9d7e0e03289dcf6a87bdff10cb8fc296bdfda911798799377bd",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashed(path: Path, expected: str | None = None) -> dict[str, str]:
    path = path.resolve()
    actual = file_sha256(path)
    if expected is not None and actual != expected:
        raise RuntimeError(f"hash drift: {path}")
    return {"path": str(path), "sha256": actual}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _verify_manifest(manifest: Any, context: str) -> None:
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError(f"invalid manifest: {context}")
    for name, record in manifest.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"invalid manifest record: {context}.{name}")
        _hashed(Path(str(record["path"])), str(record["sha256"]))


def _source_manifest(root: Path) -> dict[str, dict[str, str]]:
    return {
        name: _hashed(root / relative, LOCKED_SOURCE_SHA256.get(name))
        for name, relative in SOURCE_PATHS.items()
    }


def _input_manifest(root: Path) -> dict[str, dict[str, str]]:
    initial = root / COMMON_INITIAL_PATH
    initial_record = _hashed(initial, COMMON_INITIAL_FILE_SHA256)
    with np.load(initial, allow_pickle=False) as saved:
        array = np.ascontiguousarray(
            np.asarray(saved["initial_path"], dtype=np.float64)
        )
    array_hash = hashlib.sha256(array.tobytes()).hexdigest()
    if array.shape != (22, 66) or array_hash != COMMON_INITIAL_ARRAY_SHA256:
        raise RuntimeError("common initial-path array drifted")
    return {
        "common_initial_path": {
            **initial_record,
            "initial_path_array_sha256": array_hash,
        },
        "alanine_topology": _hashed(root / TOPOLOGY_PATH, TOPOLOGY_SHA256),
    }


def _decision_inputs(root: Path) -> dict[str, dict[str, str]]:
    records = {
        "old_current_preregistration": (OLD_CURRENT_PREREGISTRATION, OLD_CURRENT_PREREGISTRATION_SHA256),
        "old_current_report": (OLD_CURRENT_REPORT, OLD_CURRENT_REPORT_SHA256),
        "old_current_result": (member.OLD_CURRENT_RESULT, member.OLD_CURRENT_RESULT_SHA256),
        "old_current_member_audit": (OLD_CURRENT_AUDIT, OLD_CURRENT_AUDIT_SHA256),
        "old_current_command": (member.OLD_CURRENT_COMMAND, member.OLD_CURRENT_COMMAND_SHA256),
        "old_current_iperp_ctp": (OLD_CURRENT_CTP, OLD_CURRENT_CTP_SHA256),
        "old_current_postrun": (OLD_CURRENT_POSTRUN, OLD_CURRENT_POSTRUN_SHA256),
        "same_alpha_legacy_result": (LEGACY_RESULT, LEGACY_RESULT_SHA256),
        "same_alpha_legacy_audit": (LEGACY_AUDIT, LEGACY_AUDIT_SHA256),
        "same_alpha_legacy_command": (LEGACY_COMMAND, LEGACY_COMMAND_SHA256),
        "official_stage4_v4": (OFFICIAL_STAGE4_V4, OFFICIAL_STAGE4_V4_SHA256),
        "alpha15_formal_ctp_snapshot": (ALPHA15_FORMAL_CTP, ALPHA15_FORMAL_CTP_SHA256),
    }
    return {
        name: _hashed(root / relative, expected)
        for name, (relative, expected) in records.items()
    }


def _assert_no_outputs(root: Path) -> None:
    for arm in CELL_ORDER:
        paths = member.member_paths(root, arm)
        result_dir = paths["result_dir"]
        label = str(paths["label"])
        assert isinstance(result_dir, Path)
        matches = sorted(
            result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget254.npz")
        )
        companions = sorted(
            result_dir.glob(
                f"ivm_{label}_central_state_ivm_cfg*_budget254.audit.json"
            )
        )
        other_paths = [paths["audit"], paths["member_log"], paths["launcher_log"]]
        existing = matches + companions + [
            path for path in other_paths if isinstance(path, Path) and path.exists()
        ]
        if existing:
            raise FileExistsError(f"refusing pre-existing formal cell: {existing[0]}")


def _registered_commands(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    old = member.canonical_old_current_command(root)
    for arm in CELL_ORDER:
        command, label = member.single_run_command(root, arm)
        member.assert_registered_command(command, old, root, arm)
        records[arm] = {
            "label": label,
            "command": command,
            "compact_sha256": member._compact_command_sha256(command),
        }
    return records


def _verify_old_current_semantics(root: Path) -> None:
    command = member.canonical_old_current_command(root)
    options = member.command_options(command)
    expected = {
        "--alpha": "15",
        "--alpha-mode": "fixed",
        "--discrete-step-multiplier-eta": "1.1",
        "--discrete-stop-mode": "r08c10",
        "--discrete-residual-ratio": "0.8",
        "--discrete-inner-max-steps": "10",
        "--discrete-history": "7",
        "--budget": "254",
        "--reparam-mode": "state_curve",
        "--state-curve-direct-objective-acceptance": "per_image",
    }
    for flag, value in expected.items():
        if options.get(flag) != value:
            raise RuntimeError(f"old Current oracle drifted at {flag}")
    with np.load(root / member.OLD_CURRENT_RESULT, allow_pickle=False) as saved:
        reasons = np.asarray(saved["discrete_audit_stop_reason"], dtype="U64")
        if np.any(np.char.find(reasons, "absolute") >= 0):
            raise RuntimeError("old Current oracle actually used absolute-KKT stop")
        if str(np.asarray(saved["initial_path_sha256"]).item()) != (
            COMMON_INITIAL_ARRAY_SHA256
        ):
            raise RuntimeError("old Current oracle changed common initial path")


def build_preregistration(root: Path, planned_report: Path) -> dict[str, Any]:
    root = root.resolve()
    if root.name != CANONICAL_ROOT_NAME:
        raise RuntimeError("formal grid requires the canonical Ala repository")
    _assert_no_outputs(root)
    _verify_old_current_semantics(root)
    return {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "created_unix": time.time(),
        "planned_report": str(planned_report.resolve()),
        "study_role": "prospective_matched_mechanism_ablation_informed_by_old_control",
        "cell_order": list(CELL_ORDER),
        "three_new_cells": True,
        "old_alpha15_eta1p1_role": "parity_and_provenance_oracle_only",
        "current0_role": "matched_ffrbk_selector_control_with_explicit_kkt_atol_zero",
        "scalar_selector_role": (
            "replace only the fixed-solver FFrbk inverse operator by joint scalar gamma I; retain H0/LBFGS selector"
        ),
        "direct_lbfgs_only_role": (
            "same joint scalar L-BFGS but no H0 proposal and no selector call"
        ),
        "no_single_arm_predeclared_winner": True,
        "fixed_alpha": 15.0,
        "alpha_sweep_or_adaptation": False,
        "fixed_eta": 1.1,
        "eta_sweep_or_adaptation": False,
        "fixed_residual_ratio": 0.8,
        "fixed_max_evaluated_steps": 10,
        "fixed_lbfgs_history": 7,
        "fixed_kkt_atol": 0.0,
        "absolute_kkt_stop_enabled": False,
        "budget_per_image": 254,
        "replicate": "r1",
        "stop_rule": "R/R0 <= 0.8 OR evaluated accepted steps == 10",
        "terminal_budget_exhaustion_role": "incomplete_truncation_not_convergence",
        "reparameterization_policy": "state_curve/per_image from first state",
        "common_initial_and_topology": _input_manifest(root),
        "single_thread_environment": {
            "OPENMM_CPU_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
        "command_contract": (
            "all cells clone the pinned old alpha15/eta1.1 command, add the same explicit --plbfgs-kkt-atol 0, and change only isolated routing; A/B additionally change only the runtime entry point and registered scalar mode"
        ),
        "registered_commands": _registered_commands(root),
        "current0_old_oracle_safe_whole_run_parity_required_before_ctp": True,
        "current0_parity_fields": sorted(member.CURRENT0_PARITY_KEYS),
        "current0_parity_exclusions": ["configuration_sha256", "plbfgs_kkt_atol"],
        "fixed_solver_ffrbk_contract_for_scalar_arms": {
            "preconditioner_argument_is_none": True,
            "solve_normal_calls_inside_fixed_solver": 0,
            "runner_bank_argument_intercept_equals_fixed_solver_calls": True,
            "outer_recovery_and_reparameterization_unchanged": True,
            "bank_initialization_and_build_raw_openmm_nf_delta": 0,
            "solve_normal_outside_fixed_solver_allowed_and_audited": True,
        },
        "energy_or_objective_backtracking_added": False,
        "physical_energy_arrays_accessed_by_wrapper_or_launcher": False,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "densified_barrier_computation_performed": False,
        "eligible_record_definition": "initial plus reparam_committed states",
        "mesh_gate": {
            "segment_length_cv_max": member.MESH_CV_MAX,
            "segment_length_ratio_max": member.MESH_RATIO_MAX,
        },
        "cost_to_precision_definition": (
            "minimum actual total internal joint-evaluation NF at I_perp <= threshold; no interpolation"
        ),
        "reporting_anchors": list(REPORTING_ANCHORS),
        "unreachable_threshold_policy": "candidate_unreachable_is_loss",
        "pairwise_full_curve_dominance": (
            "no loss or unreachable threshold at any comparator Pareto threshold, at least one strict NF win, and tightest eligible I_perp no worse"
        ),
        "performance_candidates": ["scalar_selector", "direct_lbfgs_only"],
        "primary_comparators_for_each_performance_candidate": [
            "current_ffrbk_selector",
            "same_alpha_legacy",
            "official_stage4_v4",
        ],
        "performance_qualification_rule": (
            "each candidate independently must full-curve dominate exactly all three primary comparators"
        ),
        "winner_rule": (
            "zero qualifiers means no winner; one qualifier is the unique winner; if both qualify, lock a unique winner only when exactly one directional A-vs-B full-curve dominance holds, otherwise report co-qualified with no lock"
        ),
        "mechanism_diagnostics": {
            "ffrbk_operator_effect": "bidirectional current_ffrbk_selector versus scalar_selector full CTP",
            "selector_effect": "bidirectional scalar_selector versus direct_lbfgs_only full CTP",
            "combined_effect": "bidirectional current_ffrbk_selector versus direct_lbfgs_only full CTP",
            "crossing_interpretation": "precision-regime-dependent; no global mechanism superiority claim",
        },
        "direct_fallback_and_gamma_floor_role": (
            "registered safety semantics; report counts and mechanism sensitivity, but do not automatically invalidate performance"
        ),
        "terminal_or_raw_iperp_ranking_allowed": False,
        "post_hoc_alpha_eta_or_arm_expansion_allowed": False,
        "barrier_unembargo_allowed_by_this_protocol": False,
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
        "decision_inputs": _decision_inputs(root),
        "run_source_manifest": _source_manifest(root),
        "independent_member_paths_and_logs": True,
        "overwrite_or_force_allowed": False,
        "execution_jobs_fixed": 1,
    }


def validate_preregistration(protocol: dict[str, Any], root: Path) -> None:
    root = root.resolve()
    if protocol.get("status") != "preregistered" or protocol.get(
        "protocol_id"
    ) != PROTOCOL_ID:
        raise RuntimeError("invalid mechanism preregistration")
    exact = {
        "cell_order": list(CELL_ORDER),
        "three_new_cells": True,
        "no_single_arm_predeclared_winner": True,
        "fixed_alpha": 15.0,
        "alpha_sweep_or_adaptation": False,
        "fixed_eta": 1.1,
        "eta_sweep_or_adaptation": False,
        "fixed_residual_ratio": 0.8,
        "fixed_max_evaluated_steps": 10,
        "fixed_lbfgs_history": 7,
        "fixed_kkt_atol": 0.0,
        "absolute_kkt_stop_enabled": False,
        "budget_per_image": 254,
        "replicate": "r1",
        "energy_or_objective_backtracking_added": False,
        "physical_energy_arrays_accessed_by_wrapper_or_launcher": False,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "densified_barrier_computation_performed": False,
        "terminal_or_raw_iperp_ranking_allowed": False,
        "post_hoc_alpha_eta_or_arm_expansion_allowed": False,
        "barrier_unembargo_allowed_by_this_protocol": False,
        "overwrite_or_force_allowed": False,
        "execution_jobs_fixed": 1,
    }
    for key, expected in exact.items():
        if protocol.get(key) != expected:
            raise RuntimeError(f"preregistration drifted at {key}")
    if protocol.get("current0_parity_fields") != sorted(member.CURRENT0_PARITY_KEYS):
        raise RuntimeError("Current0 parity field contract drifted")
    if protocol.get("primary_comparators_for_each_performance_candidate") != [
        "current_ffrbk_selector",
        "same_alpha_legacy",
        "official_stage4_v4",
    ]:
        raise RuntimeError("primary comparator set drifted")
    if protocol.get("common_initial_and_topology") != _input_manifest(root):
        raise RuntimeError("input manifest drifted")
    if protocol.get("registered_commands") != _registered_commands(root):
        raise RuntimeError("registered command vectors drifted")
    _verify_manifest(protocol.get("decision_inputs"), "decision inputs")
    if protocol.get("decision_inputs") != _decision_inputs(root):
        raise RuntimeError("decision inputs drifted")
    _verify_manifest(protocol.get("run_source_manifest"), "run sources")
    if protocol.get("run_source_manifest") != _source_manifest(root):
        raise RuntimeError("run source manifest drifted")
    planned = Path(str(protocol.get("planned_report"))).resolve()
    if planned.parent != (root / "reports").resolve() or not planned.name.startswith(
        f"{PROTOCOL_ID}_N254_r1_"
    ):
        raise RuntimeError("planned report namespace drifted")
    _verify_old_current_semantics(root)
    _assert_no_outputs(root)


def _run_process(
    command: list[str], root: Path, stdout: Path, stderr: Path
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
    stdout.write_text(completed.stdout, encoding="utf-8")
    stderr.write_text(completed.stderr, encoding="utf-8")
    return completed.returncode, time.time() - started, completed.stderr


def _run_cell(root: Path, arm: str) -> dict[str, Any]:
    paths = member.member_paths(root, arm)
    launcher_log = paths["launcher_log"]
    audit_path = paths["audit"]
    assert isinstance(launcher_log, Path)
    assert isinstance(audit_path, Path)
    launcher_log.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(
            root
            / "src/run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0.py"
        ),
        "--root",
        str(root),
        "--arm",
        arm,
    ]
    (launcher_log / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    returncode, wall_seconds, error = _run_process(
        command,
        root,
        launcher_log / "stdout.txt",
        launcher_log / "stderr.txt",
    )
    row: dict[str, Any] = {
        "arm": arm,
        "returncode": returncode,
        "wall_seconds": wall_seconds,
    }
    if returncode:
        row["error"] = error[-4000:]
        return row
    audit = _read_json(audit_path)
    if audit.get("arm") != arm or audit.get("barrier_fields_accessed") is not False:
        raise RuntimeError(f"{arm} member audit identity/embargo drift")
    result = Path(str(audit["result"])).resolve()
    if file_sha256(result) != str(audit["result_sha256"]):
        raise RuntimeError(f"{arm} result hash mismatch")
    row.update(
        {
            "member_audit": str(audit_path),
            "member_audit_sha256": file_sha256(audit_path),
            "result": str(result),
            "result_sha256": file_sha256(result),
            "final_optimization_NF": int(audit["final_optimization_NF"]),
            "mesh_eligible_complete_state_count": int(
                audit["mesh_eligible_complete_state_count"]
            ),
            "registered_command_sha256": str(
                audit["scientific_command_sha256"]
            ),
        }
    )
    return row


def preregister(root: Path) -> Path:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    unique = time.time_ns()
    stem = f"{PROTOCOL_ID}_N254_r1_{unique}"
    preregistration = reports / f"{stem}_preregistered.json"
    planned_report = reports / f"{stem}.json"
    _write_json_exclusive(
        preregistration, build_preregistration(root, planned_report)
    )
    return preregistration


def launch(root: Path, preregistration: Path, jobs: int) -> Path:
    if jobs != 1:
        raise ValueError("formal mechanism grid requires --jobs 1")
    preregistration = preregistration.expanduser().resolve()
    protocol = _read_json(preregistration)
    validate_preregistration(protocol, root)
    final_report = Path(str(protocol["planned_report"])).resolve()
    if final_report.exists():
        raise FileExistsError(f"refusing to overwrite {final_report}")
    rows: list[dict[str, Any]] = []
    started = time.time()
    for arm in CELL_ORDER:
        try:
            row = _run_cell(root, arm)
        except Exception as error:
            row = {"arm": arm, "returncode": -1, "error": repr(error)}
        rows.append(row)
        if int(row.get("returncode", -1)) != 0:
            break
        registered = protocol["registered_commands"][arm]["compact_sha256"]
        if row.get("registered_command_sha256") != registered:
            raise RuntimeError(f"{arm} completed under an unregistered command")
    complete = len(rows) == len(CELL_ORDER) and all(
        int(row.get("returncode", -1)) == 0 for row in rows
    )
    report = {
        **protocol,
        "status": "complete" if complete else "incomplete",
        "preregistration": str(preregistration),
        "preregistration_sha256": file_sha256(preregistration),
        "execution_jobs_operational_only": jobs,
        "started_unix": started,
        "finished_unix": time.time(),
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
        "physical_energy_arrays_accessed_by_launcher": False,
        "barrier_fields_accessed_by_launcher": False,
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
            raise ValueError("--jobs applies only to execution and is fixed at 1")
        print(preregister(root))
        return
    print(launch(root, args.preregistration, args.jobs))


if __name__ == "__main__":
    main()
