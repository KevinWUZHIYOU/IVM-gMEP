#!/usr/bin/env python3
"""Preregister or launch the two-cell formal FFrbk selector campaign.

Preregistration and execution are separate actions.  The launcher is blind to
physical energy and barriers; a later independent read-only analyzer may rank
only complete+mesh-eligible I_perp cost-to-precision records.
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

import run_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0 as member


PROTOCOL_ID = member.MEMBER_POLICY_VERSION
CANONICAL_ROOT_NAME = "alanine_ivm_clean_four_level_20260801"
CELL_ORDER = member.ARMS
COMMON_INITIAL_PATH = "inputs/common_initial_path_equidistant.npz"
COMMON_INITIAL_FILE_SHA256 = (
    "4a08066e1f0a89ad1d0ee6ecc442d7255d511217576d1ff65e9cb5e2164d891b"
)
COMMON_INITIAL_ARRAY_SHA256 = (
    "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
)
TOPOLOGY_PATH = "inputs/ala2_fixed.pdb"
TOPOLOGY_SHA256 = (
    "7ffaf924e345b6fbbba06d0388a67d18f04c07729a2ab7d65ede9a61cd27eda2"
)

MECHANISM_CTP_REPORT = (
    "reports/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
    "N254_r1_1787260497802476000_with_stage4_ala_discrete_single_"
    "lbfgs_mechanism_matched_kkt0_direct_continuation_v1_N254_r1_"
    "1787261109219817000_iperp_ctp_readonly_v1_1787261155406090000.json"
)
MECHANISM_CTP_REPORT_SHA256 = (
    "98fe38387a76ca11308eb96c071b5448542f3eff719620f7b7d67968c3ef193d"
)

PRIOR_ARTIFACTS = {
    "mechanism_ctp_readonly": (
        MECHANISM_CTP_REPORT,
        MECHANISM_CTP_REPORT_SHA256,
    ),
    "matched_current0_preregistration": (
        "reports/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
        "N254_r1_1787260497802476000_preregistered.json",
        "04de96743da079b8b35247214a4ad3cf387157edb2134381cc08f95d3bab1f86",
    ),
    "matched_current0_campaign_report": (
        "reports/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
        "N254_r1_1787260497802476000.json",
        "c1506422a651ee664b8cdb08a19c124bbab760435130042c300b7e721e7fa02f",
    ),
    "matched_current0_postrun_mechanism": (
        "reports/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
        "N254_r1_1787260497802476000_postrun_mechanism_provenance_readonly_v1.json",
        "115056ec3d22ee102ff714d704a12913f0ea09726d4918b0fa24a1d462a049aa",
    ),
    "matched_current0_result": (
        "runs/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1/"
        "members/ivm_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
        "current_ffrbk_selector_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1_"
        "central_state_ivm_cfg66edecece9c2_budget254.npz",
        "eff9d585dfbbd966744a972f9fa5b5a7b3da11639d2079c333560e34ce98ee07",
    ),
    "matched_current0_companion_audit": (
        "runs/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1/"
        "members/ivm_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
        "current_ffrbk_selector_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1_"
        "central_state_ivm_cfg66edecece9c2_budget254.audit.json",
        "7c6ea4b394ea867f8b83b64a1bb6b910cc6f156fdf32feca71b765ef5b1a14a0",
    ),
    "matched_current0_member_audit": (
        "runs/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1/"
        "audits/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
        "current_ffrbk_selector_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1.json",
        "f8fa0c5204295b704fe3cc3b8d0fbc47e29564b1daa9276a8f55045625c0ad8d",
    ),
    "matched_current0_command": (
        member.CURRENT0_COMMAND,
        member.CURRENT0_COMMAND_SHA256,
    ),
    "old_alpha15_eta1p1_result": (
        "runs/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1/members/"
        "ivm_stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_eta1p1_"
        "r08c10_a15_N254_discrete_b040_r1_central_state_ivm_"
        "cfg02c3f3990307_budget254.npz",
        "bb189235868d63be23d335c0766db59e0315b82fd11448077a492966ce5eaa55",
    ),
    "old_alpha15_eta1p1_command": (
        "logs/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1/members/"
        "stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_eta1p1_"
        "r08c10_a15_N254_discrete_b040_r1/command.json",
        "a3f622b4be15b64574479d5832bd76fc2a0de8fb3e2032ef4103e776129fb61f",
    ),
    "old_alpha15_eta1p1_ctp": (
        "reports/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_"
        "N254_r1_1787257808883811000_iperp_ctp_readonly_v1_1787258244070171000.json",
        "0ef85ac737d57c5a5b6b7a2e725eef8684e66e225934b9e8858c167fa887d9ae",
    ),
    "same_alpha_legacy_result": (
        "runs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/members/"
        "ivm_stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
        "legacy_stop_a15_N254_legacy_r1_central_state_ivm_"
        "cfg75672fcfca59_budget254.npz",
        "2ffa7aeb8122cbf4cdfecf9d4a372da8774c6485ec82f9b6693c3047c2d247dc",
    ),
    "same_alpha_legacy_audit": (
        "runs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/audits/"
        "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
        "legacy_stop_a15_N254_legacy_r1.json",
        "fb26c9fcb09ba1fbff7058c367965c694a573ecd5ed6110ce81366cbdef3fe83",
    ),
    "same_alpha_legacy_command": (
        "logs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/"
        "members/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
        "legacy_stop_a15_N254_legacy_r1/command.json",
        "9842cbe8e47971a49e4b90203192d59d45245b19baadf06e28b36ef049d4db16",
    ),
    "official_stage4_v4": (
        "runs/pareto_archive/pareto_clean_stage4_v4_a20_phaseA34_phaseB220.npz",
        "09ccbf7db731c96c0beda9d5879c9dd46cae2b428db8b71820aad1c56cb574fc",
    ),
}

SOURCE_PATHS = {
    "grid": "src/run_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_grid.py",
    "member": "src/run_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0.py",
    "campaign_test": "tests/test_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0.py",
    "takeover_runtime": "src/run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py",
    "takeover_runtime_test": "tests/test_run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py",
    "eta_runtime": "src/run_ivm_v9_discrete_selected_eta_shim.py",
    "eta_runtime_test": "tests/test_run_ivm_v9_discrete_selected_eta_shim.py",
    "frozen_discrete": "src/frozen_discrete_b040.py",
    "frozen_discrete_test": "tests/test_frozen_discrete_b040.py",
    "production_runner": "src/run_ivm_v9_spring_projected_production_20260730.py",
    "command_helper": "src/run_stage4_uniform_optimized.py",
    "ala_core": "src/ala_core.py",
    "matched_current0_member": "src/run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0.py",
    "flexible_trajectory_validator": (
        "src/run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_"
        "direct_continuation.py"
    ),
    "mechanism_ctp_analyzer": (
        "src/analyze_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_"
        "iperp_ctp.py"
    ),
}

# External frozen dependencies are checked before preregistration.  The two
# takeover runtime hashes are filled only after their dedicated tests freeze.
LOCKED_SOURCE_SHA256 = {
    "takeover_runtime": "62c6cf784795984c8104b4ba28fe2b0e15d657593648734d8f4a36ae03edfffc",
    "takeover_runtime_test": "1801013f70928adcf0fa0655b74a61835353a6fae0adf3f4f667783603200386",
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
    record = _hashed(initial, COMMON_INITIAL_FILE_SHA256)
    with np.load(initial, allow_pickle=False) as saved:
        array = np.ascontiguousarray(np.asarray(saved["initial_path"], dtype=np.float64))
    array_hash = hashlib.sha256(array.tobytes()).hexdigest()
    if array.shape != (22, 66) or array_hash != COMMON_INITIAL_ARRAY_SHA256:
        raise RuntimeError("common initial-path array drifted")
    return {
        "common_initial_path": {
            **record,
            "initial_path_array_sha256": array_hash,
        },
        "alanine_topology": _hashed(root / TOPOLOGY_PATH, TOPOLOGY_SHA256),
    }


def _prior_artifact_manifest(root: Path) -> dict[str, dict[str, str]]:
    return {
        name: _hashed(root / relative, expected)
        for name, (relative, expected) in PRIOR_ARTIFACTS.items()
    }


def _mechanism_ctp_evidence(root: Path) -> dict[str, Any]:
    path = root / MECHANISM_CTP_REPORT
    _hashed(path, MECHANISM_CTP_REPORT_SHA256)
    report = _read_json(path)
    exact = {
        "performance_metric": "I_perp_only",
        "physical_objective_arrays_accessed": False,
        "densification_performed": False,
        "unobserved_tail_extrapolated": False,
        "fixed_alpha": 15.0,
        "fixed_eta": 1.1,
        "fixed_residual_ratio": 0.8,
        "fixed_max_evaluated_steps": 10,
        "fixed_lbfgs_history": 7,
        "fixed_kkt_atol": 0.0,
        "maximum_trajectory_length": 254,
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise RuntimeError(f"mechanism CTP evidence drifted at {key}")
    arms = report.get("arms")
    if not isinstance(arms, dict) or "current_ffrbk_selector" not in arms:
        raise RuntimeError("mechanism CTP lost matched Current0")
    return {
        **exact,
        "eligibility": report.get("eligibility"),
        "cost_to_precision": report.get("cost_to_precision"),
        "early_termination_policy": report.get("early_termination_policy"),
        "matched_current0_result_sha256": arms["current_ffrbk_selector"].get(
            "result_sha256"
        ),
        "same_alpha_legacy_result_sha256": report.get("same_alpha_legacy", {}).get(
            "result_sha256"
        ),
        "official_stage4_v4_sha256": report.get("official_stage4_v4", {}).get(
            "archive_sha256"
        ),
    }


def _registered_commands(root: Path) -> dict[str, dict[str, Any]]:
    current0 = member.current0_command(root)
    records: dict[str, dict[str, Any]] = {}
    for arm in CELL_ORDER:
        command, label = member.single_run_command(root, arm)
        member.assert_registered_command(command, current0, root, arm)
        records[arm] = {
            "label": label,
            "command": command,
            "compact_sha256": member._compact_command_sha256(command),
            "matched_current0_command_sha256": member.CURRENT0_COMMAND_SHA256,
            "only_changed_components": [
                "runtime_entry_point",
                "selector_mode_flag",
                "output_dir_routing",
                "run_label_routing",
            ],
        }
    return records


def _assert_no_outputs(root: Path) -> None:
    existing: list[Path] = []
    for arm in CELL_ORDER:
        paths = member.member_paths(root, arm)
        result_dir = paths["result_dir"]
        label = str(paths["label"])
        assert isinstance(result_dir, Path)
        existing.extend(
            sorted(result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget254.npz"))
        )
        existing.extend(
            sorted(
                result_dir.glob(
                    f"ivm_{label}_central_state_ivm_cfg*_budget254.audit.json"
                )
            )
        )
        for key in ("audit", "member_log", "launcher_log"):
            path = paths[key]
            if isinstance(path, Path) and path.exists():
                existing.append(path)
    if existing:
        raise FileExistsError(f"refusing pre-existing formal output: {existing[0]}")


def build_preregistration(root: Path, planned_report: Path) -> dict[str, Any]:
    root = root.resolve()
    if root.name != CANONICAL_ROOT_NAME:
        raise RuntimeError("formal selector campaign requires canonical Ala repository")
    _assert_no_outputs(root)
    return {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "created_unix": time.time(),
        "planned_report": str(planned_report.resolve()),
        "study_role": "matched FFrbk selector-policy ablation after scalar ablation diagnosis",
        "cell_order": list(CELL_ORDER),
        "registered_commands": _registered_commands(root),
        "scientific_command_exact_matched_current0_clone": True,
        "only_runtime_entry_mode_and_routing_changed": True,
        "fixed_alpha": 15.0,
        "fixed_eta": 1.1,
        "fixed_residual_ratio": 0.8,
        "fixed_max_evaluated_steps": 10,
        "fixed_lbfgs_history": 7,
        "fixed_kkt_atol": 0.0,
        "fixed_reparameterization": "state_curve/per_image from first state",
        "budget_per_image": 254,
        "trajectory_record_count_allowed_range": [1, 254],
        "early_stop_policy": (
            "retain valid observed prefix as nonconverged; all precision beyond its tightest observed complete+mesh state is unreachable/U"
        ),
        "early_stop_counted_as_converged": False,
        "unobserved_tail_extrapolated": False,
        "formal_performance_metric": "I_perp_only",
        "formal_eligibility": (
            "initial plus reparam_committed states, then segment_length_cv<=0.0065 and segment_length_ratio<=1.03"
        ),
        "formal_cost_to_precision": (
            "minimum actual observed optimization NF with I_perp<=threshold; no interpolation or extrapolation"
        ),
        "candidate_unreachable_threshold_policy": "U_formal_loss",
        "primary_mechanism_contrast": (
            "one_way_takeover versus matched Current0: retain strict LBFGS wins for the remainder of each fixed subproblem"
        ),
        "one_way_invalid_lbfgs_safety_fallback": (
            "if the latched L-BFGS proposal is invalid, select a valid H0 proposal without clearing the latch"
        ),
        "secondary_mechanism_contrast": (
            "forced_operator_lbfgs_selection versus one_way_takeover: force valid operator-LBFGS selection, including the empty-history tie, while preserving the FFrbk operator"
        ),
        "forced_operator_lbfgs_is_selection_not_computation_ablation": True,
        "forced_operator_lbfgs_h0_shadow_policy": (
            "H0 remains constructed as a zero-extra-NF shadow/safety candidate; it is selected only when L-BFGS is invalid"
        ),
        "identical_candidate_trajectories_allowed": True,
        "forced_operator_lbfgs_is_selection_not_computation_ablation": True,
        "identical_trajectory_interpretation": (
            "valid mechanistic null: Current0 has strict H0 switches only at outer=3 steps=2,3, while every multi-step outer has a strict L-BFGS win at step=1"
        ),
        "performance_comparators": [
            "matched_current0",
            "old_alpha15_eta1p1",
            "same_alpha_legacy",
            "official_stage4_v4",
        ],
        "mechanism_ctp_evidence": _mechanism_ctp_evidence(root),
        "energy_or_objective_backtracking_added": False,
        "physical_energy_arrays_accessed_by_wrapper_or_launcher": False,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "densified_barrier_computation_performed": False,
        "barrier_unembargo_allowed_by_this_protocol": False,
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
        "post_hoc_scientific_command_changes_allowed": False,
        "prior_artifact_manifest": _prior_artifact_manifest(root),
        "common_initial_and_topology": _input_manifest(root),
        "run_source_manifest": _source_manifest(root),
        "single_thread_environment": {
            "OPENMM_CPU_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
        "independent_member_paths_and_logs": True,
        "overwrite_or_force_allowed": False,
        "execution_jobs_fixed": 1,
    }


def validate_preregistration(protocol: dict[str, Any], root: Path) -> None:
    root = root.resolve()
    exact = {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "cell_order": list(CELL_ORDER),
        "scientific_command_exact_matched_current0_clone": True,
        "only_runtime_entry_mode_and_routing_changed": True,
        "fixed_alpha": 15.0,
        "fixed_eta": 1.1,
        "fixed_residual_ratio": 0.8,
        "fixed_max_evaluated_steps": 10,
        "fixed_lbfgs_history": 7,
        "fixed_kkt_atol": 0.0,
        "budget_per_image": 254,
        "trajectory_record_count_allowed_range": [1, 254],
        "early_stop_counted_as_converged": False,
        "unobserved_tail_extrapolated": False,
        "formal_performance_metric": "I_perp_only",
        "candidate_unreachable_threshold_policy": "U_formal_loss",
        "identical_candidate_trajectories_allowed": True,
        "energy_or_objective_backtracking_added": False,
        "physical_energy_arrays_accessed_by_wrapper_or_launcher": False,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "densified_barrier_computation_performed": False,
        "barrier_unembargo_allowed_by_this_protocol": False,
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
        "post_hoc_scientific_command_changes_allowed": False,
        "overwrite_or_force_allowed": False,
        "execution_jobs_fixed": 1,
    }
    for key, expected in exact.items():
        if protocol.get(key) != expected:
            raise RuntimeError(f"selector preregistration drifted at {key}")
    if protocol.get("registered_commands") != _registered_commands(root):
        raise RuntimeError("registered selector commands drifted")
    if protocol.get("mechanism_ctp_evidence") != _mechanism_ctp_evidence(root):
        raise RuntimeError("mechanism CTP evidence drifted")
    _verify_manifest(protocol.get("prior_artifact_manifest"), "prior artifacts")
    if protocol.get("prior_artifact_manifest") != _prior_artifact_manifest(root):
        raise RuntimeError("prior artifact manifest drifted")
    _verify_manifest(protocol.get("common_initial_and_topology"), "inputs")
    if protocol.get("common_initial_and_topology") != _input_manifest(root):
        raise RuntimeError("input manifest drifted")
    _verify_manifest(protocol.get("run_source_manifest"), "run sources")
    if protocol.get("run_source_manifest") != _source_manifest(root):
        raise RuntimeError("source manifest drifted")
    planned = Path(str(protocol.get("planned_report"))).resolve()
    if planned.parent != (root / "reports").resolve() or not planned.name.startswith(
        f"{PROTOCOL_ID}_N254_r1_"
    ):
        raise RuntimeError("planned report namespace drifted")
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
        str(root / "src/run_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0.py"),
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
        raise RuntimeError("member audit identity/embargo drifted")
    result = Path(str(audit["result"])).resolve()
    if file_sha256(result) != str(audit["result_sha256"]):
        raise RuntimeError("member result hash mismatch")
    row.update(
        {
            "member_audit": str(audit_path),
            "member_audit_sha256": file_sha256(audit_path),
            "result": str(result),
            "result_sha256": file_sha256(result),
            "trajectory_record_count": int(audit["trajectory_record_count"]),
            "trajectory_complete_to_budget": bool(audit["trajectory_complete_to_budget"]),
            "early_stop": bool(audit["early_stop"]),
            "physical_stop_reason": str(audit["physical_stop_reason"]),
            "scientific_run_converged": False,
            "precision_beyond_tightest_reached_unreachable": bool(
                audit["precision_beyond_tightest_reached_unreachable"]
            ),
            "final_optimization_NF": int(audit["final_optimization_NF"]),
            "registered_command_sha256": str(audit["scientific_command_sha256"]),
        }
    )
    return row


def preregister(root: Path) -> Path:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stem = f"{PROTOCOL_ID}_N254_r1_{time.time_ns()}"
    preregistration = reports / f"{stem}_preregistered.json"
    planned_report = reports / f"{stem}.json"
    _write_json_exclusive(preregistration, build_preregistration(root, planned_report))
    return preregistration


def launch(root: Path, preregistration: Path, jobs: int) -> Path:
    if jobs != 1:
        raise ValueError("formal selector campaign requires --jobs 1")
    preregistration = preregistration.expanduser().resolve()
    protocol = _read_json(preregistration)
    validate_preregistration(protocol, root)
    final_report = Path(str(protocol["planned_report"])).resolve()
    if final_report.exists():
        raise FileExistsError(f"refusing to overwrite {final_report}")
    started = time.time()
    rows: list[dict[str, Any]] = []
    for arm in CELL_ORDER:
        try:
            row = _run_cell(root, arm)
        except Exception as error:
            row = {"arm": arm, "returncode": -1, "error": repr(error)}
        rows.append(row)
    complete = all(int(row.get("returncode", -1)) == 0 for row in rows)
    registered = protocol["registered_commands"]
    for row in rows:
        if int(row.get("returncode", -1)) == 0 and row.get(
            "registered_command_sha256"
        ) != registered[row["arm"]]["compact_sha256"]:
            raise RuntimeError("cell ran under an unregistered command")
    report = {
        **protocol,
        "status": "complete" if complete else "incomplete",
        "preregistration": str(preregistration),
        "preregistration_sha256": file_sha256(preregistration),
        "execution_jobs_operational_only": jobs,
        "started_unix": started,
        "finished_unix": time.time(),
        "early_stop_is_valid_execution_not_scientific_convergence": True,
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
            raise ValueError("--jobs is fixed at 1")
        print(preregister(root))
        return
    print(launch(root, args.preregistration, args.jobs))


if __name__ == "__main__":
    main()
