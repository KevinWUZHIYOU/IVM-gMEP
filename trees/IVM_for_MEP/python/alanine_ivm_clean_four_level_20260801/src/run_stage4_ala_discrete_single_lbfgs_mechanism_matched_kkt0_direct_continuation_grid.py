#!/usr/bin/env python3
"""Preregister, then separately launch the one-cell direct continuation."""

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

import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_direct_continuation as member
import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_grid as original_grid


PROTOCOL_ID = member.MEMBER_POLICY_VERSION
CANONICAL_ROOT_NAME = "alanine_ivm_clean_four_level_20260801"
CELL_ORDER = (member.ARM,)

SOURCE_PATHS = {
    "continuation_grid": (
        "src/run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_"
        "direct_continuation_grid.py"
    ),
    "continuation_member": (
        "src/run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_"
        "direct_continuation.py"
    ),
    "continuation_test": (
        "tests/test_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_"
        "direct_continuation.py"
    ),
}

ORIGINAL_REPORT = (
    "reports/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
    "N254_r1_1787260497802476000.json"
)
ORIGINAL_REPORT_SHA256 = (
    "c1506422a651ee664b8cdb08a19c124bbab760435130042c300b7e721e7fa02f"
)
CURRENT_RESULT = (
    "runs/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1/members/"
    "ivm_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
    "current_ffrbk_selector_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1_"
    "central_state_ivm_cfg66edecece9c2_budget254.npz"
)
CURRENT_RESULT_SHA256 = (
    "eff9d585dfbbd966744a972f9fa5b5a7b3da11639d2079c333560e34ce98ee07"
)
CURRENT_MEMBER_AUDIT = (
    "runs/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1/audits/"
    "stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
    "current_ffrbk_selector_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1.json"
)
CURRENT_MEMBER_AUDIT_SHA256 = (
    "f8fa0c5204295b704fe3cc3b8d0fbc47e29564b1daa9276a8f55045625c0ad8d"
)
CURRENT_COMPANION_AUDIT = CURRENT_RESULT[:-4] + ".audit.json"
CURRENT_COMPANION_AUDIT_SHA256 = (
    "7c6ea4b394ea867f8b83b64a1bb6b910cc6f156fdf32feca71b765ef5b1a14a0"
)
SCALAR_RESULT = (
    "runs/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1/members/"
    "ivm_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
    "scalar_selector_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1_"
    "central_state_ivm_cfga9f9d7ced447_budget254.npz"
)
SCALAR_RESULT_SHA256 = (
    "ef5b54cd3127c86abf82ad934bb932f89e17e3008037fbd1c38a2570d868aea4"
)
SCALAR_COMPANION_AUDIT = SCALAR_RESULT[:-4] + ".audit.json"
SCALAR_COMPANION_AUDIT_SHA256 = (
    "1753186830b182811f56ea6dfd49bcbaaa53d66c46d9917f7cc24526632d585d"
)

LOG_ROOT = "logs/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1"
CURRENT_LABEL = (
    "stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
    "current_ffrbk_selector_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1"
)
SCALAR_LABEL = (
    "stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
    "scalar_selector_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1"
)
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _log_path(tier: str, label: str, name: str) -> str:
    return f"{LOG_ROOT}/{tier}/{label}/{name}"


PRIOR_ARTIFACT_PATHS_AND_HASHES = {
    "original_v1_preregistration": (
        member.ORIGINAL_PREREGISTRATION,
        member.ORIGINAL_PREREGISTRATION_SHA256,
    ),
    "original_v1_report": (ORIGINAL_REPORT, ORIGINAL_REPORT_SHA256),
    "current0_result": (CURRENT_RESULT, CURRENT_RESULT_SHA256),
    "current0_member_audit": (
        CURRENT_MEMBER_AUDIT,
        CURRENT_MEMBER_AUDIT_SHA256,
    ),
    "current0_companion_audit": (
        CURRENT_COMPANION_AUDIT,
        CURRENT_COMPANION_AUDIT_SHA256,
    ),
    "scalar_selector_result": (SCALAR_RESULT, SCALAR_RESULT_SHA256),
    "scalar_selector_companion_audit": (
        SCALAR_COMPANION_AUDIT,
        SCALAR_COMPANION_AUDIT_SHA256,
    ),
    "current0_launcher_command_log": (
        _log_path("launcher", CURRENT_LABEL, "command.json"),
        "d2a3b0468cfceadb7baa11f368ccdcb5329f169f1704bdb6f829227145a2a05e",
    ),
    "current0_launcher_stdout_log": (
        _log_path("launcher", CURRENT_LABEL, "stdout.txt"),
        CURRENT_MEMBER_AUDIT_SHA256,
    ),
    "current0_launcher_stderr_log": (
        _log_path("launcher", CURRENT_LABEL, "stderr.txt"),
        EMPTY_SHA256,
    ),
    "current0_member_command_log": (
        _log_path("members", CURRENT_LABEL, "command.json"),
        "e33ca25d88bcc6e13867f074f0b6ff242ac0822f948fb758edb52ea62a640863",
    ),
    "current0_member_stdout_log": (
        _log_path("members", CURRENT_LABEL, "stdout.txt"),
        "4861bc83a55442915afecfcbd85e69320f4450b4fba9200435855f9a8c195a5c",
    ),
    "current0_member_stderr_log": (
        _log_path("members", CURRENT_LABEL, "stderr.txt"),
        EMPTY_SHA256,
    ),
    "scalar_launcher_command_log": (
        _log_path("launcher", SCALAR_LABEL, "command.json"),
        "e577393d29a0e62298d772549ac72f444977c23dc233719d76c4ad11ffd85274",
    ),
    "scalar_launcher_stdout_log": (
        _log_path("launcher", SCALAR_LABEL, "stdout.txt"),
        EMPTY_SHA256,
    ),
    "scalar_launcher_stderr_log": (
        _log_path("launcher", SCALAR_LABEL, "stderr.txt"),
        "55736355c8594194ea8a7a1b53852818848773cc35d496748e9108a66b6de6fe",
    ),
    "scalar_member_command_log": (
        _log_path("members", SCALAR_LABEL, "command.json"),
        "61211b81757fc47d814939add57f1265b771876f06408d5b00032acd143ac589",
    ),
    "scalar_member_stdout_log": (
        _log_path("members", SCALAR_LABEL, "stdout.txt"),
        "8d22025e1e86804c067629bdd131a2ff06dd6c1b101c5af416a201f9bdb0d69e",
    ),
    "scalar_member_stderr_log": (
        _log_path("members", SCALAR_LABEL, "stderr.txt"),
        EMPTY_SHA256,
    ),
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


def _original_protocol(root: Path) -> dict[str, Any]:
    path = root / member.ORIGINAL_PREREGISTRATION
    _hashed(path, member.ORIGINAL_PREREGISTRATION_SHA256)
    protocol = _read_json(path)
    record = protocol.get("registered_commands", {}).get(member.ARM, {})
    if record.get("compact_sha256") != member.ORIGINAL_DIRECT_COMMAND_SHA256:
        raise RuntimeError("original direct registration drifted")
    _verify_manifest(protocol.get("run_source_manifest"), "original v1 sources")
    return protocol


def _prior_artifact_manifest(root: Path) -> dict[str, dict[str, str]]:
    return {
        name: _hashed(root / relative, expected)
        for name, (relative, expected) in PRIOR_ARTIFACT_PATHS_AND_HASHES.items()
    }


def _source_manifest(root: Path) -> dict[str, dict[str, str]]:
    manifest = {
        name: _hashed(root / relative)
        for name, relative in SOURCE_PATHS.items()
    }
    original = _original_protocol(root)["run_source_manifest"]
    for name, record in original.items():
        manifest[f"original_v1_{name}"] = dict(record)
    return manifest


def _registered_command(root: Path) -> dict[str, Any]:
    original = member._original_direct_command(root)
    command, label = member.single_run_command(root)
    member.assert_registered_command(command, original, root)
    return {
        "label": label,
        "command": command,
        "compact_sha256": member._compact_command_sha256(command),
        "original_v1_compact_sha256": member.ORIGINAL_DIRECT_COMMAND_SHA256,
        "only_changed_options": list(member.ROUTING_OPTIONS),
    }


def _verify_original_outcome(root: Path) -> dict[str, Any]:
    report = _read_json(root / ORIGINAL_REPORT)
    rows = report.get("results")
    if report.get("status") != "incomplete" or not isinstance(rows, list):
        raise RuntimeError("original v1 report status drifted")
    if [row.get("arm") for row in rows] != [
        "current_ffrbk_selector",
        "scalar_selector",
    ]:
        raise RuntimeError("original v1 executed-cell order drifted")
    if rows[0].get("returncode") != 0 or rows[1].get("returncode") != 1:
        raise RuntimeError("original v1 return codes drifted")
    with np.load(root / SCALAR_RESULT, allow_pickle=False) as saved:
        record_count = int(np.asarray(saved["active_alpha"]).size)
        stop_reason = str(np.asarray(saved["stop_reason"]).item())
        final_nf = int(np.asarray(saved["total_internal_joint_evaluations"])[-1])
    companion = _read_json(root / SCALAR_COMPANION_AUDIT)
    if (
        record_count != 58
        or stop_reason != "cached_coordinate_stall"
        or final_nf != 1160
        or companion.get("consistent") is not True
    ):
        raise RuntimeError("original scalar early-stop evidence drifted")
    return {
        "original_report_status": "incomplete",
        "current0_completed_and_validated": True,
        "scalar_selector_result_valid_but_wrapper_rejected_fixed_N254_shape": True,
        "scalar_selector_trajectory_record_count": record_count,
        "scalar_selector_final_optimization_NF": final_nf,
        "scalar_selector_physical_stop_reason": stop_reason,
        "direct_lbfgs_only_was_not_launched": True,
    }


def _assert_no_outputs(root: Path) -> None:
    paths = member.member_paths(root)
    result_dir = paths["result_dir"]
    label = str(paths["label"])
    assert isinstance(result_dir, Path)
    matches = sorted(
        result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget254.npz")
    )
    companions = sorted(
        result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget254.audit.json")
    )
    others = [paths["audit"], paths["member_log"], paths["launcher_log"]]
    existing = matches + companions + [
        path for path in others if isinstance(path, Path) and path.exists()
    ]
    if existing:
        raise FileExistsError(f"refusing pre-existing continuation output: {existing[0]}")


def build_preregistration(root: Path, planned_report: Path) -> dict[str, Any]:
    root = root.resolve()
    if root.name != CANONICAL_ROOT_NAME:
        raise RuntimeError("formal continuation requires the canonical Ala repository")
    _assert_no_outputs(root)
    outcome = _verify_original_outcome(root)
    return {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "created_unix": time.time(),
        "planned_report": str(planned_report.resolve()),
        "study_role": "immutable_one_cell_completion_of_unlaunched_original_v1_direct_arm",
        "continuation_of_protocol": (
            "stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1"
        ),
        "cell_order": list(CELL_ORDER),
        "single_new_cell": True,
        "scientific_command_exact_original_direct_clone": True,
        "scientific_command_only_routing_changed": True,
        "registered_command": _registered_command(root),
        "fixed_alpha": 15.0,
        "fixed_eta": 1.1,
        "fixed_residual_ratio": 0.8,
        "fixed_max_evaluated_steps": 10,
        "fixed_lbfgs_history": 7,
        "fixed_kkt_atol": 0.0,
        "budget_per_image": 254,
        "trajectory_record_count_allowed_range": [1, 254],
        "early_stop_policy": (
            "accept scientifically valid partial trajectory; mark nonconverged and treat precision beyond tightest reached as unreachable"
        ),
        "early_stop_counted_as_converged": False,
        "unreachable_threshold_policy": "candidate_unreachable_is_loss",
        "fixed_solver_mechanism_gates_unchanged": True,
        "aborted_solver_call_count_required_zero": True,
        "energy_or_objective_backtracking_added": False,
        "physical_energy_arrays_accessed_by_wrapper_or_launcher": False,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "densified_barrier_computation_performed": False,
        "barrier_unembargo_allowed_by_this_protocol": False,
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
        "post_hoc_scientific_command_changes_allowed": False,
        "original_v1_outcome": outcome,
        "prior_artifact_manifest": _prior_artifact_manifest(root),
        "common_initial_and_topology": original_grid._input_manifest(root),
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
        "single_new_cell": True,
        "scientific_command_exact_original_direct_clone": True,
        "scientific_command_only_routing_changed": True,
        "trajectory_record_count_allowed_range": [1, 254],
        "early_stop_counted_as_converged": False,
        "unreachable_threshold_policy": "candidate_unreachable_is_loss",
        "fixed_solver_mechanism_gates_unchanged": True,
        "aborted_solver_call_count_required_zero": True,
        "energy_or_objective_backtracking_added": False,
        "physical_energy_arrays_accessed_by_wrapper_or_launcher": False,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "densified_barrier_computation_performed": False,
        "barrier_unembargo_allowed_by_this_protocol": False,
        "post_hoc_scientific_command_changes_allowed": False,
        "overwrite_or_force_allowed": False,
        "execution_jobs_fixed": 1,
    }
    for key, expected in exact.items():
        if protocol.get(key) != expected:
            raise RuntimeError(f"continuation preregistration drifted at {key}")
    if protocol.get("registered_command") != _registered_command(root):
        raise RuntimeError("continuation registered command drifted")
    if protocol.get("original_v1_outcome") != _verify_original_outcome(root):
        raise RuntimeError("continuation rationale evidence drifted")
    _verify_manifest(protocol.get("prior_artifact_manifest"), "prior artifacts")
    if protocol.get("prior_artifact_manifest") != _prior_artifact_manifest(root):
        raise RuntimeError("prior artifact manifest drifted")
    _verify_manifest(protocol.get("run_source_manifest"), "run sources")
    if protocol.get("run_source_manifest") != _source_manifest(root):
        raise RuntimeError("continuation source manifest drifted")
    if protocol.get("common_initial_and_topology") != original_grid._input_manifest(root):
        raise RuntimeError("continuation input manifest drifted")
    planned = Path(str(protocol.get("planned_report"))).resolve()
    if planned.parent != (root / "reports").resolve() or not planned.name.startswith(
        f"{PROTOCOL_ID}_N254_r1_"
    ):
        raise RuntimeError("continuation report namespace drifted")
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


def _run_cell(root: Path) -> dict[str, Any]:
    paths = member.member_paths(root)
    launcher_log = paths["launcher_log"]
    audit_path = paths["audit"]
    assert isinstance(launcher_log, Path)
    assert isinstance(audit_path, Path)
    launcher_log.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(
            root
            / "src/run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_direct_continuation.py"
        ),
        "--root",
        str(root),
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
        "arm": member.ARM,
        "returncode": returncode,
        "wall_seconds": wall_seconds,
    }
    if returncode:
        row["error"] = error[-4000:]
        return row
    audit = _read_json(audit_path)
    if (
        audit.get("arm") != member.ARM
        or audit.get("barrier_fields_accessed") is not False
    ):
        raise RuntimeError("continuation member audit identity/embargo drift")
    result = Path(str(audit["result"])).resolve()
    if file_sha256(result) != str(audit["result_sha256"]):
        raise RuntimeError("continuation result hash mismatch")
    row.update(
        {
            "member_audit": str(audit_path),
            "member_audit_sha256": file_sha256(audit_path),
            "result": str(result),
            "result_sha256": file_sha256(result),
            "trajectory_record_count": int(audit["trajectory_record_count"]),
            "trajectory_complete_to_budget": bool(
                audit["trajectory_complete_to_budget"]
            ),
            "early_stop": bool(audit["early_stop"]),
            "physical_stop_reason": str(audit["physical_stop_reason"]),
            "scientific_run_converged": False,
            "precision_beyond_tightest_reached_unreachable": bool(
                audit["precision_beyond_tightest_reached_unreachable"]
            ),
            "final_optimization_NF": int(audit["final_optimization_NF"]),
            "registered_command_sha256": str(
                audit["scientific_command_sha256"]
            ),
        }
    )
    return row


def preregister(root: Path) -> Path:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stem = f"{PROTOCOL_ID}_N254_r1_{time.time_ns()}"
    preregistration = reports / f"{stem}_preregistered.json"
    planned_report = reports / f"{stem}.json"
    _write_json_exclusive(
        preregistration, build_preregistration(root, planned_report)
    )
    return preregistration


def launch(root: Path, preregistration: Path, jobs: int) -> Path:
    if jobs != 1:
        raise ValueError("formal continuation requires --jobs 1")
    preregistration = preregistration.expanduser().resolve()
    protocol = _read_json(preregistration)
    validate_preregistration(protocol, root)
    final_report = Path(str(protocol["planned_report"])).resolve()
    if final_report.exists():
        raise FileExistsError(f"refusing to overwrite {final_report}")
    started = time.time()
    try:
        row = _run_cell(root)
    except Exception as error:
        row = {"arm": member.ARM, "returncode": -1, "error": repr(error)}
    complete = int(row.get("returncode", -1)) == 0
    if complete and row.get("registered_command_sha256") != protocol[
        "registered_command"
    ]["compact_sha256"]:
        raise RuntimeError("continuation ran under an unregistered command")
    report = {
        **protocol,
        "status": "complete" if complete else "incomplete",
        "preregistration": str(preregistration),
        "preregistration_sha256": file_sha256(preregistration),
        "execution_jobs_operational_only": jobs,
        "started_unix": started,
        "finished_unix": time.time(),
        "early_stop_is_valid_completion_of_execution_not_scientific_convergence": True,
        "physical_energy_arrays_accessed_by_launcher": False,
        "barrier_fields_accessed_by_launcher": False,
        "results": [row],
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
