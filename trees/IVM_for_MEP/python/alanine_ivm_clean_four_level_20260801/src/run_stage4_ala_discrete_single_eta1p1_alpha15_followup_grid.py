#!/usr/bin/env python3
"""Preregister, then separately launch the fixed eta=1.1/alpha=15 follow-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import run_stage4_ala_discrete_single_alpha_refinement as refinement
import run_stage4_ala_discrete_single_eta1p1_alpha15_followup as member


PROTOCOL_ID = member.MEMBER_POLICY_VERSION
CANONICAL_ROOT_NAME = "alanine_ivm_clean_four_level_20260801"
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
COMMON_INITIAL_PATH = "inputs/common_initial_path_equidistant.npz"
COMMON_INITIAL_FILE_SHA256 = (
    "4a08066e1f0a89ad1d0ee6ecc442d7255d511217576d1ff65e9cb5e2164d891b"
)
COMMON_INITIAL_ARRAY_SHA256 = (
    "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
)
ETA_ONE_RESULT = (
    "runs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/members/"
    "ivm_stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "r08c10_a15_N254_discrete_b040_r1_central_state_ivm_"
    "cfg661f507a1380_budget254.npz"
)
ETA_ONE_RESULT_SHA256 = (
    "28bf7a5894337e2fc26178cc9dc9ead86f421c50972f9faf15aef34ce1830d90"
)
ETA_ONE_AUDIT = (
    "runs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/audits/"
    "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "r08c10_a15_N254_discrete_b040_r1.json"
)
ETA_ONE_AUDIT_SHA256 = (
    "65ce120aaf470cfd95ac3ab46e1cc50ceecb76afff4e6c484216f1986e77165c"
)
ETA_ONE_COMMAND = (
    "logs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/members/"
    "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "r08c10_a15_N254_discrete_b040_r1/command.json"
)
ETA_ONE_COMMAND_SHA256 = (
    "9bf0458e6d0f898ddaf76eaa9758531887c677853b4ce32b4bf98c3ea46125c8"
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
ALPHA15_REFINEMENT_CTP = (
    "reports/stage4_ala_discrete_single_alpha_refinement_matched_v1_"
    "N254_r1_1787252714040428000_iperp_ctp_readonly_v1_"
    "1787253163640931000.json"
)
ALPHA15_REFINEMENT_CTP_SHA256 = (
    "8ff9d9824eb2d3719d28f73f33b92b62e115ff9f7449bf60be054298d7ba5212"
)
SOURCE_PATHS = {
    "followup_grid": "src/run_stage4_ala_discrete_single_eta1p1_alpha15_followup_grid.py",
    "followup_member": "src/run_stage4_ala_discrete_single_eta1p1_alpha15_followup.py",
    "followup_tests": "tests/test_stage4_ala_discrete_single_eta1p1_alpha15_followup.py",
    "eta_runtime_shim": "src/run_ivm_v9_discrete_selected_eta_shim.py",
    "frozen_discrete_b040": "src/frozen_discrete_b040.py",
    "production_runner": "src/run_ivm_v9_spring_projected_production_20260730.py",
    "alpha15_refinement_wrapper": "src/run_stage4_ala_discrete_single_alpha_refinement.py",
    "coarse_wrapper_alias": "src/run_stage4_ala_discrete_single_fixedalpha.py",
    "coarse_wrapper_implementation": "src/run_stage4_ala_dsel_single_fixedalpha.py",
    "legacy_command_helper": "src/run_stage4_uniform_optimized.py",
    "ala_core": "src/ala_core.py",
}
LOCKED_SOURCE_SHA256 = {
    "eta_runtime_shim": "a2f071ea787a008ed001fd806b8952f84d8db90335b544308478f3f6ae2facf9",
    "frozen_discrete_b040": "8384a84666af463991df2e8d8a344779ae675feaddc73c12bcc82ff6ba324e22",
    "production_runner": "9abd6e96bb9e6cead380a28c849fd11c15c5fe9a8854d77fe41a991240b0510f",
    "alpha15_refinement_wrapper": "a1504d634e9e5984363b64ac23a4df8a606de007785074feab916fad86e8bf62",
    "coarse_wrapper_alias": "dd4035a6ce95a090adaffb574c806f9aba4f87f7d16744714dc564bf86134ed6",
    "coarse_wrapper_implementation": "241e90d44f64d137f63de7451c60d506307b911473c1112caf94f5701f81f845",
    "legacy_command_helper": "990be5520093af674bc44061c0c38af9169a1e207f06c33f7bea9d53ac2d7e02",
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
        raise RuntimeError(f"Hash drift: {path}")
    return {"path": str(path), "sha256": actual}


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _verify_manifest(manifest: object, name: str) -> None:
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError(f"Invalid {name}")
    for key, record in manifest.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"Invalid {name}.{key}")
        _hashed(Path(str(record["path"])), str(record["sha256"]))


def _common_initial_record(root: Path) -> dict[str, str]:
    path = root / COMMON_INITIAL_PATH
    record = _hashed(path, COMMON_INITIAL_FILE_SHA256)
    with np.load(path, allow_pickle=False) as saved:
        initial = np.ascontiguousarray(
            np.asarray(saved["initial_path"], dtype=np.float64)
        )
    array_hash = hashlib.sha256(initial.tobytes()).hexdigest()
    if array_hash != COMMON_INITIAL_ARRAY_SHA256:
        raise RuntimeError("Canonical initial-path array hash drifted")
    return {**record, "initial_path_array_sha256": array_hash}


def _decision_inputs(root: Path) -> dict[str, dict[str, str]]:
    return {
        "same_alpha_eta_one_candidate_result": _hashed(
            root / ETA_ONE_RESULT, ETA_ONE_RESULT_SHA256
        ),
        "same_alpha_eta_one_candidate_audit": _hashed(
            root / ETA_ONE_AUDIT, ETA_ONE_AUDIT_SHA256
        ),
        "same_alpha_eta_one_candidate_command": _hashed(
            root / ETA_ONE_COMMAND, ETA_ONE_COMMAND_SHA256
        ),
        "same_alpha_legacy_result": _hashed(
            root / LEGACY_RESULT, LEGACY_RESULT_SHA256
        ),
        "same_alpha_legacy_audit": _hashed(
            root / LEGACY_AUDIT, LEGACY_AUDIT_SHA256
        ),
        "same_alpha_legacy_command": _hashed(
            root / LEGACY_COMMAND, LEGACY_COMMAND_SHA256
        ),
        "official_stage4_v4": _hashed(
            root / OFFICIAL_STAGE4_V4, OFFICIAL_STAGE4_V4_SHA256
        ),
        "alpha15_refinement_ctp_snapshot": _hashed(
            root / ALPHA15_REFINEMENT_CTP, ALPHA15_REFINEMENT_CTP_SHA256
        ),
    }


def _source_manifest(root: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for name, relative in SOURCE_PATHS.items():
        records[name] = _hashed(root / relative, LOCKED_SOURCE_SHA256.get(name))
    return records


def _verify_reference_semantics(root: Path) -> None:
    expected_candidate, _ = refinement.single_run_command(
        root, 15.0, 254, "r1", "candidate"
    )
    expected_legacy, _ = refinement.single_run_command(
        root, 15.0, 254, "r1", "matched_control"
    )
    if json.loads((root / ETA_ONE_COMMAND).read_text()) != expected_candidate:
        raise RuntimeError("Pinned alpha=15 eta=1 command is not canonical")
    if json.loads((root / LEGACY_COMMAND).read_text()) != expected_legacy:
        raise RuntimeError("Pinned alpha=15 legacy command is not canonical")
    for relative, role in ((ETA_ONE_RESULT, "candidate"), (LEGACY_RESULT, "legacy")):
        with np.load(root / relative, allow_pickle=False) as saved:
            if float(np.asarray(saved["alpha"]).item()) != 15.0 or str(
                np.asarray(saved["alpha_mode"]).item()
            ) != "fixed":
                raise RuntimeError(f"Pinned alpha=15 {role} has wrong alpha")
            if np.asarray(saved["i_perp"]).shape != (254,):
                raise RuntimeError(f"Pinned alpha=15 {role} is not N254")
            if str(np.asarray(saved["initial_path_sha256"]).item()) != (
                COMMON_INITIAL_ARRAY_SHA256
            ):
                raise RuntimeError(f"Pinned alpha=15 {role} changed initial path")
            if str(np.asarray(saved["reparam_mode"]).item()) != "state_curve" or str(
                np.asarray(saved["state_curve_direct_objective_acceptance"]).item()
            ) != "per_image":
                raise RuntimeError(f"Pinned alpha=15 {role} changed reparameterization")
            if role == "candidate":
                expected = {
                    "frozen_subproblem": "discrete_b040",
                    "discrete_stop_mode": "r08c10",
                    "discrete_residual_ratio": 0.8,
                    "discrete_inner_max_steps": 10,
                    "discrete_history": 7,
                }
                for key, value in expected.items():
                    if np.asarray(saved[key]).item() != value:
                        raise RuntimeError(f"Pinned eta=1 drifted at {key}")
            elif any(
                key in saved.files
                for key in ("discrete_implementation_id", "discrete_audit_selection")
            ):
                raise RuntimeError("Pinned legacy contains discrete fields")


def _assert_no_outputs(root: Path) -> None:
    paths = member.member_paths(root)
    result_dir = paths["result_dir"]
    audit = paths["audit"]
    member_log = paths["member_log"]
    launcher_log = paths["launcher_log"]
    assert isinstance(result_dir, Path)
    assert isinstance(audit, Path)
    assert isinstance(member_log, Path)
    assert isinstance(launcher_log, Path)
    label = str(paths["label"])
    existing = sorted(
        result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget254.npz")
    )
    companion_audits = sorted(
        result_dir.glob(
            f"ivm_{label}_central_state_ivm_cfg*_budget254.audit.json"
        )
    )
    if (
        existing
        or companion_audits
        or audit.exists()
        or member_log.exists()
        or launcher_log.exists()
    ):
        target = (
            existing[0]
            if existing
            else companion_audits[0]
            if companion_audits
            else audit
        )
        raise FileExistsError(f"Refusing pre-existing eta1.1 data: {target}")


def _registered_command_hash(root: Path) -> str:
    return member.compact_command_sha256(member.single_run_command(root)[0])


def build_preregistration(root: Path, planned_report: Path) -> dict[str, object]:
    root = root.resolve()
    if root.name != CANONICAL_ROOT_NAME:
        raise RuntimeError("Follow-up must use the canonical Ala repository")
    _assert_no_outputs(root)
    _verify_reference_semantics(root)
    return {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "created_unix": time.time(),
        "planned_report": str(planned_report.resolve()),
        "study_role": (
            "prospectively_preregistered_exploratory_followup_not_independent_confirmation"
        ),
        "single_arm": "fixed post-selection eta=1.1 at fixed alpha=15",
        "fixed_eta": 1.1,
        "scheduled_or_adaptive_eta": False,
        "eta_application_stage": member.eta_shim.APPLICATION_STAGE,
        "eta_policy_id": member.eta_shim.SHIM_POLICY_ID,
        "selector_isolated_from_eta": True,
        "same_dynamic_atom_displacement_cap_reapplied": True,
        "max_step_is_safety_cap_not_eta": True,
        "energy_or_objective_backtracking_added": False,
        "extra_physical_evaluations_from_eta_shim": 0,
        "fixed_alpha": 15.0,
        "fixed_residual_ratio": 0.8,
        "fixed_max_evaluated_steps": 10,
        "fixed_lbfgs_history": 7,
        "budget_per_image": 254,
        "replicate": "r1",
        "stop_rule": "R/R0 <= 0.8 OR evaluated accepted steps == 10",
        "reparameterization_policy": "state_curve/per_image from first state",
        "scientific_command_contract": (
            "exact existing alpha15 refinement candidate argv clone; only runner "
            "changes to the locked eta shim, output-dir/run-label are routing, "
            "and exactly one fixed eta=1.1 option is appended"
        ),
        "all_non_eta_scientific_flags_frozen": True,
        "canonical_common_initial_path": _common_initial_record(root),
        "eligible_record_definition": "initial plus reparam_committed states",
        "mesh_gate": {
            "segment_length_cv_max": member.MESH_CV_MAX,
            "segment_length_ratio_max": member.MESH_RATIO_MAX,
        },
        "cost_to_precision_definition": (
            "minimum actual optimization NF at I_perp <= threshold; no interpolation"
        ),
        "reporting_anchors": list(REPORTING_ANCHORS),
        "primary_comparators": [
            "same_alpha_eta_one_candidate",
            "same_alpha_legacy",
            "official_stage4_v4",
        ],
        "primary_success_criterion": (
            "eta1.1 must weakly dominate every comparator over every reachable "
            "complete mesh-eligible Iperp CTP threshold, strictly improve at least "
            "one threshold versus each comparator, and reach a tightest eligible "
            "precision no worse than each comparator"
        ),
        "failure_rule": (
            "if the primary criterion fails, no winner/lock claim, no follow-up "
            "eta choice from this run, and no barrier unembargo"
        ),
        "post_hoc_grid_expansion_allowed": False,
        "terminal_or_raw_iperp_ranking_allowed": False,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "sparse_barrier_role": "embargoed_not_read_not_reported_not_ranked",
        "densified_energy_barrier_role": "embargoed_for_entire_followup",
        "decision_inputs": _decision_inputs(root),
        "run_source_manifest": _source_manifest(root),
        "registered_command_sha256": _registered_command_hash(root),
        "independent_member_paths_and_logs": True,
        "overwrite_or_force_allowed": False,
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
    }


def validate_preregistration(protocol: dict[str, object], root: Path) -> None:
    root = root.resolve()
    if protocol.get("status") != "preregistered" or protocol.get(
        "protocol_id"
    ) != PROTOCOL_ID:
        raise RuntimeError("Invalid eta1.1 preregistration")
    required = {
        "scheduled_or_adaptive_eta": False,
        "selector_isolated_from_eta": True,
        "same_dynamic_atom_displacement_cap_reapplied": True,
        "max_step_is_safety_cap_not_eta": True,
        "energy_or_objective_backtracking_added": False,
        "all_non_eta_scientific_flags_frozen": True,
        "post_hoc_grid_expansion_allowed": False,
        "terminal_or_raw_iperp_ranking_allowed": False,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "overwrite_or_force_allowed": False,
    }
    for key, expected in required.items():
        if protocol.get(key) is not expected:
            raise RuntimeError(f"Eta1.1 preregistration drifted at {key}")
    if (
        float(protocol["fixed_eta"]) != 1.1
        or float(protocol["fixed_alpha"]) != 15.0
        or float(protocol["fixed_residual_ratio"]) != 0.8
        or int(protocol["fixed_max_evaluated_steps"]) != 10
        or int(protocol["fixed_lbfgs_history"]) != 7
        or int(protocol["budget_per_image"]) != 254
        or str(protocol["replicate"]) != "r1"
    ):
        raise RuntimeError("Eta1.1 frozen configuration drifted")
    if protocol.get("reparameterization_policy") != (
        "state_curve/per_image from first state"
    ):
        raise RuntimeError("Eta1.1 reparameterization drifted")
    if protocol.get("eligible_record_definition") != (
        "initial plus reparam_committed states"
    ) or protocol.get("mesh_gate") != {
        "segment_length_cv_max": 0.0065,
        "segment_length_ratio_max": 1.03,
    }:
        raise RuntimeError("Eta1.1 eligibility drifted")
    if protocol.get("reporting_anchors") != list(REPORTING_ANCHORS):
        raise RuntimeError("Eta1.1 reporting anchors drifted")
    if protocol.get("primary_comparators") != [
        "same_alpha_eta_one_candidate",
        "same_alpha_legacy",
        "official_stage4_v4",
    ]:
        raise RuntimeError("Eta1.1 comparators drifted")
    planned = Path(str(protocol["planned_report"])).resolve()
    if planned.parent != (root / "reports").resolve() or not planned.name.startswith(
        f"{PROTOCOL_ID}_N254_r1_"
    ):
        raise RuntimeError("Eta1.1 planned report namespace drifted")
    if protocol["canonical_common_initial_path"] != _common_initial_record(root):
        raise RuntimeError("Eta1.1 common initial path drifted")
    _verify_manifest(protocol["decision_inputs"], "decision inputs")
    if protocol["decision_inputs"] != _decision_inputs(root):
        raise RuntimeError("Eta1.1 decision inputs drifted")
    _verify_manifest(protocol["run_source_manifest"], "run sources")
    if protocol["run_source_manifest"] != _source_manifest(root):
        raise RuntimeError("Eta1.1 run sources drifted")
    if protocol["registered_command_sha256"] != _registered_command_hash(root):
        raise RuntimeError("Eta1.1 registered command drifted")
    _verify_reference_semantics(root)
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


def _run_cell(root: Path) -> dict[str, object]:
    paths = member.member_paths(root)
    launcher_log = paths["launcher_log"]
    audit_path = paths["audit"]
    assert isinstance(launcher_log, Path)
    assert isinstance(audit_path, Path)
    launcher_log.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(root / "src/run_stage4_ala_discrete_single_eta1p1_alpha15_followup.py"),
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
    row: dict[str, object] = {
        "alpha": 15.0,
        "eta": 1.1,
        "returncode": returncode,
        "wall_seconds": wall_seconds,
    }
    if returncode:
        row["error"] = error[-4000:]
        return row
    audit = _read_json(audit_path)
    if float(audit["alpha"]) != 15.0 or float(audit["eta"]) != 1.1:
        raise RuntimeError("Eta1.1 member audit identity mismatch")
    if audit.get("barrier_fields_accessed") is not False or audit.get(
        "energy_backtracking"
    ) is not False:
        raise RuntimeError("Eta1.1 member broke its embargo")
    result = Path(str(audit["result"])).resolve()
    if file_sha256(result) != str(audit["result_sha256"]):
        raise RuntimeError("Eta1.1 result hash mismatch")
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
            "h0_selection_count": int(audit["h0_selection_count"]),
            "lbfgs_selection_count": int(audit["lbfgs_selection_count"]),
            "eta_cap_hit_count": int(audit["eta_cap_hit_count"]),
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
        raise ValueError("Single-arm formal follow-up requires --jobs 1")
    preregistration = preregistration.expanduser().resolve()
    protocol = _read_json(preregistration)
    validate_preregistration(protocol, root)
    final_report = Path(str(protocol["planned_report"])).resolve()
    if final_report.exists():
        raise FileExistsError(f"Refusing to overwrite {final_report}")
    started = time.time()
    try:
        row = _run_cell(root)
    except Exception as error:
        row = {"alpha": 15.0, "eta": 1.1, "returncode": -1, "error": repr(error)}
    complete = int(row.get("returncode", -1)) == 0
    if complete and row.get("registered_command_sha256") != protocol.get(
        "registered_command_sha256"
    ):
        raise RuntimeError("Completed row does not match registered command hash")
    report = {
        **protocol,
        "status": "complete" if complete else "incomplete",
        "preregistration": str(preregistration),
        "preregistration_sha256": file_sha256(preregistration),
        "execution_jobs_operational_only": jobs,
        "started_unix": started,
        "finished_unix": time.time(),
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
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
            raise ValueError("--jobs applies only to execution")
        print(preregister(root))
        return
    print(launch(root, args.preregistration, args.jobs))


if __name__ == "__main__":
    main()
