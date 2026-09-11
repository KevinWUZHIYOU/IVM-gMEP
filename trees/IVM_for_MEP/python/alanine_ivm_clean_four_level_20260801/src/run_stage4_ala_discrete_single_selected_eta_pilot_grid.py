#!/usr/bin/env python3
"""Preregister, then separately launch the Ala selected-step eta pilot."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import run_ivm_v9_discrete_selected_eta_shim as eta_shim
import run_stage4_ala_discrete_single_selected_eta_pilot as member


PROTOCOL_ID = "stage4_ala_discrete_single_selected_eta_pilot_v1"
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
OLD_ETA_ONE_RESULT = member.ETA_ONE_ORACLE
OLD_ETA_ONE_RESULT_SHA256 = member.ETA_ONE_ORACLE_SHA256
OLD_ETA_ONE_AUDIT = (
    "runs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/audits/"
    "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "r08c10_a12p5_N254_discrete_b040_r1.json"
)
OLD_ETA_ONE_AUDIT_SHA256 = (
    "1cf4ec9619b41f7f5d3dc39e007e0d7fc95323f78d6900befa64f365f3b5bc5f"
)
OLD_ETA_ONE_COMMAND = (
    "logs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/members/"
    "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "r08c10_a12p5_N254_discrete_b040_r1/command.json"
)
OLD_ETA_ONE_COMMAND_SHA256 = (
    "628c7ca2d276bb48e46bbe903a1ee8ee9e5416cf9c032f69ecf1a82ffcfdf0ac"
)
LEGACY_RESULT = (
    "runs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/members/"
    "ivm_stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "legacy_stop_a12p5_N254_legacy_r1_central_state_ivm_"
    "cfg706082ab7a04_budget254.npz"
)
LEGACY_RESULT_SHA256 = (
    "236c9e8f623fa297619d403163a33419a09f3069ef3d151cddd9dafb9cd11c70"
)
LEGACY_AUDIT = (
    "runs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/audits/"
    "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "legacy_stop_a12p5_N254_legacy_r1.json"
)
LEGACY_AUDIT_SHA256 = (
    "7783ba8a42491ce8d23f7b321ee310a7fd47b48d1df23e4e42d24b30a211d2db"
)
LEGACY_COMMAND = (
    "logs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/members/"
    "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "legacy_stop_a12p5_N254_legacy_r1/command.json"
)
LEGACY_COMMAND_SHA256 = (
    "db1c35940aea1e9d94a35ff32eab186c7af8e38d3453935b34ea833810783131"
)
OFFICIAL_STAGE4_V4 = (
    "runs/pareto_archive/pareto_clean_stage4_v4_a20_phaseA34_phaseB220.npz"
)
OFFICIAL_STAGE4_V4_SHA256 = (
    "09ccbf7db731c96c0beda9d5879c9dd46cae2b428db8b71820aad1c56cb574fc"
)
SOURCE_PATHS = {
    "eta_grid_launcher": "src/run_stage4_ala_discrete_single_selected_eta_pilot_grid.py",
    "eta_member_wrapper": "src/run_stage4_ala_discrete_single_selected_eta_pilot.py",
    "eta_runtime_shim": "src/run_ivm_v9_discrete_selected_eta_shim.py",
    "eta_tests": "tests/test_stage4_ala_discrete_single_selected_eta_pilot.py",
    "frozen_discrete_b040": "src/frozen_discrete_b040.py",
    "production_runner": "src/run_ivm_v9_spring_projected_production_20260730.py",
    "frozen_alpha_refinement_wrapper": "src/run_stage4_ala_discrete_single_alpha_refinement.py",
    "frozen_single_wrapper": "src/run_stage4_ala_discrete_single_fixedalpha.py",
    "legacy_command_helper": "src/run_stage4_uniform_optimized.py",
    "ala_core": "src/ala_core.py",
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
        "old_eta_one_identity_oracle_result": _hashed(
            root / OLD_ETA_ONE_RESULT, OLD_ETA_ONE_RESULT_SHA256
        ),
        "old_eta_one_identity_oracle_audit": _hashed(
            root / OLD_ETA_ONE_AUDIT, OLD_ETA_ONE_AUDIT_SHA256
        ),
        "old_eta_one_identity_oracle_command": _hashed(
            root / OLD_ETA_ONE_COMMAND, OLD_ETA_ONE_COMMAND_SHA256
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
    }


def _source_manifest(root: Path) -> dict[str, dict[str, str]]:
    return {name: _hashed(root / path) for name, path in SOURCE_PATHS.items()}


def _verify_reference_semantics(root: Path) -> None:
    inputs = _decision_inputs(root)
    old = Path(inputs["old_eta_one_identity_oracle_result"]["path"])
    legacy = Path(inputs["same_alpha_legacy_result"]["path"])
    with np.load(old, allow_pickle=False) as saved:
        if float(np.asarray(saved["alpha"]).item()) != 12.5 or str(
            np.asarray(saved["alpha_mode"]).item()
        ) != "fixed":
            raise RuntimeError("Old eta=1 oracle is not fixed alpha=12.5")
        if str(np.asarray(saved["frozen_subproblem"]).item()) != "discrete_b040":
            raise RuntimeError("Old eta=1 oracle is not discrete_b040")
        if str(np.asarray(saved["discrete_stop_mode"]).item()) != "r08c10":
            raise RuntimeError("Old eta=1 oracle changed stop mode")
        if float(np.asarray(saved["discrete_residual_ratio"]).item()) != 0.8:
            raise RuntimeError("Old eta=1 oracle changed residual ratio")
        if int(np.asarray(saved["discrete_inner_max_steps"]).item()) != 10:
            raise RuntimeError("Old eta=1 oracle changed evaluated-step cap")
        if int(np.asarray(saved["discrete_history"]).item()) != 7:
            raise RuntimeError("Old eta=1 oracle changed L-BFGS history")
        if np.asarray(saved["i_perp"]).shape != (254,):
            raise RuntimeError("Old eta=1 oracle is not N254")
        if str(np.asarray(saved["initial_path_sha256"]).item()) != (
            COMMON_INITIAL_ARRAY_SHA256
        ):
            raise RuntimeError("Old eta=1 oracle used another initial path")
    with np.load(legacy, allow_pickle=False) as saved:
        if float(np.asarray(saved["alpha"]).item()) != 12.5 or str(
            np.asarray(saved["alpha_mode"]).item()
        ) != "fixed":
            raise RuntimeError("Legacy comparator is not fixed alpha=12.5")
        if str(np.asarray(saved["solver"]).item()) != "state_ivm" or str(
            np.asarray(saved["method"]).item()
        ) != "IVM":
            raise RuntimeError("Legacy comparator is not the registered IVM control")
        if any(
            key in saved.files
            for key in ("frozen_subproblem", "discrete_implementation_id")
        ):
            raise RuntimeError("Legacy comparator contains discrete-candidate fields")
        if np.asarray(saved["i_perp"]).shape != (254,):
            raise RuntimeError("Legacy comparator is not N254")
        if str(np.asarray(saved["initial_path_sha256"]).item()) != (
            COMMON_INITIAL_ARRAY_SHA256
        ):
            raise RuntimeError("Legacy comparator used another initial path")


def _assert_no_outputs(root: Path) -> None:
    for eta in member.ETA_VALUES:
        paths = member.member_paths(root, eta)
        result_dir, audit, member_log, launcher_log = (
            paths["result_dir"],
            paths["audit"],
            paths["member_log"],
            paths["launcher_log"],
        )
        assert isinstance(result_dir, Path)
        assert isinstance(audit, Path)
        assert isinstance(member_log, Path)
        assert isinstance(launcher_log, Path)
        label = str(paths["label"])
        existing = sorted(
            result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget254.npz")
        )
        if existing or audit.exists() or member_log.exists() or launcher_log.exists():
            target = existing[0] if existing else audit
            raise FileExistsError(f"Refusing pre-existing eta pilot data: {target}")


def _registered_command_hashes(root: Path) -> dict[str, str]:
    return {
        member.eta_tag(eta): member.compact_command_sha256(
            member.single_run_command(root, eta)[0]
        )
        for eta in member.ETA_VALUES
    }


def build_preregistration(
    root: Path, planned_report: Path
) -> dict[str, object]:
    root = root.resolve()
    if root.name != CANONICAL_ROOT_NAME:
        raise RuntimeError("Eta pilot must use the canonical Ala repository")
    if tuple(member.ETA_VALUES) != (1.0, 1.25, 1.5, 2.0) or tuple(
        eta_shim.PILOT_ETA_VALUES
    ) != tuple(member.ETA_VALUES):
        raise RuntimeError("Eta grid drifted")
    _assert_no_outputs(root)
    _verify_reference_semantics(root)
    return {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "created_unix": time.time(),
        "planned_report": str(planned_report.resolve()),
        "one_factor_pilot": "post-selection selected-step multiplier eta",
        "eta_grid": list(member.ETA_VALUES),
        "eta_member_tags": [member.eta_tag(eta) for eta in member.ETA_VALUES],
        "same_build_eta_one_control_is_newly_run": True,
        "old_eta_one_is_identity_oracle_not_reused_as_control": True,
        "eta_one_whole_run_parity_gate_required": True,
        "eta_application_stage": eta_shim.APPLICATION_STAGE,
        "eta_policy_id": eta_shim.SHIM_POLICY_ID,
        "selector_isolated_from_eta": True,
        "same_dynamic_atom_displacement_cap_reapplied": True,
        "max_step_is_safety_cap_not_eta": True,
        "energy_or_objective_backtracking_added": False,
        "extra_physical_evaluations_from_eta_shim": 0,
        "fixed_alpha": 12.5,
        "fixed_residual_ratio": 0.8,
        "fixed_max_evaluated_steps": 10,
        "fixed_lbfgs_history": 7,
        "budget_per_image": 254,
        "replicate": "r1",
        "stop_rule": "R/R0 <= 0.8 OR evaluated accepted steps == 10",
        "reparameterization_policy": "state_curve/per_image from first state",
        "scientific_command_contract": (
            "exact alpha12.5/r0.8/k10/N254 candidate argv clone; only runner "
            "changes to the registered runtime shim, output-dir/run-label are "
            "routing, and exactly one post-selection eta option is appended"
        ),
        "all_non_eta_scientific_flags_frozen": True,
        "single_run_from_canonical_common_path": True,
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
        "formal_selection_rule": (
            "full-curve Iperp CTP on complete mesh-eligible states; each eta>1 "
            "is compared separately with the newly-run same-build eta=1 control, "
            "same-alpha legacy, and official Stage-IV v4"
        ),
        "primary_success_criterion": (
            "an eta>1 must have no reachable-threshold loss, no comparator-tightest-"
            "precision loss, and at least one strict NF win versus each of all "
            "three comparators"
        ),
        "multiple_primary_winner_tiebreak": [
            "lexicographically smallest candidate-minus-eta1 NF vector tight-to-loose",
            "lexicographically smallest candidate-minus-official NF vector tight-to-loose",
            "lexicographically smallest candidate-minus-legacy NF vector tight-to-loose",
            "smallest eta only as an exact deterministic tie-break",
        ],
        "no_primary_dominator_rule": {
            "eta_lock_allowed": False,
            "winner_claim_allowed": False,
            "post_hoc_eta_grid_expansion_allowed": False,
            "barrier_unembargo_allowed": False,
            "retain_eta_one_as_control": True,
        },
        "terminal_or_raw_iperp_ranking_allowed": False,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "sparse_barrier_role": "embargoed_not_read_not_reported_not_ranked",
        "densified_energy_barrier_role": "embargoed_until_eta_primary_lock",
        "decision_inputs": _decision_inputs(root),
        "run_source_manifest": _source_manifest(root),
        "registered_command_hashes": _registered_command_hashes(root),
        "independent_member_paths_and_logs": True,
        "overwrite_or_force_allowed": False,
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
    }


def validate_preregistration(protocol: dict[str, object], root: Path) -> None:
    root = root.resolve()
    if protocol.get("status") != "preregistered" or protocol.get(
        "protocol_id"
    ) != PROTOCOL_ID:
        raise RuntimeError("Invalid eta pilot preregistration")
    if tuple(float(value) for value in protocol["eta_grid"]) != (
        1.0,
        1.25,
        1.5,
        2.0,
    ):
        raise RuntimeError("Preregistered eta grid drifted")
    required = {
        "same_build_eta_one_control_is_newly_run": True,
        "old_eta_one_is_identity_oracle_not_reused_as_control": True,
        "eta_one_whole_run_parity_gate_required": True,
        "selector_isolated_from_eta": True,
        "same_dynamic_atom_displacement_cap_reapplied": True,
        "max_step_is_safety_cap_not_eta": True,
        "energy_or_objective_backtracking_added": False,
        "all_non_eta_scientific_flags_frozen": True,
        "terminal_or_raw_iperp_ranking_allowed": False,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "overwrite_or_force_allowed": False,
    }
    for key, expected in required.items():
        if protocol.get(key) is not expected:
            raise RuntimeError(f"Eta preregistration drifted at {key}")
    if (
        float(protocol["fixed_alpha"]) != 12.5
        or float(protocol["fixed_residual_ratio"]) != 0.8
        or int(protocol["fixed_max_evaluated_steps"]) != 10
        or int(protocol["fixed_lbfgs_history"]) != 7
        or int(protocol["budget_per_image"]) != 254
        or str(protocol["replicate"]) != "r1"
    ):
        raise RuntimeError("Eta frozen configuration drifted")
    if protocol.get("reparameterization_policy") != (
        "state_curve/per_image from first state"
    ):
        raise RuntimeError("Eta reparameterization policy drifted")
    if protocol.get("eligible_record_definition") != (
        "initial plus reparam_committed states"
    ) or protocol.get("mesh_gate") != {
        "segment_length_cv_max": 0.0065,
        "segment_length_ratio_max": 1.03,
    }:
        raise RuntimeError("Eta eligibility rule drifted")
    if protocol.get("cost_to_precision_definition") != (
        "minimum actual optimization NF at I_perp <= threshold; no interpolation"
    ) or protocol.get("reporting_anchors") != list(REPORTING_ANCHORS):
        raise RuntimeError("Eta CTP rule drifted")
    no_winner = protocol.get("no_primary_dominator_rule")
    if not isinstance(no_winner, dict) or no_winner != {
        "eta_lock_allowed": False,
        "winner_claim_allowed": False,
        "post_hoc_eta_grid_expansion_allowed": False,
        "barrier_unembargo_allowed": False,
        "retain_eta_one_as_control": True,
    }:
        raise RuntimeError("Eta no-dominator rule drifted")
    planned = Path(str(protocol["planned_report"])).resolve()
    if planned.parent != (root / "reports").resolve() or not planned.name.startswith(
        f"{PROTOCOL_ID}_N254_r1_"
    ):
        raise RuntimeError("Eta planned report namespace drifted")
    if protocol["canonical_common_initial_path"] != _common_initial_record(root):
        raise RuntimeError("Eta common initial path drifted")
    _verify_manifest(protocol["decision_inputs"], "decision inputs")
    if protocol["decision_inputs"] != _decision_inputs(root):
        raise RuntimeError("Eta decision inputs drifted")
    _verify_manifest(protocol["run_source_manifest"], "run sources")
    if protocol["run_source_manifest"] != _source_manifest(root):
        raise RuntimeError("Eta run sources drifted")
    if protocol["registered_command_hashes"] != _registered_command_hashes(root):
        raise RuntimeError("Registered eta commands drifted")
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


def _run_cell(root: Path, eta: float) -> dict[str, object]:
    paths = member.member_paths(root, eta)
    launcher_log, audit_path = paths["launcher_log"], paths["audit"]
    assert isinstance(launcher_log, Path)
    assert isinstance(audit_path, Path)
    launcher_log.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(root / "src/run_stage4_ala_discrete_single_selected_eta_pilot.py"),
        "--root",
        str(root),
        "--eta",
        member.ETA_TEXT[eta],
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
        "eta": eta,
        "eta_tag": member.eta_tag(eta),
        "returncode": returncode,
        "wall_seconds": wall_seconds,
    }
    if returncode:
        row["error"] = error[-4000:]
        return row
    audit = _read_json(audit_path)
    if float(audit["eta"]) != eta:
        raise RuntimeError("Eta member audit identity mismatch")
    if audit.get("barrier_fields_accessed") is not False or audit.get(
        "energy_backtracking"
    ) is not False:
        raise RuntimeError("Eta member broke its scientific embargo")
    if eta == 1.0 and audit.get(
        "eta_one_whole_run_parity_or_not_applicable"
    ) is not True:
        raise RuntimeError("New eta=1 run failed the parity gate")
    result = Path(str(audit["result"])).resolve()
    if file_sha256(result) != str(audit["result_sha256"]):
        raise RuntimeError("Eta member result hash mismatch")
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
            "eta_one_parity_gate_passed": bool(
                eta == 1.0
                and audit["eta_one_whole_run_parity_or_not_applicable"]
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
    if jobs < 1:
        raise ValueError("jobs must be positive")
    preregistration = preregistration.expanduser().resolve()
    protocol = _read_json(preregistration)
    validate_preregistration(protocol, root)
    final_report = Path(str(protocol["planned_report"])).resolve()
    if final_report.exists():
        raise FileExistsError(f"Refusing to overwrite {final_report}")
    rows: list[dict[str, object]] = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_run_cell, root, eta): eta for eta in member.ETA_VALUES
        }
        for future in concurrent.futures.as_completed(futures):
            eta = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = {
                    "eta": eta,
                    "eta_tag": member.eta_tag(eta),
                    "returncode": -1,
                    "error": repr(error),
                }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    rows.sort(key=lambda row: float(row["eta"]))
    complete = all(int(row.get("returncode", -1)) == 0 for row in rows)
    eta_one = next((row for row in rows if float(row["eta"]) == 1.0), None)
    complete = bool(
        complete
        and eta_one is not None
        and eta_one.get("eta_one_parity_gate_passed") is True
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
            raise ValueError("--jobs applies only to execution")
        print(preregister(root))
        return
    print(launch(root, args.preregistration, args.jobs))


if __name__ == "__main__":
    main()
