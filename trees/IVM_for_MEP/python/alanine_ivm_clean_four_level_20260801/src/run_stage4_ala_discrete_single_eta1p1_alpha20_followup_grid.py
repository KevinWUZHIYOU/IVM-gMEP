#!/usr/bin/env python3
"""Preregister, then separately launch the fixed eta=1.1/alpha=20 follow-up."""

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

import run_stage4_ala_discrete_single_fixedalpha as coarse_single
import run_stage4_ala_discrete_single_eta1p1_alpha20_followup as member


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
    "runs/stage4_ala_discrete_single_fixedalpha_v2/members/"
    "ivm_stage4_ala_discrete_single_fixedalpha_v2_"
    "r08c10_a20_N254_discrete_b040_r1_central_state_ivm_"
    "cfg539237df964b_budget254.npz"
)
ETA_ONE_RESULT_SHA256 = (
    "8ad8ad19f469752e2ebbd6c4b73eca12eadbfc09eb5df2d12ba5cf92e14d63ae"
)
ETA_ONE_AUDIT = (
    "runs/stage4_ala_discrete_single_fixedalpha_v2/audits/"
    "stage4_ala_discrete_single_fixedalpha_v2_"
    "r08c10_a20_N254_discrete_b040_r1.json"
)
ETA_ONE_AUDIT_SHA256 = (
    "8f71542c46c732204c908bfda3de015680d1084820ec7688ec1f454730fc8c0e"
)
ETA_ONE_COMMAND = (
    "logs/stage4_ala_discrete_single_fixedalpha_v2/"
    "stage4_ala_discrete_single_fixedalpha_v2_"
    "r08c10_a20_N254_discrete_b040_r1/command.json"
)
ETA_ONE_COMMAND_SHA256 = (
    "97284dcd99a2d7bc82a52898f1ef11cdae5009b1088a038da58a81b4b95cff01"
)
LEGACY_RESULT = (
    "runs/stage4_ala_discrete_single_fixedalpha_v2/members/"
    "ivm_stage4_ala_discrete_single_fixedalpha_v2_"
    "legacy_stop_a20_N254_legacy_r1_central_state_ivm_"
    "cfg26035919ed6a_budget254.npz"
)
LEGACY_RESULT_SHA256 = (
    "2688b7b75fb79efb11a00ad9da3f426930a5f9e0f6f25c45c138c2945feb2176"
)
LEGACY_AUDIT = (
    "runs/stage4_ala_discrete_single_fixedalpha_v2/audits/"
    "stage4_ala_discrete_single_fixedalpha_v2_"
    "legacy_stop_a20_N254_legacy_r1.json"
)
LEGACY_AUDIT_SHA256 = (
    "556ed435e62e0700b2a303ad932c22a6db5e848b5c9241a71ebbb2684f2c98e4"
)
LEGACY_COMMAND = (
    "logs/stage4_ala_discrete_single_fixedalpha_v2/"
    "stage4_ala_discrete_single_fixedalpha_v2_"
    "legacy_stop_a20_N254_legacy_r1/command.json"
)
LEGACY_COMMAND_SHA256 = (
    "441d83aa842b2aa69ba3a9b16682afbfb198152b976a1aec58a61ea156dc0a39"
)
OFFICIAL_STAGE4_V4 = (
    "runs/pareto_archive/pareto_clean_stage4_v4_a20_phaseA34_phaseB220.npz"
)
OFFICIAL_STAGE4_V4_SHA256 = (
    "09ccbf7db731c96c0beda9d5879c9dd46cae2b428db8b71820aad1c56cb574fc"
)
ALPHA15_PREREGISTRATION = (
    "reports/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_"
    "N254_r1_1787257808883811000_preregistered.json"
)
ALPHA15_PREREGISTRATION_SHA256 = (
    "dbffae0850f39128c313f2819c28b7679d5e80ef0a6f9fe59246dc43a176df1d"
)
ALPHA15_REPORT = (
    "reports/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_"
    "N254_r1_1787257808883811000.json"
)
ALPHA15_REPORT_SHA256 = (
    "fa0bf64922d91d15a48b0b6b6949d8ddb3ff109661ff011b351f8f9caee79212"
)
ALPHA15_RESULT = (
    "runs/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1/members/"
    "ivm_stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_"
    "eta1p1_r08c10_a15_N254_discrete_b040_r1_central_state_ivm_"
    "cfg02c3f3990307_budget254.npz"
)
ALPHA15_RESULT_SHA256 = (
    "bb189235868d63be23d335c0766db59e0315b82fd11448077a492966ce5eaa55"
)
ALPHA15_AUDIT = (
    "runs/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1/audits/"
    "stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_"
    "eta1p1_r08c10_a15_N254_discrete_b040_r1.json"
)
ALPHA15_AUDIT_SHA256 = (
    "058d591e5fccc4a0e7a4dee0854c4f372b7b2648d2673dd25a9bd58363eed91a"
)
ALPHA15_FORMAL_CTP = (
    "reports/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_"
    "N254_r1_1787257808883811000_iperp_ctp_readonly_v1_"
    "1787258244070171000.json"
)
ALPHA15_FORMAL_CTP_SHA256 = (
    "0ef85ac737d57c5a5b6b7a2e725eef8684e66e225934b9e8858c167fa887d9ae"
)
ALPHA15_POSTRUN_AUDIT = (
    "reports/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_"
    "N254_r1_1787257808883811000_"
    "postrun_mechanism_provenance_readonly_v1.json"
)
ALPHA15_POSTRUN_AUDIT_SHA256 = (
    "7dbd94c4627db5938d189ec24aa2fa8f5790a62122b506410b9b34c86c1ea95a"
)
SOURCE_PATHS = {
    "followup_grid": "src/run_stage4_ala_discrete_single_eta1p1_alpha20_followup_grid.py",
    "followup_member": "src/run_stage4_ala_discrete_single_eta1p1_alpha20_followup.py",
    "followup_tests": "tests/test_stage4_ala_discrete_single_eta1p1_alpha20_followup.py",
    "eta_runtime_shim": "src/run_ivm_v9_discrete_selected_eta_shim.py",
    "frozen_discrete_b040": "src/frozen_discrete_b040.py",
    "production_runner": "src/run_ivm_v9_spring_projected_production_20260730.py",
    "coarse_wrapper_alias": "src/run_stage4_ala_discrete_single_fixedalpha.py",
    "coarse_wrapper_implementation": "src/run_stage4_ala_dsel_single_fixedalpha.py",
    "legacy_command_helper": "src/run_stage4_uniform_optimized.py",
    "ala_core": "src/ala_core.py",
    "alpha15_followup_member": "src/run_stage4_ala_discrete_single_eta1p1_alpha15_followup.py",
    "alpha15_followup_grid": "src/run_stage4_ala_discrete_single_eta1p1_alpha15_followup_grid.py",
    "alpha15_formal_ctp_analyzer": "src/analyze_stage4_ala_discrete_single_eta1p1_alpha15_iperp_ctp.py",
    "alpha15_formal_ctp_test": "tests/test_analyze_stage4_ala_discrete_single_eta1p1_alpha15_iperp_ctp.py",
    "alpha15_postrun_auditor": "src/audit_stage4_ala_discrete_single_eta1p1_alpha15_postrun.py",
    "alpha15_postrun_test": "tests/test_audit_stage4_ala_discrete_single_eta1p1_alpha15_postrun.py",
}
LOCKED_SOURCE_SHA256 = {
    "eta_runtime_shim": "a2f071ea787a008ed001fd806b8952f84d8db90335b544308478f3f6ae2facf9",
    "frozen_discrete_b040": "8384a84666af463991df2e8d8a344779ae675feaddc73c12bcc82ff6ba324e22",
    "production_runner": "9abd6e96bb9e6cead380a28c849fd11c15c5fe9a8854d77fe41a991240b0510f",
    "coarse_wrapper_alias": "dd4035a6ce95a090adaffb574c806f9aba4f87f7d16744714dc564bf86134ed6",
    "coarse_wrapper_implementation": "241e90d44f64d137f63de7451c60d506307b911473c1112caf94f5701f81f845",
    "legacy_command_helper": "990be5520093af674bc44061c0c38af9169a1e207f06c33f7bea9d53ac2d7e02",
    "ala_core": "e3d9b2d472d8b9d7e0e03289dcf6a87bdff10cb8fc296bdfda911798799377bd",
    "alpha15_followup_member": "0587bb2dd438a402293ee73a2be060a6e54d82b06add6f519d19df914a780c1a",
    "alpha15_followup_grid": "0c56283c67559abf0fcf898d44faefa8e51b82481f4bc833ecf3a3d2b05b4ab6",
    "alpha15_formal_ctp_analyzer": "109a3f0b395392dc8ff8ddf5129929884fd580acce629d8a5ce3065164cb54f6",
    "alpha15_formal_ctp_test": "b343bc24e695b94d380fdca203cfb627b75299b80673eecf98e09d3510d33cfa",
    "alpha15_postrun_auditor": "a06b86d43dc8f5470947576375b195e8281a824926d787e1924c6fb0becbe62d",
    "alpha15_postrun_test": "66fcc061724f84806c32cb337208c9f7c3c2cca8f50d781d904fff32cb6fa119",
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
        "same_eta_alpha15_preregistration": _hashed(
            root / ALPHA15_PREREGISTRATION, ALPHA15_PREREGISTRATION_SHA256
        ),
        "same_eta_alpha15_report": _hashed(
            root / ALPHA15_REPORT, ALPHA15_REPORT_SHA256
        ),
        "same_eta_alpha15_result": _hashed(
            root / ALPHA15_RESULT, ALPHA15_RESULT_SHA256
        ),
        "same_eta_alpha15_member_audit": _hashed(
            root / ALPHA15_AUDIT, ALPHA15_AUDIT_SHA256
        ),
        "same_eta_alpha15_formal_ctp": _hashed(
            root / ALPHA15_FORMAL_CTP, ALPHA15_FORMAL_CTP_SHA256
        ),
        "same_eta_alpha15_postrun_audit": _hashed(
            root / ALPHA15_POSTRUN_AUDIT, ALPHA15_POSTRUN_AUDIT_SHA256
        ),
    }


def _source_manifest(root: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for name, relative in SOURCE_PATHS.items():
        records[name] = _hashed(root / relative, LOCKED_SOURCE_SHA256.get(name))
    return records


def _verify_reference_semantics(root: Path) -> None:
    expected_candidate, _ = coarse_single.single_run_command(
        root, 20.0, 254, "r1", "candidate"
    )
    expected_legacy, _ = coarse_single.single_run_command(
        root, 20.0, 254, "r1", "matched_control"
    )
    if json.loads((root / ETA_ONE_COMMAND).read_text()) != expected_candidate:
        raise RuntimeError("Pinned alpha=20 eta=1 command is not canonical")
    if json.loads((root / LEGACY_COMMAND).read_text()) != expected_legacy:
        raise RuntimeError("Pinned alpha=20 legacy command is not canonical")
    for relative, role in ((ETA_ONE_RESULT, "candidate"), (LEGACY_RESULT, "legacy")):
        with np.load(root / relative, allow_pickle=False) as saved:
            if float(np.asarray(saved["alpha"]).item()) != 20.0 or str(
                np.asarray(saved["alpha_mode"]).item()
            ) != "fixed":
                raise RuntimeError(f"Pinned alpha=20 {role} has wrong alpha")
            if np.asarray(saved["i_perp"]).shape != (254,):
                raise RuntimeError(f"Pinned alpha=20 {role} is not N254")
            if str(np.asarray(saved["initial_path_sha256"]).item()) != (
                COMMON_INITIAL_ARRAY_SHA256
            ):
                raise RuntimeError(f"Pinned alpha=20 {role} changed initial path")
            if str(np.asarray(saved["reparam_mode"]).item()) != "state_curve" or str(
                np.asarray(saved["state_curve_direct_objective_acceptance"]).item()
            ) != "per_image":
                raise RuntimeError(f"Pinned alpha=20 {role} changed reparameterization")
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

    alpha15_prereg = _read_json(root / ALPHA15_PREREGISTRATION)
    alpha15_report = _read_json(root / ALPHA15_REPORT)
    alpha15_audit = _read_json(root / ALPHA15_AUDIT)
    alpha15_ctp = _read_json(root / ALPHA15_FORMAL_CTP)
    alpha15_postrun = _read_json(root / ALPHA15_POSTRUN_AUDIT)
    alpha15_protocol = "stage4_ala_discrete_single_eta1p1_alpha15_followup_v1"
    if (
        alpha15_prereg.get("status") != "preregistered"
        or alpha15_prereg.get("protocol_id") != alpha15_protocol
        or float(alpha15_prereg.get("fixed_alpha", -1.0)) != 15.0
        or float(alpha15_prereg.get("fixed_eta", -1.0)) != 1.1
    ):
        raise RuntimeError("Pinned alpha=15 preregistration semantics drifted")
    if (
        alpha15_report.get("status") != "complete"
        or alpha15_report.get("protocol_id") != alpha15_protocol
        or float(alpha15_report.get("fixed_alpha", -1.0)) != 15.0
        or float(alpha15_report.get("fixed_eta", -1.0)) != 1.1
        or alpha15_report.get("barrier_fields_accessed_by_launcher") is not False
    ):
        raise RuntimeError("Pinned alpha=15 report semantics drifted")
    alpha15_rows = alpha15_report.get("results")
    if not isinstance(alpha15_rows, list) or len(alpha15_rows) != 1:
        raise RuntimeError("Pinned alpha=15 report is not single-arm")
    alpha15_row = alpha15_rows[0]
    if not isinstance(alpha15_row, dict) or (
        Path(str(alpha15_row.get("result"))).resolve()
        != (root / ALPHA15_RESULT).resolve()
        or str(alpha15_row.get("result_sha256")) != ALPHA15_RESULT_SHA256
        or Path(str(alpha15_row.get("member_audit"))).resolve()
        != (root / ALPHA15_AUDIT).resolve()
        or str(alpha15_row.get("member_audit_sha256")) != ALPHA15_AUDIT_SHA256
    ):
        raise RuntimeError("Pinned alpha=15 report provenance drifted")
    alpha15_command = alpha15_audit.get("command")
    alpha20_command = member.single_run_command(root)[0]
    if not isinstance(alpha15_command, list):
        raise RuntimeError("Pinned alpha=15 audit lacks its command")
    alpha15_normalized = [str(token) for token in alpha15_command]
    alpha20_normalized = list(alpha20_command)
    for command in (alpha15_normalized, alpha20_normalized):
        options = coarse_single.command_options(command)
        if options.get(member.eta_shim.ETA_FLAG) != member.ETA_TEXT:
            raise RuntimeError("Same-eta alpha control changed eta")
        for flag in ("--alpha", "--output-dir", "--run-label"):
            command[command.index(flag) + 1] = f"<{flag}>"
    if alpha15_normalized != alpha20_normalized:
        raise RuntimeError(
            "Alpha=15 and alpha=20 eta=1.1 commands differ beyond alpha/routing"
        )
    with np.load(root / ALPHA15_RESULT, allow_pickle=False) as saved:
        alpha15_allowlist = {
            "alpha",
            "alpha_mode",
            "active_alpha",
            "i_perp",
            "initial_path_sha256",
            "reparam_mode",
            "state_curve_direct_objective_acceptance",
            "frozen_subproblem",
            "discrete_stop_mode",
            "discrete_residual_ratio",
            "discrete_inner_max_steps",
            "discrete_history",
            "discrete_step_multiplier_eta",
            "discrete_step_multiplier_energy_backtracking",
        }
        if any("barrier" in key.lower() for key in alpha15_allowlist):
            raise AssertionError("Alpha=15 control allowlist contains barrier")
        if not alpha15_allowlist.issubset(saved.files):
            raise RuntimeError("Pinned alpha=15 result is incomplete")
        if (
            float(np.asarray(saved["alpha"]).item()) != 15.0
            or str(np.asarray(saved["alpha_mode"]).item()) != "fixed"
            or not np.all(np.asarray(saved["active_alpha"], float) == 15.0)
            or np.asarray(saved["i_perp"]).shape != (254,)
            or str(np.asarray(saved["initial_path_sha256"]).item())
            != COMMON_INITIAL_ARRAY_SHA256
            or str(np.asarray(saved["reparam_mode"]).item()) != "state_curve"
            or str(
                np.asarray(
                    saved["state_curve_direct_objective_acceptance"]
                ).item()
            )
            != "per_image"
            or str(np.asarray(saved["frozen_subproblem"]).item())
            != "discrete_b040"
            or str(np.asarray(saved["discrete_stop_mode"]).item()) != "r08c10"
            or float(np.asarray(saved["discrete_residual_ratio"]).item()) != 0.8
            or int(np.asarray(saved["discrete_inner_max_steps"]).item()) != 10
            or int(np.asarray(saved["discrete_history"]).item()) != 7
            or float(np.asarray(saved["discrete_step_multiplier_eta"]).item())
            != 1.1
            or bool(
                np.asarray(
                    saved["discrete_step_multiplier_energy_backtracking"]
                ).item()
            )
        ):
            raise RuntimeError("Pinned alpha=15 result semantics drifted")
    if (
        float(alpha15_ctp.get("fixed_alpha", -1.0)) != 15.0
        or float(alpha15_ctp.get("fixed_eta", -1.0)) != 1.1
        or alpha15_ctp.get("barrier_data_accessed") is not False
        or alpha15_ctp.get("barrier_computation_or_densification_performed")
        is not False
    ):
        raise RuntimeError("Pinned alpha=15 formal CTP semantics drifted")
    if (
        alpha15_postrun.get("status") != "passed"
        or alpha15_postrun.get("fixed_alpha_15_throughout") is not True
        or alpha15_postrun.get("fixed_r08_k10_m7_throughout") is not True
        or alpha15_postrun.get("energy_or_objective_fallback_used") is not False
        or alpha15_postrun.get("barrier_metric_values_accessed") is not False
        or alpha15_postrun.get("barrier_computation_or_ranking_performed")
        is not False
    ):
        raise RuntimeError("Pinned alpha=15 postrun audit semantics drifted")


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
        "single_arm": "fixed post-selection eta=1.1 at fixed alpha=20",
        "fixed_eta": 1.1,
        "scheduled_or_adaptive_eta": False,
        "eta_application_stage": member.eta_shim.APPLICATION_STAGE,
        "eta_policy_id": member.eta_shim.SHIM_POLICY_ID,
        "selector_isolated_from_eta": True,
        "same_dynamic_atom_displacement_cap_reapplied": True,
        "max_step_is_safety_cap_not_eta": True,
        "energy_or_objective_backtracking_added": False,
        "extra_physical_evaluations_from_eta_shim": 0,
        "fixed_alpha": 20.0,
        "fixed_residual_ratio": 0.8,
        "fixed_max_evaluated_steps": 10,
        "fixed_lbfgs_history": 7,
        "budget_per_image": 254,
        "replicate": "r1",
        "stop_rule": "R/R0 <= 0.8 OR evaluated accepted steps == 10",
        "reparameterization_policy": "state_curve/per_image from first state",
        "scientific_command_contract": (
            "exact existing alpha20 fixedalpha_v2 candidate argv clone; only runner "
            "changes to the locked eta shim, output-dir/run-label are routing, "
            "and exactly one fixed eta=1.1 option is appended"
        ),
        "all_non_eta_scientific_flags_frozen": True,
        "same_eta_alpha15_control_in_primary_success_gate": False,
        "same_eta_alpha15_alpha_only_difference_verified": True,
        "same_eta_alpha15_control_role": (
            "secondary strict control-variable alpha comparison; alpha is the only "
            "scientific argv difference after normalizing routing"
        ),
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
            "precision no worse than each comparator; alpha15 eta1.1 is reported "
            "as a same-eta alpha control and does not alter this three-comparator gate"
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
        "same_eta_alpha15_control_in_primary_success_gate": False,
        "same_eta_alpha15_alpha_only_difference_verified": True,
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
        or float(protocol["fixed_alpha"]) != 20.0
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
    if protocol.get("same_eta_alpha15_control_role") != (
        "secondary strict control-variable alpha comparison; alpha is the only "
        "scientific argv difference after normalizing routing"
    ):
        raise RuntimeError("Eta1.1 alpha-control role drifted")
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
        str(root / "src/run_stage4_ala_discrete_single_eta1p1_alpha20_followup.py"),
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
        "alpha": 20.0,
        "eta": 1.1,
        "returncode": returncode,
        "wall_seconds": wall_seconds,
    }
    if returncode:
        row["error"] = error[-4000:]
        return row
    audit = _read_json(audit_path)
    if float(audit["alpha"]) != 20.0 or float(audit["eta"]) != 1.1:
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
        row = {"alpha": 20.0, "eta": 1.1, "returncode": -1, "error": repr(error)}
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
