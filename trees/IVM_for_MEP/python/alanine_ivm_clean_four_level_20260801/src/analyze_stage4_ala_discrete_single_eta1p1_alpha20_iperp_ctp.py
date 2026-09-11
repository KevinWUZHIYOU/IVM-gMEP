#!/usr/bin/env python3
"""Barrier-blind I_perp CTP audit for fixed eta=1.1, alpha=20.

The formal candidate is compared against the three preregistered primary
comparators (same-alpha eta=1, same-alpha legacy, and official Stage-IV v4).
The already completed eta=1.1/alpha=15 arm is reported separately as a
strict control-variable alpha comparison and cannot affect the primary gate.

Only observed ``total_internal_joint_evaluations`` values are used.  Eligible
records are the initial state or a reparameterization-committed state that
also passes the frozen mesh gate.  There is no interpolation and candidate
unreachability is a formal loss.  No energy or barrier NPZ member is indexed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

import analyze_stage4_ala_discrete_single_alpha_refinement_iperp_ctp as base
import analyze_stage4_ala_discrete_single_eta1p1_alpha15_iperp_ctp as alpha15
import analyze_stage4_ala_discrete_single_selected_eta_iperp_ctp as eta_base


ANALYSIS_ID = "stage4_ala_discrete_single_eta1p1_alpha20_iperp_ctp_v1"
PROTOCOL_ID = "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1"
ETA = 1.1
CANONICAL_ALPHA = 20.0
SECONDARY_ALPHA = 15.0
CANONICAL_RATIO = 0.8
CANONICAL_INNER_CAP = 10
CANONICAL_HISTORY = 7
CANONICAL_BUDGET = 254
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
OFFICIAL_DOMAIN_MAX = 0.007
ETA_FLAG = "--discrete-step-multiplier-eta"
ROUTING_OPTIONS = frozenset({"--output-dir", "--run-label"})
REPORTING_ANCHORS = eta_base.REPORTING_ANCHORS

EXPECTED_REPORT_SHA256 = (
    "ca041bf07ee83f53d8ce09ffd39f908a8742247ffca4be79a4a6b67f76eb9f09"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "00a724380da87017debefa8c595f811a7f0cbaa4e1468001188e66ff33f684b3"
)
EXPECTED_CANDIDATE_RESULT_SHA256 = (
    "076144572af9c39f6a670f6323562eb6591e3d66d6986736b3b795cc7b4f5a45"
)
EXPECTED_CANDIDATE_AUDIT_SHA256 = (
    "ad503a2420b56327f244cb0a9a11c3b75ed0a623d05699896ed823f869a437fc"
)
EXPECTED_REGISTERED_COMMAND_SHA256 = (
    "8a9cdbea057c2714b2e6de2332bd1f53170d8925cc6972a8b8907095e16d25c7"
)

EXPECTED_COMMON_INITIAL_FILE_SHA256 = (
    "4a08066e1f0a89ad1d0ee6ecc442d7255d511217576d1ff65e9cb5e2164d891b"
)
EXPECTED_COMMON_INITIAL_ARRAY_SHA256 = (
    "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
)
EXPECTED_ETA_ONE_RESULT_SHA256 = (
    "8ad8ad19f469752e2ebbd6c4b73eca12eadbfc09eb5df2d12ba5cf92e14d63ae"
)
EXPECTED_ETA_ONE_AUDIT_SHA256 = (
    "8f71542c46c732204c908bfda3de015680d1084820ec7688ec1f454730fc8c0e"
)
EXPECTED_ETA_ONE_COMMAND_FILE_SHA256 = (
    "97284dcd99a2d7bc82a52898f1ef11cdae5009b1088a038da58a81b4b95cff01"
)
EXPECTED_ETA_ONE_COMMAND_SHA256 = (
    "42cff1cd9540358e4aee3b34c6fb2af3a7cebcdb67839704b8b770697f7ba30e"
)
EXPECTED_LEGACY_RESULT_SHA256 = (
    "2688b7b75fb79efb11a00ad9da3f426930a5f9e0f6f25c45c138c2945feb2176"
)
EXPECTED_LEGACY_AUDIT_SHA256 = (
    "556ed435e62e0700b2a303ad932c22a6db5e848b5c9241a71ebbb2684f2c98e4"
)
EXPECTED_LEGACY_COMMAND_FILE_SHA256 = (
    "441d83aa842b2aa69ba3a9b16682afbfb198152b976a1aec58a61ea156dc0a39"
)
EXPECTED_LEGACY_COMMAND_SHA256 = (
    "fb27da50ead126cfc07bf8a09e1ac6ceaeed8bef8670f9b7167e459c83771d0f"
)
EXPECTED_OFFICIAL_ARCHIVE_SHA256 = (
    "09ccbf7db731c96c0beda9d5879c9dd46cae2b428db8b71820aad1c56cb574fc"
)

EXPECTED_ALPHA15_PREREGISTRATION_SHA256 = (
    "dbffae0850f39128c313f2819c28b7679d5e80ef0a6f9fe59246dc43a176df1d"
)
EXPECTED_ALPHA15_REPORT_SHA256 = (
    "fa0bf64922d91d15a48b0b6b6949d8ddb3ff109661ff011b351f8f9caee79212"
)
EXPECTED_ALPHA15_RESULT_SHA256 = (
    "bb189235868d63be23d335c0766db59e0315b82fd11448077a492966ce5eaa55"
)
EXPECTED_ALPHA15_AUDIT_SHA256 = (
    "058d591e5fccc4a0e7a4dee0854c4f372b7b2648d2673dd25a9bd58363eed91a"
)
EXPECTED_ALPHA15_FORMAL_CTP_SHA256 = (
    "0ef85ac737d57c5a5b6b7a2e725eef8684e66e225934b9e8858c167fa887d9ae"
)
EXPECTED_ALPHA15_POSTRUN_AUDIT_SHA256 = (
    "7dbd94c4627db5938d189ec24aa2fa8f5790a62122b506410b9b34c86c1ea95a"
)

# These post-run files were not decision variables.  Pinning them here makes
# the independent audit chain complete without decoding the runner companion
# audit, which may contain sparse barrier diagnostics.
EXPECTED_AUXILIARY_ARTIFACTS = {
    "runner_companion_audit": (
        "runs/stage4_ala_discrete_single_eta1p1_alpha20_followup_v1/members/"
        "ivm_stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_eta1p1_"
        "r08c10_a20_N254_discrete_b040_r1_central_state_ivm_"
        "cfg72a2d169845b_budget254.audit.json",
        "77641bd0a2a16d6cab4b7d6e2737b046563693241c0da63361621691690e72d3",
    ),
    "member_command": (
        "logs/stage4_ala_discrete_single_eta1p1_alpha20_followup_v1/members/"
        "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_eta1p1_"
        "r08c10_a20_N254_discrete_b040_r1/command.json",
        "d734592e10fc7e934449ed61bd6bb0badab4b8078c32f54b63f765b9d5b58bd3",
    ),
    "launcher_command": (
        "logs/stage4_ala_discrete_single_eta1p1_alpha20_followup_v1/launcher/"
        "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_eta1p1_"
        "r08c10_a20_N254_discrete_b040_r1/command.json",
        "81b3f5e751f1ee2de80257eefd8f322ae0c0a14f4ce1f50d21746003ee02d058",
    ),
    "member_stdout": (
        "logs/stage4_ala_discrete_single_eta1p1_alpha20_followup_v1/members/"
        "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_eta1p1_"
        "r08c10_a20_N254_discrete_b040_r1/stdout.txt",
        "7e39dbb31ef7f385c2902847ae67c9bf51acc9f5a21fd30073e865d8b8eeb8e6",
    ),
    "member_stderr": (
        "logs/stage4_ala_discrete_single_eta1p1_alpha20_followup_v1/members/"
        "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_eta1p1_"
        "r08c10_a20_N254_discrete_b040_r1/stderr.txt",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "launcher_stdout": (
        "logs/stage4_ala_discrete_single_eta1p1_alpha20_followup_v1/launcher/"
        "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_eta1p1_"
        "r08c10_a20_N254_discrete_b040_r1/stdout.txt",
        "ad503a2420b56327f244cb0a9a11c3b75ed0a623d05699896ed823f869a437fc",
    ),
    "launcher_stderr": (
        "logs/stage4_ala_discrete_single_eta1p1_alpha20_followup_v1/launcher/"
        "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_eta1p1_"
        "r08c10_a20_N254_discrete_b040_r1/stderr.txt",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
}

EXPECTED_DECISION_INPUT_HASHES = {
    "same_alpha_eta_one_candidate_result": EXPECTED_ETA_ONE_RESULT_SHA256,
    "same_alpha_eta_one_candidate_audit": EXPECTED_ETA_ONE_AUDIT_SHA256,
    "same_alpha_eta_one_candidate_command": EXPECTED_ETA_ONE_COMMAND_FILE_SHA256,
    "same_alpha_legacy_result": EXPECTED_LEGACY_RESULT_SHA256,
    "same_alpha_legacy_audit": EXPECTED_LEGACY_AUDIT_SHA256,
    "same_alpha_legacy_command": EXPECTED_LEGACY_COMMAND_FILE_SHA256,
    "official_stage4_v4": EXPECTED_OFFICIAL_ARCHIVE_SHA256,
    "same_eta_alpha15_preregistration": EXPECTED_ALPHA15_PREREGISTRATION_SHA256,
    "same_eta_alpha15_report": EXPECTED_ALPHA15_REPORT_SHA256,
    "same_eta_alpha15_result": EXPECTED_ALPHA15_RESULT_SHA256,
    "same_eta_alpha15_member_audit": EXPECTED_ALPHA15_AUDIT_SHA256,
    "same_eta_alpha15_formal_ctp": EXPECTED_ALPHA15_FORMAL_CTP_SHA256,
    "same_eta_alpha15_postrun_audit": EXPECTED_ALPHA15_POSTRUN_AUDIT_SHA256,
}
EXPECTED_DECISION_INPUT_KEYS = frozenset(EXPECTED_DECISION_INPUT_HASHES)

EXPECTED_RUN_SOURCE_BASENAMES = {
    "followup_grid": "run_stage4_ala_discrete_single_eta1p1_alpha20_followup_grid.py",
    "followup_member": "run_stage4_ala_discrete_single_eta1p1_alpha20_followup.py",
    "followup_tests": "test_stage4_ala_discrete_single_eta1p1_alpha20_followup.py",
    "eta_runtime_shim": "run_ivm_v9_discrete_selected_eta_shim.py",
    "frozen_discrete_b040": "frozen_discrete_b040.py",
    "production_runner": "run_ivm_v9_spring_projected_production_20260730.py",
    "coarse_wrapper_alias": "run_stage4_ala_discrete_single_fixedalpha.py",
    "coarse_wrapper_implementation": "run_stage4_ala_dsel_single_fixedalpha.py",
    "legacy_command_helper": "run_stage4_uniform_optimized.py",
    "ala_core": "ala_core.py",
    "alpha15_followup_member": "run_stage4_ala_discrete_single_eta1p1_alpha15_followup.py",
    "alpha15_followup_grid": "run_stage4_ala_discrete_single_eta1p1_alpha15_followup_grid.py",
    "alpha15_formal_ctp_analyzer": "analyze_stage4_ala_discrete_single_eta1p1_alpha15_iperp_ctp.py",
    "alpha15_formal_ctp_test": "test_analyze_stage4_ala_discrete_single_eta1p1_alpha15_iperp_ctp.py",
    "alpha15_postrun_auditor": "audit_stage4_ala_discrete_single_eta1p1_alpha15_postrun.py",
    "alpha15_postrun_test": "test_audit_stage4_ala_discrete_single_eta1p1_alpha15_postrun.py",
}
EXPECTED_RUN_SOURCE_KEYS = frozenset(EXPECTED_RUN_SOURCE_BASENAMES)

DECISION_SOURCE_RELATIVE_PATHS = (
    "src/analyze_stage4_ala_discrete_single_eta1p1_alpha20_iperp_ctp.py",
    "tests/test_analyze_stage4_ala_discrete_single_eta1p1_alpha20_iperp_ctp.py",
    "src/analyze_stage4_ala_discrete_single_eta1p1_alpha15_iperp_ctp.py",
    "src/analyze_stage4_ala_discrete_single_alpha_refinement_iperp_ctp.py",
    "src/analyze_stage4_ala_discrete_single_selected_eta_iperp_ctp.py",
    "src/analyze_stage4_ala_discrete_single_residual_ratio_iperp_ctp.py",
)

CANDIDATE_ALLOWLIST = eta_base.ETA_MEMBER_ALLOWLIST
BASELINE_CANDIDATE_ALLOWLIST = base.COMMON_RESULT_KEYS | base.CANDIDATE_RESULT_KEYS
LEGACY_ALLOWLIST = base.COMMON_RESULT_KEYS
NPZ_ACCESS_ALLOWLISTS = {
    "eta1p1_alpha20_candidate": CANDIDATE_ALLOWLIST,
    "same_alpha_eta_one_candidate": BASELINE_CANDIDATE_ALLOWLIST,
    "same_alpha_legacy": LEGACY_ALLOWLIST,
    "same_eta_alpha15_candidate": CANDIDATE_ALLOWLIST,
    "canonical_initial": frozenset({"initial_path"}),
    "official_archive": base.OFFICIAL_ARCHIVE_KEYS,
    "official_combined": eta_base.OFFICIAL_COMBINED_ALLOWLIST,
    "official_components": base.OFFICIAL_COMPONENT_KEYS,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def read_command(path: Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(x, str) for x in payload):
        raise RuntimeError(f"Expected command vector: {path}")
    return payload


def command_sha256(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def scalar(saved: np.lib.npyio.NpzFile, key: str):
    return np.asarray(saved[key]).item()


def require_keys(saved: np.lib.npyio.NpzFile, keys: frozenset[str], context: str) -> None:
    missing = sorted(keys.difference(saved.files))
    if missing:
        raise RuntimeError(f"{context} is missing: {', '.join(missing)}")


def _inside_root(path: Path, root: Path, context: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{context} escapes the canonical root: {path}") from error


def verify_hash_record(record: object, context: str, root: Path) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise RuntimeError(f"Invalid hash record: {context}")
    path = Path(str(record["path"])).expanduser().resolve()
    _inside_root(path, root, context)
    if not path.is_file() or file_sha256(path) != str(record["sha256"]):
        raise RuntimeError(f"Hash mismatch: {context}")
    return path


def verify_named_manifest(
    manifest: object,
    expected_keys: frozenset[str],
    context: str,
    root: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise RuntimeError(f"Unexpected {context} key set")
    paths: dict[str, Path] = {}
    records: dict[str, dict[str, str]] = {}
    for name in sorted(expected_keys):
        path = verify_hash_record(manifest[name], f"{context}.{name}", root)
        paths[name] = path
        records[name] = {"path": str(path), "sha256": file_sha256(path)}
    return paths, records


def decision_source_hashes(root: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for relative in DECISION_SOURCE_RELATIVE_PATHS:
        path = (root / relative).resolve()
        if not path.is_file():
            raise RuntimeError(f"Missing decision source: {path}")
        records[str(path)] = file_sha256(path)
    return records


def validate_preregistration(prereg: dict[str, Any], report_path: Path) -> None:
    if prereg.get("status") != "preregistered":
        raise RuntimeError("Alpha20 preregistration status mismatch")
    if prereg.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Alpha20 preregistration protocol mismatch")
    if Path(str(prereg.get("planned_report"))).resolve() != report_path.resolve():
        raise RuntimeError("Alpha20 report path was not preregistered")
    numeric = {
        "fixed_eta": ETA,
        "fixed_alpha": CANONICAL_ALPHA,
        "fixed_residual_ratio": CANONICAL_RATIO,
        "fixed_max_evaluated_steps": CANONICAL_INNER_CAP,
        "fixed_lbfgs_history": CANONICAL_HISTORY,
        "budget_per_image": CANONICAL_BUDGET,
    }
    for key, expected in numeric.items():
        if float(prereg.get(key, float("nan"))) != float(expected):
            raise RuntimeError(f"Alpha20 preregistration drift at {key}")
    exact = {
        "single_arm": "fixed post-selection eta=1.1 at fixed alpha=20",
        "scheduled_or_adaptive_eta": False,
        "eta_application_stage": eta_base.ETA_APPLICATION_STAGE,
        "eta_policy_id": eta_base.ETA_POLICY_ID,
        "selector_isolated_from_eta": True,
        "same_dynamic_atom_displacement_cap_reapplied": True,
        "max_step_is_safety_cap_not_eta": True,
        "energy_or_objective_backtracking_added": False,
        "extra_physical_evaluations_from_eta_shim": 0,
        "replicate": "r1",
        "stop_rule": "R/R0 <= 0.8 OR evaluated accepted steps == 10",
        "reparameterization_policy": "state_curve/per_image from first state",
        "all_non_eta_scientific_flags_frozen": True,
        "same_eta_alpha15_control_in_primary_success_gate": False,
        "same_eta_alpha15_alpha_only_difference_verified": True,
        "post_hoc_grid_expansion_allowed": False,
        "terminal_or_raw_iperp_ranking_allowed": False,
        "independent_member_paths_and_logs": True,
        "overwrite_or_force_allowed": False,
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
    }
    for key, expected in exact.items():
        if prereg.get(key) != expected or type(prereg.get(key)) is not type(expected):
            raise RuntimeError(f"Alpha20 preregistration drift at {key}")
    if prereg.get("eligible_record_definition") != "initial plus reparam_committed states":
        raise RuntimeError("Alpha20 eligibility definition drifted")
    if prereg.get("mesh_gate") != {
        "segment_length_cv_max": MESH_CV_MAX,
        "segment_length_ratio_max": MESH_RATIO_MAX,
    }:
        raise RuntimeError("Alpha20 mesh gate drifted")
    if prereg.get("cost_to_precision_definition") != (
        "minimum actual optimization NF at I_perp <= threshold; no interpolation"
    ):
        raise RuntimeError("Alpha20 CTP definition drifted")
    if prereg.get("primary_comparators") != [
        "same_alpha_eta_one_candidate",
        "same_alpha_legacy",
        "official_stage4_v4",
    ]:
        raise RuntimeError("Alpha20 primary comparator set/order drifted")
    if prereg.get("reporting_anchors") != list(REPORTING_ANCHORS):
        raise RuntimeError("Alpha20 reporting anchors drifted")
    if set(prereg.get("decision_inputs", {})) != EXPECTED_DECISION_INPUT_KEYS:
        raise RuntimeError("Alpha20 decision-input set drifted")
    if set(prereg.get("run_source_manifest", {})) != EXPECTED_RUN_SOURCE_KEYS:
        raise RuntimeError("Alpha20 run-source manifest drifted")
    if prereg.get("registered_command_sha256") != EXPECTED_REGISTERED_COMMAND_SHA256:
        raise RuntimeError("Alpha20 registered command hash drifted")
    common = prereg.get("canonical_common_initial_path")
    if not isinstance(common, dict) or common.get("sha256") != (
        EXPECTED_COMMON_INITIAL_FILE_SHA256
    ) or common.get("initial_path_array_sha256") != (
        EXPECTED_COMMON_INITIAL_ARRAY_SHA256
    ):
        raise RuntimeError("Alpha20 common-initial provenance drifted")
    alpha15.assert_false_barrier_metadata(prereg, "alpha20_preregistration")


def _replace_option(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"Expected exactly one valued {flag}")
    command[positions[0] + 1] = value


def assert_reference_commands(candidate: list[str], legacy: list[str]) -> None:
    base.assert_member_command(candidate, CANONICAL_ALPHA, "candidate")
    base.assert_member_command(legacy, CANONICAL_ALPHA, "matched_control")
    base.assert_matched_commands(candidate, legacy, CANONICAL_ALPHA)
    if command_sha256(candidate) != EXPECTED_ETA_ONE_COMMAND_SHA256:
        raise RuntimeError("Formal alpha20 eta=1 command content drifted")
    if command_sha256(legacy) != EXPECTED_LEGACY_COMMAND_SHA256:
        raise RuntimeError("Formal alpha20 legacy command content drifted")


def assert_eta1p1_command(
    command: list[str], reference: list[str], shim_path: Path
) -> None:
    expected = list(reference)
    expected[1] = str(shim_path.resolve())
    options = base.command_options(command)
    if float(options.get(ETA_FLAG, -1.0)) != ETA:
        raise RuntimeError("Alpha20 command has the wrong eta")
    for flag in ROUTING_OPTIONS:
        value = options.get(flag)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"Alpha20 command lacks isolated routing {flag}")
        _replace_option(expected, flag, value)
    expected.extend([ETA_FLAG, "1.1"])
    if command != expected:
        raise RuntimeError(
            "Alpha20 eta1.1 command is not the exact eta=1 clone plus shim/eta/routing"
        )
    base.assert_member_command(command, CANONICAL_ALPHA, "candidate")
    if command_sha256(command) != EXPECTED_REGISTERED_COMMAND_SHA256:
        raise RuntimeError("Alpha20 candidate command content hash drifted")


def assert_alpha_only_command_difference(
    alpha20_command: list[str], alpha15_command: list[str]
) -> dict[str, object]:
    options20 = base.command_options(alpha20_command)
    options15 = base.command_options(alpha15_command)
    for options, alpha in ((options20, CANONICAL_ALPHA), (options15, SECONDARY_ALPHA)):
        if float(options.get("--alpha", -1.0)) != alpha:
            raise RuntimeError("Same-eta alpha-control command has wrong alpha")
        if float(options.get(ETA_FLAG, -1.0)) != ETA:
            raise RuntimeError("Same-eta alpha-control command changed eta")
    for flag in ("--alpha", "--output-dir", "--run-label"):
        options20.pop(flag, None)
        options15.pop(flag, None)
    if alpha20_command[:2] != alpha15_command[:2] or options20 != options15:
        raise RuntimeError("Alpha15/alpha20 eta1.1 commands differ beyond alpha/routing")
    return {
        "verified": True,
        "scientific_difference": {"--alpha": [SECONDARY_ALPHA, CANONICAL_ALPHA]},
        "ignored_routing_options": ["--output-dir", "--run-label"],
        "all_other_scientific_options_equal": True,
    }


def load_eta_candidate(result: Path, expected_alpha: float) -> dict[str, object]:
    with np.load(result, allow_pickle=False) as saved:
        require_keys(saved, CANDIDATE_ALLOWLIST, str(result))
        if float(scalar(saved, "alpha")) != expected_alpha:
            raise RuntimeError("Eta1.1 candidate alpha mismatch")
        if str(scalar(saved, "alpha_mode")) != "fixed":
            raise RuntimeError("Eta1.1 candidate alpha mode mismatch")
        active = np.asarray(saved["active_alpha"], dtype=float)
        if not active.size or not np.all(active == expected_alpha):
            raise RuntimeError("Eta1.1 candidate changed alpha within the trajectory")
        expected_scalars = {
            "frozen_subproblem": "discrete_b040",
            "discrete_stop_mode": "r08c10",
            "discrete_inner_max_steps": CANONICAL_INNER_CAP,
            "discrete_residual_ratio": CANONICAL_RATIO,
            "discrete_history": CANONICAL_HISTORY,
            "reparam_mode": "state_curve",
            "state_curve_direct_objective_acceptance": "per_image",
            "stop_complete_iperp": -1.0,
            "discrete_implementation_id": eta_base.eta_implementation_id(ETA),
            "discrete_step_multiplier_policy_id": eta_base.ETA_POLICY_ID,
            "discrete_step_multiplier_base_implementation_id": (
                eta_base.BASE_DISCRETE_IMPLEMENTATION
            ),
            "discrete_step_multiplier_implementation_id": (
                eta_base.eta_implementation_id(ETA)
            ),
            "discrete_step_multiplier_eta": ETA,
            "discrete_step_multiplier_application_stage": (
                eta_base.ETA_APPLICATION_STAGE
            ),
        }
        for key, expected in expected_scalars.items():
            if scalar(saved, key) != expected:
                raise RuntimeError(f"Eta1.1 result identity mismatch at {key}")
        if not bool(scalar(saved, "discrete_all_returned_states_evaluated")):
            raise RuntimeError("Eta1.1 result contains an unevaluated returned state")
        if not bool(scalar(saved, "discrete_step_multiplier_selector_isolated")):
            raise RuntimeError("Eta1.1 selector was not isolated")
        if not bool(scalar(saved, "discrete_step_multiplier_dynamic_cap_reapplied")):
            raise RuntimeError("Eta1.1 dynamic cap was not reapplied")
        if bool(scalar(saved, "discrete_step_multiplier_energy_backtracking")):
            raise RuntimeError("Eta1.1 enabled forbidden energy backtracking")
        if int(scalar(saved, "discrete_step_multiplier_extra_physical_evaluations")):
            raise RuntimeError("Eta1.1 shim added physical evaluations")
        if int(scalar(saved, "discrete_step_multiplier_aborted_selection_count")):
            raise RuntimeError("Eta1.1 contains aborted selector events")
        if np.any(np.asarray(saved["subproblem_line_search_backtracks"], int) != 0):
            raise RuntimeError("Eta1.1 contains forbidden line-search backtracking")

        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        mesh_ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        if cost.size != CANONICAL_BUDGET or active.shape != cost.shape:
            raise RuntimeError("Eta1.1 result is not the registered N254 trajectory")
        curve = eta_base.build_eligible_points(cost, i_perp, committed, cv, mesh_ratio)
        return {
            "eta": ETA,
            "alpha": float(expected_alpha),
            "result": str(result),
            "result_sha256": file_sha256(result),
            "configuration_sha256": str(scalar(saved, "configuration_sha256")),
            "initial_path_sha256": str(scalar(saved, "initial_path_sha256")),
            "final_optimization_NF": int(cost[-1]),
            "npz_fields_accessed": sorted(CANDIDATE_ALLOWLIST),
            **curve,
        }


def _verify_candidate_audit(
    audit_path: Path, result: Path, result_hash: str
) -> tuple[dict[str, Any], list[str]]:
    audit = read_json(audit_path)
    alpha15.assert_false_barrier_metadata(audit, "alpha20_candidate_audit")
    required = {
        "member_policy_version": PROTOCOL_ID,
        "member_role": "fixed_eta1p1_exploratory_followup",
        "eta": ETA,
        "alpha": CANONICAL_ALPHA,
        "fixed_alpha_throughout": True,
        "residual_ratio": CANONICAL_RATIO,
        "max_evaluated_steps": CANONICAL_INNER_CAP,
        "lbfgs_history": CANONICAL_HISTORY,
        "budget_per_image": CANONICAL_BUDGET,
        "scheduled_or_adaptive_eta": False,
        "selector_isolated_from_eta": True,
        "energy_backtracking": False,
        "extra_physical_evaluations_from_shim": 0,
        "stop_rule": "R/R0 <= 0.8 OR evaluated accepted steps == 10",
        "reparameterization_policy": "state_curve/per_image from first state",
        "formal_ranking_deferred_to_complete_mesh_iperp_ctp": True,
        "barrier_fields_accessed": False,
        "barrier_computation_or_densification_performed_by_wrapper": False,
    }
    for key, expected in required.items():
        if audit.get(key) != expected:
            raise RuntimeError(f"Alpha20 candidate audit drift at {key}")
    if Path(str(audit.get("result"))).resolve() != result:
        raise RuntimeError("Alpha20 candidate audit result path mismatch")
    if audit.get("result_sha256") != result_hash:
        raise RuntimeError("Alpha20 candidate audit result hash mismatch")
    alpha15.assert_no_barrier_result_fields(
        audit.get("result_fields_accessed"), "alpha20_candidate_audit"
    )
    command = audit.get("command")
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise RuntimeError("Alpha20 candidate audit command is invalid")
    return audit, command


def _verify_auxiliary_artifacts(
    root: Path,
    candidate_command: list[str],
    run_sources: dict[str, Path],
) -> dict[str, dict[str, str]]:
    verified: dict[str, dict[str, str]] = {}
    for name, (relative, expected) in EXPECTED_AUXILIARY_ARTIFACTS.items():
        path = (root / relative).resolve()
        _inside_root(path, root, f"auxiliary.{name}")
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"Auxiliary artifact drifted: {name}")
        verified[name] = {"path": str(path), "sha256": expected}
    member_command = read_command(Path(verified["member_command"]["path"]))
    if member_command != candidate_command:
        raise RuntimeError("Member-log command differs from the candidate audit")
    launcher = read_command(Path(verified["launcher_command"]["path"]))
    expected_launcher = [
        candidate_command[0],
        str(run_sources["followup_member"]),
        "--root",
        str(root),
    ]
    if launcher != expected_launcher:
        raise RuntimeError("Launcher command provenance drifted")
    return verified


def _verify_alpha15_control(
    paths: dict[str, Path],
) -> tuple[dict[str, object], list[str], dict[str, object]]:
    prereg_path = paths["same_eta_alpha15_preregistration"]
    report_path = paths["same_eta_alpha15_report"]
    result_path = paths["same_eta_alpha15_result"]
    audit_path = paths["same_eta_alpha15_member_audit"]
    formal_ctp_path = paths["same_eta_alpha15_formal_ctp"]
    postrun_path = paths["same_eta_alpha15_postrun_audit"]

    prereg = read_json(prereg_path)
    alpha15.validate_preregistration(prereg, report_path)
    report = read_json(report_path)
    if report.get("status") != "complete" or report.get("protocol_id") != alpha15.PROTOCOL_ID:
        raise RuntimeError("Alpha15 control report identity drifted")
    if Path(str(report.get("preregistration"))).resolve() != prereg_path:
        raise RuntimeError("Alpha15 report/preregistration path mismatch")
    if report.get("preregistration_sha256") != EXPECTED_ALPHA15_PREREGISTRATION_SHA256:
        raise RuntimeError("Alpha15 report/preregistration hash mismatch")
    for key, value in prereg.items():
        if key != "status" and report.get(key) != value:
            raise RuntimeError(f"Alpha15 report drifted from preregistration at {key}")
    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError("Alpha15 control report is not one successful arm")
    row = rows[0]
    if int(row.get("returncode", -1)) != 0:
        raise RuntimeError("Alpha15 control member failed")
    if (
        Path(str(row.get("result"))).resolve() != result_path
        or row.get("result_sha256") != EXPECTED_ALPHA15_RESULT_SHA256
        or Path(str(row.get("member_audit"))).resolve() != audit_path
        or row.get("member_audit_sha256") != EXPECTED_ALPHA15_AUDIT_SHA256
    ):
        raise RuntimeError("Alpha15 control report provenance drifted")

    # Recompute the complete barrier-blind alpha15 audit rather than trusting
    # only its stored conclusion.  This verifies its full transitive hash chain.
    recomputed = alpha15.analyze(report_path)
    formal = read_json(formal_ctp_path)
    if formal.get("analysis_id") != alpha15.ANALYSIS_ID:
        raise RuntimeError("Alpha15 formal CTP identity drifted")
    if formal.get("eta1p1_report_sha256") != EXPECTED_ALPHA15_REPORT_SHA256:
        raise RuntimeError("Alpha15 formal CTP report provenance drifted")
    if formal.get("eta1p1_member_audit_sha256") != EXPECTED_ALPHA15_AUDIT_SHA256:
        raise RuntimeError("Alpha15 formal CTP member provenance drifted")
    if formal.get("candidate") != recomputed.get("candidate"):
        raise RuntimeError("Alpha15 formal CTP candidate snapshot drifted")
    if (
        formal.get("barrier_data_accessed") is not False
        or formal.get("barrier_computation_or_densification_performed") is not False
        or formal.get("barrier_unembargoed") is not False
        or formal.get("barrier_remains_embargoed") is not True
    ):
        raise RuntimeError("Alpha15 formal CTP broke its barrier embargo")

    audit, command = alpha15._verify_candidate_audit(
        audit_path, result_path, EXPECTED_ALPHA15_RESULT_SHA256
    )
    if audit.get("scientific_command_sha256") != formal.get(
        "registered_scientific_command_sha256"
    ):
        raise RuntimeError("Alpha15 command provenance drifted")

    postrun = read_json(postrun_path)
    expected_postrun = {
        "status": "passed",
        "formal_report_sha256": EXPECTED_ALPHA15_REPORT_SHA256,
        "preregistration_sha256": EXPECTED_ALPHA15_PREREGISTRATION_SHA256,
        "fixed_alpha_15_throughout": True,
        "fixed_r08_k10_m7_throughout": True,
        "energy_or_objective_fallback_used": False,
        "extra_physical_evaluations_from_eta_shim": 0,
        "barrier_metric_values_accessed": False,
        "barrier_computation_or_ranking_performed": False,
    }
    for key, expected in expected_postrun.items():
        if postrun.get(key) != expected:
            raise RuntimeError(f"Alpha15 postrun provenance drift at {key}")
    return recomputed["candidate"], command, {
        "preregistration_validated": True,
        "report_validated": True,
        "formal_ctp_recomputed_and_snapshot_matched": True,
        "postrun_mechanism_audit_passed": True,
        "formal_ctp_primary_success": bool(
            formal.get("primary_success_criterion_satisfied")
        ),
    }


def comparison_with_wtlu(
    candidate: list[dict[str, object]],
    comparator: list[dict[str, object]],
    name: str,
) -> dict[str, object]:
    return eta_base.comparison_with_wtlu(candidate, comparator, name)


def deficit_vector(
    candidate: list[dict[str, object]], comparator: list[dict[str, object]]
) -> dict[str, object]:
    return eta_base.deficit_vector(candidate, comparator)


def primary_success(comparisons: dict[str, dict[str, object]]) -> bool:
    required = {
        "same_alpha_eta_one_candidate",
        "same_alpha_legacy",
        "official_stage4_v4",
    }
    if set(comparisons) != required:
        raise RuntimeError("Primary comparison set is not exactly the registered triple")
    return all(bool(comparisons[name]["full_curve_dominates"]) for name in required)


def analyze(report_path: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    report_path = report_path.expanduser().resolve()
    _inside_root(report_path, root, "formal report")
    if file_sha256(report_path) != EXPECTED_REPORT_SHA256:
        raise RuntimeError("Unexpected formal eta1.1/alpha20 report snapshot")
    report = read_json(report_path)
    alpha15.assert_false_barrier_metadata(report, "alpha20_report")
    if report.get("status") != "complete" or report.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Alpha20 report is not a complete registered protocol")
    prereg_path = Path(str(report.get("preregistration"))).resolve()
    _inside_root(prereg_path, root, "preregistration")
    if (
        file_sha256(prereg_path) != EXPECTED_PREREGISTRATION_SHA256
        or report.get("preregistration_sha256") != EXPECTED_PREREGISTRATION_SHA256
    ):
        raise RuntimeError("Alpha20 preregistration hash mismatch")
    prereg = read_json(prereg_path)
    validate_preregistration(prereg, report_path)
    for key, value in prereg.items():
        if key != "status" and report.get(key) != value:
            raise RuntimeError(f"Alpha20 report drifted from preregistration at {key}")
    if report.get("barrier_fields_accessed_by_launcher") is not False:
        raise RuntimeError("Alpha20 launcher reports barrier access")

    run_source_paths, run_source_records = verify_named_manifest(
        prereg["run_source_manifest"],
        EXPECTED_RUN_SOURCE_KEYS,
        "run_source_manifest",
        root,
    )
    for name, expected_basename in EXPECTED_RUN_SOURCE_BASENAMES.items():
        if run_source_paths[name].name != expected_basename:
            raise RuntimeError(f"Run-source path identity drifted: {name}")
    decision_paths, decision_records = verify_named_manifest(
        prereg["decision_inputs"],
        EXPECTED_DECISION_INPUT_KEYS,
        "decision_inputs",
        root,
    )
    for name, expected_hash in EXPECTED_DECISION_INPUT_HASHES.items():
        if decision_records[name]["sha256"] != expected_hash:
            raise RuntimeError(f"Registered decision input drifted: {name}")

    common_record = prereg["canonical_common_initial_path"]
    if not isinstance(common_record, dict) or set(common_record) != {
        "path", "sha256", "initial_path_array_sha256"
    }:
        raise RuntimeError("Invalid canonical common-initial record")
    common_path = Path(str(common_record["path"])).resolve()
    _inside_root(common_path, root, "canonical common initial")
    if file_sha256(common_path) != EXPECTED_COMMON_INITIAL_FILE_SHA256:
        raise RuntimeError("Canonical common-initial file hash drifted")
    with np.load(common_path, allow_pickle=False) as saved:
        require_keys(saved, NPZ_ACCESS_ALLOWLISTS["canonical_initial"], str(common_path))
        common = np.ascontiguousarray(np.asarray(saved["initial_path"], dtype=np.float64))
    common_array_hash = hashlib.sha256(common.tobytes()).hexdigest()
    if common_array_hash != EXPECTED_COMMON_INITIAL_ARRAY_SHA256:
        raise RuntimeError("Canonical common-initial array hash drifted")

    reference_command = read_command(
        decision_paths["same_alpha_eta_one_candidate_command"]
    )
    legacy_command = read_command(decision_paths["same_alpha_legacy_command"])
    assert_reference_commands(reference_command, legacy_command)

    # Coarse v2 audit JSONs contain sparse barrier diagnostics.  Their exact
    # whole-file hashes are verified above, but they are intentionally not
    # decoded.  Result and command semantics are reconstructed independently.
    eta_one = base.load_member_result(
        decision_paths["same_alpha_eta_one_candidate_result"],
        CANONICAL_ALPHA,
        "candidate",
    )
    legacy = base.load_member_result(
        decision_paths["same_alpha_legacy_result"],
        CANONICAL_ALPHA,
        "matched_control",
    )
    official = base.load_official_curve(decision_paths["official_stage4_v4"])
    official_transitive = eta_base.verify_official_transitive_hashes(
        decision_paths["official_stage4_v4"], official
    )

    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError("Alpha20 report must contain exactly one member")
    row = rows[0]
    if (
        int(row.get("returncode", -1)) != 0
        or float(row.get("eta", -1.0)) != ETA
        or float(row.get("alpha", -1.0)) != CANONICAL_ALPHA
    ):
        raise RuntimeError("Alpha20 report member identity/failure drifted")
    result = Path(str(row.get("result"))).resolve()
    audit_path = Path(str(row.get("member_audit"))).resolve()
    _inside_root(result, root, "candidate result")
    _inside_root(audit_path, root, "candidate audit")
    if (
        row.get("result_sha256") != EXPECTED_CANDIDATE_RESULT_SHA256
        or file_sha256(result) != EXPECTED_CANDIDATE_RESULT_SHA256
        or row.get("member_audit_sha256") != EXPECTED_CANDIDATE_AUDIT_SHA256
        or file_sha256(audit_path) != EXPECTED_CANDIDATE_AUDIT_SHA256
    ):
        raise RuntimeError("Alpha20 result/member-audit hash drifted")
    audit, candidate_command = _verify_candidate_audit(
        audit_path, result, EXPECTED_CANDIDATE_RESULT_SHA256
    )
    assert_eta1p1_command(
        candidate_command, reference_command, run_source_paths["eta_runtime_shim"]
    )
    if audit.get("scientific_command_sha256") != EXPECTED_REGISTERED_COMMAND_SHA256:
        raise RuntimeError("Alpha20 member-audit command hash drifted")
    auxiliary_records = _verify_auxiliary_artifacts(
        root, candidate_command, run_source_paths
    )
    if Path(auxiliary_records["launcher_stdout"]["path"]).read_bytes() != audit_path.read_bytes():
        raise RuntimeError("Launcher stdout is not the immutable member audit")

    candidate = load_eta_candidate(result, CANONICAL_ALPHA)
    if candidate["initial_path_sha256"] != common_array_hash:
        raise RuntimeError("Alpha20 candidate changed the canonical initial path")
    for key in ("complete_state_NF", "mesh_eligible_complete_state_NF"):
        if list(audit.get(key, [])) != candidate[key]:
            raise RuntimeError(f"Alpha20 member-audit mismatch at {key}")
    if int(audit.get("final_optimization_NF", -1)) != candidate["final_optimization_NF"]:
        raise RuntimeError("Alpha20 member-audit terminal NF mismatch")

    alpha15_candidate, alpha15_command, alpha15_provenance = _verify_alpha15_control(
        decision_paths
    )
    alpha_only_command = assert_alpha_only_command_difference(
        candidate_command, alpha15_command
    )
    initial_hashes = {
        candidate["initial_path_sha256"],
        eta_one["initial_path_sha256"],
        legacy["initial_path_sha256"],
        alpha15_candidate["initial_path_sha256"],
    }
    if initial_hashes != {common_array_hash}:
        raise RuntimeError("Candidate/comparators do not share the canonical initial path")

    candidate_points = candidate["eligible_points"]
    comparisons = {
        "same_alpha_eta_one_candidate": comparison_with_wtlu(
            candidate_points,
            eta_one["eligible_points"],
            "same_alpha_eta_one_candidate",
        ),
        "same_alpha_legacy": comparison_with_wtlu(
            candidate_points,
            legacy["eligible_points"],
            "same_alpha_legacy",
        ),
        "official_stage4_v4": comparison_with_wtlu(
            candidate_points,
            official["eligible_points"],
            "official_stage4_v4_post_activation_mesh_eligible_tail",
        ),
    }
    secondary = comparison_with_wtlu(
        candidate_points,
        alpha15_candidate["eligible_points"],
        "same_eta_alpha15_strict_alpha_control",
    )
    success = primary_success(comparisons)
    anchors = [float(value) for value in prereg["reporting_anchors"]]
    return {
        "analysis_id": ANALYSIS_ID,
        "created_unix": time.time(),
        "read_only_formal_input_audit": True,
        "barrier_data_accessed": False,
        "barrier_computation_or_densification_performed": False,
        "barrier_unembargoed": False,
        "barrier_remains_embargoed": True,
        "alpha20_report": str(report_path),
        "alpha20_report_sha256": file_sha256(report_path),
        "alpha20_preregistration": str(prereg_path),
        "alpha20_preregistration_sha256": file_sha256(prereg_path),
        "canonical_common_initial_path": str(common_path),
        "canonical_common_initial_file_sha256": file_sha256(common_path),
        "canonical_common_initial_array_sha256": common_array_hash,
        "verified_run_source_hashes": run_source_records,
        "verified_decision_input_hashes": decision_records,
        "verified_decision_source_hashes": decision_source_hashes(root),
        "verified_auxiliary_artifact_hashes": auxiliary_records,
        "verified_report_prereg_source_command_result_audit_hashes": True,
        "coarse_comparator_audits_hash_verified_but_not_decoded": True,
        "coarse_comparator_audit_decode_policy": (
            "whole-file hashes only because the frozen coarse audits contain "
            "sparse barrier diagnostics; result/command semantics independently reconstructed"
        ),
        "alpha20_member_audit": str(audit_path),
        "alpha20_member_audit_sha256": file_sha256(audit_path),
        "registered_scientific_command_sha256": EXPECTED_REGISTERED_COMMAND_SHA256,
        "fixed_eta": ETA,
        "fixed_alpha": CANONICAL_ALPHA,
        "fixed_residual_ratio": CANONICAL_RATIO,
        "fixed_max_evaluated_steps": CANONICAL_INNER_CAP,
        "fixed_lbfgs_history": CANONICAL_HISTORY,
        "fixed_budget_per_image": CANONICAL_BUDGET,
        "npz_access_allowlists": {
            name: sorted(keys) for name, keys in NPZ_ACCESS_ALLOWLISTS.items()
        },
        "eligibility": (
            "initial or reparam_committed, then segment_length_cv<=0.0065 "
            "and segment_length_ratio<=1.03"
        ),
        "cost_to_precision": (
            "minimum actual optimization NF with I_perp<=threshold; no interpolation"
        ),
        "unreachable_policy": "candidate unreachable is U and is a formal loss",
        "candidate": candidate,
        "same_alpha_eta_one_candidate": eta_one,
        "same_alpha_legacy": legacy,
        "official_stage4_v4": official,
        "official_transitive_hashes": official_transitive,
        "comparisons": comparisons,
        "deficit_vectors": {
            "same_alpha_eta_one_candidate": deficit_vector(
                candidate_points, eta_one["eligible_points"]
            ),
            "same_alpha_legacy": deficit_vector(
                candidate_points, legacy["eligible_points"]
            ),
            "official_stage4_v4": deficit_vector(
                candidate_points, official["eligible_points"]
            ),
        },
        "reporting_anchors": {
            "same_alpha_eta_one_candidate": base.anchor_rows(
                candidate_points, eta_one["eligible_points"], anchors
            ),
            "same_alpha_legacy": base.anchor_rows(
                candidate_points, legacy["eligible_points"], anchors
            ),
            "official_stage4_v4": base.anchor_rows(
                candidate_points,
                official["eligible_points"],
                anchors,
                domain_max=OFFICIAL_DOMAIN_MAX,
            ),
        },
        "secondary_same_eta_alpha15": {
            "role": (
                "strict control-variable alpha comparison; excluded from primary success gate"
            ),
            "in_primary_success_gate": False,
            "alpha_only_command_difference": alpha_only_command,
            "provenance": alpha15_provenance,
            "comparator": alpha15_candidate,
            "comparison": secondary,
            "deficit_vector": deficit_vector(
                candidate_points, alpha15_candidate["eligible_points"]
            ),
            "reporting_anchors": base.anchor_rows(
                candidate_points, alpha15_candidate["eligible_points"], anchors
            ),
        },
        "triple_tightest_precision_ok": all(
            bool(row["candidate_does_not_lose_tightest_precision"])
            for row in comparisons.values()
        ),
        "primary_success_criterion_satisfied": success,
        "fixed_eta1p1_alpha20_full_curve_dominates_all_three": success,
        "winner_claim_made": success,
        "post_hoc_eta_or_alpha_expansion_allowed": False,
        "barrier_unembargo_allowed_by_this_analysis": False,
    }


def write_exclusive(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    alpha15.assert_false_barrier_metadata(payload, "analysis_output")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta1p1-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    report = args.eta1p1_report.expanduser().resolve()
    if not report.is_file():
        raise FileNotFoundError(report)
    payload = analyze(report)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else report.parent
    )
    output = output_dir / f"{report.stem}_iperp_ctp_readonly_v1_{time.time_ns()}.json"
    payload["analysis_script"] = str(Path(__file__).resolve())
    payload["analysis_script_sha256"] = file_sha256(Path(__file__).resolve())
    write_exclusive(payload, output)
    print(output)


if __name__ == "__main__":
    main()
