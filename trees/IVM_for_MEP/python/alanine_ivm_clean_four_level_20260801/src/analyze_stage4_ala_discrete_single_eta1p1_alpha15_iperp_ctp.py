#!/usr/bin/env python3
"""Barrier-blind I_perp CTP audit for the fixed eta=1.1, alpha=15 arm.

This is deliberately a separate analyzer from the locked selected-eta grid
analyzer.  It accepts one preregistered single-arm report, independently
reconstructs the candidate complete+mesh curve from its NPZ, and compares it
with exactly three frozen comparators: the existing alpha=15 discrete eta=1
trajectory, the matched alpha=15 legacy trajectory, and official Stage-IV v4.

All costs are observed ``total_internal_joint_evaluations`` values.  No
interpolation, terminal-only ranking, energy field, or barrier field is used.
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
import analyze_stage4_ala_discrete_single_selected_eta_iperp_ctp as eta_base


ANALYSIS_ID = "stage4_ala_discrete_single_eta1p1_alpha15_iperp_ctp_v1"
PROTOCOL_ID = "stage4_ala_discrete_single_eta1p1_alpha15_followup_v1"
FORMAL_REFINEMENT_ANALYSIS_ID = (
    "stage4_ala_discrete_single_alpha_refinement_iperp_ctp_v1"
)

ETA = 1.1
CANONICAL_ALPHA = 15.0
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

EXPECTED_FORMAL_CTP_SHA256 = (
    "8ff9d9824eb2d3719d28f73f33b92b62e115ff9f7449bf60be054298d7ba5212"
)
EXPECTED_ETA1P1_REPORT_SHA256 = (
    "fa0bf64922d91d15a48b0b6b6949d8ddb3ff109661ff011b351f8f9caee79212"
)
EXPECTED_ETA1P1_PREREGISTRATION_SHA256 = (
    "dbffae0850f39128c313f2819c28b7679d5e80ef0a6f9fe59246dc43a176df1d"
)
EXPECTED_REFINEMENT_REPORT_SHA256 = (
    "68571385e40e523065916c36fbd72b91445c4b7239f2925c0b8cdfc7a2ff63c0"
)
EXPECTED_REFINEMENT_PREREG_SHA256 = (
    "16739061c15d8f575a48eefbd4cf2edf7cc453c38defbda987e9a3899714fc74"
)
EXPECTED_ETA_ONE_RESULT_SHA256 = (
    "28bf7a5894337e2fc26178cc9dc9ead86f421c50972f9faf15aef34ce1830d90"
)
EXPECTED_ETA_ONE_AUDIT_SHA256 = (
    "65ce120aaf470cfd95ac3ab46e1cc50ceecb76afff4e6c484216f1986e77165c"
)
EXPECTED_ETA_ONE_COMMAND_FILE_SHA256 = (
    "9bf0458e6d0f898ddaf76eaa9758531887c677853b4ce32b4bf98c3ea46125c8"
)
EXPECTED_ETA_ONE_COMMAND_SHA256 = (
    "a7d2d3f427677e6ef322ee8e5ca6c13d93ab96f9cba6c1edf821effd43a63fc4"
)
EXPECTED_LEGACY_RESULT_SHA256 = (
    "2ffa7aeb8122cbf4cdfecf9d4a372da8774c6485ec82f9b6693c3047c2d247dc"
)
EXPECTED_LEGACY_AUDIT_SHA256 = (
    "fb26c9fcb09ba1fbff7058c367965c694a573ecd5ed6110ce81366cbdef3fe83"
)
EXPECTED_LEGACY_COMMAND_FILE_SHA256 = (
    "9842cbe8e47971a49e4b90203192d59d45245b19baadf06e28b36ef049d4db16"
)
EXPECTED_LEGACY_COMMAND_SHA256 = (
    "423994b566dfe5e061c6569af707c9befad1a48117f1f87dc40c6ed0690fec8d"
)
EXPECTED_COMMON_INITIAL_FILE_SHA256 = (
    "4a08066e1f0a89ad1d0ee6ecc442d7255d511217576d1ff65e9cb5e2164d891b"
)
EXPECTED_COMMON_INITIAL_ARRAY_SHA256 = (
    "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
)
EXPECTED_OFFICIAL_ARCHIVE_SHA256 = (
    "09ccbf7db731c96c0beda9d5879c9dd46cae2b428db8b71820aad1c56cb574fc"
)

EXPECTED_DECISION_INPUT_KEYS = frozenset(
    {
        "same_alpha_eta_one_candidate_result",
        "same_alpha_eta_one_candidate_audit",
        "same_alpha_eta_one_candidate_command",
        "same_alpha_legacy_result",
        "same_alpha_legacy_audit",
        "same_alpha_legacy_command",
        "official_stage4_v4",
        "alpha15_refinement_ctp_snapshot",
    }
)
EXPECTED_RUN_SOURCE_KEYS = frozenset(
    {
        "followup_grid",
        "followup_member",
        "followup_tests",
        "eta_runtime_shim",
        "frozen_discrete_b040",
        "production_runner",
        "alpha15_refinement_wrapper",
        "coarse_wrapper_alias",
        "coarse_wrapper_implementation",
        "legacy_command_helper",
        "ala_core",
    }
)
REQUIRED_RUN_SOURCE_BASENAMES = frozenset(
    {
        "run_stage4_ala_discrete_single_eta1p1_alpha15_followup.py",
        "run_ivm_v9_discrete_selected_eta_shim.py",
        "frozen_discrete_b040.py",
        "run_ivm_v9_spring_projected_production_20260730.py",
        "run_stage4_ala_discrete_single_alpha_refinement.py",
        "run_stage4_ala_discrete_single_fixedalpha.py",
        "run_stage4_ala_dsel_single_fixedalpha.py",
        "run_stage4_uniform_optimized.py",
        "ala_core.py",
    }
)
DECISION_SOURCE_RELATIVE_PATHS = (
    "src/analyze_stage4_ala_discrete_single_eta1p1_alpha15_iperp_ctp.py",
    "tests/test_analyze_stage4_ala_discrete_single_eta1p1_alpha15_iperp_ctp.py",
    "src/analyze_stage4_ala_discrete_single_alpha_refinement_iperp_ctp.py",
    "src/analyze_stage4_ala_discrete_single_selected_eta_iperp_ctp.py",
    "src/analyze_stage4_ala_discrete_single_residual_ratio_iperp_ctp.py",
)

CANDIDATE_ALLOWLIST = eta_base.ETA_MEMBER_ALLOWLIST
BASELINE_CANDIDATE_ALLOWLIST = base.COMMON_RESULT_KEYS | base.CANDIDATE_RESULT_KEYS
LEGACY_ALLOWLIST = base.COMMON_RESULT_KEYS
NPZ_ACCESS_ALLOWLISTS = {
    "eta1p1_candidate": CANDIDATE_ALLOWLIST,
    "same_alpha_eta_one_candidate": BASELINE_CANDIDATE_ALLOWLIST,
    "same_alpha_legacy": LEGACY_ALLOWLIST,
    "canonical_initial": frozenset({"initial_path"}),
    "official_archive": base.OFFICIAL_ARCHIVE_KEYS,
    "official_combined": eta_base.OFFICIAL_COMBINED_ALLOWLIST,
    "official_components": base.OFFICIAL_COMPONENT_KEYS,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def read_command(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(x, str) for x in payload):
        raise RuntimeError(f"Expected command vector: {path}")
    return payload


def scalar(saved: np.lib.npyio.NpzFile, key: str):
    return np.asarray(saved[key]).item()


def require_keys(saved: np.lib.npyio.NpzFile, keys: frozenset[str], context: str) -> None:
    missing = sorted(keys.difference(saved.files))
    if missing:
        raise RuntimeError(f"{context} is missing: {', '.join(missing)}")


def command_sha256(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_hash_record(record: object, context: str) -> Path:
    if not isinstance(record, dict) or set(record).issuperset({"path", "sha256"}) is False:
        raise RuntimeError(f"Invalid hash record: {context}")
    path = Path(str(record["path"])).resolve()
    if not path.is_file() or file_sha256(path) != str(record["sha256"]):
        raise RuntimeError(f"Hash mismatch: {context}")
    return path


def verify_manifest(manifest: object, context: str) -> dict[str, str]:
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError(f"Invalid manifest: {context}")
    verified: dict[str, str] = {}
    for name, record in manifest.items():
        path = verify_hash_record(record, f"{context}.{name}")
        verified[str(path)] = file_sha256(path)
    return verified


def require_source_basenames(
    verified: dict[str, str], required: frozenset[str], context: str
) -> None:
    names = {Path(path).name for path in verified}
    missing = sorted(required.difference(names))
    if missing:
        raise RuntimeError(f"{context} lacks pinned source(s): {', '.join(missing)}")


_BARRIER_METADATA_SCHEMA = {
    "barrier_fields_accessed_by_wrapper_or_launcher": False,
    "barrier_fields_accessed_by_launcher": False,
    "barrier_fields_accessed": False,
    "barrier_computation_or_densification_performed_by_wrapper": False,
    "barrier_data_accessed": False,
    "barrier_computation_or_densification_performed": False,
    "barrier_unembargoed": False,
    "barrier_unembargo_allowed_by_this_analysis": False,
    "barrier_remains_embargoed": True,
    "sparse_barrier_role": "embargoed_not_read_not_reported_not_ranked",
    "densified_energy_barrier_role": "embargoed_for_entire_followup",
    "barrier_embargo": "all energy-barrier fields ignored until alpha lock",
}


def assert_false_barrier_metadata(payload: object, context: str = "payload") -> None:
    """Validate barrier metadata by field schema without touching NPZ barrier data."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{context}.{key}"
            if "barrier" in str(key).lower():
                if key not in _BARRIER_METADATA_SCHEMA:
                    raise RuntimeError(f"Unknown barrier metadata field at {child}")
                expected = _BARRIER_METADATA_SCHEMA[key]
                if value != expected or type(value) is not type(expected):
                    raise RuntimeError(f"Invalid barrier metadata at {child}")
            else:
                assert_false_barrier_metadata(value, child)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            assert_false_barrier_metadata(value, f"{context}[{index}]")


def assert_no_barrier_result_fields(fields: object, context: str) -> list[str]:
    if not isinstance(fields, list) or not all(isinstance(x, str) for x in fields):
        raise RuntimeError(f"Invalid result-field audit at {context}")
    forbidden = [
        field
        for field in fields
        if "barrier" in field.lower() or "densif" in field.lower()
    ]
    if forbidden:
        raise RuntimeError(f"Barrier result field accessed at {context}: {forbidden}")
    return fields


def decision_source_hashes(root: Path) -> dict[str, str]:
    verified: dict[str, str] = {}
    for relative in DECISION_SOURCE_RELATIVE_PATHS:
        path = (root / relative).resolve()
        if not path.is_file():
            raise RuntimeError(f"Missing decision source: {path}")
        verified[str(path)] = file_sha256(path)
    return verified


def _replace_option(command: list[str], flag: str, value: str) -> None:
    positions = [i for i, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"Expected exactly one valued {flag}")
    command[positions[0] + 1] = value


def assert_reference_commands(candidate: list[str], legacy: list[str]) -> None:
    base.assert_member_command(candidate, CANONICAL_ALPHA, "candidate")
    base.assert_member_command(legacy, CANONICAL_ALPHA, "matched_control")
    if command_sha256(candidate) != EXPECTED_ETA_ONE_COMMAND_SHA256:
        raise RuntimeError("Formal alpha15 eta=1 command content drifted")
    if command_sha256(legacy) != EXPECTED_LEGACY_COMMAND_SHA256:
        raise RuntimeError("Formal alpha15 legacy command content drifted")


def assert_eta1p1_command(
    command: list[str], reference: list[str], shim_path: Path
) -> None:
    if not isinstance(command, list) or len(command) < 3:
        raise RuntimeError("Invalid eta1.1 command")
    expected = list(reference)
    expected[1] = str(shim_path.resolve())
    candidate_options = base.command_options(command)
    if float(candidate_options.get(ETA_FLAG, -1.0)) != ETA:
        raise RuntimeError("Eta1.1 command has the wrong multiplier")
    for flag in ROUTING_OPTIONS:
        value = candidate_options.get(flag)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"Eta1.1 command lacks isolated routing {flag}")
        _replace_option(expected, flag, value)
    expected.extend([ETA_FLAG, "1.1"])
    if command != expected:
        raise RuntimeError(
            "Eta1.1 command is not an exact alpha15 eta=1 clone plus shim/eta/routing"
        )
    base.assert_member_command(command, CANONICAL_ALPHA, "candidate")


def load_eta1p1_result(result: Path) -> dict[str, object]:
    with np.load(result, allow_pickle=False) as saved:
        require_keys(saved, CANDIDATE_ALLOWLIST, str(result))
        if float(scalar(saved, "alpha")) != CANONICAL_ALPHA:
            raise RuntimeError("Eta1.1 candidate alpha mismatch")
        if str(scalar(saved, "alpha_mode")) != "fixed":
            raise RuntimeError("Eta1.1 candidate alpha mode mismatch")
        active = np.asarray(saved["active_alpha"], dtype=float)
        if not active.size or not np.all(active == CANONICAL_ALPHA):
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
        required_true = (
            "discrete_all_returned_states_evaluated",
            "discrete_step_multiplier_selector_isolated",
            "discrete_step_multiplier_dynamic_cap_reapplied",
        )
        if not all(bool(scalar(saved, key)) for key in required_true):
            raise RuntimeError("Eta1.1 result failed evaluated/selector/cap identity")
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
        curve = eta_base.build_eligible_points(
            cost, i_perp, committed, cv, mesh_ratio
        )
        return {
            "eta": ETA,
            "alpha": CANONICAL_ALPHA,
            "result": str(result),
            "result_sha256": file_sha256(result),
            "configuration_sha256": str(scalar(saved, "configuration_sha256")),
            "initial_path_sha256": str(scalar(saved, "initial_path_sha256")),
            "final_optimization_NF": int(cost[-1]),
            "npz_fields_accessed": sorted(CANDIDATE_ALLOWLIST),
            **curve,
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


def triple_success(comparisons: dict[str, dict[str, object]]) -> bool:
    required = {
        "same_alpha_eta_one_candidate",
        "same_alpha_legacy",
        "official_stage4_v4",
    }
    if set(comparisons) != required:
        raise RuntimeError("Single-arm comparison set is not exactly the registered triple")
    return all(bool(comparisons[name]["full_curve_dominates"]) for name in required)


def _assert_curve_snapshot(
    loaded: dict[str, object], snapshot: dict[str, object], context: str
) -> None:
    for key in ("result", "result_sha256", "configuration_sha256", "initial_path_sha256"):
        if loaded[key] != snapshot[key]:
            raise RuntimeError(f"{context} snapshot drift at {key}")
    for key in (
        "complete_state_NF",
        "mesh_eligible_complete_state_NF",
        "eligible_points",
        "pareto_records",
    ):
        if loaded[key] != snapshot[key]:
            raise RuntimeError(f"{context} CTP snapshot drift at {key}")


def _verify_embedded_path_hashes(
    records: object, context: str
) -> dict[str, str]:
    if not isinstance(records, dict) or not records:
        raise RuntimeError(f"Missing embedded hash map: {context}")
    verified: dict[str, str] = {}
    for raw_path, raw_hash in records.items():
        path = Path(str(raw_path)).resolve()
        expected = str(raw_hash)
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"Embedded hash drift at {context}: {path}")
        verified[str(path)] = expected
    return verified


def _verify_reference_audit(
    audit_path: Path,
    result_path: Path,
    result_hash: str,
    command: list[str],
    role: str,
) -> dict[str, object]:
    audit = read_json(audit_path)
    assert_false_barrier_metadata(audit, f"{role}_audit")
    required = {
        "member_policy_version": "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1",
        "member_role": role,
        "alpha": CANONICAL_ALPHA,
        "alpha_mode": "fixed",
        "fixed_alpha_throughout": True,
        "budget_per_image": CANONICAL_BUDGET,
        "replicate": "r1",
        "single_run_from_canonical_common_path": True,
        "reparameterization_policy": "state_curve/per_image from first state",
        "iperp_only_scientific_analysis": True,
        "barrier_fields_accessed": False,
        "barrier_computation_or_densification_performed_by_wrapper": False,
        "formal_ranking_deferred_to_complete_mesh_iperp_ctp": True,
    }
    for key, expected in required.items():
        if audit.get(key) != expected:
            raise RuntimeError(f"Alpha15 {role} audit drift at {key}")
    if Path(str(audit.get("result"))).resolve() != result_path:
        raise RuntimeError(f"Alpha15 {role} audit result path mismatch")
    if audit.get("result_sha256") != result_hash:
        raise RuntimeError(f"Alpha15 {role} audit result hash mismatch")
    if audit.get("command") != command:
        raise RuntimeError(f"Alpha15 {role} audit command mismatch")
    assert_no_barrier_result_fields(
        audit.get("result_fields_accessed"), f"alpha15_{role}_audit"
    )
    return audit


def load_formal_alpha15_bundle(
    formal_path: Path, decision_paths: dict[str, Path]
) -> dict[str, object]:
    if file_sha256(formal_path) != EXPECTED_FORMAL_CTP_SHA256:
        raise RuntimeError("Unexpected formal alpha-refinement CTP snapshot")
    formal = read_json(formal_path)
    if formal.get("analysis_id") != FORMAL_REFINEMENT_ANALYSIS_ID:
        raise RuntimeError("Formal alpha-refinement analysis identity mismatch")
    if (
        formal.get("barrier_data_accessed") is not False
        or formal.get("barrier_computation_or_densification_performed") is not False
        or formal.get("barrier_unembargoed") is not False
    ):
        raise RuntimeError("Formal alpha-refinement CTP broke the barrier embargo")
    if formal.get("barrier_remains_embargoed") is not True:
        raise RuntimeError("Formal alpha-refinement CTP did not retain embargo")
    if formal.get("refinement_report_sha256") != EXPECTED_REFINEMENT_REPORT_SHA256:
        raise RuntimeError("Formal refinement report provenance drifted")
    if formal.get("refinement_preregistration_sha256") != (
        EXPECTED_REFINEMENT_PREREG_SHA256
    ):
        raise RuntimeError("Formal refinement preregistration provenance drifted")
    if formal.get("verified_all_refinement_and_coarse_report_result_audit_hashes") is not True:
        raise RuntimeError("Formal refinement snapshot did not verify its hash chain")

    upstream_artifacts: dict[str, str] = {}
    for stem in (
        "refinement_report",
        "refinement_preregistration",
        "coarse_report",
        "coarse_preregistration",
        "coarse_ctp_report",
    ):
        path = Path(str(formal.get(stem))).resolve()
        expected = str(formal.get(f"{stem}_sha256"))
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"Formal upstream artifact drifted: {stem}")
        upstream_artifacts[str(path)] = expected
    formal_source_hashes = _verify_embedded_path_hashes(
        formal.get("verified_source_hashes"), "formal verified sources"
    )
    formal_result_hashes = _verify_embedded_path_hashes(
        formal.get("formal_result_hashes"), "formal member results"
    )
    formal_audit_hashes = _verify_embedded_path_hashes(
        formal.get("formal_member_audit_hashes"), "formal member audits"
    )

    pairs = [row for row in formal.get("pairs", []) if float(row.get("alpha", -1)) == CANONICAL_ALPHA]
    if len(pairs) != 1 or pairs[0].get("source") != "refinement":
        raise RuntimeError("Formal snapshot lacks exactly one alpha15 refinement pair")
    pair = pairs[0]
    eta_one_path = decision_paths["same_alpha_eta_one_candidate_result"]
    legacy_path = decision_paths["same_alpha_legacy_result"]
    if file_sha256(eta_one_path) != EXPECTED_ETA_ONE_RESULT_SHA256:
        raise RuntimeError("Alpha15 eta=1 result hash drifted")
    if file_sha256(legacy_path) != EXPECTED_LEGACY_RESULT_SHA256:
        raise RuntimeError("Alpha15 legacy result hash drifted")
    eta_one = base.load_member_result(eta_one_path, CANONICAL_ALPHA, "candidate")
    legacy = base.load_member_result(legacy_path, CANONICAL_ALPHA, "matched_control")
    _assert_curve_snapshot(eta_one, pair["candidate"], "alpha15 eta=1")
    _assert_curve_snapshot(legacy, pair["matched_control"], "alpha15 legacy")

    for key, expected in (
        ("same_alpha_eta_one_candidate_audit", EXPECTED_ETA_ONE_AUDIT_SHA256),
        ("same_alpha_legacy_audit", EXPECTED_LEGACY_AUDIT_SHA256),
    ):
        audit_path = decision_paths[key]
        if file_sha256(audit_path) != expected or formal_audit_hashes.get(str(audit_path)) != expected:
            raise RuntimeError(f"Formal comparator audit drifted: {key}")

    official_path = decision_paths["official_stage4_v4"]
    if file_sha256(official_path) != EXPECTED_OFFICIAL_ARCHIVE_SHA256:
        raise RuntimeError("Official archive hash drifted")
    official = base.load_official_curve(official_path)
    if official != formal["official_stage4_v4"]:
        raise RuntimeError("Official curve drifted from the formal CTP snapshot")
    official_transitive = eta_base.verify_official_transitive_hashes(
        official_path, official
    )
    return {
        "formal": formal,
        "eta_one": eta_one,
        "legacy": legacy,
        "official": official,
        "official_transitive": official_transitive,
        "upstream_artifact_hashes": upstream_artifacts,
        "formal_source_hashes": formal_source_hashes,
        "formal_result_hashes": formal_result_hashes,
        "formal_audit_hashes": formal_audit_hashes,
    }


def validate_preregistration(
    prereg: dict[str, object], report_path: Path
) -> None:
    if prereg.get("status") != "preregistered":
        raise RuntimeError("Eta1.1 preregistration status mismatch")
    if prereg.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Eta1.1 preregistration protocol mismatch")
    if Path(str(prereg.get("planned_report"))).resolve() != report_path.resolve():
        raise RuntimeError("Eta1.1 report path was not preregistered")
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
            raise RuntimeError(f"Eta1.1 preregistration drift at {key}")
    exact = {
        "single_arm": "fixed post-selection eta=1.1 at fixed alpha=15",
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
        "post_hoc_grid_expansion_allowed": False,
        "terminal_or_raw_iperp_ranking_allowed": False,
        "independent_member_paths_and_logs": True,
        "overwrite_or_force_allowed": False,
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
    }
    for key, expected in exact.items():
        if prereg.get(key) != expected or type(prereg.get(key)) is not type(expected):
            raise RuntimeError(f"Eta1.1 preregistration drift at {key}")
    if prereg.get("eligible_record_definition") != "initial plus reparam_committed states":
        raise RuntimeError("Eta1.1 eligibility definition drifted")
    if prereg.get("mesh_gate") != {
        "segment_length_cv_max": MESH_CV_MAX,
        "segment_length_ratio_max": MESH_RATIO_MAX,
    }:
        raise RuntimeError("Eta1.1 mesh gate drifted")
    if prereg.get("cost_to_precision_definition") != (
        "minimum actual optimization NF at I_perp <= threshold; no interpolation"
    ):
        raise RuntimeError("Eta1.1 CTP definition drifted")
    expected_comparators = [
        "same_alpha_eta_one_candidate",
        "same_alpha_legacy",
        "official_stage4_v4",
    ]
    if prereg.get("primary_comparators") != expected_comparators:
        raise RuntimeError("Eta1.1 comparator set/order drifted")
    if prereg.get("primary_success_criterion") != (
        "eta1.1 must weakly dominate every comparator over every reachable "
        "complete mesh-eligible Iperp CTP threshold, strictly improve at least "
        "one threshold versus each comparator, and reach a tightest eligible "
        "precision no worse than each comparator"
    ):
        raise RuntimeError("Eta1.1 primary success criterion drifted")
    if prereg.get("reporting_anchors") != list(REPORTING_ANCHORS):
        raise RuntimeError("Eta1.1 reporting anchors drifted")
    if set(prereg.get("decision_inputs", {})) != EXPECTED_DECISION_INPUT_KEYS:
        raise RuntimeError("Eta1.1 decision-input set drifted")
    if set(prereg.get("run_source_manifest", {})) != EXPECTED_RUN_SOURCE_KEYS:
        raise RuntimeError("Eta1.1 run-source manifest drifted")
    registered = prereg.get("registered_command_sha256")
    if not isinstance(registered, str) or len(registered) != 64:
        raise RuntimeError("Eta1.1 registered command hash is invalid")
    common = prereg.get("canonical_common_initial_path")
    if not isinstance(common, dict) or common.get("sha256") != (
        EXPECTED_COMMON_INITIAL_FILE_SHA256
    ) or common.get("initial_path_array_sha256") != (
        EXPECTED_COMMON_INITIAL_ARRAY_SHA256
    ):
        raise RuntimeError("Eta1.1 common-initial provenance drifted")
    assert_false_barrier_metadata(prereg, "preregistration")


def _verify_candidate_audit(
    audit_path: Path, result: Path, result_hash: str
) -> tuple[dict[str, object], list[str]]:
    audit = read_json(audit_path)
    assert_false_barrier_metadata(audit, "candidate_audit")
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
            raise RuntimeError(f"Eta1.1 candidate audit drift at {key}")
    if Path(str(audit.get("result"))).resolve() != result:
        raise RuntimeError("Eta1.1 candidate audit result path mismatch")
    if audit.get("result_sha256") != result_hash:
        raise RuntimeError("Eta1.1 candidate audit result hash mismatch")
    assert_no_barrier_result_fields(
        audit.get("result_fields_accessed"), "eta1p1_candidate_audit"
    )
    command = audit.get("command")
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise RuntimeError("Eta1.1 candidate audit command is invalid")
    return audit, command


def analyze(report_path: Path) -> dict[str, object]:
    report_path = report_path.resolve()
    if file_sha256(report_path) != EXPECTED_ETA1P1_REPORT_SHA256:
        raise RuntimeError("Unexpected formal eta1.1/alpha15 report snapshot")
    report = read_json(report_path)
    assert_false_barrier_metadata(report, "report")
    if report.get("status") != "complete" or report.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Eta1.1 report is not a complete registered protocol")
    prereg_path = Path(str(report.get("preregistration"))).resolve()
    if (
        file_sha256(prereg_path) != EXPECTED_ETA1P1_PREREGISTRATION_SHA256
        or report.get("preregistration_sha256")
        != EXPECTED_ETA1P1_PREREGISTRATION_SHA256
    ):
        raise RuntimeError("Eta1.1 preregistration hash mismatch")
    prereg = read_json(prereg_path)
    validate_preregistration(prereg, report_path)
    for key, value in prereg.items():
        if key != "status" and report.get(key) != value:
            raise RuntimeError(f"Eta1.1 report drifted from preregistration at {key}")
    if report.get("barrier_fields_accessed_by_launcher") is not False:
        raise RuntimeError("Eta1.1 report says launcher accessed barrier data")
    if report.get("formal_ranking_deferred_to_separate_readonly_iperp_ctp") is not True:
        raise RuntimeError("Eta1.1 launcher ranked before the readonly audit")

    run_sources = verify_manifest(prereg["run_source_manifest"], "run sources")
    require_source_basenames(run_sources, REQUIRED_RUN_SOURCE_BASENAMES, "run sources")
    root = Path(__file__).resolve().parents[1]
    decision_sources = decision_source_hashes(root)
    decision_paths = {
        name: verify_hash_record(record, f"decision_inputs.{name}")
        for name, record in prereg["decision_inputs"].items()
    }
    expected_hashes = {
        "alpha15_refinement_ctp_snapshot": EXPECTED_FORMAL_CTP_SHA256,
        "same_alpha_eta_one_candidate_result": EXPECTED_ETA_ONE_RESULT_SHA256,
        "same_alpha_eta_one_candidate_audit": EXPECTED_ETA_ONE_AUDIT_SHA256,
        "same_alpha_eta_one_candidate_command": EXPECTED_ETA_ONE_COMMAND_FILE_SHA256,
        "same_alpha_legacy_result": EXPECTED_LEGACY_RESULT_SHA256,
        "same_alpha_legacy_audit": EXPECTED_LEGACY_AUDIT_SHA256,
        "same_alpha_legacy_command": EXPECTED_LEGACY_COMMAND_FILE_SHA256,
        "official_stage4_v4": EXPECTED_OFFICIAL_ARCHIVE_SHA256,
    }
    for name, expected in expected_hashes.items():
        if file_sha256(decision_paths[name]) != expected:
            raise RuntimeError(f"Registered formal input drifted: {name}")

    common_record = prereg["canonical_common_initial_path"]
    common_path = verify_hash_record(common_record, "canonical_common_initial_path")
    with np.load(common_path, allow_pickle=False) as saved:
        require_keys(saved, NPZ_ACCESS_ALLOWLISTS["canonical_initial"], "common initial")
        common = np.ascontiguousarray(np.asarray(saved["initial_path"], dtype=np.float64))
    common_array_hash = hashlib.sha256(common.tobytes()).hexdigest()
    if common_array_hash != EXPECTED_COMMON_INITIAL_ARRAY_SHA256:
        raise RuntimeError("Canonical initial-path array hash drifted")

    reference_command = read_command(
        decision_paths["same_alpha_eta_one_candidate_command"]
    )
    legacy_command = read_command(decision_paths["same_alpha_legacy_command"])
    assert_reference_commands(reference_command, legacy_command)

    formal_bundle = load_formal_alpha15_bundle(
        decision_paths["alpha15_refinement_ctp_snapshot"], decision_paths
    )
    _verify_reference_audit(
        decision_paths["same_alpha_eta_one_candidate_audit"],
        decision_paths["same_alpha_eta_one_candidate_result"],
        EXPECTED_ETA_ONE_RESULT_SHA256,
        reference_command,
        "candidate",
    )
    _verify_reference_audit(
        decision_paths["same_alpha_legacy_audit"],
        decision_paths["same_alpha_legacy_result"],
        EXPECTED_LEGACY_RESULT_SHA256,
        legacy_command,
        "matched_control",
    )
    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError("Eta1.1 report must contain exactly one member")
    row = rows[0]
    if not isinstance(row, dict) or int(row.get("returncode", -1)) != 0:
        raise RuntimeError("Eta1.1 report contains a failed member")
    if float(row.get("eta", -1)) != ETA or float(row.get("alpha", -1)) != CANONICAL_ALPHA:
        raise RuntimeError("Eta1.1 report member identity mismatch")
    result = Path(str(row.get("result"))).resolve()
    audit_path = Path(str(row.get("member_audit"))).resolve()
    result_hash = str(row.get("result_sha256"))
    if file_sha256(result) != result_hash:
        raise RuntimeError("Eta1.1 result hash mismatch")
    if file_sha256(audit_path) != str(row.get("member_audit_sha256")):
        raise RuntimeError("Eta1.1 member-audit hash mismatch")
    audit, command = _verify_candidate_audit(audit_path, result, result_hash)
    shim_path = next(
        Path(path) for path in run_sources if Path(path).name == "run_ivm_v9_discrete_selected_eta_shim.py"
    )
    assert_eta1p1_command(command, reference_command, shim_path)
    registered_command = str(prereg["registered_command_sha256"])
    if command_sha256(command) != registered_command or audit.get(
        "scientific_command_sha256"
    ) != registered_command:
        raise RuntimeError("Eta1.1 registered command hash mismatch")

    candidate = load_eta1p1_result(result)
    if candidate["initial_path_sha256"] != common_array_hash:
        raise RuntimeError("Eta1.1 candidate changed the canonical initial path")
    for key in ("complete_state_NF", "mesh_eligible_complete_state_NF"):
        if list(audit.get(key, [])) != candidate[key]:
            raise RuntimeError(f"Eta1.1 member-audit mismatch at {key}")
    if int(audit.get("final_optimization_NF", -1)) != candidate["final_optimization_NF"]:
        raise RuntimeError("Eta1.1 member-audit terminal NF mismatch")
    eta_one = formal_bundle["eta_one"]
    legacy = formal_bundle["legacy"]
    official = formal_bundle["official"]
    initial_hashes = {
        candidate["initial_path_sha256"],
        eta_one["initial_path_sha256"],
        legacy["initial_path_sha256"],
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
    success = triple_success(comparisons)
    anchors = [float(value) for value in prereg.get("reporting_anchors", REPORTING_ANCHORS)]
    return {
        "analysis_id": ANALYSIS_ID,
        "created_unix": time.time(),
        "read_only_formal_input_audit": True,
        "barrier_data_accessed": False,
        "barrier_computation_or_densification_performed": False,
        "barrier_unembargoed": False,
        "barrier_remains_embargoed": True,
        "eta1p1_report": str(report_path),
        "eta1p1_report_sha256": file_sha256(report_path),
        "eta1p1_preregistration": str(prereg_path),
        "eta1p1_preregistration_sha256": file_sha256(prereg_path),
        "canonical_common_initial_path": str(common_path),
        "canonical_common_initial_file_sha256": file_sha256(common_path),
        "canonical_common_initial_array_sha256": common_array_hash,
        "verified_run_source_hashes": run_sources,
        "verified_decision_source_hashes": decision_sources,
        "verified_decision_input_hashes": {
            str(path): file_sha256(path) for path in decision_paths.values()
        },
        "verified_formal_upstream_artifact_hashes": formal_bundle[
            "upstream_artifact_hashes"
        ],
        "verified_formal_source_hashes": formal_bundle["formal_source_hashes"],
        "verified_formal_result_hashes": formal_bundle["formal_result_hashes"],
        "verified_formal_member_audit_hashes": formal_bundle[
            "formal_audit_hashes"
        ],
        "verified_report_prereg_source_command_result_audit_hashes": True,
        "eta1p1_member_audit": str(audit_path),
        "eta1p1_member_audit_sha256": file_sha256(audit_path),
        "registered_scientific_command_sha256": registered_command,
        "fixed_eta": ETA,
        "fixed_alpha": CANONICAL_ALPHA,
        "fixed_residual_ratio": CANONICAL_RATIO,
        "fixed_max_evaluated_steps": CANONICAL_INNER_CAP,
        "fixed_lbfgs_history": CANONICAL_HISTORY,
        "fixed_budget_per_image": CANONICAL_BUDGET,
        "excluded_formal_controls": ["alpha10", "alpha12.5", "eta1.25"],
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
        "official_transitive_hashes": formal_bundle["official_transitive"],
        "formal_alpha15_refinement_ctp": str(
            decision_paths["alpha15_refinement_ctp_snapshot"]
        ),
        "formal_alpha15_refinement_ctp_sha256": EXPECTED_FORMAL_CTP_SHA256,
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
        "triple_tightest_precision_ok": all(
            bool(row["candidate_does_not_lose_tightest_precision"])
            for row in comparisons.values()
        ),
        "primary_success_criterion_satisfied": success,
        "fixed_eta1p1_full_curve_dominates_all_three": success,
        "single_candidate_no_multiwinner_or_tiebreak": True,
        "winner_claim_made": success,
        "post_hoc_eta_or_alpha_expansion_allowed": False,
        "barrier_unembargo_allowed_by_this_analysis": False,
    }


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
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{report.stem}_iperp_ctp_readonly_v1_{time.time_ns()}.json"
    payload["analysis_script"] = str(Path(__file__).resolve())
    payload["analysis_script_sha256"] = file_sha256(Path(__file__).resolve())
    assert_false_barrier_metadata(payload, "analysis_output")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
