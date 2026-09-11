#!/usr/bin/env python3
"""Barrier-blind I_perp CTP audit for the FFrbk selector takeover study.

Only the two candidate NPZs' I_perp, actual optimization-NF, completeness,
mesh-quality, and immutable run-identity fields are indexed.  The matched
Current0, alpha=15 legacy, and official-v4 curves are consumed from the
already formal, hash-pinned matched-kkt0 I_perp audit.  No physical objective,
barrier, densified, stored-path, or energy array is opened.

The registered rule is unchanged: eligible observations are the initial state
plus reparameterization-committed states that pass CV<=0.0065 and segment
ratio<=1.03.  Cost-to-precision is the minimum *observed* optimization NF at
or below a comparator Pareto threshold.  Missing thresholds are U/losses;
there is no interpolation, continuation, repetition, or extrapolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

import analyze_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_iperp_ctp as matched


ANALYSIS_ID = (
    "stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_iperp_ctp_v1"
)
PROTOCOL_ID = "stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1"
ARMS = ("one_way_takeover", "forced_operator_lbfgs_selection")
PRIMARY_COMPARATORS = (
    "matched_current0",
    "same_alpha_legacy",
    "official_stage4_v4",
)

CANONICAL_ALPHA = 15.0
CANONICAL_ETA = 1.1
CANONICAL_RATIO = 0.8
CANONICAL_INNER_CAP = 10
CANONICAL_HISTORY = 7
CANONICAL_KKT_ATOL = 0.0
CANONICAL_BUDGET = 254
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
COMMON_INITIAL_ARRAY_SHA256 = (
    "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
)

EXPECTED_CAMPAIGN_REPORT_SHA256 = (
    "c64c471d612a22161ff1308bd969f83588b5de4a2aa67e62b4e4b848e96514ad"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "67dd26b93f216a2d7f429a1b31710bdf1b933c24e10f17779e2499621705fa4e"
)
EXPECTED_PRIOR_CTP_SHA256 = (
    "98fe38387a76ca11308eb96c071b5448542f3eff719620f7b7d67968c3ef193d"
)
EXPECTED_CURRENT0_RESULT_SHA256 = (
    "eff9d585dfbbd966744a972f9fa5b5a7b3da11639d2079c333560e34ce98ee07"
)
EXPECTED_LEGACY_RESULT_SHA256 = (
    "2ffa7aeb8122cbf4cdfecf9d4a372da8774c6485ec82f9b6693c3047c2d247dc"
)
EXPECTED_OFFICIAL_ARCHIVE_SHA256 = (
    "09ccbf7db731c96c0beda9d5879c9dd46cae2b428db8b71820aad1c56cb574fc"
)
EXPECTED_ARM_RESULT_SHA256 = {
    "one_way_takeover": (
        "1c81ad6fe876a0eb94f2fc55d5d2567737face95ff52b6fe1bdfffc536d16787"
    ),
    "forced_operator_lbfgs_selection": (
        "5d8169d1d16dab2836bef43affcaaec2dabf539a8e0f992c9050cd909faf4050"
    ),
}

COMMON_CURVE_KEYS = frozenset(
    {
        "active_alpha",
        "alpha",
        "alpha_mode",
        "configuration_sha256",
        "discrete_all_returned_states_evaluated",
        "discrete_history",
        "discrete_implementation_id",
        "discrete_inner_max_steps",
        "discrete_residual_ratio",
        "discrete_stop_mode",
        "frozen_subproblem",
        "i_perp",
        "initial_path_sha256",
        "plbfgs_kkt_atol",
        "reparam_committed",
        "reparam_mode",
        "segment_length_cv",
        "segment_length_ratio",
        "stop_complete_iperp",
        "stop_reason",
        "total_internal_joint_evaluations",
    }
)
SELECTOR_IDENTITY_KEYS = frozenset(
    {
        "discrete_ffrbk_selector_aborted_selection_count",
        "discrete_ffrbk_selector_aborted_solver_calls",
        "discrete_ffrbk_selector_alpha",
        "discrete_ffrbk_selector_eta",
        "discrete_ffrbk_selector_extra_physical_evaluations",
        "discrete_ffrbk_selector_final_implementation_id",
        "discrete_ffrbk_selector_h0_shadow_constructed",
        "discrete_ffrbk_selector_history",
        "discrete_ffrbk_selector_inner_max_steps",
        "discrete_ffrbk_selector_invalid_lbfgs_h0_fallback_enabled",
        "discrete_ffrbk_selector_kkt_atol",
        "discrete_ffrbk_selector_mode",
        "discrete_ffrbk_selector_mode_policy_id",
        "discrete_ffrbk_selector_one_way_latch_is_preference",
        "discrete_ffrbk_selector_operator_policy_id",
        "discrete_ffrbk_selector_operator_preserved",
        "discrete_ffrbk_selector_post_select_eta_composed",
        "discrete_ffrbk_selector_reparameterization_inside_subproblem",
        "discrete_ffrbk_selector_residual_ratio",
        "discrete_ffrbk_selector_runtime_policy_id",
        "discrete_ffrbk_selector_selector_only_change",
        "discrete_ffrbk_selector_stop_mode",
        "discrete_ffrbk_selector_true_lbfgs_only_construction",
        "discrete_ffrbk_selector_uncommitted_selection_count",
    }
)
NPZ_ACCESS_ALLOWLIST = COMMON_CURVE_KEYS | SELECTOR_IDENTITY_KEYS

MODE_POLICY_IDS = {
    "one_way_takeover": "one-way-strict-lbfgs-takeover-per-fixed-subproblem-v1",
    "forced_operator_lbfgs_selection": (
        "forced-operator-lbfgs-selection-with-h0-shadow-v1"
    ),
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


def scalar(saved: np.lib.npyio.NpzFile, key: str):
    return np.asarray(saved[key]).item()


def require_keys(saved: np.lib.npyio.NpzFile, keys: frozenset[str], context: str) -> None:
    missing = sorted(keys.difference(saved.files))
    if missing:
        raise RuntimeError(f"{context} is missing: {', '.join(missing)}")


def verify_hash_record(record: object, context: str) -> Path:
    if not isinstance(record, dict) or not {"path", "sha256"}.issubset(record):
        raise RuntimeError(f"Invalid hash record: {context}")
    path = Path(str(record["path"])).resolve()
    if not path.is_file() or file_sha256(path) != str(record["sha256"]):
        raise RuntimeError(f"Hash mismatch: {context}")
    return path


def command_sha256(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _hash_array_sequence(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _validate_preregistered_report(
    report_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    report_path = report_path.resolve()
    if file_sha256(report_path) != EXPECTED_CAMPAIGN_REPORT_SHA256:
        raise RuntimeError("Unexpected takeover campaign report snapshot")
    report = read_json(report_path)
    exact = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "cell_order": list(ARMS),
        "scientific_command_exact_matched_current0_clone": True,
        "only_runtime_entry_mode_and_routing_changed": True,
        "fixed_alpha": CANONICAL_ALPHA,
        "fixed_eta": CANONICAL_ETA,
        "fixed_residual_ratio": CANONICAL_RATIO,
        "fixed_max_evaluated_steps": CANONICAL_INNER_CAP,
        "fixed_lbfgs_history": CANONICAL_HISTORY,
        "fixed_kkt_atol": CANONICAL_KKT_ATOL,
        "budget_per_image": CANONICAL_BUDGET,
        "trajectory_record_count_allowed_range": [1, CANONICAL_BUDGET],
        "early_stop_counted_as_converged": False,
        "unobserved_tail_extrapolated": False,
        "formal_performance_metric": "I_perp_only",
        "candidate_unreachable_threshold_policy": "U_formal_loss",
        "energy_or_objective_backtracking_added": False,
        "physical_energy_arrays_accessed_by_wrapper_or_launcher": False,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "densified_barrier_computation_performed": False,
        "barrier_unembargo_allowed_by_this_protocol": False,
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
        "post_hoc_scientific_command_changes_allowed": False,
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise RuntimeError(f"Takeover report drift at {key}")
    if report.get("formal_eligibility") != (
        "initial plus reparam_committed states, then "
        "segment_length_cv<=0.0065 and segment_length_ratio<=1.03"
    ):
        raise RuntimeError("Takeover eligibility rule drifted")
    if report.get("formal_cost_to_precision") != (
        "minimum actual observed optimization NF with I_perp<=threshold; "
        "no interpolation or extrapolation"
    ):
        raise RuntimeError("Takeover CTP rule drifted")

    preregistration = Path(str(report.get("preregistration"))).resolve()
    if (
        not preregistration.is_file()
        or file_sha256(preregistration) != EXPECTED_PREREGISTRATION_SHA256
        or report.get("preregistration_sha256")
        != EXPECTED_PREREGISTRATION_SHA256
    ):
        raise RuntimeError("Takeover preregistration hash mismatch")
    prereg = read_json(preregistration)
    if prereg.get("status") != "preregistered":
        raise RuntimeError("Takeover preregistration status mismatch")
    for key, value in prereg.items():
        if key != "status" and report.get(key) != value:
            raise RuntimeError(f"Takeover report drifted from preregistration at {key}")

    # Every preregistered source/input/prior artifact remains byte-identical.
    verified: dict[str, str] = {}
    for manifest_name in (
        "prior_artifact_manifest",
        "run_source_manifest",
        "common_initial_and_topology",
    ):
        manifest = report.get(manifest_name)
        if not isinstance(manifest, dict) or not manifest:
            raise RuntimeError(f"Missing hash manifest: {manifest_name}")
        if manifest_name == "common_initial_and_topology":
            records = manifest
        else:
            records = manifest
        for name, record in records.items():
            path = verify_hash_record(record, f"{manifest_name}.{name}")
            verified[f"{manifest_name}.{name}"] = file_sha256(path)

    prior = report["prior_artifact_manifest"]
    if prior["mechanism_ctp_readonly"]["sha256"] != EXPECTED_PRIOR_CTP_SHA256:
        raise RuntimeError("Prior formal CTP snapshot drifted")
    if prior["matched_current0_result"]["sha256"] != EXPECTED_CURRENT0_RESULT_SHA256:
        raise RuntimeError("Matched Current0 result provenance drifted")
    if prior["same_alpha_legacy_result"]["sha256"] != EXPECTED_LEGACY_RESULT_SHA256:
        raise RuntimeError("Legacy result provenance drifted")
    if prior["official_stage4_v4"]["sha256"] != EXPECTED_OFFICIAL_ARCHIVE_SHA256:
        raise RuntimeError("Official-v4 archive provenance drifted")
    return report, preregistration, verified


def _validate_member_audit(path: Path, arm: str, result_hash: str) -> dict[str, Any]:
    audit = read_json(path)
    exact = {
        "member_policy_version": PROTOCOL_ID,
        "arm": arm,
        "alpha": CANONICAL_ALPHA,
        "eta": CANONICAL_ETA,
        "residual_ratio": CANONICAL_RATIO,
        "max_evaluated_steps": CANONICAL_INNER_CAP,
        "lbfgs_history": CANONICAL_HISTORY,
        "kkt_atol": CANONICAL_KKT_ATOL,
        "absolute_kkt_stop_enabled": False,
        "budget_per_image": CANONICAL_BUDGET,
        "early_stop_is_scientific_convergence": False,
        "energy_or_objective_backtracking": False,
        "barrier_fields_accessed": False,
        "barrier_computation_or_densification_performed_by_wrapper": False,
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
        "result_sha256": result_hash,
        "physical_energy_or_barrier_arrays_accessed": False,
    }
    for key, expected in exact.items():
        if audit.get(key) != expected:
            raise RuntimeError(f"{arm} member audit drift at {key}")
    return audit


def extract_campaign_results(
    report_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, dict[str, Any]], dict[str, str]]:
    report, preregistration, verified = _validate_preregistered_report(report_path)
    rows = report.get("results")
    if not isinstance(rows, list) or [row.get("arm") for row in rows] != list(ARMS):
        raise RuntimeError("Takeover result order/cell set drifted")
    registered = report.get("registered_commands")
    if not isinstance(registered, dict) or set(registered) != set(ARMS):
        raise RuntimeError("Takeover registered-command set drifted")

    extracted: dict[str, dict[str, Any]] = {}
    root = Path(__file__).resolve().parents[1]
    for row in rows:
        arm = str(row["arm"])
        if int(row.get("returncode", -1)) != 0:
            raise RuntimeError(f"Takeover arm failed: {arm}")
        result = Path(str(row.get("result"))).resolve()
        result_hash = str(row.get("result_sha256"))
        if (
            not result.is_file()
            or file_sha256(result) != result_hash
            or result_hash != EXPECTED_ARM_RESULT_SHA256[arm]
        ):
            raise RuntimeError(f"{arm} result provenance mismatch")
        count = int(row.get("trajectory_record_count", -1))
        if not 1 <= count <= CANONICAL_BUDGET:
            raise RuntimeError(f"{arm} trajectory length is outside 1..254")
        early = count < CANONICAL_BUDGET
        if bool(row.get("early_stop")) != early:
            raise RuntimeError(f"{arm} early-stop flag is inconsistent")
        if bool(row.get("trajectory_complete_to_budget")) == early:
            raise RuntimeError(f"{arm} completion flag is inconsistent")
        if row.get("scientific_run_converged") is not False:
            raise RuntimeError(f"{arm} was incorrectly counted as convergence")
        if early and row.get("precision_beyond_tightest_reached_unreachable") is not True:
            raise RuntimeError(f"{arm} early tail was not declared unreachable")

        command_record = registered[arm]
        command = command_record.get("command")
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            raise RuntimeError(f"{arm} registered command is invalid")
        compact = command_sha256(command)
        if (
            command_record.get("compact_sha256") != compact
            or row.get("registered_command_sha256") != compact
        ):
            raise RuntimeError(f"{arm} registered command hash mismatch")
        label = str(command_record.get("label"))
        command_log = root / "logs" / PROTOCOL_ID / "members" / label / "command.json"
        if json.loads(command_log.read_text(encoding="utf-8")) != command:
            raise RuntimeError(f"{arm} executed command log drifted")

        member_audit = Path(str(row.get("member_audit"))).resolve()
        if (
            not member_audit.is_file()
            or file_sha256(member_audit) != row.get("member_audit_sha256")
        ):
            raise RuntimeError(f"{arm} member-audit hash mismatch")
        audit = _validate_member_audit(member_audit, arm, result_hash)
        if audit.get("scientific_command_sha256") != compact or audit.get("command") != command:
            raise RuntimeError(f"{arm} member-audit command provenance mismatch")
        extracted[arm] = {
            "row": row,
            "result": result,
            "result_sha256": result_hash,
            "member_audit": member_audit,
            "member_audit_sha256": file_sha256(member_audit),
            "command_log": command_log,
            "command_log_sha256": file_sha256(command_log),
            "registered_command_sha256": compact,
        }
    return report, preregistration, extracted, verified


def _validate_common_identity(
    saved: np.lib.npyio.NpzFile, active: np.ndarray, arm: str
) -> None:
    exact = {
        "alpha": CANONICAL_ALPHA,
        "alpha_mode": "fixed",
        "discrete_all_returned_states_evaluated": True,
        "discrete_history": CANONICAL_HISTORY,
        "discrete_inner_max_steps": CANONICAL_INNER_CAP,
        "discrete_residual_ratio": CANONICAL_RATIO,
        "discrete_stop_mode": "r08c10",
        "frozen_subproblem": "discrete_b040",
        "initial_path_sha256": COMMON_INITIAL_ARRAY_SHA256,
        "plbfgs_kkt_atol": CANONICAL_KKT_ATOL,
        "reparam_mode": "state_curve",
        "stop_complete_iperp": -1.0,
    }
    for key, expected in exact.items():
        observed = scalar(saved, key)
        if observed != expected:
            raise RuntimeError(f"{arm} identity mismatch at {key}")
    if active.ndim != 1 or not active.size or not np.all(active == CANONICAL_ALPHA):
        raise RuntimeError(f"{arm} did not keep alpha=15")


def _validate_selector_identity(saved: np.lib.npyio.NpzFile, arm: str) -> None:
    exact = {
        "discrete_ffrbk_selector_mode": arm,
        "discrete_ffrbk_selector_mode_policy_id": MODE_POLICY_IDS[arm],
        "discrete_ffrbk_selector_alpha": CANONICAL_ALPHA,
        "discrete_ffrbk_selector_eta": CANONICAL_ETA,
        "discrete_ffrbk_selector_history": CANONICAL_HISTORY,
        "discrete_ffrbk_selector_inner_max_steps": CANONICAL_INNER_CAP,
        "discrete_ffrbk_selector_residual_ratio": CANONICAL_RATIO,
        "discrete_ffrbk_selector_kkt_atol": CANONICAL_KKT_ATOL,
        "discrete_ffrbk_selector_stop_mode": "r08c10",
        "discrete_ffrbk_selector_runtime_policy_id": (
            "ala-discrete-ffrbk-selector-runtime-v1"
        ),
        "discrete_ffrbk_selector_operator_policy_id": (
            "frozen-ffrbk-h0-and-operator-initialized-lbfgs-v1"
        ),
        "discrete_ffrbk_selector_one_way_latch_is_preference": (
            arm == "one_way_takeover"
        ),
        "discrete_ffrbk_selector_selector_only_change": True,
        "discrete_ffrbk_selector_operator_preserved": True,
        "discrete_ffrbk_selector_reparameterization_inside_subproblem": False,
        "discrete_ffrbk_selector_h0_shadow_constructed": True,
        "discrete_ffrbk_selector_invalid_lbfgs_h0_fallback_enabled": True,
        "discrete_ffrbk_selector_true_lbfgs_only_construction": False,
        "discrete_ffrbk_selector_post_select_eta_composed": True,
        "discrete_ffrbk_selector_extra_physical_evaluations": 0,
        "discrete_ffrbk_selector_aborted_selection_count": 0,
        "discrete_ffrbk_selector_aborted_solver_calls": 0,
        "discrete_ffrbk_selector_uncommitted_selection_count": 0,
    }
    for key, expected in exact.items():
        if scalar(saved, key) != expected:
            raise RuntimeError(f"{arm} selector identity mismatch at {key}")
    implementation = str(scalar(saved, "discrete_ffrbk_selector_final_implementation_id"))
    if implementation != str(scalar(saved, "discrete_implementation_id")):
        raise RuntimeError(f"{arm} final implementation identity is inconsistent")


def load_arm_result(
    result: Path,
    arm: str,
    *,
    expected_sha256: str,
    expected_record_count: int,
) -> dict[str, object]:
    if arm not in ARMS:
        raise ValueError(f"Unknown takeover arm: {arm}")
    result = result.resolve()
    if file_sha256(result) != expected_sha256:
        raise RuntimeError(f"{arm} result hash mismatch")
    with np.load(result, allow_pickle=False) as saved:
        require_keys(saved, NPZ_ACCESS_ALLOWLIST, str(result))
        active = np.asarray(saved["active_alpha"], dtype=float)
        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        mesh_ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        _validate_common_identity(saved, active, arm)
        _validate_selector_identity(saved, arm)
        if cost.size != expected_record_count or active.shape != cost.shape:
            raise RuntimeError(f"{arm} trajectory length/identity is inconsistent")
        curve = matched.build_observed_eligible_curve(
            cost, i_perp, committed, cv, mesh_ratio
        )
        stop_reason = str(scalar(saved, "stop_reason"))
        if not stop_reason or (
            cost.size < CANONICAL_BUDGET and stop_reason == "evaluation_budget"
        ):
            raise RuntimeError(f"{arm} termination identity is inconsistent")
        return {
            "arm": arm,
            "result": str(result),
            "result_sha256": expected_sha256,
            "configuration_sha256": str(scalar(saved, "configuration_sha256")),
            "initial_path_sha256": str(scalar(saved, "initial_path_sha256")),
            "implementation_id": str(scalar(saved, "discrete_implementation_id")),
            "termination_reason": stop_reason,
            "terminated_before_N254": bool(cost.size < CANONICAL_BUDGET),
            "final_observed_optimization_NF": int(cost[-1]),
            "unobserved_tail_extrapolated": False,
            "ctp_trajectory_sha256": _hash_array_sequence(
                [cost, i_perp, committed, cv, mesh_ratio]
            ),
            "npz_fields_accessed": sorted(NPZ_ACCESS_ALLOWLIST),
            **curve,
        }


def _validate_frozen_curve(curve: object, context: str) -> dict[str, Any]:
    checked = matched._validate_snapshot_curve(curve, context)
    points = checked["eligible_points"]
    if any(
        int(row["NF"]) <= 0 or not math.isfinite(float(row["i_perp"]))
        for row in points
    ):
        raise RuntimeError(f"Frozen curve contains an invalid point: {context}")
    return checked


def load_frozen_comparators(
    report: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    record = report["prior_artifact_manifest"]["mechanism_ctp_readonly"]
    snapshot_path = verify_hash_record(record, "prior mechanism CTP")
    if file_sha256(snapshot_path) != EXPECTED_PRIOR_CTP_SHA256:
        raise RuntimeError("Prior mechanism CTP snapshot hash drifted")
    snapshot = read_json(snapshot_path)
    if (
        snapshot.get("analysis_id") != matched.ANALYSIS_ID
        or snapshot.get("performance_metric") != "I_perp_only"
        or snapshot.get("physical_objective_arrays_accessed") is not False
        or snapshot.get("densification_performed") is not False
        or snapshot.get("unobserved_tail_extrapolated") is not False
    ):
        raise RuntimeError("Prior mechanism CTP identity drifted")
    current = _validate_frozen_curve(
        snapshot.get("arms", {}).get("current_ffrbk_selector"), "matched Current0"
    )
    legacy = _validate_frozen_curve(snapshot.get("same_alpha_legacy"), "legacy")
    official = _validate_frozen_curve(snapshot.get("official_stage4_v4"), "official")
    if current.get("result_sha256") != EXPECTED_CURRENT0_RESULT_SHA256:
        raise RuntimeError("Matched Current0 frozen curve provenance drifted")
    if legacy.get("result_sha256") != EXPECTED_LEGACY_RESULT_SHA256:
        raise RuntimeError("Legacy frozen curve provenance drifted")
    if official.get("archive_sha256") != EXPECTED_OFFICIAL_ARCHIVE_SHA256:
        raise RuntimeError("Official frozen curve provenance drifted")
    return current, legacy, official, snapshot_path


def compare_bidirectional(
    one_way: list[dict[str, object]],
    forced: list[dict[str, object]],
) -> dict[str, object]:
    forward = matched.comparison_with_wtlu(
        one_way, forced, "forced_operator_lbfgs_selection"
    )
    reverse = matched.comparison_with_wtlu(
        forced, one_way, "one_way_takeover"
    )
    dominators = [
        arm
        for arm, row in (
            ("one_way_takeover", forward),
            ("forced_operator_lbfgs_selection", reverse),
        )
        if bool(row["full_curve_dominates"])
    ]
    return {
        "one_way_takeover_vs_forced_operator_lbfgs_selection": forward,
        "forced_operator_lbfgs_selection_vs_one_way_takeover": reverse,
        "eligible_curves_exactly_identical": one_way == forced,
        "directional_full_curve_dominators": dominators,
        "unique_directional_dominator": dominators[0] if len(dominators) == 1 else None,
    }


def performance_comparisons(
    candidates: dict[str, list[dict[str, object]]],
    current: list[dict[str, object]],
    legacy: list[dict[str, object]],
    official: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    comparators = {
        "matched_current0": current,
        "same_alpha_legacy": legacy,
        "official_stage4_v4": official,
    }
    output: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        comparisons = {
            name: matched.comparison_with_wtlu(candidates[arm], curve, name)
            for name, curve in comparators.items()
        }
        tightest_ok = all(
            bool(row["candidate_does_not_lose_tightest_precision"])
            for row in comparisons.values()
        )
        triple = all(
            bool(row["full_curve_dominates"]) for row in comparisons.values()
        )
        output[arm] = {
            "comparisons": comparisons,
            "primary_comparator_set_exact": list(PRIMARY_COMPARATORS),
            "triple_tightest_precision_ok": tightest_ok,
            "triple_full_curve_dominance": triple,
            "qualifies": bool(tightest_ok and triple),
        }
    return output


def decide_winner(
    performance: dict[str, dict[str, object]], pair: dict[str, object]
) -> dict[str, object]:
    qualifiers = [arm for arm in ARMS if bool(performance[arm]["qualifies"])]
    if not qualifiers:
        return {
            "qualifiers": [],
            "winner_status": "no_winner_no_triple_full_curve_dominator",
            "locked_winner": None,
            "winner_claim_made": False,
        }
    if len(qualifiers) == 1:
        return {
            "qualifiers": qualifiers,
            "winner_status": "unique_triple_full_curve_dominator",
            "locked_winner": qualifiers[0],
            "winner_claim_made": True,
        }
    directional = pair.get("unique_directional_dominator")
    if directional in qualifiers:
        return {
            "qualifiers": qualifiers,
            "winner_status": "coqualified_unique_bidirectional_CTP_dominator",
            "locked_winner": directional,
            "winner_claim_made": True,
        }
    return {
        "qualifiers": qualifiers,
        "winner_status": "coqualified_no_unique_bidirectional_CTP_dominator",
        "locked_winner": None,
        "winner_claim_made": False,
    }


def _contains_report_hash_link(value: object, report: Path, digest: str) -> bool:
    if isinstance(value, dict):
        if {"path", "sha256"}.issubset(value):
            try:
                if Path(str(value["path"])).resolve() == report and value["sha256"] == digest:
                    return True
            except (OSError, ValueError):
                pass
        return any(_contains_report_hash_link(item, report, digest) for item in value.values())
    if isinstance(value, list):
        return any(_contains_report_hash_link(item, report, digest) for item in value)
    return False


def verify_optional_postrun(report_path: Path) -> dict[str, object]:
    matches = sorted(
        report_path.parent.glob(f"{report_path.stem}_postrun*.json")
    )
    if not matches:
        return {"available": False, "verified": False, "reason": "not_present"}
    if len(matches) != 1:
        raise RuntimeError("Expected at most one takeover postrun audit")
    path = matches[0].resolve()
    payload = read_json(path)
    digest = file_sha256(report_path)
    if not _contains_report_hash_link(payload, report_path.resolve(), digest):
        raise RuntimeError("Takeover postrun audit is not hash-linked to campaign report")
    return {
        "available": True,
        "verified": True,
        "path": str(path),
        "sha256": file_sha256(path),
        "campaign_report_hash_link_verified": True,
    }


def analyze(report_path: Path) -> dict[str, object]:
    report_path = report_path.resolve()
    report, preregistration, extracted, verified_manifest = extract_campaign_results(
        report_path
    )
    current, legacy, official, prior_snapshot = load_frozen_comparators(report)

    arms: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        record = extracted[arm]
        row = record["row"]
        loaded = load_arm_result(
            record["result"],
            arm,
            expected_sha256=record["result_sha256"],
            expected_record_count=int(row["trajectory_record_count"]),
        )
        if loaded["final_observed_optimization_NF"] != int(row["final_optimization_NF"]):
            raise RuntimeError(f"{arm} result/report final NF mismatch")
        if loaded["termination_reason"] != row["physical_stop_reason"]:
            raise RuntimeError(f"{arm} result/report stop-reason mismatch")
        loaded["member_audit"] = str(record["member_audit"])
        loaded["member_audit_sha256"] = record["member_audit_sha256"]
        loaded["registered_command_sha256"] = record["registered_command_sha256"]
        arms[arm] = loaded
    if {row["initial_path_sha256"] for row in arms.values()} != {
        COMMON_INITIAL_ARRAY_SHA256
    }:
        raise RuntimeError("Takeover arms do not share the registered initial path")

    curves = {arm: arms[arm]["eligible_points"] for arm in ARMS}
    pair = compare_bidirectional(curves[ARMS[0]], curves[ARMS[1]])
    pair["ctp_trajectory_sha256_equal"] = (
        arms[ARMS[0]]["ctp_trajectory_sha256"]
        == arms[ARMS[1]]["ctp_trajectory_sha256"]
    )
    performance = performance_comparisons(
        curves,
        current["eligible_points"],
        legacy["eligible_points"],
        official["eligible_points"],
    )
    winner = decide_winner(performance, pair)
    return {
        "analysis_id": ANALYSIS_ID,
        "created_unix": time.time(),
        "read_only_formal_input_audit": True,
        "performance_metric": "I_perp_only",
        "physical_objective_arrays_accessed": False,
        "physical_energy_arrays_accessed": False,
        "barrier_arrays_accessed": False,
        "densified_arrays_accessed": False,
        "densification_performed": False,
        "unobserved_tail_extrapolated": False,
        "campaign_report": str(report_path),
        "campaign_report_sha256": file_sha256(report_path),
        "preregistration": str(preregistration),
        "preregistration_sha256": file_sha256(preregistration),
        "prior_formal_mechanism_ctp": str(prior_snapshot),
        "prior_formal_mechanism_ctp_sha256": file_sha256(prior_snapshot),
        "provenance_manifest_verified": verified_manifest,
        "optional_postrun_audit": verify_optional_postrun(report_path),
        "fixed_alpha": CANONICAL_ALPHA,
        "fixed_eta": CANONICAL_ETA,
        "fixed_residual_ratio": CANONICAL_RATIO,
        "fixed_max_evaluated_steps": CANONICAL_INNER_CAP,
        "fixed_lbfgs_history": CANONICAL_HISTORY,
        "fixed_kkt_atol": CANONICAL_KKT_ATOL,
        "maximum_trajectory_length": CANONICAL_BUDGET,
        "eligibility": (
            "initial plus reparam_committed states, then "
            "segment_length_cv<=0.0065 and segment_length_ratio<=1.03"
        ),
        "cost_to_precision": (
            "minimum actual observed optimization NF with I_perp<=threshold; "
            "no interpolation or extrapolation"
        ),
        "unreachable_threshold_policy": "U_formal_loss",
        "winner_rule": (
            "candidate must full-curve dominate matched Current0, same-alpha "
            "legacy, and official-v4, with no loss of tightest eligible precision"
        ),
        "npz_access_allowlist": sorted(NPZ_ACCESS_ALLOWLIST),
        "arms": arms,
        "matched_current0": current,
        "same_alpha_legacy": legacy,
        "official_stage4_v4": official,
        "arm_bidirectional_comparison": pair,
        "performance_candidate_comparisons": performance,
        "performance_qualifiers": winner["qualifiers"],
        **winner,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    report = args.campaign_report.expanduser().resolve()
    payload = analyze(report)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else report.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"{report.stem}_iperp_ctp_readonly_v1_{time.time_ns()}.json"
    )
    script = Path(__file__).resolve()
    payload["analysis_script"] = str(script)
    payload["analysis_script_sha256"] = file_sha256(script)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
