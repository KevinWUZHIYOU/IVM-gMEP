#!/usr/bin/env python3
"""Read-only I_perp CTP audit for the matched L-BFGS mechanism ablation.

The prospective v1 launch stopped after ``scalar_selector`` terminated early,
before the launcher could run ``direct_lbfgs_only``.  This analyzer therefore
combines four immutable inputs without pretending that the failed launcher
produced a complete N254 grid:

* Current0 from the original incomplete v1 report;
* the observed, early-stalled scalar-selector NPZ left by that launch;
* direct-LBFGS-only from a separately audited continuation report; and
* the frozen alpha=15 legacy and official-v4 curves already recorded by the
  formal alpha15 eta=1.1 I_perp-only audit.

Every arm is evaluated only over its actually stored trajectory (1..254
states).  Complete records are the initial state plus reparameterization-
committed states, followed by the registered mesh gate.  Missing precision
after an early stop is always unreachable (U); no tail is interpolated,
continued, repeated, or extrapolated.  No physical objective or densified
quantity is indexed from any NPZ.
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

import analyze_stage4_ala_discrete_single_alpha_refinement_iperp_ctp as base
import analyze_stage4_ala_discrete_single_selected_eta_iperp_ctp as eta_base


ANALYSIS_ID = (
    "stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_iperp_ctp_v1"
)
PROTOCOL_ID = "stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1"
DIRECT_CONTINUATION_PROTOCOL_ID = (
    "stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_"
    "direct_continuation_v1"
)
ARMS = (
    "current_ffrbk_selector",
    "scalar_selector",
    "direct_lbfgs_only",
)
PERFORMANCE_CANDIDATES = ("scalar_selector", "direct_lbfgs_only")
PRIMARY_COMPARATORS = (
    "current_ffrbk_selector",
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

EXPECTED_ORIGINAL_REPORT_SHA256 = (
    "c1506422a651ee664b8cdb08a19c124bbab760435130042c300b7e721e7fa02f"
)
EXPECTED_ORIGINAL_PREREGISTRATION_SHA256 = (
    "04de96743da079b8b35247214a4ad3cf387157edb2134381cc08f95d3bab1f86"
)
EXPECTED_CURRENT0_RESULT_SHA256 = (
    "eff9d585dfbbd966744a972f9fa5b5a7b3da11639d2079c333560e34ce98ee07"
)
EXPECTED_SCALAR_EARLY_STALL_RESULT_SHA256 = (
    "ef5b54cd3127c86abf82ad934bb932f89e17e3008037fbd1c38a2570d868aea4"
)
EXPECTED_ALPHA15_FORMAL_CTP_SHA256 = (
    "0ef85ac737d57c5a5b6b7a2e725eef8684e66e225934b9e8858c167fa887d9ae"
)
EXPECTED_LEGACY_RESULT_SHA256 = (
    "2ffa7aeb8122cbf4cdfecf9d4a372da8774c6485ec82f9b6693c3047c2d247dc"
)
EXPECTED_OFFICIAL_ARCHIVE_SHA256 = (
    "09ccbf7db731c96c0beda9d5879c9dd46cae2b428db8b71820aad1c56cb574fc"
)

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
        "state_curve_direct_objective_acceptance",
        "stop_complete_iperp",
        "stop_reason",
        "total_internal_joint_evaluations",
    }
)
SCALAR_IDENTITY_KEYS = frozenset(
    {
        "discrete_scalar_lbfgs_direct_only",
        "discrete_scalar_lbfgs_ffrbk_applied_inside_fixed_solver",
        "discrete_scalar_lbfgs_ffrbk_solve_normal_calls_inside_fixed_solver",
        "discrete_scalar_lbfgs_fixed_solver_preconditioner_is_none",
        "discrete_scalar_lbfgs_implementation_id",
        "discrete_scalar_lbfgs_mode",
        "discrete_scalar_lbfgs_selector_competition",
    }
)
NPZ_ACCESS_ALLOWLISTS = {
    "current_ffrbk_selector": COMMON_CURVE_KEYS,
    "scalar_selector": COMMON_CURVE_KEYS | SCALAR_IDENTITY_KEYS,
    "direct_lbfgs_only": COMMON_CURVE_KEYS | SCALAR_IDENTITY_KEYS,
}

MECHANISM_PAIR_SPECS = (
    ("ffrbk_operator_effect", "current_ffrbk_selector", "scalar_selector"),
    ("selector_effect", "scalar_selector", "direct_lbfgs_only"),
    ("combined_effect", "current_ffrbk_selector", "direct_lbfgs_only"),
)
CONTINUATION_LINK_KEYS = (
    "source_incomplete_report",
    "original_incomplete_report",
    "v1_incomplete_report",
    "incomplete_report",
    "parent_report",
    "source_report",
)


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


def require_keys(
    saved: np.lib.npyio.NpzFile, keys: frozenset[str], context: str
) -> None:
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


def _hash_array_sequence(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def build_observed_eligible_curve(
    cost,
    i_perp,
    committed,
    cv,
    mesh_ratio,
) -> dict[str, object]:
    """Build a complete+mesh curve from exactly the observed trajectory."""
    cost = np.asarray(cost, dtype=int)
    i_perp = np.asarray(i_perp, dtype=float)
    committed = np.asarray(committed, dtype=bool)
    cv = np.asarray(cv, dtype=float)
    mesh_ratio = np.asarray(mesh_ratio, dtype=float)
    base.validate_curve_arrays(cost, i_perp, cv, mesh_ratio, "observed trajectory")
    if not 1 <= cost.size <= CANONICAL_BUDGET:
        raise RuntimeError("Observed trajectory length is outside 1..254")
    if committed.shape != cost.shape:
        raise RuntimeError("Observed completeness array is misaligned")

    complete = committed.copy()
    complete[0] = True
    eligible = complete & (cv <= MESH_CV_MAX) & (mesh_ratio <= MESH_RATIO_MAX)
    points = [
        {
            "NF": int(cost[index]),
            "i_perp": float(i_perp[index]),
            "segment_length_cv": float(cv[index]),
            "segment_length_ratio": float(mesh_ratio[index]),
            "is_initial": bool(index == 0),
            "reparam_committed": bool(committed[index]),
        }
        for index in np.flatnonzero(eligible)
    ]
    return {
        "observed_trajectory_length": int(cost.size),
        "complete_state_NF": cost[complete].tolist(),
        "mesh_eligible_complete_state_NF": cost[eligible].tolist(),
        "eligible_points": points,
        "pareto_records": base.pareto_records(points),
    }


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
        "state_curve_direct_objective_acceptance": "per_image",
        "stop_complete_iperp": -1.0,
    }
    for key, expected in exact.items():
        observed = scalar(saved, key)
        if observed != expected or type(observed) is not type(expected):
            # numpy scalar conversion makes integer/float type strictness noisy.
            if isinstance(expected, (int, float)) and float(observed) == float(expected):
                continue
            raise RuntimeError(f"{arm} identity mismatch at {key}")
    if active.ndim != 1 or not active.size or not np.all(active == CANONICAL_ALPHA):
        raise RuntimeError(f"{arm} did not keep alpha=15 over its observed states")


def _validate_scalar_identity(saved: np.lib.npyio.NpzFile, arm: str) -> None:
    expected_direct = arm == "direct_lbfgs_only"
    exact = {
        "discrete_scalar_lbfgs_mode": arm,
        "discrete_scalar_lbfgs_direct_only": expected_direct,
        "discrete_scalar_lbfgs_selector_competition": not expected_direct,
        "discrete_scalar_lbfgs_fixed_solver_preconditioner_is_none": True,
        "discrete_scalar_lbfgs_ffrbk_applied_inside_fixed_solver": False,
        "discrete_scalar_lbfgs_ffrbk_solve_normal_calls_inside_fixed_solver": 0,
    }
    for key, expected in exact.items():
        if scalar(saved, key) != expected:
            raise RuntimeError(f"{arm} scalar identity mismatch at {key}")
    implementation = str(scalar(saved, "discrete_scalar_lbfgs_implementation_id"))
    if implementation != str(scalar(saved, "discrete_implementation_id")):
        raise RuntimeError(f"{arm} implementation identity is inconsistent")
    if arm not in implementation and (
        "direct-lbfgs-only" not in implementation
        if expected_direct
        else "scalar-h0-vs-lbfgs" not in implementation
    ):
        raise RuntimeError(f"{arm} implementation identity lacks its mode policy")


def load_arm_result(
    result: Path,
    arm: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Load only I_perp/mesh/completeness fields from one observed arm."""
    if arm not in ARMS:
        raise ValueError(f"Unknown mechanism arm: {arm}")
    result = result.resolve()
    actual_hash = file_sha256(result)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise RuntimeError(f"{arm} result hash mismatch")
    allowlist = NPZ_ACCESS_ALLOWLISTS[arm]
    with np.load(result, allow_pickle=False) as saved:
        require_keys(saved, allowlist, str(result))
        active = np.asarray(saved["active_alpha"], dtype=float)
        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        mesh_ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        _validate_common_identity(saved, active, arm)
        if active.shape != cost.shape:
            raise RuntimeError(f"{arm} active-alpha array is misaligned")
        if arm in PERFORMANCE_CANDIDATES:
            _validate_scalar_identity(saved, arm)
        elif "discrete_scalar_lbfgs_mode" in saved.files:
            raise RuntimeError("Current0 unexpectedly contains the scalar runtime")
        curve = build_observed_eligible_curve(
            cost, i_perp, committed, cv, mesh_ratio
        )
        stop_reason = str(scalar(saved, "stop_reason"))
        if not stop_reason:
            raise RuntimeError(f"{arm} has an empty stop reason")
        if cost.size < CANONICAL_BUDGET and stop_reason == "evaluation_budget":
            raise RuntimeError(f"{arm} claims budget exhaustion before N254")
        return {
            "arm": arm,
            "result": str(result),
            "result_sha256": actual_hash,
            "configuration_sha256": str(scalar(saved, "configuration_sha256")),
            "initial_path_sha256": str(scalar(saved, "initial_path_sha256")),
            "termination_reason": stop_reason,
            "terminated_before_N254": bool(cost.size < CANONICAL_BUDGET),
            "final_observed_optimization_NF": int(cost[-1]),
            "unobserved_tail_extrapolated": False,
            "ctp_trajectory_sha256": _hash_array_sequence(
                [cost, i_perp, committed, cv, mesh_ratio]
            ),
            "npz_fields_accessed": sorted(allowlist),
            **curve,
        }


def comparison_with_wtlu(
    candidate: list[dict[str, object]],
    comparator: list[dict[str, object]],
    comparator_name: str,
) -> dict[str, object]:
    comparison = eta_base.comparison_with_wtlu(
        candidate, comparator, comparator_name
    )
    comparison["unobserved_tail_extrapolated"] = False
    comparison["candidate_unreachable_policy"] = "U_formal_loss"
    return comparison


def bidirectional_mechanism_comparisons(
    curves: dict[str, list[dict[str, object]]]
) -> dict[str, dict[str, object]]:
    if set(curves) != set(ARMS):
        raise RuntimeError("Mechanism curve set must contain exactly Current0/A/B")
    output: dict[str, dict[str, object]] = {}
    for effect, left, right in MECHANISM_PAIR_SPECS:
        left_vs_right = comparison_with_wtlu(
            curves[left], curves[right], right
        )
        right_vs_left = comparison_with_wtlu(
            curves[right], curves[left], left
        )
        directional = [
            arm
            for arm, row in ((left, left_vs_right), (right, right_vs_left))
            if bool(row["full_curve_dominates"])
        ]
        output[effect] = {
            "left_arm": left,
            "right_arm": right,
            f"{left}_vs_{right}": left_vs_right,
            f"{right}_vs_{left}": right_vs_left,
            "directional_full_curve_dominators": directional,
            "global_mechanism_superiority_claim_allowed": len(directional) == 1,
            "crossing_or_no_global_dominator": len(directional) == 0,
        }
    return output


def performance_comparisons(
    curves: dict[str, list[dict[str, object]]],
    legacy: list[dict[str, object]],
    official: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for candidate in PERFORMANCE_CANDIDATES:
        comparisons = {
            "current_ffrbk_selector": comparison_with_wtlu(
                curves[candidate],
                curves["current_ffrbk_selector"],
                "current_ffrbk_selector",
            ),
            "same_alpha_legacy": comparison_with_wtlu(
                curves[candidate], legacy, "same_alpha_legacy"
            ),
            "official_stage4_v4": comparison_with_wtlu(
                curves[candidate], official, "official_stage4_v4"
            ),
        }
        qualified = all(
            bool(comparisons[name]["full_curve_dominates"])
            for name in PRIMARY_COMPARATORS
        )
        output[candidate] = {
            "comparisons": comparisons,
            "full_curve_dominates_all_three": qualified,
            "primary_comparator_set_exact": list(PRIMARY_COMPARATORS),
        }
    return output


def decide_winner(
    performance: dict[str, dict[str, object]],
    mechanisms: dict[str, dict[str, object]],
) -> dict[str, object]:
    qualifiers = [
        arm
        for arm in PERFORMANCE_CANDIDATES
        if bool(performance[arm]["full_curve_dominates_all_three"])
    ]
    if len(qualifiers) == 0:
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
    selector = mechanisms["selector_effect"]
    directional = list(selector["directional_full_curve_dominators"])
    if len(directional) == 1:
        return {
            "qualifiers": qualifiers,
            "winner_status": "both_qualified_unique_directional_A_B_dominator",
            "locked_winner": directional[0],
            "winner_claim_made": True,
        }
    return {
        "qualifiers": qualifiers,
        "winner_status": "coqualified_no_unique_directional_A_B_dominator",
        "locked_winner": None,
        "winner_claim_made": False,
    }


def _successful_row(report: dict[str, Any], arm: str) -> dict[str, Any]:
    rows = report.get("results")
    if not isinstance(rows, list):
        raise RuntimeError("Report lacks a results list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("arm") == arm]
    if len(matches) != 1:
        raise RuntimeError(f"Report does not contain exactly one {arm} row")
    row = matches[0]
    if int(row.get("returncode", -1)) != 0:
        raise RuntimeError(f"Report {arm} row is not successful")
    if not {"result", "result_sha256"}.issubset(row):
        raise RuntimeError(f"Report {arm} row lacks result provenance")
    result = Path(str(row["result"])).resolve()
    if not result.is_file() or file_sha256(result) != str(row["result_sha256"]):
        raise RuntimeError(f"Report {arm} result hash mismatch")
    return row


def validate_original_incomplete_report(
    report_path: Path,
) -> tuple[dict[str, Any], Path]:
    report_path = report_path.resolve()
    if file_sha256(report_path) != EXPECTED_ORIGINAL_REPORT_SHA256:
        raise RuntimeError("Unexpected original incomplete v1 report snapshot")
    report = read_json(report_path)
    if report.get("protocol_id") != PROTOCOL_ID or report.get("status") != "incomplete":
        raise RuntimeError("Original v1 report identity/status mismatch")
    preregistration = Path(str(report.get("preregistration"))).resolve()
    if (
        file_sha256(preregistration) != EXPECTED_ORIGINAL_PREREGISTRATION_SHA256
        or report.get("preregistration_sha256")
        != EXPECTED_ORIGINAL_PREREGISTRATION_SHA256
    ):
        raise RuntimeError("Original v1 preregistration hash mismatch")
    exact = {
        "cell_order": list(ARMS),
        "fixed_alpha": CANONICAL_ALPHA,
        "fixed_eta": CANONICAL_ETA,
        "fixed_residual_ratio": CANONICAL_RATIO,
        "fixed_max_evaluated_steps": CANONICAL_INNER_CAP,
        "fixed_lbfgs_history": CANONICAL_HISTORY,
        "fixed_kkt_atol": CANONICAL_KKT_ATOL,
        "budget_per_image": CANONICAL_BUDGET,
        "eligible_record_definition": "initial plus reparam_committed states",
        "unreachable_threshold_policy": "candidate_unreachable_is_loss",
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise RuntimeError(f"Original v1 report drift at {key}")
    if report.get("mesh_gate") != {
        "segment_length_cv_max": MESH_CV_MAX,
        "segment_length_ratio_max": MESH_RATIO_MAX,
    }:
        raise RuntimeError("Original v1 mesh gate drifted")
    rows = report.get("results")
    if not isinstance(rows, list) or [row.get("arm") for row in rows] != [
        "current_ffrbk_selector",
        "scalar_selector",
    ]:
        raise RuntimeError("Original v1 failure prefix is not Current0 then scalar")
    _successful_row(report, "current_ffrbk_selector")
    scalar_row = rows[1]
    if int(scalar_row.get("returncode", 0)) == 0:
        raise RuntimeError("Original scalar row was expected to be launcher-incomplete")
    return report, preregistration


def resolve_scalar_early_stall_result(
    root: Path,
    report: dict[str, Any],
    explicit_result: Path | None = None,
) -> Path:
    registered = report.get("registered_commands")
    if not isinstance(registered, dict):
        raise RuntimeError("Original report lacks registered commands")
    scalar_record = registered.get("scalar_selector")
    if not isinstance(scalar_record, dict) or not isinstance(
        scalar_record.get("label"), str
    ):
        raise RuntimeError("Original report lacks the scalar registered label")
    label = str(scalar_record["label"])
    member_dir = root / "runs" / PROTOCOL_ID / "members"
    matches = sorted(
        member_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget254.npz")
    )
    if len(matches) != 1:
        raise RuntimeError("Could not resolve exactly one scalar early-stall NPZ")
    resolved = matches[0].resolve()
    if explicit_result is not None and explicit_result.resolve() != resolved:
        raise RuntimeError("Explicit scalar NPZ differs from registered orphan output")
    if file_sha256(resolved) != EXPECTED_SCALAR_EARLY_STALL_RESULT_SHA256:
        raise RuntimeError("Scalar early-stall NPZ hash drifted")
    return resolved


def _continuation_link_matches(
    report: dict[str, Any], original_report: Path
) -> bool:
    original_report = original_report.resolve()
    original_hash = file_sha256(original_report)
    for key in CONTINUATION_LINK_KEYS:
        if key not in report:
            continue
        value = report[key]
        if isinstance(value, dict):
            if (
                Path(str(value.get("path"))).resolve() == original_report
                and value.get("sha256") == original_hash
            ):
                return True
        elif Path(str(value)).resolve() == original_report:
            if report.get(f"{key}_sha256") == original_hash:
                return True
    prior = report.get("prior_artifact_manifest")
    if isinstance(prior, dict):
        record = prior.get("original_v1_report")
        if (
            isinstance(record, dict)
            and Path(str(record.get("path"))).resolve() == original_report
            and record.get("sha256") == original_hash
        ):
            return True
    return False


def extract_direct_continuation_result(
    continuation_report_path: Path, original_report_path: Path
) -> tuple[dict[str, Any], Path, str]:
    continuation_report_path = continuation_report_path.resolve()
    report = read_json(continuation_report_path)
    if report.get("status") != "complete":
        raise RuntimeError("Direct-only continuation report is not complete")
    exact = {
        "protocol_id": DIRECT_CONTINUATION_PROTOCOL_ID,
        "continuation_of_protocol": PROTOCOL_ID,
        "cell_order": ["direct_lbfgs_only"],
        "single_new_cell": True,
        "scientific_command_exact_original_direct_clone": True,
        "scientific_command_only_routing_changed": True,
        "fixed_alpha": CANONICAL_ALPHA,
        "fixed_eta": CANONICAL_ETA,
        "fixed_residual_ratio": CANONICAL_RATIO,
        "fixed_max_evaluated_steps": CANONICAL_INNER_CAP,
        "fixed_lbfgs_history": CANONICAL_HISTORY,
        "fixed_kkt_atol": CANONICAL_KKT_ATOL,
        "budget_per_image": CANONICAL_BUDGET,
        "trajectory_record_count_allowed_range": [1, CANONICAL_BUDGET],
        "early_stop_counted_as_converged": False,
        "unreachable_threshold_policy": "candidate_unreachable_is_loss",
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
        "post_hoc_scientific_command_changes_allowed": False,
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise RuntimeError(f"Direct continuation report drift at {key}")
    preregistration = Path(str(report.get("preregistration"))).resolve()
    if (
        not preregistration.is_file()
        or file_sha256(preregistration) != report.get("preregistration_sha256")
    ):
        raise RuntimeError("Direct continuation preregistration hash mismatch")
    prereg = read_json(preregistration)
    for key, value in prereg.items():
        if key != "status" and report.get(key) != value:
            raise RuntimeError(f"Direct continuation drifted from preregistration at {key}")
    if not _continuation_link_matches(report, original_report_path):
        raise RuntimeError("Direct-only report is not hash-linked to original v1")
    row = _successful_row(report, "direct_lbfgs_only")
    record_count = int(row.get("trajectory_record_count", -1))
    if not 1 <= record_count <= CANONICAL_BUDGET:
        raise RuntimeError("Direct continuation row has invalid trajectory length")
    early = record_count < CANONICAL_BUDGET
    if bool(row.get("early_stop")) != early:
        raise RuntimeError("Direct continuation early-stop flag is inconsistent")
    if bool(row.get("trajectory_complete_to_budget")) == early:
        raise RuntimeError("Direct continuation completion flag is inconsistent")
    if early and row.get("precision_beyond_tightest_reached_unreachable") is not True:
        raise RuntimeError("Direct early-stop tail was not declared unreachable")
    return report, Path(str(row["result"])).resolve(), str(row["result_sha256"])


def _validate_snapshot_curve(curve: object, context: str) -> dict[str, Any]:
    if not isinstance(curve, dict):
        raise RuntimeError(f"Invalid frozen curve: {context}")
    points = curve.get("eligible_points")
    records = curve.get("pareto_records")
    if not isinstance(points, list) or not points:
        raise RuntimeError(f"Frozen curve has no eligible points: {context}")
    if base.pareto_records(points) != records:
        raise RuntimeError(f"Frozen Pareto records drifted: {context}")
    for row in points:
        if not isinstance(row, dict) or not {"NF", "i_perp"}.issubset(row):
            raise RuntimeError(f"Frozen curve point is invalid: {context}")
        if int(row["NF"]) <= 0 or not math.isfinite(float(row["i_perp"])):
            raise RuntimeError(f"Frozen curve point is nonphysical: {context}")
    return curve


def load_frozen_comparators(
    original_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    decisions = original_report.get("decision_inputs")
    if not isinstance(decisions, dict):
        raise RuntimeError("Original report lacks decision inputs")
    snapshot_path = verify_hash_record(
        decisions.get("old_current_iperp_ctp"), "old_current_iperp_ctp"
    )
    if file_sha256(snapshot_path) != EXPECTED_ALPHA15_FORMAL_CTP_SHA256:
        raise RuntimeError("Unexpected frozen alpha15 formal CTP snapshot")
    snapshot = read_json(snapshot_path)
    if (
        snapshot.get("analysis_id")
        != "stage4_ala_discrete_single_eta1p1_alpha15_iperp_ctp_v1"
        or float(snapshot.get("fixed_alpha", -1)) != CANONICAL_ALPHA
        or float(snapshot.get("fixed_eta", -1)) != CANONICAL_ETA
    ):
        raise RuntimeError("Frozen alpha15 formal CTP identity mismatch")
    old_current = _validate_snapshot_curve(snapshot.get("candidate"), "old Current")
    legacy = _validate_snapshot_curve(snapshot.get("same_alpha_legacy"), "legacy")
    official = _validate_snapshot_curve(snapshot.get("official_stage4_v4"), "official")
    if old_current.get("initial_path_sha256") != COMMON_INITIAL_ARRAY_SHA256:
        raise RuntimeError("Frozen old Current initial path drifted")
    if legacy.get("result_sha256") != EXPECTED_LEGACY_RESULT_SHA256:
        raise RuntimeError("Frozen legacy result provenance drifted")
    if official.get("archive_sha256") != EXPECTED_OFFICIAL_ARCHIVE_SHA256:
        raise RuntimeError("Frozen official archive provenance drifted")
    return old_current, legacy, official, snapshot_path


def _assert_current0_curve_parity(
    current: dict[str, object], old_current: dict[str, Any]
) -> None:
    for key in (
        "complete_state_NF",
        "mesh_eligible_complete_state_NF",
        "eligible_points",
        "pareto_records",
    ):
        if current[key] != old_current[key]:
            raise RuntimeError(f"Current0/old-current CTP parity failed at {key}")


def analyze(
    original_report_path: Path,
    direct_continuation_report_path: Path,
    *,
    scalar_result_path: Path | None = None,
) -> dict[str, object]:
    original_report_path = original_report_path.resolve()
    continuation_path = direct_continuation_report_path.resolve()
    if not continuation_path.is_file():
        raise FileNotFoundError(
            "Direct-only continuation report must exist before formal analysis: "
            f"{continuation_path}"
        )
    original, preregistration = validate_original_incomplete_report(
        original_report_path
    )
    root = Path(__file__).resolve().parents[1]
    current_row = _successful_row(original, "current_ffrbk_selector")
    scalar_path = resolve_scalar_early_stall_result(
        root, original, scalar_result_path
    )
    continuation, direct_path, direct_hash = extract_direct_continuation_result(
        continuation_path, original_report_path
    )

    current = load_arm_result(
        Path(str(current_row["result"])),
        "current_ffrbk_selector",
        expected_sha256=EXPECTED_CURRENT0_RESULT_SHA256,
    )
    scalar_curve = load_arm_result(
        scalar_path,
        "scalar_selector",
        expected_sha256=EXPECTED_SCALAR_EARLY_STALL_RESULT_SHA256,
    )
    direct = load_arm_result(
        direct_path, "direct_lbfgs_only", expected_sha256=direct_hash
    )
    if not bool(scalar_curve["terminated_before_N254"]):
        raise RuntimeError("Registered scalar orphan is not an early termination")
    if scalar_curve["termination_reason"] != "cached_coordinate_stall":
        raise RuntimeError("Registered scalar orphan stop reason drifted")
    if {
        current["initial_path_sha256"],
        scalar_curve["initial_path_sha256"],
        direct["initial_path_sha256"],
    } != {COMMON_INITIAL_ARRAY_SHA256}:
        raise RuntimeError("Current0/A/B do not share the canonical initial path")

    old_current, legacy, official, formal_snapshot = load_frozen_comparators(
        original
    )
    _assert_current0_curve_parity(current, old_current)
    arms = {
        "current_ffrbk_selector": current,
        "scalar_selector": scalar_curve,
        "direct_lbfgs_only": direct,
    }
    curves = {
        arm: value["eligible_points"] for arm, value in arms.items()
    }
    mechanisms = bidirectional_mechanism_comparisons(curves)
    performance = performance_comparisons(
        curves,
        legacy["eligible_points"],
        official["eligible_points"],
    )
    winner = decide_winner(performance, mechanisms)
    return {
        "analysis_id": ANALYSIS_ID,
        "created_unix": time.time(),
        "read_only_formal_input_audit": True,
        "performance_metric": "I_perp_only",
        "physical_objective_arrays_accessed": False,
        "densification_performed": False,
        "unobserved_tail_extrapolated": False,
        "original_incomplete_report": str(original_report_path),
        "original_incomplete_report_sha256": file_sha256(original_report_path),
        "original_preregistration": str(preregistration),
        "original_preregistration_sha256": file_sha256(preregistration),
        "direct_continuation_report": str(continuation_path),
        "direct_continuation_report_sha256": file_sha256(continuation_path),
        "direct_continuation_protocol_id": continuation.get("protocol_id"),
        "frozen_alpha15_ctp_snapshot": str(formal_snapshot),
        "frozen_alpha15_ctp_snapshot_sha256": file_sha256(formal_snapshot),
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
        "early_termination_policy": (
            "precision thresholds not reached by the observed prefix are U "
            "formal losses; the unobserved tail is never imputed"
        ),
        "npz_access_allowlists": {
            arm: sorted(keys) for arm, keys in NPZ_ACCESS_ALLOWLISTS.items()
        },
        "arms": arms,
        "same_alpha_legacy": legacy,
        "official_stage4_v4": official,
        "mechanism_bidirectional_comparisons": mechanisms,
        "performance_candidate_comparisons": performance,
        "performance_qualifiers": winner["qualifiers"],
        **winner,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-incomplete-report", type=Path, required=True)
    parser.add_argument("--direct-continuation-report", type=Path, required=True)
    parser.add_argument("--scalar-result", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    original = args.original_incomplete_report.expanduser().resolve()
    direct = args.direct_continuation_report.expanduser().resolve()
    scalar_result = (
        args.scalar_result.expanduser().resolve()
        if args.scalar_result is not None
        else None
    )
    payload = analyze(original, direct, scalar_result_path=scalar_result)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else original.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"{original.stem}_with_{direct.stem}_iperp_ctp_readonly_v1_"
        f"{time.time_ns()}.json"
    )
    payload["analysis_script"] = str(Path(__file__).resolve())
    payload["analysis_script_sha256"] = file_sha256(Path(__file__).resolve())
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
