#!/usr/bin/env python3
"""Barrier-blind I_perp CTP audit for the Ala residual-ratio ablation.

The analyzer hash-verifies the immutable ratio preregistration/report, the
reused r=0.8 candidate and alpha=12.5 legacy control, the official Stage-IV
v4 comparator, every recorded source/command/result/audit, and the canonical
initial path.  It then merges r={0.90,0.80,0.50} and applies exactly the
preregistered complete-state mesh gate, no-interpolation I_perp
cost-to-precision definition, primary joint-dominance rule, tightest-precision
condition, and multi-winner tie-break.

All NPZ access is through explicit allowlists below.  Energy-barrier fields
are neither indexed nor computed.  The output JSON is created exclusively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

import analyze_stage4_ala_discrete_single_alpha_refinement_iperp_ctp as base


ANALYSIS_ID = "stage4_ala_discrete_single_residual_ratio_iperp_ctp_v1"
RATIO_PROTOCOL_ID = "stage4_ala_discrete_single_residual_ratio_ablation_v1"
FORMAL_REFINEMENT_ANALYSIS_ID = (
    "stage4_ala_discrete_single_alpha_refinement_iperp_ctp_v1"
)
RATIOS = (0.9, 0.8, 0.5)
NEW_RATIOS = (0.9, 0.5)
REFERENCE_RATIO = 0.8
CANONICAL_ALPHA = 12.5
CANONICAL_BUDGET = 254
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
OFFICIAL_DOMAIN_MAX = 0.007
EXPECTED_DISCRETE_IMPLEMENTATION = (
    "ala-frozen-discrete-ffrbk-dsel-m7-evaluated-v1"
)
ROUTING_OPTIONS = frozenset({"--output-dir", "--run-label"})
RATIO_OPTION = "--discrete-residual-ratio"
DSEL_OPTIONS = frozenset(
    {
        "--frozen-subproblem",
        "--discrete-inner-max-steps",
        "--discrete-stop-mode",
        "--discrete-residual-ratio",
        "--discrete-history",
    }
)

COMMON_RESULT_KEYS = frozenset(
    {
        "active_alpha",
        "alpha",
        "alpha_mode",
        "configuration_sha256",
        "i_perp",
        "initial_path_sha256",
        "reparam_committed",
        "reparam_mode",
        "segment_length_cv",
        "segment_length_ratio",
        "state_curve_direct_objective_acceptance",
        "stop_complete_iperp",
        "total_internal_joint_evaluations",
    }
)
CANDIDATE_RESULT_KEYS = frozenset(
    {
        "discrete_all_returned_states_evaluated",
        "discrete_history",
        "discrete_implementation_id",
        "discrete_inner_max_steps",
        "discrete_residual_ratio",
        "discrete_stop_mode",
        "frozen_subproblem",
    }
)
CONTROL_FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "ala_dsel_implementation_id",
        "ala_dsel_audit_selection",
        "ala_dsel_first_activation_nf",
        "discrete_implementation_id",
        "discrete_audit_selection",
    }
)
OFFICIAL_ARCHIVE_KEYS = base.OFFICIAL_ARCHIVE_KEYS
OFFICIAL_COMBINED_KEYS = base.OFFICIAL_COMBINED_KEYS
OFFICIAL_COMPONENT_KEYS = base.OFFICIAL_COMPONENT_KEYS
NPZ_ACCESS_ALLOWLISTS = {
    "candidate": COMMON_RESULT_KEYS | CANDIDATE_RESULT_KEYS,
    "legacy_control": COMMON_RESULT_KEYS,
    "official_archive": OFFICIAL_ARCHIVE_KEYS,
    "official_combined": OFFICIAL_COMBINED_KEYS,
    "official_components": OFFICIAL_COMPONENT_KEYS,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def scalar(saved, key: str):
    return np.asarray(saved[key]).item()


def require_keys(saved, keys, context: str) -> None:
    missing = sorted(set(keys).difference(saved.files))
    if missing:
        raise RuntimeError(f"{context} is missing: {', '.join(missing)}")


def verify_hash_record(record: object, context: str) -> Path:
    if not isinstance(record, dict):
        raise RuntimeError(f"Invalid hash record: {context}")
    path = Path(str(record["path"])).resolve()
    if not path.is_file() or file_sha256(path) != str(record["sha256"]):
        raise RuntimeError(f"Hash mismatch: {context}")
    return path


def verify_hash_manifest(manifest: object, context: str) -> dict[str, str]:
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError(f"Invalid hash manifest: {context}")
    verified = {}
    for name, record in manifest.items():
        path = verify_hash_record(record, f"{context}.{name}")
        verified[str(path)] = file_sha256(path)
    return verified


def command_sha256(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def command_options(command: list[str]) -> dict[str, str | bool]:
    return base.command_options(command)


def assert_candidate_command(command: list[str], ratio: float) -> None:
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        raise RuntimeError("Invalid candidate command")
    options = command_options(command)
    numeric = {
        "--alpha": CANONICAL_ALPHA,
        "--budget": CANONICAL_BUDGET,
        "--history": 7,
        "--discrete-history": 7,
        "--discrete-inner-max-steps": 10,
        "--stop-complete-iperp": -1.0,
        "--reparam-every": 1,
        RATIO_OPTION: float(ratio),
    }
    for flag, expected in numeric.items():
        if float(options[flag]) != float(expected):
            raise RuntimeError(f"Candidate command drift at {flag}")
    exact = {
        "--alpha-mode": "fixed",
        "--initial-path-file": "inputs/common_initial_path_equidistant.npz",
        "--frozen-subproblem": "discrete_b040",
        "--discrete-stop-mode": "r08c10",
        "--reparam-mode": "state_curve",
        "--state-curve-direct-objective-acceptance": "per_image",
    }
    for flag, expected in exact.items():
        if options.get(flag) != expected:
            raise RuntimeError(f"Candidate command drift at {flag}")


def assert_legacy_command(command: list[str]) -> None:
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        raise RuntimeError("Invalid legacy command")
    options = command_options(command)
    numeric = {
        "--alpha": CANONICAL_ALPHA,
        "--budget": CANONICAL_BUDGET,
        "--history": 7,
        "--stop-complete-iperp": -1.0,
        "--reparam-every": 1,
    }
    for flag, expected in numeric.items():
        if float(options[flag]) != float(expected):
            raise RuntimeError(f"Legacy command drift at {flag}")
    if options.get("--alpha-mode") != "fixed":
        raise RuntimeError("Legacy alpha is not fixed")
    if options.get("--initial-path-file") != (
        "inputs/common_initial_path_equidistant.npz"
    ):
        raise RuntimeError("Legacy common path drifted")
    if options.get("--frozen-subproblem") != "legacy":
        raise RuntimeError("Legacy command changed subproblem")
    if options.get("--reparam-mode") != "state_curve":
        raise RuntimeError("Legacy command changed state_curve")
    if options.get("--state-curve-direct-objective-acceptance") != "per_image":
        raise RuntimeError("Legacy command changed per-image acceptance")
    leaked = (DSEL_OPTIONS - {"--frozen-subproblem"}).intersection(options)
    if leaked:
        raise RuntimeError("Legacy command contains discrete options")


def candidate_scientific_signature(command: list[str]) -> dict[str, object]:
    options = command_options(command)
    for flag in ROUTING_OPTIONS | {RATIO_OPTION}:
        options.pop(flag, None)
    return {"entry_point": command[:2], "options": options}


def load_member_result(
    result: Path, role: str, residual_ratio: float | None = None
) -> dict[str, object]:
    with np.load(result, allow_pickle=False) as saved:
        allowlist = NPZ_ACCESS_ALLOWLISTS[role]
        require_keys(saved, allowlist, str(result))
        if float(scalar(saved, "alpha")) != CANONICAL_ALPHA:
            raise RuntimeError("Result alpha mismatch")
        if str(scalar(saved, "alpha_mode")) != "fixed":
            raise RuntimeError("Result alpha mode mismatch")
        active = np.asarray(saved["active_alpha"], dtype=float)
        if not active.size or not np.all(active == CANONICAL_ALPHA):
            raise RuntimeError("Result changed alpha")
        if str(scalar(saved, "reparam_mode")) != "state_curve":
            raise RuntimeError("Result reparameterization mismatch")
        if str(scalar(saved, "state_curve_direct_objective_acceptance")) != (
            "per_image"
        ):
            raise RuntimeError("Result acceptance mismatch")
        if float(scalar(saved, "stop_complete_iperp")) != -1.0:
            raise RuntimeError("Result used Iperp early stop")

        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        mesh_ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        base.validate_curve_arrays(cost, i_perp, cv, mesh_ratio, str(result))
        if cost.size != CANONICAL_BUDGET:
            raise RuntimeError("Result is not N254")
        if committed.shape != cost.shape or active.shape != cost.shape:
            raise RuntimeError("Result completeness/alpha shape mismatch")

        if role == "candidate":
            if residual_ratio is None:
                raise RuntimeError("Candidate residual ratio is required")
            if str(scalar(saved, "discrete_implementation_id")) != (
                EXPECTED_DISCRETE_IMPLEMENTATION
            ):
                raise RuntimeError("Candidate implementation mismatch")
            if str(scalar(saved, "frozen_subproblem")) != "discrete_b040":
                raise RuntimeError("Candidate subproblem mismatch")
            if str(scalar(saved, "discrete_stop_mode")) != "r08c10":
                raise RuntimeError("Candidate stop selector mismatch")
            if int(scalar(saved, "discrete_inner_max_steps")) != 10:
                raise RuntimeError("Candidate cap mismatch")
            if float(scalar(saved, "discrete_residual_ratio")) != float(
                residual_ratio
            ):
                raise RuntimeError("Candidate residual ratio mismatch")
            if int(scalar(saved, "discrete_history")) != 7:
                raise RuntimeError("Candidate history mismatch")
            if not bool(
                scalar(saved, "discrete_all_returned_states_evaluated")
            ):
                raise RuntimeError("Candidate returned unevaluated state")
        elif role == "legacy_control":
            leaked = CONTROL_FORBIDDEN_RESULT_KEYS.intersection(saved.files)
            if leaked:
                raise RuntimeError("Legacy result contains discrete fields")
        else:
            raise RuntimeError(f"Unknown role: {role}")

        complete = committed.copy()
        complete[0] = True
        eligible = complete & (cv <= MESH_CV_MAX) & (
            mesh_ratio <= MESH_RATIO_MAX
        )
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
            "result": str(result),
            "result_sha256": file_sha256(result),
            "configuration_sha256": str(scalar(saved, "configuration_sha256")),
            "initial_path_sha256": str(scalar(saved, "initial_path_sha256")),
            "complete_state_NF": cost[complete].tolist(),
            "mesh_eligible_complete_state_NF": cost[eligible].tolist(),
            "eligible_points": points,
            "pareto_records": base.pareto_records(points),
            "npz_fields_accessed": sorted(allowlist),
        }


def deficit_vector(
    candidate: list[dict[str, object]],
    comparator: list[dict[str, object]],
) -> dict[str, object]:
    thresholds = sorted(
        float(row["i_perp"])
        for row in base.pareto_records(comparator)
    )
    rows = [base.ctp_row(candidate, comparator, threshold) for threshold in thresholds]
    serialized = []
    unreachable = []
    for row in rows:
        if row["candidate_NF"] is None:
            serialized.append(None)
            unreachable.append(True)
        else:
            value = int(row["candidate_minus_comparator_NF"])
            serialized.append(value)
            unreachable.append(False)
    return {
        "thresholds_tight_to_loose": thresholds,
        "candidate_minus_comparator_NF_tight_to_loose": serialized,
        "candidate_unreachable_tight_to_loose": unreachable,
        "threshold_rows_tight_to_loose": rows,
    }


def deficit_ordering_key(vector: dict[str, object]) -> tuple[tuple[int, int], ...]:
    """Return an internal, finite key with unreachable entries ordered last."""
    values = vector["candidate_minus_comparator_NF_tight_to_loose"]
    unreachable = vector["candidate_unreachable_tight_to_loose"]
    if not isinstance(values, list) or not isinstance(unreachable, list):
        raise RuntimeError("Invalid registered deficit vector")
    if len(values) != len(unreachable):
        raise RuntimeError("Deficit vector reachability shape mismatch")
    key = []
    for value, is_unreachable in zip(values, unreachable):
        if bool(is_unreachable):
            if value is not None:
                raise RuntimeError("Unreachable deficit must serialize as null")
            key.append((1, 0))
        else:
            if value is None:
                raise RuntimeError("Reachable deficit cannot serialize as null")
            key.append((0, int(value)))
    return tuple(key)


def decide_primary_winner(
    comparisons: list[dict[str, object]],
) -> dict[str, object]:
    joint = [
        row for row in comparisons if bool(row["joint_full_curve_dominates"])
    ]
    rankings = []
    for row in joint:
        ratio = float(row["residual_ratio"])
        official = deficit_ordering_key(row["official_deficit_vector"])
        legacy = deficit_ordering_key(row["legacy_deficit_vector"])
        key = (official, legacy, abs(math.log(ratio / REFERENCE_RATIO)))
        rankings.append((key, ratio))
    rankings.sort(key=lambda item: item[0])

    selected = None
    unresolved_tie = False
    if len(rankings) == 1:
        selected = rankings[0][1]
    elif len(rankings) > 1:
        unresolved_tie = rankings[0][0] == rankings[1][0]
        if not unresolved_tie:
            selected = rankings[0][1]
    return {
        "joint_full_curve_dominator_ratios": [
            float(row["residual_ratio"]) for row in joint
        ],
        "selected_primary_winner_ratio": selected,
        "multiple_primary_winner_tiebreak_applied": len(rankings) > 1,
        "multiple_primary_winner_exact_tie_unresolved": unresolved_tie,
        "ratio_lock_status": (
            "locked_by_primary_success"
            if selected is not None
            else "not_locked_no_primary_dominator"
            if not rankings
            else "not_locked_unresolved_registered_tiebreak"
        ),
        "winner_claim_made": selected is not None,
        "ratio_lock_allowed": selected is not None,
        "barrier_unembargo_allowed": selected is not None,
        "barrier_unembargoed_by_this_analyzer": False,
    }


def _verify_audit_result(
    audit_path: Path,
    result: Path,
    expected_result_hash: str,
    context: str,
) -> tuple[dict[str, object], list[str]]:
    audit = read_json(audit_path)
    if Path(str(audit["result"])).resolve() != result:
        raise RuntimeError(f"{context} audit result path mismatch")
    if str(audit["result_sha256"]) != expected_result_hash:
        raise RuntimeError(f"{context} audit result hash mismatch")
    if audit.get("barrier_fields_accessed") is not False:
        raise RuntimeError(f"{context} audit reports barrier access")
    command = audit.get("command")
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise RuntimeError(f"{context} audit command is invalid")
    return audit, command


def analyze(ratio_report_path: Path) -> dict[str, object]:
    ratio_report_path = ratio_report_path.resolve()
    report = read_json(ratio_report_path)
    if report.get("status") != "complete":
        raise RuntimeError("Ratio report is not complete")
    if report.get("protocol_id") != RATIO_PROTOCOL_ID:
        raise RuntimeError("Unexpected ratio protocol")
    prereg_path = Path(str(report["preregistration"])).resolve()
    if file_sha256(prereg_path) != str(report["preregistration_sha256"]):
        raise RuntimeError("Ratio preregistration hash mismatch")
    prereg = read_json(prereg_path)
    if prereg.get("status") != "preregistered":
        raise RuntimeError("Ratio preregistration status mismatch")
    if prereg.get("protocol_id") != RATIO_PROTOCOL_ID:
        raise RuntimeError("Ratio preregistration protocol mismatch")
    if Path(str(prereg["planned_report"])).resolve() != ratio_report_path:
        raise RuntimeError("Ratio report differs from preregistered path")
    if tuple(float(x) for x in prereg["display_grid"]) != RATIOS:
        raise RuntimeError("Ratio display grid mismatch")
    if tuple(float(x) for x in prereg["new_residual_ratios"]) != NEW_RATIOS:
        raise RuntimeError("New ratio grid mismatch")
    if float(prereg["fixed_alpha"]) != CANONICAL_ALPHA:
        raise RuntimeError("Fixed alpha mismatch")
    if int(prereg["budget_per_image"]) != CANONICAL_BUDGET:
        raise RuntimeError("Budget mismatch")
    if prereg.get("barrier_fields_accessed_by_wrapper_or_launcher") is not False:
        raise RuntimeError("Ratio campaign broke barrier embargo")
    no_primary = prereg["no_primary_dominator_rule"]
    if not all(
        no_primary[key] is False
        for key in (
            "winner_claim_allowed",
            "ratio_lock_allowed",
            "post_hoc_grid_expansion_allowed",
            "barrier_unembargo_allowed",
        )
    ):
        raise RuntimeError("No-primary preregistration is not fail-closed")

    verified_sources = verify_hash_manifest(
        prereg["run_source_manifest"], "run_source_manifest"
    )
    formal_path = verify_hash_record(
        prereg["formal_refinement_analysis"], "formal_refinement_analysis"
    )
    formal = read_json(formal_path)
    if formal.get("analysis_id") != FORMAL_REFINEMENT_ANALYSIS_ID:
        raise RuntimeError("Unexpected formal refinement analysis")
    if formal.get("barrier_data_accessed") is not False or formal.get(
        "barrier_computation_or_densification_performed"
    ) is not False:
        raise RuntimeError("Formal refinement analysis broke barrier embargo")
    if formal.get("barrier_remains_embargoed") is not True:
        raise RuntimeError("Formal refinement analysis released barrier")
    if formal.get("joint_full_curve_dominator_alphas") != []:
        raise RuntimeError("Ratio ablation was not authorized by no-winner state")
    if float(formal["stop_ablation_center_alpha"]) != CANONICAL_ALPHA:
        raise RuntimeError("Formal analysis selected another alpha")
    if formal.get("stop_ablation_center_is_not_a_winner") is not True:
        raise RuntimeError("Formal analysis mislabeled ablation center")

    common_record = prereg["canonical_common_initial_path"]
    common_path = verify_hash_record(common_record, "canonical_common_initial_path")
    with np.load(common_path, allow_pickle=False) as saved:
        common_initial = np.ascontiguousarray(
            np.asarray(saved["initial_path"], dtype=np.float64)
        )
    common_array_hash = hashlib.sha256(common_initial.tobytes()).hexdigest()
    if common_array_hash != str(common_record["initial_path_array_sha256"]):
        raise RuntimeError("Canonical initial-path array hash mismatch")

    reuse_paths = {
        name: verify_hash_record(record, f"reused_inputs.{name}")
        for name, record in prereg["reused_inputs"].items()
    }
    reference_command_file = json.loads(
        reuse_paths["reference_command"].read_text(encoding="utf-8")
    )
    legacy_command_file = json.loads(
        reuse_paths["legacy_command"].read_text(encoding="utf-8")
    )
    assert_candidate_command(reference_command_file, REFERENCE_RATIO)
    assert_legacy_command(legacy_command_file)

    reference_audit, reference_command = _verify_audit_result(
        reuse_paths["reference_audit"],
        reuse_paths["reference_result"],
        str(prereg["reused_inputs"]["reference_result"]["sha256"]),
        "reused r=0.8",
    )
    legacy_audit, legacy_command = _verify_audit_result(
        reuse_paths["legacy_audit"],
        reuse_paths["legacy_result"],
        str(prereg["reused_inputs"]["legacy_result"]["sha256"]),
        "reused legacy",
    )
    if reference_command != reference_command_file:
        raise RuntimeError("Reused r=0.8 command file/audit mismatch")
    if legacy_command != legacy_command_file:
        raise RuntimeError("Reused legacy command file/audit mismatch")

    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) != len(NEW_RATIOS):
        raise RuntimeError("Ratio report does not contain exactly two new rows")
    new_records = {}
    for row in rows:
        if not isinstance(row, dict) or int(row.get("returncode", -1)) != 0:
            raise RuntimeError("Ratio report contains failed member")
        ratio = float(row["residual_ratio"])
        if ratio not in NEW_RATIOS or ratio in new_records:
            raise RuntimeError("Ratio report contains unexpected/duplicate member")
        result = Path(str(row["result"])).resolve()
        audit_path = Path(str(row["member_audit"])).resolve()
        if file_sha256(result) != str(row["result_sha256"]):
            raise RuntimeError("New ratio result hash mismatch")
        if file_sha256(audit_path) != str(row["member_audit_sha256"]):
            raise RuntimeError("New ratio audit hash mismatch")
        audit, command = _verify_audit_result(
            audit_path, result, str(row["result_sha256"]), f"r={ratio:g}"
        )
        if float(audit["residual_ratio"]) != ratio:
            raise RuntimeError("New ratio audit identity mismatch")
        assert_candidate_command(command, ratio)
        tag = str(row["ratio_tag"])
        expected_command_hash = str(prereg["registered_command_hashes"][tag])
        if command_sha256(command) != expected_command_hash:
            raise RuntimeError("New ratio registered command hash mismatch")
        if str(audit["scientific_command_sha256"]) != expected_command_hash:
            raise RuntimeError("New ratio audit command hash mismatch")
        new_records[ratio] = {
            "result": result,
            "audit": audit,
            "audit_path": audit_path,
            "command": command,
        }
    if set(new_records) != set(NEW_RATIOS):
        raise RuntimeError("Ratio report is missing a new ratio")

    reference_signature = candidate_scientific_signature(reference_command)
    for ratio, record in new_records.items():
        if candidate_scientific_signature(record["command"]) != reference_signature:
            raise RuntimeError(
                f"r={ratio:g} changed a frozen scientific command option"
            )

    legacy = load_member_result(
        reuse_paths["legacy_result"], "legacy_control"
    )
    candidates = {
        REFERENCE_RATIO: load_member_result(
            reuse_paths["reference_result"], "candidate", REFERENCE_RATIO
        )
    }
    for ratio, record in new_records.items():
        candidates[ratio] = load_member_result(
            record["result"], "candidate", ratio
        )
        if list(record["audit"]["complete_state_NF"]) != candidates[ratio][
            "complete_state_NF"
        ]:
            raise RuntimeError("New ratio complete-state audit mismatch")
        if list(record["audit"]["mesh_eligible_complete_state_NF"]) != (
            candidates[ratio]["mesh_eligible_complete_state_NF"]
        ):
            raise RuntimeError("New ratio mesh audit mismatch")
    if list(reference_audit["complete_state_NF"]) != candidates[REFERENCE_RATIO][
        "complete_state_NF"
    ]:
        raise RuntimeError("Reused r=0.8 complete-state audit mismatch")
    if list(legacy_audit["complete_state_NF"]) != legacy["complete_state_NF"]:
        raise RuntimeError("Reused legacy complete-state audit mismatch")

    initial_hashes = {legacy["initial_path_sha256"]} | {
        candidate["initial_path_sha256"] for candidate in candidates.values()
    }
    if initial_hashes != {common_array_hash}:
        raise RuntimeError("Merged members do not use canonical initial path")

    official_record = formal["official_stage4_v4"]
    official_path = Path(str(official_record["archive"])).resolve()
    if file_sha256(official_path) != str(official_record["archive_sha256"]):
        raise RuntimeError("Official archive hash mismatch")
    official = base.load_official_curve(official_path)

    anchors = [float(value) for value in prereg["reporting_anchors"]]
    comparisons = []
    for ratio in RATIOS:
        candidate = candidates[ratio]
        candidate_points = candidate["eligible_points"]
        legacy_comparison = base.compare_curves(
            candidate_points,
            legacy["eligible_points"],
            "same_alpha_legacy_control",
        )
        official_comparison = base.compare_curves(
            candidate_points,
            official["eligible_points"],
            "official_stage4_v4_post_activation_mesh_eligible_tail",
        )
        legacy_vector = deficit_vector(candidate_points, legacy["eligible_points"])
        official_vector = deficit_vector(
            candidate_points, official["eligible_points"]
        )
        comparisons.append(
            {
                "residual_ratio": ratio,
                "source": "reused_reference" if ratio == 0.8 else "new_ablation",
                "candidate": candidate,
                "same_alpha_legacy_comparison": legacy_comparison,
                "official_stage4_v4_comparison": official_comparison,
                "joint_full_curve_dominates": bool(
                    legacy_comparison["full_curve_dominates"]
                    and official_comparison["full_curve_dominates"]
                ),
                "legacy_deficit_vector": legacy_vector,
                "official_deficit_vector": official_vector,
                "reporting_anchors_same_alpha_legacy": base.anchor_rows(
                    candidate_points, legacy["eligible_points"], anchors
                ),
                "reporting_anchors_official_stage4_v4": base.anchor_rows(
                    candidate_points,
                    official["eligible_points"],
                    anchors,
                    domain_max=OFFICIAL_DOMAIN_MAX,
                ),
            }
        )

    decision = decide_primary_winner(comparisons)
    if decision["selected_primary_winner_ratio"] is None:
        if any(
            decision[key]
            for key in (
                "winner_claim_made",
                "ratio_lock_allowed",
                "barrier_unembargo_allowed",
            )
        ):
            raise RuntimeError("No-winner decision is not fail-closed")

    formal_result_hashes = {
        legacy["result"]: legacy["result_sha256"],
        **{
            candidate["result"]: candidate["result_sha256"]
            for candidate in candidates.values()
        },
    }
    formal_audit_hashes = {
        str(reuse_paths["legacy_audit"]): file_sha256(reuse_paths["legacy_audit"]),
        str(reuse_paths["reference_audit"]): file_sha256(
            reuse_paths["reference_audit"]
        ),
        **{
            str(record["audit_path"]): file_sha256(record["audit_path"])
            for record in new_records.values()
        },
    }
    return {
        "analysis_id": ANALYSIS_ID,
        "created_unix": time.time(),
        "read_only_formal_input_audit": True,
        "barrier_data_accessed": False,
        "barrier_computation_or_densification_performed": False,
        "barrier_unembargoed": False,
        "ratio_report": str(ratio_report_path),
        "ratio_report_sha256": file_sha256(ratio_report_path),
        "ratio_preregistration": str(prereg_path),
        "ratio_preregistration_sha256": file_sha256(prereg_path),
        "formal_refinement_analysis": str(formal_path),
        "formal_refinement_analysis_sha256": file_sha256(formal_path),
        "verified_source_hashes": verified_sources,
        "verified_report_prereg_source_command_result_audit_hashes": True,
        "canonical_initial_path": str(common_path),
        "canonical_initial_path_file_sha256": file_sha256(common_path),
        "canonical_initial_path_array_sha256": common_array_hash,
        "merged_residual_ratio_grid": list(RATIOS),
        "fixed_alpha": CANONICAL_ALPHA,
        "fixed_budget_per_image": CANONICAL_BUDGET,
        "command_identity": (
            "new candidates differ from the reused r=0.8 scientific command "
            "only in --discrete-residual-ratio; routing is isolated"
        ),
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
        "official_comparison_scope": (
            "frozen official-v4 complete mesh-eligible post-activation tail "
            "with I_perp<=0.007"
        ),
        "primary_success_criterion": prereg["primary_success_criterion"],
        "multiple_primary_winner_tiebreak": prereg[
            "multiple_primary_winner_tiebreak"
        ],
        "same_alpha_legacy_control": legacy,
        "official_stage4_v4": official,
        "formal_result_hashes": formal_result_hashes,
        "formal_member_audit_hashes": formal_audit_hashes,
        "comparisons": comparisons,
        **decision,
        "no_primary_dominator_rule_applied": (
            decision["selected_primary_winner_ratio"] is None
        ),
        "no_primary_dominator_actions": (
            {
                "winner_claim_allowed": False,
                "ratio_lock_allowed": False,
                "post_hoc_grid_expansion_allowed": False,
                "barrier_unembargo_allowed": False,
            }
            if decision["selected_primary_winner_ratio"] is None
            else None
        ),
        "barrier_remains_embargoed": (
            decision["selected_primary_winner_ratio"] is None
        ),
        "reporting_anchors": anchors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = args.ratio_report.expanduser().resolve()
    if not report.is_file():
        raise FileNotFoundError(report)
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
    payload["analysis_script"] = str(Path(__file__).resolve())
    payload["analysis_script_sha256"] = file_sha256(Path(__file__).resolve())
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
