#!/usr/bin/env python3
"""Read-only I_perp CTP audit for the Ala alpha-only refinement.

The analyzer merges the four immutable refinement pairs with the reused
coarse alpha=10 and alpha=20 pairs.  It validates the complete provenance
chain and applies only the preregistered complete-state, mesh, and I_perp CTP
rules.  Energy-barrier fields are never indexed, copied, or computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


ANALYSIS_ID = "stage4_ala_discrete_single_alpha_refinement_iperp_ctp_v1"
REFINEMENT_PROTOCOL_ID = (
    "stage4_ala_discrete_single_alpha_refinement_matched_v1"
)
COARSE_PROTOCOL_ID = "stage4_ala_discrete_single_fixedalpha_matched_v2"
COARSE_CTP_ANALYSIS_ID = "stage4_ala_discrete_single_iperp_ctp_readonly_v1"
MERGED_ALPHAS = (5.0, 7.5, 10.0, 12.5, 15.0, 20.0)
NEW_ALPHAS = (5.0, 7.5, 12.5, 15.0)
REUSED_ALPHAS = (10.0, 20.0)
MEMBER_ROLES = ("candidate", "matched_control")
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
OFFICIAL_DOMAIN_MAX = 0.007
DSEL_COMMAND_OPTIONS = frozenset(
    {
        "--frozen-subproblem",
        "--discrete-inner-max-steps",
        "--discrete-stop-mode",
        "--discrete-residual-ratio",
        "--discrete-history",
        "--ala-dsel-trigger-iperp",
    }
)
ROUTING_AND_ALPHA_OPTIONS = frozenset(
    {"--output-dir", "--run-label", "--alpha"}
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
EXPECTED_DISCRETE_IMPLEMENTATION = (
    "ala-frozen-discrete-ffrbk-dsel-m7-evaluated-v1"
)
OFFICIAL_ARCHIVE_KEYS = frozenset(
    {
        "activation_iperp",
        "candidate_calls",
        "candidate_i_perp",
        "complete_state_candidates_only",
        "source_result",
        "source_result_sha256",
    }
)
OFFICIAL_COMBINED_KEYS = frozenset(
    {"phase_a_result", "phase_b_result", "phase_transition_force_offset"}
)
OFFICIAL_COMPONENT_KEYS = frozenset(
    {
        "i_perp",
        "segment_length_cv",
        "segment_length_ratio",
        "total_internal_joint_evaluations",
    }
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(saved, key: str):
    return np.asarray(saved[key]).item()


def alpha_key(alpha: float) -> str:
    return f"{float(alpha):g}"


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


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
    options: dict[str, str | bool] = {}
    index = 0
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            index += 1
            continue
        if token in options:
            raise RuntimeError(f"Duplicate command option: {token}")
        if index + 1 == len(command) or command[index + 1].startswith("--"):
            options[token] = True
            index += 1
        else:
            options[token] = command[index + 1]
            index += 2
    return options


def assert_member_command(
    command: list[str], alpha: float, member_role: str
) -> None:
    if not isinstance(command, list) or len(command) < 3:
        raise RuntimeError("Invalid member command")
    options = command_options(command)
    expected = {
        "--alpha": float(alpha),
        "--budget": 254,
        "--history": 7,
        "--stop-complete-iperp": -1.0,
        "--reparam-every": 1,
    }
    for flag, value in expected.items():
        if float(options[flag]) != value:
            raise RuntimeError(f"Command drift at {flag}")
    if options["--alpha-mode"] != "fixed":
        raise RuntimeError("Command alpha is not fixed")
    if options["--initial-path-file"] != "inputs/common_initial_path_equidistant.npz":
        raise RuntimeError("Command changed the common initial path")
    if options["--reparam-mode"] != "state_curve":
        raise RuntimeError("Command changed the reparameterization mode")
    if options["--state-curve-direct-objective-acceptance"] != "per_image":
        raise RuntimeError("Command changed per-image acceptance")
    if member_role == "candidate":
        required = {
            "--frozen-subproblem": "discrete_b040",
            "--discrete-stop-mode": "r08c10",
            "--discrete-inner-max-steps": "10",
            "--discrete-residual-ratio": "0.8",
            "--discrete-history": "7",
        }
        for flag, value in required.items():
            if options.get(flag) != value:
                raise RuntimeError(f"Candidate command drift at {flag}")
    elif member_role == "matched_control":
        if options.get("--frozen-subproblem") != "legacy":
            raise RuntimeError("Control is not legacy")
        leaked = (
            DSEL_COMMAND_OPTIONS - {"--frozen-subproblem"}
        ).intersection(options)
        if leaked:
            raise RuntimeError("Control contains discrete options")
    else:
        raise RuntimeError(f"Unknown member role: {member_role}")


def assert_matched_commands(
    candidate: list[str], control: list[str], alpha: float
) -> None:
    assert_member_command(candidate, alpha, "candidate")
    assert_member_command(control, alpha, "matched_control")
    candidate_options = command_options(candidate)
    control_options = command_options(control)
    for flag in DSEL_COMMAND_OPTIONS | {"--run-label"}:
        candidate_options.pop(flag, None)
        control_options.pop(flag, None)
    if candidate_options != control_options:
        raise RuntimeError(f"Matched-pair command drift at alpha={alpha:g}")


def scientific_command_signature(command: list[str]) -> dict[str, object]:
    options = command_options(command)
    for flag in ROUTING_AND_ALPHA_OPTIONS:
        options.pop(flag, None)
    return {"entry_point": command[:2], "options": options}


def validate_curve_arrays(cost, i_perp, cv, ratio, context: str) -> None:
    arrays = [np.asarray(value) for value in (cost, i_perp, cv, ratio)]
    if len({array.shape for array in arrays}) != 1 or arrays[0].ndim != 1:
        raise RuntimeError(f"{context} has inconsistent arrays")
    if not arrays[0].size or np.any(np.diff(arrays[0]) <= 0):
        raise RuntimeError(f"{context} has invalid optimization NF")
    if not all(np.all(np.isfinite(array)) for array in arrays[1:]):
        raise RuntimeError(f"{context} has non-finite diagnostics")
    if np.any(arrays[1] < 0.0) or np.any(arrays[2] < 0.0) or np.any(arrays[3] < 1.0):
        raise RuntimeError(f"{context} has invalid diagnostics")


def load_member_result(
    result: Path, alpha: float, member_role: str
) -> dict[str, object]:
    with np.load(result, allow_pickle=False) as saved:
        require_keys(saved, COMMON_RESULT_KEYS, str(result))
        if float(scalar(saved, "alpha")) != float(alpha):
            raise RuntimeError("Result alpha mismatch")
        if str(scalar(saved, "alpha_mode")) != "fixed":
            raise RuntimeError("Result alpha mode mismatch")
        active = np.asarray(saved["active_alpha"], dtype=float)
        if not active.size or not np.all(active == float(alpha)):
            raise RuntimeError("Result changed alpha within the trajectory")
        if str(scalar(saved, "reparam_mode")) != "state_curve":
            raise RuntimeError("Result reparameterization mismatch")
        if str(scalar(saved, "state_curve_direct_objective_acceptance")) != "per_image":
            raise RuntimeError("Result acceptance mismatch")
        if float(scalar(saved, "stop_complete_iperp")) != -1.0:
            raise RuntimeError("Result used I_perp early stopping")

        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        validate_curve_arrays(cost, i_perp, cv, ratio, str(result))
        if committed.shape != cost.shape or active.shape != cost.shape:
            raise RuntimeError("Result completeness/alpha shape mismatch")
        if cost.size != 254:
            raise RuntimeError("Result does not contain the N254 trajectory")

        if member_role == "candidate":
            require_keys(saved, CANDIDATE_RESULT_KEYS, str(result))
            if str(scalar(saved, "discrete_implementation_id")) != EXPECTED_DISCRETE_IMPLEMENTATION:
                raise RuntimeError("Candidate implementation mismatch")
            if str(scalar(saved, "frozen_subproblem")) != "discrete_b040":
                raise RuntimeError("Candidate subproblem mismatch")
            if str(scalar(saved, "discrete_stop_mode")) != "r08c10":
                raise RuntimeError("Candidate stop mismatch")
            if int(scalar(saved, "discrete_inner_max_steps")) != 10:
                raise RuntimeError("Candidate evaluated-step cap mismatch")
            if float(scalar(saved, "discrete_residual_ratio")) != 0.8:
                raise RuntimeError("Candidate residual ratio mismatch")
            if int(scalar(saved, "discrete_history")) != 7:
                raise RuntimeError("Candidate history mismatch")
            if not bool(scalar(saved, "discrete_all_returned_states_evaluated")):
                raise RuntimeError("Candidate returned an unevaluated state")
        elif member_role == "matched_control":
            leaked = CONTROL_FORBIDDEN_RESULT_KEYS.intersection(saved.files)
            if leaked:
                raise RuntimeError("Control contains discrete/DSEL outputs")
        else:
            raise RuntimeError(f"Unknown member role: {member_role}")

        complete = committed.copy()
        complete[0] = True
        eligible = complete & (cv <= MESH_CV_MAX) & (ratio <= MESH_RATIO_MAX)
        points = [
            {
                "NF": int(cost[index]),
                "i_perp": float(i_perp[index]),
                "segment_length_cv": float(cv[index]),
                "segment_length_ratio": float(ratio[index]),
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
            "pareto_records": pareto_records(points),
        }


def pareto_records(points: list[dict[str, object]]) -> list[dict[str, object]]:
    records = []
    best = math.inf
    for row in sorted(points, key=lambda item: int(item["NF"])):
        value = float(row["i_perp"])
        if value < best:
            records.append(dict(row))
            best = value
    return records


def cost_to_precision(points: list[dict[str, object]], threshold: float):
    costs = [
        int(row["NF"])
        for row in points
        if float(row["i_perp"]) <= float(threshold)
    ]
    return min(costs) if costs else None


def minimum_precision(points: list[dict[str, object]]):
    return min((float(row["i_perp"]) for row in points), default=None)


def ctp_row(
    candidate: list[dict[str, object]],
    comparator: list[dict[str, object]],
    threshold: float,
) -> dict[str, object]:
    candidate_nf = cost_to_precision(candidate, threshold)
    comparator_nf = cost_to_precision(comparator, threshold)
    if comparator_nf is None:
        outcome = "comparator_unreachable"
    elif candidate_nf is None:
        outcome = "candidate_unreachable_loss"
    elif candidate_nf < comparator_nf:
        outcome = "strict_win"
    elif candidate_nf == comparator_nf:
        outcome = "tie"
    else:
        outcome = "loss"
    return {
        "threshold_i_perp": float(threshold),
        "candidate_NF": candidate_nf,
        "comparator_NF": comparator_nf,
        "candidate_minus_comparator_NF": (
            candidate_nf - comparator_nf
            if candidate_nf is not None and comparator_nf is not None
            else None
        ),
        "outcome": outcome,
    }


def compare_curves(
    candidate: list[dict[str, object]],
    comparator: list[dict[str, object]],
    comparator_name: str,
) -> dict[str, object]:
    comparator_pareto = pareto_records(comparator)
    rows = [
        ctp_row(candidate, comparator, float(point["i_perp"]))
        for point in comparator_pareto
    ]
    losses = [
        row
        for row in rows
        if row["outcome"] in {"loss", "candidate_unreachable_loss"}
    ]
    wins = [row for row in rows if row["outcome"] == "strict_win"]
    candidate_min = minimum_precision(candidate)
    comparator_min = minimum_precision(comparator)
    tightest_ok = bool(
        candidate_min is not None
        and comparator_min is not None
        and candidate_min <= comparator_min
    )
    return {
        "comparator": comparator_name,
        "comparator_pareto_record_count": len(comparator_pareto),
        "threshold_results": rows,
        "weakly_no_slower_at_every_comparator_pareto_threshold": bool(rows)
        and not losses,
        "strictly_faster_at_least_one_threshold": bool(wins),
        "candidate_tightest_eligible_i_perp": candidate_min,
        "comparator_tightest_eligible_i_perp": comparator_min,
        "candidate_does_not_lose_tightest_precision": tightest_ok,
        "full_curve_dominates": bool(rows and not losses and wins and tightest_ok),
        "exact_deficit": {
            "loss_threshold_count": len(losses),
            "candidate_unreachable_loss_count": sum(
                row["outcome"] == "candidate_unreachable_loss" for row in losses
            ),
            "strict_win_count": len(wins),
            "losses": losses,
        },
    }


def anchor_rows(
    candidate: list[dict[str, object]],
    comparator: list[dict[str, object]],
    anchors: list[float],
    domain_max: float | None = None,
) -> list[dict[str, object]]:
    rows = []
    for threshold in anchors:
        if domain_max is not None and threshold > domain_max:
            rows.append(
                {
                    "threshold_i_perp": threshold,
                    "domain_status": "outside_official_post_activation_domain",
                    "candidate_NF": cost_to_precision(candidate, threshold),
                    "comparator_NF": None,
                    "outcome": "not_compared",
                }
            )
        else:
            row = ctp_row(candidate, comparator, threshold)
            row["domain_status"] = "inside_comparator_domain"
            rows.append(row)
    return rows


def _component_mesh_map(path: Path, offset: int) -> dict[int, dict[str, float]]:
    with np.load(path, allow_pickle=False) as saved:
        require_keys(saved, OFFICIAL_COMPONENT_KEYS, str(path))
        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        validate_curve_arrays(cost, i_perp, cv, ratio, str(path))
        return {
            int(offset + cost[index]): {
                "i_perp": float(i_perp[index]),
                "segment_length_cv": float(cv[index]),
                "segment_length_ratio": float(ratio[index]),
            }
            for index in range(cost.size)
        }


def load_official_curve(archive: Path) -> dict[str, object]:
    with np.load(archive, allow_pickle=False) as saved:
        require_keys(saved, OFFICIAL_ARCHIVE_KEYS, str(archive))
        if not bool(scalar(saved, "complete_state_candidates_only")):
            raise RuntimeError("Official archive is not complete-state-only")
        activation = float(scalar(saved, "activation_iperp"))
        if activation != OFFICIAL_DOMAIN_MAX:
            raise RuntimeError("Official activation domain changed")
        source = Path(str(scalar(saved, "source_result"))).resolve()
        if file_sha256(source) != str(scalar(saved, "source_result_sha256")):
            raise RuntimeError("Official source result hash mismatch")
        calls = np.asarray(saved["candidate_calls"], dtype=int)
        values = np.asarray(saved["candidate_i_perp"], dtype=float)
        if calls.shape != values.shape or not calls.size:
            raise RuntimeError("Official candidate arrays are invalid")
    with np.load(source, allow_pickle=False) as combined:
        require_keys(combined, OFFICIAL_COMBINED_KEYS, str(source))
        first = Path(str(scalar(combined, "phase_a_result"))).resolve()
        second = Path(str(scalar(combined, "phase_b_result"))).resolve()
        offset = int(scalar(combined, "phase_transition_force_offset"))
    mesh = _component_mesh_map(first, 0)
    second_mesh = _component_mesh_map(second, offset)
    if set(mesh).intersection(second_mesh):
        raise RuntimeError("Official phase cost maps overlap")
    mesh.update(second_mesh)
    eligible = []
    rejected = []
    for nf, value in zip(calls, values):
        nf = int(nf)
        if nf not in mesh or float(mesh[nf]["i_perp"]) != float(value):
            raise RuntimeError("Official archive/source mesh mismatch")
        row = {"NF": nf, "i_perp": float(value), **mesh[nf]}
        if row["segment_length_cv"] <= MESH_CV_MAX and row["segment_length_ratio"] <= MESH_RATIO_MAX:
            eligible.append(row)
        else:
            rejected.append(row)
    return {
        "archive": str(archive),
        "archive_sha256": file_sha256(archive),
        "source_result": str(source),
        "source_result_sha256": file_sha256(source),
        "component_results": [str(first), str(second)],
        "component_result_sha256": [file_sha256(first), file_sha256(second)],
        "domain_max_i_perp": activation,
        "eligible_points": eligible,
        "mesh_rejected_points": rejected,
        "pareto_records": pareto_records(eligible),
    }


def official_tail_deficit(
    alpha: float,
    candidate: list[dict[str, object]],
    official: list[dict[str, object]],
) -> tuple[dict[str, object], tuple[object, ...]]:
    thresholds = sorted(
        (float(row["i_perp"]) for row in pareto_records(official))
    )
    rows = [ctp_row(candidate, official, threshold) for threshold in thresholds]
    unreachable = sum(row["candidate_NF"] is None for row in rows)
    tightest = minimum_precision(candidate)
    if tightest is None:
        raise RuntimeError("Candidate has no eligible point")
    serialized_vector = [
        row["candidate_minus_comparator_NF"]
        if row["candidate_NF"] is not None
        else None
        for row in rows
    ]
    comparison_vector = tuple(
        math.inf if value is None else int(value) for value in serialized_vector
    )
    audit = {
        "alpha": alpha,
        "unreachable_official_tail_threshold_count": unreachable,
        "tightest_eligible_complete_mesh_i_perp": tightest,
        "official_thresholds_tight_to_loose": thresholds,
        "candidate_minus_official_NF_tight_to_loose": serialized_vector,
        "threshold_rows_tight_to_loose": rows,
    }
    key = (unreachable, tightest, comparison_vector, alpha)
    return audit, key


def _audit_all_report_rows(
    report: dict[str, object], context: str
) -> dict[tuple[float, str], dict[str, object]]:
    rows = report.get("results")
    if not isinstance(rows, list):
        raise RuntimeError(f"{context} has no result rows")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or int(row.get("returncode", -1)) != 0:
            raise RuntimeError(f"{context} contains a failed result")
        result = Path(str(row["result"])).resolve()
        audit_path = Path(str(row["member_audit"])).resolve()
        if file_sha256(result) != str(row["result_sha256"]):
            raise RuntimeError(f"{context} result hash mismatch")
        if file_sha256(audit_path) != str(row["member_audit_sha256"]):
            raise RuntimeError(f"{context} audit hash mismatch")
        audit = read_json(audit_path)
        alpha = float(row["alpha"])
        role = str(row["member_role"])
        if float(audit["alpha"]) != alpha or audit["member_role"] != role:
            raise RuntimeError(f"{context} audit identity mismatch")
        if Path(str(audit["result"])).resolve() != result:
            raise RuntimeError(f"{context} audit result path mismatch")
        if str(audit["result_sha256"]) != file_sha256(result):
            raise RuntimeError(f"{context} audit result hash mismatch")
        command = audit.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise RuntimeError(f"{context} audit command is invalid")
        assert_member_command(command, alpha, role)
        key = (alpha, role)
        if key in indexed:
            raise RuntimeError(f"{context} contains duplicate members")
        indexed[key] = {
            "row": row,
            "audit": audit,
            "audit_path": audit_path,
            "command": command,
        }
    return indexed


def _load_selected_member(
    record: dict[str, object], alpha: float, role: str, refinement_member: bool
) -> dict[str, object]:
    row = record["row"]
    audit = record["audit"]
    result = Path(str(row["result"])).resolve()
    loaded = load_member_result(result, alpha, role)
    if refinement_member:
        if audit.get("barrier_fields_accessed") is not False:
            raise RuntimeError("Refinement audit reports barrier access")
        if audit.get("barrier_computation_or_densification_performed_by_wrapper") is not False:
            raise RuntimeError("Refinement audit reports barrier computation")
        if list(audit["complete_state_NF"]) != loaded["complete_state_NF"]:
            raise RuntimeError("Refinement complete-state audit mismatch")
        if list(audit["mesh_eligible_complete_state_NF"]) != loaded[
            "mesh_eligible_complete_state_NF"
        ]:
            raise RuntimeError("Refinement mesh-eligible audit mismatch")
    else:
        if list(audit["complete_candidate_state_NF"]) != loaded["complete_state_NF"]:
            raise RuntimeError("Coarse complete-state audit mismatch")
    return {
        **loaded,
        "member_audit": str(record["audit_path"]),
        "member_audit_sha256": file_sha256(record["audit_path"]),
        "command_sha256": command_sha256(record["command"]),
    }


def analyze(refinement_report_path: Path) -> dict[str, object]:
    refinement_report_path = refinement_report_path.resolve()
    refinement_report = read_json(refinement_report_path)
    if refinement_report.get("status") != "complete":
        raise RuntimeError("Refinement report is not complete")
    if refinement_report.get("protocol_id") != REFINEMENT_PROTOCOL_ID:
        raise RuntimeError("Unexpected refinement protocol")
    prereg_path = Path(str(refinement_report["preregistration"])).resolve()
    if file_sha256(prereg_path) != str(refinement_report["preregistration_sha256"]):
        raise RuntimeError("Refinement preregistration hash mismatch")
    prereg = read_json(prereg_path)
    if prereg.get("status") != "preregistered" or prereg.get("protocol_id") != REFINEMENT_PROTOCOL_ID:
        raise RuntimeError("Invalid refinement preregistration")
    if tuple(float(value) for value in prereg["new_alpha_grid"]) != NEW_ALPHAS:
        raise RuntimeError("Refinement alpha grid mismatch")
    if Path(str(prereg["planned_report"])).resolve() != refinement_report_path:
        raise RuntimeError("Refinement report differs from its preregistered path")
    if prereg.get("alpha_lock_requires_primary_success") is not True:
        raise RuntimeError("Preregistration does not require primary success")
    no_primary = prereg["no_primary_dominator_rule"]
    if no_primary["winner_claim_allowed"] is not False or no_primary["barrier_unembargo_allowed"] is not False:
        raise RuntimeError("Invalid no-dominator preregistration")

    source_hashes = {}
    for name in (
        "run_source_manifest",
        "decision_source_manifest",
    ):
        source_hashes.update(verify_hash_manifest(prereg[name], name))
    source_hashes.update(
        verify_hash_manifest(
            prereg["coarse_decision_findings"][
                "frozen_coarse_source_manifest_verified"
            ],
            "frozen_coarse_sources",
        )
    )
    decision_inputs = prereg["decision_inputs"]
    verified_inputs = {
        name: verify_hash_record(record, f"decision_inputs.{name}")
        for name, record in decision_inputs.items()
    }
    coarse_report_path = verified_inputs["coarse_complete_report"]
    coarse_prereg_path = verified_inputs["coarse_preregistration"]
    coarse_ctp_path = verified_inputs["coarse_readonly_iperp_ctp_report"]
    official_path = verified_inputs["official_stage4_v4_comparator"]
    coarse_report = read_json(coarse_report_path)
    coarse_prereg = read_json(coarse_prereg_path)
    coarse_ctp = read_json(coarse_ctp_path)
    if coarse_report.get("status") != "complete" or coarse_report.get("protocol_id") != COARSE_PROTOCOL_ID:
        raise RuntimeError("Invalid coarse report")
    if coarse_prereg.get("status") != "preregistered" or coarse_prereg.get("protocol_id") != COARSE_PROTOCOL_ID:
        raise RuntimeError("Invalid coarse preregistration")
    if Path(str(coarse_report["preregistration"])).resolve() != coarse_prereg_path:
        raise RuntimeError("Coarse report/preregistration path mismatch")
    if str(coarse_report["preregistration_sha256"]) != file_sha256(coarse_prereg_path):
        raise RuntimeError("Coarse preregistration hash mismatch")
    if coarse_ctp.get("analysis_id") != COARSE_CTP_ANALYSIS_ID:
        raise RuntimeError("Invalid coarse CTP input")
    if coarse_ctp.get("barrier_data_accessed") is not False or coarse_ctp.get(
        "barrier_computation_or_densification_performed"
    ) is not False:
        raise RuntimeError("Coarse CTP input broke the barrier embargo")
    if Path(str(coarse_ctp["formal_grid_report"])).resolve() != coarse_report_path:
        raise RuntimeError("Coarse CTP report path mismatch")
    if str(coarse_ctp["formal_grid_report_sha256"]) != file_sha256(coarse_report_path):
        raise RuntimeError("Coarse CTP report hash mismatch")

    refinement_index = _audit_all_report_rows(refinement_report, "refinement")
    coarse_index = _audit_all_report_rows(coarse_report, "coarse")
    if set(refinement_index) != {
        (alpha, role) for alpha in NEW_ALPHAS for role in MEMBER_ROLES
    }:
        raise RuntimeError("Refinement report does not contain the exact 8 cells")
    for alpha in NEW_ALPHAS:
        assert_matched_commands(
            refinement_index[(alpha, "candidate")]["command"],
            refinement_index[(alpha, "matched_control")]["command"],
            alpha,
        )
        registered = prereg["registered_command_hashes"][f"alpha{alpha:g}"]
        for role in MEMBER_ROLES:
            expected_key = f"{role}_command_sha256"
            actual = command_sha256(refinement_index[(alpha, role)]["command"])
            if actual != registered[expected_key]:
                raise RuntimeError("Refinement command hash mismatch")
    for alpha in REUSED_ALPHAS:
        assert_matched_commands(
            coarse_index[(alpha, "candidate")]["command"],
            coarse_index[(alpha, "matched_control")]["command"],
            alpha,
        )

    reference_signatures = {
        role: scientific_command_signature(coarse_index[(10.0, role)]["command"])
        for role in MEMBER_ROLES
    }
    for alpha in MERGED_ALPHAS:
        source = refinement_index if alpha in NEW_ALPHAS else coarse_index
        for role in MEMBER_ROLES:
            if scientific_command_signature(source[(alpha, role)]["command"]) != reference_signatures[role]:
                raise RuntimeError("Scientific command identity drift across merged alpha grid")

    reference_records = prereg["coarse_decision_findings"][
        "reference_cells_reused_not_rerun"
    ]
    pairs = []
    merged: dict[float, dict[str, dict[str, object]]] = {}
    for alpha in MERGED_ALPHAS:
        source = refinement_index if alpha in NEW_ALPHAS else coarse_index
        refinement_member = alpha in NEW_ALPHAS
        members = {
            role: _load_selected_member(
                source[(alpha, role)], alpha, role, refinement_member
            )
            for role in MEMBER_ROLES
        }
        if alpha in REUSED_ALPHAS:
            recorded = reference_records[f"alpha{alpha:g}"]
            for role in MEMBER_ROLES:
                if members[role]["result_sha256"] != recorded[role]["result_sha256"]:
                    raise RuntimeError("Reused coarse result differs from preregistration")
                if members[role]["member_audit_sha256"] != recorded[role]["member_audit_sha256"]:
                    raise RuntimeError("Reused coarse audit differs from preregistration")
        merged[alpha] = members

    official = load_official_curve(official_path)
    anchors = [float(value) for value in prereg["reporting_anchors"]]
    if official["archive_sha256"] != decision_inputs[
        "official_stage4_v4_comparator"
    ]["sha256"]:
        raise RuntimeError("Official archive hash mismatch")
    for alpha in MERGED_ALPHAS:
        candidate = merged[alpha]["candidate"]["eligible_points"]
        control = merged[alpha]["matched_control"]["eligible_points"]
        same = compare_curves(candidate, control, "same_alpha_legacy_control")
        official_comparison = compare_curves(
            candidate,
            official["eligible_points"],
            "official_stage4_v4_post_activation_mesh_eligible_tail",
        )
        pairs.append(
            {
                "alpha": alpha,
                "source": "refinement" if alpha in NEW_ALPHAS else "reused_coarse",
                "candidate": merged[alpha]["candidate"],
                "matched_control": merged[alpha]["matched_control"],
                "same_alpha_control_comparison": same,
                "official_stage4_v4_comparison": official_comparison,
                "joint_full_curve_dominates": bool(
                    same["full_curve_dominates"]
                    and official_comparison["full_curve_dominates"]
                ),
                "reporting_anchors_same_alpha_control": anchor_rows(
                    candidate, control, anchors
                ),
                "reporting_anchors_official_stage4_v4": anchor_rows(
                    candidate,
                    official["eligible_points"],
                    anchors,
                    domain_max=OFFICIAL_DOMAIN_MAX,
                ),
            }
        )

    joint = [pair["alpha"] for pair in pairs if pair["joint_full_curve_dominates"]]
    tail_audits = []
    tail_keys = {}
    for pair in pairs:
        audit, key = official_tail_deficit(
            float(pair["alpha"]),
            pair["candidate"]["eligible_points"],
            official["eligible_points"],
        )
        tail_audits.append(audit)
        tail_keys[float(pair["alpha"])] = key
    tail_audits.sort(key=lambda row: tail_keys[float(row["alpha"])])
    for rank, row in enumerate(tail_audits, start=1):
        row["official_tail_deficit_rank"] = rank

    if joint:
        stop_center = None
        no_dominator_applied = False
        alpha_lock_status = (
            "one_primary_dominator" if len(joint) == 1 else "multiple_primary_dominators"
        )
    else:
        stop_center = float(tail_audits[0]["alpha"])
        no_dominator_applied = True
        alpha_lock_status = "not_locked_no_joint_dominator"

    formal_result_hashes = {
        member["result"]: member["result_sha256"]
        for members in merged.values()
        for member in members.values()
    }
    formal_audit_hashes = {
        member["member_audit"]: member["member_audit_sha256"]
        for members in merged.values()
        for member in members.values()
    }
    return {
        "analysis_id": ANALYSIS_ID,
        "created_unix": time.time(),
        "read_only_formal_input_audit": True,
        "barrier_data_accessed": False,
        "barrier_computation_or_densification_performed": False,
        "barrier_unembargoed": False,
        "refinement_report": str(refinement_report_path),
        "refinement_report_sha256": file_sha256(refinement_report_path),
        "refinement_preregistration": str(prereg_path),
        "refinement_preregistration_sha256": file_sha256(prereg_path),
        "coarse_report": str(coarse_report_path),
        "coarse_report_sha256": file_sha256(coarse_report_path),
        "coarse_preregistration": str(coarse_prereg_path),
        "coarse_preregistration_sha256": file_sha256(coarse_prereg_path),
        "coarse_ctp_report": str(coarse_ctp_path),
        "coarse_ctp_report_sha256": file_sha256(coarse_ctp_path),
        "verified_source_hashes": source_hashes,
        "verified_all_refinement_and_coarse_report_result_audit_hashes": True,
        "verified_input_report_member_counts": {
            "refinement": len(refinement_index),
            "coarse": len(coarse_index),
            "merged_for_analysis": len(formal_result_hashes),
        },
        "merged_alpha_grid": list(MERGED_ALPHAS),
        "command_identity": (
            "all merged commands satisfy fixed N254 state_curve/per_image policy; "
            "new command hashes match preregistration; scientific signatures match "
            "the frozen coarse alpha10 template; every candidate/control pair is exact"
        ),
        "eligibility": (
            "initial or reparam_committed, then segment_length_cv<=0.0065 "
            "and segment_length_ratio<=1.03"
        ),
        "cost_to_precision": (
            "minimum actual optimization NF with I_perp<=threshold; no interpolation"
        ),
        "official_comparison_scope": (
            "literal frozen official-v4 complete mesh-eligible post-activation "
            "tail with I_perp<=0.007"
        ),
        "primary_success_criterion": (
            "simultaneous full-curve dominance over same-alpha control and official-v4"
        ),
        "official_stage4_v4": official,
        "formal_result_hashes": formal_result_hashes,
        "formal_member_audit_hashes": formal_audit_hashes,
        "joint_full_curve_dominator_alphas": joint,
        "alpha_lock_status": alpha_lock_status,
        "winner_claim_made": False,
        "no_primary_dominator_rule_applied": no_dominator_applied,
        "stop_ablation_center_alpha": stop_center,
        "stop_ablation_center_is_not_a_winner": bool(stop_center is not None),
        "barrier_remains_embargoed": True,
        "official_tail_deficit_order": (
            "unreachable official thresholds count; tightest eligible I_perp; "
            "candidate-minus-official NF vector tight-to-loose; numeric alpha tie-break"
        ),
        "official_tail_deficit_ranking": tail_audits,
        "reporting_anchors": anchors,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refinement-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = args.refinement_report.expanduser().resolve()
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
