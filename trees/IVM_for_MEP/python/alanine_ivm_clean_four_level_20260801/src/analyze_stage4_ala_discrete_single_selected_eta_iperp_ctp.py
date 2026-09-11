#!/usr/bin/env python3
"""Barrier-blind I_perp CTP audit for the Ala selected-step eta pilot.

The only formal input is the completed eta-grid report.  This analyzer
hash-verifies its preregistration, member results and audits, registered
commands, run-source manifest, canonical initial path, same-alpha legacy
control, eta=1 identity oracle, and the frozen official Stage-IV v4 chain.

Every NPZ access uses an explicit allowlist.  No energy-barrier field is
indexed or computed.  Candidate costs are actual optimization NF and are
never interpolated.  Only initial or reparam-committed, mesh-eligible states
enter the I_perp cost-to-precision curves.
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
import analyze_stage4_ala_discrete_single_residual_ratio_iperp_ctp as ratio_base


ANALYSIS_ID = "stage4_ala_discrete_single_selected_eta_iperp_ctp_v1"
PROTOCOL_ID = "stage4_ala_discrete_single_selected_eta_pilot_v1"
ETA_GRID = (1.0, 1.25, 1.5, 2.0)
CANDIDATE_ETAS = (1.25, 1.5, 2.0)
ETA_TAGS = {1.0: "eta1", 1.25: "eta1p25", 1.5: "eta1p5", 2.0: "eta2"}
ETA_TEXT = {1.0: "1", 1.25: "1.25", 1.5: "1.5", 2.0: "2"}
ETA_FLAG = "--discrete-step-multiplier-eta"
ETA_POLICY_ID = "post-select-selected-step-eta-recap-v1"
ETA_APPLICATION_STAGE = "after-original-select-before-trial"
BASE_DISCRETE_IMPLEMENTATION = (
    "ala-frozen-discrete-ffrbk-dsel-m7-evaluated-v1"
)
CANONICAL_ALPHA = 12.5
CANONICAL_RATIO = 0.8
CANONICAL_INNER_CAP = 10
CANONICAL_HISTORY = 7
CANONICAL_BUDGET = 254
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
OFFICIAL_DOMAIN_MAX = 0.007
REPORTING_ANCHORS = (
    1.0,
    0.1,
    0.05,
    0.02,
    0.01,
    0.005,
    0.002,
    0.001,
    5.0e-4,
    2.0e-4,
    1.5e-4,
)
EXPECTED_SOURCE_MANIFEST_KEYS = frozenset(
    {
        "eta_grid_launcher",
        "eta_member_wrapper",
        "eta_runtime_shim",
        "eta_tests",
        "frozen_discrete_b040",
        "production_runner",
        "frozen_alpha_refinement_wrapper",
        "frozen_single_wrapper",
        "legacy_command_helper",
        "ala_core",
    }
)
EXPECTED_DECISION_INPUT_KEYS = frozenset(
    {
        "old_eta_one_identity_oracle_result",
        "old_eta_one_identity_oracle_audit",
        "old_eta_one_identity_oracle_command",
        "same_alpha_legacy_result",
        "same_alpha_legacy_audit",
        "same_alpha_legacy_command",
        "official_stage4_v4",
    }
)

# The archive hashes its combined source, but that combined file stores only
# component paths.  Pin the already-formal component content as well so a
# component replacement cannot pass through the official loader unnoticed.
EXPECTED_OFFICIAL_SOURCE_SHA256 = (
    "f5be61d5121a46d349247d036ba2ffe9d258efe5caf864ee659ef2bb18aa9622"
)
EXPECTED_OFFICIAL_COMPONENT_SHA256 = (
    "9d9a5f692a5a65c82b0921d385dd9a36b8477b9fa9efa1df82c629f4fd8d9de5",
    "840b97a508242c3c187e390ed3bab34913fd50f9827dea26b1ab8963add10918",
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
        "subproblem_line_search_backtracks",
        "total_internal_joint_evaluations",
    }
)
DISCRETE_RESULT_KEYS = frozenset(
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
ETA_RESULT_KEYS = frozenset(
    {
        "discrete_step_multiplier_policy_id",
        "discrete_step_multiplier_base_implementation_id",
        "discrete_step_multiplier_implementation_id",
        "discrete_step_multiplier_eta",
        "discrete_step_multiplier_application_stage",
        "discrete_step_multiplier_selector_isolated",
        "discrete_step_multiplier_dynamic_cap_reapplied",
        "discrete_step_multiplier_energy_backtracking",
        "discrete_step_multiplier_extra_physical_evaluations",
        "discrete_step_multiplier_aborted_selection_count",
    }
)
ETA_MEMBER_ALLOWLIST = COMMON_RESULT_KEYS | DISCRETE_RESULT_KEYS | ETA_RESULT_KEYS
OFFICIAL_COMBINED_ALLOWLIST = base.OFFICIAL_COMBINED_KEYS | frozenset(
    {"phase_a_result_sha256", "phase_b_result_sha256"}
)
NPZ_ACCESS_ALLOWLISTS = {
    "eta_member": ETA_MEMBER_ALLOWLIST,
    "legacy_control": ratio_base.NPZ_ACCESS_ALLOWLISTS["legacy_control"],
    "identity_oracle": ratio_base.NPZ_ACCESS_ALLOWLISTS["candidate"],
    "canonical_initial": frozenset({"initial_path"}),
    "official_archive": base.OFFICIAL_ARCHIVE_KEYS,
    "official_combined": OFFICIAL_COMBINED_ALLOWLIST,
    "official_components": base.OFFICIAL_COMPONENT_KEYS,
}
CTP_PARITY_KEYS = (
    "active_alpha",
    "i_perp",
    "reparam_committed",
    "segment_length_cv",
    "segment_length_ratio",
    "total_internal_joint_evaluations",
)


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


def read_command_json(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise RuntimeError(f"Expected a JSON command vector: {path}")
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
    if not {"path", "sha256"}.issubset(record):
        raise RuntimeError(f"Incomplete hash record: {context}")
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


def eta_implementation_id(eta: float) -> str:
    token = float(eta).hex().replace(".", "d").replace("+", "p").replace("-", "m")
    return f"{BASE_DISCRETE_IMPLEMENTATION}+{ETA_POLICY_ID}+eta-{token}"


def _replace_option(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"Expected one valued {flag} option")
    command[positions[0] + 1] = value


def assert_eta_command(
    command: list[str],
    canonical_command: list[str],
    eta: float,
    tag: str,
    shim_path: Path,
) -> None:
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise RuntimeError("Invalid eta command")
    if eta not in ETA_GRID or tag != ETA_TAGS[eta]:
        raise RuntimeError("Eta command identity mismatch")
    expected = list(canonical_command)
    expected[1] = str(shim_path.resolve())
    _replace_option(
        expected,
        "--output-dir",
        f"runs/{PROTOCOL_ID}/members",
    )
    _replace_option(
        expected,
        "--run-label",
        f"{PROTOCOL_ID}_{tag}_r08c10_a12p5_N254_discrete_b040_r1",
    )
    expected.extend([ETA_FLAG, ETA_TEXT[eta]])
    if command != expected:
        raise RuntimeError(f"eta={eta:g} command drifted from its exact clone")

    options = command_options(command)
    numeric = {
        "--alpha": CANONICAL_ALPHA,
        "--budget": CANONICAL_BUDGET,
        "--history": CANONICAL_HISTORY,
        "--discrete-history": CANONICAL_HISTORY,
        "--discrete-inner-max-steps": CANONICAL_INNER_CAP,
        "--discrete-residual-ratio": CANONICAL_RATIO,
        "--stop-complete-iperp": -1.0,
        "--reparam-every": 1,
        ETA_FLAG: eta,
    }
    for flag, expected_value in numeric.items():
        if float(options[flag]) != float(expected_value):
            raise RuntimeError(f"Eta command drift at {flag}")
    exact = {
        "--alpha-mode": "fixed",
        "--initial-path-file": "inputs/common_initial_path_equidistant.npz",
        "--frozen-subproblem": "discrete_b040",
        "--discrete-stop-mode": "r08c10",
        "--reparam-mode": "state_curve",
        "--state-curve-direct-objective-acceptance": "per_image",
    }
    for flag, expected_value in exact.items():
        if options.get(flag) != expected_value:
            raise RuntimeError(f"Eta command drift at {flag}")


def _hash_array_sequence(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for raw in arrays:
        array = np.ascontiguousarray(np.asarray(raw))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def ctp_trajectory_sha256(result: Path) -> str:
    with np.load(result, allow_pickle=False) as saved:
        require_keys(saved, CTP_PARITY_KEYS, str(result))
        return _hash_array_sequence(
            [np.asarray(saved[key]) for key in CTP_PARITY_KEYS]
        )


def build_eligible_points(
    cost,
    i_perp,
    committed,
    cv,
    mesh_ratio,
) -> dict[str, object]:
    cost = np.asarray(cost, dtype=int)
    i_perp = np.asarray(i_perp, dtype=float)
    committed = np.asarray(committed, dtype=bool)
    cv = np.asarray(cv, dtype=float)
    mesh_ratio = np.asarray(mesh_ratio, dtype=float)
    base.validate_curve_arrays(cost, i_perp, cv, mesh_ratio, "eta trajectory")
    if committed.shape != cost.shape:
        raise RuntimeError("Eta completeness shape mismatch")
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
        "complete_state_NF": cost[complete].tolist(),
        "mesh_eligible_complete_state_NF": cost[eligible].tolist(),
        "eligible_points": points,
        "pareto_records": base.pareto_records(points),
    }


def load_eta_result(result: Path, eta: float) -> dict[str, object]:
    with np.load(result, allow_pickle=False) as saved:
        require_keys(saved, ETA_MEMBER_ALLOWLIST, str(result))
        if float(scalar(saved, "alpha")) != CANONICAL_ALPHA:
            raise RuntimeError("Eta result alpha mismatch")
        if str(scalar(saved, "alpha_mode")) != "fixed":
            raise RuntimeError("Eta result alpha mode mismatch")
        active = np.asarray(saved["active_alpha"], dtype=float)
        if not active.size or not np.all(active == CANONICAL_ALPHA):
            raise RuntimeError("Eta result changed alpha")
        if str(scalar(saved, "reparam_mode")) != "state_curve":
            raise RuntimeError("Eta result reparameterization mismatch")
        if str(scalar(saved, "state_curve_direct_objective_acceptance")) != (
            "per_image"
        ):
            raise RuntimeError("Eta result acceptance mismatch")
        if float(scalar(saved, "stop_complete_iperp")) != -1.0:
            raise RuntimeError("Eta result used Iperp early stop")
        if str(scalar(saved, "frozen_subproblem")) != "discrete_b040":
            raise RuntimeError("Eta result subproblem mismatch")
        if str(scalar(saved, "discrete_stop_mode")) != "r08c10":
            raise RuntimeError("Eta result stop selector mismatch")
        if int(scalar(saved, "discrete_inner_max_steps")) != CANONICAL_INNER_CAP:
            raise RuntimeError("Eta result inner cap mismatch")
        if float(scalar(saved, "discrete_residual_ratio")) != CANONICAL_RATIO:
            raise RuntimeError("Eta result residual ratio mismatch")
        if int(scalar(saved, "discrete_history")) != CANONICAL_HISTORY:
            raise RuntimeError("Eta result history mismatch")
        if not bool(scalar(saved, "discrete_all_returned_states_evaluated")):
            raise RuntimeError("Eta result returned unevaluated state")

        expected_id = eta_implementation_id(eta)
        if str(scalar(saved, "discrete_implementation_id")) != expected_id:
            raise RuntimeError("Eta result implementation mismatch")
        if str(scalar(saved, "discrete_step_multiplier_policy_id")) != (
            ETA_POLICY_ID
        ):
            raise RuntimeError("Eta result policy mismatch")
        if str(
            scalar(saved, "discrete_step_multiplier_base_implementation_id")
        ) != BASE_DISCRETE_IMPLEMENTATION:
            raise RuntimeError("Eta result base implementation mismatch")
        if str(
            scalar(saved, "discrete_step_multiplier_implementation_id")
        ) != expected_id:
            raise RuntimeError("Eta metadata implementation mismatch")
        if float(scalar(saved, "discrete_step_multiplier_eta")) != float(eta):
            raise RuntimeError("Eta result multiplier mismatch")
        if str(
            scalar(saved, "discrete_step_multiplier_application_stage")
        ) != ETA_APPLICATION_STAGE:
            raise RuntimeError("Eta application stage mismatch")
        if not bool(
            scalar(saved, "discrete_step_multiplier_selector_isolated")
        ):
            raise RuntimeError("Eta contaminated the selector")
        if not bool(
            scalar(saved, "discrete_step_multiplier_dynamic_cap_reapplied")
        ):
            raise RuntimeError("Eta did not reapply the dynamic cap")
        if bool(scalar(saved, "discrete_step_multiplier_energy_backtracking")):
            raise RuntimeError("Eta enabled forbidden energy backtracking")
        if int(
            scalar(saved, "discrete_step_multiplier_extra_physical_evaluations")
        ) != 0:
            raise RuntimeError("Eta shim added physical evaluations")
        if int(
            scalar(saved, "discrete_step_multiplier_aborted_selection_count")
        ) != 0:
            raise RuntimeError("Eta result contains aborted selector events")

        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        mesh_ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        if cost.size != CANONICAL_BUDGET or active.shape != cost.shape:
            raise RuntimeError("Eta result is not N254")
        backtracks = np.asarray(saved["subproblem_line_search_backtracks"], dtype=int)
        if np.any(backtracks != 0):
            raise RuntimeError("Eta result recorded forbidden backtracking")
        curve = build_eligible_points(cost, i_perp, committed, cv, mesh_ratio)
        return {
            "eta": float(eta),
            "result": str(result),
            "result_sha256": file_sha256(result),
            "configuration_sha256": str(scalar(saved, "configuration_sha256")),
            "initial_path_sha256": str(scalar(saved, "initial_path_sha256")),
            "final_optimization_NF": int(cost[-1]),
            "ctp_trajectory_sha256": _hash_array_sequence(
                [np.asarray(saved[key]) for key in CTP_PARITY_KEYS]
            ),
            "npz_fields_accessed": sorted(ETA_MEMBER_ALLOWLIST),
            **curve,
        }


def comparison_with_wtlu(
    candidate: list[dict[str, object]],
    comparator: list[dict[str, object]],
    comparator_name: str,
) -> dict[str, object]:
    comparison = base.compare_curves(candidate, comparator, comparator_name)
    rows = comparison["threshold_results"]
    counts = {
        "W": sum(row["outcome"] == "strict_win" for row in rows),
        "T": sum(row["outcome"] == "tie" for row in rows),
        "L": sum(row["outcome"] == "loss" for row in rows),
        "U": sum(row["outcome"] == "candidate_unreachable_loss" for row in rows),
    }
    if sum(counts.values()) != int(comparison["comparator_pareto_record_count"]):
        raise RuntimeError("W/T/L/U does not cover every comparator threshold")
    comparison["W_T_L_U"] = counts
    comparison["loss_count_including_unreachable"] = counts["L"] + counts["U"]
    return comparison


def deficit_vector(
    candidate: list[dict[str, object]],
    comparator: list[dict[str, object]],
) -> dict[str, object]:
    return ratio_base.deficit_vector(candidate, comparator)


def decide_primary_winner(
    comparisons: list[dict[str, object]],
) -> dict[str, object]:
    joint = [
        row
        for row in comparisons
        if bool(row["joint_full_curve_dominates_all_three"])
    ]
    ranked = []
    for row in joint:
        key = (
            ratio_base.deficit_ordering_key(row["eta_one_deficit_vector"]),
            ratio_base.deficit_ordering_key(row["official_deficit_vector"]),
            ratio_base.deficit_ordering_key(row["legacy_deficit_vector"]),
            float(row["eta"]),
        )
        ranked.append((key, float(row["eta"])))
    ranked.sort(key=lambda item: item[0])
    selected = ranked[0][1] if ranked else None
    locked = selected is not None
    return {
        "joint_full_curve_dominator_etas": [float(row["eta"]) for row in joint],
        "selected_primary_winner_eta": selected,
        "multiple_primary_winner_tiebreak_applied": len(ranked) > 1,
        "eta_lock_status": (
            "locked_by_primary_success"
            if locked
            else "not_locked_no_joint_full_curve_dominator"
        ),
        "winner_claim_made": locked,
        "eta_lock_allowed": locked,
        "barrier_unembargo_allowed": locked,
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
    if audit.get("barrier_computation_or_densification_performed_by_wrapper") not in (
        None,
        False,
    ):
        raise RuntimeError(f"{context} audit reports barrier computation")
    command = audit.get("command")
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise RuntimeError(f"{context} audit command is invalid")
    return audit, command


def _verify_common_initial(record: object) -> tuple[Path, str]:
    path = verify_hash_record(record, "canonical_common_initial_path")
    with np.load(path, allow_pickle=False) as saved:
        require_keys(saved, NPZ_ACCESS_ALLOWLISTS["canonical_initial"], str(path))
        initial = np.ascontiguousarray(
            np.asarray(saved["initial_path"], dtype=np.float64)
        )
    array_hash = hashlib.sha256(initial.tobytes()).hexdigest()
    if array_hash != str(record["initial_path_array_sha256"]):
        raise RuntimeError("Canonical initial-path array hash mismatch")
    return path, array_hash


def verify_official_transitive_hashes(
    archive: Path, official: dict[str, object]
) -> dict[str, object]:
    """Verify archive -> combined -> phase files without barrier access."""
    with np.load(archive, allow_pickle=False) as saved:
        require_keys(saved, base.OFFICIAL_ARCHIVE_KEYS, str(archive))
        calls = np.asarray(saved["candidate_calls"], dtype=int)
        values = np.asarray(saved["candidate_i_perp"], dtype=float)
        if (
            calls.shape != values.shape
            or not calls.size
            or np.any(calls <= 0)
            or np.any(np.diff(calls) <= 0)
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > OFFICIAL_DOMAIN_MAX)
        ):
            raise RuntimeError("Official archive candidate sequence is invalid")
        source = Path(str(scalar(saved, "source_result"))).resolve()
        if file_sha256(source) != str(scalar(saved, "source_result_sha256")):
            raise RuntimeError("Official archive combined-source hash mismatch")
    with np.load(source, allow_pickle=False) as combined:
        require_keys(combined, OFFICIAL_COMBINED_ALLOWLIST, str(source))
        phases = (
            Path(str(scalar(combined, "phase_a_result"))).resolve(),
            Path(str(scalar(combined, "phase_b_result"))).resolve(),
        )
        embedded = (
            str(scalar(combined, "phase_a_result_sha256")),
            str(scalar(combined, "phase_b_result_sha256")),
        )
    observed = tuple(file_sha256(path) for path in phases)
    if observed != EXPECTED_OFFICIAL_COMPONENT_SHA256:
        raise RuntimeError("Official phase content drifted from the frozen comparator")
    if str(source) != str(official["source_result"]):
        raise RuntimeError("Official loader/source path mismatch")
    if tuple(str(path) for path in phases) != tuple(official["component_results"]):
        raise RuntimeError("Official loader/component path mismatch")
    if tuple(official["component_result_sha256"]) != observed:
        raise RuntimeError("Official loader/component hash mismatch")
    return {
        "archive_sha256": file_sha256(archive),
        "combined_source_sha256": file_sha256(source),
        "phase_result_sha256": list(observed),
        "combined_embedded_phase_result_sha256": list(embedded),
        "combined_embedded_phase_hash_matches_current": [
            expected == actual for expected, actual in zip(embedded, observed)
        ],
        "component_hash_authority": (
            "frozen formal component hashes; historical combined Phase-A "
            "embedded hash is retained as an audited stale metadata field"
        ),
        "candidate_call_count": int(calls.size),
    }


def validate_preregistration(
    prereg: dict[str, object], eta_report_path: Path
) -> None:
    if prereg.get("status") != "preregistered":
        raise RuntimeError("Eta preregistration status mismatch")
    if prereg.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Unexpected eta preregistration protocol")
    if Path(str(prereg["planned_report"])).resolve() != eta_report_path.resolve():
        raise RuntimeError("Eta report differs from preregistered path")
    if tuple(float(value) for value in prereg["eta_grid"]) != ETA_GRID:
        raise RuntimeError("Eta grid mismatch")
    if tuple(str(value) for value in prereg["eta_member_tags"]) != tuple(
        ETA_TAGS[eta] for eta in ETA_GRID
    ):
        raise RuntimeError("Eta member tags mismatch")
    frozen_numeric = {
        "fixed_alpha": CANONICAL_ALPHA,
        "fixed_residual_ratio": CANONICAL_RATIO,
        "fixed_max_evaluated_steps": CANONICAL_INNER_CAP,
        "fixed_lbfgs_history": CANONICAL_HISTORY,
        "budget_per_image": CANONICAL_BUDGET,
    }
    for key, expected in frozen_numeric.items():
        if float(prereg[key]) != float(expected):
            raise RuntimeError(f"Eta preregistration drift at {key}")
    required_identity = {
        "same_build_eta_one_control_is_newly_run": True,
        "old_eta_one_is_identity_oracle_not_reused_as_control": True,
        "eta_one_whole_run_parity_gate_required": True,
        "selector_isolated_from_eta": True,
        "same_dynamic_atom_displacement_cap_reapplied": True,
        "max_step_is_safety_cap_not_eta": True,
        "energy_or_objective_backtracking_added": False,
        "extra_physical_evaluations_from_eta_shim": 0,
        "all_non_eta_scientific_flags_frozen": True,
        "terminal_or_raw_iperp_ranking_allowed": False,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "overwrite_or_force_allowed": False,
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
    }
    for key, expected in required_identity.items():
        if prereg.get(key) != expected:
            raise RuntimeError(f"Eta preregistration drift at {key}")
    if prereg.get("eta_policy_id") != ETA_POLICY_ID or prereg.get(
        "eta_application_stage"
    ) != ETA_APPLICATION_STAGE:
        raise RuntimeError("Eta policy identity mismatch")
    if prereg.get("reparameterization_policy") != (
        "state_curve/per_image from first state"
    ):
        raise RuntimeError("Eta reparameterization policy mismatch")
    if prereg.get("eligible_record_definition") != (
        "initial plus reparam_committed states"
    ) or prereg.get("mesh_gate") != {
        "segment_length_cv_max": MESH_CV_MAX,
        "segment_length_ratio_max": MESH_RATIO_MAX,
    }:
        raise RuntimeError("Eta eligibility rule mismatch")
    if prereg.get("cost_to_precision_definition") != (
        "minimum actual optimization NF at I_perp <= threshold; no interpolation"
    ):
        raise RuntimeError("Eta CTP definition mismatch")
    if tuple(float(value) for value in prereg["reporting_anchors"]) != (
        REPORTING_ANCHORS
    ):
        raise RuntimeError("Eta reporting anchors mismatch")
    if prereg.get("multiple_primary_winner_tiebreak") != [
        "lexicographically smallest candidate-minus-eta1 NF vector tight-to-loose",
        "lexicographically smallest candidate-minus-official NF vector tight-to-loose",
        "lexicographically smallest candidate-minus-legacy NF vector tight-to-loose",
        "smallest eta only as an exact deterministic tie-break",
    ]:
        raise RuntimeError("Eta primary tiebreak mismatch")
    if prereg.get("no_primary_dominator_rule") != {
        "eta_lock_allowed": False,
        "winner_claim_allowed": False,
        "post_hoc_eta_grid_expansion_allowed": False,
        "barrier_unembargo_allowed": False,
        "retain_eta_one_as_control": True,
    }:
        raise RuntimeError("Eta no-dominator rule is not fail-closed")
    if set(prereg["decision_inputs"]) != EXPECTED_DECISION_INPUT_KEYS:
        raise RuntimeError("Eta decision-input manifest mismatch")
    if set(prereg["run_source_manifest"]) != EXPECTED_SOURCE_MANIFEST_KEYS:
        raise RuntimeError("Eta source manifest mismatch")
    if set(prereg["registered_command_hashes"]) != {
        ETA_TAGS[eta] for eta in ETA_GRID
    }:
        raise RuntimeError("Eta registered-command manifest mismatch")


def analyze(eta_report_path: Path) -> dict[str, object]:
    eta_report_path = eta_report_path.resolve()
    report = read_json(eta_report_path)
    if report.get("status") != "complete":
        raise RuntimeError("Eta report is not complete")
    if report.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Unexpected eta report protocol")
    prereg_path = Path(str(report["preregistration"])).resolve()
    if file_sha256(prereg_path) != str(report["preregistration_sha256"]):
        raise RuntimeError("Eta preregistration hash mismatch")
    prereg = read_json(prereg_path)
    validate_preregistration(prereg, eta_report_path)
    for key, value in prereg.items():
        if key != "status" and report.get(key) != value:
            raise RuntimeError(f"Eta report drifted from preregistration at {key}")
    if report.get("barrier_fields_accessed_by_launcher") is not False:
        raise RuntimeError("Eta launcher reports barrier access")
    if report.get("formal_ranking_deferred_to_separate_readonly_iperp_ctp") is not True:
        raise RuntimeError("Eta launcher ranked before the readonly audit")

    verified_sources = verify_hash_manifest(
        prereg["run_source_manifest"], "run_source_manifest"
    )
    decision_paths = {
        name: verify_hash_record(record, f"decision_inputs.{name}")
        for name, record in prereg["decision_inputs"].items()
    }
    common_path, common_array_hash = _verify_common_initial(
        prereg["canonical_common_initial_path"]
    )

    old_command = read_command_json(
        decision_paths["old_eta_one_identity_oracle_command"]
    )
    legacy_command_file = read_command_json(
        decision_paths["same_alpha_legacy_command"]
    )
    base.assert_member_command(old_command, CANONICAL_ALPHA, "candidate")
    ratio_base.assert_legacy_command(legacy_command_file)
    old_audit, old_audit_command = _verify_audit_result(
        decision_paths["old_eta_one_identity_oracle_audit"],
        decision_paths["old_eta_one_identity_oracle_result"],
        str(prereg["decision_inputs"]["old_eta_one_identity_oracle_result"]["sha256"]),
        "old eta=1 identity oracle",
    )
    legacy_audit, legacy_audit_command = _verify_audit_result(
        decision_paths["same_alpha_legacy_audit"],
        decision_paths["same_alpha_legacy_result"],
        str(prereg["decision_inputs"]["same_alpha_legacy_result"]["sha256"]),
        "same-alpha legacy",
    )
    if old_audit_command != old_command:
        raise RuntimeError("Old eta=1 command file/audit mismatch")
    if legacy_audit_command != legacy_command_file:
        raise RuntimeError("Legacy command file/audit mismatch")

    source_manifest = prereg["run_source_manifest"]
    shim_path = Path(str(source_manifest["eta_runtime_shim"]["path"])).resolve()
    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) != len(ETA_GRID):
        raise RuntimeError("Eta report does not contain exactly four rows")
    records: dict[float, dict[str, object]] = {}
    member_audit_hashes = {}
    for row in rows:
        if not isinstance(row, dict) or int(row.get("returncode", -1)) != 0:
            raise RuntimeError("Eta report contains a failed member")
        eta = float(row["eta"])
        tag = str(row["eta_tag"])
        if eta not in ETA_GRID or eta in records or tag != ETA_TAGS[eta]:
            raise RuntimeError("Eta report contains unexpected/duplicate member")
        result = Path(str(row["result"])).resolve()
        audit_path = Path(str(row["member_audit"])).resolve()
        if file_sha256(result) != str(row["result_sha256"]):
            raise RuntimeError("Eta result hash mismatch")
        if file_sha256(audit_path) != str(row["member_audit_sha256"]):
            raise RuntimeError("Eta audit hash mismatch")
        audit, command = _verify_audit_result(
            audit_path, result, str(row["result_sha256"]), f"eta={eta:g}"
        )
        expected_role = "same_build_eta1_control" if eta == 1.0 else "eta_candidate"
        if (
            float(audit["eta"]) != eta
            or str(audit["member_role"]) != expected_role
            or str(audit["eta_policy"]) != ETA_POLICY_ID
            or str(audit["eta_application_stage"]) != ETA_APPLICATION_STAGE
        ):
            raise RuntimeError("Eta member audit identity mismatch")
        if audit.get("energy_backtracking") is not False or int(
            audit.get("extra_physical_evaluations_from_shim", -1)
        ) != 0:
            raise RuntimeError("Eta member audit changed query policy")
        if audit.get("formal_ranking_deferred_to_complete_mesh_iperp_ctp") is not True:
            raise RuntimeError("Eta member audit ranked prematurely")
        if any("barrier" in str(key).lower() for key in audit["result_fields_accessed"]):
            raise RuntimeError("Eta member audit allowlist contains barrier data")
        assert_eta_command(command, old_command, eta, tag, shim_path)
        registered_hash = str(prereg["registered_command_hashes"][tag])
        if command_sha256(command) != registered_hash or str(
            audit["scientific_command_sha256"]
        ) != registered_hash:
            raise RuntimeError("Eta registered command hash mismatch")

        loaded = load_eta_result(result, eta)
        if list(audit["complete_state_NF"]) != loaded["complete_state_NF"]:
            raise RuntimeError("Eta complete-state audit mismatch")
        if list(audit["mesh_eligible_complete_state_NF"]) != loaded[
            "mesh_eligible_complete_state_NF"
        ]:
            raise RuntimeError("Eta mesh audit mismatch")
        if int(audit["final_optimization_NF"]) != loaded["final_optimization_NF"]:
            raise RuntimeError("Eta terminal NF audit mismatch")
        if int(row["final_optimization_NF"]) != loaded["final_optimization_NF"]:
            raise RuntimeError("Eta report terminal NF mismatch")
        if int(row["mesh_eligible_complete_state_count"]) != len(
            loaded["mesh_eligible_complete_state_NF"]
        ):
            raise RuntimeError("Eta report mesh count mismatch")
        parity = audit.get("eta_one_whole_run_parity_or_not_applicable")
        row_parity = row.get("eta_one_parity_gate_passed")
        if eta == 1.0:
            if parity is not True or row_parity is not True:
                raise RuntimeError("New eta=1 failed its parity gate")
        elif parity is not False or row_parity is not False:
            raise RuntimeError("Non-control eta carries a parity claim")
        records[eta] = {
            "row": row,
            "audit": audit,
            "audit_path": audit_path,
            "command": command,
            "curve": loaded,
        }
        member_audit_hashes[str(audit_path)] = file_sha256(audit_path)
    if set(records) != set(ETA_GRID):
        raise RuntimeError("Eta report is missing a registered member")

    legacy = ratio_base.load_member_result(
        decision_paths["same_alpha_legacy_result"], "legacy_control"
    )
    identity_oracle = ratio_base.load_member_result(
        decision_paths["old_eta_one_identity_oracle_result"],
        "candidate",
        CANONICAL_RATIO,
    )
    if list(legacy_audit["complete_state_NF"]) != legacy["complete_state_NF"]:
        raise RuntimeError("Legacy complete-state audit mismatch")
    if list(old_audit["complete_state_NF"]) != identity_oracle["complete_state_NF"]:
        raise RuntimeError("Identity-oracle complete-state audit mismatch")
    eta_one = records[1.0]["curve"]
    oracle_ctp_hash = ctp_trajectory_sha256(
        decision_paths["old_eta_one_identity_oracle_result"]
    )
    if eta_one["ctp_trajectory_sha256"] != oracle_ctp_hash:
        raise RuntimeError("New eta=1 failed independent CTP trajectory parity")

    initial_hashes = {
        legacy["initial_path_sha256"],
        identity_oracle["initial_path_sha256"],
        *(record["curve"]["initial_path_sha256"] for record in records.values()),
    }
    if initial_hashes != {common_array_hash}:
        raise RuntimeError("Eta members do not share the canonical initial path")

    official_path = decision_paths["official_stage4_v4"]
    official = base.load_official_curve(official_path)
    if official["source_result_sha256"] != EXPECTED_OFFICIAL_SOURCE_SHA256:
        raise RuntimeError("Official combined source hash drifted")
    official_transitive_hashes = verify_official_transitive_hashes(
        official_path, official
    )
    official_calls = [int(row["NF"]) for row in official["eligible_points"] + official["mesh_rejected_points"]]
    if any(call <= 0 for call in official_calls) or len(set(official_calls)) != len(official_calls):
        raise RuntimeError("Official archive contains invalid/duplicate NF")
    if any(
        float(row["i_perp"]) > OFFICIAL_DOMAIN_MAX
        for row in official["eligible_points"] + official["mesh_rejected_points"]
    ):
        raise RuntimeError("Official archive exceeds its activation domain")
    if not official["pareto_records"]:
        raise RuntimeError("Official comparator has no Pareto records")

    anchors = [float(value) for value in prereg["reporting_anchors"]]
    comparisons = []
    for eta in CANDIDATE_ETAS:
        candidate = records[eta]["curve"]
        candidate_points = candidate["eligible_points"]
        eta_one_comparison = comparison_with_wtlu(
            candidate_points,
            eta_one["eligible_points"],
            "same_build_eta1_control",
        )
        legacy_comparison = comparison_with_wtlu(
            candidate_points,
            legacy["eligible_points"],
            "same_alpha_legacy_control",
        )
        official_comparison = comparison_with_wtlu(
            candidate_points,
            official["eligible_points"],
            "official_stage4_v4_post_activation_mesh_eligible_tail",
        )
        joint = bool(
            eta_one_comparison["full_curve_dominates"]
            and legacy_comparison["full_curve_dominates"]
            and official_comparison["full_curve_dominates"]
        )
        comparisons.append(
            {
                "eta": eta,
                "candidate": candidate,
                "same_build_eta1_comparison": eta_one_comparison,
                "same_alpha_legacy_comparison": legacy_comparison,
                "official_stage4_v4_comparison": official_comparison,
                "triple_tightest_precision_ok": bool(
                    eta_one_comparison["candidate_does_not_lose_tightest_precision"]
                    and legacy_comparison["candidate_does_not_lose_tightest_precision"]
                    and official_comparison["candidate_does_not_lose_tightest_precision"]
                ),
                "joint_full_curve_dominates_all_three": joint,
                "eta_one_deficit_vector": deficit_vector(
                    candidate_points, eta_one["eligible_points"]
                ),
                "official_deficit_vector": deficit_vector(
                    candidate_points, official["eligible_points"]
                ),
                "legacy_deficit_vector": deficit_vector(
                    candidate_points, legacy["eligible_points"]
                ),
                "reporting_anchors_same_build_eta1": base.anchor_rows(
                    candidate_points, eta_one["eligible_points"], anchors
                ),
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
    no_winner = decision["selected_primary_winner_eta"] is None
    if no_winner and any(
        decision[key]
        for key in (
            "winner_claim_made",
            "eta_lock_allowed",
            "barrier_unembargo_allowed",
        )
    ):
        raise RuntimeError("No-winner eta decision is not fail-closed")

    result_hashes = {
        legacy["result"]: legacy["result_sha256"],
        identity_oracle["result"]: identity_oracle["result_sha256"],
        **{
            record["curve"]["result"]: record["curve"]["result_sha256"]
            for record in records.values()
        },
    }
    audit_hashes = {
        str(decision_paths["same_alpha_legacy_audit"]): file_sha256(
            decision_paths["same_alpha_legacy_audit"]
        ),
        str(decision_paths["old_eta_one_identity_oracle_audit"]): file_sha256(
            decision_paths["old_eta_one_identity_oracle_audit"]
        ),
        **member_audit_hashes,
    }
    return {
        "analysis_id": ANALYSIS_ID,
        "created_unix": time.time(),
        "read_only_formal_input_audit": True,
        "barrier_data_accessed": False,
        "barrier_computation_or_densification_performed": False,
        "barrier_unembargoed": False,
        "eta_report": str(eta_report_path),
        "eta_report_sha256": file_sha256(eta_report_path),
        "eta_preregistration": str(prereg_path),
        "eta_preregistration_sha256": file_sha256(prereg_path),
        "verified_source_hashes": verified_sources,
        "verified_report_prereg_source_command_result_audit_hashes": True,
        "canonical_initial_path": str(common_path),
        "canonical_initial_path_file_sha256": file_sha256(common_path),
        "canonical_initial_path_array_sha256": common_array_hash,
        "eta_grid": list(ETA_GRID),
        "candidate_etas": list(CANDIDATE_ETAS),
        "fixed_alpha": CANONICAL_ALPHA,
        "fixed_residual_ratio": CANONICAL_RATIO,
        "fixed_max_evaluated_steps": CANONICAL_INNER_CAP,
        "fixed_budget_per_image": CANONICAL_BUDGET,
        "command_identity": (
            "all eta members are exact scientific argv clones of the frozen "
            "alpha12.5/r0.8/k10/N254 command except the registered runtime shim "
            "and one post-selection eta option; routing is isolated"
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
        "unreachable_policy": "candidate unreachable is U and is a formal loss",
        "official_comparison_scope": (
            "frozen official-v4 complete mesh-eligible post-activation tail "
            "with I_perp<=0.007"
        ),
        "primary_success_criterion": prereg["primary_success_criterion"],
        "multiple_primary_winner_tiebreak": prereg[
            "multiple_primary_winner_tiebreak"
        ],
        "new_eta_one_control": eta_one,
        "old_eta_one_identity_oracle": identity_oracle,
        "new_eta_one_ctp_parity_with_oracle": True,
        "same_alpha_legacy_control": legacy,
        "official_stage4_v4": official,
        "official_transitive_hashes": official_transitive_hashes,
        "formal_result_hashes": result_hashes,
        "formal_member_audit_hashes": audit_hashes,
        "comparisons": comparisons,
        **decision,
        "no_primary_dominator_rule_applied": no_winner,
        "no_primary_dominator_actions": (
            {
                "winner_claim_allowed": False,
                "eta_lock_allowed": False,
                "post_hoc_eta_grid_expansion_allowed": False,
                "barrier_unembargo_allowed": False,
                "retain_eta_one_as_control": True,
            }
            if no_winner
            else None
        ),
        "barrier_remains_embargoed": no_winner,
        "reporting_anchors": anchors,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    report = args.eta_report.expanduser().resolve()
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
