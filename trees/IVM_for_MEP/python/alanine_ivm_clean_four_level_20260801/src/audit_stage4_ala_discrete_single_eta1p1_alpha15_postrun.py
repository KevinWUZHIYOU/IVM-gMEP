#!/usr/bin/env python3
"""Independent read-only mechanism/provenance audit for eta=1.1, alpha=15.

The completed formal run is immutable.  This auditor pins its preregistration,
formal report, result, member audit, runner companion audit, command records,
and every registered run-source/decision-input hash.  It then independently
recomputes the pre-eta H0/L-BFGS winner and the post-selection eta recap.

No energy-barrier NPZ member is ever indexed.  Whole-file SHA256 computation
does read artifact bytes, but no barrier value is loaded, decoded, summarized,
or ranked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np


AUDIT_ID = (
    "stage4_ala_discrete_single_eta1p1_alpha15_"
    "postrun_mechanism_provenance_readonly_v1"
)
PROTOCOL_ID = "stage4_ala_discrete_single_eta1p1_alpha15_followup_v1"

FORMAL_REPORT_RELATIVE = (
    "reports/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_"
    "N254_r1_1787257808883811000.json"
)
FORMAL_REPORT_SHA256 = (
    "fa0bf64922d91d15a48b0b6b6949d8ddb3ff109661ff011b351f8f9caee79212"
)
PREREGISTRATION_RELATIVE = (
    "reports/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_"
    "N254_r1_1787257808883811000_preregistered.json"
)
PREREGISTRATION_SHA256 = (
    "dbffae0850f39128c313f2819c28b7679d5e80ef0a6f9fe59246dc43a176df1d"
)
RESULT_RELATIVE = (
    "runs/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1/members/"
    "ivm_stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_eta1p1_"
    "r08c10_a15_N254_discrete_b040_r1_central_state_ivm_"
    "cfg02c3f3990307_budget254.npz"
)
RESULT_SHA256 = (
    "bb189235868d63be23d335c0766db59e0315b82fd11448077a492966ce5eaa55"
)
MEMBER_AUDIT_RELATIVE = (
    "runs/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1/audits/"
    "stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_eta1p1_"
    "r08c10_a15_N254_discrete_b040_r1.json"
)
MEMBER_AUDIT_SHA256 = (
    "058d591e5fccc4a0e7a4dee0854c4f372b7b2648d2673dd25a9bd58363eed91a"
)
COMPANION_AUDIT_RELATIVE = RESULT_RELATIVE[: -len(".npz")] + ".audit.json"
COMPANION_AUDIT_SHA256 = (
    "c6bf4bd4dcf8df9bb909780ebed631a5b72e726f9a168bfc0036223b62aed498"
)
MEMBER_COMMAND_RELATIVE = (
    "logs/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1/members/"
    "stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_eta1p1_"
    "r08c10_a15_N254_discrete_b040_r1/command.json"
)
MEMBER_COMMAND_SHA256 = (
    "a3f622b4be15b64574479d5832bd76fc2a0de8fb3e2032ef4103e776129fb61f"
)
LAUNCHER_COMMAND_RELATIVE = (
    "logs/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1/launcher/"
    "stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_eta1p1_"
    "r08c10_a15_N254_discrete_b040_r1/command.json"
)
LAUNCHER_COMMAND_SHA256 = (
    "3b9574d6cb86054b1ec3909d53ebaeb7126a829a8303aef77057eaa96418a948"
)

ETA = 1.1
ALPHA = 15.0
RESIDUAL_RATIO = 0.8
MAX_EVALUATED_STEPS = 10
HISTORY = 7
BUDGET = 254
CONFIGURATION_SHA256 = (
    "02c3f3990307b894013cd41e97182d0045314466b500159cd059f93a42578a18"
)
INITIAL_PATH_SHA256 = (
    "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
)
SCIENTIFIC_COMMAND_SHA256 = (
    "6e01b50f8e13a627a7b9f72f6b1e6e8f8ec8f9aa3ddf605eebe950720a946b8c"
)

# Names are discovered from the archive directory only.  Values under these
# names are never indexed.  A future new barrier name is safe because read()
# rejects every name containing "barrier", not just this registered set.
EXPECTED_BARRIER_FIELD_NAMES = frozenset(
    {
        "optimization_barrier_is_sparse",
        "barrier_metric_for_figures",
        "barrier_kcal",
    }
)

MECHANISM_FIELDS = frozenset(
    {
        "configuration_sha256",
        "initial_path_sha256",
        "alpha",
        "alpha_mode",
        "active_alpha",
        "alpha_action",
        "budget",
        "frozen_subproblem",
        "discrete_stop_mode",
        "discrete_inner_max_steps",
        "discrete_residual_ratio",
        "discrete_history",
        "discrete_audit_history_count",
        "discrete_audit_outer",
        "discrete_audit_step",
        "discrete_audit_stop_reason",
        "discrete_audit_global_relative_residual",
        "discrete_audit_selection",
        "discrete_audit_h0_capped_predicted_decrease",
        "discrete_audit_lbfgs_capped_predicted_decrease",
        "discrete_h0_selection_count",
        "discrete_lbfgs_selection_count",
        "discrete_step_multiplier_eta",
        "discrete_step_multiplier_selector_isolated",
        "discrete_step_multiplier_dynamic_cap_reapplied",
        "discrete_step_multiplier_energy_backtracking",
        "discrete_step_multiplier_extra_physical_evaluations",
        "discrete_step_multiplier_audit_selection",
        "discrete_step_multiplier_audit_requested",
        "discrete_step_multiplier_audit_applied",
        "discrete_step_multiplier_audit_cap_hit",
        "discrete_step_multiplier_audit_dynamic_max_step",
        "discrete_step_multiplier_audit_base_max_atomic_displacement",
        "discrete_step_multiplier_audit_requested_max_atomic_displacement",
        "discrete_step_multiplier_audit_applied_max_atomic_displacement",
        "discrete_step_multiplier_uncommitted_selection_count",
        "discrete_step_multiplier_aborted_selection_count",
        "subproblem_line_search_backtracks",
        "reparam_mode",
        "state_curve_direct_objective_acceptance",
        "state_curve_projection_uses_spline",
        "geometric_reparam_joint_evaluations",
        "stored_path_calls",
        "total_internal_joint_evaluations",
        "counter_vectors",
        "cache_hit_vectors",
        "energy_request_vectors",
        "force_request_vectors",
    }
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pinned(root: Path, relative: str, expected: str, label: str) -> dict[str, str]:
    path = (root / relative).resolve()
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"Pinned {label} hash drifted: {path}")
    return {"path": str(path), "sha256": actual}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _compact_command_sha256(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _exact_path(value: object, expected: Path, label: str) -> None:
    if Path(str(value)).expanduser().resolve() != expected.resolve():
        raise RuntimeError(f"{label} path drifted")


def _verify_hash_manifest(
    root: Path, manifest: object, label: str
) -> dict[str, dict[str, str]]:
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError(f"Invalid {label} manifest")
    verified: dict[str, dict[str, str]] = {}
    for name, raw_record in sorted(manifest.items()):
        if not isinstance(name, str) or not isinstance(raw_record, dict):
            raise RuntimeError(f"Invalid {label} record")
        if set(raw_record) != {"path", "sha256"}:
            raise RuntimeError(f"Unexpected {label}.{name} schema")
        path = Path(str(raw_record["path"])).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"{label}.{name} escapes canonical root") from error
        expected = str(raw_record["sha256"])
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"{label}.{name} hash drifted: {path}")
        verified[name] = {"path": str(path), "sha256": actual}
    return verified


def verify_provenance(root: Path, formal_report: Path) -> dict[str, Any]:
    """Verify every immutable formal artifact and registered input/source."""
    root = Path(root).expanduser().resolve()
    formal_report = Path(formal_report).expanduser().resolve()
    expected_report = (root / FORMAL_REPORT_RELATIVE).resolve()
    if formal_report != expected_report:
        raise RuntimeError("Unregistered formal report supplied to audit")

    artifacts = {
        "formal_report": _pinned(
            root, FORMAL_REPORT_RELATIVE, FORMAL_REPORT_SHA256, "formal report"
        ),
        "preregistration": _pinned(
            root,
            PREREGISTRATION_RELATIVE,
            PREREGISTRATION_SHA256,
            "preregistration",
        ),
        "result": _pinned(root, RESULT_RELATIVE, RESULT_SHA256, "result"),
        "member_audit": _pinned(
            root, MEMBER_AUDIT_RELATIVE, MEMBER_AUDIT_SHA256, "member audit"
        ),
        "runner_companion_audit": _pinned(
            root,
            COMPANION_AUDIT_RELATIVE,
            COMPANION_AUDIT_SHA256,
            "runner companion audit",
        ),
        "member_command": _pinned(
            root,
            MEMBER_COMMAND_RELATIVE,
            MEMBER_COMMAND_SHA256,
            "member command",
        ),
        "launcher_command": _pinned(
            root,
            LAUNCHER_COMMAND_RELATIVE,
            LAUNCHER_COMMAND_SHA256,
            "launcher command",
        ),
    }

    report = _read_json(expected_report)
    prereg_path = (root / PREREGISTRATION_RELATIVE).resolve()
    prereg = _read_json(prereg_path)
    if report.get("status") != "complete" or report.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Formal report is not the completed registered protocol")
    if prereg.get("status") != "preregistered" or prereg.get(
        "protocol_id"
    ) != PROTOCOL_ID:
        raise RuntimeError("Pinned preregistration identity drifted")
    _exact_path(report.get("preregistration"), prereg_path, "preregistration")
    if report.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise RuntimeError("Formal report/preregistration hash link drifted")
    _exact_path(prereg.get("planned_report"), expected_report, "planned report")

    # The launcher clones every preregistered field, changing only status and
    # adding execution results/provenance.  Check that immutable clone exactly.
    for key, value in prereg.items():
        if key == "status":
            continue
        if key not in report or report[key] != value:
            raise RuntimeError(f"Formal report changed preregistered field {key}")

    fixed = {
        "fixed_eta": ETA,
        "fixed_alpha": ALPHA,
        "fixed_residual_ratio": RESIDUAL_RATIO,
        "fixed_max_evaluated_steps": MAX_EVALUATED_STEPS,
        "fixed_lbfgs_history": HISTORY,
        "budget_per_image": BUDGET,
        "replicate": "r1",
        "selector_isolated_from_eta": True,
        "same_dynamic_atom_displacement_cap_reapplied": True,
        "energy_or_objective_backtracking_added": False,
        "extra_physical_evaluations_from_eta_shim": 0,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
    }
    for key, expected in fixed.items():
        if prereg.get(key) != expected:
            raise RuntimeError(f"Preregistered fixed field drifted at {key}")
    if report.get("barrier_fields_accessed_by_launcher") is not False:
        raise RuntimeError("Formal launcher claims barrier access")
    if report.get("formal_ranking_deferred_to_separate_readonly_iperp_ctp") is not True:
        raise RuntimeError("Formal report did not defer scientific ranking")

    if report.get("run_source_manifest") != prereg.get("run_source_manifest"):
        raise RuntimeError("Formal source manifest differs from preregistration")
    if report.get("decision_inputs") != prereg.get("decision_inputs"):
        raise RuntimeError("Formal decision inputs differ from preregistration")
    sources = _verify_hash_manifest(
        root, prereg.get("run_source_manifest"), "run source"
    )
    decision_inputs = _verify_hash_manifest(
        root, prereg.get("decision_inputs"), "decision input"
    )

    results = report.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError("Formal report is not the registered single arm")
    row = results[0]
    if not isinstance(row, dict):
        raise RuntimeError("Malformed formal result row")
    if (
        row.get("returncode") != 0
        or row.get("eta") != ETA
        or row.get("alpha") != ALPHA
    ):
        raise RuntimeError("Formal result row identity/completion drifted")
    result_path = (root / RESULT_RELATIVE).resolve()
    member_audit_path = (root / MEMBER_AUDIT_RELATIVE).resolve()
    _exact_path(row.get("result"), result_path, "formal result")
    _exact_path(row.get("member_audit"), member_audit_path, "member audit")
    if row.get("result_sha256") != RESULT_SHA256 or row.get(
        "member_audit_sha256"
    ) != MEMBER_AUDIT_SHA256:
        raise RuntimeError("Formal row artifact hash link drifted")

    member_audit = _read_json(member_audit_path)
    companion = _read_json((root / COMPANION_AUDIT_RELATIVE).resolve())
    member_command_path = (root / MEMBER_COMMAND_RELATIVE).resolve()
    launcher_command_path = (root / LAUNCHER_COMMAND_RELATIVE).resolve()
    member_command = json.loads(member_command_path.read_text(encoding="utf-8"))
    launcher_command = json.loads(launcher_command_path.read_text(encoding="utf-8"))
    if not isinstance(member_command, list) or not all(
        isinstance(token, str) for token in member_command
    ):
        raise RuntimeError("Member command record is not an argv vector")
    if not isinstance(launcher_command, list) or not all(
        isinstance(token, str) for token in launcher_command
    ):
        raise RuntimeError("Launcher command record is not an argv vector")
    if member_audit.get("command") != member_command:
        raise RuntimeError("Member audit/command provenance link drifted")
    command_hash = _compact_command_sha256(member_command)
    if command_hash != SCIENTIFIC_COMMAND_SHA256:
        raise RuntimeError("Scientific command hash drifted")
    for location, value in (
        ("preregistration", prereg.get("registered_command_sha256")),
        ("formal report", report.get("registered_command_sha256")),
        ("formal row", row.get("registered_command_sha256")),
        ("member audit", member_audit.get("scientific_command_sha256")),
    ):
        if value != command_hash:
            raise RuntimeError(f"{location} command hash link drifted")

    _exact_path(member_audit.get("result"), result_path, "member result")
    _exact_path(companion.get("result_file"), result_path, "companion result")
    if member_audit.get("result_sha256") != RESULT_SHA256:
        raise RuntimeError("Member audit/result hash link drifted")
    if member_audit.get("barrier_fields_accessed") is not False or member_audit.get(
        "barrier_computation_or_densification_performed_by_wrapper"
    ) is not False:
        raise RuntimeError("Member audit claims barrier work")
    if member_audit.get("energy_backtracking") is not False or member_audit.get(
        "extra_physical_evaluations_from_shim"
    ) != 0:
        raise RuntimeError("Member audit claims fallback or extra evaluations")
    if companion.get("consistent") is not True:
        raise RuntimeError("Runner companion audit is inconsistent")

    # The launcher command is operational only, but its pinned record must invoke
    # the registered one-arm wrapper and canonical root exactly once.
    expected_wrapper = str(
        (root / "src/run_stage4_ala_discrete_single_eta1p1_alpha15_followup.py").resolve()
    )
    if launcher_command.count(expected_wrapper) != 1 or launcher_command.count(
        str(root)
    ) != 1:
        raise RuntimeError("Launcher command provenance drifted")

    return {
        "immutable_artifacts": artifacts,
        "registered_run_sources": sources,
        "registered_decision_inputs": decision_inputs,
        "run_source_count": len(sources),
        "decision_input_count": len(decision_inputs),
        "scientific_command_sha256": command_hash,
        "configuration_sha256": CONFIGURATION_SHA256,
        "formal_result_row": {
            "eta": float(row["eta"]),
            "alpha": float(row["alpha"]),
            "returncode": int(row["returncode"]),
            "final_optimization_NF": int(row["final_optimization_NF"]),
            "h0_selection_count": int(row["h0_selection_count"]),
            "lbfgs_selection_count": int(row["lbfgs_selection_count"]),
            "eta_cap_hit_count": int(row["eta_cap_hit_count"]),
        },
        "formal_preregistration_clone_verified": True,
        "all_registered_source_hashes_verified": True,
        "all_registered_decision_input_hashes_verified": True,
        "all_artifact_cross_links_verified": True,
    }


def _failure_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)))


def _require_zero(failures: Mapping[str, int], prefix: str) -> None:
    nonzero = {key: value for key, value in failures.items() if value}
    if nonzero:
        raise RuntimeError(prefix + ": " + json.dumps(nonzero, sort_keys=True))


def audit_mechanism(result: Path) -> dict[str, Any]:
    """Recompute selector, eta recap, fixed controls, and physical accounting."""
    accessed: set[str] = set()
    result = Path(result).expanduser().resolve()
    with np.load(result, allow_pickle=False) as saved:
        field_names = set(saved.files)
        barrier_names = {name for name in field_names if "barrier" in name.lower()}
        if barrier_names != set(EXPECTED_BARRIER_FIELD_NAMES):
            raise RuntimeError("Unexpected barrier-field namespace by name")
        missing = sorted(MECHANISM_FIELDS.difference(field_names))
        if missing:
            raise RuntimeError("Mechanism result missing: " + ", ".join(missing))

        def read(name: str, dtype: Any = None) -> np.ndarray:
            if "barrier" in name.lower():
                raise RuntimeError("Barrier values are embargoed in this audit")
            accessed.add(name)
            value = np.asarray(saved[name])
            return value if dtype is None else np.asarray(value, dtype=dtype)

        def scalar(name: str) -> Any:
            return read(name).item()

        fixed_scalar_expectations = {
            "configuration_sha256": CONFIGURATION_SHA256,
            "initial_path_sha256": INITIAL_PATH_SHA256,
            "alpha": ALPHA,
            "alpha_mode": "fixed",
            "budget": BUDGET,
            "frozen_subproblem": "discrete_b040",
            "discrete_stop_mode": "r08c10",
            "discrete_inner_max_steps": MAX_EVALUATED_STEPS,
            "discrete_residual_ratio": RESIDUAL_RATIO,
            "discrete_history": HISTORY,
            "reparam_mode": "state_curve",
            "state_curve_direct_objective_acceptance": "per_image",
            "state_curve_projection_uses_spline": False,
            "geometric_reparam_joint_evaluations": 0,
            "discrete_step_multiplier_eta": ETA,
            "discrete_step_multiplier_selector_isolated": True,
            "discrete_step_multiplier_dynamic_cap_reapplied": True,
            "discrete_step_multiplier_energy_backtracking": False,
            "discrete_step_multiplier_extra_physical_evaluations": 0,
            "discrete_step_multiplier_uncommitted_selection_count": 0,
            "discrete_step_multiplier_aborted_selection_count": 0,
        }
        for name, expected in fixed_scalar_expectations.items():
            if scalar(name) != expected:
                raise RuntimeError(f"Fixed mechanism field drifted at {name}")
        if MAX_EVALUATED_STEPS <= 3:
            raise RuntimeError("Registered maximum evaluated steps is not >3")

        active_alpha = read("active_alpha", float)
        alpha_action = read("alpha_action", "U16")
        if active_alpha.shape != (BUDGET,) or not np.all(active_alpha == ALPHA):
            raise RuntimeError("Alpha was not exactly 15 throughout")
        if alpha_action.shape != (BUDGET,) or alpha_action[0] != "initial" or not np.all(
            alpha_action[1:] == "fixed"
        ):
            raise RuntimeError("Alpha action history is not fixed")

        selection = read("discrete_audit_selection", "U8")
        eta_selection = read("discrete_step_multiplier_audit_selection", "U8")
        h0_score = read("discrete_audit_h0_capped_predicted_decrease", float)
        lbfgs_score = read(
            "discrete_audit_lbfgs_capped_predicted_decrease", float
        )
        requested_multiplier = read(
            "discrete_step_multiplier_audit_requested", float
        )
        applied_multiplier = read("discrete_step_multiplier_audit_applied", float)
        cap_hit = read("discrete_step_multiplier_audit_cap_hit", bool)
        dynamic = read("discrete_step_multiplier_audit_dynamic_max_step", float)
        base = read(
            "discrete_step_multiplier_audit_base_max_atomic_displacement", float
        )
        requested = read(
            "discrete_step_multiplier_audit_requested_max_atomic_displacement",
            float,
        )
        applied = read(
            "discrete_step_multiplier_audit_applied_max_atomic_displacement",
            float,
        )
        aligned = (
            eta_selection,
            h0_score,
            lbfgs_score,
            requested_multiplier,
            applied_multiplier,
            cap_hit,
            dynamic,
            base,
            requested,
            applied,
        )
        if selection.ndim != 1 or not selection.size or any(
            array.shape != selection.shape for array in aligned
        ):
            raise RuntimeError("Mechanism vectors are not aligned")
        numeric = (
            h0_score,
            lbfgs_score,
            requested_multiplier,
            applied_multiplier,
            dynamic,
            base,
            requested,
            applied,
        )
        if any(not np.all(np.isfinite(array)) for array in numeric):
            raise RuntimeError("Mechanism vectors contain non-finite values")
        if np.any(base <= 0.0) or np.any(dynamic <= 0.0):
            raise RuntimeError("Mechanism displacement audit is non-positive")

        expected_selection = np.where(h0_score >= lbfgs_score, "H0", "LBFGS")
        expected_requested = ETA * base
        expected_applied = np.minimum(expected_requested, dynamic)
        selector_failures = {
            "selector_winner_mismatch_count": _failure_count(
                selection != expected_selection
            ),
            "eta_selection_label_mismatch_count": _failure_count(
                eta_selection != selection
            ),
            "recorded_h0_count_mismatch_count": int(
                int(scalar("discrete_h0_selection_count"))
                != np.count_nonzero(selection == "H0")
            ),
            "recorded_lbfgs_count_mismatch_count": int(
                int(scalar("discrete_lbfgs_selection_count"))
                != np.count_nonzero(selection == "LBFGS")
            ),
        }
        recap_failures = {
            "requested_multiplier_mismatch_count": _failure_count(
                requested_multiplier != ETA
            ),
            "base_exceeds_dynamic_cap_count": _failure_count(
                base > dynamic * (1.0 + 1.0e-12)
            ),
            "requested_eta_times_base_mismatch_count": _failure_count(
                ~np.isclose(
                    requested,
                    expected_requested,
                    rtol=1.0e-13,
                    atol=1.0e-14,
                )
            ),
            "applied_below_base_count": _failure_count(
                applied < base * (1.0 - 1.0e-12)
            ),
            "applied_min_eta_base_dynamic_cap_mismatch_count": _failure_count(
                ~np.isclose(
                    applied,
                    expected_applied,
                    rtol=1.0e-12,
                    atol=1.0e-14,
                )
            ),
            "applied_multiplier_mismatch_count": _failure_count(
                ~np.isclose(
                    applied_multiplier,
                    applied / base,
                    rtol=1.0e-12,
                    atol=1.0e-14,
                )
            ),
            "cap_hit_mismatch_count": _failure_count(
                cap_hit != (expected_requested > dynamic)
            ),
        }
        _require_zero(selector_failures, "Selector invariant failed")
        _require_zero(recap_failures, "Eta recap invariant failed")

        backtracks = read("subproblem_line_search_backtracks", int)
        if backtracks.shape != (BUDGET,):
            raise RuntimeError("Backtrack vector is not N254")
        backtrack_count = int(np.sum(backtracks))
        if backtrack_count != 0:
            raise RuntimeError("Forbidden subproblem backtracking was recorded")

        outer = read("discrete_audit_outer", int)
        step = read("discrete_audit_step", int)
        stop_reason = read("discrete_audit_stop_reason", "U64")
        final_ratio = read("discrete_audit_global_relative_residual", float)
        history_count = read("discrete_audit_history_count", int)
        if (
            outer.shape != selection.shape
            or step.shape != selection.shape
            or history_count.shape != selection.shape
            or stop_reason.ndim != 1
            or final_ratio.shape != stop_reason.shape
            or not np.array_equal(np.unique(outer), np.arange(stop_reason.size))
        ):
            raise RuntimeError("Inner-stop audit vectors are inconsistent")
        if np.any(history_count < 0) or np.any(history_count > HISTORY):
            raise RuntimeError("L-BFGS history exceeded fixed m=7")
        evaluated_step_counts = []
        for outer_index, reason in enumerate(stop_reason):
            recorded_steps = step[outer == outer_index]
            expected_steps = np.arange(recorded_steps.size)
            if not np.array_equal(recorded_steps, expected_steps):
                raise RuntimeError("Evaluated-step indices are not contiguous")
            evaluated_step_counts.append(int(recorded_steps.size))
            if recorded_steps.size > MAX_EVALUATED_STEPS:
                raise RuntimeError("Inner solve exceeded fixed kmax=10")
            if reason == "evaluated-relative-natural-residual":
                if final_ratio[outer_index] > RESIDUAL_RATIO * (1.0 + 1.0e-12):
                    raise RuntimeError("Residual stop occurred above fixed r=0.8")
            elif reason == "evaluated-max-steps":
                if recorded_steps.size != MAX_EVALUATED_STEPS:
                    raise RuntimeError("kmax stop did not occur at 10 steps")
            elif reason != "per-image-new-call-limit":
                raise RuntimeError(f"Unexpected inner stop reason: {reason}")

        stored_calls = read("stored_path_calls", int)
        total_nf = read("total_internal_joint_evaluations", int)
        counter = read("counter_vectors", int)
        cache_hits = read("cache_hit_vectors", int)
        energy_requests = read("energy_request_vectors", int)
        force_requests = read("force_request_vectors", int)
        if (
            stored_calls.shape != (BUDGET,)
            or total_nf.shape != (BUDGET,)
            or counter.shape != (BUDGET, 22)
            or any(
                array.shape != counter.shape
                for array in (cache_hits, energy_requests, force_requests)
            )
        ):
            raise RuntimeError("Physical-evaluation accounting shapes drifted")
        nf_failures = {
            "energy_force_request_identity_mismatch_count": _failure_count(
                energy_requests != force_requests
            ),
            "request_counter_cache_identity_mismatch_count": _failure_count(
                energy_requests != counter + cache_hits
            ),
            "internal_nf_sum_mismatch_count": _failure_count(
                total_nf != np.sum(counter[:, 1:-1], axis=1)
            ),
            "nonincreasing_total_nf_count": _failure_count(np.diff(total_nf) <= 0),
        }
        _require_zero(nf_failures, "Physical NF accounting invariant failed")

    if accessed.intersection(EXPECTED_BARRIER_FIELD_NAMES) or any(
        "barrier" in name.lower() for name in accessed
    ):
        raise RuntimeError("Audit unexpectedly indexed a barrier field")

    return {
        "result": str(result),
        "configuration_sha256": CONFIGURATION_SHA256,
        "eta": ETA,
        "alpha": ALPHA,
        "trajectory_record_count": BUDGET,
        "joint_selected_step_count": int(selection.size),
        "h0_selection_count": int(np.count_nonzero(selection == "H0")),
        "lbfgs_selection_count": int(np.count_nonzero(selection == "LBFGS")),
        "selector_tie_count": int(np.count_nonzero(h0_score == lbfgs_score)),
        "cap_hit_count": int(np.count_nonzero(cap_hit)),
        "base_max_atomic_displacement_min": float(np.min(base)),
        "base_max_atomic_displacement_max": float(np.max(base)),
        "dynamic_cap_min": float(np.min(dynamic)),
        "dynamic_cap_max": float(np.max(dynamic)),
        "applied_multiplier_min": float(np.min(applied_multiplier)),
        "applied_multiplier_max": float(np.max(applied_multiplier)),
        "evaluated_step_count_min": int(np.min(evaluated_step_counts)),
        "evaluated_step_count_max": int(np.max(evaluated_step_counts)),
        "residual_stop_count": int(
            np.count_nonzero(stop_reason == "evaluated-relative-natural-residual")
        ),
        "budget_interrupted_inner_stop_count": int(
            np.count_nonzero(stop_reason == "per-image-new-call-limit")
        ),
        "final_optimization_NF": int(total_nf[-1]),
        "subproblem_line_search_backtrack_count": backtrack_count,
        "shim_extra_physical_evaluations": 0,
        **selector_failures,
        **recap_failures,
        **nf_failures,
        "alpha_exactly_15_throughout": True,
        "fixed_r08_k10_m7_verified": True,
        "max_evaluated_steps_strictly_greater_than_3": True,
        "existing_ala_state_curve_per_image_reparameterization_verified": True,
        "spline_reparameterization_used": False,
        "energy_or_objective_fallback_used": False,
        "barrier_field_names_seen_by_name_not_indexed": sorted(barrier_names),
        "barrier_metric_values_accessed": False,
        "mechanism_fields_accessed": sorted(accessed),
    }


def verify_companion_accounting(root: Path, mechanism: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-check NPZ physical accounting against the pinned runner audit."""
    root = Path(root).expanduser().resolve()
    companion = _read_json((root / COMPANION_AUDIT_RELATIVE).resolve())
    member = _read_json((root / MEMBER_AUDIT_RELATIVE).resolve())
    required_companion = {
        "consistent": True,
        "discrete_all_returned_states_evaluated": True,
        "frozen_subproblem": "discrete_b040",
        "discrete_inner_max_steps": MAX_EVALUATED_STEPS,
        "discrete_stop_mode": "r08c10",
        "discrete_residual_ratio": RESIDUAL_RATIO,
        "discrete_history": HISTORY,
        "discrete_reparameterization_inside_subproblem": False,
        "reparam_mode": "state_curve",
        "state_curve_direct_objective_acceptance": "per_image",
        "state_curve_projection_uses_spline": False,
    }
    for key, expected in required_companion.items():
        if companion.get(key) != expected:
            raise RuntimeError(f"Companion accounting drifted at {key}")
    if companion.get("backend_raw_joint_evaluations") != companion.get(
        "sum_image_joint_evaluations"
    ):
        raise RuntimeError("Companion joint-evaluation sum is inconsistent")
    for key in ("per_image_request_identity_ok", "per_image_query_outcome_identity_ok"):
        values = companion.get(key)
        if not isinstance(values, list) or not values or not all(values):
            raise RuntimeError(f"Companion request identity failed at {key}")
    blocked = companion.get("per_image_budget_blocked_misses")
    if not isinstance(blocked, list) or any(int(value) != 0 for value in blocked):
        raise RuntimeError("Companion contains budget-blocked physical misses")
    if int(member.get("final_optimization_NF", -1)) != int(
        mechanism["final_optimization_NF"]
    ):
        raise RuntimeError("Member audit/NPZ NF mismatch")
    if (
        int(member.get("h0_selection_count", -1))
        != int(mechanism["h0_selection_count"])
        or int(member.get("lbfgs_selection_count", -1))
        != int(mechanism["lbfgs_selection_count"])
        or int(member.get("eta_cap_hit_count", -1))
        != int(mechanism["cap_hit_count"])
    ):
        raise RuntimeError("Member audit/NPZ selector summary mismatch")
    return {
        "runner_accounting_consistent": True,
        "backend_raw_joint_evaluations": int(
            companion["backend_raw_joint_evaluations"]
        ),
        "sum_image_joint_evaluations": int(
            companion["sum_image_joint_evaluations"]
        ),
        "per_image_budget_blocked_miss_count": 0,
        "all_request_and_query_identities_verified": True,
        "member_summary_cross_checked": True,
    }


def build_audit(root: Path, formal_report: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    provenance = verify_provenance(root, formal_report)
    mechanism = audit_mechanism(root / RESULT_RELATIVE)
    accounting = verify_companion_accounting(root, mechanism)
    row = provenance["formal_result_row"]
    if (
        int(row["final_optimization_NF"]) != mechanism["final_optimization_NF"]
        or int(row["h0_selection_count"]) != mechanism["h0_selection_count"]
        or int(row["lbfgs_selection_count"]) != mechanism["lbfgs_selection_count"]
        or int(row["eta_cap_hit_count"]) != mechanism["cap_hit_count"]
    ):
        raise RuntimeError("Formal report/mechanism summary mismatch")
    return {
        "status": "passed",
        "audit_id": AUDIT_ID,
        "created_unix": time.time(),
        "read_only_postrun_audit": True,
        "formal_report": str(Path(formal_report).expanduser().resolve()),
        "formal_report_sha256": FORMAL_REPORT_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "provenance": provenance,
        "mechanism_recomputation": mechanism,
        "physical_evaluation_accounting": accounting,
        "original_h0_lbfgs_selector_recomputed_before_eta": True,
        "post_selection_recap_recomputed_as_min_1p1_base_dynamic_cap": True,
        "base_step_never_exceeded_dynamic_cap": True,
        "applied_step_never_below_base": True,
        "fixed_alpha_15_throughout": True,
        "fixed_r08_k10_m7_throughout": True,
        "energy_or_objective_fallback_used": False,
        "extra_physical_evaluations_from_eta_shim": 0,
        "subproblem_line_search_backtrack_count": 0,
        "barrier_npz_members_indexed": [],
        "barrier_metric_values_accessed": False,
        "barrier_computation_or_ranking_performed": False,
    }


def write_exclusive_strict(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--formal-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    payload = build_audit(args.root, args.formal_report)
    write_exclusive_strict(args.output, payload)
    output = Path(args.output).expanduser().resolve()
    print(json.dumps({"output": str(output), "sha256": file_sha256(output)}))


if __name__ == "__main__":
    main()
