#!/usr/bin/env python3
"""Read-only mechanism/provenance audit for the formal eta=1.1, alpha=20 run.

The formal run is immutable.  This auditor pins the preregistration, report,
result, both audits, both command records, and every registered source and
decision input.  It independently replays the H0/L-BFGS selector and the
post-selection ``min(1.1 * base_step, dynamic_cap)`` recap from saved arrays.

The invariant engine is the preregistered alpha=15 postrun auditor, used under
a temporary alpha=20 control context and restored after every call.  The
alpha=20 provenance and command checks below are separate.  Barrier members
are discovered by archive name only and are never indexed or decoded.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterator, Mapping

import numpy as np

import audit_stage4_ala_discrete_single_eta1p1_alpha15_postrun as _engine


AUDIT_ID = (
    "stage4_ala_discrete_single_eta1p1_alpha20_"
    "postrun_mechanism_provenance_readonly_v1"
)
PROTOCOL_ID = "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1"

FORMAL_REPORT_RELATIVE = (
    "reports/stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_"
    "N254_r1_1787258614120573000.json"
)
FORMAL_REPORT_SHA256 = (
    "ca041bf07ee83f53d8ce09ffd39f908a8742247ffca4be79a4a6b67f76eb9f09"
)
PREREGISTRATION_RELATIVE = (
    "reports/stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_"
    "N254_r1_1787258614120573000_preregistered.json"
)
PREREGISTRATION_SHA256 = (
    "00a724380da87017debefa8c595f811a7f0cbaa4e1468001188e66ff33f684b3"
)
RESULT_RELATIVE = (
    "runs/stage4_ala_discrete_single_eta1p1_alpha20_followup_v1/members/"
    "ivm_stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_eta1p1_"
    "r08c10_a20_N254_discrete_b040_r1_central_state_ivm_"
    "cfg72a2d169845b_budget254.npz"
)
RESULT_SHA256 = (
    "076144572af9c39f6a670f6323562eb6591e3d66d6986736b3b795cc7b4f5a45"
)
MEMBER_AUDIT_RELATIVE = (
    "runs/stage4_ala_discrete_single_eta1p1_alpha20_followup_v1/audits/"
    "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_eta1p1_"
    "r08c10_a20_N254_discrete_b040_r1.json"
)
MEMBER_AUDIT_SHA256 = (
    "ad503a2420b56327f244cb0a9a11c3b75ed0a623d05699896ed823f869a437fc"
)
COMPANION_AUDIT_RELATIVE = RESULT_RELATIVE[: -len(".npz")] + ".audit.json"
COMPANION_AUDIT_SHA256 = (
    "77641bd0a2a16d6cab4b7d6e2737b046563693241c0da63361621691690e72d3"
)
MEMBER_COMMAND_RELATIVE = (
    "logs/stage4_ala_discrete_single_eta1p1_alpha20_followup_v1/members/"
    "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_eta1p1_"
    "r08c10_a20_N254_discrete_b040_r1/command.json"
)
MEMBER_COMMAND_SHA256 = (
    "d734592e10fc7e934449ed61bd6bb0badab4b8078c32f54b63f765b9d5b58bd3"
)
LAUNCHER_COMMAND_RELATIVE = (
    "logs/stage4_ala_discrete_single_eta1p1_alpha20_followup_v1/launcher/"
    "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_eta1p1_"
    "r08c10_a20_N254_discrete_b040_r1/command.json"
)
LAUNCHER_COMMAND_SHA256 = (
    "81b3f5e751f1ee2de80257eefd8f322ae0c0a14f4ce1f50d21746003ee02d058"
)

ETA = 1.1
ALPHA = 20.0
RESIDUAL_RATIO = 0.8
MAX_EVALUATED_STEPS = 10
HISTORY = 7
BUDGET = 254
CONFIGURATION_SHA256 = (
    "72a2d169845b70641950c859b5e47130d2d6a44b097691de0b66627893b28c5a"
)
INITIAL_PATH_SHA256 = (
    "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
)
INITIAL_PATH_FILE_SHA256 = (
    "4a08066e1f0a89ad1d0ee6ecc442d7255d511217576d1ff65e9cb5e2164d891b"
)
SCIENTIFIC_COMMAND_SHA256 = (
    "8a9cdbea057c2714b2e6de2332bd1f53170d8925cc6972a8b8907095e16d25c7"
)

EXPECTED_RUN_SOURCE_KEYS = frozenset(
    {
        "followup_grid",
        "followup_member",
        "followup_tests",
        "eta_runtime_shim",
        "frozen_discrete_b040",
        "production_runner",
        "coarse_wrapper_alias",
        "coarse_wrapper_implementation",
        "legacy_command_helper",
        "ala_core",
        "alpha15_followup_member",
        "alpha15_followup_grid",
        "alpha15_formal_ctp_analyzer",
        "alpha15_formal_ctp_test",
        "alpha15_postrun_auditor",
        "alpha15_postrun_test",
    }
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
        "same_eta_alpha15_preregistration",
        "same_eta_alpha15_report",
        "same_eta_alpha15_result",
        "same_eta_alpha15_member_audit",
        "same_eta_alpha15_formal_ctp",
        "same_eta_alpha15_postrun_audit",
    }
)

file_sha256 = _engine.file_sha256
_pinned = _engine._pinned
_read_json = _engine._read_json
_compact_command_sha256 = _engine._compact_command_sha256
_exact_path = _engine._exact_path
_verify_hash_manifest = _engine._verify_hash_manifest
write_exclusive_strict = _engine.write_exclusive_strict


def _argv_value(command: list[str], flag: str) -> str:
    if command.count(flag) != 1:
        raise RuntimeError(f"Scientific command must contain {flag} exactly once")
    index = command.index(flag)
    if index + 1 >= len(command):
        raise RuntimeError(f"Scientific command has no value for {flag}")
    return command[index + 1]


def _verify_common_initial(root: Path, record: object) -> dict[str, str]:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "sha256",
        "initial_path_array_sha256",
    }:
        raise RuntimeError("Canonical initial-path record schema drifted")
    path = (root / "inputs/common_initial_path_equidistant.npz").resolve()
    _exact_path(record["path"], path, "canonical initial path")
    if file_sha256(path) != INITIAL_PATH_FILE_SHA256 or record["sha256"] != INITIAL_PATH_FILE_SHA256:
        raise RuntimeError("Canonical initial-path file hash drifted")
    with np.load(path, allow_pickle=False) as saved:
        initial = np.ascontiguousarray(np.asarray(saved["initial_path"], dtype=np.float64))
    array_hash = hashlib.sha256(initial.tobytes()).hexdigest()
    if array_hash != INITIAL_PATH_SHA256 or record["initial_path_array_sha256"] != array_hash:
        raise RuntimeError("Canonical initial-path array hash drifted")
    return {
        "path": str(path),
        "sha256": INITIAL_PATH_FILE_SHA256,
        "initial_path_array_sha256": array_hash,
    }


def _verify_scientific_command(
    root: Path,
    command: list[str],
    decision_inputs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    expected_runner = str((root / "src/run_ivm_v9_discrete_selected_eta_shim.py").resolve())
    expected_values = {
        "--root": str(root),
        "--initial-path-file": "inputs/common_initial_path_equidistant.npz",
        "--output-dir": "runs/stage4_ala_discrete_single_eta1p1_alpha20_followup_v1/members",
        "--run-label": (
            "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_eta1p1_"
            "r08c10_a20_N254_discrete_b040_r1"
        ),
        "--solver": "state_ivm",
        "--alpha": "20",
        "--alpha-mode": "fixed",
        "--max-step": "1",
        "--history": "7",
        "--budget": "254",
        "--stop-iperp": "-1",
        "--reparam-mode": "state_curve",
        "--reparam-every": "1",
        "--state-curve-direct-objective-acceptance": "per_image",
        "--stop-complete-iperp": "-1",
        "--frozen-subproblem": "discrete_b040",
        "--discrete-inner-max-steps": "10",
        "--discrete-stop-mode": "r08c10",
        "--discrete-residual-ratio": "0.8",
        "--discrete-history": "7",
        "--discrete-step-multiplier-eta": "1.1",
    }
    if len(command) < 2 or command[1] != expected_runner:
        raise RuntimeError("Scientific runner path drifted")
    for flag, expected in expected_values.items():
        if _argv_value(command, flag) != expected:
            raise RuntimeError(f"Scientific command drifted at {flag}")
    for switch in (
        "--save-all-paths",
        "--state-curve-direct-solve",
        "--state-curve-immediate-block-relock",
        "--state-curve-stall-free-polish",
    ):
        if command.count(switch) != 1:
            raise RuntimeError(f"Scientific command switch drifted at {switch}")

    alpha15_audit_path = Path(
        decision_inputs["same_eta_alpha15_member_audit"]["path"]
    ).resolve()
    alpha15_command = _read_json(alpha15_audit_path).get("command")
    if not isinstance(alpha15_command, list) or not all(
        isinstance(token, str) for token in alpha15_command
    ):
        raise RuntimeError("Pinned alpha15 control command is malformed")
    if len(alpha15_command) != len(command):
        raise RuntimeError("Alpha15/alpha20 command lengths differ")
    differences = [
        {"index": index, "alpha15": left, "alpha20": right}
        for index, (left, right) in enumerate(zip(alpha15_command, command))
        if left != right
    ]
    expected_differences = [
        {
            "index": 7,
            "alpha15": "runs/stage4_ala_discrete_single_eta1p1_alpha15_followup_v1/members",
            "alpha20": "runs/stage4_ala_discrete_single_eta1p1_alpha20_followup_v1/members",
        },
        {
            "index": 9,
            "alpha15": (
                "stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_eta1p1_"
                "r08c10_a15_N254_discrete_b040_r1"
            ),
            "alpha20": (
                "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_eta1p1_"
                "r08c10_a20_N254_discrete_b040_r1"
            ),
        },
        {"index": 13, "alpha15": "15", "alpha20": "20"},
    ]
    if differences != expected_differences:
        raise RuntimeError("Alpha20 differs from alpha15 beyond routing and alpha")
    return {
        "argv_length": len(command),
        "scientific_command_sha256": _compact_command_sha256(command),
        "fixed_controls_and_no_early_stop_verified": True,
        "alpha15_alpha20_difference_records": differences,
        "alpha_is_only_scientific_argv_difference_from_alpha15": True,
    }


def verify_provenance(root: Path, formal_report: Path) -> dict[str, Any]:
    """Pin the formal chain and recompute every registered hash/cross-link."""
    root = Path(root).expanduser().resolve()
    formal_report = Path(formal_report).expanduser().resolve()
    expected_report = (root / FORMAL_REPORT_RELATIVE).resolve()
    if formal_report != expected_report:
        raise RuntimeError("Unregistered formal report supplied to audit")

    artifacts = {
        "formal_report": _pinned(root, FORMAL_REPORT_RELATIVE, FORMAL_REPORT_SHA256, "formal report"),
        "preregistration": _pinned(root, PREREGISTRATION_RELATIVE, PREREGISTRATION_SHA256, "preregistration"),
        "result": _pinned(root, RESULT_RELATIVE, RESULT_SHA256, "result"),
        "member_audit": _pinned(root, MEMBER_AUDIT_RELATIVE, MEMBER_AUDIT_SHA256, "member audit"),
        "runner_companion_audit": _pinned(root, COMPANION_AUDIT_RELATIVE, COMPANION_AUDIT_SHA256, "runner companion audit"),
        "member_command": _pinned(root, MEMBER_COMMAND_RELATIVE, MEMBER_COMMAND_SHA256, "member command"),
        "launcher_command": _pinned(root, LAUNCHER_COMMAND_RELATIVE, LAUNCHER_COMMAND_SHA256, "launcher command"),
    }
    report = _read_json(expected_report)
    prereg_path = (root / PREREGISTRATION_RELATIVE).resolve()
    prereg = _read_json(prereg_path)
    if report.get("status") != "complete" or report.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Formal report is not the completed registered protocol")
    if prereg.get("status") != "preregistered" or prereg.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Pinned preregistration identity drifted")
    _exact_path(report.get("preregistration"), prereg_path, "preregistration")
    if report.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise RuntimeError("Formal report/preregistration hash link drifted")
    _exact_path(prereg.get("planned_report"), expected_report, "planned report")
    for key, value in prereg.items():
        if key != "status" and report.get(key) != value:
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
        "scheduled_or_adaptive_eta": False,
        "same_eta_alpha15_alpha_only_difference_verified": True,
        "same_eta_alpha15_control_in_primary_success_gate": False,
        "post_hoc_grid_expansion_allowed": False,
        "overwrite_or_force_allowed": False,
        "terminal_or_raw_iperp_ranking_allowed": False,
    }
    for key, expected in fixed.items():
        if prereg.get(key) != expected:
            raise RuntimeError(f"Preregistered fixed field drifted at {key}")
    if report.get("barrier_fields_accessed_by_launcher") is not False:
        raise RuntimeError("Formal launcher claims barrier access")
    if report.get("formal_ranking_deferred_to_separate_readonly_iperp_ctp") is not True:
        raise RuntimeError("Formal report did not defer scientific ranking")

    source_manifest = prereg.get("run_source_manifest")
    decision_manifest = prereg.get("decision_inputs")
    if report.get("run_source_manifest") != source_manifest or report.get("decision_inputs") != decision_manifest:
        raise RuntimeError("Formal report manifest differs from preregistration")
    if not isinstance(source_manifest, dict) or set(source_manifest) != EXPECTED_RUN_SOURCE_KEYS:
        raise RuntimeError("Registered run-source set drifted")
    if not isinstance(decision_manifest, dict) or set(decision_manifest) != EXPECTED_DECISION_INPUT_KEYS:
        raise RuntimeError("Registered decision-input set drifted")
    sources = _verify_hash_manifest(root, source_manifest, "run source")
    decisions = _verify_hash_manifest(root, decision_manifest, "decision input")
    common_initial = _verify_common_initial(root, prereg.get("canonical_common_initial_path"))

    results = report.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise RuntimeError("Formal report is not the registered single arm")
    row = results[0]
    if row.get("returncode") != 0 or row.get("eta") != ETA or row.get("alpha") != ALPHA:
        raise RuntimeError("Formal result row identity/completion drifted")
    result_path = (root / RESULT_RELATIVE).resolve()
    member_audit_path = (root / MEMBER_AUDIT_RELATIVE).resolve()
    _exact_path(row.get("result"), result_path, "formal result")
    _exact_path(row.get("member_audit"), member_audit_path, "member audit")
    if row.get("result_sha256") != RESULT_SHA256 or row.get("member_audit_sha256") != MEMBER_AUDIT_SHA256:
        raise RuntimeError("Formal row artifact hash link drifted")

    member_audit = _read_json(member_audit_path)
    companion = _read_json((root / COMPANION_AUDIT_RELATIVE).resolve())
    member_command_path = (root / MEMBER_COMMAND_RELATIVE).resolve()
    launcher_command_path = (root / LAUNCHER_COMMAND_RELATIVE).resolve()
    member_command = json.loads(member_command_path.read_text(encoding="utf-8"))
    launcher_command = json.loads(launcher_command_path.read_text(encoding="utf-8"))
    if not isinstance(member_command, list) or not all(isinstance(token, str) for token in member_command):
        raise RuntimeError("Member command record is not an argv vector")
    if not isinstance(launcher_command, list) or not all(isinstance(token, str) for token in launcher_command):
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
    command_contract = _verify_scientific_command(root, member_command, decisions)

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
    expected_wrapper = str(
        (root / "src/run_stage4_ala_discrete_single_eta1p1_alpha20_followup.py").resolve()
    )
    if launcher_command.count(expected_wrapper) != 1 or launcher_command.count(str(root)) != 1:
        raise RuntimeError("Launcher command provenance drifted")

    return {
        "immutable_artifacts": artifacts,
        "registered_run_sources": sources,
        "registered_decision_inputs": decisions,
        "canonical_common_initial_path": common_initial,
        "run_source_count": len(sources),
        "decision_input_count": len(decisions),
        "scientific_command_contract": command_contract,
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


@contextmanager
def _alpha20_engine_controls() -> Iterator[None]:
    replacements = {
        "RESULT_RELATIVE": RESULT_RELATIVE,
        "MEMBER_AUDIT_RELATIVE": MEMBER_AUDIT_RELATIVE,
        "COMPANION_AUDIT_RELATIVE": COMPANION_AUDIT_RELATIVE,
        "ETA": ETA,
        "ALPHA": ALPHA,
        "RESIDUAL_RATIO": RESIDUAL_RATIO,
        "MAX_EVALUATED_STEPS": MAX_EVALUATED_STEPS,
        "HISTORY": HISTORY,
        "BUDGET": BUDGET,
        "CONFIGURATION_SHA256": CONFIGURATION_SHA256,
        "INITIAL_PATH_SHA256": INITIAL_PATH_SHA256,
    }
    original = {name: getattr(_engine, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(_engine, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(_engine, name, value)


class _StopReasonAliasArchive:
    """NPZ context proxy that normalizes one documented equivalent label."""

    def __init__(self, archive: Any):
        self._archive = archive
        self.files = archive.files

    def __enter__(self) -> "_StopReasonAliasArchive":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._archive.close()

    def __getitem__(self, name: str) -> Any:
        value = self._archive[name]
        if name != "discrete_audit_stop_reason":
            return value
        normalized = np.asarray(value, dtype="U64").copy()
        normalized[normalized == "evaluated-inner-max-steps"] = "evaluated-max-steps"
        return normalized


@contextmanager
def _normalized_kmax_stop_loader(result: Path) -> Iterator[dict[str, int]]:
    """Patch only the engine load call after validating the raw label set."""
    result = Path(result).expanduser().resolve()
    original_load = np.load
    with original_load(result, allow_pickle=False) as saved:
        raw = np.asarray(saved["discrete_audit_stop_reason"], dtype="U64")
    allowed = {
        "evaluated-relative-natural-residual",
        "evaluated-inner-max-steps",
        "per-image-new-call-limit",
    }
    unknown = set(raw.tolist()).difference(allowed)
    if unknown:
        raise RuntimeError(f"Unexpected raw inner-stop labels: {sorted(unknown)}")
    counts = {
        label: int(np.count_nonzero(raw == label)) for label in sorted(allowed)
    }

    def normalized_load(path: Any, *args: Any, **kwargs: Any) -> Any:
        archive = original_load(path, *args, **kwargs)
        if Path(path).expanduser().resolve() == result:
            return _StopReasonAliasArchive(archive)
        return archive

    # _engine.np and this module's np are the same imported module; restoring
    # in finally is therefore mandatory and is tested explicitly.
    np.load = normalized_load
    try:
        yield counts
    finally:
        np.load = original_load


def audit_mechanism(result: Path) -> dict[str, Any]:
    """Recompute selector, eta recap, controls, stop rule, and NF identities."""
    with _normalized_kmax_stop_loader(result) as raw_stop_counts:
        with _alpha20_engine_controls():
            record = _engine.audit_mechanism(result)
    if record.pop("alpha_exactly_15_throughout", None) is not True:
        raise RuntimeError("Invariant engine did not verify fixed alpha")
    record["alpha_exactly_20_throughout"] = True
    record["raw_inner_stop_reason_counts"] = raw_stop_counts
    record["raw_kmax_stop_label"] = "evaluated-inner-max-steps"
    record["raw_kmax_stop_count"] = raw_stop_counts["evaluated-inner-max-steps"]
    record["kmax_label_normalized_only_for_registered_invariant_engine"] = True
    record["invariant_engine_controls_restored_after_audit"] = True
    return record


def verify_companion_accounting(root: Path, mechanism: Mapping[str, Any]) -> dict[str, Any]:
    with _alpha20_engine_controls():
        return _engine.verify_companion_accounting(root, mechanism)


def build_audit(root: Path, formal_report: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    formal_report = Path(formal_report).expanduser().resolve()
    provenance = verify_provenance(root, formal_report)
    mechanism = audit_mechanism(root / RESULT_RELATIVE)
    accounting = verify_companion_accounting(root, mechanism)
    row = provenance["formal_result_row"]
    for key, mechanism_key in (
        ("final_optimization_NF", "final_optimization_NF"),
        ("h0_selection_count", "h0_selection_count"),
        ("lbfgs_selection_count", "lbfgs_selection_count"),
        ("eta_cap_hit_count", "cap_hit_count"),
    ):
        if int(row[key]) != int(mechanism[mechanism_key]):
            raise RuntimeError("Formal report/mechanism summary mismatch")
    source = Path(__file__).resolve()
    test_source = root / "tests/test_audit_stage4_ala_discrete_single_eta1p1_alpha20_postrun.py"
    return {
        "status": "passed",
        "audit_id": AUDIT_ID,
        "created_unix": time.time(),
        "read_only_postrun_audit": True,
        "formal_report": str(formal_report),
        "formal_report_sha256": FORMAL_REPORT_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "audit_implementation": {
            "source": {"path": str(source), "sha256": file_sha256(source)},
            "test": {"path": str(test_source.resolve()), "sha256": file_sha256(test_source)},
            "registered_invariant_engine": provenance["registered_run_sources"][
                "alpha15_postrun_auditor"
            ],
        },
        "provenance": provenance,
        "mechanism_recomputation": mechanism,
        "physical_evaluation_accounting": accounting,
        "original_h0_lbfgs_selector_recomputed_before_eta": True,
        "post_selection_recap_recomputed_as_min_1p1_base_dynamic_cap": True,
        "base_step_never_exceeded_dynamic_cap": True,
        "applied_step_never_below_base": True,
        "fixed_alpha_20_throughout": True,
        "fixed_r08_k10_m7_throughout": True,
        "energy_or_objective_fallback_used": False,
        "extra_physical_evaluations_from_eta_shim": 0,
        "subproblem_line_search_backtrack_count": 0,
        "barrier_npz_members_indexed": [],
        "barrier_metric_values_accessed": False,
        "barrier_computation_or_ranking_performed": False,
    }


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
