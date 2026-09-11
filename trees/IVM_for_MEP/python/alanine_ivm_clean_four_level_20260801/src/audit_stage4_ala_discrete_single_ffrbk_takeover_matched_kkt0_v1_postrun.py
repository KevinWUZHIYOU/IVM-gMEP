#!/usr/bin/env python3
"""Independent barrier-blind post-run audit for the FFrbk takeover campaign.

This module is deliberately read-only.  It pins the formal report and
preregistration, verifies every registered source and prior-artifact hash,
pins both result/audit/command chains, and independently recomputes the saved
selector/latch truth tables and post-selection eta recap.

Physical energy and barrier arrays are never indexed.  Whole-file SHA256
verification necessarily hashes the bytes of an NPZ, but no physical energy
or barrier member is loaded, decoded, summarized, or ranked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


AUDIT_ID = (
    "stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_"
    "postrun_mechanism_provenance_readonly_v1"
)
PROTOCOL_ID = "stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1"
FORMAL_REPORT_RELATIVE = (
    "reports/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_"
    "N254_r1_1787262084320785000.json"
)
FORMAL_REPORT_SHA256 = (
    "c64c471d612a22161ff1308bd969f83588b5de4a2aa67e62b4e4b848e96514ad"
)
PREREGISTRATION_RELATIVE = (
    "reports/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_"
    "N254_r1_1787262084320785000_preregistered.json"
)
PREREGISTRATION_SHA256 = (
    "67dd26b93f216a2d7f429a1b31710bdf1b933c24e10f17779e2499621705fa4e"
)
SIDECAR_RELATIVE = (
    "reports/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_"
    "N254_r1_1787262084320785000_postrun_mechanism_provenance_"
    "readonly_v1.json"
)

ARMS = ("one_way_takeover", "forced_operator_lbfgs_selection")
FIXED_ALPHA = 15.0
FIXED_ETA = 1.1
FIXED_RATIO = 0.8
FIXED_MAX_STEPS = 10
FIXED_HISTORY = 7
FIXED_KKT_ATOL = 0.0
FIXED_BUDGET = 254
BASE_IMPLEMENTATION_ID = "ala-frozen-discrete-ffrbk-dsel-m7-evaluated-v1"
RUNTIME_POLICY_ID = "ala-discrete-ffrbk-selector-runtime-v1"
OPERATOR_POLICY_ID = "frozen-ffrbk-h0-and-operator-initialized-lbfgs-v1"
ETA_POLICY_ID = "post-select-selected-step-eta-recap-v1"
ETA_APPLICATION_STAGE = "after-original-select-before-trial"
MODE_POLICY_IDS = {
    "one_way_takeover": (
        "one-way-strict-lbfgs-takeover-per-fixed-subproblem-v1"
    ),
    "forced_operator_lbfgs_selection": (
        "forced-operator-lbfgs-selection-with-h0-shadow-v1"
    ),
}
CONFIGURATION_SHA256 = {
    "one_way_takeover": (
        "1804ab58ab7551de2573320a04f07bb369872f5d333d50610d49d4ff7c200df6"
    ),
    "forced_operator_lbfgs_selection": (
        "dbf51ea22ca44f4e7aa5d71e91209d1ae0f83e8b779b97cb9cfcdc425b2bb6ca"
    ),
}
INITIAL_PATH_SHA256 = (
    "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
)

RESULT_RELATIVE = {
    "one_way_takeover": (
        "runs/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1/"
        "members/ivm_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_"
        "v1_one_way_takeover_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1_"
        "central_state_ivm_cfg1804ab58ab75_budget254.npz"
    ),
    "forced_operator_lbfgs_selection": (
        "runs/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1/"
        "members/ivm_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_"
        "v1_forced_operator_lbfgs_selection_eta1p1_r08c10_kkt0_a15_N254_"
        "discrete_b040_r1_central_state_ivm_cfgdbf51ea22ca4_budget254.npz"
    ),
}
RESULT_SHA256 = {
    "one_way_takeover": (
        "1c81ad6fe876a0eb94f2fc55d5d2567737face95ff52b6fe1bdfffc536d16787"
    ),
    "forced_operator_lbfgs_selection": (
        "5d8169d1d16dab2836bef43affcaaec2dabf539a8e0f992c9050cd909faf4050"
    ),
}
MEMBER_AUDIT_RELATIVE = {
    arm: (
        "runs/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1/"
        "audits/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_"
        + arm
        + "_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1.json"
    )
    for arm in ARMS
}
MEMBER_AUDIT_SHA256 = {
    "one_way_takeover": (
        "bd1caa6b744ede59e95b849eef8bf92a80ca321e849754c28d5f726e1147969f"
    ),
    "forced_operator_lbfgs_selection": (
        "c9ae9d885a5b32dc07c5b45f5446d5511fefe79c8f75be24608c74d2ff232912"
    ),
}
COMPANION_AUDIT_SHA256 = {
    "one_way_takeover": (
        "d71bf5d14e7f2d070a3da4e70e85c9276bceb43920a9598de81b789ff2b63534"
    ),
    "forced_operator_lbfgs_selection": (
        "600518cea61b3cca033817ac55c92c98846e6a1a8cc01d93d7d54e035fd59323"
    ),
}
MEMBER_COMMAND_SHA256 = {
    "one_way_takeover": (
        "a970b975338c14b8f39cb6b131ccb80fe44f36c7b3801ea18101cca0f54ef93d"
    ),
    "forced_operator_lbfgs_selection": (
        "85b0c1b7367bd5f72dc3306419ad2a4ddc5a105b2fe904acbb50954449b9d344"
    ),
}
LAUNCHER_COMMAND_SHA256 = {
    "one_way_takeover": (
        "955a5e78dd0cc785a8321a66c3da9d9377e992975634c27dbd7cd2840cb9a94e"
    ),
    "forced_operator_lbfgs_selection": (
        "abeb78562810cbfff22eabbda9c53e4bb700fbcea2683c3f07642be3fdffb4ab"
    ),
}

LABELS = {
    arm: (
        "stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_"
        + arm
        + "_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1"
    )
    for arm in ARMS
}

# These are physical/scientific trajectory fields, not selector identity
# fields.  They deliberately exclude configuration/run labels, mechanism
# labels, wall time, and every physical energy or barrier member.
SAFE_SCIENTIFIC_TRAJECTORY_FIELDS = (
    "active_alpha",
    "alpha",
    "alpha_mode",
    "i_perp",
    "initial_path_sha256",
    "physical_fmax",
    "plbfgs_kkt_atol",
    "reparam_committed",
    "reparam_mode",
    "segment_length_cv",
    "segment_length_ratio",
    "state_curve_direct_objective_acceptance",
    "stop_complete_iperp",
    "stored_path_calls",
    "stored_paths",
    "subproblem_line_search_backtracks",
    "subproblem_alpha_halvings",
    "total_internal_joint_evaluations",
    "stop_reason",
    "initial_path",
    "final_path",
    "calls_per_image",
    "counter_vectors",
    "cache_hit_vectors",
    "force_request_vectors",
    "geometric_reparam_joint_evaluations",
    "reparam_attempted",
    "reparam_accepted_fraction",
    "reparam_cv_after",
    "reparam_cv_before",
    "reparam_direct_joint_evaluations",
    "reparam_recovery_joint_evaluations",
    "state_reparam_accepts",
    "state_reparam_attempts",
    "state_reparam_recovery_joint_evaluations",
    "state_curve_projection_openmm_calls",
    "state_curve_projection_uses_spline",
    "state_curve_projection_method",
    "state_curve_projection_dtau",
    "state_curve_projection_tolerance",
    "state_curve_projection_max_iter",
    "state_curve_relock_count",
    "state_curve_stall_free_polish",
    "tangent_definition",
    "discrete_audit_outer",
    "discrete_audit_step",
    "discrete_audit_h0_raw_predicted_decrease",
    "discrete_audit_lbfgs_raw_predicted_decrease",
    "discrete_audit_h0_capped_predicted_decrease",
    "discrete_audit_lbfgs_capped_predicted_decrease",
    "discrete_audit_h0_cap_scale",
    "discrete_audit_lbfgs_cap_scale",
    "discrete_audit_history_count",
    "discrete_audit_curvature_status",
    "discrete_audit_global_relative_residual",
    "discrete_audit_stop_reason",
    "discrete_step_multiplier_audit_requested",
    "discrete_step_multiplier_audit_applied",
    "discrete_step_multiplier_audit_cap_hit",
    "discrete_step_multiplier_audit_dynamic_max_step",
    "discrete_step_multiplier_audit_base_max_atomic_displacement",
    "discrete_step_multiplier_audit_requested_max_atomic_displacement",
    "discrete_step_multiplier_audit_applied_max_atomic_displacement",
)

# These names contain "energy" only because they are scalar negative-control
# declarations.  They are not energy samples and are the only such names this
# auditor permits itself to index.
ALLOWED_NEGATIVE_CONTROL_FIELDS = frozenset(
    {
        "discrete_ffrbk_selector_energy_backtracking",
        "discrete_step_multiplier_energy_backtracking",
    }
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _compact_command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _pin(path: Path, expected: str, label: str) -> Dict[str, str]:
    path = Path(path).resolve()
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"pinned {label} hash drifted: {path}")
    return {"path": str(path), "sha256": actual}


def _exact_path(value: object, expected: Path, label: str) -> None:
    if Path(str(value)).expanduser().resolve() != Path(expected).resolve():
        raise RuntimeError(f"{label} path drifted")


def _verify_manifest(
    root: Path, manifest: object, label: str
) -> Dict[str, Dict[str, str]]:
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError(f"invalid {label} manifest")
    verified: Dict[str, Dict[str, str]] = {}
    for name, record in sorted(manifest.items()):
        if not isinstance(name, str) or not isinstance(record, dict):
            raise RuntimeError(f"invalid {label} record")
        if set(record) != {"path", "sha256"}:
            raise RuntimeError(f"unexpected {label}.{name} schema")
        path = Path(str(record["path"])).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"{label}.{name} escapes canonical root") from error
        expected = str(record["sha256"])
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"{label}.{name} hash drifted: {path}")
        verified[name] = {"path": str(path), "sha256": actual}
    return verified


def _mode_implementation_ids(arm: str) -> tuple[str, str]:
    pre_eta = "+".join(
        (
            BASE_IMPLEMENTATION_ID,
            RUNTIME_POLICY_ID,
            MODE_POLICY_IDS[arm],
            OPERATOR_POLICY_ID,
            "alpha15-m7-r08c10-kkt0",
        )
    )
    final = "+".join(
        (pre_eta, ETA_POLICY_ID, "eta-0x1d199999999999app0")
    )
    return pre_eta, final


def _option_map(command: Sequence[str]) -> Dict[str, str]:
    command = list(command)
    options: Dict[str, str] = {}
    index = 2
    while index < len(command):
        token = command[index]
        if not isinstance(token, str) or not token.startswith("--"):
            raise RuntimeError(f"unexpected command token: {token!r}")
        if token in options:
            raise RuntimeError(f"duplicate command option: {token}")
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            options[token] = command[index + 1]
            index += 2
        else:
            options[token] = ""
            index += 1
    return options


def verify_provenance(root: Path, formal_report: Path) -> Dict[str, Any]:
    """Verify the immutable formal chain without indexing NPZ data members."""
    root = Path(root).expanduser().resolve()
    formal_report = Path(formal_report).expanduser().resolve()
    expected_report = (root / FORMAL_REPORT_RELATIVE).resolve()
    prereg_path = (root / PREREGISTRATION_RELATIVE).resolve()
    if formal_report != expected_report:
        raise RuntimeError("unregistered formal report supplied")

    artifacts: Dict[str, Any] = {
        "formal_report": _pin(
            expected_report, FORMAL_REPORT_SHA256, "formal report"
        ),
        "preregistration": _pin(
            prereg_path, PREREGISTRATION_SHA256, "preregistration"
        ),
        "arms": {},
    }
    report = _read_json(expected_report)
    prereg = _read_json(prereg_path)
    if report.get("status") != "complete" or report.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("formal report identity/status drifted")
    if prereg.get("status") != "preregistered" or prereg.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("preregistration identity/status drifted")
    _exact_path(report.get("preregistration"), prereg_path, "preregistration")
    _exact_path(prereg.get("planned_report"), expected_report, "planned report")
    if report.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise RuntimeError("formal report/preregistration hash link drifted")
    for key, value in prereg.items():
        if key == "status":
            continue
        if key not in report or report[key] != value:
            raise RuntimeError(f"formal report changed preregistered field {key}")

    fixed = {
        "fixed_alpha": FIXED_ALPHA,
        "fixed_eta": FIXED_ETA,
        "fixed_residual_ratio": FIXED_RATIO,
        "fixed_max_evaluated_steps": FIXED_MAX_STEPS,
        "fixed_lbfgs_history": FIXED_HISTORY,
        "fixed_kkt_atol": FIXED_KKT_ATOL,
        "budget_per_image": FIXED_BUDGET,
        "fixed_reparameterization": "state_curve/per_image from first state",
        "energy_or_objective_backtracking_added": False,
        "physical_energy_arrays_accessed_by_wrapper_or_launcher": False,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "densified_barrier_computation_performed": False,
    }
    for key, expected in fixed.items():
        if report.get(key) != expected:
            raise RuntimeError(f"formal fixed declaration drifted at {key}")
    if report.get("cell_order") != list(ARMS):
        raise RuntimeError("formal arm order drifted")

    sources = _verify_manifest(root, prereg.get("run_source_manifest"), "source")
    prior = _verify_manifest(
        root, prereg.get("prior_artifact_manifest"), "prior artifact"
    )
    common = prereg.get("common_initial_and_topology")
    if not isinstance(common, dict) or set(common) != {
        "common_initial_path",
        "alanine_topology",
    }:
        raise RuntimeError("common initial/topology manifest drifted")
    verified_common: Dict[str, Dict[str, str]] = {}
    for name, record in sorted(common.items()):
        if not isinstance(record, dict) or not {"path", "sha256"}.issubset(record):
            raise RuntimeError(f"invalid common artifact {name}")
        path = Path(str(record["path"])).expanduser().resolve()
        actual = file_sha256(path)
        if actual != str(record["sha256"]):
            raise RuntimeError(f"common artifact hash drifted: {path}")
        verified_common[name] = {"path": str(path), "sha256": actual}
    if (
        common["common_initial_path"].get("initial_path_array_sha256")
        != INITIAL_PATH_SHA256
    ):
        raise RuntimeError("registered initial-path array identity drifted")

    registered = prereg.get("registered_commands")
    rows = report.get("results")
    if not isinstance(registered, dict) or not isinstance(rows, list):
        raise RuntimeError("formal command/result schema drifted")
    if [row.get("arm") for row in rows] != list(ARMS):
        raise RuntimeError("formal result rows drifted")

    current0_command_path = Path(
        str(prior["matched_current0_command"]["path"])
    ).resolve()
    current0_command = json.loads(current0_command_path.read_text(encoding="utf-8"))
    if not isinstance(current0_command, list):
        raise RuntimeError("matched Current0 command is not a vector")

    for arm, row in zip(ARMS, rows):
        registration = registered.get(arm)
        if not isinstance(registration, dict):
            raise RuntimeError(f"missing registered command for {arm}")
        command = registration.get("command")
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            raise RuntimeError(f"invalid registered command for {arm}")
        compact = _compact_command_sha256(command)
        if compact != registration.get("compact_sha256"):
            raise RuntimeError(f"registered compact command hash drifted for {arm}")
        if compact != row.get("registered_command_sha256"):
            raise RuntimeError(f"formal result command hash drifted for {arm}")

        label = LABELS[arm]
        member_command_path = (
            root / "logs" / PROTOCOL_ID / "members" / label / "command.json"
        ).resolve()
        launcher_command_path = (
            root / "logs" / PROTOCOL_ID / "launcher" / label / "command.json"
        ).resolve()
        artifacts["arms"][arm] = {
            "result": _pin(
                root / RESULT_RELATIVE[arm], RESULT_SHA256[arm], f"{arm} result"
            ),
            "member_audit": _pin(
                root / MEMBER_AUDIT_RELATIVE[arm],
                MEMBER_AUDIT_SHA256[arm],
                f"{arm} member audit",
            ),
            "runner_companion_audit": _pin(
                root / (RESULT_RELATIVE[arm][:-4] + ".audit.json"),
                COMPANION_AUDIT_SHA256[arm],
                f"{arm} runner companion audit",
            ),
            "member_command": _pin(
                member_command_path,
                MEMBER_COMMAND_SHA256[arm],
                f"{arm} member command",
            ),
            "launcher_command": _pin(
                launcher_command_path,
                LAUNCHER_COMMAND_SHA256[arm],
                f"{arm} launcher command",
            ),
        }
        actual_command = json.loads(member_command_path.read_text(encoding="utf-8"))
        if actual_command != command:
            raise RuntimeError(f"actual member command differs from registration: {arm}")

        expected_launcher = [
            "/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python",
            str(
                (
                    root
                    / "src/run_stage4_ala_discrete_single_ffrbk_takeover_"
                    "matched_kkt0.py"
                ).resolve()
            ),
            "--root",
            str(root),
            "--arm",
            arm,
        ]
        actual_launcher = json.loads(
            launcher_command_path.read_text(encoding="utf-8")
        )
        if actual_launcher != expected_launcher:
            raise RuntimeError(f"launcher command drifted for {arm}")

        result_path = (root / RESULT_RELATIVE[arm]).resolve()
        member_audit_path = (root / MEMBER_AUDIT_RELATIVE[arm]).resolve()
        _exact_path(row.get("result"), result_path, f"{arm} formal result")
        _exact_path(
            row.get("member_audit"), member_audit_path, f"{arm} formal member audit"
        )
        if row.get("result_sha256") != RESULT_SHA256[arm]:
            raise RuntimeError(f"formal result hash link drifted for {arm}")
        if row.get("member_audit_sha256") != MEMBER_AUDIT_SHA256[arm]:
            raise RuntimeError(f"formal audit hash link drifted for {arm}")
        if row.get("returncode") != 0 or row.get("trajectory_record_count") != 254:
            raise RuntimeError(f"formal execution was not complete for {arm}")

        member_audit = _read_json(member_audit_path)
        safe_member_expected = {
            "member_policy_version": PROTOCOL_ID,
            "arm": arm,
            "alpha": FIXED_ALPHA,
            "eta": FIXED_ETA,
            "residual_ratio": FIXED_RATIO,
            "max_evaluated_steps": FIXED_MAX_STEPS,
            "lbfgs_history": FIXED_HISTORY,
            "kkt_atol": FIXED_KKT_ATOL,
            "absolute_kkt_stop_enabled": False,
            "budget_per_image": FIXED_BUDGET,
            "reparameterization_policy": "state_curve/per_image from first state",
            "energy_or_objective_backtracking": False,
            "barrier_fields_accessed": False,
            "barrier_computation_or_densification_performed_by_wrapper": False,
            "result_sha256": RESULT_SHA256[arm],
            "scientific_command_sha256": compact,
            "matched_current0_command_sha256": (
                registration.get("matched_current0_command_sha256")
            ),
            "physical_energy_or_barrier_arrays_accessed": False,
        }
        for key, expected in safe_member_expected.items():
            if member_audit.get(key) != expected:
                raise RuntimeError(f"member audit drifted at {arm}.{key}")
        _exact_path(member_audit.get("result"), result_path, f"{arm} audit result")
        if member_audit.get("command") != command:
            raise RuntimeError(f"member audit command drifted for {arm}")

        # Prove the scientific vector is the registered Current0 clone: only
        # entry point, the two routing values, and the appended mode pair vary.
        if command[-2:] != ["--discrete-ffrbk-selector-mode", arm]:
            raise RuntimeError(f"selector mode routing drifted for {arm}")
        if len(command) != len(current0_command) + 2:
            raise RuntimeError(f"command length drifted for {arm}")
        differences = [
            index
            for index, (old, new) in enumerate(zip(current0_command, command[:-2]))
            if old != new
        ]
        allowed = sorted(
            [1]
            + [
                current0_command.index(flag) + 1
                for flag in ("--output-dir", "--run-label")
            ]
        )
        if differences != allowed:
            raise RuntimeError(f"nonregistered Current0 command drift for {arm}")
        options = _option_map(command)
        required_options = {
            "--alpha": "15",
            "--alpha-mode": "fixed",
            "--max-step": "1",
            "--history": "7",
            "--budget": "254",
            "--reparam-mode": "state_curve",
            "--state-curve-direct-objective-acceptance": "per_image",
            "--frozen-subproblem": "discrete_b040",
            "--discrete-inner-max-steps": "10",
            "--discrete-stop-mode": "r08c10",
            "--discrete-residual-ratio": "0.8",
            "--discrete-history": "7",
            "--discrete-step-multiplier-eta": "1.1",
            "--plbfgs-kkt-atol": "0",
            "--discrete-ffrbk-selector-mode": arm,
        }
        for flag, expected in required_options.items():
            if options.get(flag) != expected:
                raise RuntimeError(f"fixed command option drifted: {arm} {flag}")

    return {
        "artifacts": artifacts,
        "registered_source_count": len(sources),
        "registered_prior_artifact_count": len(prior),
        "registered_common_artifact_count": len(verified_common),
        "all_registered_source_hashes_verified": True,
        "all_registered_prior_artifact_hashes_verified": True,
        "all_result_audit_command_cross_links_verified": True,
        "scientific_commands_exact_matched_current0_clones": True,
        "only_entry_mode_and_routing_changed": True,
    }


class SafeNpzReader:
    """NPZ accessor that fails closed on physical energy/barrier members."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._saved: Any = None
        self.accessed: set[str] = set()

    def __enter__(self) -> "SafeNpzReader":
        self._saved = np.load(self.path, allow_pickle=False)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._saved is not None:
            self._saved.close()
        self._saved = None

    def read(self, key: str) -> np.ndarray:
        lowered = key.lower()
        if (
            any(token in lowered for token in ("energy", "barrier", "densif"))
            and key not in ALLOWED_NEGATIVE_CONTROL_FIELDS
        ):
            raise RuntimeError(f"forbidden physical-objective field requested: {key}")
        if self._saved is None:
            raise RuntimeError("NPZ reader is not open")
        if key not in self._saved.files:
            raise RuntimeError(f"missing result field: {key}")
        self.accessed.add(key)
        return np.asarray(self._saved[key])

    def scalar(self, key: str) -> Any:
        value = self.read(key)
        if value.ndim != 0:
            raise RuntimeError(f"expected scalar result field: {key}")
        return value.item()


def _exact_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _count_mismatch(left: np.ndarray, right: np.ndarray) -> int:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return max(int(left.size), int(right.size), 1)
    return int(np.count_nonzero(left != right))


def _require_scalar(reader: SafeNpzReader, key: str, expected: Any) -> None:
    actual = reader.scalar(key)
    if actual != expected:
        raise RuntimeError(f"result scalar drifted at {key}: {actual!r}")


def _require_zero_mismatches(record: Mapping[str, Any]) -> None:
    failures = {
        key: value
        for key, value in record.items()
        if key.endswith("_mismatch_count") and int(value) != 0
    }
    if failures:
        raise RuntimeError(f"mechanism truth-table mismatch: {failures}")


def recompute_selector_truth(arrays: Mapping[str, np.ndarray], arm: str) -> Dict[str, Any]:
    """Recompute all saved selector/latch truth bits from safe mechanism data."""
    if arm not in ARMS:
        raise ValueError("unknown selector arm")
    selection = np.asarray(arrays["selection"], dtype="U16")
    solver_call = np.asarray(arrays["solver_call"], dtype=int)
    selector_call = np.asarray(arrays["selector_call"], dtype=int)
    before = np.asarray(arrays["latched_before"], dtype=bool)
    after = np.asarray(arrays["latched_after"], dtype=bool)
    trigger = np.asarray(arrays["strict_win_trigger"], dtype=bool)
    forced = np.asarray(arrays["forced_lbfgs"], dtype=bool)
    empty_tie = np.asarray(arrays["empty_history_tie"], dtype=bool)
    failover = np.asarray(arrays["h0_failover"], dtype=bool)
    h0_valid = np.asarray(arrays["h0_valid"], dtype=bool)
    lbfgs_valid = np.asarray(arrays["lbfgs_valid"], dtype=bool)
    h0_decrease = np.asarray(arrays["h0_capped_decrease"], dtype=float)
    lbfgs_decrease = np.asarray(arrays["lbfgs_capped_decrease"], dtype=float)
    values = (
        selection,
        solver_call,
        selector_call,
        before,
        after,
        trigger,
        forced,
        empty_tie,
        failover,
        h0_valid,
        lbfgs_valid,
        h0_decrease,
        lbfgs_decrease,
    )
    if not values or any(value.shape != selection.shape for value in values):
        raise RuntimeError("selector truth arrays are misaligned")
    if not selection.size or np.any(~np.isin(selection, ["H0", "LBFGS"])):
        raise RuntimeError("selector truth contains no events or unknown labels")
    if np.any(~np.isfinite(h0_decrease)) or np.any(~np.isfinite(lbfgs_decrease)):
        raise RuntimeError("selector capped decrease is non-finite")
    if np.any(~(h0_valid | lbfgs_valid)):
        raise RuntimeError("both selector candidates were invalid")

    unique_calls = list(dict.fromkeys(solver_call.tolist()))
    if unique_calls != list(range(len(unique_calls))):
        raise RuntimeError("fixed solver call IDs are not contiguous")
    local_index_mismatches = 0
    latch_chain_mismatches = 0
    for call in unique_calls:
        mask = solver_call == call
        local = selector_call[mask]
        local_index_mismatches += _count_mismatch(local, np.arange(local.size))
        local_before = before[mask]
        local_after = after[mask]
        if local_before.size:
            latch_chain_mismatches += int(local_before[0])
            latch_chain_mismatches += _count_mismatch(
                local_before[1:], local_after[:-1]
            )

    # The frozen operator makes H0 and operator-LBFGS the same proposal at the
    # first (empty-history) selector call.  Recompute the persisted truth bit
    # using that call identity plus exact capped-decrease equality.
    expected_empty_tie = (
        (selector_call == 0)
        & h0_valid
        & lbfgs_valid
        & (h0_decrease == lbfgs_decrease)
    )

    if arm == "one_way_takeover":
        expected_trigger = (
            ~before & h0_valid & lbfgs_valid & (lbfgs_decrease > h0_decrease)
        )
        expected_after = before | expected_trigger
        expected_forced = before & lbfgs_valid
        expected_failover = before & ~lbfgs_valid & h0_valid
        expected_selection = np.empty(selection.shape, dtype="U16")
        unlatched_h0 = ~before & h0_valid & (
            ~lbfgs_valid | (h0_decrease >= lbfgs_decrease)
        )
        unlatched_lbfgs = ~before & lbfgs_valid & (
            ~h0_valid | (lbfgs_decrease > h0_decrease)
        )
        expected_selection[unlatched_h0] = "H0"
        expected_selection[unlatched_lbfgs] = "LBFGS"
        expected_selection[before & lbfgs_valid] = "LBFGS"
        expected_selection[before & ~lbfgs_valid & h0_valid] = "H0"
    else:
        expected_trigger = np.zeros(selection.shape, dtype=bool)
        expected_after = np.zeros(selection.shape, dtype=bool)
        expected_forced = lbfgs_valid.copy()
        expected_failover = ~lbfgs_valid & h0_valid
        expected_selection = np.where(lbfgs_valid, "LBFGS", "H0")
        latch_chain_mismatches += int(np.count_nonzero(before))

    record = {
        "selection_event_count": int(selection.size),
        "fixed_solver_call_count_from_events": len(unique_calls),
        "h0_selection_count": int(np.count_nonzero(selection == "H0")),
        "lbfgs_selection_count": int(np.count_nonzero(selection == "LBFGS")),
        "strict_win_trigger_count": int(np.count_nonzero(trigger)),
        "forced_lbfgs_count": int(np.count_nonzero(forced)),
        "empty_history_tie_count": int(np.count_nonzero(empty_tie)),
        "h0_failover_count": int(np.count_nonzero(failover)),
        "local_selector_index_mismatch_count": local_index_mismatches,
        "latch_chain_mismatch_count": latch_chain_mismatches,
        "empty_history_tie_truth_mismatch_count": _count_mismatch(
            empty_tie, expected_empty_tie
        ),
        "strict_win_trigger_truth_mismatch_count": _count_mismatch(
            trigger, expected_trigger
        ),
        "latched_after_truth_mismatch_count": _count_mismatch(
            after, expected_after
        ),
        "forced_lbfgs_truth_mismatch_count": _count_mismatch(
            forced, expected_forced
        ),
        "h0_failover_truth_mismatch_count": _count_mismatch(
            failover, expected_failover
        ),
        "selection_truth_mismatch_count": _count_mismatch(
            selection, expected_selection
        ),
        "selected_invalid_candidate_count": int(
            np.count_nonzero(((selection == "H0") & ~h0_valid))
            + np.count_nonzero(((selection == "LBFGS") & ~lbfgs_valid))
        ),
    }
    if record["selected_invalid_candidate_count"]:
        raise RuntimeError("selector selected an invalid candidate")
    _require_zero_mismatches(record)
    return record


def audit_arm(result: Path, arm: str) -> Dict[str, Any]:
    """Audit one result without reading a physical energy/barrier member."""
    if arm not in ARMS:
        raise ValueError("unknown selector arm")
    pre_eta_id, final_id = _mode_implementation_ids(arm)
    with SafeNpzReader(result) as reader:
        scalar_expected = {
            "configuration_sha256": CONFIGURATION_SHA256[arm],
            "initial_path_sha256": INITIAL_PATH_SHA256,
            "alpha": FIXED_ALPHA,
            "alpha_mode": "fixed",
            "budget": FIXED_BUDGET,
            "frozen_subproblem": "discrete_b040",
            "discrete_inner_max_steps": FIXED_MAX_STEPS,
            "discrete_stop_mode": "r08c10",
            "discrete_residual_ratio": FIXED_RATIO,
            "discrete_history": FIXED_HISTORY,
            "plbfgs_kkt_atol": FIXED_KKT_ATOL,
            "reparam_mode": "state_curve",
            "state_curve_direct_objective_acceptance": "per_image",
            "stop_complete_iperp": -1.0,
            "discrete_ffrbk_selector_runtime_policy_id": RUNTIME_POLICY_ID,
            "discrete_ffrbk_selector_mode": arm,
            "discrete_ffrbk_selector_mode_policy_id": MODE_POLICY_IDS[arm],
            "discrete_ffrbk_selector_operator_policy_id": OPERATOR_POLICY_ID,
            "discrete_ffrbk_selector_base_implementation_id": (
                BASE_IMPLEMENTATION_ID
            ),
            "discrete_ffrbk_selector_pre_eta_implementation_id": pre_eta_id,
            "discrete_ffrbk_selector_final_implementation_id": final_id,
            "discrete_implementation_id": final_id,
            "discrete_ffrbk_selector_alpha": FIXED_ALPHA,
            "discrete_ffrbk_selector_eta": FIXED_ETA,
            "discrete_ffrbk_selector_stop_mode": "r08c10",
            "discrete_ffrbk_selector_residual_ratio": FIXED_RATIO,
            "discrete_ffrbk_selector_inner_max_steps": FIXED_MAX_STEPS,
            "discrete_ffrbk_selector_history": FIXED_HISTORY,
            "discrete_ffrbk_selector_kkt_atol": FIXED_KKT_ATOL,
            "discrete_step_multiplier_policy_id": ETA_POLICY_ID,
            "discrete_step_multiplier_application_stage": ETA_APPLICATION_STAGE,
            "discrete_step_multiplier_base_implementation_id": pre_eta_id,
            "discrete_step_multiplier_implementation_id": final_id,
            "discrete_step_multiplier_eta": FIXED_ETA,
            "stop_reason": "evaluation_budget",
        }
        for key, expected in scalar_expected.items():
            _require_scalar(reader, key, expected)
        bool_expected = {
            "discrete_all_returned_states_evaluated": True,
            "discrete_ffrbk_selector_operator_preserved": True,
            "discrete_ffrbk_selector_selector_only_change": True,
            "discrete_ffrbk_selector_h0_shadow_constructed": True,
            "discrete_ffrbk_selector_true_lbfgs_only_construction": False,
            "discrete_ffrbk_selector_invalid_lbfgs_h0_fallback_enabled": True,
            "discrete_ffrbk_selector_one_way_latch_is_preference": (
                arm == "one_way_takeover"
            ),
            "discrete_ffrbk_selector_post_select_eta_composed": True,
            "discrete_ffrbk_selector_energy_backtracking": False,
            "discrete_ffrbk_selector_reparameterization_inside_subproblem": False,
            "discrete_step_multiplier_selector_isolated": True,
            "discrete_step_multiplier_dynamic_cap_reapplied": True,
            "discrete_step_multiplier_energy_backtracking": False,
            "state_curve_projection_uses_spline": False,
        }
        for key, expected in bool_expected.items():
            if bool(reader.scalar(key)) is not expected:
                raise RuntimeError(f"result boolean drifted at {arm}.{key}")
        zero_scalars = (
            "discrete_ffrbk_selector_extra_physical_evaluations",
            "discrete_ffrbk_selector_aborted_solver_calls",
            "discrete_ffrbk_selector_apply_h0_inverse_preconditioner_none_calls",
            "discrete_ffrbk_selector_uncommitted_selection_count",
            "discrete_ffrbk_selector_aborted_selection_count",
            "discrete_step_multiplier_extra_physical_evaluations",
            "discrete_step_multiplier_uncommitted_selection_count",
            "discrete_step_multiplier_aborted_selection_count",
            "geometric_reparam_joint_evaluations",
            "state_curve_projection_openmm_calls",
        )
        for key in zero_scalars:
            if int(reader.scalar(key)) != 0:
                raise RuntimeError(f"forbidden nonzero counter at {arm}.{key}")

        active_alpha = np.asarray(reader.read("active_alpha"), dtype=float)
        total_nf = np.asarray(
            reader.read("total_internal_joint_evaluations"), dtype=int
        )
        if active_alpha.shape != (FIXED_BUDGET,) or not np.all(
            active_alpha == FIXED_ALPHA
        ):
            raise RuntimeError(f"alpha was not fixed at 15 throughout {arm}")
        if total_nf.shape != (FIXED_BUDGET,) or np.any(np.diff(total_nf) <= 0):
            raise RuntimeError(f"optimization NF trajectory is invalid for {arm}")

        prefix = "discrete_ffrbk_selector_audit_"
        suffixes = (
            "selection",
            "solver_call",
            "selector_call",
            "latched_before",
            "latched_after",
            "strict_win_trigger",
            "forced_lbfgs",
            "empty_history_tie",
            "h0_failover",
            "h0_valid",
            "lbfgs_valid",
            "h0_capped_decrease",
            "lbfgs_capped_decrease",
        )
        arrays = {suffix: reader.read(prefix + suffix) for suffix in suffixes}
        truth = recompute_selector_truth(arrays, arm)
        selection = np.asarray(arrays["selection"], dtype="U16")
        if not np.array_equal(
            selection,
            np.asarray(reader.read("discrete_audit_selection"), dtype="U16"),
        ):
            raise RuntimeError(f"selector and frozen audit selections differ for {arm}")
        if not np.array_equal(
            selection,
            np.asarray(
                reader.read("discrete_step_multiplier_audit_selection"),
                dtype="U16",
            ),
        ):
            raise RuntimeError(f"selector and eta recap selections differ for {arm}")
        if not _exact_equal(
            arrays["h0_capped_decrease"],
            reader.read("discrete_audit_h0_capped_predicted_decrease"),
        ):
            raise RuntimeError(f"H0 capped-decrease truth drifted for {arm}")
        if not _exact_equal(
            arrays["lbfgs_capped_decrease"],
            reader.read("discrete_audit_lbfgs_capped_predicted_decrease"),
        ):
            raise RuntimeError(f"LBFGS capped-decrease truth drifted for {arm}")
        if not _exact_equal(arrays["solver_call"], reader.read("discrete_audit_outer")):
            raise RuntimeError(f"selector solver-call alignment drifted for {arm}")
        if not _exact_equal(arrays["selector_call"], reader.read("discrete_audit_step")):
            raise RuntimeError(f"selector local-call alignment drifted for {arm}")

        fixed_calls = int(reader.scalar("discrete_ffrbk_selector_fixed_solver_calls"))
        latch_resets = int(reader.scalar("discrete_ffrbk_selector_latch_reset_count"))
        committed = int(
            reader.scalar("discrete_ffrbk_selector_committed_selection_count")
        )
        apply_calls = int(
            reader.scalar("discrete_ffrbk_selector_apply_h0_inverse_calls")
        )
        joint_calls = int(
            reader.scalar("discrete_ffrbk_selector_joint_lbfgs_direction_calls")
        )
        if fixed_calls != truth["fixed_solver_call_count_from_events"]:
            raise RuntimeError(f"fixed-solver count drifted for {arm}")
        if latch_resets != fixed_calls:
            raise RuntimeError(f"latch reset count drifted for {arm}")
        if committed != selection.size:
            raise RuntimeError(f"committed selector count drifted for {arm}")
        if joint_calls != int(np.count_nonzero(arrays["selector_call"] > 0)):
            raise RuntimeError(f"operator-LBFGS call count drifted for {arm}")
        if apply_calls != int(selection.size) + joint_calls:
            raise RuntimeError(f"FFrbk H0/operator application count drifted for {arm}")
        for key, expected in (
            ("discrete_h0_selection_count", truth["h0_selection_count"]),
            ("discrete_lbfgs_selection_count", truth["lbfgs_selection_count"]),
        ):
            if int(reader.scalar(key)) != expected:
                raise RuntimeError(f"selection count scalar drifted for {arm}")

        stop_reasons = np.asarray(reader.read("discrete_audit_stop_reason"), dtype="U64")
        stop_relative = np.asarray(
            reader.read("discrete_audit_global_relative_residual"), dtype=float
        )
        if stop_reasons.shape != (fixed_calls,) or stop_relative.shape != (fixed_calls,):
            raise RuntimeError(f"fixed-stop audits are misaligned for {arm}")
        relative_count = 0
        max_step_count = 0
        solver_call = np.asarray(arrays["solver_call"], dtype=int)
        for call, reason, ratio in zip(range(fixed_calls), stop_reasons, stop_relative):
            local_count = int(np.count_nonzero(solver_call == call))
            if reason == "evaluated-relative-natural-residual":
                relative_count += 1
                if not math.isfinite(float(ratio)) or float(ratio) > FIXED_RATIO:
                    raise RuntimeError(f"relative stop truth drifted for {arm}")
            elif reason == "evaluated-inner-max-steps":
                max_step_count += 1
                if local_count != FIXED_MAX_STEPS:
                    raise RuntimeError(f"max-step stop truth drifted for {arm}")
            else:
                raise RuntimeError(f"unregistered fixed-stop reason for {arm}: {reason}")

        eta_selection = np.asarray(
            reader.read("discrete_step_multiplier_audit_selection"), dtype="U16"
        )
        requested_multiplier = np.asarray(
            reader.read("discrete_step_multiplier_audit_requested"), dtype=float
        )
        applied_multiplier = np.asarray(
            reader.read("discrete_step_multiplier_audit_applied"), dtype=float
        )
        base_displacement = np.asarray(
            reader.read(
                "discrete_step_multiplier_audit_base_max_atomic_displacement"
            ),
            dtype=float,
        )
        requested_displacement = np.asarray(
            reader.read(
                "discrete_step_multiplier_audit_requested_max_atomic_displacement"
            ),
            dtype=float,
        )
        applied_displacement = np.asarray(
            reader.read(
                "discrete_step_multiplier_audit_applied_max_atomic_displacement"
            ),
            dtype=float,
        )
        dynamic_cap = np.asarray(
            reader.read("discrete_step_multiplier_audit_dynamic_max_step"),
            dtype=float,
        )
        cap_hit = np.asarray(
            reader.read("discrete_step_multiplier_audit_cap_hit"), dtype=bool
        )
        eta_arrays = (
            eta_selection,
            requested_multiplier,
            applied_multiplier,
            base_displacement,
            requested_displacement,
            applied_displacement,
            dynamic_cap,
            cap_hit,
        )
        if any(value.shape != selection.shape for value in eta_arrays):
            raise RuntimeError(f"eta recap arrays are misaligned for {arm}")
        expected_requested = FIXED_ETA * base_displacement
        expected_applied = np.minimum(expected_requested, dynamic_cap)
        expected_applied_multiplier = expected_applied / base_displacement
        expected_cap_hit = expected_requested > dynamic_cap
        eta_record = {
            "eta_selection_alignment_mismatch_count": _count_mismatch(
                eta_selection, selection
            ),
            "eta_requested_multiplier_mismatch_count": int(
                np.count_nonzero(requested_multiplier != FIXED_ETA)
            ),
            "eta_requested_displacement_mismatch_count": int(
                np.count_nonzero(
                    ~np.isclose(
                        requested_displacement,
                        expected_requested,
                        rtol=1.0e-13,
                        atol=1.0e-14,
                    )
                )
            ),
            "eta_applied_min_cap_mismatch_count": int(
                np.count_nonzero(
                    ~np.isclose(
                        applied_displacement,
                        expected_applied,
                        rtol=1.0e-13,
                        atol=1.0e-14,
                    )
                )
            ),
            "eta_applied_multiplier_mismatch_count": int(
                np.count_nonzero(
                    ~np.isclose(
                        applied_multiplier,
                        expected_applied_multiplier,
                        rtol=1.0e-13,
                        atol=1.0e-14,
                    )
                )
            ),
            "eta_cap_hit_truth_mismatch_count": _count_mismatch(
                cap_hit, expected_cap_hit
            ),
            "eta_cap_hit_count": int(np.count_nonzero(cap_hit)),
        }
        _require_zero_mismatches(eta_record)

        line_backtracks = np.asarray(
            reader.read("subproblem_line_search_backtracks"), dtype=int
        )
        alpha_halvings = np.asarray(reader.read("subproblem_alpha_halvings"), dtype=int)
        if np.any(line_backtracks) or np.any(alpha_halvings):
            raise RuntimeError(f"backtrack/halving occurred for {arm}")

        accessed = sorted(reader.accessed)
        forbidden_accessed = [
            key
            for key in accessed
            if any(token in key.lower() for token in ("barrier", "densif"))
            or (
                "energy" in key.lower()
                and key not in ALLOWED_NEGATIVE_CONTROL_FIELDS
            )
        ]
        if forbidden_accessed:
            raise RuntimeError(f"physical objective fields were accessed: {forbidden_accessed}")

    return {
        "arm": arm,
        "result": str(Path(result).resolve()),
        "result_sha256": RESULT_SHA256[arm],
        "trajectory_record_count": int(active_alpha.size),
        "final_optimization_NF": int(total_nf[-1]),
        "alpha_exactly_15_throughout": True,
        "fixed_r08_or_k10_m7_kkt0_verified": True,
        "fixed_solver_call_count": fixed_calls,
        "relative_stop_count": relative_count,
        "k10_stop_count": max_step_count,
        "ffrbk_operator_preserved": True,
        "ffrbk_apply_h0_inverse_call_count": apply_calls,
        "operator_lbfgs_direction_call_count": joint_calls,
        "preconditioner_none_call_count": 0,
        "selector_truth": truth,
        "eta_recap": eta_record,
        "reparameterization_unchanged_state_curve_per_image": True,
        "reparameterization_inside_fixed_subproblem": False,
        "state_curve_projection_uses_spline": False,
        "state_curve_projection_openmm_calls": 0,
        "energy_or_objective_backtracking_used": False,
        "subproblem_line_search_backtrack_count": int(np.sum(line_backtracks)),
        "subproblem_alpha_halving_count": int(np.sum(alpha_halvings)),
        "extra_physical_evaluations_from_selector_or_eta": 0,
        "mechanism_fields_accessed": accessed,
        "physical_energy_or_barrier_arrays_accessed": False,
    }


def compare_safe_trajectories(
    one_way_result: Path, forced_result: Path
) -> Dict[str, Any]:
    """Exact comparison on a declared, physical-objective-free field set."""
    mismatches = []
    with SafeNpzReader(one_way_result) as one_way, SafeNpzReader(
        forced_result
    ) as forced:
        for key in SAFE_SCIENTIFIC_TRAJECTORY_FIELDS:
            if not _exact_equal(one_way.read(key), forced.read(key)):
                mismatches.append(key)
        accessed = sorted(one_way.accessed | forced.accessed)
    return {
        "comparison": "one_way_takeover_vs_forced_operator_lbfgs_selection",
        "safe_scientific_field_count": len(SAFE_SCIENTIFIC_TRAJECTORY_FIELDS),
        "safe_scientific_fields": list(SAFE_SCIENTIFIC_TRAJECTORY_FIELDS),
        "mismatch_fields": mismatches,
        "all_safe_scientific_fields_exact": not mismatches,
        "stored_coordinate_trajectory_exact": "stored_paths" not in mismatches,
        "optimization_nf_trajectory_exact": (
            "total_internal_joint_evaluations" not in mismatches
        ),
        "iperp_trajectory_exact": "i_perp" not in mismatches,
        "reparameterization_trajectory_exact": all(
            key not in mismatches
            for key in (
                "reparam_committed",
                "segment_length_cv",
                "segment_length_ratio",
                "state_reparam_attempts",
                "state_reparam_accepts",
            )
        ),
        "mechanism_selection_labels_expected_to_differ": True,
        "physical_energy_or_barrier_arrays_accessed": False,
        "fields_accessed": accessed,
    }


def build_audit(root: Path, formal_report: Path) -> Dict[str, Any]:
    root = Path(root).expanduser().resolve()
    formal_report = Path(formal_report).expanduser().resolve()
    provenance = verify_provenance(root, formal_report)
    arms = {
        arm: audit_arm(root / RESULT_RELATIVE[arm], arm) for arm in ARMS
    }
    comparison = compare_safe_trajectories(
        root / RESULT_RELATIVE["one_way_takeover"],
        root / RESULT_RELATIVE["forced_operator_lbfgs_selection"],
    )
    if not comparison["all_safe_scientific_fields_exact"]:
        raise RuntimeError("takeover arms do not have exact safe trajectories")
    all_accessed = sorted(
        set().union(
            *(set(record["mechanism_fields_accessed"]) for record in arms.values()),
            set(comparison["fields_accessed"]),
        )
    )
    return {
        "audit_id": AUDIT_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "created_unix": time.time(),
        "read_only_postrun_audit": True,
        "formal_report": str(formal_report),
        "formal_report_sha256": FORMAL_REPORT_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "p0_findings": [],
        "provenance": provenance,
        "fixed_controls": {
            "alpha": FIXED_ALPHA,
            "eta": FIXED_ETA,
            "residual_ratio": FIXED_RATIO,
            "inner_max_evaluated_steps": FIXED_MAX_STEPS,
            "lbfgs_history": FIXED_HISTORY,
            "kkt_atol": FIXED_KKT_ATOL,
            "stop_rule": "R/R0 <= 0.8 OR evaluated accepted steps == 10",
            "reparameterization": "state_curve/per_image from first state",
        },
        "arms": arms,
        "safe_scientific_trajectory_comparison": comparison,
        "trajectory_conclusion": (
            "The two selector policies produced byte-exact values for every "
            "declared safe scientific trajectory field; only selector-policy "
            "identity/selection audit fields differ."
        ),
        "energy_or_objective_backtracking_used": False,
        "extra_physical_evaluations_from_selector_or_eta": 0,
        "physical_energy_or_barrier_arrays_accessed": False,
        "densified_barrier_computation_performed": False,
        "npz_safe_fields_accessed": all_accessed,
    }


def write_exclusive_strict(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--formal-report", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    root = args.root.expanduser().resolve()
    report = (
        args.formal_report.expanduser().resolve()
        if args.formal_report is not None
        else (root / FORMAL_REPORT_RELATIVE).resolve()
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (root / SIDECAR_RELATIVE).resolve()
    )
    payload = build_audit(root, report)
    write_exclusive_strict(output, payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
