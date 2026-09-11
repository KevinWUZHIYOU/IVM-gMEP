#!/usr/bin/env python3
"""Read-only post-run parity and mechanism audit for the selected-eta pilot.

This auditor is intentionally separate from every preregistered run source.
It never indexes an energy-barrier array.  It pins the completed campaign and
result hashes, compares every common non-barrier eta=1 field except explicitly
registered routing/identity/timing differences, and independently recomputes
the original selector and selected-step recap invariants for all four arms.
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


AUDIT_ID = (
    "stage4_ala_discrete_single_selected_eta_postrun_"
    "mechanism_parity_readonly_v1"
)
PROTOCOL_ID = "stage4_ala_discrete_single_selected_eta_pilot_v1"
FORMAL_REPORT_RELATIVE = (
    "reports/stage4_ala_discrete_single_selected_eta_pilot_v1_"
    "N254_r1_1787256460847550000.json"
)
FORMAL_REPORT_SHA256 = (
    "d5212daa7f612cb916fdc901ac66dfa8f987aaaac82822a9e27d2f884d2dae5b"
)
PREREGISTRATION_SHA256 = (
    "f8c8e6aee28dfc41172a3323b6faf75498c2fa44bfe415e9a55b25c4fa44f547"
)
OLD_ETA_ONE_RELATIVE = (
    "runs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/members/"
    "ivm_stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "r08c10_a12p5_N254_discrete_b040_r1_central_state_ivm_"
    "cfgfaf71a0dca2c_budget254.npz"
)
OLD_ETA_ONE_SHA256 = (
    "15df24a92b0a4a73e972be0574b5979d2e4eb7e7b980bc76cd58d474f9d8e835"
)
NEW_RESULT_RELATIVE = {
    1.0: (
        "runs/stage4_ala_discrete_single_selected_eta_pilot_v1/members/"
        "ivm_stage4_ala_discrete_single_selected_eta_pilot_v1_eta1_"
        "r08c10_a12p5_N254_discrete_b040_r1_central_state_ivm_"
        "cfgef60d94a19a7_budget254.npz"
    ),
    1.25: (
        "runs/stage4_ala_discrete_single_selected_eta_pilot_v1/members/"
        "ivm_stage4_ala_discrete_single_selected_eta_pilot_v1_eta1p25_"
        "r08c10_a12p5_N254_discrete_b040_r1_central_state_ivm_"
        "cfgcf29f46d660f_budget254.npz"
    ),
    1.5: (
        "runs/stage4_ala_discrete_single_selected_eta_pilot_v1/members/"
        "ivm_stage4_ala_discrete_single_selected_eta_pilot_v1_eta1p5_"
        "r08c10_a12p5_N254_discrete_b040_r1_central_state_ivm_"
        "cfgf9c224fc8a23_budget254.npz"
    ),
    2.0: (
        "runs/stage4_ala_discrete_single_selected_eta_pilot_v1/members/"
        "ivm_stage4_ala_discrete_single_selected_eta_pilot_v1_eta2_"
        "r08c10_a12p5_N254_discrete_b040_r1_central_state_ivm_"
        "cfgcbff96fe1106_budget254.npz"
    ),
}
NEW_RESULT_SHA256 = {
    1.0: "64de80fb5cae1f5a8ca299bd3a7f23405ccd1186dc3219f756d812fbb1b54df9",
    1.25: "c839e8db95472f872a4750b783c3e78ee98d0bfe655678995fe17b7aeda59344",
    1.5: "8cc8fde35b0ec808aaaff64661e059ebbd481e7c2a752bd9348a788175a79869",
    2.0: "de6a9843d5460bb9a10fcb874bc5690b18c80f9f2f192eb059f2d21ea558b8b8",
}
ETA_VALUES = (1.0, 1.25, 1.5, 2.0)

# These common fields are expected to differ without changing the trajectory.
PARITY_IDENTITY_EXCLUSIONS = frozenset(
    {
        "run_label",
        "configuration_sha256",
        "discrete_implementation_id",
        "wall_seconds",
    }
)
# Explicit names prevent a newly introduced barrier field from being silently
# generalized away.  Names are inspected; values are never indexed.
PARITY_BARRIER_EXCLUSIONS = frozenset(
    {
        "optimization_barrier_is_sparse",
        "barrier_metric_for_figures",
        "barrier_kcal",
    }
)
EXPECTED_ETA_ONLY_FIELDS = frozenset(
    {
        "discrete_step_multiplier_aborted_selection_count",
        "discrete_step_multiplier_application_stage",
        "discrete_step_multiplier_audit_applied",
        "discrete_step_multiplier_audit_applied_max_atomic_displacement",
        "discrete_step_multiplier_audit_base_max_atomic_displacement",
        "discrete_step_multiplier_audit_cap_hit",
        "discrete_step_multiplier_audit_dynamic_max_step",
        "discrete_step_multiplier_audit_requested",
        "discrete_step_multiplier_audit_requested_max_atomic_displacement",
        "discrete_step_multiplier_audit_selection",
        "discrete_step_multiplier_base_implementation_id",
        "discrete_step_multiplier_dynamic_cap_reapplied",
        "discrete_step_multiplier_energy_backtracking",
        "discrete_step_multiplier_eta",
        "discrete_step_multiplier_extra_physical_evaluations",
        "discrete_step_multiplier_implementation_id",
        "discrete_step_multiplier_policy_id",
        "discrete_step_multiplier_selector_isolated",
        "discrete_step_multiplier_uncommitted_selection_count",
    }
)
MECHANISM_FIELDS = frozenset(
    {
        "configuration_sha256",
        "discrete_audit_selection",
        "discrete_audit_h0_capped_predicted_decrease",
        "discrete_audit_lbfgs_capped_predicted_decrease",
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
    }
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pinned(path: Path, expected: str, label: str) -> str:
    path = Path(path).resolve()
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"Pinned {label} hash drifted: {path}")
    return actual


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.dtype.kind != right.dtype.kind:
        return False
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def compare_eta_one_safe_fields(
    old_result: Path, new_result: Path
) -> dict[str, Any]:
    """Compare all common safe fields without indexing a barrier value."""
    accessed: set[str] = set()
    with np.load(old_result, allow_pickle=False) as old, np.load(
        new_result, allow_pickle=False
    ) as new:
        old_fields = set(old.files)
        new_fields = set(new.files)
        common = old_fields & new_fields
        barrier_named = {key for key in common if "barrier" in key.lower()}
        if barrier_named != set(PARITY_BARRIER_EXCLUSIONS):
            raise RuntimeError(
                "Unexpected common barrier-field namespace: "
                + ", ".join(sorted(barrier_named))
            )
        exclusions = set(PARITY_IDENTITY_EXCLUSIONS) | set(
            PARITY_BARRIER_EXCLUSIONS
        )
        missing_exclusions = exclusions.difference(common)
        if missing_exclusions:
            raise RuntimeError(
                "Eta=1 parity exclusions missing from common fields: "
                + ", ".join(sorted(missing_exclusions))
            )
        new_only = new_fields - old_fields
        old_only = old_fields - new_fields
        if new_only != set(EXPECTED_ETA_ONLY_FIELDS) or old_only:
            raise RuntimeError("Eta=1 result schemas differ outside eta metadata")
        safe_fields = sorted(common - exclusions)
        differing = []
        for key in safe_fields:
            accessed.add(key)
            if not _array_equal(np.asarray(old[key]), np.asarray(new[key])):
                differing.append(key)
        if differing:
            raise RuntimeError(
                "Eta=1 safe-field parity failed: " + ", ".join(differing)
            )
    if accessed.intersection(PARITY_BARRIER_EXCLUSIONS):
        raise RuntimeError("Eta=1 parity audit indexed a barrier field")
    return {
        "common_field_count": len(common),
        "safe_common_field_count": len(safe_fields),
        "safe_common_fields_compared": safe_fields,
        "safe_common_difference_count": len(differing),
        "expected_identity_timing_exclusions": sorted(
            PARITY_IDENTITY_EXCLUSIONS
        ),
        "barrier_fields_seen_by_name_not_indexed": sorted(
            PARITY_BARRIER_EXCLUSIONS
        ),
        "eta_only_new_fields": sorted(new_only),
        "barrier_values_accessed": False,
    }


def _scalar(saved: np.lib.npyio.NpzFile, key: str) -> Any:
    return np.asarray(saved[key]).item()


def _failure_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)))


def audit_mechanism_arm(result: Path, registered_eta: float) -> dict[str, Any]:
    """Independently recompute selector and post-select recap invariants."""
    eta = float(registered_eta)
    accessed: set[str] = set()
    with np.load(result, allow_pickle=False) as saved:
        missing = sorted(MECHANISM_FIELDS.difference(saved.files))
        if missing:
            raise RuntimeError("Mechanism result missing: " + ", ".join(missing))
        if any(key in PARITY_BARRIER_EXCLUSIONS for key in MECHANISM_FIELDS):
            raise RuntimeError("Mechanism allowlist contains a barrier field")

        def read(key: str, dtype: Any = None) -> np.ndarray:
            accessed.add(key)
            value = np.asarray(saved[key])
            return value if dtype is None else np.asarray(value, dtype=dtype)

        recorded_eta = float(read("discrete_step_multiplier_eta").item())
        if recorded_eta != eta:
            raise RuntimeError("Mechanism arm eta differs from registration")
        for key, expected in (
            ("discrete_step_multiplier_selector_isolated", True),
            ("discrete_step_multiplier_dynamic_cap_reapplied", True),
            ("discrete_step_multiplier_energy_backtracking", False),
        ):
            if bool(read(key).item()) is not expected:
                raise RuntimeError(f"Mechanism metadata drifted at {key}")
        if int(read("discrete_step_multiplier_extra_physical_evaluations").item()):
            raise RuntimeError("Eta shim reports extra physical evaluations")
        if int(read("discrete_step_multiplier_uncommitted_selection_count").item()):
            raise RuntimeError("Eta arm contains an uncommitted selector event")
        if int(read("discrete_step_multiplier_aborted_selection_count").item()):
            raise RuntimeError("Eta arm contains an aborted selector event")

        selection = read("discrete_audit_selection", "U8")
        eta_selection = read(
            "discrete_step_multiplier_audit_selection", "U8"
        )
        h0_score = read(
            "discrete_audit_h0_capped_predicted_decrease", float
        )
        lbfgs_score = read(
            "discrete_audit_lbfgs_capped_predicted_decrease", float
        )
        requested_multiplier = read(
            "discrete_step_multiplier_audit_requested", float
        )
        applied_multiplier = read(
            "discrete_step_multiplier_audit_applied", float
        )
        cap_hit = read("discrete_step_multiplier_audit_cap_hit", bool)
        dynamic = read(
            "discrete_step_multiplier_audit_dynamic_max_step", float
        )
        base = read(
            "discrete_step_multiplier_audit_base_max_atomic_displacement",
            float,
        )
        requested = read(
            "discrete_step_multiplier_audit_requested_max_atomic_displacement",
            float,
        )
        applied = read(
            "discrete_step_multiplier_audit_applied_max_atomic_displacement",
            float,
        )
        vectors = (
            selection,
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
        if selection.ndim != 1 or any(
            vector.shape != selection.shape for vector in vectors
        ):
            raise RuntimeError("Mechanism vectors do not align")
        if not selection.size:
            raise RuntimeError("Mechanism arm has no selected steps")
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
        if any(not np.all(np.isfinite(vector)) for vector in numeric):
            raise RuntimeError("Mechanism vectors contain non-finite values")

        expected_selection = np.where(h0_score >= lbfgs_score, "H0", "LBFGS")
        selector_mismatch = selection != expected_selection
        eta_label_mismatch = selection != eta_selection
        requested_multiplier_mismatch = requested_multiplier != eta
        base_over_dynamic = base > dynamic * (1.0 + 1.0e-12)
        expected_requested = eta * base
        requested_mismatch = ~np.isclose(
            requested, expected_requested, rtol=1.0e-13, atol=1.0e-14
        )
        applied_below_base = applied < base * (1.0 - 1.0e-12)
        expected_applied = np.minimum(expected_requested, dynamic)
        recap_mismatch = ~np.isclose(
            applied, expected_applied, rtol=1.0e-12, atol=1.0e-14
        )
        multiplier_mismatch = ~np.isclose(
            applied_multiplier,
            applied / base,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        cap_hit_mismatch = cap_hit != (expected_requested > dynamic)
        failures = {
            "selector_winner_mismatch_count": _failure_count(selector_mismatch),
            "eta_selection_label_mismatch_count": _failure_count(
                eta_label_mismatch
            ),
            "requested_multiplier_mismatch_count": _failure_count(
                requested_multiplier_mismatch
            ),
            "base_exceeds_dynamic_cap_count": _failure_count(base_over_dynamic),
            "requested_eta_times_base_mismatch_count": _failure_count(
                requested_mismatch
            ),
            "applied_below_base_count": _failure_count(applied_below_base),
            "applied_recap_target_mismatch_count": _failure_count(
                recap_mismatch
            ),
            "applied_multiplier_mismatch_count": _failure_count(
                multiplier_mismatch
            ),
            "cap_hit_mismatch_count": _failure_count(cap_hit_mismatch),
        }
        if any(failures.values()):
            raise RuntimeError(
                "Mechanism invariant failed: "
                + json.dumps(failures, sort_keys=True)
            )
        configuration = str(read("configuration_sha256").item())
    if accessed.intersection(PARITY_BARRIER_EXCLUSIONS):
        raise RuntimeError("Mechanism audit indexed a barrier field")
    return {
        "eta": eta,
        "configuration_sha256": configuration,
        "joint_selected_step_count": int(selection.size),
        "h0_selection_count": int(np.count_nonzero(selection == "H0")),
        "lbfgs_selection_count": int(np.count_nonzero(selection == "LBFGS")),
        "cap_hit_count": int(np.count_nonzero(cap_hit)),
        "applied_multiplier_min": float(np.min(applied_multiplier)),
        "applied_multiplier_max": float(np.max(applied_multiplier)),
        **failures,
        "mechanism_fields_accessed": sorted(accessed),
        "barrier_values_accessed": False,
    }


def build_audit(root: Path, formal_report: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    formal_report = Path(formal_report).expanduser().resolve()
    expected_report = (root / FORMAL_REPORT_RELATIVE).resolve()
    if formal_report != expected_report:
        raise RuntimeError("Post-run audit received an unregistered formal report")
    formal_hash = _pinned(
        formal_report, FORMAL_REPORT_SHA256, "completed eta formal report"
    )
    report = _read_json(formal_report)
    if report.get("status") != "complete" or report.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Eta formal report is not complete or has wrong protocol")
    if report.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise RuntimeError("Eta formal report points to another preregistration")
    if report.get("barrier_fields_accessed_by_launcher") is not False:
        raise RuntimeError("Eta formal launcher accessed a barrier field")
    if report.get("formal_ranking_deferred_to_separate_readonly_iperp_ctp") is not True:
        raise RuntimeError("Eta formal report did not defer scientific ranking")

    old_path = (root / OLD_ETA_ONE_RELATIVE).resolve()
    old_hash = _pinned(old_path, OLD_ETA_ONE_SHA256, "old eta=1 oracle")
    results = report.get("results")
    if not isinstance(results, list) or len(results) != len(ETA_VALUES):
        raise RuntimeError("Eta formal report does not contain four arms")
    rows = {float(row["eta"]): row for row in results}
    if set(rows) != set(ETA_VALUES):
        raise RuntimeError("Eta formal report grid differs")

    result_records: dict[str, Any] = {}
    mechanism_records = []
    for eta in ETA_VALUES:
        row = rows[eta]
        if int(row.get("returncode", -1)) != 0:
            raise RuntimeError(f"Eta={eta:g} formal arm did not complete")
        result_path = (root / NEW_RESULT_RELATIVE[eta]).resolve()
        if Path(str(row["result"])).resolve() != result_path:
            raise RuntimeError(f"Eta={eta:g} formal result path drifted")
        result_hash = _pinned(
            result_path, NEW_RESULT_SHA256[eta], f"eta={eta:g} result"
        )
        if str(row["result_sha256"]) != result_hash:
            raise RuntimeError(f"Eta={eta:g} report/result hash mismatch")
        result_records[f"eta={eta:g}"] = {
            "path": str(result_path),
            "sha256": result_hash,
        }
        mechanism_records.append(audit_mechanism_arm(result_path, eta))

    new_eta_one_path = (root / NEW_RESULT_RELATIVE[1.0]).resolve()
    parity = compare_eta_one_safe_fields(old_path, new_eta_one_path)
    if parity["safe_common_difference_count"] != 0:
        raise RuntimeError("Eta=1 post-run parity is not exact")
    configurations = {
        record["configuration_sha256"] for record in mechanism_records
    }
    if len(configurations) != len(ETA_VALUES):
        raise RuntimeError("Eta arm configuration hashes are not unique")
    if any(record["barrier_values_accessed"] for record in mechanism_records):
        raise RuntimeError("Mechanism audit accessed a barrier value")

    return {
        "status": "passed",
        "audit_id": AUDIT_ID,
        "created_unix": time.time(),
        "read_only": True,
        "formal_report": str(formal_report),
        "formal_report_sha256": formal_hash,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "old_eta_one_oracle": {
            "path": str(old_path),
            "sha256": old_hash,
        },
        "new_eta_one_result": result_records["eta=1"],
        "formal_eta_results": result_records,
        "eta_one_all_common_safe_field_parity": parity,
        "mechanism_recomputation": mechanism_records,
        "all_configuration_hashes_unique": True,
        "selector_recomputed_from_original_capped_scores": True,
        "recap_recomputed_as_min_eta_times_base_dynamic_cap": True,
        "barrier_field_names_seen_only": sorted(PARITY_BARRIER_EXCLUSIONS),
        "barrier_values_accessed": False,
        "sparse_or_densified_barrier_computed": False,
        "scientific_ranking_performed": False,
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
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
    write_exclusive(args.output, payload)
    print(Path(args.output).expanduser().resolve())


if __name__ == "__main__":
    main()
