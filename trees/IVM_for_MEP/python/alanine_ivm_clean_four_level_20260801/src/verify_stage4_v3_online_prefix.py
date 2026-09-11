#!/usr/bin/env python3
"""Fail-closed comparison of the v3 online run and its v2 exact-prefix replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SCIENTIFIC_ARRAYS = (
    "alpha",
    "phase_labels",
    "phase_transition_macro_index",
    "phase_transition_force_offset",
    "macro_calls",
    "total_force_evaluations",
    "i_perp",
    "sparse_barrier_kcal",
    "physical_fmax",
    "segment_length_cv",
    "equidistance_relative_defect",
    "phase",
    "stored_force_evaluations",
    "stored_macro_calls",
    "stored_phase",
    "stored_paths",
    "final_path",
    "raw_iperp_rises",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(saved, key):
    return np.asarray(saved[key]).item()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    actual_path = (
        root
        / "runs/stage4_inexact_optimized_resample/"
        "stage4_uniform_v3_a20_phaseA80_phaseB220.npz"
    )
    replay_path = (
        root
        / "reports/v3_materialized_prefix_reference/"
        "stage4_uniform_v3_a20_phaseA80_phaseB220.materialized-prefix.npz"
    )
    comparisons = {}
    with np.load(actual_path, allow_pickle=False) as actual, np.load(
        replay_path, allow_pickle=False
    ) as replay:
        for key in SCIENTIFIC_ARRAYS:
            left = np.asarray(actual[key])
            right = np.asarray(replay[key])
            same_shape = left.shape == right.shape
            exact_equal = bool(same_shape and np.array_equal(left, right))
            record = {
                "shape_actual": list(left.shape),
                "shape_replay": list(right.shape),
                "exact_equal": exact_equal,
            }
            if same_shape and np.issubdtype(left.dtype, np.number):
                record["max_abs_difference"] = float(
                    np.max(np.abs(left.astype(float) - right.astype(float)))
                    if left.size
                    else 0.0
                )
            comparisons[key] = record
        metadata = {
            "policy_version": str(scalar(actual, "policy_version")),
            "alpha_mode": str(scalar(actual, "alpha_mode")),
            "common_initial_path_sha256": str(
                scalar(actual, "common_initial_path_sha256")
            ),
            "stop_complete_iperp": float(scalar(actual, "stop_complete_iperp")),
            "stop_rule_fired": bool(scalar(actual, "stop_rule_fired")),
            "stop_reason": str(scalar(actual, "stop_reason")),
            "terminal_reparam_committed": bool(
                scalar(actual, "terminal_reparam_committed")
            ),
            "materialization_mode": str(scalar(actual, "materialization_mode")),
            "independent_online_rerun_status": str(
                scalar(actual, "independent_online_rerun_status")
            ),
            "terminal_force_evaluations": int(
                np.asarray(actual["total_force_evaluations"])[-1]
            ),
            "terminal_i_perp": float(np.asarray(actual["i_perp"])[-1]),
        }
    all_exact = all(item["exact_equal"] for item in comparisons.values())
    online_rule_valid = bool(
        metadata["policy_version"] == "stage4_uniform_v3"
        and metadata["alpha_mode"] == "fixed_across_both_phases"
        and metadata["stop_rule_fired"]
        and metadata["stop_reason"]
        == "complete_reparameterized_i_perp_target"
        and metadata["terminal_reparam_committed"]
        and metadata["terminal_i_perp"] <= metadata["stop_complete_iperp"]
        and metadata["materialization_mode"] == "independent_online_execution"
        and metadata["independent_online_rerun_status"] == "confirmed"
    )
    report = {
        "actual": str(actual_path),
        "actual_sha256": file_sha256(actual_path),
        "materialized_prefix_reference": str(replay_path),
        "materialized_prefix_reference_sha256": file_sha256(replay_path),
        "scientific_arrays_exactly_equal": all_exact,
        "online_rule_valid": online_rule_valid,
        "metadata": metadata,
        "array_comparisons": comparisons,
        "status": "passed" if all_exact and online_rule_valid else "failed",
    }
    output = root / "reports/stage4_v3_online_prefix_verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
