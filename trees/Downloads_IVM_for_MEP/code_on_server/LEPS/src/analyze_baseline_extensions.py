#!/usr/bin/env python3
"""Audit long LEPS baseline extensions and identify correct-branch candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from leps_core import write_json


FAMILIES = {
    "static_neb": ("runs/neb", "extension1000_static_*_budget1000.npz"),
    "original_string": (
        "runs/original_string",
        "extension1000_v2dense_original_*_budget1000.npz",
    ),
    "obsm_fire": (
        "runs/obsm_fire",
        "extension1000_v2dense_fire_*_budget1000.npz",
    ),
    "obsm_lbfgs": (
        "runs/obsm_lbfgs",
        "extension1000_v2dense_lbfgs_*_budget1000.npz",
    ),
}


def first_crossing(calls, values, threshold):
    indices = np.flatnonzero(values <= threshold)
    return int(calls[indices[0]]) if len(indices) else None


def load_row(path: Path, root: Path):
    with np.load(path, allow_pickle=False) as saved:
        calls = np.asarray(saved["calls_per_image"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        error = np.asarray(saved["barrier_error"], dtype=float)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        counter_ok = bool(saved["counter_audit_consistent"])
        stop_reason = (
            str(saved["stop_reason"])
            if "stop_reason" in saved.files
            else None
        )
    audit = json.loads(path.with_suffix(".audit.json").read_text())
    stop_reason = stop_reason or audit["configuration"].get(
        "stop_reason", "unknown"
    )
    finite = bool(
        np.all(np.isfinite(i_perp))
        and np.all(np.isfinite(error))
        and np.all(np.isfinite(cv))
    )
    return {
        "result": str(path.relative_to(root)),
        "configuration": audit["configuration"],
        "stop_reason": stop_reason,
        "counter_audit_consistent": counter_ok,
        "finite": finite,
        "final_call": int(calls[-1]),
        "final_i_perp": float(i_perp[-1]),
        "final_barrier_error": float(error[-1]),
        "minimum_barrier_error": float(np.nanmin(error)),
        "final_segment_length_cv": float(cv[-1]),
        "first_barrier_error_le_0.20": first_crossing(calls, error, 0.20),
        "first_barrier_error_le_0.10": first_crossing(calls, error, 0.10),
        "first_barrier_error_le_0.02": first_crossing(calls, error, 0.02),
        "correct_branch_reached": first_crossing(calls, error, 0.20)
        is not None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    report = {
        "purpose": (
            "Post-scan extension audit. A run reaches the correct branch when "
            "its absolute dense-barrier error first falls to <= 0.20 eV."
        ),
        "families": {},
    }
    for family, (directory, pattern) in FAMILIES.items():
        rows = [
            load_row(path, root)
            for path in sorted((root / directory).glob(pattern))
        ]
        correct = [
            row
            for row in rows
            if row["correct_branch_reached"]
            and row["counter_audit_consistent"]
        ]
        correct.sort(
            key=lambda row: (
                row["first_barrier_error_le_0.20"],
                row["final_i_perp"],
                row["result"],
            )
        )
        report["families"][family] = {
            "run_count": len(rows),
            "correct_branch_count": len(correct),
            "earliest_correct_branch": correct[0] if correct else None,
            "runs": rows,
        }
    output = root / "reports/leps_baseline_extension_audit.json"
    write_json(output, report)
    print(output)
    for family, payload in report["families"].items():
        selected = payload["earliest_correct_branch"]
        crossing = (
            selected["first_barrier_error_le_0.20"] if selected else None
        )
        print(f"{family}: crossing={crossing}, run={selected and selected['result']}")


if __name__ == "__main__":
    main()
