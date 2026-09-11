#!/usr/bin/env python3
"""Verify that a shorter single-thread IR-IVM run is an exact complete-state prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXCLUDED_RECORD_ARRAYS = {"wall_seconds"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(saved: np.lib.npyio.NpzFile, key: str):
    return np.asarray(saved[key]).item()


def equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype.kind in "fc" and right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def compare(short_path: Path, long_path: Path) -> dict:
    short_path = short_path.expanduser().resolve()
    long_path = long_path.expanduser().resolve()
    with np.load(short_path, allow_pickle=False) as short, np.load(
        long_path, allow_pickle=False
    ) as long:
        for key in (
            "counter_vectors",
            "stored_paths",
            "reparam_committed",
            "total_internal_joint_evaluations",
        ):
            if key not in short.files or key not in long.files:
                raise ValueError(f"missing required prefix array: {key}")
        committed = np.asarray(short["reparam_committed"], dtype=bool)
        complete = committed.copy()
        complete[0] = True
        last_complete_index = int(np.flatnonzero(complete)[-1])
        prefix_length = last_complete_index + 1
        short_records = len(short["total_internal_joint_evaluations"])
        long_records = len(long["total_internal_joint_evaluations"])
        if long_records < prefix_length:
            raise ValueError("long run ends before the short run's last complete state")

        compared = []
        mismatches = []
        for key in sorted(set(short.files).intersection(long.files)):
            if key in EXCLUDED_RECORD_ARRAYS:
                continue
            left = np.asarray(short[key])
            right = np.asarray(long[key])
            if (
                left.ndim < 1
                or right.ndim < 1
                or left.shape[0] != short_records
                or right.shape[0] != long_records
                or left.shape[1:] != right.shape[1:]
            ):
                continue
            compared.append(key)
            if not equal(left[:prefix_length], right[:prefix_length]):
                mismatches.append(key)

        short_counter = np.asarray(short["counter_vectors"], dtype=np.int64)
        last_complete_total_nf = int(short_counter[last_complete_index].sum())
        initial_paths_equal = equal(
            np.asarray(short["initial_path"]), np.asarray(long["initial_path"])
        )
        algorithm_controls_equal = all(
            scalar(short, key) == scalar(long, key)
            for key in (
                "alpha",
                "discrete_residual_ratio",
                "discrete_inner_max_steps",
                "discrete_step_multiplier_eta",
                "max_step",
                "reparam_mode",
                "reparam_every",
                "state_curve_direct_objective_acceptance",
                "frozen_subproblem",
            )
        )
        exact = not mismatches and initial_paths_equal and algorithm_controls_equal
        return {
            "status": "passed" if exact else "failed",
            "short_result": str(short_path),
            "short_result_sha256": sha256(short_path),
            "long_result": str(long_path),
            "long_result_sha256": sha256(long_path),
            "comparison_scope": "all record-aligned arrays through the shorter run's last reparam_committed state",
            "short_record_count": short_records,
            "long_record_count": long_records,
            "last_complete_record_index": last_complete_index,
            "last_complete_total_nf": last_complete_total_nf,
            "compared_array_count": len(compared),
            "compared_arrays": compared,
            "excluded_non_scientific_record_arrays": sorted(EXCLUDED_RECORD_ARRAYS),
            "mismatched_arrays": mismatches,
            "initial_paths_bitwise_equal": initial_paths_equal,
            "algorithm_controls_equal": algorithm_controls_equal,
            "exact_complete_state_prefix": exact,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", type=Path, required=True)
    parser.add_argument("--long", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.short, args.long)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
