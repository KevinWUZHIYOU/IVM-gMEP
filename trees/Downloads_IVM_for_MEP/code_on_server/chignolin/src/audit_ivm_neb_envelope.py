#!/usr/bin/env python3
"""Audit one IVM trajectory against every compatible NEB trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def scalar_text(value) -> str:
    return str(np.asarray(value).item())


def final_records(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        calls = np.asarray(saved["calls_per_image"], dtype=int)
        keep = np.r_[calls[1:] != calls[:-1], True]
        return {
            "path": str(path.resolve()),
            "initial_path_sha256": scalar_text(
                saved["initial_path_sha256"]
            ),
            "calls": calls[keep],
            "i_perp": np.asarray(saved["i_perp"], dtype=float)[keep],
            "barrier_kcal": np.asarray(
                saved["barrier_kcal"], dtype=float
            )[keep],
        }


def first_passage(calls, values, threshold):
    reached = np.flatnonzero(np.minimum.accumulate(values) <= threshold)
    return int(calls[reached[0]]) if reached.size else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--ivm", required=True)
    parser.add_argument("--dense-ivm", required=True)
    parser.add_argument("--dense-neb", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    ivm = final_records(Path(args.ivm).expanduser().resolve())
    budget = int(ivm["calls"][-1])
    envelope_i = np.full(budget + 1, np.inf)
    envelope_barrier = np.full(budget + 1, np.inf)
    compatible_neb = []
    for path in sorted((root / "runs" / "neb").glob("*.npz")):
        try:
            run = final_records(path)
        except KeyError:
            continue
        if run["initial_path_sha256"] != ivm["initial_path_sha256"]:
            continue
        compatible_neb.append(run["path"])
        # ODE updates images asynchronously, so the maximum per-image
        # counter can occasionally skip an integer label.  At each budget,
        # compare with the last state available at or before that budget;
        # never fill from a future state.
        indices = np.searchsorted(
            run["calls"], np.arange(1, budget + 1), side="right"
        ) - 1
        available = indices >= 0
        budget_calls = np.arange(1, budget + 1)[available]
        run_indices = indices[available]
        envelope_i[budget_calls] = np.minimum(
            envelope_i[budget_calls], run["i_perp"][run_indices]
        )
        envelope_barrier[budget_calls] = np.minimum(
            envelope_barrier[budget_calls],
            run["barrier_kcal"][run_indices],
        )

    calls = ivm["calls"]
    if np.any(~np.isfinite(envelope_i[calls])):
        raise RuntimeError("NEB envelope does not cover every IVM call")
    i_ratio = envelope_i[calls] / ivm["i_perp"]
    barrier_ratio = envelope_barrier[calls] / ivm["barrier_kcal"]
    comparison = calls >= 2
    i_violations = calls[
        comparison & (ivm["i_perp"] > envelope_i[calls])
    ]
    barrier_violations = calls[
        comparison
        & (ivm["barrier_kcal"] > envelope_barrier[calls])
    ]

    ranges = []
    for lower, upper in ((2, 50), (51, 200), (201, budget), (2, budget)):
        selected = (calls >= lower) & (calls <= upper)
        i_index = np.flatnonzero(selected)[
            np.argmin(i_ratio[selected])
        ]
        barrier_index = np.flatnonzero(selected)[
            np.argmin(barrier_ratio[selected])
        ]
        ranges.append(
            {
                "calls": [lower, upper],
                "minimum_neb_over_ivm_i_perp_ratio": float(
                    i_ratio[i_index]
                ),
                "i_perp_worst_lead_call": int(calls[i_index]),
                "minimum_neb_over_ivm_sparse_barrier_ratio": float(
                    barrier_ratio[barrier_index]
                ),
                "barrier_worst_lead_call": int(calls[barrier_index]),
            }
        )

    with np.load(args.dense_ivm, allow_pickle=False) as saved:
        dense_ivm_calls = np.asarray(saved["calls"], dtype=int)
        dense_ivm_barrier = np.asarray(
            saved["adaptive_barrier_kcal"], dtype=float
        )
    with np.load(args.dense_neb, allow_pickle=False) as saved:
        dense_neb_calls = np.asarray(saved["calls"], dtype=int)
        dense_neb_barrier = np.asarray(
            saved["adaptive_barrier_kcal"], dtype=float
        )
    dense_rows = []
    for call, threshold in zip(dense_ivm_calls, dense_ivm_barrier):
        neb_call = first_passage(
            dense_neb_calls, dense_neb_barrier, threshold
        )
        if neb_call is None:
            continue
        ivm_call = first_passage(
            dense_ivm_calls, dense_ivm_barrier, threshold
        )
        dense_rows.append(
            {
                "threshold_kcal_per_mol": float(threshold),
                "ivm_first_passage_calls": ivm_call,
                "neb_first_passage_calls": neb_call,
                "speedup_neb_over_ivm": float(neb_call / ivm_call),
            }
        )

    payload = {
        "passed": not len(i_violations) and not len(barrier_violations),
        "ivm": ivm["path"],
        "initial_path_sha256": ivm["initial_path_sha256"],
        "compatible_neb_runs": compatible_neb,
        "comparison_calls": [2, budget],
        "i_perp_violation_calls": i_violations.tolist(),
        "sparse_barrier_violation_calls": barrier_violations.tolist(),
        "pointwise_lead_ranges": ranges,
        "dense_barrier_common_reachable_thresholds": dense_rows,
        "dense_barrier_best_observed": {
            "ivm_kcal_per_mol": float(
                np.min(dense_ivm_barrier)
            ),
            "neb_kcal_per_mol": float(
                np.min(dense_neb_barrier)
            ),
        },
    }
    output = root / "reports" / args.output
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
