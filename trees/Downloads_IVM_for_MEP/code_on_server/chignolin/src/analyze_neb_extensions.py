#!/usr/bin/env python3
"""Select the best fixed-k pure-ODE NEB after the 2000-call reruns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from chignolin_core import TANGENT_DEFINITION


FIXED_INTERFACE_ALPHA = 1e-4


def forward_fill(calls, values, grid):
    calls = np.asarray(calls, dtype=int)
    values = np.asarray(values, dtype=float)
    order = np.argsort(calls)
    calls, values = calls[order], values[order]
    indices = np.searchsorted(calls, grid, side="right") - 1
    indices = np.clip(indices, 0, len(calls) - 1)
    return values[indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    paths = sorted(
        (root / "runs" / "neb").glob(
            "neb_top3_central_*_ode_budget2000.npz"
        )
    )
    if len(paths) != 3:
        raise RuntimeError(f"Expected exactly three 2000-call reruns, found {len(paths)}")

    grid = np.arange(1, 2001, dtype=int)
    runs = []
    initial_hashes = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            optimizer = str(saved["optimizer"]).lower()
            alpha = float(saved["optimizer_alpha"])
            if optimizer != "ode" or not np.isclose(alpha, FIXED_INTERFACE_ALPHA):
                raise RuntimeError(f"Unexpected optimizer configuration: {path}")
            if str(saved["tangent_definition"]) != TANGENT_DEFINITION:
                raise RuntimeError(f"Non-central tangent result: {path}")
            initial_hashes.add(str(saved["initial_path_sha256"]))
            calls = np.asarray(saved["calls_per_image"], dtype=int)
            iperp = forward_fill(calls, saved["i_perp"], grid)
            barrier = forward_fill(calls, saved["barrier_kcal"], grid)
            fmax = forward_fill(calls, saved["physical_fmax"], grid)
            tail = grid >= 1901
            runs.append(
                {
                    "path": str(path),
                    "spring_k": float(saved["spring_k"]),
                    "optimizer": optimizer,
                    "alpha": alpha,
                    "iperp": iperp,
                    "barrier": barrier,
                    "fmax": fmax,
                    "tail_i_endpoint_change_percent": float(
                        abs(iperp[-1] - iperp[tail][0])
                        / max(iperp[tail][0], np.finfo(float).tiny)
                        * 100.0
                    ),
                    "tail_i_range_percent": float(
                        (np.max(iperp[tail]) - np.min(iperp[tail]))
                        / max(np.mean(iperp[tail]), np.finfo(float).tiny)
                        * 100.0
                    ),
                    "tail_barrier_drift_kcal": float(
                        abs(barrier[-1] - barrier[tail][0])
                    ),
                }
            )
    if len(initial_hashes) != 1:
        raise RuntimeError(f"Reruns used multiple initial paths: {initial_hashes}")

    iperp_stack = np.asarray([run["iperp"] for run in runs])
    barrier_stack = np.asarray([run["barrier"] for run in runs])
    envelope_i = np.min(iperp_stack, axis=0)
    envelope_b = np.min(barrier_stack, axis=0)
    for run in runs:
        i_ratio = np.maximum(run["iperp"] / envelope_i, 1.0)
        b_ratio = np.maximum(run["barrier"] / envelope_b, 1.0)
        run["score"] = float(
            np.mean(np.log10(i_ratio[1:]))
            + np.mean(np.log10(b_ratio[1:]))
            + 0.5 * np.log10(i_ratio[-1])
            + 0.5 * np.log10(b_ratio[-1])
        )
    runs.sort(key=lambda item: item["score"])

    rows = []
    for rank, run in enumerate(runs, start=1):
        rows.append(
            {
                "rank": rank,
                "spring_k": run["spring_k"],
                "score": run["score"],
                "i_perp_at_2000": float(run["iperp"][-1]),
                "barrier_at_2000_kcal": float(run["barrier"][-1]),
                "physical_fmax_at_2000": float(run["fmax"][-1]),
                "tail_i_endpoint_change_percent": run[
                    "tail_i_endpoint_change_percent"
                ],
                "tail_i_range_percent": run["tail_i_range_percent"],
                "tail_barrier_drift_kcal": run["tail_barrier_drift_kcal"],
                "path": run["path"],
            }
        )
    report = {
        "initial_path_sha256": next(iter(initial_hashes)),
        "selection_variables": ["spring_k"],
        "optimizer": "NEBOptimizer(method='ODE')",
        "fixed_interface_alpha": FIXED_INTERFACE_ALPHA,
        "alpha_used_by_ode": False,
        "score_definition": (
            "log-area ratios to the three-run pointwise I_perp and raw-barrier "
            "envelopes over calls 2-2000, plus half-weight final ratios"
        ),
        "selected_fixed_neb": rows[0],
        "ranking": rows,
    }
    report_path = root / "reports" / "neb_top3_2000_analysis.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    with (root / "reports" / "neb_top3_2000_ranking.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Selected k={rows[0]['spring_k']:g}: {report_path}")


if __name__ == "__main__":
    main()
