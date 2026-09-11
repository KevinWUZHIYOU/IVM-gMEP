#!/usr/bin/env python3
"""Rank focused LEPS baseline scans with one preregistered whole-curve score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from leps_core import write_json


FAMILIES = {
    "static_neb": ("runs/neb", "refine_static_*_budget300.npz"),
    "original_string": (
        "runs/original_string",
        "refine_v2dense_original_*_budget300.npz",
    ),
    "obsm_fire": (
        "runs/obsm_fire",
        "refine_v2dense_obsm_fire_*_budget300.npz",
    ),
    "obsm_lbfgs": (
        "runs/obsm_lbfgs",
        "refine_v2dense_obsm_lbfgs_*_budget300.npz",
    ),
}


def step_log_auc(calls, values, budget, floor):
    grid = np.arange(1, int(budget) + 1)
    indices = np.searchsorted(calls, grid, side="right") - 1
    sampled = values[np.clip(indices, 0, len(values) - 1)]
    return float(np.mean(np.log10(np.maximum(sampled, floor))))


def load_row(path: Path, root: Path, budget: int):
    with np.load(path, allow_pickle=False) as saved:
        calls = np.asarray(saved["calls_per_image"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        barrier_error = np.asarray(saved["barrier_error"], dtype=float)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        i_arc = np.asarray(saved["i_arc"], dtype=float)
        counter_ok = bool(saved["counter_audit_consistent"])
    audit_path = path.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text())
    finite = bool(
        np.all(np.isfinite(i_perp))
        and np.all(np.isfinite(barrier_error))
        and np.all(np.isfinite(cv))
        and np.all(np.isfinite(i_arc))
    )
    score_i = step_log_auc(calls, i_perp, budget, 1e-14) if finite else np.inf
    score_b = (
        step_log_auc(calls, barrier_error, budget, 1e-8)
        if finite
        else np.inf
    )
    return {
        "result": str(path.relative_to(root)),
        "configuration": audit["configuration"],
        "finite": finite,
        "counter_audit_consistent": counter_ok,
        "score_i_perp": score_i,
        "score_barrier_error": score_b,
        "score_total": score_i + score_b,
        "final_call": int(calls[-1]),
        "final_i_perp": float(i_perp[-1]),
        "final_barrier_error": float(barrier_error[-1]),
        "final_segment_length_cv": float(cv[-1]),
        "final_i_arc": float(i_arc[-1]),
        "spacing_feasible": bool(finite and cv[-1] <= 0.10),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=300)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    report = {
        "selection_contract": {
            "budget": args.budget,
            "score": (
                "mean log10 step-held I_perp + mean log10 step-held absolute "
                "dense-barrier error over calls 1..budget"
            ),
            "floors": {"i_perp": 1e-14, "barrier_error": 1e-8},
            "eligibility": (
                "finite complete trace, consistent counter audit, final "
                "segment-length CV <= 0.10"
            ),
            "tie_break": (
                "lower total score, then final I_perp, final barrier error, "
                "and lexical result path"
            ),
        },
        "families": {},
    }
    for family, (directory, pattern) in FAMILIES.items():
        rows = [
            load_row(path, root, args.budget)
            for path in sorted((root / directory).glob(pattern))
        ]
        eligible = [
            row
            for row in rows
            if row["finite"]
            and row["counter_audit_consistent"]
            and row["spacing_feasible"]
        ]
        eligible.sort(
            key=lambda row: (
                row["score_total"],
                row["final_i_perp"],
                row["final_barrier_error"],
                row["result"],
            )
        )
        report["families"][family] = {
            "run_count": len(rows),
            "eligible_count": len(eligible),
            "selected": eligible[0] if eligible else None,
            "top10": eligible[:10],
            "ineligible_count": len(rows) - len(eligible),
        }
    output = root / "reports/leps_baseline_refinement_selection.json"
    write_json(output, report)
    print(output)
    for family, payload in report["families"].items():
        selected = payload["selected"]
        print(f"{family}: {selected['result'] if selected else 'NO ELIGIBLE RUN'}")


if __name__ == "__main__":
    main()
