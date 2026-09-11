#!/usr/bin/env python3
"""Deduplicate and select across both preregistered sync-one grid stages."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
CAMPAIGN_ROOT = HERE.parents[1]
LEPS_ROOT = HERE.parents[2]
STAGE1 = LEPS_ROOT / "leps_irivm_ala_chig_every1_sync1_20260830"
STAGE2 = CAMPAIGN_ROOT
OUTDIR = STAGE2 / "reports/acceptance"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(root, stage):
    path = root / "reports/acceptance/candidate_summary.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row in rows:
        row["stage"] = stage
    return rows, path


def key(row):
    return float(row["alpha"]), float(row["r"]), int(row["m_max"])


def rank(row):
    return (
        row["strict_pass"] != "True",
        float(row["worst_joint_ratio"]),
        float(row["mean_joint_ratio"]),
        int(row["terminal_article_nf"]),
        float(row["alpha"]),
        float(row["r"]),
        int(row["m_max"]),
    )


def result_path(root, row):
    return root / "runs" / f"{row['run_id']}.npz"


def verify_duplicate(first, second):
    p1 = result_path(STAGE1, first)
    p2 = result_path(STAGE2, second)
    fields = (
        "article_nf",
        "outer",
        "state_phase",
        "path_history",
        "energy_history",
        "gradient_history",
        "per_image_joint_evaluations",
        "direct_trial_nf",
        "direct_trial_path_history",
    )
    with np.load(p1, allow_pickle=False) as left, np.load(p2, allow_pickle=False) as right:
        for field in fields:
            if not np.array_equal(left[field], right[field]):
                raise RuntimeError(f"overlapping grid result differs for {key(first)}: {field}")


def main():
    rows1, summary1 = read_rows(STAGE1, "initial_48")
    rows2, summary2 = read_rows(STAGE2, "refined_150")
    first = {key(row): row for row in rows1}
    second = {key(row): row for row in rows2}
    overlaps = sorted(set(first) & set(second))
    for controls in overlaps:
        verify_duplicate(first[controls], second[controls])

    # Prefer the second-stage copy for an overlapping registered combination;
    # the independently verified trajectories are exactly identical.
    combined = dict(first)
    combined.update(second)
    rows = sorted(combined.values(), key=rank)
    fields = list(rows[0])
    out_csv = OUTDIR / "combined_candidate_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    winner = rows[0]
    manifest = {
        "stage1_registered_count": len(rows1),
        "stage2_registered_count": len(rows2),
        "overlapping_parameter_combinations": len(overlaps),
        "overlap_trajectories_exactly_equal": True,
        "unique_parameter_combinations": len(rows),
        "strict_passing_combinations": sum(row["strict_pass"] == "True" for row in rows),
        "combinations_passing_19_of_21": sum(int(row["passing_checks"]) == 19 for row in rows),
        "selection_rule": "strict pass, worst joint ratio, mean joint ratio, terminal NoF, alpha, r, m_max",
        "winner": winner,
        "stage1_summary": str(summary1.relative_to(LEPS_ROOT)),
        "stage1_summary_sha256": sha256(summary1),
        "stage2_summary": str(summary2.relative_to(LEPS_ROOT)),
        "stage2_summary_sha256": sha256(summary2),
        "combined_summary": str(out_csv.relative_to(LEPS_ROOT)),
        "combined_summary_sha256": sha256(out_csv),
        "selection_source": str(HERE.relative_to(LEPS_ROOT)),
        "selection_source_sha256": sha256(HERE),
    }
    out_json = OUTDIR / "combined_selection_manifest.json"
    out_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
