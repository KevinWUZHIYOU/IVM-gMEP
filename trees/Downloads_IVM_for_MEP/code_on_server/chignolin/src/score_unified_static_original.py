#!/usr/bin/env python3
"""Apply one absolute score to String and Static-NEB screening runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


WEIGHTS = {
    "i_perp_log_auc": 0.30,
    "barrier_log_auc": 0.25,
    "log10_final_i_perp": 0.20,
    "log10_final_barrier_kcal": 0.15,
    "log10_one_plus_final_equidistance": 0.10,
}


def absolute_score(row: dict) -> tuple[float, dict]:
    components = {
        "i_perp_log_auc": float(row["i_perp_log_auc"]),
        "barrier_log_auc": float(row["barrier_log_auc"]),
        "log10_final_i_perp": math.log10(float(row["final_i_perp"])),
        "log10_final_barrier_kcal": math.log10(
            float(row["final_barrier_kcal"])
        ),
        "log10_one_plus_final_equidistance": math.log10(
            1.0 + float(row["final_equidistance_integral"])
        ),
    }
    score = sum(WEIGHTS[key] * components[key] for key in WEIGHTS)
    return float(score), components


def method_name(row: dict) -> str:
    method = row["configuration"]["method"]
    if method == "static_neb":
        return "Static-NEB"
    if method == "original_string":
        return "Original String"
    raise ValueError(f"Unexpected method: {method}")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "unified_rank",
        "method_rank",
        "method",
        "absolute_score",
        "configuration",
        "i_perp_log_auc",
        "barrier_log_auc",
        "final_i_perp",
        "final_barrier_kcal",
        "final_equidistance_integral",
        "result_file",
        "audit_file",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row[key], sort_keys=True)
                        if key == "configuration"
                        else row[key]
                    )
                    for key in fields
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--static-manifest", type=Path, required=True)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--report-namespace", required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    sources = [
        args.static_manifest.expanduser().resolve(),
        args.original_manifest.expanduser().resolve(),
    ]
    rows = []
    for source in sources:
        payload = json.loads(source.read_text())
        for source_row in payload["records"]:
            if not source_row.get("stable"):
                continue
            score, components = absolute_score(source_row)
            rows.append(
                {
                    **source_row,
                    "method": method_name(source_row),
                    "absolute_score": score,
                    "absolute_score_components": components,
                    "source_manifest": str(source),
                }
            )
    rows.sort(
        key=lambda row: (
            row["absolute_score"],
            row["final_i_perp"],
            row["final_barrier_kcal"],
        )
    )
    method_counts = {"Static-NEB": 0, "Original String": 0}
    for unified_rank, row in enumerate(rows, start=1):
        method_counts[row["method"]] += 1
        row["unified_rank"] = unified_rank
        row["method_rank"] = method_counts[row["method"]]
    report_dir = root / "reports" / args.report_namespace
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "manifest.json"
    output.write_text(
        json.dumps(
            {
                "score_name": (
                    "shared absolute weighted log-geometric objective"
                ),
                "lower_is_better": True,
                "same_formula_for_all_methods": True,
                "no_family_specific_rank_normalization": True,
                "screening_barrier_is_sparse": True,
                "weights": WEIGHTS,
                "formula": (
                    "0.30*AUC(log10(I_perp)) + "
                    "0.25*AUC(log10(barrier_kcal)) + "
                    "0.20*log10(final_I_perp) + "
                    "0.15*log10(final_barrier_kcal) + "
                    "0.10*log10(1+final_equidistance_integral)"
                ),
                "source_manifests": [str(path) for path in sources],
                "n_stable_candidates": len(rows),
                "records": rows,
            },
            indent=2,
        )
        + "\n"
    )
    write_csv(report_dir / "ranking.csv", rows)
    print(output, flush=True)


if __name__ == "__main__":
    main()
