#!/usr/bin/env python3
"""Shared absolute sparse-barrier score for all non-IVM 1000-call runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


WEIGHTS = {
    "i_perp_log_auc": 0.30,
    "barrier_log_auc": 0.25,
    "log10_final_i_perp": 0.20,
    "log10_final_barrier_kcal": 0.15,
    "log10_one_plus_final_equidistance": 0.10,
}


METHOD_NAMES = {
    "original_string": "Original String",
    "static_neb": "Static-NEB",
    "ode_neb": "ODE-NEB",
    "obsm_fire": "OBSM-FIRE",
}


def log_auc(cost: np.ndarray, values: np.ndarray) -> float:
    x = np.asarray(cost, dtype=float)
    y = np.maximum(np.asarray(values, dtype=float), 1e-300)
    return float(np.trapz(np.log10(y), x) / (x[-1] - x[0]))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "unified_rank",
        "method_rank",
        "method",
        "name",
        "absolute_sparse_score",
        "configuration",
        "i_perp_log_auc",
        "barrier_log_auc",
        "final_i_perp",
        "final_barrier_kcal",
        "final_equidistance_integral",
        "result_file",
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
    parser.add_argument("--production-manifest", type=Path, required=True)
    parser.add_argument("--report-namespace", required=True)
    parser.add_argument("--top-per-method", type=int, default=3)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    source = args.production_manifest.expanduser().resolve()
    payload = json.loads(source.read_text())
    source_by_name = {row["name"]: row for row in payload["records"]}
    rows = []
    for source_row in payload["records"]:
        result = Path(source_row["result_file"])
        with np.load(result, allow_pickle=False) as saved:
            cost = np.asarray(saved["counter_vectors"], dtype=int).sum(
                axis=1
            )
            i_perp = np.asarray(saved["i_perp"], dtype=float)
            barrier = np.asarray(saved["barrier_kcal"], dtype=float)
            equidistance = np.asarray(
                saved["equidistance_integral"], dtype=float
            )
            initial_hash = str(saved["initial_path_sha256"])
        components = {
            "i_perp_log_auc": log_auc(cost, i_perp),
            "barrier_log_auc": log_auc(cost, barrier),
            "log10_final_i_perp": math.log10(float(i_perp[-1])),
            "log10_final_barrier_kcal": math.log10(
                float(barrier[-1])
            ),
            "log10_one_plus_final_equidistance": math.log10(
                1.0 + float(equidistance[-1])
            ),
        }
        score = sum(
            WEIGHTS[key] * components[key] for key in WEIGHTS
        )
        method = METHOD_NAMES[source_row["configuration"]["method"]]
        rows.append(
            {
                **source_row,
                "method": method,
                "absolute_sparse_score": float(score),
                "absolute_sparse_score_components": components,
                "i_perp_log_auc": components["i_perp_log_auc"],
                "barrier_log_auc": components["barrier_log_auc"],
                "final_i_perp": float(i_perp[-1]),
                "final_barrier_kcal": float(barrier[-1]),
                "final_equidistance_integral": float(equidistance[-1]),
                "initial_path_sha256": initial_hash,
                "final_number_of_force_evaluations": int(cost[-1]),
            }
        )
    rows.sort(
        key=lambda row: (
            row["absolute_sparse_score"],
            row["final_i_perp"],
            row["final_barrier_kcal"],
        )
    )
    method_counts = {name: 0 for name in METHOD_NAMES.values()}
    selected_names = []
    for unified_rank, row in enumerate(rows, start=1):
        method_counts[row["method"]] += 1
        row["unified_rank"] = unified_rank
        row["method_rank"] = method_counts[row["method"]]
        if row["method_rank"] <= int(args.top_per_method):
            selected_names.append(row["name"])
    selected_records = [source_by_name[name] for name in selected_names]
    report_dir = root / "reports" / args.report_namespace
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "manifest.json"
    selected_output = report_dir / "selected_production_manifest.json"
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
                "source_production_manifest": str(source),
                "top_per_method_for_dense_validation": int(
                    args.top_per_method
                ),
                "selected_names": selected_names,
                "records": rows,
            },
            indent=2,
        )
        + "\n"
    )
    selected_output.write_text(
        json.dumps(
            {
                "source_production_manifest": str(source),
                "selection_score_manifest": str(output),
                "records": selected_records,
            },
            indent=2,
        )
        + "\n"
    )
    write_csv(report_dir / "ranking.csv", rows)
    print(output, flush=True)
    print(selected_output, flush=True)


if __name__ == "__main__":
    main()
