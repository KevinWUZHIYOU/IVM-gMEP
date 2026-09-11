#!/usr/bin/env python3
"""Use the shared absolute objective with densified production barriers."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


WEIGHTS = {
    "i_perp_log_auc": 0.30,
    "dense_barrier_log_auc": 0.25,
    "log10_final_i_perp": 0.20,
    "log10_final_dense_barrier_kcal": 0.15,
    "log10_one_plus_final_equidistance": 0.10,
}


def log_auc(cost: np.ndarray, values: np.ndarray) -> float:
    x = np.asarray(cost, dtype=float)
    y = np.maximum(np.asarray(values, dtype=float), 1e-300)
    if x.size < 2 or x[-1] <= x[0]:
        raise ValueError("AUC requires an increasing multi-point cost axis")
    return float(np.trapz(np.log10(y), x) / (x[-1] - x[0]))


def method_name(configuration: dict) -> str:
    names = {
        "static_neb": "Static-NEB",
        "original_string": "Original String",
        "ode_neb": "ODE-NEB",
        "obsm_fire": "OBSM-FIRE",
    }
    try:
        return names[configuration["method"]]
    except KeyError as exc:
        raise ValueError(
            f"Unexpected configuration: {configuration}"
        ) from exc


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "unified_rank",
        "method_rank",
        "method",
        "name",
        "absolute_dense_score",
        "configuration",
        "i_perp_log_auc",
        "dense_barrier_log_auc",
        "final_i_perp",
        "final_dense_barrier_kcal",
        "final_equidistance_integral",
        "result_file",
        "dense_result_file",
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
    parser.add_argument("--dense-manifest", type=Path, required=True)
    parser.add_argument("--report-namespace", required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    production_path = args.production_manifest.expanduser().resolve()
    dense_path = args.dense_manifest.expanduser().resolve()
    production_payload = json.loads(production_path.read_text())
    dense_payload = json.loads(dense_path.read_text())
    dense_by_name = {
        row["name"]: row for row in dense_payload["records"]
    }
    rows = []
    for production in production_payload["records"]:
        name = production["name"]
        dense_record = dense_by_name[name]
        result = Path(production["result_file"])
        dense_result = Path(dense_record["result_file"])
        with np.load(result, allow_pickle=False) as saved:
            calls = np.asarray(saved["calls_per_image"], dtype=int)
            counter_vectors = np.asarray(saved["counter_vectors"], dtype=int)
            cost = counter_vectors.sum(axis=1)
            i_perp = np.asarray(saved["i_perp"], dtype=float)
            equidistance = np.asarray(
                saved["equidistance_integral"], dtype=float
            )
            initial_hash = str(saved["initial_path_sha256"])
        with np.load(dense_result, allow_pickle=False) as saved:
            dense_calls = np.asarray(saved["calls"], dtype=int)
            dense_barrier = np.asarray(
                saved["adaptive_barrier_kcal"], dtype=float
            )
        dense_cost = []
        for call in dense_calls:
            match = np.flatnonzero(calls == int(call))
            if not match.size:
                raise RuntimeError(
                    f"Dense call {call} missing from {result}"
                )
            dense_cost.append(int(cost[match[-1]]))
        dense_cost = np.asarray(dense_cost, dtype=float)
        components = {
            "i_perp_log_auc": log_auc(cost, i_perp),
            "dense_barrier_log_auc": log_auc(
                dense_cost, dense_barrier
            ),
            "log10_final_i_perp": math.log10(float(i_perp[-1])),
            "log10_final_dense_barrier_kcal": math.log10(
                float(dense_barrier[-1])
            ),
            "log10_one_plus_final_equidistance": math.log10(
                1.0 + float(equidistance[-1])
            ),
        }
        score = sum(
            WEIGHTS[key] * components[key] for key in WEIGHTS
        )
        rows.append(
            {
                "name": name,
                "method": method_name(production["configuration"]),
                "configuration": production["configuration"],
                "absolute_dense_score": float(score),
                "absolute_dense_score_components": components,
                "i_perp_log_auc": components["i_perp_log_auc"],
                "dense_barrier_log_auc": components[
                    "dense_barrier_log_auc"
                ],
                "final_i_perp": float(i_perp[-1]),
                "final_dense_barrier_kcal": float(dense_barrier[-1]),
                "final_equidistance_integral": float(equidistance[-1]),
                "initial_path_sha256": initial_hash,
                "final_number_of_force_evaluations": int(cost[-1]),
                "result_file": str(result.resolve()),
                "dense_result_file": str(dense_result.resolve()),
                "production_audit_file": production["audit_file"],
                "dense_audit_file": dense_record["audit_file"],
            }
        )
    rows.sort(
        key=lambda row: (
            row["absolute_dense_score"],
            row["final_i_perp"],
            row["final_dense_barrier_kcal"],
        )
    )
    method_counts = {
        "Static-NEB": 0,
        "Original String": 0,
        "ODE-NEB": 0,
        "OBSM-FIRE": 0,
    }
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
                    "shared absolute weighted log-geometric objective "
                    "with densified barrier"
                ),
                "lower_is_better": True,
                "same_formula_for_all_methods": True,
                "no_family_specific_rank_normalization": True,
                "weights": WEIGHTS,
                "formula": (
                    "0.30*AUC(log10(I_perp)) + "
                    "0.25*AUC(log10(densified_barrier_kcal)) + "
                    "0.20*log10(final_I_perp) + "
                    "0.15*log10(final_densified_barrier_kcal) + "
                    "0.10*log10(1+final_equidistance_integral)"
                ),
                "production_manifest": str(production_path),
                "dense_manifest": str(dense_path),
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
