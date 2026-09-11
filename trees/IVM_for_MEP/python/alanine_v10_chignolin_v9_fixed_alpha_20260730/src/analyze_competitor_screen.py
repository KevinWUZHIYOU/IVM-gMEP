#!/usr/bin/env python3
"""Rank one competitor screen without consulting IVM or another method."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


WEIGHTS = {
    "iperp_log_auc": 0.35,
    "barrier_log_auc": 0.20,
    "terminal_log_iperp": 0.20,
    "terminal_barrier": 0.10,
    "terminal_log_spacing_cv": 0.10,
    "terminal_log_spring_residual": 0.05,
}


def scalar(saved, name):
    return np.asarray(saved[name]).item()


def percentile_rank(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.lexsort((np.arange(len(values)), values))
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    if len(values) > 1:
        ranks /= len(values) - 1
    return ranks


def hold_last(cost: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(cost, grid, side="right") - 1
    indices = np.clip(indices, 0, len(values) - 1)
    return values[indices]


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        counters = np.asarray(saved["counter_vectors"], dtype=int)
        cost = np.sum(counters, axis=1)
        result = {
            "path": str(path.resolve()),
            "method": str(scalar(saved, "method")),
            "cost": cost,
            "i_perp": np.asarray(saved["i_perp"], dtype=float),
            "barrier": np.asarray(saved["barrier_kcal"], dtype=float),
            "spacing_cv": np.asarray(saved["segment_length_cv"], dtype=float),
            "spring_residual": np.asarray(
                saved["max_unit_spring_force"], dtype=float
            ),
            "final_call": int(np.asarray(saved["calls_per_image"])[-1]),
            "initial_path_sha256": str(scalar(saved, "initial_path_sha256")),
            "stop_reason": str(scalar(saved, "stop_reason")),
        }
        if "config_json" in saved.files:
            result["config"] = json.loads(str(scalar(saved, "config_json")))
        else:
            result["config"] = {
                "spring_k": float(scalar(saved, "spring_k")),
                "alpha": float(scalar(saved, "optimizer_alpha")),
                "optimizer": str(scalar(saved, "optimizer")),
            }
    if not (
        len(result["cost"])
        == len(result["i_perp"])
        == len(result["barrier"])
        == len(result["spacing_cv"])
        == len(result["spring_residual"])
    ):
        raise RuntimeError(f"Metric length mismatch: {path}")
    if np.any(np.diff(result["cost"]) < 0):
        raise RuntimeError(f"Nonmonotone real-cost counter: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--method",
        choices=(
            "original",
            "original_fine",
            "original_finalists",
            "original_rank1_long",
            "static",
            "static_fine",
            "static_finalists",
            "bfgs",
            "bfgs_stage",
            "bfgs_curvature_stage",
            "bfgs_carry_stage",
            "bfgs_identity_small_stage",
            "bfgs_identity_mid_stage",
            "bfgs_carry_mid_stage",
            "bfgs_finalists",
            "bfgs_finalists_r5",
            "bfgs_diagnostic_finalists",
            "fire_stage",
            "fire_refine",
            "fire_finalists",
            "fire",
        ),
        required=True,
    )
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    launch_path = (
        root
        / "reports"
        / f"competitor_screen_{args.method}_budget{args.budget}_launch.json"
    )
    launch = json.loads(launch_path.read_text())
    paths = [
        Path(row["result"])
        for row in launch["rows"]
        if row["returncode"] == 0 and row["result"]
    ]
    items = [load(path) for path in paths]
    if not items:
        raise RuntimeError("No successful candidates")
    initial_hashes = sorted({item["initial_path_sha256"] for item in items})
    if len(initial_hashes) != 1:
        raise RuntimeError(f"Initial-path mismatch: {initial_hashes}")
    common_cost = min(int(item["cost"][-1]) for item in items)
    lower = max(int(item["cost"][0]) for item in items)
    grid = np.unique(
        np.rint(np.geomspace(max(lower, 1), max(common_cost, lower), 240)).astype(int)
    )
    raw = {name: [] for name in WEIGHTS}
    for item in items:
        iperp = hold_last(item["cost"], item["i_perp"], grid)
        barrier = hold_last(item["cost"], item["barrier"], grid)
        raw["iperp_log_auc"].append(
            float(np.mean(np.log10(np.maximum(iperp / iperp[0], 1e-14))))
        )
        raw["barrier_log_auc"].append(
            float(np.mean(np.log10(np.maximum(barrier / barrier[0], 1e-14))))
        )
        raw["terminal_log_iperp"].append(
            float(np.log10(max(float(item["i_perp"][-1]), 1e-300)))
        )
        raw["terminal_barrier"].append(float(item["barrier"][-1]))
        raw["terminal_log_spacing_cv"].append(
            float(np.log10(max(float(item["spacing_cv"][-1]), 1e-14)))
        )
        raw["terminal_log_spring_residual"].append(
            float(np.log10(max(float(item["spring_residual"][-1]), 1e-14)))
        )
    component_ranks = {
        name: percentile_rank(values) for name, values in raw.items()
    }
    score = np.zeros(len(items), dtype=float)
    for name, weight in WEIGHTS.items():
        score += weight * component_ranks[name]
    ranking = []
    for index, item in enumerate(items):
        ranking.append(
            {
                "rank": 0,
                "score": float(score[index]),
                "path": item["path"],
                "config": item["config"],
                "stop_reason": item["stop_reason"],
                "final_call": item["final_call"],
                "final_whole_path_force_calls": int(item["cost"][-1]),
                "final_i_perp": float(item["i_perp"][-1]),
                "minimum_i_perp": float(np.min(item["i_perp"])),
                "final_sparse_barrier_kcal": float(item["barrier"][-1]),
                "minimum_sparse_barrier_kcal": float(np.min(item["barrier"])),
                "final_spacing_cv": float(item["spacing_cv"][-1]),
                "raw_components": {
                    name: float(values[index]) for name, values in raw.items()
                },
                "component_percentile_ranks": {
                    name: float(values[index])
                    for name, values in component_ranks.items()
                },
            }
        )
    ranking.sort(key=lambda row: (row["score"], row["path"]))
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    payload = {
        "method": args.method,
        "budget": int(args.budget),
        "successful_candidates": len(items),
        "failed_candidates": len(launch["rows"]) - len(items),
        "common_whole_path_cost_ceiling": common_cost,
        "weights": WEIGHTS,
        "selection_uses_ivm_or_other_competitors": False,
        "barrier_metric_at_screen": "raw sparse 22-image forward barrier",
        "dense_barrier_required_for_finalist_selection": True,
        "single_initial_path_sha256": initial_hashes[0],
        "ranking": ranking,
    }
    output = (
        root
        / "reports"
        / f"competitor_screen_{args.method}_budget{args.budget}_ranking.json"
    )
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, output)
    print(output)
    for row in ranking[:10]:
        print(
            row["rank"],
            f"score={row['score']:.5f}",
            f"Iperp={row['final_i_perp']:.6e}",
            f"barrier={row['final_sparse_barrier_kcal']:.6f}",
            row["config"],
        )


if __name__ == "__main__":
    main()
