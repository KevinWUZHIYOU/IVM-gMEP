#!/usr/bin/env python3
"""Clean-campaign-only alpha mechanism: Stage I r=0.1 versus Stage IV."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_stage4_v4_alpha_mechanism import (
    PRODUCTION_ALPHAS,
    add_common_key,
    configure_style,
    draw_panel,
    load_production,
    record_frontier,
    save_figure,
)


EXPECTED_INITIAL_SHA256 = (
    "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
)


def scalar(saved, key):
    return np.asarray(saved[key]).item()


def alpha_key(alpha: float) -> str:
    return "inf" if math.isinf(alpha) else f"{alpha:g}"


def load_clean_stage1(study: Path):
    directory = study / "runs/stage1_exact_no_resample"
    sources = sorted(directory.glob("ivm_clean_stage1_r0p1_*budget160.npz"))
    if len(sources) != len(PRODUCTION_ALPHAS):
        raise RuntimeError(
            f"Expected {len(PRODUCTION_ALPHAS)} clean Stage-I r=0.1 runs; "
            f"found {len(sources)}"
        )

    curves = {}
    metadata = {}
    for source in sources:
        with np.load(source, allow_pickle=False) as saved:
            alpha = float(scalar(saved, "alpha"))
            rtol = float(scalar(saved, "plbfgs_kkt_rtol"))
            initial_sha = str(scalar(saved, "initial_path_sha256"))
            force = np.asarray(
                saved["total_internal_joint_evaluations"], dtype=int
            )
            iperp = np.asarray(saved["i_perp"], dtype=float)
        if not np.isclose(rtol, 0.1):
            raise RuntimeError(f"Unexpected Stage-I tolerance in {source}: {rtol}")
        if initial_sha != EXPECTED_INITIAL_SHA256:
            raise RuntimeError(f"Common-initial-path mismatch in {source}")
        if np.any(np.diff(force) <= 0):
            raise RuntimeError(f"Non-increasing force counter in {source}")
        curves[alpha] = record_frontier(force, iperp)
        metadata[alpha] = {
            "source": str(source),
            "rtol": rtol,
            "resampling": False,
            "common_initial_path_sha256": initial_sha,
        }

    observed = sorted(
        alpha_key(alpha) for alpha in curves
    )
    expected = sorted(alpha_key(alpha) for alpha in PRODUCTION_ALPHAS)
    if observed != expected:
        raise RuntimeError(
            f"Stage-I alpha grid mismatch: observed={observed}, expected={expected}"
        )
    return curves, metadata


def main() -> None:
    study = Path(__file__).resolve().parents[1]
    output_dir = (
        study
        / "figures/clean_four_level_audit/alpha_mechanism_clean_campaign_v5"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    stage1, stage1_metadata = load_clean_stage1(study)
    stage4, stage4_metadata = load_production(study)

    figure, axes = plt.subplots(1, 2, figsize=(8.3, 2.75), sharey=True)
    draw_panel(
        axes[0],
        stage1,
        PRODUCTION_ALPHAS,
        r"Clean Stage I: exact-like, no resampling ($r=0.1$)",
        3e-5,
        selected_alpha=20.0,
    )
    draw_panel(
        axes[1],
        stage4,
        PRODUCTION_ALPHAS,
        r"Clean Stage IV: optimized inexact + resampling",
        1e-4,
        selected_alpha=20.0,
    )
    axes[1].set_ylabel("")
    for label, axis in zip(("a", "b"), axes):
        axis.text(
            -0.15,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
    add_common_key(figure)
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.875), pad=1.0, w_pad=1.1)
    figure.text(
        0.5,
        0.01,
        (
            "Both panels use the new clean-start campaign and the same initial "
            "path; Stage-IV reference-free screen work is fully charged."
        ),
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#555555",
    )
    stem = output_dir / (
        "alanine_alpha_iperp_cost_to_precision_clean_stage1_vs_stage4_v5"
    )
    save_figure(figure, stem)

    rows = []
    for family, curves in (
        ("clean_stage1_exact_like_r0p1_no_resampling", stage1),
        ("clean_stage4_uniform_v4_pareto", stage4),
    ):
        for alpha in PRODUCTION_ALPHAS:
            precision, cost = curves[alpha]
            for item_precision, item_cost in zip(precision, cost):
                rows.append(
                    {
                        "family": family,
                        "alpha": "inf" if math.isinf(alpha) else alpha,
                        "precision_target": item_precision,
                        "number_of_forces": item_cost,
                        "point_type": "strict_first_passage_record",
                    }
                )
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "status": "valid_clean_campaign_only",
        "all_numeric_curves_from_clean_four_level_campaign": True,
        "common_initial_path_sha256": EXPECTED_INITIAL_SHA256,
        "left_panel": {
            "protocol": "Stage I exact-like IVM; r=0.1; no resampling",
            "run_count": len(stage1_metadata),
            "runs": {
                alpha_key(alpha): stage1_metadata[alpha]
                for alpha in PRODUCTION_ALPHAS
            },
        },
        "right_panel": {
            "protocol": (
                "Stage IV uniform Phase-A34/Phase-B220 optimized inexact IVM "
                "plus reference-free charged Pareto archive"
            ),
            "run_count": len(stage4_metadata),
            "runs": {
                alpha_key(alpha): stage4_metadata[alpha]
                for alpha in PRODUCTION_ALPHAS
            },
        },
        "stage2_and_stage3_are_separate_audit_figures": True,
        "selected_alpha": 20,
        "main_tex_modified": False,
        "output": str(stem.with_suffix(".pdf")),
    }
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
