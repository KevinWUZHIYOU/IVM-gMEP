#!/usr/bin/env python3
"""Keep the controlled r=0.1 panel and replace production with clean v4 data."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CONTROLLED_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 10000.0, math.inf)
PRODUCTION_ALPHAS = (
    0.01, 0.1, 1.0, 10.0, 20.0, 30.0, 100.0, 1000.0, 10000.0, math.inf
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h", "<")
ALPHA_CMAP = mpl.colormaps["viridis_r"]
ALPHA_NORM = mpl.colors.Normalize(vmin=-2.0, vmax=4.0)


def alpha_color(alpha: float):
    return "#000000" if math.isinf(alpha) else ALPHA_CMAP(ALPHA_NORM(math.log10(alpha)))


def alpha_tag(alpha: float) -> str:
    if math.isinf(alpha):
        return "inf"
    return f"{alpha:g}".replace(".", "p")


def same_alpha(left: float, right: float) -> bool:
    if math.isinf(left) or math.isinf(right):
        return math.isinf(left) and math.isinf(right)
    return math.isclose(left, right, rel_tol=5e-11, abs_tol=1e-14)


def record_frontier(cost, precision):
    cost = np.asarray(cost, dtype=float)
    precision = np.asarray(precision, dtype=float)
    finite = np.isfinite(cost) & np.isfinite(precision) & (cost > 0) & (precision > 0)
    cost, precision = cost[finite], precision[finite]
    order = np.argsort(cost, kind="stable")
    cost, precision = cost[order], precision[order]
    record_cost, record_precision, incumbent = [], [], math.inf
    for item_cost in np.unique(cost):
        item_precision = float(np.min(precision[cost == item_cost]))
        if item_precision < incumbent:
            record_cost.append(float(item_cost))
            record_precision.append(item_precision)
            incumbent = item_precision
    return np.asarray(record_precision), np.asarray(record_cost)


def load_controlled(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    curves = {}
    for alpha in CONTROLLED_ALPHAS:
        selected = [row for row in rows if same_alpha(float(row["alpha"]), alpha)]
        if not selected:
            raise RuntimeError(f"Controlled panel is missing alpha={alpha}")
        curves[alpha] = (
            np.asarray([float(row["precision_target"]) for row in selected]),
            np.asarray([float(row["number_of_forces"]) for row in selected]),
        )
    return curves


def scalar(saved, key):
    return np.asarray(saved[key]).item()


def load_production(study: Path):
    curves, metadata = {}, {}
    for alpha in PRODUCTION_ALPHAS:
        tag = alpha_tag(alpha)
        combined_path = (
            study / "runs/stage4_inexact_optimized_resample"
            / f"stage4_uniform_v3_a{tag}_phaseA34_phaseB220.npz"
        )
        archive_path = (
            study / "runs/pareto_archive"
            / f"pareto_clean_stage4_v4_a{tag}_phaseA34_phaseB220.npz"
        )
        with np.load(combined_path, allow_pickle=False) as combined, np.load(
            archive_path, allow_pickle=False
        ) as archive:
            raw_force = np.asarray(combined["total_force_evaluations"], dtype=int)
            raw_i = np.asarray(combined["i_perp"], dtype=float)
            candidate_calls = np.asarray(archive["candidate_calls"], dtype=int)
            if candidate_calls.size:
                prefix = raw_force < int(candidate_calls[0])
                costs = np.asarray(
                    archive["candidate_screen_force_equivalent_cost"], dtype=int
                )
                accepted = np.asarray(archive["candidate_accepted"], dtype=bool)
                accepted_cost = candidate_calls[accepted] + np.cumsum(costs)[accepted]
                accepted_i = np.asarray(archive["candidate_i_perp"], dtype=float)[accepted]
                force = np.concatenate([raw_force[prefix], accepted_cost])
                i_perp = np.concatenate([raw_i[prefix], accepted_i])
            else:
                force, i_perp = raw_force, raw_i
            curves[alpha] = record_frontier(force, i_perp)
            selected_i = float(scalar(archive, "selected_i_perp"))
            metadata[alpha] = {
                "combined_result": str(combined_path),
                "pareto_archive": str(archive_path),
                "target_reached": bool(scalar(archive, "target_reached")),
                "selected_i_perp": selected_i if math.isfinite(selected_i) else None,
                "selected_call": int(scalar(archive, "selected_call")),
                "screen_force_equivalent_evaluations": int(
                    scalar(archive, "screen_force_equivalent_evaluations")
                ),
                "total_charged_force_equivalent_evaluations": int(
                    scalar(archive, "total_charged_force_equivalent_evaluations")
                ),
                "reference_barrier_used": bool(
                    scalar(archive, "reference_barrier_used")
                ),
            }
    if any(item["reference_barrier_used"] for item in metadata.values()):
        raise RuntimeError("The production alpha panel must be reference-free")
    return curves, metadata


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def draw_panel(ax, curves, alphas, title, xminimum, selected_alpha=None):
    finite_alphas = [alpha for alpha in alphas if not math.isinf(alpha)]
    marker_by_alpha = {
        alpha: MARKERS[index % len(MARKERS)] for index, alpha in enumerate(finite_alphas)
    }
    for alpha in alphas:
        precision, cost = curves[alpha]
        selected = selected_alpha is not None and same_alpha(alpha, selected_alpha)
        ax.plot(
            precision,
            cost,
            color=alpha_color(alpha),
            linewidth=2.0 if selected else (1.55 if math.isinf(alpha) else 1.2),
            marker="o" if math.isinf(alpha) else marker_by_alpha[alpha],
            markersize=3.8 if selected else 3.0,
            markevery=max(1, len(cost) // 5),
            markeredgewidth=0.6 if selected else 0.3,
            markeredgecolor="#202020" if selected else alpha_color(alpha),
            linestyle="--" if math.isinf(alpha) else "-",
            zorder=4 if selected else 2,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(30.0, xminimum)
    ax.set_ylim(20.0, 26000.0)
    ax.set_xlabel(r"Target $I_\perp$")
    ax.set_ylabel("Number of Forces")
    ax.set_title(title, pad=5)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    ax.spines[["top", "right"]].set_visible(False)


def add_common_key(figure):
    axis = figure.add_axes([0.18, 0.925, 0.55, 0.018])
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=ALPHA_NORM, cmap=ALPHA_CMAP),
        cax=axis,
        orientation="horizontal",
    )
    colorbar.set_ticks([-2, -1, 0, 1, 2, 3, 4])
    colorbar.set_ticklabels(["-2", "-1", "0", "1", "2", "3", "4"])
    colorbar.ax.tick_params(labelsize=6.5, length=2.0, pad=1.5)
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.set_title(r"Finite values: $\log_{10}\alpha$", fontsize=7.0, pad=3.0)
    infinity = Line2D(
        [0], [0], color="#000000", linewidth=1.55, linestyle="--",
        marker="o", markersize=3.0, label=r"$\alpha=\infty$"
    )
    selected = Line2D(
        [0], [0], color=alpha_color(20.0), linewidth=2.0, marker="v",
        markeredgecolor="#202020", markersize=4.0, label=r"selected $\alpha=20$"
    )
    figure.legend(
        handles=[selected, infinity], frameon=False, loc="upper right",
        bbox_to_anchor=(0.985, 0.997), handlelength=1.6
    )


def save_figure(figure, stem: Path):
    for suffix in (".pdf", ".svg"):
        figure.savefig(stem.with_suffix(suffix), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    study = Path(__file__).resolve().parents[1]
    controlled_csv = (
        Path(__file__).resolve().parents[3]
        / "python/alanine_production_alpha_20260730/figures/alpha_mechanism"
        / "alanine_alpha_iperp_cost_to_precision_controlled_r0p1.csv"
    )
    output = study / "figures/clean_four_level_audit/alpha_mechanism_stage4_v4"
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    controlled = load_controlled(controlled_csv)
    production, metadata = load_production(study)
    figure, axes = plt.subplots(1, 2, figsize=(8.3, 2.75), sharey=True)
    draw_panel(
        axes[0], controlled, CONTROLLED_ALPHAS,
        r"Controlled inner solves ($r=0.1$)", 3e-4
    )
    draw_panel(
        axes[1], production, PRODUCTION_ALPHAS,
        r"Production inexact + Pareto archive", 1e-4, selected_alpha=20.0
    )
    axes[1].set_ylabel("")
    for label, ax in zip(("a", "b"), axes):
        ax.text(
            -0.15, 1.05, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top"
        )
    add_common_key(figure)
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.875), pad=1.0, w_pad=1.1)
    figure.text(
        0.5, 0.01,
        (
            "Left panel unchanged; right panel uses clean A34/B220 runs and "
            "charges every reference-free barrier10 screen as force-equivalent cost."
        ),
        ha="center", va="bottom", fontsize=7.0, color="#555555"
    )
    stem = output / "alanine_alpha_iperp_cost_to_precision_controlled_vs_production_v4_pareto"
    save_figure(figure, stem)

    rows = []
    for family, curves, alphas in (
        ("controlled_r0p1", controlled, CONTROLLED_ALPHAS),
        ("production_v4_pareto", production, PRODUCTION_ALPHAS),
    ):
        for alpha in alphas:
            precision, cost = curves[alpha]
            for item_precision, item_cost in zip(precision, cost):
                rows.append(
                    {
                        "family": family,
                        "alpha": alpha,
                        "precision_target": item_precision,
                        "number_of_forces": item_cost,
                        "point_type": "strict_running_best_record",
                    }
                )
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "controlled_panel_unchanged_source": str(controlled_csv),
        "controlled_alphas": ["inf" if math.isinf(a) else a for a in CONTROLLED_ALPHAS],
        "production_alphas": ["inf" if math.isinf(a) else a for a in PRODUCTION_ALPHAS],
        "production_policy": "stage4_uniform_v4_pareto_archive; Phase A=34; Phase B<=220",
        "production_rule_identical_across_alpha": True,
        "selected_alpha": 20,
        "selected_target_i_perp": 1.9e-4,
        "production_runs": {
            "inf" if math.isinf(alpha) else f"{alpha:g}": metadata[alpha]
            for alpha in PRODUCTION_ALPHAS
        },
        "output": str(stem.with_suffix(".pdf")),
    }
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
