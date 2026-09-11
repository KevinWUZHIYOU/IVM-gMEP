#!/usr/bin/env python3
"""Visualize the preregistered refined LEPS IR-IVM grid and selection."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


HERE = Path(__file__).resolve()
CAMPAIGN_ROOT = HERE.parents[1]
SUMMARY = CAMPAIGN_ROOT / "reports/acceptance/candidate_summary.csv"
SELECTION = CAMPAIGN_ROOT / "reports/acceptance/selection_manifest.json"
OUTDIR = CAMPAIGN_ROOT / "reports/figure"
PDF = HERE.parents[4] / "output/pdf/leps_irivm_refined_grid_selection.pdf"
PNG = OUTDIR / "leps_irivm_refined_grid_selection.png"

ALPHAS = (20.0, 50.0, 100.0, 200.0, 500.0)
R_VALUES = (0.001, 0.003, 0.01, 0.03, 0.1)
M_MAX_VALUES = (3, 4, 5, 6, 8, 10)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank(row):
    return (
        row["strict_pass"] != "True",
        -int(row["passing_checks"]),
        float(row["worst_joint_ratio"]),
        float(row["mean_joint_ratio"]),
        int(row["terminal_article_nf"]),
        float(row["alpha"]),
        float(row["r"]),
        int(row["m_max"]),
    )


def main():
    rows = list(csv.DictReader(SUMMARY.open(encoding="utf-8")))
    manifest = json.loads(SELECTION.read_text(encoding="utf-8"))
    winner = manifest["winner"]
    by_pair = {}
    for alpha in ALPHAS:
        for r_value in R_VALUES:
            subset = [
                row for row in rows
                if float(row["alpha"]) == alpha and float(row["r"]) == r_value
            ]
            by_pair[(alpha, r_value)] = min(subset, key=rank)

    checks = np.asarray(
        [[int(by_pair[(alpha, r_value)]["passing_checks"]) for r_value in R_VALUES]
         for alpha in ALPHAS],
        dtype=float,
    )
    means = np.asarray(
        [[float(by_pair[(alpha, r_value)]["mean_joint_ratio"]) for r_value in R_VALUES]
         for alpha in ALPHAS],
        dtype=float,
    )

    chosen_rows = [
        row for row in rows
        if float(row["alpha"]) == float(winner["alpha"])
        and float(row["r"]) == float(winner["r"])
    ]
    chosen_rows.sort(key=lambda row: int(row["m_max"]))
    m_values = np.asarray([int(row["m_max"]) for row in chosen_rows])
    m_checks = np.asarray([int(row["passing_checks"]) for row in chosen_rows])
    m_nof = np.asarray([int(row["terminal_article_nf"]) for row in chosen_rows])

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.2,
        "axes.labelsize": 8.8,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 68 / 25.4), dpi=220)
    r_labels = [r"$10^{-3}$", r"$3\times10^{-3}$", r"$10^{-2}$", r"$3\times10^{-2}$", r"$10^{-1}$"]

    axes[0].imshow(checks, cmap="viridis", vmin=0, vmax=21, aspect="auto")
    for i in range(checks.shape[0]):
        for j in range(checks.shape[1]):
            color = "white" if checks[i, j] < 10 else "black"
            axes[0].text(j, i, f"{int(checks[i, j])}", ha="center", va="center", color=color, fontsize=7.5)
    axes[0].set_title("a  Best full-course checks")
    axes[0].set_xlabel(r"Main stopping ratio $r$")
    axes[0].set_ylabel(r"$\alpha$")

    axes[1].imshow(
        means,
        cmap="magma_r",
        norm=LogNorm(vmin=max(np.min(means), 1e-2), vmax=np.max(means)),
        aspect="auto",
    )
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            label = f"{means[i, j]:.3f}" if means[i, j] < 10 else r"$>10^6$"
            axes[1].text(j, i, label, ha="center", va="center", color="black" if means[i, j] < 10 else "white", fontsize=6.5)
    selected_i = ALPHAS.index(float(winner["alpha"]))
    selected_j = R_VALUES.index(float(winner["r"]))
    axes[1].add_patch(
        plt.Rectangle((selected_j - 0.48, selected_i - 0.48), 0.96, 0.96,
                      fill=False, edgecolor="#00A087", linewidth=2.0)
    )
    axes[1].set_title("b  Best mean joint ratio (lower is better)")
    axes[1].set_xlabel(r"Main stopping ratio $r$")
    axes[1].set_ylabel(r"$\alpha$")

    colors = ["#00A087" if value == int(winner["m_max"]) else "#4C78A8" for value in m_values]
    bars = axes[2].bar(m_values, m_checks, width=0.72, color=colors, zorder=2)
    axes[2].axhline(21, color="#D55E00", linestyle="--", linewidth=1.1, label="Strict target: 21/21")
    axes[2].set_ylim(0, 22.5)
    axes[2].set_xticks(m_values)
    axes[2].set_xlabel(r"Maximum inner steps $m_{\max}$")
    axes[2].set_ylabel("Checks passed")
    axes[2].set_title(r"c  $m_{\max}$ at $\alpha=100,\ r=3\times10^{-3}$")
    second = axes[2].twinx()
    second.plot(m_values, m_nof, "o-", color="#7A5195", linewidth=1.25, markersize=3.5, label="Terminal NoF")
    second.set_ylabel("Terminal Number of Forces", color="#7A5195")
    second.tick_params(axis="y", colors="#7A5195")
    axes[2].bar_label(bars, labels=[str(value) for value in m_checks], padding=1.5, fontsize=7.0)
    handles1, labels1 = axes[2].get_legend_handles_labels()
    handles2, labels2 = second.get_legend_handles_labels()
    axes[2].legend(handles1 + handles2, labels1 + labels2, loc="lower right", frameon=False)
    axes[2].grid(axis="y", color="#dddddd", linewidth=0.55, zorder=0)

    for axis in axes[:2]:
        axis.set_xticks(range(len(R_VALUES)), r_labels, rotation=28, ha="right")
        axis.set_yticks(range(len(ALPHAS)), [f"{int(value)}" for value in ALPHAS])

    fig.subplots_adjust(left=0.055, right=0.92, bottom=0.23, top=0.88, wspace=0.58)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF, dpi=600, metadata={"Title": "LEPS IR-IVM refined grid selection"})
    fig.savefig(PNG, dpi=260, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "figure_pdf": str(PDF),
        "figure_pdf_sha256": sha256(PDF),
        "figure_png": str(PNG.relative_to(LEPS_ROOT := HERE.parents[2])),
        "figure_png_sha256": sha256(PNG),
        "plot_source": str(HERE.relative_to(LEPS_ROOT)),
        "plot_source_sha256": sha256(HERE),
        "candidate_summary": str(SUMMARY.relative_to(LEPS_ROOT)),
        "candidate_summary_sha256": sha256(SUMMARY),
        "selection_manifest": str(SELECTION.relative_to(LEPS_ROOT)),
        "selection_manifest_sha256": sha256(SELECTION),
        "winner": winner,
    }
    path = OUTDIR / "leps_irivm_refined_grid_selection_provenance.json"
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(PDF)
    print(PNG)
    print(path)


if __name__ == "__main__":
    main()
