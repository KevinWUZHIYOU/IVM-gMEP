#!/usr/bin/env python3
"""Paper-grade fixed-alpha V2 sensitivity from preregistered raw scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


FORCE_BUDGETS = (1000, 3000, 6000, 10000)
BUDGET_COLORS = ("#0072B2", "#D55E00", "#009E73", "#7A5195")
BUDGET_MARKERS = ("o", "s", "^", "D")
BUDGET_LINESTYLES = ("-", "--", "-.", ":")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def clean_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(
        which="major", color="#D9D9D9", lw=0.45, alpha=0.65, zorder=0
    )
    axis.tick_params(direction="out", length=3.0, width=0.7)


def panel_label(axis, label: str) -> None:
    axis.text(
        -0.02,
        1.04,
        label,
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=9.5,
        va="bottom",
    )


def save_figure(figure, output_dir: Path, stem: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 600}),
    ):
        path = output_dir / f"{stem}{suffix}"
        figure.savefig(path, **kwargs)
        outputs.append(str(path.resolve()))
    return outputs


def draw(
    alphas: np.ndarray,
    selected_alpha: float,
    i_values: np.ndarray,
    barrier_values: np.ndarray,
    output_dir: Path,
    y_scale: str,
    primary_stem: str,
) -> list[str]:
    setup_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.12))
    measures = (i_values, barrier_values)
    ylabels = (
        r"Trailing mean unnormalized $I_\perp$",
        "Trailing RMS dense-barrier error\n"
        r"(kcal mol$^{-1}$)",
    )
    for panel, (axis, measured, ylabel) in enumerate(
        zip(axes, measures, ylabels)
    ):
        for budget_index, budget in enumerate(FORCE_BUDGETS):
            axis.plot(
                alphas,
                measured[:, budget_index],
                color=BUDGET_COLORS[budget_index],
                marker=BUDGET_MARKERS[budget_index],
                linestyle=BUDGET_LINESTYLES[budget_index],
                ms=3.6,
                markerfacecolor="white",
                markeredgewidth=0.8,
                label=rf"$N_F\leq {budget:,}$",
                zorder=3 + budget_index,
            )
        axis.axvline(
            selected_alpha,
            color="#333333",
            lw=0.95,
            ls=(0, (2.0, 2.0)),
            zorder=2,
        )
        selected_index = int(
            np.flatnonzero(np.isclose(alphas, selected_alpha))[0]
        )
        axis.scatter(
            np.full(len(FORCE_BUDGETS), selected_alpha),
            measured[selected_index],
            s=38,
            facecolors="none",
            edgecolors="#222222",
            linewidths=0.9,
            zorder=10,
        )
        axis.set_xscale("log")
        axis.set_yscale(y_scale)
        axis.set_xlim(
            float(alphas[0]) / 1.25, float(alphas[-1]) * 1.25
        )
        axis.set_xlabel(r"Fixed proximal parameter $\alpha$")
        axis.set_ylabel(ylabel)
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + panel))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        handlelength=2.6,
        columnspacing=1.35,
        handletextpad=0.5,
    )
    figure.text(
        0.5,
        0.015,
        rf"Open circles mark the selected $\alpha={selected_alpha:g}$",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#4D4D4D",
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.23,
        top=0.79,
        wspace=0.31,
    )
    stem = (
        primary_stem
        if y_scale == "log"
        else f"{primary_stem}_linear_y"
    )
    outputs = save_figure(figure, output_dir, stem)
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stem",
        default="chignolin_ivm_fixed_alpha_sensitivity_total_nf",
    )
    args = parser.parse_args()

    scores_path = args.scores.expanduser().resolve()
    scores = json.loads(scores_path.read_text())
    selected_alpha = scores["selected_alpha"]
    if selected_alpha is None or scores["selection_rejected"]:
        raise RuntimeError("No preregistered production alpha was selected")
    rows = sorted(scores["records"], key=lambda row: float(row["alpha"]))
    alphas = np.asarray([row["alpha"] for row in rows], dtype=float)
    if (
        len(alphas) < 3
        or np.any(alphas <= 0.0)
        or np.any(np.diff(alphas) <= 0.0)
        or not np.any(np.isclose(alphas, float(selected_alpha)))
    ):
        raise RuntimeError("Alpha grid is incomplete or not strictly ordered")
    i_values = np.asarray(
        [
            [
                row[f"i_perp_trailing_mean_nf_le_{budget}"]
                for budget in FORCE_BUDGETS
            ]
            for row in rows
        ],
        dtype=float,
    )
    barrier_values = np.asarray(
        [
            [
                row[f"dense_barrier_rms_error_nf_le_{budget}"]
                for budget in FORCE_BUDGETS
            ]
            for row in rows
        ],
        dtype=float,
    )
    if (
        np.any(~np.isfinite(i_values))
        or np.any(i_values <= 0.0)
        or np.any(~np.isfinite(barrier_values))
        or np.any(barrier_values <= 0.0)
    ):
        raise RuntimeError("Sensitivity metrics must be positive and finite")

    output_dir = args.output_dir.expanduser().resolve()
    log_outputs = draw(
        alphas,
        float(selected_alpha),
        i_values,
        barrier_values,
        output_dir,
        "log",
        args.stem,
    )
    linear_outputs = draw(
        alphas,
        float(selected_alpha),
        i_values,
        barrier_values,
        output_dir,
        "linear",
        args.stem,
    )

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for alpha_index, alpha in enumerate(alphas):
        for budget_index, budget in enumerate(FORCE_BUDGETS):
            flat_rows.append(
                {
                    "alpha": float(alpha),
                    "requested_total_number_of_forces": int(budget),
                    "trailing_mean_unnormalized_i_perp": float(
                        i_values[alpha_index, budget_index]
                    ),
                    "trailing_rms_adaptive_dense_barrier_error_kcal": (
                        float(barrier_values[alpha_index, budget_index])
                    ),
                    "selected_alpha": bool(
                        np.isclose(alpha, float(selected_alpha))
                    ),
                }
            )
    csv_path = data_dir / f"{args.stem}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(flat_rows[0])
        )
        writer.writeheader()
        writer.writerows(flat_rows)
    summary = {
        "source_scores": str(scores_path),
        "source_scores_sha256": sha256(scores_path),
        "selected_alpha": float(selected_alpha),
        "force_budgets": list(FORCE_BUDGETS),
        "plotted_metrics": {
            "panel_a": (
                "Arithmetic mean of every measured unnormalized I_perp "
                "record in the last 30% of the actually attained N_F<=B "
                "window."
            ),
            "panel_b": (
                "RMS error over seven real-OpenMM adaptively densified "
                "barriers in the same trailing window."
            ),
        },
        "barrier_densification": (
            "10 and 20 linear Cartesian subdivisions per adjacent-image "
            "segment; 40/80 if successive barriers differ by >0.05 kcal/mol."
        ),
        "sparse_22_image_barrier_used": False,
        "curve_definition": (
            "Every measured alpha is shown and adjacent alpha values are "
            "connected directly in increasing order; no regression, "
            "smoothing, interpolation, or point removal."
        ),
        "i_perp_strict_single_well_all_budgets": scores[
            "i_perp_strict_single_well_all_budgets"
        ],
        "figure_files": {
            "log_y": log_outputs,
            "linear_y": linear_outputs,
        },
        "data_file": str(csv_path.resolve()),
    }
    summary_path = output_dir / f"{args.stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
