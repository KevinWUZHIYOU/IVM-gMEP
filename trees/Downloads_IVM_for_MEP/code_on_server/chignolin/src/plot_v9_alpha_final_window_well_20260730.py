#!/usr/bin/env python3
"""Paper-grade V9 fixed-alpha final-window sensitivity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BUDGET_COLORS = ("#0072B2", "#D55E00", "#009E73", "#7A5195")
BUDGET_MARKERS = ("o", "s", "^", "D")
BUDGET_LINESTYLES = ("-", "--", "-.", ":")
SELECTED_ALPHA = 1.0


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


def strict_single_well(values: np.ndarray, center: int) -> bool:
    values = np.asarray(values, dtype=float)
    return bool(
        np.all(np.diff(values[: center + 1]) < 0.0)
        and np.all(np.diff(values[center:]) > 0.0)
    )


def moderate_unique_interior_well(
    values: np.ndarray, alphas: np.ndarray
) -> bool:
    local_minima = [
        index
        for index in range(1, len(values) - 1)
        if values[index] < values[index - 1]
        and values[index] < values[index + 1]
    ]
    global_index = int(np.argmin(values))
    return bool(
        len(local_minima) == 1
        and local_minima[0] == global_index
        and 0.75 <= alphas[global_index] <= 1.0
    )


def draw(
    alphas: np.ndarray,
    budgets: tuple[int, ...],
    i_values: np.ndarray,
    barrier_values: np.ndarray,
    output_dir: Path,
    stem: str,
    yscale: str,
) -> list[str]:
    setup_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.12))
    measures = (i_values, barrier_values)
    ylabels = (
        r"Trailing mean unnormalized $I_\perp$",
        "Trailing RMS dense-barrier error\n"
        r"(kcal mol$^{-1}$)",
    )
    selected_index = int(
        np.flatnonzero(np.isclose(alphas, SELECTED_ALPHA))[0]
    )
    for panel, (axis, measured, ylabel) in enumerate(
        zip(axes, measures, ylabels)
    ):
        for budget_index, budget in enumerate(budgets):
            axis.plot(
                alphas,
                measured[:, budget_index],
                color=BUDGET_COLORS[budget_index],
                marker=BUDGET_MARKERS[budget_index],
                linestyle=BUDGET_LINESTYLES[budget_index],
                ms=3.8,
                markerfacecolor="white",
                markeredgewidth=0.8,
                label=rf"$N_F\leq {budget:,}$",
                zorder=3 + budget_index,
            )
        axis.axvline(
            SELECTED_ALPHA,
            color="#333333",
            lw=0.95,
            ls=(0, (2.0, 2.0)),
            zorder=2,
        )
        axis.axvspan(
            0.75,
            1.0,
            color="#BDBDBD",
            alpha=0.10,
            lw=0,
            zorder=1,
        )
        axis.scatter(
            np.full(len(budgets), SELECTED_ALPHA),
            measured[selected_index],
            s=40,
            facecolors="none",
            edgecolors="#222222",
            linewidths=0.95,
            zorder=10,
        )
        axis.set_xscale("log")
        axis.set_yscale(yscale)
        axis.set_xlim(float(alphas[0]) / 1.18, float(alphas[-1]) * 1.18)
        axis.set_xticks(alphas)
        axis.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        axis.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        axis.set_xlabel(r"Fixed proximal parameter $\alpha$")
        axis.set_ylabel(ylabel)
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + panel))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=len(budgets),
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
        r"Black rings mark the selected $\alpha=1$",
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
    suffix = "" if yscale == "log" else "_linear_y"
    outputs = save_figure(figure, output_dir, stem + suffix)
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-summary", type=Path, required=True)
    parser.add_argument("--dense-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stem",
        default="chignolin_ivm_fixed_alpha_sensitivity_total_nf",
    )
    args = parser.parse_args()

    summary_path = args.grid_summary.expanduser().resolve()
    dense_path = args.dense_manifest.expanduser().resolve()
    summary = json.loads(summary_path.read_text())
    dense = json.loads(dense_path.read_text())
    budgets = tuple(
        int(value) for value in summary["budgets_total_number_of_forces"]
    )
    if budgets != tuple(
        int(value) for value in dense["requested_total_force_budgets"]
    ):
        raise RuntimeError("Optimization and dense budget grids differ")
    rows = sorted(summary["records"], key=lambda row: float(row["alpha"]))
    dense_by_alpha = {
        float(row["alpha"]): row for row in dense["records"]
    }
    alphas = np.asarray([float(row["alpha"]) for row in rows])
    if (
        len(budgets) != 4
        or len(BUDGET_COLORS) != len(budgets)
        or not np.any(np.isclose(alphas, SELECTED_ALPHA))
    ):
        raise RuntimeError("Unexpected alpha/budget grid")

    i_values = np.asarray(
        [
            [
                record["budget_records"][budget_index][
                    "trailing_integral_mean_unnormalized_i_perp"
                ]
                for budget_index in range(len(budgets))
            ]
            for record in rows
        ],
        dtype=float,
    )
    barrier_values = np.asarray(
        [
            dense_by_alpha[float(alpha)][
                "trailing_rms_barrier_error_kcal_by_budget"
            ]
            for alpha in alphas
        ],
        dtype=float,
    )
    if (
        np.any(~np.isfinite(i_values))
        or np.any(i_values <= 0.0)
        or np.any(~np.isfinite(barrier_values))
        or np.any(barrier_values <= 0.0)
    ):
        raise RuntimeError(
            "Alpha-sensitivity scores must be positive and finite"
        )
    selected_index = int(
        np.flatnonzero(np.isclose(alphas, SELECTED_ALPHA))[0]
    )
    i_wells = [
        strict_single_well(i_values[:, index], selected_index)
        for index in range(len(budgets))
    ]
    i_moderate_wells = [
        bool(
            moderate_unique_interior_well(
                i_values[:, index], alphas
            )
            and np.isclose(
                alphas[int(np.argmin(i_values[:, index]))],
                SELECTED_ALPHA,
            )
        )
        for index in range(len(budgets))
    ]
    barrier_wells = [
        strict_single_well(barrier_values[:, index], selected_index)
        for index in range(len(budgets))
    ]
    barrier_moderate_wells = [
        moderate_unique_interior_well(
            barrier_values[:, index], alphas
        )
        for index in range(len(budgets))
    ]

    output_dir = args.output_dir.expanduser().resolve()
    log_outputs = draw(
        alphas,
        budgets,
        i_values,
        barrier_values,
        output_dir,
        args.stem,
        "log",
    )
    linear_outputs = draw(
        alphas,
        budgets,
        i_values,
        barrier_values,
        output_dir,
        args.stem,
        "linear",
    )

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    flat = []
    for alpha_index, alpha in enumerate(alphas):
        for budget_index, budget in enumerate(budgets):
            flat.append(
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
                        np.isclose(alpha, SELECTED_ALPHA)
                    ),
                }
            )
    csv_path = data_dir / f"{args.stem}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    report = {
        "source_grid_summary": str(summary_path),
        "source_grid_summary_sha256": sha256(summary_path),
        "source_dense_manifest": str(dense_path),
        "source_dense_manifest_sha256": sha256(dense_path),
        "selected_alpha": SELECTED_ALPHA,
        "force_budgets": list(budgets),
        "primary_statistics": {
            "panel_a": (
                "Force-axis arithmetic mean of complete-macrostate "
                "unnormalized I_perp over the final 30% of the requested "
                "budget through the complete endpoint."
            ),
            "panel_b": (
                "RMS error of seven force-uniform, real-OpenMM adaptively "
                "densified barriers in the final 30% force window."
            ),
        },
        "barrier_densification": dense["barrier_protocol"],
        "sparse_22_image_barrier_used": False,
        "diagnostic_calls_excluded_from_optimization_axis": True,
        "curve_definition": (
            "Every measured alpha is connected directly in increasing order; "
            "no fit, regression, smoothing, interpolation, point deletion, "
            "or missing-value carry-forward."
        ),
        "i_perp_strict_single_well_by_budget": i_wells,
        "i_perp_unique_interior_well_at_alpha_1_by_budget": (
            i_moderate_wells
        ),
        "dense_barrier_strict_single_well_by_budget": barrier_wells,
        "dense_barrier_unique_moderate_interior_well_by_budget": (
            barrier_moderate_wells
        ),
        "i_perp_strict_single_well_all_budgets": bool(all(i_wells)),
        "i_perp_unique_interior_well_at_alpha_1_all_budgets": bool(
            all(i_moderate_wells)
        ),
        "dense_barrier_strict_single_well_all_budgets": bool(
            all(barrier_wells)
        ),
        "dense_barrier_unique_moderate_interior_well_all_budgets": bool(
            all(barrier_moderate_wells)
        ),
        "figures": {"log_y": log_outputs, "linear_y": linear_outputs},
        "data_file": str(csv_path),
    }
    report_path = output_dir / f"{args.stem}_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
