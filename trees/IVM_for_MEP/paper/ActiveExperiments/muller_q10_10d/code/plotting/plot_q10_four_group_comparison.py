#!/usr/bin/env python3
"""Compare the four fixed-alpha Q10 Muller IVM experiment groups."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


GROUPS = (
    ("exact_no_resample", "Group 1: exact IVM\nno resample", 1e-3),
    ("exact_resample", "Group 2: exact IVM\n+ cubic resample", 1e-3),
    ("inexact_baseline_resample", "Group 3: baseline inexact IVM\n+ cubic resample", 1e-3),
    ("inexact_tuned_resample", "Group 4: tuned inexact IVM\n+ cubic resample", 1e-3),
)
COMMON_ALPHAS = (
    1e-4, 3e-4, 1e-3, 1.3e-3, 3e-3, 1e-2, 3e-2, 1e-1,
    3e-1, 1.0, 3.0, 10.0, 100.0, 1000.0, 10000.0,
)
EXACT_R = 0.1
SELECTED_ALPHA = 1e-3
GROUP_COLORS = ("#0072B2", "#D55E00", "#5B9279", "#CC79A7")


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.7,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def load_rows(stage: Path, group: str) -> list[dict]:
    with (stage / "reports" / f"q10_{group}_summary.json").open() as handle:
        rows = json.load(handle)
    if group.startswith("exact_"):
        rows = [
            row for row in rows
            if np.isclose(row["config"]["kkt_rtol"], EXACT_R)
        ]
    return rows


def metric_arrays(stage: Path, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(stage / row["output"], allow_pickle=False)
    forces = np.asarray(data["total_force_evaluations"], dtype=float)
    i_perp = np.maximum(np.asarray(data["i_perp"], dtype=float), 1e-30)
    reference = float(np.ravel(data["reference_barrier"])[0])
    barrier = np.maximum(
        np.abs(np.asarray(data["delta_e"], dtype=float) - reference), 1e-16
    )
    return forces, i_perp, barrier


def alpha_color(alpha: float):
    lo = np.log10(min(COMMON_ALPHAS))
    hi = np.log10(max(COMMON_ALPHAS))
    value = (np.log10(alpha) - lo) / (hi - lo)
    return plt.get_cmap("cividis")(0.05 + 0.90 * value)


def format_axis(axis, ylabel: str | None = None) -> None:
    axis.set_xscale("linear")
    axis.set_yscale("log")
    axis.set_xlim(0.0, 5000.0)
    axis.grid(True, which="major", color="#d8d8d8", linewidth=0.5, alpha=0.65)
    axis.set_xlabel("Number of Forces")
    if ylabel:
        axis.set_ylabel(ylabel)


def alpha_overview(stage: Path, output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(2, 4, figsize=(14.6, 5.35), constrained_layout=True)
    available_alphas = set()
    for column, (group, title, _) in enumerate(GROUPS):
        rows = load_rows(stage, group)
        allowed = set(COMMON_ALPHAS)
        if group == "inexact_tuned_resample":
            allowed.add(SELECTED_ALPHA)
        rows = [
            row for row in rows
            if any(np.isclose(row["config"]["alpha"], alpha) for alpha in allowed)
        ]
        for row in sorted(rows, key=lambda item: item["config"]["alpha"]):
            alpha = float(row["config"]["alpha"])
            available_alphas.add(alpha)
            forces, i_perp, barrier = metric_arrays(stage, row)
            selected = group == "inexact_tuned_resample" and np.isclose(
                alpha, SELECTED_ALPHA
            )
            style = {
                "color": "#D55E00" if selected else alpha_color(alpha),
                "linewidth": 2.35 if selected else 1.05,
                "alpha": 1.0 if selected else 0.78,
                "zorder": 30 if selected else 3,
            }
            axes[0, column].plot(forces, i_perp, **style)
            axes[1, column].plot(forces, barrier, **style)
        exact_suffix = rf", $r={EXACT_R:g}$" if group.startswith("exact_") else ""
        axes[0, column].set_title(title + exact_suffix)
        format_axis(axes[0, column], r"$I_{\perp}$" if column == 0 else None)
        format_axis(
            axes[1, column],
            r"$|\Delta E_{\mathrm{dense}}-\Delta E_{\mathrm{ref}}|$"
            if column == 0 else None,
        )
    legend_alphas = sorted(available_alphas)
    handles = [
        Line2D([0], [0], color=("#D55E00" if np.isclose(alpha, SELECTED_ALPHA)
                               else alpha_color(alpha)),
               linewidth=2.35 if np.isclose(alpha, SELECTED_ALPHA) else 1.2,
               label=rf"$\alpha={alpha:g}$")
        for alpha in legend_alphas
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.015),
               ncol=8, frameon=False)
    fig.suptitle("Fixed-Q 10D Muller-22: four fixed-alpha IVM groups", y=1.025)
    return save_all(fig, output_dir / "q10_four_group_alpha_overview", stage)


def selected_overlay(stage: Path, output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.05), constrained_layout=True)
    selected_records = []
    for index, (group, title, alpha) in enumerate(GROUPS):
        rows = load_rows(stage, group)
        row = next(
            item for item in rows if np.isclose(item["config"]["alpha"], alpha)
        )
        forces, i_perp, barrier = metric_arrays(stage, row)
        label = title.replace("\n", " ") + rf", $\alpha={alpha:g}$"
        if group.startswith("exact_"):
            label += rf", $r={EXACT_R:g}$"
        style = {
            "color": GROUP_COLORS[index],
            "linewidth": 2.3 if group == "inexact_tuned_resample" else 1.65,
            "zorder": 20 if group == "inexact_tuned_resample" else 3 + index,
            "label": label,
        }
        axes[0].plot(forces, i_perp, **style)
        axes[1].plot(forces, barrier, **style)
        selected_records.append({
            "group": group, "alpha": alpha,
            "r": EXACT_R if group.startswith("exact_") else None,
            "output": row["output"],
        })
    format_axis(axes[0], r"$I_{\perp}$")
    format_axis(axes[1], r"$|\Delta E_{\mathrm{dense}}-\Delta E_{\mathrm{ref}}|$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.035),
               ncol=2, frameon=False)
    fig.suptitle("Representative comparison of the four IVM groups", y=1.035)
    outputs = save_all(fig, output_dir / "q10_four_group_selected_comparison", stage)
    return outputs, selected_records


def save_all(fig, stem: Path, stage: Path) -> list[str]:
    outputs = []
    for extension in ("pdf", "svg", "png"):
        output = Path(f"{stem}.{extension}")
        fig.savefig(output, dpi=600 if extension == "png" else None)
        outputs.append(str(output.relative_to(stage)))
    plt.close(fig)
    return outputs


def main() -> None:
    set_style()
    stage = Path(__file__).resolve().parents[1]
    output_dir = stage / "figures" / "four_group_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_outputs = alpha_overview(stage, output_dir)
    selected_outputs, selected_records = selected_overlay(stage, output_dir)
    with (output_dir / "q10_four_group_comparison_manifest.json").open("w") as handle:
        json.dump({
            "case": "fixed-Q unperturbed 10D Muller-Brown",
            "images_total": 22,
            "x_axis": "Number of Forces",
            "cost_definition": "mean uncached force calls per internal image",
            "x_scale": "linear",
            "x_limit": [0, 1000],
            "y_scale": "log",
            "exact_r_shown_in_overview": EXACT_R,
            "exact_r_full_grid": [0.2, 0.1, 0.05],
            "metrics": ["I_perp", "500-point dense Delta E error"],
            "overview_outputs": overview_outputs,
            "selected_outputs": selected_outputs,
            "selected_records": selected_records,
        }, handle, indent=2, sort_keys=True)
    print(output_dir / "q10_four_group_alpha_overview.pdf")
    print(output_dir / "q10_four_group_selected_comparison.pdf")


if __name__ == "__main__":
    main()
