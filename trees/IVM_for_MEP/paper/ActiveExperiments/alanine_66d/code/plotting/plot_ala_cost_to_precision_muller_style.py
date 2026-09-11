#!/usr/bin/env python3
"""Build audited Alanine cost-to-precision candidates from the current data.

The candidates retain the clean Stage-IV v4 Pareto IVM series and the real
long-budget baseline histories used by the official linear-cost figure.  They
change only the presentation to 1x3 or metric-only 1x2 cost-to-precision form.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np


REPO = Path(__file__).resolve().parents[5]
STUDY = REPO / "python/alanine_ivm_clean_four_level_20260801"
LEGACY = REPO / "python/alanine_v10_chignolin_v9_fixed_alpha_20260730"
SOURCE_DIR = STUDY / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import plot_all_methods_linear_speedup_v5 as current  # noqa: E402


OUTPUT_DIR = REPO / "paper/ActiveExperiments/alanine_66d/figures/active_candidates"
LEGACY_REFERENCE_PDF = (
    REPO
    / "paper/Figures/ala2_66d/archive_legacy_pre_clean_20260802"
    / "top_level_old_paper_assets/fig_1x3_cost_to_precision_legacy_pre_v49.pdf"
)
OFFICIAL_SOURCE_PDF = (
    REPO
    / "paper/Figures/ala2_66d/official_20260802"
    / "alanine_all_methods_two_metric_descent_linear_cost_speedup.pdf"
)
RAMA_CACHE = LEGACY / "formal_figure_inputs/ala2_scan_cache.pkl"
IPERP_COMPARISON_THRESHOLD = 2.1e-4
BARRIER_COMPARISON_TARGET = 2.4721831748734475
DISPLAY_RANGES = {
    "1x3": {"i_perp": (1.0e-4, 1.0e0), "barrier": (2.3, 4.0)},
    "1x2": {"i_perp": (1.0e-4, 1.0e-1), "barrier": (2.4, 3.0)},
}
IVM_COMBINED_PATH = (
    STUDY
    / "runs/stage4_inexact_optimized_resample"
    / "stage4_uniform_v3_a20_phaseA34_phaseB220.npz"
)
IVM_ARCHIVE_PATH = (
    STUDY
    / "runs/pareto_archive"
    / "pareto_clean_stage4_v4_a20_phaseA34_phaseB220.npz"
)
IVM_DENSE_PREFIX_PATH = (
    STUDY
    / "runs/diagnostics"
    / "dense_barrier_clean_stage4_v4_a20_phaseA34_preactivation.npz"
)
IVM_DENSE_SCREEN_PATH = (
    STUDY
    / "runs/diagnostics"
    / "dense_barrier_clean_stage4_pareto_screen_a20_phaseA34_phaseB220.npz"
)
ODE_METRICS_PATH = (
    LEGACY
    / "formal_figure_inputs"
    / "metrics_ala_0to1_N22_k50_alpha0.0001_ode_results.npz"
)
ODE_RESULT_PATH = (
    LEGACY
    / "formal_figure_inputs"
    / "ala_0to1_N22_k50_alpha0.0001_ode_results.npz"
)
COMMON_INITIAL_PATH = STUDY / "inputs/common_initial_path_equidistant.npz"


METHOD_ORDER = (
    "ivm",
    "ode_neb",
    "static_neb",
    "original_string",
    "obsm_fire",
    "obsm_bfgs",
)
METHOD_STYLE = {
    "ivm": {
        "label": r"IVM ($\alpha=20$)",
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": 1.85,
        "marker": "o",
        "markersize": 3.1,
        "zorder": 20,
    },
    "ode_neb": {
        "label": "ODE-NEB",
        "color": "#D55E00",
        "linestyle": "--",
        "linewidth": 1.65,
        "marker": "s",
        "markersize": 2.9,
        "zorder": 6,
    },
    "static_neb": {
        "label": "Static NEB",
        "color": "#B279A2",
        "linestyle": "--",
        "linewidth": 1.65,
        "marker": "v",
        "markersize": 2.8,
        "zorder": 4,
    },
    "original_string": {
        "label": "Original String",
        "color": "#777777",
        "linestyle": ":",
        "linewidth": 1.65,
        "marker": "D",
        "markersize": 2.7,
        "zorder": 3,
    },
    "obsm_fire": {
        "label": "OBSM-FIRE",
        "color": "#5B9279",
        "linestyle": "-.",
        "linewidth": 1.65,
        "marker": "s",
        "markersize": 2.8,
        "zorder": 4,
    },
    "obsm_bfgs": {
        "label": "OBSM-BFGS",
        "color": "#CC79A7",
        "linestyle": (0, (5, 1.5)),
        "linewidth": 1.65,
        "marker": "P",
        "markersize": 2.8,
        "zorder": 5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--layout",
        choices=("1x3", "1x2"),
        default="1x3",
        help="Use 1x2 to omit the independently selected path panel.",
    )
    parser.add_argument(
        "--scope",
        choices=("both", "all_methods", "ivm_ode"),
        default="both",
        help="Limit output to one method scope when updating provenance.",
    )
    parser.add_argument(
        "--provenance-only",
        action="store_true",
        help="Refresh CSV/manifest without rewriting an existing PDF.",
    )
    return parser.parse_args()


def mm_to_inch(value: float) -> float:
    return value / 25.4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.6,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "it",
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "legend.frameon": False,
            "legend.handlelength": 2.2,
            "legend.handletextpad": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 220,
            "savefig.dpi": 600,
            "savefig.bbox": None,
            "savefig.transparent": False,
        }
    )


def load_data():
    module, methods, disclosure = current.load_methods(STUDY, REPO)
    by_key = {method.key: method for method in methods}
    ordered = [by_key[key] for key in METHOD_ORDER]
    if not LEGACY_REFERENCE_PDF.exists():
        raise FileNotFoundError(LEGACY_REFERENCE_PDF)
    if not OFFICIAL_SOURCE_PDF.exists():
        raise FileNotFoundError(OFFICIAL_SOURCE_PDF)
    if not RAMA_CACHE.exists():
        raise FileNotFoundError(RAMA_CACHE)
    with RAMA_CACHE.open("rb") as handle:
        rama = module._ContourCacheUnpickler(handle).load()
    return module, ordered, disclosure, rama


def metric_history(module, method, metric: str):
    if metric == "i_perp":
        counts = np.asarray(method.force, dtype=float)
        values = np.asarray(method.iperp, dtype=float)
    elif metric == "barrier":
        counts = np.asarray(method.barrier_force, dtype=float)
        values = np.asarray(method.barrier, dtype=float)
    elif metric == "barrier_error":
        counts = np.asarray(method.barrier_force, dtype=float)
        values = np.abs(
            np.asarray(method.barrier, dtype=float) - module.REFERENCE_BARRIER_KCAL
        )
    else:
        raise ValueError(metric)
    finite = np.isfinite(counts) & np.isfinite(values) & (counts > 0) & (values > 0)
    return counts[finite], values[finite]


def cost_to_precision(module, method, metric: str, thresholds: np.ndarray):
    counts, values = metric_history(module, method, metric)
    return module.cost_to_reach(counts, values, thresholds, min_count=1)


def pareto_record_points(module, method, metric: str) -> dict:
    """Return checkpoints that establish a strictly lower plotted-metric record."""
    counts, plotted_values = metric_history(module, method, metric)
    scores = plotted_values.copy()
    order = np.argsort(counts, kind="mergesort")
    counts = counts[order]
    plotted_values = plotted_values[order]
    scores = scores[order]

    unique_counts, first = np.unique(counts, return_index=True)
    group_ends = np.r_[first[1:], len(counts)]
    best_score_at_count = []
    plotted_value_at_count = []
    for start, end in zip(first, group_ends):
        local = int(start + np.argmin(scores[start:end]))
        best_score_at_count.append(float(scores[local]))
        plotted_value_at_count.append(float(plotted_values[local]))

    record_metric_value = []
    record_score = []
    record_cost = []
    running_best = np.inf
    for count, score, plotted_value in zip(
        unique_counts, best_score_at_count, plotted_value_at_count
    ):
        if score < running_best:
            running_best = float(score)
            record_metric_value.append(float(plotted_value))
            record_score.append(running_best)
            record_cost.append(float(count))
    return {
        "metric_value": np.asarray(record_metric_value, dtype=float),
        "selection_error": np.asarray(record_score, dtype=float),
        "minimum_NF": np.asarray(record_cost, dtype=float),
    }


def force_formatter(value: float, _position: int) -> str:
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def legend_handle(key: str) -> Line2D:
    style = METHOD_STYLE[key]
    return Line2D(
        [0],
        [0],
        label=style["label"],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        marker=style["marker"],
        markersize=style["markersize"],
        markerfacecolor="white",
        markeredgewidth=0.6,
    )


def initial_handle() -> Line2D:
    return Line2D(
        [0],
        [0],
        label="Initial (IDPP)",
        color="#777777",
        linestyle=":",
        linewidth=1.45,
        marker="D",
        markersize=2.7,
        markerfacecolor="white",
        markeredgewidth=0.55,
    )


def annotate_speedup(
    axis,
    threshold: float,
    low_cost: float,
    high_cost: float,
    label: str,
    label_x=None,
) -> None:
    axis.axvline(
        threshold, color="#767676", linewidth=0.55, linestyle=":", zorder=1
    )
    axis.annotate(
        "",
        xy=(threshold, low_cost),
        xytext=(threshold, high_cost),
        arrowprops={"arrowstyle": "<->", "color": "#202020", "lw": 0.8},
        zorder=25,
    )
    axis.text(
        threshold * 1.38 if label_x is None else label_x,
        0.5 * (low_cost + high_cost),
        label,
        ha="right",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color="#202020",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.8},
        zorder=26,
    )


def pairwise_comparison_record(
    module, methods, metric: str, threshold: float
) -> dict:
    """Compute an explicit marginal IVM-versus-ODE first-passage ratio."""
    by_key = {method.key: method for method in methods}
    ivm_cost = cost_to_precision(
        module, by_key["ivm"], metric, np.asarray([threshold], dtype=float)
    )[0]
    ode_cost = cost_to_precision(
        module, by_key["ode_neb"], metric, np.asarray([threshold], dtype=float)
    )[0]
    if not (np.isfinite(ivm_cost) and np.isfinite(ode_cost)):
        raise RuntimeError(
            f"Missing observed {metric} passage at comparison threshold {threshold:g}"
        )
    return {
        "metric": metric,
        "target": float(threshold),
        "ivm_marginal_first_passage_NF": int(ivm_cost),
        "ode_neb_marginal_first_passage_NF": int(ode_cost),
        "ode_neb_over_ivm": float(ode_cost / ivm_cost),
        "passages_are_metric_specific": True,
        "simultaneous_same_checkpoint_claim": False,
    }


def style_cost_axis(axis, metric: str, y_max: float, layout: str) -> None:
    display_range = DISPLAY_RANGES[layout][metric]
    axis.set_yscale("linear")
    axis.set_ylim(0, y_max)
    if y_max >= 1.0e5:
        axis.set_yticks((0, 25000, 50000, 75000, 100000))
    elif y_max >= 6.0e4:
        axis.set_yticks((0, 15000, 30000, 45000, 60000))
    else:
        axis.set_yticks((0, 5000, 10000, 15000, 20000))
    axis.yaxis.set_major_formatter(FuncFormatter(force_formatter))
    axis.grid(True, which="major", color="#D9D9D6", linewidth=0.4, alpha=0.62)
    axis.set_ylabel(r"Whole-path force evaluations, $N_F$")
    axis.set_box_aspect(1.0)
    if metric == "i_perp":
        axis.set_xscale("log")
        axis.set_xlim(display_range[1], display_range[0])
        axis.set_xlabel(r"$I_\perp$")
    else:
        axis.set_xscale("log")
        axis.set_xlim(display_range[1], display_range[0])
        axis.set_xticks(
            (3, 2.8, 2.6, 2.5, 2.4)
            if layout == "1x2"
            else (4, 3.5, 3, 2.5, 2.3)
        )
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel(r"Densified $\Delta E$ (kcal mol$^{-1}$)")


def draw_path_panel(module, axis, methods, rama) -> object:
    phi_grid, psi_grid = np.meshgrid(
        np.asarray(rama["phi_g"], dtype=float),
        np.asarray(rama["psi_g"], dtype=float),
        indexing="ij",
    )
    pes = np.asarray(rama["PES"], dtype=float)
    energy_grid = (pes - float(np.nanmin(pes))) * float(module.EV_TO_KCAL)
    contour = axis.contourf(
        phi_grid,
        psi_grid,
        energy_grid,
        levels=np.linspace(0.0, 15.0, 31),
        cmap="Greys_r",
        alpha=0.92,
        extend="max",
        zorder=-10,
    )
    axis.contour(
        phi_grid,
        psi_grid,
        energy_grid,
        levels=np.linspace(0.0, 15.0, 9),
        colors="0.25",
        linewidths=0.35,
        alpha=0.78,
        zorder=-5,
    )
    initial = np.asarray(methods[0].initial_path, dtype=float)
    initial_phi, initial_psi = module.path_to_dihedrals(initial)
    axis.plot(
        initial_phi,
        initial_psi,
        color="#777777",
        linestyle=":",
        linewidth=1.45,
        marker="D",
        markersize=2.7,
        markerfacecolor="white",
        markeredgewidth=0.55,
        zorder=10,
    )
    for method in reversed(methods):
        style = METHOD_STYLE[method.key]
        phi, psi = module.path_to_dihedrals(method.selected_path)
        axis.plot(
            phi,
            psi,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"],
            markerfacecolor="white",
            markeredgewidth=0.45,
            zorder=30 + int(style["zorder"]),
        )
    axis.scatter(
        initial_phi[0],
        initial_psi[0],
        s=30,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=0.9,
        zorder=80,
    )
    axis.scatter(
        initial_phi[-1],
        initial_psi[-1],
        s=58,
        marker="*",
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=85,
    )
    axis.set_xlim(-180, 0)
    axis.set_ylim(0, 180)
    axis.set_xticks((-180, -120, -60, 0))
    axis.set_yticks((0, 60, 120, 180))
    axis.set_xlabel(r"$\phi$ (deg)")
    axis.set_ylabel(r"$\psi$ (deg)")
    axis.set_box_aspect(1.0)
    return contour


def draw_figure(
    module,
    all_methods,
    rama,
    scope: str,
    output: Path,
    layout: str,
    write_pdf: bool = True,
):
    if scope == "all_methods":
        methods = list(all_methods)
    elif scope == "ivm_ode":
        methods = [
            method for method in all_methods if method.key in {"ivm", "ode_neb"}
        ]
    else:
        raise ValueError(scope)

    curves = {}
    for method in methods:
        curves[method.key] = {
            "i_perp": pareto_record_points(module, method, "i_perp"),
            "barrier": pareto_record_points(module, method, "barrier"),
        }

    figure = plt.figure(figsize=(mm_to_inch(180), mm_to_inch(72)), dpi=220)
    if layout == "1x3":
        grid = GridSpec(
            1,
            4,
            figure=figure,
            width_ratios=(1.0, 1.0, 1.05, 0.045),
            left=0.066,
            right=0.925,
            bottom=0.17,
            top=0.72,
            wspace=0.40,
        )
        axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
        colorbar_axis = figure.add_subplot(grid[0, 3])
    elif layout == "1x2":
        grid = GridSpec(
            1,
            2,
            figure=figure,
            left=0.065,
            right=0.975,
            bottom=0.17,
            top=0.72,
            wspace=0.24,
        )
        axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
        colorbar_axis = None
    else:
        raise ValueError(layout)

    for method in methods:
        style = METHOD_STYLE[method.key]
        for axis, curve in (
            (axes[0], curves[method.key]["i_perp"]),
            (axes[1], curves[method.key]["barrier"]),
        ):
            axis.plot(
                curve["metric_value"],
                curve["minimum_NF"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor="white",
                markeredgewidth=0.55,
                markevery=max(1, len(curve["metric_value"]) // 8),
                alpha=1.0 if method.key == "ivm" else 0.92,
                zorder=style["zorder"],
            )
    style_cost_axis(axes[0], "i_perp", 1.1e5, layout)
    barrier_y_max = 6.5e4 if scope == "all_methods" else 2.2e4
    style_cost_axis(axes[1], "barrier", barrier_y_max, layout)
    if layout == "1x2":
        for axis in axes:
            axis.set_box_aspect(None)

    iperp_comparison = pairwise_comparison_record(
        module, all_methods, "i_perp", IPERP_COMPARISON_THRESHOLD
    )
    barrier_comparison = pairwise_comparison_record(
        module, all_methods, "barrier", BARRIER_COMPARISON_TARGET
    )
    annotate_speedup(
        axes[0],
        IPERP_COMPARISON_THRESHOLD,
        iperp_comparison["ivm_marginal_first_passage_NF"],
        iperp_comparison["ode_neb_marginal_first_passage_NF"],
        "ODE-NEB / IVM\n"
        + rf"{iperp_comparison['ode_neb_over_ivm']:.1f}$\times$",
    )
    annotate_speedup(
        axes[1],
        BARRIER_COMPARISON_TARGET,
        barrier_comparison["ivm_marginal_first_passage_NF"],
        barrier_comparison["ode_neb_marginal_first_passage_NF"],
        "ODE-NEB / IVM\n"
        + rf"{barrier_comparison['ode_neb_over_ivm']:.1f}$\times$",
        label_x=3.05 if layout == "1x3" else 2.78,
    )

    if layout == "1x3":
        contour = draw_path_panel(module, axes[2], methods, rama)
        axes[2].set_title("Independently selected paths", fontsize=6.5, pad=2.0)
        colorbar = figure.colorbar(contour, cax=colorbar_axis, extend="max")
        colorbar.set_ticks((0, 3, 6, 9, 12, 15))
        colorbar.ax.tick_params(labelsize=6.2, width=0.65, length=2.6)
        colorbar.outline.set_linewidth(0.75)
        colorbar.set_label(
            r"$\Delta E$ (kcal mol$^{-1}$)", fontsize=6.8, labelpad=2.5
        )

    figure.canvas.draw()
    for axis, letter in zip(axes, "abc" if layout == "1x3" else "ab"):
        position = axis.get_position()
        figure.text(
            position.x0 - 0.027,
            position.y1 + 0.012,
            letter,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )

    handles = [legend_handle(method.key) for method in methods]
    if layout == "1x3":
        handles = [initial_handle()] + handles
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.49, 0.975),
        ncol=(4 if scope == "all_methods" else 3)
        if layout == "1x3"
        else (3 if scope == "all_methods" else 2),
        columnspacing=1.35,
        handlelength=2.35,
        handletextpad=0.5,
        frameon=False,
    )
    figure.text(
        0.49,
        0.785,
        (
            "Panels (a,b): minimum observed $N_F$ for each improved precision; "
            "(c) independently selected paths."
            if layout == "1x3"
            else "Minimum observed $N_F$ for each improved precision."
        ),
        ha="center",
        va="center",
        fontsize=6.2,
        color="#4A4A4A",
    )
    if write_pdf:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output,
            format="pdf",
            dpi=600,
            metadata={
                "Title": f"Alanine 66D cost to precision ({scope}, {layout})",
                "Subject": "Clean Stage-IV v4 and audited long-budget baselines",
            },
        )
    elif not output.exists():
        raise FileNotFoundError(output)
    plt.close(figure)
    return (
        methods,
        curves,
        iperp_comparison,
        barrier_comparison,
    )


def source_rows(module, scope: str, methods, curves, include_paths: bool):
    rows = []
    with np.load(IVM_ARCHIVE_PATH, allow_pickle=False) as archive:
        candidate_calls = np.asarray(archive["candidate_calls"], dtype=int)
        candidate_cost = np.asarray(
            archive["candidate_screen_force_equivalent_cost"], dtype=int
        )
        candidate_accepted = np.asarray(archive["candidate_accepted"], dtype=bool)
    accepted_charged_cost = set(
        (
            candidate_calls[candidate_accepted]
            + np.cumsum(candidate_cost)[candidate_accepted]
        ).tolist()
    )

    def point_source(method, metric: str, nf: int) -> tuple[str, Path]:
        if method.key == "ivm":
            accepted = nf in accepted_charged_cost
            if metric == "i_perp":
                return (
                    ("ivm_pareto_accepted", IVM_ARCHIVE_PATH)
                    if accepted
                    else ("ivm_combined_prefix", IVM_COMBINED_PATH)
                )
            return (
                ("ivm_dense_screen_accepted", IVM_DENSE_SCREEN_PATH)
                if accepted
                else ("ivm_dense_prefix", IVM_DENSE_PREFIX_PATH)
            )
        if method.key == "ode_neb" and metric == "i_perp":
            return "ode_metrics", ODE_METRICS_PATH
        return (
            ("raw_result", Path(method.result_path))
            if metric == "i_perp"
            else ("dense_diagnostic", Path(method.dense_path))
        )

    for method in methods:
        for metric in ("i_perp", "barrier"):
            curve = curves[method.key][metric]
            selection_errors = curve.get("selection_error")
            for index, (metric_value, value) in enumerate(
                zip(curve["metric_value"], curve["minimum_NF"])
            ):
                nf = int(value)
                source_stage, source = point_source(method, metric, nf)
                rows.append(
                    {
                        "record_type": "cost_to_precision_record",
                        "scope": scope,
                        "panel": "a" if metric == "i_perp" else "b",
                        "method": method.key,
                        "metric": metric,
                        "metric_value": float(metric_value),
                        "selection_error": (
                            float(selection_errors[index])
                            if selection_errors is not None
                            else ""
                        ),
                        "NF": nf,
                        "image_index": "",
                        "phi_deg": "",
                        "psi_deg": "",
                        "source_stage": source_stage,
                        "source_file": str(source),
                    }
                )
    if include_paths:
        for method_name, path, source in (
            (
                "initial",
                methods[0].initial_path,
                STUDY / "inputs/common_initial_path_equidistant.npz",
            ),
            *tuple(
                (method.key, method.selected_path, method.result_path)
                for method in methods
            ),
        ):
            phi, psi = module.path_to_dihedrals(path)
            for image_index, (phi_value, psi_value) in enumerate(zip(phi, psi)):
                rows.append(
                    {
                        "record_type": "selected_path",
                        "scope": scope,
                        "panel": "c",
                        "method": method_name,
                        "metric": "",
                        "metric_value": "",
                        "selection_error": "",
                        "NF": "",
                        "image_index": image_index,
                        "phi_deg": float(phi_value),
                        "psi_deg": float(psi_value),
                        "source_stage": "selected_path",
                        "source_file": str(source),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def input_manifest(methods) -> dict:
    paths = {
        "legacy_reference_pdf": LEGACY_REFERENCE_PDF,
        "official_linear_cost_pdf": OFFICIAL_SOURCE_PDF,
        "ramachandran_pes_cache": RAMA_CACHE,
        "common_initial": COMMON_INITIAL_PATH,
        "ivm_combined": IVM_COMBINED_PATH,
        "ivm_dense_prefix": IVM_DENSE_PREFIX_PATH,
        "ode_neb_metrics": ODE_METRICS_PATH,
    }
    for method in methods:
        paths[f"{method.key}_result"] = Path(method.result_path)
        paths[f"{method.key}_dense"] = Path(method.dense_path)
    return {
        key: {"path": str(path), "sha256": sha256(path)}
        for key, path in paths.items()
    }


def serializable_style(key: str) -> dict:
    result = dict(METHOD_STYLE[key])
    result["linestyle"] = str(result["linestyle"])
    return result


def plotting_code_manifest() -> dict:
    paths = {
        "candidate_plotter": Path(__file__).resolve(),
        "executable_loader": SOURCE_DIR / "plot_all_methods_linear_speedup_v5.py",
        "ivm_pareto_loader": SOURCE_DIR / "plot_stage4_v4_pareto_all_methods.py",
        "legacy_method_loader": (
            LEGACY / "src/plot_ala_integer20_all_methods.py"
        ),
    }
    return {
        key: {"path": str(path), "sha256": sha256(path)}
        for key, path in paths.items()
    }


def initial_path_audit() -> dict:
    with np.load(COMMON_INITIAL_PATH, allow_pickle=False) as saved:
        common = np.asarray(saved["initial_path"], dtype=float)
    with np.load(ODE_RESULT_PATH, allow_pickle=False) as saved:
        ode_stored = np.asarray(saved["images_seq"][0], dtype=float)
    maximum_difference = float(np.max(np.abs(common - ode_stored)))
    return {
        "common_initial_array_sha256": array_sha256(common),
        "ode_stored_first_path_array_sha256": array_sha256(ode_stored),
        "maximum_absolute_coordinate_difference_angstrom": maximum_difference,
        "exact_same_initial_path": bool(maximum_difference <= 5.0e-7),
        "disclosure": (
            "The frozen ODE artifact is a historical mixed-generation input. "
            "Do not describe this six-method candidate as an exact same-IDPP-input "
            "comparison. The 1x2 metric panels do not plot paths."
        ),
    }


def main() -> None:
    args = parse_args()
    configure_style()
    module, all_methods, disclosure, rama = load_data()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = input_manifest(all_methods)
    outputs = []
    scopes = (
        ("all_methods", "ivm_ode")
        if args.scope == "both"
        else (args.scope,)
    )
    for scope in scopes:
        stem = (
            f"alanine66d_stage4_v4_cost_to_precision_"
            f"{args.layout}_{scope}_muller_style"
        )
        pdf_path = args.output_dir / f"{stem}.pdf"
        csv_path = args.output_dir / f"{stem}_source.csv"
        manifest_path = args.output_dir / f"{stem}_manifest.json"
        methods, curves, iperp_comparison, barrier_comparison = draw_figure(
            module,
            all_methods,
            rama,
            scope,
            pdf_path,
            args.layout,
            write_pdf=not args.provenance_only,
        )
        rows = source_rows(
            module,
            scope,
            methods,
            curves,
            include_paths=args.layout == "1x3",
        )
        write_csv(csv_path, rows)
        metric_panels = {
            "a": {
                "metric": "I_perp cost to precision",
                "display_range": list(DISPLAY_RANGES[args.layout]["i_perp"]),
                "marginal_comparison_threshold": IPERP_COMPARISON_THRESHOLD,
            },
            "b": {
                "metric": "actual densified Delta E cost to precision",
                "display_range": list(DISPLAY_RANGES[args.layout]["barrier"]),
                "marginal_comparison_target": BARRIER_COMPARISON_TARGET,
                "point_selection": (
                    "strict running-lowest actual Delta E records in increasing N_F"
                ),
            },
        }
        if args.layout == "1x3":
            metric_panels["c"] = {
                "metric": "selected paths on the frozen 0 K constrained-minimized PES",
                "path_methods": [method.key for method in methods],
                "checkpoint_relation": (
                    "independently selected paths; not tied to panel-a or panel-b "
                    "cost-to-precision record checkpoints"
                ),
            }
        manifest = {
            "status": "active candidate; not cited by paper/main.tex",
            "change_scope": (
                "Panels a and b are Pareto-record cost-to-precision curves for lower "
                "I_perp and lower actual densified Delta E, respectively. No dynamics "
                "or selected scientific input changed. "
                + (
                    "The independent selected-path panel is retained."
                    if args.layout == "1x3"
                    else "The independent selected-path panel is omitted."
                )
            ),
            "scope": scope,
            "layout": f"{args.layout} at 180 mm publication width",
            "path_panel_included": args.layout == "1x3",
            "pdf_rewritten_in_this_run": not args.provenance_only,
            "axis_scales": {
                "panel_a_x": "reversed logarithmic",
                "panel_b_x": "reversed logarithmic",
                "whole_path_N_F_y": "linear",
            },
            "cost_definition": (
                "For each method and metric, points are processed in increasing N_F; "
                "a checkpoint is retained only when it strictly lowers the running-best "
                "metric. Thus each vertex is the minimum observed N_F needed to reach "
                "that I_perp or actual densified Delta E. Lines connect only observed "
                "record vertices, with no interpolation used in the cost calculation."
            ),
            "metric_panels": metric_panels,
            "comparison_audit": {
                "i_perp": iperp_comparison,
                "densified_barrier": barrier_comparison,
            },
            "comparison_scope_note": (
                "Both panels show pairwise IVM-versus-ODE-NEB marginal first-passage "
                "ratios at their displayed targets. Panel-b target is the tightest "
                "actual densified Delta E achieved by ODE-NEB; no absolute-error or "
                "tolerance definition is used."
            ),
            "recorded_horizons_NF": {
                method.key: {
                    "i_perp": int(np.nanmax(method.force)),
                    "densified_barrier": int(np.nanmax(method.barrier_force)),
                }
                for method in methods
            },
            "force_accounting_note": (
                "Baseline densified-barrier diagnostic calls are excluded from N_F. "
                "All IVM reference-free screen calls are charged; the disclosed "
                "screen total is recorded separately below."
            ),
            "initial_path_audit": initial_path_audit(),
            "method_style": {
                method.key: serializable_style(method.key) for method in methods
            },
            "plotting_code": plotting_code_manifest(),
            "ivm_screen_force_equivalent_evaluations": disclosure[
                "screen_force_equivalent_evaluations"
            ],
            "inputs": inputs,
            "source_csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
            "output_pdf": {"path": str(pdf_path), "sha256": sha256(pdf_path)},
            "plotter": str(Path(__file__).resolve()),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        outputs.append(str(pdf_path))
    print(json.dumps({"outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
