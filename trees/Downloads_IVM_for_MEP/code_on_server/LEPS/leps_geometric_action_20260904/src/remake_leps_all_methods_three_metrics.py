#!/usr/bin/env python3
"""Rebuild the registered LEPS comparison as three square metric panels."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


HERE = Path(__file__).resolve()
CAMPAIGN = HERE.parents[1]
LEPS = HERE.parents[2]
PROJECT = LEPS.parents[1]
UPSTREAM = LEPS / "leps_irivm_sync1_refined_grid_20260830"
UPSTREAM_PLOT = UPSTREAM / "src/remake_leps_all_methods_selected.py"
REGISTRY = UPSTREAM / "reports/figure/leps_all_methods_metrics_provenance.json"
SOURCE_CSV = UPSTREAM / "reports/figure/leps_all_methods_metrics_source.csv"
FORMAL = PROJECT / "Figures/LEPS/leps_all_methods_metrics.pdf"
OUTPUT = PROJECT / "output/pdf/leps_all_methods_metrics.pdf"
SOURCE_COPY = LEPS / "figures/leps_all_methods_metrics.pdf"
OLD_TWO_PANEL_SHA = "9faaa4e643cd14329bdb65825f954886bb0232db04b0a425b7a7ae14a9051a25"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometric_action(path: np.ndarray) -> float:
    """22-node trapezoidal integral of the complete gradient norm."""
    physical = plot.flat_to_physical(path)
    _, gradient2 = plot.leps_energy_gradient(physical[:, 0], physical[:, 1])
    gradient = np.zeros_like(path)
    gradient[:, 0] = gradient2[:, 0]
    gradient[:, 3] = gradient2[:, 1]
    gradient_norm = np.linalg.norm(gradient, axis=1)
    segment_length = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return float(np.sum(0.5 * (gradient_norm[:-1] + gradient_norm[1:]) * segment_length))


def log_tick(value, _position):
    if not np.isfinite(value) or value <= 0:
        return ""
    exponent = int(np.rint(np.log10(value)))
    return f"1e{exponent}" if np.isclose(value, 10.0**exponent) else ""


spec = importlib.util.spec_from_file_location("registered_two_metric_plot", UPSTREAM_PLOT)
selected = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selected)
plot = selected.plot


def main() -> None:
    old_registry = json.loads(REGISTRY.read_text())
    traces = {"IVM": plot.ir_ivm_trace(plot.WINNER)}
    traces.update({name: plot.baseline_trace(path) for name, path in plot.OTHER_METHODS.items()})
    results = {"IVM": plot.WINNER, **plot.OTHER_METHODS}
    assert tuple(traces) == tuple(plot.METHOD_ORDER)

    plotted = {}
    rows = []
    for name in plot.METHOD_ORDER:
        raw = results[name]
        registered = old_registry["plotted_results"][name]
        assert raw == LEPS / registered["path"]
        assert sha256(raw) == registered["sha256"]
        nof, paths = traces[name]
        keep = nof <= plot.FORCE_MAX
        nof, paths = nof[keep], paths[keep]
        i_perp, barrier = plot.metrics(paths)
        action = np.asarray([geometric_action(path) for path in paths])
        plotted[name] = (nof, i_perp, barrier, action)
        for index, values in enumerate(zip(nof, i_perp, barrier, action)):
            cost, residual, delta_e, action_eV = values
            rows.append({
                "method": name,
                "plotted_index": index,
                "number_of_forces": int(cost),
                "i_perp": float(residual),
                "densified_barrier_eV": float(delta_e),
                "geometric_action_eV": float(action_eV),
                "densification": "500 total coordinate-wise linear interpolation points (barrier only)",
                "geometric_action_quadrature": "22-node segment-length trapezoidal rule; no densification",
                "production_result": registered["path"],
                "production_result_sha256": registered["sha256"],
            })

    SOURCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({
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
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(200 / 25.4, 88 / 25.4), dpi=220)
    for name in plot.METHOD_ORDER:
        nof, i_perp, barrier, action = plotted[name]
        marker, size = plot.MARKERS[name]
        for axis, values in zip(axes, (i_perp, barrier, action)):
            axis.plot(
                nof,
                values,
                label=name,
                color=plot.COLORS[name],
                linestyle=plot.STYLES[name],
                linewidth=2.05 if name == "IVM" else 1.5,
                marker=marker,
                markersize=size,
                markerfacecolor="white",
                markeredgewidth=0.7,
                markevery=1 if name == "IVM" else max(len(nof) // 8, 1),
                zorder=30 if name == "IVM" else 5,
            )

    reference = float(plot.reference_saddle()["forward_barrier"])
    axes[1].axhline(reference, color="#222222", linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
    axes[1].text(plot.FORCE_MAX * 0.985, reference + 0.035, r"$\Delta E^\ast$",
                 ha="right", va="bottom", fontsize=6.8)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(log_tick))
    axes[2].set_ylim(0, max(float(values[3].max()) for values in plotted.values()) * 1.06)
    for axis in axes:
        axis.set_xlim(0, plot.FORCE_MAX)
        axis.set_xticks(plot.FORCE_TICKS)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.set_xlabel("Number of Forces")
        axis.set_box_aspect(1)
        axis.grid(False)
    axes[0].set_ylabel(r"$I_{\perp}$")
    axes[1].set_ylabel(r"Densified energy barrier $\Delta E$ (eV)")
    axes[2].set_ylabel(r"Discrete geometric action $\widehat{S}$ (eV)")

    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.52, 0.985), ncol=3, columnspacing=1.25,
               handlelength=2.35, handletextpad=0.5)
    fig.subplots_adjust(left=0.067, right=0.965, bottom=0.17, top=0.76, wspace=0.39)
    fig.canvas.draw()
    display_ratios = []
    for axis in axes:
        bbox = axis.get_window_extent()
        display_ratios.append(float(bbox.width / bbox.height))
    np.testing.assert_allclose(display_ratios, 1.0, rtol=0, atol=1e-12)
    panel_y = max(axis.get_position().y1 for axis in axes) + 0.012
    for axis, letter in zip(axes, ("a", "b", "c")):
        position = axis.get_position()
        fig.text(position.x0 - 0.035, panel_y, letter, ha="left", va="bottom",
                 fontsize=9.0, fontweight="bold")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, metadata={
        "Title": "LEPS all-method convergence: three metrics",
        "CreationDate": None,
        "ModDate": None,
    })
    plt.close(fig)
    FORMAL.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_COPY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUTPUT, FORMAL)
    shutil.copyfile(OUTPUT, SOURCE_COPY)
    assert sha256(OUTPUT) == sha256(FORMAL) == sha256(SOURCE_COPY)

    registry = dict(old_registry)
    registry["figure"] = str(FORMAL.relative_to(PROJECT))
    registry["figure_sha256"] = sha256(FORMAL)
    registry["plot_script"] = str(HERE.relative_to(LEPS))
    registry["plot_script_sha256"] = sha256(HERE)
    registry["source_csv"] = str(SOURCE_CSV.relative_to(LEPS))
    registry["source_csv_sha256"] = sha256(SOURCE_CSV)
    registry["figure_layout"] = {
        "panels": ["I_perp", "densified energy barrier Delta E", "discrete geometric action S_hat"],
        "arrangement": "1x3",
        "each_axes_box_aspect": 1.0,
        "verified_display_width_to_height": display_ratios,
    }
    registry["metric_contract"] = {
        "i_perp": "article path-length quadrature with Henkelman-Jonsson improved tangent",
        "densified_barrier": "500 total coordinate-wise linearly interpolated points",
        "geometric_action": (
            "sum over the 21 original segments of 0.5*(norm(grad V_i)+"
            "norm(grad V_{i+1}))*norm(x_{i+1}-x_i); all 22 nodes including "
            "endpoints, full gradient norm, no densification"
        ),
        "geometric_action_tangent_note": (
            "the full-gradient action is tangent-independent; the same "
            "Henkelman-Jonsson improved tangent is retained for I_perp"
        ),
        "number_of_forces": "uncached joint energy-force evaluations over the 20 internal images",
        "offline_metric_evaluations_counted_as_nof": False,
    }
    registry["three_panel_replacement_authorized"] = True
    registry["three_panel_replacement_date"] = "2026-09-06"
    registry["superseded_two_panel_figure"] = {
        "path": str(FORMAL.relative_to(PROJECT)),
        "sha256": OLD_TWO_PANEL_SHA,
    }
    REGISTRY.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "figure": str(FORMAL),
        "figure_sha256": sha256(FORMAL),
        "source_csv_sha256": sha256(SOURCE_CSV),
        "axes_display_ratios": display_ratios,
        "plotted_rows": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
