#!/usr/bin/env python3
"""Temporary all-methods overlay of the two best LEPS sync-one r values."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


HERE = Path(__file__).resolve()
CAMPAIGN_ROOT = HERE.parents[1]
LEPS_ROOT = HERE.parents[2]
REPO_ROOT = HERE.parents[4]
PRIMITIVES = LEPS_ROOT / "ir_ivm_qm10_v2p20_microrefine_20260829/src/remake_leps_all_methods.py"
spec = importlib.util.spec_from_file_location("_leps_all_methods_r_comparison_primitives", PRIMITIVES)
if spec is None or spec.loader is None:
    raise ImportError(PRIMITIVES)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

SELECTED = CAMPAIGN_ROOT / "runs/3c62d93a051be116.npz"  # alpha100, r3e-3, m5
PRIOR = (
    LEPS_ROOT
    / "leps_irivm_ala_chig_every1_sync1_20260830/runs/356307bb5a618535.npz"
)  # alpha100, r1e-3, m5
PDF = REPO_ROOT / "output/pdf/leps_all_methods_metrics_r_comparison.pdf"
OUTDIR = CAMPAIGN_ROOT / "reports/figure"
SOURCE_CSV = OUTDIR / "leps_all_methods_metrics_r_comparison_source.csv"
PROVENANCE = OUTDIR / "leps_all_methods_metrics_r_comparison_provenance.json"

SELECTED_NAME = r"IR-IVM ($r=3\times10^{-3}$)"
PRIOR_NAME = r"IR-IVM ($r=10^{-3}$)"
METHOD_ORDER = (
    SELECTED_NAME,
    PRIOR_NAME,
    "Article IVM",
    "ODE-NEB",
    "Static-NEB",
    "Original String",
    "OBSM-FIRE",
    "OBSM-L-BFGS",
)
COLORS = {
    SELECTED_NAME: "#0072B2",
    PRIOR_NAME: "#56B4E9",
    **{key: value for key, value in base.COLORS.items() if key != "IR-IVM"},
}
STYLES = {
    SELECTED_NAME: "-",
    PRIOR_NAME: (0, (3, 1.4)),
    **{key: value for key, value in base.STYLES.items() if key != "IR-IVM"},
}
MARKERS = {
    SELECTED_NAME: ("o", 3.2),
    PRIOR_NAME: ("D", 2.8),
    **{key: value for key, value in base.MARKERS.items() if key != "IR-IVM"},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    traces = {
        SELECTED_NAME: base.ir_ivm_trace(SELECTED),
        PRIOR_NAME: base.ir_ivm_trace(PRIOR),
        "Article IVM": base.article_ivm_trace(base.ARTICLE_IVM),
    }
    traces.update({name: base.baseline_trace(path) for name, path in base.OTHER_METHODS.items()})
    results = {
        SELECTED_NAME: SELECTED,
        PRIOR_NAME: PRIOR,
        "Article IVM": base.ARTICLE_IVM,
        **base.OTHER_METHODS,
    }
    plotted = {}
    rows = []
    for name in METHOD_ORDER:
        nof, paths = traces[name]
        keep = nof <= base.FORCE_MAX
        nof = nof[keep]
        paths = paths[keep]
        i_perp, barrier = base.metrics(paths)
        plotted[name] = (nof, i_perp, barrier)
        for index, (cost, residual, delta_e) in enumerate(zip(nof, i_perp, barrier)):
            rows.append({
                "method": name,
                "plotted_index": index,
                "number_of_forces": int(cost),
                "i_perp": float(residual),
                "densified_barrier_eV": float(delta_e),
                "densification": "500 total coordinate-wise linear interpolation points",
                "production_result": str(results[name].relative_to(LEPS_ROOT)),
                "production_result_sha256": sha256(results[name]),
            })

    OUTDIR.mkdir(parents=True, exist_ok=True)
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
        "legend.fontsize": 6.1,
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
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 76 / 25.4), dpi=220)
    for name in METHOD_ORDER:
        nof, i_perp, barrier = plotted[name]
        marker, size = MARKERS[name]
        is_ir = name in (SELECTED_NAME, PRIOR_NAME)
        markevery = 1 if is_ir else max(len(nof) // 8, 1)
        width = 2.05 if name == SELECTED_NAME else (1.75 if name == PRIOR_NAME else (1.8 if name == "Article IVM" else 1.5))
        zorder = 32 if name == SELECTED_NAME else (30 if name == PRIOR_NAME else (22 if name == "Article IVM" else 5))
        for axis, values in zip(axes, (i_perp, barrier)):
            axis.plot(
                nof,
                values,
                label=name,
                color=COLORS[name],
                linestyle=STYLES[name],
                linewidth=width,
                marker=marker,
                markersize=size,
                markerfacecolor="white",
                markeredgewidth=0.7,
                markevery=markevery,
                zorder=zorder,
            )

    reference = float(base.reference_saddle()["forward_barrier"])
    axes[1].axhline(reference, color="#222222", linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
    axes[1].text(base.FORCE_MAX * 0.985, reference + 0.035, r"$\Delta E^\ast$", ha="right", va="bottom", fontsize=6.8)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(base.log_tick))
    for axis in axes:
        axis.set_xlim(0, base.FORCE_MAX)
        axis.set_xticks(base.FORCE_TICKS)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.set_xlabel("Number of Forces")
        axis.grid(False)
    axes[0].set_ylabel(r"$I_{\perp}$")
    axes[1].set_ylabel(r"Densified energy barrier $\Delta E$ (eV)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        columnspacing=0.95,
        handlelength=2.2,
        handletextpad=0.42,
    )
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.15, top=0.73, wspace=0.31)
    fig.canvas.draw()
    panel_y = max(axis.get_position().y1 for axis in axes) + 0.01
    for axis, letter in zip(axes, ("a", "b")):
        position = axis.get_position()
        fig.text(position.x0 - 0.04, panel_y, letter, ha="left", va="bottom", fontsize=9.0, fontweight="bold")

    PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF, dpi=600, metadata={"Title": "LEPS IR-IVM r comparison and article methods"})
    plt.close(fig)

    provenance = {
        "figure": str(PDF),
        "figure_sha256": sha256(PDF),
        "plot_source": str(HERE.relative_to(LEPS_ROOT)),
        "plot_source_sha256": sha256(HERE),
        "source_csv": str(SOURCE_CSV.relative_to(LEPS_ROOT)),
        "source_csv_sha256": sha256(SOURCE_CSV),
        "metric_contract": {
            "i_perp": "article Henkelman-Jonsson improved-tangent integral",
            "densified_barrier": "500 total coordinate-wise linearly interpolated points",
            "number_of_forces": "uncached joint energy-force evaluations over the 20 internal images",
        },
        "compared_ir_ivm": {
            SELECTED_NAME: {"path": str(SELECTED.relative_to(LEPS_ROOT)), "sha256": sha256(SELECTED)},
            PRIOR_NAME: {"path": str(PRIOR.relative_to(LEPS_ROOT)), "sha256": sha256(PRIOR)},
        },
        "temporary_comparison": True,
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(PDF)
    print(SOURCE_CSV)
    print(PROVENANCE)


if __name__ == "__main__":
    main()
