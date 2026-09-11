#!/usr/bin/env python3
"""Plot all four fixed-Q Muller-10D alpha studies in article style."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


GROUPS = (
    ("exact_no_resample", "Exact IVM, no resample"),
    ("exact_resample", "Exact IVM + cubic resample"),
    ("inexact_baseline_resample", "Baseline inexact IVM + cubic resample"),
    ("inexact_tuned_resample", "Tuned inexact IVM + cubic resample"),
)


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "legend.fontsize": 6.7,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def subsets(group: str, rows: list[dict]):
    if group.startswith("exact_"):
        for r_value in sorted(
            {row["config"]["kkt_rtol"] for row in rows}, reverse=True
        ):
            yield f"r_{r_value:g}", rf"$r={r_value:g}$", [
                row for row in rows if row["config"]["kkt_rtol"] == r_value
            ]
    else:
        yield "all_alpha", "", rows


def plot_subset(stage: Path, output_dir: Path, group: str, title: str,
                suffix: str, subtitle: str, rows: list[dict]) -> list[str]:
    rows = sorted(rows, key=lambda row: row["config"]["alpha"])
    colors = plt.get_cmap("cividis")(np.linspace(0.05, 0.95, len(rows)))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.72), constrained_layout=True)
    for row, color in zip(rows, colors):
        data = np.load(stage / row["output"], allow_pickle=False)
        alpha = float(row["config"]["alpha"])
        selected = group == "inexact_tuned_resample" and np.isclose(alpha, 1e-3)
        plot_style = {
            "color": "#D55E00" if selected else color,
            "linewidth": 2.0 if selected else 1.05,
            "alpha": 1.0 if selected else 0.82,
            "zorder": 20 if selected else 3,
            "label": rf"$\alpha={alpha:g}$",
        }
        x = np.asarray(data["total_force_evaluations"], dtype=float)
        barrier = np.abs(
            np.asarray(data["delta_e"], dtype=float)
            - float(np.ravel(data["reference_barrier"])[0])
        )
        axes[0].plot(x, np.maximum(np.asarray(data["i_perp"], dtype=float), 1e-30), **plot_style)
        axes[1].plot(x, np.maximum(barrier, 1e-16), **plot_style)
    for axis in axes:
        axis.set_xscale("linear")
        axis.set_yscale("log")
        axis.set_xlim(0.0, 5000.0)
        axis.grid(True, which="major", color="#d9d9d9", linewidth=0.5, alpha=0.65)
        axis.set_xlabel("Number of Forces")
    axes[0].set_ylabel(r"$I_{\perp}$")
    axes[1].set_ylabel(r"$|\Delta E_{\mathrm{dense}}-\Delta E_{\mathrm{ref}}|$")
    fig.suptitle(title if not subtitle else f"{title}, {subtitle}", y=1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.01),
               ncol=5, frameon=False)
    stem = output_dir / f"{group}_{suffix.replace('.', 'p')}_alpha_audit"
    outputs = []
    for extension in ("pdf", "svg", "png"):
        output = Path(f"{stem}.{extension}")
        fig.savefig(output, dpi=600 if extension == "png" else None)
        outputs.append(str(output.relative_to(stage)))
    plt.close(fig)
    return outputs


def main() -> None:
    style()
    stage = Path(__file__).resolve().parents[1]
    output_dir = stage / "figures" / "alpha_studies"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for group, title in GROUPS:
        with (stage / "reports" / f"q10_{group}_summary.json").open() as handle:
            rows = json.load(handle)
        for suffix, subtitle, selected_rows in subsets(group, rows):
            outputs = plot_subset(
                stage, output_dir, group, title, suffix, subtitle, selected_rows
            )
            manifest.append({
                "group": group,
                "subset": suffix,
                "alphas": [row["config"]["alpha"] for row in selected_rows],
                "metrics": ["I_perp", "500-point dense Delta E error"],
                "force_accounting": "mean uncached joint energy-gradient evaluations per internal image",
                "xscale": "linear",
                "xlim": [0, 1000],
                "outputs": outputs,
            })
    with (output_dir / "q10_alpha_study_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(output_dir)


if __name__ == "__main__":
    main()
