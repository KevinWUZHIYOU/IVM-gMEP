#!/usr/bin/env python3
"""Render the active Muller Q10 comparison as a 1x3 linear-force figure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FuncFormatter


PLOTTING_DIR = Path(__file__).resolve().parent
ACTIVE_ROOT = PLOTTING_DIR.parents[1]
REPO_ROOT = ACTIVE_ROOT.parents[2]
if str(PLOTTING_DIR) not in sys.path:
    sys.path.insert(0, str(PLOTTING_DIR))

from plot_q10_article_muller22 import (  # noqa: E402
    LABELS,
    extend_styles,
    load_formal_namespace,
    make_run,
)


FORCE_MIN = 0.0
FORCE_MAX = 3000.0
FORCE_TICKS = (0.0, 1000.0, 2000.0, 3000.0)
FROZEN_RUNS = {
    "ivm": "data/frozen_runs/tuned_ivm_alpha1e3_selected.npz",
    "ode_neb": "data/frozen_runs/ode_neb_k1000_selected.npz",
    "static_neb": "data/frozen_runs/static_neb_selected.npz",
    "original_string": "data/frozen_runs/original_string_selected.npz",
    "obsm_fire": "data/frozen_runs/obsm_fire_selected.npz",
    "obsm_lbfgs": "data/frozen_runs/obsm_lbfgs_selected.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def apply_linear_force_axis(ax) -> None:
    ax.set_xscale("linear")
    ax.set_xlim(FORCE_MIN, FORCE_MAX)
    ax.set_xlabel("Number of Forces")
    ax.xaxis.set_major_locator(FixedLocator(FORCE_TICKS))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))


def main() -> None:
    args = parse_args()
    namespace = load_formal_namespace(REPO_ROOT, ACTIVE_ROOT)
    extend_styles(namespace)

    report_dir = ACTIVE_ROOT / "data" / "reports"
    with (report_dir / "q10_inexact_tuned_resample_summary.json").open() as handle:
        ivm_rows = json.load(handle)
    ivm_row = next(
        row for row in ivm_rows
        if row["config"]["alpha"] == 1e-3 and row["stop_reason"] == "stationary"
    )
    ivm_details = "IVM: " + json.dumps(ivm_row["config"], sort_keys=True)
    ivm = make_run(ACTIVE_ROOT, FROZEN_RUNS["ivm"], "IVM", ivm_details, namespace)

    with (report_dir / "baseline_final_selection.json").open() as handle:
        selected = json.load(handle)["selected"]
    runs = [ivm]
    for method in ("ode_neb", "static_neb", "original_string", "obsm_fire", "obsm_lbfgs"):
        item = selected[method]
        label = LABELS[method]
        details = label + ": " + json.dumps(item["config"], sort_keys=True)
        runs.append(make_run(ACTIVE_ROOT, FROZEN_RUNS[method], label, details, namespace))

    fig = plt.figure(figsize=(namespace["mm_to_inch"](180), namespace["mm_to_inch"](70)), dpi=220)
    grid = GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=(1.0, 1.0, 1.0),
        left=0.068,
        right=0.94,
        bottom=0.16,
        top=0.84,
        wspace=0.42,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]

    namespace["plot_metric_on_axis"](
        ax=axes[0],
        runs=runs,
        metric_name="loss_perp",
        ylabel=namespace["METRIC_INFO"]["loss_perp"]["ylabel"],
        use_compressed_y=True,
        panel_letter="a",
    )
    apply_linear_force_axis(axes[0])

    namespace["plot_metric_on_axis"](
        ax=axes[1],
        runs=runs,
        metric_name="Delta_Energy",
        ylabel=namespace["METRIC_INFO"]["Delta_Energy"]["ylabel"],
        use_compressed_y=True,
        panel_letter="b",
    )
    apply_linear_force_axis(axes[1])

    namespace["draw_contour_with_paths"](
        ax=axes[2],
        path_specs=namespace["build_path_specs_from_runs"](runs),
        V_func=namespace["V_muller10_np"],
        Q_np=namespace["Q_np"],
        n_grid=namespace["CONTOUR_N_GRID"],
        xlim=namespace["CONTOUR_XLIM"],
        ylim=namespace["CONTOUR_YLIM"],
        potential_level_min=namespace["POTENTIAL_LEVEL_MIN"],
        potential_level_max=namespace["POTENTIAL_LEVEL_MAX"],
        potential_n_levels=namespace["POTENTIAL_N_LEVELS"],
        contour_labels=namespace["CONTOUR_LABELS"],
        show_colorbar=namespace["SHOW_CONTOUR_COLORBAR"],
        colorbar_inset=False,
        show_start_end=namespace["SHOW_START_END"],
        panel_letter="c",
    )

    for artist in tuple(axes[2].texts):
        if artist.get_text().strip().lower() in {"start", "end"}:
            artist.remove()

    for axis in axes:
        axis.set_box_aspect(1.0)

    fig.canvas.draw()
    panel_letter_y = max(axis.get_position().y1 for axis in axes) + 0.01
    for axis, letter in zip(axes, ("a", "b", "c")):
        axis_position = axis.get_position()
        for artist in axis.texts:
            if artist.get_text().strip() == letter:
                artist.set_transform(fig.transFigure)
                artist.set_position((axis_position.x0 - 0.04, panel_letter_y))
                artist.set_horizontalalignment("left")
                artist.set_verticalalignment("bottom")
                artist.set_bbox(None)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.5,
        handletextpad=0.55,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
