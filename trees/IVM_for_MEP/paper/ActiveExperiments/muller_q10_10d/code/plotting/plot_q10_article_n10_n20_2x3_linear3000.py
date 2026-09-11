#!/usr/bin/env python3
"""Render N10-interior and N20-interior Muller comparisons as one 2x3 figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter


PLOTTING_DIR = Path(__file__).resolve().parent
N20_ROOT = PLOTTING_DIR.parents[1]
N10_ROOT = N20_ROOT / "image12"
N10_PLOTTING_DIR = N10_ROOT / "code" / "plotting"
if str(N10_PLOTTING_DIR) not in sys.path:
    sys.path.insert(0, str(N10_PLOTTING_DIR))

from plot_q10_article_muller12 import (  # noqa: E402
    LABELS,
    extend_styles,
    load_formal_namespace,
    make_run,
)


FORCE_MAX = 3000.0
FORCE_TICKS = (0.0, 1000.0, 2000.0, 3000.0)
METHOD_ORDER = (
    "ivm", "ode_neb", "static_neb", "original_string",
    "obsm_fire", "obsm_lbfgs",
)
CONTOUR_LABELS = {"IVM", "ODE-NEB", "OBSM-LBFGS"}
N10_FROZEN_RUNS = {
    "ivm": "data/frozen_runs/tuned_ivm_alpha2e3_selected.npz",
    "ode_neb": "data/frozen_runs/ode_neb_k2000_selected.npz",
    "static_neb": "data/frozen_runs/static_neb_selected.npz",
    "original_string": "data/frozen_runs/original_string_selected.npz",
    "obsm_fire": "data/frozen_runs/obsm_fire_selected.npz",
    "obsm_lbfgs": "data/frozen_runs/obsm_lbfgs_selected.npz",
}
N20_FROZEN_RUNS = {
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


def apply_force_axis(axis) -> None:
    axis.set_xscale("linear")
    axis.set_xlim(0.0, FORCE_MAX)
    axis.set_xlabel("Number of Forces")
    axis.xaxis.set_major_locator(FixedLocator(FORCE_TICKS))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_n10_runs(namespace: dict) -> tuple[list[dict], dict]:
    selection_path = N10_ROOT / "data" / "reports" / "n12_final_selection.json"
    with selection_path.open() as handle:
        report = json.load(handle)
    selected = report["selected"]
    runs = []
    for method in METHOD_ORDER:
        item = selected[method]
        label = LABELS[method]
        details = label + ": " + json.dumps(item["config"], sort_keys=True)
        runs.append(
            make_run(N10_ROOT, N10_FROZEN_RUNS[method], label, details, namespace)
        )
    return runs, report


def load_n20_runs(namespace: dict) -> tuple[list[dict], dict]:
    report_dir = N20_ROOT / "data" / "reports"
    with (report_dir / "q10_inexact_tuned_resample_summary.json").open() as handle:
        ivm_rows = json.load(handle)
    ivm_row = next(
        row for row in ivm_rows
        if math.isclose(float(row["config"]["alpha"]), 1e-3)
        and row["stop_reason"] == "stationary"
    )
    with (report_dir / "baseline_final_selection.json").open() as handle:
        baseline_report = json.load(handle)
    selected = {"ivm": ivm_row, **baseline_report["selected"]}
    runs = []
    for method in METHOD_ORDER:
        item = selected[method]
        label = LABELS[method]
        details = label + ": " + json.dumps(item["config"], sort_keys=True)
        runs.append(
            make_run(N20_ROOT, N20_FROZEN_RUNS[method], label, details, namespace)
        )
    return runs, {
        "selected": selected,
        "selection_report": str(report_dir / "baseline_final_selection.json"),
        "ivm_report": str(report_dir / "q10_inexact_tuned_resample_summary.json"),
    }


def contour_runs_at_budget(runs: list[dict]) -> tuple[list[dict], dict]:
    contour_runs = []
    checkpoints = {}
    for run in runs:
        if run["label"] not in CONTOUR_LABELS:
            continue
        eligible = (run["count_seq"] <= FORCE_MAX).nonzero()[0]
        index = int(eligible[-1]) if len(eligible) else 0
        clipped = dict(run)
        clipped["images_seq"] = run["images_seq"][:index + 1]
        contour_runs.append(clipped)
        checkpoints[run["label"]] = float(run["count_seq"][index])
    return contour_runs, checkpoints


def draw_row(namespace: dict, axes: list, runs: list[dict]) -> dict:
    namespace["plot_metric_on_axis"](
        ax=axes[0], runs=runs, metric_name="loss_perp",
        ylabel=namespace["METRIC_INFO"]["loss_perp"]["ylabel"],
        use_compressed_y=True, panel_letter="placeholder",
    )
    apply_force_axis(axes[0])
    namespace["plot_metric_on_axis"](
        ax=axes[1], runs=runs, metric_name="Delta_Energy",
        ylabel=namespace["METRIC_INFO"]["Delta_Energy"]["ylabel"],
        use_compressed_y=False, panel_letter="placeholder",
    )
    apply_force_axis(axes[1])

    contour_runs, checkpoints = contour_runs_at_budget(runs)
    namespace["draw_contour_with_paths"](
        ax=axes[2],
        path_specs=namespace["build_path_specs_from_runs"](contour_runs),
        V_func=namespace["V_muller10_np"], Q_np=namespace["Q_np"],
        n_grid=namespace["CONTOUR_N_GRID"],
        xlim=namespace["CONTOUR_XLIM"], ylim=namespace["CONTOUR_YLIM"],
        potential_level_min=namespace["POTENTIAL_LEVEL_MIN"],
        potential_level_max=namespace["POTENTIAL_LEVEL_MAX"],
        potential_n_levels=namespace["POTENTIAL_N_LEVELS"],
        contour_labels=namespace["CONTOUR_LABELS"],
        show_colorbar=namespace["SHOW_CONTOUR_COLORBAR"],
        colorbar_inset=False, show_start_end=namespace["SHOW_START_END"],
        panel_letter="placeholder",
    )
    for axis in axes:
        for artist in tuple(axis.texts):
            if artist.get_text().strip().lower() in {
                "start", "end", "placeholder",
            }:
                artist.remove()
        axis.set_box_aspect(1.0)
    return checkpoints


def write_source_csv(path: Path, case_name: str, runs: list[dict],
                     append: bool) -> None:
    rows = []
    for run in runs:
        metrics = run["metrics"]
        for index in range(len(metrics["grad_count_seq"])):
            rows.append({
                "case": case_name,
                "method": run["label"],
                "number_of_forces": float(metrics["grad_count_seq"][index]),
                "I_perp": float(metrics["loss_perp"][index]),
                "Delta_E_dense": float(metrics["Delta_Energy"][index]),
                "frozen_file": str(run["file"]),
            })
    mode = "a" if append else "w"
    with path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    namespace = load_formal_namespace(N10_ROOT)
    extend_styles(namespace)
    n10_runs, n10_report = load_n10_runs(namespace)
    n20_runs, n20_report = load_n20_runs(namespace)

    fig = plt.figure(
        figsize=(namespace["mm_to_inch"](180), namespace["mm_to_inch"](126)),
        dpi=220,
    )
    grid = GridSpec(
        2, 3, figure=fig, width_ratios=(1.0, 1.0, 1.0),
        left=0.068, right=0.94, bottom=0.08, top=0.90,
        wspace=0.42, hspace=0.12,
    )
    axes = [[fig.add_subplot(grid[row, column]) for column in range(3)]
            for row in range(2)]
    n10_checkpoints = draw_row(namespace, axes[0], n10_runs)
    n20_checkpoints = draw_row(namespace, axes[1], n20_runs)
    for axis in axes[0]:
        axis.set_xlabel("")

    fig.canvas.draw()
    row_titles = (
        r"$\mathbf{N}_{\mathrm{int}}=\mathbf{10}$",
        r"$\mathbf{N}_{\mathrm{int}}=\mathbf{20}$",
    )
    for row, (letters, row_title) in enumerate(zip(
        (("a", "b", "c"), ("d", "e", "f")), row_titles
    )):
        row_left = axes[row][0].get_position().x0
        row_right = axes[row][2].get_position().x1
        row_top = max(axis.get_position().y1 for axis in axes[row])
        header_y = row_top + 0.033
        header_center = 0.5 * (row_left + row_right)
        title_half_gap = 0.060
        for line_start, line_end in (
            (row_left, header_center - title_half_gap),
            (header_center + title_half_gap, row_right),
        ):
            fig.add_artist(Line2D(
                [line_start, line_end], [header_y, header_y],
                transform=fig.transFigure, color="#C2C2BE",
                linewidth=0.6, solid_capstyle="butt", clip_on=False, zorder=2,
            ))
        fig.text(
            header_center, header_y, row_title,
            ha="center", va="center", color="#272727",
            fontsize=8.4, fontweight="normal", zorder=3,
        )
        for axis, letter in zip(axes[row], letters):
            position = axis.get_position()
            fig.text(
                position.x0 - 0.04, row_top + 0.001, letter,
                ha="left", va="bottom", fontweight="bold", zorder=3,
                fontsize=namespace["plt"].rcParams["font.size"] + 1,
            )

    handles, labels = axes[0][0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(
        unique.values(), unique.keys(), loc="upper center",
        bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False,
        columnspacing=1.6, handlelength=2.5, handletextpad=0.55,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", dpi=600)
    plt.close(fig)

    source_csv = args.output.with_name(args.output.stem + "_source.csv")
    write_source_csv(source_csv, "10 interior images", n10_runs, append=False)
    write_source_csv(source_csv, "20 interior images", n20_runs, append=True)
    frozen = {
        "10_interior": {
            method: {
                "path": str(N10_ROOT / relative),
                "sha256": sha256(N10_ROOT / relative),
            }
            for method, relative in N10_FROZEN_RUNS.items()
        },
        "20_interior": {
            method: {
                "path": str(N20_ROOT / relative),
                "sha256": sha256(N20_ROOT / relative),
            }
            for method, relative in N20_FROZEN_RUNS.items()
        },
    }
    manifest = {
        "case": "fixed-Q unperturbed 10D Muller-Brown",
        "layout": "2x3; 10 interior images above 20 interior images",
        "xscale": "linear",
        "xlim": [0, 3000],
        "cost_axis": "direct sum of uncached force calls over all interior images",
        "energy_barrier_metric": "500-point coordinate-wise cubic densified Delta E",
        "energy_barrier_y_axis": "standard logarithmic; no compressed-axis threshold",
        "contour_methods": sorted(CONTOUR_LABELS),
        "contour_checkpoint_policy": "last checkpoint with total forces <= 3000",
        "contour_checkpoint_forces": {
            "10_interior": n10_checkpoints,
            "20_interior": n20_checkpoints,
        },
        "n10_selection": n10_report,
        "n20_selection": n20_report,
        "frozen_runs": frozen,
        "source_csv": str(source_csv),
        "formal_notebook": str(
            N10_PLOTTING_DIR / "reference" / "figure_M10_N20_v2.ipynb"
        ),
        "formal_notebook_cell": 7,
        "ivm_highest_zorder": True,
    }
    manifest_path = args.output.with_name(args.output.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
