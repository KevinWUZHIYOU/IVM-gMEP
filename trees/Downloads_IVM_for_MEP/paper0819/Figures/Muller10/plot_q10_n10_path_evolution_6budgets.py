#!/usr/bin/env python3
"""Plot the N_int=10, 12-point path evolution at six common NoF budgets."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


HERE = Path(__file__).resolve().parent
REPOSITORY = Path("/Users/wuzhiyou/Downloads/IVM_for_MEP")
OUTPUT_PDF = REPOSITORY / "output/pdf/q10_n10_path_evolution_6budgets.pdf"
QA_PNG = REPOSITORY / "tmp/pdfs/q10_n10_path_evolution_6budgets.png"
SOURCE_CSV = OUTPUT_PDF.with_name(OUTPUT_PDF.stem + "_source.csv")
MANIFEST = OUTPUT_PDF.with_name(OUTPUT_PDF.stem + "_manifest.json")
TARGET_BUDGETS = (0.0, 600.0, 1200.0, 1800.0, 2400.0, 3000.0)

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plot_muller_q10_article_method_comparison_n10_n20_2x4_geometric_action as formal  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_snapshot(run: dict, budget: float) -> dict:
    paths = np.asarray(run["images_seq"], dtype=float)
    nof = np.asarray(run["metrics"]["grad_count_seq"], dtype=float)
    if paths.shape[0] != nof.size:
        raise RuntimeError(
            f"path/checkpoint mismatch for {run['label']}: "
            f"{paths.shape[0]} != {nof.size}"
        )
    if paths.shape[1:] != (12, 10):
        raise RuntimeError(
            f"expected a 12-point 10D path for {run['label']}, got {paths.shape}"
        )

    if budget == 0.0:
        index = 0
        effective_nof = 0.0
        policy = "common initial geometry before force evaluation"
    else:
        eligible = np.flatnonzero(nof <= budget + 1.0e-12)
        if eligible.size == 0:
            raise RuntimeError(
                f"no checkpoint at or below NoF budget {budget:g} for {run['label']}"
            )
        index = int(eligible[-1])
        effective_nof = float(nof[index])
        policy = "last stored checkpoint with Number of Forces <= target budget"

    path = paths[index].copy()
    if not np.all(np.isfinite(path)):
        raise FloatingPointError(
            f"non-finite path for {run['label']} at budget {budget:g}"
        )
    return {
        "path": path,
        "source_record_index": index,
        "source_recorded_nof": float(nof[index]),
        "effective_nof": effective_nof,
        "selection_policy": policy,
    }


def style_axis(axis: plt.Axes, namespace: dict) -> None:
    axis.set_xlim(namespace["CONTOUR_XLIM"])
    axis.set_ylim(namespace["CONTOUR_YLIM"])
    axis.set_aspect("equal", adjustable="box")
    axis.spines["left"].set_linewidth(0.75)
    axis.spines["bottom"].set_linewidth(0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.0,
        width=0.75,
        pad=2.0,
    )
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(False)


def main() -> int:
    namespace = formal.registered.base.load_formal_namespace(
        formal.registered.base.N10_ROOT
    )
    n10_runs, _, formal_summaries = formal.load_runs(namespace)
    labels = [run["label"] for run in n10_runs]
    expected_labels = [
        "IVM",
        "ODE-NEB",
        "Static NEB",
        "Original String",
        "OBSM-FIRE",
        "OBSM-LBFGS",
    ]
    if labels != expected_labels:
        raise RuntimeError(f"unexpected N_int=10 method order: {labels}")

    initial = np.asarray(n10_runs[0]["images_seq"][0], dtype=float)
    initial_differences = {
        run["label"]: float(
            np.max(np.abs(np.asarray(run["images_seq"][0]) - initial))
        )
        for run in n10_runs
    }
    if max(initial_differences.values()) != 0.0:
        raise RuntimeError(
            f"methods do not share an identical initial path: {initial_differences}"
        )

    snapshots = {
        budget: {
            run["label"]: selected_snapshot(run, budget)
            for run in n10_runs
        }
        for budget in TARGET_BUDGETS
    }

    X, Y, Z = namespace["compute_contour_grid"](
        V_func=namespace["V_muller10_np"],
        Q_np=namespace["Q_np"],
        n_grid=namespace["CONTOUR_N_GRID"],
        xlim=namespace["CONTOUR_XLIM"],
        ylim=namespace["CONTOUR_YLIM"],
    )
    levels = np.linspace(
        namespace["POTENTIAL_LEVEL_MIN"],
        namespace["POTENTIAL_LEVEL_MAX"],
        namespace["POTENTIAL_N_LEVELS"],
    )
    line_levels = levels[::4]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(7.45, 5.25), dpi=220)
    contour_set = None
    source_rows = []

    for panel_index, (axis, budget) in enumerate(
        zip(axes.ravel(), TARGET_BUDGETS)
    ):
        contour_set = axis.contourf(
            X,
            Y,
            Z,
            levels=levels,
            cmap="Greys_r",
            alpha=0.92,
            extend="both",
            zorder=-10,
        )
        axis.contour(
            X,
            Y,
            Z,
            levels=line_levels,
            colors="0.25",
            linewidths=0.35,
            alpha=0.78,
            zorder=-5,
        )

        first_projected = None
        for run in n10_runs:
            label = run["label"]
            snapshot = snapshots[budget][label]
            path = snapshot["path"]
            projected = namespace["project_path_to_z01"](
                path, Q_np=namespace["Q_np"]
            )
            if first_projected is None:
                first_projected = projected
            style = namespace["get_plot_style"](run)
            axis.plot(
                projected[:, 0],
                projected[:, 1],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=min(float(style["linewidth"]), 1.55),
                markersize=max(float(style["markersize"]), 2.6),
                color=style["color"],
                alpha=0.92,
                markerfacecolor="white",
                markeredgewidth=0.45,
                markevery=1,
                label=label,
                zorder=20 + int(style["zorder"]),
            )

            source_file = Path(run["file"])
            for node_index, (point, z01) in enumerate(zip(path, projected)):
                row = {
                    "target_nof_budget": f"{budget:g}",
                    "method": label,
                    "source_record_index": snapshot["source_record_index"],
                    "source_recorded_nof": (
                        f"{snapshot['source_recorded_nof']:.17g}"
                    ),
                    "effective_nof": f"{snapshot['effective_nof']:.17g}",
                    "selection_policy": snapshot["selection_policy"],
                    "node_index": node_index,
                    "z0": f"{float(z01[0]):.17g}",
                    "z1": f"{float(z01[1]):.17g}",
                    "source_file": str(source_file),
                    "source_sha256": sha256(source_file),
                }
                row.update(
                    {
                        f"x{coordinate}": f"{float(value):.17g}"
                        for coordinate, value in enumerate(point)
                    }
                )
                source_rows.append(row)

        if first_projected is None:
            raise RuntimeError("no path was plotted")
        axis.scatter(
            first_projected[0, 0],
            first_projected[0, 1],
            s=24,
            marker="o",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=80,
        )
        axis.scatter(
            first_projected[-1, 0],
            first_projected[-1, 1],
            s=46,
            marker="*",
            facecolor="white",
            edgecolor="black",
            linewidth=0.7,
            zorder=85,
        )
        axis.set_title(f"NoF budget = {budget:g}", pad=4.0, fontsize=8.4)
        axis.text(
            -0.12,
            1.025,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )
        if budget == 0.0:
            axis.text(
                0.03,
                0.035,
                "common initial path",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=6.7,
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
                zorder=90,
            )
        style_axis(axis, namespace)

    for row in range(2):
        axes[row, 0].set_ylabel("z1")
    for column in range(3):
        axes[1, column].set_xlabel("z0")

    handles = []
    for run in n10_runs:
        style = namespace["get_plot_style"](run)
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=min(float(style["linewidth"]), 1.55),
                marker=style["marker"],
                markersize=max(float(style["markersize"]), 2.6),
                markerfacecolor="white",
                markeredgewidth=0.45,
                label=run["label"],
            )
        )
    figure.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.49, 0.925),
        ncol=3,
        frameon=False,
        columnspacing=1.15,
        handlelength=2.2,
        handletextpad=0.45,
        fontsize=7.0,
    )
    figure.suptitle(
        r"Q-transformed Muller: $N_{\mathrm{int}}=10$ path evolution",
        x=0.49,
        y=0.985,
        fontsize=10.0,
        fontweight="semibold",
    )
    if contour_set is None:
        raise RuntimeError("contour background was not created")
    color_axis = figure.add_axes([0.905, 0.20, 0.018, 0.56])
    colorbar = figure.colorbar(contour_set, cax=color_axis)
    colorbar.set_label("Potential energy", fontsize=7.2)
    colorbar.ax.tick_params(labelsize=6.6, length=2.5, width=0.6)

    figure.text(
        0.49,
        0.025,
        "At each budget, each method uses its last stored checkpoint with NoF <= budget; exact checkpoints are recorded in the source CSV.",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#444444",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.875,
        bottom=0.095,
        top=0.84,
        wspace=0.24,
        hspace=0.30,
    )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    QA_PNG.parent.mkdir(parents=True, exist_ok=True)
    staged_pdf = OUTPUT_PDF.with_name(OUTPUT_PDF.stem + ".stage.pdf")
    figure.savefig(
        staged_pdf,
        format="pdf",
        dpi=600,
        metadata={
            "Title": "Q-transformed Muller N_int=10 path evolution",
            "Subject": "Six-method 12-point paths at six common NoF budgets",
        },
    )
    figure.savefig(QA_PNG, format="png", dpi=300)
    plt.close(figure)
    staged_pdf.replace(OUTPUT_PDF)

    fieldnames = [
        "target_nof_budget",
        "method",
        "source_record_index",
        "source_recorded_nof",
        "effective_nof",
        "selection_policy",
        "node_index",
        *[f"x{coordinate}" for coordinate in range(10)],
        "z0",
        "z1",
        "source_file",
        "source_sha256",
    ]
    with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)

    checkpoint_table = {
        f"{budget:g}": {
            label: {
                key: value
                for key, value in snapshots[budget][label].items()
                if key != "path"
            }
            for label in labels
        }
        for budget in TARGET_BUDGETS
    }
    manifest = {
        "schema_version": 1,
        "status": "awaiting_visual_qa",
        "case": "fixed-Q unperturbed 10D Muller-Brown",
        "resolution": {
            "interior_images": 10,
            "total_path_points": 12,
            "path_dimension": 10,
        },
        "target_nof_budgets": list(TARGET_BUDGETS),
        "methods": labels,
        "formal_ivm_source_role": "new unified IR-IVM labeled IVM",
        "formal_ivm_summary": formal_summaries["N10"],
        "selection_policy": (
            "budget zero uses the identical common initial geometry; every "
            "positive budget uses the last actually stored checkpoint with "
            "Number of Forces no greater than the target"
        ),
        "checkpoint_table": checkpoint_table,
        "initial_path_maximum_pairwise_difference": max(
            initial_differences.values()
        ),
        "coordinate_projection": "z=Q@x; displayed coordinates are z0,z1",
        "q_array_sha256": formal.EXPECTED_Q_ARRAY_SHA256,
        "plot_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "source_csv": {
            "path": str(SOURCE_CSV),
            "sha256": sha256(SOURCE_CSV),
            "row_count": len(source_rows),
        },
        "output_pdf": {
            "path": str(OUTPUT_PDF),
            "sha256": sha256(OUTPUT_PDF),
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_PDF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
