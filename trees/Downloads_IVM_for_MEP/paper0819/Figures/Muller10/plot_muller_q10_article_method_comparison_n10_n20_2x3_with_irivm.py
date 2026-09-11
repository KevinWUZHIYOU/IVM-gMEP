#!/usr/bin/env python3
"""Redraw the registered QM10 2x3 article figure with both IR-IVM runs.

The six-panel structure and all pre-existing method data are inherited from
the registered article plotter.  Only two hash-audited overlays are added:
the exact-envelope predecessor (``Previous IR-IVM``) and the horizon-first
control-probe result (``IR-IVM``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


HERE = Path(__file__).resolve().parent
DOWNLOAD_REPOSITORY = Path(__file__).resolve().parents[3]
MAIN_REPOSITORY = Path("/Users/wuzhiyou/IVM_for_MEP")
BASE_PLOT_DIR = (
    MAIN_REPOSITORY
    / "paper/ActiveExperiments/muller_q10_10d/code/plotting"
)
if str(BASE_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_PLOT_DIR))

import plot_q10_article_n10_n20_2x3_linear3000 as base  # noqa: E402


PREVIOUS_LABEL = "Previous IR-IVM"
IRIVM_LABEL = "IR-IVM"
DATA_DIR = HERE / "data/irivm_overlay"
IR_INPUTS = {
    10: {
        PREVIOUS_LABEL: (
            DATA_DIR / "previous_n10_trajectory.npz",
            DATA_DIR / "previous_n10_summary.json",
        ),
        IRIVM_LABEL: (
            DATA_DIR / "irivm_n10_trajectory.npz",
            DATA_DIR / "irivm_n10_summary.json",
        ),
    },
    20: {
        PREVIOUS_LABEL: (
            DATA_DIR / "previous_n20_trajectory.npz",
            DATA_DIR / "previous_n20_summary.json",
        ),
        IRIVM_LABEL: (
            DATA_DIR / "irivm_n20_trajectory.npz",
            DATA_DIR / "irivm_n20_summary.json",
        ),
    },
}
DEFAULT_OUTPUT = (
    HERE / "muller_q10_article_method_comparison_n10_n20_2x3_linear3000.pdf"
)
BASE_FIGURE = (
    MAIN_REPOSITORY
    / "paper/ActiveExperiments/muller_q10_10d/figures/active_candidates/"
    "muller_q10_article_method_comparison_n10_n20_2x3_linear3000.pdf"
)
BASE_MANIFEST = BASE_FIGURE.with_name(BASE_FIGURE.stem + "_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_irivm_styles(namespace: dict) -> None:
    base.extend_styles(namespace)
    namespace["METHOD_STYLE"].update(
        {
            PREVIOUS_LABEL: {
                "linestyle": "--",
                "linewidth": 1.75,
                "color": "#6A51A3",
                "marker": "x",
                "markersize": 2.7,
                "zorder": 25,
            },
            IRIVM_LABEL: {
                "linestyle": "-",
                "linewidth": 2.0,
                "color": "#E15759",
                "marker": "o",
                "markersize": 2.6,
                "zorder": 24,
            },
        }
    )
    for method in (PREVIOUS_LABEL, IRIVM_LABEL):
        namespace["GRAD_EVALS_PER_COUNT_BY_METHOD"][method] = 1
    # The contour panels retain their registered three-method policy.  The two
    # supplemental runs are added only to the two cost-to-precision panels.


def make_irivm_run(
    trajectory_path: Path,
    summary_path: Path,
    label: str,
    n_internal: int,
) -> tuple[dict, dict]:
    summary = load_json(summary_path)
    if int(summary["n_internal"]) != int(n_internal):
        raise RuntimeError(f"resolution mismatch in {summary_path}")
    if int(summary["nof_per_image_sum"]) != int(summary["terminal_nof"]):
        raise RuntimeError(f"NoF ledger does not close in {summary_path}")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        nof = np.asarray(archive["public_nf"], dtype=float)
        paths = np.asarray(archive["public_path"], dtype=float)
        i_perp = np.asarray(archive["public_i_perp"], dtype=float)
        delta_e = np.asarray(archive["public_delta_e"], dtype=float)
    if not (
        nof.ndim == 1
        and paths.shape[0] == nof.size
        and i_perp.shape == nof.shape
        and delta_e.shape == nof.shape
        and np.all(np.diff(nof) > 0.0)
        and np.all(np.isfinite(paths))
        and np.all(np.isfinite(i_perp))
        and np.all(np.isfinite(delta_e))
        and np.all(i_perp > 0.0)
        and np.all(delta_e > 0.0)
    ):
        raise RuntimeError(f"invalid IR-IVM trajectory: {trajectory_path}")
    run = {
        "method": label,
        "label": label,
        "short_label": label,
        "details": (
            f"{label}: alpha={summary['alpha']}, r={summary['r']}, "
            f"m_max={summary['m_max']}"
        ),
        "file": trajectory_path,
        "images_seq": paths,
        "count_seq": nof,
        "metrics": {
            "count_seq": nof,
            "grad_count_seq": nof,
            "loss_perp": i_perp,
            "Delta_Energy": delta_e,
        },
    }
    return run, summary


def validate_pair(
    n_internal: int,
    previous: dict,
    current: dict,
    previous_summary: dict,
    current_summary: dict,
) -> None:
    expected = 79 if n_internal == 10 else 98
    if int(current_summary["i_perp_rise_count"]) != 0:
        raise RuntimeError(f"N={n_internal} current IR-IVM is not zero-rise")
    for field in (
        "strict_i_perp_count",
        "strict_barrier_count",
        "strict_joint_count",
        "checkpoint_count",
    ):
        if int(current_summary[field]) != expected:
            raise RuntimeError(
                f"N={n_internal} current IR-IVM fails {field}: "
                f"{current_summary[field]} != {expected}"
            )
    expected_previous_rises = 0 if n_internal == 10 else 1
    if int(previous_summary["i_perp_rise_count"]) != expected_previous_rises:
        raise RuntimeError(f"unexpected predecessor rise count for N={n_internal}")
    if not np.array_equal(previous["count_seq"], current["count_seq"]):
        raise RuntimeError(f"N={n_internal} IR-IVM NoF arrays differ")
    if n_internal == 10:
        fields = ("images_seq",)
        metric_fields = ("loss_perp", "Delta_Energy")
        if any(
            not np.array_equal(previous[field], current[field])
            for field in fields
        ) or any(
            not np.array_equal(
                previous["metrics"][field], current["metrics"][field]
            )
            for field in metric_fields
        ):
            raise RuntimeError("N=10 predecessor and current trajectories differ")
    else:
        after_100 = previous["count_seq"] >= 100.0
        if not np.array_equal(
            previous["images_seq"][after_100],
            current["images_seq"][after_100],
        ):
            raise RuntimeError("N=20 paths differ from NoF 100 onward")
        for field in ("loss_perp", "Delta_Energy"):
            if not np.array_equal(
                previous["metrics"][field][after_100],
                current["metrics"][field][after_100],
            ):
                raise RuntimeError(
                    f"N=20 {field} differs from NoF 100 onward"
                )


def write_source_csv(path: Path, rows_by_case: list[tuple[str, list[dict]]]) -> None:
    fieldnames = (
        "case",
        "method",
        "number_of_forces",
        "I_perp",
        "Delta_E_dense_linear500",
        "source_file",
        "source_sha256",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case_name, runs in rows_by_case:
            for run in runs:
                metrics = run["metrics"]
                file_path = Path(run["file"])
                file_hash = sha256(file_path)
                for index, nof in enumerate(metrics["grad_count_seq"]):
                    writer.writerow(
                        {
                            "case": case_name,
                            "method": run["label"],
                            "number_of_forces": f"{float(nof):.17g}",
                            "I_perp": f"{float(metrics['loss_perp'][index]):.17g}",
                            "Delta_E_dense_linear500": (
                                f"{float(metrics['Delta_Energy'][index]):.17g}"
                            ),
                            "source_file": str(file_path),
                            "source_sha256": file_hash,
                        }
                    )


def draw_figure(
    namespace: dict,
    n10_runs: list[dict],
    n20_runs: list[dict],
) -> tuple[plt.Figure, dict, dict]:
    figure = plt.figure(
        figsize=(namespace["mm_to_inch"](180), namespace["mm_to_inch"](126)),
        dpi=220,
    )
    grid = GridSpec(
        2,
        3,
        figure=figure,
        width_ratios=(1.0, 1.0, 1.0),
        left=0.068,
        right=0.94,
        bottom=0.08,
        top=0.90,
        wspace=0.42,
        hspace=0.12,
    )
    axes = [
        [figure.add_subplot(grid[row, column]) for column in range(3)]
        for row in range(2)
    ]
    n10_checkpoints = base.draw_row(namespace, axes[0], n10_runs)
    n20_checkpoints = base.draw_row(namespace, axes[1], n20_runs)
    for axis in axes[0]:
        axis.set_xlabel("")

    figure.canvas.draw()
    row_titles = (
        r"$\mathbf{N}_{\mathrm{int}}=\mathbf{10}$",
        r"$\mathbf{N}_{\mathrm{int}}=\mathbf{20}$",
    )
    for row, (letters, row_title) in enumerate(
        zip((("a", "b", "c"), ("d", "e", "f")), row_titles)
    ):
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
            figure.add_artist(
                Line2D(
                    [line_start, line_end],
                    [header_y, header_y],
                    transform=figure.transFigure,
                    color="#C2C2BE",
                    linewidth=0.6,
                    solid_capstyle="butt",
                    clip_on=False,
                    zorder=2,
                )
            )
        figure.text(
            header_center,
            header_y,
            row_title,
            ha="center",
            va="center",
            color="#272727",
            fontsize=8.4,
            fontweight="normal",
            zorder=3,
        )
        for axis, letter in zip(axes[row], letters):
            position = axis.get_position()
            figure.text(
                position.x0 - 0.04,
                row_top + 0.001,
                letter,
                ha="left",
                va="bottom",
                fontweight="bold",
                zorder=3,
                fontsize=namespace["plt"].rcParams["font.size"] + 1,
            )

    handles, labels = axes[0][0].get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    # Matplotlib fills a multi-row figure legend column by column.  This
    # interleaving yields the intended visual rows:
    # IVM / Previous IR-IVM / IR-IVM / ODE-NEB, then the other four methods.
    legend_order = (
        "IVM",
        "Static NEB",
        PREVIOUS_LABEL,
        "Original String",
        IRIVM_LABEL,
        "OBSM-FIRE",
        "ODE-NEB",
        "OBSM-LBFGS",
    )
    figure.legend(
        [unique[label] for label in legend_order],
        list(legend_order),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=4,
        frameon=False,
        columnspacing=1.05,
        handlelength=2.2,
        handletextpad=0.45,
        fontsize=6.5,
    )
    return figure, n10_checkpoints, n20_checkpoints


def main() -> int:
    arguments = parse_args()
    namespace = base.load_formal_namespace(base.N10_ROOT)
    configure_irivm_styles(namespace)
    n10_runs, _ = base.load_n10_runs(namespace)
    n20_runs, _ = base.load_n20_runs(namespace)

    ir_summaries: dict[str, dict] = {}
    for n_internal, runs in ((10, n10_runs), (20, n20_runs)):
        ir_runs = {}
        for label in (PREVIOUS_LABEL, IRIVM_LABEL):
            trajectory_path, summary_path = IR_INPUTS[n_internal][label]
            run, summary = make_irivm_run(
                trajectory_path, summary_path, label, n_internal
            )
            ir_runs[label] = run
            ir_summaries[f"N{n_internal}:{label}"] = summary
        validate_pair(
            n_internal,
            ir_runs[PREVIOUS_LABEL],
            ir_runs[IRIVM_LABEL],
            ir_summaries[f"N{n_internal}:{PREVIOUS_LABEL}"],
            ir_summaries[f"N{n_internal}:{IRIVM_LABEL}"],
        )
        runs[1:1] = [ir_runs[PREVIOUS_LABEL], ir_runs[IRIVM_LABEL]]

    figure, n10_checkpoints, n20_checkpoints = draw_figure(
        namespace, n10_runs, n20_runs
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    staged_pdf = arguments.output.with_name(arguments.output.stem + ".stage.pdf")
    output_png = arguments.output.with_suffix(".png")
    staged_png = output_png.with_name(output_png.stem + ".stage.png")
    figure.savefig(staged_pdf, format="pdf", dpi=600)
    figure.savefig(staged_png, format="png", dpi=300)
    plt.close(figure)
    staged_pdf.replace(arguments.output)
    staged_png.replace(output_png)

    source_csv = arguments.output.with_name(arguments.output.stem + "_source.csv")
    staged_csv = source_csv.with_name(source_csv.stem + ".stage.csv")
    write_source_csv(
        staged_csv,
        [
            ("10 interior images", n10_runs),
            ("20 interior images", n20_runs),
        ],
    )
    staged_csv.replace(source_csv)

    base_frozen = {
        "10_interior": {
            method: {
                "path": str(base.N10_ROOT / relative),
                "sha256": sha256(base.N10_ROOT / relative),
            }
            for method, relative in base.N10_FROZEN_RUNS.items()
        },
        "20_interior": {
            method: {
                "path": str(base.N20_ROOT / relative),
                "sha256": sha256(base.N20_ROOT / relative),
            }
            for method, relative in base.N20_FROZEN_RUNS.items()
        },
    }
    ir_inputs = {
        f"N{n_internal}:{label}": {
            "trajectory": str(IR_INPUTS[n_internal][label][0]),
            "trajectory_sha256": sha256(IR_INPUTS[n_internal][label][0]),
            "summary": str(IR_INPUTS[n_internal][label][1]),
            "summary_sha256": sha256(IR_INPUTS[n_internal][label][1]),
        }
        for n_internal in (10, 20)
        for label in (PREVIOUS_LABEL, IRIVM_LABEL)
    }
    manifest = {
        "schema_version": 2,
        "status": "awaiting_visual_qa",
        "case": "fixed-Q unperturbed 10D Muller-Brown",
        "layout": "2x3; 10 interior images above 20 interior images",
        "xscale": "linear",
        "xlim": [0, 3000],
        "cost_axis": (
            "direct sum of uncached force calls over all interior images"
        ),
        "energy_barrier_metric": (
            "500-point coordinate-wise linear densified Delta E"
        ),
        "energy_barrier_y_axis": "raw Delta E on standard logarithmic scale",
        "methods": [run["label"] for run in n10_runs],
        "contour_methods": sorted(base.CONTOUR_LABELS),
        "contour_checkpoint_policy": (
            "last actually paid checkpoint with total forces <= 3000"
        ),
        "contour_checkpoint_forces": {
            "10_interior": n10_checkpoints,
            "20_interior": n20_checkpoints,
        },
        "base_registered_figure": {
            "path": str(BASE_FIGURE),
            "sha256": sha256(BASE_FIGURE),
            "manifest": str(BASE_MANIFEST),
            "manifest_sha256": sha256(BASE_MANIFEST),
        },
        "base_frozen_runs": base_frozen,
        "irivm_inputs": ir_inputs,
        "irivm_validation": {
            "N10_predecessor_equals_current_bitwise": True,
            "N20_predecessor_rises": 1,
            "N20_current_rises": 0,
            "N20_current_joint_strict_checkpoints": "98/98",
            "N20_predecessor_equals_current_from_nof": 100,
        },
        "source_csv": {
            "path": str(source_csv),
            "sha256": sha256(source_csv),
        },
        "plot_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "outputs": {
            str(arguments.output): sha256(arguments.output),
            str(output_png): sha256(output_png),
        },
    }
    manifest_path = arguments.output.with_name(
        arguments.output.stem + "_manifest.json"
    )
    staged_manifest = manifest_path.with_name(manifest_path.stem + ".stage.json")
    staged_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staged_manifest.replace(manifest_path)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
