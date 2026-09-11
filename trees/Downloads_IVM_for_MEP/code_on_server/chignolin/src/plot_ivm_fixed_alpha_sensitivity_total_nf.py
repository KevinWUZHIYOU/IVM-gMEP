#!/usr/bin/env python3
"""Paper figure for fixed-alpha Chignolin IVM sensitivity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SELECTED_ALPHA = 1.0
I_PERP_LIMITS = (3.0e-2, 1.0e2)
BARRIER_LIMITS_KCAL = (20.0, 50.0)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.5,
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


def measured_first_passage_frontier(
    cost: np.ndarray,
    precision: np.ndarray,
    lower_limit: float,
    upper_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return connected measured vertices of the first-passage frontier.

    Each output vertex is an actually evaluated state that establishes a new
    best (smaller) precision value.  No threshold-grid repetition, smoothing,
    or interpolation is used, so a normal line plot cannot create an
    artificial staircase.
    """

    cost = np.asarray(cost, dtype=float)
    precision = np.asarray(precision, dtype=float)
    if cost.shape != precision.shape:
        raise ValueError("Cost and precision histories must have one shape")
    finite = np.isfinite(cost) & np.isfinite(precision)
    indices = np.flatnonzero(finite)
    if indices.size == 0:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=int),
        )
    indices = indices[np.argsort(cost[indices], kind="stable")]
    best = np.inf
    frontier_precision = []
    frontier_cost = []
    frontier_indices = []
    for index in indices:
        value = float(precision[index])
        tolerance = max(1.0e-14, 1.0e-12 * abs(best)) if np.isfinite(best) else 0.0
        if value < best - tolerance:
            best = value
            if lower_limit <= value <= upper_limit:
                frontier_precision.append(value)
                frontier_cost.append(float(cost[index]))
                frontier_indices.append(int(index))
    return (
        np.asarray(frontier_precision, dtype=float),
        np.asarray(frontier_cost, dtype=float),
        np.asarray(frontier_indices, dtype=int),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--report-namespace",
        default="ivm_fixed_alpha_sensitivity_total_nf_20260727a",
    )
    parser.add_argument(
        "--output-directory",
        default="figures/paper_methods_total_nf_20260727a",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    report_dir = root / "reports" / args.report_namespace
    manifest = json.loads((report_dir / "manifest.json").read_text())
    expected_globalization = manifest.get("state_globalization")
    dense_manifest = json.loads(
        (report_dir / "dense_manifest.json").read_text()
    )
    dense_by_alpha = {
        float(record["alpha"]): record
        for record in dense_manifest["records"]
    }
    runs = {}
    initial_hashes = set()
    for record in manifest["records"]:
        if not record.get("valid", False):
            raise RuntimeError(
                f"Invalid alpha run: alpha={record['alpha']}"
            )
        path = Path(record["result_file"]).resolve()
        if sha256(path) != record["result_sha256"]:
            raise RuntimeError(f"Alpha run changed: {path}")
        with np.load(path, allow_pickle=False) as saved:
            counters = np.asarray(saved["counter_vectors"], dtype=int)
            globalization = (
                str(saved["state_globalization"])
                if "state_globalization" in saved.files
                else None
            )
            runs[float(record["alpha"])] = {
                "path": path,
                "number_force": counters.sum(axis=1),
                "calls": np.asarray(
                    saved["calls_per_image"], dtype=int
                ),
                "i_perp": np.asarray(saved["i_perp"], dtype=float),
                "initial_hash": str(saved["initial_path_sha256"]),
                "state_globalization": globalization,
            }
        initial_hashes.add(runs[float(record["alpha"])]["initial_hash"])
    if len(initial_hashes) != 1:
        raise RuntimeError("Fixed-alpha runs do not share one initial path")
    if expected_globalization is not None and any(
        run["state_globalization"] != expected_globalization
        for run in runs.values()
    ):
        raise RuntimeError(
            "Fixed-alpha runs do not share the declared globalization"
        )

    alphas = np.asarray(sorted(runs), dtype=float)
    frontiers = {}
    output_rows = []
    for alpha in alphas:
        run = runs[float(alpha)]
        dense_record = dense_by_alpha[float(alpha)]
        dense_path = Path(dense_record["result_file"]).resolve()
        if sha256(dense_path) != dense_record["result_sha256"]:
            raise RuntimeError(f"Dense result changed: {dense_path}")
        checkpoint_by_call = {
            int(row["selected_call_per_image"]): row
            for row in dense_record["checkpoints"]
        }
        with np.load(dense_path, allow_pickle=False) as saved:
            dense_calls = np.asarray(saved["calls"], dtype=int)
            dense_barrier = np.asarray(
                saved["adaptive_barrier_kcal"], dtype=float
            )
            dense_subdivisions = np.asarray(
                saved["selected_subdivisions"], dtype=int
            )
        if any(int(call) not in checkpoint_by_call for call in dense_calls):
            raise RuntimeError(
                f"Dense force mapping is incomplete for alpha={alpha:g}"
            )
        dense_force = np.asarray(
            [
                checkpoint_by_call[int(call)][
                    "actual_number_of_force_evaluations"
                ]
                for call in dense_calls
            ],
            dtype=int,
        )
        i_precision, i_force, i_indices = measured_first_passage_frontier(
            run["number_force"],
            run["i_perp"],
            *I_PERP_LIMITS,
        )
        (
            barrier_precision,
            barrier_force,
            barrier_indices,
        ) = measured_first_passage_frontier(
            dense_force,
            dense_barrier,
            *BARRIER_LIMITS_KCAL,
        )
        frontiers[float(alpha)] = {
            "i_perp": (i_precision, i_force),
            "barrier": (barrier_precision, barrier_force),
        }
        for metric, precision, force, source_indices in (
            ("i_perp", i_precision, i_force, i_indices),
            (
                "densified_barrier_kcal",
                barrier_precision,
                barrier_force,
                barrier_indices,
            ),
        ):
            for vertex, (value, number_force, source_index) in enumerate(
                zip(precision, force, source_indices)
            ):
                subdivision = (
                    ""
                    if metric == "i_perp"
                    else int(dense_subdivisions[int(source_index)])
                )
                source_call = (
                    int(run["calls"][int(source_index)])
                    if metric == "i_perp"
                    else int(dense_calls[int(source_index)])
                )
                output_rows.append(
                    {
                        "alpha": float(alpha),
                        "metric": metric,
                        "frontier_vertex": int(vertex),
                        "attained_precision": float(value),
                        "number_of_force_evaluations": int(number_force),
                        "source_call_per_image": source_call,
                        "selected_subdivisions_per_segment": subdivision,
                        "run_sha256": sha256(run["path"]),
                        "dense_sha256": dense_record["result_sha256"],
                    }
                )

    setup_style()
    figure, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.15), sharey=True
    )
    color_norm = mpl.colors.LogNorm(
        vmin=float(np.min(alphas)), vmax=float(np.max(alphas))
    )
    color_map = mpl.colormaps["viridis"]
    handles_by_alpha = {}
    plot_order = [
        float(alpha)
        for alpha in alphas
        if not np.isclose(alpha, SELECTED_ALPHA)
    ] + [SELECTED_ALPHA]
    for alpha in plot_order:
        selected = bool(np.isclose(alpha, SELECTED_ALPHA))
        color = color_map(color_norm(alpha))
        marker = MARKERS[
            int(np.flatnonzero(np.isclose(alphas, alpha))[0])
        ]
        label = (
            rf"$\alpha={alpha:g}$ (selected)"
            if selected
            else rf"$\alpha={alpha:g}$"
        )
        for axis, key in zip(axes, ("i_perp", "barrier")):
            precision, number_force = frontiers[alpha][key]
            if precision.size == 0:
                continue
            markevery = max(1, int(np.ceil(precision.size / 13)))
            (line,) = axis.plot(
                precision,
                number_force,
                color=color,
                marker=marker,
                ms=3.5 if selected else 2.5,
                markevery=markevery,
                markeredgewidth=0.35,
                lw=2.35 if selected else 1.15,
                alpha=1.0 if selected else 0.78,
                zorder=8 if selected else 3,
                label=label,
            )
            if key == "i_perp":
                handles_by_alpha[alpha] = line
    axes[0].set_xscale("log")
    axes[0].set_xlim(I_PERP_LIMITS[1], I_PERP_LIMITS[0])
    axes[1].set_xlim(BARRIER_LIMITS_KCAL[1], BARRIER_LIMITS_KCAL[0])
    axes[0].set_xlabel(
        r"$I_\perp$ precision threshold (smaller is stricter)"
    )
    axes[1].set_xlabel(
        "Densified barrier threshold (kcal mol$^{-1}$)\n"
        "(smaller is stricter)"
    )
    axes[0].set_ylabel("Number of Force Evaluations")
    maximum_force = max(
        float(np.max(run["number_force"])) for run in runs.values()
    )
    for index, axis in enumerate(axes):
        axis.set_ylim(0.0, 1.06 * maximum_force)
        axis.yaxis.set_major_formatter(
            mpl.ticker.StrMethodFormatter("{x:,.0f}")
        )
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    ordered_handles = [handles_by_alpha[float(alpha)] for alpha in alphas]
    ordered_labels = [
        (
            rf"$\alpha={alpha:g}$ (selected)"
            if np.isclose(alpha, SELECTED_ALPHA)
            else rf"$\alpha={alpha:g}$"
        )
        for alpha in alphas
    ]
    figure.legend(
        ordered_handles,
        ordered_labels,
        ncol=6,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        columnspacing=1.0,
        handlelength=2.0,
        handletextpad=0.45,
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.21,
        top=0.72,
        wspace=0.22,
    )
    output_dir = root / args.output_directory
    outputs = save_figure(
        figure,
        output_dir,
        "chignolin_ivm_fixed_alpha_sensitivity_total_nf",
    )
    plt.close(figure)

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    table = data_dir / "chignolin_ivm_fixed_alpha_sensitivity.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0])
        )
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "figure_files": outputs,
        "data_file": str(table.resolve()),
        "selected_alpha": SELECTED_ALPHA,
        "plot_definition": (
            "Each alpha curve connects the actually measured vertices of its "
            "first-passage precision frontier with ordinary straight line "
            "segments. No step drawstyle, threshold-grid repetition, "
            "smoothing, or interpolation is used."
        ),
        "i_perp_precision_limits": list(I_PERP_LIMITS),
        "dense_barrier_precision_limits_kcal": list(
            BARRIER_LIMITS_KCAL
        ),
        "cost_axis": "total joint energy-force cache misses over 22 images",
        "cost_label": "Number of Force Evaluations",
        "diagnostic_calls_excluded_from_optimization_axis": True,
        "state_globalization": expected_globalization,
        "outer_i_perp_acceptance_guard": manifest.get(
            "outer_i_perp_acceptance_guard"
        ),
        "proximal_objective_line_search": manifest.get(
            "proximal_objective_line_search"
        ),
    }
    summary_path = (
        output_dir / "chignolin_ivm_fixed_alpha_sensitivity_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
