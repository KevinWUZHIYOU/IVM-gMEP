#!/usr/bin/env python3
"""Create the unified LEPS IVM and five-baseline production comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from leps_core import write_json


METHODS = {
    "IVM": ("runs/ivm", "ivm_production_alpha30_tight_reparam_*budget500.npz"),
    "ODE-NEB": ("runs/neb", "neb_production_k10_*budget5000.npz"),
    "Static-NEB": (
        "runs/neb",
        "production_static_k10_dt0.05_*budget3000.npz",
    ),
    "Original String": (
        "runs/original_string",
        "production_original_dt0.07_r1_*budget3000.npz",
    ),
    "OBSM-FIRE": (
        "runs/obsm_fire",
        "production_fire_stable_l0.85_n4_dt0.18_*budget3000.npz",
    ),
    "OBSM-L-BFGS": (
        "runs/obsm_lbfgs",
        "production_lbfgs_r5_l0.95_n1_s0.1_*budget3000.npz",
    ),
}
COLORS = {
    "IVM": "#0072B2",
    "ODE-NEB": "#D55E00",
    "Static-NEB": "#B279A2",
    "Original String": "#777777",
    "OBSM-FIRE": "#5B9279",
    "OBSM-L-BFGS": "#CC79A7",
}
STYLES = {
    "IVM": "-",
    "ODE-NEB": "--",
    "Static-NEB": "--",
    "Original String": ":",
    "OBSM-FIRE": "-.",
    "OBSM-L-BFGS": (0, (5, 1.5)),
}
LINEWIDTHS = {
    "IVM": 1.85,
    "ODE-NEB": 1.65,
    "Static-NEB": 1.65,
    "Original String": 1.65,
    "OBSM-FIRE": 1.65,
    "OBSM-L-BFGS": 1.65,
}
MARKERS = {
    "IVM": ("o", 3.1),
    "ODE-NEB": ("s", 2.9),
    "Static-NEB": ("v", 2.8),
    "Original String": ("D", 2.7),
    "OBSM-FIRE": ("s", 2.8),
    "OBSM-L-BFGS": ("P", 2.8),
}
ZORDERS = {
    "IVM": 20,
    "ODE-NEB": 4,
    "Static-NEB": 3,
    "Original String": 2,
    "OBSM-FIRE": 3,
    "OBSM-L-BFGS": 4,
}
FORCE_MAX = 20000
FORCE_TICKS = (0, 5000, 10000, 15000, 20000)


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def committed_ivm_view(data):
    """Exclude pre-reparameterization trial states from the IVM trace."""
    cv = np.asarray(data["segment_length_cv"], dtype=float)
    mask = cv <= 1e-10
    mask[0] = True
    indices = np.flatnonzero(mask)
    result = {}
    trace_length = len(data["calls_per_image"])
    for key, value in data.items():
        array = np.asarray(value)
        if array.ndim and array.shape[0] == trace_length:
            result[key] = array[indices]
        else:
            result[key] = value
    return result


def metric(data, key):
    if key == "barrier_error":
        return np.asarray(data["barrier_error"], dtype=float)
    if key == "densified_barrier":
        source = "dense_barrier" if "dense_barrier" in data else "barrier"
        return np.asarray(data[source], dtype=float)
    return np.asarray(data[key], dtype=float)


def number_of_forces(data):
    """Return total uncached force evaluations over the 20 internal images.

    Every evaluator cache miss computes energy and force jointly, so the
    internal-image joint-evaluation ledger is the actual force cost.  Fixed
    endpoint evaluations and cached/repeated API requests are excluded.
    """
    trace_length = len(data["calls_per_image"])
    candidates = []
    if "total_internal_calls" in data:
        candidates.append(
            ("total_internal_calls", np.asarray(data["total_internal_calls"]))
        )
    if "counter_vector" in data:
        counters = np.asarray(data["counter_vector"])
        if counters.ndim != 2 or counters.shape != (trace_length, 22):
            raise ValueError(f"Unexpected counter_vector shape: {counters.shape}")
        candidates.append(("counter_vector", np.sum(counters[:, 1:-1], axis=1)))
    if "joint_evaluations_history" in data:
        counters = np.asarray(data["joint_evaluations_history"])
        if counters.ndim != 2 or counters.shape != (trace_length, 22):
            raise ValueError(
                "Unexpected joint_evaluations_history shape: "
                f"{counters.shape}"
            )
        candidates.append(
            ("joint_evaluations_history", np.sum(counters[:, 1:-1], axis=1))
        )
    if not candidates:
        raise KeyError("No auditable per-image force-evaluation ledger")

    name, values = candidates[0]
    values = np.asarray(values)
    if values.shape != (trace_length,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"Unexpected {name} cost vector: {values.shape}, {values.dtype}")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(np.diff(values) < 0):
        raise ValueError(f"Non-monotone force ledger in {name}")
    for other_name, other_values in candidates[1:]:
        other_values = np.asarray(other_values)
        if not np.array_equal(values, other_values):
            raise ValueError(f"Force ledgers disagree: {name} versus {other_name}")
    return values


def first_call(data, key, threshold):
    values = metric(data, key)
    indices = np.flatnonzero(values <= threshold)
    return (
        int(np.asarray(data["calls_per_image"])[indices[0]])
        if len(indices)
        else None
    )


def save_figure(fig, base, *, tight=True):
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 600}
        if tight:
            kwargs["bbox_inches"] = "tight"
        fig.savefig(base.with_suffix(f".{suffix}"), **kwargs)
    plt.close(fig)


def publication_log_tick(value, _position):
    if not np.isfinite(value) or value <= 0:
        return ""
    exponent = int(np.rint(np.log10(value)))
    if not np.isclose(value, 10.0**exponent):
        return ""
    return f"1e{exponent}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    runs = {}
    paths = {}
    for name, (directory, pattern) in METHODS.items():
        matches = sorted((root / directory).glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"{name}: expected one {pattern}, found {matches}")
        paths[name] = matches[0]
        runs[name] = load(matches[0])
    runs["IVM"] = committed_ivm_view(runs["IVM"])

    rows = []
    for name, data in runs.items():
        calls = np.asarray(data["calls_per_image"], dtype=int)
        forces = number_of_forces(data)
        error = metric(data, "barrier_error")
        i_perp = metric(data, "i_perp")
        cv = np.asarray(data["segment_length_cv"], dtype=float)
        rows.append(
            {
                "method": name,
                "result": str(paths[name].relative_to(root)),
                "final_call": int(calls[-1]),
                "final_number_of_forces": int(forces[-1]),
                "final_i_perp": float(i_perp[-1]),
                "minimum_i_perp": float(np.min(i_perp)),
                "final_barrier_error_eV": float(error[-1]),
                "minimum_barrier_error_eV": float(np.min(error)),
                "final_segment_length_cv": float(cv[-1]),
                "calls_to_i_perp_1e-4": first_call(
                    data, "i_perp", 1e-4
                ),
                "calls_to_barrier_error_0.20_eV": first_call(
                    data, "barrier_error", 0.20
                ),
                "calls_to_barrier_error_0.10_eV": first_call(
                    data, "barrier_error", 0.10
                ),
                "calls_to_barrier_error_0.03_eV": first_call(
                    data, "barrier_error", 0.03
                ),
                "counter_audit_consistent": bool(
                    data["counter_audit_consistent"]
                ),
            }
        )

    table = root / "tables/leps_all_methods_production.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot_source = root / "tables/leps_all_methods_metrics_source.csv"
    plot_rows = []
    for name, data in runs.items():
        forces = number_of_forces(data)
        selected = np.flatnonzero(forces <= FORCE_MAX)
        barrier_field = "dense_barrier" if "dense_barrier" in data else "barrier"
        for plotted_index, trace_index in enumerate(selected):
            plot_rows.append(
                {
                    "method": name,
                    "plotted_index": int(plotted_index),
                    "number_of_forces": int(forces[trace_index]),
                    "i_perp": float(metric(data, "i_perp")[trace_index]),
                    "densified_barrier_eV": float(
                        metric(data, "densified_barrier")[trace_index]
                    ),
                    "barrier_npz_field": barrier_field,
                    "production_result": str(paths[name].relative_to(root)),
                }
            )
    with plot_source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
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
            "legend.borderaxespad": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 70 / 25.4), dpi=220)
    for name, data in runs.items():
        forces = number_of_forces(data)
        mask = forces <= FORCE_MAX
        markevery = max(int(np.count_nonzero(mask)) // 8, 1)
        marker, markersize = MARKERS[name]
        for ax, key in zip(axes, ("i_perp", "densified_barrier")):
            ax.plot(
                forces[mask],
                metric(data, key)[mask],
                label=name,
                color=COLORS[name],
                linestyle=STYLES[name],
                linewidth=LINEWIDTHS[name],
                marker=marker,
                markersize=markersize,
                markerfacecolor="white",
                markeredgewidth=0.7,
                markevery=markevery,
                zorder=ZORDERS[name],
            )
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(publication_log_tick))
    for ax in axes:
        ax.set_xlim(0, FORCE_MAX)
        ax.set_xticks(FORCE_TICKS)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.set_xlabel("Number of Forces")
        ax.grid(False)
    axes[0].set_ylabel(r"$I_{\perp}$")
    axes[1].set_ylabel(r"Densified energy barrier $\Delta E$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.5,
        handletextpad=0.55,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.98,
        bottom=0.16,
        top=0.78,
        wspace=0.30,
    )
    fig.canvas.draw()
    panel_letter_y = max(ax.get_position().y1 for ax in axes) + 0.01
    for ax, letter in zip(axes, ("a", "b")):
        position = ax.get_position()
        fig.text(
            position.x0 - 0.04,
            panel_letter_y,
            letter,
            ha="left",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
        )
    save_figure(fig, root / "figures/leps_all_methods_metrics", tight=False)

    i_thresholds = np.logspace(-14, 1, 16)
    barrier_thresholds = np.asarray(
        [3.2, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.03, 0.02]
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.75))
    for name, data in runs.items():
        i_cost = [first_call(data, "i_perp", value) for value in i_thresholds]
        b_cost = [
            first_call(data, "barrier_error", value)
            for value in barrier_thresholds
        ]
        for ax, thresholds, costs in (
            (axes[0], i_thresholds, i_cost),
            (axes[1], barrier_thresholds, b_cost),
        ):
            valid = np.asarray([value is not None for value in costs])
            ax.plot(
                thresholds[valid],
                np.asarray([value or np.nan for value in costs])[valid],
                color=COLORS[name],
                linestyle=STYLES[name],
                marker="o",
                markersize=2.6,
                linewidth=1.25,
                label=name,
            )
    axes[0].set_xlabel(r"Requested $I_\perp$ threshold")
    axes[1].set_xlabel(r"Requested barrier-error threshold (eV)")
    axes[0].set_ylabel("First-attainment calls per image")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.grid(True, which="major", color="#D9D9D9", linewidth=0.45)
    axes[0].text(0.02, 0.96, "a", transform=axes[0].transAxes, va="top", fontweight="bold")
    axes[1].text(0.02, 0.96, "b", transform=axes[1].transAxes, va="top", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, root / "figures/leps_all_methods_cost_to_precision")

    payload = {
        "leps_all_methods_metrics": {
            "x_axis": (
                "total uncached joint energy-force evaluations summed over "
                "the 20 internal images; fixed endpoints and cache hits excluded"
            ),
            "x_limit": [0, FORCE_MAX],
            "source_table": str(plot_source.relative_to(root)),
            "left_metric": "i_perp",
            "right_metric": (
                "densified forward barrier: dense_barrier for IVM/NEB; "
                "barrier for string-family runs"
            ),
        },
        "selection": {
            "static_neb": "k=10, static step=0.05",
            "original_string": "full-force Euler dt=0.07, reparameterize every step",
            "obsm_fire": (
                "lambda=0.85, 4 inner FIRE steps, dt=0.18, beta=0.1; "
                "selected over the faster-crossing dt=0.22 candidate by the "
                "3000-call joint whole-curve score"
            ),
            "obsm_lbfgs": (
                "lambda=0.95, 1 inner L-BFGS step, initial inverse-Hessian "
                "scale=0.1, history=7, reparameterize every 5 outer steps"
            ),
        },
        "rows": rows,
    }
    report = root / "reports/leps_all_methods_production.json"
    write_json(report, payload)
    print(table)
    print(report)


if __name__ == "__main__":
    main()
