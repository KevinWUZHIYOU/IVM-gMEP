#!/usr/bin/env python3
"""Select LEPS production runs, export tables, and make publication figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

from leps_core import (
    REACTANT,
    flat_to_physical,
    leps_energy,
    reference_saddle,
)


IVM_COLOR = "#0072B2"
NEB_COLOR = "#D55E00"
GREY = "#6E6E6E"
LIGHT_GREY = "#B8B8B8"
LOG_FLOORS = {"i_perp": 1e-14, "barrier_error": 1e-8}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.minor.width": 0.55,
            "ytick.minor.width": 0.55,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.7,
            "ytick.minor.size": 1.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def load_run(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    data["_path"] = path
    return data


def scalar(data: dict, key: str):
    value = data[key]
    return value.item() if np.asarray(value).shape == () else value


def step_sample(data: dict, key: str, grid: np.ndarray) -> np.ndarray:
    calls = np.asarray(data["calls_per_image"], dtype=int)
    values = np.asarray(data[key], dtype=float)
    indices = np.searchsorted(calls, grid, side="right") - 1
    return values[np.clip(indices, 0, len(values) - 1)]


def value_at_call(data: dict, key: str, call: int) -> float:
    return float(step_sample(data, key, np.asarray([call], dtype=int))[0])


def precision_from_label(label: str) -> str:
    if "_near_exact_" in label:
        return "near_exact"
    if "_tight_" in label:
        return "tight"
    return "inexact"


def log_auc(data: dict, key: str, budget: int, floor: float) -> float:
    grid = np.arange(1, budget + 1)
    values = step_sample(data, key, grid)
    return float(np.mean(np.log10(np.maximum(values, floor))))


def committed_ivm_view(data: dict) -> dict:
    """Return initial and post-reparameterization outer iterates only."""
    if str(scalar(data, "reparameterize")) != "post_hermite":
        return data
    cv = np.asarray(data["segment_length_cv"], dtype=float)
    mask = cv <= 1e-10
    mask[0] = True
    indices = np.flatnonzero(mask)
    result = {}
    trace_length = len(data["calls_per_image"])
    for key, value in data.items():
        if key.startswith("_"):
            result[key] = value
        elif np.asarray(value).ndim and np.asarray(value).shape[0] == trace_length:
            result[key] = np.asarray(value)[indices]
        else:
            result[key] = value
    return result


def discover_neb_scan(root: Path) -> list[dict]:
    return [
        load_run(path)
        for path in sorted((root / "runs" / "neb").glob("neb_scan_*_budget500.npz"))
    ]


def discover_ivm_grid(root: Path) -> list[dict]:
    return [
        load_run(path)
        for path in sorted(
            (root / "runs" / "ivm").glob("ivm_alpha_*_budget*.npz")
        )
    ]


def neb_scan_row(data: dict) -> dict:
    final = -1
    score = log_auc(data, "i_perp", 500, LOG_FLOORS["i_perp"])
    score += log_auc(data, "barrier_error", 500, LOG_FLOORS["barrier_error"])
    cv = float(data["segment_length_cv"][final])
    barrier_error = float(data["barrier_error"][final])
    feasible = cv <= 0.10 and barrier_error <= 0.10
    return {
        "k": float(scalar(data, "spring_constant")),
        "log_auc_i_perp": log_auc(
            data, "i_perp", 500, LOG_FLOORS["i_perp"]
        ),
        "log_auc_barrier_error": log_auc(
            data, "barrier_error", 500, LOG_FLOORS["barrier_error"]
        ),
        "composite_log_auc": score,
        "final_i_perp": float(data["i_perp"][final]),
        "final_fmax": float(data["fmax"][final]),
        "final_dense_barrier": float(data["dense_barrier"][final]),
        "final_barrier_error": barrier_error,
        "final_equidistance_integral": float(
            data["equidistance_integral"][final]
        ),
        "final_segment_length_cv": cv,
        "spacing_feasible": feasible,
        "result_file": str(data["_path"]),
    }


def neb_production_row(data: dict) -> dict:
    budget = 5000
    score = log_auc(data, "i_perp", budget, LOG_FLOORS["i_perp"])
    score += log_auc(
        data, "barrier_error", budget, LOG_FLOORS["barrier_error"]
    )
    return {
        "k": float(scalar(data, "spring_constant")),
        "log_auc_i_perp": log_auc(
            data, "i_perp", budget, LOG_FLOORS["i_perp"]
        ),
        "log_auc_barrier_error": log_auc(
            data, "barrier_error", budget, LOG_FLOORS["barrier_error"]
        ),
        "composite_log_auc": score,
        "final_call": int(data["calls_per_image"][-1]),
        "final_i_perp": float(data["i_perp"][-1]),
        "final_fmax": float(data["fmax"][-1]),
        "final_dense_barrier": float(data["dense_barrier"][-1]),
        "final_barrier_error": float(data["barrier_error"][-1]),
        "final_equidistance_integral": float(
            data["equidistance_integral"][-1]
        ),
        "final_segment_length_cv": float(data["segment_length_cv"][-1]),
        "result_file": str(data["_path"]),
    }


def ivm_grid_row(data: dict) -> dict:
    label = str(scalar(data, "run_label"))
    precision = precision_from_label(label)
    reparameterization = (
        "post_hermite"
        if str(scalar(data, "reparameterize")) == "post_hermite"
        else "none"
    )
    if "subproblem_kkt_fraction" in data:
        raw_kkt_fraction = np.asarray(
            data["subproblem_kkt_fraction"], dtype=float
        )
        finite_kkt = raw_kkt_fraction[np.isfinite(raw_kkt_fraction)]
    else:
        finite_kkt = np.asarray([], dtype=float)
    score = log_auc(data, "i_perp", 200, LOG_FLOORS["i_perp"])
    score += log_auc(data, "barrier_error", 200, LOG_FLOORS["barrier_error"])
    return {
        "alpha": float(scalar(data, "alpha_initial")),
        "precision": precision,
        "reparameterization": reparameterization,
        "kkt_rtol": float(scalar(data, "kkt_rtol")),
        "inner_calls": int(scalar(data, "inner_calls")),
        "subproblem_solver": str(scalar(data, "subproblem_solver")),
        "mean_image_subproblem_kkt_pass_fraction": (
            float(np.mean(finite_kkt)) if finite_kkt.size else np.nan
        ),
        "minimum_outer_subproblem_kkt_pass_fraction": (
            float(np.min(finite_kkt)) if finite_kkt.size else np.nan
        ),
        "fraction_outer_solves_with_all_images_kkt_satisfied": (
            float(np.mean(finite_kkt >= 1.0 - 1e-12))
            if finite_kkt.size
            else np.nan
        ),
        "composite_log_auc": score,
        "log_auc_i_perp": log_auc(
            data, "i_perp", 200, LOG_FLOORS["i_perp"]
        ),
        "log_auc_barrier_error": log_auc(
            data, "barrier_error", 200, LOG_FLOORS["barrier_error"]
        ),
        "comparison_call": 200,
        "i_perp_at_200": value_at_call(data, "i_perp", 200),
        "fmax_at_200": value_at_call(data, "fmax", 200),
        "dense_barrier_at_200": value_at_call(data, "dense_barrier", 200),
        "barrier_error_at_200": value_at_call(data, "barrier_error", 200),
        "i_arc_at_200": value_at_call(data, "i_arc", 200),
        "spring_force_max_unit_at_200": value_at_call(
            data, "spring_force_max_unit", 200
        ),
        "segment_length_cv_at_200": value_at_call(
            data, "segment_length_cv", 200
        ),
        "actual_final_call": int(data["calls_per_image"][-1]),
        "actual_final_i_perp": float(data["i_perp"][-1]),
        "actual_final_dense_barrier": float(data["dense_barrier"][-1]),
        "actual_final_i_arc": float(data["i_arc"][-1]),
        "actual_final_spring_force_max_unit": float(
            data["spring_force_max_unit"][-1]
        ),
        "stop_reason": str(scalar(data, "stop_reason")),
        "result_file": str(data["_path"]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            base.with_suffix(f".{suffix}"),
            dpi=600 if suffix == "png" else None,
        )
    plt.close(fig)


def draw_metric(
    ax,
    ivm: dict,
    neb: dict,
    key: str,
    ylabel: str,
    log: bool,
    budget: int = 500,
) -> None:
    styles = (
        (ivm, "IVM", IVM_COLOR, "-", "o"),
        (neb, "ODE-NEB", NEB_COLOR, "--", "s"),
    )
    for data, label, color, linestyle, marker in styles:
        calls = np.asarray(data["calls_per_image"])
        values = np.asarray(data[key])
        mask = calls <= budget
        ax.plot(
            calls[mask],
            values[mask],
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
            marker=marker,
            markersize=2.7,
            markevery=max(1, int(np.count_nonzero(mask) / 10)),
            label=label,
        )
    if log:
        ax.set_yscale("log")
    ax.set_xlim(0, budget)
    ax.set_xlabel("Max evaluations per internal image")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.45)
    ax.grid(True, which="minor", color="#ECECEC", linewidth=0.3)


def plot_main(root: Path, ivm: dict, neb: dict, saddle: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.08, 5.55))
    draw_metric(
        axes[0, 0], ivm, neb, "i_perp", r"$I_{\perp}$", log=True
    )
    draw_metric(
        axes[0, 1],
        ivm,
        neb,
        "i_arc",
        r"$I_{\mathrm{arc}}$",
        log=True,
    )
    draw_metric(
        axes[1, 0],
        ivm,
        neb,
        "dense_barrier",
        r"Forward barrier $\Delta E$ (eV)",
        log=False,
    )
    axes[1, 0].axhline(
        saddle["forward_barrier"],
        color=GREY,
        linewidth=0.9,
        linestyle=":",
        label=r"Exact saddle $\Delta E^\ast$",
    )

    ax = axes[1, 1]
    coordinate = np.linspace(0.55, 8.15, 420)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate)
    relative = leps_energy(x_grid, y_grid) - float(
        leps_energy(REACTANT[0], REACTANT[1])
    )
    levels = np.linspace(0.0, 6.5, 40)
    clipped = np.clip(relative, levels[0], levels[-1])
    contour = ax.contourf(
        x_grid,
        y_grid,
        clipped,
        levels=levels,
        cmap="Greys_r",
        norm=BoundaryNorm(levels, 256),
        extend="max",
    )
    ax.contour(
        x_grid,
        y_grid,
        clipped,
        levels=levels[::3],
        colors="#777777",
        linewidths=0.28,
        alpha=0.48,
    )
    initial = flat_to_physical(np.asarray(ivm["initial_path"]))
    ivm_path = flat_to_physical(np.asarray(ivm["final_path"]))
    neb_path = flat_to_physical(np.asarray(neb["final_path"]))
    ax.plot(
        initial[:, 0],
        initial[:, 1],
        color=LIGHT_GREY,
        linestyle=":",
        linewidth=1.2,
        label="Initial path",
    )
    ax.plot(
        ivm_path[:, 0],
        ivm_path[:, 1],
        color=IVM_COLOR,
        linewidth=1.7,
        marker="o",
        markersize=2.5,
        label=f"IVM ({int(ivm['calls_per_image'][-1])} calls)",
    )
    ax.plot(
        neb_path[:, 0],
        neb_path[:, 1],
        color=NEB_COLOR,
        linewidth=1.5,
        linestyle="--",
        marker="s",
        markersize=2.3,
        label=f"ODE-NEB ({int(neb['calls_per_image'][-1])} calls)",
    )
    ax.scatter(
        saddle["point"][0],
        saddle["point"][1],
        marker="*",
        s=38,
        facecolor="#CC79A7",
        edgecolor="#333333",
        linewidth=0.4,
        zorder=8,
        label="Exact saddle",
    )
    ax.set_xlim(0.55, 8.15)
    ax.set_ylim(0.55, 8.15)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$r_{\mathrm{AB}}$")
    ax.set_ylabel(r"$r_{\mathrm{BC}}$")
    colorbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(r"$V-V_{\mathrm{reactant}}$ (eV)")

    for index, axis in enumerate(axes.flat):
        axis.text(
            -0.15,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=8.5,
            fontweight="bold",
            va="top",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, loc="upper right")
    axes[1, 0].legend(loc="best")
    ax.legend(loc="upper right", handlelength=1.8)
    fig.tight_layout(w_pad=1.2, h_pad=1.2)
    save_figure(fig, root / "figures" / "leps_ivm_vs_neb")


def plot_neb_scan(root: Path, rows: list[dict], selected_k: float) -> None:
    rows = sorted(rows, key=lambda row: row["k"])
    k = np.asarray([row["k"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.25), sharex=True)
    metrics = (
        ("final_i_perp", r"$I_{\perp}$"),
        ("final_barrier_error", r"$|\Delta E-\Delta E^\ast|$ (eV)"),
        ("final_segment_length_cv", "Segment-length CV"),
    )
    for ax, (key, ylabel) in zip(axes, metrics):
        values = np.asarray([row[key] for row in rows])
        feasible = np.asarray([row["spacing_feasible"] for row in rows])
        ax.plot(k, values, color=GREY, linewidth=0.9, zorder=1)
        ax.scatter(
            k[~feasible],
            values[~feasible],
            s=18,
            facecolors="none",
            edgecolors=LIGHT_GREY,
            linewidths=0.8,
            label="Excluded" if key == "final_i_perp" else None,
        )
        ax.scatter(
            k[feasible],
            values[feasible],
            s=20,
            color=NEB_COLOR,
            label="Feasible" if key == "final_i_perp" else None,
        )
        selected = np.isclose(k, selected_k)
        ax.scatter(
            k[selected],
            values[selected],
            marker="*",
            s=55,
            color="#CC79A7",
            edgecolor="#333333",
            linewidth=0.45,
            zorder=4,
            label="Selected" if key == "final_i_perp" else None,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("NEB spring constant $k$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", color="#E0E0E0", linewidth=0.4)
    axes[0].legend(loc="best")
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, root / "figures" / "leps_neb_k_scan")


def plot_alpha_scan(root: Path, rows: list[dict], selected_alpha: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.08, 5.0))
    style = {
        ("near_exact", "post_hermite"): (
            "#6A3D9A", "-", "o", "Near-exact + resampling"
        ),
        ("near_exact", "none"): (
            "#CAB2D6", "--", "s", "Near-exact, no resampling"
        ),
        ("tight", "post_hermite"): (IVM_COLOR, "-", "^", "Tight + resampling"),
        ("tight", "none"): ("#56B4E9", "--", "v", "Tight, no resampling"),
        ("inexact", "post_hermite"): (
            "#009E73", "-", "D", "Inexact + resampling"
        ),
        ("inexact", "none"): ("#E69F00", "--", "P", "Inexact, no resampling"),
    }
    metrics = (
        ("i_perp_at_200", r"$I_{\perp}$ at 200 calls/image", True),
        (
            "dense_barrier_at_200",
            r"$\Delta E$ at 200 calls/image (eV)",
            False,
        ),
        ("i_arc_at_200", r"$I_{\mathrm{arc}}$ at 200 calls/image", True),
        (
            "spring_force_max_unit_at_200",
            r"Max unit-$k$ spring force at 200 calls/image",
            True,
        ),
    )
    for (precision, reparam), (color, ls, marker, label) in style.items():
        subset = sorted(
            [
                row
                for row in rows
                if row["precision"] == precision
                and row["reparameterization"] == reparam
            ],
            key=lambda row: row["alpha"],
        )
        x = np.asarray([row["alpha"] for row in subset])
        for ax, (key, ylabel, log_y) in zip(axes.flat, metrics):
            y = np.asarray([row[key] for row in subset])
            ax.plot(
                x,
                y,
                color=color,
                linestyle=ls,
                marker=marker,
                markersize=3.0,
                linewidth=1.35,
                label=label,
            )
            ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")
            ax.set_xlabel(r"Fixed proximal parameter $\alpha$")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="major", color="#E0E0E0", linewidth=0.4)
    for axis in axes.flat:
        axis.axvline(
            selected_alpha, color=GREY, linestyle=":", linewidth=0.8
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        columnspacing=0.8,
        handlelength=1.7,
    )
    reference_line = axes[0, 1].axhline(
        reference_saddle()["forward_barrier"],
        color=GREY,
        linestyle=":",
        linewidth=0.8,
        label=r"Exact saddle $\Delta E^\ast$",
    )
    axes[0, 1].legend(handles=[reference_line], loc="best")
    for index, axis in enumerate(axes.flat):
        axis.text(
            -0.16,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=8.5,
            fontweight="bold",
            va="top",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.1, h_pad=1.2)
    save_figure(fig, root / "figures" / "leps_alpha_sensitivity")


def plot_ablation(root: Path, grid: list[dict], alpha: float) -> None:
    selected = []
    for data in grid:
        if np.isclose(float(scalar(data, "alpha_initial")), alpha):
            selected.append(data)
    fig, axes = plt.subplots(2, 2, figsize=(7.08, 5.0))
    palette = {
        ("near_exact", "post_hermite"): (
            "#6A3D9A", "-", "Near-exact + resampling"
        ),
        ("near_exact", "none"): (
            "#CAB2D6", "--", "Near-exact, no resampling"
        ),
        ("tight", "post_hermite"): (IVM_COLOR, "-", "Tight + resampling"),
        ("tight", "none"): ("#56B4E9", "--", "Tight, no resampling"),
        ("inexact", "post_hermite"): ("#009E73", "-", "Inexact + resampling"),
        ("inexact", "none"): ("#E69F00", "--", "Inexact, no resampling"),
    }
    metrics = (
        ("i_perp", r"$I_{\perp}$", True),
        ("dense_barrier", r"Forward barrier $\Delta E$ (eV)", False),
        ("i_arc", r"$I_{\mathrm{arc}}$", True),
        (
            "spring_force_max_unit",
            r"Max unit-$k$ spring force",
            True,
        ),
    )
    for data in selected:
        data = committed_ivm_view(data)
        label = str(scalar(data, "run_label"))
        precision = precision_from_label(label)
        reparam = (
            "post_hermite"
            if str(scalar(data, "reparameterize")) == "post_hermite"
            else "none"
        )
        color, ls, display = palette[(precision, reparam)]
        calls = np.asarray(data["calls_per_image"])
        for ax, (key, ylabel, log_y) in zip(axes.flat, metrics):
            values = np.asarray(data[key])
            ax.plot(
                calls,
                values,
                color=color,
                linestyle=ls,
                linewidth=1.35,
                label=display,
            )
            if log_y:
                ax.set_yscale("log")
            ax.set_xlim(0, 200)
            ax.set_xlabel("Max evaluations per internal image")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="major", color="#E0E0E0", linewidth=0.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        columnspacing=0.8,
        handlelength=1.7,
    )
    reference_line = axes[0, 1].axhline(
        reference_saddle()["forward_barrier"],
        color=GREY,
        linestyle=":",
        linewidth=0.8,
        label=r"Exact saddle $\Delta E^\ast$",
    )
    axes[0, 1].legend(handles=[reference_line], loc="best")
    for index, axis in enumerate(axes.flat):
        axis.text(
            -0.16,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=8.5,
            fontweight="bold",
            va="top",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.1, h_pad=1.2)
    save_figure(fig, root / "figures" / "leps_subproblem_reparam_ablation")


def plot_alpha_trajectories(
    root: Path,
    grid: list[dict],
    selected_alpha: float,
    saddle: dict,
) -> None:
    """Plot all fixed-alpha near-exact, reparameterized trajectories."""
    selected = []
    for data in grid:
        label = str(scalar(data, "run_label"))
        if (
            precision_from_label(label) == "near_exact"
            and str(scalar(data, "reparameterize")) == "post_hermite"
        ):
            selected.append(committed_ivm_view(data))
    selected.sort(key=lambda data: float(scalar(data, "alpha_initial")))
    if not selected:
        return

    fig, axes = plt.subplots(2, 2, figsize=(7.08, 5.0))
    metrics = (
        ("i_perp", r"$I_{\perp}$", True),
        ("dense_barrier", r"Forward barrier $\Delta E$ (eV)", False),
        ("i_arc", r"$I_{\mathrm{arc}}$", True),
        ("spring_force_max_unit", r"Max unit-$k$ spring force", True),
    )
    colors = plt.get_cmap("viridis")(
        np.linspace(0.05, 0.95, len(selected))
    )
    for color, data in zip(colors, selected):
        alpha = float(scalar(data, "alpha_initial"))
        calls = np.asarray(data["calls_per_image"], dtype=int)
        mask = calls <= 100
        width = 2.25 if np.isclose(alpha, selected_alpha) else 1.05
        zorder = 5 if np.isclose(alpha, selected_alpha) else 2
        label = (
            rf"$\alpha={alpha:g}$ (selected)"
            if np.isclose(alpha, selected_alpha)
            else rf"$\alpha={alpha:g}$"
        )
        for ax, (key, ylabel, log_y) in zip(axes.flat, metrics):
            ax.plot(
                calls[mask],
                np.asarray(data[key], dtype=float)[mask],
                color=color,
                linewidth=width,
                label=label,
                zorder=zorder,
            )
            if log_y:
                ax.set_yscale("log")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Max evaluations per internal image")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="major", color="#E0E0E0", linewidth=0.4)
    reference_line = axes[0, 1].axhline(
        saddle["forward_barrier"],
        color=GREY,
        linestyle=":",
        linewidth=0.9,
        label=r"Exact saddle $\Delta E^\ast$",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=5,
        columnspacing=0.7,
        handlelength=1.6,
    )
    axes[0, 1].legend(handles=[reference_line], loc="best")
    for index, axis in enumerate(axes.flat):
        axis.text(
            -0.16,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=8.5,
            fontweight="bold",
            va="top",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.1, h_pad=1.2)
    save_figure(fig, root / "figures" / "leps_alpha_trajectories")


def path_at_call(data: dict, call: int) -> np.ndarray:
    if call <= 0:
        return np.asarray(data["initial_path"], dtype=float)
    calls = np.asarray(data["calls_per_image"], dtype=int)
    index = int(np.searchsorted(calls, call, side="right") - 1)
    index = int(np.clip(index, 0, len(calls) - 1))
    return np.asarray(data["path"][index], dtype=float)


def plot_path_evolution_1x4(
    root: Path,
    ivm: dict,
    neb: dict,
    saddle: dict,
) -> None:
    """Article-style equal-cost snapshots on the exact LEPS PES."""
    snapshot_calls = (0, 30, 200, 500)
    coordinate = np.linspace(0.4, 8.2, 440)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate)
    relative = leps_energy(x_grid, y_grid) - float(
        leps_energy(REACTANT[0], REACTANT[1])
    )
    levels = np.linspace(0.0, 6.5, 41)
    clipped = np.clip(relative, levels[0], levels[-1])

    fig, axes = plt.subplots(
        1, 4, figsize=(7.08, 2.12), sharex=True, sharey=True
    )
    contour = None
    for panel, (ax, call) in enumerate(zip(axes, snapshot_calls)):
        contour = ax.contourf(
            x_grid,
            y_grid,
            clipped,
            levels=levels,
            cmap="Greys_r",
            norm=BoundaryNorm(levels, 256),
            extend="max",
        )
        ax.contour(
            x_grid,
            y_grid,
            clipped,
            levels=levels[::2],
            colors="#6F6F6F",
            linewidths=0.24,
            alpha=0.55,
        )
        if panel == 0:
            path = flat_to_physical(path_at_call(ivm, 0))
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=GREY,
                linestyle=":",
                linewidth=1.25,
                marker="o",
                markersize=1.9,
                label="Common initial path",
                zorder=5,
            )
        else:
            ivm_path = flat_to_physical(path_at_call(ivm, call))
            neb_path = flat_to_physical(path_at_call(neb, call))
            ax.plot(
                ivm_path[:, 0],
                ivm_path[:, 1],
                color=IVM_COLOR,
                linewidth=1.55,
                marker="o",
                markersize=1.9,
                label="IVM",
                zorder=6,
            )
            ax.plot(
                neb_path[:, 0],
                neb_path[:, 1],
                color=NEB_COLOR,
                linestyle="--",
                linewidth=1.45,
                marker="s",
                markersize=1.7,
                label="ODE-NEB",
                zorder=5,
            )
        ax.scatter(
            saddle["point"][0],
            saddle["point"][1],
            marker="*",
            s=25,
            color="#CC79A7",
            edgecolor="#333333",
            linewidth=0.35,
            zorder=8,
        )
        ax.set_xlim(0.4, 8.2)
        ax.set_ylim(0.4, 8.2)
        ax.set_aspect("equal")
        ax.set_title(
            "Initial path"
            if call == 0
            else f"{call} evaluations/image"
        )
        ax.set_xlabel(r"$r_{\mathrm{AB}}$")
        if panel == 0:
            ax.set_ylabel(r"$r_{\mathrm{BC}}$")
        ax.text(
            -0.13,
            1.06,
            chr(ord("a") + panel),
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            va="top",
        )
    handles, labels = axes[1].get_legend_handles_labels()
    initial_handles, initial_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        initial_handles + handles,
        initial_labels + labels,
        loc="upper center",
        bbox_to_anchor=(0.48, 1.04),
        ncol=3,
        frameon=False,
    )
    colorbar_axis = fig.add_axes((0.905, 0.20, 0.012, 0.58))
    colorbar = fig.colorbar(contour, cax=colorbar_axis)
    colorbar.set_label(r"$V-V_{\mathrm{reactant}}$ (eV)")
    fig.subplots_adjust(
        left=0.065, right=0.885, bottom=0.19, top=0.82, wspace=0.12
    )
    save_figure(fig, root / "figures" / "leps_path_evolution_1x4")


def first_attainment(data: dict, key: str, threshold: float) -> float:
    values = np.asarray(data[key], dtype=float)
    mask = np.isfinite(values) & (values <= threshold)
    if not np.any(mask):
        return np.nan
    return float(np.asarray(data["calls_per_image"])[np.flatnonzero(mask)[0]])


def plot_cost_to_precision(
    root: Path, ivm: dict, neb: dict
) -> list[dict]:
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.55))
    definitions = (
        ("i_perp", r"$I_{\perp}$ threshold"),
        (
            "barrier_error",
            r"$|\Delta E-\Delta E^\ast|$ threshold (eV)",
        ),
    )
    representative = {"i_perp": 1e-4, "barrier_error": 0.03}
    for ax, (key, xlabel) in zip(axes, definitions):
        ivm_values = np.asarray(ivm[key], dtype=float)
        neb_values = np.asarray(neb[key], dtype=float)
        lower = max(
            float(np.nanmin(ivm_values)),
            float(np.nanmin(neb_values)),
            LOG_FLOORS[key],
        )
        upper = min(float(ivm_values[0]), float(neb_values[0]))
        thresholds = np.geomspace(upper, lower, 70)
        ivm_cost = np.asarray(
            [first_attainment(ivm, key, threshold) for threshold in thresholds]
        )
        neb_cost = np.asarray(
            [first_attainment(neb, key, threshold) for threshold in thresholds]
        )
        common = np.isfinite(ivm_cost) & np.isfinite(neb_cost)
        thresholds = thresholds[common]
        ivm_cost = ivm_cost[common]
        neb_cost = neb_cost[common]
        ax.plot(
            thresholds,
            ivm_cost,
            color=IVM_COLOR,
            linewidth=1.7,
            label="IVM",
        )
        ax.plot(
            thresholds,
            neb_cost,
            color=NEB_COLOR,
            linewidth=1.6,
            linestyle="--",
            label="ODE-NEB",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Minimum evaluations per image")
        ax.grid(True, which="major", color="#E0E0E0", linewidth=0.4)
        target = representative[key]
        one = first_attainment(ivm, key, target)
        two = first_attainment(neb, key, target)
        if np.isfinite(one) and np.isfinite(two):
            ax.scatter(
                [target, target],
                [one, two],
                s=18,
                color=[IVM_COLOR, NEB_COLOR],
                zorder=5,
            )
            ax.annotate(
                f"{two / one:.1f}×",
                xy=(target, np.sqrt(one * two)),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=7.0,
                va="center",
                color="#333333",
            )
        for threshold, one, two in zip(thresholds, ivm_cost, neb_cost):
            rows.append(
                {
                    "metric": key,
                    "threshold": float(threshold),
                    "ivm_first_call": float(one),
                    "neb_first_call": float(two),
                    "neb_over_ivm_speedup": float(two / one),
                }
            )
    axes[0].legend(loc="best")
    fig.tight_layout(w_pad=1.3)
    save_figure(fig, root / "figures" / "leps_cost_to_precision")
    return rows


def plot_fmax(root: Path, ivm: dict, neb: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    draw_metric(
        ax,
        ivm,
        neb,
        "fmax",
        r"Physical normal $f_{\max}$ (eV)",
        log=True,
    )
    ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, root / "figures" / "leps_fmax")


def plot_counter_audit(root: Path, ivm: dict, neb: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.5))
    image = np.arange(22)
    for ax, data, title, color in (
        (axes[0], ivm, "IVM", IVM_COLOR),
        (axes[1], neb, "ODE-NEB", NEB_COLOR),
    ):
        ax.plot(
            image,
            np.asarray(data["counter_vector"][-1]),
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.35,
            label="Joint evaluations",
        )
        ax.plot(
            image,
            np.asarray(data["cache_hit_vector"][-1]),
            color=GREY,
            marker="s",
            markersize=2.7,
            linewidth=1.1,
            linestyle="--",
            label="Cache hits",
        )
        ax.set_title(title)
        ax.set_xlabel("Image index")
        ax.set_ylabel("Count")
        ax.set_xticks([0, 5, 10, 15, 21])
        ax.grid(True, which="major", color="#E0E0E0", linewidth=0.4)
    axes[0].legend(loc="best")
    fig.tight_layout(w_pad=1.3)
    save_figure(fig, root / "figures" / "leps_counter_audit")


def trajectory_rows(method: str, data: dict) -> list[dict]:
    rows = []
    calls = np.asarray(data["calls_per_image"])
    for index, call in enumerate(calls):
        rows.append(
            {
                "method": method,
                "call_max_per_image": int(call),
                "call_mean_per_image": float(data["mean_calls"][index]),
                "total_internal_evaluations": int(
                    data["total_internal_calls"][index]
                ),
                "i_perp": float(data["i_perp"][index]),
                "fmax": float(data["fmax"][index]),
                "dense_barrier_eV": float(data["dense_barrier"][index]),
                "barrier_error_eV": float(data["barrier_error"][index]),
                "i_arc": float(data["i_arc"][index]),
                "equidistance_integral": float(
                    data["equidistance_integral"][index]
                ),
                "spring_force_max_unit": float(
                    data["spring_force_max_unit"][index]
                ),
                "segment_length_cv": float(data["segment_length_cv"][index]),
            }
        )
    return rows


def summarize_audit(data: dict, root: Path) -> dict:
    counter = np.asarray(data["counter_vector"][-1], dtype=int)
    internal = counter[1:-1]
    backend = int(scalar(data, "backend_raw_joint_evaluations"))
    return {
        "result_file": str(Path(data["_path"]).resolve().relative_to(root)),
        "counter_audit_consistent": bool(
            scalar(data, "counter_audit_consistent")
        ),
        "max_unique_evaluations_per_internal_image": int(np.max(internal)),
        "mean_unique_evaluations_per_internal_image": float(np.mean(internal)),
        "sum_internal_image_evaluations": int(np.sum(internal)),
        "endpoint_evaluations": [int(counter[0]), int(counter[-1])],
        "sum_all_image_evaluations": int(np.sum(counter)),
        "backend_raw_joint_evaluations": backend,
        "strict_backend_identity": bool(np.sum(counter) == backend),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    configure_style()

    neb_scan = discover_neb_scan(root)
    ivm_grid = discover_ivm_grid(root)
    neb_rows = [neb_scan_row(data) for data in neb_scan]
    ivm_rows = [ivm_grid_row(data) for data in ivm_grid]
    for row in neb_rows + ivm_rows:
        row["result_file"] = str(
            Path(row["result_file"]).resolve().relative_to(root)
        )

    feasible_neb = sorted(
        [row for row in neb_rows if row["spacing_feasible"]],
        key=lambda row: row["composite_log_auc"],
    )
    finalist_scan_rows = feasible_neb[:3]
    neb_finalists = []
    for row in finalist_scan_rows:
        production_path = next(
            (root / "runs" / "neb").glob(
                f"neb_production_k{row['k']:g}_*_budget5000.npz"
            )
        )
        neb_finalists.append(load_run(production_path))
    neb_finalist_rows = [neb_production_row(data) for data in neb_finalists]
    for row in neb_finalist_rows:
        row["result_file"] = str(
            Path(row["result_file"]).resolve().relative_to(root)
        )
    selected_neb_row = min(
        neb_finalist_rows, key=lambda row: row["composite_log_auc"]
    )
    eligible_ivm = [
        row
        for row in ivm_rows
        if row["precision"] == "tight"
        and row["reparameterization"] == "post_hermite"
    ]
    selected_ivm_row = min(
        eligible_ivm, key=lambda row: row["composite_log_auc"]
    )
    selected_k = selected_neb_row["k"]
    selected_alpha = selected_ivm_row["alpha"]

    neb_production_path = root / selected_neb_row["result_file"]
    ivm_production_path = next(
        (root / "runs" / "ivm").glob(
            f"ivm_production_alpha{selected_alpha:g}_tight_reparam_*budget500.npz"
        )
    )
    neb = load_run(neb_production_path)
    ivm_raw = load_run(ivm_production_path)
    ivm = committed_ivm_view(ivm_raw)
    saddle = reference_saddle()

    write_csv(root / "tables" / "neb_k_scan_summary.csv", neb_rows)
    write_csv(
        root / "tables" / "neb_production_finalists_summary.csv",
        neb_finalist_rows,
    )
    write_csv(root / "tables" / "ivm_alpha_ablation_summary.csv", ivm_rows)
    write_csv(
        root / "tables" / "main_trajectories.csv",
        trajectory_rows("IVM", ivm) + trajectory_rows("ODE-NEB", neb),
    )
    cost_rows = plot_cost_to_precision(root, ivm, neb)
    write_csv(root / "tables" / "cost_to_precision.csv", cost_rows)

    plot_main(root, ivm, neb, saddle)
    plot_neb_scan(root, neb_rows, selected_k)
    plot_alpha_scan(root, ivm_rows, selected_alpha)
    plot_ablation(root, ivm_grid, selected_alpha)
    plot_alpha_trajectories(root, ivm_grid, selected_alpha, saddle)
    plot_path_evolution_1x4(root, ivm, neb, saddle)
    plot_counter_audit(root, ivm_raw, neb)
    plot_fmax(root, ivm, neb)

    initial_hashes = {
        str(scalar(data, "initial_path_sha256"))
        for data in neb_scan + ivm_grid + [neb, ivm_raw]
    }
    tangent_definitions = {
        str(scalar(data, "tangent_definition"))
        for data in neb_scan + ivm_grid + [neb, ivm_raw]
    }
    summary = {
        "selection_rule": {
            "neb": (
                "At 500 calls/image require final segment-length CV <= 0.10 "
                "and final barrier error <= 0.10 eV. Extend the three lowest "
                "equal-weight log-AUC candidates to 5000 calls/image; select "
                "the finalist with the lowest equal-weight sum of log10 "
                "I_perp AUC and log10 barrier-error AUC over calls 1-5000."
            ),
            "ivm": (
                "Within tight, post-Hermite runs, minimize the equal-weight "
                "sum of log10 I_perp AUC and log10 barrier-error AUC over "
                "calls 1-200."
            ),
            "floors": LOG_FLOORS,
        },
        "neb_500_call_finalists": finalist_scan_rows,
        "neb_5000_call_finalists": neb_finalist_rows,
        "selected_neb": selected_neb_row,
        "selected_ivm": selected_ivm_row,
        "production_results": {
            "IVM": str(ivm_production_path.resolve().relative_to(root)),
            "ODE-NEB": str(neb_production_path.resolve().relative_to(root)),
        },
        "reference_saddle": {
            "point": saddle["point"].tolist(),
            "energy_eV": saddle["energy"],
            "reactant_energy_eV": saddle["reactant_energy"],
            "forward_barrier_eV": saddle["forward_barrier"],
            "hessian_eigenvalues": saddle["hessian_eigenvalues"].tolist(),
            "gradient_norm": saddle["gradient_norm"],
        },
        "common_initial_path": {
            "unique_hashes": sorted(initial_hashes),
            "strictly_common": len(initial_hashes) == 1,
        },
        "tangent": {
            "definitions": sorted(tangent_definitions),
            "strictly_common": len(tangent_definitions) == 1,
        },
        "counter_audit": {
            "IVM": summarize_audit(ivm_raw, root),
            "ODE-NEB": summarize_audit(neb, root),
        },
        "final_production_metrics": {
            "IVM": trajectory_rows("IVM", ivm)[-1],
            "ODE-NEB": trajectory_rows("ODE-NEB", neb)[-1],
        },
    }
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
