#!/usr/bin/env python3
"""Compare discrete IVM/NEB with continuous NN-IVM and audited gMAM."""

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

from continuous_paths import (
    FixedEndpointPathNet,
    dense_linear_metrics,
    dense_neural_metrics,
    load_parameter_vector,
    model_coordinates,
)
from leps_core import REACTANT, leps_energy, leps_energy_gradient, reference_saddle


COLORS = {
    "Discrete IVM": "#0072B2",
    "ODE–NEB": "#D55E00",
    "NN–IVM": "#009E73",
    "gMAM (autodiff)": "#CC79A7",
}
STYLES = {
    "Discrete IVM": "-",
    "ODE–NEB": "--",
    "NN–IVM": "-",
    "gMAM (autodiff)": ":",
}


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    data["_path"] = str(path)
    return data


def save_figure(fig, base):
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            base.with_suffix(f".{suffix}"),
            dpi=600 if suffix == "png" else None,
        )
    plt.close(fig)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dense_linear_trajectory(paths, subdivisions=200, chunk_size=40):
    """Vectorized dense diagnostics for a trajectory of discrete paths."""
    physical = np.asarray(paths, dtype=np.float64)[..., [0, 3]]
    count = physical.shape[0]
    result = {
        "dense_i_perp": np.empty(count),
        "dense_fmax": np.empty(count),
        "dense_barrier": np.empty(count),
        "path_length": np.empty(count),
        "spring_force_max_unit": np.empty(count),
        "segment_length_cv": np.empty(count),
    }
    fractions = np.linspace(0.0, 1.0, int(subdivisions) + 1)
    for start in range(0, count, int(chunk_size)):
        stop = min(count, start + int(chunk_size))
        band = physical[start:stop]
        displacement = np.diff(band, axis=1)
        lengths = np.linalg.norm(displacement, axis=2)
        if np.any(lengths <= 1e-14):
            raise RuntimeError("Discrete path contains a zero-length segment")
        tangent = displacement / lengths[..., None]
        points = (
            band[:, :-1, None, :]
            + fractions[None, None, :, None] * displacement[:, :, None, :]
        )
        energies, gradient = leps_energy_gradient(
            points[..., 0],
            points[..., 1],
        )
        forces = -gradient
        normal = forces - np.sum(
            forces * tangent[:, :, None, :],
            axis=-1,
            keepdims=True,
        ) * tangent[:, :, None, :]
        normal_squared = np.sum(normal**2, axis=-1)
        segment_integral = np.trapz(
            normal_squared,
            dx=1.0 / float(subdivisions),
            axis=2,
        ) * lengths
        sl = slice(start, stop)
        result["dense_i_perp"][sl] = np.sum(segment_integral, axis=1)
        result["dense_fmax"][sl] = np.max(
            np.sqrt(normal_squared),
            axis=(1, 2),
        )
        result["dense_barrier"][sl] = (
            np.max(energies, axis=(1, 2)) - energies[:, 0, 0]
        )
        result["path_length"][sl] = np.sum(lengths, axis=1)
        result["spring_force_max_unit"][sl] = np.max(
            np.abs(np.diff(lengths, axis=1)),
            axis=1,
        )
        result["segment_length_cv"][sl] = (
            np.std(lengths, axis=1) / np.mean(lengths, axis=1)
        )
    return result


def deduplicate_last(x, arrays):
    x = np.asarray(x, dtype=np.float64)
    keep = np.r_[np.diff(x) != 0.0, True]
    return x[keep], {key: np.asarray(value)[keep] for key, value in arrays.items()}


def discrete_view(label, data, saddle):
    metrics = dense_linear_trajectory(data["path"], subdivisions=200)
    metrics["dense_barrier_error"] = np.abs(
        metrics["dense_barrier"] - saddle["forward_barrier"]
    )
    x, metrics = deduplicate_last(data["calls_per_image"], metrics)
    return {
        "label": label,
        "x": x,
        "max_calls": np.asarray(data["calls_per_image"])[
            np.r_[np.diff(data["calls_per_image"]) != 0, True]
        ],
        "metrics": metrics,
        "data": data,
        "cost_definition": "20-image max calls/image",
    }


def continuous_view(label, data):
    arrays = {
        key: np.asarray(data[key], dtype=np.float64)
        for key in (
            "dense_i_perp",
            "dense_fmax",
            "dense_barrier",
            "dense_barrier_error",
            "path_length",
            "spring_force_max_unit",
            "segment_length_cv",
        )
    }
    max_calls = np.asarray(data["calls_per_image"], dtype=float)
    x, arrays = deduplicate_last(max_calls, arrays)
    keep = np.r_[np.diff(max_calls) != 0.0, True]
    return {
        "label": label,
        "x": x,
        "max_calls": np.asarray(data["calls_per_image"])[keep],
        "metrics": arrays,
        "data": data,
        "cost_definition": "20-image max calls/image",
    }


def draw_trajectory(ax, view, key, *, log=False, x_limit):
    mask = view["x"] <= float(x_limit)
    ax.plot(
        view["x"][mask],
        view["metrics"][key][mask],
        color=COLORS[view["label"]],
        linestyle=STYLES[view["label"]],
        linewidth=1.55,
        label=view["label"],
        drawstyle=(
            "steps-post"
            if view["label"] == "NN–IVM"
            else "default"
        ),
    )
    if log:
        ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1.0, 1.05 * float(x_limit))
    ax.grid(True, which="major", color="#DEDEDE", linewidth=0.4)
    ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.25)


def plot_comparison(root, views, saddle):
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.65))
    x_limit = max(float(np.max(view["x"])) for view in views)
    for view in views:
        draw_trajectory(
            axes[0],
            view,
            "dense_i_perp",
            log=True,
            x_limit=x_limit,
        )
        draw_trajectory(
            axes[1],
            view,
            "dense_barrier",
            x_limit=x_limit,
        )
        draw_trajectory(
            axes[2],
            view,
            "dense_barrier_error",
            log=True,
            x_limit=x_limit,
        )
    axes[0].set_ylabel(r"Dense $I_{\perp}$")
    axes[1].set_ylabel(r"Forward barrier $\Delta E$ (eV)")
    axes[1].axhline(
        saddle["forward_barrier"],
        color="#555555",
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        label=r"Index-1 saddle $\Delta E^\ast$",
    )
    axes[2].set_ylabel(r"$|\Delta E-\Delta E^\ast|$ (eV)")
    for ax in axes:
        ax.set_xlabel("Max joint evaluations per internal image")
    for index, ax in enumerate(axes):
        ax.text(
            -0.13,
            1.05,
            chr(ord("a") + index),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=8.5,
            va="top",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    ref_handles, ref_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles + ref_handles[-1:],
        labels + ref_labels[-1:],
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.0)
    save_figure(
        fig,
        root / "figures" / "leps_continuous_methods_comparison",
    )


def plot_per_stream_comparison(root, views):
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.62))
    for view in views:
        for ax, key in zip(
            axes,
            ("dense_i_perp", "dense_barrier_error"),
        ):
            mask = view["max_calls"] <= 500
            ax.plot(
                view["max_calls"][mask],
                view["metrics"][key][mask],
                color=COLORS[view["label"]],
                linestyle=STYLES[view["label"]],
                linewidth=1.55,
                label=view["label"],
                drawstyle=(
                    "steps-post"
                    if view["label"] == "NN–IVM"
                    else "default"
                ),
            )
            ax.set_yscale("log")
            ax.set_xlim(0, 500)
            ax.grid(True, which="major", color="#DEDEDE", linewidth=0.4)
            ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.25)
    axes[0].set_ylabel(r"Dense $I_{\perp}$")
    axes[1].set_ylabel(r"$|\Delta E-\Delta E^\ast|$ (eV)")
    for index, ax in enumerate(axes):
        ax.set_xlabel("Max evaluations per internal stream")
        ax.text(
            -0.13,
            1.05,
            chr(ord("a") + index),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=8.5,
            va="top",
        )
    axes[0].legend(loc="best")
    fig.tight_layout(w_pad=1.25)
    save_figure(
        fig,
        root / "figures" / "leps_continuous_per_stream_comparison",
    )


def plot_counter_audit(root, views):
    fig, axes = plt.subplots(1, 4, figsize=(7.08, 2.15), sharey=False)
    for index, (ax, view) in enumerate(zip(axes, views)):
        data = view["data"]
        joint = np.asarray(data["counter_vector"][-1], dtype=float)
        hits = np.asarray(data["cache_hit_vector"][-1], dtype=float)
        image_coordinate = np.linspace(0.0, 1.0, joint.size)
        ax.plot(
            image_coordinate,
            joint,
            color=COLORS[view["label"]],
            linewidth=1.25,
            marker="o",
            markersize=2.2,
            label="Joint evaluations",
        )
        ax.plot(
            image_coordinate,
            hits,
            color="#777777",
            linewidth=1.0,
            linestyle="--",
            label="Cache hits",
        )
        ax.set_title(view["label"])
        ax.set_xlabel("Path fraction / evaluator")
        if index == 0:
            ax.set_ylabel("Run-local count")
        ax.grid(True, which="major", color="#E2E2E2", linewidth=0.4)
        ax.text(
            -0.16,
            1.07,
            chr(ord("a") + index),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=8.5,
            va="top",
        )
    axes[0].legend(loc="best")
    fig.tight_layout(w_pad=0.9)
    save_figure(
        fig,
        root / "figures" / "leps_continuous_counter_audit",
    )


def first_attainment(view, key, threshold):
    values = view["metrics"][key]
    mask = np.isfinite(values) & (values <= float(threshold))
    if not np.any(mask):
        return np.nan
    return float(view["x"][np.flatnonzero(mask)[0]])


def plot_cost_to_precision(root, views):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.62))
    definitions = (
        ("dense_i_perp", r"Dense $I_{\perp}$ threshold", 1e-3),
        (
            "dense_barrier_error",
            r"$|\Delta E-\Delta E^\ast|$ threshold (eV)",
            1e-6,
        ),
    )
    for ax, (key, xlabel, floor) in zip(axes, definitions):
        upper = max(view["metrics"][key][0] for view in views)
        thresholds = np.geomspace(upper, floor, 140)
        for view in views:
            costs = np.asarray(
                [
                    first_attainment(view, key, threshold)
                    for threshold in thresholds
                ]
            )
            finite = np.isfinite(costs)
            ax.plot(
                thresholds[finite],
                costs[finite],
                color=COLORS[view["label"]],
                linestyle=STYLES[view["label"]],
                linewidth=1.55,
                label=view["label"],
            )
            for threshold, cost in zip(thresholds, costs):
                rows.append(
                    {
                        "metric": key,
                        "threshold": float(threshold),
                        "method": view["label"],
                        "first_max_calls_per_internal_image": (
                            float(cost) if np.isfinite(cost) else ""
                        ),
                    }
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Minimum max evaluations / internal image")
        ax.grid(True, which="major", color="#DEDEDE", linewidth=0.4)
        ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.25)
    axes[0].legend(loc="best")
    fig.tight_layout(w_pad=1.25)
    save_figure(
        fig,
        root / "figures" / "leps_continuous_cost_to_precision",
    )
    return rows


def network_model_from_data(data):
    return FixedEndpointPathNet(
        width=int(np.asarray(data["width"]).item()),
        depth=int(np.asarray(data["depth"]).item()),
        fourier_order=int(np.asarray(data["fourier_order"]).item()),
        seed=int(np.asarray(data["seed"]).item()),
    )


def path_from_view(view, cost, sample_count=101):
    index = int(np.searchsorted(view["x"], cost, side="right") - 1)
    index = int(np.clip(index, 0, len(view["x"]) - 1))
    data = view["data"]
    if view["label"] in ("Discrete IVM", "ODE–NEB"):
        raw_calls = np.asarray(data["calls_per_image"], dtype=float)
        raw_index = int(np.searchsorted(raw_calls, view["max_calls"][index], side="right") - 1)
        return np.asarray(data["path"][raw_index])[..., [0, 3]]
    max_calls = np.asarray(data["calls_per_image"], dtype=float)
    raw_index = int(
        np.searchsorted(max_calls, view["x"][index], side="right") - 1
    )
    model = network_model_from_data(data)
    load_parameter_vector(model, data["model_state"][raw_index])
    return model_coordinates(model, np.linspace(0.0, 1.0, sample_count))


def plot_paths(root, views, saddle):
    snapshots = (0, 30, 150, 300)
    coordinate = np.linspace(0.4, 8.2, 440)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate)
    relative = leps_energy(x_grid, y_grid) - float(
        leps_energy(REACTANT[0], REACTANT[1])
    )
    levels = np.linspace(0.0, 6.5, 41)
    clipped = np.clip(relative, levels[0], levels[-1])
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.08, 2.12),
        sharex=True,
        sharey=True,
    )
    contour = None
    for panel, (ax, cost) in enumerate(zip(axes, snapshots)):
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
            path = path_from_view(views[0], 0)
            ax.plot(
                path[:, 0],
                path[:, 1],
                color="#666666",
                linestyle=":",
                linewidth=1.25,
                label="Common initial path",
            )
        else:
            for view in views:
                path = path_from_view(view, cost)
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    color=COLORS[view["label"]],
                    linestyle=STYLES[view["label"]],
                    linewidth=1.25,
                    label=view["label"],
                )
        ax.scatter(
            saddle["point"][0],
            saddle["point"][1],
            marker="*",
            s=24,
            color="#F0E442",
            edgecolor="#333333",
            linewidth=0.35,
            zorder=8,
        )
        ax.set_xlim(0.4, 8.2)
        ax.set_ylim(0.4, 8.2)
        ax.set_aspect("equal")
        ax.set_title(
            "Initial path" if panel == 0 else f"{cost} calls / image"
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
    handles0, labels0 = axes[0].get_legend_handles_labels()
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles0 + handles,
        labels0 + labels,
        loc="upper center",
        bbox_to_anchor=(0.47, 1.05),
        ncol=5,
    )
    colorbar_axis = fig.add_axes((0.905, 0.20, 0.012, 0.58))
    colorbar = fig.colorbar(contour, cax=colorbar_axis)
    colorbar.set_label(r"$V-V_{\mathrm{reactant}}$ (eV)")
    fig.subplots_adjust(
        left=0.065,
        right=0.885,
        bottom=0.19,
        top=0.80,
        wspace=0.12,
    )
    save_figure(
        fig,
        root / "figures" / "leps_continuous_path_evolution_1x4",
    )


def convergence_rows(views):
    rows = []
    for view in views:
        data = view["data"]
        if view["label"] in ("Discrete IVM", "ODE–NEB"):
            path = np.asarray(data["path"][-1])
            for subdivisions in (50, 100, 200, 400):
                metric = dense_linear_metrics(
                    path,
                    subdivisions=subdivisions,
                )
                rows.append(
                    {
                        "method": view["label"],
                        "diagnostic_resolution": (
                            f"{subdivisions} subdivisions/segment"
                        ),
                        "dense_i_perp": metric["dense_i_perp"],
                        "dense_barrier": metric["dense_barrier"],
                    }
                )
        else:
            model = network_model_from_data(data)
            load_parameter_vector(model, data["model_state"][-1])
            for points in (2049, 4097, 8193, 16385):
                metric = dense_neural_metrics(model, dense_count=points)
                rows.append(
                    {
                        "method": view["label"],
                        "diagnostic_resolution": f"{points} points",
                        "dense_i_perp": metric["dense_i_perp"],
                        "dense_barrier": metric["dense_barrier"],
                    }
                )
    return rows


def summary_payload(views, convergence, saddle):
    methods = {}
    for view in views:
        data = view["data"]
        final = -1
        minimum_i = int(np.nanargmin(view["metrics"]["dense_i_perp"]))
        minimum_barrier = int(
            np.nanargmin(view["metrics"]["dense_barrier_error"])
        )
        methods[view["label"]] = {
            "result_file": data["_path"],
            "cost_definition": view["cost_definition"],
            "final_max_calls_per_internal_image": float(view["x"][final]),
            "final_max_calls_per_internal_stream": int(
                view["max_calls"][final]
            ),
            "final_dense_i_perp": float(
                view["metrics"]["dense_i_perp"][final]
            ),
            "minimum_dense_i_perp": float(
                view["metrics"]["dense_i_perp"][minimum_i]
            ),
            "cost_at_minimum_dense_i_perp": float(view["x"][minimum_i]),
            "final_dense_barrier": float(
                view["metrics"]["dense_barrier"][final]
            ),
            "minimum_barrier_error": float(
                view["metrics"]["dense_barrier_error"][minimum_barrier]
            ),
            "cost_at_minimum_barrier_error": float(
                view["x"][minimum_barrier]
            ),
            "final_spring_force_max_unit": float(
                view["metrics"]["spring_force_max_unit"][final]
            ),
            "backend_raw_joint_evaluations": int(
                np.asarray(data["backend_raw_joint_evaluations"]).item()
            ),
            "counter_audit_consistent": bool(
                np.asarray(data["counter_audit_consistent"]).item()
            ),
            "initial_path_sha256": str(
                np.asarray(data["initial_path_sha256"]).item()
            ),
        }
    hashes = {item["initial_path_sha256"] for item in methods.values()}
    return {
        "reference_forward_barrier": saddle["forward_barrier"],
        "common_initial_path_hash_count": len(hashes),
        "methods": methods,
        "dense_convergence": convergence,
        "interpretation": {
            "nn_ivm": (
                "The continuous representation removes the piecewise-linear "
                "tangent floor when enough audited arclength collocation "
                "points are used."
            ),
            "gmam": (
                "From the common straight path, gMAM converges toward a "
                "high-barrier stationary branch rather than the lower "
                "index-1-saddle valley; this is reported, not hidden."
            ),
        },
    }


def main(args):
    configure_style()
    root = Path(args.root).expanduser().resolve()
    saddle = reference_saddle()
    ivm = load_npz(
        root
        / "runs/ivm/"
        "ivm_production_alpha30_tight_reparam_cfg29674d42197b_budget500.npz"
    )
    neb = load_npz(
        root
        / "runs/neb/"
        "neb_production_k10_cfgfe491f5a6c6f_budget5000.npz"
    )
    nn_ivm = load_npz(
        root
        / "runs/continuous_v3/"
        "nn_ivm_adaptive_al_a100_j015_b500.npz"
    )
    gmam = load_npz(
        root
        / "runs/continuous_v2/"
        "gmam_autodiff20_production_lr001_b500.npz"
    )
    views = [
        discrete_view("Discrete IVM", ivm, saddle),
        discrete_view("ODE–NEB", neb, saddle),
        continuous_view("NN–IVM", nn_ivm),
        continuous_view("gMAM (autodiff)", gmam),
    ]

    plot_comparison(root, views, saddle)
    plot_per_stream_comparison(root, views)
    cost_rows = plot_cost_to_precision(root, views)
    plot_paths(root, views, saddle)
    plot_counter_audit(root, views)

    trajectory_rows = []
    for view in views:
        for index, cost in enumerate(view["x"]):
            trajectory_rows.append(
                {
                    "method": view["label"],
                    "max_calls_per_internal_image": float(cost),
                    "max_calls_per_internal_stream": int(
                        view["max_calls"][index]
                    ),
                    **{
                        key: float(value[index])
                        for key, value in view["metrics"].items()
                    },
                }
            )
    write_csv(
        root / "tables" / "continuous_method_trajectories.csv",
        trajectory_rows,
    )
    write_csv(
        root / "tables" / "continuous_cost_to_precision.csv",
        cost_rows,
    )
    convergence = convergence_rows(views)
    write_csv(
        root / "tables" / "continuous_dense_convergence.csv",
        convergence,
    )
    payload = summary_payload(views, convergence, saddle)
    report = root / "reports" / "continuous_methods_summary.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
