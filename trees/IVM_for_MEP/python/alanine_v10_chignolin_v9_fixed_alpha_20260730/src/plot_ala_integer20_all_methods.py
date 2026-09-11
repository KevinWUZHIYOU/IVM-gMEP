#!/usr/bin/env python3
"""Build the audited Ala fixed-alpha=20 comparison figures.

The script compares independently tuned Original String, static NEB,
ODE--NEB, OBSM--BFGS, and OBSM--FIRE against the V150 -> V155 IVM lineage.
Every horizontal coordinate is the sum of the per-image force counters.  The
separate dense-barrier evaluations are diagnostic only and are never charged
to that optimization cost.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

from plot_v49_formal_ala_1x3 import (
    EV_TO_KCAL,
    _ContourCacheUnpickler,
    configure_original_style,
    cost_to_reach,
    path_to_dihedrals,
)
from plot_v67_continuation_figures import clean_axis, panel_label, save_figure


REFERENCE_BARRIER_KCAL = 2.4560825131253083
SELECTED_V155_CALL = 142
SELECTED_BFGS_CALL = 91


@dataclass
class MethodSeries:
    key: str
    label: str
    short_label: str
    color: str
    marker: str
    linestyle: str
    force: np.ndarray
    iperp: np.ndarray
    barrier_force: np.ndarray
    barrier: np.ndarray
    selected_call: int
    selected_force: int
    selected_iperp: float
    selected_barrier: float
    selected_path: np.ndarray
    initial_path: np.ndarray
    path_force: np.ndarray
    paths: np.ndarray
    configuration: str
    result_path: Path
    dense_path: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(saved, key):
    return np.asarray(saved[key]).reshape(()).item()


def history_index(saved, call: int) -> int:
    indices = np.flatnonzero(
        np.asarray(saved["calls_per_image"], dtype=int) == int(call)
    )
    if not indices.size:
        raise RuntimeError(f"No history record at call {call}")
    return int(indices[-1])


def stored_path(saved, call: int) -> np.ndarray:
    indices = np.flatnonzero(
        np.asarray(saved["stored_path_calls"], dtype=int) == int(call)
    )
    if not indices.size:
        raise RuntimeError(f"No stored path at call {call}")
    return np.asarray(saved["stored_paths"][indices[-1]], dtype=float)


def force_at_call(saved, call: int) -> int:
    return int(
        np.asarray(saved["counter_vectors"], dtype=np.int64)[
            history_index(saved, call)
        ].sum()
    )


def dense_at_call(dense, call: int) -> float:
    indices = np.flatnonzero(np.asarray(dense["calls"], dtype=int) == int(call))
    if not indices.size:
        raise RuntimeError(f"No dense barrier at call {call}")
    return float(np.asarray(dense["adaptive_barrier_kcal"])[indices[-1]])


def generic_path_history(saved, selected_call: int | None = None):
    calls = np.asarray(saved["stored_path_calls"], dtype=int)
    paths = np.asarray(saved["stored_paths"], dtype=float)
    keep = np.ones(len(calls), dtype=bool)
    if selected_call is not None:
        keep &= calls <= int(selected_call)
    calls = calls[keep]
    paths = paths[keep]
    forces = np.asarray([force_at_call(saved, int(call)) for call in calls])
    return forces, paths


def load_generic(
    *,
    key: str,
    label: str,
    short_label: str,
    color: str,
    marker: str,
    linestyle: str,
    result_path: Path,
    dense_path: Path,
    selected_call: int,
    configuration: str,
    truncate_paths_at_selection: bool = False,
) -> MethodSeries:
    saved = np.load(result_path, allow_pickle=False)
    dense = np.load(dense_path, allow_pickle=False)
    calls = np.asarray(saved["calls_per_image"], dtype=int)
    force = np.asarray(saved["counter_vectors"], dtype=np.int64).sum(axis=1)
    iperp = np.asarray(saved["i_perp"], dtype=float)
    dense_calls = np.asarray(dense["calls"], dtype=int)
    barrier_force = np.asarray(
        [force_at_call(saved, int(call)) for call in dense_calls], dtype=int
    )
    barrier = np.asarray(dense["adaptive_barrier_kcal"], dtype=float)
    selected_index = history_index(saved, selected_call)
    path_force, paths = generic_path_history(
        saved, selected_call if truncate_paths_at_selection else None
    )
    return MethodSeries(
        key=key,
        label=label,
        short_label=short_label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        force=force,
        iperp=iperp,
        barrier_force=barrier_force,
        barrier=barrier,
        selected_call=int(selected_call),
        selected_force=int(force[selected_index]),
        selected_iperp=float(iperp[selected_index]),
        selected_barrier=dense_at_call(dense, selected_call),
        selected_path=stored_path(saved, selected_call),
        initial_path=np.asarray(saved["initial_path"], dtype=float),
        path_force=path_force,
        paths=paths,
        configuration=configuration,
        result_path=result_path,
        dense_path=dense_path,
    )


def load_ode(
    result_path: Path,
    metrics_path: Path,
    dense_path: Path,
    common_initial_path: np.ndarray,
) -> MethodSeries:
    saved = np.load(result_path, allow_pickle=False)
    metrics = np.load(metrics_path, allow_pickle=False)
    dense = np.load(dense_path, allow_pickle=False)
    all_counts = np.asarray(metrics["count_seq"], dtype=int)
    valid = all_counts >= 1
    counts = all_counts[valid]
    force = 20 * counts + 2
    dense_calls = np.asarray(dense["calls"], dtype=int)
    stored_counts = np.asarray(saved["count_seq"], dtype=int)
    paths = np.asarray(saved["images_seq"], dtype=float)
    return MethodSeries(
        key="ode_neb",
        label="ODE--NEB ($k=50$)",
        short_label="ODE--NEB",
        color="#D55E00",
        marker="o",
        linestyle=(0, (5, 2)),
        force=force,
        iperp=np.asarray(metrics["loss_perp"], dtype=float)[valid],
        barrier_force=20 * dense_calls + 2,
        barrier=np.asarray(dense["adaptive_barrier_kcal"], dtype=float),
        selected_call=int(counts[-1]),
        selected_force=int(force[-1]),
        selected_iperp=float(np.asarray(metrics["loss_perp"], dtype=float)[valid][-1]),
        selected_barrier=float(dense["adaptive_barrier_kcal"][-1]),
        selected_path=paths[-1],
        initial_path=np.asarray(common_initial_path, dtype=float),
        path_force=np.concatenate(
            [np.asarray([0], dtype=int), 20 * stored_counts + 2]
        ),
        paths=np.concatenate(
            [np.asarray(common_initial_path, dtype=float)[None, ...], paths], axis=0
        ),
        configuration="adaptive ODE--NEB, k=50, alpha=1e-4",
        result_path=result_path,
        dense_path=dense_path,
    )


def load_ivm(
    v150_path: Path,
    v155_path: Path,
    dense_v150_path: Path,
    dense_v155_path: Path,
) -> MethodSeries:
    v150 = np.load(v150_path, allow_pickle=False)
    v155 = np.load(v155_path, allow_pickle=False)
    dense_v150 = np.load(dense_v150_path, allow_pickle=False)
    dense_v155 = np.load(dense_v155_path, allow_pickle=False)

    for name, saved in (("V150", v150), ("V155", v155)):
        if scalar(saved, "alpha_mode") != "fixed":
            raise RuntimeError(f"{name} is not fixed-alpha")
        if not np.isclose(float(scalar(saved, "alpha")), 20.0):
            raise RuntimeError(f"{name} does not use alpha=20")
        if not np.allclose(np.asarray(saved["active_alpha"], dtype=float), 20.0):
            raise RuntimeError(f"{name} active alpha is not constant at 20")

    selected_index = history_index(v155, SELECTED_V155_CALL)
    if not bool(np.asarray(v155["reparam_committed"], dtype=bool)[selected_index]):
        raise RuntimeError("Selected V155 state is not a completed macrostate")

    v150_force = np.asarray(v150["counter_vectors"], dtype=np.int64).sum(axis=1)
    offset = int(v150_force[-1])
    committed = np.flatnonzero(np.asarray(v155["reparam_committed"], dtype=bool))
    committed = committed[
        np.asarray(v155["calls_per_image"], dtype=int)[committed]
        <= SELECTED_V155_CALL
    ]
    start = np.asarray([0], dtype=int)
    continuation_indices = np.unique(np.concatenate([start, committed]))
    v155_force = (
        np.asarray(v155["counter_vectors"], dtype=np.int64)[continuation_indices].sum(axis=1)
        + offset
    )
    force = np.concatenate([v150_force, v155_force])
    iperp = np.concatenate(
        [
            np.asarray(v150["i_perp"], dtype=float),
            np.asarray(v155["i_perp"], dtype=float)[continuation_indices],
        ]
    )

    barrier_force = []
    barrier = []
    for call, value in zip(
        np.asarray(dense_v150["calls"], dtype=int),
        np.asarray(dense_v150["adaptive_barrier_kcal"], dtype=float),
    ):
        barrier_force.append(force_at_call(v150, int(call)))
        barrier.append(float(value))
    allowed = {1}.union(
        int(np.asarray(v155["calls_per_image"], dtype=int)[index])
        for index in committed
    )
    for call, value in zip(
        np.asarray(dense_v155["calls"], dtype=int),
        np.asarray(dense_v155["adaptive_barrier_kcal"], dtype=float),
    ):
        if int(call) > SELECTED_V155_CALL or int(call) not in allowed:
            continue
        barrier_force.append(offset + force_at_call(v155, int(call)))
        barrier.append(float(value))

    path_force = []
    paths = []
    for call, path in zip(
        np.asarray(v150["stored_path_calls"], dtype=int),
        np.asarray(v150["stored_paths"], dtype=float),
    ):
        path_force.append(force_at_call(v150, int(call)))
        paths.append(path)
    for index in continuation_indices:
        call = int(np.asarray(v155["calls_per_image"], dtype=int)[index])
        path_force.append(offset + force_at_call(v155, call))
        paths.append(stored_path(v155, call))

    selected_force = offset + force_at_call(v155, SELECTED_V155_CALL)
    selected_barrier = dense_at_call(dense_v155, SELECTED_V155_CALL)
    return MethodSeries(
        key="ivm",
        label="IVM ($\\alpha=20$)",
        short_label="IVM",
        color="#0072B2",
        marker="^",
        linestyle="-",
        force=np.asarray(force, dtype=int),
        iperp=np.asarray(iperp, dtype=float),
        barrier_force=np.asarray(barrier_force, dtype=int),
        barrier=np.asarray(barrier, dtype=float),
        selected_call=SELECTED_V155_CALL,
        selected_force=int(selected_force),
        selected_iperp=float(np.asarray(v155["i_perp"])[selected_index]),
        selected_barrier=float(selected_barrier),
        selected_path=stored_path(v155, SELECTED_V155_CALL),
        initial_path=np.asarray(v150["initial_path"], dtype=float),
        path_force=np.asarray(path_force, dtype=int),
        paths=np.asarray(paths, dtype=float),
        configuration=(
            "V150 -> V155 complete operator-split macrostates; fixed integer "
            "alpha=20; selected V155 call 142"
        ),
        result_path=v155_path,
        dense_path=dense_v155_path,
    )


def path_at_or_before(method: MethodSeries, target: int | None) -> np.ndarray:
    if target is None:
        return method.selected_path
    eligible = np.flatnonzero(method.path_force <= int(target))
    if not eligible.size:
        return method.initial_path
    return np.asarray(method.paths[eligible[-1]], dtype=float)


def method_handles(methods: list[MethodSeries]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=method.color,
            lw=2.0 if method.key == "ivm" else 1.35,
            ls=method.linestyle,
            marker=method.marker,
            ms=4.8,
            mec="black",
            mew=0.35,
            label=method.label,
        )
        for method in methods
    ]


def plot_descent(methods: list[MethodSeries], output_dir: Path):
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.65), dpi=200)
    for method in methods:
        width = 2.25 if method.key == "ivm" else 1.1
        zorder = 8 if method.key == "ivm" else 3
        axes[0].plot(
            method.force,
            method.iperp,
            color=method.color,
            ls=method.linestyle,
            lw=width,
            alpha=1.0 if method.key == "ivm" else 0.86,
            label=method.label,
            zorder=zorder,
        )
        axes[0].scatter(
            method.selected_force,
            method.selected_iperp,
            color=method.color,
            marker=method.marker,
            s=30,
            edgecolor="black",
            linewidth=0.4,
            zorder=10,
        )
        axes[1].plot(
            method.barrier_force,
            method.barrier,
            color=method.color,
            ls=method.linestyle,
            lw=width,
            marker=method.marker,
            ms=2.8,
            markevery=max(1, len(method.barrier) // 14),
            alpha=1.0 if method.key == "ivm" else 0.86,
            zorder=zorder,
        )
        axes[1].scatter(
            method.selected_force,
            method.selected_barrier,
            color=method.color,
            marker=method.marker,
            s=30,
            edgecolor="black",
            linewidth=0.4,
            zorder=10,
        )
    axes[1].axhline(
        REFERENCE_BARRIER_KCAL,
        color="#222222",
        lw=0.8,
        ls=":",
        label=r"reference $\Delta E^*$",
    )
    axes[0].set_ylabel(r"$I_\perp$")
    axes[1].set_ylabel(r"Densified $\Delta E$ (kcal mol$^{-1}$)")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    for index, axis in enumerate(axes):
        axis.set_xscale("log")
        axis.set_xlim(20, 1.12e5)
        axis.set_xlabel(r"Whole-path force evaluations, $N_F$")
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    axes[1].set_ylim(2.3, 50)
    axes[0].legend(
        handles=method_handles(methods),
        loc="upper right",
        fontsize=7.0,
        ncol=2,
        columnspacing=1.0,
        handlelength=2.5,
    )
    selected_exponent = int(np.floor(np.log10(methods[0].selected_iperp)))
    selected_mantissa = methods[0].selected_iperp / (10.0 ** selected_exponent)
    axes[0].annotate(
        rf"selected $I_\perp={selected_mantissa:.2f}\times10^{{{selected_exponent}}}$",
        xy=(methods[0].selected_force, methods[0].selected_iperp),
        xytext=(18, 18),
        textcoords="offset points",
        color=methods[0].color,
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color=methods[0].color, lw=0.8),
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.95, wspace=0.26)
    return save_figure(figure, output_dir / "alanine_integer20_all_methods_two_metric_descent")


def plot_cost_and_paths(
    methods: list[MethodSeries], rama, output_dir: Path
):
    iperp_thresholds = np.geomspace(1e-4, 1e2, 34)
    barrier_thresholds = np.geomspace(5e-3, 4e1, 34)
    costs = {}
    rows = []
    for method in methods:
        iperp_cost = cost_to_reach(
            method.force, method.iperp, iperp_thresholds, min_count=1
        )
        barrier_error = np.abs(method.barrier - REFERENCE_BARRIER_KCAL)
        barrier_cost = cost_to_reach(
            method.barrier_force,
            barrier_error,
            barrier_thresholds,
            min_count=1,
        )
        costs[method.key] = (iperp_cost, barrier_cost)
        for metric, thresholds, values in (
            ("i_perp", iperp_thresholds, iperp_cost),
            ("dense_barrier_abs_error_kcal_per_mol", barrier_thresholds, barrier_cost),
        ):
            for threshold, value in zip(thresholds, values):
                rows.append(
                    [
                        method.key,
                        metric,
                        float(threshold),
                        "" if not np.isfinite(value) else int(value),
                    ]
                )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.7, 4.25),
        dpi=200,
        gridspec_kw={"width_ratios": [1, 1, 1.17]},
        constrained_layout=True,
    )
    for method in methods:
        for axis, thresholds, values in (
            (axes[0], iperp_thresholds, costs[method.key][0]),
            (axes[1], barrier_thresholds, costs[method.key][1]),
        ):
            axis.plot(
                thresholds,
                values,
                color=method.color,
                marker=method.marker,
                ls=method.linestyle,
                lw=2.0 if method.key == "ivm" else 1.15,
                ms=4.0,
                mec="black",
                mew=0.3,
                markevery=3,
                alpha=1.0 if method.key == "ivm" else 0.86,
                zorder=8 if method.key == "ivm" else 3,
            )
    for index, axis in enumerate(axes[:2]):
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_yscale("log")
        axis.set_ylabel(r"Whole-path force evaluations, $N_F$")
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{value / 1000:g}k")
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(True, which="major", ls="--", lw=0.4, alpha=0.4)
        panel_label(axis, chr(ord("a") + index), formal=True)
    axes[0].set_xlabel(r"$I_\perp$ threshold")
    axes[1].set_xlabel(r"$|\Delta E-\Delta E^*|$ (kcal mol$^{-1}$)")

    annotations = {}
    method_by_key = {method.key: method for method in methods}
    for axis, metric, threshold, values_index in (
        (axes[0], "i_perp", 2.1e-4, 0),
        (axes[1], "dense_barrier_error", 1.7e-2, 1),
    ):
        ivm = method_by_key["ivm"]
        ode = method_by_key["ode_neb"]
        if values_index == 0:
            c_ivm = cost_to_reach(ivm.force, ivm.iperp, np.asarray([threshold]))[0]
            c_ode = cost_to_reach(ode.force, ode.iperp, np.asarray([threshold]))[0]
        else:
            c_ivm = cost_to_reach(
                ivm.barrier_force,
                np.abs(ivm.barrier - REFERENCE_BARRIER_KCAL),
                np.asarray([threshold]),
            )[0]
            c_ode = cost_to_reach(
                ode.barrier_force,
                np.abs(ode.barrier - REFERENCE_BARRIER_KCAL),
                np.asarray([threshold]),
            )[0]
        if np.isfinite(c_ivm) and np.isfinite(c_ode):
            speedup = float(c_ode / c_ivm)
            annotations[metric] = {
                "threshold": threshold,
                "ivm_force": int(c_ivm),
                "ode_neb_force": int(c_ode),
                "speedup": speedup,
            }
            axis.annotate(
                "",
                xy=(threshold, c_ivm),
                xytext=(threshold, c_ode),
                arrowprops=dict(arrowstyle="<->", color="#222222", lw=0.8),
            )
            axis.text(
                threshold,
                np.sqrt(c_ivm * c_ode),
                f"  {speedup:.1f}×",
                fontsize=8.0,
                va="center",
                fontweight="bold",
            )

    phi_grid, psi_grid = np.meshgrid(rama["phi_g"], rama["psi_g"], indexing="ij")
    energy_grid = (rama["PES"] - np.nanmin(rama["PES"])) * EV_TO_KCAL
    axis = axes[2]
    contour = axis.contourf(
        phi_grid,
        psi_grid,
        energy_grid,
        levels=np.linspace(0, 15, 16),
        cmap="plasma",
        extend="max",
    )
    axis.contour(
        phi_grid,
        psi_grid,
        energy_grid,
        levels=np.linspace(0, 15, 8),
        colors="white",
        linewidths=0.3,
        alpha=0.5,
    )
    colorbar = figure.colorbar(contour, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(r"$\Delta E$ (kcal mol$^{-1}$)", fontsize=9)
    phi, psi = path_to_dihedrals(methods[0].initial_path)
    axis.plot(
        phi,
        psi,
        "--",
        color="#777777",
        lw=1.5,
        marker="s",
        ms=3.7,
        mec="black",
        mew=0.3,
        label="Initial (IDPP)",
        zorder=2,
    )
    for method in reversed(methods):
        phi, psi = path_to_dihedrals(method.selected_path)
        axis.plot(
            phi,
            psi,
            color=method.color,
            ls=method.linestyle,
            lw=2.2 if method.key == "ivm" else 1.15,
            marker=method.marker,
            ms=4.2,
            mec="black",
            mew=0.3,
            label=method.label,
            alpha=1.0 if method.key == "ivm" else 0.84,
            zorder=8 if method.key == "ivm" else 4,
        )
    for minimum in rama["unique"][:2]:
        axis.scatter(
            minimum["phi"],
            minimum["psi"],
            s=180,
            marker="*",
            c="gold",
            edgecolor="black",
            zorder=10,
            linewidth=0.7,
        )
    axis.set_xlim(-180, 180)
    axis.set_ylim(-180, 180)
    axis.set_xticks(range(-180, 181, 90))
    axis.set_yticks(range(-180, 181, 90))
    axis.set_xlabel(r"$\phi$ (deg)")
    axis.set_ylabel(r"$\psi$ (deg)")
    axis.set_aspect("equal")
    panel_label(axis, "c", formal=True)
    figure.legend(
        handles=method_handles(methods),
        loc="upper center",
        bbox_to_anchor=(0.49, 1.035),
        ncol=6,
        fontsize=7.1,
        columnspacing=0.9,
        handlelength=2.2,
        frameon=False,
    )
    outputs = save_figure(
        figure, output_dir / "alanine_integer20_all_methods_cost_to_precision_and_paths"
    )
    return outputs, rows, annotations


def plot_path_evolution(
    methods: list[MethodSeries], rama, output_dir: Path
):
    phi_grid, psi_grid = np.meshgrid(rama["phi_g"], rama["psi_g"], indexing="ij")
    energy_grid = (rama["PES"] - np.nanmin(rama["PES"])) * EV_TO_KCAL
    panel_config = [
        (0, "Common IDPP initialization"),
        (1000, r"$N_F\leq1$k"),
        (10000, r"$N_F\leq10$k"),
        (None, "Selected checkpoints"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(16.0, 4.15), dpi=200)
    figure.subplots_adjust(left=0.04, right=0.955, bottom=0.15, top=0.80, wspace=0.18)
    contour = None
    for panel_index, (axis, (target, title)) in enumerate(zip(axes, panel_config)):
        contour = axis.contourf(
            phi_grid,
            psi_grid,
            energy_grid,
            levels=np.linspace(0, 15, 16),
            cmap="pink",
            extend="max",
        )
        axis.contour(
            phi_grid,
            psi_grid,
            energy_grid,
            levels=np.linspace(0, 15, 8),
            colors="white",
            linewidths=0.3,
            alpha=0.5,
        )
        if target == 0:
            phi, psi = path_to_dihedrals(methods[0].initial_path)
            axis.plot(
                phi,
                psi,
                "--",
                color="#777777",
                lw=1.7,
                marker="s",
                ms=4.2,
                mec="black",
                mew=0.35,
                zorder=4,
            )
        else:
            for method in reversed(methods):
                path = path_at_or_before(method, target)
                phi, psi = path_to_dihedrals(path)
                axis.plot(
                    phi,
                    psi,
                    color=method.color,
                    ls=method.linestyle,
                    lw=2.1 if method.key == "ivm" else 1.1,
                    marker=method.marker,
                    ms=3.7,
                    mec="black",
                    mew=0.25,
                    alpha=1.0 if method.key == "ivm" else 0.80,
                    zorder=8 if method.key == "ivm" else 4,
                )
        for minimum in rama["unique"][:2]:
            axis.scatter(
                minimum["phi"],
                minimum["psi"],
                s=180,
                marker="*",
                c="gold",
                edgecolor="black",
                zorder=10,
                linewidth=0.7,
            )
        axis.set_xlim(-180, 0)
        axis.set_ylim(0, 180)
        axis.set_xticks(range(-180, 1, 60))
        axis.set_yticks(range(0, 181, 60))
        axis.set_xlabel(r"$\phi$ (deg)")
        axis.set_ylabel(r"$\psi$ (deg)")
        axis.set_aspect("equal")
        axis.set_title(title, fontsize=9.5)
        panel_label(axis, chr(ord("a") + panel_index))
    figure.legend(
        handles=[
            Line2D(
                [0], [0], color="#777777", ls="--", marker="s", label="Initial (IDPP)"
            )
        ]
        + method_handles(methods),
        loc="upper center",
        bbox_to_anchor=(0.49, 0.965),
        ncol=7,
        fontsize=7.2,
        columnspacing=0.85,
        handlelength=2.2,
        frameon=False,
    )
    colorbar = figure.colorbar(contour, ax=axes, fraction=0.025, pad=0.018)
    colorbar.set_label(r"$\Delta E$ (kcal mol$^{-1}$)", fontsize=9)
    return save_figure(
        figure, output_dir / "alanine_integer20_all_methods_1x4_path_evolution"
    )


def main() -> None:
    study = Path(__file__).resolve().parents[1]
    repo = Path(__file__).resolve().parents[3]
    output_dir = study / "figures/integer_alpha20_all_methods_20260801"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "v150": study / "runs/ivm/ivm_ala_v150_integer20_v49equiv_v9_pathsum_r0p96_stall_to_r0p95_trust1p5_stop1p5e3_a20_central_state_ivm_cfg714ae84c407c_budget220.npz",
        "v155": study / "runs/ivm/ivm_ala_v155_integer20_smooth_from_v150_v9_state_curve_recovery_r0p8_trust2_eqc0p2_persistent_a20_central_state_ivm_cfga4bfde1e0227_budget220.npz",
        "dense_v150": study / "runs/diagnostics/dense_barrier_ala_v150_integer20_logsample_dense.npz",
        "dense_v155": study / "runs/diagnostics/dense_barrier_ala_v155_integer20_eqc0p2_commits_dense.npz",
        "original": study / "runs/competitor_screen_original_finalists/competitor_screen_original_finalists_finalists20260731_original_nolambda_dt0p014_r1_cfg587b411dd523_budget3000.npz",
        "dense_original": study / "runs/diagnostics/dense_barrier_ala_original_dt0p014_finalist.npz",
        "static": study / "runs/neb/neb_finalists20260731_central_k30_alpha0p013_static_budget3000.npz",
        "dense_static": study / "runs/diagnostics/dense_barrier_ala_static_k30_dt0p013_finalist.npz",
        "ode": study / "formal_figure_inputs/ala_0to1_N22_k50_alpha0.0001_ode_results.npz",
        "ode_metrics": study / "formal_figure_inputs/metrics_ala_0to1_N22_k50_alpha0.0001_ode_results.npz",
        "dense_ode": study / "runs/diagnostics/dense_barrier_ala_ode_neb_k50_logsample_dense.npz",
        "bfgs": study / "runs/competitor_screen_bfgs_diagnostic_finalists/competitor_screen_bfgs_diagnostic_finalists_diagnostic_finalists20260731_bfgs_lam0p3_inner1_r1_nocap_cfg3958c11e3826_budget300.npz",
        "dense_bfgs": study / "runs/diagnostics/dense_barrier_ala_obsm_bfgs_lam0p3_r1_finalist.npz",
        "fire": study / "runs/competitor_screen_fire_finalists/competitor_screen_fire_finalists_finalists20260731_fire_lam0p85_inner3_r1_nocap_cfg3416aa96d5bf_budget2000.npz",
        "dense_fire": study / "runs/diagnostics/dense_barrier_ala_obsm_fire_lam0p85_inner3_dt0p1_finalist.npz",
        "common_initial": study / "inputs/common_initial_path_equidistant.npz",
        "rama": study / "formal_figure_inputs/ala2_scan_cache.pkl",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    ivm = load_ivm(paths["v150"], paths["v155"], paths["dense_v150"], paths["dense_v155"])
    original = load_generic(
        key="original_string",
        label="Original String ($dt=0.014$)",
        short_label="Original String",
        color="#009E73",
        marker="D",
        linestyle=(0, (1.5, 1.4)),
        result_path=paths["original"],
        dense_path=paths["dense_original"],
        selected_call=3000,
        configuration="Original String; dt=0.014; reparameterize every step",
    )
    static = load_generic(
        key="static_neb",
        label="Static NEB ($k=30$, $dt=0.013$)",
        short_label="Static NEB",
        color="#CC79A7",
        marker="X",
        linestyle=(0, (3, 1.4, 1, 1.4)),
        result_path=paths["static"],
        dense_path=paths["dense_static"],
        selected_call=3000,
        configuration="static central-tangent NEB; k=30; dt=0.013",
    )
    common_initial = np.asarray(
        np.load(paths["common_initial"], allow_pickle=False)["initial_path"],
        dtype=float,
    )
    ode = load_ode(
        paths["ode"], paths["ode_metrics"], paths["dense_ode"], common_initial
    )
    bfgs = load_generic(
        key="obsm_bfgs",
        label="OBSM--BFGS (bonded $H_0$)",
        short_label="OBSM--BFGS",
        color="#E69F00",
        marker="v",
        linestyle=(0, (6, 1.5, 1.2, 1.5)),
        result_path=paths["bfgs"],
        dense_path=paths["dense_bfgs"],
        selected_call=SELECTED_BFGS_CALL,
        configuration=(
            "OBSM full dense BFGS; lambda=0.3; one inner step; call-free bonded "
            "H0; disclosed cross-outer curvature carry; selected balanced call 91"
        ),
        truncate_paths_at_selection=True,
    )
    fire = load_generic(
        key="obsm_fire",
        label="OBSM--FIRE",
        short_label="OBSM--FIRE",
        color="#56B4E9",
        marker="P",
        linestyle=(0, (2.5, 1.4)),
        result_path=paths["fire"],
        dense_path=paths["dense_fire"],
        selected_call=1999,
        configuration="OBSM--FIRE; lambda=0.85; 3 inner steps; dt=0.1; mass=1; beta=0.1",
    )
    methods = [ivm, ode, original, static, bfgs, fire]

    common = ivm.initial_path
    initial_differences = {
        method.key: float(np.max(np.abs(method.initial_path - common)))
        for method in methods
    }
    if any(value > 5e-7 for value in initial_differences.values()):
        raise RuntimeError(f"Initial paths are not common: {initial_differences}")

    with paths["rama"].open("rb") as handle:
        rama = _ContourCacheUnpickler(handle).load()
    configure_original_style()
    descent_outputs = plot_descent(methods, output_dir)
    cost_outputs, cost_rows, annotations = plot_cost_and_paths(methods, rama, output_dir)
    evolution_outputs = plot_path_evolution(methods, rama, output_dir)

    summary_path = output_dir / "alanine_integer20_selected_methods.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "configuration",
                "selected_call",
                "selected_whole_path_force_evaluations",
                "selected_i_perp",
                "selected_dense_barrier_kcal_per_mol",
                "selected_abs_barrier_error_kcal_per_mol",
                "result_file",
                "dense_barrier_file",
            ]
        )
        for method in methods:
            writer.writerow(
                [
                    method.short_label,
                    method.configuration,
                    method.selected_call,
                    method.selected_force,
                    method.selected_iperp,
                    method.selected_barrier,
                    abs(method.selected_barrier - REFERENCE_BARRIER_KCAL),
                    str(method.result_path.relative_to(repo)),
                    str(method.dense_path.relative_to(repo)),
                ]
            )

    cost_path = output_dir / "alanine_integer20_cost_to_precision.csv"
    with cost_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "metric", "threshold", "first_passage_whole_path_force_evaluations"])
        writer.writerows(cost_rows)

    ivm_macro_path = output_dir / "alanine_integer20_ivm_published_macrostates.csv"
    v155 = np.load(paths["v155"], allow_pickle=False)
    with ivm_macro_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stage",
                "stage_call",
                "cumulative_whole_path_force_evaluations",
                "i_perp",
                "segment_length_cv",
                "dense_barrier_kcal_per_mol",
            ]
        )
        v150_total = int(
            np.asarray(np.load(paths["v150"])["counter_vectors"], dtype=np.int64)[-1].sum()
        )
        dense_v155 = np.load(paths["dense_v155"], allow_pickle=False)
        for call in np.asarray(dense_v155["calls"], dtype=int):
            if call > SELECTED_V155_CALL:
                continue
            index = history_index(v155, int(call))
            writer.writerow(
                [
                    "V155",
                    int(call),
                    v150_total + force_at_call(v155, int(call)),
                    float(v155["i_perp"][index]),
                    float(v155["segment_length_cv"][index]),
                    dense_at_call(dense_v155, int(call)),
                ]
            )

    audit = {
        "scope": "Ala 66D independently tuned all-method comparison; paper directory not modified",
        "force_accounting": "sum of all 22 per-image force counters, including endpoints and rejected/hidden work",
        "dense_barrier_accounting": "separate diagnostic OpenMM evaluations, never charged to N_F",
        "barrier_reference_kcal_per_mol": REFERENCE_BARRIER_KCAL,
        "ivm": {
            "alpha": 20,
            "alpha_is_integer": True,
            "alpha_mode": "fixed",
            "selected_stage": "V155",
            "selected_call": SELECTED_V155_CALL,
            "selected_is_complete_macrostate": True,
            "selected_whole_path_force_evaluations": ivm.selected_force,
            "selected_i_perp": ivm.selected_iperp,
            "selected_dense_barrier_kcal_per_mol": ivm.selected_barrier,
            "selected_abs_barrier_error_kcal_per_mol": abs(ivm.selected_barrier - REFERENCE_BARRIER_KCAL),
        },
        "ode_neb_terminal": {
            "whole_path_force_evaluations": ode.selected_force,
            "i_perp": ode.selected_iperp,
            "dense_barrier_kcal_per_mol": ode.selected_barrier,
            "abs_barrier_error_kcal_per_mol": abs(ode.selected_barrier - REFERENCE_BARRIER_KCAL),
        },
        "ivm_beats_ode_terminal_i_perp": bool(ivm.selected_iperp < ode.selected_iperp),
        "ivm_beats_ode_terminal_barrier_error": bool(
            abs(ivm.selected_barrier - REFERENCE_BARRIER_KCAL)
            < abs(ode.selected_barrier - REFERENCE_BARRIER_KCAL)
        ),
        "selected_methods": [
            {
                "key": method.key,
                "label": method.label,
                "configuration": method.configuration,
                "selected_call": method.selected_call,
                "selected_whole_path_force_evaluations": method.selected_force,
                "selected_i_perp": method.selected_iperp,
                "selected_dense_barrier_kcal_per_mol": method.selected_barrier,
                "selected_abs_barrier_error_kcal_per_mol": abs(
                    method.selected_barrier - REFERENCE_BARRIER_KCAL
                ),
            }
            for method in methods
        ],
        "ivm_vs_ode_first_passage_speedups": annotations,
        "initial_path_max_abs_differences_angstrom": initial_differences,
        "bfgs_disclosure": (
            "The displayed OBSM--BFGS finalist uses full dense BFGS updates with a "
            "call-free bonded-force-field H0 and cross-outer curvature carry. This is "
            "a disclosed tuned implementation choice, not the strict identity-H0 paper baseline."
        ),
        "paper_files_modified": False,
        "inputs": {
            str(path.relative_to(repo)): sha256(path) for path in paths.values()
        },
        "outputs": {
            "two_metric_descent": descent_outputs,
            "cost_to_precision_and_paths": cost_outputs,
            "path_evolution_1x4": evolution_outputs,
            "selected_methods_csv": sha256(summary_path),
            "cost_to_precision_csv": sha256(cost_path),
            "ivm_macrostates_csv": sha256(ivm_macro_path),
        },
    }
    audit_path = output_dir / "alanine_integer20_all_methods_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
