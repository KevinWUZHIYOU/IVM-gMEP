#!/usr/bin/env python3
"""Plot the fixed-alpha V49 -> V50 -> V147 Ala production lineage.

V147 publishes only complete operator-split macrostates: an ordinary
frozen-normal fixed-alpha IVM update, a coordinate-only identical-spring mesh
refresh, and a fixed-alpha IVM recovery satisfying both the physical and mesh
guards.  All hidden recovery and rejected force queries remain charged on the
horizontal axis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from plot_v49_formal_ala_1x3 import (
    EV_TO_KCAL,
    _ContourCacheUnpickler,
    configure_original_style,
    cost_to_reach,
    path_to_dihedrals,
)
from plot_v67_continuation_figures import (
    COLOR_INIT,
    COLOR_IVM,
    COLOR_NEB,
    clean_axis,
    panel_label,
    save_figure,
    stable_stage_points,
)


SELECTED_V147_CALL = 176


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_name(path: Path, repo: Path) -> str:
    """Return a stable repository-relative name when possible (Python 3.8+)."""
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def scalar(saved, name):
    return np.asarray(saved[name]).reshape(()).item()


def history_index_at_call(saved, call: int) -> int:
    indices = np.flatnonzero(
        np.asarray(saved["calls_per_image"], dtype=int) == int(call)
    )
    if not indices.size:
        raise RuntimeError(f"Missing history record at call {call}")
    return int(indices[-1])


def path_at_call(saved, call: int) -> np.ndarray:
    indices = np.flatnonzero(
        np.asarray(saved["stored_path_calls"], dtype=int) == int(call)
    )
    if not indices.size:
        raise RuntimeError(f"Missing stored path at call {call}")
    return np.asarray(saved["stored_paths"][indices[-1]], dtype=float)


def counter_total_at_call(saved, call: int) -> int:
    index = history_index_at_call(saved, call)
    return int(np.asarray(saved["counter_vectors"], dtype=np.int64)[index].sum())


def v147_macro_points(saved, selected_call: int) -> list[dict]:
    committed = np.flatnonzero(np.asarray(saved["reparam_committed"], dtype=bool))
    calls = np.asarray(saved["calls_per_image"], dtype=int)
    iperp = np.asarray(saved["i_perp"], dtype=float)
    cv = np.asarray(saved["segment_length_cv"], dtype=float)
    counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
    points = []
    for index in committed:
        call = int(calls[index])
        if call > selected_call:
            continue
        points.append(
            {
                "call": call,
                "total_force": int(counters[index].sum()),
                "iperp": float(iperp[index]),
                "cv": float(cv[index]),
                "path": path_at_call(saved, call),
                "macrostate": True,
            }
        )
    if not points or points[-1]["call"] != selected_call:
        raise RuntimeError(
            f"Selected V147 call {selected_call} is not a committed macrostate"
        )
    if np.any(np.diff([point["iperp"] for point in points]) > 1e-12):
        raise RuntimeError("Committed V147 I_perp sequence is not monotone")
    return points


def main() -> None:
    study = Path(__file__).resolve().parents[1]
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v49",
        type=Path,
        default=study / "runs/ivm/ivm_ala_v49_production_full_alpha_v9_pathsum_r0p96_stall_to_r0p95_trust1p5_stop1p5e3_a20p8_central_state_ivm_cfgd17c08618165_budget220.npz",
    )
    parser.add_argument(
        "--v50",
        type=Path,
        default=study / "runs/ivm/ivm_ala_v50_continue_v49_alpha_screen_v9_pathsum_r0p96_stall_to_r0p95_trust1p5_a20p8_central_state_ivm_cfgab44264ee7d6_budget120.npz",
    )
    parser.add_argument(
        "--v147",
        type=Path,
        default=study / "runs/ivm/ivm_ala_v147_fixeda20p8_curve_eqc0p05_persistent_central_state_ivm_cfg6d7cb44ca5d4_budget300.npz",
    )
    parser.add_argument("--selected-v147-call", type=int, default=SELECTED_V147_CALL)
    parser.add_argument(
        "--dense-v49",
        type=Path,
        default=study / "runs/diagnostics/dense_barrier_v49_production_a20p8_dense25.npz",
    )
    parser.add_argument(
        "--dense-v50",
        type=Path,
        default=study / "runs/diagnostics/dense_barrier_ala_v50_terminal_for_v67_lineage.npz",
    )
    parser.add_argument(
        "--dense-v147",
        type=Path,
        default=study / "runs/diagnostics/dense_barrier_ala_v147_curve_eqc0p05_persistent_commits_dense.npz",
    )
    parser.add_argument(
        "--neb",
        type=Path,
        default=study / "formal_figure_inputs/ala_0to1_N22_k50_alpha0.0001_ode_results.npz",
    )
    parser.add_argument(
        "--neb-metrics",
        type=Path,
        default=study / "formal_figure_inputs/metrics_ala_0to1_N22_k50_alpha0.0001_ode_results.npz",
    )
    parser.add_argument(
        "--dense-neb",
        type=Path,
        default=study / "runs/diagnostics/dense_barrier_ala_ode_neb_k50_logsample_dense.npz",
    )
    parser.add_argument(
        "--rama-cache",
        type=Path,
        default=study / "formal_figure_inputs/ala2_scan_cache.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=study / "figures/fixed_alpha_high_precision_v147_20260731",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = [
        args.v49,
        args.v50,
        args.v147,
        args.dense_v49,
        args.dense_v50,
        args.dense_v147,
        args.neb,
        args.neb_metrics,
        args.dense_neb,
        args.rama_cache,
    ]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)

    v49 = np.load(args.v49, allow_pickle=False)
    v50 = np.load(args.v50, allow_pickle=False)
    v147 = np.load(args.v147, allow_pickle=False)
    dense_v49 = np.load(args.dense_v49, allow_pickle=False)
    dense_v50 = np.load(args.dense_v50, allow_pickle=False)
    dense_v147 = np.load(args.dense_v147, allow_pickle=False)
    neb = np.load(args.neb, allow_pickle=False)
    neb_metrics = np.load(args.neb_metrics, allow_pickle=False)
    dense_neb = np.load(args.dense_neb, allow_pickle=False)
    with args.rama_cache.open("rb") as handle:
        rama = _ContourCacheUnpickler(handle).load()

    for label, saved in (("V49", v49), ("V50", v50), ("V147", v147)):
        if scalar(saved, "alpha_mode") != "fixed" or not np.isclose(
            float(scalar(saved, "alpha")), 20.8
        ):
            raise RuntimeError(f"{label} is not the declared fixed-alpha=20.8 run")
        if not np.allclose(np.asarray(saved["active_alpha"]), 20.8):
            raise RuntimeError(f"{label} contains a nonconstant active alpha")

    v49_total = int(np.asarray(v49["counter_vectors"], dtype=np.int64)[-1].sum())
    v50_total = int(np.asarray(v50["counter_vectors"], dtype=np.int64)[-1].sum())
    offsets = [0, v49_total, v49_total + v50_total]
    stage_specs = [
        (v49, stable_stage_points(v49), "V49"),
        (v50, stable_stage_points(v50), "V50"),
        (v147, v147_macro_points(v147, args.selected_v147_call), "V147"),
    ]
    lineage = []
    for stage_index, (_saved, points, label) in enumerate(stage_specs):
        for point in points:
            candidate = dict(point)
            candidate["stage"] = label
            candidate["cumulative_total_force"] = offsets[stage_index] + int(
                point["total_force"]
            )
            if lineage and np.array_equal(lineage[-1]["path"], candidate["path"]):
                continue
            if lineage and candidate["iperp"] > lineage[-1]["iperp"] + 1e-12:
                continue
            lineage.append(candidate)

    ivm_force = np.asarray([point["cumulative_total_force"] for point in lineage])
    ivm_iperp = np.asarray([point["iperp"] for point in lineage])
    ivm_cv = np.asarray([point["cv"] for point in lineage])
    final_point = lineage[-1]
    if final_point["stage"] != "V147" or final_point["call"] != args.selected_v147_call:
        raise RuntimeError("Published lineage did not end at the selected V147 macrostate")

    barrier_force = []
    barrier_values = []
    for saved, diagnostic, offset, allowed_calls in (
        (v49, dense_v49, offsets[0], None),
        (v50, dense_v50, offsets[1], None),
        (
            v147,
            dense_v147,
            offsets[2],
            {1}.union(point["call"] for point in stage_specs[-1][1]),
        ),
    ):
        for call, barrier in zip(
            np.asarray(diagnostic["calls"], dtype=int),
            np.asarray(diagnostic["adaptive_barrier_kcal"], dtype=float),
        ):
            if allowed_calls is not None and int(call) not in allowed_calls:
                continue
            barrier_force.append(offset + counter_total_at_call(saved, int(call)))
            barrier_values.append(float(barrier))
    barrier_force = np.asarray(barrier_force, dtype=np.int64)
    barrier_values = np.asarray(barrier_values, dtype=float)
    order = np.argsort(barrier_force, kind="stable")
    barrier_force = barrier_force[order]
    barrier_values = barrier_values[order]

    neb_count = np.asarray(neb_metrics["count_seq"], dtype=np.int64)
    valid = neb_count >= 1
    neb_force = 20 * neb_count[valid] + 2
    neb_iperp = np.asarray(neb_metrics["loss_perp"], dtype=float)[valid]
    neb_dense_calls = np.asarray(dense_neb["calls"], dtype=np.int64)
    neb_barrier_force = 20 * neb_dense_calls + 2
    neb_barrier = np.asarray(dense_neb["adaptive_barrier_kcal"], dtype=float)
    barrier_reference = float(neb_barrier[-1])

    configure_original_style()
    method_labels = ("Best tuned ODE--NEB ($k=50$)", "Fixed-$\\alpha$ IVM (V49--V147)")

    # Best-method two-observable descent plot.
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    axes[0].plot(neb_force, neb_iperp, color=COLOR_NEB, lw=1.35, label=method_labels[0])
    axes[0].plot(ivm_force, ivm_iperp, color=COLOR_IVM, lw=2.2, marker="o", ms=2.6, label=method_labels[1], zorder=5)
    axes[1].plot(neb_barrier_force, neb_barrier, color=COLOR_NEB, lw=1.35, marker="o", ms=2.4, label=method_labels[0])
    axes[1].plot(barrier_force, barrier_values, color=COLOR_IVM, lw=2.2, marker="s", ms=3.0, label=method_labels[1], zorder=5)
    axes[1].axhline(barrier_reference, color="black", lw=0.7, ls=":", label="NEB terminal barrier")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$I_\perp$")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"Densified 0 K barrier $\Delta E$ (kcal mol$^{-1}$)")
    for index, axis in enumerate(axes):
        axis.set_xscale("log")
        axis.set_xlim(20, 1.05 * max(neb_force[-1], neb_barrier_force[-1]))
        axis.set_xlabel("Total Number of Force Evaluations, $N_F$")
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    axes[0].legend(frameon=False, loc="upper right", fontsize=7.3)
    axes[0].annotate(
        f"$I_\\perp={final_point['iperp']:.2e}$",
        xy=(final_point["cumulative_total_force"], final_point["iperp"]),
        xytext=(18, 20),
        textcoords="offset points",
        color=COLOR_IVM,
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color=COLOR_IVM, lw=0.8),
    )
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.91, wspace=0.31)
    loss_outputs = save_figure(figure, args.output_dir / "alanine_v147_best_methods_two_metric_loss")

    # Measured first-passage cost-to-precision plus final path overlay.
    iperp_thresholds = np.geomspace(1e-1, float(neb_iperp[-1]), 12)
    barrier_thresholds = np.asarray([10, 7, 5, 4, 3.5, 3, 2.8, 2.65, 2.55, 2.52, 2.50, 2.49])
    neb_cost_iperp = cost_to_reach(neb_force, neb_iperp, iperp_thresholds)
    ivm_cost_iperp = cost_to_reach(ivm_force, ivm_iperp, iperp_thresholds)
    neb_cost_barrier = cost_to_reach(neb_barrier_force, neb_barrier, barrier_thresholds)
    ivm_cost_barrier = cost_to_reach(barrier_force, barrier_values, barrier_thresholds)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.0, 4.0),
        dpi=200,
        gridspec_kw={"width_ratios": [1, 1, 1.18]},
        constrained_layout=True,
    )
    speedups = {}

    def cost_panel(axis, thresholds, neb_cost, ivm_cost, xlabel, label, key, logx):
        both = np.isfinite(neb_cost) & np.isfinite(ivm_cost)
        thresholds = thresholds[both]
        neb_cost = neb_cost[both]
        ivm_cost = ivm_cost[both]
        if not len(thresholds):
            raise RuntimeError(f"No common cost-to-precision threshold for {key}")
        plotter = axis.semilogx if logx else axis.plot
        plotter(thresholds, neb_cost, marker="o", ms=6.5, lw=1.6, color=COLOR_NEB, label="ODE--NEB ($k=50$)")
        plotter(thresholds, ivm_cost, marker="s", ms=6.0, lw=1.8, color=COLOR_IVM, label="Fixed-$\\alpha$ IVM")
        axis.invert_xaxis()
        strict = -1
        speedup = float(neb_cost[strict] / ivm_cost[strict])
        speedups[key] = {
            "threshold": float(thresholds[strict]),
            "neb_force_evaluations": float(neb_cost[strict]),
            "ivm_force_evaluations": float(ivm_cost[strict]),
            "speedup": speedup,
        }
        axis.annotate("", xy=(thresholds[strict], ivm_cost[strict]), xytext=(thresholds[strict], neb_cost[strict]), arrowprops=dict(arrowstyle="<->", color="black", lw=0.9, shrinkA=4, shrinkB=4))
        axis.text(thresholds[strict], np.sqrt(neb_cost[strict] * ivm_cost[strict]), f"  {speedup:.1f}x", va="center", fontsize=9, fontweight="bold")
        axis.set_yscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Total force evaluations, $N_F$")
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value / 1000:g}k"))
        axis.legend(frameon=False, loc="upper left", fontsize=7.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(True, which="major", ls="--", lw=0.4, alpha=0.4)
        panel_label(axis, label, formal=True)

    cost_panel(axes[0], iperp_thresholds, neb_cost_iperp, ivm_cost_iperp, r"$I_\perp$ threshold", "a", "i_perp", True)
    cost_panel(axes[1], barrier_thresholds, neb_cost_barrier, ivm_cost_barrier, r"Densified barrier target (kcal mol$^{-1}$)", "b", "dense_barrier", False)

    phi_grid, psi_grid = np.meshgrid(rama["phi_g"], rama["psi_g"], indexing="ij")
    energy_grid = (rama["PES"] - np.nanmin(rama["PES"])) * EV_TO_KCAL
    axis = axes[2]
    contour = axis.contourf(phi_grid, psi_grid, energy_grid, levels=np.linspace(0, 15, 16), cmap="plasma", extend="max")
    axis.contour(phi_grid, psi_grid, energy_grid, levels=np.linspace(0, 15, 8), colors="white", linewidths=0.3, alpha=0.5)
    colorbar = figure.colorbar(contour, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(r"$\Delta E$ (kcal mol$^{-1}$)", fontsize=9)
    initial_path = np.asarray(v49["initial_path"], dtype=float)
    final_neb_path = np.asarray(neb["images_seq"][-1], dtype=float)
    final_ivm_path = np.asarray(final_point["path"], dtype=float)
    for path, style, color, marker, label, zorder in (
        (initial_path, "--", COLOR_INIT, "s", "Initial (IDPP)", 3),
        (final_neb_path, "-", COLOR_NEB, "o", "ODE--NEB ($k=50$)", 4),
        (final_ivm_path, "-", COLOR_IVM, "^", "Fixed-$\\alpha$ IVM", 5),
    ):
        phi, psi = path_to_dihedrals(path)
        axis.plot(phi, psi, style, color=color, lw=1.8, marker=marker, ms=5.0, mec="black", mew=0.4, label=label, zorder=zorder)
    for minimum in rama["unique"][:2]:
        axis.scatter(minimum["phi"], minimum["psi"], s=220, marker="*", c="gold", edgecolor="black", zorder=6, linewidth=0.8)
    axis.set_xlim(-180, 180)
    axis.set_ylim(-180, 180)
    axis.set_xticks(range(-180, 181, 90))
    axis.set_yticks(range(-180, 181, 90))
    axis.set_xlabel(r"$\phi$ (deg)")
    axis.set_ylabel(r"$\psi$ (deg)")
    axis.set_aspect("equal")
    axis.legend(loc="upper right", fontsize=7.2, facecolor="white", edgecolor="black", framealpha=0.9)
    panel_label(axis, "c", formal=True)
    cost_outputs = save_figure(figure, args.output_dir / "alanine_v147_cost_to_precision_and_path")

    def ivm_path_at_or_before(target: int) -> np.ndarray:
        indices = np.flatnonzero(ivm_force <= int(target))
        return np.asarray(lineage[indices[-1] if indices.size else 0]["path"])

    def neb_path_at_or_before(target: int) -> np.ndarray:
        costs = 20 * np.asarray(neb["count_seq"], dtype=np.int64) + 2
        indices = np.flatnonzero(costs <= int(target))
        return np.asarray(neb["images_seq"][indices[-1] if indices.size else 0])

    panel_config = [
        (0, "Number of Forces: 0"),
        (1000, "Number of Forces: 1k"),
        (4000, "Number of Forces: 4k"),
        (None, f"selected IVM: {final_point['cumulative_total_force'] / 1000:.1f}k"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=200)
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.12, top=0.90, wspace=0.18)
    contour = None
    for axis, (target, title) in zip(axes, panel_config):
        contour = axis.contourf(phi_grid, psi_grid, energy_grid, levels=np.linspace(0, 15, 16), cmap="pink", extend="max")
        axis.contour(phi_grid, psi_grid, energy_grid, levels=np.linspace(0, 15, 8), colors="white", linewidths=0.3, alpha=0.5)
        if target == 0:
            phi, psi = path_to_dihedrals(initial_path)
            axis.plot(phi, psi, "--", color=COLOR_INIT, lw=1.6, marker="s", ms=4.5, mec="black", mew=0.4, label="Initial (IDPP)", zorder=3)
        else:
            neb_path = final_neb_path if target is None else neb_path_at_or_before(target)
            ivm_path = final_ivm_path if target is None else ivm_path_at_or_before(target)
            neb_phi, neb_psi = path_to_dihedrals(neb_path)
            ivm_phi, ivm_psi = path_to_dihedrals(ivm_path)
            axis.plot(neb_phi, neb_psi, "-", color=COLOR_NEB, lw=1.8, marker="o", ms=4.0, mec="black", mew=0.4, label="ODE--NEB", zorder=4)
            axis.plot(ivm_phi, ivm_psi, "-", color=COLOR_IVM, lw=1.8, marker="^", ms=5.0, mec="black", mew=0.4, label="fixed-$\\alpha$ IVM", zorder=5)
        for minimum in rama["unique"][:2]:
            axis.scatter(minimum["phi"], minimum["psi"], s=220, marker="*", c="gold", edgecolor="black", zorder=6, linewidth=0.8)
        axis.set_xlim(-180, 0)
        axis.set_ylim(0, 180)
        axis.set_xticks(range(-180, 1, 60))
        axis.set_yticks(range(0, 181, 60))
        axis.set_xlabel(r"$\phi$ (deg)")
        axis.set_ylabel(r"$\psi$ (deg)")
        axis.set_aspect("equal")
        axis.set_title(title, fontsize=10)
        axis.legend(loc="upper right", fontsize=6.5, facecolor="white", edgecolor="black", framealpha=0.9)
    colorbar = figure.colorbar(contour, ax=axes, fraction=0.03, pad=0.02, location="right")
    colorbar.set_label(r"$\Delta E$ (kcal mol$^{-1}$)", fontsize=9)
    contour_outputs = save_figure(figure, args.output_dir / "alanine_v147_1x4_path_evolution")

    csv_path = args.output_dir / "alanine_v49_v50_v147_published_macrostates.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "stage_call", "cumulative_total_force", "i_perp", "segment_length_cv"])
        for point in lineage:
            writer.writerow([point["stage"], point["call"], point["cumulative_total_force"], point["iperp"], point["cv"]])

    selected_history = history_index_at_call(v147, args.selected_v147_call)
    selected_barrier_index = np.flatnonzero(
        np.asarray(dense_v147["calls"], dtype=int) == args.selected_v147_call
    )
    if not selected_barrier_index.size:
        raise RuntimeError("Selected V147 call has no dense barrier diagnostic")
    selected_barrier = float(dense_v147["adaptive_barrier_kcal"][selected_barrier_index[-1]])
    audit = {
        "policy": "fixed alpha=20.8; V49->V50 followed by complete V147 operator-split macrostates only",
        "selected_v147_call": args.selected_v147_call,
        "selected_cumulative_total_force": int(final_point["cumulative_total_force"]),
        "selected_i_perp": float(v147["i_perp"][selected_history]),
        "selected_segment_length_cv": float(v147["segment_length_cv"][selected_history]),
        "selected_dense_barrier_kcal_per_mol": selected_barrier,
        "neb_terminal_i_perp": float(neb_iperp[-1]),
        "neb_terminal_dense_barrier_kcal_per_mol": barrier_reference,
        "active_alpha_unique_by_stage": {
            label: np.unique(np.asarray(saved["active_alpha"], dtype=float)).tolist()
            for label, saved in (("V49", v49), ("V50", v50), ("V147", v147))
        },
        "selected_i_perp_strictly_below_neb_terminal": bool(
            float(v147["i_perp"][selected_history]) < float(neb_iperp[-1])
        ),
        "selected_i_perp_improvement_fraction_vs_neb": float(
            1.0 - float(v147["i_perp"][selected_history]) / float(neb_iperp[-1])
        ),
        "selected_dense_barrier_absolute_error_vs_neb_kcal_per_mol": float(
            abs(selected_barrier - barrier_reference)
        ),
        "v147_committed_i_perp_monotone_through_selection": bool(
            np.all(
                np.diff([point["iperp"] for point in stage_specs[-1][1]])
                <= 1e-12
            )
        ),
        "v147_dense_barrier_through_selection": [
            {
                "call": int(call),
                "kcal_per_mol": float(barrier),
            }
            for call, barrier in zip(
                np.asarray(dense_v147["calls"], dtype=int),
                np.asarray(dense_v147["adaptive_barrier_kcal"], dtype=float),
            )
            if int(call) <= args.selected_v147_call
        ],
        "stage_offsets_total_force": offsets,
        "published_points": len(lineage),
        "maximum_v147_macrostate_cv_through_selection": float(
            max(point["cv"] for point in stage_specs[-1][1])
        ),
        "speedups": speedups,
        "paper_files_modified": False,
        "inputs": {
            provenance_name(path, repo): sha256(path)
            for path in inputs
        },
        "outputs": {
            "loss_descent": loss_outputs,
            "cost_to_precision": cost_outputs,
            "contours_1x4": contour_outputs,
            "trajectory_csv": sha256(csv_path),
        },
    }
    audit_path = args.output_dir / "alanine_v147_fixed_alpha_figures.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
