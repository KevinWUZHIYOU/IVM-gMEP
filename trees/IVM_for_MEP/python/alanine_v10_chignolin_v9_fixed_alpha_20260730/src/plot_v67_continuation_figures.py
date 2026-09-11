#!/usr/bin/env python3
"""Plot the fully charged V49 -> V67 Ala continuation lineage.

The visual layout is inherited from the formal V49 Ala scripts.  The only new
adapter concatenates independently audited continuation stages while charging
all force queries from every stage.  Only genuine path changes with the
declared spacing bound and monotone physical merit are published.
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
    log_thresholds_uniform,
    path_to_dihedrals,
)


COLOR_NEB = "#D55E00"
COLOR_IVM = "#0072B2"
COLOR_INIT = "#888888"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_name(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def scalar(saved, name):
    return np.asarray(saved[name]).reshape(()).item()


def path_change_indices(paths: np.ndarray) -> np.ndarray:
    changed = [0]
    for index in range(1, len(paths)):
        if not np.array_equal(paths[index], paths[changed[-1]]):
            changed.append(index)
    return np.asarray(changed, dtype=int)


def stable_stage_points(saved, *, max_cv: float = 1e-3) -> list[dict]:
    history_calls = np.asarray(saved["calls_per_image"], dtype=int)
    history_iperp = np.asarray(saved["i_perp"], dtype=float)
    history_cv = np.asarray(saved["segment_length_cv"], dtype=float)
    history_counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
    stored_calls = np.asarray(saved["stored_path_calls"], dtype=int)
    stored_paths = np.asarray(saved["stored_paths"], dtype=float)
    call_to_history = {int(call): i for i, call in enumerate(history_calls)}
    points = []
    best = np.inf
    for stored_index in path_change_indices(stored_paths):
        call = int(stored_calls[stored_index])
        history_index = call_to_history[call]
        iperp = float(history_iperp[history_index])
        cv = float(history_cv[history_index])
        if cv > max_cv + 1e-15 or iperp > best + 1e-12:
            continue
        best = min(best, iperp)
        points.append(
            {
                "call": call,
                "total_force": int(history_counters[history_index].sum()),
                "iperp": iperp,
                "cv": cv,
                "path": stored_paths[stored_index].copy(),
            }
        )
    final_path = np.asarray(saved["final_path"], dtype=float)
    final_history = len(history_calls) - 1
    final = {
        "call": int(history_calls[final_history]),
        "total_force": int(history_counters[final_history].sum()),
        "iperp": float(history_iperp[final_history]),
        "cv": float(history_cv[final_history]),
        "path": final_path.copy(),
    }
    if final["cv"] <= max_cv + 1e-15 and final["iperp"] <= best + 1e-12:
        if not points or not np.array_equal(points[-1]["path"], final_path):
            points.append(final)
    if not points:
        raise RuntimeError(f"No stable points in {scalar(saved, 'run_label')}")
    return points


def counter_total_at_call(saved, call: int) -> int:
    calls = np.asarray(saved["calls_per_image"], dtype=int)
    indices = np.flatnonzero(calls == int(call))
    if not indices.size:
        raise RuntimeError(f"Missing metric record for call {call}")
    counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
    return int(counters[indices[-1]].sum())


def save_figure(figure, stem: Path) -> dict[str, str]:
    outputs = {}
    for suffix, kwargs in ((".pdf", {}), (".svg", {}), (".png", {"dpi": 300})):
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", **kwargs)
        outputs[suffix] = sha256(path)
    plt.close(figure)
    return outputs


def clean_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.55)


def panel_label(axis, label: str, *, formal: bool = False) -> None:
    axis.text(
        -0.20 if formal else 0.0,
        1.02 if formal else 1.04,
        label,
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=14 if formal else 9,
        va="bottom",
    )


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
        "--v57",
        type=Path,
        default=study / "runs/ivm/ivm_ala_v57_locked_dense_alpha_from_v50_v9_pathsum_r0p9_trust1p5_locked_a28_central_state_ivm_cfgdff662b0d079_budget260.npz",
    )
    parser.add_argument(
        "--v61",
        type=Path,
        default=study / "runs/ivm/ivm_ala_v61_cv_relock_from_v57_a28_v9_pathsum_r0p96_stall_to_r0p95_trust1p5_cv_relock_all_t0p1_a25_central_state_ivm_cfg7360a0078a6a_budget500.npz",
    )
    parser.add_argument(
        "--v67",
        type=Path,
        default=study / "runs/ivm/ivm_ala_v67_counter_closed_from_v61_v9_pathsum_r0p96_stall_to_r0p95_trust1p5_cv_relock_all_t0p1_a30_central_state_ivm_cfg89be1c54fc9c_budget700.npz",
    )
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
        "--dense-v57",
        type=Path,
        default=study / "runs/diagnostics/dense_barrier_ala_v57_terminal_for_v67_lineage.npz",
    )
    parser.add_argument(
        "--dense-v61",
        type=Path,
        default=study / "runs/diagnostics/dense_barrier_ala_v61_terminal_for_v67_lineage.npz",
    )
    parser.add_argument(
        "--dense-v67",
        type=Path,
        default=study / "runs/diagnostics/dense_barrier_ala_v67_a30_final_dense.npz",
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
        "--rama-cache",
        type=Path,
        default=study / "formal_figure_inputs/ala2_scan_cache.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=study / "figures/formal_v67_continuation",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage_paths = [args.v49, args.v50, args.v57, args.v61, args.v67]
    dense_paths = [
        args.dense_v49,
        args.dense_v50,
        args.dense_v57,
        args.dense_v61,
        args.dense_v67,
    ]
    inputs = stage_paths + dense_paths + [args.neb, args.neb_metrics, args.rama_cache]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)

    stages = [np.load(path, allow_pickle=False) for path in stage_paths]
    dense = [np.load(path, allow_pickle=False) for path in dense_paths]
    neb = np.load(args.neb, allow_pickle=False)
    neb_metrics = np.load(args.neb_metrics, allow_pickle=False)
    with args.rama_cache.open("rb") as handle:
        rama = _ContourCacheUnpickler(handle).load()

    lineage = []
    total_offset = 0
    stage_offsets = []
    for stage_index, saved in enumerate(stages):
        stage_offsets.append(total_offset)
        for point in stable_stage_points(saved):
            candidate = dict(point)
            candidate["stage"] = stage_index
            candidate["cumulative_total_force"] = total_offset + point["total_force"]
            if lineage and np.array_equal(lineage[-1]["path"], candidate["path"]):
                continue
            if lineage and candidate["iperp"] > lineage[-1]["iperp"] + 1e-12:
                continue
            lineage.append(candidate)
        total_offset += int(np.asarray(saved["counter_vectors"], dtype=np.int64)[-1].sum())

    ivm_force = np.asarray([point["cumulative_total_force"] for point in lineage])
    ivm_iperp = np.asarray([point["iperp"] for point in lineage])
    ivm_cv = np.asarray([point["cv"] for point in lineage])
    neb_count = np.asarray(neb_metrics["count_seq"], dtype=np.int64)
    # The historical cache begins with a count-zero placeholder whose stored
    # residual is exactly zero.  The original formal script excluded it via
    # cost_to_reach(min_count=1) before converting to total force cost.
    neb_valid = neb_count >= 1
    neb_force = 20 * neb_count[neb_valid] + 2
    neb_iperp = np.asarray(neb_metrics["loss_perp"], dtype=float)[neb_valid]
    neb_barrier = (
        np.asarray(neb_metrics["Delta_Energy"], dtype=float)[neb_valid]
        * EV_TO_KCAL
    )

    barrier_force = []
    barrier_values = []
    for stage_index, (saved, diagnostic) in enumerate(zip(stages, dense)):
        for call, barrier in zip(
            np.asarray(diagnostic["calls"], dtype=int),
            np.asarray(diagnostic["adaptive_barrier_kcal"], dtype=float),
        ):
            barrier_force.append(stage_offsets[stage_index] + counter_total_at_call(saved, int(call)))
            barrier_values.append(float(barrier))
    barrier_force = np.asarray(barrier_force, dtype=np.int64)
    barrier_values = np.asarray(barrier_values, dtype=float)
    order = np.argsort(barrier_force, kind="stable")
    barrier_force = barrier_force[order]
    barrier_values = barrier_values[order]

    configure_original_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    method_labels = ("Best tuned ODE--NEB ($k=50$)", "Best IVM (V49--V67)")
    axes[0].plot(neb_force, neb_iperp, color=COLOR_NEB, lw=1.35, label=method_labels[0])
    axes[0].plot(ivm_force, ivm_iperp, color=COLOR_IVM, lw=2.2, label=method_labels[1], zorder=5)
    axes[1].plot(neb_force, neb_barrier, color=COLOR_NEB, lw=1.35, marker="o", ms=2.0, markevery=max(1, len(neb_barrier) // 10), label=method_labels[0])
    axes[1].plot(barrier_force, barrier_values, color=COLOR_IVM, lw=2.2, marker="o", ms=3.0, label=method_labels[1], zorder=5)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel(r"$I_\perp$")
    axes[1].set_ylabel(r"Densified 0 K barrier $\Delta E$ (kcal mol$^{-1}$)")
    maximum_force = int(max(neb_force[-1], total_offset))
    for index, axis in enumerate(axes):
        axis.set_xlim(0, 1.005 * maximum_force)
        axis.set_xlabel("Number of Force Evaluations, $N_F$")
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].annotate(
        f"V67: $I_\\perp={ivm_iperp[-1]:.2e}$",
        xy=(ivm_force[-1], ivm_iperp[-1]),
        xytext=(28, 28),
        textcoords="offset points",
        color=COLOR_IVM,
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color=COLOR_IVM, lw=0.8),
    )
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.90, wspace=0.29)
    metrics_outputs = save_figure(figure, args.output_dir / "alanine_v67_main_metrics_total_nf")

    # The requested "loss" view is a method comparison, not an IVM-only
    # continuation trace.  Plot both physical observables simultaneously and
    # retain the complete best-tuned ODE--NEB record for an honest cost axis.
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].plot(neb_force, neb_iperp, color=COLOR_NEB, lw=1.35, label=method_labels[0])
    axes[0].plot(ivm_force, ivm_iperp, color=COLOR_IVM, lw=2.2, label=method_labels[1], zorder=5)
    axes[1].plot(neb_force, neb_barrier, color=COLOR_NEB, lw=1.35, marker="o", ms=2.0, markevery=max(1, len(neb_barrier) // 10), label=method_labels[0])
    axes[1].plot(barrier_force, barrier_values, color=COLOR_IVM, lw=2.2, marker="o", ms=3.0, label=method_labels[1], zorder=5)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel(r"$I_\perp$")
    axes[1].set_ylabel(r"Densified 0 K energy barrier $\Delta E$ (kcal mol$^{-1}$)")
    for index, axis in enumerate(axes):
        axis.set_xlim(0, 1.005 * maximum_force)
        axis.set_xlabel("Total Number of Force Evaluations, $N_F$")
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    axes[0].legend(frameon=False, loc="upper right")
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.90, wspace=0.31)
    loss_outputs = save_figure(figure, args.output_dir / "alanine_best_methods_two_metric_loss_total_nf")

    delta_e_star = float(np.nanmin(np.concatenate([neb_barrier, barrier_values])))
    neb_barrier_error = np.abs(neb_barrier - delta_e_star)
    ivm_barrier_error = np.abs(barrier_values - delta_e_star)
    thresh_lp = log_thresholds_uniform(2e-4, 1e0, 4)[::-1]
    thresh_de = log_thresholds_uniform(1e-2, 1e0, 8)[::-1]
    neb_cost_lp = cost_to_reach(neb_force, neb_iperp, thresh_lp)
    ivm_cost_lp = cost_to_reach(ivm_force, ivm_iperp, thresh_lp)
    neb_cost_de = cost_to_reach(neb_force, neb_barrier_error, thresh_de)
    ivm_cost_de = cost_to_reach(barrier_force, ivm_barrier_error, thresh_de)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.0, 4.0),
        dpi=200,
        gridspec_kw={"width_ratios": [1, 1, 1.18]},
        constrained_layout=True,
    )
    speedups = {}

    def plot_cost_panel(axis, thresholds, neb_cost, ivm_cost, xlabel, label, metric):
        both = np.isfinite(neb_cost) & np.isfinite(ivm_cost)
        if not np.any(both):
            axis.set_visible(False)
            return
        last = np.flatnonzero(both)[-1]
        thresholds = thresholds[: last + 1]
        neb_cost = neb_cost[: last + 1]
        ivm_cost = ivm_cost[: last + 1]
        axis.semilogx(thresholds, neb_cost, marker="o", markersize=7.5, linewidth=1.6, color=COLOR_NEB, mfc=COLOR_NEB, mec="black", mew=0.6, label="NEB (k=50)", zorder=3)
        axis.semilogx(thresholds, ivm_cost, marker="s", markersize=7.0, linewidth=1.6, color=COLOR_IVM, mfc=COLOR_IVM, mec="black", mew=0.6, label="IVM continuation", zorder=4)
        axis.invert_xaxis()
        threshold = float(thresholds[-1])
        speedup = float(neb_cost[-1] / ivm_cost[-1])
        speedups[metric] = {
            "threshold": threshold,
            "neb_force_evaluations": float(neb_cost[-1]),
            "ivm_force_evaluations": float(ivm_cost[-1]),
            "speedup": speedup,
        }
        axis.annotate("", xy=(threshold, ivm_cost[-1]), xytext=(threshold, neb_cost[-1]), arrowprops=dict(arrowstyle="<->", color="black", lw=0.9, shrinkA=4, shrinkB=4))
        axis.text(threshold, np.sqrt(neb_cost[-1] * ivm_cost[-1]), f"  {speedup:.1f}×", ha="left", va="center", fontsize=9, fontweight="bold")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Number of force computations")
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value / 1000:.0f}k"))
        axis.legend(loc="upper left")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.4)
        panel_label(axis, label, formal=True)

    plot_cost_panel(axes[0], thresh_lp, neb_cost_lp, ivm_cost_lp, r"$I_{\perp}$", "a", "i_perp")
    plot_cost_panel(axes[1], thresh_de, neb_cost_de, ivm_cost_de, r"$|\Delta E - \Delta E^{*}|$  (kcal mol$^{-1}$)", "b", "barrier_error_kcal_per_mol")

    phi_grid, psi_grid = np.meshgrid(rama["phi_g"], rama["psi_g"], indexing="ij")
    energy_grid = (rama["PES"] - np.nanmin(rama["PES"])) * EV_TO_KCAL
    axis = axes[2]
    contour = axis.contourf(phi_grid, psi_grid, energy_grid, levels=np.linspace(0, 15, 16), cmap="plasma", extend="max")
    axis.contour(phi_grid, psi_grid, energy_grid, levels=np.linspace(0, 15, 8), colors="white", linewidths=0.3, alpha=0.5)
    colorbar = figure.colorbar(contour, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(r"$\Delta E$  (kcal mol$^{-1}$)", fontsize=9)
    initial_phi, initial_psi = path_to_dihedrals(stages[0]["initial_path"])
    neb_phi, neb_psi = path_to_dihedrals(neb["images_seq"][-1])
    ivm_phi, ivm_psi = path_to_dihedrals(stages[-1]["final_path"])
    axis.plot(initial_phi, initial_psi, "--", color=COLOR_INIT, lw=1.6, marker="s", ms=4.5, mfc=COLOR_INIT, mec="black", mew=0.4, label="Initial (IDPP)", zorder=3)
    axis.plot(neb_phi, neb_psi, "-", color=COLOR_NEB, lw=1.8, marker="o", ms=5.0, mfc=COLOR_NEB, mec="black", mew=0.4, label="NEB (k=50)", zorder=4)
    axis.plot(ivm_phi, ivm_psi, "-", color=COLOR_IVM, lw=1.8, marker="^", ms=5.5, mfc=COLOR_IVM, mec="black", mew=0.4, label="IVM (V67)", zorder=5)
    for index in (0, 1):
        minimum = rama["unique"][index]
        axis.scatter(minimum["phi"], minimum["psi"], s=220, marker="*", c="gold", edgecolor="black", zorder=6, linewidth=0.8)
    axis.set_xlim(-180, 180)
    axis.set_ylim(-180, 180)
    axis.set_xticks(range(-180, 181, 90))
    axis.set_yticks(range(-180, 181, 90))
    axis.set_xlabel(r"$\phi$  (deg)")
    axis.set_ylabel(r"$\psi$  (deg)")
    axis.legend(loc="upper right", fontsize=7.5, facecolor="white", edgecolor="black", framealpha=0.9)
    axis.set_aspect("equal")
    panel_label(axis, "c", formal=True)
    formal_outputs = save_figure(figure, args.output_dir / "fig_1x3_cost_to_precision_v67_continuation")

    def ivm_path_at_or_before(target: int) -> np.ndarray:
        indices = np.flatnonzero(ivm_force <= int(target))
        return np.asarray(lineage[indices[-1] if indices.size else 0]["path"])

    def neb_path_at_or_before(target: int) -> np.ndarray:
        full_force = 20 * np.asarray(neb["count_seq"], dtype=np.int64) + 2
        indices = np.flatnonzero(full_force <= int(target))
        return np.asarray(neb["images_seq"][indices[-1] if indices.size else 0])

    panel_config = [
        (0, "Number of Forces: 0"),
        (1000, "Number of Forces: 1k"),
        (10000, "Number of Forces: 10k"),
        (None, "converged"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=200)
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.12, top=0.90, wspace=0.18)
    contour = None
    for axis, (target, title) in zip(axes, panel_config):
        contour = axis.contourf(phi_grid, psi_grid, energy_grid, levels=np.linspace(0, 15, 16), cmap="pink", extend="max")
        axis.contour(phi_grid, psi_grid, energy_grid, levels=np.linspace(0, 15, 8), colors="white", linewidths=0.3, alpha=0.5)
        if target == 0:
            phi, psi = path_to_dihedrals(stages[0]["initial_path"])
            axis.plot(phi, psi, "--", color=COLOR_INIT, lw=1.6, marker="s", ms=4.5, mfc=COLOR_INIT, mec="black", mew=0.4, label="Initial (IDPP)", zorder=3)
        else:
            neb_path = neb["images_seq"][-1] if target is None else neb_path_at_or_before(target)
            ivm_path = stages[-1]["final_path"] if target is None else ivm_path_at_or_before(target)
            neb_phi, neb_psi = path_to_dihedrals(neb_path)
            ivm_phi, ivm_psi = path_to_dihedrals(ivm_path)
            axis.plot(neb_phi, neb_psi, "-", color=COLOR_NEB, lw=1.8, marker="o", ms=4.0, mfc=COLOR_NEB, mec="black", mew=0.4, label="NEB (k=50)", zorder=4)
            axis.plot(ivm_phi, ivm_psi, "-", color=COLOR_IVM, lw=1.8, marker="^", ms=5.0, mfc=COLOR_IVM, mec="black", mew=0.4, label="IVM continuation", zorder=5)
        for index in (0, 1):
            minimum = rama["unique"][index]
            axis.scatter(minimum["phi"], minimum["psi"], s=220, marker="*", c="gold", edgecolor="black", zorder=6, linewidth=0.8)
        axis.set_xlim(-180, 0)
        axis.set_ylim(0, 180)
        axis.set_xticks(range(-180, 1, 60))
        axis.set_yticks(range(0, 181, 60))
        axis.set_xlabel(r"$\phi$  (deg)")
        axis.set_ylabel(r"$\psi$  (deg)")
        axis.set_aspect("equal")
        axis.set_title(title, fontsize=10)
        axis.legend(loc="upper right", fontsize=6.5, facecolor="white", edgecolor="black", framealpha=0.9)
    colorbar = figure.colorbar(contour, ax=axes, fraction=0.03, pad=0.02, location="right")
    colorbar.set_label(r"$\Delta E$  (kcal mol$^{-1}$)", fontsize=9)
    contour_outputs = save_figure(figure, args.output_dir / "fig_1x4_contours_v67_continuation")

    csv_path = args.output_dir / "alanine_v49_to_v67_published_trajectory.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "stage_call", "cumulative_total_force", "i_perp", "segment_length_cv"])
        for point in lineage:
            writer.writerow([point["stage"], point["call"], point["cumulative_total_force"], point["iperp"], point["cv"]])

    audit = {
        "policy": "fully charged V49->V67 continuation; stable published macrostates only",
        "stage_offsets_total_force": stage_offsets,
        "stage_terminal_total_force": [int(np.asarray(stage["counter_vectors"], dtype=np.int64)[-1].sum()) for stage in stages],
        "final_cumulative_total_force": int(total_offset),
        "published_points": len(lineage),
        "final_i_perp": float(ivm_iperp[-1]),
        "maximum_published_spacing_cv": float(np.max(ivm_cv)),
        "delta_e_star_kcal_per_mol": delta_e_star,
        "speedups": speedups,
        "contour_total_force_targets": [0, 1000, 10000, None],
        "inputs": {provenance_name(path, repo): sha256(path) for path in inputs},
        "outputs": {
            "main_metrics": metrics_outputs,
            "loss_descent": loss_outputs,
            "formal_1x3": formal_outputs,
            "contours_1x4": contour_outputs,
            "trajectory_csv": sha256(csv_path),
        },
    }
    audit_path = args.output_dir / "alanine_v67_continuation_figures.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
