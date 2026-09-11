#!/usr/bin/env python3
"""Plot V49 metric histories and the original Ala 1x4 contour evolution.

The metric-history layout follows Chignolin's
``plot_paper_method_comparison.py::plot_main`` with the horizontal coordinate
equal to the summed force counters over the whole path.  The contour block is
adapted directly from cell 12 of the historical ``figure_ala_N20.ipynb``;
colors, contour levels, crop, markers, and layout are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_v49_formal_ala_1x3 import (
    EV_TO_KCAL,
    _ContourCacheUnpickler,
    configure_original_style,
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


def clean_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.55)


def panel_label(axis, label: str) -> None:
    axis.text(
        0.0,
        1.04,
        label,
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=9,
        va="bottom",
    )


def save_figure(figure, stem: Path) -> dict[str, str]:
    outputs = {}
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 300}),
    ):
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", **kwargs)
        outputs[suffix] = sha256(path)
    plt.close(figure)
    return outputs


def last_total_force_for_calls(run, calls: np.ndarray) -> np.ndarray:
    run_calls = np.asarray(run["calls_per_image"], dtype=int)
    totals = np.asarray(run["counter_vectors"], dtype=np.int64).sum(axis=1)
    result = []
    for call in np.asarray(calls, dtype=int):
        indices = np.flatnonzero(run_calls == call)
        if not indices.size:
            raise RuntimeError(f"Dense checkpoint {call} missing in V49 run")
        result.append(int(totals[indices[-1]]))
    return np.asarray(result, dtype=int)


def path_at_or_before(paths, calls, target_call: int) -> np.ndarray:
    calls = np.asarray(calls, dtype=int)
    indices = np.flatnonzero(calls <= target_call)
    if not indices.size:
        return np.asarray(paths[0])
    return np.asarray(paths[indices[-1]])


def main() -> None:
    study = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v49",
        type=Path,
        default=study
        / "runs/ivm/ivm_ala_v49_production_full_alpha_v9_pathsum_r0p96_"
        "stall_to_r0p95_trust1p5_stop1p5e3_a20p8_central_state_ivm_"
        "cfgd17c08618165_budget220.npz",
    )
    parser.add_argument(
        "--v49-dense",
        type=Path,
        default=study
        / "runs/diagnostics/dense_barrier_v49_production_a20p8_dense25.npz",
    )
    parser.add_argument(
        "--neb",
        type=Path,
        default=study
        / "formal_figure_inputs/ala_0to1_N22_k50_alpha0.0001_ode_results.npz",
    )
    parser.add_argument(
        "--neb-metrics",
        type=Path,
        default=study
        / "formal_figure_inputs/metrics_ala_0to1_N22_k50_alpha0.0001_ode_results.npz",
    )
    parser.add_argument(
        "--rama-cache",
        type=Path,
        default=study / "formal_figure_inputs/ala2_scan_cache.pkl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=study / "figures/formal_v49"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v49 = np.load(args.v49, allow_pickle=False)
    v49_dense = np.load(args.v49_dense, allow_pickle=False)
    neb = np.load(args.neb, allow_pickle=False)
    neb_metrics = np.load(args.neb_metrics, allow_pickle=False)
    with args.rama_cache.open("rb") as handle:
        rama = _ContourCacheUnpickler(handle).load()

    configure_original_style()

    # Chignolin-style main histories: metric on y, whole-path force cost on x.
    v49_force = np.asarray(v49["counter_vectors"], dtype=np.int64).sum(axis=1)
    neb_force = 20 * np.asarray(neb_metrics["count_seq"], dtype=np.int64) + 2
    v49_dense_force = last_total_force_for_calls(v49, v49_dense["calls"])
    neb_barrier = np.asarray(neb_metrics["Delta_Energy"]) * EV_TO_KCAL

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].plot(
        neb_force,
        neb_metrics["loss_perp"],
        color=COLOR_NEB,
        lw=1.35,
        label="NEB (k=50)",
    )
    axes[0].plot(
        v49_force,
        v49["i_perp"],
        color=COLOR_IVM,
        lw=2.2,
        label="IVM (V49)",
        zorder=5,
    )
    axes[1].plot(
        neb_force,
        neb_barrier,
        color=COLOR_NEB,
        lw=1.35,
        marker="o",
        ms=2.0,
        markevery=max(1, len(neb_barrier) // 10),
    )
    axes[1].plot(
        v49_dense_force,
        v49_dense["adaptive_barrier_kcal"],
        color=COLOR_IVM,
        lw=2.2,
        marker="o",
        ms=3.0,
        zorder=5,
    )
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel(r"$I_\perp$")
    axes[1].set_ylabel(
        r"Densified 0 K barrier $\Delta E$ (kcal mol$^{-1}$)"
    )
    maximum_force = int(max(neb_force[-1], v49_force[-1]))
    for index, axis in enumerate(axes):
        axis.set_xlim(0, 1.005 * maximum_force)
        axis.set_xlabel("Number of Force Evaluations, $N_F$")
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].annotate(
        f"V49 stops: $N_F={int(v49_force[-1]):,}$",
        xy=(v49_force[-1], v49["i_perp"][-1]),
        xytext=(12, 12),
        textcoords="offset points",
        color=COLOR_IVM,
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color=COLOR_IVM, lw=0.8),
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.19,
        top=0.90,
        wspace=0.29,
    )
    metrics_stem = args.output_dir / "alanine_v49_main_metrics_total_nf_preview"
    metrics_outputs = save_figure(figure, metrics_stem)

    # Original Ala cell-12 1x4 contour block.  V49's short trajectory replaces
    # the obsolete 10k milestone with measured 500 and 1000 total-force states.
    panel_config = [
        {"total_force": 0, "title": "Number of Forces: 0"},
        {"total_force": 500, "title": "Number of Forces: 500"},
        {"total_force": 1000, "title": "Number of Forces: 1k"},
        {"total_force": None, "title": "converged"},
    ]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=200)
    figure.subplots_adjust(
        left=0.04,
        right=0.96,
        bottom=0.12,
        top=0.90,
        wspace=0.18,
        hspace=0.0,
    )
    phi_grid, psi_grid = np.meshgrid(
        rama["phi_g"], rama["psi_g"], indexing="ij"
    )
    e_kcal_grid = (rama["PES"] - np.nanmin(rama["PES"])) * EV_TO_KCAL
    for axis, config in zip(axes, panel_config):
        cs = axis.contourf(
            phi_grid,
            psi_grid,
            e_kcal_grid,
            levels=np.linspace(0, 15, 16),
            cmap="pink",
            extend="max",
        )
        axis.contour(
            phi_grid,
            psi_grid,
            e_kcal_grid,
            levels=np.linspace(0, 15, 8),
            colors="white",
            linewidths=0.3,
            alpha=0.5,
        )
        total_force = config["total_force"]
        if total_force == 0:
            initial_phi, initial_psi = path_to_dihedrals(v49["initial_path"])
            axis.plot(
                initial_phi,
                initial_psi,
                "--",
                color=COLOR_INIT,
                lw=1.6,
                marker="s",
                ms=4.5,
                mfc=COLOR_INIT,
                mec="black",
                mew=0.4,
                label="Initial (IDPP)",
                zorder=3,
            )
        else:
            if total_force is None:
                neb_path = neb["images_seq"][-1]
                v49_path = v49["final_path"]
            else:
                per_image_call = int(total_force // 20)
                neb_path = path_at_or_before(
                    neb["images_seq"], neb["count_seq"], per_image_call
                )
                v49_path = path_at_or_before(
                    v49["stored_paths"],
                    v49["stored_path_calls"],
                    per_image_call,
                )
            neb_phi, neb_psi = path_to_dihedrals(neb_path)
            v49_phi, v49_psi = path_to_dihedrals(v49_path)
            axis.plot(
                neb_phi,
                neb_psi,
                "-",
                color=COLOR_NEB,
                lw=1.8,
                marker="o",
                ms=4.0,
                mfc=COLOR_NEB,
                mec="black",
                mew=0.4,
                label="NEB (k=50)",
                zorder=4,
            )
            axis.plot(
                v49_phi,
                v49_psi,
                "-",
                color=COLOR_IVM,
                lw=1.8,
                marker="^",
                ms=5.0,
                mfc=COLOR_IVM,
                mec="black",
                mew=0.4,
                label="IVM (V49)",
                zorder=5,
            )
        for index in (0, 1):
            minimum = rama["unique"][index]
            axis.scatter(
                minimum["phi"],
                minimum["psi"],
                s=220,
                marker="*",
                c="gold",
                edgecolor="black",
                zorder=6,
                linewidth=0.8,
            )
        axis.set_xlim(-180, 0)
        axis.set_ylim(0, 180)
        axis.set_xticks(range(-180, 1, 60))
        axis.set_yticks(range(0, 181, 60))
        axis.set_xlabel(r"$\phi$  (deg)")
        axis.set_ylabel(r"$\psi$  (deg)")
        axis.set_aspect("equal")
        axis.set_title(config["title"], fontsize=10)
        axis.legend(
            loc="upper right",
            fontsize=6.5,
            facecolor="white",
            edgecolor="black",
            framealpha=0.9,
        )
    colorbar = figure.colorbar(
        cs, ax=axes, fraction=0.03, pad=0.02, location="right"
    )
    colorbar.set_label(r"$\Delta E$  (kcal mol$^{-1}$)", fontsize=9)
    contour_stem = args.output_dir / "fig_1x4_contours_v49_preview"
    contour_outputs = save_figure(figure, contour_stem)

    audit = {
        "metric_plot_source": (
            "legacy:code_on_server/chignolin/src/"
            "plot_paper_method_comparison.py::plot_main"
        ),
        "contour_plot_source": (
            "legacy:code_on_server/IVM_on_server/scripts/"
            "figure_ala_N20.ipynb, cell 12"
        ),
        "v49_final_total_path_force_evaluations": int(v49_force[-1]),
        "neb_final_total_path_force_evaluations": int(neb_force[-1]),
        "contour_intermediate_total_force_targets": [500, 1000],
        "inputs": {
            str(path): sha256(path)
            for path in (
                args.v49,
                args.v49_dense,
                args.neb,
                args.neb_metrics,
                args.rama_cache,
            )
        },
        "outputs": {
            "main_metrics": metrics_outputs,
            "contours": contour_outputs,
        },
    }
    audit_path = args.output_dir / "alanine_v49_force_history_and_contours.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
