#!/usr/bin/env python3
"""Article-style 1x4 continuous NN-IVM path evolution for Q10 Muller-22."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures" / "nn_ivm"
INPUTS = ROOT / "inputs"
RUN = (
    ROOT / "runs" / "nn_ivm_fine_production" /
    "production_fixed_alpha_nn_ivm" / "q10_nn_ivm_fourier_9ba4d6d6c194.npz"
)
SNAPSHOT_FORCES = (0, 500, 1500, 3500)

D = np.array([-200.0, -100.0, -170.0, 15.0])
A = np.array([-1.0, -1.0, -6.5, 0.7])
B = np.array([0.0, 0.0, 11.0, 0.6])
C = np.array([-10.0, -10.0, -6.5, 0.7])
X0 = np.array([1.0, 0.0, -0.5, -1.0])
Y0 = np.array([0.0, 0.5, 1.5, 1.0])


def muller_energy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    value = np.zeros_like(x, dtype=float)
    for di, ai, bi, ci, xi, yi in zip(D, A, B, C, X0, Y0):
        dx = x - xi
        dy = y - yi
        value += di * np.exp(ai * dx * dx + bi * dx * dy + ci * dy * dy)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    q_path = INPUTS / "Q_data.npz"
    with np.load(q_path) as archive:
        q = np.asarray(archive["Q"], dtype=float)
    with np.load(RUN) as archive:
        paths = np.asarray(archive["dense_sample_path"], dtype=float)
        costs = np.asarray(archive["accepted_number_of_forces"], dtype=int)

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.7,
        "font.size": 8.0,
        "axes.titlesize": 8.2,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    x = np.linspace(-1.5, 0.9, 360)
    y = np.linspace(-0.5, 1.9, 360)
    xx, yy = np.meshgrid(x, y)
    energy = np.clip(muller_energy(xx, yy), -175.0, 105.0)
    levels = np.linspace(-175.0, 105.0, 41)
    fig, axes = plt.subplots(
        1, 4, figsize=(7.08, 2.18), sharex=True, sharey=True,
        constrained_layout=False,
    )
    records: list[dict[str, float | int]] = []
    actual_costs: dict[str, int] = {}
    contour = None

    for panel, (axis, requested) in enumerate(zip(axes, SNAPSHOT_FORCES)):
        eligible = np.flatnonzero(costs <= requested)
        index = int(eligible[-1])
        actual = int(costs[index])
        natural = paths[index] @ q.T
        xy = natural[:, :2]
        markers = np.linspace(0, len(xy) - 1, 22).round().astype(int)

        contour = axis.contourf(
            xx, yy, energy, levels=levels, cmap="Greys_r", extend="both"
        )
        axis.contour(
            xx, yy, energy, levels=levels[::4], colors="#686868",
            linewidths=0.25, alpha=0.55,
        )
        color = "#646464" if panel == 0 else "#D1495B"
        linestyle = ":" if panel == 0 else "-"
        axis.plot(xy[:, 0], xy[:, 1], color=color, linestyle=linestyle,
                  linewidth=1.45, zorder=7)
        axis.plot(xy[markers, 0], xy[markers, 1], linestyle="none", marker="o",
                  markersize=2.35, markerfacecolor="white", markeredgecolor=color,
                  markeredgewidth=0.5, zorder=8)
        axis.scatter(xy[0, 0], xy[0, 1], s=20, marker="o", facecolor="white",
                     edgecolor="black", linewidth=0.65, zorder=9)
        axis.scatter(xy[-1, 0], xy[-1, 1], s=26, marker="*", facecolor="black",
                     edgecolor="white", linewidth=0.35, zorder=9)
        axis.set_title("Initial path" if panel == 0 else f"{actual} forces")
        axis.text(0.025, 0.965, f"({chr(97 + panel)})", transform=axis.transAxes,
                  ha="left", va="top", fontweight="bold")
        axis.set_xlim(-1.5, 0.9)
        axis.set_ylim(-0.5, 1.9)
        axis.set_xlabel(r"$z_1$")
        axis.tick_params(direction="in", top=True, right=True, length=2.5)
        actual_costs[str(requested)] = actual
        for sample, point in enumerate(natural):
            records.append({
                "requested_number_of_forces": requested,
                "actual_number_of_forces": actual,
                "sample": sample,
                "z1": float(point[0]),
                "z2": float(point[1]),
                "maximum_inactive_coordinate": float(np.max(np.abs(point[2:]))),
            })

    axes[0].set_ylabel(r"$z_2$")
    handles = [
        Line2D([0], [0], color="#D1495B", marker="o", markerfacecolor="white",
               linewidth=1.45, markersize=3.0, label="NN-IVM"),
        Line2D([0], [0], color="#646464", linestyle=":", marker="o",
               markerfacecolor="white", linewidth=1.25, markersize=3.0,
               label="Initial path"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.47, 1.02),
               ncol=2, frameon=False, handlelength=2.2, columnspacing=1.3)
    colorbar_axis = fig.add_axes([0.925, 0.17, 0.012, 0.68])
    colorbar = fig.colorbar(
        contour, cax=colorbar_axis, ticks=[-150, -100, -50, 0, 50, 100]
    )
    colorbar.set_label(r"$U_M(z_1,z_2)$", labelpad=2)
    fig.subplots_adjust(left=0.065, right=0.91, bottom=0.18, top=0.82, wspace=0.08)

    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "muller22_q10_nn_ivm_path_evolution_1x4"
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600 if suffix == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)

    csv_path = stem.with_name(stem.name + "_source_data.csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    manifest = {
        "figure": "1x4 continuous NN-IVM path evolution",
        "coordinate_system": "natural Muller coordinates z = Qx",
        "snapshot_number_of_forces": list(SNAPSHOT_FORCES),
        "actual_number_of_forces": actual_costs,
        "force_cost_definition": "direct sum of all 20 sample-force streams",
        "continuous_path_points": 5001,
        "displayed_image_markers": 22,
        "saddle_markers": False,
        "alpha": 1.0e-3,
        "tangent": "NEB Henkelman-Jonsson improved tangent",
        "run": str(RUN.relative_to(ROOT)),
        "q_file": str(q_path.relative_to(ROOT)),
        "q_sha256": sha256(q_path),
        "main_tex_updated": False,
        "source_data": str(csv_path.relative_to(ROOT)),
    }
    stem.with_name(stem.name + "_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
