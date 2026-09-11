#!/usr/bin/env python3
"""Article-style 1x4 path evolution for the fixed-Q 10D Muller-22 case."""

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
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures" / "path_evolution"
INPUTS = ROOT / "inputs"
SNAPSHOT_FORCES = (0.0, 5.0, 20.0, 100.0)

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


def resolve_run(reference: str) -> Path:
    path = Path(reference)
    return path if path.is_absolute() else ROOT / path


def latest_path(
    paths: np.ndarray, calls: np.ndarray, requested: float
) -> tuple[np.ndarray, float]:
    if requested <= 0.0:
        return paths[0], 0.0
    eligible = np.flatnonzero(calls <= requested + 1.0e-12)
    index = int(eligible[-1]) if len(eligible) else 0
    return paths[index], float(calls[index])


def natural_xy(path: np.ndarray, q: np.ndarray) -> np.ndarray:
    # Column convention is z = Q @ x; paths are stored row-wise.
    return (np.asarray(path, dtype=float) @ q.T)[:, :2]


def main() -> None:
    selection = json.loads((REPORTS / "q10_final_selection.json").read_text())
    baselines = json.loads((REPORTS / "baseline_final_selection.json").read_text())
    ivm_path = resolve_run(selection["selected_ivm"]["output"])
    ode_path = resolve_run(baselines["selected"]["ode_neb"]["run"])
    q_path = INPUTS / "Q_data.npz"

    with np.load(q_path) as archive:
        q = np.asarray(archive["Q"], dtype=float)
    with np.load(ivm_path) as archive:
        ivm_paths = np.asarray(archive["path_history"], dtype=float)
        ivm_calls = np.asarray(archive["mean_calls_per_image"], dtype=float)
        initial = (
            np.asarray(archive["transformed_initial_path"], dtype=float)
            if "transformed_initial_path" in archive
            else ivm_paths[0]
        )
    with np.load(ode_path) as archive:
        ode_paths = np.asarray(archive["path_history"], dtype=float)
        ode_calls = np.asarray(archive["mean_calls_per_image"], dtype=float)

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
    records: list[dict[str, float | int | str]] = []
    actual_calls: dict[str, dict[str, float]] = {}
    initial_xy = natural_xy(initial, q)

    contour = None
    for panel, (axis, requested) in enumerate(zip(axes, SNAPSHOT_FORCES)):
        contour = axis.contourf(
            xx, yy, energy, levels=levels, cmap="Greys_r", extend="both"
        )
        axis.contour(
            xx, yy, energy, levels=levels[::4], colors="#686868",
            linewidths=0.25, alpha=0.55,
        )
        if panel == 0:
            axis.plot(
                initial_xy[:, 0], initial_xy[:, 1], color="#646464",
                linestyle=":", linewidth=1.25, marker="o", markersize=2.4,
                markerfacecolor="white", markeredgewidth=0.55, zorder=6,
            )
            axis.set_title("Initial path")
            actual_calls["0"] = {"IVM": 0.0, "ODE-NEB": 0.0}
            for image, point in enumerate(initial_xy):
                records.append({
                    "requested_forces": 0.0, "actual_forces": 0.0,
                    "method": "Initial", "image": image,
                    "z1": float(point[0]), "z2": float(point[1]),
                })
        else:
            ivm_snapshot, ivm_actual = latest_path(ivm_paths, ivm_calls, requested)
            ode_snapshot, ode_actual = latest_path(ode_paths, ode_calls, requested)
            ivm_xy = natural_xy(ivm_snapshot, q)
            ode_xy = natural_xy(ode_snapshot, q)
            axis.plot(
                ivm_xy[:, 0], ivm_xy[:, 1], color="#0072B2",
                linewidth=1.45, marker="o", markersize=2.35,
                markeredgewidth=0.35, zorder=7,
            )
            axis.plot(
                ode_xy[:, 0], ode_xy[:, 1], color="#D55E00",
                linestyle="--", linewidth=1.25, marker="s", markersize=2.2,
                markerfacecolor="white", markeredgewidth=0.5, zorder=6,
            )
            axis.set_title(f"{int(requested)} forces / image")
            actual_calls[str(int(requested))] = {
                "IVM": ivm_actual, "ODE-NEB": ode_actual,
            }
            for method, actual, points in (
                ("IVM", ivm_actual, ivm_xy),
                ("ODE-NEB", ode_actual, ode_xy),
            ):
                for image, point in enumerate(points):
                    records.append({
                        "requested_forces": requested,
                        "actual_forces": actual,
                        "method": method,
                        "image": image,
                        "z1": float(point[0]), "z2": float(point[1]),
                    })

        axis.scatter(
            initial_xy[0, 0], initial_xy[0, 1], s=20, marker="o",
            facecolor="white", edgecolor="black", linewidth=0.65, zorder=9,
        )
        axis.scatter(
            initial_xy[-1, 0], initial_xy[-1, 1], s=26, marker="*",
            facecolor="black", edgecolor="white", linewidth=0.35, zorder=9,
        )
        axis.text(
            0.025, 0.965, f"({chr(97 + panel)})", transform=axis.transAxes,
            ha="left", va="top", fontweight="bold",
        )
        axis.set_xlim(-1.5, 0.9)
        axis.set_ylim(-0.5, 1.9)
        axis.set_xlabel(r"$z_1$")
        axis.tick_params(direction="in", top=True, right=True, length=2.5)
    axes[0].set_ylabel(r"$z_2$")

    handles = [
        Line2D([0], [0], color="#0072B2", marker="o", linewidth=1.45,
               markersize=3.0, label="IVM"),
        Line2D([0], [0], color="#D55E00", linestyle="--", marker="s",
               markerfacecolor="white", linewidth=1.25, markersize=3.0,
               label="ODE-NEB"),
        Line2D([0], [0], color="#646464", linestyle=":", marker="o",
               markerfacecolor="white", linewidth=1.25, markersize=3.0,
               label="Initial path"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.47, 1.02),
        ncol=3, frameon=False, handlelength=2.2, columnspacing=1.3,
    )
    colorbar_axis = fig.add_axes([0.925, 0.17, 0.012, 0.68])
    colorbar = fig.colorbar(
        contour, cax=colorbar_axis, ticks=[-150, -100, -50, 0, 50, 100]
    )
    colorbar.set_label(r"$U_M(z_1,z_2)$", labelpad=2)
    fig.subplots_adjust(
        left=0.065, right=0.91, bottom=0.18, top=0.82, wspace=0.08
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "q10_muller_path_evolution_1x4"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    csv_path = stem.with_name(stem.name + "_source_data.csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    manifest = {
        "figure": "1x4 matched-cost path evolution",
        "coordinate_system": "natural Muller coordinates z = Qx",
        "snapshot_force_budgets": list(SNAPSHOT_FORCES),
        "actual_force_counts": actual_calls,
        "force_cost_definition": "mean force calls per movable image",
        "ivm_run": str(ivm_path.relative_to(ROOT)),
        "ode_neb_run": str(ode_path.relative_to(ROOT)),
        "q_file": str(q_path.relative_to(ROOT)),
        "q_sha256": sha256(q_path),
        "outputs": [
            str(stem.with_suffix(ext).relative_to(ROOT))
            for ext in (".pdf", ".svg", ".png")
        ],
        "source_data": str(csv_path.relative_to(ROOT)),
    }
    manifest_path = stem.with_name(stem.name + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
