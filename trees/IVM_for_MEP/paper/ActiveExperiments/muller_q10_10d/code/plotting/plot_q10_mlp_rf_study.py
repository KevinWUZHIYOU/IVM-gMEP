#!/usr/bin/env python3
"""5001-point comparison of spline, full-MLP, and random-feature NN-IVM."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from q10_ivm_core import REFERENCE_BARRIER, improved_tangents_arbitrary
from q10_potential import Q, energy_gradient, energy_values


ROOT = Path(__file__).resolve().parents[1]
DENSE_COUNT = 5001
SOURCES = {
    "Full MLP NN-IVM": (
        "runs/nnivm_full_mlp_production_audit/"
        "q10_nn_ivm_mlp_lbfgs_d2d2b7a8139f.npz"
    ),
    "RF NN-IVM": (
        "runs/nnivm_rbf_production_audit/"
        "q10_nn_ivm_mlp_lbfgs_2f62d1475247.npz"
    ),
}


def dense_metrics(path: np.ndarray) -> tuple[float, float, float]:
    path = np.asarray(path, dtype=np.float64)
    spline = CubicSpline(np.linspace(0.0, 1.0, len(path)), path, axis=0)
    points = spline(np.linspace(0.0, 1.0, DENSE_COUNT))
    energies = energy_values(points)
    tangent = improved_tangents_arbitrary(points, energies)
    gradients = np.asarray([energy_gradient(point)[1] for point in points[1:-1]])
    normal = gradients - np.sum(gradients * tangent, axis=1, keepdims=True) * tangent
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    weights = 0.5 * (segments[:-1] + segments[1:])
    i_perp = float(np.sum(weights * np.sum(normal * normal, axis=1)))
    barrier = float(abs(np.max(energies) - energies[0] - REFERENCE_BARRIER))
    inactive = float(np.max(np.abs(points @ Q.T)[:, 2:]))
    return i_perp, barrier, inactive


def process_neural(label: str, source: str) -> tuple[Path, dict]:
    archive = np.load(ROOT / source)
    paths = np.asarray(archive["dense_path_history"], dtype=np.float64)
    cost = np.asarray(archive["number_of_forces"], dtype=np.int64)
    metric = np.asarray([dense_metrics(path) for path in paths])
    output_directory = ROOT / "runs" / "mlp_rf_dense_postprocessed"
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / (
        "full_mlp.npz" if label.startswith("Full") else "random_feature.npz"
    )
    np.savez_compressed(
        output,
        method=np.asarray(label),
        source=np.asarray(source),
        number_of_forces=cost,
        dense_i_perp=metric[:, 0],
        dense_barrier_error=metric[:, 1],
        inactive_coordinate_max=metric[:, 2],
        dense_count=np.asarray(DENSE_COUNT),
        tangent=np.asarray("NEB Henkelman-Jonsson improved tangent"),
    )
    minimum = int(np.argmin(metric[:, 0]))
    barrier_gate = np.flatnonzero(metric[:, 1] <= 1.0e-4)
    summary = {
        "method": label,
        "source": source,
        "output": str(output.relative_to(ROOT)),
        "final_number_of_forces": int(cost[-1]),
        "final_dense_i_perp": float(metric[-1, 0]),
        "final_dense_barrier_error": float(metric[-1, 1]),
        "minimum_dense_i_perp": float(metric[minimum, 0]),
        "minimum_i_perp_number_of_forces": int(cost[minimum]),
        "barrier_error_at_minimum_i_perp": float(metric[minimum, 1]),
        "best_i_perp_with_barrier_below_1e-4": (
            float(np.min(metric[barrier_gate, 0])) if len(barrier_gate) else None
        ),
        "maximum_inactive_coordinate": float(np.max(metric[:, 2])),
    }
    return output, summary


def main() -> None:
    files: dict[str, Path] = {
        "Spline NN-IVM": ROOT / "runs/dense_postprocessed_improved_5001/nn_ivm.npz",
        "IVM": ROOT / "runs/dense_postprocessed_improved_5001/ivm.npz",
    }
    summaries = []
    for label, source in SOURCES.items():
        files[label], summary = process_neural(label, source)
        summaries.append(summary)

    for label in ("Spline NN-IVM", "IVM"):
        archive = np.load(files[label])
        cost = np.asarray(archive["number_of_forces"])
        i_perp = np.asarray(archive["dense_i_perp"])
        barrier = np.asarray(archive["dense_delta_e_error"])
        keep = cost <= 5000
        summaries.append({
            "method": label,
            "output": str(files[label].relative_to(ROOT)),
            "final_number_of_forces": int(cost[keep][-1]),
            "final_dense_i_perp": float(i_perp[keep][-1]),
            "final_dense_barrier_error": float(barrier[keep][-1]),
            "minimum_dense_i_perp": float(np.min(i_perp[keep])),
        })

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "font.size": 10.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    styles = {
        "Full MLP NN-IVM": dict(color="#2A9D8F", linestyle="-.", marker="^", zorder=8),
        "RF NN-IVM": dict(color="#E9A23B", linestyle="--", marker="s", zorder=10),
        "Spline NN-IVM": dict(color="#D1495B", linestyle="-", marker="o", zorder=16),
        "IVM": dict(color="#0072B2", linestyle="-", marker="o", zorder=20),
    }
    plot_order = ("Full MLP NN-IVM", "RF NN-IVM", "Spline NN-IVM", "IVM")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.75))
    handles = {}
    for label in plot_order:
        archive = np.load(files[label])
        x = np.asarray(archive["number_of_forces"], dtype=float)
        i_perp = np.maximum(np.asarray(archive["dense_i_perp"], dtype=float), 1.0e-30)
        barrier_key = (
            "dense_delta_e_error" if "dense_delta_e_error" in archive.files
            else "dense_barrier_error"
        )
        barrier = np.maximum(np.asarray(archive[barrier_key], dtype=float), 1.0e-16)
        keep = x <= 5000
        markevery = max(1, int(np.count_nonzero(keep) / 9))
        line, = axes[0].plot(
            x[keep], i_perp[keep], label=label, linewidth=2.0,
            markersize=3.0, markevery=markevery, **styles[label]
        )
        axes[1].plot(
            x[keep], barrier[keep], label=label, linewidth=2.0,
            markersize=3.0, markevery=markevery, **styles[label]
        )
        handles[label] = line
    for axis in axes:
        axis.set_xlim(0.0, 5000.0)
        axis.set_yscale("log")
        axis.set_xlabel("Number of Forces")
        axis.grid(True, which="major", alpha=0.22, linewidth=0.55)
        axis.tick_params(direction="in", top=True, right=True)
    axes[0].set_ylabel(r"Densified $I_{\perp}$")
    axes[1].set_ylabel(r"Densified $|\Delta E-\Delta E^*|$")
    legend_order = ("IVM", "Spline NN-IVM", "RF NN-IVM", "Full MLP NN-IVM")
    fig.legend(
        [handles[label] for label in legend_order], legend_order,
        loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.985),
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.78, wspace=0.27)
    output_directory = ROOT / "figures" / "nn_ivm"
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "muller22_q10_mlp_random_feature_study"
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            stem.with_suffix(f".{suffix}"),
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    report = {
        "case": "fixed-Q unperturbed 10D Muller-Brown, 22 images",
        "alpha": 1.0e-3,
        "sample_count": 20,
        "dense_count": DENSE_COUNT,
        "tangent": "NEB Henkelman-Jonsson improved tangent",
        "force_cost": "direct sum over all 20 sample streams",
        "main_tex_updated": False,
        "runs": summaries,
        "figure": str(stem.with_suffix(".pdf").relative_to(ROOT)),
    }
    (ROOT / "reports" / "q10_mlp_random_feature_study.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
