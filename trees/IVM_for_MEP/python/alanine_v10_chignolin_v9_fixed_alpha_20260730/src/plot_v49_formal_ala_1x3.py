#!/usr/bin/env python3
"""Rebuild the formal Ala 1x3 figure with the V49 IVM trajectory.

The plotting block is the original final cell (cell 8) of
``code_on_server/IVM_on_server/scripts/figure_ala_N20.ipynb``.  Only the data
adapter has changed: the historical IVM arrays are replaced by V49's audited
``i_perp``, densified barrier, and final Cartesian path.  The historical
best-tuned NEB k=50 baseline, threshold grid, force-count convention, panel
layout, colors, and Ramachandran background are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


EV_TO_KCAL = 23.0605
FORCE_EVAL_FACTOR = 20
PHI_IDX = (4, 6, 8, 14)
PSI_IDX = (6, 8, 14, 16)


class _IgnoredAseObject:
    """Placeholder for ASE objects embedded in the historical contour cache."""

    def __new__(cls, *_args, **_kwargs):
        return super().__new__(cls)

    def __setstate__(self, state):
        self.state = state


class _ContourCacheUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("ase."):
            return _IgnoredAseObject
        if module.startswith("numpy._core"):
            # NumPy 2 writes private-module pickle names that NumPy 1.x does
            # not expose.  The cached Ramachandran arrays use compatible
            # ndarray/scalar payloads, so map only the module namespace.
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


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


def dihedral_degrees(x_flat: np.ndarray, indices: tuple[int, ...]) -> float:
    """ASE-compatible Cartesian dihedral, wrapped to [-180, 180)."""
    points = np.asarray(x_flat, dtype=float).reshape(22, 3)[list(indices)]
    b0 = -(points[1] - points[0])
    b1 = points[2] - points[1]
    b2 = points[3] - points[2]
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    angle = np.degrees(
        np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))
    )
    return float(((angle + 180.0) % 360.0) - 180.0)


def path_to_dihedrals(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phis = [dihedral_degrees(x, PHI_IDX) for x in path]
    psis = [dihedral_degrees(x, PSI_IDX) for x in path]
    return np.asarray(phis), np.asarray(psis)


# Original helper, unchanged.
def cost_to_reach(counts, values, thresholds, min_count=1):
    counts = np.asarray(counts, float)
    values = np.asarray(values, float)
    m = np.isfinite(counts) & np.isfinite(values) & (counts >= min_count)
    counts, values = counts[m], values[m]
    order = np.argsort(counts)
    counts, values = counts[order], values[order]
    running_min = np.minimum.accumulate(values)

    out = np.full_like(thresholds, np.nan, dtype=float)
    for i, tau in enumerate(thresholds):
        idx = np.where(running_min <= tau)[0]
        if idx.size > 0:
            out[i] = counts[idx[0]]
    return out


# Original helper, unchanged.
def log_thresholds_uniform(lo, hi, points_per_decade=3):
    n_decades = int(np.ceil(np.log10(hi)) - np.floor(np.log10(lo)))
    n_total = n_decades * points_per_decade + 1
    candidates = np.logspace(
        np.floor(np.log10(lo)), np.ceil(np.log10(hi)), n_total
    )
    mandatory = [
        10**d
        for d in range(
            int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1
        )
    ]
    all_vals = np.unique(np.concatenate([candidates, mandatory]))
    return np.array([v for v in all_vals if lo <= v <= hi])


def configure_original_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
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
    parser.add_argument(
        "--stem", default="fig_1x3_cost_to_precision_v49_preview"
    )
    args = parser.parse_args()

    inputs = [
        args.v49,
        args.v49_dense,
        args.neb,
        args.neb_metrics,
        args.rama_cache,
    ]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)

    ivm_run = np.load(args.v49, allow_pickle=False)
    ivm_dense = np.load(args.v49_dense, allow_pickle=False)
    neb_run = np.load(args.neb, allow_pickle=False)
    neb_metrics = np.load(args.neb_metrics, allow_pickle=False)
    # The cache stores ASE Atoms objects beside the numeric minima.  Plotting
    # needs only phi/psi/PES, so ASE is deliberately not a runtime dependency.
    with args.rama_cache.open("rb") as handle:
        rama_cache = _ContourCacheUnpickler(handle).load()

    # Data adapter: V49 replaces only the historical IVM series/path.
    neb_c_lp = neb_metrics["count_seq"]
    neb_y_lp = neb_metrics["loss_perp"]
    ivm_c_lp = ivm_run["calls_per_image"]
    ivm_y_lp = ivm_run["i_perp"]

    neb_c_de = neb_metrics["count_seq"]
    neb_de_kcal = neb_metrics["Delta_Energy"] * EV_TO_KCAL
    ivm_c_de = ivm_dense["calls"]
    ivm_de_kcal = ivm_dense["adaptive_barrier_kcal"]
    delta_e_star = np.nanmin(np.concatenate([neb_de_kcal, ivm_de_kcal]))
    neb_err_de = np.abs(neb_de_kcal - delta_e_star)
    ivm_err_de = np.abs(ivm_de_kcal - delta_e_star)

    thresh_lp = log_thresholds_uniform(2e-4, 1e0, 4)[::-1]
    thresh_de = log_thresholds_uniform(1e-2, 1e0, 8)[::-1]
    neb_cost_lp = cost_to_reach(neb_c_lp, neb_y_lp, thresh_lp)
    ivm_cost_lp = cost_to_reach(ivm_c_lp, ivm_y_lp, thresh_lp)
    neb_cost_de = cost_to_reach(neb_c_de, neb_err_de, thresh_de)
    ivm_cost_de = cost_to_reach(ivm_c_de, ivm_err_de, thresh_de)

    configure_original_style()
    color_neb = "#D55E00"
    color_ivm = "#0072B2"
    color_init = "#888888"
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.0, 4.0),
        dpi=200,
        gridspec_kw={"width_ratios": [1, 1, 1.18]},
        constrained_layout=True,
    )

    speedups = {}

    # Original plotting helper; labels retain the formal-figure convention.
    def plot_cost_panel(
        ax, thresholds, neb_cost, ivm_cost, xlabel, panel_letter, metric
    ):
        neb_cost = neb_cost * FORCE_EVAL_FACTOR
        ivm_cost = ivm_cost * FORCE_EVAL_FACTOR
        both_ok = np.isfinite(neb_cost) & np.isfinite(ivm_cost)
        if not both_ok.any():
            ax.set_visible(False)
            return
        last_valid = np.where(both_ok)[0][-1]
        thresholds = thresholds[: last_valid + 1]
        neb_cost = neb_cost[: last_valid + 1]
        ivm_cost = ivm_cost[: last_valid + 1]

        ax.semilogx(
            thresholds,
            neb_cost,
            marker="o",
            markersize=7.5,
            linewidth=1.6,
            color=color_neb,
            mfc=color_neb,
            mec="black",
            mew=0.6,
            label="NEB (k=50)",
            zorder=3,
        )
        ax.semilogx(
            thresholds,
            ivm_cost,
            marker="s",
            markersize=7.0,
            linewidth=1.6,
            color=color_ivm,
            mfc=color_ivm,
            mec="black",
            mew=0.6,
            label="IVM (V49)",
            zorder=4,
        )
        ax.invert_xaxis()

        tau_s = thresholds[-1]
        c_neb = neb_cost[-1]
        c_ivm = ivm_cost[-1]
        speedup = c_neb / c_ivm
        speedups[metric] = {
            "threshold": float(tau_s),
            "neb_force_evaluations": float(c_neb),
            "ivm_force_evaluations": float(c_ivm),
            "speedup": float(speedup),
        }
        ax.annotate(
            "",
            xy=(tau_s, c_ivm),
            xytext=(tau_s, c_neb),
            arrowprops=dict(
                arrowstyle="<->", color="black", lw=0.9, shrinkA=4, shrinkB=4
            ),
        )
        ax.text(
            tau_s,
            np.sqrt(c_neb * c_ivm),
            f"  {speedup:.1f}×",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of force computations")
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, _position: f"{x / 1000:.0f}k")
        )
        ax.legend(loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.4)
        ax.text(
            -0.20,
            1.02,
            panel_letter,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
        )

    plot_cost_panel(
        axes[0], thresh_lp, neb_cost_lp, ivm_cost_lp, r"$I_{\perp}$", "a", "i_perp"
    )
    plot_cost_panel(
        axes[1],
        thresh_de,
        neb_cost_de,
        ivm_cost_de,
        r"$|\Delta E - \Delta E^{*}|$  (kcal mol$^{-1}$)",
        "b",
        "barrier_error_kcal_per_mol",
    )

    # Original Ramachandran panel, with V49 supplying only the blue path.
    ax = axes[2]
    phi_g = rama_cache["phi_g"]
    psi_g = rama_cache["psi_g"]
    e_kcal_grid = (rama_cache["PES"] - np.nanmin(rama_cache["PES"])) * EV_TO_KCAL
    phi_grid, psi_grid = np.meshgrid(phi_g, psi_g, indexing="ij")
    cs = ax.contourf(
        phi_grid,
        psi_grid,
        e_kcal_grid,
        levels=np.linspace(0, 15, 16),
        cmap="plasma",
        extend="max",
    )
    ax.contour(
        phi_grid,
        psi_grid,
        e_kcal_grid,
        levels=np.linspace(0, 15, 8),
        colors="white",
        linewidths=0.3,
        alpha=0.5,
    )
    cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta E$  (kcal mol$^{-1}$)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)

    init_phi, init_psi = path_to_dihedrals(neb_run["images_seq"][0])
    neb_phi, neb_psi = path_to_dihedrals(neb_run["images_seq"][-1])
    ivm_phi, ivm_psi = path_to_dihedrals(ivm_run["final_path"])
    ax.plot(
        init_phi,
        init_psi,
        "--",
        color=color_init,
        lw=1.6,
        marker="s",
        ms=4.5,
        mfc=color_init,
        mec="black",
        mew=0.4,
        label="Initial (IDPP)",
        zorder=3,
    )
    ax.plot(
        neb_phi,
        neb_psi,
        "-",
        color=color_neb,
        lw=1.8,
        marker="o",
        ms=5.0,
        mfc=color_neb,
        mec="black",
        mew=0.4,
        label="NEB (k=50)",
        zorder=4,
    )
    ax.plot(
        ivm_phi,
        ivm_psi,
        "-",
        color=color_ivm,
        lw=1.8,
        marker="^",
        ms=5.5,
        mfc=color_ivm,
        mec="black",
        mew=0.4,
        label="IVM (V49)",
        zorder=5,
    )
    for index in (0, 1):
        minimum = rama_cache["unique"][index]
        ax.scatter(
            minimum["phi"],
            minimum["psi"],
            s=220,
            marker="*",
            c="gold",
            edgecolor="black",
            zorder=6,
            linewidth=0.8,
        )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(range(-180, 181, 90))
    ax.set_yticks(range(-180, 181, 90))
    ax.set_xlabel(r"$\phi$  (deg)")
    ax.set_ylabel(r"$\psi$  (deg)")
    ax.legend(
        loc="upper right",
        fontsize=7.5,
        facecolor="white",
        edgecolor="black",
        framealpha=0.9,
    )
    ax.set_aspect("equal")
    ax.text(
        -0.20,
        1.02,
        "c",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.stem
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    initial_max_abs_difference = float(
        np.max(np.abs(neb_run["images_seq"][0] - ivm_run["initial_path"]))
    )
    audit = {
        "plot_source": (
            "legacy:code_on_server/IVM_on_server/scripts/"
            "figure_ala_N20.ipynb, final plotting cell 8"
        ),
        "plot_policy": "original formal 1x3 code and thresholds; IVM replaced by V49",
        "force_evaluation_factor": FORCE_EVAL_FACTOR,
        "delta_e_star_kcal_per_mol": float(delta_e_star),
        "speedups": speedups,
        "initial_path_max_abs_difference_angstrom": initial_max_abs_difference,
        "inputs": {
            provenance_name(path, repo): sha256(path)
            for path in inputs
        },
        "outputs": {
            extension: sha256(stem.with_suffix(extension))
            for extension in (".pdf", ".png", ".svg")
        },
    }
    with stem.with_suffix(".json").open("w") as handle:
        json.dump(audit, handle, indent=2)
        handle.write("\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
