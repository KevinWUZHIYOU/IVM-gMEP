#!/usr/bin/env python3
"""Build the three data-backed LEPS figures used by paper0819."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
LEPS = HERE.parents[2]
PROJECT = LEPS.parents[1]
PAPER = PROJECT / "paper0819"
PAPER_FIGURES = PAPER / "Figures/LEPS"
REPORTS = ROOT / "reports"
IR_RESULT = LEPS / "leps_irivm_sync1_refined_grid_20260830/runs/3c62d93a051be116.npz"
ODE_RESULT = LEPS / "runs/neb/neb_production_k10_cfgfe491f5a6c6f_budget5000.npz"
NN_RESULT = LEPS / "runs/continuous_v3/nn_ivm_adaptive_al_a100_j015_b500.npz"
GMAM_RESULT = LEPS / "runs/continuous_v2/gmam_autodiff20_production_lr001_b500.npz"
METRICS_FIGURE = PROJECT / "Figures/LEPS/leps_all_methods_metrics.pdf"
METRICS_REGISTRY = LEPS / "leps_action_baselines_20260910/reports/leps_all_methods_metrics_provenance.json"
METRICS_CSV = LEPS / "leps_action_baselines_20260910/reports/leps_all_methods_metrics_source.csv"
LEPS_SRC = LEPS / "src"
sys.path.insert(0, str(LEPS_SRC))

from analyze_continuous_study import dense_linear_trajectory  # noqa: E402
from leps_core import REACTANT, flat_to_physical, leps_energy, reference_saddle  # noqa: E402


COLORS = {
    "IVM": "#0072B2",
    "ODE-NEB": "#D55E00",
    "NN-IVM": "#009E73",
    "gMAM (autodiff)": "#CC79A7",
}
STYLES = {"IVM": "-", "ODE-NEB": "--", "NN-IVM": "-.", "gMAM (autodiff)": ":"}
PATH_SNAPSHOTS = (200, 1500, "final")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        result = {key: saved[key] for key in saved.files}
    result["_path"] = str(path)
    return result


def configure_style() -> None:
    plt.rcParams.update({
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
    })


def latest_at_or_below(nof: np.ndarray, paths: np.ndarray, budget: int):
    index = int(np.searchsorted(nof, budget, side="right") - 1)
    if index < 0:
        raise ValueError(f"no path available at budget {budget}")
    return int(nof[index]), np.asarray(paths[index], dtype=float)


def path_at_snapshot(nof: np.ndarray, paths: np.ndarray, snapshot: int | str):
    if snapshot == "final":
        return int(nof[-1]), np.asarray(paths[-1], dtype=float)
    return latest_at_or_below(nof, paths, int(snapshot))


def path_figure(ivm: dict, ode: dict, saddle: dict) -> tuple[Path, list[dict]]:
    ivm_nof = np.asarray(ivm["article_nf"], dtype=int)
    ivm_paths = np.asarray(ivm["path_history"], dtype=float)
    ode_nof = np.asarray(ode["total_internal_calls"], dtype=int)
    ode_paths = np.asarray(ode["path"], dtype=float)
    np.testing.assert_array_equal(ivm_paths[0], ode_paths[0])

    coordinate = np.linspace(0.4, 8.2, 440)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate)
    relative = leps_energy(x_grid, y_grid) - float(leps_energy(REACTANT[0], REACTANT[1]))
    levels = np.linspace(0.0, 6.5, 41)
    clipped = np.clip(relative, levels[0], levels[-1])
    fig, axes = plt.subplots(1, 4, figsize=(7.08, 2.18), sharex=True, sharey=True)
    rows, contour = [], None
    for panel, axis in enumerate(axes):
        contour = axis.contourf(x_grid, y_grid, clipped, levels=levels, cmap="Greys_r",
                                norm=BoundaryNorm(levels, 256), extend="max")
        axis.contour(x_grid, y_grid, clipped, levels=levels[::2], colors="#6F6F6F",
                     linewidths=0.24, alpha=0.55)
        if panel == 0:
            physical = flat_to_physical(ivm_paths[0])
            axis.plot(physical[:, 0], physical[:, 1], color="#777777", linestyle=":",
                      linewidth=1.25, marker="o", markersize=1.9,
                      label="Common initial path", zorder=5)
            for image, point in enumerate(physical):
                rows.append({"panel": "a", "budget_nof": "", "method": "Common initial path",
                             "actual_nof": int(ivm_nof[0]), "image": image,
                             "r_AB": float(point[0]), "r_BC": float(point[1])})
            title = "Initial path"
        else:
            snapshot = PATH_SNAPSHOTS[panel - 1]
            title = "Final paths" if snapshot == "final" else rf"NoF $\leq {snapshot}$"
            actual_nof = {}
            for name, nof, paths in (("IVM", ivm_nof, ivm_paths), ("ODE-NEB", ode_nof, ode_paths)):
                actual, path = path_at_snapshot(nof, paths, snapshot)
                actual_nof[name] = actual
                physical = flat_to_physical(path)
                axis.plot(physical[:, 0], physical[:, 1], color=COLORS[name],
                          linestyle=STYLES[name], linewidth=1.55 if name == "IVM" else 1.45,
                          marker="o" if name == "IVM" else "s",
                          markersize=1.9 if name == "IVM" else 1.7,
                          label=name, zorder=6 if name == "IVM" else 5)
                for image, point in enumerate(physical):
                    rows.append({"panel": chr(ord("a") + panel), "budget_nof": snapshot,
                             "method": name, "actual_nof": actual, "image": image,
                             "r_AB": float(point[0]), "r_BC": float(point[1])})
            if snapshot == "final":
                axis.text(
                    0.97,
                    0.96,
                    f"IVM: $\\approx 1500$ NoF\nODE-NEB: {actual_nof['ODE-NEB']} NoF",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=5.4,
                    bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none", "pad": 1.0},
                    zorder=9,
                )
        axis.scatter(saddle["point"][0], saddle["point"][1], marker="*", s=25,
                     color="#CC79A7", edgecolor="#333333", linewidth=0.35, zorder=8)
        axis.set_xlim(0.4, 8.2)
        axis.set_ylim(0.4, 8.2)
        axis.set_aspect("equal")
        axis.set_box_aspect(1)
        axis.set_title(title)
        axis.set_xlabel(r"$r_{\mathrm{AB}}$")
        if panel == 0:
            axis.set_ylabel(r"$r_{\mathrm{BC}}$")
        axis.text(-0.13, 1.06, chr(ord("a") + panel), transform=axis.transAxes,
                  fontsize=8.5, fontweight="bold", va="top")
    initial_handles, initial_labels = axes[0].get_legend_handles_labels()
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(initial_handles + handles, initial_labels + labels, loc="upper center",
               bbox_to_anchor=(0.48, 1.02), ncol=3, frameon=False)
    colorbar_axis = fig.add_axes((0.905, 0.20, 0.012, 0.58))
    colorbar = fig.colorbar(contour, cax=colorbar_axis)
    colorbar.set_label(r"$V-V_{\mathrm{reactant}}$ (eV)")
    fig.subplots_adjust(left=0.065, right=0.885, bottom=0.19, top=0.80, wspace=0.12)
    fig.canvas.draw()
    ratios = [float(axis.get_window_extent().width / axis.get_window_extent().height) for axis in axes]
    np.testing.assert_allclose(ratios, 1.0, atol=1e-12, rtol=0)
    output = PAPER_FIGURES / "leps_path_evolution_1x4.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, metadata={"Title": "LEPS equal-total-NoF path evolution",
                                           "CreationDate": None, "ModDate": None})
    plt.close(fig)
    return output, rows


def continuous_figure(ivm: dict, ode: dict, nn: dict, gmam: dict, saddle: dict):
    ivm_metrics = dense_linear_trajectory(ivm["path_history"], subdivisions=200)
    ode_metrics = dense_linear_trajectory(ode["path"], subdivisions=200)
    views = [
        ("IVM", np.asarray(ivm["article_nf"], dtype=float), ivm_metrics),
        ("ODE-NEB", np.asarray(ode["total_internal_calls"], dtype=float), ode_metrics),
        ("NN-IVM", np.asarray(nn["total_internal_calls"], dtype=float), nn),
        ("gMAM (autodiff)", np.asarray(gmam["total_internal_calls"], dtype=float), gmam),
    ]
    for _, _, metrics in views:
        if "dense_barrier_error" not in metrics:
            metrics["dense_barrier_error"] = np.abs(
                np.asarray(metrics["dense_barrier"]) - float(saddle["forward_barrier"])
            )

    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.85))
    definitions = (
        ("dense_i_perp", r"Dense $I_{\perp}$", True),
        ("dense_barrier", r"Forward barrier $\Delta E$ (eV)", False),
        ("dense_barrier_error", r"$|\Delta E-\Delta E^\ast|$ (eV)", True),
    )
    rows, summaries = [], {}
    for name, nof, metrics in views:
        for axis, (key, _, log_y) in zip(axes, definitions):
            axis.plot(nof, np.asarray(metrics[key], dtype=float), color=COLORS[name],
                      linestyle=STYLES[name], linewidth=1.75 if name == "IVM" else 1.45,
                      label=name, zorder=10 if name == "IVM" else 4)
            if log_y:
                axis.set_yscale("log")
        dense_i = np.asarray(metrics["dense_i_perp"], dtype=float)
        error = np.asarray(metrics["dense_barrier_error"], dtype=float)
        i_index, error_index = int(np.nanargmin(dense_i)), int(np.nanargmin(error))
        summaries[name] = {
            "last_nof": int(nof[-1]),
            "last_dense_i_perp": float(dense_i[-1]),
            "minimum_dense_i_perp": float(dense_i[i_index]),
            "nof_at_minimum_dense_i_perp": int(nof[i_index]),
            "last_barrier_eV": float(np.asarray(metrics["dense_barrier"])[-1]),
            "minimum_barrier_error_eV": float(error[error_index]),
            "nof_at_minimum_barrier_error": int(nof[error_index]),
        }
        for index, cost in enumerate(nof):
            rows.append({"method": name, "plotted_index": index, "number_of_forces": int(cost),
                         "dense_i_perp": float(dense_i[index]),
                         "dense_barrier_eV": float(np.asarray(metrics["dense_barrier"])[index]),
                         "dense_barrier_error_eV": float(error[index])})

    axes[1].axhline(saddle["forward_barrier"], color="#555555", linewidth=0.8,
                    linestyle=(0, (3, 2)), label=r"Index-1 saddle $\Delta E^\ast$")
    for axis, (_, ylabel, _) in zip(axes, definitions):
        axis.set_xscale("log")
        axis.set_xlim(20, 105000)
        axis.set_xlabel("Number of Forces")
        axis.set_ylabel(ylabel)
        axis.set_box_aspect(1)
        axis.grid(True, which="major", color="#DEDEDE", linewidth=0.4)
        axis.grid(True, which="minor", color="#EEEEEE", linewidth=0.25)
    for index, axis in enumerate(axes):
        axis.text(-0.16, 1.05, chr(ord("a") + index), transform=axis.transAxes,
                  fontweight="bold", fontsize=8.5, va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    ref_handles, ref_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles + ref_handles[-1:], labels + ref_labels[-1:], loc="upper center",
               ncol=5, bbox_to_anchor=(0.5, 0.985), frameon=False)
    fig.subplots_adjust(left=0.073, right=0.98, bottom=0.17, top=0.78, wspace=0.42)
    fig.canvas.draw()
    ratios = [float(axis.get_window_extent().width / axis.get_window_extent().height) for axis in axes]
    np.testing.assert_allclose(ratios, 1.0, atol=1e-12, rtol=0)
    output = PAPER_FIGURES / "leps_continuous_methods_comparison.pdf"
    fig.savefig(output, dpi=600, metadata={"Title": "LEPS path-representation comparison by total NoF",
                                           "CreationDate": None, "ModDate": None})
    plt.close(fig)
    return output, rows, summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    configure_style()
    registration = json.loads(METRICS_REGISTRY.read_text())
    assert sha256(METRICS_FIGURE) == registration["figure_sha256"]
    inputs = {"selected_ir_ivm": IR_RESULT, "ode_neb": ODE_RESULT,
              "nn_ivm": NN_RESULT, "gmam": GMAM_RESULT}
    for name, path in inputs.items():
        if name in ("selected_ir_ivm", "ode_neb"):
            registered_name = "IVM" if name == "selected_ir_ivm" else "ODE-NEB"
            assert sha256(path) == registration["plotted_results"][registered_name]["sha256"]
    ivm, ode, nn, gmam = (load(path) for path in inputs.values())
    hashes = {sha256(IR_RESULT), sha256(ODE_RESULT), sha256(NN_RESULT), sha256(GMAM_RESULT)}
    assert len(hashes) == 4
    initial_hashes = {
        hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        for array in (ivm["path_history"][0], ode["initial_path"], nn["initial_path"], gmam["initial_path"])
    }
    assert len(initial_hashes) == 1
    saddle = reference_saddle()

    path_pdf, path_rows = path_figure(ivm, ode, saddle)
    continuous_pdf, continuous_rows, continuous_summary = continuous_figure(ivm, ode, nn, gmam, saddle)
    metrics_target = PAPER_FIGURES / "leps_all_methods_metrics.pdf"
    shutil.copyfile(METRICS_FIGURE, metrics_target)
    assert sha256(metrics_target) == registration["figure_sha256"]
    write_csv(REPORTS / "leps_path_evolution_source.csv", path_rows)
    write_csv(REPORTS / "leps_continuous_methods_source.csv", continuous_rows)

    with METRICS_CSV.open(newline="", encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))
    ivm_rows = [row for row in metric_rows if row["method"] == "IVM"]
    i = np.asarray([float(row["i_perp"]) for row in ivm_rows])
    nof = np.asarray([int(row["number_of_forces"]) for row in ivm_rows])
    action = np.asarray([float(row["geometric_action_eV"]) for row in ivm_rows])
    threshold = np.flatnonzero(i <= 1e-4)
    all_methods_summary = {
        "ivm_first_nof_at_i_perp_le_1e-4": int(nof[threshold[0]]),
        "ivm_last_saved_path_nof": int(nof[-1]),
        "ivm_terminal_ledger_nof": int(ivm["terminal_article_nf"]),
        "ivm_last_i_perp": float(i[-1]),
        "ivm_last_geometric_action_eV": float(action[-1]),
        "ivm_minimum_geometric_action_eV": float(np.min(action)),
        "ivm_nof_at_minimum_geometric_action": int(nof[np.argmin(action)]),
    }
    provenance = {
        "date": "2026-09-06",
        "purpose": "paper0819 LEPS three-figure and prose update from frozen experiment results",
        "script": {"path": str(HERE.relative_to(PROJECT)), "sha256": sha256(HERE)},
        "figures": {
            "path_evolution": {"path": str(path_pdf.relative_to(PROJECT)), "sha256": sha256(path_pdf)},
            "all_methods_metrics": {"path": str(metrics_target.relative_to(PROJECT)), "sha256": sha256(metrics_target)},
            "continuous_comparison": {"path": str(continuous_pdf.relative_to(PROJECT)), "sha256": sha256(continuous_pdf)},
        },
        "inputs": {name: {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)}
                   for name, path in inputs.items()},
        "metrics_registry": {"path": str(METRICS_REGISTRY.relative_to(PROJECT)),
                             "sha256": sha256(METRICS_REGISTRY)},
        "metrics_source_csv": {"path": str(METRICS_CSV.relative_to(PROJECT)),
                               "sha256": sha256(METRICS_CSV)},
        "derived_sources": {
            "path_evolution": "reports/leps_path_evolution_source.csv",
            "continuous_comparison": "reports/leps_continuous_methods_source.csv",
        },
        "shared_initial_path_array_sha256": next(iter(initial_hashes)),
        "path_snapshots": list(PATH_SNAPSHOTS),
        "path_snapshot_actual_nof": {
            str(snapshot): {
                "IVM": path_at_snapshot(
                    ivm["article_nf"], ivm["path_history"], snapshot
                )[0],
                "ODE-NEB": path_at_snapshot(
                    ode["total_internal_calls"], ode["path"], snapshot
                )[0],
            }
            for snapshot in PATH_SNAPSHOTS
        },
        "all_methods_summary": all_methods_summary,
        "continuous_summary": continuous_summary,
        "metric_contracts": {
            "all_methods": registration["metric_contract"],
            "continuous_dense_discrete": "200 linear subdivisions per each of 21 segments",
            "continuous_neural": "stored dense diagnostic arrays from each frozen production result",
            "cost": "total uncached joint force evaluations across 20 internal images/streams",
            "postprocessing_excluded_from_nof": True,
        },
        "checks": {"all_initial_paths_identical": True,
                   "all_axes_boxes_square": True,
                   "metrics_figure_matches_current_registration": True},
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"figures": provenance["figures"],
                      "path_snapshot_actual_nof": provenance["path_snapshot_actual_nof"],
                      "all_methods": all_methods_summary,
                      "continuous": continuous_summary}, indent=2))


if __name__ == "__main__":
    main()
