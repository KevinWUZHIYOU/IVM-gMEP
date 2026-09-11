#!/usr/bin/env python3
"""Append discrete geometric action to the Delta-aware Ala comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np


REPO = Path(__file__).resolve().parents[5]
PLOT_DIR = Path(__file__).resolve().parent
METRIC_DIR = PLOT_DIR.parent / "metrics"
if str(METRIC_DIR) not in sys.path:
    sys.path.insert(0, str(METRIC_DIR))

import postprocess_ala_geometric_action as action_source  # noqa: E402


DELTA_PLOTTER = PLOT_DIR / "plot_ala_delta_aware_grid_ivm_replacement_preview.py"
OUTPUT_DIR = REPO / "output/pdf"
STEM = "alanine66d_ivm_geometric_action_1x3"
MANUSCRIPT_TEX = Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819/main.tex")
MANUSCRIPT_FIGURE = (
    MANUSCRIPT_TEX.parent / "Figures/ala2_66d" / f"{STEM}.pdf"
)
ACTION_TARGET = 5.0
EXPECTED_ACTION_FIRST_PASSAGES = {"ivm": 2233, "ode_neb": 91382}
Y_MAX = 110000
PANEL_METRICS = {"a": "i_perp", "b": "densified_Delta_E", "c": "geometric_action"}
FIGURE4_PLOTTER = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819/Figures/Muller10/"
    "plot_muller_q10_article_method_comparison_n10_n20_2x3_with_irivm.py"
)
FIGURE4_STYLE_SOURCE = (
    REPO
    / "paper/ActiveExperiments/muller_q10_10d/image12/code/plotting/"
    "plot_q10_article_muller12.py"
)
FIGURE4_MARKER_POLICY = {
    "ivm": {"marker": "o", "markersize": 3.1},
    "ode_neb": {"marker": "s", "markersize": 2.9},
    "static_neb": {"marker": "v", "markersize": 2.8},
    "original_string": {"marker": "D", "markersize": 2.7},
    "obsm_fire": {"marker": "s", "markersize": 2.8},
    "obsm_bfgs": {"marker": "P", "markersize": 2.8},
}
FIGURE4_MARKER_EDGEWIDTH = 0.7
FIGURE4_TARGET_MARKERS_PER_CURVE = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_delta_plotter():
    spec = importlib.util.spec_from_file_location("ala_delta_preview", DELTA_PLOTTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {DELTA_PLOTTER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_action_data() -> tuple[dict, dict, dict]:
    contract = json.loads(action_source.CONTRACT.read_text())
    audit_path = action_source.OUT.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text())
    if audit["contract_sha256"] != sha256(action_source.CONTRACT):
        raise RuntimeError("Action contract hash changed")
    if audit["result_sha256"] != sha256(action_source.OUT):
        raise RuntimeError("Action result hash changed")
    if audit["diagnostic_joint_evaluations"] != contract[
        "expected_diagnostic_joint_evaluations"
    ]:
        raise RuntimeError("Action diagnostic NoF changed")
    with np.load(action_source.OUT, allow_pickle=False) as saved:
        data = {key: np.asarray(saved[key]) for key in saved.files}
    reconstructed = (
        data["segment_length_angstrom"]
        * (
            data["gradient_norm_ev_per_angstrom"][:, :-1]
            + data["gradient_norm_ev_per_angstrom"][:, 1:]
        )
        * 0.5
    ).sum(axis=1) * contract["ev_to_kcal"]
    np.testing.assert_allclose(reconstructed, data["action_kcal"], rtol=5e-15)
    np.testing.assert_array_equal(data["diagnostic_joint_evaluations"], 22)
    if len(data["method"]) != contract["selected_path_count"]:
        raise RuntimeError("Action path count changed")
    return data, contract, audit


def strict_frontier(
    values: np.ndarray, costs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    costs = np.asarray(costs, dtype=int)
    order = np.argsort(costs, kind="mergesort")
    values = values[order]
    costs = costs[order]
    unique_costs, starts = np.unique(costs, return_index=True)
    ends = np.r_[starts[1:], len(costs)]
    chosen = np.asarray(
        [start + int(np.argmin(values[start:end])) for start, end in zip(starts, ends)]
    )
    candidate_values = values[chosen]
    source_indices = order[chosen]
    keep = []
    best = np.inf
    for index, value in enumerate(candidate_values):
        if value < best:
            best = float(value)
            keep.append(index)
    keep = np.asarray(keep, dtype=int)
    return candidate_values[keep], unique_costs[keep], source_indices[keep]


def first_passage(data: dict, method: str, target: float) -> int | None:
    take = (data["method"] == method) & (data["action_kcal"] <= target)
    return int(data["number_of_forces"][take].min()) if np.any(take) else None


def force_formatter(value: float, _position: int) -> str:
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def apply_figure4_marker_policy(base) -> None:
    """Freeze the shared-method marker assignments to manuscript Figure 4."""
    if tuple(FIGURE4_MARKER_POLICY) != tuple(base.METHOD_ORDER):
        raise RuntimeError("Figure 4 marker policy no longer covers every method")
    for method, marker_style in FIGURE4_MARKER_POLICY.items():
        base.METHOD_STYLE[method].update(marker_style)


def write_source_csv(path: Path, delta_rows: list[dict], action_rows: list[dict]) -> None:
    fieldnames = (
        "panel",
        "method",
        "metric",
        "metric_value",
        "running_best_metric",
        "minimum_NF",
        "source_call",
        "source_record_index",
        "selected_path_sha256",
        "cost_source_file",
        "metric_source_file",
        "diagnostic_calls_charged_to_NF",
        "source_stage",
    )
    rows = []
    for row in delta_rows:
        rows.append(
            {
                key: row.get(key, "")
                for key in fieldnames
            }
        )
    for row in action_rows:
        rows.append({key: row.get(key, "") for key in fieldnames})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    delta = load_delta_plotter()
    shared = delta.load_shared()
    shared.validate_registered_inputs()
    base = shared.load_base_plotter()
    base.configure_style()
    apply_figure4_marker_policy(base)
    fieldnames, frozen_baselines = shared.baseline_rows()
    del fieldnames
    grid_rows, grid_info = shared.load_grid_ivm()
    delta_rows = grid_rows + frozen_baselines
    action, contract, audit = load_action_data()

    passages = {
        method: first_passage(action, method, ACTION_TARGET)
        for method in ("ivm", "ode_neb")
    }
    if passages != EXPECTED_ACTION_FIRST_PASSAGES:
        raise RuntimeError(f"Action first passages changed: {passages}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / f"{STEM}.pdf"
    csv_path = args.output_dir / f"{STEM}_source.csv"
    manifest_path = args.output_dir / f"{STEM}_manifest.json"
    existing = [path for path in (pdf_path, csv_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    figure = plt.figure(figsize=(base.mm_to_inch(180), base.mm_to_inch(67)), dpi=220)
    grid = GridSpec(
        1,
        3,
        figure=figure,
        left=0.065,
        right=0.985,
        bottom=0.19,
        top=0.76,
        wspace=0.27,
    )
    axes = [figure.add_subplot(grid[0, index], sharey=None if index == 0 else None) for index in range(3)]

    for method in base.METHOD_ORDER:
        style = base.METHOD_STYLE[method]
        for axis, panel in zip(axes[:2], ("a", "b")):
            values, costs = shared.curve(delta_rows, method, panel)
            axis.plot(
                values,
                costs,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor="white",
                markeredgewidth=FIGURE4_MARKER_EDGEWIDTH,
                markevery=max(1, len(values) // FIGURE4_TARGET_MARKERS_PER_CURVE),
                alpha=1.0 if method == "ivm" else 0.92,
                zorder=style["zorder"],
            )

    action_source_rows = []
    for method in base.METHOD_ORDER:
        take = action["method"] == method
        source_positions = np.flatnonzero(take)
        values, costs, local_indices = strict_frontier(
            action["action_kcal"][take], action["number_of_forces"][take]
        )
        style = base.METHOD_STYLE[method]
        axes[2].plot(
            values,
            costs,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"],
            markerfacecolor="white",
            markeredgewidth=FIGURE4_MARKER_EDGEWIDTH,
            markevery=max(1, len(values) // FIGURE4_TARGET_MARKERS_PER_CURVE),
            alpha=1.0 if method == "ivm" else 0.92,
            zorder=style["zorder"],
        )
        for value, cost, local_index in zip(values, costs, local_indices):
            index = int(source_positions[local_index])
            action_source_rows.append(
                {
                    "panel": "c",
                    "method": method,
                    "metric": "geometric_action",
                    "metric_value": float(action["action_kcal"][index]),
                    "running_best_metric": float(value),
                    "minimum_NF": int(cost),
                    "source_call": int(action["source_call"][index]),
                    "source_record_index": int(action["source_record_index"][index]),
                    "selected_path_sha256": str(action["selected_path_sha256"][index]),
                    "cost_source_file": str(action["source_result"][index]),
                    "metric_source_file": str(action_source.OUT),
                    "diagnostic_calls_charged_to_NF": False,
                    "source_stage": "discrete_22_image_full_gradient_trapezoid",
                }
            )

    base.style_cost_axis(axes[0], "i_perp", Y_MAX, "1x2")
    base.style_cost_axis(axes[1], "barrier", Y_MAX, "1x2")
    axes[1].set_xlabel(r"$\Delta E$ (kcal mol$^{-1}$)")
    axes[2].set_xscale("log")
    axes[2].set_xlim(3000.0, 2.5)
    axes[2].set_xticks((1000.0, 100.0, 10.0, 5.0))
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].set_xlabel(r"$\widehat{S}$ (kcal mol$^{-1}$)")

    for axis in axes:
        axis.set_ylim(0, Y_MAX)
        axis.set_yticks((0, 25000, 50000, 75000, 100000))
        axis.yaxis.set_major_formatter(FuncFormatter(force_formatter))
        axis.set_ylabel("Number of Forces")
        axis.set_box_aspect(None)
        axis.grid(True, which="major", color="#D9D9D6", linewidth=0.4, alpha=0.62)

    ivm_iperp = shared.first_passage(delta_rows, "ivm", "a", shared.IPERP_TARGET)
    ode_iperp = shared.first_passage(delta_rows, "ode_neb", "a", shared.IPERP_TARGET)
    ivm_barrier = shared.first_passage(delta_rows, "ivm", "b", shared.BARRIER_TARGET)
    ode_barrier = shared.first_passage(delta_rows, "ode_neb", "b", shared.BARRIER_TARGET)
    shared.annotate_speedup(
        axes[0], shared.IPERP_TARGET, ivm_iperp, ode_iperp,
        rf"{ode_iperp / ivm_iperp:.1f}$\times$", 1.72e-4,
    )
    shared.annotate_speedup(
        axes[1], shared.BARRIER_TARGET, ivm_barrier, ode_barrier,
        rf"{ode_barrier / ivm_barrier:.1f}$\times$", 2.455,
    )
    shared.annotate_speedup(
        axes[2], ACTION_TARGET, passages["ivm"], passages["ode_neb"],
        rf"{passages['ode_neb'] / passages['ivm']:.1f}$\times$", 4.6,
    )

    figure.canvas.draw()
    for axis, letter in zip(axes, "abc"):
        position = axis.get_position()
        figure.text(
            position.x0 - 0.025,
            position.y1 + 0.012,
            letter,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
    base.METHOD_STYLE["ivm"]["label"] = "IVM"
    handles = [base.legend_handle(method) for method in base.METHOD_ORDER]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        columnspacing=1.35,
        handlelength=2.35,
        handletextpad=0.5,
        frameon=False,
    )
    figure.savefig(
        pdf_path,
        format="pdf",
        dpi=600,
        metadata={
            "Title": "Alanine 66D IVM comparison with discrete geometric action",
            "Subject": (
                "alpha=10, r=0.7, m_max=5; 22-image full-gradient trapezoid "
                "action displayed as S-hat without a subscript"
            ),
        },
    )
    plt.close(figure)

    write_source_csv(csv_path, delta_rows, action_source_rows)
    manifest = {
        "status": "registered_and_cited_in_paper0819",
        "purpose": "registered 1x3 cost-to-precision figure with discrete geometric action",
        "manuscript_registration": {
            "status": "registered",
            "main_tex": str(MANUSCRIPT_TEX),
            "figure_copy": str(MANUSCRIPT_FIGURE),
            "figure_label": "fig:Ala2_convergence",
            "registered_date": "2026-09-06",
        },
        "selected_ivm_controls": contract["selected_ivm_controls"],
        "action_definition": {
            "display_symbol": r"\widehat{S}",
            "continuous": "integral_path norm(grad V) d ell",
            "discrete": contract["formula"],
            "tangent": contract["tangent"],
            "quadrature": contract["quadrature"],
            "densification": contract["densification"],
            "unit": contract["output_unit"],
        },
        "action_comparison": {
            "target_kcal_per_mol": ACTION_TARGET,
            "threshold_policy": "fixed round plotting threshold selected after calculation for illustration only",
            "ivm_first_passage_NF": passages["ivm"],
            "ode_neb_first_passage_NF": passages["ode_neb"],
            "ode_neb_over_ivm": passages["ode_neb"] / passages["ivm"],
            "ivm_best_kcal_per_mol": float(action["action_kcal"][action["method"] == "ivm"].min()),
            "ode_neb_best_kcal_per_mol": float(action["action_kcal"][action["method"] == "ode_neb"].min()),
        },
        "common_y_axis": {"label": "Number of Forces", "minimum": 0, "maximum": Y_MAX},
        "marker_policy": {
            "reference": "paper0819 Figure 4 shared-method trajectory markers",
            "mapping": FIGURE4_MARKER_POLICY,
            "marker_face_color": "white",
            "marker_edge_width": FIGURE4_MARKER_EDGEWIDTH,
            "target_markers_per_curve": FIGURE4_TARGET_MARKERS_PER_CURVE,
            "figure4_plotter": {
                "path": str(FIGURE4_PLOTTER),
                "sha256": sha256(FIGURE4_PLOTTER),
            },
            "figure4_style_source": {
                "path": str(FIGURE4_STYLE_SOURCE),
                "sha256": sha256(FIGURE4_STYLE_SOURCE),
            },
        },
        "diagnostic": {
            "contract": {"path": str(action_source.CONTRACT), "sha256": sha256(action_source.CONTRACT)},
            "result": {"path": str(action_source.OUT), "sha256": sha256(action_source.OUT)},
            "audit": {"path": str(action_source.OUT.with_suffix('.audit.json')), "sha256": sha256(action_source.OUT.with_suffix('.audit.json'))},
            "selected_path_count": audit["selected_path_count"],
            "diagnostic_joint_evaluations": audit["diagnostic_joint_evaluations"],
            "diagnostic_cost_excluded_from_optimization_axis": True,
        },
        "original_two_panel_input": {
            "manifest": {"path": str(action_source.FIGURE_MANIFEST), "sha256": sha256(action_source.FIGURE_MANIFEST)},
            "source_csv": {"path": str(action_source.FIGURE_SOURCE), "sha256": sha256(action_source.FIGURE_SOURCE)},
            "grid_ivm": grid_info,
        },
        "code": {
            "plotter": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "action_evaluator": {"path": str(Path(action_source.__file__).resolve()), "sha256": sha256(Path(action_source.__file__).resolve())},
            "delta_plotter": {"path": str(DELTA_PLOTTER), "sha256": sha256(DELTA_PLOTTER)},
        },
        "outputs": {
            "pdf": {"path": str(pdf_path), "sha256": sha256(pdf_path)},
            "source_csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
