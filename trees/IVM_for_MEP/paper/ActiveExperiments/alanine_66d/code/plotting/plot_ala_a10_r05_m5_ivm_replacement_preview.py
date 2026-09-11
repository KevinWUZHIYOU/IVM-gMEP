#!/usr/bin/env python3
"""Preview alpha=10, r=0.5, m_max=5 as the IVM series in the Ala 1x2 figure.

The five registered baseline curves are copied unchanged.  The left-panel
comparison remains the registered I_perp <= 2.1e-4 first passage.  The
selected IVM does not reach the registered densified Delta E target, so the
right panel reports that miss instead of drawing a speedup arrow.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


REPO = Path(__file__).resolve().parents[5]
SHARED_PLOTTER = Path(__file__).with_name(
    "plot_ala_grid_recommended_ivm_replacement_preview.py"
)
OUTPUT_DIR = REPO / "output/pdf"
STEM = "alanine66d_a10_r05_m5_ivm_replacement_1x2_preview"
CELL_ID = "e1ecaee043e8a0b2"
EXPECTED_RESULT_SHA256 = "9582aad8db77b4d57ae1f09b0e4b81d92c24d2060ac989fb90adf6011957773d"
EXPECTED_CONTROLS = {
    "alpha": 10.0,
    "residual_ratio": 0.5,
    "inner_max_steps": 5,
    "total_internal_nf_budget": 5000,
}
EXPECTED_IPERP_COSTS = (2265, 83522)
EXPECTED_STRICT_IPERP_COST = 2816
EXPECTED_ODE_BARRIER_COST = 19222
EXPECTED_BEST_BARRIER = 2.4783444495828424
EXPECTED_BEST_BARRIER_NF = 568
EXPECTED_REGISTERED_IVM_COSTS = (2150, 911)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_shared():
    spec = importlib.util.spec_from_file_location("ala_grid_preview_shared", SHARED_PLOTTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SHARED_PLOTTER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CELL_ID = CELL_ID
    module.EXPECTED_RESULT_SHA256 = EXPECTED_RESULT_SHA256
    module.EXPECTED_CONTROLS = EXPECTED_CONTROLS
    module.EXPECTED_ELIGIBLE_STATE_COUNT = 22
    module.EXPECTED_IPERP_RECORD_COUNT = 22
    module.EXPECTED_BARRIER_RECORD_COUNT = 6
    module.EXPECTED_BARRIER_ERROR_CTP_NF = None
    module.EXPECTED_IPERP_COSTS = EXPECTED_IPERP_COSTS
    module.EXPECTED_BEST_BARRIER = EXPECTED_BEST_BARRIER
    module.EXPECTED_BEST_BARRIER_NF = EXPECTED_BEST_BARRIER_NF
    return module


def registered_rows(shared) -> list[dict]:
    with shared.REGISTERED_SOURCE.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def validate_selected_run(shared) -> dict:
    preregistration = shared.load_json(shared.METRIC_PREREGISTRATION)
    matches = [cell for cell in preregistration["cells"] if cell["cell_id"] == CELL_ID]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one preregistered run {CELL_ID}")
    cell = matches[0]
    evidence = dict(cell["inner_cap_stop_evidence"])
    if (
        cell["family"] != "inexact_reparam"
        or cell["controls"] != EXPECTED_CONTROLS
        or int(evidence["inner_cap_stop_count"]) != 1
        or not bool(evidence["inner_cap_binding"])
    ):
        raise RuntimeError("Selected-run controls or inner-limit evidence changed")
    return evidence


def draw_figure(shared, base, rows: list[dict], output: Path) -> dict:
    figure = plt.figure(figsize=(base.mm_to_inch(180), base.mm_to_inch(72)), dpi=220)
    grid = GridSpec(
        1,
        2,
        figure=figure,
        left=0.065,
        right=0.975,
        bottom=0.17,
        top=0.78,
        wspace=0.24,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    for method in base.METHOD_ORDER:
        style = base.METHOD_STYLE[method]
        for axis, panel in zip(axes, ("a", "b")):
            values, costs = shared.curve(rows, method, panel)
            axis.plot(
                values,
                costs,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor="white",
                markeredgewidth=0.55,
                markevery=max(1, len(values) // 8),
                alpha=1.0 if method == "ivm" else 0.92,
                zorder=style["zorder"],
            )

    base.style_cost_axis(axes[0], "i_perp", 1.1e5, "1x2")
    base.style_cost_axis(axes[1], "barrier", 4.5e4, "1x2")
    for axis in axes:
        axis.set_ylabel("Number of Forces")
        axis.set_box_aspect(None)
    axes[1].set_yticks((0, 15000, 30000, 45000))
    axes[1].set_xlabel(r"$\Delta E$ (kcal mol$^{-1}$)")

    ivm_iperp = shared.first_passage(rows, "ivm", "a", shared.IPERP_TARGET)
    ode_iperp = shared.first_passage(rows, "ode_neb", "a", shared.IPERP_TARGET)
    strict_iperp = shared.first_passage(rows, "ivm", "a", 1.5e-4)
    ivm_barrier = shared.first_passage(rows, "ivm", "b", shared.BARRIER_TARGET)
    ode_barrier = shared.first_passage(rows, "ode_neb", "b", shared.BARRIER_TARGET)
    if (int(ivm_iperp), int(ode_iperp)) != EXPECTED_IPERP_COSTS:
        raise RuntimeError("I_perp first-passage drift")
    if int(strict_iperp) != EXPECTED_STRICT_IPERP_COST:
        raise RuntimeError("Strict I_perp first-passage drift")
    if np.isfinite(ivm_barrier) or int(ode_barrier) != EXPECTED_ODE_BARRIER_COST:
        raise RuntimeError("Densified Delta E first-passage status changed")

    shared.annotate_speedup(
        axes[0],
        shared.IPERP_TARGET,
        ivm_iperp,
        ode_iperp,
        rf"{ode_iperp / ivm_iperp:.1f}$\times$",
        1.72e-4,
    )

    axes[1].axvline(
        shared.BARRIER_TARGET,
        color="#767676",
        linewidth=0.55,
        linestyle=":",
        zorder=1,
    )
    ivm_barriers, ivm_barrier_costs = shared.curve(rows, "ivm", "b")
    best_index = int(np.argmin(ivm_barriers))
    best_barrier = float(ivm_barriers[best_index])
    best_barrier_nf = int(ivm_barrier_costs[best_index])
    if not np.isclose(best_barrier, EXPECTED_BEST_BARRIER, rtol=0, atol=1e-12):
        raise RuntimeError("Best densified Delta E changed")
    if best_barrier_nf != EXPECTED_BEST_BARRIER_NF:
        raise RuntimeError("Best densified Delta E cost changed")
    axes[1].annotate(
        "target not reached",
        xy=(best_barrier, best_barrier_nf),
        xytext=(2.515, 3200),
        ha="right",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="#202020",
        arrowprops={"arrowstyle": "->", "color": "#202020", "lw": 0.7},
        zorder=26,
    )

    figure.canvas.draw()
    for axis, letter in zip(axes, "ab"):
        position = axis.get_position()
        figure.text(
            position.x0 - 0.027,
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
        bbox_to_anchor=(0.49, 0.975),
        ncol=3,
        columnspacing=1.35,
        handlelength=2.35,
        handletextpad=0.5,
        frameon=False,
    )
    figure.savefig(
        output,
        format="pdf",
        dpi=600,
        metadata={
            "Title": "Alanine 66D alpha10 r0.5 m5 IVM replacement preview",
            "Subject": (
                "alpha=10, r=0.5, inner_max_steps=5; densified Delta E "
                "target not reached"
            ),
        },
    )
    plt.close(figure)
    return {
        "i_perp": {
            "figure_target": shared.IPERP_TARGET,
            "ivm_first_passage_NF": int(ivm_iperp),
            "ode_neb_first_passage_NF": int(ode_iperp),
            "ode_neb_over_ivm": float(ode_iperp / ivm_iperp),
            "strict_target": 1.5e-4,
            "strict_target_first_passage_NF": int(strict_iperp),
        },
        "densified_Delta_E": {
            "figure_target_kcal_per_mol": shared.BARRIER_TARGET,
            "ivm_target_reached": False,
            "ivm_first_passage_NF": None,
            "ode_neb_first_passage_NF": int(ode_barrier),
            "ivm_best_record_kcal_per_mol": best_barrier,
            "ivm_best_record_NF": best_barrier_nf,
            "miss_above_target_kcal_per_mol": float(
                best_barrier - shared.BARRIER_TARGET
            ),
        },
        "same_recorded_state_claim": False,
    }


def main() -> None:
    args = parse_args()
    shared = load_shared()
    shared.validate_registered_inputs()
    base = shared.load_base_plotter()
    base.configure_style()
    inner_limit_evidence = validate_selected_run(shared)
    fieldnames, frozen_baselines = shared.baseline_rows()
    original_rows = registered_rows(shared)
    grid_rows, grid_info = shared.load_grid_ivm()
    rows = grid_rows + frozen_baselines

    registered_iperp = shared.first_passage(
        original_rows, "ivm", "a", shared.IPERP_TARGET
    )
    registered_barrier = shared.first_passage(
        original_rows, "ivm", "b", shared.BARRIER_TARGET
    )
    if (int(registered_iperp), int(registered_barrier)) != EXPECTED_REGISTERED_IVM_COSTS:
        raise RuntimeError("Registered IVM comparison values changed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / f"{STEM}.pdf"
    csv_path = args.output_dir / f"{STEM}_source.csv"
    manifest_path = args.output_dir / f"{STEM}_manifest.json"
    existing = [path for path in (pdf_path, csv_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    comparisons = draw_figure(shared, base, rows, pdf_path)
    shared.write_csv(csv_path, fieldnames, rows)
    comparisons["replacement_effect_vs_registered_ivm"] = {
        "i_perp_first_passage_NF_before": int(registered_iperp),
        "i_perp_first_passage_NF_after": comparisons["i_perp"][
            "ivm_first_passage_NF"
        ],
        "densified_Delta_E_first_passage_NF_before": int(registered_barrier),
        "densified_Delta_E_first_passage_NF_after": None,
    }
    manifest = {
        "status": "preview_not_for_citation",
        "purpose": "replace only the IVM series in the existing active-candidate 1x2 figure",
        "formal_grid_conclusion": "no_unique_winner",
        "selection_role": "user-requested post-hoc replacement preview",
        "selected_cell_id": CELL_ID,
        "selected_controls": EXPECTED_CONTROLS,
        "selected_run_inner_iteration_limit_evidence": inner_limit_evidence,
        "grid_ivm": grid_info,
        "baseline_policy": (
            "all five plotted baseline record vertices copied unchanged from the "
            "registered source CSV"
        ),
        "comparisons": comparisons,
        "inputs": {
            "registered_source_csv": {
                "path": str(shared.REGISTERED_SOURCE),
                "sha256": shared.EXPECTED_HASHES[shared.REGISTERED_SOURCE],
            },
            "metric_preregistration": {
                "path": str(shared.METRIC_PREREGISTRATION),
                "sha256": shared.EXPECTED_HASHES[shared.METRIC_PREREGISTRATION],
            },
            "metric_analysis": {
                "path": str(shared.METRIC_ANALYSIS),
                "sha256": shared.EXPECTED_HASHES[shared.METRIC_ANALYSIS],
            },
            "plotter": {
                "path": str(Path(__file__).resolve()),
                "sha256": shared.sha256(Path(__file__).resolve()),
            },
            "shared_plotter": {
                "path": str(SHARED_PLOTTER),
                "sha256": shared.sha256(SHARED_PLOTTER),
            },
            "base_plotter": {
                "path": str(shared.BASE_PLOTTER),
                "sha256": shared.sha256(shared.BASE_PLOTTER),
            },
        },
        "outputs": {
            "pdf": {"path": str(pdf_path), "sha256": shared.sha256(pdf_path)},
            "source_csv": {"path": str(csv_path), "sha256": shared.sha256(csv_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pdf": str(pdf_path), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
