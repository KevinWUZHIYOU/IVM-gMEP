#!/usr/bin/env python3
"""Preview the Delta-E-aware grid IVM in the existing Ala 1x2 figure.

Selection is made against the two fixed comparison values already used by the
figure: I_perp <= 2.1e-4 and actual densified Delta E <=
2.4721831748734475 kcal/mol.  The two first passages remain metric-specific;
they do not occur at the same recorded path state.
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
STEM = "alanine66d_delta_aware_grid_ivm_replacement_1x2_preview"
CELL_ID = "a736dd230285d61c"
EXPECTED_RESULT_SHA256 = "0da1bbab469de81ca5b9229c4ebef656b7688ccb79c822ec14f0f4c92226d161"
EXPECTED_CONTROLS = {
    "alpha": 10.0,
    "residual_ratio": 0.7,
    "inner_max_steps": 5,
    "total_internal_nf_budget": 5000,
}
EXPECTED_IPERP_COSTS = (3386, 83522)
EXPECTED_BARRIER_COSTS = (672, 19222)
EXPECTED_BEST_BARRIER = 2.4626005431228144
EXPECTED_TESTED_LIMITS = [5, 8, 10, 15, 20, 50]


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
    module.EXPECTED_BARRIER_RECORD_COUNT = 7
    module.EXPECTED_BARRIER_ERROR_CTP_NF = 672
    module.EXPECTED_IPERP_COSTS = EXPECTED_IPERP_COSTS
    return module


def validate_inner_limit_invariance(shared) -> list[dict]:
    preregistration = shared.load_json(shared.METRIC_PREREGISTRATION)
    candidates = [
        cell
        for cell in preregistration["cells"]
        if cell["family"] == "inexact_reparam"
        and cell["controls"]["alpha"] == 10.0
        and cell["controls"]["residual_ratio"] == 0.7
    ]
    candidates.sort(key=lambda cell: int(cell["controls"]["inner_max_steps"]))
    limits = [int(cell["controls"]["inner_max_steps"]) for cell in candidates]
    if limits != EXPECTED_TESTED_LIMITS:
        raise RuntimeError(f"Unexpected tested inner-iteration limits: {limits}")
    for cell in candidates:
        if int(cell["inner_cap_stop_evidence"]["inner_cap_stop_count"]) != 0:
            raise RuntimeError("An alpha=10, r=0.7 run stopped at its iteration limit")
        result_path = Path(cell["source_result"])
        if shared.sha256(result_path) != str(cell["source_result_sha256"]):
            raise RuntimeError(f"Frozen result hash changed: {result_path}")

    keys = (
        "stored_path_calls",
        "total_internal_joint_evaluations",
        "i_perp",
        "stored_paths",
        "reparam_committed",
        "segment_length_cv",
        "segment_length_ratio",
    )
    reference_path = Path(candidates[0]["source_result"])
    with np.load(reference_path, allow_pickle=False) as reference:
        reference_arrays = {key: np.asarray(reference[key]) for key in keys}
    for cell in candidates[1:]:
        with np.load(Path(cell["source_result"]), allow_pickle=False) as saved:
            for key in keys:
                if not np.array_equal(reference_arrays[key], np.asarray(saved[key]), equal_nan=True):
                    raise RuntimeError(
                        f"Inner-iteration limit changed scientific array {key} "
                        f"for m_max={cell['controls']['inner_max_steps']}"
                    )
    return [
        {
            "cell_id": cell["cell_id"],
            "inner_max_steps": int(cell["controls"]["inner_max_steps"]),
            "result": cell["source_result"],
            "result_sha256": cell["source_result_sha256"],
            "iteration_limit_stop_count": int(
                cell["inner_cap_stop_evidence"]["inner_cap_stop_count"]
            ),
        }
        for cell in candidates
    ]


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
    ivm_barrier = shared.first_passage(rows, "ivm", "b", shared.BARRIER_TARGET)
    ode_barrier = shared.first_passage(rows, "ode_neb", "b", shared.BARRIER_TARGET)
    if (int(ivm_iperp), int(ode_iperp)) != EXPECTED_IPERP_COSTS:
        raise RuntimeError("I_perp first-passage drift")
    if (int(ivm_barrier), int(ode_barrier)) != EXPECTED_BARRIER_COSTS:
        raise RuntimeError("Delta E first-passage drift")
    shared.annotate_speedup(
        axes[0],
        shared.IPERP_TARGET,
        ivm_iperp,
        ode_iperp,
        rf"{ode_iperp / ivm_iperp:.1f}$\times$",
        1.72e-4,
    )
    shared.annotate_speedup(
        axes[1],
        shared.BARRIER_TARGET,
        ivm_barrier,
        ode_barrier,
        rf"{ode_barrier / ivm_barrier:.1f}$\times$",
        2.455,
    )

    ivm_barriers, _ = shared.curve(rows, "ivm", "b")
    if not np.isclose(np.min(ivm_barriers), EXPECTED_BEST_BARRIER, rtol=0, atol=1e-12):
        raise RuntimeError("Best densified Delta E changed")

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
            "Title": "Alanine 66D Delta-E-aware grid IVM replacement preview",
            "Subject": "alpha=10, r=0.7, inner_max_steps=5; metric-specific first passages",
        },
    )
    plt.close(figure)
    return {
        "i_perp": {
            "target": shared.IPERP_TARGET,
            "ivm_first_passage_NF": int(ivm_iperp),
            "ode_neb_first_passage_NF": int(ode_iperp),
            "ode_neb_over_ivm": float(ode_iperp / ivm_iperp),
        },
        "densified_Delta_E": {
            "target_kcal_per_mol": shared.BARRIER_TARGET,
            "ivm_first_passage_NF": int(ivm_barrier),
            "ode_neb_first_passage_NF": int(ode_barrier),
            "ode_neb_over_ivm": float(ode_barrier / ivm_barrier),
            "best_record_kcal_per_mol": float(np.min(ivm_barriers)),
        },
        "metric_specific_first_passages": True,
        "same_recorded_state_claim": False,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    shared = load_shared()
    shared.validate_registered_inputs()
    base = shared.load_base_plotter()
    base.configure_style()
    invariant_limits = validate_inner_limit_invariance(shared)
    fieldnames, frozen_baselines = shared.baseline_rows()
    grid_rows, grid_info = shared.load_grid_ivm()
    rows = grid_rows + frozen_baselines

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / f"{STEM}.pdf"
    csv_path = args.output_dir / f"{STEM}_source.csv"
    manifest_path = args.output_dir / f"{STEM}_manifest.json"
    existing = [path for path in (pdf_path, csv_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    comparisons = draw_figure(shared, base, rows, pdf_path)
    write_csv(csv_path, fieldnames, rows)
    manifest = {
        "status": "preview_not_for_citation",
        "purpose": "replace only the IVM series in the existing active-candidate 1x2 figure",
        "formal_grid_conclusion": "no_unique_winner",
        "selection_rule": (
            "reach both displayed comparison values separately; choose the smallest "
            "tested inactive inner-iteration limit among identical trajectories"
        ),
        "selected_cell_id": CELL_ID,
        "selected_controls": EXPECTED_CONTROLS,
        "same_recorded_state_claim": False,
        "grid_ivm": grid_info,
        "inner_iteration_limit_invariance": invariant_limits,
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
