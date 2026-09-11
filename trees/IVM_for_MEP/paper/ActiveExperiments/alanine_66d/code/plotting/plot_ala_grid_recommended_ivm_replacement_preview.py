#!/usr/bin/env python3
"""Preview the registered grid-recommended IVM in the existing Ala 1x2 figure.

The five baseline curves are copied exactly from the registered source CSV of
the existing active candidate.  Only the IVM series is replaced.  The new IVM
series is the engineering follow-up configuration alpha=10, r=0.5, and
inner_max_steps=8 from the completed discrete-grid campaign.  This selection
is not a formal winner: the preregistered rule returned no unique winner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


REPO = Path(__file__).resolve().parents[5]
STUDY = REPO / "python/alanine_ivm_clean_four_level_20260801"
PLOT_DIR = REPO / "paper/ActiveExperiments/alanine_66d/code/plotting"
BASE_PLOTTER = PLOT_DIR / "plot_ala_cost_to_precision_muller_style.py"
REGISTERED_SOURCE = (
    REPO
    / "paper/ActiveExperiments/alanine_66d/figures/active_candidates"
    / "alanine66d_discrete_ivm_cost_to_precision_1x2_source.csv"
)
METRIC_STEM = "stage4_ala_discrete_grid_metric_campaign_v2_1787608620070816000"
METRIC_PREREGISTRATION = STUDY / "reports" / f"{METRIC_STEM}_preregistered.json"
METRIC_ANALYSIS = STUDY / "reports" / f"{METRIC_STEM}_metric_analysis.json"
LOCAL_HOST_REPORT = STUDY / "reports" / f"{METRIC_STEM}_local_host_report.json"
REMOTE_HOST_REPORT = STUDY / "reports" / f"{METRIC_STEM}_remote_host_report.json"
OUTPUT_DIR = REPO / "output/pdf"
STEM = "alanine66d_grid_recommended_ivm_replacement_1x2_preview"
CELL_ID = "c6b759e30de39e91"

EXPECTED_HASHES = {
    REGISTERED_SOURCE: "742723564379da1530ee5a6b1a9bd07e0d762f9e40456fd8ab45b9c01c3efe42",
    METRIC_PREREGISTRATION: "5fcfad3228e72c267ea1d768dd6f677528ae1fc8e8c6f180ea0d3498503b98a3",
    METRIC_ANALYSIS: "e6724142f780aefb567d060abfca8843b47fd27bf6cbc3bda34ea6156d194b3a",
}
EXPECTED_RESULT_SHA256 = "5e2911ea5f85af4214c608cf704fc2ac7ba03c4b1fbe7acd62ca085e35e34b4a"
EXPECTED_CONTROLS = {
    "alpha": 10.0,
    "residual_ratio": 0.5,
    "inner_max_steps": 8,
    "total_internal_nf_budget": 5000,
}
EXPECTED_ELIGIBLE_STATE_COUNT = 18
EXPECTED_IPERP_RECORD_COUNT = 18
EXPECTED_BARRIER_RECORD_COUNT = 6
EXPECTED_BARRIER_ERROR_CTP_NF = None
IPERP_TARGET = 2.1e-4
BARRIER_TARGET = 2.4721831748734475
EXPECTED_IPERP_COSTS = (2276, 83522)
EXPECTED_ODE_BARRIER_COST = 19222
EXPECTED_BEST_BARRIER = 2.483138098503456
EXPECTED_BEST_BARRIER_NF = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def load_base_plotter():
    spec = importlib.util.spec_from_file_location("ala_cost_base", BASE_PLOTTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_PLOTTER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_registered_inputs() -> None:
    for path, expected in EXPECTED_HASHES.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"Registered input drift for {path}: expected {expected}, got {observed}"
            )


def baseline_rows() -> tuple[list[str], list[dict]]:
    with REGISTERED_SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader if row["method"] != "ivm"]
    if not fieldnames or not rows:
        raise RuntimeError("Registered baseline source CSV is empty")
    return fieldnames, rows


def host_artifact_hashes() -> dict[str, str]:
    result = {}
    for path in (LOCAL_HOST_REPORT, REMOTE_HOST_REPORT):
        payload = load_json(path)
        for row in payload["rows"]:
            task_id = str(row["task_id"])
            artifact_hash = str(row["artifact_sha256"])
            if task_id in result and result[task_id] != artifact_hash:
                raise RuntimeError(f"Conflicting artifact hashes for task {task_id}")
            result[task_id] = artifact_hash
    return result


def strict_records(rows: list[dict], value_key: str) -> list[dict]:
    records = []
    running_best = np.inf
    for row in sorted(rows, key=lambda item: int(item["optimization_nf"])):
        value = float(row[value_key])
        if value < running_best:
            running_best = value
            copied = dict(row)
            copied["running_best"] = running_best
            records.append(copied)
    return records


def load_grid_ivm() -> tuple[list[dict], dict]:
    preregistration = load_json(METRIC_PREREGISTRATION)
    analysis = load_json(METRIC_ANALYSIS)
    cells = [cell for cell in preregistration["cells"] if cell["cell_id"] == CELL_ID]
    if len(cells) != 1:
        raise RuntimeError(f"Expected exactly one preregistered cell {CELL_ID}")
    cell = cells[0]
    expected_controls = EXPECTED_CONTROLS
    if cell["family"] != "inexact_reparam" or cell["controls"] != expected_controls:
        raise RuntimeError(f"Unexpected controls for {CELL_ID}: {cell['controls']}")
    if cell["complete_mesh_eligible_state_count"] != EXPECTED_ELIGIBLE_STATE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_ELIGIBLE_STATE_COUNT} complete states satisfying "
            "the mesh criteria"
        )
    result_path = Path(cell["source_result"])
    if str(cell["source_result_sha256"]) != EXPECTED_RESULT_SHA256:
        raise RuntimeError("Preregistered result hash changed")
    if sha256(result_path) != EXPECTED_RESULT_SHA256:
        raise RuntimeError("Frozen grid result file hash changed")

    task_specs = {str(task["task_id"]): task for task in preregistration["tasks"]}
    attested_hashes = host_artifact_hashes()
    task_artifacts = []
    state_rows = []
    with np.load(result_path, allow_pickle=False) as saved:
        paths = np.asarray(saved["stored_paths"], dtype=float)
        calls = np.asarray(saved["stored_path_calls"], dtype=int)
        force = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        iperp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        for state in cell["states"]:
            if not (state["formal_complete"] and state["mesh_eligible"]):
                raise RuntimeError("Cell state list contains an ineligible state")
            index = int(state["trajectory_index"])
            if array_sha256(paths[index]) != str(state["path_sha256"]):
                raise RuntimeError(f"Path hash mismatch at trajectory index {index}")
            checks = (
                (int(calls[index]), int(state["calls_per_image"]), "stored call"),
                (int(force[index]), int(state["optimization_nf"]), "optimization NF"),
            )
            for observed, expected, label in checks:
                if observed != expected:
                    raise RuntimeError(f"{label} mismatch at trajectory index {index}")
            if not np.isclose(iperp[index], float(state["raw_i_perp"]), rtol=0, atol=0):
                raise RuntimeError(f"I_perp mismatch at trajectory index {index}")
            if not np.isclose(cv[index], float(state["segment_length_cv"]), rtol=0, atol=0):
                raise RuntimeError(f"CV mismatch at trajectory index {index}")
            if not np.isclose(
                ratio[index], float(state["segment_length_ratio"]), rtol=0, atol=0
            ):
                raise RuntimeError(f"segment ratio mismatch at trajectory index {index}")

            task_id = str(state["path_task_id"])
            task_spec = task_specs[task_id]
            artifact_path = Path(task_spec["artifact"])
            observed_artifact_hash = sha256(artifact_path)
            if observed_artifact_hash != attested_hashes.get(task_id):
                raise RuntimeError(f"Dense metric artifact hash mismatch for task {task_id}")
            artifact = load_json(artifact_path)
            metric = artifact["metric"]
            if (
                artifact["status"] != "complete"
                or artifact["path_sha256"] != state["path_sha256"]
                or not metric["converged"]
                or int(metric["selected_subdivisions"]) != 160
                or int(metric["optimization_nf_charged"]) != 0
            ):
                raise RuntimeError(f"Invalid dense metric artifact for task {task_id}")
            state_rows.append(
                {
                    **state,
                    "reparam_committed": bool(committed[index]),
                    "barrier": float(metric["selected_barrier_kcal_per_mol"]),
                    "dense_subdivisions": int(metric["selected_subdivisions"]),
                    "dense_difference": float(
                        metric["final_level_difference_kcal_per_mol"]
                    ),
                    "artifact": str(artifact_path),
                    "artifact_sha256": observed_artifact_hash,
                }
            )
            task_artifacts.append(
                {"task_id": task_id, "path": str(artifact_path), "sha256": observed_artifact_hash}
            )

    iperp_records = strict_records(state_rows, "raw_i_perp")
    barrier_records = strict_records(state_rows, "barrier")
    if (
        len(iperp_records) != EXPECTED_IPERP_RECORD_COUNT
        or len(barrier_records) != EXPECTED_BARRIER_RECORD_COUNT
    ):
        raise RuntimeError("Unexpected number of strict IVM record vertices")

    summaries = [
        item
        for item in analysis["cell_summaries"]
        if item["family"] == "inexact_reparam" and item["cell_id"] == CELL_ID
    ]
    if len(summaries) != 1:
        raise RuntimeError("Metric analysis does not contain the selected cell exactly once")
    summary = summaries[0]
    if summary["targets"]["i_perp_ctp_nf"] != EXPECTED_IPERP_COSTS[0]:
        raise RuntimeError("Selected-cell I_perp first passage changed")
    if summary["targets"]["barrier_error_ctp_nf"] != EXPECTED_BARRIER_ERROR_CTP_NF:
        raise RuntimeError("Selected-cell barrier-error first passage changed")
    if analysis["winner"]["status"] != "no_unique_winner":
        raise RuntimeError("The formal grid conclusion changed")

    grid_rows = []
    common = {
        "method": "ivm",
        "diagnostic_calls_charged_to_NF": False,
        "cost_source_file": str(result_path),
    }
    for record in iperp_records:
        grid_rows.append(
            {
                **common,
                "panel": "a",
                "metric": "i_perp",
                "metric_value": record["raw_i_perp"],
                "running_best_metric": record["running_best"],
                "minimum_NF": record["optimization_nf"],
                "source_call": record["calls_per_image"],
                "is_initial": record["trajectory_index"] == 0,
                "reparam_committed": record["reparam_committed"],
                "segment_length_cv": record["segment_length_cv"],
                "segment_length_ratio": record["segment_length_ratio"],
                "dense_subdivisions": "",
                "dense_final_level_difference_kcal": "",
                "metric_source_file": str(result_path),
                "source_stage": "grid_complete_mesh_raw_result",
            }
        )
    for record in barrier_records:
        grid_rows.append(
            {
                **common,
                "panel": "b",
                "metric": "barrier",
                "metric_value": record["barrier"],
                "running_best_metric": record["running_best"],
                "minimum_NF": record["optimization_nf"],
                "source_call": record["calls_per_image"],
                "is_initial": record["trajectory_index"] == 0,
                "reparam_committed": record["reparam_committed"],
                "segment_length_cv": record["segment_length_cv"],
                "segment_length_ratio": record["segment_length_ratio"],
                "dense_subdivisions": record["dense_subdivisions"],
                "dense_final_level_difference_kcal": record["dense_difference"],
                "metric_source_file": record["artifact"],
                "source_stage": "grid_complete_mesh_dense160",
            }
        )
    info = {
        "controls": expected_controls,
        "result": {"path": str(result_path), "sha256": EXPECTED_RESULT_SHA256},
        "task_artifacts": task_artifacts,
        "eligible_state_count": len(state_rows),
        "iperp_record_count": len(iperp_records),
        "barrier_record_count": len(barrier_records),
        "best_barrier": float(barrier_records[-1]["barrier"]),
        "best_barrier_nf": int(barrier_records[-1]["optimization_nf"]),
    }
    return grid_rows, info


def curve(rows: list[dict], method: str, panel: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["method"] == method and row["panel"] == panel]
    selected.sort(key=lambda row: float(row["minimum_NF"]))
    return (
        np.asarray([float(row["metric_value"]) for row in selected], dtype=float),
        np.asarray([float(row["minimum_NF"]) for row in selected], dtype=float),
    )


def first_passage(rows: list[dict], method: str, panel: str, target: float) -> float:
    values, costs = curve(rows, method, panel)
    reached = costs[values <= target]
    return float(np.min(reached)) if reached.size else np.inf


def annotate_speedup(axis, target: float, low: float, high: float, label: str, x: float):
    axis.axvline(target, color="#767676", linewidth=0.55, linestyle=":", zorder=1)
    axis.annotate(
        "",
        xy=(target, low),
        xytext=(target, high),
        arrowprops={"arrowstyle": "<->", "color": "#202020", "lw": 0.8},
        zorder=25,
    )
    axis.text(
        x,
        0.5 * (low + high),
        label,
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color="#202020",
        zorder=26,
    )


def draw_figure(base, rows: list[dict], output: Path) -> dict:
    figure = plt.figure(figsize=(base.mm_to_inch(180), base.mm_to_inch(72)), dpi=220)
    grid = GridSpec(
        1, 2, figure=figure, left=0.065, right=0.975, bottom=0.17,
        top=0.78, wspace=0.24,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    for method in base.METHOD_ORDER:
        style = base.METHOD_STYLE[method]
        for axis, panel in zip(axes, ("a", "b")):
            values, costs = curve(rows, method, panel)
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

    ivm_iperp = first_passage(rows, "ivm", "a", IPERP_TARGET)
    ode_iperp = first_passage(rows, "ode_neb", "a", IPERP_TARGET)
    if (int(ivm_iperp), int(ode_iperp)) != EXPECTED_IPERP_COSTS:
        raise RuntimeError("I_perp first-passage drift")
    annotate_speedup(
        axes[0], IPERP_TARGET, ivm_iperp, ode_iperp,
        rf"{ode_iperp / ivm_iperp:.1f}$\times$", 1.72e-4,
    )

    ivm_barrier = first_passage(rows, "ivm", "b", BARRIER_TARGET)
    ode_barrier = first_passage(rows, "ode_neb", "b", BARRIER_TARGET)
    if np.isfinite(ivm_barrier) or int(ode_barrier) != EXPECTED_ODE_BARRIER_COST:
        raise RuntimeError("Barrier first-passage status changed")
    axes[1].axvline(
        BARRIER_TARGET, color="#767676", linewidth=0.55, linestyle=":", zorder=1
    )
    ivm_values, ivm_costs = curve(rows, "ivm", "b")
    best_index = int(np.argmin(ivm_values))
    best_value = float(ivm_values[best_index])
    best_cost = float(ivm_costs[best_index])
    if not np.isclose(best_value, EXPECTED_BEST_BARRIER, rtol=0, atol=1e-12):
        raise RuntimeError("Best grid IVM barrier changed")
    if int(best_cost) != EXPECTED_BEST_BARRIER_NF:
        raise RuntimeError("Best grid IVM barrier cost changed")
    axes[1].annotate(
        "target not reached",
        xy=(best_value, best_cost),
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
            "Title": "Alanine 66D grid-recommended IVM replacement preview",
            "Subject": "Engineering follow-up alpha=10, r=0.5, inner_max_steps=8",
        },
    )
    plt.close(figure)
    return {
        "i_perp": {
            "target": IPERP_TARGET,
            "ivm_first_passage_NF": int(ivm_iperp),
            "ode_neb_first_passage_NF": int(ode_iperp),
            "ode_neb_over_ivm": float(ode_iperp / ivm_iperp),
        },
        "barrier": {
            "target_kcal_per_mol": BARRIER_TARGET,
            "ivm_first_passage_NF": None,
            "ode_neb_first_passage_NF": int(ode_barrier),
            "ivm_best_record_kcal_per_mol": best_value,
            "ivm_best_record_NF": int(best_cost),
        },
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_registered_inputs()
    base = load_base_plotter()
    base.configure_style()
    fieldnames, frozen_baselines = baseline_rows()
    grid_rows, grid_info = load_grid_ivm()
    rows = grid_rows + frozen_baselines

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / f"{STEM}.pdf"
    csv_path = args.output_dir / f"{STEM}_source.csv"
    manifest_path = args.output_dir / f"{STEM}_manifest.json"
    existing = [path for path in (pdf_path, csv_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    comparisons = draw_figure(base, rows, pdf_path)
    write_csv(csv_path, fieldnames, rows)
    manifest = {
        "status": "preview_not_for_citation",
        "purpose": "replace only the IVM series in the existing active-candidate 1x2 figure",
        "formal_grid_conclusion": "no_unique_winner",
        "selection_role": "post-hoc engineering follow-up configuration",
        "selected_cell_id": CELL_ID,
        "selected_controls": grid_info["controls"],
        "grid_ivm": grid_info,
        "baseline_policy": "all five plotted baseline record vertices copied unchanged from the registered source CSV",
        "comparisons": comparisons,
        "inputs": {
            "registered_source_csv": {
                "path": str(REGISTERED_SOURCE),
                "sha256": EXPECTED_HASHES[REGISTERED_SOURCE],
            },
            "metric_preregistration": {
                "path": str(METRIC_PREREGISTRATION),
                "sha256": EXPECTED_HASHES[METRIC_PREREGISTRATION],
            },
            "metric_analysis": {
                "path": str(METRIC_ANALYSIS),
                "sha256": EXPECTED_HASHES[METRIC_ANALYSIS],
            },
            "plotter": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "base_plotter": {"path": str(BASE_PLOTTER), "sha256": sha256(BASE_PLOTTER)},
        },
        "outputs": {
            "pdf": {"path": str(pdf_path), "sha256": sha256(pdf_path)},
            "source_csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pdf": str(pdf_path), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
