#!/usr/bin/env python3
"""Plot the registered Ala fixed-alpha comparison as a 2x3 CTP figure.

The columns form two controlled comparisons.  The first and second columns
hold r=0.1 and the implementation safety cap fixed while changing only
whether reparameterization is applied.  The second and third columns retain
reparameterization while changing the inner stopping rule.  The upper and
lower rows report raw I_perp and the independently densified energy barrier.

This is deliberately a thin companion to the registered 2x2 plotter.  It
imports that module for frozen inputs, dense-metric validation, styling, and
strict-record construction.  The only separate data loader is for the
exact-no-reparameterization family: every formally complete saved state is
included, whether or not it passes the mesh diagnostic.  Its observed mesh
status is nevertheless checked and written to the source table.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "plot_ala_alpha_cost_to_precision_2x2.py"
_SPEC = importlib.util.spec_from_file_location("ala_alpha_ctp_2x2", BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot import companion plotter: {BASE_PATH}")
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)


STEM = "alanine66d_alpha_cost_to_precision_2x3"
NOREPARAM_SPEC = {
    "residual_ratio": 0.1,
    "inner_max_steps": 5000,
    "total_internal_nf_budget": base.TOTAL_NF_BUDGET,
}
IPERP_XLIM = (1.0e3, 1.0e-4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--noreparam-metric-preregistration",
        type=Path,
        required=True,
        help=(
            "Registered dense-metric manifest for the r=0.1 "
            "exact-no-reparameterization alpha sweep."
        ),
    )
    parser.add_argument(
        "--noreparam-metric-aggregate",
        type=Path,
        required=True,
        help="Completed aggregate belonging to --noreparam-metric-preregistration.",
    )
    parser.add_argument(
        "--reparam-r01-metric-preregistration",
        "--left-metric-preregistration",
        dest="reparam_r01_metric_preregistration",
        type=Path,
        required=True,
        help=(
            "Registered dense-metric manifest for r=0.1, m_max=5000 with "
            "reparameterization."
        ),
    )
    parser.add_argument(
        "--reparam-r01-metric-aggregate",
        "--left-metric-aggregate",
        dest="reparam_r01_metric_aggregate",
        type=Path,
        required=True,
        help=(
            "Completed aggregate belonging to "
            "--reparam-r01-metric-preregistration."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=base.OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and summarize all three registered inputs without writing outputs.",
    )
    return parser.parse_args()


def validate_noreparam_result_identity(
    saved: np.lib.npyio.NpzFile,
    alpha: float,
) -> None:
    required = {
        "alpha": alpha,
        "alpha_mode": "fixed",
        "discrete_residual_ratio": NOREPARAM_SPEC["residual_ratio"],
        "discrete_inner_max_steps": NOREPARAM_SPEC["inner_max_steps"],
        "reparam_every": 0,
        "total_internal_nf_budget": NOREPARAM_SPEC["total_internal_nf_budget"],
        "exact_no_reparam_enabled": True,
    }
    for key, expected in required.items():
        observed = base.scalar(saved, key)
        if isinstance(expected, str):
            matched = str(observed) == expected
        elif isinstance(expected, bool):
            matched = bool(observed) is expected
        else:
            matched = base.same_number(observed, expected)
        if not matched:
            raise base.FigureInputError(
                f"Exact-no-reparameterization result mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            )
    if "state_reparam_attempts" in saved.files:
        if int(base.scalar(saved, "state_reparam_attempts")) != 0:
            raise base.FigureInputError(
                "The no-reparameterization result records reparameterization attempts"
            )
    if "reparam_committed" in saved.files:
        if np.any(np.asarray(saved["reparam_committed"], dtype=bool)):
            raise base.FigureInputError(
                "The no-reparameterization result contains committed reparameterizations"
            )


def dense_metric_for_state(
    state: Mapping[str, Any],
    task_specs: Mapping[str, Mapping[str, Any]],
    attested_hashes: Mapping[str, str],
) -> tuple[float, float, str, Path, str]:
    task_id = str(state.get("path_task_id", ""))
    task = task_specs.get(task_id)
    if task is None:
        raise base.FigureInputError(f"No dense metric task for {task_id}")
    artifact_path = Path(str(task.get("artifact", ""))).resolve()
    if not artifact_path.is_file():
        raise base.FigureInputError(f"Dense metric artifact is missing: {artifact_path}")
    artifact_hash = base.sha256(artifact_path)
    if attested_hashes and attested_hashes.get(task_id) != artifact_hash:
        raise base.FigureInputError(
            f"Dense metric artifact is not task-report-attested: {task_id}"
        )
    artifact = base.load_json(artifact_path)
    metric = artifact.get("metric")
    if not isinstance(metric, Mapping):
        raise base.FigureInputError(f"Dense metric payload is missing: {artifact_path}")
    if artifact.get("status") != "complete" or artifact.get("task_id") != task_id:
        raise base.FigureInputError(f"Dense metric task identity changed: {task_id}")
    if artifact.get("path_sha256") != state.get("path_sha256"):
        raise base.FigureInputError(f"Dense metric path identity changed: {task_id}")
    if metric.get("converged") is not True:
        raise base.FigureInputError(f"Dense metric did not converge: {task_id}")
    if int(metric.get("selected_subdivisions", -1)) != base.DENSE_SUBDIVISIONS:
        raise base.FigureInputError(f"Dense metric did not select level 160: {task_id}")
    if int(metric.get("optimization_nf_charged", -1)) != 0:
        raise base.FigureInputError(f"Dense diagnostic was charged to N_F: {task_id}")
    difference = float(
        metric.get("final_level_difference_kcal_per_mol", math.inf)
    )
    if not math.isfinite(difference) or difference > base.DENSE_CONVERGENCE_TOL:
        raise base.FigureInputError(f"Dense metric convergence check failed: {task_id}")
    barrier = float(metric.get("selected_barrier_kcal_per_mol", math.nan))
    if not math.isfinite(barrier) or barrier <= 0.0:
        raise base.FigureInputError(f"Invalid densified barrier: {task_id}")
    return barrier, difference, task_id, artifact_path, artifact_hash


def load_noreparam_run_states(
    run: Mapping[str, Any],
    task_specs: Mapping[str, Mapping[str, Any]],
    attested_hashes: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    controls = run.get("controls")
    if not isinstance(controls, Mapping):
        raise base.FigureInputError("No-reparameterization run lacks controls")
    alpha = float(controls["alpha"])
    result_path = Path(str(run.get("source_result", ""))).resolve()
    expected_result_hash = str(run.get("source_result_sha256", ""))
    if not result_path.is_file() or not expected_result_hash:
        raise base.FigureInputError(
            f"Registered no-reparameterization result is missing for alpha={alpha:g}"
        )
    if base.sha256(result_path) != expected_result_hash:
        raise base.FigureInputError(f"Optimizer result hash changed: {result_path}")

    registered_states = run.get("states")
    if not isinstance(registered_states, list) or not registered_states:
        raise base.FigureInputError(
            f"No formally complete no-reparameterization states for alpha={alpha:g}"
        )
    formal_count = int(run.get("formal_complete_state_count", -1))
    if formal_count != len(registered_states):
        raise base.FigureInputError(
            "The no-reparameterization metric manifest does not contain every "
            f"formally complete state for alpha={alpha:g}: "
            f"registered={len(registered_states)}, formal={formal_count}"
        )
    cap_evidence = run.get("inner_cap_stop_evidence")
    if (
        not isinstance(cap_evidence, Mapping)
        or int(cap_evidence.get("inner_cap_stop_count", -1)) != 0
        or cap_evidence.get("inner_cap_binding") is not False
    ):
        raise base.FigureInputError(
            "The no-reparameterization m_max=5000 safety cap was not "
            "certified nonbinding"
        )

    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, str]] = []
    seen_indices: set[int] = set()
    seen_state_ids: set[str] = set()
    with np.load(result_path, allow_pickle=False) as saved:
        validate_noreparam_result_identity(saved, alpha)
        paths = np.asarray(saved["stored_paths"], dtype=float)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=int)
        force_cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        mesh_cv = np.asarray(saved["segment_length_cv"], dtype=float)
        mesh_ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        complete_mask = np.asarray(
            saved["exact_no_reparam_complete_state_mask"], dtype=bool
        )
        lengths = {
            len(paths), len(stored_calls), len(force_cost), len(i_perp),
            len(mesh_cv), len(mesh_ratio), len(complete_mask),
        }
        if len(lengths) != 1:
            raise base.FigureInputError(
                f"No-reparameterization optimizer arrays are misaligned: {result_path}"
            )
        formal_indices = set(np.flatnonzero(complete_mask).tolist())
        registered_indices = {
            int(state.get("trajectory_index", -1)) for state in registered_states
        }
        if registered_indices != formal_indices:
            raise base.FigureInputError(
                "No-reparameterization registered states are not exactly the "
                "formal-complete mask"
            )

        for state in registered_states:
            index = int(state.get("trajectory_index", -1))
            state_id = str(state.get("state_id", ""))
            if (
                index < 0
                or index >= len(paths)
                or not state_id
                or index in seen_indices
                or state_id in seen_state_ids
            ):
                raise base.FigureInputError(
                    "Invalid or duplicated no-reparameterization state identity"
                )
            seen_indices.add(index)
            seen_state_ids.add(state_id)
            if state.get("formal_complete") is not True or not bool(complete_mask[index]):
                raise base.FigureInputError(
                    "No-reparameterization metric manifest includes a noncomplete state"
                )

            observed_mesh_eligible = bool(
                float(mesh_cv[index]) <= base.MESH_CV_MAX
                and float(mesh_ratio[index]) <= base.MESH_RATIO_MAX
            )
            if not isinstance(state.get("mesh_eligible"), bool):
                raise base.FigureInputError(
                    "No-reparameterization state lacks an explicit mesh_eligible diagnostic"
                )
            if bool(state["mesh_eligible"]) != observed_mesh_eligible:
                raise base.FigureInputError(
                    f"mesh_eligible differs from the stored mesh diagnostics at index {index}"
                )
            if base.array_sha256(paths[index]) != str(state.get("path_sha256", "")):
                raise base.FigureInputError("Registered path SHA-256 differs from stored_paths")
            if int(stored_calls[index]) != int(state.get("calls_per_image", -1)):
                raise base.FigureInputError(f"Stored call differs at trajectory index {index}")
            if int(force_cost[index]) != int(state.get("optimization_nf", -1)):
                raise base.FigureInputError(f"N_F differs at trajectory index {index}")
            if not base.same_number(i_perp[index], state.get("raw_i_perp"), atol=0.0):
                raise base.FigureInputError(f"Raw I_perp differs at trajectory index {index}")
            if not base.same_number(
                mesh_cv[index], state.get("segment_length_cv"), atol=0.0
            ):
                raise base.FigureInputError(f"Mesh CV differs at trajectory index {index}")
            if not base.same_number(
                mesh_ratio[index], state.get("segment_length_ratio"), atol=0.0
            ):
                raise base.FigureInputError(f"Mesh ratio differs at trajectory index {index}")

            barrier, difference, task_id, artifact_path, artifact_hash = (
                dense_metric_for_state(state, task_specs, attested_hashes)
            )
            raw_i_perp = float(i_perp[index])
            if not math.isfinite(raw_i_perp) or raw_i_perp <= 0.0:
                raise base.FigureInputError(
                    f"Invalid raw I_perp at trajectory index {index}"
                )
            rows.append(
                {
                    "setting": "exact_no_reparameterization",
                    "alpha": alpha,
                    "optimization_nf": int(force_cost[index]),
                    "trajectory_index": index,
                    "stored_call": int(stored_calls[index]),
                    "state_id": state_id,
                    "raw_i_perp": raw_i_perp,
                    "densified_delta_e_kcal_per_mol": barrier,
                    "path_sha256": str(state["path_sha256"]),
                    "segment_length_cv": float(mesh_cv[index]),
                    "segment_length_ratio": float(mesh_ratio[index]),
                    "mesh_eligible": observed_mesh_eligible,
                    "result_file": str(result_path),
                    "result_sha256": expected_result_hash,
                    "metric_task_id": task_id,
                    "metric_artifact": str(artifact_path),
                    "metric_artifact_sha256": artifact_hash,
                    "dense_subdivisions": base.DENSE_SUBDIVISIONS,
                    "dense_final_level_difference_kcal_per_mol": difference,
                }
            )
            artifacts.append(
                {"task_id": task_id, "path": str(artifact_path), "sha256": artifact_hash}
            )

    mesh_count = sum(bool(row["mesh_eligible"]) for row in rows)
    recorded_mesh_count = run.get("complete_mesh_eligible_state_count")
    if recorded_mesh_count is not None and int(recorded_mesh_count) != mesh_count:
        raise base.FigureInputError(
            "No-reparameterization mesh-eligible count differs from the manifest"
        )
    return rows, {
        "alpha": alpha,
        "controls": dict(controls),
        "result": {"path": str(result_path), "sha256": expected_result_hash},
        "inner_cap_stop_evidence": dict(cap_evidence),
        "formal_complete_path_count": len(rows),
        "mesh_eligible_path_count": mesh_count,
        "mesh_ineligible_path_count": len(rows) - mesh_count,
        "eligible_path_metric_artifacts": artifacts,
    }


def load_noreparam_setting(
    preregistration_path: Path,
    aggregate_path: Path,
) -> dict[str, Any]:
    preregistration_path = preregistration_path.expanduser().resolve()
    aggregate_path = aggregate_path.expanduser().resolve()
    preregistration = base.load_json(preregistration_path)
    base.validate_dense_policy(preregistration)
    base.validate_reference_barrier(preregistration)
    aggregate = base.validate_aggregate(
        preregistration_path, aggregate_path, preregistration
    )
    selected = base.select_alpha_runs(
        preregistration,
        NOREPARAM_SPEC,
        "exact no reparameterization",
    )
    raw_tasks = preregistration.get("tasks")
    if not isinstance(raw_tasks, list):
        raise base.FigureInputError("Metric preregistration has no dense task list")
    task_specs = {str(task["task_id"]): task for task in raw_tasks}
    attested_hashes = base.host_artifact_hashes(
        preregistration_path, preregistration, aggregate
    )

    curves: dict[float, dict[str, list[dict[str, Any]]]] = {}
    runs: list[dict[str, Any]] = []
    all_states: list[dict[str, Any]] = []
    for run in selected:
        states, run_info = load_noreparam_run_states(
            run, task_specs, attested_hashes
        )
        alpha = float(run_info["alpha"])
        i_records = base.strict_records(states, "raw_i_perp")
        barrier_records = base.strict_records(
            states, "densified_delta_e_kcal_per_mol"
        )
        run_info["raw_i_perp_record_count"] = len(i_records)
        run_info["densified_delta_e_record_count"] = len(barrier_records)
        curves[alpha] = {
            "raw_i_perp": i_records,
            "densified_delta_e_kcal_per_mol": barrier_records,
        }
        runs.append(run_info)
        all_states.extend(states)

    undershoots = [
        state
        for state in all_states
        if float(state["densified_delta_e_kcal_per_mol"])
        < base.REFERENCE_BARRIER_KCAL_PER_MOL - base.REFERENCE_ABS_TOL
    ]
    if undershoots:
        minimum = min(
            float(state["densified_delta_e_kcal_per_mol"])
            for state in undershoots
        )
        raise base.FigureInputError(
            "The exact-no-reparameterization data undershoot the reference "
            f"densified barrier (minimum Delta E={minimum:.15g}). Raw Delta E "
            "would incorrectly reward underestimation; the lower row must be "
            "changed to abs(Delta E-Delta E_ref) and labeled densified barrier error."
        )

    return {
        "setting": "exact_no_reparameterization",
        "specification": dict(NOREPARAM_SPEC),
        "selection": (
            "all formal-complete saved states; mesh_eligible is audited and "
            "reported but is not a selection gate"
        ),
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": base.sha256(preregistration_path),
        },
        "aggregate": {
            "path": str(aggregate_path),
            "sha256": base.sha256(aggregate_path),
        },
        "registered_code": base.registered_code_files(preregistration),
        "runs": runs,
        "curves": curves,
    }


def style_axis(axis: plt.Axes, metric: str, column: int) -> None:
    base.style_axis(axis, metric)
    if metric == "raw_i_perp":
        axis.set_xlim(*IPERP_XLIM)
    if column:
        axis.set_ylabel("")
        axis.tick_params(axis="y", labelleft=False)


def draw_figure(
    noreparam: Mapping[str, Any],
    tight_reparam: Mapping[str, Any],
    production: Mapping[str, Any],
    output: Path,
) -> None:
    base.configure_style()
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(265 / 25.4, 118 / 25.4),
        sharey=True,
    )
    settings = (noreparam, tight_reparam, production)
    metrics = ("raw_i_perp", "densified_delta_e_kcal_per_mol")
    for column, setting in enumerate(settings):
        for row, metric in enumerate(metrics):
            axis = axes[row, column]
            for alpha in base.ALPHAS:
                base.draw_curve(axis, setting["curves"][alpha][metric], alpha, metric)
            style_axis(axis, metric, column)

    titles = (
        "Without reparameterization\n"
        + r"($r=0.1$, $m_{\max}=5000$; cap nonbinding)",
        "With reparameterization\n"
        + r"($r=0.1$, $m_{\max}=5000$; cap nonbinding)",
        "With reparameterization\n" + r"($r=0.7$, $m_{\max}=5$)",
    )
    for axis, title in zip(axes[0], titles):
        axis.set_title(title, pad=6)

    handles = [
        Line2D(
            [0], [0],
            color=base.ALPHA_COLORS[index],
            linewidth=1.2,
            marker=base.ALPHA_MARKERS[index],
            markersize=3.5,
            markerfacecolor=base.ALPHA_COLORS[index],
            markeredgecolor="#202020",
            markeredgewidth=0.28,
            label=base.alpha_label(alpha),
        )
        for index, alpha in enumerate(base.ALPHAS)
    ]
    handles.append(
        Line2D(
            [0], [0],
            color="#686868",
            linestyle=":",
            linewidth=0.7,
            label=r"$\Delta E_{\mathrm{ref}}$",
        )
    )
    figure.legend(
        handles=handles,
        title=r"Fixed proximal parameter $\alpha$",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=9,
        frameon=False,
        handlelength=1.7,
        handletextpad=0.4,
        columnspacing=0.85,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.99,
        bottom=0.105,
        top=0.80,
        wspace=0.13,
        hspace=0.35,
    )
    figure.canvas.draw()
    for axis, label in zip(axes.ravel(), "abcdef"):
        base.panel_label(figure, axis, label)
    figure.savefig(
        output,
        format="pdf",
        dpi=600,
        metadata={
            "Title": "Alanine 66D fixed-alpha and reparameterization comparison",
            "Subject": (
                "Controlled comparison of no reparameterization, tighter "
                "reparameterized inner solves, and production inner solves"
            ),
        },
    )
    plt.close(figure)


def curve_rows(settings: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    setting_order = {
        "exact_no_reparameterization": 0,
        "tighter_inner_solves": 1,
        "production_inner_solves": 2,
    }
    rows: list[dict[str, Any]] = []
    for setting in settings:
        setting_name = str(setting["setting"])
        column = setting_order[setting_name]
        for alpha in base.ALPHAS:
            for row_index, metric in enumerate(
                ("raw_i_perp", "densified_delta_e_kcal_per_mol")
            ):
                panel = "abcdef"[row_index * 3 + column]
                for record in setting["curves"][alpha][metric]:
                    rows.append(
                        {
                            "panel": panel,
                            "setting": setting_name,
                            "alpha": f"{alpha:g}",
                            "residual_ratio": setting["specification"]["residual_ratio"],
                            "inner_max_steps": setting["specification"]["inner_max_steps"],
                            "reparameterization": setting_name
                            != "exact_no_reparameterization",
                            "metric": metric,
                            "metric_value": record[metric],
                            "running_best_metric": record["running_best_metric"],
                            "minimum_NF": record["optimization_nf"],
                            "trajectory_index": record["trajectory_index"],
                            "stored_call": record["stored_call"],
                            "state_id": record["state_id"],
                            "formal_complete": True,
                            "mesh_eligible": bool(record.get("mesh_eligible", True)),
                            "segment_length_cv": record["segment_length_cv"],
                            "segment_length_ratio": record["segment_length_ratio"],
                            "path_sha256": record["path_sha256"],
                            "result_file": record["result_file"],
                            "result_sha256": record["result_sha256"],
                            "metric_task_id": record["metric_task_id"],
                            "metric_artifact": record["metric_artifact"],
                            "metric_artifact_sha256": record["metric_artifact_sha256"],
                            "dense_subdivisions": (
                                record["dense_subdivisions"]
                                if metric == "densified_delta_e_kcal_per_mol"
                                else ""
                            ),
                            "dense_final_level_difference_kcal_per_mol": (
                                record["dense_final_level_difference_kcal_per_mol"]
                                if metric == "densified_delta_e_kcal_per_mol"
                                else ""
                            ),
                            "interpolation_used": False,
                            "extrapolation_used": False,
                            "carry_forward_used": False,
                            "diagnostic_calls_charged_to_NF": False,
                        }
                    )
    return rows


def public_setting(setting: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in setting.items() if key != "curves"}


def main() -> None:
    args = parse_args()
    base.validate_frozen_right_inputs()
    tight_optimizer = base.validate_frozen_left_optimizer()
    noreparam = load_noreparam_setting(
        args.noreparam_metric_preregistration,
        args.noreparam_metric_aggregate,
    )
    tight_reparam = base.load_setting(
        args.reparam_r01_metric_preregistration,
        args.reparam_r01_metric_aggregate,
        base.LEFT_SPEC,
        "tighter_inner_solves",
        tight_optimizer["runs_by_alpha"],
    )
    production = base.load_setting(
        base.RIGHT_PREREGISTRATION,
        base.RIGHT_AGGREGATE,
        base.RIGHT_SPEC,
        "production_inner_solves",
    )
    summary = {
        "noreparam_runs": len(noreparam["runs"]),
        "tight_reparam_runs": len(tight_reparam["runs"]),
        "production_runs": len(production["runs"]),
        "alpha_grid": [f"{alpha:g}" for alpha in base.ALPHAS],
        "noreparam_formal_complete_paths": sum(
            int(run["formal_complete_path_count"]) for run in noreparam["runs"]
        ),
        "noreparam_mesh_eligible_paths": sum(
            int(run["mesh_eligible_path_count"]) for run in noreparam["runs"]
        ),
        "noreparam_inner_cap_nonbinding": all(
            run["inner_cap_stop_evidence"]["inner_cap_binding"] is False
            for run in noreparam["runs"]
        ),
        "reference_barrier_kcal_per_mol": base.REFERENCE_BARRIER_KCAL_PER_MOL,
    }
    if args.validate_only:
        print(json.dumps(summary, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / f"{STEM}.pdf"
    csv_path = args.output_dir / f"{STEM}_source.csv"
    manifest_path = args.output_dir / f"{STEM}_manifest.json"
    existing = [path for path in (pdf_path, csv_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    rows = curve_rows((noreparam, tight_reparam, production))
    draw_figure(noreparam, tight_reparam, production, pdf_path)
    if not rows:
        raise base.FigureInputError("No plotted records were produced")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "status": "generated_from_completed_registered_metrics",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Controlled fixed-alpha comparison of reparameterization and the "
            "inner stopping rule in the discrete Ala IVM"
        ),
        "alpha_grid": [f"{alpha:g}" for alpha in base.ALPHAS],
        "columns": {
            "left": public_setting(noreparam),
            "middle": public_setting(tight_reparam),
            "right": public_setting(production),
        },
        "control_comparisons": {
            "left_vs_middle": (
                "r=0.1 and m_max=5000 fixed; reparameterization changes"
            ),
            "middle_vs_right": (
                "reparameterization fixed; inner stopping rule changes"
            ),
        },
        "rows": {
            "top": "raw I_perp",
            "bottom": "actual densified Delta E in kcal/mol",
        },
        "cost_to_precision_definition": {
            "no_reparameterization_paths": (
                "all metric-preregistered formal-complete saved paths; no mesh gate"
            ),
            "reparameterized_paths": (
                "metric-preregistered formal-complete and common-mesh-eligible "
                "saved paths"
            ),
            "equal_NF_rule": "retain the lowest actually observed metric value",
            "curve_vertices": "strictly decreasing observed records",
            "interpolation": False,
            "extrapolation": False,
            "carry_forward_to_budget": False,
            "optimization_cost_source": "total_internal_joint_evaluations",
            "offline_dense_diagnostics_charged": False,
        },
        "densified_delta_e": {
            "quantity": "actual barrier, not reference error",
            "reference_barrier_kcal_per_mol": base.REFERENCE_BARRIER_KCAL_PER_MOL,
            "reference_line_drawn": True,
            "no_reparameterization_undershoot_gate": (
                "fail if any formal-complete no-reparameterization path lies below "
                "Delta E_ref by more than 1e-12 kcal/mol; require barrier-error row"
            ),
            "piecewise_linear_cartesian_subdivisions_per_segment": (
                base.DENSE_SUBDIVISIONS
            ),
            "convergence_tolerance_kcal_per_mol": base.DENSE_CONVERGENCE_TOL,
        },
        "common_axes": {
            "number_of_forces": {"scale": "log", "limits": list(base.NF_YLIM)},
            "raw_i_perp": {"scale": "log reversed", "limits": list(IPERP_XLIM)},
            "densified_delta_e": {
                "scale": "log reversed",
                "limits": list(base.BARRIER_XLIM),
            },
        },
        "plotting_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.sha256(Path(__file__).resolve()),
            "imported_companion": {
                "path": str(BASE_PATH),
                "sha256": base.sha256(BASE_PATH),
            },
        },
        "outputs": {
            "pdf": {"path": str(pdf_path), "sha256": base.sha256(pdf_path)},
            "source_csv": {"path": str(csv_path), "sha256": base.sha256(csv_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                **summary,
                "pdf": str(pdf_path),
                "source_csv": str(csv_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
