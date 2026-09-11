#!/usr/bin/env python3
"""Audited medium-alpha IVM efficiency well for Chignolin."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


FORCE_BUDGETS = (1002, 3002, 6002)
FOCUS_ALPHA_LIMITS = (0.03, 100.0)
SELECTED_ALPHA = 1.0
IVM_BLUE = "#0072B2"
BUDGET_COLORS = ("#0072B2", "#D55E00", "#009E73")
BUDGET_MARKERS = ("o", "s", "^")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(saved, key: str):
    value = np.asarray(saved[key])
    return value.item() if value.ndim == 0 else value


def geometric_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Geometric-mean inputs must be positive and finite")
    return float(np.exp(np.mean(np.log(values))))


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def clean_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(
        which="major", color="#D9D9D9", lw=0.45, alpha=0.65, zorder=0
    )
    axis.tick_params(direction="out", length=3.0, width=0.7)


def panel_label(axis, label: str) -> None:
    axis.text(
        -0.02,
        1.04,
        label,
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=9.5,
        va="bottom",
    )


def save_figure(figure, output_dir: Path, stem: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 600}),
    ):
        path = output_dir / f"{stem}{suffix}"
        figure.savefig(path, **kwargs)
        outputs.append(str(path.resolve()))
    return outputs


def load_namespace(root: Path, namespace: str) -> list[dict]:
    report_dir = root / "reports" / namespace
    manifest_path = report_dir / "manifest.json"
    dense_manifest_path = report_dir / "dense_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dense_manifest = json.loads(dense_manifest_path.read_text())
    dense_by_alpha = {
        float(record["alpha"]): record
        for record in dense_manifest["records"]
    }
    output = []
    for record in manifest["records"]:
        alpha = float(record["alpha"])
        if not record.get("valid", False):
            raise RuntimeError(f"Invalid run in {namespace}: alpha={alpha:g}")
        result = Path(record["result_file"]).resolve()
        if sha256(result) != record["result_sha256"]:
            raise RuntimeError(f"Optimization result changed: {result}")
        dense_record = dense_by_alpha[alpha]
        dense_result = Path(dense_record["result_file"]).resolve()
        if sha256(dense_result) != dense_record["result_sha256"]:
            raise RuntimeError(f"Dense result changed: {dense_result}")
        with np.load(result, allow_pickle=False) as saved:
            counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
            number_force = counters.sum(axis=1)
            i_perp = np.asarray(saved["i_perp"], dtype=float)
            initial_hash = str(scalar(saved, "initial_path_sha256"))
            active_alpha = np.unique(
                np.asarray(saved["active_alpha"], dtype=float)
            )
            alpha_mode = str(scalar(saved, "alpha_mode"))
            globalization = str(scalar(saved, "state_globalization"))
            tangent = str(scalar(saved, "tangent_definition"))
        audit = json.loads(Path(record["audit_file"]).read_text())
        if (
            audit["backend_raw_joint_evaluations"]
            != audit["sum_image_joint_evaluations"]
            or alpha_mode != "fixed"
            or globalization != "proximal_only"
            or active_alpha.shape != (1,)
            or active_alpha[0] != alpha
            or "central difference" not in tangent.lower()
            or np.any(np.diff(counters, axis=0) < 0)
        ):
            raise RuntimeError(f"Optimization audit failed: alpha={alpha:g}")
        with np.load(dense_result, allow_pickle=False) as saved:
            dense_calls = np.asarray(saved["calls"], dtype=int)
            dense_barrier = np.asarray(
                saved["adaptive_barrier_kcal"], dtype=float
            )
            dense_subdivisions = np.asarray(
                saved["selected_subdivisions"], dtype=int
            )
        checkpoint_by_call = {
            int(row["selected_call_per_image"]): row
            for row in dense_record["checkpoints"]
        }
        dense_force = np.asarray(
            [
                checkpoint_by_call[int(call)][
                    "actual_number_of_force_evaluations"
                ]
                for call in dense_calls
            ],
            dtype=int,
        )
        dense_audit = json.loads(Path(dense_record["audit_file"]).read_text())
        if (
            dense_audit["diagnostic_backend_raw_joint_evaluations"]
            != dense_audit["diagnostic_evaluator_cache_misses"]
            or not dense_audit["counts_excluded_from_optimization_axis"]
        ):
            raise RuntimeError(
                f"Dense diagnostic audit failed: alpha={alpha:g}"
            )
        output.append(
            {
                "alpha": alpha,
                "result": result,
                "dense_result": dense_result,
                "number_force": number_force,
                "i_perp": i_perp,
                "dense_force": dense_force,
                "dense_barrier": dense_barrier,
                "dense_subdivisions": dense_subdivisions,
                "initial_hash": initial_hash,
                "optimization_evaluations": int(
                    audit["backend_raw_joint_evaluations"]
                ),
                "diagnostic_evaluations": int(
                    dense_audit[
                        "diagnostic_backend_raw_joint_evaluations"
                    ]
                ),
            }
        )
    return output


def draw_well(
    alphas: np.ndarray,
    i_by_budget: np.ndarray,
    barrier_by_budget: np.ndarray,
    output_dir: Path,
    y_scale: str,
) -> list[str]:
    setup_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    values = (i_by_budget, barrier_by_budget)
    labels = (
        r"Best attained $I_\perp$",
        r"Lowest densified barrier (kcal mol$^{-1}$)",
    )
    for index, (axis, measured, label) in enumerate(
        zip(axes, values, labels)
    ):
        for budget_index, budget in enumerate(FORCE_BUDGETS):
            axis.plot(
                alphas,
                measured[:, budget_index],
                color=BUDGET_COLORS[budget_index],
                marker=BUDGET_MARKERS[budget_index],
                ms=4.1,
                markerfacecolor="white",
                markeredgewidth=0.9,
                lw=1.65,
                zorder=3 + budget_index,
                label=rf"$N_F\leq {budget:,}$",
            )
        axis.axvline(
            SELECTED_ALPHA,
            color="#4D4D4D",
            ls=":",
            lw=1.05,
            alpha=0.85,
            zorder=2,
            label=r"Selected $\alpha=1$",
        )
        axis.set_xscale("log")
        axis.set_yscale(y_scale)
        if y_scale == "log" and index == 1:
            axis.yaxis.set_major_locator(
                mpl.ticker.FixedLocator(
                    [20.0, 24.0, 30.0, 40.0, 50.0]
                )
            )
            axis.yaxis.set_major_formatter(
                mpl.ticker.FormatStrFormatter("%.0f")
            )
            axis.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        axis.set_xlim(FOCUS_ALPHA_LIMITS)
        axis.set_xlabel(r"Fixed proximal parameter $\alpha$")
        axis.set_ylabel(label)
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    handles, labels_legend = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels_legend,
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.3,
        handlelength=2.2,
        handletextpad=0.5,
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.19,
        top=0.79,
        wspace=0.30,
    )
    stem = f"chignolin_ivm_fixed_alpha_well_{y_scale}y"
    outputs = save_figure(figure, output_dir, stem)
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--report-namespaces",
        default=(
            "ivm_fixed_alpha_proximal_only_20260728a,"
            "ivm_fixed_alpha_well_fill_20260728a"
        ),
    )
    parser.add_argument(
        "--high-alpha-namespaces",
        default=(
            "ivm_fixed_alpha_large_probe_20260728a,"
            "ivm_fixed_alpha_large_long_20260728a"
        ),
    )
    parser.add_argument(
        "--output-directory",
        default="figures/ivm_alpha_well_20260728a",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output_dir = root / args.output_directory
    namespaces = [
        item.strip()
        for item in args.report_namespaces.split(",")
        if item.strip()
    ]
    runs = []
    for namespace in namespaces:
        runs.extend(load_namespace(root, namespace))
    by_alpha = {}
    for run in runs:
        if FOCUS_ALPHA_LIMITS[0] <= run["alpha"] <= FOCUS_ALPHA_LIMITS[1]:
            if run["alpha"] in by_alpha:
                raise RuntimeError(f"Duplicate focus alpha={run['alpha']:g}")
            by_alpha[run["alpha"]] = run
    if len(by_alpha) < 12 or SELECTED_ALPHA not in by_alpha:
        raise RuntimeError("Insufficient independent alpha coverage")
    initial_hashes = {run["initial_hash"] for run in by_alpha.values()}
    if len(initial_hashes) != 1:
        raise RuntimeError("Fixed-alpha runs do not share one initial path")

    rows = []
    for alpha, run in sorted(by_alpha.items()):
        i_values = []
        barrier_values = []
        barrier_subdivisions = []
        for budget in FORCE_BUDGETS:
            eligible = np.flatnonzero(run["number_force"] <= budget)
            if eligible.size == 0:
                raise RuntimeError(
                    f"No optimization state under N_F={budget}: alpha={alpha:g}"
                )
            i_values.append(
                float(np.min(run["i_perp"][eligible]))
            )
            dense_eligible = np.flatnonzero(run["dense_force"] <= budget)
            if dense_eligible.size == 0:
                raise RuntimeError(
                    f"No dense state under N_F={budget}: alpha={alpha:g}"
                )
            eligible_barriers = run["dense_barrier"][dense_eligible]
            best_dense = int(
                dense_eligible[int(np.argmin(eligible_barriers))]
            )
            barrier_values.append(float(eligible_barriers.min()))
            barrier_subdivisions.append(
                int(run["dense_subdivisions"][best_dense])
            )
        i_score = float(i_values[-1])
        barrier_score = float(barrier_values[-1])
        row = {
            "alpha": alpha,
            "i_perp_score": i_score,
            "lowest_dense_barrier_kcal_nf_le_6002": barrier_score,
            "optimization_result": str(run["result"]),
            "optimization_sha256": sha256(run["result"]),
            "dense_result": str(run["dense_result"]),
            "dense_sha256": sha256(run["dense_result"]),
        }
        for budget, i_value, barrier_value, subdivisions in zip(
            FORCE_BUDGETS,
            i_values,
            barrier_values,
            barrier_subdivisions,
        ):
            row[f"best_i_perp_nf_le_{budget}"] = i_value
            row[
                f"lowest_measured_dense_barrier_kcal_nf_le_{budget}"
            ] = barrier_value
            row[
                f"barrier_subdivisions_at_best_nf_le_{budget}"
            ] = subdivisions
        rows.append(row)

    alphas = np.asarray([row["alpha"] for row in rows], dtype=float)
    i_by_budget = np.asarray(
        [
            [
                row[f"best_i_perp_nf_le_{budget}"]
                for budget in FORCE_BUDGETS
            ]
            for row in rows
        ],
        dtype=float,
    )
    barrier_by_budget = np.asarray(
        [
            [
                row[
                    f"lowest_measured_dense_barrier_kcal_nf_le_{budget}"
                ]
                for budget in FORCE_BUDGETS
            ]
            for row in rows
        ],
        dtype=float,
    )
    log_outputs = draw_well(
        alphas,
        i_by_budget,
        barrier_by_budget,
        output_dir,
        "log",
    )
    linear_outputs = draw_well(
        alphas,
        i_by_budget,
        barrier_by_budget,
        output_dir,
        "linear",
    )

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    table = data_dir / "chignolin_ivm_fixed_alpha_well.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    high_alpha_records = []
    high_namespaces = [
        item.strip()
        for item in args.high_alpha_namespaces.split(",")
        if item.strip()
    ]
    for namespace in high_namespaces:
        manifest_path = root / "reports" / namespace / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["records"]:
            result = Path(record["result_file"]).resolve()
            with np.load(result, allow_pickle=False) as saved:
                number_force = np.asarray(
                    saved["counter_vectors"], dtype=int
                ).sum(axis=1)
                i_perp = np.asarray(saved["i_perp"], dtype=float)
                calls = np.asarray(saved["calls_per_image"], dtype=int)
                stop_reason = str(scalar(saved, "stop_reason"))
                run_budget = int(scalar(saved, "budget"))
            high_alpha_records.append(
                {
                    "namespace": namespace,
                    "alpha": float(record["alpha"]),
                    "budget_calls_per_image": run_budget,
                    "final_max_calls_per_internal_image": int(calls[-1]),
                    "final_number_of_force_evaluations": int(
                        number_force[-1]
                    ),
                    "minimum_i_perp": float(np.min(i_perp)),
                    "minimum_i_perp_number_of_force_evaluations": int(
                        number_force[int(np.argmin(i_perp))]
                    ),
                    "final_i_perp": float(i_perp[-1]),
                    "stop_reason": stop_reason,
                    "result": str(result),
                    "result_sha256": sha256(result),
                }
            )
    high_table = data_dir / "chignolin_ivm_high_alpha_saturation.csv"
    with high_table.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(high_alpha_records[0])
        )
        writer.writeheader()
        writer.writerows(high_alpha_records)

    summary = {
        "figure_files": {
            "log_y": log_outputs,
            "linear_y": linear_outputs,
        },
        "raw_data_file": str(table.resolve()),
        "high_alpha_saturation_file": str(high_table.resolve()),
        "focus_alpha_limits": list(FOCUS_ALPHA_LIMITS),
        "selected_alpha": SELECTED_ALPHA,
        "force_budgets": list(FORCE_BUDGETS),
        "metric_definition": {
            "i_perp": (
                "For each displayed total-force budget, the minimum I_perp "
                "actually attained at or below that budget."
            ),
            "barrier": (
                "For each displayed total-force budget, the lowest "
                "independently densified raw 0 K forward barrier measured at "
                "or below that budget."
            ),
        },
        "plot_definition": {
            "x_scale": "log alpha",
            "connection": (
                "Within each force budget, adjacent measured alpha values are "
                "connected directly in ascending-alpha order."
            ),
            "regression": False,
            "smoothing": False,
            "interpolation": False,
            "same_raw_curves_used_for_log_and_linear_y_figures": True,
        },
        "optimization_axis": (
            "total joint energy-force cache misses over all 22 images"
        ),
        "diagnostic_calls_excluded_from_optimization_budgets": True,
        "initial_path_sha256": next(iter(initial_hashes)),
        "optimization_joint_evaluations_focus_runs": int(
            sum(run["optimization_evaluations"] for run in by_alpha.values())
        ),
        "diagnostic_joint_evaluations_focus_runs": int(
            sum(run["diagnostic_evaluations"] for run in by_alpha.values())
        ),
        "theory_interpretation": (
            "The article theorem is conditional asymptotic convergence for "
            "the exact unreparameterized path-space iteration at any fixed "
            "alpha > 0. It is not an unconditional rate or finite-inner-solver "
            "guarantee. In this implementation alpha >= O(10^2) approaches a "
            "trust-cap-dominated saturation regime, so those values are kept "
            "outside the main medium-alpha connected curves rather than "
            "portrayed as divergent."
        ),
    }
    summary_path = output_dir / "chignolin_ivm_fixed_alpha_well_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
