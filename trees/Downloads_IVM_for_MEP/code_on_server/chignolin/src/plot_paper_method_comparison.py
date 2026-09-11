#!/usr/bin/env python3
"""Paper-grade Chignolin comparison on total path force evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


INITIAL_HASH = (
    "57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996"
)
METHOD_ORDER = (
    "IVM",
    "ODE-NEB",
    "Static-NEB",
    "Original String",
    "OBSM-FIRE",
)
COLORS = {
    "IVM": "#0072B2",
    "ODE-NEB": "#D55E00",
    "Static-NEB": "#B279A2",
    "Original String": "#777777",
    "OBSM-FIRE": "#5B9279",
}
LINESTYLES = {
    "IVM": "-",
    "ODE-NEB": "-",
    "Static-NEB": "--",
    "Original String": ":",
    "OBSM-FIRE": "-.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.5,
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


def load_run(path: Path, maximum_call: int = 500) -> dict:
    path = path.resolve()
    with np.load(path, allow_pickle=False) as saved:
        calls = np.asarray(saved["calls_per_image"], dtype=int)
        keep = calls <= int(maximum_call)
        counters = np.asarray(saved["counter_vectors"], dtype=int)[keep]
        data = {
            "path": path,
            "calls_per_image": calls[keep],
            "number_of_force_evaluations": counters.sum(axis=1),
            "counter_vectors": counters,
            "i_perp": np.asarray(saved["i_perp"], dtype=float)[keep],
            "barrier_kcal": np.asarray(
                saved["barrier_kcal"], dtype=float
            )[keep],
            "physical_fmax": np.asarray(
                saved["physical_fmax"], dtype=float
            )[keep],
            "equidistance_integral": np.asarray(
                saved["equidistance_integral"], dtype=float
            )[keep],
            "segment_length_cv": np.asarray(
                saved["segment_length_cv"], dtype=float
            )[keep],
            "energy_request_vectors": np.asarray(
                saved["energy_request_vectors"], dtype=int
            )[keep],
            "force_request_vectors": np.asarray(
                saved["force_request_vectors"], dtype=int
            )[keep],
            "cache_hit_vectors": np.asarray(
                saved["cache_hit_vectors"], dtype=int
            )[keep],
            "initial_path_sha256": str(saved["initial_path_sha256"]),
            "tangent_definition": str(saved["tangent_definition"]),
            "state_globalization": (
                str(saved["state_globalization"])
                if "state_globalization" in saved.files
                else "not_applicable"
            ),
            "result_sha256": sha256(path),
        }
    if not np.any(keep):
        raise RuntimeError(f"No records at call <= {maximum_call}: {path}")
    audit_path = path.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text())
    if (
        not audit.get("consistent", False)
        or audit.get("backend_raw_joint_evaluations")
        != audit.get("sum_image_joint_evaluations")
    ):
        raise RuntimeError(f"Counter audit failed for {path}")
    data["audit_path"] = audit_path
    data["audit_sha256"] = sha256(audit_path)
    return data


def load_dense(path: Path, run: dict) -> dict:
    path = path.resolve()
    with np.load(path, allow_pickle=False) as saved:
        calls = np.asarray(saved["calls"], dtype=int)
        barrier = np.asarray(saved["adaptive_barrier_kcal"], dtype=float)
        source_hash = str(saved["source_initial_path_sha256"])
        backend_calls = int(saved["diagnostic_backend_raw_joint_evaluations"])
        evaluator_misses = int(saved["diagnostic_evaluator_cache_misses"])
        subdivisions = np.asarray(
            saved["selected_subdivisions"], dtype=int
        )
    if source_hash != run["initial_path_sha256"]:
        raise RuntimeError(f"Dense initial-path mismatch in {path}")
    if backend_calls != evaluator_misses:
        raise RuntimeError(f"Dense diagnostic counter mismatch in {path}")
    number_force = []
    for call in calls:
        indices = np.flatnonzero(run["calls_per_image"] == int(call))
        if not indices.size:
            raise RuntimeError(
                f"Dense call {call} absent from source run {run['path']}"
            )
        number_force.append(
            int(run["number_of_force_evaluations"][indices[-1]])
        )
    return {
        "path": path,
        "calls_per_image": calls,
        "number_of_force_evaluations": np.asarray(number_force, dtype=int),
        "barrier_kcal": barrier,
        "selected_subdivisions": subdivisions,
        "diagnostic_joint_evaluations": backend_calls,
        "result_sha256": sha256(path),
    }


def first_passage(
    force_evaluations: np.ndarray,
    values: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    order = np.argsort(force_evaluations, kind="stable")
    x = np.asarray(force_evaluations, dtype=float)[order]
    y = np.asarray(values, dtype=float)[order]
    running_best = np.minimum.accumulate(y)
    output = np.full(np.asarray(thresholds).shape, np.nan, dtype=float)
    for index, threshold in enumerate(thresholds):
        reached = np.flatnonzero(running_best <= threshold)
        if reached.size:
            output[index] = x[reached[0]]
    return output


def log_auc(force_evaluations: np.ndarray, values: np.ndarray) -> float:
    x = np.asarray(force_evaluations, dtype=float)
    y = np.maximum(np.asarray(values, dtype=float), 1e-300)
    if x.size < 2 or x[-1] <= x[0]:
        raise ValueError("AUC requires an increasing multi-point cost axis")
    return float(np.trapz(np.log10(y), x) / (x[-1] - x[0]))


def pareto_frontier(
    force_evaluations: np.ndarray,
    values: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the measured first-passage frontier as connected vertices.

    Plotting a dense threshold grid repeats the same first-passage cost over
    many adjacent thresholds and therefore produces an artificial staircase.
    This representation keeps only the actual record-setting observations and
    joins those measured Pareto vertices by straight segments.
    """

    order = np.argsort(force_evaluations, kind="stable")
    costs = np.asarray(force_evaluations, dtype=float)[order]
    metric = np.asarray(values, dtype=float)[order]
    frontier_metric = []
    frontier_cost = []
    running_best = np.inf
    for cost, value in zip(costs, metric):
        if not np.isfinite(cost) or not np.isfinite(value):
            continue
        tolerance = 16.0 * np.finfo(float).eps * max(
            1.0, abs(running_best) if np.isfinite(running_best) else 1.0
        )
        if value < running_best - tolerance:
            running_best = float(value)
            if lower_bound <= running_best <= upper_bound:
                if frontier_cost and cost == frontier_cost[-1]:
                    frontier_metric[-1] = running_best
                else:
                    frontier_metric.append(running_best)
                    frontier_cost.append(float(cost))
    return (
        np.asarray(frontier_metric, dtype=float),
        np.asarray(frontier_cost, dtype=float),
    )


def add_speedup_arrow(
    axis,
    target: float,
    arrow_x: float,
    ivm_cost: float,
    baseline_cost: float,
    baseline_label: str,
    text_side: str,
) -> float:
    """Draw a black two-headed same-precision cost arrow.

    The arrow is shifted slightly away from the common target so that the
    ODE-NEB and OBSM-FIRE comparisons can be read independently.  Thin
    horizontal connectors preserve the exact same-threshold interpretation.
    """

    if text_side not in {"left", "right"}:
        raise ValueError("text_side must be left or right")
    speedup = float(baseline_cost) / float(ivm_cost)
    short_label = {
        "ODE-NEB": "vs ODE",
        "OBSM-FIRE": "vs OBSM",
    }[baseline_label]
    axis.plot(
        [target, arrow_x],
        [ivm_cost, ivm_cost],
        color="#111111",
        lw=0.65,
        zorder=9,
    )
    axis.plot(
        [target, arrow_x],
        [baseline_cost, baseline_cost],
        color="#111111",
        lw=0.65,
        zorder=9,
    )
    axis.annotate(
        "",
        xy=(arrow_x, baseline_cost),
        xytext=(arrow_x, ivm_cost),
        arrowprops={
            "arrowstyle": "<->",
            "color": "#111111",
            "lw": 0.95,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 8.0,
        },
        zorder=10,
    )
    axis.annotate(
        rf"{speedup:.1f}$\times$" + "\n" + short_label,
        xy=(arrow_x, 0.5 * (float(ivm_cost) + float(baseline_cost))),
        xytext=((-5 if text_side == "left" else 5), 0),
        textcoords="offset points",
        color="#111111",
        fontsize=6.4,
        fontweight="normal",
        ha=("right" if text_side == "left" else "left"),
        va="center",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 0.35,
        },
        zorder=11,
    )
    return speedup


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_selected(
    root: Path,
    ivm_result: Optional[Path] = None,
    ivm_dense_result: Optional[Path] = None,
) -> tuple[dict, dict, dict]:
    confirmed_production_manifest = json.loads(
        (
            root
            / "reports"
            / "confirmed1000_productions_20260728c"
            / "manifest.json"
        ).read_text()
    )
    confirmed_production = {
        row["name"]: Path(row["result_file"])
        for row in confirmed_production_manifest["records"]
    }
    legacy_dense_manifest = json.loads(
        (
            root
            / "reports"
            / "paper_total_nf_dense_20260727a"
            / "manifest.json"
        ).read_text()
    )
    legacy_dense_paths = {
        row["name"]: Path(row["result_file"])
        for row in legacy_dense_manifest["records"]
    }
    confirmed_dense_manifest = json.loads(
        (
            root
            / "reports"
            / "confirmed1000_dense_20260728c"
            / "manifest.json"
        ).read_text()
    )
    confirmed_dense_paths = {
        row["name"]: Path(row["result_file"])
        for row in confirmed_dense_manifest["records"]
    }
    selected_paths = {
        "IVM": (
            ivm_result
            if ivm_result is not None
            else root
            / "runs"
            / "ivm"
            / "ivm_ivm_equal_idpp_v52_fixed_alpha1_production_central_state_ivm_cfg31b81924ce51_budget500.npz"
        ),
        "ODE-NEB": (
            confirmed_production["confirmed1000_ode_k50"]
        ),
        "Static-NEB": confirmed_production[
            "confirmed1000_static_k40_a0p012575"
        ],
        "Original String": confirmed_production[
            "confirmed1000_original_dt0p012575"
        ],
        "OBSM-FIRE": confirmed_production[
            "confirmed1000_obsm_l0p9_n5_dt0p12"
        ],
    }
    selected_dense = {
        "IVM": (
            ivm_dense_result
            if ivm_dense_result is not None
            else legacy_dense_paths["ivm_alpha1"]
        ),
        "ODE-NEB": confirmed_dense_paths["confirmed1000_ode_k50"],
        "Static-NEB": confirmed_dense_paths[
            "confirmed1000_static_k40_a0p012575"
        ],
        "Original String": confirmed_dense_paths[
            "confirmed1000_original_dt0p012575"
        ],
        "OBSM-FIRE": confirmed_dense_paths[
            "confirmed1000_obsm_l0p9_n5_dt0p12"
        ],
    }
    runs = {
        method: load_run(
            path, maximum_call=(500 if method == "IVM" else 1000)
        )
        for method, path in selected_paths.items()
    }
    dense = {
        method: load_dense(selected_dense[method], runs[method])
        for method in METHOD_ORDER
    }
    hashes = {run["initial_path_sha256"] for run in runs.values()}
    if hashes != {INITIAL_HASH}:
        raise RuntimeError(f"Selected initial-path mismatch: {hashes}")
    if any(
        "central difference" not in run["tangent_definition"].lower()
        for run in runs.values()
    ):
        raise RuntimeError("A selected method does not declare central tangent")
    return confirmed_production_manifest, runs, dense


def plot_main(
    output_dir: Path, runs: dict, dense: dict, generated: dict
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for method in METHOD_ORDER:
        width = 2.2 if method == "IVM" else 1.35
        axes[0].plot(
            runs[method]["number_of_force_evaluations"],
            runs[method]["i_perp"],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            lw=width,
            zorder=5 if method == "IVM" else 3,
            label=method,
        )
        axes[1].plot(
            dense[method]["number_of_force_evaluations"],
            dense[method]["barrier_kcal"],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            marker="o",
            ms=2.0,
            markevery=max(1, len(dense[method]["barrier_kcal"]) // 10),
            lw=width,
            zorder=5 if method == "IVM" else 3,
        )
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel(r"Unnormalized $I_\perp$")
    axes[1].set_ylabel(
        r"Densified 0 K barrier $\Delta E$ (kcal mol$^{-1}$)"
    )
    maximum_force = max(
        int(runs[method]["number_of_force_evaluations"][-1])
        for method in METHOD_ORDER
    )
    for index, axis in enumerate(axes):
        axis.set_xlim(0, 1.005 * maximum_force)
        axis.set_xlabel("Number of Force Evaluations")
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        columnspacing=1.25,
        handlelength=2.7,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.19,
        top=0.79,
        wspace=0.29,
    )
    generated["main_metrics"] = save_figure(
        figure, output_dir, "chignolin_main_metrics_total_nf"
    )
    plt.close(figure)


def plot_cost_and_speedup(
    output_dir: Path,
    runs: dict,
    dense: dict,
    generated: dict,
    data_dir: Path,
) -> None:
    thresholds_i = np.logspace(4.5, -2.0, 320)
    all_dense_values = np.concatenate(
        [dense[method]["barrier_kcal"] for method in METHOD_ORDER]
    )
    # Match the cost-to-precision convention used by the manuscript:
    # precision threshold on x, first-passage cost on y.  The initial IDPP
    # barrier is thousands of kcal/mol and is not a useful precision target,
    # so the displayed barrier range starts at 100 kcal/mol and resolves the
    # scientifically relevant approach to the converged barrier.
    thresholds_b = np.linspace(
        min(100.0, float(np.max(all_dense_values))),
        19.5,
        320,
    )
    passages_i = {}
    passages_b = {}
    rows = []
    frontier_rows = []
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for method in METHOD_ORDER:
        passages_i[method] = first_passage(
            runs[method]["number_of_force_evaluations"],
            runs[method]["i_perp"],
            thresholds_i,
        )
        passages_b[method] = first_passage(
            dense[method]["number_of_force_evaluations"],
            dense[method]["barrier_kcal"],
            thresholds_b,
        )
        frontier_i, frontier_i_cost = pareto_frontier(
            runs[method]["number_of_force_evaluations"],
            runs[method]["i_perp"],
            float(np.min(thresholds_i)),
            float(np.max(thresholds_i)),
        )
        frontier_b, frontier_b_cost = pareto_frontier(
            dense[method]["number_of_force_evaluations"],
            dense[method]["barrier_kcal"],
            float(np.min(thresholds_b)),
            float(np.max(thresholds_b)),
        )
        for axis, frontier, frontier_cost in (
            (axes[0], frontier_i, frontier_i_cost),
            (axes[1], frontier_b, frontier_b_cost),
        ):
            axis.plot(
                frontier,
                frontier_cost,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                lw=(
                    2.25
                    if method == "IVM"
                    else 1.7
                    if method == "ODE-NEB"
                    else 1.15
                ),
                alpha=1.0 if method in ("IVM", "ODE-NEB") else 0.9,
                zorder=(
                    6
                    if method == "IVM"
                    else 5
                    if method == "ODE-NEB"
                    else 3
                ),
                label=method,
            )
        for metric, frontier, frontier_cost in (
            ("i_perp", frontier_i, frontier_i_cost),
            ("densified_barrier_kcal", frontier_b, frontier_b_cost),
        ):
            for threshold, force in zip(frontier, frontier_cost):
                frontier_rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "measured_record_threshold": float(threshold),
                        "number_of_force_evaluations": int(force),
                    }
                )
        for metric, thresholds, passage in (
            ("i_perp", thresholds_i, passages_i[method]),
            ("densified_barrier_kcal", thresholds_b, passages_b[method]),
        ):
            for threshold, force in zip(thresholds, passage):
                rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "target_threshold": float(threshold),
                        "minimum_number_of_force_evaluations": (
                            "" if not np.isfinite(force) else int(force)
                        ),
                    }
                )
    panels = (
        (r"$I_\perp$ threshold", True),
        (
            r"Densified barrier threshold (kcal mol$^{-1}$)",
            False,
        ),
    )
    maximum_force = max(
        int(runs[method]["number_of_force_evaluations"][-1])
        for method in METHOD_ORDER
    )
    for index, (axis, (xlabel, log_x)) in enumerate(zip(axes, panels)):
        if log_x:
            axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_ylim(0.0, 1.10 * maximum_force)
        axis.yaxis.set_major_formatter(
            mpl.ticker.StrMethodFormatter("{x:,.0f}")
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(
            "Minimum Number of\nForce Evaluations",
            labelpad=4,
        )
        axis.margins(x=0.035)
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + index))

    compared_methods = ("IVM", "ODE-NEB", "OBSM-FIRE")
    strictest_common_i = max(
        float(np.nanmin(runs[method]["i_perp"]))
        for method in compared_methods
    )
    strictest_common_b = max(
        float(np.nanmin(dense[method]["barrier_kcal"]))
        for method in compared_methods
    )
    annotation_rows = []
    for (
        axis,
        metric,
        target,
        source,
        value_key,
        arrow_positions,
    ) in (
        (
            axes[0],
            "i_perp",
            strictest_common_i,
            runs,
            "i_perp",
            {
                "ODE-NEB": strictest_common_i * 0.68,
                "OBSM-FIRE": strictest_common_i * 0.38,
            },
        ),
        (
            axes[1],
            "densified_barrier_kcal",
            strictest_common_b,
            dense,
            "barrier_kcal",
            {
                "ODE-NEB": strictest_common_b - 1.7,
                "OBSM-FIRE": strictest_common_b - 3.7,
            },
        ),
    ):
        ivm_cost = float(
            first_passage(
                source["IVM"]["number_of_force_evaluations"],
                source["IVM"][value_key],
                np.asarray([target]),
            )[0]
        )
        for baseline in ("ODE-NEB", "OBSM-FIRE"):
            baseline_cost = float(
                first_passage(
                    source[baseline]["number_of_force_evaluations"],
                    source[baseline][value_key],
                    np.asarray([target]),
                )[0]
            )
            arrow_x = arrow_positions[baseline]
            speedup = add_speedup_arrow(
                axis,
                target,
                arrow_x,
                ivm_cost,
                baseline_cost,
                baseline,
                "left" if baseline == "ODE-NEB" else "right",
            )
            annotation_rows.append(
                {
                    "metric": metric,
                    "shared_precision_threshold": target,
                    "baseline": baseline,
                    "ivm_number_of_force_evaluations": int(ivm_cost),
                    "baseline_number_of_force_evaluations": int(
                        baseline_cost
                    ),
                    "ivm_speedup_vs_baseline": speedup,
                }
            )
    handles, labels_legend = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels_legend,
        ncol=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        columnspacing=1.25,
        handlelength=2.7,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.21,
        top=0.79,
        wspace=0.34,
    )
    generated["cost_to_precision"] = save_figure(
        figure, output_dir, "chignolin_cost_to_precision_total_nf"
    )
    plt.close(figure)
    write_csv(data_dir / "chignolin_cost_to_precision_total_nf.csv", rows)
    write_csv(
        data_dir / "chignolin_cost_to_precision_frontier_total_nf.csv",
        frontier_rows,
    )
    write_csv(
        data_dir / "chignolin_cost_to_precision_annotations_total_nf.csv",
        annotation_rows,
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    speedup_rows = []
    for method in ("IVM", "Static-NEB", "Original String", "OBSM-FIRE"):
        for axis, thresholds, method_passage, ode_passage, metric in (
            (
                axes[0],
                thresholds_i,
                passages_i[method],
                passages_i["ODE-NEB"],
                "i_perp",
            ),
            (
                axes[1],
                thresholds_b,
                passages_b[method],
                passages_b["ODE-NEB"],
                "densified_barrier_kcal",
            ),
        ):
            valid = (
                np.isfinite(method_passage)
                & np.isfinite(ode_passage)
                & (method_passage > 0)
            )
            speedup = ode_passage[valid] / method_passage[valid]
            axis.plot(
                thresholds[valid],
                speedup,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                lw=2.2 if method == "IVM" else 1.35,
                label=method,
            )
            for threshold, value in zip(thresholds[valid], speedup):
                speedup_rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "target_threshold": float(threshold),
                        "speedup_vs_ode_neb": float(value),
                    }
                )
    for index, (axis, xlabel) in enumerate(
        zip(
            axes,
            (
                r"Target $I_\perp$",
                r"Target densified $\Delta E$ (kcal mol$^{-1}$)",
            ),
        )
    ):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.invert_xaxis()
        axis.axhline(1.0, color="#888888", lw=0.8, linestyle="--")
        axis.axhline(2.0, color="#BBBBBB", lw=0.65, linestyle=":")
        axis.axhline(3.0, color="#BBBBBB", lw=0.65, linestyle=":")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(r"Speedup vs ODE-NEB")
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    handles, labels_legend = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels_legend,
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.19,
        top=0.79,
        wspace=0.29,
    )
    generated["speedup_vs_ode"] = save_figure(
        figure, output_dir, "chignolin_speedup_vs_ode_total_nf"
    )
    plt.close(figure)
    write_csv(data_dir / "chignolin_speedup_vs_ode_total_nf.csv", speedup_rows)


def plot_diagnostics(
    output_dir: Path, runs: dict, generated: dict
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    specifications = (
        ("physical_fmax", r"Physical normal $f_{\max}$ (eV Å$^{-1}$)"),
        (
            "equidistance_integral",
            r"Equidistance integral $\mathcal{I}_{\rm eq}$",
        ),
        ("segment_length_cv", "Segment-length CV"),
    )
    for method in METHOD_ORDER:
        for axis, (key, _) in zip(axes, specifications):
            axis.plot(
                runs[method]["number_of_force_evaluations"],
                np.maximum(runs[method][key], 1e-24),
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                lw=2.1 if method == "IVM" else 1.25,
                label=method,
            )
    maximum_force = max(
        int(runs[method]["number_of_force_evaluations"][-1])
        for method in METHOD_ORDER
    )
    for index, (axis, (_, ylabel)) in enumerate(zip(axes, specifications)):
        axis.set_yscale("log")
        axis.set_xlim(0, 1.005 * maximum_force)
        axis.set_xlabel("Number of Force Evaluations")
        axis.set_ylabel(ylabel)
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        columnspacing=1.05,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.20,
        top=0.77,
        wspace=0.42,
    )
    generated["diagnostics"] = save_figure(
        figure, output_dir, "chignolin_method_diagnostics_total_nf"
    )
    plt.close(figure)


def plot_static_original_resolution(
    output_dir: Path,
    runs: dict,
    dense: dict,
    generated: dict,
    data_dir: Path,
) -> None:
    """Resolve the small but systematic Static/String performance difference."""

    original = runs["Original String"]
    static = runs["Static-NEB"]
    if not np.array_equal(
        original["number_of_force_evaluations"],
        static["number_of_force_evaluations"],
    ):
        raise RuntimeError("Static/String production cost grids differ")
    if not np.array_equal(
        dense["Original String"]["number_of_force_evaluations"],
        dense["Static-NEB"]["number_of_force_evaluations"],
    ):
        raise RuntimeError("Static/String dense checkpoint grids differ")
    force = original["number_of_force_evaluations"]
    ratio = static["i_perp"] / original["i_perp"]
    dense_force = dense["Original String"]["number_of_force_evaluations"]
    barrier_difference = (
        dense["Static-NEB"]["barrier_kcal"]
        - dense["Original String"]["barrier_kcal"]
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    axes[0].plot(force, ratio, color="#333333", lw=1.45)
    axes[0].axhline(1.0, color="#888888", lw=0.8, linestyle="--")
    axes[0].set_ylabel(
        r"$I_\perp^{\rm Static}/I_\perp^{\rm String}$"
    )
    axes[0].text(
        0.97,
        0.95,
        "String lower",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        color=COLORS["Original String"],
        fontsize=6.7,
    )
    axes[0].text(
        0.97,
        0.06,
        "Static lower",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        color=COLORS["Static-NEB"],
        fontsize=6.7,
    )
    axes[1].plot(
        dense_force,
        barrier_difference,
        color="#333333",
        marker="o",
        ms=2.1,
        lw=1.35,
    )
    axes[1].axhline(0.0, color="#888888", lw=0.8, linestyle="--")
    axes[1].set_ylabel(
        r"$\Delta E_{\rm Static}-\Delta E_{\rm String}$"
        "\n"
        r"(kcal mol$^{-1}$)"
    )
    axes[1].text(
        0.97,
        0.95,
        "String lower",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        color=COLORS["Original String"],
        fontsize=6.7,
    )
    axes[1].text(
        0.97,
        0.06,
        "Static lower",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        color=COLORS["Static-NEB"],
        fontsize=6.7,
    )
    for method in ("Static-NEB", "Original String"):
        axes[2].plot(
            runs[method]["number_of_force_evaluations"][1:],
            np.maximum(runs[method]["equidistance_integral"][1:], 1e-24),
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            lw=1.55,
            label=method,
        )
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"Equidistance integral $\mathcal{I}_{\rm eq}$")
    axes[2].legend(frameon=False, loc="best")
    maximum_force = int(
        max(
            original["number_of_force_evaluations"][-1],
            static["number_of_force_evaluations"][-1],
        )
    )
    for index, axis in enumerate(axes):
        axis.set_xlim(0, 1.005 * maximum_force)
        axis.set_xlabel("Number of Force Evaluations")
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    figure.subplots_adjust(
        left=0.085,
        right=0.99,
        bottom=0.21,
        top=0.93,
        wspace=0.42,
    )
    generated["static_original_resolution"] = save_figure(
        figure,
        output_dir,
        "chignolin_static_neb_vs_original_string_resolved_total_nf",
    )
    plt.close(figure)
    rows = []
    for cost, value in zip(force, ratio):
        rows.append(
            {
                "metric": "static_to_original_i_perp_ratio",
                "number_of_force_evaluations": int(cost),
                "value": float(value),
            }
        )
    for cost, value in zip(dense_force, barrier_difference):
        rows.append(
            {
                "metric": "static_minus_original_dense_barrier_kcal",
                "number_of_force_evaluations": int(cost),
                "value": float(value),
            }
        )
    write_csv(
        data_dir / "chignolin_static_neb_vs_original_string_resolved.csv",
        rows,
    )


def plot_force_audit(
    output_dir: Path, runs: dict, generated: dict
) -> None:
    matrix = np.asarray(
        [runs[method]["counter_vectors"][-1] for method in METHOD_ORDER],
        dtype=float,
    )
    total_queries = np.asarray(
        [
            np.sum(runs[method]["energy_request_vectors"][-1])
            + np.sum(runs[method]["force_request_vectors"][-1])
            for method in METHOD_ORDER
        ],
        dtype=float,
    )
    total_hits = np.asarray(
        [
            np.sum(runs[method]["cache_hit_vectors"][-1])
            for method in METHOD_ORDER
        ],
        dtype=float,
    )
    hit_fraction = total_hits / np.maximum(total_queries, 1.0)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    image = axes[0].imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    axes[0].set_yticks(np.arange(len(METHOD_ORDER)), METHOD_ORDER)
    axes[0].set_xticks(
        [0, 1, 5, 10, 15, 20, 21],
        ["0", "1", "5", "10", "15", "20", "21"],
    )
    axes[0].set_xlabel("Image index")
    axes[0].set_ylabel("Method")
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03)
    colorbar.set_label("Joint energy–force evaluations")
    panel_label(axes[0], "a")
    axes[1].bar(
        np.arange(len(METHOD_ORDER)),
        hit_fraction,
        color=[COLORS[method] for method in METHOD_ORDER],
        width=0.72,
    )
    axes[1].set_xticks(
        np.arange(len(METHOD_ORDER)),
        METHOD_ORDER,
        rotation=28,
        ha="right",
    )
    axes[1].set_ylabel("Cache-hit fraction of API requests")
    axes[1].set_ylim(0, max(0.05, 1.12 * float(np.max(hit_fraction))))
    clean_axis(axes[1])
    panel_label(axes[1], "b")
    figure.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.25,
        top=0.94,
        wspace=0.35,
    )
    generated["force_audit"] = save_figure(
        figure, output_dir, "chignolin_force_call_audit_total_nf"
    )
    plt.close(figure)


def plot_static_sensitivity(
    root: Path, output_dir: Path, generated: dict, data_dir: Path
) -> None:
    manifest = json.loads(
        (
            root
            / "reports"
            / "static_ultrafine_20260728a"
            / "manifest.json"
        ).read_text()
    )
    rows = manifest["records"]
    display_k = (34.0, 36.0, 38.0, 40.0, 42.0, 44.0)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    palette = plt.get_cmap("viridis")(
        np.linspace(0.06, 0.92, len(display_k))
    )
    for spring_k, color in zip(display_k, palette):
        subset = sorted(
            (
                row
                for row in rows
                if np.isclose(
                    float(row["configuration"]["spring_k"]), spring_k
                )
            ),
            key=lambda row: float(row["configuration"]["alpha"]),
        )
        stable = [row for row in subset if row.get("stable")]
        unstable = [row for row in subset if not row.get("stable")]
        if stable:
            axes[0].plot(
                [row["configuration"]["alpha"] for row in stable],
                [row["final_i_perp"] for row in stable],
                marker="o",
                ms=2.7,
                lw=1.15,
                color=color,
                label=rf"$k={spring_k:g}$",
            )
        if unstable and stable:
            top = max(row["final_i_perp"] for row in stable) * 1.3
            axes[0].scatter(
                [row["configuration"]["alpha"] for row in unstable],
                [top] * len(unstable),
                marker="x",
                s=16,
                color=color,
            )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Static Euler step $\alpha$")
    axes[0].set_ylabel(r"Terminal unnormalized $I_\perp$")
    axes[0].legend(
        frameon=False, ncol=2, loc="best", columnspacing=0.9
    )
    clean_axis(axes[0])
    panel_label(axes[0], "a")
    selected_alpha = 0.012575
    selected_rows = sorted(
        (
            row
            for row in rows
            if row.get("stable")
            and np.isclose(
                float(row["configuration"]["alpha"]), selected_alpha
            )
        ),
        key=lambda row: float(row["configuration"]["spring_k"]),
    )
    axes[1].plot(
        [row["configuration"]["spring_k"] for row in selected_rows],
        [row["final_equidistance_integral"] for row in selected_rows],
        color=COLORS["Static-NEB"],
        marker="o",
        ms=3.0,
    )
    chosen = next(
        row
        for row in selected_rows
        if np.isclose(float(row["configuration"]["spring_k"]), 40.0)
    )
    axes[1].scatter(
        [40.0],
        [chosen["final_equidistance_integral"]],
        marker="*",
        s=65,
        color=COLORS["ODE-NEB"],
        zorder=5,
        label=r"selected: $k=40$",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Spring coefficient $k$")
    axes[1].set_ylabel(r"Terminal $\mathcal{I}_{\rm eq}$")
    axes[1].legend(frameon=False, loc="best")
    axes[1].text(
        0.04,
        0.97,
        rf"$\alpha={selected_alpha:g}$",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
    )
    clean_axis(axes[1])
    panel_label(axes[1], "b")
    figure.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.20,
        top=0.94,
        wspace=0.30,
    )
    generated["static_sensitivity"] = save_figure(
        figure, output_dir, "chignolin_static_neb_k_alpha_sensitivity"
    )
    plt.close(figure)
    write_csv(data_dir / "chignolin_static_neb_sensitivity.csv", rows)


def plot_original_sensitivity(
    root: Path, output_dir: Path, generated: dict, data_dir: Path
) -> None:
    manifest = json.loads(
        (
            root
            / "reports"
            / "original_ultrafine_20260728a"
            / "manifest.json"
        ).read_text()
    )
    rows = sorted(
        manifest["records"], key=lambda row: row["configuration"]["dt"]
    )
    stable = [row for row in rows if row.get("stable")]
    unstable = [row for row in rows if not row.get("stable")]
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    fields = (
        ("final_i_perp", r"Terminal $I_\perp$"),
        (
            "final_barrier_kcal",
            r"Terminal sparse $\Delta E$ (kcal mol$^{-1}$)",
        ),
        (
            "final_equidistance_integral",
            r"Terminal $\mathcal{I}_{\rm eq}$",
        ),
    )
    for index, (axis, (key, ylabel)) in enumerate(zip(axes, fields)):
        axis.plot(
            [row["configuration"]["dt"] for row in stable],
            [row[key] for row in stable],
            color=COLORS["Original String"],
            marker="o",
            ms=2.8,
            lw=1.3,
        )
        if unstable:
            top = max(row[key] for row in stable) * 1.4
            axis.scatter(
                [row["configuration"]["dt"] for row in unstable],
                [top] * len(unstable),
                marker="x",
                color=COLORS["ODE-NEB"],
                s=18,
                label="unstable" if index == 0 else None,
            )
        selected = min(
            stable,
            key=lambda row: abs(
                row["configuration"]["dt"] - 0.012575
            ),
        )
        axis.scatter(
            [selected["configuration"]["dt"]],
            [selected[key]],
            marker="*",
            s=55,
            color=COLORS["IVM"],
            zorder=5,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"String descent step $\Delta t$")
        axis.set_ylabel(ylabel)
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    if unstable:
        axes[0].legend(frameon=False, loc="best")
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.20,
        top=0.94,
        wspace=0.42,
    )
    generated["original_sensitivity"] = save_figure(
        figure, output_dir, "chignolin_original_string_dt_sensitivity"
    )
    plt.close(figure)
    write_csv(data_dir / "chignolin_original_string_sensitivity.csv", rows)


def heatmap(
    axis,
    rows: list[dict],
    xkey: str,
    ykey: str,
    value_key: str,
    title: str,
):
    xvalues = sorted({float(row["configuration"][xkey]) for row in rows})
    yvalues = sorted({float(row["configuration"][ykey]) for row in rows})
    matrix = np.full((len(yvalues), len(xvalues)), np.nan)
    for row in rows:
        xindex = xvalues.index(float(row["configuration"][xkey]))
        yindex = yvalues.index(float(row["configuration"][ykey]))
        if row.get("stable"):
            matrix[yindex, xindex] = np.log10(
                max(float(row[value_key]), 1e-300)
            )
    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="cividis",
    )
    axis.set_xticks(np.arange(len(xvalues)), [f"{value:g}" for value in xvalues])
    axis.set_yticks(np.arange(len(yvalues)), [f"{value:g}" for value in yvalues])
    axis.set_xlabel(xkey.replace("_", " "))
    axis.set_ylabel(ykey.replace("_", " "))
    axis.set_title(title)
    return image


def plot_obsm_sensitivity(
    root: Path, output_dir: Path, generated: dict, data_dir: Path
) -> None:
    dt_manifest = json.loads(
        (
            root
            / "reports"
            / "obsm_fire_dt_extension_20260727a"
            / "manifest.json"
        ).read_text()
    )
    dt_rows = [
        row
        for row in dt_manifest["records"]
        if row.get("stable")
        and np.isclose(row["configuration"]["lambda_mix"], 0.9)
        and row["configuration"]["inner_steps"] in {5, 10, 15, 20}
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    palette = plt.get_cmap("viridis")(np.linspace(0.08, 0.88, 4))
    for inner_steps, color in zip((5, 10, 15, 20), palette):
        subset = sorted(
            (
                row
                for row in dt_rows
                if row["configuration"]["inner_steps"] == inner_steps
            ),
            key=lambda row: row["configuration"]["fire_dt"],
        )
        for axis, key in zip(
            axes, ("final_i_perp", "final_barrier_kcal")
        ):
            axis.plot(
                [row["configuration"]["fire_dt"] for row in subset],
                [row[key] for row in subset],
                marker="o",
                ms=2.8,
                color=color,
                label=rf"$n_{{\rm inner}}={inner_steps}$",
            )
    for index, (axis, ylabel) in enumerate(
        zip(
            axes,
            (
                r"Terminal $I_\perp$",
                r"Terminal sparse $\Delta E$ (kcal mol$^{-1}$)",
            ),
        )
    ):
        axis.set_yscale("log")
        axis.set_xlabel(r"FIRE step $\Delta t$")
        axis.set_ylabel(ylabel)
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    axes[0].legend(frameon=False, loc="best")
    figure.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.19,
        top=0.94,
        wspace=0.30,
    )
    generated["obsm_dt_sensitivity"] = save_figure(
        figure, output_dir, "chignolin_obsm_fire_dt_sensitivity"
    )
    plt.close(figure)

    beta_manifest = json.loads(
        (
            root
            / "reports"
            / "obsm_fire_beta_reparam_20260727a"
            / "manifest.json"
        ).read_text()
    )
    beta_rows = [
        row
        for row in beta_manifest["records"]
        if np.isclose(row["configuration"]["lambda_mix"], 0.9)
        and row["configuration"]["inner_steps"] == 10
        and np.isclose(row["configuration"]["fire_dt"], 0.12)
    ]
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    fields = (
        ("final_i_perp", r"$\log_{10} I_\perp$"),
        ("final_barrier_kcal", r"$\log_{10}\Delta E$"),
        (
            "final_equidistance_integral",
            r"$\log_{10}\mathcal{I}_{\rm eq}$",
        ),
    )
    for index, (axis, (key, colorbar_label)) in enumerate(zip(axes, fields)):
        image = heatmap(
            axis,
            beta_rows,
            "reparam_interval",
            "fire_beta",
            key,
            colorbar_label,
        )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        colorbar.ax.tick_params(labelsize=6.5)
        panel_label(axis, chr(ord("a") + index))
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.20,
        top=0.88,
        wspace=0.42,
    )
    generated["obsm_beta_reparam_sensitivity"] = save_figure(
        figure, output_dir, "chignolin_obsm_beta_reparam_sensitivity"
    )
    plt.close(figure)
    write_csv(data_dir / "chignolin_obsm_dt_sensitivity.csv", dt_rows)
    write_csv(
        data_dir / "chignolin_obsm_beta_reparam_sensitivity.csv",
        beta_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-directory",
        default="figures/paper_methods_total_nf_20260727a",
    )
    parser.add_argument("--ivm-result", type=Path)
    parser.add_argument("--ivm-dense-result", type=Path)
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output_dir = root / args.output_directory
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    production_manifest, runs, dense = load_selected(
        root,
        ivm_result=(
            None
            if args.ivm_result is None
            else args.ivm_result.expanduser().resolve()
        ),
        ivm_dense_result=(
            None
            if args.ivm_dense_result is None
            else args.ivm_dense_result.expanduser().resolve()
        ),
    )
    generated = {}
    plot_main(output_dir, runs, dense, generated)
    plot_cost_and_speedup(output_dir, runs, dense, generated, data_dir)
    if not args.primary_only:
        plot_diagnostics(output_dir, runs, generated)
        plot_static_original_resolution(
            output_dir, runs, dense, generated, data_dir
        )
        plot_force_audit(output_dir, runs, generated)
        plot_static_sensitivity(root, output_dir, generated, data_dir)
        plot_original_sensitivity(root, output_dir, generated, data_dir)
        plot_obsm_sensitivity(root, output_dir, generated, data_dir)

    metrics_rows = []
    dense_rows = []
    summary_methods = {}
    for method in METHOD_ORDER:
        for index in range(len(runs[method]["i_perp"])):
            metrics_rows.append(
                {
                    "method": method,
                    "number_of_force_evaluations": int(
                        runs[method]["number_of_force_evaluations"][index]
                    ),
                    "max_calls_per_internal_image": int(
                        runs[method]["calls_per_image"][index]
                    ),
                    "i_perp": float(runs[method]["i_perp"][index]),
                    "sparse_barrier_kcal": float(
                        runs[method]["barrier_kcal"][index]
                    ),
                    "physical_fmax_ev_per_angstrom": float(
                        runs[method]["physical_fmax"][index]
                    ),
                    "equidistance_integral": float(
                        runs[method]["equidistance_integral"][index]
                    ),
                    "segment_length_cv": float(
                        runs[method]["segment_length_cv"][index]
                    ),
                }
            )
        for index in range(len(dense[method]["barrier_kcal"])):
            dense_rows.append(
                {
                    "method": method,
                    "number_of_force_evaluations": int(
                        dense[method]["number_of_force_evaluations"][index]
                    ),
                    "max_calls_per_internal_image": int(
                        dense[method]["calls_per_image"][index]
                    ),
                    "adaptive_dense_barrier_kcal": float(
                        dense[method]["barrier_kcal"][index]
                    ),
                    "selected_subdivisions_per_segment": int(
                        dense[method]["selected_subdivisions"][index]
                    ),
                }
            )
        summary_methods[method] = {
            "run": str(runs[method]["path"]),
            "run_sha256": runs[method]["result_sha256"],
            "run_audit": str(runs[method]["audit_path"]),
            "run_audit_sha256": runs[method]["audit_sha256"],
            "dense": str(dense[method]["path"]),
            "dense_sha256": dense[method]["result_sha256"],
            "final_number_of_force_evaluations": int(
                runs[method]["number_of_force_evaluations"][-1]
            ),
            "final_i_perp": float(runs[method]["i_perp"][-1]),
            "final_dense_barrier_kcal": float(
                dense[method]["barrier_kcal"][-1]
            ),
            "final_physical_fmax": float(
                runs[method]["physical_fmax"][-1]
            ),
            "final_equidistance_integral": float(
                runs[method]["equidistance_integral"][-1]
            ),
            "diagnostic_joint_evaluations_excluded_from_cost": int(
                dense[method]["diagnostic_joint_evaluations"]
            ),
        }
    write_csv(data_dir / "chignolin_method_metrics_total_nf.csv", metrics_rows)
    write_csv(
        data_dir / "chignolin_dense_barrier_total_nf.csv", dense_rows
    )
    summary = {
        "primary_cost_axis": "counter_vectors.sum(axis=1)",
        "primary_cost_label": "Number of Force Evaluations",
        "cost_includes_all_22_images": True,
        "fixed_endpoint_first_evaluations_included": True,
        "dense_diagnostic_calls_excluded_from_optimization_axis": True,
        "initial_path_sha256": INITIAL_HASH,
        "tangent": "normalized central difference: x[i+1] - x[i-1]",
        "selected_obsm": {
            "lambda_mix": 0.9,
            "inner_steps": 5,
            "fire_dt": 0.12,
            "fire_beta": 1.0,
            "reparam_interval": 1,
            "selection": (
                "shared absolute-score winner after focused 1000-call "
                "screening and densified-barrier validation"
            ),
        },
        "selected_ivm": {
            "alpha": 1.0,
            "state_globalization": runs["IVM"]["state_globalization"],
            "outer_i_perp_acceptance_guard": (
                runs["IVM"]["state_globalization"] != "proximal_only"
            ),
            "proximal_objective_line_search": True,
        },
        "selected_static_neb": {
            "spring_k": 40.0,
            "alpha": 0.012575,
            "selection": (
                "shared absolute-score winner after ultrafine screening "
                "and 1000-call densified-barrier validation"
            ),
        },
        "selected_original_string": {
            "dt": 0.012575,
            "descent": "full physical force without tangent projection",
            "reparam_interval": 1,
            "reparameterization": (
                "Simplified String Method Appendix A cubic spline"
            ),
            "selection": (
                "shared absolute-score winner after ultrafine screening "
                "and 1000-call densified-barrier validation"
            ),
        },
        "selected_ode_neb": {
            "spring_k": 50.0,
            "optimizer": "ODE",
            "selection": (
                "shared absolute-score winner after eight-k 1000-call "
                "screening and densified-barrier validation"
            ),
        },
        "methods": summary_methods,
        "generated_figures": generated,
        "production_manifest_records": production_manifest["records"],
    }
    head_to_head = {}
    for method in ("Static-NEB", "Original String"):
        head_to_head[method] = {
            "i_perp_log_auc": log_auc(
                runs[method]["number_of_force_evaluations"],
                runs[method]["i_perp"],
            ),
            "dense_barrier_log_auc": log_auc(
                dense[method]["number_of_force_evaluations"],
                dense[method]["barrier_kcal"],
            ),
            "final_i_perp": float(runs[method]["i_perp"][-1]),
            "final_dense_barrier_kcal": float(
                dense[method]["barrier_kcal"][-1]
            ),
            "final_equidistance_integral": float(
                runs[method]["equidistance_integral"][-1]
            ),
        }
    metric_weights = {
        "i_perp_log_auc": 0.30,
        "dense_barrier_log_auc": 0.25,
        "final_i_perp": 0.20,
        "final_dense_barrier_kcal": 0.15,
        "final_equidistance_integral": 0.10,
    }
    metric_winners = {}
    for field in metric_weights:
        winner = min(head_to_head, key=lambda method: head_to_head[method][field])
        metric_winners[field] = winner
    absolute_scores = {}
    for method, values in head_to_head.items():
        absolute_scores[method] = (
            0.30 * values["i_perp_log_auc"]
            + 0.25 * values["dense_barrier_log_auc"]
            + 0.20 * np.log10(values["final_i_perp"])
            + 0.15 * np.log10(values["final_dense_barrier_kcal"])
            + 0.10 * np.log10(1.0 + values["final_equidistance_integral"])
        )
    overall_winner = min(absolute_scores, key=absolute_scores.get)
    summary["static_original_head_to_head"] = {
        "shared_absolute_score_formula": (
            "0.30*AUC(log10(I_perp)) + "
            "0.25*AUC(log10(densified_barrier_kcal)) + "
            "0.20*log10(final_I_perp) + "
            "0.15*log10(final_densified_barrier_kcal) + "
            "0.10*log10(1+final_equidistance_integral)"
        ),
        "predeclared_absolute_score_weights": metric_weights,
        "metrics": head_to_head,
        "metric_winners": metric_winners,
        "absolute_scores_lower_is_better": absolute_scores,
        "overall_winner": overall_winner,
        "interpretation": (
            "Original String has the lower prespecified shared absolute score; "
            "Static-NEB has the lower terminal I_perp and most intermediate "
            "dense barriers, while Original String has the lower terminal "
            "dense barrier and far better parameterization quality."
        ),
    }
    summary_path = output_dir / "chignolin_method_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
