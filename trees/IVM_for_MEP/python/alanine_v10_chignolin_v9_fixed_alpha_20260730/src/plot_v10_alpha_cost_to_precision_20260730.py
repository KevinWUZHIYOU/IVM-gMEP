#!/usr/bin/env python3
"""Rebuild the two-panel Ala alpha figure with the V10 production grid.

The left controlled r=0.1 data are read unchanged from the frozen CSV. The
right panel accepts only an alpha grid whose individual raw-trajectory audits
all passed, preventing a visually attractive lower envelope from replacing a
failed production trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from run_ala_v10_policy_20260730 import POLICIES


CONTROLLED_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 10000.0, math.inf)
PRODUCTION_ALPHAS = (
    0.01,
    0.1,
    1.0,
    10.0,
    20.8,
    30.0,
    100.0,
    10000.0,
    math.inf,
)
MARKERS = ("o", "s", "^", "D", "v", "P")
CMAP = mpl.colormaps["viridis_r"]
NORM = mpl.colors.Normalize(vmin=-2.0, vmax=4.0)


def scalar(saved, name):
    return np.asarray(saved[name]).reshape(()).item()


def same_alpha(left, right):
    if math.isinf(left) or math.isinf(right):
        return math.isinf(left) and math.isinf(right)
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-14)


def alpha_label(alpha):
    if math.isinf(alpha):
        return "inf"
    return f"{alpha:g}".replace(".", "p").replace("-", "m")


def alpha_text(alpha):
    return r"\infty" if math.isinf(alpha) else f"{alpha:g}"


def color(alpha):
    return "#000000" if math.isinf(alpha) else CMAP(NORM(math.log10(alpha)))


def record_frontier(cost, precision):
    order = np.argsort(np.asarray(cost, dtype=float), kind="stable")
    cost = np.asarray(cost, dtype=float)[order]
    precision = np.asarray(precision, dtype=float)[order]
    records_cost = []
    records_precision = []
    best = math.inf
    for unique_cost in np.unique(cost):
        value = float(np.min(precision[cost == unique_cost]))
        if value < best:
            records_cost.append(float(unique_cost))
            records_precision.append(value)
            best = value
    return np.asarray(records_precision), np.asarray(records_cost)


def load_controlled(reference_csv: Path):
    with reference_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    curves = {}
    for alpha in CONTROLLED_ALPHAS:
        selected = [
            row for row in rows
            if same_alpha(float(row["alpha"]), alpha)
            and row.get("metric", "i_perp") == "i_perp"
        ]
        if not selected:
            raise ValueError(f"controlled CSV has no alpha={alpha:g}")
        curves[alpha] = (
            np.asarray([float(row["precision_target"]) for row in selected]),
            np.asarray([float(row["number_of_forces"]) for row in selected]),
        )
    return curves


def load_production(
    root: Path,
    policy: str,
    budget: int,
    study_label: str,
    max_spacing_cv: float,
):
    policy_spec = POLICIES[policy]
    curves = {}
    provenance = {}
    for alpha in PRODUCTION_ALPHAS:
        token = alpha_label(alpha)
        matches = sorted(
            (root / "runs" / "ivm").glob(
                f"ivm_{study_label}_{policy}_a{token}_central_state_ivm_*_budget{budget}.npz"
            )
        )
        if len(matches) != 1:
            raise ValueError(f"expected one V10 result for alpha={alpha:g}, found {matches}")
        result = matches[0]
        audit_path = result.with_suffix(".audit.json")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("consistent", False):
            raise RuntimeError(f"counter audit failed: {result}")
        with np.load(result, allow_pickle=False) as saved:
            expected = {
                "solver": "state_ivm",
                "alpha_mode": "fixed",
                "reparam_mode": "state_curve_projected",
                "state_globalization": "iperp_monotone",
                "plbfgs_kkt_rtol": float(policy_spec.get("kkt_rtol", 0.7)),
                "inner_calls": int(policy_spec.get("inner_calls", 30)),
                "state_curve_direct_trial_calls": int(policy_spec.get("inner_calls", 30)),
                "state_curve_direct_objective_acceptance": policy_spec.get(
                    "direct_objective_acceptance", "per_image"
                ),
                "state_trust_fraction": float(policy_spec["state_trust_fraction"]),
                "state_kkt_stall_tighten_target": float(
                    policy_spec.get("stall_kkt_target", -1.0)
                ),
                "stop_iperp": float(policy_spec.get("stop_iperp", -1.0)),
            }
            for name, target in expected.items():
                actual = scalar(saved, name)
                matches_target = (
                    math.isclose(float(actual), target, rel_tol=0.0, abs_tol=1e-14)
                    if isinstance(target, float)
                    else actual == target
                )
                if not matches_target:
                    raise RuntimeError(f"{result.name}: {name}={actual!r}, expected {target!r}")
            active_alpha = np.asarray(saved["active_alpha"], dtype=float)
            if not all(same_alpha(float(item), alpha) for item in active_alpha):
                raise RuntimeError(f"{result.name}: active alpha was not fixed")
            counters = np.asarray(saved["counter_vectors"], dtype=int)
            image_count = int(np.asarray(saved["initial_path"]).shape[0])
            if counters.shape[1] != image_count:
                raise RuntimeError(
                    f"{result.name}: counter width does not match initial path"
                )
            cost = np.sum(counters, axis=1)
            precision = np.asarray(saved["i_perp"], dtype=float)
            spacing_cv = np.asarray(saved["segment_length_cv"], dtype=float)
            violations = np.flatnonzero(spacing_cv > max_spacing_cv)
            prefix_stop = int(violations[0]) if violations.size else len(cost)
            if prefix_stop < 1:
                raise RuntimeError(f"{result.name}: initial state fails spacing gate")
            curves[alpha] = record_frontier(
                cost[:prefix_stop], precision[:prefix_stop]
            )
            provenance[alpha_text(alpha)] = {
                "result": str(result),
                "result_audit": str(audit_path),
                "plotted_prefix_records": prefix_stop,
                "truncated_at_spacing_gate": bool(violations.size),
                "first_spacing_violation_call": (
                    None
                    if not violations.size
                    else int(saved["calls_per_image"][violations[0]])
                ),
                "maximum_full_run_spacing_cv": float(np.max(spacing_cv)),
                "final_total_forces": int(cost[-1]),
                "minimum_i_perp": float(np.min(precision)),
                "final_i_perp": float(precision[-1]),
            }
    return curves, provenance


def configure_style():
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 8.0,
        "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "legend.fontsize": 7.0, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def draw_panel(axis, curves, title, alphas, provenance=None):
    for index, alpha in enumerate(alphas):
        precision, cost = curves[alpha]
        metadata = None if provenance is None else provenance[alpha_text(alpha)]
        truncated = bool(
            metadata is not None
            and metadata["truncated_at_spacing_gate"]
        )
        selected = provenance is not None and same_alpha(alpha, 20.8)
        axis.plot(
            precision, cost, color=color(alpha),
            linewidth=2.0 if selected else 1.55 if math.isinf(alpha) else 1.2,
            linestyle=":" if truncated else "--" if math.isinf(alpha) else "-",
            marker="o" if math.isinf(alpha) else MARKERS[index % len(MARKERS)],
            markersize=3.0, markeredgewidth=0.3,
            markeredgecolor="#000000" if selected else color(alpha),
            markevery=max(1, len(cost) // 5),
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(30.0, 3e-4)
    maximum_cost = max(float(np.max(values[1])) for values in curves.values())
    axis.set_ylim(20.0, max(100.0, 1.15 * maximum_cost))
    axis.set_xlabel(r"Target $I_\perp$")
    axis.set_ylabel("Number of Forces")
    axis.set_title(title, pad=5)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    axis.spines[["top", "right"]].set_visible(False)


def add_key(figure):
    color_axis = figure.add_axes([0.18, 0.925, 0.55, 0.018])
    colorbar = figure.colorbar(mpl.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=color_axis, orientation="horizontal")
    ticks = [-2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{tick:.2f}".rstrip("0").rstrip(".") for tick in ticks])
    colorbar.ax.tick_params(labelsize=6.5, length=2.0, pad=1.5)
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.set_title(r"Color scale: $\log_{10}\alpha$", fontsize=7.0, pad=3.0)
    figure.legend(
        handles=[
            Line2D([0], [0], color="#000000", linewidth=1.55,
                   linestyle="--", marker="o", markersize=3.0,
                   label=r"$\alpha=\infty$"),
            Line2D([0], [0], color=color(20.8), linewidth=2.0,
                   marker="o", markeredgecolor="#000000", markersize=3.5,
                   label=r"selected $\alpha=20.8$"),
        ],
        frameon=False, loc="upper right", bbox_to_anchor=(0.99, 1.01),
        handlelength=1.6,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--policy",
        default="v9_strict",
        choices=tuple(POLICIES),
    )
    parser.add_argument("--budget", type=int, default=650)
    parser.add_argument("--study-label", default="ala_v10")
    parser.add_argument("--max-spacing-cv", type=float, default=1e-3)
    parser.add_argument("--controlled-reference-csv", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = root / "figures" / "alpha_mechanism"
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    controlled = load_controlled(args.controlled_reference_csv.expanduser().resolve())
    production, provenance = load_production(
        root,
        args.policy,
        args.budget,
        args.study_label,
        args.max_spacing_cv,
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.3, 2.75), sharey=True)
    draw_panel(
        axes[0], controlled, r"Controlled inner solves ($r=0.1$)",
        CONTROLLED_ALPHAS,
    )
    draw_panel(
        axes[1], production,
        r"Inexact production IVM ($r:0.96\to0.95$, cap 15)",
        PRODUCTION_ALPHAS, provenance,
    )
    axes[1].set_ylabel("")
    for label, axis in zip(("a", "b"), axes):
        axis.text(-0.15, 1.05, label, transform=axis.transAxes, fontsize=10, fontweight="bold", va="top")
    add_key(figure)
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.875), pad=1.0, w_pad=1.1)
    figure.text(
        0.5, 0.01,
        "Raw monotone production trajectories; optimization force calls only.",
        ha="center", va="bottom", fontsize=7.0, color="#555555",
    )
    stem = output / "alanine_alpha_iperp_cost_to_precision_controlled_vs_v49_production"
    for suffix in (".pdf", ".svg"):
        figure.savefig(stem.with_suffix(suffix), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)
    rows = []
    for family, curves, alphas in (
        ("controlled_r0p1_frozen", controlled, CONTROLLED_ALPHAS),
        ("production_v49_state_forced", production, PRODUCTION_ALPHAS),
    ):
        for alpha in alphas:
            precision, cost = curves[alpha]
            rows.extend({"family": family, "alpha": "inf" if math.isinf(alpha) else alpha, "precision_target": float(item_precision), "number_of_forces": float(item_cost), "point_type": "running_best_record"} for item_precision, item_cost in zip(precision, cost))
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "left_panel": {"status": "frozen_controlled_r0p1", "source_csv": str(args.controlled_reference_csv.expanduser().resolve())},
        "right_panel": {
            "policy": args.policy,
            "study_label": args.study_label,
            "budget_per_internal_image": args.budget,
            "algorithm": (
                "fixed-alpha Ala port of Chignolin-v9; inexact KKT r=0.96 "
                "tightened to 0.95 on a state stall, cap 15, path-summed "
                "proximal-objective and refreshed-global-I_perp acceptance, "
                "I_perp convergence target 1.5e-3"
            ),
            "mesh_gate": (
                "each curve is truncated before its first raw state with "
                f"segment-length CV > {args.max_spacing_cv:g}"
            ),
            "selected_alpha": 20.8,
            "selected_stability_audit": str(
                root / "reports" / "stability_v49_a20p8_production.json"
            ),
            "runs": provenance,
        },
        "output": str(stem.with_suffix(".pdf")),
    }
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
