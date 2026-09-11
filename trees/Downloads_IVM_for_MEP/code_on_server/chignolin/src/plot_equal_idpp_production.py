#!/usr/bin/env python3
"""Plot the audited equal-IDPP Chignolin production comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


IVM_COLOR = "#0072B2"
NEB_COLOR = "#D55E00"


def final_record_per_call(saved, maximum=None):
    calls = np.asarray(saved["calls_per_image"], dtype=int)
    keep = np.r_[calls[1:] != calls[:-1], True]
    calls = calls[keep]
    if maximum is not None:
        selected = calls <= int(maximum)
        calls = calls[selected]
    else:
        selected = slice(None)
    result = {"calls": calls}
    for key in (
        "i_perp",
        "barrier_kcal",
        "physical_fmax",
        "equidistance_integral",
        "segment_length_cv",
    ):
        if key in saved:
            values = np.asarray(saved[key], dtype=float)[keep]
            result[key] = values[selected]
    return result


def value_at_or_before(series, key, call):
    indices = np.flatnonzero(series["calls"] <= int(call))
    if not indices.size:
        raise ValueError(f"No record at or before call {call}")
    return float(series[key][indices[-1]])


def first_call_to_threshold(series, key, threshold):
    running_best = np.minimum.accumulate(series[key])
    reached = np.flatnonzero(running_best <= float(threshold))
    return int(series["calls"][reached[0]]) if reached.size else None


def save_figure(fig, stem):
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
    )


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def plot_metrics(ivm, neb, dense_ivm, dense_neb, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    ax = axes[0, 0]
    ax.plot(ivm["calls"], ivm["i_perp"], color=IVM_COLOR, label="IVM")
    ax.plot(neb["calls"], neb["i_perp"], color=NEB_COLOR, label="NEB")
    ax.set_yscale("log")
    ax.set_ylabel(r"$I_{\perp}$")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[0, 1]
    ax.plot(
        dense_ivm["calls"],
        dense_ivm["adaptive_barrier_kcal"],
        color=IVM_COLOR,
        marker="o",
        markersize=3.0,
        label="IVM",
    )
    neb_visible = dense_neb["calls"] <= 500
    ax.plot(
        dense_neb["calls"][neb_visible],
        dense_neb["adaptive_barrier_kcal"][neb_visible],
        color=NEB_COLOR,
        marker="s",
        markersize=3.0,
        label="NEB",
    )
    ax.set_ylabel("Densified barrier (kcal mol$^{-1}$)")

    ax = axes[1, 0]
    ax.plot(
        ivm["calls"],
        ivm["physical_fmax"],
        color=IVM_COLOR,
    )
    ax.plot(
        neb["calls"],
        neb["physical_fmax"],
        color=NEB_COLOR,
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"$f_{\max}$ (eV $\AA^{-1}$)")
    ax.set_xlabel("Max joint evaluations per internal image")

    ax = axes[1, 1]
    ax.plot(
        ivm["calls"],
        np.maximum(ivm["equidistance_integral"], 1e-22),
        color=IVM_COLOR,
    )
    ax.plot(
        neb["calls"],
        np.maximum(neb["equidistance_integral"], 1e-22),
        color=NEB_COLOR,
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"$\int_0^1(\phi'\!\cdot\!\phi'')^2\,ds$")
    ax.set_xlabel("Max joint evaluations per internal image")

    for label, ax in zip(("a", "b", "c", "d"), axes.flat):
        ax.text(
            -0.16,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
            va="bottom",
        )
        ax.set_xlim(0, 500)
        ax.tick_params(direction="out", length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.55)
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.10,
        top=0.98,
        wspace=0.34,
        hspace=0.34,
    )
    save_figure(fig, output_dir / "chignolin_equal_idpp_metrics")
    plt.close(fig)


def build_cost_rows(ivm, neb, dense_ivm, dense_neb):
    rows = []
    i_calls = (10, 20, 50, 100, 200, 300, 400, 500)
    for nominal_call in i_calls:
        threshold = value_at_or_before(
            ivm, "i_perp", nominal_call
        )
        ivm_call = first_call_to_threshold(ivm, "i_perp", threshold)
        neb_call = first_call_to_threshold(neb, "i_perp", threshold)
        if (
            rows
            and rows[-1]["metric"] == "i_perp"
            and rows[-1]["ivm_min_calls"] == ivm_call
            and rows[-1]["neb_min_calls"] == neb_call
            and np.isclose(rows[-1]["threshold"], threshold)
        ):
            continue
        rows.append(
            {
                "metric": "i_perp",
                "threshold": threshold,
                "ivm_min_calls": ivm_call,
                "neb_min_calls": neb_call,
                "speedup_neb_over_ivm": (
                    float(neb_call) / float(ivm_call)
                    if neb_call is not None
                    else np.nan
                ),
            }
        )
    for threshold, nominal_call in zip(
        dense_ivm["adaptive_barrier_kcal"],
        dense_ivm["calls"],
    ):
        ivm_reached = np.flatnonzero(
            np.minimum.accumulate(
                dense_ivm["adaptive_barrier_kcal"]
            )
            <= threshold
        )
        neb_reached = np.flatnonzero(
            np.minimum.accumulate(
                dense_neb["adaptive_barrier_kcal"]
            )
            <= threshold
        )
        ivm_call = int(dense_ivm["calls"][ivm_reached[0]])
        neb_call = (
            int(dense_neb["calls"][neb_reached[0]])
            if neb_reached.size
            else None
        )
        # A cost-to-threshold comparison is only defined when both methods
        # actually attain the threshold.  Do not extrapolate the extended
        # NEB curve below its best observed densified barrier.
        if neb_call is None:
            continue
        rows.append(
            {
                "metric": "densified_barrier_kcal",
                "threshold": float(threshold),
                "ivm_min_calls": ivm_call,
                "neb_min_calls": neb_call,
                "speedup_neb_over_ivm": float(neb_call) / float(ivm_call),
            }
        )
    return rows


def plot_cost(rows, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    panels = (
        ("i_perp", r"$I_{\perp}$ threshold", True),
        (
            "densified_barrier_kcal",
            "Densified barrier threshold (kcal mol$^{-1}$)",
            False,
        ),
    )
    for label, ax, (metric, xlabel, log_x) in zip(
        ("a", "b"), axes, panels
    ):
        subset = [row for row in rows if row["metric"] == metric]
        threshold = np.asarray([row["threshold"] for row in subset])
        ivm_calls = np.asarray([row["ivm_min_calls"] for row in subset])
        neb_calls = np.asarray([row["neb_min_calls"] for row in subset])
        order = np.argsort(threshold)
        ax.plot(
            threshold[order],
            ivm_calls[order],
            color=IVM_COLOR,
            marker="o",
            markersize=3.2,
            label="IVM",
        )
        ax.plot(
            threshold[order],
            neb_calls[order],
            color=NEB_COLOR,
            marker="s",
            markersize=3.2,
            label="NEB",
        )
        if log_x:
            ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Minimum joint evaluations / image")
        ax.text(
            -0.16,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
            va="bottom",
        )
        final = subset[-1]
        ax.annotate(
            f"{final['speedup_neb_over_ivm']:.2f}\u00d7",
            xy=(final["threshold"], final["neb_min_calls"]),
            xytext=(7, 1),
            textcoords="offset points",
            color=NEB_COLOR,
            fontsize=7.5,
            va="bottom",
        )
        ax.tick_params(direction="out", length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.55)
    axes[0].legend(frameon=False, loc="upper left")
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.20,
        top=0.96,
        wspace=0.34,
    )
    save_figure(fig, output_dir / "chignolin_equal_idpp_cost_to_precision")
    plt.close(fig)


def write_metrics_csv(ivm, neb, output):
    fields = [
        "calls_per_image",
        "ivm_i_perp",
        "neb_i_perp",
        "ivm_sparse_barrier_kcal",
        "neb_sparse_barrier_kcal",
        "ivm_fmax_ev_a",
        "neb_fmax_ev_a",
        "ivm_equidistance_integral",
        "neb_equidistance_integral",
        "ivm_segment_cv",
        "neb_segment_cv",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for call in range(1, 501):
            writer.writerow(
                {
                    "calls_per_image": call,
                    "ivm_i_perp": value_at_or_before(
                        ivm, "i_perp", call
                    ),
                    "neb_i_perp": value_at_or_before(
                        neb, "i_perp", call
                    ),
                    "ivm_sparse_barrier_kcal": value_at_or_before(
                        ivm, "barrier_kcal", call
                    ),
                    "neb_sparse_barrier_kcal": value_at_or_before(
                        neb, "barrier_kcal", call
                    ),
                    "ivm_fmax_ev_a": value_at_or_before(
                        ivm, "physical_fmax", call
                    ),
                    "neb_fmax_ev_a": value_at_or_before(
                        neb, "physical_fmax", call
                    ),
                    "ivm_equidistance_integral": value_at_or_before(
                        ivm, "equidistance_integral", call
                    ),
                    "neb_equidistance_integral": value_at_or_before(
                        neb, "equidistance_integral", call
                    ),
                    "ivm_segment_cv": value_at_or_before(
                        ivm, "segment_length_cv", call
                    ),
                    "neb_segment_cv": value_at_or_before(
                        neb, "segment_length_cv", call
                    ),
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ivm", required=True)
    parser.add_argument("--neb", required=True)
    parser.add_argument("--dense-ivm", required=True)
    parser.add_argument("--dense-neb", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.ivm, allow_pickle=False) as saved:
        ivm = final_record_per_call(saved, maximum=500)
        ivm_initial_hash = str(saved["initial_path_sha256"].item())
        ivm_budget = int(saved["budget"].item())
        ivm_kkt_rtol = float(saved["plbfgs_kkt_rtol"].item())
        ivm_final = {
            "i_perp": float(saved["i_perp"][-1]),
            "sparse_barrier_kcal": float(saved["barrier_kcal"][-1]),
            "fmax_ev_a": float(saved["physical_fmax"][-1]),
            "equidistance_integral": float(
                saved["equidistance_integral"][-1]
            ),
            "segment_cv": float(saved["segment_length_cv"][-1]),
        }
    with np.load(args.neb, allow_pickle=False) as saved:
        neb = final_record_per_call(saved, maximum=500)
        neb_full = final_record_per_call(saved)
        neb_initial_hash = str(saved["initial_path_sha256"].item())
        neb_budget = int(saved["budget"].item())
        neb_k = float(saved["spring_k"].item())
        neb_final = {
            "i_perp": float(saved["i_perp"][-1]),
            "sparse_barrier_kcal": float(saved["barrier_kcal"][-1]),
            "fmax_ev_a": float(saved["physical_fmax"][-1]),
            "equidistance_integral": float(
                saved["equidistance_integral"][-1]
            ),
            "segment_cv": float(saved["segment_length_cv"][-1]),
        }
    with np.load(args.dense_ivm, allow_pickle=False) as saved:
        dense_ivm = {
            key: np.asarray(saved[key])
            for key in (
                "calls",
                "adaptive_barrier_kcal",
                "selected_subdivisions",
            )
        }
    with np.load(args.dense_neb, allow_pickle=False) as saved:
        dense_neb = {
            key: np.asarray(saved[key])
            for key in (
                "calls",
                "adaptive_barrier_kcal",
                "selected_subdivisions",
            )
        }

    if ivm_initial_hash != neb_initial_hash:
        raise RuntimeError("IVM and NEB initial paths are not identical")
    configure_style()
    plot_metrics(ivm, neb, dense_ivm, dense_neb, output_dir)
    cost_rows = build_cost_rows(ivm, neb_full, dense_ivm, dense_neb)
    plot_cost(cost_rows, output_dir)
    write_metrics_csv(
        ivm,
        neb,
        output_dir / "chignolin_equal_idpp_metrics.csv",
    )
    with (
        output_dir / "chignolin_equal_idpp_cost_to_precision.csv"
    ).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cost_rows[0]))
        writer.writeheader()
        writer.writerows(cost_rows)

    summary = {
        "initial_path_sha256": ivm_initial_hash,
        "ivm_result": str(Path(args.ivm).resolve()),
        "neb_result": str(Path(args.neb).resolve()),
        "ivm_budget": ivm_budget,
        "ivm_fixed_relative_kkt_tolerance": ivm_kkt_rtol,
        "neb_budget": neb_budget,
        "fixed_neb_k": neb_k,
        "selection_reason": (
            "The k=50 ODE-only baseline retained nearly the best I_perp "
            "while giving the strongest balanced densified barrier and "
            "spring distribution among the extended finalists."
        ),
        "ivm_at_500": ivm_final,
        f"neb_at_{neb_budget}": neb_final,
        "i_perp_final_threshold": cost_rows[7],
        "dense_barrier_final_threshold": cost_rows[-1],
        "diagnostic_calls_excluded_from_optimization_axis": True,
    }
    with (
        output_dir / "chignolin_equal_idpp_summary.json"
    ).open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
