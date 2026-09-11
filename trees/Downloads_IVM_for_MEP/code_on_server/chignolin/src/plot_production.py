#!/usr/bin/env python3
"""Publication figures and machine-readable tables for fixed IVM/NEB runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


IVM_COLOR = "#0072B2"
NEB_COLOR = "#D55E00"


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.7,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", which="major", color="0.88", linewidth=0.45)
    axis.tick_params(which="both", pad=2)


def panel_label(axis, label):
    axis.text(
        -0.15,
        1.03,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )


def save_figure(figure, output_stem: Path):
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight"
    )
    plt.close(figure)


def load_run(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.asarray(saved[name]) for name in saved.files}


def load_dense(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.asarray(saved[name]) for name in saved.files}


def truncate_records(payload, call_key, stop_call):
    """Retain complete records up to an audited stopping call."""
    calls = np.asarray(payload[call_key])
    mask = calls <= int(stop_call)
    truncated = {}
    for name, values in payload.items():
        values = np.asarray(values)
        if values.ndim and values.shape[0] == calls.shape[0]:
            truncated[name] = values[mask]
        else:
            truncated[name] = values
    if not np.any(mask):
        raise ValueError(
            f"No {call_key} record exists at or before stop call {stop_call}"
        )
    return truncated


def running_first_passage(calls, values, thresholds):
    calls = np.asarray(calls, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(calls) & np.isfinite(values)
    order = np.argsort(calls[valid])
    calls = calls[valid][order]
    values = values[valid][order]
    running = np.minimum.accumulate(values)
    output = np.full(len(thresholds), np.nan)
    for index, threshold in enumerate(thresholds):
        reached = np.flatnonzero(running <= threshold)
        if reached.size:
            output[index] = calls[reached[0]]
    return output


def common_thresholds(ivm_values, neb_values, points=48):
    ivm_values = np.asarray(ivm_values)
    neb_values = np.asarray(neb_values)
    positive = np.concatenate(
        [
            ivm_values[np.isfinite(ivm_values) & (ivm_values > 0)],
            neb_values[np.isfinite(neb_values) & (neb_values > 0)],
        ]
    )
    if positive.size == 0:
        return np.array([])
    low = max(float(np.min(positive)), 1e-12)
    high = float(np.max(positive))
    geometric = np.geomspace(high, low, points)
    exact_limits = np.array(
        [
            np.min(ivm_values[np.isfinite(ivm_values) & (ivm_values > 0)]),
            np.min(neb_values[np.isfinite(neb_values) & (neb_values > 0)]),
        ]
    )
    # Keep the smooth logarithmic sampling, but never omit the exact finest
    # precision reached by either production curve.
    return np.sort(np.unique(np.concatenate([geometric, exact_limits])))[::-1]


def common_barrier_targets(ivm_values, neb_values, points=48):
    """Absolute barrier ceilings reached by both fixed production runs."""
    ivm_values = np.asarray(ivm_values, dtype=float)
    neb_values = np.asarray(neb_values, dtype=float)
    ivm_valid = ivm_values[np.isfinite(ivm_values) & (ivm_values > 0)]
    neb_valid = neb_values[np.isfinite(neb_values) & (neb_values > 0)]
    if ivm_valid.size == 0 or neb_valid.size == 0:
        return np.array([])
    finest_common = max(float(np.min(ivm_valid)), float(np.min(neb_valid)))
    coarsest = max(float(np.max(ivm_valid)), float(np.max(neb_valid)))
    geometric = np.geomspace(coarsest, finest_common, points)
    return np.sort(
        np.unique(np.concatenate([geometric, [finest_common]]))
    )[::-1]


def plot_raw(ivm, neb, dense_ivm, dense_neb, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    methods = [
        ("IVM", ivm, dense_ivm, IVM_COLOR, 2.0),
        ("NEB", neb, dense_neb, NEB_COLOR, 1.65),
    ]
    for label, run, dense, color, width in methods:
        calls = np.asarray(run["calls_per_image"], dtype=float)
        mask = calls <= 500
        axes[0].plot(
            calls[mask],
            np.asarray(run["i_perp"])[mask],
            color=color,
            linewidth=width,
            label=label,
        )
        dense_calls = np.asarray(dense["calls"], dtype=float)
        dense_mask = dense_calls <= 500
        axes[1].plot(
            dense_calls[dense_mask],
            np.asarray(dense["adaptive_barrier_kcal"])[dense_mask],
            color=color,
            linewidth=width,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel(r"Unnormalized $I_{\perp}$")
    axes[1].set_ylabel(
        r"0 K forward barrier (kcal mol$^{-1}$)"
    )
    for axis in axes:
        axis.set_xlim(0, 500)
        axis.set_xlabel("Max evaluations per internal image")
        clean_axis(axis)
    axes[0].legend(loc="upper right")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.95, wspace=0.30)
    save_figure(figure, output_dir / "chignolin_raw_metrics")


def plot_fmax(ivm, neb, output_dir):
    figure, axis = plt.subplots(figsize=(3.55, 2.8))
    for label, run, color, width in (
        ("IVM", ivm, IVM_COLOR, 2.0),
        ("NEB", neb, NEB_COLOR, 1.65),
    ):
        calls = np.asarray(run["calls_per_image"], dtype=float)
        mask = calls <= 500
        axis.plot(
            calls[mask],
            np.asarray(run["physical_fmax"])[mask],
            color=color,
            linewidth=width,
            label=label,
        )
    axis.axhline(0.05, color="0.35", linewidth=0.8, linestyle="--")
    axis.text(495, 0.055, "0.05 eV Å$^{-1}$", ha="right", va="bottom")
    axis.set_yscale("log")
    axis.set_xlim(0, 500)
    axis.set_xlabel("Max evaluations per internal image")
    axis.set_ylabel(r"Physical normal $f_{\max}$ (eV Å$^{-1}$)")
    axis.legend(loc="upper right")
    clean_axis(axis)
    figure.subplots_adjust(left=0.18, right=0.97, bottom=0.18, top=0.96)
    save_figure(figure, output_dir / "chignolin_fmax")


def equidistance_records(run):
    if {
        "equidistance_integral",
        "segment_length_cv",
        "segment_length_ratio",
    }.issubset(run):
        return (
            np.asarray(run["calls_per_image"], dtype=float),
            np.asarray(run["equidistance_integral"], dtype=float),
            np.asarray(run["segment_length_cv"], dtype=float),
            np.asarray(run["segment_length_ratio"], dtype=float),
        )
    paths = np.asarray(run["stored_paths"], dtype=float)
    calls = np.asarray(run["stored_path_calls"], dtype=float)
    ds = 1.0 / float(paths.shape[1] - 1)
    first = (paths[:, 2:] - paths[:, :-2]) / (2.0 * ds)
    second = (
        paths[:, 2:] - 2.0 * paths[:, 1:-1] + paths[:, :-2]
    ) / (ds * ds)
    speed_derivative_half = np.sum(first * second, axis=2)
    integral = np.sum(speed_derivative_half**2, axis=1) * ds
    lengths = np.linalg.norm(paths[:, 1:] - paths[:, :-1], axis=2)
    means = np.mean(lengths, axis=1)
    cv = np.std(lengths, axis=1) / means
    ratio = np.max(lengths, axis=1) / np.min(lengths, axis=1)
    return calls, integral, cv, ratio


def plot_equidistance(ivm, neb, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    for label, run, color, width in (
        ("IVM", ivm, IVM_COLOR, 2.0),
        ("NEB", neb, NEB_COLOR, 1.65),
    ):
        calls, integral, cv, _ = equidistance_records(run)
        mask = calls <= 500
        axes[0].plot(
            calls[mask],
            cv[mask],
            color=color,
            linewidth=width,
            label=label,
        )
        axes[1].plot(
            calls[mask],
            integral[mask],
            color=color,
            linewidth=width,
            label=label,
        )
    axes[0].set_ylabel("Segment-length coefficient of variation")
    axes[1].set_ylabel(
        r"$\int_0^1[\phi'(s)\!\cdot\!\phi''(s)]^2\,ds$"
    )
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlim(0, 500)
        axis.set_xlabel("Max evaluations per internal image")
        clean_axis(axis)
    axes[0].legend(loc="upper left")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.18,
        top=0.95,
        wspace=0.33,
    )
    save_figure(figure, output_dir / "chignolin_equidistance_diagnostics")


def plot_cost_to_precision(
    ivm,
    neb,
    dense_ivm,
    dense_neb,
    output_dir,
):
    ivm_i = np.asarray(ivm["i_perp"], dtype=float)
    neb_i = np.asarray(neb["i_perp"], dtype=float)
    ivm_barrier = np.asarray(
        dense_ivm["adaptive_barrier_kcal"], dtype=float
    )
    neb_barrier = np.asarray(
        dense_neb["adaptive_barrier_kcal"], dtype=float
    )
    specs = [
        (
            "iperp",
            np.asarray(ivm["calls_per_image"], dtype=float),
            ivm_i,
            np.asarray(neb["calls_per_image"], dtype=float),
            neb_i,
            common_thresholds(ivm_i, neb_i),
            r"Target $I_{\perp}$",
        ),
        (
            "barrier_target",
            np.asarray(dense_ivm["calls"], dtype=float),
            ivm_barrier,
            np.asarray(dense_neb["calls"], dtype=float),
            neb_barrier,
            common_barrier_targets(ivm_barrier, neb_barrier),
            r"Target forward barrier (kcal mol$^{-1}$)",
        ),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    speedup_payload = {}
    for panel_index, (
        key,
        ivm_calls,
        ivm_values,
        neb_calls,
        neb_values,
        thresholds,
        xlabel,
    ) in enumerate(specs):
        ivm_cost = running_first_passage(ivm_calls, ivm_values, thresholds)
        neb_cost = running_first_passage(neb_calls, neb_values, thresholds)
        ivm_reached = np.isfinite(ivm_cost)
        neb_reached = np.isfinite(neb_cost)
        common = np.isfinite(ivm_cost) & np.isfinite(neb_cost)
        axis = axes[panel_index]
        axis.plot(
            thresholds[ivm_reached],
            ivm_cost[ivm_reached],
            color=IVM_COLOR,
            marker="s",
            markersize=3.3,
            label="IVM",
        )
        axis.plot(
            thresholds[neb_reached],
            neb_cost[neb_reached],
            color=NEB_COLOR,
            marker="o",
            markersize=3.3,
            label="NEB",
        )
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Minimum evaluations per internal image")
        clean_axis(axis)
        common_indices = np.flatnonzero(common)
        ivm_only_indices = np.flatnonzero(ivm_reached & ~neb_reached)
        if common_indices.size:
            finest = int(common_indices[-1])
            ratio = float(neb_cost[finest] / ivm_cost[finest])
            threshold = float(thresholds[finest])
            axis.annotate(
                "",
                xy=(threshold, ivm_cost[finest]),
                xytext=(threshold, neb_cost[finest]),
                arrowprops=dict(
                    arrowstyle="<->", color="0.2", linewidth=0.8
                ),
            )
            axis.annotate(
                f"{ratio:.1f}×",
                xy=(threshold, 0.5 * (ivm_cost[finest] + neb_cost[finest])),
                xytext=(
                    (-34, -12) if ivm_only_indices.size else (5, 0)
                ),
                textcoords="offset points",
                ha="right" if ivm_only_indices.size else "left",
                va="center",
                fontsize=8,
            )
            speedup_payload[key] = {
                (
                    "finest_common_threshold"
                    if key == "iperp"
                    else "finest_common_barrier_kcal_per_mol"
                ): threshold,
                "ivm_first_passage_calls": float(ivm_cost[finest]),
                "neb_first_passage_calls": float(neb_cost[finest]),
                "speedup": ratio,
                "thresholds": thresholds[common].tolist(),
                "ivm_cost": ivm_cost[common].tolist(),
                "neb_cost": neb_cost[common].tolist(),
            }
        if ivm_only_indices.size:
            finest = int(ivm_only_indices[-1])
            threshold = float(thresholds[finest])
            neb_budget = float(np.max(neb_calls))
            lower_bound = float(neb_budget / ivm_cost[finest])
            axis.scatter(
                [threshold],
                [neb_budget],
                marker="^",
                s=28,
                facecolors="none",
                edgecolors=NEB_COLOR,
                linewidths=0.9,
                zorder=4,
            )
            axis.annotate(
                "",
                xy=(threshold, ivm_cost[finest]),
                xytext=(threshold, neb_budget),
                arrowprops=dict(
                    arrowstyle="<->",
                    color="0.2",
                    linewidth=0.8,
                    linestyle="--",
                ),
            )
            axis.text(
                0.69,
                0.72,
                f">{lower_bound:.1f}×\n(censored)",
                transform=axis.transAxes,
                ha="left",
                va="center",
                fontsize=7.5,
            )
            speedup_payload.setdefault(key, {})
            speedup_payload[key]["finest_ivm_only_target"] = {
                "threshold": threshold,
                "ivm_first_passage_calls": float(ivm_cost[finest]),
                "neb_budget_calls": neb_budget,
                "neb_reached": False,
                "speedup_lower_bound": lower_bound,
            }
    axes[0].legend(loc="upper left")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.19, top=0.95, wspace=0.31)
    save_figure(figure, output_dir / "chignolin_cost_to_precision")
    return speedup_payload


def plot_counter_audit(ivm, neb, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), sharey=True)
    for axis, label, run, color in (
        (axes[0], "IVM", ivm, IVM_COLOR),
        (axes[1], "NEB", neb, NEB_COLOR),
    ):
        counts = np.asarray(run["counter_vectors"], dtype=int)[-1]
        image_indices = np.arange(len(counts))
        axis.bar(image_indices, counts, color=color, width=0.72)
        axis.axhline(
            np.max(counts[1:-1]),
            color="0.25",
            linewidth=0.7,
            linestyle="--",
        )
        axis.set_xlabel("Image index")
        axis.set_title(label)
        clean_axis(axis)
    axes[0].set_ylabel("Joint OpenMM energy–force evaluations")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.89, wspace=0.12)
    save_figure(figure, output_dir / "chignolin_counter_audit")


def export_tables(ivm, neb, dense_ivm, dense_neb, barrier_reference, output_dir):
    for label, run, dense in (
        ("ivm", ivm, dense_ivm),
        ("neb", neb, dense_neb),
    ):
        eq_calls, eq_integral, eq_cv, eq_ratio = equidistance_records(run)
        equidistance_by_call = {
            int(call): (float(integral), float(cv), float(ratio))
            for call, integral, cv, ratio in zip(
                eq_calls, eq_integral, eq_cv, eq_ratio
            )
        }
        dense_by_call = {
            int(call): float(value)
            for call, value in zip(
                dense["calls"], dense["adaptive_barrier_kcal"]
            )
        }
        path = output_dir / f"chignolin_{label}_metrics.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "max_evaluations_per_internal_image",
                    "mean_evaluations_per_internal_image",
                    "total_internal_openmm_evaluations",
                    "unnormalized_i_perp",
                    "physical_fmax_eV_per_A",
                    "equidistance_integral",
                    "segment_length_cv",
                    "segment_length_ratio",
                    "raw_22_image_barrier_kcal_per_mol",
                    "adaptive_dense_barrier_kcal_per_mol",
                    "dense_barrier_error_kcal_per_mol",
                ]
            )
            for index, call in enumerate(run["calls_per_image"]):
                dense_value = dense_by_call.get(int(call), np.nan)
                equidistance = equidistance_by_call.get(
                    int(call), (np.nan, np.nan, np.nan)
                )
                writer.writerow(
                    [
                        int(call),
                        float(run["mean_calls_per_internal_image"][index]),
                        int(run["total_internal_joint_evaluations"][index]),
                        float(run["i_perp"][index]),
                        float(run["physical_fmax"][index]),
                        equidistance[0],
                        equidistance[1],
                        equidistance[2],
                        float(run["barrier_kcal"][index]),
                        dense_value,
                        (
                            abs(dense_value - barrier_reference)
                            if np.isfinite(dense_value)
                            else np.nan
                        ),
                    ]
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--ivm", required=True)
    parser.add_argument("--neb", required=True)
    parser.add_argument("--dense-ivm", required=True)
    parser.add_argument("--dense-neb", required=True)
    parser.add_argument("--barrier-reference", type=float, default=np.nan)
    parser.add_argument(
        "--ivm-stop-call",
        type=int,
        default=0,
        help="Declared IVM budget; truncates cost, counters, and CSV exports.",
    )
    parser.add_argument(
        "--neb-stop-call",
        type=int,
        default=0,
        help="Audited NEB stop; truncates cost, counters, and CSV exports.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output_dir = root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    ivm = load_run(Path(args.ivm))
    neb = load_run(Path(args.neb))
    dense_ivm = load_dense(Path(args.dense_ivm))
    dense_neb = load_dense(Path(args.dense_neb))
    if args.ivm_stop_call > 0:
        ivm = truncate_records(
            ivm, "calls_per_image", args.ivm_stop_call
        )
        dense_ivm = truncate_records(
            dense_ivm, "calls", args.ivm_stop_call
        )
    if args.neb_stop_call > 0:
        neb = truncate_records(
            neb, "calls_per_image", args.neb_stop_call
        )
        dense_neb = truncate_records(
            dense_neb, "calls", args.neb_stop_call
        )
    barrier_reference = args.barrier_reference
    if not np.isfinite(barrier_reference):
        barrier_reference = float(
            min(
                dense_ivm["adaptive_barrier_kcal"][-1],
                dense_neb["adaptive_barrier_kcal"][-1],
            )
        )

    configure_style()
    plot_raw(ivm, neb, dense_ivm, dense_neb, output_dir)
    plot_fmax(ivm, neb, output_dir)
    plot_equidistance(ivm, neb, output_dir)
    speedups = plot_cost_to_precision(
        ivm,
        neb,
        dense_ivm,
        dense_neb,
        output_dir,
    )
    ivm_endpoint_i = float(ivm["i_perp"][-1])
    neb_i_cost = running_first_passage(
        neb["calls_per_image"], neb["i_perp"], [ivm_endpoint_i]
    )[0]
    ivm_dense_barrier = np.asarray(
        dense_ivm["adaptive_barrier_kcal"], dtype=float
    )
    neb_dense_barrier = np.asarray(
        dense_neb["adaptive_barrier_kcal"], dtype=float
    )
    finest_common_barrier = max(
        float(np.min(ivm_dense_barrier)),
        float(np.min(neb_dense_barrier)),
    )
    ivm_barrier_cost = running_first_passage(
        dense_ivm["calls"],
        ivm_dense_barrier,
        [finest_common_barrier],
    )[0]
    neb_barrier_cost = running_first_passage(
        dense_neb["calls"],
        neb_dense_barrier,
        [finest_common_barrier],
    )[0]
    endpoint_speedups = {
        "iperp": {
            "target_ivm_at_declared_stop": ivm_endpoint_i,
            "ivm_calls": int(ivm["calls_per_image"][-1]),
            "neb_first_passage_calls": (
                float(neb_i_cost) if np.isfinite(neb_i_cost) else None
            ),
            "neb_reached": bool(np.isfinite(neb_i_cost)),
            "speedup": (
                float(neb_i_cost / float(ivm["calls_per_image"][-1]))
                if np.isfinite(neb_i_cost)
                else None
            ),
            "speedup_lower_bound": (
                None
                if np.isfinite(neb_i_cost)
                else float(
                    np.max(neb["calls_per_image"])
                    / float(ivm["calls_per_image"][-1])
                )
            ),
        },
        "barrier_target": {
            "target_forward_barrier_kcal_per_mol": finest_common_barrier,
            "ivm_first_passage_calls": float(ivm_barrier_cost),
            "neb_first_passage_calls": (
                float(neb_barrier_cost)
                if np.isfinite(neb_barrier_cost)
                else None
            ),
            "neb_reached": bool(np.isfinite(neb_barrier_cost)),
            "speedup": (
                float(
                    neb_barrier_cost / ivm_barrier_cost
                )
                if np.isfinite(neb_barrier_cost)
                and np.isfinite(ivm_barrier_cost)
                else None
            ),
            "speedup_lower_bound": (
                None
                if np.isfinite(neb_barrier_cost)
                else float(
                    np.max(dense_neb["calls"])
                    / ivm_barrier_cost
                )
            ),
        },
    }
    plot_counter_audit(ivm, neb, output_dir)
    export_tables(
        ivm,
        neb,
        dense_ivm,
        dense_neb,
        barrier_reference,
        output_dir,
    )
    summary = {
        "ivm_color": IVM_COLOR,
        "neb_color": NEB_COLOR,
        "legacy_barrier_reference_kcal_per_mol": barrier_reference,
        "legacy_barrier_reference_used_in_cost_plot": False,
        "barrier_cost_definition": (
            "minimum calls to reach an absolute forward-barrier ceiling"
        ),
        "speedups": speedups,
        "endpoint_and_budget_speedups": endpoint_speedups,
        "ivm_result": str(Path(args.ivm).resolve()),
        "neb_result": str(Path(args.neb).resolve()),
        "dense_ivm": str(Path(args.dense_ivm).resolve()),
        "dense_neb": str(Path(args.dense_neb).resolve()),
        "ivm_declared_stop_call": (
            int(args.ivm_stop_call) if args.ivm_stop_call > 0 else None
        ),
        "neb_audited_stop_call": (
            int(args.neb_stop_call) if args.neb_stop_call > 0 else None
        ),
    }
    (output_dir / "chignolin_figure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
