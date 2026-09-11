#!/usr/bin/env python3
"""Plot the fixed-alpha=1 Chignolin comparison and reparameterization ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


IVM_COLOR = "#0072B2"
NO_REPARAM_COLOR = "#009E73"
NEB_COLOR = "#D55E00"
SPARSE_COLOR = "#6F6F6F"

MAIN_LABEL = r"IVM, $\alpha=1$ + equal-spring"
NO_REPARAM_LABEL = r"IVM, $\alpha=1$, no reparam."
NEB_LABEL = r"ODE-NEB, $k=50$"
SPARSE_LABEL = "IVM v46, sparse reparam."


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(saved, key, default=None):
    if key not in saved:
        return default
    value = np.asarray(saved[key])
    if value.size != 1:
        raise ValueError(f"{key} is not scalar")
    return value.item()


def finite_or_none(value):
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def final_record_per_call(saved, maximum=None):
    original_calls = np.asarray(saved["calls_per_image"], dtype=int)
    keep = np.r_[original_calls[1:] != original_calls[:-1], True]
    calls = original_calls[keep]
    selected = np.ones(len(calls), dtype=bool)
    if maximum is not None:
        selected &= calls <= int(maximum)
    result = {"calls": calls[selected]}
    for key in (
        "i_perp",
        "barrier_kcal",
        "physical_fmax",
        "equidistance_integral",
        "segment_length_cv",
    ):
        if key in saved:
            result[key] = np.asarray(saved[key], dtype=float)[keep][selected]
    return result


def load_run(path, maximum=None):
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as saved:
        series = final_record_per_call(saved, maximum=maximum)
        metadata = {
            "path": str(path),
            "sha256": sha256(path),
            "initial_path_sha256": str(scalar(saved, "initial_path_sha256")),
            "budget": int(scalar(saved, "budget")),
            "stop_reason": str(scalar(saved, "stop_reason", "")),
            "alpha": finite_or_none(scalar(saved, "alpha", np.nan)),
            "alpha_mode": str(scalar(saved, "alpha_mode", "")),
            "reparam_mode": str(scalar(saved, "reparam_mode", "")),
            "reparam_every": int(scalar(saved, "reparam_every", -1)),
            "state_reparam_attempts": int(
                scalar(saved, "state_reparam_attempts", 0)
            ),
            "state_reparam_accepts": int(
                scalar(saved, "state_reparam_accepts", 0)
            ),
            "spring_k": finite_or_none(scalar(saved, "spring_k", np.nan)),
            "last_recorded_call": int(series["calls"][-1]),
        }
        if "active_alpha" in saved:
            metadata["unique_active_alpha"] = np.unique(
                np.asarray(saved["active_alpha"], dtype=float)
            ).tolist()
        if "reparam_attempted" in saved:
            metadata["explicit_reparam_attempt_records"] = int(
                np.count_nonzero(np.asarray(saved["reparam_attempted"], dtype=bool))
            )
        else:
            metadata["explicit_reparam_attempt_records"] = 0
        if "reparam_committed" in saved:
            metadata["explicit_reparam_commit_records"] = int(
                np.count_nonzero(np.asarray(saved["reparam_committed"], dtype=bool))
            )
        else:
            metadata["explicit_reparam_commit_records"] = 0
        metadata["final"] = {
            key: float(series[key][-1])
            for key in (
                "i_perp",
                "barrier_kcal",
                "physical_fmax",
                "equidistance_integral",
                "segment_length_cv",
            )
        }
    return series, metadata


def load_dense(path, maximum=None):
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as saved:
        calls = np.asarray(saved["calls"], dtype=int)
        selected = np.ones(len(calls), dtype=bool)
        if maximum is not None:
            selected &= calls <= int(maximum)
        dense = {
            key: np.asarray(saved[key])[selected]
            for key in (
                "calls",
                "adaptive_barrier_kcal",
                "selected_subdivisions",
                "final_level_difference_kcal",
            )
            if key in saved
        }
        metadata = {
            "path": str(path),
            "sha256": sha256(path),
            "source_result": str(scalar(saved, "source_result", "")),
            "source_initial_path_sha256": str(
                scalar(saved, "source_initial_path_sha256", "")
            ),
            "diagnostic_backend_raw_joint_evaluations": int(
                scalar(saved, "diagnostic_backend_raw_joint_evaluations", 0)
            ),
        }
    return dense, metadata


def value_at_or_before(series, key, call, extend=False):
    call = int(call)
    if not extend and call > int(series["calls"][-1]):
        return None
    indices = np.flatnonzero(series["calls"] <= call)
    if not indices.size:
        return None
    return float(series[key][indices[-1]])


def first_call_to_threshold(series, key, threshold):
    running_best = np.minimum.accumulate(np.asarray(series[key], dtype=float))
    reached = np.flatnonzero(running_best <= float(threshold))
    return int(series["calls"][reached[0]]) if reached.size else None


def first_dense_call_to_threshold(dense, threshold):
    return first_call_to_threshold(
        {
            "calls": dense["calls"],
            "adaptive_barrier_kcal": dense["adaptive_barrier_kcal"],
        },
        "adaptive_barrier_kcal",
        threshold,
    )


def save_figure(fig, stem):
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")


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
            "legend.fontsize": 7.0,
            "lines.linewidth": 1.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def clean_axes(axes, xmaximum=500):
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", np.ravel(axes)):
        ax.text(
            -0.16,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
            va="bottom",
        )
        ax.set_xlim(0, xmaximum)
        ax.tick_params(direction="out", length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.55)


def plot_metrics(
    main,
    no_reparam,
    sparse,
    neb,
    dense_main,
    dense_no,
    dense_sparse,
    dense_neb,
    out,
):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    styles = [
        (main, IVM_COLOR, "-", MAIN_LABEL),
        (no_reparam, NO_REPARAM_COLOR, "--", NO_REPARAM_LABEL),
        (neb, NEB_COLOR, "-", NEB_LABEL),
    ]
    if sparse is not None:
        styles.insert(2, (sparse, SPARSE_COLOR, "-.", SPARSE_LABEL))
    ax = axes[0, 0]
    for series, color, linestyle, label in styles:
        ax.plot(
            series["calls"],
            series["i_perp"],
            color=color,
            linestyle=linestyle,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_ylabel(r"$I_{\perp}$")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[0, 1]
    dense_styles = [
        (dense_main, IVM_COLOR, "-", "o"),
        (dense_no, NO_REPARAM_COLOR, "--", "^"),
        (dense_neb, NEB_COLOR, "-", "s"),
    ]
    if dense_sparse is not None:
        dense_styles.insert(2, (dense_sparse, SPARSE_COLOR, "-.", "D"))
    for dense, color, linestyle, marker in dense_styles:
        visible = dense["calls"] <= 500
        ax.plot(
            dense["calls"][visible],
            dense["adaptive_barrier_kcal"][visible],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=3.0,
        )
    ax.set_ylabel("Densified barrier (kcal mol$^{-1}$)")

    ax = axes[1, 0]
    for series, color, linestyle, _ in styles:
        ax.plot(
            series["calls"],
            series["physical_fmax"],
            color=color,
            linestyle=linestyle,
        )
    ax.set_yscale("log")
    ax.set_ylabel(r"$f_{\max}$ (eV $\AA^{-1}$)")
    ax.set_xlabel("Max joint evaluations per internal image")

    ax = axes[1, 1]
    for series, color, linestyle, _ in styles:
        ax.plot(
            series["calls"],
            np.maximum(series["equidistance_integral"], 1e-24),
            color=color,
            linestyle=linestyle,
        )
    ax.set_yscale("log")
    ax.set_ylabel(r"$\int_0^1(\phi'\!\cdot\!\phi'')^2\,ds$")
    ax.set_xlabel("Max joint evaluations per internal image")

    clean_axes(axes)
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.10,
        top=0.98,
        wspace=0.34,
        hspace=0.34,
    )
    save_figure(fig, out / "chignolin_fixed_alpha1_metrics")
    plt.close(fig)


def plot_reparam_ablation(main, no_reparam, sparse, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    styles = [
        (main, IVM_COLOR, "-", MAIN_LABEL),
        (no_reparam, NO_REPARAM_COLOR, "--", NO_REPARAM_LABEL),
    ]
    if sparse is not None:
        styles.append((sparse, SPARSE_COLOR, "-.", SPARSE_LABEL))
    panels = (
        ("i_perp", r"$I_{\perp}$", True),
        (
            "equidistance_integral",
            r"$\int_0^1(\phi'\!\cdot\!\phi'')^2\,ds$",
            True,
        ),
        ("segment_length_cv", "Segment-length CV", True),
    )
    for ax, (key, ylabel, log_y) in zip(axes, panels):
        for series, color, linestyle, label in styles:
            values = np.asarray(series[key], dtype=float)
            if log_y:
                values = np.maximum(values, 1e-24)
            ax.plot(
                series["calls"],
                values,
                color=color,
                linestyle=linestyle,
                label=label,
            )
        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Max joint evaluations per internal image")
    axes[0].legend(frameon=False, loc="upper right")
    clean_axes(axes)
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.20,
        top=0.94,
        wspace=0.40,
    )
    save_figure(fig, out / "chignolin_reparameterization_ablation")
    plt.close(fig)


def common_thresholds(main, neb, key, count=14, logarithmic=True):
    series = (main, neb)
    common_floor = max(float(np.min(item[key])) for item in series)
    common_ceiling = min(float(item[key][0]) for item in series)
    if common_ceiling <= common_floor:
        raise RuntimeError(f"No common threshold range for {key}")
    if logarithmic:
        return np.geomspace(common_ceiling, common_floor, count)
    return np.linspace(common_ceiling, common_floor, count)


def build_cost_rows(main, no_reparam, neb, dense_main, dense_no, dense_neb):
    rows = []
    thresholds = common_thresholds(
        main, neb, "i_perp", count=14, logarithmic=True
    )
    for threshold in thresholds:
        main_call = first_call_to_threshold(main, "i_perp", threshold)
        no_call = first_call_to_threshold(no_reparam, "i_perp", threshold)
        neb_call = first_call_to_threshold(neb, "i_perp", threshold)
        rows.append(
            {
                "metric": "i_perp",
                "threshold": float(threshold),
                "ivm_equal_spring_min_calls": main_call,
                "ivm_no_reparam_min_calls": no_call,
                "neb_min_calls": neb_call,
                "equal_spring_speedup_neb_over_ivm": (
                    float(neb_call) / float(main_call)
                ),
                "no_reparam_speedup_neb_over_ivm": (
                    float(neb_call) / float(no_call)
                    if no_call is not None
                    else None
                ),
            }
        )

    common_floor = max(
        float(np.min(item["adaptive_barrier_kcal"]))
        for item in (dense_main, dense_neb)
    )
    common_ceiling = min(
        float(item["adaptive_barrier_kcal"][0])
        for item in (dense_main, dense_neb)
    )
    thresholds = np.linspace(common_ceiling, common_floor, 12).tolist()
    no_reparam_floor = float(np.min(dense_no["adaptive_barrier_kcal"]))
    if common_floor < no_reparam_floor < common_ceiling:
        thresholds.append(no_reparam_floor)
    thresholds = np.asarray(sorted(set(thresholds), reverse=True))
    for threshold in thresholds:
        main_call = first_dense_call_to_threshold(dense_main, threshold)
        no_call = first_dense_call_to_threshold(dense_no, threshold)
        neb_call = first_dense_call_to_threshold(dense_neb, threshold)
        if None in (main_call, neb_call):
            continue
        rows.append(
            {
                "metric": "densified_barrier_kcal",
                "threshold": float(threshold),
                "ivm_equal_spring_min_calls": main_call,
                "ivm_no_reparam_min_calls": no_call,
                "neb_min_calls": neb_call,
                "equal_spring_speedup_neb_over_ivm": (
                    float(neb_call) / float(main_call)
                ),
                "no_reparam_speedup_neb_over_ivm": (
                    float(neb_call) / float(no_call)
                    if no_call is not None
                    else None
                ),
            }
        )
    return rows


def plot_cost(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    panels = (
        ("i_perp", r"$I_{\perp}$ threshold", True),
        (
            "densified_barrier_kcal",
            "Densified barrier threshold (kcal mol$^{-1}$)",
            False,
        ),
    )
    for label, ax, (metric, xlabel, log_x) in zip("ab", axes, panels):
        subset = [row for row in rows if row["metric"] == metric]
        thresholds = np.asarray([row["threshold"] for row in subset])
        main_calls = np.asarray(
            [row["ivm_equal_spring_min_calls"] for row in subset]
        )
        no_calls = np.asarray(
            [
                (
                    row["ivm_no_reparam_min_calls"]
                    if row["ivm_no_reparam_min_calls"] is not None
                    else np.nan
                )
                for row in subset
            ],
            dtype=float,
        )
        neb_calls = np.asarray([row["neb_min_calls"] for row in subset])
        order = np.argsort(thresholds)
        ax.plot(
            thresholds[order],
            main_calls[order],
            color=IVM_COLOR,
            marker="o",
            markersize=3.2,
            label=MAIN_LABEL,
        )
        ax.plot(
            thresholds[order],
            no_calls[order],
            color=NO_REPARAM_COLOR,
            linestyle="--",
            marker="^",
            markersize=3.2,
            label=NO_REPARAM_LABEL,
        )
        ax.plot(
            thresholds[order],
            neb_calls[order],
            color=NEB_COLOR,
            marker="s",
            markersize=3.2,
            label=NEB_LABEL,
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
        strictest = min(subset, key=lambda row: row["threshold"])
        ax.annotate(
            f"{strictest['equal_spring_speedup_neb_over_ivm']:.2f}\u00d7",
            xy=(
                strictest["threshold"],
                strictest["ivm_equal_spring_min_calls"],
            ),
            xytext=(6, -1),
            textcoords="offset points",
            color=IVM_COLOR,
            fontsize=7.2,
            va="top",
        )
        no_subset = [
            row
            for row in subset
            if row["ivm_no_reparam_min_calls"] is not None
        ]
        no_strictest = min(no_subset, key=lambda row: row["threshold"])
        ax.annotate(
            f"{no_strictest['no_reparam_speedup_neb_over_ivm']:.2f}\u00d7",
            xy=(
                no_strictest["threshold"],
                no_strictest["ivm_no_reparam_min_calls"],
            ),
            xytext=(6, 2),
            textcoords="offset points",
            color=NO_REPARAM_COLOR,
            fontsize=7.2,
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
    save_figure(fig, out / "chignolin_fixed_alpha1_cost_to_precision")
    plt.close(fig)


def write_metrics_csv(main, no_reparam, neb, output):
    fields = ["calls_per_image"]
    for prefix in ("ivm_equal_spring", "ivm_no_reparam", "neb"):
        fields.extend(
            [
                f"{prefix}_i_perp",
                f"{prefix}_sparse_barrier_kcal",
                f"{prefix}_fmax_ev_a",
                f"{prefix}_equidistance_integral",
                f"{prefix}_segment_cv",
            ]
        )
    mappings = (
        ("ivm_equal_spring", main),
        ("ivm_no_reparam", no_reparam),
        ("neb", neb),
    )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for call in range(1, 501):
            row = {"calls_per_image": call}
            for prefix, series in mappings:
                row[f"{prefix}_i_perp"] = value_at_or_before(
                    series, "i_perp", call
                )
                row[f"{prefix}_sparse_barrier_kcal"] = value_at_or_before(
                    series, "barrier_kcal", call
                )
                row[f"{prefix}_fmax_ev_a"] = value_at_or_before(
                    series, "physical_fmax", call
                )
                row[f"{prefix}_equidistance_integral"] = value_at_or_before(
                    series, "equidistance_integral", call
                )
                row[f"{prefix}_segment_cv"] = value_at_or_before(
                    series, "segment_length_cv", call
                )
            writer.writerow(row)


def write_dense_csv(datasets, output):
    all_calls = sorted(
        {
            int(call)
            for _, dense in datasets
            for call in np.asarray(dense["calls"], dtype=int)
            if int(call) <= 500
        }
    )
    fields = ["calls_per_image"]
    for prefix, _ in datasets:
        fields.extend(
            [
                f"{prefix}_densified_barrier_kcal",
                f"{prefix}_selected_subdivisions",
            ]
        )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for call in all_calls:
            row = {"calls_per_image": call}
            for prefix, dense in datasets:
                indices = np.flatnonzero(dense["calls"] == call)
                if indices.size:
                    index = int(indices[-1])
                    row[f"{prefix}_densified_barrier_kcal"] = float(
                        dense["adaptive_barrier_kcal"][index]
                    )
                    row[f"{prefix}_selected_subdivisions"] = int(
                        dense["selected_subdivisions"][index]
                    )
                else:
                    row[f"{prefix}_densified_barrier_kcal"] = None
                    row[f"{prefix}_selected_subdivisions"] = None
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ivm-main", required=True)
    parser.add_argument("--ivm-no-reparam", required=True)
    parser.add_argument("--ivm-sparse")
    parser.add_argument("--neb", required=True)
    parser.add_argument("--dense-main", required=True)
    parser.add_argument("--dense-no-reparam", required=True)
    parser.add_argument("--dense-sparse")
    parser.add_argument("--dense-neb", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    main_series, main_meta = load_run(args.ivm_main, maximum=500)
    no_series, no_meta = load_run(args.ivm_no_reparam, maximum=500)
    sparse_series = None
    sparse_meta = None
    if args.ivm_sparse:
        sparse_series, sparse_meta = load_run(args.ivm_sparse, maximum=500)
    neb_visible, neb_meta = load_run(args.neb, maximum=500)
    neb_full, _ = load_run(args.neb)
    dense_main, dense_main_meta = load_dense(
        args.dense_main, maximum=main_meta["last_recorded_call"]
    )
    dense_no, dense_no_meta = load_dense(args.dense_no_reparam, maximum=500)
    dense_sparse = None
    dense_sparse_meta = None
    if args.dense_sparse:
        dense_sparse, dense_sparse_meta = load_dense(
            args.dense_sparse, maximum=500
        )
    dense_neb_visible, dense_neb_meta = load_dense(args.dense_neb, maximum=500)
    dense_neb_full, _ = load_dense(args.dense_neb)

    hashes = {
        main_meta["initial_path_sha256"],
        no_meta["initial_path_sha256"],
        neb_meta["initial_path_sha256"],
    }
    if sparse_meta is not None:
        hashes.add(sparse_meta["initial_path_sha256"])
    if len(hashes) != 1:
        raise RuntimeError(f"Initial paths differ: {sorted(hashes)}")
    if main_meta["alpha"] != 1 or main_meta["alpha_mode"] != "fixed":
        raise RuntimeError("Main IVM is not fixed alpha=1")
    if no_meta["alpha"] != 1 or no_meta["alpha_mode"] != "fixed":
        raise RuntimeError("No-reparameterization IVM is not fixed alpha=1")
    if (
        no_meta["reparam_every"] != 0
        or no_meta["state_reparam_attempts"] != 0
        or no_meta["explicit_reparam_attempt_records"] != 0
        or no_meta["explicit_reparam_commit_records"] != 0
    ):
        raise RuntimeError("No-reparameterization run contains reparameterization")

    configure_style()
    plot_metrics(
        main_series,
        no_series,
        sparse_series,
        neb_visible,
        dense_main,
        dense_no,
        dense_sparse,
        dense_neb_visible,
        output,
    )
    plot_reparam_ablation(main_series, no_series, sparse_series, output)
    cost_rows = build_cost_rows(
        main_series,
        no_series,
        neb_full,
        dense_main,
        dense_no,
        dense_neb_full,
    )
    plot_cost(cost_rows, output)

    write_metrics_csv(
        main_series,
        no_series,
        neb_visible,
        output / "chignolin_fixed_alpha1_metrics.csv",
    )
    dense_csv_datasets = [
        ("ivm_equal_spring", dense_main),
        ("ivm_no_reparam", dense_no),
    ]
    if dense_sparse is not None:
        dense_csv_datasets.append(("ivm_sparse_reparam_v46", dense_sparse))
    dense_csv_datasets.append(("neb", dense_neb_full))
    write_dense_csv(
        dense_csv_datasets,
        output / "chignolin_fixed_alpha1_dense_barrier.csv",
    )
    with (
        output / "chignolin_fixed_alpha1_cost_to_precision.csv"
    ).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cost_rows[0]))
        writer.writeheader()
        writer.writerows(cost_rows)

    i_rows = [row for row in cost_rows if row["metric"] == "i_perp"]
    barrier_rows = [
        row for row in cost_rows
        if row["metric"] == "densified_barrier_kcal"
    ]
    summary = {
        "status": (
            "fixed-alpha1 selected presentation; adaptive v52 remains "
            "preserved and unchanged"
        ),
        "initial_path_sha256": hashes.pop(),
        "tangent": "central difference throughout",
        "main_ivm": main_meta,
        "no_reparameterization_ivm": no_meta,
        "historical_sparse_reparameterization_reference": sparse_meta,
        "historical_sparse_reference_caveat": (
            "v46 is included only as a historical sparse-reparameterization "
            "reference; it is not claimed to be a one-factor controlled "
            "ablation of v52."
        ),
        "neb": neb_meta,
        "dense_diagnostics": {
            "main_ivm": dense_main_meta,
            "no_reparameterization_ivm": dense_no_meta,
            "historical_sparse_reparameterization_reference": dense_sparse_meta,
            "neb": dense_neb_meta,
            "excluded_from_optimization_axis": True,
        },
        "strictest_common_i_perp_threshold": i_rows[-1],
        "strictest_common_densified_barrier_threshold": barrier_rows[-1],
        "main_curve_not_extended_after_honest_stop": True,
    }
    with (
        output / "chignolin_fixed_alpha1_summary.json"
    ).open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
