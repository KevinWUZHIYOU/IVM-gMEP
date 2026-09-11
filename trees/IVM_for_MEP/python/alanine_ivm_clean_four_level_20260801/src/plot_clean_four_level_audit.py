#!/usr/bin/env python3
"""Plot and tabulate the clean four-level fixed-alpha Ala IVM audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


REFERENCE_BARRIER_KCAL = 2.4560825131253083
ALPHAS = (0.01, 0.1, 1.0, 10.0, 20.0, 30.0, 100.0, 1000.0, 10000.0, math.inf)
STAGE_GLOBS = {
    "stage1": "runs/stage1_exact_no_resample/*budget160.npz",
    "stage2": "runs/stage2_exact_resample/*budget160.npz",
    "stage3": "runs/stage3_inexact_baseline_resample/*budget220.npz",
    "stage4": "runs/stage4_inexact_optimized_resample/stage4_uniform_v3_a*_phaseA80_phaseB220.npz",
}
STAGE_TITLES = {
    "stage1": "Exact-like, no resampling",
    "stage2": "Exact-like + resampling",
    "stage3": "Baseline inexact + resampling",
    "stage4": "Uniform optimized inexact + resampling",
}


@dataclass
class Series:
    stage: str
    rtol: float
    alpha: float
    source: Path
    dense_source: Path
    force: np.ndarray
    iperp: np.ndarray
    dense_force: np.ndarray
    dense_barrier: np.ndarray
    cv: np.ndarray
    equidistance: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alpha_label(alpha: float) -> str:
    return r"$\infty$" if math.isinf(alpha) else f"{alpha:g}"


def number_label(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:g}".replace(".", "p").replace("-", "m")


def alpha_style(alpha: float):
    if math.isinf(alpha):
        return "#202020", (0, (3, 1.5))
    finite = np.asarray(ALPHAS[:-1], dtype=float)
    scaled = (np.log10(alpha) - np.log10(finite).min()) / (
        np.log10(finite).max() - np.log10(finite).min()
    )
    return mpl.colormaps["viridis"](0.05 + 0.9 * scaled), "-"


def dense_path(root: Path, stage: str, rtol: float, alpha: float) -> Path:
    policy_label = "_v3" if stage == "stage4" else ""
    path = root / "runs/diagnostics" / (
        f"dense_barrier_clean_{stage}{policy_label}_r{number_label(rtol)}_"
        f"a{number_label(alpha)}_common_checkpoints.npz"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_series(root: Path) -> list[Series]:
    output = []
    for stage, pattern in STAGE_GLOBS.items():
        for source in sorted(root.glob(pattern)):
            with np.load(source, allow_pickle=False) as saved:
                alpha = float(np.asarray(saved["alpha"]).item())
                if stage == "stage4":
                    rtol = 0.8
                    force = np.asarray(saved["total_force_evaluations"], dtype=int)
                else:
                    rtol = float(np.asarray(saved["plbfgs_kkt_rtol"]).item())
                    force = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
                iperp = np.asarray(saved["i_perp"], dtype=float)
                cv = np.asarray(saved["segment_length_cv"], dtype=float)
                equidistance = np.asarray(saved["equidistance_relative_defect"], dtype=float)
                if len(force) != len(iperp) or not np.all(np.diff(force) > 0):
                    raise RuntimeError(f"Invalid force accounting in {source}")
            diagnostic = dense_path(root, stage, rtol, alpha)
            with np.load(diagnostic, allow_pickle=False) as dense:
                labels = np.asarray(dense["calls"], dtype=int)
                barrier = np.asarray(dense["adaptive_barrier_kcal"], dtype=float)
            if stage == "stage4":
                dense_force = labels
            else:
                with np.load(source, allow_pickle=False) as saved:
                    mapping = {
                        int(call): int(cost)
                        for call, cost in zip(saved["calls_per_image"], force)
                    }
                dense_force = np.asarray([mapping[int(call)] for call in labels], dtype=int)
            if not np.all(np.diff(dense_force) > 0):
                raise RuntimeError(f"Dense cost labels are not increasing in {diagnostic}")
            output.append(
                Series(
                    stage, rtol, alpha, source, diagnostic, force, iperp,
                    dense_force, barrier, cv, equidistance
                )
            )
    return output


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(axis):
    axis.grid(True, which="major", ls="--", lw=0.45, alpha=0.35)
    axis.tick_params(direction="out", length=3)


def panel_label(axis, label: str):
    axis.text(-0.13, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")


def legend_handles():
    handles = []
    for alpha in ALPHAS:
        color, linestyle = alpha_style(alpha)
        handles.append(Line2D([0], [0], color=color, ls=linestyle, lw=1.7, label=alpha_label(alpha)))
    return handles


def plot_panel(axis_i, axis_b, members: list[Series], title: str):
    for member in sorted(members, key=lambda item: item.alpha):
        color, linestyle = alpha_style(member.alpha)
        width = 2.15 if np.isclose(member.alpha, 20.0) else 1.0
        order = 8 if np.isclose(member.alpha, 20.0) else 3
        axis_i.plot(member.force, member.iperp, color=color, ls=linestyle, lw=width, zorder=order)
        axis_b.plot(
            member.dense_force, member.dense_barrier, color=color, ls=linestyle,
            lw=width, marker="o", ms=2.2, zorder=order
        )
    axis_b.axhline(REFERENCE_BARRIER_KCAL, color="#555555", lw=0.8, ls=":")
    axis_i.set_xscale("log")
    axis_i.set_yscale("log")
    axis_b.set_xscale("log")
    axis_b.set_yscale("log")
    axis_i.set_title(title)
    axis_i.set_ylabel(r"$I_\perp$")
    axis_b.set_ylabel(r"Densified $\Delta E$ (kcal mol$^{-1}$)")
    axis_b.set_xlabel(r"Whole-path force evaluations, $N_F$")
    axis_i.set_ylim(1e-5, 1.2e3)
    axis_b.set_ylim(2.3, 55)
    clean_axis(axis_i)
    clean_axis(axis_b)


def save(figure, stem: Path):
    outputs = []
    for suffix in ("pdf", "png", "svg"):
        output = stem.with_suffix(f".{suffix}")
        figure.savefig(output, dpi=300)
        outputs.append(output)
    plt.close(figure)
    return outputs


def plot_exact(series: list[Series], output_dir: Path):
    columns = [
        ("stage1", 0.2), ("stage1", 0.1),
        ("stage2", 0.2), ("stage2", 0.1),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(14.6, 6.1), dpi=200)
    for column, (stage, rtol) in enumerate(columns):
        members = [item for item in series if item.stage == stage and np.isclose(item.rtol, rtol)]
        plot_panel(axes[0, column], axes[1, column], members, f"{STAGE_TITLES[stage]}, r={rtol:g}")
        panel_label(axes[0, column], chr(ord("a") + column))
        panel_label(axes[1, column], chr(ord("e") + column))
    figure.legend(
        handles=legend_handles(), title=r"Fixed $\alpha$", loc="upper center",
        bbox_to_anchor=(0.5, 1.035), ncol=10, frameon=False
    )
    figure.subplots_adjust(left=0.055, right=0.99, bottom=0.11, top=0.88, wspace=0.32, hspace=0.30)
    return save(figure, output_dir / "alanine_clean_exact_alpha_two_metric_2x4")


def plot_inexact(series: list[Series], output_dir: Path):
    columns = ["stage3", "stage4"]
    figure, axes = plt.subplots(2, 2, figsize=(9.1, 6.25), dpi=200)
    for column, stage in enumerate(columns):
        members = [item for item in series if item.stage == stage]
        plot_panel(axes[0, column], axes[1, column], members, STAGE_TITLES[stage])
        panel_label(axes[0, column], chr(ord("a") + column))
        panel_label(axes[1, column], chr(ord("c") + column))
    figure.legend(
        handles=legend_handles(), title=r"Fixed $\alpha$", loc="upper center",
        bbox_to_anchor=(0.5, 1.03), ncol=5, frameon=False
    )
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.105, top=0.84, wspace=0.25, hspace=0.28)
    return save(figure, output_dir / "alanine_clean_inexact_alpha_two_metric_2x2")


def cost_to_reach(cost: np.ndarray, metric: np.ndarray, thresholds: np.ndarray):
    values = []
    for threshold in thresholds:
        indices = np.flatnonzero(metric <= threshold)
        values.append(np.nan if not indices.size else float(cost[indices[0]]))
    return np.asarray(values)


def plot_stage4_cost(series: list[Series], output_dir: Path):
    members = [item for item in series if item.stage == "stage4"]
    thresholds_i = np.geomspace(1e-4, 1e2, 42)
    thresholds_b = np.geomspace(5e-3, 20, 42)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.65), dpi=200)
    for member in sorted(members, key=lambda item: item.alpha):
        color, linestyle = alpha_style(member.alpha)
        width = 2.2 if np.isclose(member.alpha, 20.0) else 1.0
        axes[0].plot(
            thresholds_i, cost_to_reach(member.force, member.iperp, thresholds_i),
            color=color, ls=linestyle, lw=width
        )
        error = np.abs(member.dense_barrier - REFERENCE_BARRIER_KCAL)
        axes[1].plot(
            thresholds_b, cost_to_reach(member.dense_force, error, thresholds_b),
            color=color, ls=linestyle, lw=width
        )
    axes[0].set_xlabel(r"$I_\perp$ threshold")
    axes[1].set_xlabel(r"$|\Delta E-\Delta E^*|$ (kcal mol$^{-1}$)")
    for index, axis in enumerate(axes):
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_yscale("log")
        axis.set_ylabel(r"Whole-path force evaluations, $N_F$")
        clean_axis(axis)
        panel_label(axis, chr(ord("a") + index))
    axes[0].legend(handles=legend_handles(), title=r"Fixed $\alpha$", ncol=2, frameon=False, loc="upper right")
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.96, wspace=0.28)
    return save(figure, output_dir / "alanine_clean_stage4_alpha_cost_to_precision_1x2")


def write_summary(series: list[Series], output_dir: Path):
    rows = []
    for member in sorted(series, key=lambda item: (item.stage, item.rtol, item.alpha)):
        barrier_error = np.abs(member.dense_barrier - REFERENCE_BARRIER_KCAL)
        rows.append(
            {
                "stage": member.stage,
                "rtol": member.rtol,
                "alpha": "inf" if math.isinf(member.alpha) else member.alpha,
                "terminal_force": int(member.force[-1]),
                "terminal_iperp": float(member.iperp[-1]),
                "minimum_iperp": float(member.iperp.min()),
                "raw_iperp_rises": int(np.count_nonzero(np.diff(member.iperp) > 1e-12)),
                "terminal_dense_barrier_kcal_per_mol": float(member.dense_barrier[-1]),
                "terminal_dense_barrier_abs_error": float(barrier_error[-1]),
                "minimum_sampled_dense_barrier_abs_error": float(barrier_error.min()),
                "sampled_dense_barrier_rises": int(np.count_nonzero(np.diff(member.dense_barrier) > 1e-10)),
                "maximum_sampled_dense_barrier_rise": float(np.maximum(np.diff(member.dense_barrier), 0).max(initial=0)),
                "terminal_segment_length_cv": float(member.cv[-1]),
                "terminal_equidistance_relative_defect": float(member.equidistance[-1]),
                "source": str(member.source),
                "source_sha256": sha256(member.source),
                "dense_source": str(member.dense_source),
                "dense_source_sha256": sha256(member.dense_source),
            }
        )
    csv_path = output_dir / "alanine_clean_four_level_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "alanine_clean_four_level_audit.json"
    json_path.write_text(
        json.dumps(
            {
                "reference_barrier_kcal_per_mol": REFERENCE_BARRIER_KCAL,
                "dense_checkpoint_policy": "common log-cost and I_perp first-passage checkpoints; raw sampled values",
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    return [csv_path, json_path]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "figures/clean_four_level_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    series = load_series(root)
    if len(series) != 60:
        raise RuntimeError(f"Expected 60 clean-grid series, found {len(series)}")
    outputs = []
    outputs += plot_exact(series, output_dir)
    outputs += plot_inexact(series, output_dir)
    outputs += plot_stage4_cost(series, output_dir)
    outputs += write_summary(series, output_dir)
    manifest = {
        "outputs": [{"path": str(path), "sha256": sha256(path)} for path in outputs],
        "series": len(series),
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
