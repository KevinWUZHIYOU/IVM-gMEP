#!/usr/bin/env python3
"""Clear, protocol-separated figures for the four clean-start Ala IVM levels."""

from __future__ import annotations

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
}
STAGE_NAMES = {
    "stage1": "Stage I — exact-like IVM, no resampling",
    "stage2": "Stage II — exact-like IVM + resampling",
    "stage3": "Stage III — baseline inexact IVM + resampling",
    "stage4": "Stage IV — optimized inexact IVM + resampling",
}


@dataclass
class RawSeries:
    stage: str
    rtol: float
    alpha: float
    source: Path
    dense_source: Path
    force: np.ndarray
    iperp: np.ndarray
    dense_force: np.ndarray
    barrier_error: np.ndarray


@dataclass
class ProductionSeries:
    alpha: float
    combined_source: Path
    archive_source: Path
    raw_force: np.ndarray
    raw_iperp: np.ndarray
    candidate_force: np.ndarray
    candidate_charged_force: np.ndarray
    candidate_iperp: np.ndarray
    candidate_barrier_error10: np.ndarray
    accepted: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(saved, key):
    return np.asarray(saved[key]).item()


def alpha_tag(alpha: float) -> str:
    return "inf" if math.isinf(alpha) else f"{alpha:g}".replace(".", "p")


def alpha_label(alpha: float) -> str:
    return r"$\infty$" if math.isinf(alpha) else f"{alpha:g}"


def alpha_color(alpha: float):
    if math.isinf(alpha):
        return "#222222"
    finite = np.asarray(ALPHAS[:-1], dtype=float)
    value = (np.log10(alpha) - np.log10(finite).min()) / (
        np.log10(finite).max() - np.log10(finite).min()
    )
    return mpl.colormaps["viridis"](0.05 + 0.90 * value)


def alpha_style(alpha: float):
    selected = np.isclose(alpha, 20.0)
    return {
        "color": alpha_color(alpha),
        "ls": (0, (3.0, 1.7)) if math.isinf(alpha) else "-",
        "lw": 2.8 if selected else 1.15,
        "alpha": 1.0 if selected or math.isinf(alpha) else 0.82,
        "zorder": 10 if selected else 3,
    }


def legend_handles():
    handles = []
    for alpha in ALPHAS:
        style = alpha_style(alpha)
        handles.append(
            Line2D(
                [0], [0], color=style["color"], ls=style["ls"],
                lw=style["lw"], marker="o" if np.isclose(alpha, 20.0) else None,
                ms=4.0, markeredgecolor="#222222", markeredgewidth=0.6,
                label=alpha_label(alpha),
            )
        )
    return handles


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.2,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def style_axis(axis, xlabel=True):
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, which="major", color="#d7d7d7", ls="--", lw=0.55, alpha=0.72)
    axis.tick_params(direction="out", length=3.0, width=0.7)
    if xlabel:
        axis.set_xlabel(r"Whole-path force evaluations, $N_F$")


def panel_label(axis, label: str):
    axis.text(-0.115, 1.045, label, transform=axis.transAxes, fontweight="bold")


def dense_path(root: Path, stage: str, rtol: float, alpha: float) -> Path:
    token_r = f"{rtol:g}".replace(".", "p")
    token_a = alpha_tag(alpha)
    path = root / "runs/diagnostics" / (
        f"dense_barrier_clean_{stage}_r{token_r}_a{token_a}_common_checkpoints.npz"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_raw_series(root: Path) -> list[RawSeries]:
    records = []
    for stage, pattern in STAGE_GLOBS.items():
        for source in sorted(root.glob(pattern)):
            with np.load(source, allow_pickle=False) as saved:
                alpha = float(scalar(saved, "alpha"))
                rtol = float(scalar(saved, "plbfgs_kkt_rtol"))
                force = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
                iperp = np.asarray(saved["i_perp"], dtype=float)
                calls_per_image = np.asarray(saved["calls_per_image"], dtype=int)
            diagnostic = dense_path(root, stage, rtol, alpha)
            with np.load(diagnostic, allow_pickle=False) as dense:
                labels = np.asarray(dense["calls"], dtype=int)
                barrier = np.asarray(dense["adaptive_barrier_kcal"], dtype=float)
            mapping = {int(call): int(cost) for call, cost in zip(calls_per_image, force)}
            dense_force = np.asarray([mapping[int(call)] for call in labels], dtype=int)
            if not (len(force) == len(iperp) and np.all(np.diff(force) > 0)):
                raise RuntimeError(f"Invalid force history: {source}")
            if not np.all(np.diff(dense_force) > 0):
                raise RuntimeError(f"Invalid dense force history: {diagnostic}")
            records.append(
                RawSeries(
                    stage=stage,
                    rtol=rtol,
                    alpha=alpha,
                    source=source,
                    dense_source=diagnostic,
                    force=force,
                    iperp=iperp,
                    dense_force=dense_force,
                    barrier_error=np.maximum(
                        np.abs(barrier - REFERENCE_BARRIER_KCAL), 1e-8
                    ),
                )
            )
    expected = {"stage1": 20, "stage2": 20, "stage3": 10}
    counts = {stage: sum(item.stage == stage for item in records) for stage in expected}
    if counts != expected:
        raise RuntimeError(f"Unexpected Stage I--III counts: {counts}")
    return records


def load_production_series(root: Path) -> list[ProductionSeries]:
    records = []
    for alpha in ALPHAS:
        tag = alpha_tag(alpha)
        combined_source = root / "runs/stage4_inexact_optimized_resample" / (
            f"stage4_uniform_v3_a{tag}_phaseA34_phaseB220.npz"
        )
        archive_source = root / "runs/pareto_archive" / (
            f"pareto_clean_stage4_v4_a{tag}_phaseA34_phaseB220.npz"
        )
        with np.load(combined_source, allow_pickle=False) as combined, np.load(
            archive_source, allow_pickle=False
        ) as archive:
            raw_force = np.asarray(combined["total_force_evaluations"], dtype=int)
            raw_iperp = np.asarray(combined["i_perp"], dtype=float)
            candidate_force = np.asarray(archive["candidate_calls"], dtype=int)
            screen_cost = np.asarray(
                archive["candidate_screen_force_equivalent_cost"], dtype=int
            )
            candidate_charged_force = candidate_force + np.cumsum(screen_cost)
            candidate_iperp = np.asarray(archive["candidate_i_perp"], dtype=float)
            candidate_barrier = np.asarray(
                archive["candidate_screen_barrier_kcal"], dtype=float
            )
            accepted = np.asarray(archive["candidate_accepted"], dtype=bool)
            if bool(scalar(archive, "reference_barrier_used")):
                raise RuntimeError(f"Reference barrier leaked into {archive_source}")
        records.append(
            ProductionSeries(
                alpha=alpha,
                combined_source=combined_source,
                archive_source=archive_source,
                raw_force=raw_force,
                raw_iperp=raw_iperp,
                candidate_force=candidate_force,
                candidate_charged_force=candidate_charged_force,
                candidate_iperp=candidate_iperp,
                candidate_barrier_error10=np.maximum(
                    np.abs(candidate_barrier - REFERENCE_BARRIER_KCAL), 1e-8
                ),
                accepted=accepted,
            )
        )
    return records


def plot_raw_metric(axis, members, metric: str):
    for member in sorted(members, key=lambda item: item.alpha):
        style = alpha_style(member.alpha)
        if metric == "iperp":
            x, y = member.force, member.iperp
        else:
            x, y = member.dense_force, member.barrier_error
        axis.plot(x, y, **style)
        if np.isclose(member.alpha, 20.0):
            axis.scatter(
                x[-1], y[-1], s=25, color=style["color"], edgecolor="#222222",
                linewidth=0.7, zorder=12,
            )


def add_shared_legend(figure, y=0.985):
    figure.legend(
        handles=legend_handles(), title=r"Fixed $\alpha$ (20 emphasized)",
        loc="upper center", bbox_to_anchor=(0.5, y), ncol=5, frameon=False,
        handlelength=2.4, columnspacing=1.4,
    )


def finish_metric_axis(axis, metric: str, title: str):
    style_axis(axis)
    axis.set_title(title)
    if metric == "iperp":
        axis.set_ylabel(r"Raw perpendicular residual, $I_\perp$")
        axis.set_ylim(1e-5, 2e3)
    else:
        axis.set_ylabel(r"$|\Delta E-\Delta E^*|$ (kcal/mol)")
        axis.set_ylim(1e-4, 1e2)


def save_figure(figure, stem: Path):
    outputs = []
    for suffix in ("pdf", "svg", "png"):
        output = stem.with_suffix(f".{suffix}")
        figure.savefig(output, dpi=320)
        outputs.append(output)
    plt.close(figure)
    return outputs


def plot_exact_stage(records, stage: str, output_dir: Path):
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.3), sharex="col")
    for column, rtol in enumerate((0.2, 0.1)):
        members = [
            item for item in records
            if item.stage == stage and np.isclose(item.rtol, rtol)
        ]
        plot_raw_metric(axes[0, column], members, "iperp")
        plot_raw_metric(axes[1, column], members, "barrier")
        finish_metric_axis(axes[0, column], "iperp", f"KKT relative tolerance r = {rtol:g}")
        finish_metric_axis(axes[1, column], "barrier", f"KKT relative tolerance r = {rtol:g}")
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    for index, axis in enumerate(axes.flat):
        panel_label(axis, chr(ord("a") + index))
    figure.suptitle(STAGE_NAMES[stage], y=0.845, fontweight="bold")
    add_shared_legend(figure, y=0.985)
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.09, top=0.755, wspace=0.18, hspace=0.30)
    return save_figure(figure, output_dir / f"alanine_clear_{stage}_two_metric")


def plot_stage3(records, output_dir: Path):
    members = [item for item in records if item.stage == "stage3"]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.15))
    plot_raw_metric(axes[0], members, "iperp")
    plot_raw_metric(axes[1], members, "barrier")
    finish_metric_axis(axes[0], "iperp", "Raw IVM dynamics")
    finish_metric_axis(axes[1], "barrier", "Independent adaptive densification")
    for index, axis in enumerate(axes):
        panel_label(axis, chr(ord("a") + index))
    figure.suptitle(STAGE_NAMES["stage3"] + " (r = 0.8, five inner calls)", y=0.805, fontweight="bold")
    add_shared_legend(figure, y=0.985)
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.15, top=0.70, wspace=0.32)
    return save_figure(figure, output_dir / "alanine_clear_stage3_two_metric")


def plot_stage4(records, output_dir: Path):
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    for member in sorted(records, key=lambda item: item.alpha):
        style = alpha_style(member.alpha)
        axes[0, 0].plot(member.raw_force, member.raw_iperp, **style)
        if member.candidate_force.size:
            axes[0, 1].plot(
                member.candidate_force, member.candidate_barrier_error10, **style
            )
            axes[0, 1].scatter(
                member.candidate_force[member.accepted],
                member.candidate_barrier_error10[member.accepted],
                s=24 if np.isclose(member.alpha, 20.0) else 13,
                color=style["color"], edgecolor="#222222", linewidth=0.55,
                zorder=style["zorder"] + 1,
            )
            axes[1, 0].plot(
                member.candidate_charged_force[member.accepted],
                member.candidate_iperp[member.accepted], **style,
            )
            axes[1, 0].scatter(
                member.candidate_charged_force[member.accepted],
                member.candidate_iperp[member.accepted],
                s=24 if np.isclose(member.alpha, 20.0) else 13,
                color=style["color"], edgecolor="#222222", linewidth=0.55,
                zorder=style["zorder"] + 1,
            )
            axes[1, 1].plot(
                member.candidate_charged_force[member.accepted],
                member.candidate_barrier_error10[member.accepted], **style,
            )
            axes[1, 1].scatter(
                member.candidate_charged_force[member.accepted],
                member.candidate_barrier_error10[member.accepted],
                s=24 if np.isclose(member.alpha, 20.0) else 13,
                color=style["color"], edgecolor="#222222", linewidth=0.55,
                zorder=style["zorder"] + 1,
            )
    titles = (
        "Raw dynamics — optimization cost",
        "All screened complete states — optimization cost",
        "Accepted Pareto incumbents — fully charged cost",
        "Accepted Pareto incumbents — fully charged cost",
    )
    for index, axis in enumerate(axes.flat):
        style_axis(axis)
        axis.set_title(titles[index])
        panel_label(axis, chr(ord("a") + index))
    axes[0, 0].set_ylabel(r"Raw perpendicular residual, $I_\perp$")
    axes[0, 1].set_ylabel(r"Barrier10 error (kcal/mol)")
    axes[1, 0].set_ylabel(r"Accepted $I_\perp$")
    axes[1, 1].set_ylabel(r"Accepted barrier10 error (kcal/mol)")
    axes[0, 0].set_ylim(1e-5, 2e3)
    axes[0, 1].set_ylim(1e-4, 1.0)
    axes[1, 0].set_ylim(1e-4, 1e-2)
    axes[1, 1].set_ylim(1e-4, 1.0)
    axes[1, 0].axhline(1.9e-4, color="#777777", ls=":", lw=0.9)
    axes[1, 0].text(
        0.98, 0.08, r"target $1.9\times10^{-4}$", transform=axes[1, 0].transAxes,
        ha="right", va="bottom", color="#555555", fontsize=8.2,
    )
    axes[0, 1].text(
        0.03, 0.06, r"$\alpha=0.01,0.1$ never reach the screen activation threshold",
        transform=axes[0, 1].transAxes, ha="left", va="bottom", fontsize=7.8,
        color="#555555",
    )
    figure.suptitle(
        STAGE_NAMES["stage4"] + " (uniform A34/B220 policy)",
        y=0.845, fontweight="bold",
    )
    add_shared_legend(figure, y=0.985)
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.09, top=0.755, wspace=0.29, hspace=0.32)
    return save_figure(figure, output_dir / "alanine_clear_stage4_raw_and_accepted")


def selected_alpha20_stage4(root: Path, production: list[ProductionSeries]):
    member = next(item for item in production if np.isclose(item.alpha, 20.0))
    accepted_calls = member.candidate_force[member.accepted]
    accepted_cost = member.candidate_charged_force[member.accepted]
    accepted_i = member.candidate_iperp[member.accepted]
    dense_path = root / "runs/diagnostics" / (
        "dense_barrier_clean_stage4_pareto_screen_a20_phaseA34_phaseB220.npz"
    )
    with np.load(dense_path, allow_pickle=False) as dense:
        dense_calls = np.asarray(dense["calls"], dtype=int)
        barrier20 = np.asarray(dense["barrier_20_kcal"], dtype=float)
    mapping = {int(call): float(value) for call, value in zip(dense_calls, barrier20)}
    accepted_error = np.asarray(
        [abs(mapping[int(call)] - REFERENCE_BARRIER_KCAL) for call in accepted_calls]
    )
    return accepted_cost, accepted_i, accepted_error, dense_path


def plot_overview(root: Path, raw, production, output_dir: Path):
    figure, axes = plt.subplots(4, 2, figsize=(10.6, 11.4), sharex=False)
    row_specs = (
        ("stage1", "Stage I: exact-like, no resampling"),
        ("stage2", "Stage II: exact-like + resampling"),
        ("stage3", "Stage III: baseline inexact + resampling"),
    )
    selected_sources = []
    for row, (stage, title) in enumerate(row_specs):
        members = [item for item in raw if item.stage == stage and np.isclose(item.alpha, 20.0)]
        for member in sorted(members, key=lambda item: item.rtol, reverse=True):
            label = f"r={member.rtol:g}" if stage in ("stage1", "stage2") else "r=0.8"
            linestyle = "--" if np.isclose(member.rtol, 0.2) else "-"
            axes[row, 0].plot(member.force, member.iperp, lw=2.0, ls=linestyle, label=label)
            axes[row, 1].plot(
                member.dense_force, member.barrier_error, lw=2.0, ls=linestyle, label=label
            )
            selected_sources.extend((member.source, member.dense_source))
        axes[row, 0].text(
            0.02, 0.08, title, transform=axes[row, 0].transAxes,
            ha="left", va="bottom", fontsize=9.2, fontweight="bold",
        )
        axes[row, 0].legend(frameon=False, loc="upper right")
    accepted_cost, accepted_i, accepted_error, dense20_source = selected_alpha20_stage4(root, production)
    axes[3, 0].plot(accepted_cost, accepted_i, lw=2.8, marker="o", ms=4.0)
    axes[3, 1].plot(accepted_cost, accepted_error, lw=2.8, marker="o", ms=4.0)
    axes[3, 0].axhline(1.9e-4, color="#777777", ls=":", lw=0.9)
    axes[3, 0].text(
        0.02, 0.08, "Stage IV: optimized + Pareto archive (charged)",
        transform=axes[3, 0].transAxes, ha="left", va="bottom",
        fontsize=9.2, fontweight="bold",
    )
    for row in range(4):
        style_axis(axes[row, 0])
        style_axis(axes[row, 1])
        axes[row, 0].set_ylim(1e-5, 2e3)
        axes[row, 1].set_ylim(1e-4, 1e2)
        panel_label(axes[row, 0], chr(ord("a") + 2 * row))
        panel_label(axes[row, 1], chr(ord("a") + 2 * row + 1))
    axes[0, 0].set_title(r"Perpendicular residual $I_\perp$")
    axes[0, 1].set_title(r"Densified barrier error $|\Delta E-\Delta E^*|$")
    for row in range(4):
        axes[row, 0].set_ylabel(r"$I_\perp$")
        axes[row, 1].set_ylabel(r"kcal mol$^{-1}$")
    figure.suptitle(
        r"Clean-start four-level audit at fixed integer $\alpha=20$",
        y=0.992, fontweight="bold",
    )
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.06, top=0.96, wspace=0.24, hspace=0.33)
    outputs = save_figure(figure, output_dir / "alanine_clear_four_level_alpha20_overview")
    return outputs, selected_sources + [dense20_source]


def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "figures/clean_four_level_audit/clear_four_groups_v4"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    raw = load_raw_series(root)
    production = load_production_series(root)
    outputs = []
    outputs += plot_exact_stage(raw, "stage1", output_dir)
    outputs += plot_exact_stage(raw, "stage2", output_dir)
    outputs += plot_stage3(raw, output_dir)
    outputs += plot_stage4(production, output_dir)
    overview_outputs, overview_sources = plot_overview(root, raw, production, output_dir)
    outputs += overview_outputs
    sources = sorted(
        {item.source for item in raw}
        | {item.dense_source for item in raw}
        | {item.combined_source for item in production}
        | {item.archive_source for item in production}
        | set(overview_sources)
    )
    manifest = {
        "status": "complete",
        "figure_scope": "four clean-start levels plotted separately plus alpha20 overview",
        "stage_counts": {"stage1": 20, "stage2": 20, "stage3": 10, "stage4": 10},
        "stage4_policy": "uniform Phase-A34/Phase-B220 with reference-free charged Pareto archive",
        "stage4_barrier_panel": (
            "10-subdivision screen values; error to reference computed only for plotting, "
            "never used by the archive decision"
        ),
        "overview_stage4_barrier": "independent 20-subdivision reporting diagnostic",
        "outputs": [{"path": str(path), "sha256": sha256(path)} for path in outputs],
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in sources],
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
