#!/usr/bin/env python3
"""Long-budget alpha audit with complete histories and honest budget endpoints.

This script fixes the visual ambiguity of the v5 first-passage-only plot.  It
keeps every raw optimization point in a full-history figure and augments the
cost-to-precision frontiers with dashed tails to the actual terminal charged
cost whenever later work fails to improve the incumbent precision.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ALPHAS = (0.01, 0.1, 1.0, 10.0, 20.0, 30.0, 100.0, 1000.0, 10000.0, math.inf)
EXPECTED_INITIAL_SHA256 = (
    "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
)
TARGET_IPERP = 1.9e-4
CMAP = mpl.colormaps["viridis_r"]
NORM = mpl.colors.Normalize(vmin=-2.0, vmax=4.0)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h", "<")


@dataclass
class AlphaRun:
    family: str
    alpha: float
    raw_source: Path
    archive_source: Optional[Path]
    raw_force: np.ndarray
    raw_iperp: np.ndarray
    frontier_force: np.ndarray
    frontier_iperp: np.ndarray
    terminal_charged_force: int
    terminal_best_iperp: float
    raw_stop_reason: str
    target_reached: bool
    screen_force: int
    initial_sha256: str
    protocol_budget: str = ""
    target_force: Optional[int] = None
    target_iperp: Optional[float] = None
    common_force_cutoff: Optional[int] = None
    last_actual_raw_force: Optional[int] = None
    cutoff_carried_forward: bool = False


def scalar(saved, key):
    return np.asarray(saved[key]).item()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alpha_tag(alpha: float) -> str:
    return "inf" if math.isinf(alpha) else f"{alpha:g}".replace(".", "p")


def alpha_label(alpha: float) -> str:
    if math.isinf(alpha):
        return r"$\infty$"
    exponent = round(math.log10(alpha))
    if np.isclose(alpha, 10.0**exponent):
        return rf"$10^{{{exponent}}}$"
    return f"{alpha:g}"


def alpha_color(alpha: float):
    return "#161616" if math.isinf(alpha) else CMAP(NORM(math.log10(alpha)))


def alpha_style(alpha: float) -> dict:
    selected = np.isclose(alpha, 20.0)
    return {
        "color": alpha_color(alpha),
        "ls": (0, (4.0, 2.0)) if math.isinf(alpha) else "-",
        "lw": 2.7 if selected else 1.15,
        "alpha": 1.0 if selected or math.isinf(alpha) else 0.82,
        "zorder": 12 if selected else 3,
    }


def marker_for(alpha: float) -> str:
    if math.isinf(alpha):
        return "o"
    finite = ALPHAS[:-1]
    return MARKERS[finite.index(alpha) % len(MARKERS)]


def record_frontier(force, iperp):
    force = np.asarray(force, dtype=float)
    iperp = np.asarray(iperp, dtype=float)
    keep = np.isfinite(force) & np.isfinite(iperp) & (force > 0) & (iperp > 0)
    force, iperp = force[keep], iperp[keep]
    order = np.argsort(force, kind="stable")
    force, iperp = force[order], iperp[order]
    result_force, result_iperp, incumbent = [], [], math.inf
    for item_force in np.unique(force):
        item_iperp = float(np.min(iperp[force == item_force]))
        if item_iperp < incumbent:
            result_force.append(float(item_force))
            result_iperp.append(item_iperp)
            incumbent = item_iperp
    return np.asarray(result_force), np.asarray(result_iperp)


def validate_alpha_grid(records: list[AlphaRun], family: str) -> None:
    observed = sorted(alpha_tag(item.alpha) for item in records)
    expected = sorted(alpha_tag(alpha) for alpha in ALPHAS)
    if observed != expected:
        raise RuntimeError(f"{family} alpha grid mismatch: {observed} != {expected}")


def load_stage1(root: Path) -> list[AlphaRun]:
    sources = sorted(
        (root / "runs/stage1_exact_no_resample").glob(
            "ivm_clean_stage1_r0p1_*budget320.npz"
        )
    )
    records = []
    for source in sources:
        with np.load(source, allow_pickle=False) as saved:
            alpha = float(scalar(saved, "alpha"))
            rtol = float(scalar(saved, "plbfgs_kkt_rtol"))
            initial_sha = str(scalar(saved, "initial_path_sha256"))
            force = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
            iperp = np.asarray(saved["i_perp"], dtype=float)
            stop_reason = str(scalar(saved, "stop_reason"))
            budget = int(scalar(saved, "budget"))
        if not np.isclose(rtol, 0.1) or budget != 320:
            raise RuntimeError(f"Unexpected Stage-I protocol in {source}")
        if initial_sha != EXPECTED_INITIAL_SHA256:
            raise RuntimeError(f"Common-initial-path mismatch in {source}")
        if len(force) != len(iperp) or np.any(np.diff(force) <= 0):
            raise RuntimeError(f"Invalid Stage-I history in {source}")
        frontier_force, frontier_iperp = record_frontier(force, iperp)
        records.append(
            AlphaRun(
                family="stage1_exact_like_r0p1_no_resampling_budget320",
                alpha=alpha,
                raw_source=source,
                archive_source=None,
                raw_force=force,
                raw_iperp=iperp,
                frontier_force=frontier_force,
                frontier_iperp=frontier_iperp,
                terminal_charged_force=int(force[-1]),
                terminal_best_iperp=float(np.min(iperp)),
                raw_stop_reason=stop_reason,
                target_reached=False,
                screen_force=0,
                initial_sha256=initial_sha,
            )
        )
    validate_alpha_grid(records, "Stage I")
    return records


def load_stage1_minimum_force(
    root: Path,
    minimum_terminal_force: int,
    minimum_macro_budget: int = 1000,
) -> list[AlphaRun]:
    """Choose the shortest new Stage-I run above a common force floor."""
    grouped = {alpha: [] for alpha in ALPHAS}
    sources = sorted(
        (root / "runs/stage1_exact_no_resample").glob(
            "ivm_clean_stage1_r0p1_*budget*.npz"
        )
    )
    for source in sources:
        with np.load(source, allow_pickle=False) as saved:
            alpha = float(scalar(saved, "alpha"))
            rtol = float(scalar(saved, "plbfgs_kkt_rtol"))
            initial_sha = str(scalar(saved, "initial_path_sha256"))
            budget = int(scalar(saved, "budget"))
            force = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
            iperp = np.asarray(saved["i_perp"], dtype=float)
            stop_reason = str(scalar(saved, "stop_reason"))
        if not np.isclose(rtol, 0.1) or initial_sha != EXPECTED_INITIAL_SHA256:
            continue
        if budget < minimum_macro_budget or int(force[-1]) <= minimum_terminal_force:
            continue
        if len(force) != len(iperp) or np.any(np.diff(force) <= 0):
            raise RuntimeError(f"Invalid Stage-I history in {source}")
        match = next(
            (
                value
                for value in ALPHAS
                if (math.isinf(value) and math.isinf(alpha))
                or (not math.isinf(value) and np.isclose(value, alpha))
            ),
            None,
        )
        if match is not None:
            grouped[match].append(
                (int(force[-1]), budget, source, force, iperp, stop_reason, initial_sha)
            )

    records = []
    for alpha in ALPHAS:
        if not grouped[alpha]:
            raise RuntimeError(
                f"No Stage-I alpha={alpha} run exceeds NF={minimum_terminal_force}"
            )
        terminal, budget, source, force, iperp, stop_reason, initial_sha = min(
            grouped[alpha], key=lambda item: (item[1], item[0])
        )
        frontier_force, frontier_iperp = record_frontier(force, iperp)
        records.append(
            AlphaRun(
                family=(
                    "stage1_exact_like_r0p1_no_resampling_"
                    f"minimumNF{minimum_terminal_force}"
                ),
                alpha=alpha,
                raw_source=source,
                archive_source=None,
                raw_force=force,
                raw_iperp=iperp,
                frontier_force=frontier_force,
                frontier_iperp=frontier_iperp,
                terminal_charged_force=terminal,
                terminal_best_iperp=float(np.min(iperp)),
                raw_stop_reason=stop_reason,
                target_reached=False,
                screen_force=0,
                initial_sha256=initial_sha,
                protocol_budget=f"macro_budget={budget}",
            )
        )
    validate_alpha_grid(records, "Stage I minimum-force")
    return records


def load_stage4(root: Path) -> list[AlphaRun]:
    records = []
    for alpha in ALPHAS:
        tag = alpha_tag(alpha)
        source = root / "runs/stage4_inexact_optimized_resample" / (
            f"stage4_uniform_v3_a{tag}_phaseA34_phaseB440_fullbudget.npz"
        )
        archive_source = root / "runs/pareto_archive" / (
            f"pareto_clean_stage4_v6_fullbudget_a{tag}_phaseA34_phaseB440.npz"
        )
        with np.load(source, allow_pickle=False) as raw, np.load(
            archive_source, allow_pickle=False
        ) as archive:
            observed_alpha = float(scalar(raw, "alpha"))
            if not (
                (math.isinf(alpha) and math.isinf(observed_alpha))
                or np.isclose(alpha, observed_alpha)
            ):
                raise RuntimeError(f"Alpha mismatch in {source}")
            initial_sha = str(scalar(raw, "common_initial_path_sha256"))
            force = np.asarray(raw["total_force_evaluations"], dtype=int)
            iperp = np.asarray(raw["i_perp"], dtype=float)
            stop_reason = str(scalar(raw, "stop_reason"))
            phase_a = int(scalar(raw, "phase_a_budget"))
            phase_b = int(scalar(raw, "phase_b_budget"))
            stop_complete_iperp = float(scalar(raw, "stop_complete_iperp"))
            candidate_calls = np.asarray(archive["candidate_calls"], dtype=int)
            candidate_iperp = np.asarray(archive["candidate_i_perp"], dtype=float)
            screen_cost = np.asarray(
                archive["candidate_screen_force_equivalent_cost"], dtype=int
            )
            accepted = np.asarray(archive["candidate_accepted"], dtype=bool)
            target_reached = bool(scalar(archive, "target_reached"))
            total_charged = int(
                scalar(archive, "total_charged_force_equivalent_evaluations")
            )
            total_screen = int(
                scalar(archive, "screen_force_equivalent_evaluations")
            )
            reference_used = bool(scalar(archive, "reference_barrier_used"))
        if (
            initial_sha != EXPECTED_INITIAL_SHA256
            or phase_a != 34
            or phase_b != 440
            or stop_complete_iperp != -1.0
            or stop_reason != "evaluation_budget"
        ):
            raise RuntimeError(f"Unexpected Stage-IV protocol in {source}")
        if reference_used:
            raise RuntimeError(f"Reference barrier leaked into {archive_source}")
        if len(force) != len(iperp) or np.any(np.diff(force) <= 0):
            raise RuntimeError(f"Invalid Stage-IV history in {source}")
        if not (
            len(candidate_calls)
            == len(candidate_iperp)
            == len(screen_cost)
            == len(accepted)
        ):
            raise RuntimeError(f"Invalid Pareto archive in {archive_source}")

        if candidate_calls.size:
            prefix = force < int(candidate_calls[0])
            charged_candidate = candidate_calls + np.cumsum(screen_cost)
            joint_force = np.concatenate((force[prefix], charged_candidate[accepted]))
            joint_iperp = np.concatenate((iperp[prefix], candidate_iperp[accepted]))
        else:
            joint_force, joint_iperp = force, iperp
        frontier_force, frontier_iperp = record_frontier(joint_force, joint_iperp)
        if not len(frontier_force):
            raise RuntimeError(f"Empty joint frontier in {archive_source}")
        if total_charged < frontier_force[-1]:
            raise RuntimeError(f"Terminal charged cost precedes frontier in {archive_source}")
        records.append(
            AlphaRun(
                family=(
                    "stage4_inexact_resample_A34_B440_fullbudget_"
                    "reference_free_pareto"
                ),
                alpha=alpha,
                raw_source=source,
                archive_source=archive_source,
                raw_force=force,
                raw_iperp=iperp,
                frontier_force=frontier_force,
                frontier_iperp=frontier_iperp,
                terminal_charged_force=total_charged,
                terminal_best_iperp=float(frontier_iperp[-1]),
                raw_stop_reason=stop_reason,
                target_reached=target_reached,
                screen_force=total_screen,
                initial_sha256=initial_sha,
            )
        )
    validate_alpha_grid(records, "Stage IV")
    return records


def load_stage4_minimum_force(
    root: Path,
    minimum_terminal_force: int,
    minimum_phase_b: int,
    archive_prefix: str,
) -> list[AlphaRun]:
    """Load uncensored Stage-IV runs and complete-screen audit archives."""
    records = []
    directory = root / "runs/stage4_inexact_optimized_resample"
    for alpha in ALPHAS:
        tag = alpha_tag(alpha)
        candidates = []
        for source in sorted(
            directory.glob(
                f"stage4_uniform_v3_a{tag}_phaseA34_phaseB*_fullbudget.npz"
            )
        ):
            with np.load(source, allow_pickle=False) as raw:
                phase_b = int(scalar(raw, "phase_b_budget"))
                force = np.asarray(raw["total_force_evaluations"], dtype=int)
            if phase_b >= minimum_phase_b and int(force[-1]) > minimum_terminal_force:
                candidates.append((phase_b, int(force[-1]), source))
        if not candidates:
            raise RuntimeError(
                f"No Stage-IV alpha={alpha} run exceeds NF={minimum_terminal_force}"
            )
        phase_b, _, source = min(candidates, key=lambda item: (item[0], item[1]))
        archive_source = root / "runs/pareto_archive" / (
            f"pareto_{archive_prefix}_a{tag}_phaseA34_phaseB{phase_b}.npz"
        )
        with np.load(source, allow_pickle=False) as raw, np.load(
            archive_source, allow_pickle=False
        ) as archive:
            initial_sha = str(scalar(raw, "common_initial_path_sha256"))
            force = np.asarray(raw["total_force_evaluations"], dtype=int)
            iperp = np.asarray(raw["i_perp"], dtype=float)
            stop_reason = str(scalar(raw, "stop_reason"))
            phase_a = int(scalar(raw, "phase_a_budget"))
            stop_complete_iperp = float(scalar(raw, "stop_complete_iperp"))
            candidate_calls = np.asarray(archive["candidate_calls"], dtype=int)
            candidate_iperp = np.asarray(archive["candidate_i_perp"], dtype=float)
            screen_cost = np.asarray(
                archive["candidate_screen_force_equivalent_cost"], dtype=int
            )
            accepted = np.asarray(archive["candidate_accepted"], dtype=bool)
            target_reached = bool(scalar(archive, "target_reached"))
            continue_after_target = bool(scalar(archive, "continue_after_target"))
            total_charged = int(
                scalar(
                    archive,
                    "full_audit_total_charged_force_equivalent_evaluations",
                )
            )
            total_screen = int(
                scalar(archive, "full_audit_screen_force_equivalent_evaluations")
            )
            first_target_force = int(
                scalar(
                    archive,
                    "first_target_total_charged_force_equivalent_evaluations",
                )
            )
            first_target_i = float(scalar(archive, "first_target_i_perp"))
            reference_used = bool(scalar(archive, "reference_barrier_used"))
        if (
            initial_sha != EXPECTED_INITIAL_SHA256
            or phase_a != 34
            or phase_b < minimum_phase_b
            or stop_complete_iperp != -1.0
            or stop_reason != "evaluation_budget"
            or not continue_after_target
            or reference_used
            or int(force[-1]) <= minimum_terminal_force
        ):
            raise RuntimeError(f"Unexpected Stage-IV audit protocol in {source}")
        if len(force) != len(iperp) or np.any(np.diff(force) <= 0):
            raise RuntimeError(f"Invalid Stage-IV history in {source}")
        if candidate_calls.size:
            prefix = force < int(candidate_calls[0])
            charged_candidate = candidate_calls + np.cumsum(screen_cost)
            joint_force = np.concatenate((force[prefix], charged_candidate[accepted]))
            joint_iperp = np.concatenate((iperp[prefix], candidate_iperp[accepted]))
        else:
            joint_force, joint_iperp = force, iperp
        frontier_force, frontier_iperp = record_frontier(joint_force, joint_iperp)
        records.append(
            AlphaRun(
                family=(
                    "stage4_inexact_resample_uncensored_"
                    f"minimumNF{minimum_terminal_force}_reference_free_pareto"
                ),
                alpha=alpha,
                raw_source=source,
                archive_source=archive_source,
                raw_force=force,
                raw_iperp=iperp,
                frontier_force=frontier_force,
                frontier_iperp=frontier_iperp,
                terminal_charged_force=total_charged,
                terminal_best_iperp=float(frontier_iperp[-1]),
                raw_stop_reason=stop_reason,
                target_reached=target_reached,
                screen_force=total_screen,
                initial_sha256=initial_sha,
                protocol_budget=f"Phase A=34; Phase B={phase_b}",
                target_force=first_target_force if target_reached else None,
                target_iperp=first_target_i if target_reached else None,
            )
        )
    validate_alpha_grid(records, "Stage IV minimum-force")
    return records


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.2,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.2,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def grid(axis) -> None:
    axis.grid(True, which="major", color="#d8d8d8", lw=0.48, ls="--", alpha=0.72)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=3.0, width=0.7)


def plot_raw(
    axis,
    records: list[AlphaRun],
    show_ylabel: bool,
    xmax: Optional[float] = None,
) -> None:
    for item in records:
        style = alpha_style(item.alpha)
        marker = marker_for(item.alpha)
        axis.plot(
            item.raw_force,
            item.raw_iperp,
            marker=marker,
            markersize=2.5 if np.isclose(item.alpha, 20.0) else 2.0,
            markevery=max(1, len(item.raw_force) // 12),
            markeredgewidth=0.25,
            **style,
        )
        completed = item.raw_stop_reason == "complete_reparameterized_i_perp_target"
        axis.scatter(
            item.raw_force[-1],
            item.raw_iperp[-1],
            marker="*" if completed else "X",
            s=48 if completed else 37,
            facecolor=style["color"],
            edgecolor="#202020",
            linewidth=0.55,
            zorder=style["zorder"] + 2,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(
        16,
        xmax
        if xmax is not None
        else max(1.25e4, 1.15 * max(item.raw_force[-1] for item in records)),
    )
    axis.set_ylim(4e-6, 1.5e3)
    axis.set_xlabel(r"Whole-path force evaluations, $N_F$")
    axis.set_ylabel(r"Raw perpendicular residual, $I_\perp$" if show_ylabel else "")
    grid(axis)


def plot_budget_frontier(axis, records: list[AlphaRun], show_ylabel: bool) -> None:
    for item in records:
        style = alpha_style(item.alpha)
        marker = marker_for(item.alpha)
        axis.plot(
            item.frontier_iperp,
            item.frontier_force,
            marker=marker,
            markersize=3.0 if np.isclose(item.alpha, 20.0) else 2.3,
            markevery=max(1, len(item.frontier_force) // 9),
            markeredgewidth=0.25,
            **style,
        )
        if item.terminal_charged_force > item.frontier_force[-1]:
            axis.plot(
                [item.terminal_best_iperp, item.terminal_best_iperp],
                [item.frontier_force[-1], item.terminal_charged_force],
                color=style["color"],
                ls=(0, (2.0, 2.0)),
                lw=1.25 if np.isclose(item.alpha, 20.0) else 0.95,
                alpha=0.92,
                zorder=style["zorder"] + 1,
            )
        axis.scatter(
            item.terminal_best_iperp,
            item.terminal_charged_force,
            marker=("*" if item.target_reached else "X")
            if item.target_force is None
            else "X",
            s=(50 if item.target_reached else 38)
            if item.target_force is None
            else 38,
            facecolor=style["color"],
            edgecolor="#202020",
            linewidth=0.55,
            zorder=style["zorder"] + 3,
        )
        if item.target_force is not None and item.target_iperp is not None:
            axis.scatter(
                item.target_iperp,
                item.target_force,
                marker="*",
                s=54,
                facecolor=style["color"],
                edgecolor="#202020",
                linewidth=0.6,
                zorder=style["zorder"] + 4,
            )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1.2e3, 4e-6)
    axis.set_ylim(16, 1.55e4)
    axis.set_xlabel(r"Target $I_\perp$ (smaller is better)")
    axis.set_ylabel("Number of Forces" if show_ylabel else "")
    grid(axis)


def alpha_legend(
    figure,
    y: float,
    records: Optional[list[AlphaRun]] = None,
    title: Optional[str] = "auto",
) -> None:
    alphas = ALPHAS if records is None else tuple(item.alpha for item in records)
    handles = []
    for alpha in alphas:
        style = alpha_style(alpha)
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                ls=style["ls"],
                lw=style["lw"],
                marker=marker_for(alpha),
                ms=3.7,
                markeredgecolor="#202020",
                markeredgewidth=0.35,
                label=alpha_label(alpha),
            )
        )
    resolved_title = title
    if title == "auto":
        resolved_title = (
            r"Fixed $\alpha$ (20 emphasized)"
            if any(np.isclose(alpha, 20.0) for alpha in alphas)
            else r"Fixed $\alpha$ (decade grid)"
        )
    figure.legend(
        handles=handles,
        title=resolved_title,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=10,
        frameon=False,
        handlelength=2.0,
        columnspacing=0.95,
    )


def endpoint_legend(
    axis,
    raw: bool,
    complete_audit: bool = False,
    show_target_passage: bool = True,
) -> None:
    if raw:
        handles = [
            Line2D(
                [0], [0], ls="", marker="X", ms=5.2,
                markerfacecolor="#b0b0b0", markeredgecolor="#202020",
                markeredgewidth=0.5, label="full optimization budget exhausted"
            )
        ]
    else:
        handles = [
            Line2D(
                [0], [0], color="#777777", ls=(0, (2.0, 2.0)), lw=1.0,
                label="spent after last precision improvement",
            ),
            Line2D(
                [0], [0], ls="", marker="X", ms=5.2,
                markerfacecolor="#b0b0b0", markeredgecolor="#202020",
                markeredgewidth=0.5,
                label=(
                    "full charged audit endpoint"
                    if complete_audit
                    else "joint target missed after full budget"
                ),
            ),
        ]
        if show_target_passage:
            handles.append(
                Line2D(
                    [0], [0], ls="", marker="*", ms=6.5,
                    markerfacecolor="#b0b0b0", markeredgecolor="#202020",
                    markeredgewidth=0.5,
                    label=(
                        "first joint-target passage"
                        if complete_audit
                        else "joint target reached"
                    ),
                )
            )
    axis.legend(
        handles=handles,
        frameon=False,
        loc="lower left" if raw else "upper left",
        fontsize=6.7,
    )


def panel_label(axis, label: str) -> None:
    axis.text(-0.13, 1.045, label, transform=axis.transAxes, fontweight="bold", fontsize=10)


def save(figure, stem: Path) -> list[Path]:
    outputs = []
    for suffix in ("pdf", "svg"):
        output = stem.with_suffix(f".{suffix}")
        figure.savefig(output, bbox_inches="tight")
        outputs.append(output)
    output = stem.with_suffix(".png")
    figure.savefig(output, dpi=500, bbox_inches="tight")
    outputs.append(output)
    plt.close(figure)
    return outputs


def make_full_history(
    stage1,
    stage4,
    output_dir: Path,
    version: str = "v6",
    stage1_title: str = r"Stage I: exact-like, no resampling ($r=0.1$; 320 steps)",
    stage4_title: str = "Stage IV: optimized inexact + resampling (full A34/B440)",
    raw_xmax: Optional[float] = None,
    raw_endpoint_label: Optional[str] = None,
) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(9.3, 3.35), sharey=True)
    plot_raw(axes[0], stage1, True, xmax=raw_xmax)
    plot_raw(axes[1], stage4, False, xmax=raw_xmax)
    axes[0].set_title(stage1_title)
    axes[1].set_title(stage4_title)
    axes[1].axhline(TARGET_IPERP, color="#6a6a6a", ls=":", lw=0.9, zorder=1)
    axes[1].text(
        0.98, 0.055, r"$I_\perp$ component of joint target $=1.9\times10^{-4}$",
        transform=axes[1].transAxes, ha="right", va="bottom", fontsize=6.9,
        color="#505050",
    )
    endpoint_legend(axes[1], raw=True)
    if raw_endpoint_label:
        axes[1].get_legend().get_texts()[0].set_text(raw_endpoint_label)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    alpha_legend(figure, 0.995, stage1)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.76, wspace=0.12)
    return save(
        figure,
        output_dir / f"alanine_{version}_alpha_complete_raw_histories_stage1_vs_stage4",
    )


def annotate_joint_endpoints(axis, stage4: list[AlphaRun]) -> None:
    by_alpha = {item.alpha: item for item in stage4}
    selected = by_alpha.get(20.0)
    slow = by_alpha.get(0.01)
    ymax = axis.get_ylim()[1]
    if selected is not None:
        target_force = selected.target_force or selected.terminal_charged_force
        target_iperp = selected.target_iperp or selected.terminal_best_iperp
        axis.annotate(
            rf"$\alpha=20$: target at {target_force:,}",
            xy=(target_iperp, target_force),
            xytext=(8.0e-4, min(2.7e3, 0.48 * ymax)),
            fontsize=6.8,
            ha="center",
            color="#303030",
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.65},
            zorder=30,
        )
    if slow is not None:
        axis.annotate(
            rf"$\alpha=10^{{-2}}$: {slow.terminal_charged_force:,} forces, still {slow.terminal_best_iperp:.2e}",
            xy=(slow.terminal_best_iperp, slow.terminal_charged_force),
            xytext=(2.2e-1, 0.82 * ymax),
            fontsize=6.8,
            ha="center",
            color="#303030",
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.65},
            zorder=30,
        )


def make_budget_frontier(
    stage1,
    stage4,
    output_dir: Path,
    version: str = "v6",
    complete_audit: bool = False,
    endpoint_label: Optional[str] = None,
    minimum_ymax: float = 1.55e4,
    stage1_title: str = "Stage I: first-passage records + full terminal budget",
    stage4_title: str = "Stage IV: joint Pareto records + fully charged terminal budget",
    legend_title: Optional[str] = "auto",
) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(9.3, 3.55), sharey=True)
    plot_budget_frontier(axes[0], stage1, True)
    plot_budget_frontier(axes[1], stage4, False)
    axes[0].set_title(stage1_title)
    axes[1].set_title(stage4_title)
    axes[1].axvline(TARGET_IPERP, color="#6a6a6a", ls=":", lw=0.9, zorder=1)
    axes[1].text(
        0.68, 0.055, r"joint target $1.9\times10^{-4}$",
        transform=axes[1].transAxes, ha="center", va="bottom", fontsize=6.9,
        color="#505050",
    )
    shared_ymax = max(
        minimum_ymax,
        1.35 * max(item.terminal_charged_force for item in stage1 + stage4),
    )
    axes[0].set_ylim(16, shared_ymax)
    axes[1].set_ylim(16, shared_ymax)
    endpoint_legend(
        axes[0],
        raw=False,
        complete_audit=complete_audit,
        show_target_passage=any(item.target_force is not None for item in stage4),
    )
    if endpoint_label:
        axes[0].get_legend().get_texts()[1].set_text(endpoint_label)
    annotate_joint_endpoints(axes[1], stage4)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    alpha_legend(figure, 0.995, stage1, title=legend_title)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.76, wspace=0.12)
    return save(
        figure,
        output_dir / f"alanine_{version}_alpha_budget_aware_cost_to_precision_stage1_vs_stage4",
    )


def make_audit(
    stage1,
    stage4,
    output_dir: Path,
    version: str = "v6",
    stage1_title: str = r"Stage I full history ($r=0.1$; 320 steps)",
    stage4_title: str = "Stage IV full history (uncensored A34/B440)",
    complete_audit: bool = False,
    raw_xmax: Optional[float] = None,
    raw_endpoint_label: Optional[str] = None,
    endpoint_label: Optional[str] = None,
    minimum_ymax: float = 1.55e4,
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(9.6, 6.7))
    plot_raw(axes[0, 0], stage1, True, xmax=raw_xmax)
    plot_raw(axes[0, 1], stage4, False, xmax=raw_xmax)
    plot_budget_frontier(axes[1, 0], stage1, True)
    plot_budget_frontier(axes[1, 1], stage4, False)
    axes[0, 0].set_title(stage1_title)
    axes[0, 1].set_title(stage4_title)
    axes[1, 0].set_title("Stage I first-passage records + terminal budget")
    axes[1, 1].set_title("Stage IV joint records + fully charged terminal budget")
    axes[0, 1].axhline(TARGET_IPERP, color="#6a6a6a", ls=":", lw=0.9, zorder=1)
    axes[1, 1].axvline(TARGET_IPERP, color="#6a6a6a", ls=":", lw=0.9, zorder=1)
    endpoint_legend(axes[0, 1], raw=True)
    if raw_endpoint_label:
        axes[0, 1].get_legend().get_texts()[0].set_text(raw_endpoint_label)
    shared_ymax = max(
        minimum_ymax,
        1.35 * max(item.terminal_charged_force for item in stage1 + stage4),
    )
    axes[1, 0].set_ylim(16, shared_ymax)
    axes[1, 1].set_ylim(16, shared_ymax)
    endpoint_legend(
        axes[1, 0],
        raw=False,
        complete_audit=complete_audit,
        show_target_passage=any(item.target_force is not None for item in stage4),
    )
    if endpoint_label:
        axes[1, 0].get_legend().get_texts()[1].set_text(endpoint_label)
    annotate_joint_endpoints(axes[1, 1], stage4)
    for label, axis in zip(("a", "b", "c", "d"), axes.flat):
        panel_label(axis, label)
    alpha_legend(figure, 0.995, stage1)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.085, top=0.86, wspace=0.16, hspace=0.34)
    return save(
        figure,
        output_dir / f"alanine_{version}_alpha_full_budget_audit_stage1_vs_stage4",
    )


def write_data(
    stage1,
    stage4,
    output_dir: Path,
    version: str = "v6",
    stage1_protocol: str = "exact-like IVM; r=0.1; no resampling; 320 macro-steps",
    stage4_protocol: str = (
        "uniform optimized inexact IVM + resampling; Phase A=34; Phase B=440; "
        "complete-I_perp early stop disabled only to expose uncensored dynamics; "
        "reference-free barrier10 Pareto screen; all screen work charged"
    ),
) -> tuple[Path, Path]:
    csv_path = output_dir / f"alanine_{version}_alpha_full_budget_data.csv"
    fields = (
        "family",
        "alpha",
        "point_type",
        "number_of_forces",
        "i_perp",
        "stop_reason",
        "target_reached",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in stage1 + stage4:
            alpha = "inf" if math.isinf(item.alpha) else item.alpha
            for raw_index, (force, iperp) in enumerate(
                zip(item.raw_force, item.raw_iperp)
            ):
                carried = (
                    item.cutoff_carried_forward
                    and raw_index == len(item.raw_force) - 1
                )
                writer.writerow(
                    {
                        "family": item.family,
                        "alpha": alpha,
                        "point_type": (
                            "budget_cutoff_carry_forward"
                            if carried
                            else "complete_raw_history"
                        ),
                        "number_of_forces": int(force),
                        "i_perp": float(iperp),
                        "stop_reason": item.raw_stop_reason,
                        "target_reached": item.target_reached,
                    }
                )
            for force, iperp in zip(item.frontier_force, item.frontier_iperp):
                writer.writerow(
                    {
                        "family": item.family,
                        "alpha": alpha,
                        "point_type": "strict_first_passage_or_joint_record",
                        "number_of_forces": float(force),
                        "i_perp": float(iperp),
                        "stop_reason": item.raw_stop_reason,
                        "target_reached": item.target_reached,
                    }
                )
            writer.writerow(
                {
                    "family": item.family,
                    "alpha": alpha,
                    "point_type": "terminal_charged_budget_at_best_accepted_precision",
                    "number_of_forces": item.terminal_charged_force,
                    "i_perp": item.terminal_best_iperp,
                    "stop_reason": item.raw_stop_reason,
                    "target_reached": item.target_reached,
                }
            )
            if item.target_force is not None and item.target_iperp is not None:
                writer.writerow(
                    {
                        "family": item.family,
                        "alpha": alpha,
                        "point_type": "first_joint_target_passage",
                        "number_of_forces": item.target_force,
                        "i_perp": item.target_iperp,
                        "stop_reason": item.raw_stop_reason,
                        "target_reached": True,
                    }
                )

    manifest_path = output_dir / "MANIFEST.json"
    records = {}
    for item in stage1 + stage4:
        key = f"{item.family}/alpha_{alpha_tag(item.alpha)}"
        records[key] = {
            "alpha": "inf" if math.isinf(item.alpha) else item.alpha,
            "raw_source": str(item.raw_source),
            "raw_source_sha256": sha256(item.raw_source),
            "archive_source": str(item.archive_source) if item.archive_source else None,
            "archive_source_sha256": sha256(item.archive_source) if item.archive_source else None,
            "raw_terminal_force": int(item.raw_force[-1]),
            "raw_terminal_i_perp": float(item.raw_iperp[-1]),
            "raw_best_i_perp": float(np.min(item.raw_iperp)),
            "raw_stop_reason": item.raw_stop_reason,
            "screen_force_equivalent_evaluations": item.screen_force,
            "terminal_charged_force": item.terminal_charged_force,
            "terminal_best_accepted_i_perp": item.terminal_best_iperp,
            "joint_target_reached": item.target_reached,
            "first_joint_target_force": item.target_force,
            "first_joint_target_i_perp": item.target_iperp,
            "protocol_budget": item.protocol_budget,
            "common_force_cutoff": item.common_force_cutoff,
            "last_actual_raw_force": item.last_actual_raw_force,
            "cutoff_carried_forward": item.cutoff_carried_forward,
            "common_initial_path_sha256": item.initial_sha256,
        }
    report = {
        "status": "complete_long_budget_alpha_audit",
        "purpose": (
            "Show complete dynamics and retain all terminal cost after the last "
            "precision improvement; corrects the v5 first-passage-only ambiguity."
        ),
        "common_initial_path_sha256": EXPECTED_INITIAL_SHA256,
        "version": version,
        "stage1_protocol": stage1_protocol,
        "stage4_protocol": stage4_protocol,
        "all_alpha_policies_identical_within_each_stage": True,
        "selected_alpha": 20,
        "joint_target_i_perp": TARGET_IPERP,
        "records": records,
        "csv": str(csv_path),
        "main_tex_modified": False,
    }
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return csv_path, manifest_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "figures/clean_four_level_audit/alpha_full_budget_v6"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    stage1 = load_stage1(root)
    stage4 = load_stage4(root)
    outputs = []
    outputs += make_full_history(stage1, stage4, output_dir)
    outputs += make_budget_frontier(stage1, stage4, output_dir)
    outputs += make_audit(stage1, stage4, output_dir)
    csv_path, manifest_path = write_data(stage1, stage4, output_dir)
    result = {
        "outputs": [str(path) for path in outputs],
        "data": str(csv_path),
        "manifest": str(manifest_path),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
