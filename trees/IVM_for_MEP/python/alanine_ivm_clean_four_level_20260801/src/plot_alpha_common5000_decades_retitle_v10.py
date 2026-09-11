#!/usr/bin/env python3
"""Retitle the decade-alpha cost-to-precision candidate with protocol names."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_alpha_common5000_decades_v9 import (
    ARCHIVE_PREFIX,
    COMMON_FORCE_CUTOFF,
    DECADE_LABELS,
    filter_decades,
)
from plot_alpha_common5000_v8 import (
    truncate_stage1,
    truncate_stage4,
    validate_common_cutoff,
)
from plot_alpha_full_budget_v6 import (
    TARGET_IPERP,
    alpha_legend,
    alpha_style,
    configure_style,
    grid,
    load_stage1_minimum_force,
    load_stage4_minimum_force,
    marker_for,
    panel_label,
    save,
)


VERSION = "v10_cut5000_decades_protocol_titles"


def scalar(saved, key):
    return np.asarray(saved[key]).item()


def validate_title_protocol(stage1, stage4) -> dict:
    stage1_r = set()
    stage4_phase_a_r = set()
    stage4_phase_b_r = set()
    stage4_reparam_every = set()
    for item in stage1:
        with np.load(item.raw_source, allow_pickle=False) as saved:
            stage1_r.add(float(scalar(saved, "plbfgs_kkt_rtol")))
    for item in stage4:
        with np.load(item.raw_source, allow_pickle=False) as combined:
            phase_a_path = Path(str(scalar(combined, "phase_a_result")))
            phase_b_path = Path(str(scalar(combined, "phase_b_result")))
        with np.load(phase_a_path, allow_pickle=False) as phase_a, np.load(
            phase_b_path, allow_pickle=False
        ) as phase_b:
            stage4_phase_a_r.add(float(scalar(phase_a, "plbfgs_kkt_rtol")))
            stage4_phase_b_r.add(float(scalar(phase_b, "plbfgs_kkt_rtol")))
            stage4_reparam_every.update(
                [
                    int(scalar(phase_a, "reparam_every")),
                    int(scalar(phase_b, "reparam_every")),
                ]
            )
    if stage1_r != {0.1}:
        raise RuntimeError(f"Unexpected Stage-I r values: {stage1_r}")
    if stage4_phase_a_r != {0.96} or stage4_phase_b_r != {0.8}:
        raise RuntimeError(
            f"Unexpected Stage-IV r values: {stage4_phase_a_r}, {stage4_phase_b_r}"
        )
    if stage4_reparam_every != {1}:
        raise RuntimeError(
            f"Unexpected Stage-IV reparameterization schedule: {stage4_reparam_every}"
        )
    return {
        "stage1_r_values": sorted(stage1_r),
        "stage4_phase_a_r_values": sorted(stage4_phase_a_r),
        "stage4_phase_b_r_values": sorted(stage4_phase_b_r),
        "stage4_reparam_every_values": sorted(stage4_reparam_every),
        "validated_alpha_count_per_stage": len(stage1),
    }


def fully_charged_raw_trajectory(item) -> tuple[np.ndarray, np.ndarray]:
    """Return uncensored raw states through the exact charged NF=5000 cutoff."""
    last_actual = item.last_actual_raw_force or COMMON_FORCE_CUTOFF
    actual = np.asarray(item.raw_force, dtype=int) <= int(last_actual)
    raw_force = np.asarray(item.raw_force, dtype=int)[actual]
    raw_iperp = np.asarray(item.raw_iperp, dtype=float)[actual]
    charged_force = raw_force.copy()

    if item.archive_source is not None:
        with np.load(item.archive_source, allow_pickle=False) as archive:
            candidate_calls = np.asarray(archive["candidate_calls"], dtype=int)
            screen_cost = np.asarray(
                archive["candidate_screen_force_equivalent_cost"], dtype=int
            )
            reference_used = bool(scalar(archive, "reference_barrier_used"))
        if reference_used:
            raise RuntimeError(f"Reference barrier leaked into {item.archive_source}")
        cumulative_screen = np.cumsum(screen_cost)
        completed = np.searchsorted(candidate_calls, raw_force, side="right")
        charged_force = raw_force + np.asarray(
            [cumulative_screen[index - 1] if index else 0 for index in completed],
            dtype=int,
        )

    keep = (
        np.isfinite(raw_iperp)
        & (raw_iperp > 0)
        & (charged_force > 0)
        & (charged_force <= COMMON_FORCE_CUTOFF)
    )
    charged_force = charged_force[keep]
    raw_iperp = raw_iperp[keep]
    if charged_force.size == 0:
        raise RuntimeError(f"No complete state within cutoff for {item.raw_source}")

    # At an exact external budget cutoff the current complete state remains
    # the reported path while an unfinished next operation consumes the tail.
    # This is a solid raw trajectory segment, not a best-so-far extrapolation.
    if int(charged_force[-1]) < COMMON_FORCE_CUTOFF:
        charged_force = np.append(charged_force, COMMON_FORCE_CUTOFF)
        raw_iperp = np.append(raw_iperp, raw_iperp[-1])
    return charged_force, raw_iperp


def plot_raw_cost_trajectory(axis, records, show_ylabel: bool) -> dict:
    audit = {}
    for item in records:
        force, iperp = fully_charged_raw_trajectory(item)
        style = alpha_style(item.alpha)
        marker = marker_for(item.alpha)
        axis.plot(
            iperp,
            force,
            marker=marker,
            markersize=3.0,
            markevery=max(1, len(force) // 11),
            markeredgewidth=0.25,
            **style,
        )
        axis.scatter(
            iperp[-1],
            force[-1],
            marker="X",
            s=40,
            facecolor=style["color"],
            edgecolor="#202020",
            linewidth=0.55,
            zorder=style["zorder"] + 3,
        )
        audit[f"{item.alpha:g}"] = {
            "complete_states_and_cutoff_endpoint": int(len(force)),
            "terminal_force": int(force[-1]),
            "terminal_raw_i_perp": float(iperp[-1]),
            "raw_i_perp_increases": int(np.count_nonzero(np.diff(iperp) > 0)),
        }
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1.2e3, 4.0e-6)
    axis.set_ylim(16, 6500)
    axis.set_xlabel(r"Raw $I_\perp$ (smaller is better)")
    axis.set_ylabel("Number of Forces" if show_ylabel else "")
    grid(axis)
    return audit


def make_raw_common5000(stage1, stage4, output_dir: Path) -> tuple[list[Path], dict]:
    figure, axes = plt.subplots(1, 2, figsize=(9.3, 3.55), sharey=True)
    trajectory_audit = {
        "stage1": plot_raw_cost_trajectory(axes[0], stage1, True),
        "stage4": plot_raw_cost_trajectory(axes[1], stage4, False),
    }
    axes[0].set_title(r"Exact-like subminimization ($r=0.1$), no resampling")
    axes[1].set_title(
        r"Optimized inexact subminimization + resampling "
        r"($r=0.96\,\rightarrow\,0.8$)"
    )
    axes[1].axvline(TARGET_IPERP, color="#6a6a6a", ls=":", lw=0.9, zorder=1)
    axes[1].text(
        0.68,
        0.055,
        r"joint target $1.9\times10^{-4}$",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=6.9,
        color="#505050",
    )
    axes[0].legend(
        handles=[
            Line2D(
                [0], [0], ls="", marker="X", ms=5.2,
                markerfacecolor="#b0b0b0", markeredgecolor="#202020",
                markeredgewidth=0.5,
                label=r"common charged $N_F=5000$ cutoff",
            )
        ],
        frameon=False,
        loc="upper left",
        fontsize=6.7,
    )
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    alpha_legend(figure, 0.995, stage1, title=None)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.76, wspace=0.12)
    outputs = save(
        figure,
        output_dir
        / f"alanine_{VERSION}_alpha_budget_aware_cost_to_precision_stage1_vs_stage4",
    )
    return outputs, trajectory_audit


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "figures/clean_four_level_audit/alpha_common5000_decades_v10_titles"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    stage1_all = truncate_stage1(
        load_stage1_minimum_force(
            root,
            minimum_terminal_force=COMMON_FORCE_CUTOFF,
            minimum_macro_budget=1000,
        )
    )
    stage4_all = truncate_stage4(
        load_stage4_minimum_force(
            root,
            minimum_terminal_force=COMMON_FORCE_CUTOFF,
            minimum_phase_b=600,
            archive_prefix=ARCHIVE_PREFIX,
        )
    )
    validate_common_cutoff(stage1_all, stage4_all)
    stage1 = filter_decades(stage1_all)
    stage4 = filter_decades(stage4_all)
    protocol_validation = validate_title_protocol(stage1, stage4)

    outputs, trajectory_audit = make_raw_common5000(
        stage1,
        stage4,
        output_dir,
    )
    report = {
        "status": "complete",
        "alpha_grid": list(DECADE_LABELS),
        "common_force_cutoff": COMMON_FORCE_CUTOFF,
        "stage1_title_basis": (
            "Stage-I exact-like no-resampling run metadata: plbfgs_kkt_rtol=0.1"
        ),
        "stage4_title_basis": (
            "Stage-IV Phase A metadata: plbfgs_kkt_rtol=0.96; "
            "Phase B metadata: plbfgs_kkt_rtol=0.8; resampling every macrostep"
        ),
        "overall_legend_title_removed": True,
        "display_semantics": (
            "uncensored raw I_perp trajectories through exact NF=5000; "
            "Stage-IV reference-free screen work is charged"
        ),
        "best_so_far_frontier_used": False,
        "spent_after_last_improvement_dashed_tail_used": False,
        "trajectory_audit": trajectory_audit,
        "protocol_validation": protocol_validation,
        "outputs": [str(path) for path in outputs],
        "new_dynamics_run": False,
        "main_tex_modified": False,
    }
    report_path = output_dir / "TITLE_PROTOCOL_AUDIT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
