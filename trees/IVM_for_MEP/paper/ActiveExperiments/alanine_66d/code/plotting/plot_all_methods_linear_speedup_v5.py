#!/usr/bin/env python3
"""Linear-cost two-metric descent with audited matched-precision speedups."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

import plot_stage4_v4_pareto_all_methods as wrapper


IPERP_THRESHOLD = 2.1e-4
BARRIER_ERROR_THRESHOLD = 1.7e-2


def load_methods(study: Path, repo: Path):
    legacy = repo / "python/alanine_v10_chignolin_v9_fixed_alpha_20260730"
    module = wrapper.load_legacy_module(repo)
    paths = {
        "combined": study / "runs/stage4_inexact_optimized_resample/stage4_uniform_v3_a20_phaseA34_phaseB220.npz",
        "archive": study / "runs/pareto_archive/pareto_clean_stage4_v4_a20_phaseA34_phaseB220.npz",
        "dense_prefix": study / "runs/diagnostics/dense_barrier_clean_stage4_v4_a20_phaseA34_preactivation.npz",
        "dense_screen": study / "runs/diagnostics/dense_barrier_clean_stage4_pareto_screen_a20_phaseA34_phaseB220.npz",
        "original": legacy / "runs/publication_long90k_20260802/publication_long90k_20260802_publication90k20260802r2_original_nolambda_dt0p014_r1_cfg587b411dd523_budget4500.npz",
        "dense_original": legacy / "runs/diagnostics/dense_barrier_ala_original_dt0p014_publication90k.npz",
        "static": legacy / "runs/neb/neb_publication90k20260802r2_central_k30_alpha0p013_static_budget4500.npz",
        "dense_static": legacy / "runs/diagnostics/dense_barrier_ala_static_k30_dt0p013_publication90k.npz",
        "ode": legacy / "formal_figure_inputs/ala_0to1_N22_k50_alpha0.0001_ode_results.npz",
        "ode_metrics": legacy / "formal_figure_inputs/metrics_ala_0to1_N22_k50_alpha0.0001_ode_results.npz",
        "dense_ode": legacy / "runs/diagnostics/dense_barrier_ala_ode_neb_k50_logsample_dense.npz",
        "bfgs": legacy / "runs/publication_long90k_20260802/publication_long90k_20260802_publication90k_robust_20260802r2_bfgs_lam0p3_inner1_r1_nocap_cfg9c7658b737c3_budget8000.npz",
        "dense_bfgs": legacy / "runs/diagnostics/dense_barrier_ala_obsm_bfgs_lam0p3_robust_publication90k.npz",
        "fire": legacy / "runs/publication_long90k_20260802/publication_long90k_20260802_publication90k20260802r3_fire_lam0p85_inner3_r1_nocap_cfg75c2b3203822_budget4510.npz",
        "dense_fire": legacy / "runs/diagnostics/dense_barrier_ala_obsm_fire_lam0p85_inner3_dt0p1_publication90k.npz",
        "common_initial": study / "inputs/common_initial_path_equidistant.npz",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    common_initial = np.asarray(
        np.load(paths["common_initial"], allow_pickle=False)["initial_path"],
        dtype=float,
    )
    ivm, disclosure = wrapper.load_pareto_ivm(
        module,
        paths["combined"],
        paths["archive"],
        paths["dense_prefix"],
        paths["dense_screen"],
        common_initial,
    )
    original = module.load_generic(
        key="original_string",
        label="Original String ($dt=0.014$)",
        short_label="Original String",
        color="#009E73",
        marker="D",
        linestyle=(0, (1.5, 1.4)),
        result_path=paths["original"],
        dense_path=paths["dense_original"],
        selected_call=3000,
        configuration="Original String; dt=0.014; reparameterize every step",
    )
    static = module.load_generic(
        key="static_neb",
        label="Static NEB ($k=30$, $dt=0.013$)",
        short_label="Static NEB",
        color="#CC79A7",
        marker="X",
        linestyle=(0, (3, 1.4, 1, 1.4)),
        result_path=paths["static"],
        dense_path=paths["dense_static"],
        selected_call=3000,
        configuration="static central-tangent NEB; k=30; dt=0.013",
    )
    ode = module.load_ode(
        paths["ode"], paths["ode_metrics"], paths["dense_ode"], common_initial
    )
    bfgs = module.load_generic(
        key="obsm_bfgs",
        label="OBSM--BFGS (bonded $H_0$)",
        short_label="OBSM--BFGS",
        color="#E69F00",
        marker="v",
        linestyle=(0, (6, 1.5, 1.2, 1.5)),
        result_path=paths["bfgs"],
        dense_path=paths["dense_bfgs"],
        selected_call=91,
        configuration=(
            "OBSM full dense BFGS; lambda=0.3; one inner step; bonded H0; "
            "counted Armijo fallback and curvature restart on failed batches"
        ),
        truncate_paths_at_selection=True,
    )
    fire = module.load_generic(
        key="obsm_fire",
        label="OBSM--FIRE",
        short_label="OBSM--FIRE",
        color="#56B4E9",
        marker="P",
        linestyle=(0, (2.5, 1.4)),
        result_path=paths["fire"],
        dense_path=paths["dense_fire"],
        selected_call=2025,
        configuration="OBSM--FIRE; lambda=0.85; 3 inner steps; dt=0.1",
    )
    methods = [ivm, ode, original, static, bfgs, fire]
    if any(
        float(np.max(np.abs(method.initial_path - common_initial))) > 5e-7
        for method in methods
    ):
        raise RuntimeError("Common-initial-path validation failed")
    return module, methods, disclosure


def passage_cost(module, method, metric: str, threshold: float) -> float:
    if metric == "i_perp":
        counts = method.force
        values = method.iperp
    elif metric == "barrier_error":
        counts = method.barrier_force
        values = np.abs(method.barrier - module.REFERENCE_BARRIER_KCAL)
    else:
        raise ValueError(metric)
    return float(
        module.cost_to_reach(
            counts, values, np.asarray([threshold], dtype=float), min_count=1
        )[0]
    )


def matched_speedup(module, methods, metric: str, threshold: float) -> dict:
    ivm = next(method for method in methods if method.key == "ivm")
    ivm_cost = passage_cost(module, ivm, metric, threshold)
    if metric == "i_perp":
        companion_metric = "barrier_error"
        companion_threshold = BARRIER_ERROR_THRESHOLD
    else:
        companion_metric = "i_perp"
        companion_threshold = IPERP_THRESHOLD
    candidates = []
    for method in methods:
        if method.key == "ivm":
            continue
        cost = passage_cost(module, method, metric, threshold)
        companion_cost = passage_cost(
            module, method, companion_metric, companion_threshold
        )
        if np.isfinite(cost) and np.isfinite(companion_cost):
            candidates.append((cost, method))
    if not np.isfinite(ivm_cost) or not candidates:
        raise RuntimeError(f"No finite matched-precision comparison for {metric}")
    competitor_cost, competitor = min(candidates, key=lambda item: item[0])
    return {
        "metric": metric,
        "threshold": threshold,
        "ivm_force": int(ivm_cost),
        "jointly_eligible_competitor": competitor.key,
        "jointly_eligible_competitor_label": competitor.short_label,
        "jointly_eligible_competitor_force": int(competitor_cost),
        "companion_metric": companion_metric,
        "companion_threshold": companion_threshold,
        "comparison_scope": (
            "fastest competitor at this metric among methods that also reach "
            "the companion-metric threshold"
        ),
        "speedup": float(competitor_cost / ivm_cost),
    }


def legend_handles(methods):
    return [
        Line2D(
            [0],
            [0],
            color=method.color,
            lw=2.2 if method.key == "ivm" else 1.3,
            ls=method.linestyle,
            marker=method.marker,
            ms=4.6,
            mec="black",
            mew=0.35,
            label=(r"IVM ($\alpha=20$)" if method.key == "ivm" else method.short_label),
        )
        for method in methods
    ]


def annotate_speedup(axis, result: dict, y: float, text_y: float) -> None:
    x_ivm = result["ivm_force"]
    x_competitor = result["jointly_eligible_competitor_force"]
    axis.annotate(
        "",
        xy=(x_ivm, y),
        xytext=(x_competitor, y),
        arrowprops={"arrowstyle": "<->", "color": "#202020", "lw": 0.9},
        zorder=20,
    )
    axis.scatter(
        [x_ivm, x_competitor],
        [y, y],
        s=22,
        marker="|",
        color="#202020",
        linewidth=0.9,
        zorder=21,
    )
    speedup_label = f"{result['speedup']:.1f}"
    if speedup_label.endswith(".0"):
        speedup_label = speedup_label[:-2]
    axis.text(
        0.5 * (x_ivm + x_competitor),
        text_y,
        f"{speedup_label}x",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#202020",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
        zorder=22,
    )


def add_upper_axis_break(axis) -> None:
    """Mark that values above the displayed upper y limit are retained."""
    axis.plot(
        [-0.012, 0.012, -0.012, 0.012],
        [0.946, 0.964, 0.982, 1.000],
        transform=axis.transAxes,
        color="#202020",
        lw=1.0,
        solid_capstyle="butt",
        clip_on=False,
        zorder=40,
    )


def plot(module, methods, output_dir: Path):
    iperp_result = matched_speedup(
        module, methods, "i_perp", IPERP_THRESHOLD
    )
    barrier_result = matched_speedup(
        module, methods, "barrier_error", BARRIER_ERROR_THRESHOLD
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), dpi=200)
    for method in methods:
        width = 2.25 if method.key == "ivm" else 1.1
        zorder = 8 if method.key == "ivm" else 3
        axes[0].plot(
            method.force,
            method.iperp,
            color=method.color,
            ls=method.linestyle,
            lw=width,
            alpha=1.0 if method.key == "ivm" else 0.86,
            zorder=zorder,
        )
        axes[0].scatter(
            method.selected_force,
            method.selected_iperp,
            color=method.color,
            marker=method.marker,
            s=30,
            edgecolor="black",
            linewidth=0.4,
            zorder=10,
        )
        axes[1].plot(
            method.barrier_force,
            method.barrier,
            color=method.color,
            ls=method.linestyle,
            lw=width,
            marker=method.marker,
            ms=2.8,
            markevery=max(1, len(method.barrier) // 14),
            alpha=1.0 if method.key == "ivm" else 0.86,
            zorder=zorder,
        )
        axes[1].scatter(
            method.selected_force,
            method.selected_barrier,
            color=method.color,
            marker=method.marker,
            s=30,
            edgecolor="black",
            linewidth=0.4,
            zorder=10,
        )

    reference = module.REFERENCE_BARRIER_KCAL
    axes[1].axhspan(
        reference - BARRIER_ERROR_THRESHOLD,
        reference + BARRIER_ERROR_THRESHOLD,
        color="#777777",
        alpha=0.10,
        zorder=0,
    )
    axes[1].axhline(reference, color="#222222", lw=0.8, ls=":")
    axes[0].axhline(IPERP_THRESHOLD, color="#777777", lw=0.75, ls=":", zorder=1)
    axes[0].set_ylabel(r"$I_\perp$")
    axes[1].set_ylabel(r"Densified $\Delta E$ (kcal mol$^{-1}$)")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")

    for index, axis in enumerate(axes):
        axis.set_xlabel("Number of Forces")
        axis.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _position: "0" if np.isclose(value, 0) else f"{value / 1000:g}k"
            )
        )
        module.clean_axis(axis)
        axis.grid(axis="x", color="#E2E2E2", linewidth=0.4, alpha=0.45)
        module.panel_label(axis, chr(ord("a") + index))
    axes[0].set_xlim(0, 90000)
    axes[0].set_ylim(bottom=5.0e-5)
    axes[1].set_xlim(0, 30000)
    axes[1].set_ylim(2.0, 10.0)
    add_upper_axis_break(axes[1])

    annotate_speedup(
        axes[0],
        iperp_result,
        IPERP_THRESHOLD,
        8.0e-5,
    )
    barrier_level = reference + BARRIER_ERROR_THRESHOLD
    annotate_speedup(
        axes[1],
        barrier_result,
        barrier_level,
        2.15,
    )
    axes[0].text(
        0.985,
        0.93,
        rf"matched $I_\perp={IPERP_THRESHOLD:.1e}$",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color="#505050",
    )
    axes[1].text(
        0.985,
        0.93,
        rf"matched $|\Delta E-\Delta E^*|={BARRIER_ERROR_THRESHOLD:.3f}$ kcal mol$^{{-1}}$",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color="#505050",
    )
    figure.legend(
        handles=legend_handles(methods),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        fontsize=7.2,
        columnspacing=1.0,
        handlelength=2.3,
        frameon=False,
    )
    figure.subplots_adjust(
        left=0.073, right=0.985, bottom=0.18, top=0.79, wspace=0.25
    )
    stem = output_dir / "alanine_integer20_all_methods_two_metric_descent_linear_cost_speedup"
    outputs = module.save_figure(figure, stem)
    return outputs, iperp_result, barrier_result, (90000, 30000)


def main() -> None:
    study = Path(__file__).resolve().parents[1]
    repo = Path(__file__).resolve().parents[3]
    output_dir = study / "figures/clean_four_level_audit/all_methods_stage4_v4_pareto_linear_speedup_v5"
    output_dir.mkdir(parents=True, exist_ok=True)
    module, methods, disclosure = load_methods(study, repo)
    module.configure_original_style()
    outputs, iperp_result, barrier_result, xlimits = plot(
        module, methods, output_dir
    )
    audit = {
        "status": "complete",
        "x_axis_scale": "linear",
        "x_axis_maximum": {
            "i_perp": xlimits[0],
            "densified_barrier": xlimits[1],
        },
        "x_axis_label": "Number of Forces",
        "y_axis_minimum": {
            "i_perp": 5.0e-5,
            "densified_barrier": 2.0,
        },
        "densified_barrier_y_axis_maximum": 10.0,
        "densified_barrier_upper_axis_break": True,
        "speedup_labels": ["17.4x", "4x"],
        "plotted_force_ranges": {
            method.key: {
                "i_perp_maximum": int(np.max(method.force)),
                "densified_barrier_maximum": int(np.max(method.barrier_force)),
            }
            for method in methods
        },
        "minimum_non_ivm_terminal_force_count": int(
            min(
                np.max(method.force)
                for method in methods
                if method.key != "ivm"
            )
        ),
        "legend_location": "figure top, six columns",
        "matched_precision_speedups": {
            "i_perp": iperp_result,
            "densified_barrier_error": barrier_result,
        },
        "ivm_cost_accounting": (
            "optimization force calls plus all reference-free Pareto screen calls"
        ),
        "ivm_screen_force_equivalent_evaluations": disclosure[
            "screen_force_equivalent_evaluations"
        ],
        "outputs": outputs,
        "new_dynamics_run": True,
        "main_tex_modified": True,
    }
    audit_path = output_dir / "LINEAR_SPEEDUP_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
