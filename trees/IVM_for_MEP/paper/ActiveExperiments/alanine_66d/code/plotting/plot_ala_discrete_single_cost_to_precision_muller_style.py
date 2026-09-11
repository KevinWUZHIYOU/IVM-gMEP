#!/usr/bin/env python3
"""Plot the no-Phase-A/B discrete Ala IVM against frozen long baselines.

The IVM curve is the formal one-way-takeover single run with fixed alpha=15,
post-selection eta=1.1, and the sole inner stopping rule R/R0 <= 0.8 OR ten
evaluated steps.  Only complete, mesh-qualified states enter either panel.
Panel (b) uses a dedicated post-hoc densification of those same states; the
result file's sparse ``barrier_kcal`` diagnostic is intentionally not used.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


REPO = Path(__file__).resolve().parents[5]
STUDY = REPO / "python/alanine_ivm_clean_four_level_20260801"
LEGACY = REPO / "python/alanine_v10_chignolin_v9_fixed_alpha_20260730"
SOURCE_DIR = STUDY / "src"
BASE_PLOTTER = Path(__file__).with_name("plot_ala_cost_to_precision_muller_style.py")
OUTPUT_DIR = REPO / "paper/ActiveExperiments/alanine_66d/figures/active_candidates"
STEM = "alanine66d_discrete_ivm_cost_to_precision_1x2"

NEW_RESULT = (
    STUDY
    / "runs/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1/members"
    / "ivm_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_"
    "one_way_takeover_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1_"
    "central_state_ivm_cfg1804ab58ab75_budget254.npz"
)
FORCED_IDENTITY_RESULT = (
    STUDY
    / "runs/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1/members"
    / "ivm_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_"
    "forced_operator_lbfgs_selection_eta1p1_r08c10_kkt0_a15_N254_"
    "discrete_b040_r1_central_state_ivm_cfgdbf51ea22ca4_budget254.npz"
)
NEW_DENSE = (
    STUDY
    / "runs/diagnostics/dense_barrier_stage4_discrete_single_one_way_"
    "eta1p1_r08c10_kkt0_a15_complete_mesh.npz"
)
NEW_DENSE_AUDIT = Path(str(NEW_DENSE).replace(".npz", ".audit.json"))
REPORT_PREFIX = (
    "stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_"
    "N254_r1_1787262084320785000"
)
CAMPAIGN_REPORT = STUDY / "reports" / f"{REPORT_PREFIX}.json"
PREREGISTRATION = STUDY / "reports" / f"{REPORT_PREFIX}_preregistered.json"
FORMAL_IPERP_CTP = (
    STUDY
    / "reports"
    / f"{REPORT_PREFIX}_iperp_ctp_readonly_v1_1787262442236424000.json"
)
POSTRUN_AUDIT = (
    STUDY
    / "reports"
    / f"{REPORT_PREFIX}_postrun_mechanism_provenance_readonly_v1.json"
)
COMMON_INITIAL = STUDY / "inputs/common_initial_path_equidistant.npz"
TOPOLOGY = STUDY / "inputs/ala2_fixed.pdb"
POSTPROCESSOR = SOURCE_DIR / "postprocess_dense_barrier.py"

EXPECTED_HASHES = {
    NEW_RESULT: "1c81ad6fe876a0eb94f2fc55d5d2567737face95ff52b6fe1bdfffc536d16787",
    FORCED_IDENTITY_RESULT: "5d8169d1d16dab2836bef43affcaaec2dabf539a8e0f992c9050cd909faf4050",
    NEW_DENSE: "da2eb76b4c40347212914e493abe0e440617a36052f9e64609131af39c1e342a",
    NEW_DENSE_AUDIT: "b3c9b9b12900410cd1e5f3809ee47f0f75f8e35ecdb35494e13e7a286da03c46",
    CAMPAIGN_REPORT: "c64c471d612a22161ff1308bd969f83588b5de4a2aa67e62b4e4b848e96514ad",
    PREREGISTRATION: "67dd26b93f216a2d7f429a1b31710bdf1b933c24e10f17779e2499621705fa4e",
    FORMAL_IPERP_CTP: "e9f5008381617fa98b04fabeb05cbc8b347e809dce9cdbd0901b7a0fc5a3e80d",
    POSTRUN_AUDIT: "28b3c12692ba6ac21576dd193677916b77355370a41a009ba9e33b0a615ed538",
}

EXPECTED_ELIGIBLE_INDICES = np.asarray(
    [0, 3, 6, 8, 49, 55, 61, 85, 95, 101, 107, 113, 133, 139, 165, 189, 213, 253],
    dtype=int,
)
EXPECTED_ELIGIBLE_CALLS = np.asarray(
    [1, 4, 7, 9, 50, 56, 62, 86, 96, 102, 108, 114, 134, 140, 166, 190, 214, 254],
    dtype=int,
)
EXPECTED_ELIGIBLE_NF = np.asarray(
    [20, 80, 140, 180, 911, 1035, 1157, 1460, 1569, 1688, 1820, 1941,
     2150, 2282, 2519, 2781, 3045, 3389],
    dtype=int,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this new candidate's PDF/CSV/manifest if they already exist.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def scalar(saved, key: str):
    return np.asarray(saved[key]).reshape(()).item()


def load_base_plotter():
    spec = importlib.util.spec_from_file_location("ala_cost_base", BASE_PLOTTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base plotter: {BASE_PLOTTER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_registered_inputs() -> None:
    for path, expected in EXPECTED_HASHES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"Registered input drift for {path}: expected {expected}, got {observed}"
            )


def load_new_ivm(old_ivm):
    with np.load(NEW_RESULT, allow_pickle=False) as saved:
        n_records = len(np.asarray(saved["stored_path_calls"]))
        if n_records != 254:
            raise RuntimeError(f"Expected 254 single-run records, got {n_records}")
        controls = {
            "alpha": float(scalar(saved, "alpha")),
            "eta": float(scalar(saved, "discrete_step_multiplier_eta")),
            "residual_ratio": float(scalar(saved, "discrete_residual_ratio")),
            "inner_max_steps": int(scalar(saved, "discrete_inner_max_steps")),
            "history": int(scalar(saved, "discrete_history")),
            "kkt_atol": float(scalar(saved, "plbfgs_kkt_atol")),
            "selector_mode": str(scalar(saved, "discrete_ffrbk_selector_mode")),
            "stop_mode": str(scalar(saved, "discrete_stop_mode")),
            "reparam_mode": str(scalar(saved, "reparam_mode")),
            "reparam_objective_acceptance": str(
                scalar(saved, "state_curve_direct_objective_acceptance")
            ),
        }
        expected_controls = {
            "alpha": 15.0,
            "eta": 1.1,
            "residual_ratio": 0.8,
            "inner_max_steps": 10,
            "history": 7,
            "kkt_atol": 0.0,
            "selector_mode": "one_way_takeover",
            "stop_mode": "r08c10",
            "reparam_mode": "state_curve",
            "reparam_objective_acceptance": "per_image",
        }
        if controls != expected_controls:
            raise RuntimeError(f"Unexpected new-run controls: {controls}")
        active_alpha = np.asarray(saved["active_alpha"], dtype=float)
        if not np.array_equal(active_alpha, np.full(n_records, 15.0)):
            raise RuntimeError("The new single run did not keep alpha fixed at 15")

        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        complete = (np.arange(n_records) == 0) | committed
        eligible = complete & (cv <= 0.0065) & (ratio <= 1.03)
        indices = np.flatnonzero(eligible)
        if not np.array_equal(indices, EXPECTED_ELIGIBLE_INDICES):
            raise RuntimeError(f"Complete+mesh index drift: {indices.tolist()}")

        calls_all = np.asarray(saved["stored_path_calls"], dtype=int)
        nf_all = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        iperp_all = np.asarray(saved["i_perp"], dtype=float)
        paths_all = np.asarray(saved["stored_paths"], dtype=float)
        calls = calls_all[eligible]
        nf = nf_all[eligible]
        iperp = iperp_all[eligible]
        paths = paths_all[eligible]
        if not np.array_equal(calls, EXPECTED_ELIGIBLE_CALLS):
            raise RuntimeError("Eligible stored-path calls changed")
        if not np.array_equal(nf, EXPECTED_ELIGIBLE_NF):
            raise RuntimeError("Eligible whole-path NF values changed")
        if not np.all(np.diff(iperp) < 0):
            raise RuntimeError("Formal eligible I_perp values are not strict records")
        initial_path = np.asarray(saved["initial_path"], dtype=float)

    with np.load(NEW_DENSE, allow_pickle=False) as dense:
        dense_calls = np.asarray(dense["calls"], dtype=int)
        barrier = np.asarray(dense["adaptive_barrier_kcal"], dtype=float)
        subdivisions = np.asarray(dense["selected_subdivisions"], dtype=int)
        dense_difference = np.asarray(dense["final_level_difference_kcal"], dtype=float)
        dense_queries = int(scalar(dense, "diagnostic_backend_raw_joint_evaluations"))
        if not np.array_equal(dense_calls, calls):
            raise RuntimeError("Dense diagnostics do not align with eligible single-run calls")
        if not np.array_equal(subdivisions, np.full(len(calls), 20)):
            raise RuntimeError("Expected all dense records to select 20 subdivisions")
        if dense_queries != 7544:
            raise RuntimeError(f"Unexpected dense diagnostic query count: {dense_queries}")
        if str(scalar(dense, "source_result")) != str(NEW_RESULT):
            raise RuntimeError("Dense diagnostic points to a different source result")
        if str(scalar(dense, "source_stored_paths_sha256")) != array_sha256(paths):
            raise RuntimeError("Dense selected-path hash does not match the plotted paths")

    with np.load(FORCED_IDENTITY_RESULT, allow_pickle=False) as forced:
        for key, expected in (
            ("stored_path_calls", calls_all),
            ("total_internal_joint_evaluations", nf_all),
            ("i_perp", iperp_all),
            ("stored_paths", paths_all),
            ("reparam_committed", committed),
            ("segment_length_cv", cv),
            ("segment_length_ratio", ratio),
        ):
            if not np.array_equal(np.asarray(forced[key]), expected, equal_nan=True):
                raise RuntimeError(f"Forced identity arm differs in safe field {key}")

    method = replace(
        old_ivm,
        key="ivm",
        label="IVM",
        short_label="IVM",
        color="#0072B2",
        marker="o",
        linestyle="-",
        force=nf.astype(float),
        iperp=iperp,
        barrier_force=nf.astype(float),
        barrier=barrier,
        selected_call=int(calls[-1]),
        selected_force=int(nf[-1]),
        selected_iperp=float(iperp[-1]),
        selected_barrier=float(barrier[-1]),
        selected_path=paths[-1],
        initial_path=initial_path,
        path_force=nf.astype(float),
        paths=paths,
        configuration=(
            "single continuous run; no Phase A/B; discrete fixed-path FFrbk/operator-"
            "LBFGS one-way takeover; alpha=15 fixed; eta=1.1; R/R0<=0.8 OR "
            "10 evaluated steps; m=7; kkt_atol=0; state_curve/per_image reparameterization"
        ),
        result_path=NEW_RESULT,
        dense_path=NEW_DENSE,
    )
    info = {
        "controls": controls,
        "eligible_indices": indices,
        "eligible_calls": calls,
        "eligible_nf": nf,
        "eligible_iperp": iperp,
        "eligible_cv": cv[eligible],
        "eligible_ratio": ratio[eligible],
        "eligible_committed": committed[eligible],
        "dense_barrier": barrier,
        "dense_subdivisions": subdivisions,
        "dense_final_level_difference": dense_difference,
        "dense_queries": dense_queries,
    }
    return method, info


def load_methods(base):
    """Load the five frozen baselines without opening any old Phase-A/B IVM."""
    module = base.current.wrapper.load_legacy_module(REPO)
    paths = {
        "original": LEGACY / "runs/publication_long90k_20260802/publication_long90k_20260802_publication90k20260802r2_original_nolambda_dt0p014_r1_cfg587b411dd523_budget4500.npz",
        "dense_original": LEGACY / "runs/diagnostics/dense_barrier_ala_original_dt0p014_publication90k.npz",
        "static": LEGACY / "runs/neb/neb_publication90k20260802r2_central_k30_alpha0p013_static_budget4500.npz",
        "dense_static": LEGACY / "runs/diagnostics/dense_barrier_ala_static_k30_dt0p013_publication90k.npz",
        "ode": LEGACY / "formal_figure_inputs/ala_0to1_N22_k50_alpha0.0001_ode_results.npz",
        "ode_metrics": base.ODE_METRICS_PATH,
        "dense_ode": LEGACY / "runs/diagnostics/dense_barrier_ala_ode_neb_k50_logsample_dense.npz",
        "bfgs": LEGACY / "runs/publication_long90k_20260802/publication_long90k_20260802_publication90k_robust_20260802r2_bfgs_lam0p3_inner1_r1_nocap_cfg9c7658b737c3_budget8000.npz",
        "dense_bfgs": LEGACY / "runs/diagnostics/dense_barrier_ala_obsm_bfgs_lam0p3_robust_publication90k.npz",
        "fire": LEGACY / "runs/publication_long90k_20260802/publication_long90k_20260802_publication90k20260802r3_fire_lam0p85_inner3_r1_nocap_cfg75c2b3203822_budget4510.npz",
        "dense_fire": LEGACY / "runs/diagnostics/dense_barrier_ala_obsm_fire_lam0p85_inner3_dt0p1_publication90k.npz",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    with np.load(COMMON_INITIAL, allow_pickle=False) as saved:
        common_initial = np.asarray(saved["initial_path"], dtype=float)

    ode = module.load_ode(
        paths["ode"], paths["ode_metrics"], paths["dense_ode"], common_initial
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

    # MethodSeries is shared by the legacy baseline loader.  Use one baseline
    # only as a dataclass template; no old IVM artifact is opened here.
    new_ivm, new_info = load_new_ivm(ode)
    by_key = {
        method.key: method for method in (new_ivm, ode, original, static, bfgs, fire)
    }
    methods = [by_key[key] for key in base.METHOD_ORDER]
    for method in methods:
        if not np.array_equal(np.asarray(method.initial_path), common_initial):
            raise RuntimeError(f"Common-initial validation failed for {method.key}")
    with np.load(paths["ode"], allow_pickle=False) as saved:
        ode_stored_first = np.asarray(saved["images_seq"][0], dtype=float)
    maximum_difference = float(np.max(np.abs(common_initial - ode_stored_first)))
    disclosure = {
        "common_initial_array_sha256": array_sha256(common_initial),
        "ode_stored_first_path_array_sha256": array_sha256(ode_stored_first),
        "maximum_absolute_coordinate_difference_angstrom": maximum_difference,
        "exact_same_initial_path_in_frozen_ode_artifact": bool(
            maximum_difference <= 5.0e-7
        ),
        "note": (
            "The historical ODE artifact stores a different first path; do not "
            "describe the frozen six-method inputs as exact same-IDPP artifacts."
        ),
    }
    base.METHOD_STYLE["ivm"]["label"] = "IVM"
    return module, methods, disclosure, new_info


def annotate_compact_speedup(
    axis,
    threshold: float,
    low_cost: float,
    high_cost: float,
    label: str,
    label_x: float,
) -> None:
    """Place a compact multiplier just right of a vertical arrow on reversed x."""
    axis.axvline(
        threshold, color="#767676", linewidth=0.55, linestyle=":", zorder=1
    )
    axis.annotate(
        "",
        xy=(threshold, low_cost),
        xytext=(threshold, high_cost),
        arrowprops={"arrowstyle": "<->", "color": "#202020", "lw": 0.8},
        zorder=25,
    )
    axis.text(
        label_x,
        0.5 * (low_cost + high_cost),
        label,
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color="#202020",
        zorder=26,
    )


def draw_figure(base, module, methods, output: Path):
    curves = {
        method.key: {
            "i_perp": base.pareto_record_points(module, method, "i_perp"),
            "barrier": base.pareto_record_points(module, method, "barrier"),
        }
        for method in methods
    }
    figure = plt.figure(figsize=(base.mm_to_inch(180), base.mm_to_inch(72)), dpi=220)
    grid = GridSpec(
        1,
        2,
        figure=figure,
        left=0.065,
        right=0.975,
        bottom=0.17,
        top=0.78,
        wspace=0.24,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    for method in methods:
        style = base.METHOD_STYLE[method.key]
        for axis, curve in (
            (axes[0], curves[method.key]["i_perp"]),
            (axes[1], curves[method.key]["barrier"]),
        ):
            axis.plot(
                curve["metric_value"],
                curve["minimum_NF"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor="white",
                markeredgewidth=0.55,
                markevery=max(1, len(curve["metric_value"]) // 8),
                alpha=1.0 if method.key == "ivm" else 0.92,
                zorder=style["zorder"],
            )

    base.style_cost_axis(axes[0], "i_perp", 1.1e5, "1x2")
    base.style_cost_axis(axes[1], "barrier", 4.5e4, "1x2")
    for axis in axes:
        axis.set_ylabel("Number of Forces")
    axes[1].set_yticks((0, 15000, 30000, 45000))
    axes[1].set_xlabel(r"$\Delta E$ (kcal mol$^{-1}$)")
    for axis in axes:
        axis.set_box_aspect(None)

    iperp_comparison = base.pairwise_comparison_record(
        module, methods, "i_perp", base.IPERP_COMPARISON_THRESHOLD
    )
    barrier_comparison = base.pairwise_comparison_record(
        module, methods, "barrier", base.BARRIER_COMPARISON_TARGET
    )
    expected_costs = {
        "iperp": (2150, 83522),
        "barrier": (911, 19222),
    }
    observed_costs = {
        "iperp": (
            iperp_comparison["ivm_marginal_first_passage_NF"],
            iperp_comparison["ode_neb_marginal_first_passage_NF"],
        ),
        "barrier": (
            barrier_comparison["ivm_marginal_first_passage_NF"],
            barrier_comparison["ode_neb_marginal_first_passage_NF"],
        ),
    }
    if observed_costs != expected_costs:
        raise RuntimeError(f"First-passage drift: {observed_costs}")

    annotate_compact_speedup(
        axes[0],
        base.IPERP_COMPARISON_THRESHOLD,
        observed_costs["iperp"][0],
        observed_costs["iperp"][1],
        rf"{iperp_comparison['ode_neb_over_ivm']:.1f}$\times$",
        label_x=1.72e-4,
    )
    annotate_compact_speedup(
        axes[1],
        base.BARRIER_COMPARISON_TARGET,
        observed_costs["barrier"][0],
        observed_costs["barrier"][1],
        rf"{barrier_comparison['ode_neb_over_ivm']:.1f}$\times$",
        label_x=2.455,
    )

    figure.canvas.draw()
    for axis, letter in zip(axes, "ab"):
        position = axis.get_position()
        figure.text(
            position.x0 - 0.027,
            position.y1 + 0.012,
            letter,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
    handles = [base.legend_handle(method.key) for method in methods]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.49, 0.975),
        ncol=3,
        columnspacing=1.35,
        handlelength=2.35,
        handletextpad=0.5,
        frameon=False,
    )
    figure.savefig(
        output,
        format="pdf",
        dpi=600,
        metadata={
            "Title": "Alanine 66D single-run discrete IVM cost to precision",
            "Subject": "No-Phase-A/B discrete IVM and audited long-budget baselines",
        },
    )
    plt.close(figure)
    return curves, iperp_comparison, barrier_comparison


def source_rows(base, methods, curves, new_info: dict) -> list[dict]:
    rows = []
    ivm_nf_to_index = {
        int(nf): index for index, nf in enumerate(new_info["eligible_nf"])
    }
    for method in methods:
        for metric, panel in (("i_perp", "a"), ("barrier", "b")):
            curve = curves[method.key][metric]
            for metric_value, nf_value in zip(
                curve["metric_value"], curve["minimum_NF"]
            ):
                nf = int(nf_value)
                record = {
                    "panel": panel,
                    "method": method.key,
                    "metric": metric,
                    "metric_value": float(metric_value),
                    "running_best_metric": float(metric_value),
                    "minimum_NF": nf,
                    "source_call": "",
                    "is_initial": "",
                    "reparam_committed": "",
                    "segment_length_cv": "",
                    "segment_length_ratio": "",
                    "dense_subdivisions": "",
                    "dense_final_level_difference_kcal": "",
                    "diagnostic_calls_charged_to_NF": False,
                    "cost_source_file": str(method.result_path),
                    "metric_source_file": (
                        str(method.result_path)
                        if metric == "i_perp"
                        else str(method.dense_path)
                    ),
                    "source_stage": (
                        "raw_result"
                        if metric == "i_perp"
                        else "posthoc_dense_diagnostic"
                    ),
                }
                if method.key == "ode_neb" and metric == "i_perp":
                    record["metric_source_file"] = str(base.ODE_METRICS_PATH)
                    record["source_stage"] = "ode_metrics"
                if method.key == "ivm":
                    index = ivm_nf_to_index[nf]
                    record.update(
                        {
                            "source_call": int(new_info["eligible_calls"][index]),
                            "is_initial": bool(index == 0),
                            "reparam_committed": bool(
                                new_info["eligible_committed"][index]
                            ),
                            "segment_length_cv": float(
                                new_info["eligible_cv"][index]
                            ),
                            "segment_length_ratio": float(
                                new_info["eligible_ratio"][index]
                            ),
                            "source_stage": (
                                "single_run_complete_mesh"
                                if metric == "i_perp"
                                else "single_run_complete_mesh_dense20"
                            ),
                            "cost_source_file": str(NEW_RESULT),
                            "metric_source_file": (
                                str(NEW_RESULT) if metric == "i_perp" else str(NEW_DENSE)
                            ),
                        }
                    )
                    if metric == "barrier":
                        record["dense_subdivisions"] = int(
                            new_info["dense_subdivisions"][index]
                        )
                        record["dense_final_level_difference_kcal"] = float(
                            new_info["dense_final_level_difference"][index]
                        )
                rows.append(record)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_record(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path)}


def main() -> None:
    args = parse_args()
    validate_registered_inputs()
    base = load_base_plotter()
    base.configure_style()
    module, methods, disclosure, new_info = load_methods(base)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / f"{STEM}.pdf"
    csv_path = args.output_dir / f"{STEM}_source.csv"
    manifest_path = args.output_dir / f"{STEM}_manifest.json"
    if not args.overwrite:
        existing = [path for path in (pdf_path, csv_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite: {existing}")

    curves, iperp_comparison, barrier_comparison = draw_figure(
        base, module, methods, pdf_path
    )
    rows = source_rows(base, methods, curves, new_info)
    write_csv(csv_path, rows)

    inputs = {
        "new_single_run_result": file_record(NEW_RESULT),
        "forced_identity_result_not_drawn": file_record(FORCED_IDENTITY_RESULT),
        "new_single_run_dense": file_record(NEW_DENSE),
        "new_single_run_dense_audit": file_record(NEW_DENSE_AUDIT),
        "campaign_report": file_record(CAMPAIGN_REPORT),
        "preregistration": file_record(PREREGISTRATION),
        "formal_iperp_ctp": file_record(FORMAL_IPERP_CTP),
        "postrun_mechanism_audit": file_record(POSTRUN_AUDIT),
        "common_initial": file_record(COMMON_INITIAL),
        "topology": file_record(TOPOLOGY),
        "dense_postprocessor": file_record(POSTPROCESSOR),
        "base_muller_style_plotter": file_record(BASE_PLOTTER),
        "current_baseline_loader": file_record(
            SOURCE_DIR / "plot_all_methods_linear_speedup_v5.py"
        ),
        "ivm_loader_dependency": file_record(
            SOURCE_DIR / "plot_stage4_v4_pareto_all_methods.py"
        ),
        "legacy_method_loader": file_record(
            LEGACY / "src/plot_ala_integer20_all_methods.py"
        ),
        "new_candidate_plotter": file_record(Path(__file__).resolve()),
        "ode_metrics": file_record(base.ODE_METRICS_PATH),
    }
    for method in methods:
        if method.key == "ivm":
            continue
        inputs[f"{method.key}_result"] = file_record(Path(method.result_path))
        inputs[f"{method.key}_dense"] = file_record(Path(method.dense_path))

    manifest = {
        "status": "active_candidate_not_cited",
        "figure": str(pdf_path),
        "source_csv": str(csv_path),
        "scientific_identity": {
            "single_run_no_phase_a_b": True,
            "old_phase_a_b_ivm_used": False,
            "representative_arm": "one_way_takeover",
            "forced_operator_arm_drawn_separately": False,
            "forced_operator_arm_scientifically_identical": True,
            "ivm_label": "IVM",
            "ivm_algorithm_identity": "one_way_takeover; alpha=15; eta=1.1",
            "fixed_controls": new_info["controls"],
            "reparameterization": "existing state_curve/per_image policy",
            "optimization_horizon_NF": int(new_info["eligible_nf"][-1]),
        },
        "eligibility": {
            "definition": (
                "initial state OR reparam_committed; then segment_length_cv <= 0.0065 "
                "and segment_length_ratio <= 1.03"
            ),
            "indices_zero_based": new_info["eligible_indices"].tolist(),
            "stored_path_calls": new_info["eligible_calls"].tolist(),
            "whole_path_NF": new_info["eligible_nf"].tolist(),
            "state_count": len(new_info["eligible_nf"]),
        },
        "panels": {
            "a": {
                "metric": "I_perp cost to precision",
                "x_scale": "log_reversed",
                "x_range": [0.1, 0.0001],
                "x_label": "I_perp",
                "y_scale": "linear",
                "y_range_NF": [0, 110000],
                "y_label": "Number of Forces",
            },
            "b": {
                "metric": "actual densified energy barrier Delta E cost to lower value",
                "x_scale": "log_reversed",
                "x_range_kcal_per_mol": [3.0, 2.4],
                "x_label": "Delta E (kcal mol^-1)",
                "y_scale": "linear",
                "y_range_NF": [0, 45000],
                "y_label": "Number of Forces",
                "new_ivm_metric_source": "adaptive_barrier_kcal from NEW_DENSE",
                "sparse_result_barrier_kcal_used": False,
                "all_selected_subdivisions": 20,
                "posthoc_diagnostic_joint_evaluations": new_info["dense_queries"],
                "posthoc_diagnostic_calls_charged_to_optimization_NF": False,
            },
        },
        "curve_definition": {
            "vertices": "strict running-lowest actual observations in increasing NF",
            "drawstyle": "continuous straight segments between observed record vertices",
            "interpolation_used_for_costs": False,
            "extrapolation": False,
        },
        "pairwise_annotations": {
            "i_perp": iperp_comparison,
            "densified_barrier": barrier_comparison,
            "comparison_is_explicitly_ivm_vs_ode_neb_only": True,
            "metric_specific_marginal_first_passages": True,
            "simultaneous_joint_checkpoint_claim": False,
        },
        "baseline_policy": {
            "all_five_baseline_inputs_unchanged_from_existing_candidate": True,
            "ode_historical_initial_path_disclosure": disclosure,
        },
        "styles": {
            key: {
                **{
                    field: value
                    for field, value in base.METHOD_STYLE[key].items()
                    if field != "linestyle"
                },
                "linestyle": str(base.METHOD_STYLE[key]["linestyle"]),
            }
            for key in base.METHOD_ORDER
        },
        "inputs": inputs,
        "outputs": {
            "pdf": file_record(pdf_path),
            "source_csv": file_record(csv_path),
            "record_rows": len(rows),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(pdf_path)
    print(csv_path)
    print(manifest_path)
    print(
        f"I_perp: {iperp_comparison['ode_neb_over_ivm']:.6f}x; "
        f"barrier: {barrier_comparison['ode_neb_over_ivm']:.6f}x"
    )


if __name__ == "__main__":
    main()
