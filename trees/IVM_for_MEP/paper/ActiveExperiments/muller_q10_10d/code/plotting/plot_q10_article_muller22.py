#!/usr/bin/env python3
"""Render fixed-Q Muller-10D results with the formal Muller-22 article code."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np


LABELS = {
    "ivm": "IVM",
    "obsm_fire": "OBSM-FIRE",
    "obsm_lbfgs": "OBSM-LBFGS",
    "original_string": "Original String",
    "ode_neb": "ODE-NEB",
    "static_neb": "Static NEB",
}


def load_formal_namespace(repo: Path, stage: Path) -> dict:
    notebook_path = (
        repo / "code_on_server" / "IVM_on_server" / "scripts" / "figure_M10_N20_v2.ipynb"
    )
    if not notebook_path.exists():
        notebook_path = (
            Path(__file__).resolve().parent
            / "reference"
            / "figure_M10_N20_v2.ipynb"
        )
    with notebook_path.open() as handle:
        notebook = json.load(handle)
    source = "".join(notebook["cells"][7]["source"])
    source = source.rsplit("runs = main()", 1)[0]
    q_path = stage / "inputs" / "Q_data.npz"
    if not q_path.exists():
        q_path = stage / "data" / "inputs" / "Q_data.npz"
    source = source.replace(
        'base_dir = Path("/home/zqcai3/wzy/IVP_IVM")',
        f"base_dir = Path({str(stage)!r})",
    )
    source = source.replace(
        'q_data_path = Path("/home/zqcai3/wzy/IVP_new/Q_data.npz")',
        f"q_data_path = Path({str(q_path)!r})",
    )
    source = source.replace(
        'Q_np = np.load(q_data_path)["Q"]',
        'Q_np = np.load(q_data_path)["Q"].T',
    )
    source = source.replace("SHOW_FIGURES = True", "SHOW_FIGURES = False")
    source = source.replace(', bbox_inches="tight"', "")
    namespace = {"__name__": "muller_q10_formal_figure_runtime"}
    exec(compile(source, str(notebook_path), "exec"), namespace)
    namespace["plt"].rcParams["savefig.bbox"] = None
    formal_plot_metric = namespace["plot_metric_on_axis"]

    def plot_metric_with_linear_force_axis(*args, **kwargs):
        result = formal_plot_metric(*args, **kwargs)
        axis = kwargs.get("ax", args[0] if args else None)
        axis.set_xscale("linear")
        axis.set_xlim(0.0, 5000.0)
        axis.set_xlabel("Number of Forces")
        return result

    namespace["plot_metric_on_axis"] = plot_metric_with_linear_force_axis
    namespace["METRIC_INFO"]["Delta_Energy"]["ylabel"] = (
        r"Densified energy barrier $\Delta E$"
    )
    return namespace


def extend_styles(namespace: dict) -> None:
    neb = dict(namespace["METHOD_STYLE"]["NEB"])
    namespace["METHOD_STYLE"].update({
        "ODE-NEB": neb,
        "Static NEB": {
            "linestyle": "--", "linewidth": 1.65, "color": "#B279A2",
            "marker": "v", "markersize": 2.8, "zorder": 3,
        },
        "Original String": {
            "linestyle": ":", "linewidth": 1.65, "color": "#777777",
            "marker": "D", "markersize": 2.7, "zorder": 2,
        },
        "OBSM-FIRE": {
            "linestyle": "-.", "linewidth": 1.65, "color": "#5B9279",
            "marker": "s", "markersize": 2.8, "zorder": 3,
        },
        "OBSM-LBFGS": {
            "linestyle": (0, (5, 1.5)), "linewidth": 1.65, "color": "#CC79A7",
            "marker": "P", "markersize": 2.8, "zorder": 4,
        },
    })
    namespace["METHOD_STYLE"]["IVM"]["zorder"] = 20
    for method in LABELS.values():
        namespace["GRAD_EVALS_PER_COUNT_BY_METHOD"][method] = 1


def make_run(stage: Path, output: str, label: str, details: str,
             namespace: dict) -> dict:
    data = np.load(stage / output, allow_pickle=False)
    paths = np.asarray(data["path_history"], dtype=float)
    normal = np.asarray(data["normal_force_history"], dtype=float)
    run = {
        "method": label,
        "label": label,
        "short_label": label,
        "details": details,
        "file": stage / output,
        "images_seq": paths,
        "F_perp": normal,
        "count_seq": np.asarray(data["total_force_evaluations"], dtype=float),
    }
    run["metrics"] = namespace["compute_selected_metrics"](
        images_seq=paths,
        count_seq=run["count_seq"],
        F_perp=normal,
        V_func=namespace["V_muller10_np"],
    )
    # Each archived value was evaluated on a 500-point coordinate-wise cubic
    # interpolation of that checkpoint. Use it directly so the article panel's
    # densified barrier definition and data provenance are fully explicit.
    run["metrics"]["Delta_Energy"] = np.asarray(data["delta_e"], dtype=float)
    namespace["add_gradient_eval_count_to_run"](run)
    return run


def selected_ivm(stage: Path, namespace: dict) -> tuple[dict, dict]:
    with (stage / "reports" / "q10_inexact_tuned_resample_summary.json").open() as handle:
        rows = json.load(handle)
    row = next(
        item for item in rows
        if np.isclose(item["config"]["alpha"], 1e-3)
        and item["stop_reason"] == "stationary"
    )
    details = "IVM: " + json.dumps(row["config"], sort_keys=True)
    return make_run(stage, row["output"], "IVM", details, namespace), row


def main() -> None:
    stage = Path(__file__).resolve().parents[1]
    repo = stage.parent
    namespace = load_formal_namespace(repo, stage)
    extend_styles(namespace)
    ivm, ivm_row = selected_ivm(stage, namespace)
    with (stage / "reports" / "baseline_final_selection.json").open() as handle:
        selected = json.load(handle)["selected"]
    order = ("ode_neb", "static_neb", "original_string", "obsm_fire", "obsm_lbfgs")
    runs = [ivm]
    for method in order:
        item = selected[method]
        label = LABELS[method]
        details = label + ": " + json.dumps(item["config"], sort_keys=True)
        runs.append(make_run(stage, item["run"], label, details, namespace))

    figure_dir = stage / "figures" / "formal_article"
    figure_dir.mkdir(parents=True, exist_ok=True)
    namespace["save_dir"] = figure_dir
    namespace["SHOW_FIGURES"] = False
    namespace["SAVE_FORMATS"] = ("pdf", "svg", "png")
    namespace["PNG_DPI"] = 600
    namespace["plot_2x2_metrics_and_contour"](
        runs=runs, save_dir=figure_dir, use_compressed_y=True
    )
    namespace["plot_1x3_metrics"](
        runs=runs, save_dir=figure_dir, use_compressed_y=True
    )
    namespace["plot_contour_only"](runs=runs, save_dir=figure_dir)

    source_rows = []
    for run in runs:
        metrics = run["metrics"]
        for index in range(len(metrics["grad_count_seq"])):
            source_rows.append({
                "method": run["label"],
                "number_of_forces": float(metrics["grad_count_seq"][index]),
                "I_perp": float(metrics["loss_perp"][index]),
                "I_arc": float(metrics["loss_arc"][index]),
                "Delta_E_dense": float(metrics["Delta_Energy"][index]),
                "production_file": str(run["file"]),
            })
    with (figure_dir / "muller22_q10_article_source.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    formal_stem = "N20_BestMethods_2x2_lossperp_lossarc_deltaE_contour_gradEvals_compressed"
    for suffix in ("pdf", "svg", "png"):
        shutil.copyfile(
            figure_dir / f"{formal_stem}.{suffix}",
            figure_dir / f"muller22_q10_article_method_comparison.{suffix}",
        )
    with (figure_dir / "muller22_q10_article_manifest.json").open("w") as handle:
        json.dump({
            "case": "fixed-Q unperturbed 10D Muller-Brown",
            "formal_notebook": str(
                repo / "code_on_server" / "IVM_on_server" / "scripts" / "figure_M10_N20_v2.ipynb"
            ),
            "formal_notebook_cell": 7,
            "q_sha256": "372962a47504e6df28a20d8b022482a6835680765434b4385a6dba0e826974bc",
            "ivm_config": ivm_row["config"],
            "ivm_output": ivm_row["output"],
            "run_order": [run["label"] for run in runs],
            "ivm_highest_zorder": True,
            "cost_axis": (
                "total uncached force calls summed over all 20 internal images"
            ),
            "xscale": "linear",
            "xlim": [0, 5000],
            "energy_barrier_metric": (
                "500-point coordinate-wise cubic densified Delta E"
            ),
        }, handle, indent=2, sort_keys=True)
    print(figure_dir / "muller22_q10_article_method_comparison.pdf")


if __name__ == "__main__":
    main()
