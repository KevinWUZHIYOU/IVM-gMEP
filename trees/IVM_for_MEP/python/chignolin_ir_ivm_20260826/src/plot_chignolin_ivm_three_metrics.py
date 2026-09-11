#!/usr/bin/env python3
"""Three-panel Chignolin cost-to-precision figure with square axes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import plot_chignolin_ivm_arc500 as ref


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json"
DEFAULT_ARC500 = ROOT / "runs/diagnostics/chignolin_ivm_arc_length_500.npz"
DEFAULT_OUTPUT = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/chignolin/active_candidates/"
    "chignolin_ir_ivm_cpu1"
)
DEFAULT_STEM = "chignolin_ir_ivm_overlay_total_nf"
ACTION_CONTRACT = ROOT / "registrations/GEOMETRIC_ACTION_DISCRETE_V1.json"
ACTION_RESULT = ROOT / "runs/diagnostics/chignolin_geometric_action_discrete_v1.npz"


def load_action_diagnostic():
    """Validate the immutable diagnostic without reopening moved optimizer files."""
    config = json.loads(ACTION_CONTRACT.read_text())
    audit_path = ACTION_RESULT.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text())
    if audit["result_sha256"] != ref.sha256_file(ACTION_RESULT):
        raise ValueError("Action result hash mismatch")
    if audit["contract_sha256"] != ref.sha256_file(ACTION_CONTRACT):
        raise ValueError("Action contract hash mismatch")
    with np.load(ACTION_RESULT, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    if len(data["method"]) != config["selected_path_count"]:
        raise ValueError("Action row count mismatch")
    if int(data["diagnostic_joint_evaluations"].sum()) != audit["diagnostic_joint_evaluations"]:
        raise ValueError("Action diagnostic-call count mismatch")
    np.testing.assert_array_equal(data["counter_vector"].sum(axis=1), data["number_of_forces"])
    np.testing.assert_array_equal(data["diagnostic_joint_evaluations"], 22)
    reconstructed = (
        data["segment_length_angstrom"]
        * (data["gradient_norm_ev_per_angstrom"][:, :-1]
           + data["gradient_norm_ev_per_angstrom"][:, 1:])
        * 0.5
    ).sum(axis=1) * config["ev_to_kcal"]
    np.testing.assert_allclose(reconstructed, data["action_kcal"], rtol=5e-15)
    np.testing.assert_allclose(data["recomputed_i_perp"], data["i_perp"], rtol=0, atol=0)
    return data, config, audit


def validate_and_report(contract_path: Path, arc500_path: Path):
    contract_path, contract = ref.load_contract(contract_path)
    dense, dense_audit = ref.load_arc500(arc500_path, contract_path, contract)
    action_data, action_contract, action_audit = load_action_diagnostic()

    histories = {}
    for method in ref.METHOD_ORDER:
        take = action_data["method"] == method
        source = ref._resolve_contract_path(contract_path, contract["methods"][method]["source_result"])
        source_hash = str(contract["methods"][method]["source_result_sha256"])
        if set(action_data["source_result"][take]) != {str(source)}:
            raise ValueError(f"Action/raw source mismatch for {method}")
        if set(action_data["source_result_sha256"][take]) != {source_hash}:
            raise ValueError(f"Action/raw source hash mismatch for {method}")
        histories[method] = {
            "number_of_forces": action_data["number_of_forces"][take],
            "i_perp": action_data["i_perp"][take],
        }

    # Every arc-length-500 row must map exactly to a row in the action diagnostic.
    with np.load(arc500_path, allow_pickle=False) as saved:
        arc = {key: saved[key] for key in (
            "method", "source_result", "source_result_sha256", "source_record_index",
            "paper_total_nf", "i_perp", "selected_path_sha256",
        )}
    action_lookup = {
        (str(action_data["method"][i]), int(action_data["source_record_index"][i])): i
        for i in range(len(action_data["method"]))
    }
    for i in range(len(arc["method"])):
        key = (str(arc["method"][i]), int(arc["source_record_index"][i]))
        j = action_lookup.get(key)
        if j is None:
            raise ValueError(f"Arc500 row absent from action diagnostic: {key}")
        for left, right in (
            (arc["source_result"][i], action_data["source_result"][j]),
            (arc["source_result_sha256"][i], action_data["source_result_sha256"][j]),
            (arc["paper_total_nf"][i], action_data["number_of_forces"][j]),
            (arc["i_perp"][i], action_data["i_perp"][j]),
            (arc["selected_path_sha256"][i], action_data["selected_path_sha256"][j]),
        ):
            if left != right:
                raise ValueError(f"Arc500/action row mismatch for {key}")

    i_threshold = max(float(histories[m]["i_perp"].min()) for m in ref.REFERENCE_METHODS)
    barrier_threshold = max(float(dense[m]["barrier_kcal"].min()) for m in ref.REFERENCE_METHODS)
    passages = {}
    for method in ref.METHOD_ORDER:
        passages[method] = {
            "i_perp": ref.first_passage(histories[method]["number_of_forces"],
                                         histories[method]["i_perp"], i_threshold),
            "barrier_500_kcal": ref.first_passage(dense[method]["number_of_forces"],
                                                   dense[method]["barrier_kcal"], barrier_threshold),
        }

    action_threshold = float(
        action_data["action_kcal"][action_data["method"] == "ODE-NEB"].min()
    )
    action_passages = {}
    for method in ref.METHOD_ORDER:
        take = action_data["method"] == method
        action_passages[method] = ref.first_passage(
            action_data["number_of_forces"][take],
            action_data["action_kcal"][take],
            action_threshold,
        )
    if action_passages["IVM"] != 905 or action_passages["ODE-NEB"] != 20002:
        raise ValueError("Unexpected action first passage")

    report = dict(
        status="validated",
        display_label_policy="The selected IR-IVM run is displayed as IVM; historical V9 IVM excluded",
        method_order=list(ref.METHOD_ORDER),
        contract=str(contract_path), contract_sha256=ref.sha256_file(contract_path),
        arc_length_500_result=str(arc500_path.resolve()),
        arc_length_500_result_sha256=ref.sha256_file(arc500_path),
        arc_length_500_audit=str(arc500_path.with_suffix(".audit.json")),
        arc_length_500_audit_sha256=ref.sha256_file(arc500_path.with_suffix(".audit.json")),
        arc_length_500_protocol=contract["arc_length_500_protocol"],
        diagnostic_counts_excluded_from_optimization_axis=True,
        paper_cost_axis="counter_vectors.sum(axis=1)", paper_cost_includes_all_22_images=True,
        selected_path_counts={m: dense[m]["selected_path_count"] for m in ref.METHOD_ORDER},
        thresholds={"i_perp": i_threshold, "barrier_500_kcal": barrier_threshold,
                    "definition": "I_perp and Delta E use strictest precision reached by IVM, ODE-NEB, OBSM-FIRE"},
        first_passage_total_nf=passages,
        speedup_vs_baselines={metric: {"ODE-NEB": passages["ODE-NEB"][metric] / passages["IVM"][metric]}
                              for metric in ("i_perp", "barrier_500_kcal")},
        source_results={m: {"path": str(ref._resolve_contract_path(contract_path, contract["methods"][m]["source_result"])),
                            "sha256": str(contract["methods"][m]["source_result_sha256"])}
                        for m in ref.METHOD_ORDER},
        figure_revision="2026-09-06: geometric action added; all three axes boxes square; action label standardized to S-hat without a discretization subscript",
        panel_order=["i_perp", "barrier_500_kcal", "action_kcal"],
        axes_box_aspect=1.0,
        delta_e_panel={
            "scale": "reversed_linear",
            "visible_xlim_kcal_per_mol": [100.0, 19.5],
            "plotting_policy": "Plot the complete Pareto polyline, then clip it to the unchanged visible x range",
            "full_recorded_range_kcal_per_mol": [
                min(float(dense[m]["barrier_kcal"].min()) for m in ref.METHOD_ORDER),
                max(float(dense[m]["barrier_kcal"].max()) for m in ref.METHOD_ORDER),
            ],
            "initial_value_kcal_per_mol": float(dense["IVM"]["barrier_kcal"][0]),
            "records_above_100_by_method": {
                m: int(np.count_nonzero(dense[m]["barrier_kcal"] > 100.0))
                for m in ref.METHOD_ORDER
            },
        },
        action_definition=action_contract["formula"],
        action_unit="kcal/mol",
        action_result=str(ACTION_RESULT), action_result_sha256=ref.sha256_file(ACTION_RESULT),
        action_contract=str(ACTION_CONTRACT), action_contract_sha256=ref.sha256_file(ACTION_CONTRACT),
        action_diagnostic_audit=action_audit,
        action_selected_path_counts={
            method: int(np.count_nonzero(action_data["method"] == method))
            for method in ref.METHOD_ORDER
        },
        source_alignment="Every arc500 row matches action method, record, NoF, Iperp, path SHA, source path and source SHA",
        source_availability_note="Historical source paths are pinned by the contracts; combined plot validates immutable diagnostic artifacts without recalculating physics",
    )
    report["thresholds"].update(
        action_kcal=action_threshold,
        action_definition="exact minimum discrete action recorded by ODE-NEB; no rounding",
    )
    report["first_passage_total_nf"] = {
        method: dict(passages[method], action_kcal=action_passages[method])
        for method in ref.METHOD_ORDER
    }
    report["speedup_vs_baselines"]["action_kcal"] = {
        "ODE-NEB": action_passages["ODE-NEB"] / action_passages["IVM"]
    }
    report["plot_code"] = str(Path(__file__).resolve())
    return contract, histories, dense, action_data, report


def draw(output_dir: Path, stem: str, histories, dense, action_data, report, replace=False):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / f"{stem}.pdf").exists() and not replace:
        raise FileExistsError("Refusing to overwrite figure without --replace")

    ref.setup_style(mpl)
    figure = plt.figure(figsize=(ref.mm_to_inch(180), ref.mm_to_inch(82)), dpi=220)
    grid = GridSpec(
        1, 3, figure=figure, left=0.058, right=0.988, bottom=0.165,
        top=0.76, wspace=0.43,
    )
    axes = [figure.add_subplot(grid[0, i]) for i in range(3)]
    action_bounds = (
        float(action_data["action_kcal"].min()) * 0.60,
        10.0 ** np.ceil(np.log10(float(action_data["action_kcal"].max()))),
    )
    barrier_max = max(float(dense[m]["barrier_kcal"].max()) for m in ref.METHOD_ORDER)
    barrier_full_bounds = (19.5, barrier_max * 1.18)
    panels = (
        dict(axis=axes[0], metric="i_perp", key="i_perp", source=histories,
             bounds=(1e-2, 10.0 ** 4.5), view_bounds=(1e-2, 10.0 ** 4.5),
             scale="log", xlabel=r"$I_\perp$"),
        dict(axis=axes[1], metric="barrier_500_kcal", key="barrier_kcal", source=dense,
             bounds=barrier_full_bounds, view_bounds=(19.5, 100.0),
             scale="linear", xlabel=r"$\Delta E$ (kcal mol$^{-1}$)"),
        dict(axis=axes[2], metric="action_kcal", key="action_kcal", source=None,
             bounds=action_bounds, view_bounds=action_bounds,
             scale="log", xlabel=r"$\widehat{S}$ (kcal mol$^{-1}$)"),
    )

    frontier_rows = []
    for method in ref.METHOD_ORDER:
        style = ref.METHOD_STYLE[method]
        for panel in panels:
            if panel["metric"] == "action_kcal":
                take = action_data["method"] == method
                nf = action_data["number_of_forces"][take]
                values = action_data["action_kcal"][take]
            else:
                nf = panel["source"][method]["number_of_forces"]
                values = panel["source"][method][panel["key"]]
            xx, yy = ref.pareto_frontier(nf, values, *panel["bounds"])
            panel["axis"].plot(
                xx, yy, color=style["color"], linestyle=style["linestyle"],
                linewidth=style["linewidth"], marker=style["marker"],
                markersize=style["markersize"], markerfacecolor="white",
                markeredgewidth=0.55, markevery=max(1, len(xx) // 8),
                alpha=1.0 if method == "IVM" else 0.92, zorder=style["zorder"],
            )
            frontier_rows.extend(
                dict(method=method, metric=panel["metric"], measured_record_threshold=float(x),
                     number_of_forces=int(y)) for x, y in zip(xx, yy)
            )

    label_x = {
        "i_perp": report["thresholds"]["i_perp"] * 0.80,
        "barrier_500_kcal": report["thresholds"]["barrier_500_kcal"] * 0.80,
        "action_kcal": report["thresholds"]["action_kcal"] * 0.80,
    }
    annotation_rows = []
    for panel in panels:
        axis = panel["axis"]
        metric = panel["metric"]
        if panel["scale"] == "log":
            axis.set_xscale("log")
        axis.set_xlim(panel["view_bounds"][1], panel["view_bounds"][0])
        if metric == "barrier_500_kcal":
            axis.set_xticks((100, 80, 60, 40, 20))
        axis.set_ylim(0, 22000)
        axis.set_yticks((0, 5000, 10000, 15000, 20000))
        axis.yaxis.set_major_formatter(FuncFormatter(ref.force_formatter))
        axis.set_xlabel(panel["xlabel"])
        axis.set_ylabel("Number of Forces")
        axis.grid(True, which="major", color="#D9D9D6", linewidth=0.4, alpha=0.62)
        axis.set_box_aspect(1)

        threshold = float(report["thresholds"][metric])
        ivm_nf = report["first_passage_total_nf"]["IVM"][metric]
        ode_nf = report["first_passage_total_nf"]["ODE-NEB"][metric]
        if ivm_nf is None or ode_nf is None:
            raise ValueError(f"Missing IVM/ODE-NEB first passage for {metric}")
        ratio = ref.add_speedup_arrow(
            axis, threshold, int(ivm_nf), int(ode_nf), float(label_x[metric])
        )
        annotation_rows.append(
            dict(metric=metric, precision_threshold=threshold, baseline="ODE-NEB",
                 ivm_number_of_forces=int(ivm_nf), baseline_number_of_forces=int(ode_nf),
                 ode_neb_over_ivm=ratio)
        )

    figure.canvas.draw()
    for axis, letter in zip(axes, "abc"):
        position = axis.get_position()
        figure.text(position.x0 - 0.026, position.y1 + 0.011, letter,
                    fontsize=8, fontweight="bold", ha="left", va="bottom")
    handles = [
        Line2D([0], [0], label=ref.METHOD_STYLE[m]["label"],
               color=ref.METHOD_STYLE[m]["color"], linestyle=ref.METHOD_STYLE[m]["linestyle"],
               linewidth=ref.METHOD_STYLE[m]["linewidth"], marker=ref.METHOD_STYLE[m]["marker"],
               markersize=ref.METHOD_STYLE[m]["markersize"], markerfacecolor="white",
               markeredgewidth=0.6)
        for m in ref.METHOD_ORDER
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.51, 0.965),
                  ncol=3, columnspacing=1.35, handlelength=2.35, handletextpad=0.5,
                  frameon=False)

    metadata = {
        "Title": "Chignolin IVM cost-to-precision comparison: three metrics",
        "Subject": "I_perp, arc-length 500-point Delta E, and discrete geometric action; square panels",
    }
    generated = {}
    for suffix, kwargs in (
        ("pdf", dict(format="pdf", dpi=600, metadata=metadata)),
        ("svg", dict(format="svg")),
        ("png", dict(format="png", dpi=600)),
    ):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, **kwargs)
        generated[suffix] = dict(path=str(path.resolve()), sha256=ref.sha256_file(path))
    plt.close(figure)

    frontier_path = output_dir / f"{stem}_frontier.csv"
    annotation_path = output_dir / f"{stem}_annotations.csv"
    ref.write_csv(frontier_path, frontier_rows)
    ref.write_csv(annotation_path, annotation_rows)
    generated["frontier"] = dict(path=str(frontier_path.resolve()), sha256=ref.sha256_file(frontier_path))
    generated["annotations"] = dict(path=str(annotation_path.resolve()), sha256=ref.sha256_file(annotation_path))
    report["generated_files"] = generated
    report["annotations"] = annotation_rows
    report["plot_code_sha256"] = ref.sha256_file(Path(__file__))
    provenance_path = output_dir / f"{stem}_provenance.json"
    provenance_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--arc500", type=Path, default=DEFAULT_ARC500)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    _, histories, dense, action_data, report = validate_and_report(
        args.contract.resolve(), args.arc500.resolve()
    )
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    draw(args.output_dir.resolve(), args.stem, histories, dense, action_data, report, args.replace)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
