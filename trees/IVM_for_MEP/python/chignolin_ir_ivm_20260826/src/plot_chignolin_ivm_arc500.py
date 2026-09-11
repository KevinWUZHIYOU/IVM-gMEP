#!/usr/bin/env python3
"""Plot Chignolin cost to precision with IR-IVM promoted to ``IVM``.

The historical V9 IVM curve is intentionally excluded.  The selected IR-IVM
run supplies the curve and legend entry named ``IVM``.  The right panel reads
only fixed, whole-path arc-length diagnostics containing 500 points including
both endpoints; the left panel continues to use the manuscript definition of
``I_perp`` from the raw optimization histories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


METHOD_ORDER = (
    "IVM",
    "ODE-NEB",
    "Static-NEB",
    "Original String",
    "OBSM-FIRE",
)
REFERENCE_METHODS = ("IVM", "ODE-NEB", "OBSM-FIRE")
METHOD_STYLE = {
    "IVM": {
        "label": "IVM",
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": 1.85,
        "marker": "o",
        "markersize": 3.1,
        "zorder": 20,
    },
    "ODE-NEB": {
        "label": "ODE-NEB",
        "color": "#D55E00",
        "linestyle": "--",
        "linewidth": 1.65,
        "marker": "s",
        "markersize": 2.9,
        "zorder": 6,
    },
    "Static-NEB": {
        "label": "Static NEB",
        "color": "#B279A2",
        "linestyle": "--",
        "linewidth": 1.65,
        "marker": "v",
        "markersize": 2.8,
        "zorder": 4,
    },
    "Original String": {
        "label": "Original String",
        "color": "#777777",
        "linestyle": ":",
        "linewidth": 1.65,
        "marker": "D",
        "markersize": 2.7,
        "zorder": 3,
    },
    "OBSM-FIRE": {
        "label": "OBSM-FIRE",
        "color": "#5B9279",
        "linestyle": "-.",
        "linewidth": 1.65,
        "marker": "s",
        "markersize": 2.8,
        "zorder": 4,
    },
}
N_DENSE_POINTS = 500


def mm_to_inch(value: float) -> float:
    return value / 25.4


def force_formatter(value: float, _position: int) -> str:
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(saved, key: str):
    return np.asarray(saved[key]).item()


def _resolve_contract_path(contract_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = contract_path.parent / path
    return path.resolve()


def pareto_frontier(
    number_of_forces: np.ndarray,
    values: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.asarray(number_of_forces), kind="stable")
    costs = np.asarray(number_of_forces, dtype=np.float64)[order]
    metric = np.asarray(values, dtype=np.float64)[order]
    frontier_value: List[float] = []
    frontier_cost: List[float] = []
    running_best = np.inf
    for cost, value in zip(costs, metric):
        if not np.isfinite(cost) or not np.isfinite(value):
            continue
        tolerance = 16.0 * np.finfo(float).eps * max(
            1.0, abs(running_best) if np.isfinite(running_best) else 1.0
        )
        if value < running_best - tolerance:
            running_best = float(value)
            if float(lower_bound) <= running_best <= float(upper_bound):
                if frontier_cost and cost == frontier_cost[-1]:
                    frontier_value[-1] = running_best
                else:
                    frontier_value.append(running_best)
                    frontier_cost.append(float(cost))
    return np.asarray(frontier_value), np.asarray(frontier_cost)


def first_passage(
    number_of_forces: np.ndarray, values: np.ndarray, threshold: float
) -> Optional[int]:
    cost = np.asarray(number_of_forces, dtype=np.int64)
    metric = np.asarray(values, dtype=np.float64)
    if cost.shape != metric.shape:
        raise ValueError("cost and metric vectors differ in shape")
    order = np.argsort(cost, kind="stable")
    running_best = np.minimum.accumulate(metric[order])
    reached = np.flatnonzero(running_best <= float(threshold))
    return None if not reached.size else int(cost[order][reached[0]])


def load_contract(path: Path) -> Tuple[Path, Dict]:
    contract_path = Path(path).expanduser().resolve()
    contract = json.loads(contract_path.read_text())
    if tuple(contract.get("methods", {})) != METHOD_ORDER:
        raise ValueError(f"contract method order must be {METHOD_ORDER}")
    protocol = contract.get("arc_length_500_protocol", {})
    expected = {
        "coordinate_space": "unweighted_3N_cartesian",
        "interpolation": "piecewise_linear",
        "number_of_points": N_DENSE_POINTS,
        "endpoint_policy": "include_both_endpoints",
        "grid_formula": "s_j=j*L/499,j=0,...,499",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"arc-length protocol mismatch for {key}")
    return contract_path, contract


def load_raw_histories(contract_path: Path, contract: Dict) -> Dict[str, Dict]:
    histories: Dict[str, Dict] = {}
    for method in METHOD_ORDER:
        record = contract["methods"][method]
        path = _resolve_contract_path(contract_path, record["source_result"])
        if sha256_file(path) != str(record["source_result_sha256"]):
            raise ValueError(f"{method} source-result hash mismatch")
        with np.load(path, allow_pickle=False) as saved:
            counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
            i_perp = np.asarray(saved["i_perp"], dtype=np.float64)
            calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
            if counters.shape != (len(i_perp), 22) or calls.shape != i_perp.shape:
                raise ValueError(f"{method} raw histories are misaligned")
            total_nf = counters.sum(axis=1, dtype=np.int64)
            if np.any(np.diff(total_nf) < 0):
                raise ValueError(f"{method} Number-of-Forces history is not cumulative")
            eligibility = str(record["eligibility"])
            if eligibility == "initial_or_reparam_committed":
                committed = np.asarray(saved["reparam_committed"], dtype=bool)
                keep = committed.copy()
                keep[0] = True
            else:
                keep = np.ones(len(i_perp), dtype=bool)
            initial_hash = str(_scalar(saved, "initial_path_sha256"))
            if initial_hash != str(contract["common_initial_path_sha256"]):
                raise ValueError(f"{method} initial-path hash mismatch")
            metadata = {}
            if method == "IVM":
                for key in (
                    "alpha",
                    "discrete_residual_ratio",
                    "discrete_inner_max_steps",
                    "reparam_mode",
                    "reparam_every",
                    "state_globalization",
                    "frozen_subproblem",
                ):
                    if key in saved.files:
                        metadata[key] = _scalar(saved, key)
        histories[method] = {
            "source_result": str(path),
            "source_result_sha256": sha256_file(path),
            "source_record_index": np.flatnonzero(keep),
            "number_of_forces": total_nf[keep],
            "calls_per_image": calls[keep],
            "i_perp": i_perp[keep],
            "metadata": metadata,
        }
    return histories


def load_arc500(
    path: Path, contract_path: Path, contract: Dict
) -> Tuple[Dict[str, Dict], Dict]:
    dense_path = Path(path).expanduser().resolve()
    audit_path = dense_path.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text())
    if audit.get("output_sha256") != sha256_file(dense_path):
        raise ValueError("arc500 output hash does not match its audit")
    if audit.get("contract_sha256") != sha256_file(contract_path):
        raise ValueError("arc500 result used another comparison contract")
    if not audit.get("every_selected_path_has_exactly_500_fresh_evaluations"):
        raise ValueError("arc500 audit does not certify 500 evaluations per path")
    if not audit.get("diagnostic_counts_excluded_from_optimization_axis"):
        raise ValueError("arc500 calls contaminate the optimization axis")
    with np.load(dense_path, allow_pickle=False) as saved:
        required = {
            "method",
            "source_result",
            "source_result_sha256",
            "source_record_index",
            "paper_total_nf",
            "i_perp",
            "selected_counter_vectors",
            "selected_path_sha256",
            "barrier_500_kcal",
            "diagnostic_joint_evaluations_per_path",
            "coordinate_space",
            "interpolation",
            "number_of_points",
            "endpoint_policy",
            "grid_formula",
            "diagnostic_counts_excluded_from_optimization_axis",
            "contract_sha256",
        }
        missing = sorted(required.difference(saved.files))
        if missing:
            raise ValueError("arc500 result is missing " + ",".join(missing))
        if (
            str(_scalar(saved, "coordinate_space")) != "unweighted_3N_cartesian"
            or str(_scalar(saved, "interpolation")) != "piecewise_linear"
            or int(_scalar(saved, "number_of_points")) != N_DENSE_POINTS
            or str(_scalar(saved, "endpoint_policy")) != "include_both_endpoints"
            or str(_scalar(saved, "grid_formula")) != "s_j=j*L/499,j=0,...,499"
            or not bool(_scalar(saved, "diagnostic_counts_excluded_from_optimization_axis"))
            or str(_scalar(saved, "contract_sha256")) != sha256_file(contract_path)
        ):
            raise ValueError("arc500 NPZ protocol metadata mismatch")
        methods = np.asarray(saved["method"]).astype(str)
        source = np.asarray(saved["source_result"]).astype(str)
        source_hash = np.asarray(saved["source_result_sha256"]).astype(str)
        record_index = np.asarray(saved["source_record_index"], dtype=np.int64)
        nf = np.asarray(saved["paper_total_nf"], dtype=np.int64)
        i_perp = np.asarray(saved["i_perp"], dtype=np.float64)
        counters = np.asarray(saved["selected_counter_vectors"], dtype=np.int64)
        barrier = np.asarray(saved["barrier_500_kcal"], dtype=np.float64)
        diagnostic_calls = np.asarray(
            saved["diagnostic_joint_evaluations_per_path"], dtype=np.int64
        )
        n = len(methods)
        if not all(len(array) == n for array in (source, source_hash, record_index, nf, i_perp, counters, barrier, diagnostic_calls)):
            raise ValueError("arc500 arrays differ in length")
        if counters.shape != (n, 22) or not np.array_equal(counters.sum(axis=1), nf):
            raise ValueError("arc500 total NoF does not equal all 22 counters")
        if not np.all(diagnostic_calls == N_DENSE_POINTS):
            raise ValueError("a selected path did not use exactly 500 diagnostic calls")

        dense: Dict[str, Dict] = {}
        for method in METHOD_ORDER:
            selected = np.flatnonzero(methods == method)
            if not selected.size:
                raise ValueError(f"arc500 result has no {method} paths")
            expected_source = _resolve_contract_path(
                contract_path, contract["methods"][method]["source_result"]
            )
            expected_hash = str(contract["methods"][method]["source_result_sha256"])
            if set(source[selected]) != {str(expected_source)} or set(source_hash[selected]) != {expected_hash}:
                raise ValueError(f"{method} arc500 source mismatch")
            order = np.argsort(nf[selected], kind="stable")
            selected = selected[order]
            dense[method] = {
                "source_record_index": record_index[selected],
                "number_of_forces": nf[selected],
                "i_perp": i_perp[selected],
                "barrier_kcal": barrier[selected],
                "selected_path_count": int(len(selected)),
            }
    return dense, audit


def setup_style(mpl) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.6,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "it",
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "legend.frameon": False,
            "legend.handlelength": 2.2,
            "legend.handletextpad": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": None,
        }
    )


def add_speedup_arrow(
    axis,
    target: float,
    ivm_cost: int,
    baseline_cost: int,
    label_x: float,
) -> float:
    speedup = float(baseline_cost) / float(ivm_cost)
    axis.axvline(target, color="#767676", linewidth=0.55, linestyle=":", zorder=1)
    axis.annotate(
        "",
        xy=(target, ivm_cost),
        xytext=(target, baseline_cost),
        arrowprops={"arrowstyle": "<->", "color": "#202020", "lw": 0.8},
        zorder=25,
    )
    axis.text(
        label_x,
        0.5 * (float(ivm_cost) + float(baseline_cost)),
        rf"{speedup:.1f}$\times$",
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color="#202020",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.8},
        zorder=26,
    )
    return speedup


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    contract_path: Path,
    contract: Dict,
    dense_path: Path,
    dense_audit: Dict,
    histories: Dict[str, Dict],
    dense: Dict[str, Dict],
) -> Dict:
    i_threshold = max(float(np.min(histories[m]["i_perp"])) for m in REFERENCE_METHODS)
    b_threshold = max(float(np.min(dense[m]["barrier_kcal"])) for m in REFERENCE_METHODS)
    passages: Dict[str, Dict[str, Optional[int]]] = {}
    for method in METHOD_ORDER:
        passages[method] = {
            "i_perp": first_passage(
                histories[method]["number_of_forces"], histories[method]["i_perp"], i_threshold
            ),
            "barrier_500_kcal": first_passage(
                dense[method]["number_of_forces"], dense[method]["barrier_kcal"], b_threshold
            ),
        }
    speedups = {}
    for metric in ("i_perp", "barrier_500_kcal"):
        ivm = passages["IVM"][metric]
        speedups[metric] = {
            baseline: (
                None
                if ivm is None or passages[baseline][metric] is None
                else float(passages[baseline][metric]) / float(ivm)
            )
            for baseline in ("ODE-NEB",)
        }
    return {
        "status": "validated",
        "display_label_policy": (
            "The selected IR-IVM run is the IVM result; the historical V9 IVM run is excluded."
        ),
        "method_order": list(METHOD_ORDER),
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "arc_length_500_result": str(dense_path),
        "arc_length_500_result_sha256": sha256_file(dense_path),
        "arc_length_500_audit": str(dense_path.with_suffix(".audit.json")),
        "arc_length_500_audit_sha256": sha256_file(dense_path.with_suffix(".audit.json")),
        "arc_length_500_protocol": contract["arc_length_500_protocol"],
        "diagnostic_counts_excluded_from_optimization_axis": True,
        "paper_cost_axis": "counter_vectors.sum(axis=1)",
        "paper_cost_includes_all_22_images": True,
        "selected_path_counts": {
            method: dense[method]["selected_path_count"] for method in METHOD_ORDER
        },
        "thresholds": {
            "i_perp": i_threshold,
            "barrier_500_kcal": b_threshold,
            "definition": "strictest precision reached by IVM, ODE-NEB, and OBSM-FIRE",
        },
        "first_passage_total_nf": passages,
        "speedup_vs_baselines": speedups,
        "ivm_algorithm_metadata": histories["IVM"]["metadata"],
        "source_results": {
            method: {
                "path": histories[method]["source_result"],
                "sha256": histories[method]["source_result_sha256"],
            }
            for method in METHOD_ORDER
        },
        "diagnostic_audit_summary": {
            key: dense_audit[key]
            for key in (
                "selected_path_count",
                "diagnostic_joint_evaluations",
                "every_selected_path_has_exactly_500_fresh_evaluations",
            )
        },
    }


def save_figure(
    output_dir: Path,
    stem: str,
    contract: Dict,
    histories: Dict[str, Dict],
    dense: Dict[str, Dict],
    report: Dict,
) -> List[Path]:
    import matplotlib as mpl  # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    from matplotlib.gridspec import GridSpec  # pylint: disable=import-outside-toplevel
    from matplotlib.lines import Line2D  # pylint: disable=import-outside-toplevel
    from matplotlib.ticker import FuncFormatter  # pylint: disable=import-outside-toplevel

    setup_style(mpl)
    figure = plt.figure(figsize=(mm_to_inch(180), mm_to_inch(72)), dpi=220)
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
    frontier_rows: List[Dict] = []
    for method in METHOD_ORDER:
        style = METHOD_STYLE[method]
        for axis, source, key, bounds, metric in (
            (axes[0], histories, "i_perp", (1e-2, 10 ** 4.5), "i_perp"),
            (axes[1], dense, "barrier_kcal", (19.5, 100.0), "barrier_500_kcal"),
        ):
            frontier, costs = pareto_frontier(
                source[method]["number_of_forces"], source[method][key], bounds[0], bounds[1]
            )
            axis.plot(
                frontier,
                costs,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor="white",
                markeredgewidth=0.55,
                markevery=max(1, len(frontier) // 8),
                alpha=1.0 if method == "IVM" else 0.92,
                zorder=style["zorder"],
            )
            for threshold, cost in zip(frontier, costs):
                frontier_rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "measured_record_threshold": float(threshold),
                        "number_of_forces": int(cost),
                    }
                )

    thresholds = report["thresholds"]
    for index, (axis, xlabel, metric) in enumerate(
        (
            (axes[0], r"$I_\perp$", "i_perp"),
            (axes[1], r"$\Delta E$ (kcal mol$^{-1}$)", "barrier_500_kcal"),
        )
    ):
        if index == 0:
            axis.set_xscale("log")
            axis.set_xlim(10 ** 4.5, 1e-2)
        else:
            axis.set_xscale("linear")
            axis.set_xlim(100.0, 19.5)
            axis.set_xticks((100, 80, 60, 40, 20))
        axis.set_ylim(0.0, 22000.0)
        axis.set_yticks((0, 5000, 10000, 15000, 20000))
        axis.yaxis.set_major_formatter(FuncFormatter(force_formatter))
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Number of Forces")
        axis.grid(True, which="major", color="#D9D9D6", linewidth=0.4, alpha=0.62)
        axis.set_box_aspect(None)

    label_positions = {
        "i_perp": {
            "ODE-NEB": float(thresholds["i_perp"]) * 0.82,
        },
        "barrier_500_kcal": {
            "ODE-NEB": float(thresholds["barrier_500_kcal"]) - 1.3,
        },
    }
    annotation_rows: List[Dict] = []
    for axis, metric in ((axes[0], "i_perp"), (axes[1], "barrier_500_kcal")):
        ivm_cost = report["first_passage_total_nf"]["IVM"][metric]
        if ivm_cost is None:
            raise ValueError(f"IVM does not reach the common {metric} threshold")
        for baseline in ("ODE-NEB",):
            baseline_cost = report["first_passage_total_nf"][baseline][metric]
            if baseline_cost is None:
                raise ValueError(f"{baseline} does not reach the common {metric} threshold")
            speedup = add_speedup_arrow(
                axis,
                float(thresholds[metric]),
                int(ivm_cost),
                int(baseline_cost),
                label_positions[metric][baseline],
            )
            annotation_rows.append(
                {
                    "metric": metric,
                    "shared_precision_threshold": float(thresholds[metric]),
                    "baseline": baseline,
                    "ivm_number_of_forces": int(ivm_cost),
                    "baseline_number_of_forces": int(baseline_cost),
                    "ivm_speedup_vs_baseline": speedup,
                }
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
    handles = [
        Line2D(
            [0],
            [0],
            label=METHOD_STYLE[method]["label"],
            color=METHOD_STYLE[method]["color"],
            linestyle=METHOD_STYLE[method]["linestyle"],
            linewidth=METHOD_STYLE[method]["linewidth"],
            marker=METHOD_STYLE[method]["marker"],
            markersize=METHOD_STYLE[method]["markersize"],
            markerfacecolor="white",
            markeredgewidth=0.6,
        )
        for method in METHOD_ORDER
    ]
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

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    metadata = {
        "Title": "Chignolin IVM replacement cost-to-precision comparison",
        "Subject": (
            "Fixed arc-length 500-point diagnostics; metric-specific IVM versus "
            "ODE-NEB first passages"
        ),
    }
    for suffix, kwargs in (
        (".pdf", {"format": "pdf", "dpi": 600, "metadata": metadata}),
        (".svg", {"format": "svg"}),
        (".png", {"format": "png", "dpi": 600}),
    ):
        destination = output_dir / f"{stem}{suffix}"
        figure.savefig(destination, **kwargs)
        generated.append(destination.resolve())
    plt.close(figure)
    write_csv(output_dir / f"{stem}_frontier.csv", frontier_rows)
    write_csv(output_dir / f"{stem}_annotations.csv", annotation_rows)
    report["generated_files"] = {
        path.name: {"path": str(path), "sha256": sha256_file(path)} for path in generated
    }
    report["frontier_csv"] = {
        "path": str((output_dir / f"{stem}_frontier.csv").resolve()),
        "sha256": sha256_file(output_dir / f"{stem}_frontier.csv"),
    }
    report["annotation_csv"] = {
        "path": str((output_dir / f"{stem}_annotations.csv").resolve()),
        "sha256": sha256_file(output_dir / f"{stem}_annotations.csv"),
    }
    provenance = output_dir / f"{stem}_provenance.json"
    provenance.write_text(json.dumps(report, indent=2) + "\n")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--arc500", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="chignolin_ir_ivm_overlay_total_nf")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    contract_path, contract = load_contract(args.contract)
    histories = load_raw_histories(contract_path, contract)
    dense, dense_audit = load_arc500(args.arc500, contract_path, contract)
    report = build_report(
        contract_path,
        contract,
        args.arc500.expanduser().resolve(),
        dense_audit,
        histories,
        dense,
    )
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    save_figure(
        args.output_dir.expanduser().resolve(),
        args.stem,
        contract,
        histories,
        dense,
        report,
    )


if __name__ == "__main__":
    main()
