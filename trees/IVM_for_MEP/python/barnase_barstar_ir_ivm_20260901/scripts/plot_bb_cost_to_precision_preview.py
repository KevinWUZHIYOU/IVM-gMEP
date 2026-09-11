#!/usr/bin/env python3
"""Draw the in-progress BB IVM versus ODE--NEB cost-to-precision preview."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from evaluate_discrete_iperp_densified_barrier import (
    EV_TO_KCAL_PER_MOL,
    N_DENSE_POINTS,
    _validate_contract,
    load_eligible_rows,
    sha256_file,
)


IVM_COLOR = "#0072B2"
NEB_COLOR = "#D55E00"


def _read_closed_wal_rows(path: Path) -> dict[int, np.ndarray]:
    records: dict[int, dict[int, float]] = defaultdict(dict)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("IR-IVM energy ledger is empty")
    header = json.loads(lines[0])
    if int(header["number_of_points_per_path"]) != N_DENSE_POINTS:
        raise ValueError("IR-IVM ledger does not use exactly 500 points")
    if int(header["force_requests"]) != 0 or int(header["optimization_N_F_contribution"]) != 0:
        raise ValueError("IR-IVM diagnostic ledger contaminates optimization N_F")
    for line in lines[1:]:
        record = json.loads(line)
        if record.get("status") != "success":
            raise ValueError("IR-IVM diagnostic ledger contains a failed attempt")
        row = int(record["row_index"])
        point = int(record["point_index"])
        if point in records[row]:
            raise ValueError("IR-IVM diagnostic ledger repeats a point")
        records[row][point] = float(record["energy_eV"])
    closed = {}
    expected = set(range(N_DENSE_POINTS))
    for row, points in records.items():
        if set(points) == expected:
            closed[row] = np.asarray([points[index] for index in range(N_DENSE_POINTS)])
    return closed


def _load_ivm(
    contract_path: Path, wal_path: Path, run_label: str | None = None
) -> dict[str, np.ndarray]:
    contract_path = contract_path.expanduser().resolve()
    contract = _validate_contract(contract_path)
    common, common_i, common_nf, committed = load_eligible_rows(contract_path, contract)
    del common
    rows = [
        {
            "label": "common_initial_path",
            "optimization_N_F": int(common_nf),
            "discrete_I_perp": float(common_i),
            "alpha": np.nan,
        },
        *committed,
    ]
    closed = _read_closed_wal_rows(wal_path)
    selected = []
    for row_index in sorted(closed):
        if row_index >= len(rows):
            raise ValueError("IR-IVM energy ledger refers to an unknown path")
        energies = closed[row_index]
        record = rows[row_index]
        if run_label is not None and row_index != 0 and record["label"] != run_label:
            continue
        selected.append(
            {
                "row_index": row_index,
                "label": record["label"],
                "alpha": float(record["alpha"]),
                "optimization_N_F": int(record["optimization_N_F"]),
                "discrete_I_perp": float(record["discrete_I_perp"]),
                "densified_Delta_E_kcal_per_mol": float((np.max(energies) - energies[0]) * EV_TO_KCAL_PER_MOL),
            }
        )
    if not selected:
        raise ValueError("no fully closed IR-IVM 500-point path is available")
    return {
        "number_of_forces": np.asarray([row["optimization_N_F"] for row in selected], dtype=np.int64),
        "i_perp": np.asarray([row["discrete_I_perp"] for row in selected], dtype=np.float64),
        "delta_e": np.asarray([row["densified_Delta_E_kcal_per_mol"] for row in selected], dtype=np.float64),
        "records": selected,
    }


def _load_ivm_result(result_path: Path, run_label: str | None = None) -> dict[str, np.ndarray]:
    """Load a sealed arc500 result without replaying its append-only WAL."""

    result_path = result_path.expanduser().resolve(strict=True)
    audit_path = result_path.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("output_sha256") != sha256_file(result_path):
        raise ValueError("IR-IVM arc500 result failed its audit")
    with np.load(result_path, allow_pickle=False) as saved:
        labels = np.asarray(saved["label"]).astype(str)
        nf = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        i_perp = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
        delta_e = np.asarray(saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64)
        forces = np.asarray(saved["diagnostic_force_requests_per_path"], dtype=np.int64)
        successes = np.asarray(saved["diagnostic_energy_successes_per_path"], dtype=np.int64)
        points = int(np.asarray(saved["number_of_points"]).item())
        optimization_contribution = int(
            np.asarray(saved["optimization_N_F_contribution"]).item()
        )
        alpha = np.asarray(saved["alpha"], dtype=np.float64)
    if points != N_DENSE_POINTS or np.any(successes != N_DENSE_POINTS):
        raise ValueError("IR-IVM result does not bind every path to 500 energies")
    if np.any(forces != 0) or optimization_contribution != 0:
        raise ValueError("IR-IVM diagnostic result contaminates optimization N_F")
    mask = labels == "common_initial_path"
    if run_label is not None:
        mask |= labels == run_label
    else:
        mask[:] = True
    if not np.any(mask & (labels != "common_initial_path")):
        raise ValueError("selected IR-IVM run has no sealed arc500 path")
    records = [
        {
            "row_index": int(index),
            "label": str(labels[index]),
            "alpha": None if not np.isfinite(alpha[index]) else float(alpha[index]),
            "optimization_N_F": int(nf[index]),
            "discrete_I_perp": float(i_perp[index]),
            "densified_Delta_E_kcal_per_mol": float(delta_e[index]),
        }
        for index in np.flatnonzero(mask)
    ]
    return {
        "number_of_forces": nf[mask],
        "i_perp": i_perp[mask],
        "delta_e": delta_e[mask],
        "records": records,
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "audit": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
    }


def _load_ode(result_paths: list[Path]) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Merge disjoint, fully audited ODE--NEB arc500 result files."""

    rows: list[tuple[int, float, float]] = []
    bindings: list[dict] = []
    for supplied_path in result_paths:
        result_path = supplied_path.expanduser().resolve(strict=True)
        audit_path = result_path.with_suffix(".audit.json")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("status") != "PASS"
            or audit.get("output_sha256") != sha256_file(result_path)
        ):
            raise ValueError("ODE--NEB arc500 result failed its audit")
        with np.load(result_path, allow_pickle=False) as saved:
            nf = np.asarray(saved["optimization_N_F"], dtype=np.int64)
            i_perp = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
            delta_e = np.asarray(
                saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64
            )
            points = int(np.asarray(saved["number_of_points"]).item())
            force = np.asarray(
                saved["diagnostic_force_requests_per_path"], dtype=np.int64
            )
        n_rows = int(nf.size)
        if not (
            n_rows > 0
            and i_perp.shape == nf.shape
            and delta_e.shape == nf.shape
            and force.shape == nf.shape
            and points == N_DENSE_POINTS
            and np.all(force == 0)
        ):
            raise ValueError("ODE--NEB result protocol metadata is inconsistent")
        expected_energy_calls = n_rows * N_DENSE_POINTS
        if (
            int(audit["diagnostic_energy_attempts"]) != expected_energy_calls
            or int(audit["diagnostic_energy_successes"]) != expected_energy_calls
        ):
            raise ValueError("ODE--NEB arc500 ledger is incomplete")
        if (
            int(audit["diagnostic_force_requests"]) != 0
            or int(audit["optimization_N_F_contribution"]) != 0
        ):
            raise ValueError("ODE--NEB diagnostic calls contaminate optimization N_F")
        rows.extend(
            (int(cost), float(residual), float(barrier))
            for cost, residual, barrier in zip(nf, i_perp, delta_e)
        )
        bindings.append(
            {
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
                "audit": str(audit_path),
                "audit_sha256": sha256_file(audit_path),
                "number_of_paths": n_rows,
                "diagnostic_energy_successes": expected_energy_calls,
            }
        )
    costs = [row[0] for row in rows]
    if len(set(costs)) != len(costs):
        raise ValueError("ODE--NEB result files contain duplicate optimization N_F")
    rows.sort(key=lambda row: row[0])
    return {
        "number_of_forces": np.asarray([row[0] for row in rows], dtype=np.int64),
        "i_perp": np.asarray([row[1] for row in rows], dtype=np.float64),
        "delta_e": np.asarray([row[2] for row in rows], dtype=np.float64),
    }, bindings


def _frontier(nf: np.ndarray, metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(nf, kind="stable")
    costs = np.asarray(nf, dtype=np.int64)[order]
    values = np.asarray(metric, dtype=np.float64)[order]
    unique_costs = np.unique(costs)
    best_at_cost = np.asarray([np.min(values[costs == cost]) for cost in unique_costs])
    out_cost, out_value = [], []
    best = np.inf
    for cost, value in zip(unique_costs, best_at_cost):
        if np.isfinite(value) and value < best:
            best = float(value)
            out_cost.append(int(cost))
            out_value.append(best)
    return np.asarray(out_value), np.asarray(out_cost)


def _first_passage(nf: np.ndarray, metric: np.ndarray, threshold: float) -> int | None:
    order = np.argsort(nf, kind="stable")
    costs = np.asarray(nf, dtype=np.int64)[order]
    values = np.asarray(metric, dtype=np.float64)[order]
    best = np.inf
    for cost in np.unique(costs):
        best = min(best, float(np.min(values[costs == cost])))
        if best <= threshold:
            return int(cost)
    return None


def _force_formatter(value: float, _position: int) -> str:
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def _setup_style(mpl) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "it",
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _add_arrow(axis, threshold: float, ivm_cost: int, neb_cost: int) -> float:
    ratio = float(neb_cost) / float(ivm_cost)
    axis.axvline(threshold, color="#767676", linewidth=0.55, linestyle=":", zorder=1)
    axis.annotate(
        "",
        xy=(threshold, ivm_cost),
        xytext=(threshold, neb_cost),
        arrowprops={"arrowstyle": "<->", "color": "#202020", "lw": 0.8},
        zorder=25,
    )
    axis.annotate(
        rf"{ratio:.1f}$\times$",
        xy=(threshold, 0.5 * (ivm_cost + neb_cost)),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color="#202020",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.8},
        zorder=26,
    )
    return ratio


def plot(
    ivm: dict, ode: dict, output: Path, preview_note: str | None = None
) -> dict:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    _setup_style(mpl)
    figure = plt.figure(figsize=(180 / 25.4, 72 / 25.4), dpi=220)
    # Keep a wider outer margin on the right so the multiplier remains to the
    # right of a near-boundary vertical arrow without being clipped.
    grid = GridSpec(1, 2, figure=figure, left=0.072, right=0.935, bottom=0.18, top=0.78, wspace=0.25)
    axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    report = {"thresholds": {}, "first_passage_number_of_forces": {}, "ODE_NEB_over_IVM": {}}
    for axis, key, xlabel, xscale in (
        (axes[0], "i_perp", r"$I_\perp$", "log"),
        (axes[1], "delta_e", r"$\Delta E$ (kcal mol$^{-1}$)", "linear"),
    ):
        ivm_x, ivm_y = _frontier(ivm["number_of_forces"], ivm[key])
        ode_x, ode_y = _frontier(ode["number_of_forces"], ode[key])
        axis.plot(ivm_x, ivm_y, color=IVM_COLOR, linestyle="-", linewidth=1.85, marker="o", markersize=3.3, markerfacecolor="white", markeredgewidth=0.6, zorder=20)
        axis.plot(ode_x, ode_y, color=NEB_COLOR, linestyle="--", linewidth=1.65, marker="s", markersize=3.1, markerfacecolor="white", markeredgewidth=0.6, zorder=6)
        threshold = max(float(np.min(ivm_x)), float(np.min(ode_x)))
        ivm_cost = _first_passage(ivm["number_of_forces"], ivm[key], threshold)
        ode_cost = _first_passage(ode["number_of_forces"], ode[key], threshold)
        if ivm_cost is None or ode_cost is None:
            raise ValueError(f"both methods must reach the common {key} threshold")
        ratio = _add_arrow(axis, threshold, ivm_cost, ode_cost)
        report["thresholds"][key] = threshold
        report["first_passage_number_of_forces"][key] = {"IVM": ivm_cost, "ODE-NEB": ode_cost}
        report["ODE_NEB_over_IVM"][key] = ratio
        all_x = np.concatenate((ivm_x, ode_x))
        if xscale == "log":
            axis.set_xscale("log")
            axis.set_xlim(float(np.max(all_x)) * 1.45, float(np.min(all_x)) / 1.45)
        else:
            span = max(float(np.ptp(all_x)), 1.0)
            axis.set_xlim(float(np.max(all_x)) + 0.06 * span, max(0.0, float(np.min(all_x)) - 0.06 * span))
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Number of Forces")
        axis.grid(True, which="major", color="#D9D9D6", linewidth=0.4, alpha=0.62)
        axis.yaxis.set_major_formatter(FuncFormatter(_force_formatter))
    ymax = max(11000.0, 1.08 * max(np.max(ivm["number_of_forces"]), np.max(ode["number_of_forces"])))
    for axis in axes:
        axis.set_ylim(0.0, ymax)
    figure.canvas.draw()
    for axis, letter in zip(axes, "ab"):
        position = axis.get_position()
        figure.text(position.x0 - 0.03, position.y1 + 0.012, letter, ha="left", va="bottom", fontsize=8.0, fontweight="bold")
    handles = [
        Line2D([0], [0], label="IVM", color=IVM_COLOR, linestyle="-", linewidth=1.85, marker="o", markersize=3.3, markerfacecolor="white", markeredgewidth=0.6),
        Line2D([0], [0], label="ODE-NEB", color=NEB_COLOR, linestyle="--", linewidth=1.65, marker="s", markersize=3.1, markerfacecolor="white", markeredgewidth=0.6),
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, columnspacing=1.5, handlelength=2.35, handletextpad=0.5, frameon=False)
    figure.text(0.975, 0.965, "in-progress preview", ha="right", va="top", fontsize=6.5, color="#666666")
    if preview_note:
        figure.text(
            0.5,
            0.845,
            preview_note,
            ha="center",
            va="top",
            fontsize=6.35,
            color="#4D4D4D",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="pdf",
        dpi=600,
        metadata={
            "Title": "Barnase--Barstar in-progress IVM versus ODE-NEB cost-to-precision preview",
            "Subject": "Discrete I_perp and equal-Cartesian-arclength 500-point Delta E; only fully closed diagnostic paths",
        },
    )
    plt.close(figure)
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ivm-contract", type=Path)
    parser.add_argument("--ivm-wal", type=Path)
    parser.add_argument("--ivm-result", type=Path)
    parser.add_argument(
        "--ivm-run-label",
        help="plot only this IR-IVM run plus the common initial path",
    )
    parser.add_argument(
        "--ode-result",
        type=Path,
        action="append",
        required=True,
        help="repeat for each disjoint audited ODE--NEB arc500 result",
    )
    parser.add_argument("--preview-note")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.ivm_result is not None:
        if args.ivm_contract is not None or args.ivm_wal is not None:
            raise ValueError("use either --ivm-result or the contract/WAL pair")
        ivm = _load_ivm_result(args.ivm_result, args.ivm_run_label)
    else:
        if args.ivm_contract is None or args.ivm_wal is None:
            raise ValueError("the IR-IVM contract and WAL must be provided together")
        ivm = _load_ivm(args.ivm_contract, args.ivm_wal, args.ivm_run_label)
    ode, ode_bindings = _load_ode(args.ode_result)
    output = args.output.expanduser().resolve()
    report = plot(ivm, ode, output, args.preview_note)
    provenance = {
        "schema": "bb-cost-to-precision-in-progress-preview-v2",
        "status": "IN_PROGRESS_PREVIEW_NOT_A_FINAL_PARAMETER_SELECTION",
        "figure": str(output),
        "figure_sha256": sha256_file(output),
        "IVM_legend_policy": "Frozen Chignolin-style IR-IVM is displayed as IVM.",
        "IVM_path_policy": (
            "Only fully closed 500-point rows from the selected run plus the common "
            "initial path; no interpolation or extrapolation beyond observed states."
        ),
        "IVM_selected_run_label": args.ivm_run_label,
        "preview_note": args.preview_note,
        "ODE_NEB_path_policy": (
            "Only explicitly selected official optimization milestones; every path "
            "has a complete audited 500-point diagnostic energy ledger."
        ),
        "IR_IVM_closed_records": ivm["records"],
        "ODE_NEB_bindings": ode_bindings,
        "diagnostic_energy_calls_excluded_from_optimization_axis": True,
        **report,
    }
    if args.ivm_result is not None:
        provenance.update(
            {
                "IR_IVM_result": ivm["result"],
                "IR_IVM_result_sha256": ivm["result_sha256"],
                "IR_IVM_audit": ivm["audit"],
                "IR_IVM_audit_sha256": ivm["audit_sha256"],
            }
        )
    else:
        provenance.update(
            {
                "IR_IVM_contract": str(args.ivm_contract.expanduser().resolve()),
                "IR_IVM_contract_sha256": sha256_file(args.ivm_contract.expanduser().resolve()),
                "IR_IVM_energy_wal": str(args.ivm_wal.expanduser().resolve()),
                "IR_IVM_energy_wal_sha256_at_plot_time": sha256_file(args.ivm_wal.expanduser().resolve()),
            }
        )
    provenance_path = output.with_name(output.stem + "_provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
