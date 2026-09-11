#!/usr/bin/env python3
"""Plot attained Ala inner-solve residual ratios against cumulative force cost.

The prescribed stopping parameter is fixed at ``r=0.1``.  The plotted
quantity is instead the actually attained global natural-residual ratio
``R_end/R_0`` at the end of each fixed-path discrete solve.  The runtime does
not persist this scalar at every inner step, so this script deliberately plots
only audited solve endpoints.

The cumulative Number of Forces includes every successful cache miss for the
20 internal images.  Consequently, horizontal gaps between solve endpoints
also include charged reparameterization and recovery work.  No residual is
invented for those gaps: the figure uses endpoint markers only, with no
interpolation, extrapolation, or carry-forward.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np


REPO = Path(__file__).resolve().parents[5]
STUDY = REPO / "python/alanine_ivm_clean_four_level_20260801"
REPORTS = STUDY / "reports"
CAMPAIGN_STEM = (
    "stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign_v1_"
    "1788923892307558000"
)
PREREGISTRATION = REPORTS / f"{CAMPAIGN_STEM}_preregistered.json"
LOCAL_REPORT = REPORTS / f"{CAMPAIGN_STEM}_local_report.json"
AGGREGATE = REPORTS / f"{CAMPAIGN_STEM}_aggregate.json"
EXPECTED_INPUT_SHA256 = {
    PREREGISTRATION: (
        "da1ded53f6dc11ac71ebe5bae64ba452f19c95fdc60467be220ff630912b6a05"
    ),
    LOCAL_REPORT: (
        "78ef8b82b7b7a8dbc2b6f4f62d8728fb216c1494fce46a95bbddce5241592362"
    ),
    AGGREGATE: (
        "50914e1cdfdafc59287921b4eef6081a5274d40366a4da926a590c06a6c9828a"
    ),
}
EXPECTED_RESULT_SHA256 = {
    0.1: "42127a3c24baca8b747f720082843f7af939858a835539979c3315458c3da698",
    1.0: "2662ba91186ce9a8b4261447e0349bbbddb726d92a664d1b76336228582cf4b2",
    10.0: "173ae496752f30d9c73f96b92355a41c931d8f49a36bf1f55bc745a44c1ad23c",
    100.0: "13faae96c9e01083e3ded7f6904fe7e756e7eb38cf9a8fdd9bfc3d6226a2110a",
    1000.0: "9c86d50d9f24c35198b94fcc512a197659556cbddd890df4ac42cb4ac086c380",
    10000.0: "c4dea9f35654f535301dd7af11402baf2bba06fe46a605398320b9f97bec7b7d",
    100000.0: "c893ff42d7943572a44136896c9a6c4928ce0fe71dc4bb53f360881c42d13af7",
    1000000.0: "339dadd0ff7eed1da1f355e18b0e1b722977e9b76dc3ef9af6f1961074d9d6d3",
}
EXPECTED_PROTOCOL_ID = (
    "stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign_v1"
)
ALPHAS = tuple(10.0**exponent for exponent in range(-1, 7))
ALPHA_TEXT = tuple(f"1e{exponent}" for exponent in range(-1, 7))
RESIDUAL_THRESHOLD = 0.1
TOTAL_NF_BUDGET = 5000
N_INTERNAL = 20
INNER_MAX_STEPS = 5000
OUTPUT_DIR = REPO / "output/pdf"
STEM = "alanine66d_r01_reparam_inner_residual_vs_nf"

# Exactly the discrete jet palette and alpha-marker order used by the 2x3
# fixed-alpha comparison.
ALPHA_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
ALPHA_COLORS = tuple(
    mpl.colormaps["jet"].resampled(len(ALPHAS))(index)
    for index in range(len(ALPHAS))
)
NF_TICKS = (20, 50, 100, 200, 500, 1000, 2000, 5000)
XLIM = (18.0, 6500.0)
YLIM = (1.5e-2, 1.0e2)
RESIDUAL_STOP = "evaluated-relative-natural-residual"
CALL_LIMIT_STOP = "per-image-new-call-limit"


class FigureInputError(RuntimeError):
    """The registered campaign or endpoint alignment failed validation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the frozen campaign and endpoint alignment without outputs.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FigureInputError(f"Cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise FigureInputError(f"Expected JSON object: {path}")
    return value


def scalar(saved: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in saved.files:
        raise FigureInputError(f"Result lacks required field {key!r}")
    value = np.asarray(saved[key])
    if value.size != 1:
        raise FigureInputError(f"Result field {key!r} is not scalar")
    return value.reshape(()).item()


def same_number(left: Any, right: Any, atol: float = 1.0e-14) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def alpha_label(alpha: float) -> str:
    exponent = int(round(math.log10(alpha)))
    return rf"$10^{{{exponent}}}$"


def validate_frozen_campaign() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path, expected_hash in EXPECTED_INPUT_SHA256.items():
        if not path.is_file():
            raise FigureInputError(f"Frozen campaign input is missing: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise FigureInputError(
                f"Frozen campaign input hash changed: {path}\n"
                f"expected {expected_hash}, observed {actual_hash}"
            )

    preregistration = load_json(PREREGISTRATION)
    local_report = load_json(LOCAL_REPORT)
    aggregate = load_json(AGGREGATE)
    prereg_hash = EXPECTED_INPUT_SHA256[PREREGISTRATION]
    local_hash = EXPECTED_INPUT_SHA256[LOCAL_REPORT]

    if (
        preregistration.get("status") != "preregistered"
        or preregistration.get("protocol_id") != EXPECTED_PROTOCOL_ID
        or preregistration.get("family") != "inexact_reparam"
        or int(preregistration.get("cell_count", -1)) != len(ALPHAS)
        or list(preregistration.get("alpha_grid", [])) != list(ALPHA_TEXT)
    ):
        raise FigureInputError("Optimizer preregistration identity changed")
    controls = preregistration.get("fixed_controls")
    if not isinstance(controls, Mapping) or not (
        same_number(controls.get("residual_ratio"), RESIDUAL_THRESHOLD)
        and int(controls.get("inner_max_steps", -1)) == INNER_MAX_STEPS
        and int(controls.get("total_internal_nf_budget", -1)) == TOTAL_NF_BUDGET
        and controls.get("reparameterization")
        == "state_curve/per_image every outer proposal"
        and controls.get("complete_state_definition")
        == "initial_or_reparam_committed"
    ):
        raise FigureInputError("Optimizer preregistered controls changed")

    planned = preregistration.get("planned_outputs")
    if not isinstance(planned, Mapping) or (
        Path(str(planned.get("local_report", ""))).resolve() != LOCAL_REPORT
        or Path(str(planned.get("aggregate", ""))).resolve() != AGGREGATE
    ):
        raise FigureInputError("Optimizer preregistration routes other outputs")

    if not (
        local_report.get("status") == "complete"
        and local_report.get("protocol_id") == EXPECTED_PROTOCOL_ID
        and Path(str(local_report.get("preregistration", ""))).resolve()
        == PREREGISTRATION
        and local_report.get("preregistration_sha256") == prereg_hash
        and int(local_report.get("jobs", -1)) == 1
    ):
        raise FigureInputError("Optimizer local report identity changed")

    if not (
        aggregate.get("status") == "complete"
        and aggregate.get("protocol_id") == EXPECTED_PROTOCOL_ID
        and Path(str(aggregate.get("preregistration", ""))).resolve()
        == PREREGISTRATION
        and aggregate.get("preregistration_sha256") == prereg_hash
        and Path(str(aggregate.get("local_report", ""))).resolve() == LOCAL_REPORT
        and aggregate.get("local_report_sha256") == local_hash
        and int(aggregate.get("cell_count", -1)) == len(ALPHAS)
        and int(aggregate.get("unique_cell_count", -1)) == len(ALPHAS)
        and aggregate.get("all_inner_caps_nonbinding") is True
    ):
        raise FigureInputError("Optimizer aggregate identity changed")
    return preregistration, local_report, aggregate


def required_array(
    saved: np.lib.npyio.NpzFile,
    key: str,
    *,
    dtype: Any,
    ndim: int = 1,
) -> np.ndarray:
    if key not in saved.files:
        raise FigureInputError(f"Result lacks required field {key!r}")
    value = np.asarray(saved[key], dtype=dtype)
    if value.ndim != ndim:
        raise FigureInputError(f"Result field {key!r} has wrong rank")
    return value


def validate_result_identity(saved: np.lib.npyio.NpzFile, alpha: float) -> None:
    expected = {
        "alpha": alpha,
        "alpha_mode": "fixed",
        "solver": "state_ivm",
        "frozen_subproblem": "discrete_b040",
        "discrete_residual_ratio": RESIDUAL_THRESHOLD,
        "discrete_inner_max_steps": INNER_MAX_STEPS,
        "discrete_stop_mode": "r08c10",
        "reparam_mode": "state_curve",
        "reparam_every": 1,
        "state_curve_direct_solve": True,
        "state_curve_direct_objective_acceptance": "per_image",
        "total_internal_nf_budget": TOTAL_NF_BUDGET,
        "total_internal_nf_consumed": TOTAL_NF_BUDGET,
        "total_internal_nf_budget_no_overshoot": True,
        "total_internal_nf_endpoint_evaluations_excluded": True,
        "total_internal_nf_complete_state_definition": "initial_or_reparam_committed",
        "discrete_ffrbk_selector_residual_ratio": RESIDUAL_THRESHOLD,
        "discrete_ffrbk_selector_inner_max_steps": INNER_MAX_STEPS,
        "discrete_ffrbk_selector_mode": "one_way_takeover",
        "discrete_step_multiplier_eta": 1.1,
    }
    for key, expected_value in expected.items():
        observed = scalar(saved, key)
        if isinstance(expected_value, str):
            matched = str(observed) == expected_value
        elif isinstance(expected_value, bool):
            matched = bool(observed) is expected_value
        else:
            matched = same_number(observed, expected_value)
        if not matched:
            raise FigureInputError(
                f"Result identity mismatch for alpha={alpha:g}, {key}: "
                f"{observed!r} != {expected_value!r}"
            )


def align_solve_endpoints(
    saved: np.lib.npyio.NpzFile,
    alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map one scalar residual audit per solve to its exact saved NoF snapshot.

    ``discrete_*_audit_solver_call`` has one entry per committed inner joint
    step, whereas ``discrete_audit_global_relative_residual`` has one entry per
    completed fixed-path solve.  Non-reparameterization history snapshots are
    the initial state followed by the paid primary-inner steps.  At most one
    additional non-reparameterization snapshot may close the terminal ledger
    after a budget interruption; it must be the final history record.
    """

    validate_result_identity(saved, alpha)
    residuals = required_array(
        saved, "discrete_audit_global_relative_residual", dtype=float
    )
    reasons = required_array(saved, "discrete_audit_stop_reason", dtype="U64")
    solver_calls = required_array(
        saved, "discrete_ffrbk_selector_audit_solver_call", dtype=int
    )
    outer_calls = required_array(saved, "discrete_audit_outer", dtype=int)
    inner_steps = required_array(saved, "discrete_audit_step", dtype=int)
    force_cost = required_array(
        saved, "total_internal_joint_evaluations", dtype=int
    )
    attempted = required_array(saved, "reparam_attempted", dtype=bool)
    committed = required_array(saved, "reparam_committed", dtype=bool)
    counters = required_array(saved, "counter_vectors", dtype=int, ndim=2)

    history_lengths = {len(force_cost), len(attempted), len(committed), len(counters)}
    if len(history_lengths) != 1 or not len(force_cost):
        raise FigureInputError(f"History arrays are misaligned for alpha={alpha:g}")
    if counters.shape[1] != N_INTERNAL + 2:
        raise FigureInputError(f"Counter-vector width changed for alpha={alpha:g}")
    if not np.array_equal(force_cost, np.sum(counters[:, 1:-1], axis=1)):
        raise FigureInputError(f"NoF differs from internal counter sum for alpha={alpha:g}")
    if np.any(np.diff(force_cost) <= 0) or int(force_cost[-1]) != TOTAL_NF_BUDGET:
        raise FigureInputError(f"NoF ledger is not strictly increasing to 5000 for alpha={alpha:g}")
    if len(residuals) != len(reasons) or not len(residuals):
        raise FigureInputError(f"Solve-level residual arrays differ for alpha={alpha:g}")
    fixed_solve_count = int(scalar(saved, "discrete_ffrbk_selector_fixed_solver_calls"))
    if fixed_solve_count != len(residuals):
        raise FigureInputError(f"Fixed-solve count differs from residual count for alpha={alpha:g}")
    if not (
        solver_calls.shape == outer_calls.shape == inner_steps.shape
        and np.array_equal(solver_calls, outer_calls)
    ):
        raise FigureInputError(f"Inner-step audit identities differ for alpha={alpha:g}")
    if not len(solver_calls):
        raise FigureInputError(f"No inner-step audit exists for alpha={alpha:g}")
    expected_solver_ids = np.arange(len(residuals), dtype=int)
    observed_solver_ids = np.unique(solver_calls)
    if not np.array_equal(observed_solver_ids, expected_solver_ids):
        raise FigureInputError(f"Solve IDs are not contiguous for alpha={alpha:g}")

    steps_per_solve = np.bincount(solver_calls, minlength=len(residuals))
    if np.any(steps_per_solve <= 0):
        raise FigureInputError(f"A solve lacks an audited inner step for alpha={alpha:g}")
    for solve_index, step_count in enumerate(steps_per_solve):
        observed_steps = inner_steps[solver_calls == solve_index]
        if not np.array_equal(observed_steps, np.arange(step_count, dtype=int)):
            raise FigureInputError(
                f"Inner-step numbering changed for alpha={alpha:g}, solve={solve_index}"
            )

    non_reparam_indices = np.flatnonzero(~attempted)
    if not len(non_reparam_indices) or int(non_reparam_indices[0]) != 0:
        raise FigureInputError(f"The initial state is not the first ledger point for alpha={alpha:g}")
    if int(force_cost[0]) != N_INTERNAL:
        raise FigureInputError(f"Initial internal force cost is not 20 for alpha={alpha:g}")
    required_non_reparam = 1 + int(np.sum(steps_per_solve))
    closure_count = len(non_reparam_indices) - required_non_reparam
    if closure_count not in (0, 1):
        raise FigureInputError(
            f"Expected at most one terminal ledger closure for alpha={alpha:g}; "
            f"observed {closure_count}"
        )
    if closure_count == 1 and int(non_reparam_indices[-1]) != len(force_cost) - 1:
        raise FigureInputError(
            f"The extra non-reparameterization record is not terminal for alpha={alpha:g}"
        )

    step_history_indices = non_reparam_indices[1:required_non_reparam]
    if len(step_history_indices) != int(np.sum(steps_per_solve)):
        raise FigureInputError(f"Inner-step ledger is incomplete for alpha={alpha:g}")
    cumulative_steps = np.cumsum(steps_per_solve)
    endpoint_indices = step_history_indices[cumulative_steps - 1]
    start_offsets = np.r_[0, cumulative_steps[:-1]]
    first_step_indices = step_history_indices[start_offsets]
    if np.any(attempted[endpoint_indices]):
        raise FigureInputError(f"A solve endpoint is marked as reparameterization for alpha={alpha:g}")
    endpoint_nf = force_cost[endpoint_indices]
    if np.any(np.diff(endpoint_nf) <= 0):
        raise FigureInputError(f"Solve endpoint NoF is not increasing for alpha={alpha:g}")

    permitted_reasons = {RESIDUAL_STOP, CALL_LIMIT_STOP}
    if not set(reasons.tolist()).issubset(permitted_reasons):
        raise FigureInputError(f"Unexpected stop reason for alpha={alpha:g}: {set(reasons)}")
    satisfied = reasons == RESIDUAL_STOP
    if np.any(residuals[satisfied] > RESIDUAL_THRESHOLD + 1.0e-12):
        raise FigureInputError(f"A residual-stop endpoint exceeds r for alpha={alpha:g}")
    if np.any(residuals[~satisfied] <= RESIDUAL_THRESHOLD):
        raise FigureInputError(f"A call-limit endpoint already satisfies r for alpha={alpha:g}")
    call_limit_indices = np.flatnonzero(~satisfied)
    if len(call_limit_indices) > 1 or (
        len(call_limit_indices) == 1 and int(call_limit_indices[0]) != len(residuals) - 1
    ):
        raise FigureInputError(f"Call-limit stop is not a unique final solve for alpha={alpha:g}")
    cap_reasons = reasons == "evaluated-inner-max-steps"
    if np.any(cap_reasons):
        raise FigureInputError(f"Implementation cap bound a solve for alpha={alpha:g}")

    rows: list[dict[str, Any]] = []
    previous_endpoint = int(force_cost[0])
    for solve_index in range(len(residuals)):
        endpoint_index = int(endpoint_indices[solve_index])
        first_step_index = int(first_step_indices[solve_index])
        start_nf = int(force_cost[first_step_index - 1])
        endpoint = int(force_cost[endpoint_index])
        preceding_non_inner = start_nf - previous_endpoint
        if preceding_non_inner < 0:
            raise FigureInputError(f"Negative between-solve NoF for alpha={alpha:g}")
        rows.append(
            {
                "alpha": f"{alpha:g}",
                "solve_index": solve_index,
                "solve_start_NF": start_nf,
                "solve_endpoint_NF": endpoint,
                "inner_solve_NF": endpoint - start_nf,
                "preceding_reparameterization_or_recovery_NF": preceding_non_inner,
                "joint_steps": int(steps_per_solve[solve_index]),
                "achieved_global_relative_natural_residual": float(
                    residuals[solve_index]
                ),
                "prescribed_residual_ratio_r": RESIDUAL_THRESHOLD,
                "stop_reason": str(reasons[solve_index]),
                "residual_threshold_satisfied": bool(satisfied[solve_index]),
                "history_endpoint_index": endpoint_index,
                "initial_path_NF": int(force_cost[0]),
                "terminal_total_NF": int(force_cost[-1]),
                "terminal_ledger_closure_present": bool(closure_count),
                "interpolation_used": False,
                "extrapolation_used": False,
                "carry_forward_used": False,
            }
        )
        previous_endpoint = endpoint

    summary = {
        "alpha": alpha,
        "solve_endpoint_count": len(rows),
        "inner_joint_step_count": int(np.sum(steps_per_solve)),
        "residual_stop_count": int(np.count_nonzero(satisfied)),
        "call_limit_stop_count": int(np.count_nonzero(~satisfied)),
        "first_endpoint_NF": int(endpoint_nf[0]),
        "last_endpoint_NF": int(endpoint_nf[-1]),
        "minimum_achieved_ratio": float(np.min(residuals)),
        "maximum_achieved_ratio": float(np.max(residuals)),
        "terminal_ledger_closure_count": int(closure_count),
        "terminal_charged_after_last_solve_NF": int(force_cost[-1] - endpoint_nf[-1]),
        "last_complete_outer_state_NF": int(
            scalar(saved, "total_internal_nf_last_complete_state_nf")
        ),
        "result_fields": {
            "relative_residual": "discrete_audit_global_relative_residual",
            "stop_reason": "discrete_audit_stop_reason",
            "inner_step_solve_id": "discrete_ffrbk_selector_audit_solver_call",
            "history_cost": "total_internal_joint_evaluations",
            "reparameterization_flag": "reparam_attempted",
        },
    }
    return rows, summary


def load_endpoint_data(
    local_report: Mapping[str, Any],
) -> tuple[dict[float, list[dict[str, Any]]], list[dict[str, Any]]]:
    report_rows = local_report.get("rows")
    if not isinstance(report_rows, list) or len(report_rows) != len(ALPHAS):
        raise FigureInputError("Local report does not contain exactly eight runs")
    by_alpha: dict[float, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    observed_order: list[float] = []
    for report_row in report_rows:
        if not isinstance(report_row, Mapping):
            raise FigureInputError("Malformed optimizer local-report row")
        alpha = float(report_row.get("alpha", math.nan))
        observed_order.append(alpha)
        expected_hash = EXPECTED_RESULT_SHA256.get(alpha)
        result_path = Path(str(report_row.get("result", ""))).resolve()
        if expected_hash is None or report_row.get("result_sha256") != expected_hash:
            raise FigureInputError(f"Unregistered optimizer result for alpha={alpha:g}")
        if not result_path.is_file() or sha256(result_path) != expected_hash:
            raise FigureInputError(f"Optimizer result hash changed: {result_path}")
        if int(report_row.get("inner_cap_stop_count", -1)) != 0:
            raise FigureInputError(f"Inner cap bound the registered run alpha={alpha:g}")
        with np.load(result_path, allow_pickle=False) as saved:
            rows, summary = align_solve_endpoints(saved, alpha)
        for row in rows:
            row["result_file"] = str(result_path)
            row["result_sha256"] = expected_hash
        summary["result_file"] = str(result_path)
        summary["result_sha256"] = expected_hash
        by_alpha[alpha] = rows
        summaries.append(summary)
    if tuple(observed_order) != ALPHAS or set(by_alpha) != set(ALPHAS):
        raise FigureInputError("Optimizer results are not in the registered alpha order")
    return by_alpha, summaries


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def format_nf(value: float, _: int) -> str:
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def draw_figure(data: Mapping[float, Sequence[Mapping[str, Any]]], output: Path) -> None:
    configure_style()
    figure, axis = plt.subplots(figsize=(178 / 25.4, 102 / 25.4))
    for index, alpha in enumerate(ALPHAS):
        rows = data[alpha]
        successful = [row for row in rows if row["residual_threshold_satisfied"]]
        limited = [row for row in rows if not row["residual_threshold_satisfied"]]
        if successful:
            axis.scatter(
                [row["solve_endpoint_NF"] for row in successful],
                [row["achieved_global_relative_natural_residual"] for row in successful],
                s=13.0,
                marker=ALPHA_MARKERS[index],
                facecolor=ALPHA_COLORS[index],
                edgecolor="#202020",
                linewidth=0.25,
                zorder=3,
            )
        if limited:
            axis.scatter(
                [row["solve_endpoint_NF"] for row in limited],
                [row["achieved_global_relative_natural_residual"] for row in limited],
                s=47.0,
                marker="X",
                facecolor="none",
                edgecolor=ALPHA_COLORS[index],
                linewidth=1.15,
                zorder=5,
            )

    axis.axhline(
        RESIDUAL_THRESHOLD,
        color="#3f3f3f",
        linestyle=(0, (2.0, 2.0)),
        linewidth=0.85,
        zorder=1,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(*XLIM)
    axis.set_ylim(*YLIM)
    axis.set_xticks(NF_TICKS)
    axis.xaxis.set_major_formatter(FuncFormatter(format_nf))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.grid(which="major", color="#d8d8d8", linewidth=0.45, alpha=0.65)
    axis.grid(which="minor", axis="y", color="#e7e7e7", linewidth=0.3, alpha=0.55)
    axis.set_xlabel("Number of Forces")
    axis.set_ylabel(r"Achieved inner residual ratio $R_k^{\mathrm{end}}/R_k^0$")
    axis.set_title(
        r"With reparameterization ($r=0.1$, $m_{\max}=5000$; cap nonbinding)",
        pad=7,
    )

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=ALPHA_MARKERS[index],
            markersize=4.2,
            markerfacecolor=ALPHA_COLORS[index],
            markeredgecolor="#202020",
            markeredgewidth=0.3,
            label=alpha_label(alpha),
        )
        for index, alpha in enumerate(ALPHAS)
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#3f3f3f",
                linestyle=(0, (2.0, 2.0)),
                linewidth=0.85,
                label=r"prescribed $r=0.1$",
            ),
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="X",
                markersize=5.4,
                markerfacecolor="none",
                markeredgecolor="#343434",
                markeredgewidth=1.0,
                label="call-limit stop",
            ),
        ]
    )
    figure.legend(
        handles=handles,
        title=r"Fixed proximal parameter $\alpha$",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        handlelength=1.55,
        handletextpad=0.42,
        columnspacing=1.0,
    )
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.14, top=0.72)
    figure.savefig(
        output,
        format="pdf",
        dpi=600,
        metadata={
            "Title": "Alanine 66D attained inner residual ratio versus force cost",
            "Subject": (
                "Fixed-path solve endpoints for r=0.1 with reparameterization; "
                "all internal force costs included"
            ),
        },
    )
    plt.close(figure)


def write_source_csv(path: Path, data: Mapping[float, Sequence[Mapping[str, Any]]]) -> None:
    rows = [dict(row) for alpha in ALPHAS for row in data[alpha]]
    if not rows:
        raise FigureInputError("No endpoint rows were produced")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    _, local_report, _ = validate_frozen_campaign()
    endpoint_data, summaries = load_endpoint_data(local_report)
    validation_summary = {
        "status": "validated",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "alpha_grid": [f"{alpha:g}" for alpha in ALPHAS],
        "prescribed_residual_ratio": RESIDUAL_THRESHOLD,
        "total_endpoint_count": sum(item["solve_endpoint_count"] for item in summaries),
        "endpoint_summaries": summaries,
    }
    args = parse_args()
    if args.validate_only:
        print(json.dumps(validation_summary, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / f"{STEM}.pdf"
    csv_path = args.output_dir / f"{STEM}_source.csv"
    manifest_path = args.output_dir / f"{STEM}_manifest.json"
    existing = [path for path in (pdf_path, csv_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    draw_figure(endpoint_data, pdf_path)
    write_source_csv(csv_path, endpoint_data)
    manifest = {
        **validation_summary,
        "status": "generated_from_frozen_registered_optimizer_results",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quantity": {
            "prescribed_parameter": "r=0.1 (constant horizontal reference)",
            "plotted_value": (
                "global natural residual at the fixed-path solve endpoint divided "
                "by that solve's initial global natural residual"
            ),
            "source_field": "discrete_audit_global_relative_residual",
            "available_resolution": "one scalar per completed fixed-path solve",
            "inner_step_residuals_available": False,
        },
        "endpoint_alignment": {
            "inner_step_solve_id": "discrete_ffrbk_selector_audit_solver_call",
            "equivalent_outer_id_checked": "discrete_audit_outer",
            "inner_step_index": "discrete_audit_step",
            "cost_field": "total_internal_joint_evaluations",
            "initial_history_record": "NF=20; no solve residual attached",
            "primary_history_selection": "reparam_attempted == false",
            "endpoint_rule": "last paid primary-inner step of each audited solve",
            "terminal_ledger_closure_gate": "zero or one; if present it must be final",
        },
        "cost_accounting": {
            "definition": (
                "successful energy/force cache misses of the 20 internal images"
            ),
            "endpoint_evaluations_excluded": True,
            "reparameterization_and_recovery_included": True,
            "interpretation_of_horizontal_gaps": (
                "charged reparameterization/recovery work between fixed-path solves"
            ),
        },
        "display": {
            "x_scale": "log",
            "x_limits": list(XLIM),
            "y_scale": "log",
            "y_limits": list(YLIM),
            "palette": "matplotlib jet resampled to eight discrete alpha colors",
            "markers": list(ALPHA_MARKERS),
            "call_limit_marker": "hollow X",
            "interpolation": False,
            "extrapolation": False,
            "carry_forward": False,
        },
        "frozen_inputs": {
            "preregistration": {
                "path": str(PREREGISTRATION),
                "sha256": EXPECTED_INPUT_SHA256[PREREGISTRATION],
            },
            "local_report": {
                "path": str(LOCAL_REPORT),
                "sha256": EXPECTED_INPUT_SHA256[LOCAL_REPORT],
            },
            "aggregate": {
                "path": str(AGGREGATE),
                "sha256": EXPECTED_INPUT_SHA256[AGGREGATE],
            },
            "optimizer_results": [
                {
                    "alpha": summary["alpha"],
                    "path": summary["result_file"],
                    "sha256": summary["result_sha256"],
                }
                for summary in summaries
            ],
        },
        "plotting_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "outputs": {
            "pdf": {"path": str(pdf_path), "sha256": sha256(pdf_path)},
            "source_csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "total_endpoint_count": validation_summary["total_endpoint_count"],
                "pdf": str(pdf_path),
                "source_csv": str(csv_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
