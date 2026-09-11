#!/usr/bin/env python3
"""Plot one-factor Chignolin IR-IVM parameter effects.

The figure deliberately distinguishes an observed first passage from a run
that did not reach the registered I_perp threshold.  A non-reaching run is
shown only at the Number of Forces of its last complete, committed
reparameterized path; that value is a lower bound on its unknown first-passage
cost, not an attained cost.

This script is isolated from the historical Chignolin comparison plotters and
only writes PNG and SVG candidates beneath the CPU1 active-candidate folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    EXPERIMENT_ROOT
    / "reports"
    / "parameter_scan_combined"
    / "parameter_settings.csv"
)
DEFAULT_CPU1_REPORT = (
    EXPERIMENT_ROOT / "reports" / "cpu1_prefix_repro" / "audit_1500.json"
)
DEFAULT_OUTPUT_DIR = (
    EXPERIMENT_ROOT
    / "figures"
    / "active_candidates"
    / "chignolin_ir_ivm_cpu1"
)
OUTPUT_STEM = "chignolin_ir_ivm_parameter_effects"

EXPECTED_ALPHA = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0)
EXPECTED_R = (0.1, 0.3, 0.5, 0.7, 0.8)
EXPECTED_M_MAX = (5, 8, 10, 15, 20, 50)
EXPECTED_INTERNAL_NF_BUDGET = 1500
EXPECTED_ETA = 1.1
EXPECTED_I_PERP_THRESHOLD = 0.21242215133173672
EXPECTED_CPU1_TOTAL_NF = 905
PAPER_TOTAL_NF_DEFINITION = "sum of all 22 image joint E/F cache misses"
COMPLETE_STATE_KIND = "reparam_committed"

REQUIRED_COLUMNS = {
    "parameter_setting_id",
    "alpha",
    "residual_ratio_r",
    "inner_max_steps_m_max",
    "internal_nf_budget",
    "eta",
    "i_perp_threshold",
    "analysis_status",
    "audit_failure_reason",
    "first_threshold_reached",
    "first_threshold_state_index",
    "first_threshold_state_kind",
    "first_threshold_total_nf",
    "first_threshold_i_perp",
    "last_complete_reparam_state_available",
    "last_complete_reparam_state_index",
    "last_complete_reparam_total_nf",
    "last_complete_reparam_i_perp",
    "last_usable_state_kind",
    "last_usable_state_index",
    "last_usable_total_nf",
    "last_usable_i_perp",
    "right_censoring_total_nf",
    "terminal_total_nf",
    "terminal_partial_tail_internal_nf",
    "terminal_partial_tail_quarantined",
}


@dataclass(frozen=True)
class Observation:
    """A plotted first-passage observation or a valid lower bound."""

    parameter_setting_id: str
    reached: bool
    number_of_forces: int
    i_perp: float
    state_index: int
    state_kind: str


@dataclass(frozen=True)
class Panel:
    """A one-factor control panel."""

    key: str
    values: Tuple[float, ...]
    rows: Tuple[Mapping[str, str], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: object, field: str = "boolean") -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} must be True or False, got {value!r}")


def parse_float(row: Mapping[str, str], field: str) -> float:
    value = str(row.get(field, "")).strip()
    if not value:
        raise ValueError(
            f"{row.get('parameter_setting_id', '<unknown>')}: missing {field}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"{row.get('parameter_setting_id', '<unknown>')}: non-finite {field}"
        )
    return result


def parse_int(row: Mapping[str, str], field: str) -> int:
    value = parse_float(row, field)
    integer = int(value)
    if value != integer:
        raise ValueError(
            f"{row.get('parameter_setting_id', '<unknown>')}: {field} is not an integer"
        )
    return integer


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def load_parameter_rows(path: Path) -> List[Dict[str, str]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing parameter table: {resolved}")
    with resolved.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(
                "parameter table is missing required columns: " + ", ".join(missing)
            )
        rows = list(reader)
    if not rows:
        raise ValueError("parameter table is empty")
    return rows


def _matches(
    row: Mapping[str, str], *, alpha: float = None, r: float = None, m_max: int = None
) -> bool:
    if parse_int(row, "internal_nf_budget") != EXPECTED_INTERNAL_NF_BUDGET:
        return False
    if not close(parse_float(row, "eta"), EXPECTED_ETA):
        return False
    if alpha is not None and not close(parse_float(row, "alpha"), alpha):
        return False
    if r is not None and not close(parse_float(row, "residual_ratio_r"), r):
        return False
    if m_max is not None and parse_int(row, "inner_max_steps_m_max") != m_max:
        return False
    return True


def _ordered_unique_panel(
    key: str,
    rows: Iterable[Mapping[str, str]],
    expected_values: Sequence[float],
    value_field: str,
) -> Panel:
    indexed: Dict[float, Mapping[str, str]] = {}
    for row in rows:
        value = parse_float(row, value_field)
        canonical = next(
            (expected for expected in expected_values if close(value, expected)),
            None,
        )
        if canonical is None:
            raise ValueError(f"unexpected {key} value in control panel: {value}")
        if canonical in indexed:
            raise ValueError(f"duplicate {key}={canonical:g} control record")
        indexed[canonical] = row
    missing = [value for value in expected_values if value not in indexed]
    if missing:
        raise ValueError(
            f"missing {key} control records: "
            + ", ".join(f"{value:g}" for value in missing)
        )
    return Panel(
        key=key,
        values=tuple(float(value) for value in expected_values),
        rows=tuple(indexed[float(value)] for value in expected_values),
    )


def select_control_panels(rows: Sequence[Mapping[str, str]]) -> Tuple[Panel, ...]:
    """Select the three registered 1500-internal-NF one-factor scans."""

    alpha_rows = [row for row in rows if _matches(row, r=0.5, m_max=5)]
    r_rows = [row for row in rows if _matches(row, alpha=10.0, m_max=5)]
    m_rows = [row for row in rows if _matches(row, alpha=10.0, r=0.5)]
    return (
        _ordered_unique_panel(
            "alpha", alpha_rows, EXPECTED_ALPHA, "alpha"
        ),
        _ordered_unique_panel(
            "r", r_rows, EXPECTED_R, "residual_ratio_r"
        ),
        _ordered_unique_panel(
            "m_max", m_rows, EXPECTED_M_MAX, "inner_max_steps_m_max"
        ),
    )


def _require_blank(row: Mapping[str, str], fields: Iterable[str]) -> None:
    for field in fields:
        if str(row.get(field, "")).strip():
            raise ValueError(
                f"{row.get('parameter_setting_id', '<unknown>')}: "
                f"{field} must be blank"
            )


def observation_from_row(row: Mapping[str, str]) -> Observation:
    """Validate and map a row without turning censoring into success."""

    setting = str(row.get("parameter_setting_id", "")).strip()
    if not setting:
        raise ValueError("parameter_setting_id is missing")
    if str(row.get("audit_failure_reason", "")).strip():
        raise ValueError(f"{setting}: audit failure cannot be plotted")
    if not close(parse_float(row, "i_perp_threshold"), EXPECTED_I_PERP_THRESHOLD):
        raise ValueError(f"{setting}: unexpected I_perp threshold")
    if not parse_bool(
        row.get("last_complete_reparam_state_available", ""),
        "last_complete_reparam_state_available",
    ):
        raise ValueError(f"{setting}: no complete reparameterized path")
    if not parse_bool(
        row.get("terminal_partial_tail_quarantined", ""),
        "terminal_partial_tail_quarantined",
    ):
        raise ValueError(f"{setting}: incomplete terminal tail was not quarantined")

    last_index = parse_int(row, "last_complete_reparam_state_index")
    last_nf = parse_int(row, "last_complete_reparam_total_nf")
    last_i_perp = parse_float(row, "last_complete_reparam_i_perp")
    if str(row.get("last_usable_state_kind", "")).strip() != COMPLETE_STATE_KIND:
        raise ValueError(f"{setting}: last usable state is not reparam_committed")
    if parse_int(row, "last_usable_state_index") != last_index:
        raise ValueError(f"{setting}: last usable state index is inconsistent")
    if parse_int(row, "last_usable_total_nf") != last_nf:
        raise ValueError(f"{setting}: last usable total NF is inconsistent")
    if not close(parse_float(row, "last_usable_i_perp"), last_i_perp):
        raise ValueError(f"{setting}: last usable I_perp is inconsistent")
    if last_nf > parse_int(row, "terminal_total_nf"):
        raise ValueError(f"{setting}: committed state exceeds terminal total NF")

    reached = parse_bool(
        row.get("first_threshold_reached", ""), "first_threshold_reached"
    )
    status = str(row.get("analysis_status", "")).strip()
    if reached:
        if status != "reached":
            raise ValueError(f"{setting}: reached flag disagrees with status")
        if str(row.get("first_threshold_state_kind", "")).strip() != COMPLETE_STATE_KIND:
            raise ValueError(f"{setting}: first passage is not reparam_committed")
        first_index = parse_int(row, "first_threshold_state_index")
        first_nf = parse_int(row, "first_threshold_total_nf")
        first_i_perp = parse_float(row, "first_threshold_i_perp")
        if first_i_perp > EXPECTED_I_PERP_THRESHOLD:
            raise ValueError(f"{setting}: reported first passage is above threshold")
        if first_index > last_index or first_nf > last_nf:
            raise ValueError(f"{setting}: first passage follows the last complete state")
        _require_blank(row, ("right_censoring_total_nf",))
        return Observation(
            parameter_setting_id=setting,
            reached=True,
            number_of_forces=first_nf,
            i_perp=first_i_perp,
            state_index=first_index,
            state_kind=COMPLETE_STATE_KIND,
        )

    if status != "not_reached":
        raise ValueError(f"{setting}: non-reaching flag disagrees with status")
    _require_blank(
        row,
        (
            "first_threshold_state_index",
            "first_threshold_state_kind",
            "first_threshold_total_nf",
            "first_threshold_i_perp",
        ),
    )
    censor_nf = parse_int(row, "right_censoring_total_nf")
    if censor_nf != last_nf:
        raise ValueError(
            f"{setting}: censoring NF is not the last complete reparam NF"
        )
    if last_i_perp <= EXPECTED_I_PERP_THRESHOLD:
        raise ValueError(f"{setting}: non-reaching record is already below threshold")
    return Observation(
        parameter_setting_id=setting,
        reached=False,
        number_of_forces=censor_nf,
        i_perp=last_i_perp,
        state_index=last_index,
        state_kind=COMPLETE_STATE_KIND,
    )


def load_cpu1_confirmation(path: Path) -> Dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing CPU1 confirmation report: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("paper_total_nf_definition") != PAPER_TOTAL_NF_DEFINITION:
        raise ValueError("CPU1 report uses a different Number of Forces definition")
    if payload.get("usable_state_rule") != "record_index_0_or_reparam_committed":
        raise ValueError("CPU1 report uses a different complete-state rule")
    results = payload.get("results", [])
    if payload.get("result_count") != 1 or len(results) != 1:
        raise ValueError("CPU1 report must contain exactly one result")
    result = results[0]
    checks = (
        (close(float(result.get("alpha", math.nan)), 10.0), "alpha"),
        (close(float(result.get("residual_ratio_r", math.nan)), 0.5), "r"),
        (int(result.get("inner_max_steps_m_max", -1)) == 5, "m_max"),
        (
            int(result.get("internal_nf_budget", -1))
            == EXPECTED_INTERNAL_NF_BUDGET,
            "internal NF budget",
        ),
        (close(float(result.get("eta", math.nan)), EXPECTED_ETA), "eta"),
        (
            close(
                float(result.get("i_perp_threshold", math.nan)),
                EXPECTED_I_PERP_THRESHOLD,
            ),
            "I_perp threshold",
        ),
        (result.get("audit_status") == "passed", "audit status"),
        (result.get("first_threshold_reached") is True, "first-passage flag"),
        (
            result.get("first_threshold_state_kind") == COMPLETE_STATE_KIND,
            "first-passage state kind",
        ),
        (
            int(result.get("first_threshold_total_nf", -1))
            == EXPECTED_CPU1_TOTAL_NF,
            "first-passage total NF",
        ),
    )
    failures = [label for passed, label in checks if not passed]
    if failures:
        raise ValueError("CPU1 confirmation mismatch: " + ", ".join(failures))
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "alpha": 10.0,
        "r": 0.5,
        "m_max": 5,
        "number_of_forces": EXPECTED_CPU1_TOTAL_NF,
        "state_kind": COMPLETE_STATE_KIND,
        "i_perp": float(result["first_threshold_i_perp"]),
    }


def build_validation_report(
    input_csv: Path, cpu1_report: Path
) -> Tuple[Tuple[Panel, ...], Dict[str, object], Dict[str, object]]:
    rows = load_parameter_rows(input_csv)
    panels = select_control_panels(rows)
    threshold_values = []
    panel_report = {}
    for panel in panels:
        observations = []
        for value, row in zip(panel.values, panel.rows):
            if parse_int(row, "internal_nf_budget") != EXPECTED_INTERNAL_NF_BUDGET:
                raise ValueError("control panel contains the wrong internal NF budget")
            if not close(parse_float(row, "eta"), EXPECTED_ETA):
                raise ValueError("control panel contains the wrong eta")
            threshold_values.append(parse_float(row, "i_perp_threshold"))
            observation = observation_from_row(row)
            observations.append(
                {
                    "parameter_value": value,
                    "status": "reached" if observation.reached else "lower_bound",
                    "number_of_forces": observation.number_of_forces,
                    "state_kind": observation.state_kind,
                    "state_index": observation.state_index,
                    "i_perp": observation.i_perp,
                    "parameter_setting_id": observation.parameter_setting_id,
                }
            )
        panel_report[panel.key] = observations
    if not all(close(value, EXPECTED_I_PERP_THRESHOLD) for value in threshold_values):
        raise ValueError("control panels do not share the registered I_perp threshold")
    cpu1 = load_cpu1_confirmation(cpu1_report)
    input_resolved = Path(input_csv).expanduser().resolve()
    report = {
        "schema_version": "chignolin-ir-ivm-parameter-effects-v1",
        "source_parameter_table": str(input_resolved),
        "source_parameter_table_sha256": sha256_file(input_resolved),
        "i_perp_threshold": EXPECTED_I_PERP_THRESHOLD,
        "number_of_forces_definition": PAPER_TOTAL_NF_DEFINITION,
        "non_reaching_value_definition": (
            "lower bound at the last reparam_committed state; not an attained cost"
        ),
        "panels": panel_report,
        "cpu1_confirmation": cpu1,
    }
    return panels, cpu1, report


def ensure_output_dir(path: Path) -> Path:
    requested = Path(path).expanduser().resolve()
    required = DEFAULT_OUTPUT_DIR.resolve()
    if requested != required:
        raise ValueError(
            "parameter-effect output is restricted to the CPU1 active-candidate "
            f"directory: {required}"
        )
    return requested


def setup_style(mpl) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def _plot_observations(axis, x_values, observations, color: str) -> None:
    reached_x = [x for x, obs in zip(x_values, observations) if obs.reached]
    reached_y = [obs.number_of_forces for obs in observations if obs.reached]
    censored_x = [x for x, obs in zip(x_values, observations) if not obs.reached]
    censored_y = [obs.number_of_forces for obs in observations if not obs.reached]
    if reached_x:
        axis.scatter(
            reached_x,
            reached_y,
            s=28,
            marker="o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )
    if censored_x:
        axis.scatter(
            censored_x,
            censored_y,
            s=31,
            marker="^",
            facecolor="white",
            edgecolor=color,
            linewidth=1.05,
            zorder=5,
        )
        for x_value, y_value in zip(censored_x, censored_y):
            axis.annotate(
                "",
                xy=(x_value, y_value + 72),
                xytext=(x_value, y_value + 10),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": 0.8,
                    "mutation_scale": 5.5,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=4,
            )


def save_plot(
    panels: Tuple[Panel, ...],
    cpu1: Mapping[str, object],
    output_dir: Path,
    overwrite: bool,
) -> List[Path]:
    import matplotlib as mpl  # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    from matplotlib.lines import Line2D  # pylint: disable=import-outside-toplevel

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / f"{OUTPUT_STEM}.png",
        output_dir / f"{OUTPUT_STEM}.svg",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "candidate output exists; pass --overwrite to replace only this plot: "
            + ", ".join(str(path) for path in existing)
        )

    setup_style(mpl)
    blue = "#0072B2"
    orange = "#E69F00"
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
    observations_by_panel = [
        [observation_from_row(row) for row in panel.rows] for panel in panels
    ]

    alpha_panel, r_panel, m_panel = panels
    alpha_x = list(alpha_panel.values)
    r_x = list(r_panel.values)
    m_x = list(range(len(m_panel.values)))
    plot_x = (alpha_x, r_x, m_x)
    cpu_x = (float(cpu1["alpha"]), float(cpu1["r"]), m_panel.values.index(5.0))
    titles = (
        r"$r=0.5,\ m_{\max}=5$",
        r"$\alpha=10,\ m_{\max}=5$",
        r"$\alpha=10,\ r=0.5$",
    )
    xlabels = (r"$\alpha$", r"$r$", r"$m_{\max}$")

    for index, (axis, panel, x_values, observations) in enumerate(
        zip(axes, panels, plot_x, observations_by_panel)
    ):
        _plot_observations(axis, x_values, observations, blue)
        axis.scatter(
            [cpu_x[index]],
            [int(cpu1["number_of_forces"])],
            s=57,
            marker="*",
            facecolor=orange,
            edgecolor="#333333",
            linewidth=0.55,
            zorder=7,
        )
        axis.set_title(titles[index], pad=6)
        axis.set_xlabel(xlabels[index], labelpad=4)
        axis.set_ylim(0, 1650)
        axis.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.7, zorder=0)
        axis.tick_params(direction="out", length=3.0, width=0.7)
        axis.text(
            -0.10,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9.5,
            va="bottom",
        )

    axes[0].set_xscale("log")
    axes[0].set_xlim(0.055, 1.8e6)
    axes[0].set_xticks(EXPECTED_ALPHA)
    axes[0].xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10))
    axes[0].xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axes[1].set_xlim(0.04, 0.86)
    axes[1].set_xticks(EXPECTED_R)
    axes[1].set_xticklabels(["0.1", "0.3", "0.5", "0.7", "0.8"])
    axes[2].set_xlim(-0.5, len(EXPECTED_M_MAX) - 0.5)
    axes[2].set_xticks(m_x)
    axes[2].set_xticklabels([str(value) for value in EXPECTED_M_MAX])
    axes[0].set_ylabel("Number of Forces", labelpad=5)

    handles = (
        Line2D(
            [], [], marker="o", linestyle="None", markersize=5.1,
            markerfacecolor=blue, markeredgecolor="white", markeredgewidth=0.55,
            label="Threshold reached",
        ),
        Line2D(
            [], [], marker=r"$\uparrow$", linestyle="None", markersize=8.0,
            color=blue, label="Not reached: lower bound",
        ),
        Line2D(
            [], [], marker="*", linestyle="None", markersize=7.5,
            markerfacecolor=orange, markeredgecolor="#333333", markeredgewidth=0.55,
            label="CPU1 confirmation",
        ),
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.35,
    )
    figure.suptitle(
        rf"First passage of $I_\perp \leq {EXPECTED_I_PERP_THRESHOLD:.4f}$",
        x=0.5,
        y=0.885,
        fontsize=8.5,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.99,
        bottom=0.21,
        top=0.69,
        wspace=0.28,
    )
    figure.savefig(outputs[0], dpi=600)
    figure.savefig(outputs[1])
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot audited one-factor Chignolin IR-IVM parameter effects."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--cpu1-report", type=Path, default=DEFAULT_CPU1_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the exact plotted observations; write nothing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this script's existing PNG and SVG candidate outputs.",
    )
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)
    panels, cpu1, report = build_validation_report(args.input_csv, args.cpu1_report)
    if args.dry_run:
        report["dry_run"] = True
        report["would_write"] = [
            str(output_dir / f"{OUTPUT_STEM}.png"),
            str(output_dir / f"{OUTPUT_STEM}.svg"),
        ]
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    generated = save_plot(panels, cpu1, output_dir, args.overwrite)
    print(
        json.dumps(
            {
                "generated": [str(path.resolve()) for path in generated],
                "source_parameter_table_sha256": report[
                    "source_parameter_table_sha256"
                ],
                "cpu1_report_sha256": cpu1["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
