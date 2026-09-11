#!/usr/bin/env python3
"""Plot the registered Ala fixed-alpha comparison as a 2x2 CTP figure.

Columns compare tighter and production inner solves while retaining the same
reparameterized IVM algorithm.  Rows show raw I_perp and the independently
densified energy barrier.  Every curve is an exact observed cost-to-precision
envelope: at a repeated force cost the best observed value is retained, and
only strict improvements are connected.  The script never interpolates,
extrapolates, or carries a terminal value forward to the force budget.

The right-hand inputs are frozen to the completed formal Ala metric-v2
campaign.  The left-hand metric preregistration and aggregate are supplied on
the command line because those eight tighter-inner-solve calculations are a
new registered experiment.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np


REPO = Path(__file__).resolve().parents[5]
STUDY = REPO / "python/alanine_ivm_clean_four_level_20260801"
RIGHT_STEM = "stage4_ala_discrete_grid_metric_campaign_v2_1787608620070816000"
RIGHT_PREREGISTRATION = STUDY / "reports" / f"{RIGHT_STEM}_preregistered.json"
RIGHT_AGGREGATE = STUDY / "reports" / f"{RIGHT_STEM}_aggregate.json"
RIGHT_EXPECTED_SHA256 = {
    RIGHT_PREREGISTRATION: (
        "5fcfad3228e72c267ea1d768dd6f677528ae1fc8e8c6f180ea0d3498503b98a3"
    ),
    RIGHT_AGGREGATE: (
        "0e53a4e00a8a30f8e60c9fd278426f9b55fecada08840dee6f514a4e93390e63"
    ),
}

LEFT_OPTIMIZER_STEM = (
    "stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign_v1_"
    "1788923892307558000"
)
LEFT_OPTIMIZER_PREREGISTRATION = (
    STUDY / "reports" / f"{LEFT_OPTIMIZER_STEM}_preregistered.json"
)
LEFT_OPTIMIZER_AGGREGATE = STUDY / "reports" / f"{LEFT_OPTIMIZER_STEM}_aggregate.json"
LEFT_OPTIMIZER_LOCAL_REPORT = (
    STUDY / "reports" / f"{LEFT_OPTIMIZER_STEM}_local_report.json"
)
LEFT_OPTIMIZER_EXPECTED_SHA256 = {
    LEFT_OPTIMIZER_PREREGISTRATION: (
        "da1ded53f6dc11ac71ebe5bae64ba452f19c95fdc60467be220ff630912b6a05"
    ),
    LEFT_OPTIMIZER_AGGREGATE: (
        "50914e1cdfdafc59287921b4eef6081a5274d40366a4da926a590c06a6c9828a"
    ),
    LEFT_OPTIMIZER_LOCAL_REPORT: (
        "78ef8b82b7b7a8dbc2b6f4f62d8728fb216c1494fce46a95bbddce5241592362"
    ),
}
LEFT_OPTIMIZER_PROTOCOL_ID = (
    "stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign_v1"
)

OUTPUT_DIR = REPO / "output/pdf"
STEM = "alanine66d_alpha_cost_to_precision_2x2"
ALPHAS = tuple(10.0**exponent for exponent in range(-1, 7))
TOTAL_NF_BUDGET = 5000
REFERENCE_BARRIER_KCAL_PER_MOL = 2.4560825131253083
REFERENCE_ABS_TOL = 1.0e-12
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
DENSE_SUBDIVISIONS = 160
DENSE_CONVERGENCE_TOL = 0.001

LEFT_SPEC = {
    "residual_ratio": 0.1,
    "inner_max_steps": 5000,
    "total_internal_nf_budget": TOTAL_NF_BUDGET,
}
RIGHT_SPEC = {
    "residual_ratio": 0.7,
    "inner_max_steps": 5,
    "total_internal_nf_budget": TOTAL_NF_BUDGET,
}

IPERP_XLIM = (1.0e3, 1.0e-4)
BARRIER_XLIM = (4.0, 2.44)
NF_YLIM = (18.0, 5500.0)
NF_TICKS = (20, 50, 100, 200, 500, 1000, 2000, 5000)
ALPHA_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
ALPHA_COLORS = tuple(
    mpl.colormaps["jet"].resampled(len(ALPHAS))(index)
    for index in range(len(ALPHAS))
)


class FigureInputError(RuntimeError):
    """A registered optimizer or dense-metric input failed validation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--left-metric-preregistration",
        type=Path,
        required=True,
        help="Registered metric manifest for r=0.1, m_max=5000 with reparameterization.",
    )
    parser.add_argument(
        "--left-metric-aggregate",
        type=Path,
        required=True,
        help="Completed aggregate belonging to --left-metric-preregistration.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and summarize both registered inputs without writing outputs.",
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


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FigureInputError(f"Cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise FigureInputError(f"Expected a JSON object: {path}")
    return value


def scalar(saved: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in saved.files:
        raise FigureInputError(f"Result lacks required field {key!r}")
    value = np.asarray(saved[key])
    if value.size != 1:
        raise FigureInputError(f"Result field {key!r} is not scalar")
    return value.reshape(()).item()


def same_number(left: Any, right: Any, *, atol: float = 1.0e-14) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def alpha_label(alpha: float) -> str:
    exponent = int(round(math.log10(alpha)))
    return rf"$10^{{{exponent}}}$"


def flatten_registered_runs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("cells")
    if isinstance(raw, list):
        runs = raw
    elif isinstance(raw, dict):
        runs = [item for family in raw.values() for item in family]
    else:
        raise FigureInputError("Metric preregistration has no run list")
    if not all(isinstance(item, dict) for item in runs):
        raise FigureInputError("Metric preregistration run list is malformed")
    return [dict(item) for item in runs]


def validate_dense_policy(preregistration: Mapping[str, Any]) -> None:
    policy = preregistration.get("dense_policy")
    if not isinstance(policy, Mapping):
        raise FigureInputError("Metric preregistration lacks dense_policy")
    if list(policy.get("subdivision_levels", [])) != [20, 40, 80, 160]:
        raise FigureInputError("Dense subdivision levels are not 20,40,80,160")
    if int(policy.get("maximum_subdivisions", -1)) != DENSE_SUBDIVISIONS:
        raise FigureInputError("Dense metric does not select subdivision level 160")
    if not same_number(
        policy.get("convergence_tolerance_kcal_per_mol"),
        DENSE_CONVERGENCE_TOL,
    ):
        raise FigureInputError("Dense convergence tolerance is not 0.001 kcal/mol")
    if policy.get("selection_rule") != "always_select_maximum_preregistered_level":
        raise FigureInputError("Dense metric selection rule has changed")


def validate_aggregate(
    preregistration_path: Path,
    aggregate_path: Path,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate = load_json(aggregate_path)
    preregistration_hash = sha256(preregistration_path)
    if aggregate.get("status") != "complete":
        raise FigureInputError(f"Metric aggregate is not complete: {aggregate_path}")
    if str(aggregate.get("preregistration_sha256", "")) != preregistration_hash:
        raise FigureInputError("Metric aggregate/preregistration hash mismatch")
    recorded_preregistration = Path(str(aggregate.get("preregistration", ""))).resolve()
    if recorded_preregistration != preregistration_path:
        raise FigureInputError("Metric aggregate points to another preregistration")
    planned = preregistration.get("planned_outputs", {})
    if isinstance(planned, Mapping) and planned.get("aggregate"):
        if Path(str(planned["aggregate"])).resolve() != aggregate_path:
            raise FigureInputError("Provided aggregate differs from the registered output")
    if "all_path_tasks_dense_converged" in aggregate:
        if aggregate.get("all_path_tasks_dense_converged") is not True:
            raise FigureInputError("Not every registered dense path is converged")
    elif "all_eligible_states_dense_converged" in aggregate:
        if aggregate.get("all_eligible_states_dense_converged") is not True:
            raise FigureInputError("Not every eligible state has a converged dense metric")
    else:
        raise FigureInputError("Metric aggregate lacks a dense-convergence certificate")
    if (
        aggregate.get("nonconverged_path_task_count") is not None
        and int(aggregate["nonconverged_path_task_count"]) != 0
    ):
        raise FigureInputError("Metric aggregate contains nonconverged paths")
    if (
        aggregate.get("all_four_dense_levels_required") is not None
        and aggregate.get("all_four_dense_levels_required") is not True
    ):
        raise FigureInputError("Metric aggregate did not require all four dense levels")
    if (
        aggregate.get("selected_subdivisions") is not None
        and int(aggregate["selected_subdivisions"]) != DENSE_SUBDIVISIONS
    ):
        raise FigureInputError("Metric aggregate did not select subdivision level 160")
    if (
        aggregate.get("dense_convergence_tolerance_kcal_per_mol") is not None
        and not same_number(
            aggregate["dense_convergence_tolerance_kcal_per_mol"],
            DENSE_CONVERGENCE_TOL,
        )
    ):
        raise FigureInputError("Metric aggregate dense tolerance is not 0.001 kcal/mol")
    if aggregate.get("metric_runtime_matches_optimizer_runtime", True) is not True:
        raise FigureInputError("Metric and optimizer runtime identities differ")
    if int(aggregate.get("optimization_nf_charged", 0)) != 0:
        raise FigureInputError("Offline dense diagnostics were charged to optimization N_F")
    return aggregate


def validate_reference_barrier(preregistration: Mapping[str, Any]) -> None:
    comparison = preregistration.get("formal_comparison")
    if comparison is None:
        return
    if not isinstance(comparison, Mapping) or not same_number(
        comparison.get("reference_barrier_kcal_per_mol"),
        REFERENCE_BARRIER_KCAL_PER_MOL,
        atol=1.0e-15,
    ):
        raise FigureInputError("Reference densified barrier has changed")


def host_artifact_hashes(
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> dict[str, str]:
    planned = preregistration.get("planned_outputs", {})
    if not isinstance(planned, Mapping):
        raise FigureInputError("Metric preregistration lacks planned outputs")

    report_paths: list[tuple[str, Path, Mapping[str, Any]]] = []
    planned_host_reports = planned.get("host_reports")
    if isinstance(planned_host_reports, Mapping):
        aggregate_host_reports = aggregate.get("host_reports")
        if not isinstance(aggregate_host_reports, Mapping):
            raise FigureInputError("Metric aggregate lacks registered host reports")
        for name, raw_path in planned_host_reports.items():
            aggregate_record = aggregate_host_reports.get(name)
            if not isinstance(aggregate_record, Mapping):
                raise FigureInputError(f"Metric aggregate lacks host report {name!r}")
            report_paths.append((str(name), Path(str(raw_path)).resolve(), aggregate_record))
    elif planned.get("run_report"):
        aggregate_record = aggregate.get("run_report")
        if not isinstance(aggregate_record, Mapping):
            raise FigureInputError("Metric aggregate lacks its registered run report")
        report_paths.append(
            ("run", Path(str(planned["run_report"])).resolve(), aggregate_record)
        )
    else:
        raise FigureInputError("Metric preregistration has no registered task report")

    result: dict[str, str] = {}
    preregistration_hash = sha256(preregistration_path)
    protocol_id = str(preregistration.get("protocol_id", ""))
    for report_name, path, aggregate_record in report_paths:
        if not path.is_file():
            raise FigureInputError(f"Registered metric task report is missing: {path}")
        if Path(str(aggregate_record.get("path", ""))).resolve() != path:
            raise FigureInputError(
                f"Metric aggregate routes {report_name!r} to another task report"
            )
        actual_report_hash = sha256(path)
        if str(aggregate_record.get("sha256", "")) != actual_report_hash:
            raise FigureInputError(
                f"Metric aggregate task-report hash changed: {path}"
            )
        report = load_json(path)
        if (
            report.get("status") != "complete"
            or str(report.get("protocol_id", "")) != protocol_id
            or Path(str(report.get("preregistration", ""))).resolve()
            != preregistration_path
            or str(report.get("preregistration_sha256", ""))
            != preregistration_hash
            or int(report.get("jobs", -1)) != 1
        ):
            raise FigureInputError(f"Metric task report identity is invalid: {path}")
        rows = report.get("rows")
        if not isinstance(rows, list):
            raise FigureInputError(f"Metric task report has no row list: {path}")
        for row in rows:
            if not isinstance(row, Mapping):
                raise FigureInputError(f"Malformed metric task report row: {path}")
            task_id = str(row.get("task_id", ""))
            artifact_hash = str(row.get("artifact_sha256", ""))
            if not task_id or not artifact_hash:
                raise FigureInputError(f"Malformed metric task report row: {path}")
            if task_id in result and result[task_id] != artifact_hash:
                raise FigureInputError(f"Conflicting metric hash for task {task_id}")
            result[task_id] = artifact_hash
    return result


def registered_code_files(preregistration: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    source_manifest = preregistration.get("source_manifest", {})
    if not isinstance(source_manifest, Mapping):
        return {}
    nested_repository_files = source_manifest.get("repository_files")
    repository_files = (
        nested_repository_files
        if isinstance(nested_repository_files, Mapping)
        else source_manifest
    )
    if not isinstance(repository_files, Mapping):
        return {}
    result: dict[str, dict[str, str]] = {}
    for name, entry in repository_files.items():
        if not isinstance(entry, Mapping) or not entry.get("path") or not entry.get("sha256"):
            continue
        path = Path(str(entry["path"])).resolve()
        expected = str(entry["sha256"])
        if not path.is_file() or sha256(path) != expected:
            raise FigureInputError(f"Registered source-code hash changed: {path}")
        result[str(name)] = {"path": str(path), "sha256": expected}
    return result


def select_alpha_runs(
    preregistration: Mapping[str, Any],
    specification: Mapping[str, Any],
    setting: str,
) -> list[dict[str, Any]]:
    selected = []
    for run in flatten_registered_runs(preregistration):
        controls = run.get("controls")
        if not isinstance(controls, Mapping):
            continue
        if not all(
            same_number(controls.get(name), expected)
            for name, expected in specification.items()
        ):
            continue
        alpha = float(controls.get("alpha", math.nan))
        if alpha in ALPHAS:
            selected.append(run)
    observed = [float(run["controls"]["alpha"]) for run in selected]
    if len(selected) != len(ALPHAS) or sorted(observed) != list(ALPHAS):
        raise FigureInputError(
            f"{setting} must contain exactly alpha=1e-1,...,1e6 once each; "
            f"observed {sorted(observed)}"
        )
    selected.sort(key=lambda run: float(run["controls"]["alpha"]))
    return selected


def validate_result_identity(
    saved: np.lib.npyio.NpzFile,
    alpha: float,
    specification: Mapping[str, Any],
) -> None:
    required = {
        "alpha": alpha,
        "alpha_mode": "fixed",
        "discrete_residual_ratio": specification["residual_ratio"],
        "discrete_inner_max_steps": specification["inner_max_steps"],
        "reparam_mode": "state_curve",
        "reparam_every": 1,
        "total_internal_nf_budget": specification["total_internal_nf_budget"],
    }
    for key, expected in required.items():
        observed = scalar(saved, key)
        if isinstance(expected, str):
            matched = str(observed) == expected
        else:
            matched = same_number(observed, expected)
        if not matched:
            raise FigureInputError(
                f"Result identity mismatch for {key}: {observed!r} != {expected!r}"
            )


def validate_inner_cap(run: Mapping[str, Any], setting: str) -> dict[str, Any]:
    evidence = run.get("inner_cap_stop_evidence")
    if not isinstance(evidence, Mapping):
        raise FigureInputError(f"{setting} run lacks inner-cap stop evidence")
    copied = dict(evidence)
    if setting == "tighter_inner_solves":
        if int(evidence.get("inner_cap_stop_count", -1)) != 0:
            raise FigureInputError(
                "The r=0.1 calculation reached m_max=5000; the left column "
                "cannot be described as having a nonbinding inner cap"
            )
        if evidence.get("inner_cap_binding") is not False:
            raise FigureInputError("The left inner cap was not certified nonbinding")
    return copied


def validate_frozen_left_optimizer() -> dict[str, Any]:
    """Validate the completed eight-run optimizer campaign before metrics."""

    for path, expected in LEFT_OPTIMIZER_EXPECTED_SHA256.items():
        if not path.is_file() or sha256(path) != expected:
            raise FigureInputError(f"Frozen tighter-inner optimizer input changed: {path}")
    preregistration = load_json(LEFT_OPTIMIZER_PREREGISTRATION)
    aggregate = load_json(LEFT_OPTIMIZER_AGGREGATE)
    local_report = load_json(LEFT_OPTIMIZER_LOCAL_REPORT)
    if (
        preregistration.get("status") != "preregistered"
        or preregistration.get("protocol_id") != LEFT_OPTIMIZER_PROTOCOL_ID
        or int(preregistration.get("cell_count", -1)) != len(ALPHAS)
    ):
        raise FigureInputError("Tighter-inner optimizer preregistration is invalid")
    if (
        aggregate.get("status") != "complete"
        or aggregate.get("protocol_id") != LEFT_OPTIMIZER_PROTOCOL_ID
        or aggregate.get("all_inner_caps_nonbinding") is not True
        or int(aggregate.get("cell_count", -1)) != len(ALPHAS)
        or int(aggregate.get("unique_cell_count", -1)) != len(ALPHAS)
        or aggregate.get("preregistration_sha256")
        != LEFT_OPTIMIZER_EXPECTED_SHA256[LEFT_OPTIMIZER_PREREGISTRATION]
        or aggregate.get("local_report_sha256")
        != LEFT_OPTIMIZER_EXPECTED_SHA256[LEFT_OPTIMIZER_LOCAL_REPORT]
    ):
        raise FigureInputError("Tighter-inner optimizer aggregate is invalid")
    if (
        Path(str(aggregate.get("preregistration", ""))).resolve()
        != LEFT_OPTIMIZER_PREREGISTRATION
        or Path(str(aggregate.get("local_report", ""))).resolve()
        != LEFT_OPTIMIZER_LOCAL_REPORT
    ):
        raise FigureInputError("Tighter-inner optimizer aggregate routing changed")
    if (
        local_report.get("status") != "complete"
        or local_report.get("protocol_id") != LEFT_OPTIMIZER_PROTOCOL_ID
        or local_report.get("preregistration_sha256")
        != LEFT_OPTIMIZER_EXPECTED_SHA256[LEFT_OPTIMIZER_PREREGISTRATION]
        or int(local_report.get("jobs", -1)) != 1
    ):
        raise FigureInputError("Tighter-inner optimizer local report is invalid")

    registered_runs = select_alpha_runs(preregistration, LEFT_SPEC, "tighter optimizer")
    registered_by_id = {str(run["cell_id"]): run for run in registered_runs}
    rows = local_report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(ALPHAS):
        raise FigureInputError("Tighter-inner optimizer report does not contain eight runs")
    by_alpha: dict[float, dict[str, Any]] = {}
    for row in rows:
        alpha = float(row.get("alpha", math.nan))
        run_id = str(row.get("cell_id", ""))
        registered = registered_by_id.get(run_id)
        if alpha not in ALPHAS or registered is None or alpha in by_alpha:
            raise FigureInputError("Tighter-inner optimizer report identity is invalid")
        if not same_number(registered["controls"]["alpha"], alpha):
            raise FigureInputError("Tighter-inner optimizer alpha identity changed")
        if int(row.get("inner_cap_stop_count", -1)) != 0:
            raise FigureInputError(f"Tighter-inner optimizer cap bound for alpha={alpha:g}")
        result_path = Path(str(row.get("result", ""))).resolve()
        result_hash = str(row.get("result_sha256", ""))
        audit_path = Path(str(row.get("member_audit", ""))).resolve()
        audit_hash = str(row.get("member_audit_sha256", ""))
        companion_path = Path(str(row.get("companion_audit", ""))).resolve()
        companion_hash = str(row.get("companion_audit_sha256", ""))
        for path, expected, subject in (
            (result_path, result_hash, "optimizer result"),
            (audit_path, audit_hash, "member audit"),
            (companion_path, companion_hash, "counter audit"),
        ):
            if not expected or not path.is_file() or sha256(path) != expected:
                raise FigureInputError(
                    f"Frozen tighter-inner {subject} changed for alpha={alpha:g}: {path}"
                )
        by_alpha[alpha] = {
            "cell_id": run_id,
            "result": {"path": str(result_path), "sha256": result_hash},
            "member_audit": {"path": str(audit_path), "sha256": audit_hash},
            "companion_audit": {
                "path": str(companion_path), "sha256": companion_hash
            },
            "inner_cap_stop_count": 0,
            "last_complete_state_NF": int(row["last_complete_state_NF"]),
            "final_optimization_NF": int(row["final_optimization_NF"]),
        }
    if sorted(by_alpha) != list(ALPHAS):
        raise FigureInputError("Tighter-inner optimizer alpha grid is incomplete")
    return {
        "protocol_id": LEFT_OPTIMIZER_PROTOCOL_ID,
        "preregistration": {
            "path": str(LEFT_OPTIMIZER_PREREGISTRATION),
            "sha256": LEFT_OPTIMIZER_EXPECTED_SHA256[LEFT_OPTIMIZER_PREREGISTRATION],
        },
        "aggregate": {
            "path": str(LEFT_OPTIMIZER_AGGREGATE),
            "sha256": LEFT_OPTIMIZER_EXPECTED_SHA256[LEFT_OPTIMIZER_AGGREGATE],
        },
        "local_report": {
            "path": str(LEFT_OPTIMIZER_LOCAL_REPORT),
            "sha256": LEFT_OPTIMIZER_EXPECTED_SHA256[LEFT_OPTIMIZER_LOCAL_REPORT],
        },
        "all_inner_caps_nonbinding": True,
        "runs_by_alpha": by_alpha,
    }


def load_run_states(
    run: Mapping[str, Any],
    specification: Mapping[str, Any],
    setting: str,
    task_specs: Mapping[str, Mapping[str, Any]],
    attested_hashes: Mapping[str, str],
    expected_optimizer_run: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    controls = run["controls"]
    alpha = float(controls["alpha"])
    result_path = Path(str(run.get("source_result", ""))).resolve()
    expected_result_hash = str(run.get("source_result_sha256", ""))
    if not result_path.is_file() or not expected_result_hash:
        raise FigureInputError(f"Registered optimizer result is missing for alpha={alpha:g}")
    if sha256(result_path) != expected_result_hash:
        raise FigureInputError(f"Optimizer result hash changed: {result_path}")
    if expected_optimizer_run is not None:
        expected_result = expected_optimizer_run["result"]
        if (
            result_path != Path(str(expected_result["path"])).resolve()
            or expected_result_hash != str(expected_result["sha256"])
        ):
            raise FigureInputError(
                f"Left metric manifest does not reference the frozen optimizer "
                f"result for alpha={alpha:g}"
            )

    registered_states = run.get("states")
    if not isinstance(registered_states, list) or not registered_states:
        raise FigureInputError(f"No registered eligible paths for alpha={alpha:g}")
    expected_count = run.get("complete_mesh_eligible_state_count")
    if expected_count is not None and int(expected_count) != len(registered_states):
        raise FigureInputError("Registered eligible-path count changed")

    rows: list[dict[str, Any]] = []
    artifact_records: list[dict[str, str]] = []
    seen_state_ids: set[str] = set()
    seen_indices: set[int] = set()
    with np.load(result_path, allow_pickle=False) as saved:
        validate_result_identity(saved, alpha, specification)
        paths = np.asarray(saved["stored_paths"], dtype=float)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=int)
        force_cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        mesh_cv = np.asarray(saved["segment_length_cv"], dtype=float)
        mesh_ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        lengths = {
            len(paths), len(stored_calls), len(force_cost), len(i_perp),
            len(committed), len(mesh_cv), len(mesh_ratio),
        }
        if len(lengths) != 1:
            raise FigureInputError(f"Optimizer arrays are misaligned: {result_path}")

        for state in registered_states:
            if state.get("formal_complete") is not True or state.get("mesh_eligible") is not True:
                raise FigureInputError("Metric manifest includes an ineligible path")
            index = int(state.get("trajectory_index", -1))
            state_id = str(state.get("state_id", ""))
            if index < 0 or index >= len(paths) or not state_id:
                raise FigureInputError("Registered path identity is invalid")
            if index in seen_indices or state_id in seen_state_ids:
                raise FigureInputError("Registered path identity is duplicated within one run")
            seen_indices.add(index)
            seen_state_ids.add(state_id)
            if index != 0 and not bool(committed[index]):
                raise FigureInputError("Noninitial eligible path was not reparameterization-committed")
            if float(mesh_cv[index]) > MESH_CV_MAX or float(mesh_ratio[index]) > MESH_RATIO_MAX:
                raise FigureInputError("Registered path violates the common mesh criteria")
            if array_sha256(paths[index]) != str(state.get("path_sha256", "")):
                raise FigureInputError("Registered path SHA-256 differs from stored_paths")
            checks = (
                (int(stored_calls[index]), int(state.get("calls_per_image", -1)), "stored call"),
                (int(force_cost[index]), int(state.get("optimization_nf", -1)), "N_F"),
            )
            for observed, expected, label in checks:
                if observed != expected:
                    raise FigureInputError(f"{label} differs at trajectory index {index}")
            if not same_number(i_perp[index], state.get("raw_i_perp"), atol=0.0):
                raise FigureInputError(f"Raw I_perp differs at trajectory index {index}")
            if not same_number(mesh_cv[index], state.get("segment_length_cv"), atol=0.0):
                raise FigureInputError(f"Mesh CV differs at trajectory index {index}")
            if not same_number(mesh_ratio[index], state.get("segment_length_ratio"), atol=0.0):
                raise FigureInputError(f"Mesh ratio differs at trajectory index {index}")

            task_id = str(state.get("path_task_id", ""))
            task = task_specs.get(task_id)
            if task is None:
                raise FigureInputError(f"No dense metric task for {task_id}")
            artifact_path = Path(str(task.get("artifact", ""))).resolve()
            if not artifact_path.is_file():
                raise FigureInputError(f"Dense metric artifact is missing: {artifact_path}")
            artifact_hash = sha256(artifact_path)
            if attested_hashes and attested_hashes.get(task_id) != artifact_hash:
                raise FigureInputError(f"Dense metric artifact is not host-attested: {task_id}")
            artifact = load_json(artifact_path)
            metric = artifact.get("metric")
            if not isinstance(metric, Mapping):
                raise FigureInputError(f"Dense metric payload is missing: {artifact_path}")
            if artifact.get("status") != "complete" or artifact.get("task_id") != task_id:
                raise FigureInputError(f"Dense metric task identity changed: {task_id}")
            if artifact.get("path_sha256") != state.get("path_sha256"):
                raise FigureInputError(f"Dense metric path identity changed: {task_id}")
            if metric.get("converged") is not True:
                raise FigureInputError(f"Dense metric did not converge: {task_id}")
            if int(metric.get("selected_subdivisions", -1)) != DENSE_SUBDIVISIONS:
                raise FigureInputError(f"Dense metric did not select level 160: {task_id}")
            if int(metric.get("optimization_nf_charged", -1)) != 0:
                raise FigureInputError(f"Dense diagnostic was charged to N_F: {task_id}")
            dense_difference = float(metric.get("final_level_difference_kcal_per_mol", math.inf))
            if not math.isfinite(dense_difference) or dense_difference > DENSE_CONVERGENCE_TOL:
                raise FigureInputError(f"Dense metric convergence check failed: {task_id}")
            barrier = float(metric.get("selected_barrier_kcal_per_mol", math.nan))
            raw_i_perp = float(i_perp[index])
            if not math.isfinite(barrier) or barrier <= 0.0:
                raise FigureInputError(f"Invalid densified barrier: {task_id}")
            if not math.isfinite(raw_i_perp) or raw_i_perp <= 0.0:
                raise FigureInputError(f"Invalid raw I_perp at trajectory index {index}")
            rows.append(
                {
                    "setting": setting,
                    "alpha": alpha,
                    "optimization_nf": int(force_cost[index]),
                    "trajectory_index": index,
                    "stored_call": int(stored_calls[index]),
                    "state_id": state_id,
                    "raw_i_perp": raw_i_perp,
                    "densified_delta_e_kcal_per_mol": barrier,
                    "path_sha256": str(state["path_sha256"]),
                    "segment_length_cv": float(mesh_cv[index]),
                    "segment_length_ratio": float(mesh_ratio[index]),
                    "result_file": str(result_path),
                    "result_sha256": expected_result_hash,
                    "metric_task_id": task_id,
                    "metric_artifact": str(artifact_path),
                    "metric_artifact_sha256": artifact_hash,
                    "dense_subdivisions": DENSE_SUBDIVISIONS,
                    "dense_final_level_difference_kcal_per_mol": dense_difference,
                }
            )
            artifact_records.append(
                {"task_id": task_id, "path": str(artifact_path), "sha256": artifact_hash}
            )

    cap_evidence = validate_inner_cap(run, setting)
    return rows, {
        "alpha": alpha,
        "controls": dict(controls),
        "result": {"path": str(result_path), "sha256": expected_result_hash},
        "inner_cap_stop_evidence": cap_evidence,
        "eligible_path_count": len(rows),
        "eligible_path_metric_artifacts": artifact_records,
    }


def strict_records(
    states: Sequence[Mapping[str, Any]], value_key: str
) -> list[dict[str, Any]]:
    """Return exact strict records after resolving repeated observed N_F."""

    best_at_nf: dict[int, Mapping[str, Any]] = {}
    for state in states:
        nf = int(state["optimization_nf"])
        value = float(state[value_key])
        previous = best_at_nf.get(nf)
        key = (value, int(state["trajectory_index"]), str(state["state_id"]))
        if previous is None:
            best_at_nf[nf] = state
        else:
            previous_key = (
                float(previous[value_key]),
                int(previous["trajectory_index"]),
                str(previous["state_id"]),
            )
            if key < previous_key:
                best_at_nf[nf] = state
    records: list[dict[str, Any]] = []
    running_best = math.inf
    for nf in sorted(best_at_nf):
        state = best_at_nf[nf]
        value = float(state[value_key])
        if value < running_best:
            copied = dict(state)
            copied["running_best_metric"] = value
            records.append(copied)
            running_best = value
    if not records:
        raise FigureInputError(f"No strict observed records for {value_key}")
    return records


def load_setting(
    preregistration_path: Path,
    aggregate_path: Path,
    specification: Mapping[str, Any],
    setting: str,
    expected_optimizer_runs: Mapping[float, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    preregistration_path = preregistration_path.expanduser().resolve()
    aggregate_path = aggregate_path.expanduser().resolve()
    preregistration = load_json(preregistration_path)
    validate_dense_policy(preregistration)
    validate_reference_barrier(preregistration)
    aggregate = validate_aggregate(
        preregistration_path, aggregate_path, preregistration
    )
    selected = select_alpha_runs(preregistration, specification, setting)
    raw_tasks = preregistration.get("tasks")
    if not isinstance(raw_tasks, list):
        raise FigureInputError("Metric preregistration has no dense task list")
    task_specs = {str(task["task_id"]): task for task in raw_tasks}
    attested_hashes = host_artifact_hashes(
        preregistration_path, preregistration, aggregate
    )

    curves: dict[float, dict[str, list[dict[str, Any]]]] = {}
    runs = []
    all_states = []
    for run in selected:
        states, run_info = load_run_states(
            run,
            specification,
            setting,
            task_specs,
            attested_hashes,
            (
                None
                if expected_optimizer_runs is None
                else expected_optimizer_runs[float(run["controls"]["alpha"])]
            ),
        )
        alpha = float(run_info["alpha"])
        i_records = strict_records(states, "raw_i_perp")
        barrier_records = strict_records(states, "densified_delta_e_kcal_per_mol")
        run_info["raw_i_perp_record_count"] = len(i_records)
        run_info["densified_delta_e_record_count"] = len(barrier_records)
        curves[alpha] = {
            "raw_i_perp": i_records,
            "densified_delta_e_kcal_per_mol": barrier_records,
        }
        runs.append(run_info)
        all_states.extend(states)

    if setting == "tighter_inner_solves":
        undershoots = [
            state
            for state in all_states
            if float(state["densified_delta_e_kcal_per_mol"])
            < REFERENCE_BARRIER_KCAL_PER_MOL - REFERENCE_ABS_TOL
        ]
        if undershoots:
            minimum = min(
                float(state["densified_delta_e_kcal_per_mol"])
                for state in undershoots
            )
            raise FigureInputError(
                "The new r=0.1 data undershoot the reference densified barrier "
                f"(minimum Delta E={minimum:.15g}). Raw Delta E would reward "
                "underestimation; redraw the lower row against "
                "abs(Delta E-Delta E_ref) and label it densified barrier error."
            )

    return {
        "setting": setting,
        "specification": dict(specification),
        "preregistration": {
            "path": str(preregistration_path), "sha256": sha256(preregistration_path)
        },
        "aggregate": {"path": str(aggregate_path), "sha256": sha256(aggregate_path)},
        "aggregate_summary": {
            key: aggregate.get(key)
            for key in (
                "status", "unique_path_task_count",
                "all_path_tasks_dense_converged",
                "all_eligible_states_dense_converged",
                "nonconverged_path_task_count",
                "total_diagnostic_raw_joint_evaluations", "optimization_nf_charged",
            )
        },
        "registered_code": registered_code_files(preregistration),
        "runs": runs,
        "curves": curves,
    }


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.4,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.4,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.minor.width": 0.55,
            "ytick.minor.width": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def nf_formatter(value: float, _position: int) -> str:
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def draw_curve(axis: plt.Axes, records: Sequence[Mapping[str, Any]], alpha: float, key: str) -> None:
    index = ALPHAS.index(alpha)
    values = np.asarray([float(row[key]) for row in records], dtype=float)
    costs = np.asarray([int(row["optimization_nf"]) for row in records], dtype=float)
    axis.plot(
        values,
        costs,
        color=ALPHA_COLORS[index],
        linewidth=1.2,
        marker=ALPHA_MARKERS[index],
        markersize=3.0,
        markerfacecolor=ALPHA_COLORS[index],
        markeredgecolor="#202020",
        markeredgewidth=0.28,
        markevery=max(1, len(records) // 9),
        alpha=0.9,
        zorder=3 + index,
    )


def style_axis(axis: plt.Axes, metric: str) -> None:
    axis.set_yscale("log")
    axis.set_ylim(*NF_YLIM)
    axis.set_yticks(NF_TICKS)
    axis.yaxis.set_major_formatter(FuncFormatter(nf_formatter))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_ylabel("Number of Forces")
    axis.grid(True, which="major", color="#D9D9D6", linewidth=0.45, alpha=0.7)
    if metric == "raw_i_perp":
        axis.set_xscale("log")
        axis.set_xlim(*IPERP_XLIM)
        axis.set_xlabel(r"Raw $I_\perp$ (smaller is better)")
    else:
        axis.set_xscale("log")
        axis.set_xlim(*BARRIER_XLIM)
        axis.set_xticks((4.0, 3.5, 3.0, 2.8, 2.6, 2.5))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel(r"Densified $\Delta E$ (kcal mol$^{-1}$)")
        axis.axvline(
            REFERENCE_BARRIER_KCAL_PER_MOL,
            color="#686868",
            linewidth=0.7,
            linestyle=":",
            zorder=1,
        )


def panel_label(figure: plt.Figure, axis: plt.Axes, label: str) -> None:
    position = axis.get_position()
    figure.text(
        position.x0 - 0.025,
        position.y1 + 0.009,
        label,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
    )


def draw_figure(left: Mapping[str, Any], right: Mapping[str, Any], output: Path) -> None:
    configure_style()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(180 / 25.4, 116 / 25.4),
        sharey=True,
    )
    settings = (left, right)
    metrics = ("raw_i_perp", "densified_delta_e_kcal_per_mol")
    for column, setting in enumerate(settings):
        for row, metric in enumerate(metrics):
            axis = axes[row, column]
            for alpha in ALPHAS:
                draw_curve(axis, setting["curves"][alpha][metric], alpha, metric)
            style_axis(axis, metric)

    axes[0, 0].set_title(
        "Tighter inner solves with reparameterization\n"
        r"($r=0.1$, $m_{\max}=5000$; cap nonbinding)",
        pad=6,
    )
    axes[0, 1].set_title(
        "Production inner solves with reparameterization\n"
        r"($r=0.7$, $m_{\max}=5$)",
        pad=6,
    )

    handles = [
        Line2D(
            [0], [0],
            color=ALPHA_COLORS[index],
            linewidth=1.2,
            marker=ALPHA_MARKERS[index],
            markersize=3.5,
            markerfacecolor=ALPHA_COLORS[index],
            markeredgecolor="#202020",
            markeredgewidth=0.28,
            label=alpha_label(alpha),
        )
        for index, alpha in enumerate(ALPHAS)
    ]
    handles.append(
        Line2D(
            [0], [0], color="#686868", linestyle=":", linewidth=0.7,
            label=r"$\Delta E_{\mathrm{ref}}$",
        )
    )
    figure.legend(
        handles=handles,
        title=r"Fixed proximal parameter $\alpha$",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=9,
        frameon=False,
        handlelength=1.7,
        handletextpad=0.4,
        columnspacing=0.85,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.105,
        top=0.80,
        wspace=0.18,
        hspace=0.35,
    )
    figure.canvas.draw()
    for axis, label in zip(axes.ravel(), "abcd"):
        panel_label(figure, axis, label)
    figure.savefig(
        output,
        format="pdf",
        dpi=600,
        metadata={
            "Title": "Alanine 66D fixed-alpha cost-to-precision comparison",
            "Subject": (
                "Same reparameterized IVM; r=0.1 with a nonbinding inner cap "
                "versus production r=0.7 and m_max=5; raw I_perp and actual "
                "densified Delta E"
            ),
        },
    )
    plt.close(figure)


def curve_rows(settings: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    panel_by_setting_metric = {
        ("tighter_inner_solves", "raw_i_perp"): "a",
        ("production_inner_solves", "raw_i_perp"): "b",
        ("tighter_inner_solves", "densified_delta_e_kcal_per_mol"): "c",
        ("production_inner_solves", "densified_delta_e_kcal_per_mol"): "d",
    }
    rows = []
    for setting in settings:
        setting_name = str(setting["setting"])
        for alpha in ALPHAS:
            for metric in ("raw_i_perp", "densified_delta_e_kcal_per_mol"):
                records = setting["curves"][alpha][metric]
                for row in records:
                    rows.append(
                        {
                            "panel": panel_by_setting_metric[(setting_name, metric)],
                            "inner_solve_setting": setting_name,
                            "alpha": f"{alpha:g}",
                            "residual_ratio": setting["specification"]["residual_ratio"],
                            "inner_max_steps": setting["specification"]["inner_max_steps"],
                            "metric": metric,
                            "metric_value": row[metric],
                            "running_best_metric": row["running_best_metric"],
                            "minimum_NF": row["optimization_nf"],
                            "trajectory_index": row["trajectory_index"],
                            "stored_call": row["stored_call"],
                            "state_id": row["state_id"],
                            "path_sha256": row["path_sha256"],
                            "segment_length_cv": row["segment_length_cv"],
                            "segment_length_ratio": row["segment_length_ratio"],
                            "result_file": row["result_file"],
                            "result_sha256": row["result_sha256"],
                            "metric_task_id": row["metric_task_id"],
                            "metric_artifact": row["metric_artifact"],
                            "metric_artifact_sha256": row["metric_artifact_sha256"],
                            "dense_subdivisions": (
                                row["dense_subdivisions"]
                                if metric == "densified_delta_e_kcal_per_mol"
                                else ""
                            ),
                            "dense_final_level_difference_kcal_per_mol": (
                                row["dense_final_level_difference_kcal_per_mol"]
                                if metric == "densified_delta_e_kcal_per_mol"
                                else ""
                            ),
                            "formal_complete": True,
                            "mesh_eligible": True,
                            "interpolation_used": False,
                            "extrapolation_used": False,
                            "carry_forward_used": False,
                            "diagnostic_calls_charged_to_NF": False,
                        }
                    )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise FigureInputError("No plotted records were produced")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def public_setting(setting: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in setting.items()
        if key != "curves"
    }


def validate_frozen_right_inputs() -> None:
    for path, expected in RIGHT_EXPECTED_SHA256.items():
        if sha256(path) != expected:
            raise FigureInputError(f"Frozen formal right-column input changed: {path}")


def main() -> None:
    args = parse_args()
    validate_frozen_right_inputs()
    left_optimizer = validate_frozen_left_optimizer()
    left = load_setting(
        args.left_metric_preregistration,
        args.left_metric_aggregate,
        LEFT_SPEC,
        "tighter_inner_solves",
        left_optimizer["runs_by_alpha"],
    )
    right = load_setting(
        RIGHT_PREREGISTRATION,
        RIGHT_AGGREGATE,
        RIGHT_SPEC,
        "production_inner_solves",
    )
    summary = {
        "left_runs": len(left["runs"]),
        "right_runs": len(right["runs"]),
        "alpha_grid": [f"{alpha:g}" for alpha in ALPHAS],
        "left_inner_cap_nonbinding": all(
            run["inner_cap_stop_evidence"]["inner_cap_binding"] is False
            for run in left["runs"]
        ),
        "reference_barrier_kcal_per_mol": REFERENCE_BARRIER_KCAL_PER_MOL,
    }
    if args.validate_only:
        print(json.dumps(summary, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / f"{STEM}.pdf"
    csv_path = args.output_dir / f"{STEM}_source.csv"
    manifest_path = args.output_dir / f"{STEM}_manifest.json"
    existing = [path for path in (pdf_path, csv_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    rows = curve_rows((left, right))
    draw_figure(left, right, pdf_path)
    write_csv(csv_path, rows)
    manifest = {
        "status": "generated_from_completed_registered_metrics",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Fixed-alpha cost-to-precision comparison under the same "
            "reparameterized Ala IVM algorithm"
        ),
        "alpha_grid": [f"{alpha:g}" for alpha in ALPHAS],
        "columns": {
            "left": public_setting(left),
            "right": public_setting(right),
        },
        "left_optimizer_campaign": left_optimizer,
        "rows": {
            "top": "raw I_perp",
            "bottom": "actual densified Delta E in kcal/mol",
        },
        "cost_to_precision_definition": {
            "eligible_paths": (
                "metric-preregistered formal-complete and common-mesh-eligible "
                "saved paths only"
            ),
            "equal_NF_rule": "retain the lowest actually observed metric value",
            "curve_vertices": "strictly decreasing observed records",
            "interpolation": False,
            "extrapolation": False,
            "carry_forward_to_budget": False,
            "optimization_cost_source": "total_internal_joint_evaluations",
            "fixed_endpoints_charged": False,
            "offline_dense_diagnostics_charged": False,
        },
        "densified_delta_e": {
            "quantity": "actual barrier, not reference error",
            "reference_barrier_kcal_per_mol": REFERENCE_BARRIER_KCAL_PER_MOL,
            "reference_line_drawn": True,
            "left_undershoot_gate": (
                "fail if any eligible left path has Delta E below Delta E_ref "
                "beyond 1e-12 kcal/mol"
            ),
            "piecewise_linear_cartesian_subdivisions_per_segment": DENSE_SUBDIVISIONS,
            "convergence_tolerance_kcal_per_mol": DENSE_CONVERGENCE_TOL,
        },
        "common_axes": {
            "number_of_forces": {"scale": "log", "limits": list(NF_YLIM)},
            "raw_i_perp": {"scale": "log reversed", "limits": list(IPERP_XLIM)},
            "densified_delta_e": {
                "scale": "log reversed", "limits": list(BARRIER_XLIM)
            },
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
                **summary,
                "pdf": str(pdf_path),
                "source_csv": str(csv_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
