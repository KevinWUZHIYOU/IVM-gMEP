#!/usr/bin/env python3
"""Candidate Chignolin IR-IVM comparison against the frozen article V9 data.

This is intentionally a thin, isolated plotter.  It does not import or edit
the historical V2/V3 plotters.  The baseline summary and its hashes are frozen
in a separate comparison contract, while the candidate IR-IVM optimization
and record-indexed dense results are supplied on the command line.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    EXPERIMENT_ROOT / "registrations" / "FIGURE_COMPARISON_CONTRACT.json"
)
ACTIVE_CANDIDATE_ROOT = EXPERIMENT_ROOT / "figures" / "active_candidates"

BASELINE_ORDER = (
    "IVM",
    "ODE-NEB",
    "Static-NEB",
    "Original String",
    "OBSM-FIRE",
)
VALID_MODES = ("overlay", "replace-current-IVM")
IR_COMPLETE_STATE_DEFINITION = "initial_or_reparam_committed"

COLORS = {
    "IR-IVM": "#0072B2",
    "IVM": "#56B4E9",
    "ODE-NEB": "#D55E00",
    "Static-NEB": "#B279A2",
    "Original String": "#777777",
    "OBSM-FIRE": "#5B9279",
}
LINESTYLES = {
    "IR-IVM": "-",
    "IVM": "--",
    "ODE-NEB": "-",
    "Static-NEB": "--",
    "Original String": ":",
    "OBSM-FIRE": "-.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    actual = sha256_file(resolved)
    if actual != str(expected):
        raise ValueError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )
    return resolved


def scalar(saved, key: str):
    return np.asarray(saved[key]).item()


def complete_ir_mask(
    reparam_committed: np.ndarray, complete_state_definition: str
) -> np.ndarray:
    if str(complete_state_definition) != IR_COMPLETE_STATE_DEFINITION:
        raise ValueError(
            "IR-IVM complete-state contract is not "
            f"{IR_COMPLETE_STATE_DEFINITION!r}"
        )
    committed = np.asarray(reparam_committed, dtype=bool)
    if committed.ndim != 1 or not len(committed):
        raise ValueError("reparam_committed must be a non-empty vector")
    complete = committed.copy()
    complete[0] = True
    return complete


def validate_total_nf(counter_vectors: np.ndarray) -> np.ndarray:
    counters = np.asarray(counter_vectors)
    if counters.ndim != 2 or counters.shape[1] != 22:
        raise ValueError("counter_vectors must contain all 22 path images")
    integer = np.asarray(counters, dtype=np.int64)
    if not np.array_equal(counters, integer) or np.any(integer < 0):
        raise ValueError("counter_vectors must be non-negative exact integers")
    if np.any(integer[:, 0] < 1) or np.any(integer[:, -1] < 1):
        raise ValueError("fixed endpoint first evaluations are missing")
    total_nf = integer.sum(axis=1, dtype=np.int64)
    if len(total_nf) > 1 and np.any(np.diff(total_nf) < 0):
        raise ValueError("total Number of Forces is not cumulative")
    return total_nf


def first_passage(
    number_of_forces: np.ndarray, values: np.ndarray, threshold: float
) -> Optional[int]:
    cost = np.asarray(number_of_forces, dtype=np.int64)
    metric = np.asarray(values, dtype=float)
    if cost.shape != metric.shape:
        raise ValueError("cost and metric vectors differ in shape")
    order = np.argsort(cost, kind="stable")
    running_best = np.minimum.accumulate(metric[order])
    reached = np.flatnonzero(running_best <= float(threshold))
    return None if not reached.size else int(cost[order][reached[0]])


def pareto_frontier(
    number_of_forces: np.ndarray,
    values: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(number_of_forces, kind="stable")
    costs = np.asarray(number_of_forces, dtype=float)[order]
    metric = np.asarray(values, dtype=float)[order]
    frontier_metric = []
    frontier_cost = []
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
                    frontier_metric[-1] = running_best
                else:
                    frontier_metric.append(running_best)
                    frontier_cost.append(float(cost))
    return (
        np.asarray(frontier_metric, dtype=float),
        np.asarray(frontier_cost, dtype=float),
    )


def validate_counter_audit(
    audit_path: Path, expected_hash: str, final_counter_vector: np.ndarray
) -> Dict:
    path = require_hash(audit_path, expected_hash, "optimization audit")
    payload = json.loads(path.read_text())
    per_image = np.asarray(
        payload.get("per_image_joint_evaluations", []), dtype=np.int64
    )
    if (
        not payload.get("consistent", False)
        or int(payload.get("backend_raw_joint_evaluations", -1))
        != int(payload.get("sum_image_joint_evaluations", -2))
        or per_image.shape != (22,)
        or not np.array_equal(
            per_image, np.asarray(final_counter_vector, dtype=np.int64)
        )
    ):
        raise ValueError(f"counter audit failed: {path}")
    return payload


def load_baseline_run(
    method: str, record: Dict, total_nf_limit: int, maximum_call: int
) -> Dict:
    run_path = require_hash(record["run"], record["run_sha256"], f"{method} run")
    with np.load(run_path, allow_pickle=False) as saved:
        counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
        total_nf = validate_total_nf(counters)
        calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        if not (len(total_nf) == len(calls) == len(i_perp)):
            raise ValueError(f"misaligned history in {run_path}")
        if method == "IVM":
            if {
                "state_free_polish_active",
                "reparam_committed",
            }.issubset(saved.files):
                free_polish = np.asarray(
                    saved["state_free_polish_active"], dtype=bool
                )
                spring_committed = np.asarray(
                    saved["reparam_committed"], dtype=bool
                )
                complete = (~free_polish) | spring_committed
            else:
                complete = np.ones(len(total_nf), dtype=bool)
            keep = (total_nf <= int(total_nf_limit)) & complete
        else:
            keep = calls <= int(maximum_call)
        initial_hash = str(scalar(saved, "initial_path_sha256"))
        tangent = str(scalar(saved, "tangent_definition"))
    if not np.any(keep):
        raise ValueError(f"no eligible records in {run_path}")
    validate_counter_audit(
        Path(record["run_audit"]),
        record["run_audit_sha256"],
        counters[-1],
    )
    return {
        "path": run_path,
        "result_sha256": record["run_sha256"],
        "number_of_forces": total_nf[keep],
        "calls_per_image": calls[keep],
        "i_perp": i_perp[keep],
        "initial_path_sha256": initial_hash,
        "tangent_definition": tangent,
    }


def load_baseline_dense(method: str, record: Dict, run: Dict) -> Dict:
    dense_path = require_hash(
        record["dense"], record["dense_sha256"], f"{method} dense result"
    )
    audit_path = require_hash(
        record["dense_audit"],
        record["dense_audit_sha256"],
        f"{method} dense audit",
    )
    audit = json.loads(audit_path.read_text())
    if (
        not audit.get("counts_excluded_from_optimization_axis", False)
        or int(audit["initial_subdivisions"]) != 10
        or int(audit["comparison_subdivisions"]) != 20
        or not np.isclose(float(audit["adaptive_threshold_kcal"]), 0.05)
        or int(audit["maximum_subdivisions"]) != 80
    ):
        raise ValueError(f"dense audit protocol mismatch: {audit_path}")
    with np.load(dense_path, allow_pickle=False) as saved:
        source = Path(str(scalar(saved, "source_result"))).resolve()
        if source != run["path"]:
            raise ValueError(f"{method} dense source-result mismatch")
        if str(scalar(saved, "source_initial_path_sha256")) != run[
            "initial_path_sha256"
        ]:
            raise ValueError(f"{method} dense initial-path mismatch")
        calls = np.asarray(saved["calls"], dtype=np.int64)
        barrier = np.asarray(saved["adaptive_barrier_kcal"], dtype=float)
    number_of_forces = []
    for call in calls:
        indices = np.flatnonzero(run["calls_per_image"] == int(call))
        if not indices.size:
            raise ValueError(
                f"{method} dense call {int(call)} is outside the plotted run"
            )
        number_of_forces.append(int(run["number_of_forces"][indices[-1]]))
    return {
        "path": dense_path,
        "number_of_forces": np.asarray(number_of_forces, dtype=np.int64),
        "barrier_kcal": barrier,
    }


def load_baselines(summary: Dict, contract: Dict) -> Tuple[Dict, Dict]:
    methods = summary.get("methods", {})
    if set(BASELINE_ORDER).difference(methods):
        raise ValueError("baseline summary does not contain every article method")
    runs = {}
    dense = {}
    for method in BASELINE_ORDER:
        runs[method] = load_baseline_run(
            method,
            methods[method],
            int(contract["total_nf_limit_for_current_ivm"]),
            int(contract["non_ivm_max_calls_per_image"]),
        )
        dense[method] = load_baseline_dense(
            method, methods[method], runs[method]
        )
    initial_hashes = {
        run["initial_path_sha256"] for run in runs.values()
    }
    if initial_hashes != {str(contract["initial_path_sha256"])}:
        raise ValueError(f"baseline initial paths differ: {initial_hashes}")
    if any(
        "central difference" not in run["tangent_definition"].lower()
        for run in runs.values()
    ):
        raise ValueError("a baseline does not use the article central tangent")
    return runs, dense


def load_ir_run(result_path: Path, contract: Dict) -> Dict:
    path = Path(result_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing IR-IVM result: {path}")
    result_sha256 = sha256_file(path)
    with np.load(path, allow_pickle=False) as saved:
        required = set(contract["required_ir_result_arrays"])
        missing = sorted(required.difference(saved.files))
        if missing:
            raise ValueError("IR-IVM result is missing: " + ",".join(missing))
        counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
        total_nf = validate_total_nf(counters)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        complete = complete_ir_mask(
            committed,
            str(scalar(saved, "total_internal_nf_complete_state_definition")),
        )
        if i_perp.shape != total_nf.shape or committed.shape != total_nf.shape:
            raise ValueError("IR-IVM history arrays differ in length")
        keep = complete & (
            total_nf <= int(contract["total_nf_limit_for_current_ivm"])
        )
        if not np.any(keep & (np.arange(len(keep)) != 0)):
            raise ValueError("IR-IVM has no committed reparameterized state")
        invariants = {
            "reparam_mode": str(scalar(saved, "reparam_mode")),
            "reparam_every": int(scalar(saved, "reparam_every")),
            "state_curve_direct_solve": bool(
                scalar(saved, "state_curve_direct_solve")
            ),
            "state_curve_direct_objective_acceptance": str(
                scalar(saved, "state_curve_direct_objective_acceptance")
            ),
            "state_globalization": str(scalar(saved, "state_globalization")),
            "frozen_subproblem": str(scalar(saved, "frozen_subproblem")),
            "eta": float(scalar(saved, "discrete_ffrbk_selector_eta")),
        }
        expected = contract["ir_algorithm_gates"]
        for key, value in expected.items():
            if isinstance(value, float):
                valid = np.isclose(float(invariants[key]), value)
            else:
                valid = invariants[key] == value
            if not valid:
                raise ValueError(
                    f"IR-IVM algorithm gate {key} failed: "
                    f"expected {value!r}, got {invariants[key]!r}"
                )
        initial_hash = str(scalar(saved, "initial_path_sha256"))
        tangent = str(scalar(saved, "tangent_definition"))
        parameters = {
            "alpha": float(scalar(saved, "alpha")),
            "r": float(scalar(saved, "discrete_residual_ratio")),
            "m_max": int(scalar(saved, "discrete_inner_max_steps")),
        }
        attempted_record_count = int(
            np.count_nonzero(np.asarray(saved["reparam_attempted"], dtype=bool))
        )
    if initial_hash != str(contract["initial_path_sha256"]):
        raise ValueError("IR-IVM and article baselines use different initial paths")
    if "central difference" not in tangent.lower():
        raise ValueError("IR-IVM tangent differs from the article definition")
    audit_path = path.with_suffix(".audit.json")
    audit_sha256 = sha256_file(audit_path)
    audit = validate_counter_audit(audit_path, audit_sha256, counters[-1])
    reparam_attempt_count = int(audit.get("state_reparam_attempts", -1))
    reparam_accept_count = int(audit.get("state_reparam_accepts", -1))
    if (
        reparam_attempt_count < reparam_accept_count
        or reparam_accept_count != int(np.count_nonzero(committed))
        or attempted_record_count < reparam_attempt_count
    ):
        raise ValueError("IR-IVM reparameterization event audit failed")
    return {
        "path": path,
        "result_sha256": result_sha256,
        "audit_path": audit_path,
        "audit_sha256": audit_sha256,
        "raw_counter_vectors": counters,
        "raw_total_nf": total_nf,
        "raw_i_perp": i_perp,
        "raw_reparam_committed": committed,
        "complete_mask": complete,
        "number_of_forces": total_nf[keep],
        "i_perp": i_perp[keep],
        "source_record_indices": np.flatnonzero(keep),
        "initial_path_sha256": initial_hash,
        "parameters": parameters,
        "reparam_attempt_count": reparam_attempt_count,
        "reparam_accept_count": reparam_accept_count,
        "reparam_attempted_record_count": attempted_record_count,
        "invariants": invariants,
    }


def validate_record_indexed_dense_arrays(
    *,
    record_index: np.ndarray,
    paper_total_nf: np.ndarray,
    i_perp: np.ndarray,
    selected_counter_vectors: np.ndarray,
    complete_state_kind: np.ndarray,
    reparam_committed: np.ndarray,
    raw_counter_vectors: np.ndarray,
    raw_total_nf: np.ndarray,
    raw_i_perp: np.ndarray,
    raw_reparam_committed: np.ndarray,
) -> None:
    indices = np.asarray(record_index, dtype=np.int64)
    nf = np.asarray(paper_total_nf, dtype=np.int64)
    metric = np.asarray(i_perp, dtype=float)
    kinds = np.asarray(complete_state_kind).astype(str)
    committed = np.asarray(reparam_committed, dtype=bool)
    selected_counters = np.asarray(selected_counter_vectors, dtype=np.int64)
    n = len(indices)
    if not all(
        len(array) == n
        for array in (nf, metric, kinds, committed, selected_counters)
    ):
        raise ValueError("record-indexed dense arrays differ in length")
    if n == 0 or (n > 1 and np.any(np.diff(indices) <= 0)):
        raise ValueError("dense record indices must be non-empty and increasing")
    if np.any(indices < 0) or np.any(indices >= len(raw_total_nf)):
        raise ValueError("dense record index is outside the source history")
    eligible = (indices == 0) | raw_reparam_committed[indices]
    if not np.all(eligible):
        raise ValueError("dense result includes an uncommitted IR-IVM state")
    expected_kinds = np.asarray(
        ["initial" if index == 0 else "reparam_committed" for index in indices]
    )
    if (
        not np.array_equal(nf, raw_total_nf[indices])
        or not np.allclose(metric, raw_i_perp[indices], rtol=0.0, atol=0.0)
        or not np.array_equal(
            selected_counters, raw_counter_vectors[indices]
        )
        or not np.array_equal(kinds, expected_kinds)
        or not np.array_equal(committed, raw_reparam_committed[indices])
    ):
        raise ValueError("record-indexed dense mapping does not match source NPZ")


def load_ir_dense(dense_path: Path, ir_run: Dict, contract: Dict) -> Dict:
    path = Path(dense_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing record-indexed IR dense result: {path}")
    dense_sha256 = sha256_file(path)
    audit_path = path.with_suffix(".audit.json")
    if not audit_path.is_file():
        raise FileNotFoundError(f"missing record-indexed dense audit: {audit_path}")
    audit = json.loads(audit_path.read_text())
    if audit.get("output_sha256") != dense_sha256:
        raise ValueError("record-indexed dense output hash audit failed")
    if (
        audit.get("source_result_sha256") != ir_run["result_sha256"]
        or audit.get("source_optimization_audit_sha256")
        != ir_run["audit_sha256"]
        or audit.get("complete_state_definition")
        != IR_COMPLETE_STATE_DEFINITION
        or not audit.get("all_selected_records_eligible", False)
        or not audit.get(
            "diagnostic_counts_excluded_from_optimization_axis", False
        )
    ):
        raise ValueError("record-indexed dense provenance audit failed")
    with np.load(path, allow_pickle=False) as saved:
        required = set(contract["required_ir_dense_arrays"])
        missing = sorted(required.difference(saved.files))
        if missing:
            raise ValueError("IR dense result is missing: " + ",".join(missing))
        if str(scalar(saved, "source_result_sha256")) != ir_run[
            "result_sha256"
        ]:
            raise ValueError("IR dense source result hash mismatch")
        if str(scalar(saved, "source_optimization_audit_sha256")) != ir_run[
            "audit_sha256"
        ]:
            raise ValueError("IR dense source audit hash mismatch")
        if str(scalar(saved, "complete_state_definition")) != (
            IR_COMPLETE_STATE_DEFINITION
        ):
            raise ValueError("IR dense complete-state definition mismatch")
        if not bool(scalar(saved, "paper_total_nf_includes_all_22_images")):
            raise ValueError("IR dense cost does not include all 22 images")
        if not bool(
            scalar(saved, "diagnostic_counts_excluded_from_paper_total_nf")
        ):
            raise ValueError("IR dense diagnostic calls contaminate paper NF")
        if (
            int(scalar(saved, "initial_subdivisions")) != 10
            or int(scalar(saved, "comparison_subdivisions")) != 20
            or not np.isclose(float(scalar(saved, "adaptive_threshold_kcal")), 0.05)
            or int(scalar(saved, "maximum_subdivisions")) != 80
        ):
            raise ValueError("IR dense barrier protocol mismatch")
        indices = np.asarray(saved["record_index"], dtype=np.int64)
        nf = np.asarray(saved["paper_total_nf"], dtype=np.int64)
        dense_i_perp = np.asarray(saved["i_perp"], dtype=float)
        barrier = np.asarray(
            saved["adaptive_dense_barrier_kcal"], dtype=float
        )
        validate_record_indexed_dense_arrays(
            record_index=indices,
            paper_total_nf=nf,
            i_perp=dense_i_perp,
            selected_counter_vectors=saved["selected_counter_vectors"],
            complete_state_kind=saved["complete_state_kind"],
            reparam_committed=saved["reparam_committed"],
            raw_counter_vectors=ir_run["raw_counter_vectors"],
            raw_total_nf=ir_run["raw_total_nf"],
            raw_i_perp=ir_run["raw_i_perp"],
            raw_reparam_committed=ir_run["raw_reparam_committed"],
        )
    if not np.all(np.isfinite(barrier)):
        raise ValueError("IR dense barriers contain non-finite values")
    keep = nf <= int(contract["total_nf_limit_for_current_ivm"])
    if not np.any(keep):
        raise ValueError("IR dense result has no point inside the paper NF window")
    return {
        "path": path,
        "dense_sha256": dense_sha256,
        "audit_path": audit_path,
        "audit_sha256": sha256_file(audit_path),
        "source_record_indices": indices[keep],
        "number_of_forces": nf[keep],
        "i_perp": dense_i_perp[keep],
        "barrier_kcal": barrier[keep],
    }


def load_contract(path: Path) -> Tuple[Dict, Dict]:
    contract_path = Path(path).expanduser().resolve()
    contract = json.loads(contract_path.read_text())
    if contract.get("allowed_modes") != list(VALID_MODES):
        raise ValueError("comparison contract modes changed")
    baseline = contract["baseline_summary"]
    baseline_path = require_hash(
        baseline["path"], baseline["sha256"], "article V9 summary"
    )
    require_hash(
        contract["source_figure"]["path"],
        contract["source_figure"]["sha256"],
        "article V9 source figure",
    )
    annotations = require_hash(
        contract["annotation_table"]["path"],
        contract["annotation_table"]["sha256"],
        "article V9 annotation table",
    )
    summary = json.loads(baseline_path.read_text())
    if summary.get("primary_cost_axis") != "counter_vectors.sum(axis=1)":
        raise ValueError("article summary does not use total Number of Forces")
    if not summary.get("cost_includes_all_22_images", False):
        raise ValueError("article summary excludes path images from its cost")
    validate_annotation_contract(annotations, contract["fixed_thresholds"])
    return contract, summary


def validate_annotation_contract(path: Path, fixed_thresholds: Dict) -> None:
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for metric, gate in fixed_thresholds.items():
        selected = [row for row in rows if row["metric"] == metric]
        if {row["baseline"] for row in selected} != set(
            gate["baseline_first_passage_total_nf"]
        ):
            raise ValueError(f"annotation baselines changed for {metric}")
        for row in selected:
            baseline = row["baseline"]
            if (
                not np.isclose(
                    float(row["shared_precision_threshold"]),
                    float(gate["value"]),
                )
                or int(row["ivm_number_of_force_evaluations"])
                != int(gate["current_ivm_first_passage_total_nf"])
                or int(row["baseline_number_of_force_evaluations"])
                != int(gate["baseline_first_passage_total_nf"][baseline])
            ):
                raise ValueError(f"frozen annotation gate changed for {metric}")


def validate_frozen_first_passages(
    runs: Dict, dense: Dict, contract: Dict
) -> None:
    for metric, gate in contract["fixed_thresholds"].items():
        source = runs if metric == "i_perp" else dense
        key = "i_perp" if metric == "i_perp" else "barrier_kcal"
        current = first_passage(
            source["IVM"]["number_of_forces"],
            source["IVM"][key],
            float(gate["value"]),
        )
        if current != int(gate["current_ivm_first_passage_total_nf"]):
            raise ValueError(f"current IVM first passage changed for {metric}")
        for baseline, expected in gate[
            "baseline_first_passage_total_nf"
        ].items():
            observed = first_passage(
                source[baseline]["number_of_forces"],
                source[baseline][key],
                float(gate["value"]),
            )
            if observed != int(expected):
                raise ValueError(
                    f"{baseline} first passage changed for {metric}"
                )


def ensure_active_candidate_output(path: Path) -> Path:
    output = Path(path).expanduser().resolve()
    root = ACTIVE_CANDIDATE_ROOT.resolve()
    try:
        output.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"output must remain under active_candidates: {root}"
        ) from error
    return output


def method_order(mode: str) -> Tuple[str, ...]:
    if mode == "overlay":
        return ("IR-IVM",) + BASELINE_ORDER
    if mode == "replace-current-IVM":
        return ("IR-IVM",) + BASELINE_ORDER[1:]
    raise ValueError(f"unsupported comparison mode: {mode}")


def build_validation_report(
    *,
    mode: str,
    output_dir: Path,
    contract_path: Path,
    contract: Dict,
    ir_run: Dict,
    ir_dense: Dict,
    runs: Dict,
    dense: Dict,
) -> Dict:
    thresholds = contract["fixed_thresholds"]
    report = {
        "status": "validated",
        "dry_run_writes_files": False,
        "mode": mode,
        "method_order": list(method_order(mode)),
        "candidate_output_directory": str(output_dir),
        "output_is_under_active_candidates": True,
        "comparison_contract": str(Path(contract_path).resolve()),
        "comparison_contract_sha256": sha256_file(contract_path),
        "baseline_summary": contract["baseline_summary"],
        "ir_ivm": {
            "result": str(ir_run["path"]),
            "result_sha256": ir_run["result_sha256"],
            "audit": str(ir_run["audit_path"]),
            "audit_sha256": ir_run["audit_sha256"],
            "parameters": ir_run["parameters"],
            "algorithm_invariants": ir_run["invariants"],
            "reparam_attempt_count": ir_run["reparam_attempt_count"],
            "reparam_accept_count": ir_run["reparam_accept_count"],
            "reparam_attempted_record_count": ir_run[
                "reparam_attempted_record_count"
            ],
            "plotted_complete_state_count": int(len(ir_run["i_perp"])),
            "plotted_state_definition": IR_COMPLETE_STATE_DEFINITION,
            "last_plotted_total_nf": int(ir_run["number_of_forces"][-1]),
            "last_plotted_i_perp": float(ir_run["i_perp"][-1]),
            "dense_result": str(ir_dense["path"]),
            "dense_result_sha256": ir_dense["dense_sha256"],
            "dense_audit": str(ir_dense["audit_path"]),
            "dense_audit_sha256": ir_dense["audit_sha256"],
            "dense_record_indices": ir_dense[
                "source_record_indices"
            ].tolist(),
        },
        "frozen_thresholds": {},
    }
    for metric, gate in thresholds.items():
        ir_source = ir_run if metric == "i_perp" else ir_dense
        key = "i_perp" if metric == "i_perp" else "barrier_kcal"
        report["frozen_thresholds"][metric] = {
            "value": float(gate["value"]),
            "ir_ivm_first_passage_total_nf": first_passage(
                ir_source["number_of_forces"],
                ir_source[key],
                float(gate["value"]),
            ),
            "current_ivm_first_passage_total_nf": int(
                gate["current_ivm_first_passage_total_nf"]
            ),
            "baseline_first_passage_total_nf": gate[
                "baseline_first_passage_total_nf"
            ],
        }
    joint_gate = (
        (ir_dense["i_perp"] <= float(thresholds["i_perp"]["value"]))
        & (
            ir_dense["barrier_kcal"]
            <= float(thresholds["densified_barrier_kcal"]["value"])
        )
    )
    joint_reached = np.flatnonzero(joint_gate)
    report["joint_registered_thresholds"] = {
        "ir_ivm_first_passage_total_nf": (
            None
            if not joint_reached.size
            else int(
                np.min(ir_dense["number_of_forces"][joint_reached])
            )
        ),
        "current_ivm_first_passage_total_nf": int(
            contract["current_ivm_joint_first_passage_total_nf"]
        ),
        "requires_same_committed_record": True,
    }
    return report


def setup_style(mpl) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def write_csv(path: Path, rows: List[Dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_candidate_plot(
    *,
    mode: str,
    output_dir: Path,
    contract: Dict,
    validation_report: Dict,
    runs: Dict,
    dense: Dict,
) -> List[str]:
    import matplotlib as mpl  # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    setup_style(mpl)
    order = method_order(mode)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    frontier_rows = []
    for method in order:
        for axis, source, key, bounds, metric in (
            (axes[0], runs, "i_perp", (1e-2, 10 ** 4.5), "i_perp"),
            (
                axes[1],
                dense,
                "barrier_kcal",
                (19.5, 100.0),
                "densified_barrier_kcal",
            ),
        ):
            frontier, costs = pareto_frontier(
                source[method]["number_of_forces"],
                source[method][key],
                bounds[0],
                bounds[1],
            )
            axis.plot(
                frontier,
                costs,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                lw=2.25 if method == "IR-IVM" else 1.7 if method == "ODE-NEB" else 1.15,
                marker="o" if len(frontier) == 1 else None,
                markersize=3.0,
                alpha=1.0 if method in ("IR-IVM", "ODE-NEB") else 0.9,
                zorder=7 if method == "IR-IVM" else 5 if method == "ODE-NEB" else 3,
                label=method,
            )
            for threshold, cost in zip(frontier, costs):
                frontier_rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "measured_record_threshold": float(threshold),
                        "number_of_force_evaluations": int(cost),
                    }
                )

    for index, (axis, xlabel, metric) in enumerate(
        (
            (axes[0], r"$I_\perp$ threshold", "i_perp"),
            (
                axes[1],
                r"Densified barrier threshold (kcal mol$^{-1}$)",
                "densified_barrier_kcal",
            ),
        )
    ):
        if index == 0:
            axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_ylim(
            0.0,
            1.10
            * max(
                int(runs[method]["number_of_forces"][-1])
                for method in order
            ),
        )
        axis.axvline(
            float(contract["fixed_thresholds"][metric]["value"]),
            color="#333333",
            lw=0.7,
            linestyle=(0, (2, 2)),
            alpha=0.7,
            zorder=1,
        )
        axis.yaxis.set_major_formatter(
            mpl.ticker.StrMethodFormatter("{x:,.0f}")
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Number of Forces", labelpad=4)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(
            which="major", color="#D9D9D9", lw=0.45, alpha=0.65, zorder=0
        )
        axis.tick_params(direction="out", length=3.0, width=0.7)
        axis.margins(x=0.035)
        axis.text(
            -0.02,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9.5,
            va="bottom",
        )
        if mode == "overlay":
            gate = validation_report["frozen_thresholds"][metric]
            ir_nf = gate["ir_ivm_first_passage_total_nf"]
            ivm_nf = gate["current_ivm_first_passage_total_nf"]
            if ir_nf is not None:
                threshold = float(contract["fixed_thresholds"][metric]["value"])
                arrow_x = threshold * 0.78 if index == 0 else threshold + 2.0
                axis.annotate(
                    "",
                    xy=(arrow_x, ir_nf),
                    xytext=(arrow_x, ivm_nf),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "#222222",
                        "lw": 0.9,
                        "mutation_scale": 7.0,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                    zorder=8,
                )
                axis.annotate(
                    f"{ivm_nf / ir_nf:.2f}x\nvs IVM",
                    xy=(arrow_x, 0.5 * (ivm_nf + ir_nf)),
                    xytext=(6, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=7.0,
                    color="#222222",
                    zorder=9,
                )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=len(order),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        columnspacing=1.05,
        handlelength=2.5,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.21,
        top=0.79,
        wspace=0.34,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"chignolin_ir_ivm_{mode.replace('-', '_')}_total_nf"
    generated = []
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 600}),
    ):
        path = output_dir / f"{stem}{suffix}"
        figure.savefig(path, **kwargs)
        generated.append(str(path.resolve()))
    plt.close(figure)
    write_csv(output_dir / f"{stem}_frontier.csv", frontier_rows)
    validation_report["generated_candidate_files"] = generated
    summary_path = output_dir / f"{stem}_provenance.json"
    summary_path.write_text(json.dumps(validation_report, indent=2) + "\n")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-result", type=Path, required=True)
    parser.add_argument("--ir-dense", type=Path, required=True)
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all data and print provenance without writing files.",
    )
    args = parser.parse_args()

    output_dir = ensure_active_candidate_output(args.output_dir)
    contract, summary = load_contract(args.contract)
    runs, dense = load_baselines(summary, contract)
    validate_frozen_first_passages(runs, dense, contract)
    ir_run = load_ir_run(args.ir_result, contract)
    ir_dense = load_ir_dense(args.ir_dense, ir_run, contract)
    runs = dict(runs)
    dense = dict(dense)
    runs["IR-IVM"] = ir_run
    dense["IR-IVM"] = ir_dense
    report = build_validation_report(
        mode=args.mode,
        output_dir=output_dir,
        contract_path=args.contract,
        contract=contract,
        ir_run=ir_run,
        ir_dense=ir_dense,
        runs=runs,
        dense=dense,
    )
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    save_candidate_plot(
        mode=args.mode,
        output_dir=output_dir,
        contract=contract,
        validation_report=report,
        runs=runs,
        dense=dense,
    )


if __name__ == "__main__":
    main()
