#!/usr/bin/env python3
"""Read-only I_perp cost-to-precision audit for the formal single-run grid.

The analysis reads only optimization cost, I_perp, completeness, and mesh
diagnostics.  It never reads or computes an energy barrier.  Formal run files
and their preregistration are immutable inputs; the only write is a new JSON
report under ``reports/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


ANALYSIS_ID = "stage4_ala_discrete_single_iperp_ctp_readonly_v1"
EXPECTED_PROTOCOL_ID = "stage4_ala_discrete_single_fixedalpha_matched_v2"
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
SINGLE_ALLOWED_KEYS = frozenset(
    {
        "alpha",
        "alpha_mode",
        "configuration_sha256",
        "initial_path_sha256",
        "total_internal_joint_evaluations",
        "i_perp",
        "reparam_committed",
        "segment_length_cv",
        "segment_length_ratio",
    }
)
OFFICIAL_ARCHIVE_ALLOWED_KEYS = frozenset(
    {
        "policy_version",
        "source_result",
        "source_result_sha256",
        "activation_iperp",
        "complete_state_candidates_only",
        "candidate_calls",
        "candidate_i_perp",
    }
)
OFFICIAL_COMBINED_ALLOWED_KEYS = frozenset(
    {
        "phase_a_result",
        "phase_b_result",
        "phase_transition_force_offset",
    }
)
OFFICIAL_COMPONENT_ALLOWED_KEYS = frozenset(
    {
        "total_internal_joint_evaluations",
        "i_perp",
        "segment_length_cv",
        "segment_length_ratio",
    }
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(saved, key):
    return np.asarray(saved[key]).item()


def alpha_key(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:g}"


def _require_keys(saved, required, context: str) -> None:
    missing = sorted(set(required).difference(saved.files))
    if missing:
        raise RuntimeError(f"{context} is missing: {', '.join(missing)}")


def validate_curve_arrays(cost, i_perp, cv, ratio, context: str) -> None:
    lengths = {len(cost), len(i_perp), len(cv), len(ratio)}
    if len(lengths) != 1 or not len(cost):
        raise RuntimeError(f"{context} has inconsistent or empty arrays")
    if np.any(np.diff(cost) <= 0):
        raise RuntimeError(f"{context} costs are not strictly increasing")
    if not (
        np.all(np.isfinite(i_perp))
        and np.all(np.isfinite(cv))
        and np.all(np.isfinite(ratio))
    ):
        raise RuntimeError(f"{context} contains non-finite diagnostics")
    if np.any(i_perp < 0.0) or np.any(cv < 0.0) or np.any(ratio < 1.0):
        raise RuntimeError(f"{context} contains invalid diagnostics")


def eligible_single_points(
    cost: np.ndarray,
    i_perp: np.ndarray,
    committed: np.ndarray,
    cv: np.ndarray,
    ratio: np.ndarray,
) -> list[dict[str, object]]:
    cost = np.asarray(cost, dtype=int)
    i_perp = np.asarray(i_perp, dtype=float)
    committed = np.asarray(committed, dtype=bool)
    cv = np.asarray(cv, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    validate_curve_arrays(cost, i_perp, cv, ratio, "single-run curve")
    if committed.shape != cost.shape:
        raise RuntimeError("single-run completeness array has the wrong shape")
    complete = committed.copy()
    complete[0] = True
    mesh = (cv <= MESH_CV_MAX) & (ratio <= MESH_RATIO_MAX)
    eligible = complete & mesh
    return [
        {
            "NF": int(cost[index]),
            "i_perp": float(i_perp[index]),
            "segment_length_cv": float(cv[index]),
            "segment_length_ratio": float(ratio[index]),
            "is_initial": bool(index == 0),
            "reparam_committed": bool(committed[index]),
        }
        for index in np.flatnonzero(eligible)
    ]


def pareto_records(points: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(points, key=lambda row: int(row["NF"]))
    records: list[dict[str, object]] = []
    best = math.inf
    for row in ordered:
        value = float(row["i_perp"])
        if value < best:
            records.append(dict(row))
            best = value
    return records


def cost_to_precision(
    points: list[dict[str, object]], threshold: float
) -> int | None:
    costs = [
        int(row["NF"])
        for row in points
        if float(row["i_perp"]) <= float(threshold)
    ]
    return min(costs) if costs else None


def minimum_precision(points: list[dict[str, object]]) -> float | None:
    if not points:
        return None
    return min(float(row["i_perp"]) for row in points)


def ctp_row(
    candidate: list[dict[str, object]],
    comparator: list[dict[str, object]],
    threshold: float,
) -> dict[str, object]:
    candidate_cost = cost_to_precision(candidate, threshold)
    comparator_cost = cost_to_precision(comparator, threshold)
    if comparator_cost is None:
        outcome = "comparator_unreachable"
    elif candidate_cost is None:
        outcome = "candidate_unreachable_loss"
    elif candidate_cost < comparator_cost:
        outcome = "strict_win"
    elif candidate_cost == comparator_cost:
        outcome = "tie"
    else:
        outcome = "loss"
    return {
        "threshold_i_perp": float(threshold),
        "candidate_NF": candidate_cost,
        "comparator_NF": comparator_cost,
        "outcome": outcome,
        "candidate_minus_comparator_NF": (
            candidate_cost - comparator_cost
            if candidate_cost is not None and comparator_cost is not None
            else None
        ),
        "speedup_comparator_over_candidate": (
            float(comparator_cost) / float(candidate_cost)
            if candidate_cost is not None
            and comparator_cost is not None
            and candidate_cost > 0
            else None
        ),
    }


def compare_curves(
    candidate: list[dict[str, object]],
    comparator: list[dict[str, object]],
    comparator_name: str,
) -> dict[str, object]:
    comparator_pareto = pareto_records(comparator)
    threshold_rows = [
        ctp_row(candidate, comparator, float(row["i_perp"]))
        for row in comparator_pareto
    ]
    losses = [
        row
        for row in threshold_rows
        if row["outcome"] in {"loss", "candidate_unreachable_loss"}
    ]
    wins = [row for row in threshold_rows if row["outcome"] == "strict_win"]
    ties = [row for row in threshold_rows if row["outcome"] == "tie"]
    candidate_min = minimum_precision(candidate)
    comparator_min = minimum_precision(comparator)
    tightest_ok = bool(
        comparator_min is not None
        and candidate_min is not None
        and candidate_min <= comparator_min
    )
    weak = not losses and bool(threshold_rows)
    strict = bool(wins)
    finite_regrets = [
        int(row["candidate_minus_comparator_NF"])
        for row in losses
        if row["candidate_minus_comparator_NF"] is not None
    ]
    unreachable_losses = sum(
        row["outcome"] == "candidate_unreachable_loss" for row in losses
    )
    return {
        "comparator": comparator_name,
        "comparator_pareto_record_count": len(comparator_pareto),
        "threshold_results": threshold_rows,
        "weakly_no_slower_at_every_comparator_pareto_threshold": weak,
        "strictly_faster_at_least_one_threshold": strict,
        "candidate_tightest_eligible_i_perp": candidate_min,
        "comparator_tightest_eligible_i_perp": comparator_min,
        "candidate_does_not_lose_tightest_precision": tightest_ok,
        "full_curve_dominates": bool(weak and strict and tightest_ok),
        "near_miss": {
            "loss_threshold_count": len(losses),
            "candidate_unreachable_loss_count": int(unreachable_losses),
            "strict_win_count": len(wins),
            "tie_count": len(ties),
            "maximum_finite_positive_NF_regret": (
                max(finite_regrets) if finite_regrets else 0
            ),
            "sum_finite_positive_NF_regret": sum(finite_regrets),
            "tightest_i_perp_excess": (
                candidate_min - comparator_min
                if candidate_min is not None and comparator_min is not None
                else None
            ),
            "losses": losses,
        },
    }


def anchor_rows(
    candidate: list[dict[str, object]],
    comparator: list[dict[str, object]],
    anchors: list[float],
    domain_max: float | None = None,
) -> list[dict[str, object]]:
    rows = []
    for threshold in anchors:
        if domain_max is not None and float(threshold) > float(domain_max):
            rows.append(
                {
                    "threshold_i_perp": float(threshold),
                    "domain_status": "outside_comparator_post_activation_domain",
                    "candidate_NF": cost_to_precision(candidate, threshold),
                    "comparator_NF": None,
                    "outcome": "not_compared",
                }
            )
            continue
        row = ctp_row(candidate, comparator, threshold)
        row["domain_status"] = "inside_comparator_domain"
        rows.append(row)
    return rows


def load_single_curve(result: Path, expected_alpha: float) -> dict[str, object]:
    with np.load(result, allow_pickle=False) as saved:
        _require_keys(saved, SINGLE_ALLOWED_KEYS, str(result))
        alpha = float(scalar(saved, "alpha"))
        if alpha != float(expected_alpha):
            raise RuntimeError(f"alpha mismatch in {result}")
        if str(scalar(saved, "alpha_mode")) != "fixed":
            raise RuntimeError(f"non-fixed alpha in {result}")
        points = eligible_single_points(
            saved["total_internal_joint_evaluations"],
            saved["i_perp"],
            saved["reparam_committed"],
            saved["segment_length_cv"],
            saved["segment_length_ratio"],
        )
        return {
            "result": str(result),
            "result_sha256": file_sha256(result),
            "configuration_sha256": str(scalar(saved, "configuration_sha256")),
            "initial_path_sha256": str(scalar(saved, "initial_path_sha256")),
            "eligible_points": points,
            "pareto_records": pareto_records(points),
        }


def _component_mesh_map(path: Path, offset: int) -> dict[int, dict[str, float]]:
    with np.load(path, allow_pickle=False) as saved:
        _require_keys(saved, OFFICIAL_COMPONENT_ALLOWED_KEYS, str(path))
        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        validate_curve_arrays(cost, i_perp, cv, ratio, str(path))
        return {
            int(offset + cost[index]): {
                "i_perp": float(i_perp[index]),
                "segment_length_cv": float(cv[index]),
                "segment_length_ratio": float(ratio[index]),
            }
            for index in range(len(cost))
        }


def load_official_curve(archive: Path) -> dict[str, object]:
    with np.load(archive, allow_pickle=False) as saved:
        _require_keys(saved, OFFICIAL_ARCHIVE_ALLOWED_KEYS, str(archive))
        if not bool(scalar(saved, "complete_state_candidates_only")):
            raise RuntimeError("official archive is not complete-state-only")
        source_result = Path(str(scalar(saved, "source_result"))).resolve()
        expected_source_sha = str(scalar(saved, "source_result_sha256"))
        if file_sha256(source_result) != expected_source_sha:
            raise RuntimeError("official archive source hash mismatch")
        activation = float(scalar(saved, "activation_iperp"))
        calls = np.asarray(saved["candidate_calls"], dtype=int)
        i_perp = np.asarray(saved["candidate_i_perp"], dtype=float)
        if calls.shape != i_perp.shape or not calls.size:
            raise RuntimeError("invalid official candidate arrays")

    with np.load(source_result, allow_pickle=False) as combined:
        _require_keys(combined, OFFICIAL_COMBINED_ALLOWED_KEYS, str(source_result))
        first = Path(str(scalar(combined, "phase_a_result"))).resolve()
        second = Path(str(scalar(combined, "phase_b_result"))).resolve()
        offset = int(scalar(combined, "phase_transition_force_offset"))
    mesh_by_nf = _component_mesh_map(first, 0)
    second_mesh_by_nf = _component_mesh_map(second, offset)
    overlap = set(mesh_by_nf).intersection(second_mesh_by_nf)
    if overlap:
        raise RuntimeError("official component cost maps overlap")
    mesh_by_nf.update(second_mesh_by_nf)

    points = []
    mesh_rejected = []
    for force, value in zip(calls, i_perp):
        force = int(force)
        if force not in mesh_by_nf:
            raise RuntimeError(f"official candidate NF={force} lacks mesh data")
        diagnostics = mesh_by_nf[force]
        if float(diagnostics["i_perp"]) != float(value):
            raise RuntimeError(f"official candidate NF={force} I_perp mismatch")
        row = {
            "NF": force,
            "i_perp": float(value),
            "segment_length_cv": float(diagnostics["segment_length_cv"]),
            "segment_length_ratio": float(
                diagnostics["segment_length_ratio"]
            ),
        }
        if (
            row["segment_length_cv"] <= MESH_CV_MAX
            and row["segment_length_ratio"] <= MESH_RATIO_MAX
        ):
            points.append(row)
        else:
            mesh_rejected.append(row)
    return {
        "archive": str(archive),
        "archive_sha256": file_sha256(archive),
        "source_result": str(source_result),
        "source_result_sha256": file_sha256(source_result),
        "component_results": [str(first), str(second)],
        "component_result_sha256": [file_sha256(first), file_sha256(second)],
        "domain": "literal official archive post-activation complete candidates",
        "domain_max_i_perp": activation,
        "eligible_points": points,
        "mesh_rejected_points": mesh_rejected,
        "pareto_records": pareto_records(points),
    }


def near_miss_vector(pair: dict[str, object]) -> tuple[float, ...]:
    same = pair["same_alpha_control_comparison"]
    official = pair["official_stage4_v4_comparison"]
    comparisons = (same, official)
    tightest_failures = sum(
        not item["candidate_does_not_lose_tightest_precision"]
        for item in comparisons
    )
    unreachable = sum(
        item["near_miss"]["candidate_unreachable_loss_count"]
        for item in comparisons
    )
    losses = sum(item["near_miss"]["loss_threshold_count"] for item in comparisons)
    max_regret = max(
        item["near_miss"]["maximum_finite_positive_NF_regret"]
        for item in comparisons
    )
    precision_excess = sum(
        max(0.0, float(item["near_miss"]["tightest_i_perp_excess"] or 0.0))
        for item in comparisons
    )
    return (
        float(tightest_failures),
        float(unreachable),
        float(losses),
        float(max_regret),
        float(precision_excess),
    )


def nondominated_near_misses(pairs: list[dict[str, object]]) -> list[str]:
    vectors = {pair["alpha_key"]: near_miss_vector(pair) for pair in pairs}
    selected = []
    for key, vector in vectors.items():
        dominated = False
        for other_key, other in vectors.items():
            if other_key == key:
                continue
            if all(a <= b for a, b in zip(other, vector)) and any(
                a < b for a, b in zip(other, vector)
            ):
                dominated = True
                break
        if not dominated:
            selected.append(key)
    return selected


def analyze(grid_report: Path) -> dict[str, object]:
    grid = json.loads(grid_report.read_text(encoding="utf-8"))
    if grid.get("status") != "complete":
        raise RuntimeError("formal grid report is not complete")
    if grid.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise RuntimeError("unexpected formal protocol")
    preregistration = Path(grid["preregistration"]).resolve()
    if file_sha256(preregistration) != grid["preregistration_sha256"]:
        raise RuntimeError("preregistration hash mismatch")
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    if prereg.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise RuntimeError("preregistration protocol mismatch")
    anchors = [float(value) for value in prereg["reporting_anchors"]]
    official_path = Path(prereg["official_stage4_v4_comparator"]).resolve()
    if file_sha256(official_path) != prereg["official_stage4_v4_comparator_sha256"]:
        raise RuntimeError("official comparator hash mismatch")
    official = load_official_curve(official_path)

    rows_by_alpha: dict[float, dict[str, dict[str, object]]] = {}
    input_hashes = {}
    for row in grid["results"]:
        if int(row["returncode"]) != 0:
            raise RuntimeError("formal grid contains a failed member")
        alpha = math.inf if row["alpha"] == "inf" else float(row["alpha"])
        role = str(row["member_role"])
        if role not in {"candidate", "matched_control"}:
            raise RuntimeError(f"unknown formal member role: {role}")
        result = Path(row["result"]).resolve()
        actual_hash = file_sha256(result)
        if actual_hash != row["result_sha256"]:
            raise RuntimeError(f"result hash mismatch: {result}")
        input_hashes[str(result)] = actual_hash
        loaded = load_single_curve(result, alpha)
        rows_by_alpha.setdefault(alpha, {})[role] = loaded
    expected_alphas = [
        math.inf if value == "inf" else float(value)
        for value in prereg["canonical_alpha_grid"]
    ]
    if set(rows_by_alpha) != set(expected_alphas):
        raise RuntimeError("formal alpha set differs from preregistration")

    pairs = []
    for alpha in expected_alphas:
        members = rows_by_alpha[alpha]
        if set(members) != {"candidate", "matched_control"}:
            raise RuntimeError(f"alpha={alpha_key(alpha)} lacks a matched pair")
        candidate = members["candidate"]["eligible_points"]
        control = members["matched_control"]["eligible_points"]
        same_comparison = compare_curves(
            candidate, control, "same_alpha_legacy_control"
        )
        official_comparison = compare_curves(
            candidate,
            official["eligible_points"],
            "official_stage4_v4_post_activation",
        )
        pair = {
            "alpha": "inf" if math.isinf(alpha) else alpha,
            "alpha_key": alpha_key(alpha),
            "claim_scope": (
                "confirmatory" if alpha == float(prereg["confirmatory_alpha"]) else "exploratory"
            ),
            "candidate": members["candidate"],
            "matched_control": members["matched_control"],
            "same_alpha_control_comparison": same_comparison,
            "official_stage4_v4_comparison": official_comparison,
            "reporting_anchors_same_alpha_control": anchor_rows(
                candidate, control, anchors
            ),
            "reporting_anchors_official_stage4_v4": anchor_rows(
                candidate,
                official["eligible_points"],
                anchors,
                domain_max=float(official["domain_max_i_perp"]),
            ),
            "jointly_dominates_both_comparators": bool(
                same_comparison["full_curve_dominates"]
                and official_comparison["full_curve_dominates"]
            ),
        }
        pair["near_miss_deficit_vector"] = list(near_miss_vector(pair))
        pairs.append(pair)

    joint = [pair["alpha_key"] for pair in pairs if pair["jointly_dominates_both_comparators"]]
    same = [
        pair["alpha_key"]
        for pair in pairs
        if pair["same_alpha_control_comparison"]["full_curve_dominates"]
    ]
    official_dominators = [
        pair["alpha_key"]
        for pair in pairs
        if pair["official_stage4_v4_comparison"]["full_curve_dominates"]
    ]
    near_frontier = nondominated_near_misses(
        [pair for pair in pairs if not pair["jointly_dominates_both_comparators"]]
    )
    return {
        "analysis_id": ANALYSIS_ID,
        "created_unix": time.time(),
        "read_only_formal_input_audit": True,
        "barrier_data_accessed": False,
        "barrier_computation_or_densification_performed": False,
        "single_run_fields_accessed": sorted(SINGLE_ALLOWED_KEYS),
        "official_archive_fields_accessed": sorted(
            OFFICIAL_ARCHIVE_ALLOWED_KEYS
        ),
        "formal_grid_report": str(grid_report),
        "formal_grid_report_sha256": file_sha256(grid_report),
        "preregistration": str(preregistration),
        "preregistration_sha256": file_sha256(preregistration),
        "formal_protocol_id": EXPECTED_PROTOCOL_ID,
        "eligibility": (
            "initial or reparam_committed, then segment_length_cv <= 0.0065 "
            "and segment_length_ratio <= 1.03"
        ),
        "cost_to_precision": (
            "minimum actual optimization NF with I_perp <= threshold; no interpolation"
        ),
        "same_alpha_control_comparison_scope": (
            "full I_perp domain represented by each same-alpha matched pair's "
            "mesh-eligible initial and reparam_committed states"
        ),
        "official_comparison_scope": (
            "literal frozen official v4 Pareto archive, post-activation I_perp "
            "domain <= 0.007; higher reporting anchors are outside domain"
        ),
        "near_miss_ordering": (
            "descriptive non-scalar Pareto frontier of preregistered dominance "
            "deficits; not a post-hoc winner rule"
        ),
        "reporting_anchors": anchors,
        "official_stage4_v4": official,
        "formal_result_hashes": input_hashes,
        "same_alpha_control_full_curve_dominators": same,
        "official_stage4_v4_full_curve_dominators": official_dominators,
        "joint_full_curve_dominators": joint,
        "near_miss_deficit_pareto_alphas": near_frontier,
        "confirmatory_alpha20_passes_joint_dominance": "20" in joint,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    grid_report = args.grid_report.expanduser().resolve()
    if not grid_report.is_file():
        raise FileNotFoundError(grid_report)
    payload = analyze(grid_report)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else grid_report.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = grid_report.stem
    output = output_dir / (
        f"{stem}_iperp_ctp_readonly_v1_{time.time_ns()}.json"
    )
    payload["analysis_script"] = str(Path(__file__).resolve())
    payload["analysis_script_sha256"] = file_sha256(Path(__file__).resolve())
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
