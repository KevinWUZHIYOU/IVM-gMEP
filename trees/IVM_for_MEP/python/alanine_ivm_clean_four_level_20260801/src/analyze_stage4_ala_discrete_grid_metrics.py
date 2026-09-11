#!/usr/bin/env python3
"""Post-metric control-variable analysis for the formal 248-cell Ala grid.

The dense-metric preregistration is the sole source of eligible states.  This
module never reopens optimizer trajectories to invent a terminal-only or a
different completeness/mesh filter.  Cost-to-precision (CTP) is the minimum
*observed* optimization NF satisfying a threshold; neither metrics nor costs
are interpolated.

Raw endpoints are reported as descriptive diagnostics only.  Cell ordering,
the control-variable effects, and the nondominated set use complete observed
curves plus the same-state joint target.  A unique winner is emitted only when
an explicitly supported rule was present in the frozen metric
preregistration; otherwise the formal result is ``no_unique_winner``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import run_stage4_ala_discrete_grid_metric_campaign as metric_campaign


ANALYSIS_ID = "stage4_ala_discrete_grid_metric_control_effects_v1"
SUPPORTED_UNIQUE_WINNER_RULE = (
    "unique_grid_cell_that_full_curve_dominates_every_other_grid_cell_and_"
    "incumbent_on_raw_i_perp_and_dense_barrier_error_reaches_same_state_"
    "joint_target_and_all_comparison_domain_paths_are_dense_converged_v1"
)
PRIMARY_FAMILY = metric_campaign.PRIMARY_FAMILY
SECONDARY_FAMILY = metric_campaign.SECONDARY_FAMILY
FAMILIES = metric_campaign.FAMILIES
INCUMBENT_FAMILY = metric_campaign.INCUMBENT_FAMILY
CONTROL_AXES = ("alpha", "residual_ratio", "inner_max_steps")


class MetricAnalysisError(RuntimeError):
    """The frozen metric inputs or requested analysis are invalid."""


def _finite_nonnegative(value: Any, subject: str) -> float:
    observed = float(value)
    if not math.isfinite(observed) or observed < 0.0:
        raise MetricAnalysisError(f"{subject} must be finite and nonnegative")
    return observed


def _level_key(value: Any) -> str:
    if isinstance(value, bool):
        raise MetricAnalysisError("boolean control level is invalid")
    if isinstance(value, (int, float)):
        observed = float(value)
        if not math.isfinite(observed):
            raise MetricAnalysisError("non-finite control level is invalid")
        return f"{observed:g}"
    return str(value)


def _sort_levels(values: Iterable[Any]) -> List[Any]:
    observed = list(values)
    try:
        return sorted(observed, key=float)
    except (TypeError, ValueError):
        return sorted(observed, key=str)


def observed_ctp(
    states: Sequence[Mapping[str, Any]], metric_key: str, threshold: float
) -> Optional[int]:
    """Return the minimum observed NF meeting ``metric <= threshold``."""

    limit = _finite_nonnegative(threshold, "CTP threshold")
    costs = [
        int(state["optimization_nf"])
        for state in states
        if state.get(metric_key) is not None
        and _finite_nonnegative(state[metric_key], metric_key) <= limit
    ]
    return min(costs) if costs else None


def observed_frontier(
    states: Sequence[Mapping[str, Any]], metric_key: str
) -> List[Dict[str, Any]]:
    """Build the exact observed CTP envelope, without interpolation.

    Multiple states may have the same NF (the exact family permits this).  At
    one NF the best actually observed metric is retained, with trajectory
    index and state identity kept for traceability.
    """

    best_at_nf: Dict[int, Mapping[str, Any]] = {}
    for state in states:
        if state.get(metric_key) is None:
            continue
        nf = int(state["optimization_nf"])
        value = _finite_nonnegative(state[metric_key], metric_key)
        previous = best_at_nf.get(nf)
        if previous is None or (value, int(state["trajectory_index"])) < (
            float(previous[metric_key]),
            int(previous["trajectory_index"]),
        ):
            best_at_nf[nf] = state
    records: List[Dict[str, Any]] = []
    running = math.inf
    for nf in sorted(best_at_nf):
        state = best_at_nf[nf]
        value = float(state[metric_key])
        if value < running:
            records.append(
                {
                    "optimization_nf": nf,
                    metric_key: value,
                    "trajectory_index": int(state["trajectory_index"]),
                    "state_id": str(state["state_id"]),
                }
            )
            running = value
    return records


def _cost_as_float(value: Optional[int]) -> float:
    return math.inf if value is None else float(value)


def full_observed_curve_order(
    left_states: Sequence[Mapping[str, Any]],
    right_states: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> Dict[str, Any]:
    """Compare two complete observed CTP curves on all observed thresholds."""

    return _full_frontier_order(
        observed_frontier(left_states, metric_key),
        observed_frontier(right_states, metric_key),
        metric_key,
    )


def _full_frontier_order(
    left_frontier: Sequence[Mapping[str, Any]],
    right_frontier: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> Dict[str, Any]:
    """Linear-time comparison of two already-built observed CTP envelopes."""

    thresholds = sorted(
        {
            _finite_nonnegative(row[metric_key], metric_key)
            for row in list(left_frontier) + list(right_frontier)
        },
        reverse=True,
    )
    if not thresholds:
        raise MetricAnalysisError("cannot compare two empty observed curves")
    nonworse = True
    strict = False
    strict_thresholds: List[float] = []
    loss_thresholds: List[float] = []
    left_index = 0
    right_index = 0
    for threshold in thresholds:
        while (
            left_index < len(left_frontier)
            and float(left_frontier[left_index][metric_key]) > threshold
        ):
            left_index += 1
        while (
            right_index < len(right_frontier)
            and float(right_frontier[right_index][metric_key]) > threshold
        ):
            right_index += 1
        left = (
            None
            if left_index == len(left_frontier)
            else int(left_frontier[left_index]["optimization_nf"])
        )
        right = (
            None
            if right_index == len(right_frontier)
            else int(right_frontier[right_index]["optimization_nf"])
        )
        left_value = _cost_as_float(left)
        right_value = _cost_as_float(right)
        if left_value > right_value:
            nonworse = False
            loss_thresholds.append(threshold)
        elif left_value < right_value:
            strict = True
            strict_thresholds.append(threshold)
    return {
        "nonworse_at_every_observed_threshold": nonworse,
        "strict_at_some_observed_threshold": strict,
        "observed_threshold_count": len(thresholds),
        "strict_thresholds": strict_thresholds,
        "loss_thresholds": loss_thresholds,
        "reverse_nonworse_at_every_observed_threshold": not strict,
        "reverse_strict_at_some_observed_threshold": bool(loss_thresholds),
        "interpolation_used": False,
    }


def same_state_joint_ctp(
    states: Sequence[Mapping[str, Any]],
    i_perp_target: float,
    barrier_error_target: float,
) -> Optional[int]:
    """Minimum observed NF at which one state meets both registered targets."""

    i_limit = _finite_nonnegative(i_perp_target, "I_perp target")
    e_limit = _finite_nonnegative(barrier_error_target, "barrier-error target")
    costs = [
        int(state["optimization_nf"])
        for state in states
        if float(state["raw_i_perp"]) <= i_limit
        and state.get("barrier_error_kcal_per_mol") is not None
        and float(state["barrier_error_kcal_per_mol"]) <= e_limit
    ]
    return min(costs) if costs else None


def _target_order(left: Optional[int], right: Optional[int]) -> str:
    if left is None and right is None:
        return "both_unreachable"
    if left is None:
        return "right_only_reaches"
    if right is None:
        return "left_only_reaches"
    if left < right:
        return "left_lower_nf"
    if right < left:
        return "right_lower_nf"
    return "tie"


def _prepare_state(
    raw: Mapping[str, Any], reference_barrier: float
) -> Dict[str, Any]:
    if raw.get("formal_complete") is not True or raw.get("mesh_eligible") is not True:
        raise MetricAnalysisError(
            "analyzer input must contain only manifest-fixed eligible states"
        )
    nf = int(raw.get("optimization_nf", -1))
    index = int(raw.get("trajectory_index", -1))
    if nf < 0 or index < 0:
        raise MetricAnalysisError("state NF/index is invalid")
    state_id = str(raw.get("state_id", ""))
    if not state_id:
        raise MetricAnalysisError("state identity is missing")
    i_perp = _finite_nonnegative(raw.get("raw_i_perp"), "raw I_perp")
    barrier = _finite_nonnegative(
        raw.get("dense_barrier_kcal_per_mol"), "dense barrier"
    )
    error = abs(barrier - reference_barrier)
    supplied_error = raw.get("barrier_error_kcal_per_mol")
    if supplied_error is not None and not math.isclose(
        float(supplied_error), error, rel_tol=0.0, abs_tol=1e-15
    ):
        raise MetricAnalysisError("supplied dense-barrier error drifted")
    dense_converged = bool(raw.get("dense_converged", True))
    return {
        "state_id": state_id,
        "trajectory_index": index,
        "optimization_nf": nf,
        "raw_i_perp": i_perp,
        "dense_barrier_kcal_per_mol": barrier,
        "diagnostic_barrier_error_kcal_per_mol": error,
        "barrier_error_kcal_per_mol": error if dense_converged else None,
        "barrier_error_metric_valid": dense_converged,
        "dense_converged": dense_converged,
        "path_task_id": str(raw.get("path_task_id", "")),
    }


def prepare_cell(
    raw: Mapping[str, Any],
    reference_barrier: float,
    i_perp_target: float,
    barrier_error_target: float,
) -> Dict[str, Any]:
    family = str(raw.get("family", ""))
    if family not in (*FAMILIES, INCUMBENT_FAMILY):
        raise MetricAnalysisError(f"unknown family: {family}")
    cell_id = str(raw.get("cell_id", ""))
    controls = raw.get("controls")
    raw_states = raw.get("states")
    if not cell_id or not isinstance(controls, Mapping):
        raise MetricAnalysisError("cell identity/controls are missing")
    if not isinstance(raw_states, list) or not raw_states:
        raise MetricAnalysisError("cell has no manifest-eligible states")
    states = [_prepare_state(state, reference_barrier) for state in raw_states]
    state_ids = [state["state_id"] for state in states]
    indices = [int(state["trajectory_index"]) for state in states]
    if len(set(state_ids)) != len(state_ids) or len(set(indices)) != len(indices):
        raise MetricAnalysisError("cell state identity/index is not unique")
    endpoint = max(states, key=lambda state: int(state["trajectory_index"]))
    i_ctp = observed_ctp(states, "raw_i_perp", i_perp_target)
    e_ctp = observed_ctp(
        states, "barrier_error_kcal_per_mol", barrier_error_target
    )
    joint = same_state_joint_ctp(states, i_perp_target, barrier_error_target)
    return {
        "family": family,
        "cell_id": cell_id,
        "controls": dict(controls),
        "inner_cap_stop_evidence": raw.get(
            "inner_cap_stop_evidence",
            {
                "status": "not_available_in_metric_manifest",
                "inner_cap_binding": None,
            },
        ),
        "manifest_eligible_state_count": len(states),
        "dense_nonconverged_state_count": sum(
            not bool(state["dense_converged"]) for state in states
        ),
        "barrier_metric_valid_state_count": sum(
            bool(state["barrier_error_metric_valid"]) for state in states
        ),
        "all_referenced_dense_paths_converged": all(
            bool(state["dense_converged"]) for state in states
        ),
        "raw_endpoint": {
            **dict(endpoint),
            "role": "descriptive_only_not_used_for_ranking",
            "definition": "largest_trajectory_index_in_metric_manifest",
        },
        "targets": {
            "i_perp_ctp_nf": i_ctp,
            "barrier_error_ctp_nf": e_ctp,
            "same_state_joint_ctp_nf": joint,
            "i_perp_target": float(i_perp_target),
            "barrier_error_target_kcal_per_mol": float(barrier_error_target),
        },
        "observed_ctp_frontiers": {
            "raw_i_perp": observed_frontier(states, "raw_i_perp"),
            "barrier_error_kcal_per_mol": observed_frontier(
                states, "barrier_error_kcal_per_mol"
            ),
        },
        "_states": states,
    }


def compare_cells(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    i_order = _full_frontier_order(
        left["observed_ctp_frontiers"]["raw_i_perp"],
        right["observed_ctp_frontiers"]["raw_i_perp"],
        "raw_i_perp",
    )
    left_e_frontier = left["observed_ctp_frontiers"][
        "barrier_error_kcal_per_mol"
    ]
    right_e_frontier = right["observed_ctp_frontiers"][
        "barrier_error_kcal_per_mol"
    ]
    barrier_curves_valid = bool(left_e_frontier and right_e_frontier)
    e_order = (
        _full_frontier_order(
            left_e_frontier,
            right_e_frontier,
            "barrier_error_kcal_per_mol",
        )
        if barrier_curves_valid
        else {
            "metric_valid": False,
            "reason": "one_or_both_cells_have_no_dense_converged_barrier_state",
            "left_valid_state_count": int(
                left["barrier_metric_valid_state_count"]
            ),
            "right_valid_state_count": int(
                right["barrier_metric_valid_state_count"]
            ),
            "interpolation_used": False,
        }
    )
    left_joint = left["targets"]["same_state_joint_ctp_nf"]
    right_joint = right["targets"]["same_state_joint_ctp_nf"]
    left_dominates = bool(
        barrier_curves_valid
        and
        i_order["nonworse_at_every_observed_threshold"]
        and e_order["nonworse_at_every_observed_threshold"]
        and (
            i_order["strict_at_some_observed_threshold"]
            or e_order["strict_at_some_observed_threshold"]
        )
    )
    right_dominates = bool(
        barrier_curves_valid
        and
        i_order["reverse_nonworse_at_every_observed_threshold"]
        and e_order["reverse_nonworse_at_every_observed_threshold"]
        and (
            i_order["reverse_strict_at_some_observed_threshold"]
            or e_order["reverse_strict_at_some_observed_threshold"]
        )
    )
    if not barrier_curves_valid:
        relation = "metric_invalid"
    elif left_dominates:
        relation = "left_dominates"
    elif right_dominates:
        relation = "right_dominates"
    elif (
        i_order["nonworse_at_every_observed_threshold"]
        and i_order["reverse_nonworse_at_every_observed_threshold"]
        and e_order["nonworse_at_every_observed_threshold"]
        and e_order["reverse_nonworse_at_every_observed_threshold"]
    ):
        relation = "equivalent"
    else:
        relation = "incomparable"
    return {
        "left_cell_id": left["cell_id"],
        "right_cell_id": right["cell_id"],
        "relation": relation,
        "left_dominates": left_dominates,
        "right_dominates": right_dominates,
        "i_perp_full_curve_order": i_order,
        "barrier_error_full_curve_order": e_order,
        "joint_target_order": _target_order(left_joint, right_joint),
        "left_reaches_same_state_joint_target": left_joint is not None,
        "right_reaches_same_state_joint_target": right_joint is not None,
        "target_ctp_orders": {
            "i_perp": _target_order(
                left["targets"]["i_perp_ctp_nf"],
                right["targets"]["i_perp_ctp_nf"],
            ),
            "barrier_error": _target_order(
                left["targets"]["barrier_error_ctp_nf"],
                right["targets"]["barrier_error_ctp_nf"],
            ),
            "same_state_joint": _target_order(left_joint, right_joint),
        },
    }


def nondominated_cell_ids(cells: Sequence[Mapping[str, Any]]) -> List[str]:
    dominated: set[str] = set()
    for left_index, left in enumerate(cells):
        for right in cells[left_index + 1 :]:
            comparison = compare_cells(left, right)
            if comparison["left_dominates"]:
                dominated.add(str(right["cell_id"]))
            elif comparison["right_dominates"]:
                dominated.add(str(left["cell_id"]))
    return sorted(
        str(cell["cell_id"])
        for cell in cells
        if str(cell["cell_id"]) not in dominated
    )


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "minimum": min(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "maximum": max(finite) if finite else None,
        "role": "descriptive_main_effect_only_not_a_winner_rule",
    }


def _level_summary(cells: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    targets = ("i_perp_ctp_nf", "barrier_error_ctp_nf", "same_state_joint_ctp_nf")
    result: Dict[str, Any] = {"cell_count": len(cells), "target_ctp": {}}
    for target in targets:
        raw = [cell["targets"][target] for cell in cells]
        result["target_ctp"][target] = {
            "reach_count": sum(value is not None for value in raw),
            "nf_distribution_among_reachers": _distribution(
                [float(value) for value in raw if value is not None]
            ),
        }
    result["raw_endpoint_descriptive"] = {
        "raw_i_perp": _distribution(
            [float(cell["raw_endpoint"]["raw_i_perp"]) for cell in cells]
        ),
        "diagnostic_barrier_error_kcal_per_mol": _distribution(
            [
                float(
                    cell["raw_endpoint"][
                        "diagnostic_barrier_error_kcal_per_mol"
                    ]
                )
                for cell in cells
            ]
        ),
    }
    return result


def control_variable_effects(
    primary_cells: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Matched one-factor-at-a-time effects for the inexact full factorial."""

    if not primary_cells:
        raise MetricAnalysisError("primary control grid is empty")
    by_key: Dict[Tuple[Any, Any, Any], Mapping[str, Any]] = {}
    for cell in primary_cells:
        controls = cell["controls"]
        try:
            key = tuple(controls[axis] for axis in CONTROL_AXES)
        except KeyError as error:
            raise MetricAnalysisError(f"primary control missing {error.args[0]}") from error
        if key in by_key:
            raise MetricAnalysisError("duplicate primary control combination")
        by_key[key] = cell

    axes: Dict[str, Any] = {}
    for axis_index, axis in enumerate(CONTROL_AXES):
        other_axes = [item for item in CONTROL_AXES if item != axis]
        levels = _sort_levels({key[axis_index] for key in by_key})
        contexts: Dict[Tuple[Any, ...], Dict[Any, Mapping[str, Any]]] = defaultdict(dict)
        for key, cell in by_key.items():
            context = tuple(
                key[CONTROL_AXES.index(other_axis)] for other_axis in other_axes
            )
            contexts[context][key[axis_index]] = cell
        if any(set(rows) != set(levels) for rows in contexts.values()):
            raise MetricAnalysisError(f"{axis} matched contexts are incomplete")
        context_is_dense_complete = {
            context: all(
                bool(cell["all_referenced_dense_paths_converged"])
                for cell in rows.values()
            )
            for context, rows in contexts.items()
        }

        conditional_rows: List[Dict[str, Any]] = []
        aggregate: Dict[Tuple[Any, Any], Dict[str, Counter]] = {}
        for left_index, left_level in enumerate(levels):
            for right_level in levels[left_index + 1 :]:
                aggregate[(left_level, right_level)] = {
                    "partial_order": Counter(),
                    "i_perp_target": Counter(),
                    "barrier_error_target": Counter(),
                    "same_state_joint_target": Counter(),
                }
        for context_key in sorted(contexts, key=lambda item: tuple(map(str, item))):
            rows = contexts[context_key]
            comparisons: List[Dict[str, Any]] = []
            dense_complete = context_is_dense_complete[context_key]
            if dense_complete:
                for left_index, left_level in enumerate(levels):
                    for right_level in levels[left_index + 1 :]:
                        comparison = compare_cells(
                            rows[left_level], rows[right_level]
                        )
                        compact = {
                            "left_level": left_level,
                            "right_level": right_level,
                            "left_cell_id": comparison["left_cell_id"],
                            "right_cell_id": comparison["right_cell_id"],
                            "partial_order_relation": comparison["relation"],
                            "target_ctp_orders": comparison["target_ctp_orders"],
                        }
                        comparisons.append(compact)
                        counts = aggregate[(left_level, right_level)]
                        counts["partial_order"][comparison["relation"]] += 1
                        counts["i_perp_target"][
                            comparison["target_ctp_orders"]["i_perp"]
                        ] += 1
                        counts["barrier_error_target"][
                            comparison["target_ctp_orders"]["barrier_error"]
                        ] += 1
                        counts["same_state_joint_target"][
                            comparison["target_ctp_orders"]["same_state_joint"]
                        ] += 1
            context_cells = [rows[level] for level in levels]
            conditional_rows.append(
                {
                    "status": (
                        "fully_converged_matched_context"
                        if dense_complete
                        else "not_compared_due_to_nonconverged_paths"
                    ),
                    "fixed_controls": {
                        name: value for name, value in zip(other_axes, context_key)
                    },
                    "cell_ids_by_level": {
                        _level_key(level): rows[level]["cell_id"] for level in levels
                    },
                    "nondominated_levels": (
                        [
                            level
                            for level in levels
                            if rows[level]["cell_id"]
                            in set(nondominated_cell_ids(context_cells))
                        ]
                        if dense_complete
                        else None
                    ),
                    "pairwise_relations": comparisons,
                }
            )
        axes[axis] = {
            "levels": levels,
            "matched_context_count": len(contexts),
            "fully_converged_matched_context_count": sum(
                context_is_dense_complete.values()
            ),
            "nonconverged_matched_context_count": sum(
                not value for value in context_is_dense_complete.values()
            ),
            "level_main_effect_summaries": {
                _level_key(level): _level_summary(
                    [
                        rows[level]
                        for context, rows in contexts.items()
                        if context_is_dense_complete[context]
                    ]
                )
                for level in levels
            },
            "matched_pairwise_main_effect_counts": [
                {
                    "left_level": left,
                    "right_level": right,
                    **{
                        name: dict(counter)
                        for name, counter in aggregate[(left, right)].items()
                    },
                }
                for left_index, left in enumerate(levels)
                for right in levels[left_index + 1 :]
            ],
            "conditional_effects": conditional_rows,
        }
    return {
        "scope": "inexact_reparam_full_factorial_only",
        "status": (
            "all_matched_contexts_dense_complete"
            if all(
                bool(cell["all_referenced_dense_paths_converged"])
                for cell in primary_cells
            )
            else "provisional_complete_matched_contexts_only"
        ),
        "definition": (
            "one control changes inside each exactly matched context; summaries "
            "do not define a scalar or post-hoc winner"
        ),
        "axes": axes,
    }


def exact_vs_inexact_effects(
    primary_cells: Sequence[Mapping[str, Any]],
    exact_cells: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for exact in sorted(exact_cells, key=lambda cell: float(cell["controls"]["alpha"])):
        alpha = float(exact["controls"]["alpha"])
        matched = [
            cell
            for cell in primary_cells
            if float(cell["controls"]["alpha"]) == alpha
            and float(cell["controls"]["residual_ratio"]) == 0.1
            and int(cell["controls"]["inner_max_steps"]) == 50
        ]
        if len(matched) != 1:
            raise MetricAnalysisError(
                f"alpha={alpha:g} lacks exactly one inexact r=0.1,c50 comparator"
            )
        comparisons = []
        for inexact in sorted(
            matched, key=lambda cell: int(cell["controls"]["inner_max_steps"])
        ):
            dense_complete = bool(
                exact["all_referenced_dense_paths_converged"]
                and inexact["all_referenced_dense_paths_converged"]
            )
            comparison = compare_cells(exact, inexact) if dense_complete else None
            comparisons.append(
                {
                    "status": (
                        "fully_converged_closest_comparator"
                        if dense_complete
                        else "not_compared_due_to_nonconverged_paths"
                    ),
                    "inexact_cell_id": inexact["cell_id"],
                    "inexact_maxstep": int(
                        inexact["controls"]["inner_max_steps"]
                    ),
                    "inexact_c50_cap_stop_evidence": inexact[
                        "inner_cap_stop_evidence"
                    ],
                    "partial_order_relation_exact_as_left": (
                        None if comparison is None else comparison["relation"]
                    ),
                    "target_ctp_orders_exact_as_left": (
                        None
                        if comparison is None
                        else comparison["target_ctp_orders"]
                    ),
                }
            )
        rows.append(
            {
                "alpha": alpha,
                "exact_cell_id": exact["cell_id"],
                "matched_inexact_r": 0.1,
                "comparison_count": len(comparisons),
                "comparisons": comparisons,
            }
        )
    return {
        "scope": "same_alpha_and_inexact_r_0.1_c50_closest_comparator_only",
        "status": (
            "all_closest_pairs_dense_complete"
            if all(
                comparison["status"] == "fully_converged_closest_comparator"
                for row in rows
                for comparison in row["comparisons"]
            )
            else "provisional_complete_closest_pairs_only"
        ),
        "causal_label": (
            "descriptive confounded family-package comparison; exact/no-reparam/"
            "operationally-unbounded versus inexact/reparam/c50 are jointly "
            "changed, so no effect is causally attributed to reparameterization"
        ),
        "c50_binding_requirement": (
            "report frozen cap-binding evidence; unavailable evidence forbids a "
            "claim that c50 approximates operationally unbounded"
        ),
        "rows": rows,
    }


def _public_cell(cell: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in cell.items() if key != "_states"}


def analyze_cells(
    raw_cells: Sequence[Mapping[str, Any]],
    formal_comparison: Mapping[str, Any],
    raw_incumbent: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    reference = _finite_nonnegative(
        formal_comparison.get("reference_barrier_kcal_per_mol"),
        "reference barrier",
    )
    i_target = _finite_nonnegative(
        formal_comparison.get("i_perp_target"), "I_perp target"
    )
    e_target = _finite_nonnegative(
        formal_comparison.get("barrier_error_target_kcal_per_mol"),
        "barrier-error target",
    )
    cells = [
        prepare_cell(cell, reference, i_target, e_target) for cell in raw_cells
    ]
    incumbent = (
        None
        if raw_incumbent is None
        else prepare_cell(raw_incumbent, reference, i_target, e_target)
    )
    cell_ids = [cell["cell_id"] for cell in cells]
    if incumbent is not None:
        cell_ids.append(incumbent["cell_id"])
    if len(set(cell_ids)) != len(cell_ids):
        raise MetricAnalysisError("analysis cell IDs are not unique")
    primary = [cell for cell in cells if cell["family"] == PRIMARY_FAMILY]
    exact = [cell for cell in cells if cell["family"] == SECONDARY_FAMILY]
    grid_nonconverged = sorted(
        str(cell["cell_id"])
        for cell in cells
        if not bool(cell["all_referenced_dense_paths_converged"])
    )
    inclusive_cells = [*cells, *([] if incumbent is None else [incumbent])]
    inclusive_nonconverged = sorted(
        str(cell["cell_id"])
        for cell in inclusive_cells
        if not bool(cell["all_referenced_dense_paths_converged"])
    )
    comparison_domain_dense_complete = bool(
        incumbent is not None and not inclusive_nonconverged
    )
    fully_converged_grid = [
        cell
        for cell in cells
        if bool(cell["all_referenced_dense_paths_converged"])
    ]
    fully_converged_inclusive = [
        cell
        for cell in inclusive_cells
        if bool(cell["all_referenced_dense_paths_converged"])
    ]
    provisional_grid_nondominated = nondominated_cell_ids(fully_converged_grid)
    provisional_inclusive_nondominated = nondominated_cell_ids(
        fully_converged_inclusive
    )
    grid_only_nondominated = (
        nondominated_cell_ids(cells) if comparison_domain_dense_complete else None
    )
    inclusive_nondominated = (
        nondominated_cell_ids(inclusive_cells)
        if comparison_domain_dense_complete
        else None
    )
    gate_qualifiers = []
    if comparison_domain_dense_complete:
        gate_qualifiers = [
            str(candidate["cell_id"])
            for candidate in cells
            if candidate["targets"]["same_state_joint_ctp_nf"] is not None
            and all(
                other["cell_id"] == candidate["cell_id"]
                or compare_cells(candidate, other)["left_dominates"]
                for other in inclusive_cells
            )
        ]

    incumbent_comparisons: List[Dict[str, Any]] = []
    if incumbent is not None:
        for candidate in cells:
            dense_complete_pair = bool(
                candidate["all_referenced_dense_paths_converged"]
                and incumbent["all_referenced_dense_paths_converged"]
            )
            comparison = (
                compare_cells(candidate, incumbent)
                if dense_complete_pair
                else None
            )
            incumbent_comparisons.append(
                {
                    "status": (
                        "fully_converged_engineering_comparison"
                        if dense_complete_pair
                        else "not_compared_due_to_nonconverged_paths"
                    ),
                    "grid_cell_id": candidate["cell_id"],
                    "partial_order_relation_grid_as_left": (
                        None if comparison is None else comparison["relation"]
                    ),
                    "grid_reaches_same_state_joint_target": (
                        candidate["targets"]["same_state_joint_ctp_nf"] is not None
                    ),
                    "grid_all_referenced_dense_paths_converged": (
                        int(candidate["dense_nonconverged_state_count"]) == 0
                    ),
                    "target_ctp_orders_grid_as_left": (
                        None
                        if comparison is None
                        else comparison["target_ctp_orders"]
                    ),
                }
            )

    registered_rule = formal_comparison.get("unique_winner_gate_policy_id")
    if incumbent is None:
        winner = {
            "status": "no_unique_winner",
            "cell_id": None,
            "reason": "incumbent_metric_input_missing",
            "registered_rule": registered_rule,
        }
    elif not comparison_domain_dense_complete:
        winner = {
            "status": "not_determined_due_to_nonconverged_paths",
            "cell_id": None,
            "reason": "not_determined_due_to_nonconverged_paths",
            "registered_rule": registered_rule,
            "nonconverged_comparison_domain_cell_ids": inclusive_nonconverged,
        }
    elif registered_rule != SUPPORTED_UNIQUE_WINNER_RULE:
        winner = {
            "status": "no_unique_winner",
            "cell_id": None,
            "reason": "supported_unique_winner_rule_not_preregistered",
            "registered_rule": registered_rule,
        }
    else:
        if len(gate_qualifiers) == 1:
            winner = {
                "status": "unique_winner",
                "cell_id": gate_qualifiers[0],
                "reason": SUPPORTED_UNIQUE_WINNER_RULE,
                "registered_rule": registered_rule,
            }
        else:
            winner = {
                "status": "no_unique_winner",
                "cell_id": None,
                "reason": "registered_rule_did_not_select_exactly_one_cell",
                "registered_rule": registered_rule,
                "registered_gate_qualifying_cell_ids": gate_qualifiers,
            }

    return {
        "analysis_id": ANALYSIS_ID,
        "analysis_semantics": {
            "eligibility": "metric_preregistration_manifest_only",
            "raw_endpoint_role": "descriptive_only_not_used_for_ranking",
            "cost_to_precision": "minimum_observed_optimization_nf",
            "interpolation_used": False,
            "terminal_only_ranking_used": False,
            "joint_gate": "same_observed_state_meets_both_targets",
            "partial_order": (
                "weakly_nonworse_on_both_full_observed_CTP_curves, with at "
                "least one strict threshold across either curve"
            ),
            "winner_rule": SUPPORTED_UNIQUE_WINNER_RULE,
            "dense_nonconvergence": (
                "any nonconverged eligible path in grid or incumbent makes both "
                "formal nondominated sets not determined and forbids a winner"
            ),
        },
        "formal_comparison": dict(formal_comparison),
        "family_cell_counts": {
            PRIMARY_FAMILY: len(primary),
            SECONDARY_FAMILY: len(exact),
        },
        "cell_summaries": [_public_cell(cell) for cell in cells],
        "raw_endpoint_analysis": {
            "role": "descriptive_only_not_used_for_ranking",
            "by_family": {
                family: _level_summary(
                    [cell for cell in cells if cell["family"] == family]
                )
                for family in FAMILIES
            },
            "incumbent": (
                None if incumbent is None else dict(incumbent["raw_endpoint"])
            ),
        },
        "control_variable_analysis": control_variable_effects(primary),
        "exact_vs_inexact": exact_vs_inexact_effects(primary, exact),
        "grid_only_nondominated": {
            "role": "control_tuning_interpretation_only_not_formal_winner",
            "status": (
                "determined"
                if comparison_domain_dense_complete
                else "not_determined_due_to_nonconverged_paths"
            ),
            "cell_ids": grid_only_nondominated,
            "cell_count": (
                None
                if grid_only_nondominated is None
                else len(grid_only_nondominated)
            ),
            "nonconverged_comparison_domain_cell_ids": grid_nonconverged,
        },
        "incumbent_inclusive_nondominated": {
            "role": "formal_winner_universe",
            "status": (
                "determined"
                if comparison_domain_dense_complete
                else "not_determined_due_to_nonconverged_paths"
            ),
            "cell_ids": inclusive_nondominated,
            "cell_count": (
                None
                if inclusive_nondominated is None
                else len(inclusive_nondominated)
            ),
            "nonconverged_comparison_domain_cell_ids": inclusive_nonconverged,
        },
        "provisional_nondominated_among_fully_converged_cells": {
            "role": "engineering_diagnostic_only_never_a_formal_winner",
            "grid_only_cell_ids": provisional_grid_nondominated,
            "incumbent_inclusive_cell_ids": provisional_inclusive_nondominated,
        },
        "incumbent_comparison": {
            "incumbent": None if incumbent is None else _public_cell(incumbent),
            "grid_rows": incumbent_comparisons,
        },
        "registered_winner_gate_qualifying_cell_ids": gate_qualifiers,
        "winner": winner,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MetricAnalysisError(f"cannot read JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise MetricAnalysisError(f"expected JSON object: {path}")
    return payload


def load_formal_metric_cells(
    root: Path, aggregate_path: Path
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Validate a completed dense aggregate and fan metrics back to 248 cells."""

    root = metric_campaign._root(root)
    aggregate_path = aggregate_path.expanduser().resolve()
    aggregate = _read_json(aggregate_path)
    preregistration_path = Path(str(aggregate.get("preregistration", ""))).resolve()
    if (
        aggregate.get("status") != "complete"
        or aggregate.get("protocol_id") != metric_campaign.PROTOCOL_ID
        or not preregistration_path.is_file()
        or metric_campaign.file_sha256(preregistration_path)
        != aggregate.get("preregistration_sha256")
    ):
        raise MetricAnalysisError("dense metric aggregate identity is invalid")
    preregistration = metric_campaign.load_and_validate_preregistration(
        root, preregistration_path
    )
    expected_aggregate = Path(
        str(preregistration["planned_outputs"]["aggregate"])
    ).resolve()
    if aggregate_path != expected_aggregate:
        raise MetricAnalysisError("dense metric aggregate route drifted")
    rebuilt = metric_campaign.validate_aggregate(
        root, preregistration_path, preregistration
    )
    if aggregate != rebuilt:
        raise MetricAnalysisError("dense metric aggregate validator disagreed")

    metrics_by_task: Dict[str, Dict[str, Any]] = {}
    artifact_hashes: Dict[str, str] = {}
    for task in preregistration["tasks"]:
        state, inspected = metric_campaign.inspect_task(
            preregistration_path, preregistration, task
        )
        if state != "verified" or inspected is None:
            raise MetricAnalysisError(f"metric task is not verified: {task['task_id']}")
        artifact_path = Path(str(task["artifact"])).resolve()
        artifact = _read_json(artifact_path)
        payload = metric_campaign._validate_metric_payload(
            artifact.get("metric"), task, preregistration["dense_policy"]
        )
        task_id = str(task["task_id"])
        metrics_by_task[task_id] = {
            "dense_barrier_kcal_per_mol": float(
                payload["selected_barrier_kcal_per_mol"]
            ),
            "dense_converged": bool(payload["converged"]),
            "selected_subdivisions": int(payload["selected_subdivisions"]),
        }
        artifact_hashes[task_id] = metric_campaign.file_sha256(artifact_path)

    def fanout_cell(registered_cell: Mapping[str, Any]) -> Dict[str, Any]:
        states = []
        for registered_state in registered_cell["states"]:
            task_id = str(registered_state["path_task_id"])
            task_metric = metrics_by_task.get(task_id)
            if task_metric is None:
                raise MetricAnalysisError("eligible state lacks a dense metric task")
            states.append({**dict(registered_state), **task_metric})
        return {
            "family": registered_cell["family"],
            "cell_id": registered_cell["cell_id"],
            "controls": dict(registered_cell["controls"]),
            "inner_cap_stop_evidence": registered_cell.get(
                "inner_cap_stop_evidence"
            ),
            "states": states,
        }

    cells = [fanout_cell(item) for item in preregistration["cells"]]
    incumbent = fanout_cell(preregistration["incumbent_comparator"])
    expected_counts = metric_campaign.EXPECTED_CELL_COUNTS
    observed_counts = {
        family: sum(cell["family"] == family for cell in cells)
        for family in FAMILIES
    }
    if observed_counts != expected_counts:
        raise MetricAnalysisError("formal metric analysis does not cover 248 cells")
    provenance = {
        "metric_aggregate": str(aggregate_path),
        "metric_aggregate_sha256": metric_campaign.file_sha256(aggregate_path),
        "metric_preregistration": str(preregistration_path),
        "metric_preregistration_sha256": metric_campaign.file_sha256(
            preregistration_path
        ),
        "metric_task_artifact_sha256": artifact_hashes,
    }
    return cells, incumbent, preregistration, provenance


def analyze_formal_metric_aggregate(
    root: Path, aggregate_path: Path
) -> Dict[str, Any]:
    cells, incumbent, preregistration, provenance = load_formal_metric_cells(
        root, aggregate_path
    )
    payload = analyze_cells(
        cells, preregistration["formal_comparison"], incumbent
    )
    payload.update(
        {
            "created_unix": time.time(),
            "formal_input_provenance": provenance,
            "dense_metric_quality": {
                "all_path_tasks_dense_converged": bool(
                    _read_json(aggregate_path)["all_path_tasks_dense_converged"]
                ),
                "nonconverged_path_task_count": int(
                    _read_json(aggregate_path)["nonconverged_path_task_count"]
                ),
            },
        }
    )
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--metric-aggregate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = analyze_formal_metric_aggregate(args.root, args.metric_aggregate)
    if args.output is None:
        preregistration = _read_json(Path(payload["formal_input_provenance"]["metric_preregistration"]))
        output = Path(str(preregistration["planned_outputs"]["analysis"])).resolve()
    else:
        output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload["analysis_script"] = str(Path(__file__).resolve())
    payload["analysis_script_sha256"] = metric_campaign.file_sha256(
        Path(__file__).resolve()
    )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
