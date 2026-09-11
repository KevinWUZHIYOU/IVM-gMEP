"""Pure, registration-driven acceptance analysis for Ala2D V17 formal MD.

This module performs no molecular dynamics and writes no files.  Public
trajectory dictionaries use the schema returned by
``ala2d_irivm_v17.load_public_trajectory``.  Fixed-path diagnostic observations are
passed explicitly to :func:`evaluate_terminal_noise_test` or
:func:`evaluate_campaign_acceptance`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union
import json

import numpy as np

from riemannian_torus_geometry import periodic_difference
from ala2d_irivm_v17 import IMPLEMENTATION_ID
from formal_contract import (
    ACCEPTANCE_SCHEMA,
    FORMAL_INCREMENTAL_NOF_BUDGET,
    PAID_STATE_KINDS,
    FormalContractError,
    validate_public_trajectory,
)


Array = np.ndarray
Trajectory = Mapping[str, Array]
Registration = Mapping[str, object]


class AcceptanceInputError(ValueError):
    """Raised when evidence cannot be matched to the registered analysis."""


def load_registration(path: Union[str, Path]) -> Dict[str, object]:
    """Load a registration JSON without modifying it."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AcceptanceInputError("registration root must be an object")
    return value


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise AcceptanceInputError(f"registration field {key!r} must be an object")
    return value


def _integer(parent: Mapping[str, object], key: str) -> int:
    value = parent.get(key)
    if type(value) is not int:
        raise AcceptanceInputError(f"registration field {key!r} must be an integer")
    return int(value)


def _number(parent: Mapping[str, object], key: str) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceInputError(f"registration field {key!r} must be numerical")
    result = float(value)
    if not np.isfinite(result):
        raise AcceptanceInputError(f"registration field {key!r} must be finite")
    return result


def _bool(parent: Mapping[str, object], key: str) -> bool:
    value = parent.get(key)
    if type(value) is not bool:
        raise AcceptanceInputError(f"registration field {key!r} must be boolean")
    return bool(value)


def _seed_list(registration: Registration, key: str) -> Tuple[int, ...]:
    values = _mapping(registration, "seeds").get(key)
    if not isinstance(values, list) or not values or any(type(item) is not int for item in values):
        raise AcceptanceInputError(f"registration seeds.{key} must be a nonempty integer list")
    result = tuple(int(item) for item in values)
    if len(set(result)) != len(result):
        raise AcceptanceInputError(f"registration seeds.{key} contains duplicates")
    return result


def _optimization_items(
    optimization_trajectories: Mapping[Union[int, str], Trajectory],
    registration: Registration,
) -> List[Tuple[int, Trajectory]]:
    result: List[Tuple[int, Trajectory]] = []
    for seed in _seed_list(registration, "optimization"):
        if seed in optimization_trajectories:
            trajectory = optimization_trajectories[seed]
        elif str(seed) in optimization_trajectories:
            trajectory = optimization_trajectories[str(seed)]
        else:
            raise AcceptanceInputError(f"missing formal optimization trajectory for seed {seed}")
        if not isinstance(trajectory, Mapping):
            raise AcceptanceInputError(f"trajectory for seed {seed} must be an object")
        result.append((seed, trajectory))
    if len(optimization_trajectories) != len(result):
        expected = {str(seed) for seed, _ in result}
        observed = {str(seed) for seed in optimization_trajectories}
        raise AcceptanceInputError(
            f"formal optimization trajectory seeds differ: expected={sorted(expected)}, "
            f"observed={sorted(observed)}"
        )
    return result


def _checkpoint_grid(section: Mapping[str, object]) -> Array:
    start = _integer(section, "start_nof")
    stop = _integer(section, "stop_nof")
    step = _integer(section, "step_nof")
    if start < 0 or stop < start or step <= 0 or (stop - start) % step:
        raise AcceptanceInputError("registered checkpoint range is invalid")
    return np.arange(start, stop + step, step, dtype=np.int64)


def _exact_indices(trajectory: Trajectory, checkpoints: Array, label: str) -> Array:
    nof = np.asarray(trajectory.get("nof_mean_force_incremental"), dtype=np.int64)
    if nof.ndim != 1 or nof.size == 0 or np.any(np.diff(nof) <= 0):
        raise AcceptanceInputError(f"{label} has an invalid NoF sequence")
    positions = np.searchsorted(nof, checkpoints)
    valid = positions < nof.size
    if not np.all(valid) or not np.array_equal(nof[positions], checkpoints):
        missing = checkpoints[~valid | (nof[np.minimum(positions, nof.size - 1)] != checkpoints)]
        raise AcceptanceInputError(
            f"{label} is missing registered checkpoints {missing.tolist()}"
        )
    return positions


def _metric_at(trajectory: Trajectory, key: str, indices: Array, label: str) -> Array:
    if key not in trajectory:
        raise AcceptanceInputError(f"{label} lacks metric {key}")
    values = np.asarray(trajectory[key], dtype=np.float64)
    nof = np.asarray(trajectory["nof_mean_force_incremental"])
    if values.shape != nof.shape:
        raise AcceptanceInputError(f"{label}.{key} shape differs from its NoF sequence")
    result = values[indices]
    if np.any(~np.isfinite(result)):
        raise AcceptanceInputError(f"{label}.{key} contains nonfinite checkpoint values")
    return result


def _operator_from_rule(rule: object):
    if not isinstance(rule, str):
        raise AcceptanceInputError("registered component comparison must be text")
    if "<=" in rule:
        return np.less_equal, "<="
    if ">=" in rule:
        return np.greater_equal, ">="
    if "<" in rule:
        return np.less, "<"
    if ">" in rule:
        return np.greater, ">"
    raise AcceptanceInputError(f"cannot determine comparison operator from {rule!r}")


def _normalized_auc(values: Array, checkpoints: Array) -> float:
    values = np.asarray(values, dtype=np.float64)
    checkpoints = np.asarray(checkpoints, dtype=np.float64)
    if values.ndim != 1 or values.size != checkpoints.size or values.size < 2:
        raise AcceptanceInputError("tail AUC requires at least two aligned checkpoints")
    width = float(checkpoints[-1] - checkpoints[0])
    if width <= 0.0:
        raise AcceptanceInputError("tail AUC checkpoint width must be positive")
    return float(np.trapz(values, checkpoints) / width)


def _performance_values(
    optimization_trajectories: Mapping[Union[int, str], Trajectory],
    historical_current_ivm: Trajectory,
    registration: Registration,
    checkpoints: Array,
) -> Tuple[List[int], Dict[str, Array], Dict[str, Array]]:
    items = _optimization_items(optimization_trajectories, registration)
    barrier_reference = _number(_mapping(registration, "barrier_reference"), "value_kcal_per_mol")
    metric_sources = {
        "I_perp_G": "I_perp_G",
        "I_arc": "I_arc",
        "Delta_F_error": "Delta_F_barrier",
    }
    candidate: Dict[str, Array] = {}
    baseline: Dict[str, Array] = {}
    baseline_indices = _exact_indices(historical_current_ivm, checkpoints, "current IVM")
    seeds = [seed for seed, _ in items]
    for seed, trajectory in items:
        try:
            validate_public_trajectory(
                trajectory, expected_budget=FORMAL_INCREMENTAL_NOF_BUDGET
            )
        except FormalContractError as error:
            raise AcceptanceInputError(
                f"IR-IVM seed {seed} violates the exact V17 public prefix: {error}"
            ) from error
    for public_name, source_name in metric_sources.items():
        rows = []
        for seed, trajectory in items:
            indices = _exact_indices(trajectory, checkpoints, f"IR-IVM seed {seed}")
            rows.append(_metric_at(trajectory, source_name, indices, f"IR-IVM seed {seed}"))
        candidate_values = np.stack(rows)
        baseline_values = _metric_at(
            historical_current_ivm, source_name, baseline_indices, "current IVM"
        )
        if public_name == "Delta_F_error":
            candidate_values = np.abs(candidate_values - barrier_reference)
            baseline_values = np.abs(baseline_values - barrier_reference)
        candidate[public_name] = candidate_values
        baseline[public_name] = baseline_values
    return seeds, candidate, baseline


def evaluate_cost_history(
    optimization_trajectories: Mapping[Union[int, str], Trajectory],
    historical_current_ivm: Trajectory,
    registration: Registration,
) -> Dict[str, object]:
    """Evaluate the registered front/middle and noise-tolerant tail rules."""

    acceptance = _mapping(registration, "acceptance")
    front_rule = _mapping(acceptance, "front_middle")
    tail_rule = _mapping(acceptance, "tail")
    component_rules = _mapping(front_rule, "required_at_every_checkpoint")
    metric_names = ("I_perp_G", "I_arc", "Delta_F_error")
    if set(component_rules) != set(metric_names):
        raise AcceptanceInputError("front/middle component rules do not name all three metrics")

    front_checkpoints = _checkpoint_grid(front_rule)
    front_seeds, front_candidate, front_baseline = _performance_values(
        optimization_trajectories,
        historical_current_ivm,
        registration,
        front_checkpoints,
    )
    front_metrics: Dict[str, object] = {}
    front_pass = True
    for name in metric_names:
        candidate_median = np.median(front_candidate[name], axis=0)
        comparison, symbol = _operator_from_rule(component_rules[name])
        passed = comparison(candidate_median, front_baseline[name])
        metric_pass = bool(np.all(passed))
        front_pass = front_pass and metric_pass
        front_metrics[name] = {
            "comparison": symbol,
            "candidate_median": candidate_median.tolist(),
            "current_ivm": front_baseline[name].tolist(),
            "checkpoint_pass": passed.tolist(),
            "failure_nof": front_checkpoints[~passed].astype(int).tolist(),
            "pass": metric_pass,
        }

    tail_checkpoints = _checkpoint_grid(tail_rule)
    registered_count = tail_rule.get("checkpoint_count")
    if (
        type(registered_count) is not int
        or registered_count != tail_checkpoints.size
        or registered_count != 20
    ):
        raise AcceptanceInputError(
            "registered tail checkpoint_count must describe exactly 20 checkpoints"
        )
    tail_seeds, tail_candidate, tail_baseline = _performance_values(
        optimization_trajectories,
        historical_current_ivm,
        registration,
        tail_checkpoints,
    )
    if tail_seeds != front_seeds:
        raise AcceptanceInputError("front and tail optimization-seed order differs")
    if len(tail_seeds) != 3:
        raise AcceptanceInputError("tail acceptance requires exactly three candidate paths")
    minimum_win_fraction = _number(
        tail_rule, "minimum_checkpoint_win_fraction_each_metric"
    )
    if not 0.0 <= minimum_win_fraction <= 1.0:
        raise AcceptanceInputError("tail win-fraction threshold is outside [0,1]")
    require_lower_median = _bool(tail_rule, "candidate_tail_median_lower_each_metric")
    require_lower_auc = _bool(
        tail_rule, "candidate_normalized_trapezoidal_auc_lower_each_metric"
    )
    minimum_individual = _integer(
        tail_rule, "minimum_individual_candidate_seeds_with_lower_auc_each_metric"
    )
    if not 0 <= minimum_individual <= len(tail_seeds):
        raise AcceptanceInputError("individual-seed tail AUC threshold is invalid")

    # The online I_perp and barrier estimates inherit restrained-MD noise, so
    # their tail gate is deliberately an aggregate curve-shape rule.  I_arc is
    # a deterministic function of the public path and is evaluated separately
    # below; treating it as another noisy observable would weaken the geometric
    # claim for no statistical reason.
    noisy_metric_names = ("I_perp_G", "Delta_F_error")
    tail_metrics: Dict[str, object] = {}
    tail_pass = True
    for name in noisy_metric_names:
        candidate_median = np.median(tail_candidate[name], axis=0)
        comparison, symbol = _operator_from_rule(component_rules[name])
        checkpoint_wins = comparison(candidate_median, tail_baseline[name])
        win_fraction = float(np.mean(checkpoint_wins))
        win_fraction_pass = bool(win_fraction >= minimum_win_fraction)
        candidate_tail_median = float(np.median(candidate_median))
        baseline_tail_median = float(np.median(tail_baseline[name]))
        median_pass = bool(candidate_tail_median < baseline_tail_median)
        candidate_auc = _normalized_auc(candidate_median, tail_checkpoints)
        baseline_auc = _normalized_auc(tail_baseline[name], tail_checkpoints)
        auc_pass = bool(candidate_auc < baseline_auc)
        individual_aucs = [
            _normalized_auc(row, tail_checkpoints) for row in tail_candidate[name]
        ]
        individual_pass = [bool(value < baseline_auc) for value in individual_aucs]
        individual_count = int(sum(individual_pass))
        individual_count_pass = bool(individual_count >= minimum_individual)
        metric_pass = bool(
            win_fraction_pass
            and (median_pass or not require_lower_median)
            and (auc_pass or not require_lower_auc)
            and individual_count_pass
        )
        tail_pass = tail_pass and metric_pass
        tail_metrics[name] = {
            "gate_kind": "noisy_curve_shape",
            "comparison": symbol,
            "candidate_checkpoint_median": candidate_median.tolist(),
            "current_ivm": tail_baseline[name].tolist(),
            "checkpoint_wins": checkpoint_wins.tolist(),
            "win_fraction": win_fraction,
            "minimum_win_fraction": minimum_win_fraction,
            "win_fraction_pass": win_fraction_pass,
            "candidate_tail_median": candidate_tail_median,
            "current_ivm_tail_median": baseline_tail_median,
            "tail_median_pass": median_pass,
            "tail_median_required": require_lower_median,
            "candidate_normalized_auc": candidate_auc,
            "current_ivm_normalized_auc": baseline_auc,
            "normalized_auc_pass": auc_pass,
            "normalized_auc_required": require_lower_auc,
            "individual_candidate_normalized_auc": individual_aucs,
            "individual_candidate_auc_pass": individual_pass,
            "individual_candidate_auc_pass_count": individual_count,
            "minimum_individual_candidate_auc_pass_count": minimum_individual,
            "pass": metric_pass,
        }

    minimum_I_arc_individual = _integer(
        tail_rule,
        "minimum_individual_candidate_paths_with_lower_I_arc_auc",
    )
    if minimum_I_arc_individual != 2:
        raise AcceptanceInputError(
            "I_arc tail AUC gate must require at least two of three candidate paths"
        )
    candidate_I_arc = tail_candidate["I_arc"]
    baseline_I_arc = tail_baseline["I_arc"]
    candidate_I_arc_median = np.median(candidate_I_arc, axis=0)
    I_arc_checkpoint_pass = candidate_I_arc_median <= baseline_I_arc
    candidate_I_arc_individual_auc = [
        _normalized_auc(row, tail_checkpoints) for row in candidate_I_arc
    ]
    baseline_I_arc_auc = _normalized_auc(baseline_I_arc, tail_checkpoints)
    I_arc_individual_pass = [
        bool(value < baseline_I_arc_auc)
        for value in candidate_I_arc_individual_auc
    ]
    I_arc_individual_count = int(sum(I_arc_individual_pass))
    I_arc_pass = bool(
        np.all(I_arc_checkpoint_pass)
        and I_arc_individual_count >= minimum_I_arc_individual
    )
    tail_pass = tail_pass and I_arc_pass
    tail_metrics["I_arc"] = {
        "gate_kind": "deterministic_path_geometry",
        "comparison": "<=",
        "candidate_checkpoint_median": candidate_I_arc_median.tolist(),
        "current_ivm": baseline_I_arc.tolist(),
        "checkpoint_pass": I_arc_checkpoint_pass.tolist(),
        "failure_nof": tail_checkpoints[~I_arc_checkpoint_pass].astype(int).tolist(),
        "all_checkpoint_pass": bool(np.all(I_arc_checkpoint_pass)),
        "candidate_individual_normalized_auc": candidate_I_arc_individual_auc,
        "current_ivm_normalized_auc": baseline_I_arc_auc,
        "individual_candidate_auc_pass": I_arc_individual_pass,
        "individual_candidate_auc_pass_count": I_arc_individual_count,
        "minimum_individual_candidate_auc_pass_count": minimum_I_arc_individual,
        "pass": I_arc_pass,
    }

    return {
        "optimization_seeds": front_seeds,
        "barrier_comparison": "absolute error relative to registered Delta_F_star",
        "front_middle": {
            "checkpoint_nof": front_checkpoints.astype(int).tolist(),
            "metrics": front_metrics,
            "pass": bool(front_pass),
        },
        "tail": {
            "checkpoint_nof": tail_checkpoints.astype(int).tolist(),
            "noisy_metric_names": list(noisy_metric_names),
            "deterministic_geometry_metric_name": "I_arc",
            "metrics": tail_metrics,
            "pass": bool(tail_pass),
        },
        "pass": bool(front_pass and tail_pass),
    }


def select_exact_paid_public_state(trajectory: Trajectory, required_nof: int) -> int:
    """Return the actual V17 public state at one exact paid checkpoint.

    A movement, exact-center HOLD, or independent warm-validation band is a
    paid public state.  The selector neither carries an older movement forward
    nor substitutes an unpurchased pending candidate.
    """

    try:
        nof = np.asarray(
            trajectory.get("nof_mean_force_incremental"), dtype=np.int64
        )
    except (TypeError, ValueError) as error:
        raise AcceptanceInputError("invalid public NoF sequence") from error
    kinds = np.asarray(trajectory.get("state_kind"))
    if (
        nof.ndim != 1
        or nof.size == 0
        or np.any(np.diff(nof) <= 0)
        or kinds.shape != nof.shape
    ):
        raise AcceptanceInputError("trajectory lacks an aligned increasing public-state ledger")
    matches = np.flatnonzero(nof == int(required_nof))
    if matches.size != 1 or int(required_nof) <= 0:
        raise AcceptanceInputError(
            f"trajectory lacks exactly one paid public state at NoF {required_nof}"
        )
    index = int(matches[0])
    if str(kinds[index]) not in PAID_STATE_KINDS:
        raise AcceptanceInputError(
            f"state at NoF {required_nof} is not one of the three registered V17 paid states"
        )
    return index


def _select_exact_final_paid_public_state(
    trajectory: Trajectory,
    required_nof: int,
) -> int:
    """Validate and select only the exact terminal V17 public state."""

    try:
        terminal = validate_public_trajectory(
            trajectory, expected_budget=int(required_nof)
        )
    except FormalContractError as error:
        raise AcceptanceInputError(str(error)) from error
    return int(terminal.index)


def _correct_path_for_seed(
    seed: int,
    trajectory: Trajectory,
    gate: Mapping[str, object],
    barrier_reference: float,
    saddle: Array,
    fes_minimax_path: Array,
    maximum_nof: int,
) -> Dict[str, object]:
    failures: List[str] = []
    nof = np.asarray(trajectory.get("nof_mean_force_incremental"), dtype=np.int64)
    paths = np.asarray(trajectory.get("full_path_unwrapped"), dtype=np.float64)
    expected_shape_raw = gate.get("shape")
    if not isinstance(expected_shape_raw, list) or any(type(item) is not int for item in expected_shape_raw):
        raise AcceptanceInputError("correct_path_gate.shape must be an integer list")
    expected_shape = tuple(int(item) for item in expected_shape_raw)
    if paths.ndim != 3 or paths.shape[1:] != expected_shape or paths.shape[0] != nof.size:
        failures.append("path_shape")

    selected_index: Optional[int]
    selection_error: Optional[str]
    try:
        selected_index = _select_exact_final_paid_public_state(
            trajectory,
            maximum_nof,
        )
        selection_error = None
    except AcceptanceInputError as error:
        selected_index = None
        selection_error = str(error)
        failures.append("missing_exact_paid_terminal_public_state")

    finite_required = _bool(gate, "finite_path_and_metrics")
    if finite_required:
        for key in ("full_path_unwrapped", "M", "I_perp_G", "I_arc", "Delta_F_barrier"):
            if key not in trajectory or np.any(~np.isfinite(np.asarray(trajectory[key], dtype=np.float64))):
                failures.append(f"nonfinite_or_missing_{key}")

    endpoint_tolerance = _number(gate, "endpoint_absolute_tolerance_rad")
    start = np.asarray(gate.get("start_endpoint_rad"), dtype=np.float64)
    target = np.asarray(gate.get("target_endpoint_rad"), dtype=np.float64)
    if paths.ndim == 3 and paths.shape[1:] == expected_shape:
        if start.shape != paths.shape[2:] or not np.allclose(
            paths[:, 0], start, rtol=0.0, atol=endpoint_tolerance
        ):
            failures.append("start_endpoint")
        if target.shape != paths.shape[2:] or not np.allclose(
            paths[:, -1], target, rtol=0.0, atol=endpoint_tolerance
        ):
            failures.append("target_endpoint")

    minimum_eigenvalue = _number(
        gate, "metric_minimum_eigenvalue_strictly_greater_than"
    )
    M = np.asarray(trajectory.get("M"), dtype=np.float64)
    internal_count = expected_shape[0] - 2 if len(expected_shape) == 2 else -1
    if M.shape != (nof.size, internal_count, expected_shape[-1], expected_shape[-1]):
        failures.append("metric_shape")
        minimum_observed_eigenvalue: Optional[float] = None
    elif np.any(~np.isfinite(M)):
        failures.append("metric_nonfinite")
        minimum_observed_eigenvalue = None
    else:
        symmetric_M = 0.5 * (M + np.swapaxes(M, -1, -2))
        minimum_observed_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric_M)))
        if minimum_observed_eigenvalue <= minimum_eigenvalue:
            failures.append("metric_not_safely_spd")

    details: Dict[str, object] = {}
    if selected_index is not None and paths.ndim == 3 and paths.shape[1:] == expected_shape:
        selected_path = paths[selected_index]
        segment_lengths = np.linalg.norm(
            periodic_difference(selected_path[1:], selected_path[:-1]), axis=1
        )
        if np.any(~np.isfinite(segment_lengths)) or np.any(segment_lengths <= 0.0):
            relative_deviation = float("inf")
            maximum_segment = float("inf")
        else:
            relative_deviation = float(
                np.max(np.abs(segment_lengths / np.mean(segment_lengths) - 1.0))
            )
            maximum_segment = float(np.max(segment_lengths))
        relative_limit = _number(
            gate, "selected_path_relative_segment_length_deviation_max"
        )
        segment_limit = _number(
            gate, "selected_path_maximum_shortest_periodic_segment_rad"
        )
        if not relative_deviation <= relative_limit:
            failures.append("selected_path_equal_arclength")

        recorded_deviation_values = np.asarray(
            trajectory.get("segment_length_relative_deviation"), dtype=np.float64
        )
        if (
            recorded_deviation_values.shape != nof.shape
            or not np.isfinite(recorded_deviation_values[selected_index])
        ):
            recorded_deviation: Optional[float] = None
            failures.append("invalid_public_chord_relative_deviation")
        else:
            recorded_deviation = float(recorded_deviation_values[selected_index])
            if not np.isclose(
                recorded_deviation,
                relative_deviation,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                failures.append("public_chord_relative_deviation_mismatch")
            if recorded_deviation > relative_limit:
                failures.append("public_chord_relative_deviation_above_limit")

        output_residual_values = np.asarray(
            trajectory.get("selected_reparameterization_output_residual"),
            dtype=np.float64,
        )
        output_residual_limit = _number(
            gate,
            "selected_path_article_spring_output_residual_strictly_less_than",
        )
        if (
            output_residual_values.shape != nof.shape
            or not np.isfinite(output_residual_values[selected_index])
        ):
            output_residual: Optional[float] = None
            failures.append("invalid_public_spring_output_residual")
        else:
            output_residual = float(output_residual_values[selected_index])
            if not 0.0 <= output_residual < output_residual_limit:
                failures.append("public_spring_output_residual_not_strictly_below_limit")
        if not maximum_segment <= segment_limit:
            failures.append("selected_path_segment_jump")

        displacement = periodic_difference(
            selected_path[:, None, :], fes_minimax_path[None, :, :]
        )
        nearest = np.min(np.linalg.norm(displacement, axis=2), axis=1)
        nearest_maximum = float(np.max(nearest))
        nearest_rms = float(np.sqrt(np.mean(nearest * nearest)))
        nearest_maximum_limit = _number(
            gate,
            "selected_path_each_image_maximum_nearest_fes_minimax_path_distance_rad",
        )
        nearest_rms_limit = _number(
            gate, "selected_path_rms_nearest_fes_minimax_path_distance_rad"
        )
        if not nearest_maximum <= nearest_maximum_limit:
            failures.append("selected_path_fes_corridor_maximum")
        if not nearest_rms <= nearest_rms_limit:
            failures.append("selected_path_fes_corridor_rms")

        saddle_distance = float(
            np.min(np.linalg.norm(periodic_difference(selected_path, saddle), axis=1))
        )
        saddle_limit = _number(gate, "selected_path_minimax_saddle_proximity_rad")
        if not saddle_distance <= saddle_limit:
            failures.append("selected_path_saddle_proximity")

        barriers = np.asarray(trajectory.get("Delta_F_barrier"), dtype=np.float64)
        if barriers.shape != nof.shape or not np.isfinite(barriers[selected_index]):
            selected_barrier = None
            barrier_error = None
            failures.append("selected_path_barrier")
        else:
            selected_barrier = float(barriers[selected_index])
            barrier_error = abs(selected_barrier - barrier_reference)
            barrier_limit = _number(
                gate, "selected_path_barrier_error_max_kcal_per_mol"
            )
            if not barrier_error <= barrier_limit:
                failures.append("selected_path_barrier_error")
        details = {
            "selected_state_index": selected_index,
            "selected_nof": int(nof[selected_index]),
            "selected_state_kind": str(np.asarray(trajectory["state_kind"])[selected_index]),
            "article_spring_output_residual": output_residual,
            "article_spring_output_residual_strict_upper_limit": output_residual_limit,
            "segment_length_relative_deviation": relative_deviation,
            "recorded_segment_length_relative_deviation": recorded_deviation,
            "segment_length_relative_deviation_upper_limit": relative_limit,
            "maximum_shortest_periodic_segment_rad": maximum_segment,
            "maximum_nearest_fes_path_distance_rad": nearest_maximum,
            "rms_nearest_fes_path_distance_rad": nearest_rms,
            "minimum_saddle_distance_rad": saddle_distance,
            "Delta_F_barrier": selected_barrier,
            "Delta_F_error": barrier_error,
        }

    return {
        "seed": int(seed),
        "minimum_observed_metric_eigenvalue": minimum_observed_eigenvalue,
        "terminal_state_selection_error": selection_error,
        **details,
        "failures": sorted(set(failures)),
        "pass": not failures,
    }


def evaluate_correct_path(
    optimization_trajectories: Mapping[Union[int, str], Trajectory],
    fes_minimax_path: Array,
    registration: Registration,
) -> Dict[str, object]:
    """Evaluate the registered correct-channel gate for every formal run."""

    fes_path = np.asarray(fes_minimax_path, dtype=np.float64)
    if fes_path.ndim != 2 or fes_path.shape[1] != 2 or np.any(~np.isfinite(fes_path)):
        raise AcceptanceInputError("FES minimax reference path must have shape (n,2)")
    gate = _mapping(_mapping(registration, "acceptance"), "correct_path_guard")
    barrier = _mapping(registration, "barrier_reference")
    saddle = np.asarray(barrier.get("saddle_coordinate_rad"), dtype=np.float64)
    if saddle.shape != (2,) or np.any(~np.isfinite(saddle)):
        raise AcceptanceInputError("registered minimax saddle coordinate is invalid")
    barrier_reference = _number(barrier, "value_kcal_per_mol")
    maximum_nof = _integer(
        _mapping(registration, "nof"), "incremental_budget_per_optimization_run"
    )
    if maximum_nof != FORMAL_INCREMENTAL_NOF_BUDGET:
        raise AcceptanceInputError("correct-path gate requires exact terminal NoF 2000")
    registered_kinds = _mapping(registration, "public_state_contract").get(
        "paid_state_kinds"
    )
    if registered_kinds != list(PAID_STATE_KINDS):
        raise AcceptanceInputError("correct-path gate requires all three V17 paid states")
    output_residual_limit = _number(
        gate,
        "selected_path_article_spring_output_residual_strictly_less_than",
    )
    if output_residual_limit != 1.0e-6:
        raise AcceptanceInputError(
            "correct-path spring output residual threshold must be exactly 1e-6"
        )
    chord_relative_limit = _number(
        gate, "selected_path_relative_segment_length_deviation_max"
    )
    if chord_relative_limit != 1.0e-4:
        raise AcceptanceInputError(
            "correct-path chord relative-deviation threshold must be exactly 1e-4"
        )
    results = [
        _correct_path_for_seed(
            seed,
            trajectory,
            gate,
            barrier_reference,
            saddle,
            fes_path,
            maximum_nof,
        )
        for seed, trajectory in _optimization_items(
            optimization_trajectories, registration
        )
    ]
    required_each = _bool(gate, "required_for_every_optimization_run")
    passed = all(bool(result["pass"]) for result in results)
    return {
        "evaluated_nof_maximum": maximum_nof,
        "required_for_every_optimization_run": required_each,
        "seeds": results,
        "pass": bool(passed if required_each else any(bool(item["pass"]) for item in results)),
    }


_TERMINAL_ANCHORS = (1620, 1800, 2000)
_TERMINAL_BOOTSTRAP_RESAMPLES = 100000
_TERMINAL_CONFIDENCE = 0.95
_TERMINAL_CANDIDATE_PATHS = 3
_TERMINAL_BASELINE_PATHS = 1
_TERMINAL_NOF_PER_FIXED_PATH = 20
_TERMINAL_TOTAL_NOF = 2400


def _integer_list(parent: Mapping[str, object], key: str) -> Tuple[int, ...]:
    values = parent.get(key)
    if not isinstance(values, list) or not values or any(
        type(item) is not int for item in values
    ):
        raise AcceptanceInputError(f"registration field {key!r} must be a nonempty integer list")
    result = tuple(int(item) for item in values)
    if len(set(result)) != len(result):
        raise AcceptanceInputError(f"registration field {key!r} contains duplicates")
    return result


def _diagnostic_candidate_entry(
    candidates: Mapping[Union[int, str], object], seed: int
) -> Mapping[str, object]:
    if seed in candidates:
        value = candidates[seed]
    elif str(seed) in candidates:
        value = candidates[str(seed)]
    else:
        raise AcceptanceInputError(f"terminal diagnostics lack candidate seed {seed}")
    if not isinstance(value, Mapping):
        raise AcceptanceInputError(f"terminal candidate diagnostics for {seed} must be an object")
    return value


def _diagnostic_matrix(
    entry: Mapping[str, object],
    key: str,
    expected_anchor_count: int,
    expected_seed_count: int,
    label: str,
) -> Array:
    if key not in entry:
        raise AcceptanceInputError(f"{label} lacks {key}")
    values = np.asarray(entry[key], dtype=np.float64)
    expected_shape = (expected_anchor_count, expected_seed_count)
    if values.shape != expected_shape or np.any(~np.isfinite(values)):
        raise AcceptanceInputError(
            f"{label}.{key} must be a finite matrix with shape {expected_shape} "
            "(anchor, diagnostic seed)"
        )
    return values


def _simultaneous_cluster_bootstrap(
    paired_log_ratio_clusters: Array,
    resamples: int,
    confidence: float,
    threshold: float,
    bootstrap_seed: int,
    cell_labels: List[str],
) -> Dict[str, object]:
    """Bootstrap whole diagnostic-seed clusters jointly across all six cells."""

    values = np.asarray(paired_log_ratio_clusters, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] != len(cell_labels)
        or np.any(~np.isfinite(values))
    ):
        raise AcceptanceInputError("paired log-ratio cluster matrix is invalid")
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, values.shape[0], size=(resamples, values.shape[0]))
    bootstrap_cell_means = np.mean(values[indices], axis=1)
    bootstrap_maximum = np.max(bootstrap_cell_means, axis=1)
    upper = float(np.percentile(bootstrap_maximum, 100.0 * confidence))
    observed_means = np.mean(values, axis=0)
    return {
        "cluster_definition": "one complete diagnostic seed across all metrics and anchors",
        "cluster_count": int(values.shape[0]),
        "cell_labels": cell_labels,
        "paired_log_ratio_cluster_matrix": values.tolist(),
        "observed_mean_paired_log_ratio_by_cell": observed_means.tolist(),
        "observed_maximum_mean_paired_log_ratio": float(np.max(observed_means)),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_resamples": int(resamples),
        "upper_one_sided_simultaneous_percentile_bound": upper,
        "confidence_level": confidence,
        "upper_bound_threshold": threshold,
        "pass": bool(upper < threshold),
    }


def evaluate_terminal_noise_test(
    optimization_trajectories: Mapping[Union[int, str], Trajectory],
    historical_current_ivm: Trajectory,
    terminal_diagnostics: Mapping[str, object],
    registration: Registration,
) -> Dict[str, object]:
    """Evaluate the fixed-anchor, fixed-path paired diagnostic design.

    Metric matrices are ordered ``(anchor, diagnostic seed)``.  A diagnostic
    seed is the bootstrap cluster: all three anchors, both noisy quantities,
    all three candidate paths, and the paired current-IVM path stay together.
    This function performs no MD and does not permit optional extension.
    """

    if not isinstance(terminal_diagnostics, Mapping):
        raise AcceptanceInputError("terminal diagnostics must be an object")
    if {"stage", "extension_required", "extension_was_run"} & set(
        terminal_diagnostics
    ):
        raise AcceptanceInputError(
            "terminal evidence must not carry optional extension state"
        )
    terminal_rule = _mapping(_mapping(registration, "acceptance"), "terminal_diagnostics")
    registered_seed_section = _mapping(registration, "seeds")
    forbidden_seed_fields = {
        "terminal_diagnostic_initial",
        "terminal_diagnostic_extension",
    }
    if forbidden_seed_fields & set(registered_seed_section):
        raise AcceptanceInputError(
            "optional terminal extension seed schedules are not permitted"
        )
    forbidden_terminal_fields = {
        "extension_trigger",
        "initial_diagnostic_nof",
        "maximum_total_diagnostic_nof",
        "further_optional_stopping",
        "I_arc_rule",
        "minimum_individual_candidate_paths_with_lower_I_arc",
    }
    if forbidden_terminal_fields & set(terminal_rule):
        raise AcceptanceInputError(
            "optional terminal extension rules are not permitted"
        )
    registered_seeds = _seed_list(registration, "terminal_diagnostic")
    if len(registered_seeds) != 10:
        raise AcceptanceInputError(
            "registered terminal diagnostic schedule must contain ten prospectively sealed seeds"
        )
    supplied_seeds_raw = terminal_diagnostics.get("diagnostic_seeds")
    if not isinstance(supplied_seeds_raw, list) or any(
        type(item) is not int for item in supplied_seeds_raw
    ):
        raise AcceptanceInputError("diagnostic_seeds must be an integer list")
    supplied_seeds = tuple(int(item) for item in supplied_seeds_raw)
    if supplied_seeds != registered_seeds:
        raise AcceptanceInputError(
            "diagnostic seeds must exactly equal the registered fixed schedule"
        )

    registered_anchors = _integer_list(terminal_rule, "anchor_nof")
    if registered_anchors != _TERMINAL_ANCHORS:
        raise AcceptanceInputError(
            "registered terminal anchors must be exactly [1620, 1800, 2000]"
        )
    supplied_anchors = terminal_diagnostics.get("anchor_nof")
    if not isinstance(supplied_anchors, list) or any(
        type(item) is not int for item in supplied_anchors
    ):
        raise AcceptanceInputError("terminal anchor_nof must be an integer list")
    anchors = tuple(int(item) for item in supplied_anchors)
    if anchors != registered_anchors:
        raise AcceptanceInputError(
            "terminal diagnostic anchors differ from the registered fixed anchors"
        )
    maximum_nof = _integer(
        _mapping(registration, "nof"), "incremental_budget_per_optimization_run"
    )
    if maximum_nof != anchors[-1]:
        raise AcceptanceInputError(
            "optimization budget must equal the final registered diagnostic anchor"
        )

    optimization_items = _optimization_items(optimization_trajectories, registration)
    candidate_path_count = _integer(
        terminal_rule, "candidate_path_count_per_anchor"
    )
    baseline_path_count = _integer(
        terminal_rule, "baseline_path_count_per_anchor"
    )
    fixed_path_count = _integer(
        terminal_rule, "fixed_path_count_per_anchor_per_diagnostic_seed"
    )
    if (
        candidate_path_count != _TERMINAL_CANDIDATE_PATHS
        or candidate_path_count != len(optimization_items)
        or baseline_path_count != _TERMINAL_BASELINE_PATHS
        or fixed_path_count != candidate_path_count + baseline_path_count
    ):
        raise AcceptanceInputError(
            "terminal fixed-path counts must be three candidate paths plus one baseline"
        )

    nof_per_path = _integer(terminal_rule, "mean_force_nof_per_fixed_path")
    registered_total_nof = _integer(terminal_rule, "total_diagnostic_nof")
    computed_total_nof = (
        len(anchors)
        * len(supplied_seeds)
        * fixed_path_count
        * nof_per_path
    )
    if (
        nof_per_path != _TERMINAL_NOF_PER_FIXED_PATH
        or registered_total_nof != _TERMINAL_TOTAL_NOF
        or computed_total_nof != registered_total_nof
    ):
        raise AcceptanceInputError(
            "terminal NoF accounting must be 3 anchors x 10 seeds x 4 paths x 20 = 2400"
        )
    supplied_total_nof = terminal_diagnostics.get("nof_mean_force_total")
    if type(supplied_total_nof) is not int or supplied_total_nof != computed_total_nof:
        raise AcceptanceInputError(
            "terminal evidence must report exactly 2400 mean-force NoF"
        )

    candidate_diagnostics = terminal_diagnostics.get("candidate")
    current_diagnostics = terminal_diagnostics.get("current_ivm")
    if not isinstance(candidate_diagnostics, Mapping) or not isinstance(
        current_diagnostics, Mapping
    ):
        raise AcceptanceInputError("terminal diagnostics require candidate/current_ivm objects")

    expected_candidate_keys = {str(seed) for seed, _ in optimization_items}
    if (
        len(candidate_diagnostics) != len(expected_candidate_keys)
        or {str(key) for key in candidate_diagnostics} != expected_candidate_keys
    ):
        raise AcceptanceInputError(
            "terminal candidate diagnostics must name exactly the formal optimization paths"
        )

    candidate_I_rows: List[Array] = []
    candidate_D_rows: List[Array] = []
    selected_nof: Dict[str, List[int]] = {}
    barrier_reference = _number(
        _mapping(registration, "barrier_reference"), "value_kcal_per_mol"
    )
    for seed, trajectory in optimization_items:
        trajectory_nof = np.asarray(trajectory["nof_mean_force_incremental"], dtype=np.int64)
        selected_indices = [
            select_exact_paid_public_state(trajectory, anchor)
            for anchor in anchors
        ]
        expected_selected_nof = [
            int(trajectory_nof[index]) for index in selected_indices
        ]
        if tuple(expected_selected_nof) != anchors:
            raise AcceptanceInputError(
                f"candidate seed {seed} lacks an exact V17 paid public state "
                "at every diagnostic anchor"
            )
        entry = _diagnostic_candidate_entry(candidate_diagnostics, seed)
        if _integer_list(entry, "anchor_nof") != anchors:
            raise AcceptanceInputError(
                f"terminal diagnostics for seed {seed} bind the wrong anchor paths"
            )
        selected_nof[str(seed)] = expected_selected_nof
        candidate_I_rows.append(
            _diagnostic_matrix(
                entry,
                "I_perp_G",
                len(anchors),
                len(supplied_seeds),
                f"candidate {seed}",
            )
        )
        candidate_barrier = _diagnostic_matrix(
            entry,
            "Delta_F_barrier",
            len(anchors),
            len(supplied_seeds),
            f"candidate {seed}",
        )
        candidate_D_rows.append(np.abs(candidate_barrier - barrier_reference))

    if _integer_list(current_diagnostics, "anchor_nof") != anchors:
        raise AcceptanceInputError("current-IVM diagnostics bind the wrong anchor paths")
    current_indices = _exact_indices(
        historical_current_ivm, np.asarray(anchors, dtype=np.int64), "current IVM"
    )
    if current_indices.size != len(anchors):
        raise AcceptanceInputError("current IVM lacks registered diagnostic anchors")
    current_I = _diagnostic_matrix(
        current_diagnostics,
        "I_perp_G",
        len(anchors),
        len(supplied_seeds),
        "current IVM",
    )
    current_barrier = _diagnostic_matrix(
        current_diagnostics,
        "Delta_F_barrier",
        len(anchors),
        len(supplied_seeds),
        "current IVM",
    )
    current_D = np.abs(current_barrier - barrier_reference)

    floor = _number(terminal_rule, "log_ratio_floor")
    if floor <= 0.0:
        raise AcceptanceInputError("terminal log-ratio floor must be positive")
    resamples = _integer(terminal_rule, "bootstrap_resamples")
    confidence = _number(terminal_rule, "one_sided_confidence_level")
    threshold = _number(terminal_rule, "paired_log_ratio_upper_bound_threshold")
    bootstrap_seed = _integer(_mapping(registration, "seeds"), "bootstrap")
    if (
        resamples != _TERMINAL_BOOTSTRAP_RESAMPLES
        or confidence != _TERMINAL_CONFIDENCE
        or threshold != 0.0
        or type(bootstrap_seed) is not int
    ):
        raise AcceptanceInputError(
            "terminal simultaneous bootstrap must use 100000 resamples, "
            "one prospectively sealed integer seed, one-sided 95%, and threshold zero"
        )

    candidate_I = np.stack(candidate_I_rows)
    candidate_D = np.stack(candidate_D_rows)
    if np.any(candidate_I < 0.0) or np.any(current_I < 0.0):
        raise AcceptanceInputError("I_perp_G diagnostic values must be nonnegative")
    candidate_metrics = {
        "I_perp_G": candidate_I,
        "Delta_F_error": candidate_D,
    }
    baseline_metrics = {
        "I_perp_G": current_I,
        "Delta_F_error": current_D,
    }
    minimum_individual = _integer(
        terminal_rule,
        "minimum_individual_candidate_paths_with_lower_auc_each_metric",
    )
    if minimum_individual != 2:
        raise AcceptanceInputError(
            "terminal point-estimate AUC gate must require two of three candidate paths"
        )

    metrics: Dict[str, object] = {}
    cluster_cells: List[Array] = []
    cell_labels: List[str] = []
    individual_gates_pass = True
    for metric_name in ("I_perp_G", "Delta_F_error"):
        candidate_values = candidate_metrics[metric_name]
        baseline_values = baseline_metrics[metric_name]
        candidate_median = np.median(candidate_values, axis=0)
        paired_log_ratio = np.log(
            np.maximum(candidate_median, floor)
            / np.maximum(baseline_values, floor)
        )
        for anchor_index, anchor in enumerate(anchors):
            cluster_cells.append(paired_log_ratio[anchor_index])
            cell_labels.append(f"{metric_name}@NoF{anchor}")

        baseline_seed_auc = np.asarray(
            [
                _normalized_auc(baseline_values[:, index], np.asarray(anchors))
                for index in range(len(supplied_seeds))
            ],
            dtype=np.float64,
        )
        candidate_seed_auc = np.asarray(
            [
                [
                    _normalized_auc(
                        candidate_values[path_index, :, diagnostic_index],
                        np.asarray(anchors),
                    )
                    for diagnostic_index in range(len(supplied_seeds))
                ]
                for path_index in range(candidate_path_count)
            ],
            dtype=np.float64,
        )
        paired_auc_difference = candidate_seed_auc - baseline_seed_auc[None, :]
        point_estimate_difference = np.mean(paired_auc_difference, axis=1)
        candidate_point_estimate_auc = np.mean(candidate_seed_auc, axis=1)
        baseline_point_estimate_auc = float(np.mean(baseline_seed_auc))
        individual_pass = (point_estimate_difference < 0.0).tolist()
        individual_count = int(sum(individual_pass))
        metric_pass = bool(individual_count >= minimum_individual)
        individual_gates_pass = individual_gates_pass and metric_pass
        metrics[metric_name] = {
            "candidate_median_by_anchor_and_diagnostic_seed": candidate_median.tolist(),
            "current_ivm_by_anchor_and_diagnostic_seed": baseline_values.tolist(),
            "paired_log_ratio_by_anchor_and_diagnostic_seed": paired_log_ratio.tolist(),
            "mean_paired_log_ratio_by_anchor": np.mean(
                paired_log_ratio, axis=1
            ).tolist(),
            "candidate_point_estimate_auc_by_path": candidate_point_estimate_auc.tolist(),
            "current_ivm_point_estimate_auc": baseline_point_estimate_auc,
            "mean_paired_auc_difference_by_path": point_estimate_difference.tolist(),
            "individual_candidate_path_auc_pass": individual_pass,
            "individual_candidate_path_auc_pass_count": individual_count,
            "minimum_individual_candidate_path_auc_pass_count": minimum_individual,
            "individual_path_auc_gate_pass": metric_pass,
        }

    cluster_matrix = np.stack(cluster_cells, axis=1)
    simultaneous = _simultaneous_cluster_bootstrap(
        cluster_matrix,
        resamples,
        confidence,
        threshold,
        bootstrap_seed,
        cell_labels,
    )
    passed = bool(simultaneous["pass"] and individual_gates_pass)

    return {
        "diagnostic_seeds": list(supplied_seeds),
        "anchor_nof": list(anchors),
        "selected_candidate_path_nof": selected_nof,
        "current_ivm_path_nof": list(anchors),
        "nof_accounting": {
            "definition": "one complete 2D mean-force vector at one internal CV point = 1 NoF",
            "anchor_count": len(anchors),
            "diagnostic_seed_count": len(supplied_seeds),
            "fixed_path_count_per_anchor_per_seed": fixed_path_count,
            "mean_force_nof_per_fixed_path": nof_per_path,
            "computed_total_diagnostic_nof": computed_total_nof,
            "reported_total_diagnostic_nof": int(supplied_total_nof),
        },
        "barrier_comparison": "absolute error relative to registered Delta_F_star",
        "metrics": metrics,
        "simultaneous_bootstrap": simultaneous,
        "individual_path_auc_gates_pass": bool(individual_gates_pass),
        "decision": "pass" if passed else "fail",
        "pass": passed,
    }


def evaluate_campaign_acceptance(
    optimization_trajectories: Mapping[Union[int, str], Trajectory],
    historical_current_ivm: Trajectory,
    fes_minimax_path: Array,
    registration: Registration,
    terminal_diagnostics: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Return the complete machine-readable registered acceptance result."""

    selected = _mapping(registration, "candidate").get("selected_parameter")
    if not isinstance(selected, Mapping):
        raise AcceptanceInputError("formal candidate has not been explicitly sealed")
    parameter_id = selected.get("parameter_id")
    if not isinstance(parameter_id, str) or not parameter_id:
        raise AcceptanceInputError("formal candidate lacks a sealed parameter_id")

    cost_history = evaluate_cost_history(
        optimization_trajectories, historical_current_ivm, registration
    )
    correct_path = evaluate_correct_path(
        optimization_trajectories, fes_minimax_path, registration
    )
    if terminal_diagnostics is None:
        terminal: Optional[Dict[str, object]] = None
        decision = "pending_terminal_diagnostics"
        passed: Optional[bool] = None
    else:
        terminal = evaluate_terminal_noise_test(
            optimization_trajectories,
            historical_current_ivm,
            terminal_diagnostics,
            registration,
        )
        if not cost_history["pass"] or not correct_path["pass"]:
            decision = "fail"
            passed = False
        else:
            passed = bool(
                cost_history["pass"]
                and correct_path["pass"]
                and terminal["pass"]
            )
            decision = "pass" if passed else "fail"
    return {
        "schema_version": ACCEPTANCE_SCHEMA,
        "analysis": ACCEPTANCE_SCHEMA,
        "implementation_id": IMPLEMENTATION_ID,
        "campaign_id": registration.get("campaign_id"),
        "candidate_parameter_id": parameter_id,
        "optimization_seeds": list(_seed_list(registration, "optimization")),
        "cost_history": cost_history,
        "correct_path": correct_path,
        "terminal_noise_test": terminal,
        "decision": decision,
        "pass": passed,
    }
