#!/usr/bin/env python3
"""Fail-closed, development-only analysis of the frozen Ala2D V15 screen.

This program never runs restrained MD and never advances an optimizer.  It
first verifies the frozen plan, the campaign execution manifest, every run
binding, every public-state ledger and every immutable public-state hash.  It
then scores each *already public* path with a newly constructed deterministic
surrogate oracle.  The archived Current IVM path history is scored by the same
oracle and is the only selection baseline.

All 48 registered trajectories must contain the exact 0, 20, ..., 2000 public
NoF-equivalent prefix before selection is possible.  ``--allow-incomplete``
only publishes an explicitly incomplete inventory: it neither carries the last
path forward nor ranks or recommends a parameter combination.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy

from ala2d_irivm_v15 import IMPLEMENTATION_ID, N_INTERNAL, load_public_trajectory
from ala2d_oracles import SurrogateOracle
from campaign_analysis import (
    evaluate_current_paths_with_surrogate,
    fes_minimax_barrier_reference,
    load_historical_current_ivm,
)
from riemannian_torus_geometry import compute_article_metrics, periodic_difference
import run_development_screen as screen_runner


Array = np.ndarray
Trajectory = Mapping[str, Array]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCREEN_ROOT = ROOT / "runs/development_screen"
ANALYSIS_SCHEMA = "ala2d.irivm.v15.development_screen.analysis.v1"
INVENTORY_SCHEMA = "ala2d.irivm.v15.development_screen.nonselecting_inventory.v1"
ANALYSIS_SOURCE_RELATIVES = (
    Path("src/analyze_development_screen.py"),
    Path("src/campaign_analysis.py"),
)

R1_CAMPAIGN_DIRECTORY = "ala2d_mfep_irivm_20260905_r1"
R1_REGISTRATION_SHA256 = (
    "15c2608eb9268945a5bbc6a2364caa7ad25dbbc0cd98128a090ed91f0e3b9a7f"
)
R1_PLOTTING_LOADER_SHA256 = (
    "1d9c20051c341102e70f2dd1d525687787354adc69fd3ccd92e518f8f1749312"
)
R1_FENEB_SHA256 = (
    "53f840618fcc194e1e3b288f9a8936020b4bac05990387068196601e2563fc49"
)


class AnalysisError(RuntimeError):
    """A frozen input, exact checkpoint, or analysis contract is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _json_digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _core_array_sha256(value: Array) -> str:
    """Reproduce the V15 typed-array digest used by its run manifest."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise AnalysisError("object arrays are inadmissible in a V15 digest")
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and not np.little_endian
    ):
        array = array.byteswap().newbyteorder("<")
    array = np.ascontiguousarray(array)
    header = json.dumps(
        {
            "schema": "BinaryArray/ala2d-irivm-v15-development",
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(header + b"\x00" + array.tobytes()).hexdigest()


def _load_json_regular(path: Path, label: str) -> Dict[str, object]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise AnalysisError(f"{label} is not a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read valid JSON from {label}: {path}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} root must be a JSON object")
    return value


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{label} must be a JSON object")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{label} must be numerical")
    result = float(value)
    if not np.isfinite(result):
        raise AnalysisError(f"{label} must be finite")
    return result


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise AnalysisError(f"{label} must be an integer")
    return int(value)


def _atomic_replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_once_or_identical(path: Path, content: bytes) -> None:
    """Publish a content-addressed artifact once, or require identical bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
            raise AnalysisError(f"content-addressed output collision: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise


def exact_indices(trajectory: Trajectory, checkpoints: Array, label: str) -> Array:
    """Return indices of exact checkpoints; last-observation carry-forward is forbidden."""

    raw_nof = np.asarray(trajectory.get("nof_mean_force_incremental"))
    requested = np.asarray(checkpoints)
    if (
        raw_nof.ndim != 1
        or raw_nof.size == 0
        or not np.issubdtype(raw_nof.dtype, np.integer)
        or np.any(np.diff(raw_nof.astype(np.int64, copy=False)) <= 0)
    ):
        raise AnalysisError(f"{label} has an invalid exact NoF sequence")
    if requested.ndim != 1 or not np.issubdtype(requested.dtype, np.integer):
        raise AnalysisError("requested checkpoints must be a one-dimensional integer array")
    nof = raw_nof.astype(np.int64, copy=False)
    requested = requested.astype(np.int64, copy=False)
    positions = np.searchsorted(nof, requested)
    valid = positions < nof.size
    observed = np.full(requested.shape, np.iinfo(np.int64).min, dtype=np.int64)
    observed[valid] = nof[positions[valid]]
    matching = valid & (observed == requested)
    if not np.all(matching):
        raise AnalysisError(
            f"{label} is missing exact checkpoints {requested[~matching].tolist()}; "
            "LOCF is forbidden"
        )
    return positions


def normalized_auc(values: Array, checkpoints: Array) -> float:
    values = np.asarray(values, dtype=np.float64)
    checkpoints = np.asarray(checkpoints, dtype=np.float64)
    if (
        values.ndim != 1
        or checkpoints.ndim != 1
        or values.size != checkpoints.size
        or values.size < 2
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(checkpoints))
        or np.any(np.diff(checkpoints) <= 0.0)
    ):
        raise AnalysisError("normalized AUC requires finite aligned increasing samples")
    return float(np.trapz(values, checkpoints) / (checkpoints[-1] - checkpoints[0]))


def _metric_values(
    trajectory: Trajectory,
    public_name: str,
    indices: Array,
    barrier_reference: float,
    label: str,
) -> Array:
    source_name = "Delta_F_barrier" if public_name == "Delta_F_error" else public_name
    if source_name not in trajectory:
        raise AnalysisError(f"{label} lacks {source_name}")
    nof = np.asarray(trajectory.get("nof_mean_force_incremental"))
    values = np.asarray(trajectory[source_name], dtype=np.float64)
    if values.shape != nof.shape or np.any(~np.isfinite(values)):
        raise AnalysisError(f"{label}.{source_name} is not one finite value per checkpoint")
    selected = values[np.asarray(indices, dtype=np.int64)]
    if public_name == "Delta_F_error":
        selected = np.abs(selected - float(barrier_reference))
    if np.any(selected < 0.0):
        raise AnalysisError(f"{label}.{public_name} contains a negative value")
    return selected


def _strict_ratio(numerator: float, denominator: float, label: str) -> float:
    if not np.isfinite(numerator) or numerator < 0.0:
        raise AnalysisError(f"invalid numerator in {label}")
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise AnalysisError(f"nonpositive baseline denominator in {label}")
    return float(numerator / denominator)


def correct_path_guard(
    path: Array,
    barrier_error: float,
    reparam_residual: float,
    minimax_path: Array,
    saddle_point: Array,
    guard: Mapping[str, object],
) -> Dict[str, object]:
    """Evaluate exactly the five frozen terminal correct-channel thresholds."""

    failures: List[str] = []
    path = np.asarray(path, dtype=np.float64)
    reference = np.asarray(minimax_path, dtype=np.float64)
    saddle = np.asarray(saddle_point, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 2 or np.any(~np.isfinite(path)):
        raise AnalysisError("terminal path must have finite shape (n>=2, 2)")
    if reference.ndim != 2 or reference.shape[0] < 2 or reference.shape[1] != 2 or np.any(~np.isfinite(reference)):
        raise AnalysisError("FES minimax path must have finite shape (n>=2, 2)")
    if saddle.shape != (2,) or np.any(~np.isfinite(saddle)):
        raise AnalysisError("FES minimax saddle must be one finite 2D point")
    barrier_error = _finite_number(barrier_error, "terminal barrier error")
    reparam_residual = _finite_number(reparam_residual, "terminal reparameterization residual")
    if barrier_error < 0.0:
        raise AnalysisError("terminal barrier error cannot be negative")
    if reparam_residual < 0.0:
        failures.append("negative_article_reparameterization_output_residual")

    segment_limit = _finite_number(guard.get("maximum_segment_rad"), "maximum_segment_rad")
    nearest_limit = _finite_number(
        guard.get("maximum_nearest_fes_minimax_path_distance_rad"),
        "maximum_nearest_fes_minimax_path_distance_rad",
    )
    rms_limit = _finite_number(
        guard.get("rms_nearest_fes_minimax_path_distance_rad"),
        "rms_nearest_fes_minimax_path_distance_rad",
    )
    saddle_limit = _finite_number(
        guard.get("minimax_saddle_proximity_rad"), "minimax_saddle_proximity_rad"
    )
    barrier_limit = _finite_number(
        guard.get("maximum_barrier_error_kcal_per_mol"),
        "maximum_barrier_error_kcal_per_mol",
    )
    residual_limit = _finite_number(
        guard.get("article_reparameterization_output_residual_strictly_less_than"),
        "article_reparameterization_output_residual_strictly_less_than",
    )
    if min(segment_limit, nearest_limit, rms_limit, saddle_limit, barrier_limit, residual_limit) <= 0.0:
        raise AnalysisError("correct-path thresholds must be positive")

    segment_lengths = np.linalg.norm(
        periodic_difference(path[1:], path[:-1]), axis=1
    )
    maximum_segment = float(np.max(segment_lengths))
    path_to_reference = periodic_difference(path[:, None, :], reference[None, :, :])
    nearest = np.min(np.linalg.norm(path_to_reference, axis=2), axis=1)
    maximum_nearest = float(np.max(nearest))
    rms_nearest = float(np.sqrt(np.mean(nearest * nearest)))
    saddle_proximity = float(
        np.min(np.linalg.norm(periodic_difference(path, saddle[None, :]), axis=1))
    )

    if maximum_segment > segment_limit:
        failures.append("maximum_segment_rad")
    if maximum_nearest > nearest_limit:
        failures.append("maximum_nearest_fes_minimax_path_distance_rad")
    if rms_nearest > rms_limit:
        failures.append("rms_nearest_fes_minimax_path_distance_rad")
    if saddle_proximity > saddle_limit:
        failures.append("minimax_saddle_proximity_rad")
    if barrier_error > barrier_limit:
        failures.append("maximum_barrier_error_kcal_per_mol")
    if not reparam_residual < residual_limit:
        failures.append("article_reparameterization_output_residual_strictly_less_than")

    return {
        "maximum_shortest_periodic_segment_rad": maximum_segment,
        "maximum_segment_rad_limit": segment_limit,
        "maximum_nearest_fes_minimax_path_distance_rad": maximum_nearest,
        "maximum_nearest_fes_minimax_path_distance_rad_limit": nearest_limit,
        "rms_nearest_fes_minimax_path_distance_rad": rms_nearest,
        "rms_nearest_fes_minimax_path_distance_rad_limit": rms_limit,
        "minimax_saddle_proximity_rad": saddle_proximity,
        "minimax_saddle_proximity_rad_limit": saddle_limit,
        "Delta_F_error_kcal_per_mol": barrier_error,
        "maximum_barrier_error_kcal_per_mol": barrier_limit,
        "article_reparameterization_output_residual": reparam_residual,
        "article_reparameterization_output_residual_strict_limit": residual_limit,
        "failures": sorted(set(failures)),
        "pass": not failures,
    }


def _checkpoint_grid(section: Mapping[str, object]) -> Array:
    start = _integer(section.get("start"), "checkpoint start")
    stop = _integer(section.get("stop"), "checkpoint stop")
    step = _integer(section.get("step"), "checkpoint step")
    if start < 0 or stop < start or step <= 0 or (stop - start) % step:
        raise AnalysisError("invalid frozen checkpoint range")
    return np.arange(start, stop + step, step, dtype=np.int64)


def _extract_components(
    trajectory: Trajectory,
    checkpoints: Array,
    barrier_reference: float,
    label: str,
) -> Dict[str, Array]:
    indices = exact_indices(trajectory, checkpoints, label)
    return {
        name: _metric_values(trajectory, name, indices, barrier_reference, label)
        for name in ("I_perp_G", "I_arc", "Delta_F_error")
    }


def evaluate_parameter_rules(
    synthetic_by_seed: Mapping[int, Trajectory],
    deterministic: Trajectory,
    baseline: Trajectory,
    barrier_reference: float,
    plan: Mapping[str, object],
    *,
    minimax_path: Array,
    saddle_point: Array,
) -> Dict[str, object]:
    """Apply the frozen pointwise, tail, deterministic, and path rules."""

    posthoc = _require_mapping(plan.get("posthoc_scoring"), "posthoc_scoring")
    front_rule = _require_mapping(posthoc.get("front_middle"), "front_middle")
    tail_rule = _require_mapping(posthoc.get("tail"), "tail")
    guard = _require_mapping(posthoc.get("correct_path_guard"), "correct_path_guard")
    if guard.get("required_for_deterministic_and_every_synthetic_repetition") is not True:
        raise AnalysisError("correct-path guard must apply to deterministic and every seed")
    terminal_nof = _integer(guard.get("terminal_checkpoint"), "terminal_checkpoint")
    if terminal_nof != 2000:
        raise AnalysisError("frozen correct-path terminal checkpoint must be 2000")
    if not np.isclose(
        float(barrier_reference),
        _finite_number(posthoc.get("barrier_reference_kcal_per_mol"), "barrier reference"),
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise AnalysisError("supplied barrier reference differs from frozen plan")

    oracle_plan = _require_mapping(plan.get("oracles"), "oracles")
    synthetic_plan = _require_mapping(
        oracle_plan.get("calibrated_synthetic_noise"), "calibrated_synthetic_noise"
    )
    expected_seeds_raw = synthetic_plan.get("master_seeds")
    if not isinstance(expected_seeds_raw, list) or any(type(item) is not int for item in expected_seeds_raw):
        raise AnalysisError("synthetic master seed list is invalid")
    expected_seeds = tuple(int(item) for item in expected_seeds_raw)
    if set(synthetic_by_seed) != set(expected_seeds) or len(synthetic_by_seed) != 3:
        raise AnalysisError(
            f"synthetic seed set differs: expected {list(expected_seeds)}, "
            f"observed {sorted(synthetic_by_seed)}"
        )

    front = _checkpoint_grid(front_rule)
    tail = _checkpoint_grid(tail_rule)
    if front[0] != 20 or front[-1] != 1600 or front.size != 80:
        raise AnalysisError("front/middle grid must be exact 20..1600 by 20")
    if tail[0] != 1620 or tail[-1] != 2000 or tail.size != 20:
        raise AnalysisError("tail grid must be exact 1620..2000 by 20")
    if _integer(tail_rule.get("checkpoint_count"), "tail checkpoint_count") != tail.size:
        raise AnalysisError("tail checkpoint_count differs from its exact grid")

    baseline_front = _extract_components(baseline, front, barrier_reference, "Current IVM")
    baseline_tail = _extract_components(baseline, tail, barrier_reference, "Current IVM")
    det_front = _extract_components(deterministic, front, barrier_reference, "deterministic IR-IVM")
    det_tail = _extract_components(deterministic, tail, barrier_reference, "deterministic IR-IVM")
    synthetic_front = {
        seed: _extract_components(synthetic_by_seed[seed], front, barrier_reference, f"synthetic seed {seed}")
        for seed in expected_seeds
    }
    synthetic_tail = {
        seed: _extract_components(synthetic_by_seed[seed], tail, barrier_reference, f"synthetic seed {seed}")
        for seed in expected_seeds
    }

    strict_names = ("I_perp_G", "Delta_F_error")
    front_metrics: Dict[str, object] = {}
    front_pass_arrays: List[Array] = []
    for name in ("I_perp_G", "I_arc", "Delta_F_error"):
        rows = np.stack([synthetic_front[seed][name] for seed in expected_seeds])
        median = np.median(rows, axis=0)
        checkpoint_pass = (
            median < baseline_front[name]
            if name in strict_names
            else median <= baseline_front[name]
        )
        front_pass_arrays.append(checkpoint_pass)
        front_metrics[name] = {
            "comparison": "<" if name in strict_names else "<=",
            "synthetic_seed_values": {
                str(seed): synthetic_front[seed][name].tolist() for seed in expected_seeds
            },
            "synthetic_median": median.tolist(),
            "current_ivm": baseline_front[name].tolist(),
            "checkpoint_pass": checkpoint_pass.tolist(),
            "failure_nof": front[~checkpoint_pass].astype(int).tolist(),
            "pass": bool(np.all(checkpoint_pass)),
        }
    joint_front = np.all(np.stack(front_pass_arrays), axis=0)
    front_pass = bool(np.all(joint_front))

    noisy_rules = _require_mapping(
        tail_rule.get("for_I_perp_G_and_Delta_F_error"),
        "tail.for_I_perp_G_and_Delta_F_error",
    )
    minimum_fraction = _finite_number(
        noisy_rules.get("minimum_median_checkpoint_win_fraction"),
        "minimum_median_checkpoint_win_fraction",
    )
    if not 0.0 <= minimum_fraction <= 1.0:
        raise AnalysisError("tail win fraction is outside [0,1]")
    if noisy_rules.get("candidate_tail_median_strictly_lower") is not True:
        raise AnalysisError("tail median rule must be strict and enabled")
    if noisy_rules.get("candidate_median_normalized_auc_strictly_lower") is not True:
        raise AnalysisError("tail median-AUC rule must be strict and enabled")
    minimum_individual = _integer(
        noisy_rules.get("minimum_individual_synthetic_seeds_with_lower_auc"),
        "minimum individual noisy-metric AUC wins",
    )
    if minimum_individual != 2:
        raise AnalysisError("noisy tail must require at least two of three seed AUC wins")

    tail_metrics: Dict[str, object] = {}
    tail_win_fractions: Dict[str, float] = {}
    for name in strict_names:
        rows = np.stack([synthetic_tail[seed][name] for seed in expected_seeds])
        median = np.median(rows, axis=0)
        wins = median < baseline_tail[name]
        win_fraction = float(np.mean(wins))
        candidate_tail_median = float(np.median(median))
        baseline_tail_median = float(np.median(baseline_tail[name]))
        candidate_auc = normalized_auc(median, tail)
        baseline_auc = normalized_auc(baseline_tail[name], tail)
        individual_auc = {
            str(seed): normalized_auc(synthetic_tail[seed][name], tail)
            for seed in expected_seeds
        }
        individual_pass = {
            key: bool(value < baseline_auc) for key, value in individual_auc.items()
        }
        passed = bool(
            win_fraction >= minimum_fraction
            and candidate_tail_median < baseline_tail_median
            and candidate_auc < baseline_auc
            and sum(individual_pass.values()) >= minimum_individual
        )
        tail_win_fractions[name] = win_fraction
        tail_metrics[name] = {
            "gate_kind": "noise_tolerant_curve_shape",
            "comparison": "<",
            "synthetic_seed_values": {
                str(seed): synthetic_tail[seed][name].tolist() for seed in expected_seeds
            },
            "synthetic_checkpoint_median": median.tolist(),
            "current_ivm": baseline_tail[name].tolist(),
            "checkpoint_wins": wins.tolist(),
            "checkpoint_win_fraction": win_fraction,
            "minimum_checkpoint_win_fraction": minimum_fraction,
            "checkpoint_win_fraction_pass": bool(win_fraction >= minimum_fraction),
            "synthetic_tail_median": candidate_tail_median,
            "current_ivm_tail_median": baseline_tail_median,
            "tail_median_pass": bool(candidate_tail_median < baseline_tail_median),
            "synthetic_median_normalized_auc": candidate_auc,
            "current_ivm_normalized_auc": baseline_auc,
            "median_normalized_auc_pass": bool(candidate_auc < baseline_auc),
            "individual_synthetic_seed_normalized_auc": individual_auc,
            "individual_seed_auc_pass": individual_pass,
            "individual_seed_auc_pass_count": int(sum(individual_pass.values())),
            "minimum_individual_seed_auc_pass_count": minimum_individual,
            "pass": passed,
        }

    arc_rules = _require_mapping(tail_rule.get("for_I_arc"), "tail.for_I_arc")
    if arc_rules.get("median_no_higher_at_every_checkpoint") is not True:
        raise AnalysisError("I_arc tail pointwise rule must be enabled")
    arc_minimum_individual = _integer(
        arc_rules.get("minimum_individual_synthetic_seeds_with_lower_auc"),
        "minimum individual I_arc AUC wins",
    )
    if arc_minimum_individual != 2:
        raise AnalysisError("I_arc tail must require at least two of three seed AUC wins")
    arc_rows = np.stack([synthetic_tail[seed]["I_arc"] for seed in expected_seeds])
    arc_median = np.median(arc_rows, axis=0)
    arc_checkpoint_pass = arc_median <= baseline_tail["I_arc"]
    arc_baseline_auc = normalized_auc(baseline_tail["I_arc"], tail)
    arc_individual_auc = {
        str(seed): normalized_auc(synthetic_tail[seed]["I_arc"], tail)
        for seed in expected_seeds
    }
    arc_individual_pass = {
        key: bool(value < arc_baseline_auc) for key, value in arc_individual_auc.items()
    }
    arc_pass = bool(
        np.all(arc_checkpoint_pass)
        and sum(arc_individual_pass.values()) >= arc_minimum_individual
    )
    tail_win_fractions["I_arc"] = float(np.mean(arc_checkpoint_pass))
    tail_metrics["I_arc"] = {
        "gate_kind": "path_geometry",
        "comparison": "<=",
        "synthetic_seed_values": {
            str(seed): synthetic_tail[seed]["I_arc"].tolist() for seed in expected_seeds
        },
        "synthetic_checkpoint_median": arc_median.tolist(),
        "current_ivm": baseline_tail["I_arc"].tolist(),
        "checkpoint_pass": arc_checkpoint_pass.tolist(),
        "failure_nof": tail[~arc_checkpoint_pass].astype(int).tolist(),
        "checkpoint_win_fraction": tail_win_fractions["I_arc"],
        "individual_synthetic_seed_normalized_auc": arc_individual_auc,
        "current_ivm_normalized_auc": arc_baseline_auc,
        "individual_seed_auc_pass": arc_individual_pass,
        "individual_seed_auc_pass_count": int(sum(arc_individual_pass.values())),
        "minimum_individual_seed_auc_pass_count": arc_minimum_individual,
        "pass": arc_pass,
    }
    tail_pass = bool(all(bool(tail_metrics[name]["pass"]) for name in tail_metrics))

    deterministic_front_metrics: Dict[str, object] = {}
    deterministic_front_pass = True
    for name in ("I_perp_G", "I_arc", "Delta_F_error"):
        checkpoint_pass = (
            det_front[name] < baseline_front[name]
            if name in strict_names
            else det_front[name] <= baseline_front[name]
        )
        metric_pass = bool(np.all(checkpoint_pass))
        deterministic_front_pass = deterministic_front_pass and metric_pass
        deterministic_front_metrics[name] = {
            "comparison": "<" if name in strict_names else "<=",
            "candidate": det_front[name].tolist(),
            "current_ivm": baseline_front[name].tolist(),
            "checkpoint_pass": checkpoint_pass.tolist(),
            "failure_nof": front[~checkpoint_pass].astype(int).tolist(),
            "pass": metric_pass,
        }
    deterministic_tail_metrics: Dict[str, object] = {}
    deterministic_tail_pass = True
    for name in ("I_perp_G", "I_arc", "Delta_F_error"):
        candidate_auc = normalized_auc(det_tail[name], tail)
        baseline_auc = normalized_auc(baseline_tail[name], tail)
        metric_pass = bool(candidate_auc < baseline_auc)
        deterministic_tail_pass = deterministic_tail_pass and metric_pass
        deterministic_tail_metrics[name] = {
            "candidate_normalized_auc": candidate_auc,
            "current_ivm_normalized_auc": baseline_auc,
            "strictly_lower": metric_pass,
        }

    all_checkpoints = np.arange(20, terminal_nof + 20, 20, dtype=np.int64)
    baseline_all = _extract_components(baseline, all_checkpoints, barrier_reference, "Current IVM")
    det_all = _extract_components(
        deterministic, all_checkpoints, barrier_reference, "deterministic IR-IVM"
    )
    deterministic_ratios = []
    for name in ("I_perp_G", "I_arc", "Delta_F_error"):
        for candidate_value, baseline_value in zip(det_all[name], baseline_all[name]):
            deterministic_ratios.append(
                _strict_ratio(float(candidate_value), float(baseline_value), f"deterministic {name} ratio")
            )
    deterministic_worst_ratio = float(max(deterministic_ratios))
    deterministic_pass = bool(deterministic_front_pass and deterministic_tail_pass)

    terminal_indices = {
        "deterministic": int(exact_indices(deterministic, np.asarray([terminal_nof]), "deterministic IR-IVM")[0])
    }
    for seed in expected_seeds:
        terminal_indices[str(seed)] = int(
            exact_indices(
                synthetic_by_seed[seed], np.asarray([terminal_nof]), f"synthetic seed {seed}"
            )[0]
        )

    def path_result(trajectory: Trajectory, index: int, label: str) -> Dict[str, object]:
        paths = np.asarray(trajectory.get("full_path_unwrapped"), dtype=np.float64)
        nof = np.asarray(trajectory.get("nof_mean_force_incremental"))
        residuals = np.asarray(
            trajectory.get("selected_reparameterization_output_residual"),
            dtype=np.float64,
        )
        if paths.ndim != 3 or paths.shape[0] != nof.size:
            raise AnalysisError(f"{label} lacks one public path per exact checkpoint")
        if residuals.shape != nof.shape:
            raise AnalysisError(
                f"{label} lacks selected_reparameterization_output_residual in "
                "load_public_trajectory output"
            )
        barrier = _metric_values(
            trajectory,
            "Delta_F_error",
            np.asarray([index]),
            barrier_reference,
            label,
        )[0]
        return correct_path_guard(
            paths[index],
            float(barrier),
            float(residuals[index]),
            minimax_path,
            saddle_point,
            guard,
        )

    deterministic_path = path_result(
        deterministic, terminal_indices["deterministic"], "deterministic IR-IVM"
    )
    synthetic_paths = {
        str(seed): path_result(
            synthetic_by_seed[seed], terminal_indices[str(seed)], f"synthetic seed {seed}"
        )
        for seed in expected_seeds
    }
    correct_path_pass = bool(
        deterministic_path["pass"]
        and all(bool(value["pass"]) for value in synthetic_paths.values())
    )

    noisy_auc_ratios: Dict[str, List[float]] = {"I_perp_G": [], "Delta_F_error": []}
    for name in noisy_auc_ratios:
        baseline_auc = normalized_auc(baseline_tail[name], tail)
        noisy_auc_ratios[name] = [
            _strict_ratio(
                normalized_auc(synthetic_tail[seed][name], tail),
                baseline_auc,
                f"seed {seed} tail {name} AUC ratio",
            )
            for seed in expected_seeds
        ]

    selection_metrics = {
        "joint_front_middle_checkpoint_win_count": int(np.sum(joint_front)),
        "front_middle_checkpoint_count": int(front.size),
        "minimum_component_tail_checkpoint_win_fraction": float(
            min(tail_win_fractions.values())
        ),
        "worst_individual_synthetic_seed_tail_I_perp_G_auc_ratio": float(
            max(noisy_auc_ratios["I_perp_G"])
        ),
        "worst_individual_synthetic_seed_tail_Delta_F_error_auc_ratio": float(
            max(noisy_auc_ratios["Delta_F_error"])
        ),
        "deterministic_worst_all_metric_checkpoint_ratio": deterministic_worst_ratio,
    }
    primary_pass = bool(front_pass and tail_pass and deterministic_pass and correct_path_pass)
    return {
        "front_middle": {
            "checkpoint_nof": front.astype(int).tolist(),
            "metrics": front_metrics,
            "joint_checkpoint_pass": joint_front.tolist(),
            "joint_checkpoint_win_count": int(np.sum(joint_front)),
            "pass": front_pass,
        },
        "tail": {
            "checkpoint_nof": tail.astype(int).tolist(),
            "metrics": tail_metrics,
            "minimum_component_checkpoint_win_fraction": selection_metrics[
                "minimum_component_tail_checkpoint_win_fraction"
            ],
            "pass": tail_pass,
        },
        "deterministic_guard": {
            "front_middle": {
                "checkpoint_nof": front.astype(int).tolist(),
                "metrics": deterministic_front_metrics,
                "pass": bool(deterministic_front_pass),
            },
            "tail": {
                "checkpoint_nof": tail.astype(int).tolist(),
                "metrics": deterministic_tail_metrics,
                "pass": bool(deterministic_tail_pass),
            },
            "worst_all_metric_checkpoint_ratio": deterministic_worst_ratio,
            "pass": deterministic_pass,
        },
        "correct_path": {
            "terminal_checkpoint": terminal_nof,
            "deterministic": deterministic_path,
            "synthetic": synthetic_paths,
            "required_for_all_four_trajectories": True,
            "pass": correct_path_pass,
        },
        "selection_metrics": selection_metrics,
        "primary_pass": primary_pass,
        "pass": primary_pass,
    }


def passing_sort_key(
    result: Mapping[str, object], controls: Mapping[str, object]
) -> Tuple[object, ...]:
    """The frozen passing_order, encoded as an ascending Python sort key."""

    metrics = _require_mapping(result.get("selection_metrics"), "selection_metrics")
    return (
        -_integer(
            metrics.get("joint_front_middle_checkpoint_win_count"),
            "joint front/middle checkpoint wins",
        ),
        -_finite_number(
            metrics.get("minimum_component_tail_checkpoint_win_fraction"),
            "minimum component tail win fraction",
        ),
        _finite_number(
            metrics.get("worst_individual_synthetic_seed_tail_I_perp_G_auc_ratio"),
            "worst individual I_perp tail AUC ratio",
        ),
        _finite_number(
            metrics.get("worst_individual_synthetic_seed_tail_Delta_F_error_auc_ratio"),
            "worst individual Delta_F tail AUC ratio",
        ),
        _finite_number(
            metrics.get("deterministic_worst_all_metric_checkpoint_ratio"),
            "deterministic worst all-metric ratio",
        ),
        _finite_number(controls.get("alpha"), "alpha"),
        _finite_number(controls.get("r"), "r"),
        _integer(controls.get("m_max"), "m_max"),
    )


def diagnostic_sort_key(
    result: Mapping[str, object], controls: Mapping[str, object]
) -> Tuple[object, ...]:
    """Pick one curve for diagnosis without turning a failure into a winner."""

    gate_count = sum(
        bool(_require_mapping(result.get(name), name).get("pass"))
        for name in ("front_middle", "tail", "deterministic_guard", "correct_path")
    )
    return (-int(bool(result.get("primary_pass"))), -gate_count) + passing_sort_key(
        result, controls
    )


def _verify_hash_mapping(
    root: Path,
    observed: object,
    expected_relatives: Iterable[Path],
    label: str,
) -> Dict[str, str]:
    table = _require_mapping(observed, label)
    expected_keys = {Path(value).as_posix() for value in expected_relatives}
    if set(table) != expected_keys:
        raise AnalysisError(
            f"{label} key set differs: expected={sorted(expected_keys)}, "
            f"observed={sorted(table)}"
        )
    verified: Dict[str, str] = {}
    for relative_text in sorted(expected_keys):
        expected_digest = table[relative_text]
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise AnalysisError(f"{label}[{relative_text!r}] is not a SHA-256 digest")
        path = root / Path(relative_text)
        if not path.is_file() or path.is_symlink():
            raise AnalysisError(f"hashed {label} file is missing or unsafe: {path}")
        observed_digest = sha256_file(path)
        if observed_digest != expected_digest:
            raise AnalysisError(
                f"{label} hash mismatch for {relative_text}: expected "
                f"{expected_digest}, observed {observed_digest}"
            )
        verified[relative_text] = observed_digest
    return verified


def _analysis_source_hashes(root: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for relative in ANALYSIS_SOURCE_RELATIVES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise AnalysisError(f"analysis source is missing or unsafe: {path}")
        result[relative.as_posix()] = sha256_file(path)
    return result


def _expected_run_order(plan: screen_runner.FrozenPlan) -> List[Dict[str, object]]:
    return [
        {
            "ordinal": spec.ordinal,
            "oracle_kind": spec.oracle_kind,
            "master_seed": spec.master_seed,
            "parameter_id": spec.parameter.parameter_id,
            "alpha": spec.parameter.controls.alpha,
            "alpha_integer_mantissa_notation": spec.parameter.alpha_label,
            "r": spec.parameter.controls.r,
            "m_max": spec.parameter.controls.m_max,
            "relative_directory": spec.relative_directory.as_posix(),
        }
        for spec in plan.run_specifications()
    ]


def validate_execution_manifest(
    *,
    root: Path,
    workspace: Path,
    screen_root: Path,
    plan: screen_runner.FrozenPlan,
) -> Tuple[Path, Dict[str, object], Dict[str, object]]:
    """Validate the execution manifest and re-hash every bound immutable input."""

    try:
        screen_runner._validate_screen_root_layout(screen_root)
    except screen_runner.ScreenError as error:
        raise AnalysisError(str(error)) from error
    manifest_path = screen_root / screen_runner.EXECUTION_MANIFEST_NAME
    manifest = _load_json_regular(manifest_path, "development execution manifest")
    if manifest_path.read_bytes() != _canonical_json_bytes(manifest):
        raise AnalysisError("development execution manifest is not canonical write-once JSON")
    exact_fields = {
        "schema_version": screen_runner.SCREEN_SCHEMA,
        "plan_id": plan.raw["plan_id"],
        "plan_sha256": plan.sha256,
        "formal_evidence": False,
        "new_restrained_md_authorized": False,
        "implementation_id": IMPLEMENTATION_ID,
        "run_count": 48,
        "per_run_incremental_query_equivalent_budget": plan.nof_budget,
        "complete_band_query_equivalents": N_INTERNAL,
        "common_initial_incremental_cost": 0,
        "root": str(root.resolve()),
        "screen_root": str(screen_root.resolve()),
    }
    for key, expected in exact_fields.items():
        if manifest.get(key) != expected or type(manifest.get(key)) is not type(expected):
            raise AnalysisError(
                f"execution manifest {key!r} differs: expected {expected!r}, "
                f"observed {manifest.get(key)!r}"
            )
    if manifest.get("run_order") != _expected_run_order(plan):
        raise AnalysisError("execution manifest run_order differs from the frozen factorial order")

    verified: Dict[str, object] = {
        "execution_manifest_sha256": sha256_file(manifest_path),
        "source_sha256": _verify_hash_mapping(
            root,
            manifest.get("source_sha256"),
            screen_runner.SOURCE_RELATIVES,
            "source_sha256",
        ),
        "specification_sha256": _verify_hash_mapping(
            root,
            manifest.get("specification_sha256"),
            screen_runner.SPECIFICATION_RELATIVES,
            "specification_sha256",
        ),
        "input_sha256": _verify_hash_mapping(
            root,
            manifest.get("input_sha256"),
            (
                screen_runner.INITIAL_RELATIVE,
                screen_runner.INITIAL_PROVENANCE_RELATIVE,
                screen_runner.FES_RELATIVE,
                screen_runner.METRIC_BANK_RELATIVE,
                screen_runner.CALIBRATION_RELATIVE,
                screen_runner.CALIBRATION_PROVENANCE_RELATIVE,
            ),
            "input_sha256",
        ),
    }
    formal = _require_mapping(
        manifest.get("formal_pdf_unchanged_sha256"),
        "formal_pdf_unchanged_sha256",
    )
    expected_formal = {
        path.as_posix(): digest
        for path, digest in screen_runner.FORMAL_PDF_SHA256.items()
    }
    if dict(formal) != expected_formal:
        raise AnalysisError("execution manifest formal-PDF hash table differs")
    for relative_text, expected_digest in expected_formal.items():
        path = workspace / relative_text
        if not path.is_file() or path.is_symlink():
            raise AnalysisError(f"formal PDF is missing or unsafe: {path}")
        observed_digest = sha256_file(path)
        if observed_digest != expected_digest:
            raise AnalysisError(
                f"formal PDF changed during development: {path}; observed {observed_digest}"
            )
    verified["formal_pdf_unchanged_sha256"] = expected_formal
    runtime = _require_mapping(manifest.get("runtime"), "execution manifest runtime")
    for key in (
        "python_executable",
        "python_version",
        "python_implementation",
        "python_full_version",
        "byteorder",
        "numpy_version",
        "scipy_version",
        "platform_system",
        "platform_release",
        "platform_machine",
    ):
        if not isinstance(runtime.get(key), str) or not runtime.get(key):
            raise AnalysisError(f"execution manifest runtime lacks {key}")
    return manifest_path, manifest, verified


def _run_core_manifest_checks(
    *,
    core_manifest: Mapping[str, object],
    spec: screen_runner.RunSpecification,
    plan: screen_runner.FrozenPlan,
    root: Path,
    initial_path: Array,
    initial_M: Array,
    initial_gradient: Array,
) -> None:
    expected_fields = {
        "implementation_id": IMPLEMENTATION_ID,
        "algorithm_variant": "v15_development_one_sided_r_resolved_lbfgs",
        "development_only": True,
        "nof_budget_mean_force_incremental": plan.nof_budget,
        "complete_20_image_band_cost_nof": N_INTERNAL,
        "common_initial_band_excluded_from_incremental_axis": True,
        "absolute_nof_additive_constant": N_INTERNAL,
        "oracle_formal_evidence": False,
        "internal_image_count": N_INTERNAL,
        "relative_residual_stop": "U(P_j) <= r*L(P_anchor)",
        "relative_residual_continue": (
            "whenever the one-sided stop U(P_j) <= r*L(P_anchor) is not proved"
        ),
        "relative_residual_ambiguous_action": (
            "continue_movement_same_frozen_outer"
        ),
        "r_stop_is_one_sided_confident_stop": True,
        "ambiguous_candidate_policy": (
            "choose valid H0 when the positive L-BFGS score advantage is not "
            "resolved above secant_noise_z shared-source standard errors; never "
            "HOLD solely to arbitrate H0 versus L-BFGS"
        ),
    }
    for key, expected in expected_fields.items():
        if core_manifest.get(key) != expected or type(core_manifest.get(key)) is not type(expected):
            raise AnalysisError(
                f"{spec.relative_directory} core manifest field {key!r} differs"
            )
    expected_core_source_hash = sha256_file(root / "src/ala2d_irivm_v15.py")
    if core_manifest.get("implementation_source_sha256") != expected_core_source_hash:
        raise AnalysisError(
            f"{spec.relative_directory} core manifest is not bound to the current "
            "ala2d_irivm_v15.py bytes"
        )
    expected_controls = asdict(spec.parameter.controls)
    if core_manifest.get("controls") != expected_controls:
        raise AnalysisError(f"{spec.relative_directory} core controls differ from run binding")
    initial_hashes = {
        "causal_initial_path_sha256": _core_array_sha256(initial_path),
        "causal_initial_M_sha256": _core_array_sha256(initial_M),
        "causal_initial_grad_F_sha256": _core_array_sha256(initial_gradient),
        "causal_initial_grad_F_block_means_sha256": _core_array_sha256(
            initial_gradient[:, None, :]
        ),
        "causal_initial_M_block_means_sha256": _core_array_sha256(
            initial_M[:, None, :, :]
        ),
        "causal_initial_block_frame_counts_sha256": _core_array_sha256(
            np.zeros((N_INTERNAL, 1), dtype=np.int64)
        ),
    }
    for key, expected in initial_hashes.items():
        if core_manifest.get(key) != expected:
            raise AnalysisError(f"{spec.relative_directory} core initial hash {key} differs")

    oracle = screen_runner._instantiate_oracle(root, plan, spec)
    if int(getattr(oracle, "call_count", -1)) != 0:
        raise AnalysisError("fresh development oracle unexpectedly has nonzero query count")
    expected_provenance = oracle.provenance()
    if core_manifest.get("oracle_provenance") != expected_provenance:
        raise AnalysisError(
            f"{spec.relative_directory} oracle provenance differs from frozen inputs"
        )
    if core_manifest.get("oracle_id") != getattr(oracle, "oracle_id"):
        raise AnalysisError(f"{spec.relative_directory} oracle identity differs")


def validate_one_run(
    *,
    root: Path,
    screen_root: Path,
    plan: screen_runner.FrozenPlan,
    spec: screen_runner.RunSpecification,
    execution_manifest_path: Path,
    initial_path: Array,
    initial_M: Array,
    initial_gradient: Array,
) -> Tuple[Dict[str, object], Optional[Dict[str, Array]]]:
    """Validate one exact bound run and return inventory plus replayed trajectory."""

    run_directory = screen_root / spec.relative_directory
    base: Dict[str, object] = {
        "ordinal": spec.ordinal,
        "oracle_kind": spec.oracle_kind,
        "master_seed": spec.master_seed,
        "parameter_id": spec.parameter.parameter_id,
        "relative_directory": spec.relative_directory.as_posix(),
        "expected_terminal_incremental_query_equivalents": plan.nof_budget,
        "complete": False,
    }
    if not run_directory.exists():
        return {**base, "status": "missing", "last_incremental_query_equivalents": None}, None
    if not run_directory.is_dir() or run_directory.is_symlink():
        raise AnalysisError(f"run directory is unsafe: {run_directory}")
    binding_path = run_directory / screen_runner.RUN_BINDING_NAME
    if not binding_path.exists():
        if any(run_directory.iterdir()):
            raise AnalysisError(f"run artifacts exist without binding: {run_directory}")
        return {**base, "status": "missing_run_binding", "last_incremental_query_equivalents": None}, None

    expected_binding = screen_runner._run_binding_payload(
        spec=spec, plan=plan, manifest_path=execution_manifest_path
    )
    screen_runner._validate_existing_run_root(run_directory, expected_binding)
    binding = _load_json_regular(binding_path, "screen run binding")
    if binding_path.read_bytes() != _canonical_json_bytes(binding):
        raise AnalysisError(f"noncanonical screen run binding: {binding_path}")
    if binding != expected_binding:
        raise AnalysisError(f"screen run binding differs: {binding_path}")

    core_manifest_path = run_directory / "run_manifest.json"
    state_directory = run_directory / "states"
    if not core_manifest_path.exists():
        if state_directory.exists() or (run_directory / screen_runner.PUBLIC_STATE_LEDGER_NAME).exists():
            raise AnalysisError(f"public evidence exists without a core manifest: {run_directory}")
        return {
            **base,
            "status": "bound_not_started",
            "last_incremental_query_equivalents": None,
            "run_binding_sha256": sha256_file(binding_path),
        }, None
    core_manifest = _load_json_regular(core_manifest_path, "V15 core run manifest")
    _run_core_manifest_checks(
        core_manifest=core_manifest,
        spec=spec,
        plan=plan,
        root=root,
        initial_path=initial_path,
        initial_M=initial_M,
        initial_gradient=initial_gradient,
    )

    state_paths = sorted(state_directory.glob("state_*.npz")) if state_directory.is_dir() else []
    if not state_paths:
        return {
            **base,
            "status": "manifest_written_no_public_state",
            "last_incremental_query_equivalents": None,
            "run_binding_sha256": sha256_file(binding_path),
            "core_run_manifest_sha256": sha256_file(core_manifest_path),
        }, None
    ledger_path = run_directory / screen_runner.PUBLIC_STATE_LEDGER_NAME
    if not ledger_path.exists():
        return {
            **base,
            "status": "public_states_present_but_ledger_missing",
            "last_incremental_query_equivalents": None,
            "public_state_file_count": len(state_paths),
            "run_binding_sha256": sha256_file(binding_path),
            "core_run_manifest_sha256": sha256_file(core_manifest_path),
        }, None
    ledger = _load_json_regular(ledger_path, "public-state ledger")
    if ledger_path.read_bytes() != _canonical_json_bytes(ledger):
        raise AnalysisError(f"noncanonical public-state ledger: {ledger_path}")
    ledger_exact = {
        "schema_version": screen_runner.LEDGER_SCHEMA,
        "formal_evidence": False,
        "implementation_id": IMPLEMENTATION_ID,
        "execution_manifest_sha256": sha256_file(execution_manifest_path),
        "run_binding_sha256": sha256_file(binding_path),
        "core_run_manifest_sha256": sha256_file(core_manifest_path),
        "oracle_kind": spec.oracle_kind,
        "master_seed": spec.master_seed,
        "parameter_id": spec.parameter.parameter_id,
    }
    for key, expected in ledger_exact.items():
        if ledger.get(key) != expected or type(ledger.get(key)) is not type(expected):
            raise AnalysisError(f"{spec.relative_directory} ledger field {key!r} differs")
    state_records = ledger.get("states")
    if not isinstance(state_records, list) or not state_records:
        raise AnalysisError(f"{spec.relative_directory} ledger has no states")
    if _integer(ledger.get("state_count_including_common_initial"), "ledger state count") != len(state_records):
        raise AnalysisError(f"{spec.relative_directory} ledger state count differs")
    if len(state_records) != len(state_paths):
        raise AnalysisError(f"{spec.relative_directory} ledger/file state counts differ")
    verified_state_hashes: List[Dict[str, object]] = []
    for index, (record, state_path) in enumerate(zip(state_records, state_paths)):
        record = _require_mapping(record, f"state ledger record {index}")
        expected_relative = f"states/state_{index:06d}.npz"
        expected_fields = {
            "state_index": index,
            "incremental_query_equivalents": index * N_INTERNAL,
            "relative_file": expected_relative,
        }
        for key, expected in expected_fields.items():
            if record.get(key) != expected or type(record.get(key)) is not type(expected):
                raise AnalysisError(
                    f"{spec.relative_directory} state-ledger record {index} {key} differs"
                )
        if state_path.name != f"state_{index:06d}.npz" or state_path.is_symlink() or not state_path.is_file():
            raise AnalysisError(f"unsafe or noncontiguous public state file: {state_path}")
        observed_digest = sha256_file(state_path)
        if record.get("sha256") != observed_digest:
            raise AnalysisError(
                f"{spec.relative_directory} public state {index} hash differs"
            )
        verified_state_hashes.append(
            {
                "state_index": index,
                "incremental_query_equivalents": index * N_INTERNAL,
                "sha256": observed_digest,
            }
        )

    trajectory = load_public_trajectory(run_directory)
    nof = np.asarray(trajectory["nof_mean_force_incremental"])
    if not np.array_equal(nof, np.arange(len(state_records), dtype=np.int64) * N_INTERNAL):
        raise AnalysisError(f"{spec.relative_directory} replayed NoF differs from ledger")
    if not np.array_equal(np.asarray(trajectory["full_path_unwrapped"])[0], initial_path):
        raise AnalysisError(f"{spec.relative_directory} common initial path differs")
    if not np.array_equal(np.asarray(trajectory["M"])[0], initial_M):
        raise AnalysisError(f"{spec.relative_directory} common initial metric differs")
    if not np.array_equal(np.asarray(trajectory["grad_F"])[0], initial_gradient):
        raise AnalysisError(f"{spec.relative_directory} common initial mean force differs")
    last_nof = int(nof[-1])
    if _integer(
        ledger.get("last_incremental_query_equivalents"), "ledger last incremental cost"
    ) != last_nof:
        raise AnalysisError(f"{spec.relative_directory} ledger terminal NoF differs")

    completion_path = run_directory / "completion.json"
    completion: Optional[Dict[str, object]] = None
    if completion_path.exists():
        completion = _load_json_regular(completion_path, "run completion")
        completion_expected = {
            "implementation_id": IMPLEMENTATION_ID,
            "development_only": True,
            "nof_budget": plan.nof_budget,
            "nof_mean_force_incremental": last_nof,
            "state_count_including_common_initial": len(state_records),
            "run_directory": str(run_directory.resolve()),
        }
        for key, expected in completion_expected.items():
            if completion.get(key) != expected or type(completion.get(key)) is not type(expected):
                raise AnalysisError(
                    f"{spec.relative_directory} completion field {key!r} differs"
                )

    expected_complete_nof = np.arange(
        0, plan.nof_budget + plan.checkpoint_spacing, plan.checkpoint_spacing,
        dtype=np.int64,
    )
    exact_complete_prefix = bool(np.array_equal(nof, expected_complete_nof))
    if exact_complete_prefix:
        if completion is None:
            raise AnalysisError(f"{spec.relative_directory} exact terminal run lacks completion")
        if completion.get("completed") is not True:
            raise AnalysisError(f"{spec.relative_directory} exact terminal run is not marked complete")
        if completion.get("paused_at_requested_stop") is not False:
            raise AnalysisError(f"{spec.relative_directory} terminal run is falsely marked paused")
        if completion.get("algorithmic_zero_nof_stop") is not None:
            raise AnalysisError(f"{spec.relative_directory} completed run reports zero-NoF stop")
        status = "complete_exact_0_to_2000"
    else:
        if completion is not None and completion.get("completed") is True:
            raise AnalysisError(f"{spec.relative_directory} incomplete prefix is marked complete")
        if completion is not None and completion.get("algorithmic_zero_nof_stop") is not None:
            status = "algorithmic_zero_nof_stop_before_exact_2000"
        elif completion is not None and completion.get("paused_at_requested_stop") is True:
            status = "paused_before_exact_2000"
        else:
            status = "incomplete_exact_prefix"

    record = {
        **base,
        "status": status,
        "complete": exact_complete_prefix,
        "last_incremental_query_equivalents": last_nof,
        "state_count_including_common_initial": len(state_records),
        "missing_exact_checkpoint_count": int(expected_complete_nof.size - nof.size),
        "algorithmic_zero_nof_stop": (
            completion.get("algorithmic_zero_nof_stop") if completion else None
        ),
        "run_binding_sha256": sha256_file(binding_path),
        "core_run_manifest_sha256": sha256_file(core_manifest_path),
        "public_state_ledger_sha256": sha256_file(ledger_path),
        "completion_sha256": sha256_file(completion_path) if completion_path.is_file() else None,
        "public_state_sha256": verified_state_hashes,
    }
    return record, trajectory


def collect_validated_runs(
    *,
    root: Path,
    screen_root: Path,
    plan: screen_runner.FrozenPlan,
    execution_manifest_path: Path,
    initial_path: Array,
    initial_M: Array,
    initial_gradient: Array,
) -> Tuple[List[Dict[str, object]], Dict[int, Dict[str, Array]]]:
    inventory: List[Dict[str, object]] = []
    trajectories: Dict[int, Dict[str, Array]] = {}
    for spec in plan.run_specifications():
        record, trajectory = validate_one_run(
            root=root,
            screen_root=screen_root,
            plan=plan,
            spec=spec,
            execution_manifest_path=execution_manifest_path,
            initial_path=initial_path,
            initial_M=initial_M,
            initial_gradient=initial_gradient,
        )
        inventory.append(record)
        if trajectory is not None:
            trajectories[spec.ordinal] = trajectory
    return inventory, trajectories


def score_public_paths_blind(
    trajectory: Trajectory,
    *,
    fes_npz: Path,
    metric_bank_npz: Path,
    label: str,
) -> Dict[str, Array]:
    """Score every public path with a fresh deterministic oracle instance."""

    nof = np.asarray(trajectory.get("nof_mean_force_incremental"))
    paths = np.asarray(trajectory.get("full_path_unwrapped"), dtype=np.float64)
    if (
        nof.ndim != 1
        or not np.issubdtype(nof.dtype, np.integer)
        or paths.ndim != 3
        or paths.shape[0] != nof.size
        or paths.shape[1:] != (22, 2)
        or np.any(~np.isfinite(paths))
    ):
        raise AnalysisError(f"{label} has invalid public paths for blind scoring")
    oracle = SurrogateOracle(fes_npz=fes_npz, metric_bank_npz=metric_bank_npz)
    rows: List[Mapping[str, Array]] = []
    metrics_M: List[Array] = []
    metrics_gradient: List[Array] = []
    for state_index, path in enumerate(paths):
        band = oracle.evaluate_band(path, int(state_index))
        metrics_M.append(np.asarray(band.M, dtype=np.float64))
        metrics_gradient.append(np.asarray(band.grad_F, dtype=np.float64))
        rows.append(compute_article_metrics(path, band.M, band.grad_F))
    expected_queries = int(paths.shape[0] * N_INTERNAL)
    if int(oracle.call_count) != expected_queries:
        raise AnalysisError(
            f"{label} blind scoring query ledger differs: expected {expected_queries}, "
            f"observed {oracle.call_count}"
        )
    result: Dict[str, Array] = {
        "nof_mean_force_incremental": nof.astype(np.int64, copy=True),
        "full_path_unwrapped": paths.copy(),
        "M": np.stack(metrics_M),
        "grad_F": np.stack(metrics_gradient),
        "I_perp_G": np.asarray([float(row["I_perp_G"]) for row in rows]),
        "I_arc": np.asarray([float(row["I_arc"]) for row in rows]),
        "Delta_F_barrier": np.asarray(
            [float(row["Delta_F_barrier"]) for row in rows]
        ),
    }
    if "state_kind" in trajectory:
        result["state_kind"] = np.asarray(trajectory["state_kind"]).copy()
    if "selected_reparameterization_output_residual" not in trajectory:
        raise AnalysisError(
            f"{label} replay interface lacks selected_reparameterization_output_residual"
        )
    residual = np.asarray(
        trajectory["selected_reparameterization_output_residual"], dtype=np.float64
    )
    if residual.shape != nof.shape:
        raise AnalysisError(f"{label} reparameterization residual shape differs")
    result["selected_reparameterization_output_residual"] = residual.copy()
    return result


def _validate_and_score_current_ivm(
    *, root: Path, barrier_reference: float
) -> Tuple[Dict[str, Array], Dict[str, object]]:
    current = load_historical_current_ivm(
        root / screen_runner.METRIC_BANK_RELATIVE,
        root / screen_runner.INITIAL_RELATIVE,
        maximum_count=100,
    )
    oracle = SurrogateOracle(
        fes_npz=root / screen_runner.FES_RELATIVE,
        metric_bank_npz=root / screen_runner.METRIC_BANK_RELATIVE,
    )
    scored = evaluate_current_paths_with_surrogate(current, oracle)
    exact = np.arange(0, 2020, 20, dtype=np.int64)
    if not np.array_equal(scored["nof_mean_force_incremental"], exact):
        raise AnalysisError("archived Current IVM lacks the exact 0..2000 baseline")
    if int(oracle.call_count) != exact.size * N_INTERNAL:
        raise AnalysisError("Current IVM blind-scoring oracle query ledger differs")
    for key in ("I_perp_G", "I_arc", "Delta_F_barrier"):
        values = np.asarray(scored[key], dtype=np.float64)
        if values.shape != exact.shape or np.any(~np.isfinite(values)):
            raise AnalysisError(f"blind-scored Current IVM {key} is invalid")
    return scored, {
        "source": "actual archived Current IVM public paths",
        "scoring": "new deterministic SurrogateOracle plus compute_article_metrics",
        "selection_baseline": True,
        "barrier_error_definition": (
            "absolute Delta_F_barrier minus frozen FES minimax barrier"
        ),
        "barrier_reference_kcal_per_mol": float(barrier_reference),
        "oracle_provenance": oracle.provenance(),
        "scored_public_path_count": int(exact.size),
        "deterministic_scoring_query_equivalents": int(oracle.call_count),
        "formal_nof_added": 0,
    }


def _minimax_saddle_coordinate(
    fes_npz: Path, minimax: Mapping[str, object]
) -> Array:
    path = np.asarray(minimax.get("grid_path"), dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or np.any(~np.isfinite(path)):
        raise AnalysisError("independent FES minimax result has invalid grid path")
    with np.load(fes_npz, allow_pickle=False) as loaded:
        X = np.asarray(loaded["X"], dtype=np.float64)
        Y = np.asarray(loaded["Y"], dtype=np.float64)
        Z = np.asarray(loaded["Z"], dtype=np.float64)
    x_axis = np.deg2rad(X[0])
    y_axis = np.deg2rad(Y[:, 0])
    values: List[float] = []
    for phi, psi in path:
        phi_difference = (x_axis - phi + np.pi) % (2.0 * np.pi) - np.pi
        psi_difference = (y_axis - psi + np.pi) % (2.0 * np.pi) - np.pi
        column = int(np.argmin(np.abs(phi_difference)))
        row = int(np.argmin(np.abs(psi_difference)))
        values.append(float(Z[row, column]))
    saddle_index = int(np.argmax(np.asarray(values)))
    observed_saddle = float(values[saddle_index])
    registered_saddle = _finite_number(minimax.get("saddle_kj_per_mol"), "minimax saddle")
    if not np.isclose(observed_saddle, registered_saddle, rtol=0.0, atol=1.0e-10):
        raise AnalysisError("minimax saddle coordinate does not reproduce saddle energy")
    return path[saddle_index].copy()


def validate_barrier_reference(
    *, root: Path, plan: screen_runner.FrozenPlan
) -> Tuple[Dict[str, object], Array, Array]:
    minimax = fes_minimax_barrier_reference(
        root / screen_runner.FES_RELATIVE,
        root / screen_runner.INITIAL_RELATIVE,
    )
    posthoc = _require_mapping(plan.raw.get("posthoc_scoring"), "posthoc_scoring")
    frozen_value = _finite_number(
        posthoc.get("barrier_reference_kcal_per_mol"),
        "frozen barrier_reference_kcal_per_mol",
    )
    observed_value = _finite_number(
        minimax.get("barrier_kcal_per_mol"), "independent FES minimax barrier"
    )
    if not np.isclose(observed_value, frozen_value, rtol=0.0, atol=1.0e-14):
        raise AnalysisError(
            "independent frozen-FES minimax barrier differs from the plan: "
            f"plan={frozen_value:.17g}, recomputed={observed_value:.17g}"
        )
    saddle = _minimax_saddle_coordinate(root / screen_runner.FES_RELATIVE, minimax)
    result = {
        key: value
        for key, value in minimax.items()
        if key != "grid_path"
    }
    result.update(
        {
            "plan_barrier_reference_kcal_per_mol": frozen_value,
            "plan_matches_independent_recomputation": True,
            "minimax_saddle_coordinate_rad": saddle.tolist(),
            "grid_path_sha256": _core_array_sha256(np.asarray(minimax["grid_path"])),
        }
    )
    return result, np.asarray(minimax["grid_path"], dtype=np.float64), saddle


def _import_verified_r1_plot_loader(root: Path, workspace: Path):
    r1_root = workspace / "code_on_server" / R1_CAMPAIGN_DIRECTORY
    registration_path = r1_root / "registrations/CAMPAIGN_PREREGISTRATION.json"
    loader_path = r1_root / "src/plot_registered_figures.py"
    if sha256_file(registration_path) != R1_REGISTRATION_SHA256:
        raise AnalysisError("archived r1 registration hash differs")
    if sha256_file(loader_path) != R1_PLOTTING_LOADER_SHA256:
        raise AnalysisError("archived r1 plotting loader hash differs")
    registration = _load_json_regular(registration_path, "archived r1 registration")
    hashes = _require_mapping(registration.get("hashes"), "r1 registration hashes")
    source_hashes = _require_mapping(hashes.get("source"), "r1 registered source hashes")
    input_hashes = _require_mapping(hashes.get("input"), "r1 registered input hashes")
    if source_hashes.get("src/plot_registered_figures.py") != R1_PLOTTING_LOADER_SHA256:
        raise AnalysisError("r1 registration does not bind the plotting loader")
    if input_hashes.get("inputs/reference/feneb_lambda0p02.npz") != R1_FENEB_SHA256:
        raise AnalysisError("r1 registration does not bind the selected FENEB archive")
    geometry_path = r1_root / "src/riemannian_torus_geometry.py"
    expected_geometry_hash = source_hashes.get("src/riemannian_torus_geometry.py")
    if not isinstance(expected_geometry_hash, str) or sha256_file(geometry_path) != expected_geometry_hash:
        raise AnalysisError("r1 registered geometry implementation differs")
    # The loader resolves the already imported geometry module.  Require the
    # r2 copy to be byte-identical to the registered r1 implementation.
    if sha256_file(root / "src/riemannian_torus_geometry.py") != expected_geometry_hash:
        raise AnalysisError("r2 geometry is not identical to the registered FENEB loader geometry")
    spec = importlib.util.spec_from_file_location(
        "_ala2d_r1_registered_plot_loader_read_only", loader_path
    )
    if spec is None or spec.loader is None:
        raise AnalysisError("cannot create an import spec for the r1 plotting loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, r1_root, registration, {
        "registration_sha256": R1_REGISTRATION_SHA256,
        "plotting_loader_sha256": R1_PLOTTING_LOADER_SHA256,
        "geometry_sha256": expected_geometry_hash,
    }


def load_archived_feneb_context(
    *, root: Path, workspace: Path, barrier_reference: float
) -> Tuple[Dict[str, Array], Dict[str, object]]:
    """Use the sealed r1 plotting loader to reconstruct FENEB, context only."""

    loader, r1_root, registration, provenance = _import_verified_r1_plot_loader(
        root, workspace
    )
    hashes = _require_mapping(registration["hashes"], "r1 hashes")
    input_hashes = _require_mapping(hashes["input"], "r1 input hashes")
    common_relative = "inputs/reference/MFEP_INITIAL_PATH_current_ivm_state0.npz"
    feneb_relative = "inputs/reference/feneb_lambda0p02.npz"
    common_path = r1_root / common_relative
    feneb_path = r1_root / feneb_relative
    for relative, path in ((common_relative, common_path), (feneb_relative, feneb_path)):
        expected = input_hashes.get(relative)
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise AnalysisError(f"archived r1 input hash differs for {relative}")
    if sha256_file(feneb_path) != R1_FENEB_SHA256:
        raise AnalysisError("archived selected FENEB digest differs")
    r2_feneb = root / "inputs/reference/feneb_lambda0p02.npz"
    if sha256_file(r2_feneb) != R1_FENEB_SHA256:
        raise AnalysisError("r2 FENEB reference copy differs from registered r1 archive")
    common = loader._load_npz(common_path, "registered causal common initial")
    feneb_record = loader._load_npz(feneb_path, "registered FENEB trajectory")
    feneb = loader._construct_feneb(feneb_record, common, registration)
    expected_nof = np.arange(0, 2020, 20, dtype=np.int64)
    if not np.array_equal(feneb["nof_mean_force_incremental"], expected_nof):
        raise AnalysisError("registered FENEB loader did not return exact 0..2000 NoF")
    provenance.update(
        {
            "feneb_npz_sha256": R1_FENEB_SHA256,
            "common_initial_sha256": input_hashes[common_relative],
            "construction": "r1 plot_registered_figures._construct_feneb",
            "selection_baseline": False,
            "role": "archived contextual comparison only",
            "barrier_reference_kcal_per_mol": float(barrier_reference),
            "formal_nof_added": 0,
        }
    )
    return feneb, provenance


def contextual_window_summary(
    trajectory: Trajectory, barrier_reference: float
) -> Dict[str, object]:
    windows = {
        "early_20_to_400": np.arange(20, 420, 20, dtype=np.int64),
        "front_middle_20_to_1600": np.arange(20, 1620, 20, dtype=np.int64),
        "tail_1620_to_2000": np.arange(1620, 2020, 20, dtype=np.int64),
    }
    result: Dict[str, object] = {}
    for window_name, checkpoints in windows.items():
        values = _extract_components(
            trajectory, checkpoints, barrier_reference, "archived FENEB context"
        )
        # Raw barrier is contextual; the registered selection quantity remains
        # its absolute error relative to the frozen FES minimax reference.
        source_barrier = np.asarray(trajectory["Delta_F_barrier"], dtype=np.float64)[
            exact_indices(trajectory, checkpoints, "archived FENEB context")
        ]
        result[window_name] = {
            "checkpoint_start": int(checkpoints[0]),
            "checkpoint_stop": int(checkpoints[-1]),
            "checkpoint_count": int(checkpoints.size),
            "I_perp_G": {
                "median": float(np.median(values["I_perp_G"])),
                "normalized_auc": normalized_auc(values["I_perp_G"], checkpoints),
                "first": float(values["I_perp_G"][0]),
                "last": float(values["I_perp_G"][-1]),
            },
            "Delta_F_barrier_kcal_per_mol": {
                "median": float(np.median(source_barrier)),
                "normalized_auc": normalized_auc(source_barrier, checkpoints),
                "first": float(source_barrier[0]),
                "last": float(source_barrier[-1]),
            },
            "Delta_F_error_kcal_per_mol": {
                "median": float(np.median(values["Delta_F_error"])),
                "normalized_auc": normalized_auc(values["Delta_F_error"], checkpoints),
                "first": float(values["Delta_F_error"][0]),
                "last": float(values["Delta_F_error"][-1]),
            },
        }
    return result


def _csv_bytes(rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for raw_row in rows:
        row: Dict[str, object] = {}
        for key in fieldnames:
            value = raw_row.get(key, "")
            if isinstance(value, (float, np.floating)):
                row[key] = format(float(value), ".17g")
            elif isinstance(value, (bool, np.bool_)):
                row[key] = "true" if bool(value) else "false"
            elif value is None:
                row[key] = ""
            else:
                row[key] = value
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _artifact_record(root: Path, path: Path, content: bytes) -> Dict[str, object]:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise AnalysisError(f"analysis artifact escapes campaign root: {path}") from error
    return {
        "relative_path": relative,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _publish_analysis_pointer(path: Path, payload: Mapping[str, object]) -> None:
    content = _canonical_json_bytes(payload)
    if path.exists():
        previous = _load_json_regular(path, "existing development analysis pointer")
        previous_binding = previous.get("analysis_input_sha256")
        current_binding = payload.get("analysis_input_sha256")
        if previous_binding == current_binding and path.read_bytes() != content:
            raise AnalysisError(
                "same analysis-input digest produced different JSON; refusing overwrite"
            )
    _atomic_replace(path, content)


def _inventory_rows(
    inventory: Sequence[Mapping[str, object]], input_digest: str
) -> List[Dict[str, object]]:
    return [
        {
            "artifact_scope": "DEVELOPMENT ONLY",
            "formal_evidence": False,
            "analysis_input_sha256": input_digest,
            "ordinal": item["ordinal"],
            "oracle_kind": item["oracle_kind"],
            "master_seed": item["master_seed"],
            "parameter_id": item["parameter_id"],
            "status": item["status"],
            "complete": item["complete"],
            "last_incremental_query_equivalents": item.get(
                "last_incremental_query_equivalents"
            ),
            "state_count_including_common_initial": item.get(
                "state_count_including_common_initial"
            ),
            "algorithmic_zero_nof_stop": (
                json.dumps(item.get("algorithmic_zero_nof_stop"), sort_keys=True)
                if item.get("algorithmic_zero_nof_stop") is not None
                else None
            ),
        }
        for item in inventory
    ]


def publish_nonselecting_inventory(
    *,
    root: Path,
    plan: screen_runner.FrozenPlan,
    inventory: Sequence[Mapping[str, object]],
    binding: Mapping[str, object],
) -> Dict[str, object]:
    input_digest = _json_digest(binding)
    token = input_digest[:16]
    complete_count = int(sum(bool(item["complete"]) for item in inventory))
    frozen_run_count = len(plan.run_specifications())
    if len(inventory) != frozen_run_count:
        raise AnalysisError(
            f"inventory row count is {len(inventory)}, expected {frozen_run_count}"
        )
    rows = _inventory_rows(inventory, input_digest)
    fieldnames = (
        "artifact_scope",
        "formal_evidence",
        "analysis_input_sha256",
        "ordinal",
        "oracle_kind",
        "master_seed",
        "parameter_id",
        "status",
        "complete",
        "last_incremental_query_equivalents",
        "state_count_including_common_initial",
        "algorithmic_zero_nof_stop",
    )
    csv_content = _csv_bytes(rows, fieldnames)
    csv_path = root / "reports" / f"development_screen_inventory.{token}.csv"
    _write_once_or_identical(csv_path, csv_content)
    status_counts: Dict[str, int] = {}
    for item in inventory:
        key = str(item["status"])
        status_counts[key] = status_counts.get(key, 0) + 1
    is_complete_inventory = complete_count == frozen_run_count
    payload: Dict[str, object] = {
        "schema_version": INVENTORY_SCHEMA,
        "status": (
            "COMPLETE_INVENTORY_NO_SELECTION"
            if is_complete_inventory
            else "INCOMPLETE_DEVELOPMENT_SCREEN"
        ),
        "formal_evidence": False,
        "new_restrained_md_started_by_analysis": False,
        "plan_id": plan.raw["plan_id"],
        "plan_sha256": plan.sha256,
        "analysis_input_sha256": input_digest,
        "frozen_run_count": frozen_run_count,
        "complete_exact_0_to_2000_count": complete_count,
        "not_complete_count": len(inventory) - complete_count,
        "status_counts": status_counts,
        "selection_performed": False,
        "recommended_parameter_id": None,
        "diagnostic_parameter_id": None,
        "last_observation_carried_forward": False,
        "reason": (
            "--allow-incomplete is an inventory-only mode. All 48 exact trajectories "
            "are present, but this invocation deliberately did not score, rank, or select."
            if is_complete_inventory
            else (
                "At least one of the 48 frozen runs lacks an exact public NoF-equivalent "
                "checkpoint at 2000. An algorithmic zero-NoF stop is valid execution "
                "evidence but is still a missing registered checkpoint and therefore "
                "cannot be selected."
            )
        ),
        "inventory": list(inventory),
        "artifacts": {
            "inventory_csv": _artifact_record(root, csv_path, csv_content),
            "static_png": [],
            "long_metrics_csv": None,
        },
        "analysis_binding": dict(binding),
    }
    _publish_analysis_pointer(
        root / "reports/development_screen_analysis.json", payload
    )
    return payload


def _summary_rows(
    combinations: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in combinations:
        rules = _require_mapping(item["rules"], "combination rules")
        selection = _require_mapping(rules["selection_metrics"], "selection metrics")
        correct = _require_mapping(rules["correct_path"], "correct path")
        rows.append(
            {
                "parameter_id": item["parameter_id"],
                "alpha": item["controls"]["alpha"],
                "alpha_integer_mantissa_notation": item[
                    "alpha_integer_mantissa_notation"
                ],
                "r": item["controls"]["r"],
                "m_max": item["controls"]["m_max"],
                "front_middle_pass": rules["front_middle"]["pass"],
                "tail_pass": rules["tail"]["pass"],
                "deterministic_guard_pass": rules["deterministic_guard"]["pass"],
                "correct_path_all_four_pass": correct["pass"],
                "primary_pass": rules["primary_pass"],
                "joint_front_middle_checkpoint_win_count": selection[
                    "joint_front_middle_checkpoint_win_count"
                ],
                "minimum_component_tail_checkpoint_win_fraction": selection[
                    "minimum_component_tail_checkpoint_win_fraction"
                ],
                "worst_individual_synthetic_seed_tail_I_perp_G_auc_ratio": selection[
                    "worst_individual_synthetic_seed_tail_I_perp_G_auc_ratio"
                ],
                "worst_individual_synthetic_seed_tail_Delta_F_error_auc_ratio": selection[
                    "worst_individual_synthetic_seed_tail_Delta_F_error_auc_ratio"
                ],
                "deterministic_worst_all_metric_checkpoint_ratio": selection[
                    "deterministic_worst_all_metric_checkpoint_ratio"
                ],
            }
        )
    return rows


def _append_long_trajectory_rows(
    rows: List[Dict[str, object]],
    *,
    trajectory: Trajectory,
    barrier_reference: float,
    series_role: str,
    parameter_id: Optional[str],
    oracle_kind: str,
    master_seed: Optional[int],
    scoring_source: str,
) -> None:
    nof = np.asarray(trajectory["nof_mean_force_incremental"], dtype=np.int64)
    metrics = {
        "I_perp_G": np.asarray(trajectory["I_perp_G"], dtype=np.float64),
        "I_arc": np.asarray(trajectory["I_arc"], dtype=np.float64),
        "Delta_F_barrier": np.asarray(
            trajectory["Delta_F_barrier"], dtype=np.float64
        ),
    }
    metrics["Delta_F_error"] = np.abs(
        metrics["Delta_F_barrier"] - float(barrier_reference)
    )
    for metric_name, values in metrics.items():
        if values.shape != nof.shape or np.any(~np.isfinite(values)):
            raise AnalysisError(
                f"long-metrics source {series_role}.{metric_name} is invalid"
            )
        for cost, value in zip(nof, values):
            rows.append(
                {
                    "series_role": series_role,
                    "parameter_id": parameter_id,
                    "oracle_kind": oracle_kind,
                    "master_seed": master_seed,
                    "incremental_query_equivalents": int(cost),
                    "metric": metric_name,
                    "value": float(value),
                    "scoring_source": scoring_source,
                    "selection_baseline": series_role == "current_ivm_selection_baseline",
                }
            )


def _configure_matplotlib():
    """Select a deterministic noninteractive style and report render dependencies."""

    matplotlib_config = Path(tempfile.gettempdir()) / "ala2d-v15-matplotlib-cache"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    # Deliberately override rather than inherit a user matplotlibrc location.
    os.environ["MPLCONFIGDIR"] = str(matplotlib_config)
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )
    from matplotlib import font_manager, ft2font
    import PIL
    import zlib

    font_path = Path(
        font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans"),
            fallback_to_default=False,
        )
    ).resolve()
    if not font_path.is_file():
        raise AnalysisError("fixed DejaVu Sans render font is unavailable")
    runtime = {
        "matplotlib_version": matplotlib.__version__,
        "pillow_version": PIL.__version__,
        "freetype_version": str(ft2font.__freetype_version__),
        "zlib_version": zlib.ZLIB_VERSION,
        "backend": str(matplotlib.get_backend()),
        "font_path": str(font_path),
        "font_sha256": sha256_file(font_path),
        "rcparams": {
            key: matplotlib.rcParams[key]
            for key in (
                "font.family",
                "mathtext.fontset",
                "axes.unicode_minus",
                "figure.facecolor",
                "axes.facecolor",
                "savefig.facecolor",
                "savefig.transparent",
            )
        },
    }
    return matplotlib, runtime


def _render_parameter_comparison_png(
    combinations: Sequence[Mapping[str, object]],
) -> bytes:
    _matplotlib, _runtime = _configure_matplotlib()
    import matplotlib.pyplot as plt

    labels = [
        f"{item['alpha_integer_mantissa_notation']}\nr={item['controls']['r']}, "
        f"m={item['controls']['m_max']}"
        for item in combinations
    ]
    selection = [item["rules"]["selection_metrics"] for item in combinations]
    primary = np.asarray([bool(item["rules"]["primary_pass"]) for item in combinations])
    colors = np.where(primary, "#0072B2", "#999999")
    x = np.arange(len(combinations))
    series = (
        (
            "Joint front/middle checkpoint wins",
            [row["joint_front_middle_checkpoint_win_count"] for row in selection],
            80.0,
            "count (of 80)",
        ),
        (
            "Minimum tail checkpoint win fraction",
            [row["minimum_component_tail_checkpoint_win_fraction"] for row in selection],
            0.8,
            "fraction",
        ),
        (
            "Worst noisy-tail individual AUC ratio",
            [
                max(
                    row["worst_individual_synthetic_seed_tail_I_perp_G_auc_ratio"],
                    row[
                        "worst_individual_synthetic_seed_tail_Delta_F_error_auc_ratio"
                    ],
                )
                for row in selection
            ],
            1.0,
            "candidate / Current IVM",
        ),
    )
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), sharex=True)
    for axis, (title, values, threshold, ylabel) in zip(axes, series):
        axis.scatter(x, values, s=48, c=colors, edgecolor="black", linewidth=0.45)
        axis.axhline(threshold, color="#D55E00", linestyle="--", linewidth=1.4)
        axis.set_title(title, loc="left", fontsize=11)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    fig.suptitle(
        "DEVELOPMENT ONLY — frozen Ala2D IR-IVM parameter screen",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Blue = complete primary pass; gray = failure. Query equivalents are not formal NoF.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.04, 1.0, 0.96))
    stream = io.BytesIO()
    fig.savefig(
        stream,
        format="png",
        dpi=180,
        metadata={"Software": "Ala2D V15 development analyzer"},
    )
    plt.close(fig)
    return stream.getvalue()


def _render_diagnostic_curves_png(
    *,
    parameter_id: str,
    candidate_by_seed: Mapping[int, Trajectory],
    deterministic: Trajectory,
    baseline: Trajectory,
    feneb: Trajectory,
    barrier_reference: float,
    primary_pass: bool,
) -> bytes:
    _matplotlib, _runtime = _configure_matplotlib()
    import matplotlib.pyplot as plt

    nof = np.asarray(deterministic["nof_mean_force_incremental"], dtype=np.int64)
    seed_order = sorted(candidate_by_seed)
    metric_specs = (
        ("I_perp_G", r"$I_\perp^G$"),
        ("I_arc", r"$I_{\mathrm{arc}}$"),
        ("Delta_F_error", r"$|\Delta F-\Delta F^\star|$ (kcal mol$^{-1}$)"),
    )

    def values(trajectory: Trajectory, metric: str) -> Array:
        if metric == "Delta_F_error":
            raw = np.abs(
                np.asarray(trajectory["Delta_F_barrier"], dtype=np.float64)
                - barrier_reference
            )
        else:
            raw = np.asarray(trajectory[metric], dtype=np.float64)
        return np.maximum(raw, np.finfo(np.float64).tiny)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharex=True)
    for axis, (metric, ylabel) in zip(axes, metric_specs):
        stack = np.stack([values(candidate_by_seed[seed], metric) for seed in seed_order])
        median = np.median(stack, axis=0)
        minimum = np.min(stack, axis=0)
        maximum = np.max(stack, axis=0)
        axis.fill_between(nof, minimum, maximum, color="#56B4E9", alpha=0.22)
        axis.plot(nof, median, color="#0072B2", linewidth=2.2, label="IR-IVM synthetic median")
        axis.plot(
            nof,
            values(deterministic, metric),
            color="#009E73",
            linewidth=1.6,
            linestyle="--",
            label="IR-IVM deterministic",
        )
        axis.plot(
            np.asarray(baseline["nof_mean_force_incremental"]),
            values(baseline, metric),
            color="#222222",
            linewidth=2.0,
            label="Current IVM (selection baseline)",
        )
        axis.plot(
            np.asarray(feneb["nof_mean_force_incremental"]),
            values(feneb, metric),
            color="#D55E00",
            linewidth=1.5,
            linestyle=":",
            label="Archived FENEB (context only)",
        )
        axis.axvspan(1620, 2000, color="#F0E442", alpha=0.10)
        axis.set_yscale("log")
        axis.set_xlim(0, 2000)
        axis.set_xlabel("Development query equivalents")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", color="#e2e2e2", linewidth=0.55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    verdict = "primary pass" if primary_pass else "diagnostic only — NOT a passing recommendation"
    fig.suptitle(
        f"DEVELOPMENT ONLY — {parameter_id} ({verdict})",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.055,
        "Candidate and Current IVM use blinded deterministic post-hoc scoring; archived FENEB is contextual and excluded from selection.",
        ha="center",
        fontsize=8.4,
    )
    fig.tight_layout(rect=(0.02, 0.13, 1.0, 0.93))
    stream = io.BytesIO()
    fig.savefig(
        stream,
        format="png",
        dpi=180,
        metadata={"Software": "Ala2D V15 development analyzer"},
    )
    plt.close(fig)
    return stream.getvalue()


def analyze_development_screen(
    *,
    root: Path = ROOT,
    screen_root: Optional[Path] = None,
    allow_incomplete: bool = False,
) -> Dict[str, object]:
    """Validate, blindly score, apply the frozen screen, and publish artifacts."""

    root = Path(root).resolve()
    workspace = root.parents[1]
    screen_root = (
        root / "runs/development_screen"
        if screen_root is None
        else Path(screen_root).resolve()
    )
    analyzer_path = Path(__file__).resolve()
    analyzer_hash_at_start = sha256_file(analyzer_path)
    analysis_source_hashes_at_start = _analysis_source_hashes(root)
    plan = screen_runner.load_frozen_plan(root)
    screen_runner._validate_required_files(root)
    screen_runner._validate_formal_pdfs(workspace)

    manifest_path = screen_root / screen_runner.EXECUTION_MANIFEST_NAME
    if not manifest_path.exists():
        if screen_root.exists():
            if not screen_root.is_dir() or screen_root.is_symlink():
                raise AnalysisError(f"development screen root is unsafe: {screen_root}")
            orphan_entries = sorted(path.name for path in screen_root.iterdir())
            if orphan_entries:
                raise AnalysisError(
                    "development artifacts exist without the required execution "
                    f"manifest: {orphan_entries}"
                )
        inventory = [
            {
                "ordinal": spec.ordinal,
                "oracle_kind": spec.oracle_kind,
                "master_seed": spec.master_seed,
                "parameter_id": spec.parameter.parameter_id,
                "relative_directory": spec.relative_directory.as_posix(),
                "expected_terminal_incremental_query_equivalents": plan.nof_budget,
                "status": "missing",
                "complete": False,
                "last_incremental_query_equivalents": None,
            }
            for spec in plan.run_specifications()
        ]
        binding = {
            "plan_sha256": plan.sha256,
            "execution_manifest_sha256": None,
            "analyzer_source_sha256": analyzer_hash_at_start,
            "analysis_source_sha256": analysis_source_hashes_at_start,
            "run_inventory_digest": _json_digest({"inventory": inventory}),
        }
        if allow_incomplete:
            return publish_nonselecting_inventory(
                root=root, plan=plan, inventory=inventory, binding=binding
            )
        raise AnalysisError(
            "development screen has no execution manifest: 0/48 exact runs; "
            "use --allow-incomplete only for a nonselecting inventory"
        )

    manifest_path, _manifest, manifest_verification = validate_execution_manifest(
        root=root,
        workspace=workspace,
        screen_root=screen_root,
        plan=plan,
    )
    initial_path, initial_M, initial_gradient = screen_runner._load_initial(root)
    inventory, trajectories = collect_validated_runs(
        root=root,
        screen_root=screen_root,
        plan=plan,
        execution_manifest_path=manifest_path,
        initial_path=initial_path,
        initial_M=initial_M,
        initial_gradient=initial_gradient,
    )

    complete_count = int(sum(bool(item["complete"]) for item in inventory))
    base_binding: Dict[str, object] = {
        "plan_sha256": plan.sha256,
        "execution_manifest_sha256": sha256_file(manifest_path),
        "execution_manifest_verification": manifest_verification,
        "analyzer_source_sha256": analyzer_hash_at_start,
        "analysis_source_sha256": analysis_source_hashes_at_start,
        "run_inventory_digest": _json_digest({"inventory": inventory}),
        "run_inventory": inventory,
    }
    if allow_incomplete:
        end_manifest_path, _end_manifest, end_verification = validate_execution_manifest(
            root=root,
            workspace=workspace,
            screen_root=screen_root,
            plan=plan,
        )
        end_inventory, _end_trajectories = collect_validated_runs(
            root=root,
            screen_root=screen_root,
            plan=plan,
            execution_manifest_path=end_manifest_path,
            initial_path=initial_path,
            initial_M=initial_M,
            initial_gradient=initial_gradient,
        )
        if (
            end_manifest_path != manifest_path
            or end_verification != manifest_verification
            or _json_digest({"inventory": end_inventory})
            != _json_digest({"inventory": inventory})
            or _analysis_source_hashes(root) != analysis_source_hashes_at_start
        ):
            raise AnalysisError("execution or inventory changed during inventory-only analysis")
        return publish_nonselecting_inventory(
            root=root,
            plan=plan,
            inventory=inventory,
            binding=base_binding,
        )
    if complete_count != len(plan.run_specifications()):
        raise AnalysisError(
            f"development screen is incomplete: {complete_count}/48 runs contain "
            "every exact checkpoint through 2000; LOCF and selection are forbidden"
        )

    expected_ordinals = {spec.ordinal for spec in plan.run_specifications()}
    if set(trajectories) != expected_ordinals:
        raise AnalysisError("complete inventory and replayed trajectory set differ")

    barrier_report, minimax_path, saddle_point = validate_barrier_reference(
        root=root, plan=plan
    )
    barrier_reference = float(barrier_report["plan_barrier_reference_kcal_per_mol"])
    baseline, baseline_provenance = _validate_and_score_current_ivm(
        root=root, barrier_reference=barrier_reference
    )
    feneb, feneb_provenance = load_archived_feneb_context(
        root=root, workspace=workspace, barrier_reference=barrier_reference
    )
    feneb_window_summary = contextual_window_summary(feneb, barrier_reference)

    scored_runs: Dict[int, Dict[str, Array]] = {}
    for spec in plan.run_specifications():
        scored_runs[spec.ordinal] = score_public_paths_blind(
            trajectories[spec.ordinal],
            fes_npz=root / screen_runner.FES_RELATIVE,
            metric_bank_npz=root / screen_runner.METRIC_BANK_RELATIVE,
            label=(
                f"{spec.oracle_kind} seed={spec.master_seed} "
                f"parameter={spec.parameter.parameter_id}"
            ),
        )

    combinations: List[Dict[str, object]] = []
    combination_curves: Dict[str, Dict[str, object]] = {}
    specifications = plan.run_specifications()
    for parameter in plan.combinations:
        deterministic_specs = [
            spec
            for spec in specifications
            if spec.parameter.parameter_id == parameter.parameter_id
            and spec.oracle_kind == "deterministic"
        ]
        synthetic_specs = [
            spec
            for spec in specifications
            if spec.parameter.parameter_id == parameter.parameter_id
            and spec.oracle_kind == "synthetic"
        ]
        if len(deterministic_specs) != 1 or len(synthetic_specs) != 3:
            raise AnalysisError("frozen run order does not provide one plus three repetitions")
        deterministic = scored_runs[deterministic_specs[0].ordinal]
        synthetic = {
            int(spec.master_seed): scored_runs[spec.ordinal]
            for spec in synthetic_specs
        }
        rules = evaluate_parameter_rules(
            synthetic,
            deterministic,
            baseline,
            barrier_reference,
            plan.raw,
            minimax_path=minimax_path,
            saddle_point=saddle_point,
        )
        item = {
            "parameter_id": parameter.parameter_id,
            "alpha_integer_mantissa_notation": parameter.alpha_label,
            "controls": asdict(parameter.controls),
            "rules": rules,
        }
        combinations.append(item)
        combination_curves[parameter.parameter_id] = {
            "deterministic": deterministic,
            "synthetic": synthetic,
        }

    passing = [item for item in combinations if item["rules"]["primary_pass"]]
    passing.sort(key=lambda item: passing_sort_key(item["rules"], item["controls"]))
    diagnostic_order = sorted(
        combinations,
        key=lambda item: diagnostic_sort_key(item["rules"], item["controls"]),
    )
    diagnostic = diagnostic_order[0]
    recommended = passing[0] if passing else None
    eligible = [item for item in combinations if item["rules"]["correct_path"]["pass"]]

    long_rows: List[Dict[str, object]] = []
    _append_long_trajectory_rows(
        long_rows,
        trajectory=baseline,
        barrier_reference=barrier_reference,
        series_role="current_ivm_selection_baseline",
        parameter_id=None,
        oracle_kind="archived_current_ivm_rescored",
        master_seed=None,
        scoring_source="deterministic_surrogate_blind_posthoc",
    )
    _append_long_trajectory_rows(
        long_rows,
        trajectory=feneb,
        barrier_reference=barrier_reference,
        series_role="archived_feneb_context_only",
        parameter_id=None,
        oracle_kind="archived_feneb",
        master_seed=None,
        scoring_source="registered_r1_article_metric_loader_not_selection_baseline",
    )
    for spec in specifications:
        _append_long_trajectory_rows(
            long_rows,
            trajectory=scored_runs[spec.ordinal],
            barrier_reference=barrier_reference,
            series_role="irivm_candidate",
            parameter_id=spec.parameter.parameter_id,
            oracle_kind=spec.oracle_kind,
            master_seed=spec.master_seed,
            scoring_source="deterministic_surrogate_blind_posthoc",
        )

    base_binding.update(
        {
            "barrier_reference": barrier_report,
            "current_ivm_baseline_provenance": baseline_provenance,
            "archived_feneb_context_provenance": feneb_provenance,
            "blind_scoring_contract": {
                "every_public_path_scored": True,
                "scoring_oracle": "new deterministic SurrogateOracle per trajectory",
                "metric_function": "compute_article_metrics",
                "algorithm_had_access_to_scores": False,
                "candidate_stored_online_metrics_used_for_selection": False,
                "formal_nof_added": 0,
            },
        }
    )
    _matplotlib, rendering_runtime = _configure_matplotlib()

    base_binding["analysis_runtime"] = {
        "python_full_version": os.sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "rendering": rendering_runtime,
    }
    input_digest = _json_digest(base_binding)
    token = input_digest[:16]

    summary_fields = (
        "parameter_id",
        "alpha",
        "alpha_integer_mantissa_notation",
        "r",
        "m_max",
        "front_middle_pass",
        "tail_pass",
        "deterministic_guard_pass",
        "correct_path_all_four_pass",
        "primary_pass",
        "joint_front_middle_checkpoint_win_count",
        "minimum_component_tail_checkpoint_win_fraction",
        "worst_individual_synthetic_seed_tail_I_perp_G_auc_ratio",
        "worst_individual_synthetic_seed_tail_Delta_F_error_auc_ratio",
        "deterministic_worst_all_metric_checkpoint_ratio",
    )
    summary_rows = _summary_rows(combinations)
    for row in summary_rows:
        row.update(
            {
                "artifact_scope": "DEVELOPMENT ONLY",
                "formal_evidence": False,
                "analysis_input_sha256": input_digest,
            }
        )
    for row in long_rows:
        row.update(
            {
                "artifact_scope": "DEVELOPMENT ONLY",
                "formal_evidence": False,
                "analysis_input_sha256": input_digest,
            }
        )
    summary_fields = (
        "artifact_scope",
        "formal_evidence",
        "analysis_input_sha256",
    ) + summary_fields
    summary_content = _csv_bytes(summary_rows, summary_fields)
    long_fields = (
        "artifact_scope",
        "formal_evidence",
        "analysis_input_sha256",
        "series_role",
        "parameter_id",
        "oracle_kind",
        "master_seed",
        "incremental_query_equivalents",
        "metric",
        "value",
        "scoring_source",
        "selection_baseline",
    )
    long_content = _csv_bytes(long_rows, long_fields)
    parameter_png = _render_parameter_comparison_png(combinations)
    curves = combination_curves[str(diagnostic["parameter_id"])]
    curve_png = _render_diagnostic_curves_png(
        parameter_id=str(diagnostic["parameter_id"]),
        candidate_by_seed=curves["synthetic"],
        deterministic=curves["deterministic"],
        baseline=baseline,
        feneb=feneb,
        barrier_reference=barrier_reference,
        primary_pass=bool(diagnostic["rules"]["primary_pass"]),
    )

    summary_path = root / "reports" / f"development_screen_summary.{token}.csv"
    long_path = root / "reports" / f"development_screen_long_metrics.{token}.csv"
    parameter_png_path = (
        root / "figures/development_screen" / f"parameter_comparison.{token}.png"
    )
    curve_png_path = (
        root
        / "figures/development_screen"
        / f"diagnostic_full_curves.{token}.png"
    )
    end_manifest_path, _end_manifest, end_manifest_verification = (
        validate_execution_manifest(
            root=root,
            workspace=workspace,
            screen_root=screen_root,
            plan=plan,
        )
    )
    if (
        end_manifest_path != manifest_path
        or end_manifest_verification != manifest_verification
    ):
        raise AnalysisError("execution binding changed during blinded analysis")
    _end_feneb, end_feneb_provenance = load_archived_feneb_context(
        root=root,
        workspace=workspace,
        barrier_reference=barrier_reference,
    )
    if end_feneb_provenance != feneb_provenance:
        raise AnalysisError("archived FENEB context binding changed during analysis")
    end_inventory, _end_trajectories = collect_validated_runs(
        root=root,
        screen_root=screen_root,
        plan=plan,
        execution_manifest_path=end_manifest_path,
        initial_path=initial_path,
        initial_M=initial_M,
        initial_gradient=initial_gradient,
    )
    if _json_digest({"inventory": end_inventory}) != _json_digest(
        {"inventory": inventory}
    ):
        raise AnalysisError(
            "run membership, ledger, state hashes, or completion changed during analysis"
        )
    if (
        sha256_file(analyzer_path) != analyzer_hash_at_start
        or _analysis_source_hashes(root) != analysis_source_hashes_at_start
    ):
        raise AnalysisError("analysis source changed during analysis; outputs not published")
    for path, content in (
        (summary_path, summary_content),
        (long_path, long_content),
        (parameter_png_path, parameter_png),
        (curve_png_path, curve_png),
    ):
        _write_once_or_identical(path, content)
    if recommended is None:
        decision = "NO_COMBINATION_PASSED"
        recommendation_text = (
            "No parameter combination passed the frozen screen. The diagnostic curve "
            "is not promoted; revise the algorithm or freeze a new complete systematic "
            "screen before formal restrained-MD work."
        )
    else:
        decision = "DEVELOPMENT_PASS"
        recommendation_text = (
            "The first parameter under the frozen passing_order is eligible only for a "
            "separately registered formal restrained-MD pilot; this is not formal acceptance."
        )
    payload: Dict[str, object] = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "COMPLETE_DEVELOPMENT_SCREEN_ANALYSIS",
        "decision": decision,
        "formal_evidence": False,
        "selection_is_not_formal_acceptance": True,
        "new_restrained_md_started_by_analysis": False,
        "plan_id": plan.raw["plan_id"],
        "plan_sha256": plan.sha256,
        "analysis_input_sha256": input_digest,
        "frozen_run_count": len(specifications),
        "complete_exact_0_to_2000_count": complete_count,
        "last_observation_carried_forward": False,
        "blind_scoring": base_binding["blind_scoring_contract"],
        "barrier_reference": barrier_report,
        "current_ivm_selection_baseline": baseline_provenance,
        "archived_feneb_context": {
            "provenance": feneb_provenance,
            "selection_baseline": False,
            "window_summary": feneb_window_summary,
        },
        "eligible_correct_path_parameter_ids": [
            item["parameter_id"] for item in eligible
        ],
        "passing_parameter_ids_in_selection_order": [
            item["parameter_id"] for item in passing
        ],
        "recommended_parameter_id": (
            recommended["parameter_id"] if recommended is not None else None
        ),
        "diagnostic_parameter_id": diagnostic["parameter_id"],
        "diagnostic_parameter_is_recommendation": bool(
            recommended is not None
            and diagnostic["parameter_id"] == recommended["parameter_id"]
        ),
        "recommendation": recommendation_text,
        "passing_order": plan.raw["selection_rule"]["passing_order"],
        "parameter_results": combinations,
        "artifacts": {
            "summary_csv": _artifact_record(root, summary_path, summary_content),
            "long_metrics_csv": _artifact_record(root, long_path, long_content),
            "parameter_comparison_png": _artifact_record(
                root, parameter_png_path, parameter_png
            ),
            "diagnostic_full_curves_png": _artifact_record(
                root, curve_png_path, curve_png
            ),
            "pdf_created_or_modified": False,
        },
        "analysis_binding": base_binding,
    }
    _publish_analysis_pointer(
        root / "reports/development_screen_analysis.json", payload
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen-root",
        type=Path,
        default=None,
        help="development-screen result root (defaults to the frozen campaign path)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="publish an inventory only (even if complete); never score, rank, or select",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    payload = analyze_development_screen(
        root=ROOT,
        screen_root=arguments.screen_root,
        allow_incomplete=bool(arguments.allow_incomplete),
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload.get("decision"),
                "complete_exact_0_to_2000_count": payload[
                    "complete_exact_0_to_2000_count"
                ],
                "recommended_parameter_id": payload["recommended_parameter_id"],
                "diagnostic_parameter_id": payload["diagnostic_parameter_id"],
                "analysis_report": str(
                    ROOT / "reports/development_screen_analysis.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AnalysisError, screen_runner.ScreenError, ValueError) as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        raise SystemExit(2)
