#!/usr/bin/env python3
"""Fail-closed audit for the low-r coherence-conditioned QM10 campaign.

This program never runs an optimizer.  It authenticates the preregistration,
every registered NPZ and receipt, closes the Number-of-Forces ledgers, then
performs the frozen latter-half LOCF comparison with the Article IVM.  A
well-formed campaign with no strict winner is a valid scientific result and
therefore exits zero.  Missing, malformed, non-finite, or unauthenticated data
are integrity failures and exit non-zero after reports have been written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_FILE = Path(__file__).resolve()
CAMPAIGN_ROOT = SCRIPT_FILE.parents[1]
SRC_ROOT = SCRIPT_FILE.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from q10_irivm_atomic_tangent_secant_core import (  # noqa: E402
    ARTICLE_N10,
    ARTICLE_N20,
    ARTICLE_NF_BUDGET,
    BACKTRACK,
    EXPERIMENT_ID,
    FORMAL_LEPS_OPTIMIZER,
    FORMAL_LEPS_REGISTRY,
    KKT_ATOL,
    METRIC_SOURCE,
    N20_GATE_MANIFEST,
    RECOVERY_BAND_EVALUATION_CAP,
    SPRING_DTAU,
    SPRING_MAX_ITER,
    SPRING_TOLERANCE,
    REGISTERED_CONTROLS,
    REGISTRATION,
    REPOSITORY_ROOT,
    THIS_FILE as CORE_SOURCE,
    TRANSACTION_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT,
    Controls,
    validate_preregistration,
)


REFERENCE_DELTA_E = 67.50188060819488
EXPECTED_ARTICLE_SHA256 = {
    10: "a09bdc481d98c37b448ae9efaff045aba6208f247b8bcbf6d296b1f1ec7a7a40",
    20: "3cf6649deb4b6bd2c11da51ca204a06513cad8e6702867e59619a2956d52162b",
}
ARTICLE_FILES = {10: ARTICLE_N10, 20: ARTICLE_N20}
EXPECTED_ARTICLE_TERMINAL_NF = {10: 951, 20: 2377}
CHECKPOINT_COUNT = 51
ELIGIBLE_STOP_REASONS = ("budget", "stationary", "outer_limit")
ENERGY_GRADIENT_ATOL = 2.0e-12
RECOVERY_EQUIDISTANCE_CONTRACTION = 0.2
AUDIT_RECOVERY_IPERP_ATOL = 1.0e-12
AUDIT_RECOVERY_EQUIDISTANCE_ATOL = 1.0e-14
EXPECTED_EXPERIMENT_ID = "q10-irivm-low-r-coherence-secant-v2-20260830"
EXPECTED_N20_GATE_MANIFEST = (
    CAMPAIGN_ROOT / "reports/n20_gate_audit/audit_manifest.json"
)
SECANT_CURVATURE_THRESHOLD = 1.0e-6
SECANT_PUBLICATION_MODES = (
    "not_applicable",
    "strong_top1",
    "conservative_all",
    "ordinary_all",
)
EXPECTED_REGISTERED_CONTROLS = tuple(
    (n_internal, alpha, ratio, 4)
    for n_internal in (10, 20)
    for alpha in (1.0e-3, 2.0e-3)
    for ratio in (0.60, 0.65)
)
RECOVERY_DEFINITION = (
    "zero-NoF adjacent-chord-coherent saddle seed, direct-energy "
    "improved-tangent chord-Newton projection, one evaluated projected band, "
    "unchanged atomic gate, coherence-conditioned accepted physical secant "
    "publication"
)
SECANT_NORMALIZED_CURVATURE_FORMULA = "(s^T y)/(||s|| ||y||)"
SECANT_STRONG_PUBLICATION = (
    "highest normalized-curvature certified pair only"
)
SECANT_ALL_PUBLICATION = "all curvature-certified pairs"
SECANT_TIE_BREAK = "smaller 1-based internal image index"


class IntegrityError(RuntimeError):
    """A result cannot be trusted or compared."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path, payload):
    atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def atomic_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def scalar(archive, name):
    require(name in archive, "archive lacks scalar field: " + name)
    value = np.asarray(archive[name])
    require(value.shape == (), name + " must be scalar")
    return value.item()


def require_finite_json(value, location="json"):
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_json(item, location + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_json(item, location + "[{}]".format(index))
    elif isinstance(value, (float, np.floating)):
        require(np.isfinite(value), location + " contains a non-finite number")


def load_json_lines(values, name):
    result = []
    for index, value in enumerate(np.asarray(values).reshape(-1)):
        try:
            decoded = json.loads(str(value))
        except Exception as error:
            raise IntegrityError(
                "{}[{}] is invalid JSON: {}".format(name, index, error)
            )
        require(isinstance(decoded, dict), name + " entries must be objects")
        require_finite_json(decoded, "{}[{}]".format(name, index))
        result.append(decoded)
    return result


def load_metric_module():
    spec = importlib.util.spec_from_file_location(
        "_q10_atomic_tangent_secant_offline_metrics", METRIC_SOURCE
    )
    if spec is None or spec.loader is None:
        raise IntegrityError("cannot load offline metric source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(
        float(module.REFERENCE_BARRIER) == REFERENCE_DELTA_E,
        "offline reference Delta E drifted",
    )
    return module


def latter_half_checkpoints(nof):
    values = np.asarray(nof, dtype=np.int64).reshape(-1)
    require(values.size > 0, "Article NoF ledger is empty")
    require(np.all(np.diff(values) > 0), "Article NoF ledger is not strictly increasing")
    lower = int(np.ceil((int(values[0]) + int(values[-1])) / 2.0))
    result = np.rint(
        np.linspace(lower, int(values[-1]), CHECKPOINT_COUNT)
    ).astype(np.int64)
    require(
        np.unique(result).size == CHECKPOINT_COUNT,
        "latter-half checkpoints are not 51 unique integers",
    )
    return result


def locf_indices(nof, checkpoints):
    values = np.asarray(nof, dtype=np.int64).reshape(-1)
    points = np.asarray(checkpoints, dtype=np.int64).reshape(-1)
    require(values.size > 0, "LOCF NoF ledger is empty")
    require(np.all(np.diff(values) > 0), "LOCF NoF ledger is not strictly increasing")
    indices = np.searchsorted(values, points, side="right") - 1
    require(np.all(indices >= 0), "a checkpoint precedes the initial public state")
    return np.asarray(indices, dtype=np.int64)


def metric_values(metric, paths, q_matrix):
    values = np.asarray(paths, dtype=np.float64)
    require(values.ndim == 3 and values.shape[2] == 10, "metric paths have invalid shape")
    i_perp = np.empty(values.shape[0], dtype=np.float64)
    delta_e = np.empty(values.shape[0], dtype=np.float64)
    for index, path in enumerate(values):
        i_perp[index] = metric.article_path_metric(path, q=q_matrix)[0]
        delta_e[index] = metric.linear500_barrier(path, q=q_matrix)
    error = np.abs(delta_e - REFERENCE_DELTA_E)
    require(np.all(np.isfinite(i_perp)), "offline I_perp contains non-finite values")
    require(np.all(np.isfinite(delta_e)), "offline densified Delta E is non-finite")
    require(np.all(np.isfinite(error)), "offline Delta E error is non-finite")
    require(np.all(i_perp >= 0.0), "offline I_perp is negative")
    return i_perp, delta_e, error


def safe_ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    require(np.all(numerator >= 0.0), "ratio numerator is negative")
    require(np.all(denominator >= 0.0), "ratio denominator is negative")
    maximum = np.finfo(np.float64).max
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, maximum),
        where=denominator > 0.0,
    )


def require_close(observed, expected, label):
    observed = float(observed)
    expected = float(expected)
    tolerance = float(
        512.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(observed), abs(expected))
    )
    require(abs(observed - expected) <= tolerance, label + " drifted")


def require_index_list(transaction, name, n_internal):
    """Load one canonical list of 1-based internal path-image indices."""

    raw = transaction.get(name)
    require(isinstance(raw, list), name + " must be a JSON list")
    values = []
    for value in raw:
        require(
            type(value) is int,  # bool is deliberately rejected
            name + " contains a non-integer index",
        )
        require(
            1 <= value <= int(n_internal),
            name + " contains an out-of-range internal image index",
        )
        values.append(int(value))
    require(values == sorted(set(values)), name + " is not sorted and unique")
    return values


def require_finite_float_list(transaction, name):
    raw = transaction.get(name)
    require(isinstance(raw, list), name + " must be a JSON list")
    values = []
    for value in raw:
        require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            name + " contains a non-numeric value",
        )
        numeric = float(value)
        require(np.isfinite(numeric), name + " contains a non-finite value")
        values.append(numeric)
    return values


def require_nonnegative_int(transaction, name):
    value = transaction.get(name)
    require(type(value) is int and value >= 0, name + " must be a nonnegative integer")
    return int(value)


def independent_secant_partition(
    direct_path,
    direct_gradient,
    projected_path,
    projected_gradient,
    publication_mode,
):
    """Recompute C/R/P/O without importing or trusting the optimizer publisher."""

    direct_path = np.asarray(direct_path, dtype=np.float64)
    direct_gradient = np.asarray(direct_gradient, dtype=np.float64)
    projected_path = np.asarray(projected_path, dtype=np.float64)
    projected_gradient = np.asarray(projected_gradient, dtype=np.float64)
    require(
        direct_path.shape == projected_path.shape
        and direct_gradient.shape == projected_gradient.shape
        and direct_path.shape == direct_gradient.shape,
        "secant reconstruction states have inconsistent shapes",
    )
    n_internal = int(direct_path.shape[0] - 2)
    certified = []
    rejected = []
    normalized = {}
    for image_index in range(1, n_internal + 1):
        step = projected_path[image_index] - direct_path[image_index]
        change = projected_gradient[image_index] - direct_gradient[image_index]
        step_norm = float(np.linalg.norm(step))
        change_norm = float(np.linalg.norm(change))
        curvature = float(np.dot(step.ravel(), change.ravel()))
        require(
            np.isfinite(step_norm)
            and np.isfinite(change_norm)
            and np.isfinite(curvature),
            "independent recovery secant contains a non-finite value",
        )
        threshold = float(
            SECANT_CURVATURE_THRESHOLD * step_norm * change_norm
        )
        if step_norm > 0.0 and change_norm > 0.0 and curvature > threshold:
            certified.append(image_index)
            normalized[image_index] = float(
                curvature / (step_norm * change_norm)
            )
        else:
            rejected.append(image_index)

    if publication_mode == "strong_top1":
        if certified:
            # An exact floating-point tie is resolved by the smaller 1-based
            # path-image index; no hidden tolerance or fitted constant exists.
            winner = max(certified, key=lambda index: (normalized[index], -index))
            published = [int(winner)]
        else:
            published = []
        omitted = [index for index in certified if index not in published]
    elif publication_mode in ("conservative_all", "ordinary_all"):
        published = list(certified)
        omitted = []
    else:
        raise IntegrityError(
            "independent partition received an invalid publication mode"
        )
    return {
        "certified": certified,
        "rejected": rejected,
        "omitted": omitted,
        "published": published,
        "published_normalized_curvatures": [
            normalized[index] for index in published
        ],
    }


def audit_secant_publication(
    transaction,
    n_internal,
    *,
    recovery_committed,
    expected_mode,
    direct_path=None,
    direct_gradient=None,
    projected_path=None,
    projected_gradient=None,
):
    """Authenticate one transaction's fixed-schema secant publication log."""

    require(
        transaction.get("accepted_recovery_secants_enabled") is True,
        "transaction disabled accepted recovery secants",
    )
    require(
        transaction.get(
            "main_memory_updates_only_after_complete_recovery_commit"
        )
        is True,
        "transaction permits recovery memory updates before an atomic commit",
    )
    require_close(
        transaction.get("accepted_recovery_secant_curvature_threshold"),
        SECANT_CURVATURE_THRESHOLD,
        "accepted recovery secant curvature threshold",
    )
    mode = transaction.get("accepted_recovery_secant_publication_mode")
    require(mode in SECANT_PUBLICATION_MODES, "unknown recovery secant publication mode")
    certified_count = require_nonnegative_int(
        transaction, "accepted_recovery_secants_certified_curvature"
    )
    rejected_count = require_nonnegative_int(
        transaction, "accepted_recovery_secants_rejected_curvature"
    )
    omitted_count = require_nonnegative_int(
        transaction, "accepted_recovery_secants_omitted_by_rank"
    )
    published_count = require_nonnegative_int(
        transaction, "accepted_recovery_secants_added"
    )
    certified = require_index_list(
        transaction, "accepted_recovery_secant_certified_indices", n_internal
    )
    rejected = require_index_list(
        transaction, "accepted_recovery_secant_rejected_indices", n_internal
    )
    omitted = require_index_list(
        transaction, "accepted_recovery_secant_omitted_indices", n_internal
    )
    published = require_index_list(
        transaction, "accepted_recovery_secant_published_indices", n_internal
    )
    published_curvatures = require_finite_float_list(
        transaction,
        "accepted_recovery_secant_published_normalized_curvatures",
    )
    require(certified_count == len(certified), "certified secant count/list differ")
    require(rejected_count == len(rejected), "curvature-rejected count/list differ")
    require(omitted_count == len(omitted), "rank-omitted count/list differ")
    require(published_count == len(published), "published secant count/list differ")
    require(
        len(published_curvatures) == len(published),
        "published secant index/normalized-curvature lists differ",
    )

    if not recovery_committed:
        require(mode == "not_applicable", "uncommitted recovery has a publication mode")
        require(
            not certified
            and not rejected
            and not omitted
            and not published
            and not published_curvatures,
            "uncommitted recovery logged a secant publication classification",
        )
        return {
            "certified": 0,
            "rejected": 0,
            "omitted": 0,
            "published": 0,
            "max_published_normalized_curvature_error": 0.0,
            "mode": mode,
        }

    require(mode == expected_mode, "accepted recovery secant branch drifted")
    require(mode != "not_applicable", "accepted recovery lacks a publication mode")
    expected = independent_secant_partition(
        direct_path,
        direct_gradient,
        projected_path,
        projected_gradient,
        mode,
    )
    require(certified == expected["certified"], "curvature-certified indices drifted")
    require(rejected == expected["rejected"], "curvature-rejected indices drifted")
    require(omitted == expected["omitted"], "rank-omitted indices drifted")
    require(published == expected["published"], "published secant indices drifted")
    all_indices = set(range(1, int(n_internal) + 1))
    require(set(certified).isdisjoint(rejected), "certified/rejected sets overlap")
    require(set(certified) | set(rejected) == all_indices, "C/R partition does not close")
    require(set(published).isdisjoint(omitted), "published/omitted sets overlap")
    require(set(published) | set(omitted) == set(certified), "P/O partition does not close")
    require(
        published_count + omitted_count + rejected_count == int(n_internal),
        "published/rank-omitted/curvature-rejected counts do not close",
    )
    require(
        certified_count == published_count + omitted_count,
        "certified count does not equal published plus rank-omitted",
    )
    if mode == "strong_top1":
        require(published_count <= 1, "strong branch published more than one secant")
    else:
        require(not omitted, "all-publication branch omitted a certified secant")
    maximum_error = 0.0
    for observed, expected_value in zip(
        published_curvatures,
        expected["published_normalized_curvatures"],
    ):
        require_close(observed, expected_value, "published normalized curvature")
        maximum_error = max(maximum_error, abs(observed - expected_value))
    return {
        "certified": certified_count,
        "rejected": rejected_count,
        "omitted": omitted_count,
        "published": published_count,
        "max_published_normalized_curvature_error": float(maximum_error),
        "mode": mode,
    }


def independent_gate_i_perp(metric, path, energies, gradients):
    """Recompute the optimizer's path-length I_perp without importing its helper."""

    points = np.asarray(path, dtype=np.float64)
    values = np.asarray(energies, dtype=np.float64)
    derivatives = np.asarray(gradients, dtype=np.float64)
    tangents = metric.improved_tangents(points, values)
    forces = -derivatives[1:-1]
    normal = forces - np.sum(forces * tangents, axis=1)[:, None] * tangents
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    weights = 0.5 * (lengths[:-1] + lengths[1:])
    result = float(np.sum(weights * np.sum(normal * normal, axis=1)))
    require(np.isfinite(result) and result >= 0.0, "independent gate I_perp is invalid")
    return result


def independent_equidistance(path):
    """Recompute the fixed path equidistance integral independently."""

    points = np.asarray(path, dtype=np.float64)
    ds = 1.0 / float(points.shape[0] - 1)
    first = (points[2:] - points[:-2]) / (2.0 * ds)
    second = (
        points[2:] - 2.0 * points[1:-1] + points[:-2]
    ) / (ds * ds)
    inner = np.sum(first * second, axis=1)
    result = float(np.sum(inner * inner) * ds)
    require(np.isfinite(result) and result >= 0.0, "independent equidistance is invalid")
    return result


def independent_polyline_equal_arclength(path):
    points = np.asarray(path, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if (
        np.any(~np.isfinite(lengths))
        or np.any(lengths <= np.finfo(np.float64).tiny)
    ):
        return None
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, float(cumulative[-1]), points.shape[0])
    result = np.empty_like(points)
    result[0], result[-1] = points[0], points[-1]
    for image, target in enumerate(targets[1:-1], start=1):
        edge = min(
            int(np.searchsorted(cumulative, target, side="right") - 1),
            lengths.size - 1,
        )
        fraction = (target - cumulative[edge]) / lengths[edge]
        result[image] = points[edge] + fraction * (
            points[edge + 1] - points[edge]
        )
    return result


def independent_improved_spring(metric, path, energies):
    result = np.asarray(path, dtype=np.float64).copy()
    fixed_energies = np.asarray(energies, dtype=np.float64)
    residual = np.inf
    iterations = 0
    for iteration in range(SPRING_MAX_ITER):
        lengths = np.linalg.norm(np.diff(result, axis=0), axis=1)
        tangents = metric.improved_tangents(result, fixed_energies)
        force = (lengths[1:] - lengths[:-1])[:, None] * tangents
        residual = float(np.max(np.linalg.norm(force, axis=1)))
        iterations = iteration + 1
        if residual <= SPRING_TOLERANCE:
            break
        result[1:-1] += SPRING_DTAU * force
    result[0], result[-1] = path[0], path[-1]
    return result, residual, iterations


def independent_chord_constraints_and_jacobian(path):
    points = np.asarray(path, dtype=np.float64)
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    units = segments / np.maximum(
        lengths[:, None], np.finfo(np.float64).tiny
    )
    n_internal = points.shape[0] - 2
    dimension = points.shape[1]
    constraints = lengths[1:] - lengths[:-1]
    jacobian = np.zeros(
        (n_internal, n_internal * dimension), dtype=np.float64
    )
    for row in range(n_internal):
        image = row + 1
        previous = units[image - 1]
        following = units[image]
        if image - 1 >= 1:
            start = (image - 2) * dimension
            jacobian[row, start : start + dimension] += previous
        start = (image - 1) * dimension
        jacobian[row, start : start + dimension] += -following - previous
        if image + 1 <= n_internal:
            start = image * dimension
            jacobian[row, start : start + dimension] += following
    return constraints, jacobian


def independent_tangent_chord_projection(metric, seed, direct_energies, ceiling):
    """Replay chord-Newton projection without importing optimizer helpers."""

    path = np.asarray(seed, dtype=np.float64).copy()
    fixed_energies = np.asarray(direct_energies, dtype=np.float64).copy()
    endpoints = (path[0].copy(), path[-1].copy())
    backtracks = 0
    for iteration in range(SPRING_MAX_ITER + 1):
        constraints, jacobian = independent_chord_constraints_and_jacobian(path)
        residual = float(np.linalg.norm(constraints))
        if independent_equidistance(path) <= (
            float(ceiling) + float(AUDIT_RECOVERY_EQUIDISTANCE_ATOL)
        ):
            return path, residual, iteration, backtracks, True
        if iteration == SPRING_MAX_ITER:
            break
        tangents = metric.improved_tangents(path, fixed_energies)
        n_internal = path.shape[0] - 2
        dimension = path.shape[1]
        basis = np.zeros(
            (n_internal * dimension, n_internal), dtype=np.float64
        )
        for index, tangent in enumerate(tangents):
            start = index * dimension
            basis[start : start + dimension, index] = tangent
        coefficients = np.linalg.lstsq(
            jacobian @ basis, -constraints, rcond=None
        )[0]
        correction = (basis @ coefficients).reshape(n_internal, dimension)
        if not np.all(np.isfinite(correction)):
            break
        accepted = False
        fraction = 1.0
        for _ in range(64):
            trial = path.copy()
            trial[1:-1] += fraction * correction
            trial[0], trial[-1] = endpoints
            trial_constraints, _ = independent_chord_constraints_and_jacobian(
                trial
            )
            if np.linalg.norm(trial_constraints) < residual:
                path = trial
                accepted = True
                break
            fraction *= BACKTRACK
            backtracks += 1
        if not accepted:
            break
    constraints, _ = independent_chord_constraints_and_jacobian(path)
    feasible = bool(
        independent_equidistance(path)
        <= float(ceiling) + float(AUDIT_RECOVERY_EQUIDISTANCE_ATOL)
    )
    return (
        path,
        float(np.linalg.norm(constraints)),
        min(SPRING_MAX_ITER, iteration),
        backtracks,
        feasible,
    )


def audit_projection_replay(metric, transaction, direct_energies, ceiling):
    """Authenticate the zero-NoF chord projection from its public inputs."""

    raw_constructed = bool(transaction.get("raw_proposal_constructed"))
    if not raw_constructed:
        require(
            transaction.get("recovery_projection_seed_path") is None
            and transaction.get("projected_attempt_path") is None
            and transaction.get("recovery_projection_residual") is None,
            "unconstructed raw proposal contains projection state",
        )
        require(
            type(transaction.get("recovery_projection_iterations")) is int
            and int(transaction["recovery_projection_iterations"]) == 0
            and type(transaction.get("recovery_projection_backtracks")) is int
            and int(transaction["recovery_projection_backtracks"]) == 0
            and transaction.get("recovery_projection_feasible") is False,
            "unconstructed raw proposal contains a projection result",
        )
        return {
            "path": None,
            "residual": None,
            "iterations": 0,
            "backtracks": 0,
            "feasible": False,
        }

    seed = np.asarray(
        transaction.get("recovery_projection_seed_path"), dtype=np.float64
    )
    expected_shape = np.asarray(direct_energies, dtype=np.float64).shape
    require(
        seed.ndim == 2
        and seed.shape[0] == expected_shape[0]
        and seed.shape[1] == 10
        and np.all(np.isfinite(seed)),
        "projection seed shape or finiteness drifted",
    )
    (
        expected_path,
        expected_residual,
        expected_iterations,
        expected_backtracks,
        expected_feasible,
    ) = independent_tangent_chord_projection(
        metric,
        seed,
        np.asarray(direct_energies, dtype=np.float64),
        float(ceiling),
    )
    attempt = np.asarray(
        transaction.get("projected_attempt_path"), dtype=np.float64
    )
    require(
        attempt.shape == expected_path.shape
        and np.array_equal(attempt, expected_path),
        "projected attempt path differs from independent chord-Newton replay",
    )
    require_close(
        transaction.get("recovery_projection_residual"),
        expected_residual,
        "recovery projection residual",
    )
    require(
        type(transaction.get("recovery_projection_iterations")) is int
        and int(transaction["recovery_projection_iterations"])
        == int(expected_iterations),
        "recovery projection iteration count drifted",
    )
    require(
        type(transaction.get("recovery_projection_backtracks")) is int
        and int(transaction["recovery_projection_backtracks"])
        == int(expected_backtracks),
        "recovery projection backtrack count drifted",
    )
    require(
        type(transaction.get("recovery_projection_feasible")) is bool
        and bool(transaction["recovery_projection_feasible"])
        == bool(expected_feasible),
        "recovery projection feasibility drifted",
    )
    return {
        "path": expected_path,
        "residual": expected_residual,
        "iterations": expected_iterations,
        "backtracks": expected_backtracks,
        "feasible": expected_feasible,
    }


def independent_recovery_trust_radius(path):
    """Recompute the frozen QM10 recovery trust radius algebraically."""

    points = np.asarray(path, dtype=np.float64)
    require(
        points.ndim == 2 and points.shape[1] == 10,
        "recovery trust-radius path shape drifted",
    )
    segment_rms = np.linalg.norm(np.diff(points, axis=0), axis=1) / np.sqrt(
        10.0
    )
    return min(2.0 * float(np.median(segment_rms)), 1.0)


def audit_raw_proposal_algebra(
    transaction,
    post_main_path,
    direct_path,
    direct_energies,
    direct_gradients,
    alpha,
    m_max,
    retry_state,
):
    """Authenticate raw displacement algebra and the same-anchor retry chain."""

    direct = np.asarray(direct_path, dtype=np.float64)
    energies = np.asarray(direct_energies, dtype=np.float64)
    gradients = np.asarray(direct_gradients, dtype=np.float64)
    require(
        direct.ndim == 2
        and gradients.shape == direct.shape
        and energies.shape == (direct.shape[0],),
        "raw-proposal direct anchor shape drifted",
    )
    expected_trust = independent_recovery_trust_radius(post_main_path)
    require_close(
        transaction.get("total_atomic_trust_radius"),
        expected_trust,
        "recovery total atomic trust radius",
    )

    state_used = bool(transaction.get("recovery_backtrack_state_used"))
    require(
        bool(transaction.get("private_recovery_backtrack_state_entering"))
        == state_used,
        "private retry-state entry/use flags differ",
    )
    if state_used:
        require(retry_state is not None, "same-anchor retry lacks prior base state")
        require(
            np.array_equal(direct, retry_state["path"])
            and np.array_equal(energies, retry_state["energies"])
            and np.array_equal(gradients, retry_state["gradients"]),
            "same-anchor retry changed its direct evaluated anchor",
        )
        require(
            transaction.get("recovery_backtrack_anchor_matched") is True,
            "same-anchor retry was not logged as an exact anchor match",
        )
    else:
        retry_state = None

    raw_constructed = bool(transaction.get("raw_proposal_constructed"))
    if not raw_constructed:
        require(
            transaction.get("raw_candidate_path") is None
            and transaction.get("raw_pre_addition_direction") is None,
            "unconstructed raw proposal contains a raw path or direction",
        )
        require(
            not bool(transaction.get("recovery_backtrack_carry_allowed")),
            "unconstructed raw proposal retained a retry state",
        )
        return None

    candidates = transaction.get("direction_candidates")
    require(
        isinstance(candidates, list) and len(candidates) == 1,
        "raw proposal lacks its unique direction-candidate record",
    )
    candidate = candidates[0]
    require(isinstance(candidate, dict), "raw direction candidate is not an object")
    raw = np.asarray(transaction.get("raw_candidate_path"), dtype=np.float64)
    require(
        raw.shape == direct.shape and np.all(np.isfinite(raw)),
        "raw proposal path shape or finiteness drifted",
    )
    require(
        np.array_equal(raw[[0, -1]], direct[[0, -1]]),
        "raw proposal changed a fixed endpoint",
    )
    logged_direction = np.asarray(
        transaction.get("raw_pre_addition_direction"), dtype=np.float64
    )
    require(
        logged_direction.shape == direct.shape
        and np.all(np.isfinite(logged_direction)),
        "raw pre-addition direction shape or finiteness drifted",
    )
    require(
        np.array_equal(
            logged_direction[[0, -1]],
            np.zeros_like(logged_direction[[0, -1]]),
        ),
        "raw pre-addition direction moves an endpoint",
    )
    step = float(transaction.get("regularized_step_length"))
    predicted = float(transaction.get("predicted_regularized_decrease"))
    fraction = float(transaction.get("recovery_backtrack_fraction"))
    require(
        np.isfinite(step) and step > 0.0 and np.isfinite(predicted) and predicted > 0.0,
        "constructed raw proposal has non-positive step or prediction",
    )
    for field, expected in (
        ("regularized_step_length", step),
        ("predicted_regularized_decrease", predicted),
        ("backtrack_fraction", fraction),
    ):
        require_close(candidate.get(field), expected, "direction candidate " + field)
    require(
        int(candidate.get("warm_pair_count", -1))
        == int(transaction.get("warm_pair_count", -2)),
        "direction candidate warm-pair count drifted",
    )
    recomputed_g_dot_d = 0.0
    recomputed_norm_squared = 0.0
    for image_index in range(1, int(direct.shape[0]) - 1):
        recomputed_g_dot_d += float(
            np.dot(gradients[image_index], logged_direction[image_index])
        )
        recomputed_norm_squared += float(
            np.dot(logged_direction[image_index], logged_direction[image_index])
        )
    candidate_g_dot_d = float(candidate.get("g_dot_d"))
    candidate_norm_squared = float(candidate.get("norm_squared"))
    require(
        candidate_g_dot_d == recomputed_g_dot_d
        and candidate_norm_squared == recomputed_norm_squared,
        "raw direction candidate scalars differ from its logged direction",
    )
    require(
        np.isfinite(candidate_g_dot_d) and candidate_g_dot_d < 0.0,
        "raw direction candidate is not a logged descent direction",
    )
    require(
        np.isfinite(candidate_norm_squared)
        and candidate_norm_squared > np.finfo(np.float64).tiny,
        "raw direction candidate has a non-positive or tiny logged norm",
    )
    expected_base_step = float(
        np.clip(
            -float(alpha) * candidate_g_dot_d / candidate_norm_squared,
            0.0,
            1.0,
        )
    )
    candidate_base_step = float(candidate.get("base_regularized_step_length"))
    require(
        candidate_base_step == expected_base_step,
        "direction candidate base step differs from its logged g dot d and norm",
    )

    if state_used:
        before = int(transaction.get("same_anchor_recovery_attempt_count_before", -1))
        require(
            before == int(retry_state["completed_attempts"]) and before >= 1,
            "same-anchor retry attempt count does not continue its base state",
        )
        expected_fraction = float(0.5**before)
        require_close(fraction, expected_fraction, "same-anchor retry fraction")
        require(
            candidate_g_dot_d == float(retry_state["g_dot_d"])
            and candidate_norm_squared == float(retry_state["norm_squared"]),
            "same-anchor retry changed its logged direction algebra",
        )
        require(
            np.array_equal(logged_direction, retry_state["logged_direction"]),
            "same-anchor retry changed its logged pre-addition direction",
        )
        require(
            candidate_base_step == float(retry_state["base_step"]),
            "same-anchor retry changed its base step",
        )
        expected_step = float(expected_base_step * expected_fraction)
        require(
            step == expected_step,
            "same-anchor retry step differs from its logged direction algebra",
        )
        expected_raw = direct + expected_fraction * np.asarray(
            retry_state["base_displacement"], dtype=np.float64
        )
        expected_raw[0], expected_raw[-1] = direct[0], direct[-1]
        require(
            np.array_equal(raw, expected_raw),
            "same-anchor raw path is not the exact base displacement fraction",
        )
        direction = np.asarray(retry_state["base_displacement"], dtype=np.float64) / float(
            retry_state["base_step"]
        )
        base_displacement = np.asarray(
            retry_state["base_displacement"], dtype=np.float64
        ).copy()
        base_step = float(retry_state["base_step"])
        require(
            candidate.get("name") == "same_anchor_backtrack"
            and transaction.get("selected_direction") == "same_anchor_backtrack",
            "retry did not select the registered same-anchor direction",
        )
    else:
        require_close(fraction, 1.0, "first recovery fraction")
        expected_step = expected_base_step
        require(
            step == expected_step and candidate_base_step == step,
            "first recovery step differs from its logged direction algebra",
        )
        require(
            candidate.get("name") == "accepted_history"
            and transaction.get("selected_direction") == "accepted_history",
            "first recovery did not select accepted history",
        )
        require(
            int(transaction.get("same_anchor_recovery_attempt_count_before", -1))
            == 0,
            "first recovery entered with a nonzero retry count",
        )
        base_displacement = raw - direct
        base_step = step
        direction = base_displacement / base_step
        expected_raw = direct + step * logged_direction
        expected_raw[0], expected_raw[-1] = direct[0], direct[-1]
        require(
            np.array_equal(raw, expected_raw),
            "first raw path differs from its logged pre-addition direction",
        )

    require(
        np.array_equal(direction[[0, -1]], np.zeros_like(direction[[0, -1]])),
        "raw recovery direction moves an endpoint",
    )
    require(
        np.all(
            np.max(np.abs(logged_direction[1:-1]), axis=1)
            <= expected_trust + 1.0e-12
        ),
        "logged recovery direction exceeds the independently reconstructed trust radius",
    )
    expected_predicted = float(
        -step * candidate_g_dot_d
        - step * step * candidate_norm_squared / (2.0 * float(alpha))
    )
    require(
        predicted == expected_predicted,
        "raw regularized prediction differs from logged direction algebra",
    )

    completed = bool(
        transaction.get("projected_trial_evaluated")
        and not transaction.get("interrupted")
    )
    before = int(transaction.get("same_anchor_recovery_attempt_count_before", -1))
    expected_after = before + 1 if completed else before
    require(
        int(transaction.get("same_anchor_recovery_attempt_count_after", -1))
        == expected_after,
        "same-anchor recovery attempt count does not close",
    )
    carry = bool(transaction.get("recovery_backtrack_carry_allowed"))
    require(
        not carry or (completed and expected_after < int(m_max)),
        "retry state survived without a completed sub-cap recovery attempt",
    )
    if not carry:
        return None
    return {
        "path": direct.copy(),
        "energies": energies.copy(),
        "gradients": gradients.copy(),
        "base_displacement": base_displacement.copy(),
        "base_step": base_step,
        "g_dot_d": candidate_g_dot_d,
        "norm_squared": candidate_norm_squared,
        "logged_direction": logged_direction.copy(),
        "completed_attempts": expected_after,
    }


def audit_chord_coherent_seed(metric, control, transaction, direct_energy):
    """Reconstruct the zero-NoF seed without importing optimizer helpers."""

    raw = np.asarray(transaction.get("raw_candidate_path"), dtype=np.float64)
    seed = np.asarray(
        transaction.get("recovery_projection_seed_path"), dtype=np.float64
    )
    expected_shape = (int(control.n_internal) + 2, 10)
    require(raw.shape == expected_shape, "raw recovery path shape drifted")
    require(seed.shape == expected_shape, "recovery seed path shape drifted")
    require(np.all(np.isfinite(raw)), "raw recovery path is non-finite")
    require(np.all(np.isfinite(seed)), "recovery seed path is non-finite")
    require(
        np.array_equal(raw[[0, -1]], seed[[0, -1]]),
        "recovery seed changed an endpoint",
    )
    stationary = bool(transaction.get("stationary_normal_probe"))
    if stationary:
        require(
            transaction.get("recovery_seed_mode")
            == "ordinary_normal_seed_for_stationarity_exhaustion",
            "stationary recovery did not use the ordinary normal seed",
        )
        require(np.array_equal(seed, raw), "stationary seed differs from raw path")
        require(
            transaction.get("adjacent_chord_cosine") is None
            and transaction.get("coherence_loss") is None
            and transaction.get("coherence_loss_limit") is None
            and transaction.get("amplified_direction_mismatch") is None
            and transaction.get("saddle_spring_weight") is None,
            "stationary probe improperly applied the chord gate",
        )
        require(
            transaction.get("saddle_image") is None
            and transaction.get("strong_recovery_selected") is False
            and transaction.get("seed_spring_residual") is None
            and type(transaction.get("seed_spring_iterations")) is int
            and int(transaction.get("seed_spring_iterations")) == 0,
            "stationary probe logged a non-ordinary seed branch",
        )
        return {"stationary": True, "strong": False, "weight": None}

    spring, residual, iterations = independent_improved_spring(
        metric, raw, direct_energy
    )
    saddle = int(np.argmax(np.asarray(direct_energy)[1:-1])) + 1
    horizon = float(int(control.m_max) - 1)
    polyline = independent_polyline_equal_arclength(raw)
    if polyline is None:
        require(
            transaction.get("recovery_seed_mode")
            == "spring_fallback_degenerate_polyline",
            "degenerate polyline did not select the registered spring fallback",
        )
        require(
            np.array_equal(seed, spring),
            "stored degenerate-polyline spring fallback is not reproducible",
        )
        require(
            int(transaction.get("saddle_image")) == saddle,
            "fallback saddle image drifted",
        )
        require(
            transaction.get("adjacent_chord_cosine") is None
            and transaction.get("coherence_loss") is None
            and transaction.get("amplified_direction_mismatch") is None,
            "degenerate-polyline fallback logged an inapplicable chord score",
        )
        require_close(
            transaction.get("coherence_loss_limit"),
            1.0 / horizon**2,
            "fallback coherence loss limit",
        )
        require(
            transaction.get("strong_recovery_selected") is False,
            "degenerate-polyline fallback selected strong recovery",
        )
        require_close(
            transaction.get("saddle_spring_weight"),
            1.0,
            "fallback saddle spring weight",
        )
        require_close(
            transaction.get("seed_spring_residual"),
            residual,
            "fallback seed spring residual",
        )
        require(
            type(transaction.get("seed_spring_iterations")) is int
            and int(transaction.get("seed_spring_iterations"))
            == int(iterations),
            "fallback seed spring iteration count drifted",
        )
        return {
            "stationary": False,
            "strong": False,
            "weight": 1.0,
            "degenerate_polyline_fallback": True,
        }

    require(
        transaction.get("recovery_seed_mode")
        == "polyline_spring_saddle_adjacent_chord_coherence",
        "nondegenerate polyline did not select the registered chord branch",
    )
    left = polyline[saddle] - polyline[saddle - 1]
    right = polyline[saddle + 1] - polyline[saddle]
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    denominator = float(left_norm * right_norm)
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).tiny:
        cosine = -1.0
    else:
        cosine = float(
            np.clip(np.dot(left, right) / denominator, -1.0, 1.0)
        )
    loss = float(1.0 - cosine)
    amplified = float(horizon * np.sqrt(max(0.0, loss)))
    strong = bool(amplified <= 1.0)
    weight = float(horizon if strong else 1.0)
    expected_seed = polyline.copy()
    expected_seed[saddle] = polyline[saddle] + weight * (
        spring[saddle] - polyline[saddle]
    )
    expected_seed[0], expected_seed[-1] = raw[0], raw[-1]
    require(
        np.array_equal(seed, expected_seed),
        "stored chord-coherent recovery seed is not independently reproducible",
    )
    require(int(transaction.get("saddle_image")) == saddle, "saddle image drifted")
    require_close(
        transaction.get("adjacent_chord_cosine"), cosine, "adjacent chord cosine"
    )
    require_close(transaction.get("coherence_loss"), loss, "coherence loss")
    require_close(
        transaction.get("coherence_loss_limit"),
        1.0 / horizon**2,
        "coherence loss limit",
    )
    require_close(
        transaction.get("amplified_direction_mismatch"),
        amplified,
        "amplified direction mismatch",
    )
    require(
        bool(transaction.get("strong_recovery_selected")) == strong,
        "strong recovery decision drifted",
    )
    require_close(
        transaction.get("saddle_spring_weight"), weight, "saddle spring weight"
    )
    require_close(
        transaction.get("seed_spring_residual"), residual, "seed spring residual"
    )
    require(
        int(transaction.get("seed_spring_iterations")) == int(iterations),
        "seed spring iteration count drifted",
    )
    return {
        "stationary": False,
        "strong": strong,
        "weight": weight,
        "degenerate_polyline_fallback": False,
    }


def load_article(metric, n_internal, registration):
    source = Path(ARTICLE_FILES[int(n_internal)])
    expected_sha = EXPECTED_ARTICLE_SHA256[int(n_internal)]
    require(source.is_file(), "missing frozen Article IVM: " + str(source))
    observed_sha = sha256(source)
    require(observed_sha == expected_sha, "frozen Article IVM hash drifted")
    dependencies = registration.get("dependency_sha256", {})
    require(
        dependencies.get(str(source)) == expected_sha,
        "preregistration does not freeze the expected Article IVM",
    )
    with np.load(source, allow_pickle=False) as saved:
        required = {
            "q_matrix",
            "q_sha256",
            "transformed_initial_path",
            "path_history",
            "total_force_evaluations",
        }
        require(required.issubset(saved.files), "Article IVM archive lacks required fields")
        q_matrix = np.asarray(saved["q_matrix"], dtype=np.float64)
        initial_path = np.asarray(saved["transformed_initial_path"], dtype=np.float64)
        paths = np.asarray(saved["path_history"], dtype=np.float64)
        nof = np.asarray(saved["total_force_evaluations"], dtype=np.int64)
        stored_q_sha = str(np.asarray(saved["q_sha256"]).item())
    expected_shape = (paths.shape[0], int(n_internal) + 2, 10)
    require(paths.shape == expected_shape, "Article IVM path shape drifted")
    require(initial_path.shape == expected_shape[1:], "Article initial path shape drifted")
    require(q_matrix.shape == (10, 10), "Article Q shape drifted")
    require(np.all(np.isfinite(paths)), "Article path contains non-finite values")
    require(np.all(np.isfinite(initial_path)), "Article initial path is non-finite")
    require(np.all(np.isfinite(q_matrix)), "Article Q is non-finite")
    require(nof.shape == (paths.shape[0],), "Article NoF shape drifted")
    require(np.all(np.diff(nof) > 0), "Article NoF is not strictly increasing")
    require(int(nof[0]) == int(n_internal), "Article initial NoF is not N_internal")
    require(
        int(nof[-1]) == EXPECTED_ARTICLE_TERMINAL_NF[int(n_internal)],
        "Article terminal NoF drifted",
    )
    q_sha = hashlib.sha256(np.ascontiguousarray(q_matrix).tobytes()).hexdigest()
    require(q_sha == stored_q_sha, "Article Q hash field drifted")
    checkpoints = latter_half_checkpoints(nof)
    expected_range = {
        10: (481, 951),
        20: (1199, 2377),
    }[int(n_internal)]
    require(
        (int(checkpoints[0]), int(checkpoints[-1])) == expected_range,
        "Article latter-half checkpoint range drifted",
    )
    strict = registration.get("strict_acceptance", {})
    registered_checkpoints = strict.get("checkpoints", {}).get(str(n_internal), {})
    require(
        int(registered_checkpoints.get("count", -1)) == CHECKPOINT_COUNT
        and tuple(registered_checkpoints.get("range", ())) == expected_range,
        "preregistered strict checkpoints drifted for N={}".format(n_internal),
    )
    indices = locf_indices(nof, checkpoints)
    unique, inverse = np.unique(indices, return_inverse=True)
    unique_i, unique_delta, unique_error = metric_values(
        metric, paths[unique], q_matrix
    )
    return {
        "source": str(source),
        "source_sha256": observed_sha,
        "q_matrix": q_matrix,
        "initial_path": initial_path,
        "paths": paths,
        "nof": nof,
        "checkpoints": checkpoints,
        "checkpoint_indices": indices,
        "checkpoint_i_perp": unique_i[inverse],
        "checkpoint_delta_e": unique_delta[inverse],
        "checkpoint_delta_e_error": unique_error[inverse],
    }


def validate_low_r_registration_contract(registration):
    """Independently pin the new campaign identity, paths, grid, and policy."""

    require(EXPERIMENT_ID == EXPECTED_EXPERIMENT_ID, "core campaign ID drifted")
    require(
        registration.get("campaign_id") == EXPECTED_EXPERIMENT_ID,
        "registration campaign ID drifted",
    )
    require(
        registration.get("registered_before_first_optimization_run") is True,
        "campaign was not registered before optimization",
    )
    require(
        tuple(tuple(item) for item in REGISTERED_CONTROLS)
        == EXPECTED_REGISTERED_CONTROLS,
        "core low-r registered controls drifted",
    )
    require(registration.get("resolutions") == [10, 20], "registered resolutions drifted")
    require(
        int(registration.get("registered_member_count", -1)) == 8,
        "registered low-r member count drifted",
    )
    require(
        registration.get("shared_grid_for_both_resolutions")
        == {
            "alpha": [1.0e-3, 2.0e-3],
            "alpha_labels": ["1e-3", "2e-3"],
            "r": [0.60, 0.65],
            "m_max": [4],
        },
        "registered low-r Cartesian grid drifted",
    )
    require(
        all(float(item[2]) < 0.7 for item in REGISTERED_CONTROLS),
        "registered control violates r < 0.7",
    )
    execution = registration.get("execution", {})
    for name, expected in {
        "n10_core_gate_manifest": (
            "reports/n20_gate_audit/audit_manifest.json"
        ),
        "n10_gate_token_is_current_manifest_sha256": True,
        "n20_gate_token": "not_applicable",
    }.items():
        require(
            execution.get(name) == expected,
            "registered staged-resolution gate drifted: " + name,
        )
    frozen = registration.get("frozen_sources", {})
    require(
        str(Path(frozen.get("optimizer", "")).resolve())
        == str(Path(CORE_SOURCE).resolve()),
        "registration optimizer path differs from this campaign core",
    )
    require(
        str(Path(frozen.get("independent_audit", "")).resolve())
        == str(SCRIPT_FILE.resolve()),
        "registration audit path differs from this script",
    )
    require(
        Path(REGISTRATION).resolve()
        == (CAMPAIGN_ROOT / "registrations/CAMPAIGN_PREREGISTRATION.json").resolve(),
        "registration path is outside the new campaign",
    )
    dependencies = registration.get("dependency_sha256", {})
    require(
        dependencies.get(str(SCRIPT_FILE)) == sha256(SCRIPT_FILE),
        "registration does not freeze this independent audit",
    )
    require(
        registration.get("implementation_sha256") == sha256(CORE_SOURCE),
        "registration does not freeze the new campaign core",
    )
    recovery_contract = registration.get("projected_recovery_contract", {})
    expected_contract = {
        "accepted_secant_curvature_threshold": SECANT_CURVATURE_THRESHOLD,
        "normalized_curvature_formula": SECANT_NORMALIZED_CURVATURE_FORMULA,
        "strong_secant_publication": SECANT_STRONG_PUBLICATION,
        "conservative_secant_publication": SECANT_ALL_PUBLICATION,
        "ordinary_secant_publication": SECANT_ALL_PUBLICATION,
        "secant_tie_break": SECANT_TIE_BREAK,
        "secant_index_convention": "1-based internal path image index",
        "rank_omitted_is_distinct_from_curvature_rejected": True,
        "publication_only_after_complete_atomic_commit": True,
    }
    for name, expected in expected_contract.items():
        require(
            recovery_contract.get(name) == expected,
            "registered recovery secant contract drifted: " + name,
        )


def expected_result_path(control):
    return CAMPAIGN_ROOT / "runs" / "n{}".format(control.n_internal) / (
        control.run_id + ".npz"
    )


def expected_receipt_path(control):
    return CAMPAIGN_ROOT / "reports/receipts" / "n{}".format(
        control.n_internal
    ) / (control.run_id + ".receipt.json")


def exact_csv_bool(value, label):
    if value == "True":
        return True
    if value == "False":
        return False
    raise IntegrityError("N20 gate CSV has invalid boolean: " + label)


def validate_n20_gate_manifest_for_n10(registration):
    """Authenticate the staged N20 decision without importing the runner."""

    require(
        Path(N20_GATE_MANIFEST).resolve()
        == EXPECTED_N20_GATE_MANIFEST.resolve(),
        "core N20 gate path drifted",
    )
    path = EXPECTED_N20_GATE_MANIFEST
    require(path.is_file(), "N10 requires the fixed N20 audit manifest")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise IntegrityError("N20 gate manifest is invalid JSON") from error
    require(isinstance(manifest, dict), "N20 gate manifest is not an object")
    require_finite_json(manifest, "n20_gate_manifest")
    controls = [
        Controls(int(item[0]), float(item[1]), float(item[2]), int(item[3]))
        for item in REGISTERED_CONTROLS
        if int(item[0]) == 20
    ]
    expected_count = len(controls)
    resolution = manifest.get("resolution_results", {}).get("20", {})
    required = {
        "campaign_id": manifest.get("campaign_id") == EXPECTED_EXPERIMENT_ID,
        "audit_status": manifest.get("audit_status") == "complete",
        "requested_resolutions": manifest.get("requested_resolutions") == [20],
        "registration": manifest.get("registration") == str(REGISTRATION),
        "registration_sha256": (
            manifest.get("registration_sha256") == sha256(REGISTRATION)
        ),
        "core_source": manifest.get("core_source") == str(CORE_SOURCE),
        "core_source_sha256": (
            manifest.get("core_source_sha256") == sha256(CORE_SOURCE)
        ),
        "audit_script": manifest.get("audit_script") == str(SCRIPT_FILE),
        "audit_script_sha256": (
            manifest.get("audit_script_sha256") == sha256(SCRIPT_FILE)
        ),
        "registration_campaign": (
            registration.get("campaign_id") == EXPECTED_EXPERIMENT_ID
        ),
        "registration_core": (
            registration.get("implementation_sha256") == sha256(CORE_SOURCE)
        ),
        "requested_member_count": (
            int(manifest.get("requested_member_count", -1)) == expected_count
        ),
        "audited_member_count": (
            int(manifest.get("audited_member_count", -1)) == expected_count
        ),
        "integrity_valid_member_count": (
            int(manifest.get("integrity_valid_member_count", -1))
            == expected_count
        ),
        "integrity_failure_count": (
            int(manifest.get("integrity_failure_count", -1)) == 0
        ),
        "integrity_errors": manifest.get("integrity_errors") == [],
        "resolution_registered": (
            int(resolution.get("registered_member_count", -1))
            == expected_count
        ),
        "resolution_integrity": (
            int(resolution.get("integrity_valid_member_count", -1))
            == expected_count
        ),
        "resolution_selectable": (
            int(resolution.get("selectable_member_count", -1))
            == expected_count
        ),
        "resolution_strict_pass": (
            int(resolution.get("strict_pass_count", 0)) > 0
        ),
        "all_requested_resolutions_strictly_pass": (
            manifest.get("all_requested_resolutions_strictly_pass") is True
        ),
    }
    failed = sorted(name for name, valid in required.items() if not valid)
    require(not failed, "N20 gate manifest failed: " + ", ".join(failed))

    expected_artifacts = {
        "candidate_summary_csv": "candidate_summary.csv",
        "checkpoint_comparison_csv": "checkpoint_comparison.csv",
        "human_report": "AUDIT_REPORT.md",
    }
    artifacts = manifest.get("artifacts")
    require(
        isinstance(artifacts, dict)
        and set(artifacts) == set(expected_artifacts),
        "N20 gate artifact set drifted",
    )
    authenticated = {}
    for name, filename in expected_artifacts.items():
        record = artifacts.get(name)
        require(isinstance(record, dict), "N20 gate artifact record drifted")
        artifact = Path(record.get("path", ""))
        expected = path.parent / filename
        require(
            artifact.is_file()
            and artifact.resolve() == expected.resolve()
            and record.get("sha256") == sha256(artifact),
            "N20 gate artifact authentication failed: " + name,
        )
        authenticated[name] = artifact

    with authenticated["candidate_summary_csv"].open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    expected_controls = {control.run_id: control for control in controls}
    observed_run_ids = [row.get("run_id", "") for row in rows]
    require(
        len(rows) == expected_count
        and len(set(observed_run_ids)) == expected_count
        and set(observed_run_ids) == set(expected_controls),
        "N20 gate summary member identity drifted",
    )

    strict_pass_count = 0
    expected_core_sha = sha256(CORE_SOURCE)
    expected_registration_sha = sha256(REGISTRATION)
    for row in rows:
        run_id = row["run_id"]
        control = expected_controls[run_id]
        try:
            row_identity = bool(
                int(row.get("n_internal", -1)) == 20
                and float(row.get("alpha", "nan")) == float(control.alpha)
                and row.get("alpha_label") == control.alpha_label
                and float(row.get("r", "nan")) == float(control.r)
                and int(row.get("m_max", -1)) == int(control.m_max)
            )
        except (TypeError, ValueError):
            row_identity = False
        require(row_identity, "N20 gate summary control drifted: " + run_id)
        require(
            exact_csv_bool(row.get("integrity_valid"), "integrity_valid")
            and row.get("integrity_error", "") == ""
            and exact_csv_bool(
                row.get("valid_for_selection"), "valid_for_selection"
            )
            and row.get("failure_message", "") == "",
            "N20 gate summary member is not selectable: " + run_id,
        )
        strict_pass_count += int(
            exact_csv_bool(row.get("strict_pass"), "strict_pass")
        )
        source = Path(row.get("source", ""))
        receipt_path = Path(row.get("receipt", ""))
        require(
            source.is_file()
            and source.resolve() == expected_result_path(control).resolve()
            and row.get("source_sha256") == sha256(source)
            and receipt_path.is_file()
            and receipt_path.resolve() == expected_receipt_path(control).resolve()
            and row.get("receipt_sha256") == sha256(receipt_path),
            "N20 gate member authentication failed: " + run_id,
        )
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise IntegrityError("N20 gate receipt is invalid: " + run_id) from error
        require_finite_json(receipt, "n20_gate_receipt." + run_id)
        source_sha = sha256(source)
        require(
            receipt.get("run_id") == run_id
            and receipt.get("controls") == control.as_dict()
            and str(Path(receipt.get("output", "")).resolve())
            == str(source.resolve())
            and receipt.get("output_sha256") == source_sha
            and receipt.get("status") in ("completed", "existing_validated")
            and receipt.get("n20_gate_manifest_sha256_token")
            == "not_applicable",
            "N20 gate receipt identity drifted: " + run_id,
        )
        with np.load(source, allow_pickle=False) as archive:
            require(
                str(scalar(archive, "experiment_id")) == EXPECTED_EXPERIMENT_ID
                and str(scalar(archive, "run_id")) == run_id
                and str(scalar(archive, "source_sha256")) == expected_core_sha
                and str(scalar(archive, "registration_sha256"))
                == expected_registration_sha
                and str(
                    scalar(archive, "n20_gate_manifest_sha256_token")
                )
                == "not_applicable",
                "N20 gate NPZ identity drifted: " + run_id,
            )
            try:
                config = json.loads(str(scalar(archive, "config_json")))
            except Exception as error:
                raise IntegrityError(
                    "N20 gate NPZ config is invalid: " + run_id
                ) from error
            terminal = int(scalar(archive, "terminal_article_nf"))
            last_complete = int(
                scalar(archive, "last_complete_article_nf")
            )
            stop_reason = str(scalar(archive, "stop_reason"))
            failure_message = str(scalar(archive, "failure_message"))
            per_image = np.asarray(
                archive["per_image_joint_evaluations"], dtype=np.int64
            )
            article_nf = np.asarray(archive["article_nf"], dtype=np.int64)
            usable = np.asarray(archive["article_usable_state"], dtype=bool)
        require(
            config == control.as_dict()
            and per_image.shape == (22,)
            and np.all(per_image >= 0)
            and int(per_image[0]) == 0
            and int(per_image[-1]) == 0
            and int(np.sum(per_image)) == terminal
            and article_nf.ndim == 1
            and article_nf.size > 0
            and np.all(np.diff(article_nf) > 0)
            and int(article_nf[0]) == 20
            and int(article_nf[-1]) == last_complete == terminal
            and 20 <= terminal <= ARTICLE_NF_BUDGET
            and usable.shape == article_nf.shape
            and np.all(usable),
            "N20 gate Number-of-Forces ledger drifted: " + run_id,
        )
        require(
            failure_message == ""
            and stop_reason in ELIGIBLE_STOP_REASONS
            and receipt.get("stop_reason") == stop_reason
            and receipt.get("failure_message", "") == failure_message
            and int(receipt.get("terminal_article_nf", -1)) == terminal
            and int(receipt.get("last_complete_article_nf", -1))
            == last_complete
            and row.get("stop_reason") == stop_reason
            and int(row.get("terminal_number_of_forces", -1)) == terminal
            and int(row.get("last_complete_number_of_forces", -1))
            == last_complete,
            "N20 gate member completion identity drifted: " + run_id,
        )

    require(
        strict_pass_count > 0
        and strict_pass_count == int(resolution.get("strict_pass_count", -1)),
        "N20 gate strict-pass count drifted",
    )
    return sha256(path), manifest


def verify_stored_energy_gradient(metric, paths, energies, gradients, q_matrix, label):
    max_energy_error = 0.0
    max_gradient_error = 0.0
    for state_index, state_path in enumerate(paths):
        for image_index, point in enumerate(state_path):
            energy, gradient = metric.energy_gradient(point, q=q_matrix)
            max_energy_error = max(
                max_energy_error,
                abs(float(energy) - float(energies[state_index, image_index])),
            )
            max_gradient_error = max(
                max_gradient_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(gradient, dtype=np.float64)
                            - gradients[state_index, image_index]
                        )
                    )
                ),
            )
    require(
        max_energy_error <= ENERGY_GRADIENT_ATOL,
        label + " stored energies fail independent recomputation",
    )
    require(
        max_gradient_error <= ENERGY_GRADIENT_ATOL,
        label + " stored gradients fail independent recomputation",
    )
    return max_energy_error, max_gradient_error


def require_state_chain_array(record, name, expected_shape, label):
    """Load one finite fixed-shape state-chain array fail-closed."""

    require(name in record, label + " lacks state-chain field: " + name)
    try:
        value = np.asarray(record[name], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise IntegrityError(
            label + " state-chain field is not numeric: " + name
        ) from error
    require(value.shape == tuple(expected_shape), name + " shape drifted")
    require(np.all(np.isfinite(value)), name + " contains a non-finite value")
    return value


def independently_evaluate_path(metric, path, q_matrix):
    points = np.asarray(path, dtype=np.float64)
    energies = np.empty(points.shape[0], dtype=np.float64)
    gradients = np.empty_like(points)
    for index, point in enumerate(points):
        energy, gradient = metric.energy_gradient(point, q=q_matrix)
        energies[index] = float(energy)
        gradients[index] = np.asarray(gradient, dtype=np.float64)
    require(
        np.all(np.isfinite(energies)) and np.all(np.isfinite(gradients)),
        "independent state-chain evaluation is non-finite",
    )
    return energies, gradients


def audit_outer_state_chain_entry(
    metric,
    record,
    n_internal,
    expected_outer_start,
    q_matrix,
    label,
):
    """Authenticate fields common to a main-stage and transaction state link."""

    path_shape = (int(n_internal) + 2, 10)
    outer_start = require_state_chain_array(
        record, "outer_start_path", path_shape, label
    )
    post_path = require_state_chain_array(
        record, "post_main_path", path_shape, label
    )
    post_energies = require_state_chain_array(
        record, "post_main_energies", (path_shape[0],), label
    )
    post_gradients = require_state_chain_array(
        record, "post_main_gradients", path_shape, label
    )
    expected_outer_start = np.asarray(expected_outer_start, dtype=np.float64)
    require(
        expected_outer_start.shape == path_shape
        and np.all(np.isfinite(expected_outer_start)),
        "expected outer-start state is invalid",
    )
    require(
        np.array_equal(outer_start, expected_outer_start),
        label + " outer-start path breaks the committed-public chain",
    )
    require(
        np.array_equal(outer_start[[0, -1]], post_path[[0, -1]]),
        label + " state chain changed an endpoint",
    )
    energy_error, gradient_error = verify_stored_energy_gradient(
        metric,
        post_path[None, ...],
        post_energies[None, ...],
        post_gradients[None, ...],
        q_matrix,
        label + " post-main state",
    )
    outer_energies, outer_gradients = independently_evaluate_path(
        metric, outer_start, q_matrix
    )
    outer_i = independent_gate_i_perp(
        metric, outer_start, outer_energies, outer_gradients
    )
    post_i = independent_gate_i_perp(
        metric, post_path, post_energies, post_gradients
    )
    return {
        "outer_start_path": outer_start,
        "outer_start_energies": outer_energies,
        "outer_start_gradients": outer_gradients,
        "outer_start_i_perp": float(outer_i),
        "post_main_path": post_path,
        "post_main_energies": post_energies,
        "post_main_gradients": post_gradients,
        "post_main_i_perp": float(post_i),
        "post_main_equidistance": float(independent_equidistance(post_path)),
        "max_post_main_energy_error": float(energy_error),
        "max_post_main_gradient_error": float(gradient_error),
    }


def audit_main_stage_state_chain_entry(
    metric,
    stage,
    n_internal,
    expected_outer_start,
    q_matrix,
):
    """Reconstruct one complete main-stage public transition."""

    state = audit_outer_state_chain_entry(
        metric,
        stage,
        n_internal,
        expected_outer_start,
        q_matrix,
        "main stage",
    )
    pre_i = float(state["outer_start_i_perp"])
    post_i = float(state["post_main_i_perp"])
    require_close(stage.get("pre_i_perp"), pre_i, "main-stage pre I_perp")
    require_close(
        stage.get("post_main_i_perp"), post_i, "main-stage post I_perp"
    )
    rolled_back = bool(stage.get("rolled_back_by_i_perp_guard"))
    candidate_i = float(stage.get("candidate_i_perp"))
    require(np.isfinite(candidate_i), "main-stage candidate I_perp is non-finite")
    require(
        rolled_back == bool(candidate_i > pre_i),
        "main-stage rollback gate differs from independent pre I_perp",
    )
    if rolled_back:
        require(
            np.array_equal(state["post_main_path"], state["outer_start_path"])
            and np.array_equal(
                state["post_main_energies"], state["outer_start_energies"]
            )
            and np.array_equal(
                state["post_main_gradients"], state["outer_start_gradients"]
            ),
            "rolled-back main stage did not restore its outer-start state",
        )
    else:
        require_close(candidate_i, post_i, "accepted main-stage candidate I_perp")
    changed = bool(
        not rolled_back
        and not np.array_equal(
            state["outer_start_path"], state["post_main_path"]
        )
    )
    require(
        stage.get("main_public_path_changed_bitwise") is changed,
        "main-stage public-path change flag drifted",
    )
    return state


def audit_transaction_state_chain_entry(
    metric,
    transaction,
    n_internal,
    expected_outer_start,
    q_matrix,
):
    """Authenticate one zero-NoF-safe outer/post-main/public state link."""

    state = audit_outer_state_chain_entry(
        metric,
        transaction,
        n_internal,
        expected_outer_start,
        q_matrix,
        "transaction",
    )
    committed = require_state_chain_array(
        transaction,
        "committed_public_path",
        state["post_main_path"].shape,
        "transaction",
    )
    require(
        np.array_equal(
            state["outer_start_path"][[0, -1]], committed[[0, -1]]
        ),
        "transaction state chain changed an endpoint",
    )
    for name in (
        "post_main_i_perp",
        "pre_i_perp",
        "acceptance_envelope_i_perp",
    ):
        require_close(transaction.get(name), state["post_main_i_perp"], name)
    require_close(
        transaction.get("pre_equidistance"),
        state["post_main_equidistance"],
        "post-main equidistance",
    )
    state["committed_public_path"] = committed
    return state


def close_transaction_state_chain_entry(transaction, state_chain, expected_committed):
    """Close one transaction link using the independently reconstructed gate."""

    expected = np.asarray(expected_committed, dtype=np.float64)
    observed = np.asarray(state_chain["committed_public_path"], dtype=np.float64)
    require(
        expected.shape == observed.shape and np.array_equal(observed, expected),
        "transaction committed-public path differs from the reconstructed gate",
    )
    movement = float(
        np.max(
            np.linalg.norm(
                observed - np.asarray(state_chain["outer_start_path"]), axis=1
            )
        )
    )
    require_close(
        transaction.get("public_path_movement"),
        movement,
        "transaction public path movement",
    )
    return observed.copy(), movement


def coordinate_sha256(point):
    value = np.ascontiguousarray(point, dtype=np.float64)
    return hashlib.sha256(value.tobytes()).hexdigest()


def audit_evaluation_miss_trace(
    trace,
    n_internal,
    initial_path,
    main_audits,
    transactions,
    direct_paths,
    metric,
    terminal_nof,
    stop_reason,
):
    """Rebuild the exact-coordinate cache and phase-local miss vectors."""

    n_internal = int(n_internal)
    total_images = n_internal + 2
    require(
        isinstance(trace, list) and len(trace) == int(terminal_nof),
        "ordered evaluation-miss trace length differs from terminal NoF",
    )
    phase_rank = {"initial": 0, "main": 1, "direct": 2, "projected": 3}
    context_vectors = {}
    context_entries = {}
    miss_lookup = {}
    previous_token = (0, 0)
    expected_fields = {
        "sequence",
        "article_nf",
        "outer",
        "phase",
        "image_index",
        "coordinate_sha256",
    }
    for offset, entry in enumerate(trace, start=1):
        require(isinstance(entry, dict), "evaluation-miss trace entry is not an object")
        require(set(entry) == expected_fields, "evaluation-miss trace schema drifted")
        require(
            type(entry.get("sequence")) is int
            and type(entry.get("article_nf")) is int
            and int(entry["sequence"]) == offset
            and int(entry["article_nf"]) == offset,
            "evaluation-miss trace sequence/NoF differs from array order",
        )
        outer = entry.get("outer")
        image = entry.get("image_index")
        phase = entry.get("phase")
        key = entry.get("coordinate_sha256")
        require(type(outer) is int and type(image) is int, "evaluation-miss trace index is not an integer")
        require(phase in phase_rank, "evaluation-miss trace phase drifted")
        require(
            isinstance(key, str)
            and len(key) == 64
            and all(character in "0123456789abcdef" for character in key),
            "evaluation-miss trace coordinate key is invalid",
        )
        require(1 <= image <= n_internal, "evaluation-miss trace charged an endpoint")
        if phase == "initial":
            require(outer == 0, "initial evaluation-miss trace outer drifted")
            token = (0, 0)
        else:
            require(outer >= 1, "non-initial evaluation-miss trace outer drifted")
            token = (outer, phase_rank[phase])
        require(token >= previous_token, "evaluation-miss trace context order decreases")
        previous_token = token
        identity = (image, key)
        require(identity not in miss_lookup, "evaluation-miss trace repeats a cached coordinate")
        miss_lookup[identity] = {"token": token, "entry": entry}
        context = (outer, phase)
        vector = context_vectors.setdefault(
            context, np.zeros(total_images, dtype=np.int64)
        )
        vector[image] += 1
        context_entries.setdefault(context, []).append(entry)

    initial = np.asarray(initial_path, dtype=np.float64)
    require(initial.shape == (total_images, 10), "miss-trace initial path shape drifted")
    initial_vector = context_vectors.get(
        (0, "initial"), np.zeros(total_images, dtype=np.int64)
    )
    expected_initial = np.zeros(total_images, dtype=np.int64)
    expected_initial[1:-1] = 1
    require(
        np.array_equal(initial_vector, expected_initial),
        "initial evaluation-miss trace does not contain one miss per internal image",
    )
    for image in range(1, total_images - 1):
        key = coordinate_sha256(initial[image])
        record = miss_lookup.get((image, key))
        require(
            record is not None and record["token"] == (0, 0),
            "initial evaluation-miss coordinate key drifted",
        )

    def context_vector(outer, phase):
        return context_vectors.get(
            (int(outer), str(phase)), np.zeros(total_images, dtype=np.int64)
        ).copy()

    def expected_complete_path_vector(path, outer, phase):
        points = np.asarray(path, dtype=np.float64)
        require(points.shape == (total_images, 10), phase + " trace path shape drifted")
        current = (int(outer), phase_rank[str(phase)])
        expected = np.zeros(total_images, dtype=np.int64)
        for image in range(1, total_images - 1):
            record = miss_lookup.get((image, coordinate_sha256(points[image])))
            require(record is not None, phase + " queried coordinate is absent from miss cache")
            require(record["token"] <= current, phase + " coordinate first appears in a future context")
            expected[image] = int(record["token"] == current)
        require(
            np.array_equal(context_vector(outer, phase), expected),
            phase + " evaluation-miss keys differ from its known path",
        )
        return expected

    def expected_partial_path_vector(path, outer, phase):
        points = np.asarray(path, dtype=np.float64)
        require(points.shape == (total_images, 10), phase + " partial path shape drifted")
        current = (int(outer), phase_rank[str(phase)])
        expected = np.zeros(total_images, dtype=np.int64)
        blocked = False
        for image in range(1, total_images - 1):
            record = miss_lookup.get((image, coordinate_sha256(points[image])))
            if record is None or record["token"] > current:
                blocked = True
                break
            expected[image] = int(record["token"] == current)
        require(blocked, phase + " partial evaluation lacks a blocked miss")
        require(
            np.array_equal(context_vector(outer, phase), expected),
            phase + " partial evaluation-miss keys differ from its known path",
        )
        return expected

    main_vectors = {}
    direct_vectors = {}
    projected_vectors = {}
    used_contexts = {(0, "initial")}
    for index, stage in enumerate(main_audits):
        outer = index + 1
        vector = context_vector(outer, "main")
        image_audits = stage.get("image_audits", [])
        require(len(image_audits) == n_internal, "main miss trace lacks image audits")
        logged = np.zeros(total_images, dtype=np.int64)
        for image, item in enumerate(image_audits, start=1):
            logged[image] = int(item.get("new_calls", -1))
        require(
            np.array_equal(vector, logged),
            "main-stage new_calls differ from the ordered miss trace",
        )
        main_vectors[outer] = vector
        used_contexts.add((outer, "main"))

    direct_paths = np.asarray(direct_paths, dtype=np.float64)
    for index, path in enumerate(direct_paths):
        outer = index + 1
        direct_vectors[outer] = expected_complete_path_vector(
            path, outer, "direct"
        )
        used_contexts.add((outer, "direct"))

    for index, transaction in enumerate(transactions):
        outer = index + 1
        attempt = transaction.get("projected_attempt_path")
        projected_evaluated = bool(transaction.get("projected_trial_evaluated"))
        interrupted = bool(transaction.get("interrupted"))
        if attempt is None:
            expected = np.zeros(total_images, dtype=np.int64)
            require(
                not projected_evaluated and not interrupted,
                "evaluated/interrupted projection lacks its attempt path",
            )
            require(
                np.array_equal(context_vector(outer, "projected"), expected),
                "projection miss trace exists without an attempt path",
            )
        elif projected_evaluated:
            expected = expected_complete_path_vector(
                attempt, outer, "projected"
            )
            require(
                np.array_equal(
                    np.asarray(attempt, dtype=np.float64),
                    np.asarray(transaction.get("projected_candidate_path"), dtype=np.float64),
                ),
                "complete projected attempt/candidate paths differ",
            )
        elif interrupted:
            expected = expected_partial_path_vector(
                attempt, outer, "projected"
            )
        else:
            expected = np.zeros(total_images, dtype=np.int64)
            require(
                np.array_equal(context_vector(outer, "projected"), expected),
                "unevaluated projection has evaluation-miss records",
            )
        projected_vectors[outer] = expected
        used_contexts.add((outer, "projected"))

    if len(main_audits) == len(transactions) + 1 and len(direct_paths) == len(transactions):
        outer = len(main_audits)
        stage = main_audits[-1]
        if (outer, "direct") in context_vectors or stop_reason == "budget":
            expected_direct, _, _ = independent_improved_spring(
                metric,
                np.asarray(stage.get("post_main_path"), dtype=np.float64),
                np.asarray(stage.get("post_main_energies"), dtype=np.float64),
            )
            direct_vectors[outer] = expected_partial_path_vector(
                expected_direct, outer, "direct"
            )
            used_contexts.add((outer, "direct"))

    unconsumed = set(context_vectors) - used_contexts
    if unconsumed:
        require(
            stop_reason in ("budget", "numerical_failure")
            and len(unconsumed) == 1,
            "evaluation-miss trace contains an unexplained context",
        )
        outer, phase = next(iter(unconsumed))
        require(
            outer == len(main_audits) + 1 and phase == "main",
            "evaluation-miss trace tail is not a partial main stage",
        )

    return {
        "main_vectors": main_vectors,
        "direct_vectors": direct_vectors,
        "projected_vectors": projected_vectors,
        "trace_length": len(trace),
    }


def audit_stop_reason_semantics(
    stop_reason,
    failure_message,
    terminal_nof,
    blocked_misses,
    main_audits,
    transactions,
    direct_count,
    m_max,
    miss_trace_audit,
    partial_tail_class,
):
    """Reconstruct which frozen run-loop exit can produce the stored stop."""

    stop_reason = str(stop_reason)
    failure_message = str(failure_message)
    terminal_nof = int(terminal_nof)
    blocked_misses = int(blocked_misses)
    main_audits = list(main_audits)
    transactions = list(transactions)
    direct_count = int(direct_count)
    m_max = int(m_max)

    def outer_calls(outer):
        total = 0
        for name in ("main_vectors", "direct_vectors", "projected_vectors"):
            vector = miss_trace_audit[name].get(int(outer))
            if vector is not None:
                total += int(np.sum(np.asarray(vector, dtype=np.int64)))
        return total

    for index, transaction in enumerate(transactions, start=1):
        require(
            int(transaction.get("outer", -1)) == index,
            "stop audit transaction outer indices are not consecutive",
        )
        if stop_reason == "numerical_failure" and transaction.get(
            "recovery_backtrack_lifecycle_action"
        ) == "not_resolved":
            require(
                index == len(transactions),
                "an unresolved numerical-failure transaction is not trailing",
            )
            continue
        interrupted = bool(transaction.get("interrupted"))
        committed = bool(transaction.get("committed"))
        rolled_back = bool(transaction.get("rolled_back"))
        projected_evaluated = bool(
            transaction.get("projected_trial_evaluated")
        )
        completed = bool(projected_evaluated and not interrupted)
        before = int(
            transaction.get("same_anchor_recovery_attempt_count_before", -1)
        )
        after = int(
            transaction.get("same_anchor_recovery_attempt_count_after", -1)
        )
        require(
            0 <= before <= m_max and after == before + int(completed) <= m_max,
            "recovery attempt counter is inconsistent with a completed trial",
        )
        require(
            bool(
                transaction.get(
                    "completed_recovery_candidate_at_current_anchor"
                )
            )
            == completed,
            "completed-recovery lifecycle flag drifted",
        )
        require(
            int(transaction.get("same_anchor_recovery_attempt_index", -1))
            == (after if completed else 0),
            "same-anchor recovery attempt index drifted",
        )
        require(
            int(transaction.get("same_anchor_recovery_attempt_cap_m_max", -1))
            == m_max,
            "same-anchor recovery attempt cap drifted",
        )
        movement = float(transaction.get("public_path_movement"))
        require(np.isfinite(movement) and movement >= 0.0, "public movement is invalid")
        carry = bool(transaction.get("recovery_backtrack_carry_allowed"))
        require(
            not carry or (completed and rolled_back and not committed and after < m_max),
            "retry carry is inconsistent with lifecycle and attempt cap",
        )
        retry = bool(transaction.get("same_anchor_recovery_retry_allowed"))
        require(
            retry == bool(carry and movement <= 1.0e-14),
            "same-anchor retry decision differs from movement/carry state",
        )
        if interrupted:
            expected_action = "clear_on_interruption"
        elif committed:
            expected_action = "clear_on_transaction_commit"
        elif carry:
            expected_action = (
                "retain_new_line_search_after_main_movement"
                if movement > 1.0e-14
                else "retain_for_same_anchor_backtrack"
            )
        else:
            expected_action = (
                "clear_on_stationary_or_attempt_cap"
                if movement <= 1.0e-14
                else "clear_on_no_descriptor_or_attempt_cap_after_main_movement"
            )
        require(
            transaction.get("recovery_backtrack_lifecycle_action")
            == expected_action,
            "recovery retry lifecycle action differs from the run-loop branch",
        )

    require(
        blocked_misses == (1 if stop_reason == "budget" else 0),
        "blocked-miss count is inconsistent with the run-loop stop branch",
    )
    if stop_reason == "budget":
        require(
            terminal_nof == ARTICLE_NF_BUDGET and failure_message == "",
            "budget stop lacks the exact terminal ledger or has a failure message",
        )
        require(
            partial_tail_class
            in ("partial_main_stage", "partial_direct_band", "none"),
            "budget stop has an impossible trailing execution shape",
        )
        interrupted_indices = [
            index
            for index, transaction in enumerate(transactions)
            if bool(transaction.get("interrupted"))
        ]
        require(
            not interrupted_indices
            or interrupted_indices == [len(transactions) - 1],
            "budget interruption is not the final transaction",
        )
        return "budget"

    if stop_reason == "outer_limit":
        require(failure_message == "", "outer-limit stop has a failure message")
        require(
            len(main_audits) == len(transactions) == direct_count == 10000,
            "outer-limit stop did not complete exactly 10000 outers",
        )
        require(partial_tail_class == "none", "outer-limit stop has a partial tail")
        for index, transaction in enumerate(transactions, start=1):
            require(not bool(transaction.get("interrupted")), "outer-limit run was interrupted")
            movement = float(transaction.get("public_path_movement"))
            carry = bool(transaction.get("recovery_backtrack_carry_allowed"))
            require(
                (movement <= 1.0e-14 and carry)
                or (movement > 1.0e-14 and outer_calls(index) > 0),
                "outer-limit run contains an earlier stationary exit condition",
            )
        return "outer_limit"

    if stop_reason == "stationary":
        require(failure_message == "", "stationary stop has a failure message")
        require(
            len(transactions) > 0
            and len(main_audits) == len(transactions) == direct_count
            and partial_tail_class == "none",
            "stationary stop lacks one final complete outer transaction",
        )
        transaction = transactions[-1]
        require(not bool(transaction.get("interrupted")), "stationary stop was interrupted")
        movement = float(transaction.get("public_path_movement"))
        carry = bool(transaction.get("recovery_backtrack_carry_allowed"))
        zero_nof_outer = bool(outer_calls(len(transactions)) == 0)
        movement_branch = bool(movement <= 1.0e-14 and not carry)
        require(
            movement_branch or zero_nof_outer,
            "stationary stop matches neither frozen run-loop exit branch",
        )
        if carry:
            require(
                zero_nof_outer and movement > 1.0e-14,
                "stationary stop retained a retry that would have continued",
            )
        if movement_branch and bool(transaction.get("rolled_back")) and bool(
            transaction.get("projected_trial_evaluated")
        ):
            require(
                int(transaction.get("same_anchor_recovery_attempt_count_after", -1))
                == m_max,
                "rejected stationary recovery stopped before exhausting retries",
            )
        return "stationary"

    require(stop_reason == "numerical_failure", "unknown stop branch")
    require(
        failure_message.startswith(
            ("CertificationFailure: ", "FloatingPointError: ", "ValueError: ", "LinAlgError: ")
        )
        and len(failure_message.split(": ", 1)[1]) > 0,
        "numerical-failure message is not a caught exception record",
    )
    require(
        partial_tail_class
        in (
            "none",
            "partial_main_stage",
            "partial_direct_band",
            "complete_direct_plus_partial_recovery_or_failure",
        ),
        "numerical failure has an impossible trailing execution shape",
    )
    require(
        not any(bool(transaction.get("interrupted")) for transaction in transactions),
        "numerical failure contains a budget interruption",
    )
    return "numerical_failure"


def advance_main_stage_nof(stage, running_nof, trace_vector=None):
    calls = int(
        sum(
            int(item.get("new_calls", -1))
            for item in stage.get("image_audits", [])
        )
    )
    require(calls >= 0, "main-stage NoF increment is invalid")
    if trace_vector is not None:
        trace_vector = np.asarray(trace_vector, dtype=np.int64)
        require(
            int(np.sum(trace_vector)) == calls,
            "main-stage calls differ from independently reconstructed miss trace",
        )
        calls = int(np.sum(trace_vector))
    expected = int(running_nof) + calls
    require(
        int(stage.get("terminal_article_nf", -1)) == expected,
        "main-stage terminal NoF breaks the per-outer ledger",
    )
    return expected


def advance_transaction_nof(
    transaction,
    direct_trial_nof,
    running_nof,
    total_images,
    direct_trace_vector=None,
    recovery_trace_vector=None,
):
    direct = np.asarray(
        transaction.get("direct_joint_evaluation_vector"), dtype=np.int64
    )
    recovery = np.asarray(
        transaction.get("recovery_joint_evaluation_vector"), dtype=np.int64
    )
    require(
        direct.shape == (int(total_images),)
        and recovery.shape == (int(total_images),)
        and np.all(direct >= 0)
        and np.all(recovery >= 0),
        "transaction state-chain NoF vectors are invalid",
    )
    if direct_trace_vector is not None:
        require(
            np.array_equal(direct, np.asarray(direct_trace_vector, dtype=np.int64)),
            "direct NoF vector differs from independently reconstructed miss trace",
        )
    if recovery_trace_vector is not None:
        require(
            np.array_equal(
                recovery, np.asarray(recovery_trace_vector, dtype=np.int64)
            ),
            "recovery NoF vector differs from independently reconstructed miss trace",
        )
    after_direct = int(running_nof) + int(np.sum(direct))
    require(
        int(direct_trial_nof) == after_direct,
        "direct-trial NoF breaks the per-outer ledger",
    )
    return after_direct + int(np.sum(recovery))


def audit_main_and_transaction_ledgers(control, main_audits, transactions):
    n_internal = int(control.n_internal)
    total_images = n_internal + 2
    main_vector = np.zeros(total_images, dtype=np.int64)
    previous_outer = 0
    for stage_index, stage in enumerate(main_audits):
        outer = int(stage.get("outer", -1))
        require(outer == previous_outer + 1, "main-stage outer indices are not consecutive")
        previous_outer = outer
        require(stage.get("recovery_reads_main_r") is False, "main audit says recovery read r")
        require(
            float(stage.get("recovery_acceptance_envelope_i_perp"))
            == float(stage.get("post_main_i_perp")),
            "main audit recovery threshold is not the post-main I_perp",
        )
        require_finite_json(stage, "main_stage_audit[{}]".format(stage_index))
        image_audits = stage.get("image_audits")
        require(isinstance(image_audits, list), "main audit lacks image audits")
        require(len(image_audits) == n_internal, "main stage omits an internal image")
        for image_offset, item in enumerate(image_audits, start=1):
            calls = int(item.get("new_calls", -1))
            accepted = int(item.get("accepted_steps", -1))
            require(0 <= calls <= max(12, 4 * int(control.m_max)), "main image call cap exceeded")
            require(0 <= accepted <= int(control.m_max), "main image m_max cap exceeded")
            require(float(item.get("target_fraction")) == float(control.r), "main image r drifted")
            main_vector[image_offset] += calls

    transaction_vector = np.zeros(total_images, dtype=np.int64)
    previous_outer = 0
    committed = 0
    rolled_back = 0
    interrupted = 0
    for index, transaction in enumerate(transactions):
        require(
            not any("window" in str(name).lower() for name in transaction),
            "transaction contains an unregistered history-window gate",
        )
        outer = int(transaction.get("outer", -1))
        require(outer > previous_outer, "transaction outer indices are not increasing")
        previous_outer = outer
        require_finite_json(transaction, "transaction[{}]".format(index))
        require(transaction.get("main_r_read_by_recovery") is False, "recovery read main r")
        require(transaction.get("recovery_acceptance_uses_main_r") is False, "recovery acceptance read main r")
        require(transaction.get("recovery_geometry_uses_main_r") is False, "recovery geometry read main r")
        require(transaction.get("regularized_prediction_uses_main_r") is False, "direction prediction improperly read r")
        require(
            float(transaction.get("acceptance_envelope_i_perp"))
            == float(transaction.get("post_main_i_perp"))
            == float(transaction.get("pre_i_perp")),
            "transaction recovery threshold is not the post-main I_perp",
        )
        require(
            transaction.get("direct_i_perp_acceptance_reference") == "post-main I_perp"
            and transaction.get("recovery_i_perp_acceptance_reference")
            == "post-main I_perp or positive actual proximal merit",
            "transaction acceptance-reference identity drifted",
        )
        require(
            transaction.get("recovery_commit_gate")
            == "post_main_i_monotone_or_positive_actual_proximal_merit",
            "transaction recovery gate identity drifted",
        )
        require(transaction.get("raw_trial_evaluated") is False, "algebraic raw proposal was evaluated")
        require(int(transaction.get("raw_recovery_new_image_evaluations", -1)) == 0, "raw recovery consumed NoF")
        require(int(transaction.get("cycles", -1)) in (0, 1), "more than one recovery band was attempted")
        require(int(transaction.get("recovery_band_evaluation_cap", -1)) == RECOVERY_BAND_EVALUATION_CAP, "recovery band cap drifted")
        require(int(transaction.get("transaction_call_cap_per_image_including_direct", -1)) == TRANSACTION_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT, "transaction per-image cap drifted")
        require(float(transaction.get("recovery_backtrack_factor")) == float(BACKTRACK), "recovery backtrack factor drifted")
        require(int(transaction.get("same_anchor_recovery_attempt_cap_m_max", -1)) == int(control.m_max), "recovery m_max audit drifted")
        raw_constructed = bool(transaction.get("raw_proposal_constructed"))
        require(
            (transaction.get("raw_candidate_path") is not None) == raw_constructed,
            "raw recovery evidence presence drifted",
        )
        require(
            (transaction.get("recovery_projection_seed_path") is not None)
            == raw_constructed,
            "recovery seed evidence presence drifted",
        )

        vectors = {}
        for name in (
            "direct_joint_evaluation_vector",
            "raw_recovery_joint_evaluation_vector",
            "projected_recovery_joint_evaluation_vector",
            "recovery_joint_evaluation_vector",
            "transaction_joint_evaluation_vector",
        ):
            vector = np.asarray(transaction.get(name), dtype=np.int64)
            require(vector.shape == (total_images,), name + " shape drifted")
            require(np.all(vector >= 0), name + " contains negative calls")
            require(vector[0] == 0 and vector[-1] == 0, name + " charges endpoints")
            vectors[name] = vector
        direct = vectors["direct_joint_evaluation_vector"]
        raw = vectors["raw_recovery_joint_evaluation_vector"]
        projected = vectors["projected_recovery_joint_evaluation_vector"]
        recovery = vectors["recovery_joint_evaluation_vector"]
        total = vectors["transaction_joint_evaluation_vector"]
        require(np.all(direct <= 1), "direct proposal charged an image more than once")
        require(np.all(raw == 0), "raw recovery vector is not zero")
        require(np.all(projected <= 1), "projected band charged an image more than once")
        require(np.array_equal(recovery, raw + projected), "recovery call vectors do not close")
        require(np.array_equal(total, direct + recovery), "transaction call vectors do not close")
        require(np.all(total <= TRANSACTION_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT), "transaction per-image call cap exceeded")
        require(int(transaction.get("direct_joint_evaluations", -1)) == int(np.sum(direct)), "direct scalar/vector NoF mismatch")
        require(int(transaction.get("projected_recovery_new_image_evaluations", -1)) == int(np.sum(projected)), "projected scalar/vector NoF mismatch")
        require(int(transaction.get("recovery_joint_evaluations", -1)) == int(np.sum(recovery)), "recovery scalar/vector NoF mismatch")
        require(int(transaction.get("recovery_new_image_evaluations", -1)) == int(np.sum(recovery)), "recovery total NoF mismatch")
        require(int(transaction.get("transaction_joint_evaluations", -1)) == int(np.sum(total)), "transaction scalar/vector NoF mismatch")
        transaction_vector += total

        is_committed = bool(transaction.get("committed"))
        is_rolled_back = bool(transaction.get("rolled_back"))
        is_interrupted = bool(transaction.get("interrupted"))
        require(not (is_committed and is_rolled_back), "transaction is both committed and rolled back")
        require(is_committed or is_rolled_back, "transaction is neither committed nor rolled back")
        if is_interrupted:
            require(is_rolled_back, "interrupted transaction was not rolled back")
        committed += int(is_committed)
        rolled_back += int(is_rolled_back)
        interrupted += int(is_interrupted)
    return {
        "main_vector": main_vector,
        "transaction_vector": transaction_vector,
        "main_calls": int(np.sum(main_vector)),
        "transaction_calls": int(np.sum(transaction_vector)),
        "committed_transactions": committed,
        "rolled_back_transactions": rolled_back,
        "interrupted_transactions": interrupted,
    }


def audit_recovery_gates(
    metric,
    control,
    main_audits,
    transactions,
    direct_nf,
    direct_paths,
    direct_energies,
    direct_gradients,
    initial_path,
    public_nof,
    public_paths,
    public_energies,
    public_gradients,
    stop_reason,
    terminal_nof,
    q_matrix,
    miss_trace_audit,
):
    """Independently reconstruct every gate, branch, and secant partition."""

    n_internal = int(control.n_internal)
    total_images = n_internal + 2
    projected_trials = 0
    projected_commits = 0
    monotone_projected_commits = 0
    merit_only_projected_commits = 0
    rejected_projected_trials = 0
    direct_commits = 0
    max_projected_energy_error = 0.0
    max_projected_gradient_error = 0.0
    max_post_main_energy_error = 0.0
    max_post_main_gradient_error = 0.0
    max_candidate_i_error = 0.0
    max_candidate_equidistance_error = 0.0
    max_actual_merit_error = 0.0
    stationary_probes = 0
    strong_seeds = 0
    conservative_seeds = 0
    secants_certified = 0
    secants_curvature_rejected = 0
    secants_rank_omitted = 0
    secants_published = 0
    strong_publication_commits = 0
    conservative_publication_commits = 0
    ordinary_publication_commits = 0
    max_published_curvature_error = 0.0
    maximum_public_path_movement_error = 0.0

    expected_outer_start = np.asarray(initial_path, dtype=np.float64)
    require(
        expected_outer_start.shape == (total_images, 10)
        and np.all(np.isfinite(expected_outer_start)),
        "initial state cannot start the transaction path chain",
    )
    main_audits = list(main_audits)
    direct_nf = np.asarray(direct_nf, dtype=np.int64)
    require(
        len(main_audits) - len(transactions) in (0, 1),
        "state chain has more than one trailing complete main stage",
    )
    require(
        direct_nf.size >= len(transactions),
        "state chain lacks a direct NoF anchor",
    )
    initial_energies, initial_gradients = independently_evaluate_path(
        metric, expected_outer_start, q_matrix
    )
    expected_public_events = []

    def append_public_event(nof_value, state_path, state_energies, state_gradients):
        event = (
            int(nof_value),
            np.asarray(state_path, dtype=np.float64).copy(),
            np.asarray(state_energies, dtype=np.float64).copy(),
            np.asarray(state_gradients, dtype=np.float64).copy(),
        )
        if expected_public_events:
            require(
                event[0] >= expected_public_events[-1][0],
                "reconstructed public-state NoF decreases",
            )
        if expected_public_events and event[0] == expected_public_events[-1][0]:
            expected_public_events[-1] = event
        else:
            expected_public_events.append(event)

    append_public_event(
        n_internal,
        expected_outer_start,
        initial_energies,
        initial_gradients,
    )
    expected_outer_energies = initial_energies
    expected_outer_gradients = initial_gradients
    expected_running_nof = int(n_internal)
    raw_retry_state = None

    for index, transaction in enumerate(transactions):
        require(
            int(transaction.get("outer", -1)) == index + 1,
            "transaction state-chain outer indices are not consecutive",
        )
        stage = main_audits[index]
        require(
            int(stage.get("outer", -1)) == index + 1,
            "main-stage state-chain outer indices are not consecutive",
        )
        expected_running_nof = advance_main_stage_nof(
            stage,
            expected_running_nof,
            miss_trace_audit["main_vectors"].get(index + 1),
        )
        stage_chain = audit_main_stage_state_chain_entry(
            metric,
            stage,
            n_internal,
            expected_outer_start,
            q_matrix,
        )
        append_public_event(
            expected_running_nof,
            stage_chain["post_main_path"],
            stage_chain["post_main_energies"],
            stage_chain["post_main_gradients"],
        )
        state_chain = audit_transaction_state_chain_entry(
            metric,
            transaction,
            n_internal,
            expected_outer_start,
            q_matrix,
        )
        expected_running_nof = advance_transaction_nof(
            transaction,
            direct_nf[index],
            expected_running_nof,
            total_images,
            miss_trace_audit["direct_vectors"].get(index + 1),
            miss_trace_audit["projected_vectors"].get(index + 1),
        )
        for name in (
            "outer_start_path",
            "post_main_path",
            "post_main_energies",
            "post_main_gradients",
        ):
            require(
                np.array_equal(state_chain[name], stage_chain[name]),
                "main-stage/transaction state-chain fields differ: " + name,
            )
        max_post_main_energy_error = max(
            max_post_main_energy_error,
            float(state_chain["max_post_main_energy_error"]),
        )
        max_post_main_gradient_error = max(
            max_post_main_gradient_error,
            float(state_chain["max_post_main_gradient_error"]),
        )
        post_main_path = np.asarray(
            state_chain["post_main_path"], dtype=np.float64
        )
        post_main_energy = np.asarray(
            state_chain["post_main_energies"], dtype=np.float64
        )
        post_main_i = float(state_chain["post_main_i_perp"])
        pre_eq = float(state_chain["post_main_equidistance"])
        direct_path = np.asarray(direct_paths[index], dtype=np.float64)
        direct_energy = np.asarray(direct_energies[index], dtype=np.float64)
        direct_gradient = np.asarray(direct_gradients[index], dtype=np.float64)
        direct_i = independent_gate_i_perp(
            metric, direct_path, direct_energy, direct_gradient
        )
        direct_eq = independent_equidistance(direct_path)
        expected_direct, expected_spring_residual, expected_spring_iterations = (
            independent_improved_spring(
                metric, post_main_path, post_main_energy
            )
        )
        require(
            np.array_equal(direct_path, expected_direct),
            "direct reparameterization is not reproducible from post-main state",
        )
        require_close(
            transaction.get("spring_residual"),
            expected_spring_residual,
            "direct spring residual",
        )
        require(
            type(transaction.get("spring_iterations")) is int
            and int(transaction.get("spring_iterations"))
            == int(expected_spring_iterations),
            "direct spring iteration count drifted",
        )
        require_close(transaction.get("direct_i_perp"), direct_i, "direct I_perp")
        require_close(
            transaction.get("direct_equidistance"),
            direct_eq,
            "direct equidistance",
        )
        require_close(
            transaction.get("regularized_prediction_alpha_eff"),
            control.alpha,
            "transaction alpha",
        )
        require(
            float(transaction.get("i_perp_absolute_tolerance"))
            == float(AUDIT_RECOVERY_IPERP_ATOL),
            "transaction I_perp tolerance drifted",
        )
        require(
            float(transaction.get("equidistance_absolute_tolerance"))
            == float(AUDIT_RECOVERY_EQUIDISTANCE_ATOL),
            "transaction equidistance tolerance drifted",
        )
        i_atol = float(AUDIT_RECOVERY_IPERP_ATOL)
        eq_atol = float(AUDIT_RECOVERY_EQUIDISTANCE_ATOL)
        ceiling = float(transaction.get("equidistance_ceiling"))
        require(np.isfinite(post_main_i), "post-main I_perp is non-finite")
        require(np.isfinite(ceiling), "equidistance ceiling is non-finite")
        expected_ceiling = float(
            max(direct_eq, RECOVERY_EQUIDISTANCE_CONTRACTION * pre_eq)
        )
        require_close(ceiling, expected_ceiling, "recovery equidistance ceiling")
        direct_ok = bool(
            direct_i <= post_main_i + i_atol
            and direct_eq <= ceiling + eq_atol
        )
        require(
            bool(transaction.get("direct_within_post_main_i_perp")) == direct_ok,
            "direct post-main acceptance gate is inconsistent",
        )
        outer_start_path = np.asarray(
            state_chain["outer_start_path"], dtype=np.float64
        )
        stationary_movement = float(
            np.max(np.linalg.norm(direct_path - outer_start_path, axis=1))
        )
        expected_stationary = bool(
            direct_ok and stationary_movement <= float(KKT_ATOL)
        )
        stationary = bool(transaction.get("stationary_normal_probe"))
        require(
            stationary == expected_stationary,
            "stationary recovery branch differs from public/direct paths",
        )
        require(
            not stationary or direct_ok,
            "stationary recovery probe lacks an acceptable direct band",
        )
        if stationary:
            stationary_probes += 1
            require_close(
                transaction.get("stationary_public_movement"),
                stationary_movement,
                "stationary public movement",
            )
            require(
                stationary_movement <= float(KKT_ATOL),
                "stationary recovery probe exceeded the public tolerance",
            )
        else:
            require(
                transaction.get("stationary_public_movement") is None,
                "non-stationary transaction logged stationary movement",
            )

        raw_constructed = bool(transaction.get("raw_proposal_constructed"))
        raw_retry_state = audit_raw_proposal_algebra(
            transaction,
            post_main_path,
            direct_path,
            direct_energy,
            direct_gradient,
            control.alpha,
            control.m_max,
            raw_retry_state,
        )
        seed_audit = None
        if raw_constructed:
            seed_audit = audit_chord_coherent_seed(
                metric, control, transaction, direct_energy
            )
            strong_seeds += int(seed_audit["strong"])
            conservative_seeds += int(
                not seed_audit["stationary"] and not seed_audit["strong"]
            )
        projection_audit = audit_projection_replay(
            metric,
            transaction,
            direct_energy,
            ceiling,
        )

        projected_evaluated = bool(transaction.get("projected_trial_evaluated"))
        require(
            bool(transaction.get("sync_trial_evaluated")) == projected_evaluated,
            "projected/synchronized evaluation flags differ",
        )
        require(
            int(transaction.get("cycles", -1)) == int(projected_evaluated),
            "recovery cycle count differs from projected evaluation",
        )
        projected_field_names = (
            "projected_candidate_path",
            "projected_candidate_energies",
            "projected_candidate_gradients",
        )
        candidate_scalar_names = (
            "candidate_i_perp",
            "candidate_equidistance",
            "actual_proximal_merit",
        )

        if not projected_evaluated:
            for name in projected_field_names + candidate_scalar_names:
                require(transaction.get(name) is None, name + " exists without a complete projected trial")
            for name in (
                "recovery_candidate_finite",
                "recovery_monotone_to_post_main_i_perp",
                "recovery_positive_actual_proximal_merit",
                "recovery_accepted_by_atomic_gate",
            ):
                require(transaction.get(name) is False, name + " is true without a complete projected trial")
            require(int(transaction.get("merged_images", -1)) == 0, "non-projected transaction merged images")
            require(
                transaction.get("recovery_projection_feasible")
                is bool(projection_audit["feasible"]),
                "unevaluated projection feasibility differs from replay",
            )
            audit_secant_publication(
                transaction,
                n_internal,
                recovery_committed=False,
                expected_mode="not_applicable",
            )
            direct_shortcut = bool(direct_ok and not stationary)
            require(bool(transaction.get("committed")) == direct_shortcut, "non-projected commit differs from direct/stationary gate")
            require(bool(transaction.get("rolled_back")) != bool(transaction.get("committed")), "non-projected commit/rollback flags are invalid")
            expected_committed_i = direct_i if direct_shortcut else post_main_i
            require_close(
                transaction.get("committed_i_perp"),
                expected_committed_i,
                "non-projected committed I_perp",
            )
            require(
                transaction.get("commit_is_monotone_to_post_main_i_perp") is True,
                "non-projected transaction is not monotone to post-main",
            )
            expected_committed_path = (
                direct_path if direct_shortcut else post_main_path
            )
            expected_committed_energies = (
                direct_energy
                if direct_shortcut
                else state_chain["post_main_energies"]
            )
            expected_committed_gradients = (
                direct_gradient
                if direct_shortcut
                else state_chain["post_main_gradients"]
            )
            expected_outer_start, public_movement = (
                close_transaction_state_chain_entry(
                    transaction, state_chain, expected_committed_path
                )
            )
            maximum_public_path_movement_error = max(
                maximum_public_path_movement_error,
                abs(
                    float(transaction.get("public_path_movement"))
                    - public_movement
                ),
            )
            expected_outer_energies = np.asarray(
                expected_committed_energies, dtype=np.float64
            ).copy()
            expected_outer_gradients = np.asarray(
                expected_committed_gradients, dtype=np.float64
            ).copy()
            append_public_event(
                expected_running_nof,
                expected_outer_start,
                expected_outer_energies,
                expected_outer_gradients,
            )
            direct_commits += int(direct_shortcut)
            continue

        projected_trials += 1
        require(
            not direct_ok or stationary,
            "projected recovery ran after a non-stationary acceptable direct band",
        )
        require(
            transaction.get("recovery_projection_feasible") is True,
            "complete projected trial was marked geometrically infeasible",
        )
        projected_path = np.asarray(
            transaction.get("projected_candidate_path"), dtype=np.float64
        )
        projected_energy = np.asarray(
            transaction.get("projected_candidate_energies"), dtype=np.float64
        )
        projected_gradient = np.asarray(
            transaction.get("projected_candidate_gradients"), dtype=np.float64
        )
        require(
            projected_path.shape == (total_images, 10)
            and projected_gradient.shape == (total_images, 10),
            "projected candidate path/gradient shape drifted",
        )
        require(
            projected_energy.shape == (total_images,),
            "projected candidate energy shape drifted",
        )
        require(
            np.all(np.isfinite(projected_path))
            and np.all(np.isfinite(projected_energy))
            and np.all(np.isfinite(projected_gradient)),
            "projected candidate contains non-finite values",
        )
        require(
            np.array_equal(projected_path[[0, -1]], direct_path[[0, -1]]),
            "projected candidate changed an endpoint",
        )
        energy_error, gradient_error = verify_stored_energy_gradient(
            metric,
            projected_path[None, ...],
            projected_energy[None, ...],
            projected_gradient[None, ...],
            q_matrix,
            "projected candidate transaction {}".format(index),
        )
        max_projected_energy_error = max(max_projected_energy_error, energy_error)
        max_projected_gradient_error = max(
            max_projected_gradient_error, gradient_error
        )

        candidate_i = independent_gate_i_perp(
            metric, projected_path, projected_energy, projected_gradient
        )
        candidate_eq = independent_equidistance(projected_path)
        displacement = projected_path[1:-1] - direct_path[1:-1]
        actual_merit = float(
            np.sum(direct_energy[1:-1] - projected_energy[1:-1])
            - np.sum(displacement * displacement)
            / (2.0 * float(control.alpha))
        )
        require(np.isfinite(actual_merit), "independent actual proximal merit is non-finite")
        observed_i = float(transaction.get("candidate_i_perp"))
        observed_eq = float(transaction.get("candidate_equidistance"))
        observed_merit = float(transaction.get("actual_proximal_merit"))
        max_candidate_i_error = max(
            max_candidate_i_error, abs(observed_i - candidate_i)
        )
        max_candidate_equidistance_error = max(
            max_candidate_equidistance_error, abs(observed_eq - candidate_eq)
        )
        max_actual_merit_error = max(
            max_actual_merit_error, abs(observed_merit - actual_merit)
        )
        require_close(observed_i, candidate_i, "projected candidate I_perp")
        require_close(observed_eq, candidate_eq, "projected candidate equidistance")
        require_close(observed_merit, actual_merit, "actual proximal merit")

        finite = bool(
            np.isfinite(candidate_i)
            and np.isfinite(candidate_eq)
            and np.isfinite(actual_merit)
            and np.all(np.isfinite(projected_path))
            and np.all(np.isfinite(projected_energy))
            and np.all(np.isfinite(projected_gradient))
        )
        changed = bool(
            np.any(
                np.linalg.norm(
                    projected_path[1:-1] - direct_path[1:-1], axis=1
                )
                > 1.0e-14
            )
        )
        monotone = bool(finite and candidate_i <= post_main_i + i_atol)
        positive_merit = bool(finite and actual_merit > 0.0)
        equidistant = bool(candidate_eq <= ceiling + eq_atol)
        accepted_by_gate = bool(
            finite and changed and equidistant and (monotone or positive_merit)
        )
        require(
            bool(transaction.get("recovery_candidate_finite")) == finite,
            "projected finite audit is inconsistent",
        )
        require(
            bool(transaction.get("recovery_monotone_to_post_main_i_perp"))
            == monotone,
            "projected monotonicity audit is inconsistent",
        )
        require(
            bool(transaction.get("recovery_positive_actual_proximal_merit"))
            == positive_merit,
            "projected proximal-merit audit is inconsistent",
        )
        require(
            bool(transaction.get("recovery_accepted_by_atomic_gate"))
            == accepted_by_gate,
            "projected atomic gate audit is inconsistent",
        )
        require(
            int(transaction.get("merged_images", -1))
            == (n_internal if accepted_by_gate else 0),
            "projected atomic merged-image count is inconsistent",
        )
        require(
            bool(transaction.get("committed")) == accepted_by_gate,
            "projected commit differs from independently reconstructed gate",
        )
        require(
            bool(transaction.get("rolled_back")) != accepted_by_gate,
            "projected rollback differs from independently reconstructed gate",
        )
        expected_committed_i = candidate_i if accepted_by_gate else post_main_i
        require_close(
            transaction.get("committed_i_perp"),
            expected_committed_i,
            "projected transaction committed I_perp",
        )
        expected_commit_monotone = bool(not accepted_by_gate or monotone)
        require(
            bool(transaction.get("commit_is_monotone_to_post_main_i_perp"))
            == expected_commit_monotone,
            "projected commit monotonicity audit is inconsistent",
        )
        expected_committed_path = (
            projected_path if accepted_by_gate else post_main_path
        )
        expected_committed_energies = (
            projected_energy
            if accepted_by_gate
            else state_chain["post_main_energies"]
        )
        expected_committed_gradients = (
            projected_gradient
            if accepted_by_gate
            else state_chain["post_main_gradients"]
        )
        expected_outer_start, public_movement = (
            close_transaction_state_chain_entry(
                transaction, state_chain, expected_committed_path
            )
        )
        maximum_public_path_movement_error = max(
            maximum_public_path_movement_error,
            abs(
                float(transaction.get("public_path_movement"))
                - public_movement
            ),
        )
        expected_outer_energies = np.asarray(
            expected_committed_energies, dtype=np.float64
        ).copy()
        expected_outer_gradients = np.asarray(
            expected_committed_gradients, dtype=np.float64
        ).copy()
        append_public_event(
            expected_running_nof,
            expected_outer_start,
            expected_outer_energies,
            expected_outer_gradients,
        )
        if accepted_by_gate:
            require(seed_audit is not None, "accepted recovery lacks an audited seed")
            if stationary:
                expected_publication_mode = "ordinary_all"
            elif bool(seed_audit["strong"]):
                expected_publication_mode = "strong_top1"
            else:
                expected_publication_mode = "conservative_all"
        else:
            expected_publication_mode = "not_applicable"
        secant_audit = audit_secant_publication(
            transaction,
            n_internal,
            recovery_committed=accepted_by_gate,
            expected_mode=expected_publication_mode,
            direct_path=direct_path,
            direct_gradient=direct_gradient,
            projected_path=projected_path,
            projected_gradient=projected_gradient,
        )
        secants_certified += int(secant_audit["certified"])
        secants_curvature_rejected += int(secant_audit["rejected"])
        secants_rank_omitted += int(secant_audit["omitted"])
        secants_published += int(secant_audit["published"])
        max_published_curvature_error = max(
            max_published_curvature_error,
            float(secant_audit["max_published_normalized_curvature_error"]),
        )
        strong_publication_commits += int(
            accepted_by_gate and expected_publication_mode == "strong_top1"
        )
        conservative_publication_commits += int(
            accepted_by_gate and expected_publication_mode == "conservative_all"
        )
        ordinary_publication_commits += int(
            accepted_by_gate and expected_publication_mode == "ordinary_all"
        )
        if accepted_by_gate:
            projected_commits += 1
            if monotone:
                monotone_projected_commits += 1
            else:
                require(
                    positive_merit and actual_merit > 0.0,
                    "non-monotone recovery commit lacks positive actual merit",
                )
                merit_only_projected_commits += 1
        else:
            rejected_projected_trials += 1

    if len(main_audits) == len(transactions) + 1:
        trailing_stage = main_audits[-1]
        require(
            int(trailing_stage.get("outer", -1)) == len(main_audits),
            "trailing main-stage outer index drifted",
        )
        expected_running_nof = advance_main_stage_nof(
            trailing_stage,
            expected_running_nof,
            miss_trace_audit["main_vectors"].get(len(main_audits)),
        )
        trailing_chain = audit_main_stage_state_chain_entry(
            metric,
            trailing_stage,
            n_internal,
            expected_outer_start,
            q_matrix,
        )
        append_public_event(
            expected_running_nof,
            trailing_chain["post_main_path"],
            trailing_chain["post_main_energies"],
            trailing_chain["post_main_gradients"],
        )
        require(
            stop_reason in ("budget", "numerical_failure"),
            "trailing complete main stage lacks an interrupted tail",
        )
        if stop_reason == "budget":
            terminal_path = trailing_chain["post_main_path"]
            terminal_energies = trailing_chain["post_main_energies"]
            terminal_gradients = trailing_chain["post_main_gradients"]
        else:
            # The numerical-failure handler restores the outer-start state.
            terminal_path = expected_outer_start
            terminal_energies = expected_outer_energies
            terminal_gradients = expected_outer_gradients
    else:
        terminal_path = expected_outer_start
        terminal_energies = expected_outer_energies
        terminal_gradients = expected_outer_gradients
    append_public_event(
        terminal_nof,
        terminal_path,
        terminal_energies,
        terminal_gradients,
    )
    require(
        expected_running_nof <= int(terminal_nof),
        "completed per-outer NoF exceeds the terminal ledger",
    )

    observed_nof = np.asarray(public_nof, dtype=np.int64)
    observed_paths = np.asarray(public_paths, dtype=np.float64)
    observed_energies = np.asarray(public_energies, dtype=np.float64)
    observed_gradients = np.asarray(public_gradients, dtype=np.float64)
    expected_nof = np.asarray(
        [event[0] for event in expected_public_events], dtype=np.int64
    )
    expected_paths = np.asarray([event[1] for event in expected_public_events])
    expected_energies = np.asarray(
        [event[2] for event in expected_public_events]
    )
    expected_gradients = np.asarray(
        [event[3] for event in expected_public_events]
    )
    require(
        np.array_equal(observed_nof, expected_nof),
        "public NoF history differs from the reconstructed state chain",
    )
    require(
        observed_paths.shape == expected_paths.shape
        and np.array_equal(observed_paths, expected_paths),
        "scientific path history differs from the reconstructed state chain",
    )
    require(
        observed_energies.shape == expected_energies.shape
        and np.allclose(
            observed_energies,
            expected_energies,
            rtol=0.0,
            atol=ENERGY_GRADIENT_ATOL,
        ),
        "scientific energy history differs from the reconstructed state chain",
    )
    require(
        observed_gradients.shape == expected_gradients.shape
        and np.allclose(
            observed_gradients,
            expected_gradients,
            rtol=0.0,
            atol=ENERGY_GRADIENT_ATOL,
        ),
        "scientific gradient history differs from the reconstructed state chain",
    )

    return {
        "projected_trials": projected_trials,
        "projected_commits": projected_commits,
        "monotone_projected_commits": monotone_projected_commits,
        "merit_only_projected_commits": merit_only_projected_commits,
        "rejected_projected_trials": rejected_projected_trials,
        "direct_commits": direct_commits,
        "stationary_probes": stationary_probes,
        "strong_chord_coherent_seeds": strong_seeds,
        "conservative_chord_coherent_seeds": conservative_seeds,
        "max_projected_energy_error": float(max_projected_energy_error),
        "max_projected_gradient_error": float(max_projected_gradient_error),
        "max_post_main_energy_error": float(max_post_main_energy_error),
        "max_post_main_gradient_error": float(max_post_main_gradient_error),
        "max_public_path_movement_error": float(
            maximum_public_path_movement_error
        ),
        "authenticated_public_state_chain_events": int(expected_nof.size),
        "max_candidate_i_perp_error": float(max_candidate_i_error),
        "max_candidate_equidistance_error": float(
            max_candidate_equidistance_error
        ),
        "max_actual_proximal_merit_error": float(max_actual_merit_error),
        "accepted_recovery_secants_certified_curvature": int(secants_certified),
        "accepted_recovery_secants_curvature_rejected": int(
            secants_curvature_rejected
        ),
        "accepted_recovery_secants_rank_omitted": int(secants_rank_omitted),
        "accepted_recovery_secants_published": int(secants_published),
        "strong_top1_publication_commits": int(strong_publication_commits),
        "conservative_all_publication_commits": int(
            conservative_publication_commits
        ),
        "ordinary_all_publication_commits": int(ordinary_publication_commits),
        "max_published_normalized_curvature_error": float(
            max_published_curvature_error
        ),
    }


def audit_candidate(
    metric,
    registration,
    control,
    article,
    expected_gate_token,
):
    source = expected_result_path(control)
    receipt_path = expected_receipt_path(control)
    require(source.is_file(), "missing registered result: " + str(source))
    require(receipt_path.is_file(), "missing registered receipt: " + str(receipt_path))
    source_sha = sha256(source)
    receipt_sha = sha256(receipt_path)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise IntegrityError("receipt is invalid JSON: " + str(error))
    require(isinstance(receipt, dict), "receipt is not an object")
    require_finite_json(receipt, "receipt")
    require(receipt.get("output_sha256") == source_sha, "receipt does not authenticate result NPZ")
    require(receipt.get("run_id") == control.run_id, "receipt run ID drifted")
    require(receipt.get("controls") == control.as_dict(), "receipt controls drifted")
    require(
        receipt.get("n20_gate_manifest_sha256_token")
        == expected_gate_token,
        "receipt staged-resolution gate token drifted",
    )
    require(
        str(Path(receipt.get("output", "")).resolve()) == str(source.resolve()),
        "receipt output path drifted",
    )

    with np.load(source, allow_pickle=False) as saved:
        archive = {name: np.asarray(saved[name]) for name in saved.files}
    required_fields = {
        "experiment_id", "run_id", "config_json", "source_sha256",
        "n20_gate_manifest_sha256_token",
        "registration_sha256", "formal_leps_optimizer_sha256",
        "formal_leps_registry_sha256", "q_matrix", "q_sha256",
        "recovery_definition", "recovery_chord_coherence_rule",
        "recovery_chord_coherence_strong_weight",
        "recovery_chord_coherence_conservative_weight",
        "accepted_recovery_secant_curvature_threshold",
        "recovery_secant_normalized_curvature_formula",
        "recovery_secant_strong_publication",
        "recovery_secant_conservative_publication",
        "recovery_secant_ordinary_publication",
        "recovery_secant_tie_break",
        "stationarity_definition",
        "recovery_acceptance_reference", "direct_acceptance_reference",
        "recovery_escape_merit",
        "optimizer_reads_v2p20_opponent", "optimizer_reads_reference_barrier",
        "optimizer_reads_densified_delta_e", "recovery_reads_main_r",
        "raw_recovery_path_charged", "projected_recovery_path_charged",
        "terminal_article_nf", "last_complete_article_nf", "blocked_misses",
        "recovery_backtrack_state_remaining", "per_image_joint_evaluations",
        "transformed_initial_path", "article_nf", "total_force_evaluations",
        "article_usable_state", "outer", "state_phase", "path_history",
        "energy_history", "gradient_history", "direct_trial_nf",
        "direct_trial_path_history", "direct_trial_energy_history",
        "direct_trial_gradient_history", "transaction_json",
        "main_stage_audit_json", "evaluation_miss_trace_json",
        "stop_reason", "failure_message",
    }
    require(required_fields.issubset(archive), "result archive lacks: " + ", ".join(sorted(required_fields - set(archive))))
    for name, value in archive.items():
        if np.issubdtype(value.dtype, np.floating):
            require(np.all(np.isfinite(value)), "result floating array is non-finite: " + name)

    require(str(scalar(archive, "experiment_id")) == EXPERIMENT_ID, "experiment ID drifted")
    require(
        str(scalar(archive, "n20_gate_manifest_sha256_token"))
        == expected_gate_token,
        "result staged-resolution gate token drifted",
    )
    require(str(scalar(archive, "run_id")) == control.run_id, "result run ID drifted")
    try:
        config = json.loads(str(scalar(archive, "config_json")))
    except Exception as error:
        raise IntegrityError("config_json is invalid: " + str(error))
    require(config == control.as_dict(), "result controls drifted")
    current_core_sha = sha256(CORE_SOURCE)
    registration_sha = sha256(REGISTRATION)
    require(registration.get("implementation_sha256") == current_core_sha, "registration/core hash mismatch")
    require(str(scalar(archive, "source_sha256")) == current_core_sha, "result source hash drifted")
    require(str(scalar(archive, "registration_sha256")) == registration_sha, "result registration hash drifted")
    require(str(scalar(archive, "formal_leps_optimizer_sha256")) == sha256(FORMAL_LEPS_OPTIMIZER), "formal LEPS optimizer hash drifted in result")
    require(str(scalar(archive, "formal_leps_registry_sha256")) == sha256(FORMAL_LEPS_REGISTRY), "formal LEPS registry hash drifted in result")
    require(bool(scalar(archive, "optimizer_reads_v2p20_opponent")) is False, "optimizer read the prior opponent")
    require(bool(scalar(archive, "optimizer_reads_reference_barrier")) is False, "optimizer read reference Delta E")
    require(bool(scalar(archive, "optimizer_reads_densified_delta_e")) is False, "optimizer read densified Delta E")
    require(bool(scalar(archive, "recovery_reads_main_r")) is False, "result says recovery read main r")
    require(
        str(scalar(archive, "recovery_definition"))
        == RECOVERY_DEFINITION,
        "result recovery definition drifted",
    )
    require(
        float(scalar(archive, "accepted_recovery_secant_curvature_threshold"))
        == SECANT_CURVATURE_THRESHOLD,
        "result accepted recovery secant curvature threshold drifted",
    )
    require(
        str(scalar(archive, "recovery_secant_normalized_curvature_formula"))
        == SECANT_NORMALIZED_CURVATURE_FORMULA,
        "result normalized-curvature formula drifted",
    )
    require(
        str(scalar(archive, "recovery_secant_strong_publication"))
        == SECANT_STRONG_PUBLICATION,
        "result strong secant publication rule drifted",
    )
    require(
        str(scalar(archive, "recovery_secant_conservative_publication"))
        == SECANT_ALL_PUBLICATION
        and str(scalar(archive, "recovery_secant_ordinary_publication"))
        == SECANT_ALL_PUBLICATION,
        "result all-certified secant publication rule drifted",
    )
    require(
        str(scalar(archive, "recovery_secant_tie_break"))
        == SECANT_TIE_BREAK,
        "result secant ranking tie break drifted",
    )
    require(
        str(scalar(archive, "recovery_chord_coherence_rule"))
        == "(m_max-1)*sqrt(1-dot(u_minus,u_plus)) <= 1",
        "result chord-coherence rule drifted",
    )
    require(
        str(scalar(archive, "recovery_chord_coherence_strong_weight"))
        == "m_max-1"
        and int(
            scalar(archive, "recovery_chord_coherence_conservative_weight")
        )
        == 1,
        "result chord-coherence recovery horizons drifted",
    )
    require(
        str(scalar(archive, "stationarity_definition"))
        == "acceptable unchanged direct band converges only after ordinary normal recovery exhausts its m_max slots",
        "result stationarity definition drifted",
    )
    require(
        str(scalar(archive, "recovery_acceptance_reference"))
        == "post-main I_perp or positive actual proximal merit",
        "result recovery acceptance reference drifted",
    )
    require(str(scalar(archive, "direct_acceptance_reference")) == "post-main I_perp", "result direct acceptance reference drifted")
    require(
        str(scalar(archive, "recovery_escape_merit"))
        == "sum_internal(E_direct-E_projected-||x_projected-x_direct||^2/(2*alpha))",
        "result recovery escape-merit identity drifted",
    )
    require(bool(scalar(archive, "raw_recovery_path_charged")) is False, "raw recovery was marked charged")
    require(bool(scalar(archive, "projected_recovery_path_charged")) is True, "projected recovery was marked uncharged")

    q_matrix = np.asarray(archive["q_matrix"], dtype=np.float64)
    initial_path = np.asarray(archive["transformed_initial_path"], dtype=np.float64)
    require(q_matrix.shape == (10, 10), "result Q shape drifted")
    require(np.array_equal(q_matrix, article["q_matrix"]), "result Q differs from Article IVM")
    require(np.array_equal(initial_path, article["initial_path"]), "result initial path differs from Article IVM")
    q_sha = hashlib.sha256(np.ascontiguousarray(q_matrix).tobytes()).hexdigest()
    require(str(scalar(archive, "q_sha256")) == q_sha, "result Q hash field drifted")

    n_internal = int(control.n_internal)
    total_images = n_internal + 2
    paths = np.asarray(archive["path_history"], dtype=np.float64)
    energies = np.asarray(archive["energy_history"], dtype=np.float64)
    gradients = np.asarray(archive["gradient_history"], dtype=np.float64)
    nof = np.asarray(archive["article_nf"], dtype=np.int64)
    total_nof = np.asarray(archive["total_force_evaluations"], dtype=np.int64)
    usable = np.asarray(archive["article_usable_state"], dtype=bool)
    outer = np.asarray(archive["outer"], dtype=np.int64)
    phases = np.asarray(archive["state_phase"]).astype(str)
    expected_shape = (paths.shape[0], total_images, 10)
    require(paths.shape == expected_shape and gradients.shape == expected_shape, "accepted path/gradient shape drifted")
    require(energies.shape == expected_shape[:2], "accepted energy shape drifted")
    require(nof.shape == (paths.shape[0],), "accepted NoF shape drifted")
    require(np.array_equal(nof, total_nof), "accepted NoF aliases differ")
    require(usable.shape == nof.shape and np.all(usable), "accepted usable-state mask drifted")
    require(outer.shape == nof.shape and phases.shape == nof.shape, "accepted metadata shape drifted")
    require(nof.size > 0 and np.all(np.diff(nof) > 0), "accepted NoF is not strictly increasing")
    require(int(nof[0]) == n_internal, "initial candidate NoF is not N_internal")
    require(np.all((nof >= 0) & (nof <= ARTICLE_NF_BUDGET)), "accepted NoF lies outside budget")
    require(phases[0] == "initial" and np.count_nonzero(phases == "initial") == 1, "initial phase ledger drifted")

    terminal = int(scalar(archive, "terminal_article_nf"))
    last_complete = int(scalar(archive, "last_complete_article_nf"))
    blocked_misses = int(scalar(archive, "blocked_misses"))
    per_image = np.asarray(archive["per_image_joint_evaluations"], dtype=np.int64)
    require(per_image.shape == (total_images,), "per-image NoF shape drifted")
    require(np.all(per_image >= 0), "per-image NoF contains negatives")
    require(per_image[0] == 0 and per_image[-1] == 0, "per-image NoF charges endpoints")
    require(int(np.sum(per_image)) == terminal, "terminal NoF does not equal per-image ledger")
    require(int(nof[-1]) == last_complete, "last complete NoF differs from accepted ledger")
    require(last_complete == terminal, "terminal held public state is missing from accepted ledger")
    require(0 <= last_complete <= terminal <= ARTICLE_NF_BUDGET, "terminal NoF is outside budget")
    require(blocked_misses >= 0, "blocked miss count is negative")
    require(bool(scalar(archive, "recovery_backtrack_state_remaining")) is False, "private recovery state leaked at termination")

    direct_nf = np.asarray(archive["direct_trial_nf"], dtype=np.int64)
    direct_paths = np.asarray(archive["direct_trial_path_history"], dtype=np.float64)
    direct_energies = np.asarray(archive["direct_trial_energy_history"], dtype=np.float64)
    direct_gradients = np.asarray(archive["direct_trial_gradient_history"], dtype=np.float64)
    direct_shape = (direct_nf.size, total_images, 10)
    require(direct_paths.shape == direct_shape and direct_gradients.shape == direct_shape, "direct path/gradient shape drifted")
    require(direct_energies.shape == direct_shape[:2], "direct energy shape drifted")
    if direct_nf.size:
        # A bitwise-identical direct anchor may be served entirely from the
        # exact-coordinate cache, so this auxiliary ledger is nondecreasing,
        # unlike the deduplicated public-state ledger above.
        require(np.all(np.diff(direct_nf) >= 0), "direct NoF decreases")
        require(np.all((direct_nf >= n_internal) & (direct_nf <= terminal)), "direct NoF lies outside ledger")

    transactions = load_json_lines(archive["transaction_json"], "transaction_json")
    main_audits = load_json_lines(archive["main_stage_audit_json"], "main_stage_audit_json")
    evaluation_miss_trace = load_json_lines(
        archive["evaluation_miss_trace_json"], "evaluation_miss_trace_json"
    )
    require(len(transactions) <= direct_nf.size, "more transactions than complete direct trials")
    trailing_direct = int(direct_nf.size - len(transactions))
    stop_reason = str(scalar(archive, "stop_reason"))
    failure_message = str(scalar(archive, "failure_message"))
    require(
        stop_reason in ELIGIBLE_STOP_REASONS + ("numerical_failure",),
        "unknown stop reason: " + stop_reason,
    )
    require(
        (stop_reason == "numerical_failure") == bool(failure_message),
        "stop reason and failure message are inconsistent",
    )
    require(trailing_direct in (0, 1), "unexplained count of trailing direct trials")
    if trailing_direct:
        require(stop_reason == "numerical_failure", "trailing direct trial without numerical failure")
    require(len(main_audits) - len(transactions) in (0, 1), "main-stage/transaction ledger misalignment")
    for stage, transaction in zip(main_audits, transactions):
        require(
            int(stage.get("outer", -1)) == int(transaction.get("outer", -2)),
            "main stage and transaction refer to different outer iterations",
        )
        require_close(
            stage.get("post_main_i_perp"),
            transaction.get("post_main_i_perp"),
            "main-stage/transaction post-main I_perp",
        )
    ledger = audit_main_and_transaction_ledgers(control, main_audits, transactions)

    initial_vector = np.zeros(total_images, dtype=np.int64)
    initial_vector[1:-1] = 1
    attributed_vector = initial_vector + ledger["main_vector"] + ledger["transaction_vector"]
    partial_tail_vector = per_image - attributed_vector
    require(np.all(partial_tail_vector >= 0), "component NoF exceeds per-image ledger")
    partial_tail = int(np.sum(partial_tail_vector))
    require(int(np.sum(attributed_vector)) + partial_tail == terminal, "component NoF closure failed")
    main_transaction_gap = int(len(main_audits) - len(transactions))
    maximum_main_tail = n_internal * max(12, 4 * int(control.m_max))
    if main_transaction_gap == 0:
        # Every completed main stage has a transaction.  Any unitemized calls
        # can only belong to the next, interrupted main solve; no complete
        # direct band may be trailing in this case.
        require(trailing_direct == 0, "trailing direct trial without a trailing main stage")
        partial_tail_class = "none" if partial_tail == 0 else "partial_main_stage"
        if partial_tail:
            require(stop_reason in ("budget", "numerical_failure"), "partial main tail without interruption")
            require(partial_tail <= maximum_main_tail, "partial main tail exceeds one complete main stage")
    else:
        # One complete main stage has no transaction, hence termination must
        # have occurred in its direct/recovery portion.
        require(stop_reason in ("budget", "numerical_failure"), "trailing main stage without interruption")
        if trailing_direct == 0:
            partial_tail_class = "partial_direct_band"
            require(partial_tail < n_internal, "partial direct tail reached a complete direct band")
        else:
            partial_tail_class = "complete_direct_plus_partial_recovery_or_failure"
            require(stop_reason == "numerical_failure", "complete trailing direct band is not a numerical failure")
            require(partial_tail <= 2 * n_internal, "direct plus recovery tail exceeds two bands")
    if ledger["interrupted_transactions"]:
        require(ledger["interrupted_transactions"] == 1, "more than one interrupted recovery transaction")
        require(bool(transactions[-1].get("interrupted")), "interrupted recovery is not the final transaction")
        require(stop_reason == "budget", "interrupted recovery did not terminate at budget")
        require(partial_tail == 0, "interrupted recovery calls were not closed in its transaction")
    if stop_reason == "budget":
        require(terminal == ARTICLE_NF_BUDGET, "budget stop did not reach the exact NoF budget")

    require(receipt.get("stop_reason") == stop_reason, "receipt/NPZ stop reason mismatch")
    require(receipt.get("failure_message", "") == failure_message, "receipt/NPZ failure message mismatch")
    for key, expected in (
        ("terminal_article_nf", terminal),
        ("last_complete_article_nf", last_complete),
        ("accepted_states", int(paths.shape[0])),
        ("transactions", len(transactions)),
        ("committed_transactions", ledger["committed_transactions"]),
        ("rolled_back_transactions", ledger["rolled_back_transactions"]),
    ):
        require(int(receipt.get(key, -1)) == int(expected), "receipt field drifted: " + key)

    valid_for_selection = stop_reason in ELIGIBLE_STOP_REASONS and failure_message == ""
    if stop_reason == "numerical_failure":
        valid_for_selection = False
    require(
        receipt.get("status") in (("failed",) if stop_reason == "numerical_failure" else ("completed", "existing_validated")),
        "receipt status does not match stop reason",
    )

    max_energy_error, max_gradient_error = verify_stored_energy_gradient(
        metric, paths, energies, gradients, q_matrix, "accepted trajectory"
    )
    max_direct_energy_error, max_direct_gradient_error = verify_stored_energy_gradient(
        metric, direct_paths, direct_energies, direct_gradients, q_matrix, "direct trials"
    )
    miss_trace_audit = audit_evaluation_miss_trace(
        evaluation_miss_trace,
        n_internal,
        initial_path,
        main_audits,
        transactions,
        direct_paths,
        metric,
        terminal,
        stop_reason,
    )
    audit_stop_reason_semantics(
        stop_reason,
        failure_message,
        terminal,
        blocked_misses,
        main_audits,
        transactions,
        direct_nf.size,
        control.m_max,
        miss_trace_audit,
        partial_tail_class,
    )
    gate_audit = audit_recovery_gates(
        metric,
        control,
        main_audits,
        transactions,
        direct_nf,
        direct_paths,
        direct_energies,
        direct_gradients,
        initial_path,
        nof,
        paths,
        energies,
        gradients,
        stop_reason,
        terminal,
        q_matrix,
        miss_trace_audit,
    )

    checkpoints = article["checkpoints"]
    indices = locf_indices(nof, checkpoints)
    unique, inverse = np.unique(indices, return_inverse=True)
    unique_i, unique_delta, unique_error = metric_values(metric, paths[unique], q_matrix)
    candidate_i = unique_i[inverse]
    candidate_delta = unique_delta[inverse]
    candidate_error = unique_error[inverse]
    article_i = article["checkpoint_i_perp"]
    article_error = article["checkpoint_delta_e_error"]
    strict_i = candidate_i < article_i
    strict_error = candidate_error < article_error
    strict_both = strict_i & strict_error
    i_ratio = safe_ratio(candidate_i, article_i)
    error_ratio = safe_ratio(candidate_error, article_error)
    joint_ratio = np.maximum(i_ratio, error_ratio)
    strict_pass = bool(valid_for_selection and np.all(strict_both))

    summary = {
        **control.as_dict(),
        "run_id": control.run_id,
        "integrity_valid": True,
        "integrity_error": "",
        "source": str(source),
        "source_sha256": source_sha,
        "receipt": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "stop_reason": stop_reason,
        "failure_message": failure_message,
        "valid_for_selection": bool(valid_for_selection),
        "accepted_states": int(paths.shape[0]),
        "terminal_number_of_forces": terminal,
        "last_complete_number_of_forces": last_complete,
        "transactions": len(transactions),
        "committed_transactions": ledger["committed_transactions"],
        "rolled_back_transactions": ledger["rolled_back_transactions"],
        "interrupted_transactions": ledger["interrupted_transactions"],
        "audited_initial_force_calls": n_internal,
        "audited_complete_main_force_calls": ledger["main_calls"],
        "audited_complete_transaction_force_calls": ledger["transaction_calls"],
        "audited_unattributed_interrupted_tail_force_calls": partial_tail,
        "audited_unattributed_tail_class": partial_tail_class,
        "per_image_number_of_forces_closes": True,
        "i_perp_strictly_better_checkpoints": int(np.sum(strict_i)),
        "delta_e_error_strictly_better_checkpoints": int(np.sum(strict_error)),
        "both_strictly_better_checkpoints": int(np.sum(strict_both)),
        "strict_pass": strict_pass,
        "worst_i_perp_ratio": float(np.max(i_ratio)),
        "worst_delta_e_error_ratio": float(np.max(error_ratio)),
        "worst_joint_ratio": float(np.max(joint_ratio)),
        "mean_joint_ratio": float(np.mean(joint_ratio)),
        "max_stored_energy_error": float(max_energy_error),
        "max_stored_gradient_error": float(max_gradient_error),
        "max_direct_energy_error": float(max_direct_energy_error),
        "max_direct_gradient_error": float(max_direct_gradient_error),
        **gate_audit,
    }
    checkpoint_rows = []
    for index, checkpoint in enumerate(checkpoints):
        checkpoint_rows.append(
            {
                "n_internal": n_internal,
                "alpha": float(control.alpha),
                "alpha_label": control.alpha_label,
                "r": float(control.r),
                "m_max": int(control.m_max),
                "run_id": control.run_id,
                "checkpoint_number_of_forces": int(checkpoint),
                "candidate_state_number_of_forces": int(nof[indices[index]]),
                "article_state_number_of_forces": int(article["nof"][article["checkpoint_indices"][index]]),
                "candidate_i_perp": float(candidate_i[index]),
                "article_i_perp": float(article_i[index]),
                "i_perp_ratio": float(i_ratio[index]),
                "candidate_densified_delta_e": float(candidate_delta[index]),
                "article_densified_delta_e": float(article["checkpoint_delta_e"][index]),
                "candidate_delta_e_error": float(candidate_error[index]),
                "article_delta_e_error": float(article_error[index]),
                "delta_e_error_ratio": float(error_ratio[index]),
                "i_perp_strictly_better": bool(strict_i[index]),
                "delta_e_error_strictly_better": bool(strict_error[index]),
                "both_strictly_better": bool(strict_both[index]),
            }
        )
    return summary, checkpoint_rows


def ranking_key(item):
    return (
        not bool(item.get("valid_for_selection")),
        not bool(item.get("strict_pass")),
        float(item.get("worst_joint_ratio", np.finfo(np.float64).max)),
        float(item.get("mean_joint_ratio", np.finfo(np.float64).max)),
        int(item.get("terminal_number_of_forces", ARTICLE_NF_BUDGET + 1)),
        float(item["alpha"]),
        float(item["r"]),
        int(item["m_max"]),
        str(item["run_id"]),
    )


SUMMARY_FIELDS = (
    "n_internal", "alpha", "alpha_label", "r", "m_max", "run_id", "integrity_valid",
    "integrity_error", "source", "source_sha256", "receipt", "receipt_sha256",
    "stop_reason", "failure_message", "valid_for_selection", "accepted_states",
    "terminal_number_of_forces", "last_complete_number_of_forces", "transactions",
    "committed_transactions", "rolled_back_transactions", "interrupted_transactions",
    "audited_initial_force_calls", "audited_complete_main_force_calls",
    "audited_complete_transaction_force_calls",
    "audited_unattributed_interrupted_tail_force_calls",
    "audited_unattributed_tail_class",
    "per_image_number_of_forces_closes", "i_perp_strictly_better_checkpoints",
    "delta_e_error_strictly_better_checkpoints",
    "both_strictly_better_checkpoints", "strict_pass", "worst_i_perp_ratio",
    "worst_delta_e_error_ratio", "worst_joint_ratio", "mean_joint_ratio",
    "max_stored_energy_error", "max_stored_gradient_error",
    "max_direct_energy_error", "max_direct_gradient_error",
    "projected_trials", "projected_commits", "monotone_projected_commits",
    "merit_only_projected_commits", "rejected_projected_trials",
    "direct_commits", "max_projected_energy_error",
    "max_post_main_energy_error", "max_post_main_gradient_error",
    "max_public_path_movement_error",
    "authenticated_public_state_chain_events",
    "stationary_probes", "strong_chord_coherent_seeds",
    "conservative_chord_coherent_seeds",
    "max_projected_gradient_error", "max_candidate_i_perp_error",
    "max_candidate_equidistance_error", "max_actual_proximal_merit_error",
    "accepted_recovery_secants_certified_curvature",
    "accepted_recovery_secants_curvature_rejected",
    "accepted_recovery_secants_rank_omitted",
    "accepted_recovery_secants_published",
    "strong_top1_publication_commits",
    "conservative_all_publication_commits",
    "ordinary_all_publication_commits",
    "max_published_normalized_curvature_error",
)


CHECKPOINT_FIELDS = (
    "n_internal", "alpha", "alpha_label", "r", "m_max", "run_id",
    "checkpoint_number_of_forces", "candidate_state_number_of_forces",
    "article_state_number_of_forces", "candidate_i_perp", "article_i_perp",
    "i_perp_ratio", "candidate_densified_delta_e",
    "article_densified_delta_e", "candidate_delta_e_error",
    "article_delta_e_error", "delta_e_error_ratio",
    "i_perp_strictly_better", "delta_e_error_strictly_better",
    "both_strictly_better",
)


def invalid_summary(control, error):
    return {
        **control.as_dict(),
        "run_id": control.run_id,
        "integrity_valid": False,
        "integrity_error": str(error),
        "valid_for_selection": False,
        "strict_pass": False,
    }


def build_markdown(manifest, summaries):
    lines = [
        "# QM10 low-r coherence-conditioned IR-IVM result audit",
        "",
        "- Campaign: `{}`".format(EXPERIMENT_ID),
        "- Audit status: **{}**".format(manifest["audit_status"]),
        "- Registered runs requested: {}".format(manifest["requested_member_count"]),
        "- Integrity-valid results: {}".format(manifest["integrity_valid_member_count"]),
        "- Integrity failures: {}".format(manifest["integrity_failure_count"]),
        "- Strict rule: at every one of 51 latter-half same-NoF LOCF checkpoints, both $I_\\perp$ and $|\\Delta E-\\Delta E_{\\rm ref}|$ must be exactly lower than frozen Article IVM; equality fails.",
        "- Recovery audit: every complete projected candidate's path, energies, and gradients, the strong/conservative/ordinary branch, stationarity, curvature-certified and curvature-rejected sets, strong-branch rank omissions, published indices, normalized curvatures, and exact tie break are independently recomputed. Optimizer publication helpers and reported scores are not trusted.",
        "- Reference densified barrier: `{:.14f}`".format(REFERENCE_DELTA_E),
        "",
        "## Outcome",
        "",
    ]
    for n_internal in manifest["requested_resolutions"]:
        result = manifest["resolution_results"].get(str(n_internal), {})
        best = result.get("best_available")
        lines.append(
            "- $N_{{\\rm int}}={}$: {} strict passing run(s).".format(
                n_internal, result.get("strict_pass_count", 0)
            )
        )
        if best:
            lines.append(
                "  Best available: `alpha={alpha_label}, r={r:g}, m_max={m_max}` "
                "(`{run_id}`), {both_strictly_better_checkpoints}/51 joint checkpoint wins, "
                "worst joint ratio {worst_joint_ratio:.6g}.".format(**best)
            )
    totals = manifest["recovery_secant_publication_audit"][
        "campaign_totals_over_integrity_valid_results"
    ]
    lines.extend(
        [
            "",
            "## Recovery secant publication totals",
            "",
            "- Curvature-certified: `{}`; published: `{}`; curvature-rejected: `{}`; rank-omitted: `{}`.".format(
                totals["certified_curvature"],
                totals["published"],
                totals["curvature_rejected"],
                totals["rank_omitted"],
            ),
            "- Accepted recovery branches: strong top-1 `{}`, conservative all `{}`, ordinary all `{}`.".format(
                totals["strong_top1_commits"],
                totals["conservative_all_commits"],
                totals["ordinary_all_commits"],
            ),
        ]
    )
    lines.extend(["", "## Integrity", ""])
    if manifest["integrity_errors"]:
        for item in manifest["integrity_errors"]:
            lines.append("- `{}`: {}".format(item["run_id"], item["message"]))
    else:
        lines.append("All requested registered results, receipts, hashes, finite-value checks, and Number-of-Forces ledgers passed.")
    lines.extend(
        [
            "",
            "A lack of a strict winner is reported as a scientific non-pass, not as a program error. Only integrity failures make this audit exit non-zero.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-internal",
        type=int,
        choices=(10, 20),
        action="append",
        help="audit one resolution; repeat for both (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CAMPAIGN_ROOT / "reports/result_audit",
    )
    args = parser.parse_args()
    requested_resolutions = sorted(set(args.n_internal or (10, 20)))
    output_dir = Path(args.output_dir)
    summaries = []
    checkpoint_rows = []
    integrity_errors = []
    registration = None
    metric = None
    articles = {}
    dependency_rows = []
    gate_tokens = {20: "not_applicable"}

    try:
        registration = validate_preregistration()
        validate_low_r_registration_contract(registration)
        require(sha256(CORE_SOURCE) == registration["implementation_sha256"], "registered core source hash drifted")
        require(
            sha256(METRIC_SOURCE) == registration["dependency_sha256"].get(str(METRIC_SOURCE)),
            "registered offline metric hash drifted",
        )
        for absolute, expected in sorted(registration["dependency_sha256"].items()):
            path = Path(absolute)
            observed = sha256(path) if path.is_file() else None
            dependency_rows.append(
                {
                    "path": absolute,
                    "registered_sha256": expected,
                    "observed_sha256": observed,
                    "valid": observed == expected,
                }
            )
            require(observed == expected, "registered dependency hash drifted: " + absolute)
        metric = load_metric_module()
        for n_internal in requested_resolutions:
            articles[n_internal] = load_article(metric, n_internal, registration)
        if 10 in requested_resolutions:
            gate_tokens[10], _ = validate_n20_gate_manifest_for_n10(
                registration
            )
    except Exception as error:
        integrity_errors.append(
            {"run_id": "campaign", "message": "{}: {}".format(type(error).__name__, error)}
        )

    controls = [Controls(*item) for item in REGISTERED_CONTROLS if int(item[0]) in requested_resolutions]
    if not integrity_errors:
        for index, control in enumerate(controls, start=1):
            try:
                summary, rows = audit_candidate(
                    metric,
                    registration,
                    control,
                    articles[control.n_internal],
                    gate_tokens[control.n_internal],
                )
                summaries.append(summary)
                checkpoint_rows.extend(rows)
            except Exception as error:
                message = "{}: {}".format(type(error).__name__, error)
                summaries.append(invalid_summary(control, message))
                integrity_errors.append({"run_id": control.run_id, "message": message})
            if index == 1 or index % 25 == 0 or index == len(controls):
                print(
                    json.dumps(
                        {
                            "audited": index,
                            "total": len(controls),
                            "integrity_failures": len(integrity_errors),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        # Close the provenance window: detect a source, dependency, result, or
        # receipt changed while the potentially long offline metric audit was
        # running.  Such a candidate is removed from selection before winners
        # are computed.
        final_provenance_ok = True
        try:
            final_registration = validate_preregistration()
            validate_low_r_registration_contract(final_registration)
            if 10 in requested_resolutions:
                final_gate_token, _ = validate_n20_gate_manifest_for_n10(
                    final_registration
                )
                require(
                    final_gate_token == gate_tokens[10],
                    "N20 gate manifest changed during N10 audit",
                )
        except Exception as error:
            final_provenance_ok = False
            integrity_errors.append(
                {
                    "run_id": "campaign-final-provenance",
                    "message": "{}: {}".format(type(error).__name__, error),
                }
            )
        for summary in summaries:
            if not summary.get("integrity_valid"):
                continue
            if not final_provenance_ok:
                summary["integrity_valid"] = False
                summary["integrity_error"] = "campaign provenance changed during audit"
                summary["valid_for_selection"] = False
                summary["strict_pass"] = False
                continue
            try:
                require(
                    sha256(Path(summary["source"])) == summary["source_sha256"],
                    "result changed during audit",
                )
                require(
                    sha256(Path(summary["receipt"])) == summary["receipt_sha256"],
                    "receipt changed during audit",
                )
            except Exception as error:
                message = "{}: {}".format(type(error).__name__, error)
                summary["integrity_valid"] = False
                summary["integrity_error"] = message
                summary["valid_for_selection"] = False
                summary["strict_pass"] = False
                integrity_errors.append(
                    {"run_id": summary["run_id"], "message": message}
                )

    resolution_results = {}
    for n_internal in requested_resolutions:
        valid = [
            item for item in summaries
            if item.get("integrity_valid") and int(item["n_internal"]) == n_internal
        ]
        selectable = [item for item in valid if item.get("valid_for_selection")]
        passing = [item for item in selectable if item.get("strict_pass")]
        best = min(selectable, key=ranking_key) if selectable else None
        winner = min(passing, key=ranking_key) if passing else None
        resolution_results[str(n_internal)] = {
            "registered_member_count": sum(int(item[0]) == n_internal for item in REGISTERED_CONTROLS),
            "integrity_valid_member_count": len(valid),
            "selectable_member_count": len(selectable),
            "strict_pass_count": len(passing),
            "strict_winner": winner,
            "best_available": best,
        }

    all_requested_resolutions_pass = bool(
        len(requested_resolutions) > 0
        and all(
            resolution_results[str(n_internal)]["strict_pass_count"] > 0
            for n_internal in requested_resolutions
        )
    )
    audit_status = "integrity_failure" if integrity_errors else "complete"
    manifest = {
        "campaign_id": EXPERIMENT_ID,
        "audit_status": audit_status,
        "audit_script": str(SCRIPT_FILE),
        "audit_script_sha256": sha256(SCRIPT_FILE),
        "registration": str(REGISTRATION),
        "registration_sha256": sha256(REGISTRATION) if REGISTRATION.is_file() else None,
        "core_source": str(CORE_SOURCE),
        "core_source_sha256": sha256(CORE_SOURCE),
        "metric_source": str(METRIC_SOURCE),
        "metric_source_sha256": sha256(METRIC_SOURCE),
        "reference_delta_e": REFERENCE_DELTA_E,
        "number_of_forces_budget": ARTICLE_NF_BUDGET,
        "requested_resolutions": requested_resolutions,
        "requested_member_count": len(controls),
        "audited_member_count": len(summaries),
        "integrity_valid_member_count": sum(bool(item.get("integrity_valid")) for item in summaries),
        "integrity_failure_count": len(integrity_errors),
        "integrity_errors": integrity_errors,
        "dependency_hash_audit": dependency_rows,
        "staged_resolution_gate": {
            "fixed_n20_manifest": str(EXPECTED_N20_GATE_MANIFEST),
            "n10_manifest_sha256_token": gate_tokens.get(10),
            "n20_token": "not_applicable",
            "independently_authenticated_for_n10": bool(
                10 not in requested_resolutions
                or 10 in gate_tokens
            ),
        },
        "energy_gradient_recomputation_performed": True,
        "checkpoint_rule": {
            "same_nof": "last complete public state at or below checkpoint (LOCF), including carry-forward after stopping",
            "strict_equality_fails": True,
            "count_per_resolution": CHECKPOINT_COUNT,
            "ranges": {"10": [481, 951], "20": [1199, 2377]},
        },
        "recovery_gate_audit": {
            "reads_r": False,
            "actual_proximal_merit": "sum_internal(E_direct-E_projected-||x_projected-x_direct||^2/(2*alpha))",
            "atomic_acceptance": "finite and changed and equidistant and (post-main-I_perp-monotone or positive actual proximal merit)",
            "complete_projected_candidate_path_energy_gradient_recomputed": True,
            "post_main_state_and_public_history_chain_recomputed": True,
            "per_outer_number_of_forces_recomputed": True,
            "merit_only_commit_requires_strictly_positive_actual_merit": True,
            "history_window": False,
        },
        "recovery_secant_publication_audit": {
            "branch_recomputed_from_geometry_and_stationarity": True,
            "stationarity_recomputed_from_outer_start_and_direct_paths": True,
            "optimizer_publisher_called_by_audit": False,
            "normalized_curvature_formula": SECANT_NORMALIZED_CURVATURE_FORMULA,
            "curvature_threshold": SECANT_CURVATURE_THRESHOLD,
            "strong_publication": SECANT_STRONG_PUBLICATION,
            "conservative_publication": SECANT_ALL_PUBLICATION,
            "ordinary_publication": SECANT_ALL_PUBLICATION,
            "tie_break": SECANT_TIE_BREAK,
            "curvature_rejected_separate_from_rank_omitted": True,
            "campaign_totals_over_integrity_valid_results": {
                "certified_curvature": int(
                    sum(
                        int(item.get("accepted_recovery_secants_certified_curvature", 0))
                        for item in summaries
                        if item.get("integrity_valid")
                    )
                ),
                "published": int(
                    sum(
                        int(item.get("accepted_recovery_secants_published", 0))
                        for item in summaries
                        if item.get("integrity_valid")
                    )
                ),
                "curvature_rejected": int(
                    sum(
                        int(item.get("accepted_recovery_secants_curvature_rejected", 0))
                        for item in summaries
                        if item.get("integrity_valid")
                    )
                ),
                "rank_omitted": int(
                    sum(
                        int(item.get("accepted_recovery_secants_rank_omitted", 0))
                        for item in summaries
                        if item.get("integrity_valid")
                    )
                ),
                "strong_top1_commits": int(
                    sum(
                        int(item.get("strong_top1_publication_commits", 0))
                        for item in summaries
                        if item.get("integrity_valid")
                    )
                ),
                "conservative_all_commits": int(
                    sum(
                        int(item.get("conservative_all_publication_commits", 0))
                        for item in summaries
                        if item.get("integrity_valid")
                    )
                ),
                "ordinary_all_commits": int(
                    sum(
                        int(item.get("ordinary_all_publication_commits", 0))
                        for item in summaries
                        if item.get("integrity_valid")
                    )
                ),
            },
        },
        "resolution_results": resolution_results,
        "all_requested_resolutions_strictly_pass": all_requested_resolutions_pass,
        "process_exit_semantics": {
            "no_strict_winner": 0,
            "integrity_failure": 2,
        },
    }

    summary_path = output_dir / "candidate_summary.csv"
    checkpoint_path = output_dir / "checkpoint_comparison.csv"
    report_path = output_dir / "AUDIT_REPORT.md"
    manifest_path = output_dir / "audit_manifest.json"
    atomic_csv(summary_path, summaries, SUMMARY_FIELDS)
    atomic_csv(checkpoint_path, checkpoint_rows, CHECKPOINT_FIELDS)
    manifest["artifacts"] = {
        "candidate_summary_csv": {
            "path": str(summary_path),
            "sha256": sha256(summary_path),
        },
        "checkpoint_comparison_csv": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
        },
    }
    atomic_text(report_path, build_markdown(manifest, summaries))
    manifest["artifacts"]["human_report"] = {
        "path": str(report_path),
        "sha256": sha256(report_path),
    }
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "audit_status": audit_status,
                "integrity_failures": len(integrity_errors),
                "all_requested_resolutions_strictly_pass": all_requested_resolutions_pass,
                "report": str(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 2 if integrity_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
