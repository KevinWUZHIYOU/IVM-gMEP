"""Complete independent integrity/scientific audit for frozen QM10 runs.

This source deliberately imports neither the optimizer nor its runtime.  The
potential, HJ tangent, spring recurrence, and two Article metrics are repeated
independently here.  The registered metric module is authenticated but never
executed by this auditor.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .formal_io import (
    FormalIOError,
    atomic_output_path,
    atomic_write_bytes,
    atomic_write_json,
    derive_campaign_paths,
    load_canonical_json,
    resolve_registered_path,
    sha256_file,
    verify_registration_seal,
)
from .result_schema import ValidationReport, coordinate_sha256, validate_result_archive


DIMENSION = 10
SPRING_DTAU = 0.05
SPRING_MAX_ITER = 500
SPRING_TOLERANCE = 1.0e-5
REFERENCE_BARRIER = 67.50188060819488
LINEAR_DENSE_COUNT = 500
Q_ARRAY_SHA256 = "372962a47504e6df28a20d8b022482a6835680765434b4385a6dba0e826974bc"
EXPECTED_CHECKPOINT_COUNTS = {10: 79, 20: 98}
EXPECTED_REGISTERED_INPUTS = {
    "metric_source": "python/unified_ir_ivm_20260827/src/q10_article_metrics.py",
    "article_n10": (
        "paper/ActiveExperiments/muller_q10_10d/image12/data/frozen_runs/"
        "tuned_ivm_alpha2e3_selected.npz"
    ),
    "article_n20": (
        "paper/ActiveExperiments/muller_q10_10d/data/frozen_runs/"
        "tuned_ivm_alpha1e3_selected.npz"
    ),
    "q_n10": "paper/ActiveExperiments/muller_q10_10d/image12/data/inputs/Q_data.npz",
    "q_n20": "paper/ActiveExperiments/muller_q10_10d/data/inputs/Q_data.npz",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_D = np.asarray([-200.0, -100.0, -170.0, 15.0], dtype=np.float64)
_A = np.asarray([-1.0, -1.0, -6.5, 0.7], dtype=np.float64)
_B = np.asarray([0.0, 0.0, 11.0, 0.6], dtype=np.float64)
_C = np.asarray([-10.0, -10.0, -6.5, 0.7], dtype=np.float64)
_X = np.asarray([1.0, 0.0, -0.5, -1.0], dtype=np.float64)
_Y = np.asarray([0.0, 0.5, 1.5, 1.0], dtype=np.float64)


class FinalAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArticleTrajectory:
    source_path: Path
    nof: np.ndarray
    paths: np.ndarray
    i_perp: np.ndarray
    delta_e: np.ndarray
    maximum_stored_i_perp_error: float
    maximum_stored_normal_force_error: float
    maximum_stored_delta_e_error: float


@dataclass(frozen=True)
class FrozenEnergySourceCandidate:
    """One archived, causally available path that may own frozen energies.

    ``availability_nof`` is the first NoF at which the *whole path state* is
    recorded as available.  This is deliberately different from the NoF at
    which an individual image was first evaluated.  Keeping both notions
    prevents a future public/recovery record from being used to justify an
    earlier spring event.
    """

    kind: str
    index: int
    availability_nof: int
    phase: str
    path: np.ndarray
    stored_energy: np.ndarray | None
    independently_recomputed_energy: np.ndarray


@dataclass(frozen=True)
class AuthenticatedFrozenEnergySource:
    candidate: FrozenEnergySourceCandidate
    maximum_paid_nof: int
    paid_nof_by_internal_image: tuple[int, ...]
    maximum_energy_absolute_error: float
    maximum_potential_energy_absolute_error: float
    maximum_endpoint_energy_absolute_error: float
    maximum_paid_energy_absolute_error: float


def _scalar(archive: Mapping[str, Any], name: str) -> Any:
    if name not in archive:
        raise FinalAuditError(f"missing scalar {name}")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise FinalAuditError(f"{name} is not scalar")
    return value.item()


def _digest(value: Any, label: str) -> str:
    text = str(value)
    if SHA256_RE.fullmatch(text) is None:
        raise FinalAuditError(f"{label} is not a lowercase SHA-256 digest")
    return text


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def q10_energy_gradient(point: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
    value = np.asarray(point, dtype=np.float64).reshape(DIMENSION)
    matrix = np.asarray(q, dtype=np.float64)
    if matrix.shape != (DIMENSION, DIMENSION):
        raise FinalAuditError("Q must have shape (10,10)")
    z = matrix @ value
    dx = z[0] - _X
    dy = z[1] - _Y
    exponent = _A * dx * dx + _B * dx * dy + _C * dy * dy
    if not np.all(np.isfinite(exponent)) or float(np.max(exponent)) > 650.0:
        raise FinalAuditError("unsafe Muller exponent")
    terms = _D * np.exp(exponent)
    energy = float(np.sum(terms) + 2.0 * np.dot(z[2:], z[2:]))
    gradient_z = np.empty(DIMENSION, dtype=np.float64)
    gradient_z[0] = np.sum(terms * (2.0 * _A * dx + _B * dy))
    gradient_z[1] = np.sum(terms * (_B * dx + 2.0 * _C * dy))
    gradient_z[2:] = 4.0 * z[2:]
    gradient = matrix.T @ gradient_z
    if not np.isfinite(energy) or not np.all(np.isfinite(gradient)):
        raise FinalAuditError("non-finite independent potential evaluation")
    return energy, np.asarray(gradient, dtype=np.float64)


def q10_energy_gradient_batch(
    points: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != DIMENSION:
        raise FinalAuditError("potential batch must have shape [K,10]")
    energies = np.empty(values.shape[0], dtype=np.float64)
    gradients = np.empty_like(values)
    for index, point in enumerate(values):
        energies[index], gradients[index] = q10_energy_gradient(point, q)
    return energies, gradients


def hj_improved_tangents(path: np.ndarray, energies: np.ndarray) -> np.ndarray:
    points = np.asarray(path, dtype=np.float64)
    values = np.asarray(energies, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != DIMENSION:
        raise FinalAuditError("HJ path must have shape [images,10]")
    if values.shape != (points.shape[0],):
        raise FinalAuditError("HJ energy/path length mismatch")
    result = np.empty((points.shape[0] - 2, DIMENSION), dtype=np.float64)
    for output, image in enumerate(range(1, points.shape[0] - 1)):
        forward = points[image + 1] - points[image]
        backward = points[image] - points[image - 1]
        previous, current, following = values[image - 1:image + 2]
        if following > current > previous:
            tangent = forward.copy()
        elif following < current < previous:
            tangent = backward.copy()
        else:
            delta_forward = abs(following - current)
            delta_backward = abs(previous - current)
            delta_max = max(delta_forward, delta_backward)
            delta_min = min(delta_forward, delta_backward)
            if following > previous:
                tangent = forward * delta_max + backward * delta_min
            else:
                tangent = forward * delta_min + backward * delta_max
        norm = float(np.linalg.norm(tangent))
        if norm <= 1.0e-14:
            tangent = forward + backward
            norm = float(np.linalg.norm(tangent))
        if norm <= 1.0e-14:
            raise FinalAuditError("degenerate independently computed HJ tangent")
        result[output] = tangent / norm
    return result


def article_path_metric(
    path: np.ndarray, q: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(path, dtype=np.float64)
    energies, gradients = q10_energy_gradient_batch(points, q)
    tangents = hj_improved_tangents(points, energies)
    forces = -gradients[1:-1]
    normal = forces - np.sum(forces * tangents, axis=1)[:, None] * tangents
    coordinate = np.linspace(0.0, 1.0, points.shape[0])
    speed = np.linalg.norm(
        np.gradient(points, coordinate, axis=0, edge_order=2), axis=1
    )
    ds = 1.0 / float(points.shape[0] - 1)
    i_perp = float(np.sum(np.sum(normal * normal, axis=1) * speed[1:-1]) * ds)
    return i_perp, normal, energies, gradients


def linear500_barrier(path: np.ndarray, q: np.ndarray) -> float:
    points = np.asarray(path, dtype=np.float64)
    old = np.linspace(0.0, 1.0, points.shape[0])
    new = np.linspace(0.0, 1.0, LINEAR_DENSE_COUNT)
    dense = np.column_stack(
        [np.interp(new, old, points[:, coordinate]) for coordinate in range(DIMENSION)]
    )
    dense_energy, _ = q10_energy_gradient_batch(dense, q)
    endpoint, _ = q10_energy_gradient(points[0], q)
    return float(np.max(dense_energy) - endpoint)


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise FinalAuditError(f"shape mismatch: {left_array.shape} != {right_array.shape}")
    return float(np.max(np.abs(left_array - right_array))) if left_array.size else 0.0


def _check_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    absolute: float,
    relative: float,
    label: str,
) -> float:
    error = _max_abs(actual, expected)
    scale = max(1.0, float(np.max(np.abs(expected))) if np.asarray(expected).size else 1.0)
    limit = absolute + relative * scale
    if error > limit:
        raise FinalAuditError(f"{label} mismatch: {error:.3e} > {limit:.3e}")
    return error


def _canonical_f64(value: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype="<f8"))


def _domain_array_digest(value: Any, *, domain: str) -> bytes:
    array = _canonical_f64(value)
    digest = hashlib.sha256()
    digest.update(b"q10-formal-array-v2\x00")
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00<f8\x00")
    digest.update(struct.pack(">Q", int(array.ndim)))
    for extent in array.shape:
        digest.update(struct.pack(">Q", int(extent)))
    digest.update(array.tobytes(order="C"))
    return digest.digest()


def _digest_row(array: np.ndarray, index: int) -> bytes:
    return np.asarray(array[index], dtype=np.uint8).tobytes()


def tangent_transcript_sha256(archive: Mapping[str, Any]) -> str:
    contexts = [str(item) for item in np.asarray(archive["tangent_context_names"]).tolist()]
    codes = np.asarray(archive["tangent_context_code"], dtype=np.int64)
    event_ids = np.asarray(archive["tangent_event_id"], dtype=np.int64)
    parents = np.asarray(archive["tangent_parent_reparam_event_id"], dtype=np.int64)
    if any(not item for item in contexts) or len(contexts) != len(set(contexts)):
        raise FinalAuditError("invalid tangent context table")
    if np.any((codes < 0) | (codes >= len(contexts))):
        raise FinalAuditError("tangent context code outside table")
    if "tangent_context_counts" not in archive:
        raise FinalAuditError("compact tangent transcript lacks context counts")
    context_counts = np.asarray(archive["tangent_context_counts"], dtype=np.int64)
    if context_counts.shape != (len(contexts),):
        raise FinalAuditError("tangent context count/table shape mismatch")
    reconstructed = np.bincount(codes, minlength=len(contexts)).astype(np.int64)
    if not np.array_equal(context_counts, reconstructed):
        raise FinalAuditError("tangent context counts differ from event codes")
    if int(_scalar(archive, "tangent_mismatch_count")) != 0:
        raise FinalAuditError("runtime independent HJ checker recorded a mismatch")
    digest = hashlib.sha256(b"q10-tangent-transcript-v2\x00")
    for index, (event_id, parent, code) in enumerate(zip(event_ids, parents, codes)):
        digest.update(int(event_id).to_bytes(8, "big", signed=False))
        digest.update(int(parent).to_bytes(8, "big", signed=True))
        context = contexts[int(code)].encode("utf-8")
        digest.update(len(context).to_bytes(4, "big", signed=False))
        digest.update(context)
        digest.update(_digest_row(archive["tangent_path_sha256"], index))
        digest.update(_digest_row(archive["tangent_energy_sha256"], index))
        digest.update(_digest_row(archive["tangent_output_sha256"], index))
    return digest.hexdigest()


def _strict_componentwise_energy_error(
    actual: Any,
    expected: Any,
    *,
    label: str,
    raise_on_mismatch: bool,
) -> float | None:
    """Compare every energy component with the formal potential tolerance."""

    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape or not (
        np.all(np.isfinite(left)) and np.all(np.isfinite(right))
    ):
        if raise_on_mismatch:
            raise FinalAuditError(f"{label} has a shape or finiteness mismatch")
        return None
    error = np.abs(left - right)
    limit = 2.0e-11 + 2.0e-13 * np.maximum(1.0, np.abs(right))
    if np.any(error > limit):
        if raise_on_mismatch:
            offending = int(np.flatnonzero(error > limit)[0])
            raise FinalAuditError(
                f"{label} component {offending} mismatch: "
                f"{float(error.flat[offending]):.3e} > "
                f"{float(limit.flat[offending]):.3e}"
            )
        return None
    return float(np.max(error)) if error.size else 0.0


def _complete_paid_band_candidates(
    archive: Mapping[str, Any], endpoint_path: np.ndarray, q: np.ndarray,
) -> list[FrozenEnergySourceCandidate]:
    """Recover only unambiguous, contiguous full-band paid evaluations.

    A charged trace does not log cache hits.  Consequently this routine does
    not try to infer a band from a partial group.  Such paths must instead be
    represented by a public or recovery-state record.  A full paid band is a
    contiguous 1..N image sequence with one outer index and one NoF category.
    """

    total_images = int(np.asarray(endpoint_path).shape[0])
    n_internal = total_images - 2
    image = np.asarray(archive["evaluation_image_index"], dtype=np.int64)
    nof = np.asarray(archive["evaluation_nof"], dtype=np.int64)
    outer = np.asarray(archive["evaluation_outer"], dtype=np.int64)
    phase = np.asarray(archive["evaluation_phase"]).astype("U")
    coordinate = np.asarray(archive["evaluation_coordinate"], dtype=np.float64)
    energy = np.asarray(archive["evaluation_energy"], dtype=np.float64)
    if not (
        image.shape == nof.shape == outer.shape == phase.shape == energy.shape
        and coordinate.shape == (image.size, DIMENSION)
    ):
        raise FinalAuditError("paid trace arrays disagree while building source bands")
    expected_images = np.arange(1, n_internal + 1, dtype=np.int64)
    candidates: list[FrozenEnergySourceCandidate] = []
    for start in np.flatnonzero(image == 1).tolist():
        stop = int(start) + n_internal
        if stop > image.size:
            continue
        trace_slice = slice(int(start), stop)
        if not np.array_equal(image[trace_slice], expected_images):
            continue
        if not np.array_equal(
            nof[trace_slice],
            np.arange(int(nof[start]), int(nof[start]) + n_internal, dtype=np.int64),
        ):
            continue
        if not (
            np.all(outer[trace_slice] == outer[start])
            and np.all(phase[trace_slice] == phase[start])
        ):
            continue
        path = np.empty((total_images, DIMENSION), dtype=np.float64)
        path[0] = endpoint_path[0]
        path[-1] = endpoint_path[-1]
        path[1:-1] = coordinate[trace_slice]
        stored_energy = np.empty(total_images, dtype=np.float64)
        stored_energy[:] = np.nan
        stored_energy[1:-1] = energy[trace_slice]
        candidates.append(
            FrozenEnergySourceCandidate(
                kind="paid_complete_band",
                index=int(start),
                availability_nof=int(nof[stop - 1]),
                phase=str(phase[start]),
                path=path,
                stored_energy=stored_energy,
                independently_recomputed_energy=q10_energy_gradient_batch(path, q)[0],
            )
        )
    return candidates


def _frozen_energy_source_candidates(
    archive: Mapping[str, Any], q: np.ndarray,
) -> list[FrozenEnergySourceCandidate]:
    public_paths = np.asarray(archive["public_path_history"], dtype=np.float64)
    if public_paths.ndim != 3 or public_paths.shape[2] != DIMENSION:
        raise FinalAuditError("public paths cannot seed frozen-energy sources")
    public_nf = np.asarray(archive["public_nf"], dtype=np.int64)
    public_phase = np.asarray(
        archive.get("public_phase", np.full(public_paths.shape[0], "public_state", dtype="U"))
    ).astype("U")
    public_energy = np.asarray(archive["public_energy_history"], dtype=np.float64)
    if not (
        public_nf.shape == public_phase.shape == (public_paths.shape[0],)
        and public_energy.shape == public_paths.shape[:2]
    ):
        raise FinalAuditError("public source path metadata has inconsistent shapes")

    candidates: list[FrozenEnergySourceCandidate] = []
    for index, (nof, path, energy) in enumerate(
        zip(public_nf.tolist(), public_paths, public_energy)
    ):
        candidates.append(
            FrozenEnergySourceCandidate(
                kind="public_state",
                index=index,
                availability_nof=int(nof),
                phase=str(public_phase[index]),
                path=np.asarray(path, dtype=np.float64),
                stored_energy=np.asarray(energy, dtype=np.float64),
                independently_recomputed_energy=q10_energy_gradient_batch(path, q)[0],
            )
        )

    recovery_nf = np.asarray(archive["recovery_state_nf"], dtype=np.int64)
    recovery_fields = (
        ("recovery_anchor", "recovery_anchor_path"),
        ("recovery_candidate", "recovery_candidate_path"),
        ("recovery_public_before", "recovery_public_before_path"),
    )
    for kind, field in recovery_fields:
        paths = np.asarray(archive[field], dtype=np.float64)
        if paths.shape != (recovery_nf.size,) + public_paths.shape[1:]:
            raise FinalAuditError(f"{field} has an inconsistent source-path shape")
        for index, (nof, path) in enumerate(zip(recovery_nf.tolist(), paths)):
            candidates.append(
                FrozenEnergySourceCandidate(
                    kind=kind,
                    index=index,
                    availability_nof=int(nof),
                    phase="recovery_state",
                    path=np.asarray(path, dtype=np.float64),
                    stored_energy=None,
                    independently_recomputed_energy=q10_energy_gradient_batch(path, q)[0],
                )
            )

    candidates.extend(_complete_paid_band_candidates(archive, public_paths[0], q))
    return candidates


def _paid_trace_lookup(
    archive: Mapping[str, Any],
) -> dict[tuple[int, bytes], tuple[int, ...]]:
    image = np.asarray(archive["evaluation_image_index"], dtype=np.int64)
    coordinate = np.asarray(archive["evaluation_coordinate"], dtype="<f8")
    if coordinate.shape != (image.size, DIMENSION):
        raise FinalAuditError("paid trace coordinate/image shape mismatch")
    mutable: dict[tuple[int, bytes], list[int]] = {}
    for trace_index, (image_index, point) in enumerate(zip(image.tolist(), coordinate)):
        key = (
            int(image_index),
            np.ascontiguousarray(point, dtype="<f8").tobytes(order="C"),
        )
        mutable.setdefault(key, []).append(trace_index)
    return {key: tuple(value) for key, value in mutable.items()}


def _authenticate_frozen_energy_source(
    candidate: FrozenEnergySourceCandidate,
    *,
    event_input_path: np.ndarray,
    fixed_energy: np.ndarray,
    nof_at_entry: int,
    archive: Mapping[str, Any],
    paid_lookup: Mapping[tuple[int, bytes], tuple[int, ...]],
) -> AuthenticatedFrozenEnergySource | None:
    """Return a causal paid-state proof for one candidate, or ``None``."""

    path = np.asarray(candidate.path, dtype=np.float64)
    fixed = np.asarray(fixed_energy, dtype=np.float64)
    if candidate.availability_nof > int(nof_at_entry):
        return None
    if path.shape != np.asarray(event_input_path).shape or fixed.shape != (path.shape[0],):
        return None
    # A normal/recovery proposal may change internal coordinates before the
    # spring, but the physical endpoints must still belong to the source state.
    if not np.array_equal(path[(0, -1), :], np.asarray(event_input_path)[(0, -1), :]):
        return None

    independently_recomputed = np.asarray(
        candidate.independently_recomputed_energy, dtype=np.float64
    )
    potential_error = _strict_componentwise_energy_error(
        fixed,
        independently_recomputed,
        label="frozen/source potential energy",
        raise_on_mismatch=False,
    )
    if potential_error is None:
        return None
    endpoint_error = _strict_componentwise_energy_error(
        fixed[[0, -1]],
        independently_recomputed[[0, -1]],
        label="frozen/source endpoint energy",
        raise_on_mismatch=False,
    )
    if endpoint_error is None:
        return None

    stored_error = 0.0
    if candidate.stored_energy is not None:
        stored = np.asarray(candidate.stored_energy, dtype=np.float64)
        # Complete paid bands deliberately leave endpoints as NaN because
        # endpoints carry no NoF.  They have just been independently checked.
        if np.all(np.isfinite(stored)):
            compared = _strict_componentwise_energy_error(
                fixed,
                stored,
                label="frozen/recorded source energy",
                raise_on_mismatch=False,
            )
        else:
            compared = _strict_componentwise_energy_error(
                fixed[1:-1],
                stored[1:-1],
                label="frozen/paid-band source energy",
                raise_on_mismatch=False,
            )
        if compared is None:
            return None
        stored_error = float(compared)

    evaluation_nof = np.asarray(archive["evaluation_nof"], dtype=np.int64)
    evaluation_coordinate = np.asarray(archive["evaluation_coordinate"], dtype=np.float64)
    evaluation_energy = np.asarray(archive["evaluation_energy"], dtype=np.float64)
    paid_nofs: list[int] = []
    maximum_paid_error = 0.0
    for image_index in range(1, path.shape[0] - 1):
        coordinate_key = (
            image_index,
            np.ascontiguousarray(path[image_index], dtype="<f8").tobytes(order="C"),
        )
        eligible = paid_lookup.get(coordinate_key, ())
        matched: list[tuple[int, float]] = []
        for trace_index in eligible:
            if int(evaluation_nof[trace_index]) > int(nof_at_entry):
                continue
            if not np.array_equal(
                evaluation_coordinate[trace_index], path[image_index]
            ):
                continue
            error = _strict_componentwise_energy_error(
                np.asarray([fixed[image_index]], dtype=np.float64),
                np.asarray([evaluation_energy[trace_index]], dtype=np.float64),
                label="frozen/paid internal energy",
                raise_on_mismatch=False,
            )
            if error is not None:
                matched.append((trace_index, float(error)))
        if not matched:
            return None
        # Exact per-image cache closure makes this unique in a validated
        # archive.  Taking the latest remains deterministic for unit fixtures.
        trace_index, paid_error = max(
            matched, key=lambda item: int(evaluation_nof[item[0]])
        )
        paid_nofs.append(int(evaluation_nof[trace_index]))
        maximum_paid_error = max(maximum_paid_error, paid_error)

    return AuthenticatedFrozenEnergySource(
        candidate=candidate,
        maximum_paid_nof=max(paid_nofs) if paid_nofs else 0,
        paid_nof_by_internal_image=tuple(paid_nofs),
        maximum_energy_absolute_error=max(
            float(endpoint_error), stored_error, maximum_paid_error
        ),
        maximum_potential_energy_absolute_error=float(potential_error),
        maximum_endpoint_energy_absolute_error=float(endpoint_error),
        maximum_paid_energy_absolute_error=maximum_paid_error,
    )


def _select_frozen_energy_source(
    *,
    candidates: Sequence[FrozenEnergySourceCandidate],
    event_id: int,
    event_input_path: np.ndarray,
    fixed_energy: np.ndarray,
    nof_at_entry: int,
    archive: Mapping[str, Any],
    paid_lookup: Mapping[tuple[int, bytes], tuple[int, ...]],
) -> tuple[AuthenticatedFrozenEnergySource, int]:
    matches: list[AuthenticatedFrozenEnergySource] = []
    for candidate in candidates:
        authenticated = _authenticate_frozen_energy_source(
            candidate,
            event_input_path=event_input_path,
            fixed_energy=fixed_energy,
            nof_at_entry=nof_at_entry,
            archive=archive,
            paid_lookup=paid_lookup,
        )
        if authenticated is not None:
            matches.append(authenticated)
    if not matches:
        raise FinalAuditError(
            f"spring event {event_id} frozen energies have no causal, fully paid source state "
            f"at or before NoF {nof_at_entry}"
        )

    def semantic_priority(item: AuthenticatedFrozenEnergySource) -> int:
        candidate = item.candidate
        # A public state is the strongest whole-state record.  If none exists,
        # the synchronized recovery contract specifically consumes the direct
        # spring-check band.  Recovery candidates then cover private evaluated
        # states (notably the two N=20 bootstrap events).  This ordering is
        # source semantics, not a NoF-dependent optimization rule.
        if candidate.kind == "public_state":
            return 50
        if (
            candidate.kind == "paid_complete_band"
            and candidate.phase == "direct_spring_check"
        ):
            return 40
        if candidate.kind == "recovery_candidate":
            return 30
        if candidate.kind == "recovery_anchor":
            return 20
        if candidate.kind == "recovery_public_before":
            return 10
        return 1

    selected = max(
        matches,
        key=lambda item: (
            semantic_priority(item),
            item.candidate.availability_nof,
            item.candidate.index,
        ),
    )
    return selected, len(matches)


def audit_compact_spring_replay(
    archive: Mapping[str, Any], q: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tangent_ids = np.asarray(archive["tangent_event_id"], dtype=np.int64)
    parents = np.asarray(archive["tangent_parent_reparam_event_id"], dtype=np.int64)
    event_ids = np.asarray(archive["reparam_event_id"], dtype=np.int64)
    if not np.array_equal(tangent_ids, np.arange(1, tangent_ids.size + 1)):
        raise FinalAuditError("tangent event IDs are not 1..G")
    if not np.array_equal(event_ids, np.arange(1, event_ids.size + 1)):
        raise FinalAuditError("reparameterization event IDs are not 1..P")
    if "tangent_call_count" in archive and int(_scalar(archive, "tangent_call_count")) != tangent_ids.size:
        raise FinalAuditError("stored tangent_call_count differs from transcript length")
    stored_transcript = _digest(_scalar(archive, "tangent_transcript_sha256"), "tangent transcript")
    if tangent_transcript_sha256(archive) != stored_transcript:
        raise FinalAuditError("ordered tangent transcript digest mismatch")
    coverage = np.zeros(tangent_ids.size, dtype=np.int64)
    source_candidates = _frozen_energy_source_candidates(archive, q)
    paid_lookup = _paid_trace_lookup(archive)
    event_rows: list[dict[str, Any]] = []
    maximum_output_error = 0.0
    maximum_residual_error = 0.0
    maximum_energy_error = 0.0
    maximum_potential_energy_error = 0.0
    maximum_endpoint_energy_error = 0.0
    maximum_paid_energy_error = 0.0
    replayed_tangents = 0
    source_kind_counts: Counter[str] = Counter()
    source_phase_counts: Counter[str] = Counter()
    source_events: list[dict[str, Any]] = []
    distinct_input_source_events = 0
    callers = np.asarray(archive["reparam_caller"]).tolist()
    for offset, event_id in enumerate(event_ids.tolist()):
        result = np.asarray(archive["reparam_input_path"][offset], dtype=np.float64).copy()
        initial = result.copy()
        fixed_energy = np.asarray(archive["reparam_fixed_energy"][offset], dtype=np.float64)
        nof_at_entry = int(archive["reparam_nof_at_entry"][offset])
        source, matching_source_count = _select_frozen_energy_source(
            candidates=source_candidates,
            event_id=event_id,
            event_input_path=initial,
            fixed_energy=fixed_energy,
            nof_at_entry=nof_at_entry,
            archive=archive,
            paid_lookup=paid_lookup,
        )
        source_path_equal_to_input = bool(
            np.array_equal(source.candidate.path, initial)
        )
        distinct_input_source_events += int(not source_path_equal_to_input)
        source_kind_counts[source.candidate.kind] += 1
        source_phase_counts[source.candidate.phase] += 1
        iterations = int(archive["reparam_iterations"][offset])
        start = int(archive["reparam_tangent_call_start"][offset])
        count = int(archive["reparam_tangent_call_count"][offset])
        if not 1 <= iterations <= SPRING_MAX_ITER or count != iterations:
            raise FinalAuditError(f"spring event {event_id} has invalid iteration/tangent count")
        if start < 0 or start + count > tangent_ids.size:
            raise FinalAuditError(f"spring event {event_id} tangent slice is outside transcript")
        coverage[start:start + count] += 1
        residual = float("inf")
        for iteration in range(iterations):
            tangents = hj_improved_tangents(result, fixed_energy)
            tangent_index = start + iteration
            checks = (
                int(parents[tangent_index]) == event_id,
                _digest_row(archive["tangent_path_sha256"], tangent_index)
                == _domain_array_digest(result, domain="tangent-input-path"),
                _digest_row(archive["tangent_energy_sha256"], tangent_index)
                == _domain_array_digest(fixed_energy, domain="tangent-input-energy"),
                _digest_row(archive["tangent_output_sha256"], tangent_index)
                == _domain_array_digest(tangents, domain="tangent-output"),
            )
            if not all(checks):
                raise FinalAuditError(
                    f"spring event {event_id} tangent digest mismatch at iteration {iteration + 1}"
                )
            lengths = np.linalg.norm(np.diff(result, axis=0), axis=1)
            spring_force = (lengths[1:] - lengths[:-1])[:, None] * tangents
            residual = float(np.max(np.linalg.norm(spring_force, axis=1)))
            if residual <= SPRING_TOLERANCE:
                if iteration != iterations - 1:
                    raise FinalAuditError(f"spring event {event_id} continued after convergence")
                break
            result[1:-1] += SPRING_DTAU * spring_force
        if iterations < SPRING_MAX_ITER and residual > SPRING_TOLERANCE:
            raise FinalAuditError(f"spring event {event_id} stopped before convergence")
        result[0] = initial[0]
        result[-1] = initial[-1]
        output_error = _max_abs(result, archive["reparam_output_path"][offset])
        residual_error = abs(residual - float(archive["reparam_final_residual"][offset]))
        energy_error = source.maximum_energy_absolute_error
        if output_error != 0.0 or residual_error != 0.0:
            raise FinalAuditError(
                f"spring event {event_id} is not bitwise-identical on independent replay"
            )
        maximum_output_error = max(maximum_output_error, output_error)
        maximum_residual_error = max(maximum_residual_error, residual_error)
        maximum_energy_error = max(maximum_energy_error, energy_error)
        maximum_potential_energy_error = max(
            maximum_potential_energy_error,
            source.maximum_potential_energy_absolute_error,
        )
        maximum_endpoint_energy_error = max(
            maximum_endpoint_energy_error,
            source.maximum_endpoint_energy_absolute_error,
        )
        maximum_paid_energy_error = max(
            maximum_paid_energy_error,
            source.maximum_paid_energy_absolute_error,
        )
        replayed_tangents += iterations
        source_event = {
            "event_id": event_id,
            "source_kind": source.candidate.kind,
            "source_index": source.candidate.index,
            "source_nof": source.candidate.availability_nof,
            "source_phase": source.candidate.phase,
            "source_maximum_paid_nof": source.maximum_paid_nof,
        }
        source_events.append(source_event)
        event_rows.append(
            {
                "n_internal": int(np.asarray(archive["public_path_history"]).shape[1] - 2),
                "event_id": event_id,
                "caller": str(callers[offset]),
                "outer": int(archive["reparam_outer"][offset]),
                "nof_at_entry": nof_at_entry,
                "energy_source_kind": source.candidate.kind,
                "energy_source_index": source.candidate.index,
                "energy_source_nof": source.candidate.availability_nof,
                "energy_source_phase": source.candidate.phase,
                "energy_source_maximum_paid_nof": source.maximum_paid_nof,
                "energy_source_paid_nof_by_image_json": json.dumps(
                    list(source.paid_nof_by_internal_image), separators=(",", ":")
                ),
                "energy_source_matching_candidate_count": matching_source_count,
                "energy_source_path_equals_reparam_input": source_path_equal_to_input,
                "iterations": iterations,
                "tangent_call_start": start,
                "tangent_call_count": count,
                "stored_residual": float(archive["reparam_final_residual"][offset]),
                "replayed_residual": residual,
                "maximum_output_absolute_error": output_error,
                "maximum_fixed_energy_absolute_error": energy_error,
                "maximum_source_potential_energy_absolute_error": (
                    source.maximum_potential_energy_absolute_error
                ),
                "maximum_endpoint_energy_absolute_error": (
                    source.maximum_endpoint_energy_absolute_error
                ),
                "maximum_paid_energy_absolute_error": (
                    source.maximum_paid_energy_absolute_error
                ),
                "replay_pass": True,
            }
        )
    if not np.array_equal(coverage, (parents > 0).astype(np.int64)):
        raise FinalAuditError("spring tangent slices overlap or leave positive-parent events uncovered")
    if "tangent_reparam_call_count" in archive and int(
        _scalar(archive, "tangent_reparam_call_count")
    ) != replayed_tangents:
        raise FinalAuditError("stored tangent_reparam_call_count differs from spring replay")
    summary = {
        "reparameterization_events": int(event_ids.size),
        "tangent_events_total": int(tangent_ids.size),
        "spring_tangent_events_independently_replayed": replayed_tangents,
        "non_spring_tangent_events_hash_only": int(np.count_nonzero(parents == -1)),
        "maximum_output_absolute_error": maximum_output_error,
        "maximum_residual_absolute_error": maximum_residual_error,
        "maximum_fixed_energy_absolute_error": maximum_energy_error,
        "maximum_source_potential_energy_absolute_error": (
            maximum_potential_energy_error
        ),
        "maximum_endpoint_energy_absolute_error": maximum_endpoint_energy_error,
        "maximum_paid_energy_absolute_error": maximum_paid_energy_error,
        "fixed_energy_sources_authenticated": int(event_ids.size),
        "fixed_energy_source_kind_counts": dict(sorted(source_kind_counts.items())),
        "fixed_energy_source_phase_counts": dict(sorted(source_phase_counts.items())),
        "events_with_distinct_input_and_energy_source": distinct_input_source_events,
        "authenticated_fixed_energy_sources": source_events,
        "source_index_convention": (
            "zero-based archive row for public/recovery states; zero-based paid-trace "
            "start row for a contiguous complete band"
        ),
        "tangent_transcript_sha256": stored_transcript,
        "proof_boundary": (
            "Every frozen energy vector was authenticated to a whole source state that "
            "was available no later than spring entry; every internal coordinate/energy "
            "was found in the prior paid trace and endpoint energies were independently "
            "recomputed. Every spring tangent was independently recomputed at every iteration. "
            "Non-spring calls retain ordered input/output digests and are supported by "
            "the separately sealed reachable-call audit, not claimed as numerical replay."
        ),
    }
    return summary, event_rows


def count_duplicate_charged_coordinates(archive: Mapping[str, Any]) -> int:
    image = np.asarray(archive["evaluation_image_index"], dtype=np.int64)
    coordinates = np.asarray(archive["evaluation_coordinate"], dtype="<f8")
    caches: dict[int, set[bytes]] = {}
    duplicates = 0
    for image_index, coordinate in zip(image.tolist(), coordinates):
        key = np.ascontiguousarray(coordinate).tobytes(order="C")
        cache = caches.setdefault(int(image_index), set())
        if key in cache:
            duplicates += 1
        else:
            cache.add(key)
    return duplicates


def audit_nof(archive: Mapping[str, Any]) -> dict[str, Any]:
    terminal = int(_scalar(archive, "terminal_nof"))
    expected = np.arange(1, terminal + 1, dtype=np.int64)
    if not np.array_equal(archive["evaluation_sequence"], expected):
        raise FinalAuditError("evaluation_sequence is not 1..terminal NoF")
    if not np.array_equal(archive["evaluation_nof"], expected):
        raise FinalAuditError("evaluation_nof is not 1..terminal NoF")
    image = np.asarray(archive["evaluation_image_index"], dtype=np.int64)
    per_image = np.asarray(archive["per_image_joint_evaluations"], dtype=np.int64)
    if not np.array_equal(np.bincount(image, minlength=per_image.size), per_image):
        raise FinalAuditError("per-image NoF ledger differs from trace")
    phases = [str(item) for item in np.asarray(archive["evaluation_phase"]).tolist()]
    names = [str(item) for item in np.asarray(archive["nof_category_names"]).tolist()]
    counts = np.asarray(archive["nof_category_counts"], dtype=np.int64)
    reconstructed = Counter(phases)
    if not set(reconstructed).issubset(set(names)):
        raise FinalAuditError("trace contains a phase absent from the NoF category ledger")
    if any(reconstructed.get(name, 0) != int(count) for name, count in zip(names, counts.tolist())):
        raise FinalAuditError("NoF category counts differ from trace")
    for index, (coordinate, stored) in enumerate(
        zip(archive["evaluation_coordinate"], archive["evaluation_coordinate_sha256"])
    ):
        if coordinate_sha256(coordinate) != str(stored):
            raise FinalAuditError(f"charged coordinate hash mismatch at NoF {index + 1}")
    optimizer_trace = np.asarray(archive["optimizer_evaluation_trace_json"]).tolist()
    if len(optimizer_trace) != terminal:
        raise FinalAuditError("optimizer trace length differs from paid trace")
    for index, raw in enumerate(optimizer_trace):
        try:
            event = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise FinalAuditError(f"optimizer trace JSON is invalid at NoF {index + 1}") from exc
        expected_fields = (
            int(event.get("sequence", -1)) == index + 1,
            int(event.get("article_nf", -1)) == index + 1,
            int(event.get("image_index", -1)) == int(image[index]),
            str(event.get("coordinate_sha256", ""))
            == str(archive["evaluation_coordinate_sha256"][index]),
        )
        if not all(expected_fields):
            raise FinalAuditError(f"optimizer and observer traces differ at NoF {index + 1}")
    duplicates = count_duplicate_charged_coordinates(archive)
    if duplicates:
        raise FinalAuditError(f"{duplicates} exact per-image cache hits were charged")
    per_image_sum = int(np.sum(per_image))
    category_sum = int(np.sum(counts))
    if len(expected) != terminal or per_image_sum != terminal or category_sum != terminal:
        raise FinalAuditError("NoF trace/per-image/category closure failed")
    return {
        "terminal_nof": terminal,
        "trace_length": len(expected),
        "per_image_sum": per_image_sum,
        "category_sum": category_sum,
        "blocked_uncached_misses": int(_scalar(archive, "blocked_uncached_misses")),
        "duplicate_charged_coordinates": duplicates,
        "category_counts": dict(sorted((str(k), int(v)) for k, v in reconstructed.items())),
    }


def audit_potential_replay(archive: Mapping[str, Any], q: np.ndarray) -> dict[str, Any]:
    coordinates = np.asarray(archive["evaluation_coordinate"], dtype=np.float64)
    energy, gradient = q10_energy_gradient_batch(coordinates, q)
    paid_energy_error = _check_close(
        archive["evaluation_energy"], energy,
        absolute=2.0e-11, relative=2.0e-13, label="charged energy",
    )
    paid_gradient_error = _check_close(
        archive["evaluation_gradient"], gradient,
        absolute=2.0e-10, relative=2.0e-13, label="charged gradient",
    )
    paths = np.asarray(archive["public_path_history"], dtype=np.float64)
    public_energy = np.empty(paths.shape[:2], dtype=np.float64)
    public_gradient = np.empty_like(paths)
    for index, path in enumerate(paths):
        public_energy[index], public_gradient[index] = q10_energy_gradient_batch(path, q)
    public_energy_error = _check_close(
        archive["public_energy_history"], public_energy,
        absolute=2.0e-11, relative=2.0e-13, label="public energy",
    )
    public_gradient_error = _check_close(
        archive["public_gradient_history"], public_gradient,
        absolute=2.0e-10, relative=2.0e-13, label="public gradient",
    )
    return {
        "evaluation_count": int(coordinates.shape[0]),
        "public_state_count": int(paths.shape[0]),
        "maximum_evaluation_energy_absolute_error": paid_energy_error,
        "maximum_evaluation_gradient_absolute_error": paid_gradient_error,
        "maximum_public_energy_absolute_error": public_energy_error,
        "maximum_public_gradient_absolute_error": public_gradient_error,
    }


def locf_indices(observation_nof: np.ndarray, checkpoints: np.ndarray) -> np.ndarray:
    observed = np.asarray(observation_nof, dtype=np.int64)
    requested = np.asarray(checkpoints, dtype=np.int64)
    if observed.ndim != 1 or requested.ndim != 1 or observed.size == 0:
        raise FinalAuditError("LOCF inputs must be nonempty 1D arrays")
    if np.any(np.diff(observed) <= 0):
        raise FinalAuditError("LOCF observations are not strictly increasing")
    indices = np.searchsorted(observed, requested, side="right") - 1
    if np.any(indices < 0):
        raise FinalAuditError("a checkpoint precedes the first public state")
    return indices.astype(np.int64)


def strict_checkpoint_comparison(
    candidate_i: np.ndarray,
    article_i: np.ndarray,
    candidate_delta: np.ndarray,
    article_delta: np.ndarray,
) -> dict[str, np.ndarray]:
    candidate_i = np.asarray(candidate_i, dtype=np.float64)
    article_i = np.asarray(article_i, dtype=np.float64)
    candidate_error = np.abs(np.asarray(candidate_delta, dtype=np.float64) - REFERENCE_BARRIER)
    article_error = np.abs(np.asarray(article_delta, dtype=np.float64) - REFERENCE_BARRIER)
    if not (candidate_i.shape == article_i.shape == candidate_error.shape == article_error.shape):
        raise FinalAuditError("strict checkpoint arrays have different shapes")
    if not all(np.all(np.isfinite(item)) for item in (
        candidate_i, article_i, candidate_error, article_error
    )):
        raise FinalAuditError("strict checkpoint arrays contain non-finite values")
    i_strict = candidate_i < article_i
    barrier_strict = candidate_error < article_error
    return {
        "candidate_i_perp": candidate_i,
        "article_i_perp": article_i,
        "candidate_barrier_error": candidate_error,
        "article_barrier_error": article_error,
        "i_perp_strict": i_strict,
        "barrier_strict": barrier_strict,
        "joint_strict": i_strict & barrier_strict,
    }


def _coerce_article_paths(paths: np.ndarray) -> np.ndarray:
    value = np.asarray(paths, dtype=np.float64)
    if value.ndim != 3:
        raise FinalAuditError("Article path_history is not three-dimensional")
    if value.shape[-1] == DIMENSION:
        return value
    if value.shape[-1] != 3 * DIMENSION:
        raise FinalAuditError("Article path dimension is neither 10 nor 30")
    positions = value.reshape(value.shape[0], value.shape[1], DIMENSION, 3)
    inactive_max = float(np.max(np.abs(positions[..., 1:]))) if positions[..., 1:].size else 0.0
    if inactive_max > 1.0e-10:
        raise FinalAuditError("Article 30D embedding has nonzero inactive coordinates")
    return np.asarray(positions[..., 0], dtype=np.float64)


def load_and_audit_article(path: Path, q: np.ndarray, n_internal: int) -> ArticleTrajectory:
    with np.load(path, allow_pickle=False) as source:
        required = {
            "path_history", "total_force_evaluations", "i_perp",
            "normal_force_history", "delta_e", "q_matrix", "q_sha256",
        }
        missing = required - set(source.files)
        if missing:
            raise FinalAuditError(f"Article archive lacks {sorted(missing)}")
        paths = _coerce_article_paths(source["path_history"])
        nof = np.asarray(source["total_force_evaluations"], dtype=np.int64)
        stored_i = np.asarray(source["i_perp"], dtype=np.float64)
        stored_normal = np.asarray(source["normal_force_history"], dtype=np.float64)
        stored_delta = np.asarray(source["delta_e"], dtype=np.float64)
        article_q = np.asarray(source["q_matrix"], dtype=np.float64)
        article_q_sha = str(np.asarray(source["q_sha256"]).item())
    state_count = EXPECTED_CHECKPOINT_COUNTS[n_internal] + 1
    if paths.shape != (state_count, n_internal + 2, DIMENSION):
        raise FinalAuditError(f"Article N={n_internal} path shape changed: {paths.shape}")
    if nof.shape != (state_count,) or int(nof[0]) != n_internal or np.any(np.diff(nof) <= 0):
        raise FinalAuditError(f"Article N={n_internal} NoF checkpoints changed")
    if not np.array_equal(article_q, q) or article_q_sha != Q_ARRAY_SHA256:
        raise FinalAuditError("Article Q identity differs from registered Q")
    recomputed_i = np.empty(state_count, dtype=np.float64)
    recomputed_delta = np.empty(state_count, dtype=np.float64)
    recomputed_normal = np.empty((state_count, n_internal, DIMENSION), dtype=np.float64)
    for index, state_path in enumerate(paths):
        recomputed_i[index], recomputed_normal[index], _, _ = article_path_metric(state_path, q)
        recomputed_delta[index] = linear500_barrier(state_path, q)
    i_error = _check_close(
        stored_i, recomputed_i, absolute=2.0e-10, relative=2.0e-13,
        label=f"Article N={n_internal} I_perp",
    )
    normal_error = _check_close(
        stored_normal, recomputed_normal, absolute=2.0e-10, relative=2.0e-13,
        label=f"Article N={n_internal} normal forces",
    )
    delta_error = _check_close(
        stored_delta, recomputed_delta, absolute=2.0e-12, relative=2.0e-14,
        label=f"Article N={n_internal} linear500 Delta E",
    )
    return ArticleTrajectory(
        source_path=path, nof=nof, paths=paths, i_perp=recomputed_i,
        delta_e=recomputed_delta,
        maximum_stored_i_perp_error=i_error,
        maximum_stored_normal_force_error=normal_error,
        maximum_stored_delta_e_error=delta_error,
    )


def _verify_hash_manifests(
    campaign_root: Path,
    repository_root: Path,
    registration: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Path]]:
    source_manifest = registration.get("source_sha256")
    dependency_manifest = registration.get("dependency_sha256")
    if not isinstance(source_manifest, dict) or not source_manifest:
        raise FinalAuditError("registration lacks source_sha256")
    if not isinstance(dependency_manifest, dict) or not dependency_manifest:
        raise FinalAuditError("registration lacks dependency_sha256")
    source_paths: dict[str, Path] = {}
    dependency_paths: dict[str, Path] = {}
    for relative, expected in source_manifest.items():
        resolved = resolve_registered_path(campaign_root, str(relative))
        if sha256_file(resolved) != _digest(expected, f"source {relative}"):
            raise FinalAuditError(f"campaign source drift: {relative}")
        source_paths[str(relative)] = resolved
    for tagged, expected in dependency_manifest.items():
        tagged_text = str(tagged)
        if not tagged_text.startswith("repo:"):
            raise FinalAuditError(f"dependency lacks repo: tag: {tagged_text}")
        relative = tagged_text[len("repo:"):]
        resolved = resolve_registered_path(repository_root, relative)
        if sha256_file(resolved) != _digest(expected, f"dependency {tagged_text}"):
            raise FinalAuditError(f"repository dependency drift: {relative}")
        dependency_paths[tagged_text] = resolved
    return source_paths, dependency_paths


def _registered_input_paths(
    repository_root: Path,
    registration: Mapping[str, Any],
    dependency_paths: Mapping[str, Path],
) -> dict[str, Path]:
    declared = registration.get("registered_inputs")
    if not isinstance(declared, dict) or set(declared) != set(EXPECTED_REGISTERED_INPUTS):
        raise FinalAuditError("registered_inputs has the wrong closed set")
    result: dict[str, Path] = {}
    for label, expected_relative in EXPECTED_REGISTERED_INPUTS.items():
        relative = str(declared[label])
        if relative != expected_relative:
            raise FinalAuditError(f"registered input {label} changed: {relative}")
        tagged = "repo:" + relative
        if tagged not in dependency_paths:
            raise FinalAuditError(f"registered input is not dependency-hash locked: {tagged}")
        result[label] = resolve_registered_path(repository_root, relative)
    return result


def _verify_sealed_evidence(source_paths: Mapping[str, Path]) -> dict[str, Any]:
    static_key = "evidence/static_contract_report.json"
    equivalence_key = "evidence/preformal_instrumentation_equivalence.json"
    if static_key not in source_paths or equivalence_key not in source_paths:
        raise FinalAuditError("sealed static/equivalence evidence is missing from source_sha256")
    static = load_canonical_json(source_paths[static_key])
    equivalence = load_canonical_json(source_paths[equivalence_key])
    if not isinstance(static, dict) or static.get("status") != "PASS" or static.get("passed") is not True:
        raise FinalAuditError("sealed static reachable-call report is not a PASS")
    if static.get("all_runtime_calls_intercepted") is not True:
        raise FinalAuditError("static report did not prove all runtime calls intercepted")
    forbidden = static.get("forbidden_reachable_counts")
    if not isinstance(forbidden, dict) or not forbidden or any(int(value) != 0 for value in forbidden.values()):
        raise FinalAuditError("static report has forbidden reachable operations")
    if "improved" not in str(static.get("tangent_target", "")).lower():
        raise FinalAuditError("static report tangent target is not improved_tangents")
    if not static.get("optimizer_reachable") or not static.get("reparam_calls"):
        raise FinalAuditError("static report lacks reachable optimizer/reparameterization evidence")
    if not isinstance(equivalence, dict):
        raise FinalAuditError("preformal instrumentation equivalence is not an object")
    equivalence_pass = equivalence.get("passed") is True or equivalence.get("all_passed") is True
    if str(equivalence.get("status", "")).upper() != "PASS" or not equivalence_pass:
        raise FinalAuditError("preformal instrumentation equivalence is not a PASS")
    return {
        "static_contract_report": {
            "relative_path": static_key,
            "sha256": sha256_file(source_paths[static_key]),
            "status": "PASS",
            "all_runtime_calls_intercepted": True,
        },
        "preformal_instrumentation_equivalence": {
            "relative_path": equivalence_key,
            "sha256": sha256_file(source_paths[equivalence_key]),
            "status": "PASS",
        },
    }


def _find_digest_by_suffix(
    manifest: Mapping[str, Any], suffixes: Sequence[str], label: str
) -> str:
    matches = [
        str(value) for key, value in manifest.items()
        if any(str(key).endswith(suffix) for suffix in suffixes)
    ]
    if len(matches) != 1:
        raise FinalAuditError(f"cannot uniquely resolve registered {label} digest")
    return _digest(matches[0], label)


def _expected_result_identity(registration: Mapping[str, Any]) -> dict[str, str]:
    explicit_paths = registration.get("result_identity_paths")
    if isinstance(explicit_paths, dict):
        expected_keys = {
            "algorithm_source_sha256", "optimizer_sha256", "adapter_sha256",
            "runtime_sha256", "backend_sha256", "runner_sha256",
            "exploratory_source_sha256", "preformal_equivalence_sha256",
            "instrumentation_equivalence_sha256",
        }
        if set(explicit_paths) != expected_keys:
            raise FinalAuditError("result_identity_paths has the wrong closed set")
        result: dict[str, str] = {}
        for field, registered_key in explicit_paths.items():
            tagged = str(registered_key)
            if tagged.startswith("campaign:"):
                expected = registration["source_sha256"].get(tagged[len("campaign:"):])
            elif tagged.startswith("repo:"):
                expected = registration["dependency_sha256"].get(tagged)
            else:
                raise FinalAuditError(f"identity path for {field} lacks campaign:/repo: tag")
            if expected is None:
                raise FinalAuditError(f"identity path for {field} is not hash-registered")
            result[field] = _digest(expected, field)
        return result
    return {
        "algorithm_source_sha256": _find_digest_by_suffix(
            registration["source_sha256"], ("src/q10_unified_cycle_optimizer.py",),
            "algorithm source",
        ),
        "optimizer_sha256": _find_digest_by_suffix(
            registration["source_sha256"], ("src/q10_unified_cycle_optimizer.py",), "optimizer"
        ),
        "adapter_sha256": _find_digest_by_suffix(
            registration["source_sha256"], ("src/registered_adapter.py",), "adapter"
        ),
        "runner_sha256": _find_digest_by_suffix(
            registration["source_sha256"], ("src/run_frozen_reproduction.py",), "runner"
        ),
        "runtime_sha256": _find_digest_by_suffix(
            registration["dependency_sha256"],
            ("/q10_irivm_atomic_tangent_secant_core.py",), "runtime",
        ),
        "backend_sha256": _find_digest_by_suffix(
            registration["dependency_sha256"], ("/q10_solver_backend.py",), "backend"
        ),
        "exploratory_source_sha256": _find_digest_by_suffix(
            registration["source_sha256"], ("evidence/exploratory_candidate.py",),
            "exploratory source",
        ),
        "preformal_equivalence_sha256": _find_digest_by_suffix(
            registration["source_sha256"], ("evidence/preformal_equivalence.json",),
            "preformal equivalence",
        ),
        "instrumentation_equivalence_sha256": _find_digest_by_suffix(
            registration["source_sha256"],
            ("evidence/preformal_instrumentation_equivalence.json",),
            "instrumentation equivalence",
        ),
    }


def _verify_result_identity(
    archive: Mapping[str, Any],
    registration: Mapping[str, Any],
    registration_hash: str,
    seal_hash: str,
) -> None:
    if str(_scalar(archive, "registration_sha256")) != registration_hash:
        raise FinalAuditError("result registration hash mismatch")
    if str(_scalar(archive, "registration_seal_sha256")) != seal_hash:
        raise FinalAuditError("result registration seal hash mismatch")
    expected_identity = _expected_result_identity(registration)
    registered_identity = registration.get("result_identity")
    if not isinstance(registered_identity, dict):
        raise FinalAuditError("registration lacks result_identity")
    for field, expected in expected_identity.items():
        if registered_identity.get(field) != expected:
            raise FinalAuditError(f"registered result identity {field} differs from manifest")
        if str(_scalar(archive, field)) != expected:
            raise FinalAuditError(f"result {field} differs from registered source")
    if registered_identity.get("q_sha256") != Q_ARRAY_SHA256:
        raise FinalAuditError("registered result identity Q digest changed")
    runtime_contract = registration.get("runtime_contract")
    expected_environment = {
        "OPENBLAS_NUM_THREADS": "8",
        "OMP_NUM_THREADS": None,
        "MKL_NUM_THREADS": None,
        "VECLIB_MAXIMUM_THREADS": None,
        "NUMEXPR_NUM_THREADS": None,
    }
    if not isinstance(runtime_contract, dict) or runtime_contract.get("jobs") != 1:
        raise FinalAuditError("registered runtime jobs contract changed")
    if runtime_contract.get("thread_environment") != expected_environment:
        raise FinalAuditError("registered OpenBLAS/thread environment changed")
    stored_runtime = json.loads(str(_scalar(archive, "runtime_identity_json")))
    if stored_runtime != runtime_contract.get("runtime_identity"):
        raise FinalAuditError("result runtime identity differs from registration")
    dependency_text = str(_scalar(archive, "dependency_manifest_json"))
    expected_text = json.dumps(
        registration["dependency_sha256"], sort_keys=True, separators=(",", ":")
    )
    if dependency_text != expected_text:
        raise FinalAuditError("result dependency manifest differs byte-for-byte")


def _audit_receipt(
    receipt_path: Path,
    result_path: Path,
    archive: Mapping[str, Any],
    n_internal: int,
    campaign_root: Path,
    registration_hash: str,
    seal_hash: str,
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = load_canonical_json(receipt_path)
    if not isinstance(receipt, dict):
        raise FinalAuditError("receipt root is not an object")
    expected = {
        "schema_version": 2,
        "campaign_id": str(_scalar(archive, "campaign_id")),
        "run_id": str(_scalar(archive, "run_id")),
        "n_internal": n_internal,
        "result_relative_path": result_path.relative_to(campaign_root).as_posix(),
        "result_sha256": sha256_file(result_path),
        "registration_relative_path": "registrations/CAMPAIGN_PREREGISTRATION.json",
        "registration_sha256": registration_hash,
        "registration_seal_sha256": seal_hash,
        "algorithm_source_sha256": str(_scalar(archive, "algorithm_source_sha256")),
        "optimizer_sha256": str(_scalar(archive, "optimizer_sha256")),
        "adapter_sha256": str(_scalar(archive, "adapter_sha256")),
        "runtime_sha256": str(_scalar(archive, "runtime_sha256")),
        "backend_sha256": str(_scalar(archive, "backend_sha256")),
        "runner_sha256": str(_scalar(archive, "runner_sha256")),
        "exploratory_source_sha256": str(_scalar(archive, "exploratory_source_sha256")),
        "preformal_equivalence_sha256": str(_scalar(archive, "preformal_equivalence_sha256")),
        "instrumentation_equivalence_sha256": str(
            _scalar(archive, "instrumentation_equivalence_sha256")
        ),
        "q_sha256": str(_scalar(archive, "q_sha256")),
        "source_sha256": registration["source_sha256"],
        "dependency_sha256": registration["dependency_sha256"],
        "terminal_nof": int(_scalar(archive, "terminal_nof")),
        "stop_reason": str(_scalar(archive, "stop_reason")),
        "runtime_identity": json.loads(str(_scalar(archive, "runtime_identity_json"))),
    }
    if set(receipt) != set(expected):
        raise FinalAuditError("receipt field set differs from runner v2 contract")
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value:
            raise FinalAuditError(f"receipt field {key} mismatch")
    return receipt


def _public_metrics(paths: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i_perp = np.empty(paths.shape[0], dtype=np.float64)
    delta_e = np.empty(paths.shape[0], dtype=np.float64)
    for index, path in enumerate(paths):
        i_perp[index] = article_path_metric(path, q)[0]
        delta_e[index] = linear500_barrier(path, q)
    return i_perp, delta_e


def _checkpoint_rows(
    n_internal: int,
    candidate_nof: np.ndarray,
    candidate_i: np.ndarray,
    candidate_delta: np.ndarray,
    article: ArticleTrajectory,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    checkpoints = article.nof[1:]
    if checkpoints.size != EXPECTED_CHECKPOINT_COUNTS[n_internal]:
        raise FinalAuditError("Article checkpoint count drift")
    candidate_indices = locf_indices(candidate_nof, checkpoints)
    comparison = strict_checkpoint_comparison(
        candidate_i[candidate_indices], article.i_perp[1:],
        candidate_delta[candidate_indices], article.delta_e[1:],
    )
    rows: list[dict[str, Any]] = []
    for offset, checkpoint in enumerate(checkpoints.tolist()):
        candidate_index = int(candidate_indices[offset])
        candidate_barrier = float(comparison["candidate_barrier_error"][offset])
        article_barrier = float(comparison["article_barrier_error"][offset])
        rows.append({
            "n_internal": n_internal,
            "checkpoint_index": offset + 1,
            "checkpoint_nof": checkpoint,
            "candidate_state_index": candidate_index,
            "candidate_state_nof": int(candidate_nof[candidate_index]),
            "article_state_index": offset + 1,
            "article_state_nof": int(article.nof[offset + 1]),
            "candidate_i_perp": float(comparison["candidate_i_perp"][offset]),
            "article_i_perp": float(comparison["article_i_perp"][offset]),
            "i_perp_ratio": float(
                comparison["candidate_i_perp"][offset] / comparison["article_i_perp"][offset]
            ),
            "i_perp_strict": bool(comparison["i_perp_strict"][offset]),
            "candidate_delta_e": float(candidate_delta[candidate_index]),
            "article_delta_e": float(article.delta_e[offset + 1]),
            "candidate_barrier_error": candidate_barrier,
            "article_barrier_error": article_barrier,
            "barrier_error_ratio": candidate_barrier / article_barrier,
            "barrier_strict": bool(comparison["barrier_strict"][offset]),
            "joint_strict": bool(comparison["joint_strict"][offset]),
        })
    return rows, comparison


def _trajectory_rows(
    n_internal: int,
    candidate_nof: np.ndarray,
    candidate_i: np.ndarray,
    candidate_delta: np.ndarray,
    article: ArticleTrajectory,
) -> list[dict[str, Any]]:
    terminal = int(article.nof[-1])
    rows: list[dict[str, Any]] = []
    for index in range(article.nof.size):
        rows.append({
            "n_internal": n_internal,
            "method": "Article IVM",
            "nof": int(article.nof[index]),
            "i_perp": float(article.i_perp[index]),
            "delta_e": float(article.delta_e[index]),
            "barrier_error": abs(float(article.delta_e[index]) - REFERENCE_BARRIER),
            "is_locf_extension": False,
            "source_state_index": index,
        })
    selected = np.flatnonzero(candidate_nof <= terminal)
    if selected.size == 0:
        raise FinalAuditError("candidate has no public state inside Article plotting range")
    for index in selected.tolist():
        rows.append({
            "n_internal": n_internal,
            "method": "IR-IVM",
            "nof": int(candidate_nof[index]),
            "i_perp": float(candidate_i[index]),
            "delta_e": float(candidate_delta[index]),
            "barrier_error": abs(float(candidate_delta[index]) - REFERENCE_BARRIER),
            "is_locf_extension": False,
            "source_state_index": index,
        })
    last_index = int(selected[-1])
    if int(candidate_nof[last_index]) < terminal:
        rows.append({
            "n_internal": n_internal,
            "method": "IR-IVM",
            "nof": terminal,
            "i_perp": float(candidate_i[last_index]),
            "delta_e": float(candidate_delta[last_index]),
            "barrier_error": abs(float(candidate_delta[last_index]) - REFERENCE_BARRIER),
            "is_locf_extension": True,
            "source_state_index": last_index,
        })
    return rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with atomic_output_path(path, suffix=".csv") as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                if set(row) != set(fieldnames):
                    raise FinalAuditError(f"CSV field mismatch for {path.name}")
                writer.writerow({name: _csv_value(row[name]) for name in fieldnames})


def _select_unique(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FinalAuditError(f"expected one {label}, found {len(matches)}")
    return matches[0]


def run_complete_audit(
    *,
    registration_path: Path,
    n10_result_path: Path,
    n20_result_path: Path,
    n10_receipt_path: Path,
    n20_receipt_path: Path,
    output_directory: Path,
) -> Path:
    paths = derive_campaign_paths(__file__)
    campaign_root = paths.campaign_root
    repository_root = paths.repository_root
    seal_path = registration_path.with_suffix(".sha256")
    registration, registration_hash = verify_registration_seal(registration_path, seal_path)
    seal_hash = sha256_file(seal_path)
    source_paths, dependency_paths = _verify_hash_manifests(
        campaign_root, repository_root, registration
    )
    sealed_evidence = _verify_sealed_evidence(source_paths)
    registered_inputs = _registered_input_paths(
        repository_root, registration, dependency_paths
    )
    with np.load(registered_inputs["q_n10"], allow_pickle=False) as archive:
        q10 = np.asarray(archive["Q"], dtype=np.float64)
    with np.load(registered_inputs["q_n20"], allow_pickle=False) as archive:
        q20 = np.asarray(archive["Q"], dtype=np.float64)
    if not np.array_equal(q10, q20) or _array_sha256(q10) != Q_ARRAY_SHA256:
        raise FinalAuditError("registered Q arrays differ or have wrong array digest")
    q = q10
    articles = {
        10: load_and_audit_article(registered_inputs["article_n10"], q, 10),
        20: load_and_audit_article(registered_inputs["article_n20"], q, 20),
    }
    members = registration.get("registered_members")
    if not isinstance(members, list) or len(members) != 2:
        raise FinalAuditError("registration must contain exactly two members")
    controls: dict[str, dict[str, Any]] = {}
    for member in members:
        n_internal = int(member.get("n_internal", -1))
        if n_internal not in (10, 20) or str(n_internal) in controls:
            raise FinalAuditError("registered members are not exactly N=10/N=20")
        controls[str(n_internal)] = {
            "alpha": float(member["alpha"]),
            "alpha_label": str(member["alpha_label"]),
            "r": float(member["r"]),
            "m_max": int(member["m_max"]),
        }
    result_paths = {10: n10_result_path, 20: n20_result_path}
    receipt_paths = {10: n10_receipt_path, 20: n20_receipt_path}
    checkpoint_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    nof_rows: list[dict[str, Any]] = []
    reparam_rows: list[dict[str, Any]] = []
    run_reports: list[dict[str, Any]] = []
    for n_internal in (10, 20):
        result_path = result_paths[n_internal].resolve(strict=True)
        receipt_path = receipt_paths[n_internal].resolve(strict=True)
        try:
            result_path.relative_to(campaign_root)
            receipt_path.relative_to(campaign_root)
        except ValueError as exc:
            raise FinalAuditError("result or receipt lies outside campaign root") from exc
        schema_report: ValidationReport = validate_result_archive(result_path)
        if schema_report.dimensions.n_internal != n_internal:
            raise FinalAuditError("result resolution/path mismatch")
        with np.load(result_path, allow_pickle=False) as saved:
            archive = {name: np.asarray(saved[name]) for name in saved.files}
        _verify_result_identity(archive, registration, registration_hash, seal_hash)
        if str(_scalar(archive, "q_sha256")) != Q_ARRAY_SHA256:
            raise FinalAuditError("result Q digest mismatch")
        if not np.array_equal(np.asarray(archive["q_matrix"]), q):
            raise FinalAuditError("result Q matrix differs from registered Q")
        if str(_scalar(archive, "failure_message")):
            raise FinalAuditError("formal optimizer reported a failure")
        if "improved" not in str(_scalar(archive, "tangent_definition")).lower():
            raise FinalAuditError("result tangent definition is not improved tangent")
        reparam_text = str(_scalar(archive, "reparameterization_definition")).lower()
        if "spring" not in reparam_text or "nearest" not in reparam_text:
            raise FinalAuditError("result reparameterization definition is not nearest-neighbor spring")
        config = json.loads(str(_scalar(archive, "config_json")))
        frozen = controls[str(n_internal)]
        for key in ("alpha", "r", "m_max"):
            if config.get(key) != frozen[key]:
                raise FinalAuditError(f"N={n_internal} result config {key} drifted")
        _audit_receipt(
            receipt_path, result_path, archive, n_internal, campaign_root,
            registration_hash, seal_hash, registration,
        )
        nof_report = audit_nof(archive)
        potential_report = audit_potential_replay(archive, q)
        spring_report, event_rows = audit_compact_spring_replay(archive, q)
        reparam_rows.extend(event_rows)
        public_nof = np.asarray(archive["public_nf"], dtype=np.int64)
        public_paths = np.asarray(archive["public_path_history"], dtype=np.float64)
        public_i, public_delta = _public_metrics(public_paths, q)
        article = articles[n_internal]
        if not np.array_equal(public_paths[0], article.paths[0]):
            raise FinalAuditError(f"N={n_internal} shared initial path differs")
        if int(public_nof[0]) != int(article.nof[0]):
            raise FinalAuditError(f"N={n_internal} shared initial NoF differs")
        member_checkpoint_rows, comparison = _checkpoint_rows(
            n_internal, public_nof, public_i, public_delta, article
        )
        expected_count = EXPECTED_CHECKPOINT_COUNTS[n_internal]
        strict_i = int(np.count_nonzero(comparison["i_perp_strict"]))
        strict_barrier = int(np.count_nonzero(comparison["barrier_strict"]))
        strict_joint = int(np.count_nonzero(comparison["joint_strict"]))
        if (strict_i, strict_barrier, strict_joint) != (
            expected_count, expected_count, expected_count
        ):
            raise FinalAuditError(
                f"N={n_internal} strict acceptance failed: I {strict_i}/{expected_count}, "
                f"Delta E {strict_barrier}/{expected_count}, joint {strict_joint}/{expected_count}"
            )
        checkpoint_rows.extend(member_checkpoint_rows)
        trajectory_rows.extend(
            _trajectory_rows(n_internal, public_nof, public_i, public_delta, article)
        )
        nof_rows.append({
            "n_internal": n_internal,
            "run_id": str(_scalar(archive, "run_id")),
            "terminal_nof": nof_report["terminal_nof"],
            "trace_length": nof_report["trace_length"],
            "per_image_sum": nof_report["per_image_sum"],
            "category_sum": nof_report["category_sum"],
            "blocked_uncached_misses": nof_report["blocked_uncached_misses"],
            "duplicate_charged_coordinates": nof_report["duplicate_charged_coordinates"],
            "category_counts_json": json.dumps(
                nof_report["category_counts"], sort_keys=True, separators=(",", ":")
            ),
            "closure_pass": True,
        })
        rises = np.diff(public_i)
        positive_rises = rises[rises > 0.0]
        run_reports.append({
            "n_internal": n_internal,
            "run_id": str(_scalar(archive, "run_id")),
            "controls": frozen,
            "result_path": result_path.relative_to(campaign_root).as_posix(),
            "result_sha256": sha256_file(result_path),
            "receipt_path": receipt_path.relative_to(campaign_root).as_posix(),
            "receipt_sha256": sha256_file(receipt_path),
            "schema": schema_report.to_jsonable(),
            "nof": nof_report,
            "potential_replay": potential_report,
            "spring_and_tangent_replay": spring_report,
            "article_source_sha256": sha256_file(article.source_path),
            "article_stored_metric_audit": {
                "maximum_i_perp_absolute_error": article.maximum_stored_i_perp_error,
                "maximum_normal_force_absolute_error": article.maximum_stored_normal_force_error,
                "maximum_delta_e_absolute_error": article.maximum_stored_delta_e_error,
            },
            "checkpoint_count": expected_count,
            "strict_i_count": strict_i,
            "strict_barrier_count": strict_barrier,
            "strict_joint_count": strict_joint,
            "public_i_perp_rise_count": int(positive_rises.size),
            "maximum_public_i_perp_rise": (
                float(np.max(positive_rises)) if positive_rises.size else 0.0
            ),
        })

    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "checkpoint_comparison.csv"
    trajectory_path = output_directory / "public_trajectory_metrics.csv"
    nof_path = output_directory / "nof_accounting.csv"
    reparam_path = output_directory / "reparam_replay.csv"
    report_path = output_directory / "AUDIT_REPORT.md"
    manifest_path = output_directory / "audit_manifest.json"
    _write_csv(checkpoint_path, tuple(checkpoint_rows[0]), checkpoint_rows)
    _write_csv(trajectory_path, tuple(trajectory_rows[0]), trajectory_rows)
    _write_csv(nof_path, tuple(nof_rows[0]), nof_rows)
    reparam_fields = (
        tuple(reparam_rows[0]) if reparam_rows else (
            "n_internal", "event_id", "caller", "outer", "nof_at_entry",
            "energy_source_kind", "energy_source_index", "energy_source_nof",
            "energy_source_phase", "energy_source_maximum_paid_nof",
            "energy_source_paid_nof_by_image_json",
            "energy_source_matching_candidate_count",
            "energy_source_path_equals_reparam_input",
            "iterations", "tangent_call_start", "tangent_call_count",
            "stored_residual", "replayed_residual", "maximum_output_absolute_error",
            "maximum_fixed_energy_absolute_error",
            "maximum_source_potential_energy_absolute_error",
            "maximum_endpoint_energy_absolute_error",
            "maximum_paid_energy_absolute_error", "replay_pass",
        )
    )
    _write_csv(reparam_path, reparam_fields, reparam_rows)
    report_lines = [
        "# Independent audit report",
        "",
        "Outcome: **complete scientific pass**.",
        "",
        "The auditor independently recomputed every charged and public-state energy/gradient, "
        "every frozen spring-energy vector against a causal prior whole-path source, every "
        "source internal image against its exact paid coordinate/energy observation, every "
        "uncharged endpoint energy, every spring-flow improved tangent at every spring iteration, "
        "every spring output, "
        "Article I_perp, and the coordinate-wise linear-500 densified energy barrier. It did "
        "not import the optimizer, its runtime, or the registered metric implementation.",
        "",
        "Non-spring tangent calls are protected by ordered input/energy/output hashes, a runtime "
        "independent-HJ mismatch guard, and the sealed reachable-call audit. They are not claimed "
        "as offline per-call numerical replays.",
        "",
        "| N_int | I_perp strict | Delta E strict | joint strict | terminal NoF | I_perp rises |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in run_reports:
        report_lines.append(
            f"| {item['n_internal']} | {item['strict_i_count']}/{item['checkpoint_count']} | "
            f"{item['strict_barrier_count']}/{item['checkpoint_count']} | "
            f"{item['strict_joint_count']}/{item['checkpoint_count']} | "
            f"{item['nof']['terminal_nof']} | {item['public_i_perp_rise_count']} |"
        )
    report_lines.extend([
        "",
        "Frozen-energy source authentication (kind/record index/availability NoF, paid-image "
        "NoFs, and evaluation phase) is reported per spring event in `reparam_replay.csv` and "
        "in the audit manifest. Complete paid-band sources are accepted only from a contiguous "
        "1..N image sequence with one outer iteration and one NoF category; partial or future "
        "evaluation groups cannot authenticate a spring.",
        "",
        "| N_int | spring sources | distinct raw/source paths | source kinds | source phases |",
        "|---:|---:|---:|---|---|",
    ])
    for item in run_reports:
        spring = item["spring_and_tangent_replay"]
        report_lines.append(
            f"| {item['n_internal']} | {spring['fixed_energy_sources_authenticated']} | "
            f"{spring['events_with_distinct_input_and_energy_source']} | "
            f"`{json.dumps(spring['fixed_energy_source_kind_counts'], sort_keys=True)}` | "
            f"`{json.dumps(spring['fixed_energy_source_phase_counts'], sort_keys=True)}` |"
        )
    report_lines.extend([
        "",
        "The shared initial state was verified and excluded. IR-IVM values at exact Article NoF "
        "checkpoints use last observation carried forward. N=10 is extended by LOCF to the "
        "Article terminal 951 NoF; N=20 is truncated at the Article terminal 2377 NoF.",
        "",
    ])
    atomic_write_bytes(report_path, ("\n".join(report_lines)).encode("utf-8"))
    artifacts = {}
    for artifact in (checkpoint_path, trajectory_path, nof_path, reparam_path, report_path):
        artifacts[artifact.name] = {
            "relative_path": artifact.relative_to(output_directory).as_posix(),
            "sha256": sha256_file(artifact),
        }
    manifest = {
        "schema_version": 1,
        "outcome": "complete_scientific_pass",
        "integrity_complete": True,
        "campaign_id": registration.get("campaign_id"),
        "registration": {
            "relative_path": registration_path.relative_to(campaign_root).as_posix(),
            "sha256": registration_hash,
            "seal_relative_path": seal_path.relative_to(campaign_root).as_posix(),
            "seal_sha256": seal_hash,
        },
        "audit_source_sha256": sha256_file(Path(__file__)),
        "sealed_evidence": sealed_evidence,
        "source_sha256": {
            key: sha256_file(value) for key, value in sorted(source_paths.items())
        },
        "dependency_sha256": {
            key: sha256_file(value) for key, value in sorted(dependency_paths.items())
        },
        "registered_inputs": {
            key: {
                "relative_path": value.relative_to(repository_root).as_posix(),
                "sha256": sha256_file(value),
            }
            for key, value in sorted(registered_inputs.items())
        },
        "q_array_sha256": Q_ARRAY_SHA256,
        "reference_barrier": REFERENCE_BARRIER,
        "linear_densification_count": LINEAR_DENSE_COUNT,
        "registered_controls": controls,
        "runs": run_reports,
        "checkpoint_counts": {
            str(item["n_internal"]): {
                "expected": item["checkpoint_count"],
                "i_perp_strict": item["strict_i_count"],
                "barrier_strict": item["strict_barrier_count"],
                "joint_strict": item["strict_joint_count"],
            }
            for item in run_reports
        },
        "artifacts": artifacts,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def main() -> int:
    paths = derive_campaign_paths(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registration", type=Path,
        default=paths.campaign_root / "registrations/CAMPAIGN_PREREGISTRATION.json",
    )
    parser.add_argument("--n10", type=Path)
    parser.add_argument("--n20", type=Path)
    parser.add_argument("--n10-receipt", type=Path)
    parser.add_argument("--n20-receipt", type=Path)
    parser.add_argument(
        "--output-directory", type=Path,
        default=paths.campaign_root / "reports/audit",
    )
    arguments = parser.parse_args()
    try:
        n10_result = arguments.n10 or _select_unique(
            paths.campaign_root / "runs/n10", "*.npz", "N=10 result"
        )
        n20_result = arguments.n20 or _select_unique(
            paths.campaign_root / "runs/n20", "*.npz", "N=20 result"
        )
        n10_receipt = arguments.n10_receipt or _select_unique(
            paths.campaign_root / "reports/receipts/n10", "*.json", "N=10 receipt"
        )
        n20_receipt = arguments.n20_receipt or _select_unique(
            paths.campaign_root / "reports/receipts/n20", "*.json", "N=20 receipt"
        )
        manifest = run_complete_audit(
            registration_path=arguments.registration,
            n10_result_path=n10_result,
            n20_result_path=n20_result,
            n10_receipt_path=n10_receipt,
            n20_receipt_path=n20_receipt,
            output_directory=arguments.output_directory,
        )
    except (FormalIOError, FinalAuditError, OSError, KeyError, ValueError) as exc:
        print(f"AUDIT REFUSED: {exc}")
        return 1
    print(f"COMPLETE SCIENTIFIC PASS: {manifest}")
    return 0
