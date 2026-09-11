"""Independent result-audit primitives for the frozen reproduction.

This module deliberately does not import the optimizer or ``irivm_runtime``.
The HJ tangent and nearest-neighbour spring replay below are independent audit
implementations. Backend/potential and Article metric sources are loaded only
after their registered hashes are verified.

The command-line entry point remains fail-closed while this directory is a
staging scaffold: it can emit a structural/HJ report, but it cannot issue a
scientific-pass manifest until the registered backend and Article input adapter
is bound and tested.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Protocol

import numpy as np

from .formal_io import (
    FormalIOError,
    atomic_write_json,
    derive_campaign_paths,
    resolve_registered_path,
    sha256_file,
    verify_registration_seal,
)
from .result_schema import ValidationReport, validate_result_archive


SPRING_DTAU = 0.05
SPRING_MAX_ITER = 500
SPRING_TOLERANCE = 1.0e-5
TANGENT_DEGENERACY_TOLERANCE = 1.0e-14
REFERENCE_BARRIER = 67.50188060819488


class IndependentAuditError(RuntimeError):
    """Raised when saved evidence violates an independently checked contract."""


class MetricAdapter(Protocol):
    def energy_gradient(
        self, point: np.ndarray, q: np.ndarray
    ) -> tuple[float, np.ndarray]: ...

    def article_path_metric(
        self, path: np.ndarray, q: np.ndarray
    ) -> tuple[float, np.ndarray]: ...

    def linear500_barrier(self, path: np.ndarray, q: np.ndarray) -> float: ...


@dataclass(frozen=True)
class SpringReplayReport:
    event_count: int
    tangent_call_count: int
    maximum_output_absolute_error: float
    maximum_tangent_absolute_error: float
    maximum_residual_absolute_error: float


@dataclass(frozen=True)
class StructuralRunAudit:
    result_path: str
    result_sha256: str
    schema: dict[str, Any]
    duplicate_charged_coordinate_count: int
    spring_replay: dict[str, Any]


def hj_improved_tangents(path: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """Independent Henkelman--Jonsson improved tangent implementation."""

    points = np.asarray(path, dtype=np.float64)
    values = np.asarray(energies, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 10:
        raise IndependentAuditError("HJ tangent path must have shape [images,10]")
    if values.shape != (points.shape[0],):
        raise IndependentAuditError("HJ tangent energy/path length mismatch")
    tangents: list[np.ndarray] = []
    for index in range(1, points.shape[0] - 1):
        forward = points[index + 1] - points[index]
        backward = points[index] - points[index - 1]
        previous, current, following = values[index - 1:index + 2]
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
        magnitude = float(np.linalg.norm(tangent))
        if magnitude <= TANGENT_DEGENERACY_TOLERANCE:
            tangent = forward + backward
            magnitude = float(np.linalg.norm(tangent))
        if magnitude <= TANGENT_DEGENERACY_TOLERANCE:
            raise IndependentAuditError("degenerate HJ improved tangent")
        tangents.append(tangent / magnitude)
    return np.asarray(tangents, dtype=np.float64)


def spring_iteration(
    path: np.ndarray, fixed_energies: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return one pre-update tangent set, spring force, and residual."""

    points = np.asarray(path, dtype=np.float64)
    tangents = hj_improved_tangents(points, fixed_energies)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    force = (lengths[1:] - lengths[:-1])[:, None] * tangents
    residual = float(np.max(np.linalg.norm(force, axis=1)))
    return tangents, force, residual


def replay_spring_event(
    input_path: np.ndarray,
    fixed_energies: np.ndarray,
    iterations: int,
    *,
    dtau: float = SPRING_DTAU,
    tolerance: float = SPRING_TOLERANCE,
) -> tuple[np.ndarray, float, list[tuple[np.ndarray, np.ndarray]]]:
    """Replay the frozen spring loop with the original operation order.

    Each tangent log item is ``(path_before_iteration, tangent_output)``. The
    stored residual is the residual computed before the final iteration's
    possible update, matching the candidate loop contract.
    """

    if not (1 <= int(iterations) <= SPRING_MAX_ITER):
        raise IndependentAuditError("spring iteration count lies outside 1..500")
    original = np.asarray(input_path, dtype=np.float64)
    result = original.copy()
    fixed = np.asarray(fixed_energies, dtype=np.float64)
    tangent_log: list[tuple[np.ndarray, np.ndarray]] = []
    residual = np.inf
    stopped = False
    for index in range(int(iterations)):
        tangents, force, residual = spring_iteration(result, fixed)
        tangent_log.append((result.copy(), tangents.copy()))
        if residual <= tolerance:
            stopped = True
            if index + 1 != int(iterations):
                raise IndependentAuditError("stored iterations continue after spring convergence")
            break
        result[1:-1] += float(dtau) * force
    if stopped is False and int(iterations) < SPRING_MAX_ITER:
        raise IndependentAuditError("spring event stopped before convergence or max iteration")
    result[0] = original[0]
    result[-1] = original[-1]
    return result, float(residual), tangent_log


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise IndependentAuditError(f"shape mismatch during replay: {left.shape} != {right.shape}")
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def audit_spring_and_tangent_events(
    archive: Mapping[str, Any], *, absolute_tolerance: float = 2.0e-13
) -> SpringReplayReport:
    event_ids = np.asarray(archive["reparam_event_id"], dtype=np.int64)
    starts = np.asarray(archive["reparam_tangent_call_start"], dtype=np.int64)
    counts = np.asarray(archive["reparam_tangent_call_count"], dtype=np.int64)
    tangent_paths = np.asarray(archive["tangent_path"], dtype=np.float64)
    tangent_energies = np.asarray(archive["tangent_energy"], dtype=np.float64)
    tangent_outputs = np.asarray(archive["tangent_output"], dtype=np.float64)
    parents = np.asarray(archive["tangent_parent_reparam_event_id"], dtype=np.int64)
    max_output = 0.0
    max_tangent = 0.0
    max_residual = 0.0

    # Independently check every tangent, including tangent contexts outside
    # spring reparameterization.
    for index in range(tangent_outputs.shape[0]):
        recomputed = hj_improved_tangents(tangent_paths[index], tangent_energies[index])
        error = _max_abs(recomputed, tangent_outputs[index])
        max_tangent = max(max_tangent, error)
        if error > absolute_tolerance:
            raise IndependentAuditError(
                f"HJ tangent event {index + 1} mismatch: max abs {error:.3e}"
            )

    for offset, event_id in enumerate(event_ids.tolist()):
        output, residual, tangent_log = replay_spring_event(
            np.asarray(archive["reparam_input_path"])[offset],
            np.asarray(archive["reparam_fixed_energy"])[offset],
            int(np.asarray(archive["reparam_iterations"])[offset]),
            dtau=float(np.asarray(archive["spring_dtau"]).item()),
            tolerance=float(np.asarray(archive["spring_tolerance"]).item()),
        )
        output_error = _max_abs(output, np.asarray(archive["reparam_output_path"])[offset])
        residual_error = abs(
            residual - float(np.asarray(archive["reparam_final_residual"])[offset])
        )
        max_output = max(max_output, output_error)
        max_residual = max(max_residual, residual_error)
        if output_error > absolute_tolerance:
            raise IndependentAuditError(
                f"spring event {event_id} output mismatch: max abs {output_error:.3e}"
            )
        if residual_error > absolute_tolerance:
            raise IndependentAuditError(
                f"spring event {event_id} residual mismatch: abs {residual_error:.3e}"
            )
        start, count = int(starts[offset]), int(counts[offset])
        if count != len(tangent_log):
            raise IndependentAuditError(
                f"spring event {event_id} tangent count {count} != replay {len(tangent_log)}"
            )
        for local, (logged_path, logged_tangent) in enumerate(tangent_log):
            global_index = start + local
            if int(parents[global_index]) != event_id:
                raise IndependentAuditError("spring tangent has wrong parent event")
            path_error = _max_abs(logged_path, tangent_paths[global_index])
            tangent_error = _max_abs(logged_tangent, tangent_outputs[global_index])
            energy_error = _max_abs(
                np.asarray(archive["reparam_fixed_energy"])[offset],
                tangent_energies[global_index],
            )
            max_tangent = max(max_tangent, tangent_error)
            if max(path_error, tangent_error, energy_error) > absolute_tolerance:
                raise IndependentAuditError(
                    f"spring event {event_id} tangent log mismatch at local call {local}"
                )

    return SpringReplayReport(
        event_count=int(event_ids.size),
        tangent_call_count=int(np.count_nonzero(parents > 0)),
        maximum_output_absolute_error=max_output,
        maximum_tangent_absolute_error=max_tangent,
        maximum_residual_absolute_error=max_residual,
    )


def count_duplicate_charged_coordinates(archive: Mapping[str, Any]) -> int:
    """Reconstruct exact per-image caches and count illegally repeated misses."""

    image_indices = np.asarray(archive["evaluation_image_index"], dtype=np.int64)
    coordinates = np.asarray(archive["evaluation_coordinate"], dtype="<f8")
    caches: dict[int, set[bytes]] = {}
    duplicates = 0
    for image_index, coordinate in zip(image_indices.tolist(), coordinates):
        key = np.ascontiguousarray(coordinate).tobytes(order="C")
        image_cache = caches.setdefault(int(image_index), set())
        if key in image_cache:
            duplicates += 1
        else:
            image_cache.add(key)
    return duplicates


def locf_indices(observation_nof: np.ndarray, checkpoints: np.ndarray) -> np.ndarray:
    observed = np.asarray(observation_nof, dtype=np.int64)
    requested = np.asarray(checkpoints, dtype=np.int64)
    if observed.ndim != 1 or requested.ndim != 1 or observed.size == 0:
        raise IndependentAuditError("LOCF inputs must be non-empty one-dimensional arrays")
    if np.any(np.diff(observed) <= 0):
        raise IndependentAuditError("LOCF observations must be strictly increasing")
    indices = np.searchsorted(observed, requested, side="right") - 1
    if np.any(indices < 0):
        raise IndependentAuditError("checkpoint precedes first available public state")
    return indices.astype(np.int64)


def strict_checkpoint_comparison(
    candidate_i: np.ndarray,
    article_i: np.ndarray,
    candidate_barrier: np.ndarray,
    article_barrier: np.ndarray,
    *,
    reference_barrier: float = REFERENCE_BARRIER,
) -> dict[str, np.ndarray]:
    ci = np.asarray(candidate_i, dtype=np.float64)
    ai = np.asarray(article_i, dtype=np.float64)
    cb = np.abs(np.asarray(candidate_barrier, dtype=np.float64) - reference_barrier)
    ab = np.abs(np.asarray(article_barrier, dtype=np.float64) - reference_barrier)
    if not (ci.shape == ai.shape == cb.shape == ab.shape):
        raise IndependentAuditError("strict-comparison arrays have mismatched shapes")
    if not all(np.all(np.isfinite(item)) for item in (ci, ai, cb, ab)):
        raise IndependentAuditError("strict-comparison arrays contain non-finite values")
    i_win = ci < ai
    barrier_win = cb < ab
    return {
        "candidate_i_perp": ci,
        "article_i_perp": ai,
        "candidate_barrier_error": cb,
        "article_barrier_error": ab,
        "i_perp_strict": i_win,
        "barrier_strict": barrier_win,
        "joint_strict": i_win & barrier_win,
    }


def load_hash_authenticated_module(path: Path, expected_sha256: str, name: str) -> ModuleType:
    if sha256_file(path) != expected_sha256:
        raise IndependentAuditError(f"registered Python dependency hash mismatch: {path}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise IndependentAuditError(f"cannot load registered module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def audit_structural_result(path: Path) -> StructuralRunAudit:
    schema_report: ValidationReport = validate_result_archive(path)
    with np.load(path, allow_pickle=False) as archive:
        duplicates = count_duplicate_charged_coordinates(archive)
        if duplicates:
            raise IndependentAuditError(
                f"{duplicates} repeated exact coordinates were charged as new misses"
            )
        spring = audit_spring_and_tangent_events(archive)
    return StructuralRunAudit(
        result_path=str(path),
        result_sha256=sha256_file(path),
        schema=schema_report.to_jsonable(),
        duplicate_charged_coordinate_count=duplicates,
        spring_replay=asdict(spring),
    )


def _source_manifest_from_registration(
    campaign_root: Path, repository_root: Path, registration: Mapping[str, Any]
) -> None:
    for relative, expected in registration.get("source_sha256", {}).items():
        actual_path = resolve_registered_path(campaign_root, relative)
        if sha256_file(actual_path) != expected:
            raise IndependentAuditError(f"registered campaign source drift: {relative}")
    for key, expected in registration.get("dependency_sha256", {}).items():
        if not key.startswith("repo:"):
            raise IndependentAuditError(f"dependency key lacks repo: root tag: {key}")
        relative = key[len("repo:"):]
        actual_path = resolve_registered_path(repository_root, relative)
        if sha256_file(actual_path) != expected:
            raise IndependentAuditError(f"registered repository dependency drift: {relative}")


def staging_structural_main() -> int:
    paths = derive_campaign_paths(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registration",
        type=Path,
        default=paths.campaign_root / "registrations" / "CAMPAIGN_PREREGISTRATION.json",
    )
    parser.add_argument("--n10", type=Path, default=paths.campaign_root / "runs/n10/result.npz")
    parser.add_argument("--n20", type=Path, default=paths.campaign_root / "runs/n20/result.npz")
    parser.add_argument(
        "--structural-report",
        type=Path,
        default=paths.campaign_root / "reports/audit/STAGING_STRUCTURAL_AUDIT.json",
    )
    arguments = parser.parse_args()
    seal = arguments.registration.with_suffix(".sha256")
    try:
        registration, registration_hash = verify_registration_seal(arguments.registration, seal)
        _source_manifest_from_registration(
            paths.campaign_root, paths.repository_root, registration
        )
        runs = [audit_structural_result(arguments.n10), audit_structural_result(arguments.n20)]
        report = {
            "status": "integrity_partial_staging",
            "registration_sha256": registration_hash,
            "runs": [asdict(item) for item in runs],
            "scientific_pass_issued": False,
            "blocking_dependency": (
                "bind and test the exact registered Q10 potential/Article input adapter, "
                "then implement public-state potential replay, Article metric/LOCF tables, "
                "receipt authentication, and audit-manifest artifact hashing"
            ),
        }
        atomic_write_json(arguments.structural_report, report)
    except (FormalIOError, IndependentAuditError, OSError, ValueError) as exc:
        print(f"INTEGRITY FAILURE: {exc}")
        return 1
    print("STRUCTURAL/HJ AUDIT COMPLETE; SCIENTIFIC PASS INTENTIONALLY NOT ISSUED")
    return 2


def main() -> int:
    """Run the complete sealed audit implemented in the independent core."""

    from .final_audit_core import main as complete_audit_main

    return complete_audit_main()


if __name__ == "__main__":
    raise SystemExit(main())
