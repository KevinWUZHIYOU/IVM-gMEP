#!/usr/bin/env python3
"""Shared LEPS IVM/NEB machinery.

The LEPS energy and force definitions match the historical notebooks exactly.
All path tangents are delegated to ASE's ImprovedTangentMethod.  Every image has
an experiment-local, exact-coordinate cache: a cache miss evaluates energy and
force jointly, while a byte-identical revisit reuses both.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.mep.neb import NEB, NEBState


N_ATOMS = 2
DIMENSION = 6
N_INTERNAL = 20
N_IMAGES = N_INTERNAL + 2

REACTANT = np.array([0.74200064, 8.0], dtype=float)
PRODUCT = np.array([8.0, 0.74200064], dtype=float)

A_LEPS = 0.05
B_LEPS = 0.30
C_LEPS = 0.05
D_AB = 4.746
D_BC = 4.746
D_AC = 3.445
R0 = 0.742
MORSE_ALPHA = 1.942

TANGENT_DEFINITION = (
    "ASE 3.23.0 ImprovedTangentMethod: energy-weighted Henkelman-Jonsson "
    "improved tangent"
)


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_save_npz(destination: Path, **arrays) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)


def write_json(destination: Path, payload) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, destination)


def project_relative(path: Path, root: Path) -> str:
    path = Path(path).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _q_and_derivative(r, dissociation):
    r = np.asarray(r, dtype=float)
    displacement = r - R0
    exp_one = np.exp(-MORSE_ALPHA * displacement)
    exp_two = exp_one * exp_one
    value = 0.5 * dissociation * (1.5 * exp_two - exp_one)
    derivative = (
        0.5
        * dissociation
        * MORSE_ALPHA
        * (-3.0 * exp_two + exp_one)
    )
    return value, derivative


def _j_and_derivative(r, dissociation):
    r = np.asarray(r, dtype=float)
    displacement = r - R0
    exp_one = np.exp(-MORSE_ALPHA * displacement)
    exp_two = exp_one * exp_one
    value = 0.25 * dissociation * (exp_two - 6.0 * exp_one)
    derivative = (
        0.25
        * dissociation
        * MORSE_ALPHA
        * (-2.0 * exp_two + 6.0 * exp_one)
    )
    return value, derivative


def leps_energy_gradient(r_ab, r_bc):
    """Return the historical LEPS energy and its two-coordinate gradient."""
    r_ab = np.asarray(r_ab, dtype=float)
    r_bc = np.asarray(r_bc, dtype=float)
    r_ac = r_ab + r_bc

    q_ab, dq_ab = _q_and_derivative(r_ab, D_AB)
    q_bc, dq_bc = _q_and_derivative(r_bc, D_BC)
    q_ac, dq_ac = _q_and_derivative(r_ac, D_AC)
    j_ab, dj_ab = _j_and_derivative(r_ab, D_AB)
    j_bc, dj_bc = _j_and_derivative(r_bc, D_BC)
    j_ac, dj_ac = _j_and_derivative(r_ac, D_AC)

    pa = 1.0 + A_LEPS
    pb = 1.0 + B_LEPS
    pc = 1.0 + C_LEPS

    term_q = q_ab / pa + q_bc / pb + q_ac / pc
    radicand = (
        j_ab**2 / pa**2
        + j_bc**2 / pb**2
        + j_ac**2 / pc**2
        - j_ab * j_bc / (pa * pb)
        - j_bc * j_ac / (pb * pc)
        - j_ab * j_ac / (pa * pc)
    )
    radicand = np.maximum(radicand, 0.0)
    root = np.sqrt(radicand)
    energy = term_q - root

    d_radicand_ab = (
        2.0 * j_ab * dj_ab / pa**2
        + 2.0 * j_ac * dj_ac / pc**2
        - dj_ab * j_bc / (pa * pb)
        - j_bc * dj_ac / (pb * pc)
        - (dj_ab * j_ac + j_ab * dj_ac) / (pa * pc)
    )
    d_radicand_bc = (
        2.0 * j_bc * dj_bc / pb**2
        + 2.0 * j_ac * dj_ac / pc**2
        - j_ab * dj_bc / (pa * pb)
        - (dj_bc * j_ac + j_bc * dj_ac) / (pb * pc)
        - j_ab * dj_ac / (pa * pc)
    )
    safe_root = np.maximum(root, 1e-300)
    grad_ab = dq_ab / pa + dq_ac / pc - 0.5 * d_radicand_ab / safe_root
    grad_bc = dq_bc / pb + dq_ac / pc - 0.5 * d_radicand_bc / safe_root
    gradient = np.stack((grad_ab, grad_bc), axis=-1)
    return energy, gradient


def leps_energy(r_ab, r_bc):
    return leps_energy_gradient(r_ab, r_bc)[0]


def physical_to_flat(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    single = points.ndim == 1
    points = points.reshape(-1, 2)
    flat = np.zeros((points.shape[0], DIMENSION), dtype=float)
    flat[:, 0] = points[:, 0]
    flat[:, 3] = points[:, 1]
    return flat[0] if single else flat


def flat_to_physical(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    single = path.ndim == 1
    path = path.reshape(-1, DIMENSION)
    physical = path[:, [0, 3]]
    return physical[0] if single else physical


class LEPSBackend:
    """Joint LEPS energy-force oracle with a raw call counter."""

    def __init__(self, label: str = "optimization"):
        self.label = str(label)
        self.raw_joint_evaluations = 0

    def evaluate_joint(self, positions_flat: np.ndarray):
        physical = flat_to_physical(positions_flat)
        energy, gradient = leps_energy_gradient(physical[0], physical[1])
        force = np.zeros(DIMENSION, dtype=float)
        force[0] = -float(gradient[0])
        force[3] = -float(gradient[1])
        self.raw_joint_evaluations += 1
        return float(energy), force

    def snapshot(self):
        return {
            "label": self.label,
            "raw_joint_evaluations": int(self.raw_joint_evaluations),
        }

    def restore(self, state):
        if str(state["label"]) != self.label:
            raise ValueError("LEPS backend checkpoint label mismatch")
        self.raw_joint_evaluations = int(state["raw_joint_evaluations"])


class EvaluationBudgetReached(RuntimeError):
    pass


@dataclass
class EvaluationCounters:
    energy_requests: int = 0
    force_requests: int = 0
    cache_hits: int = 0
    joint_evaluations: int = 0
    total_query_events: int = 0
    both_requests: int = 0
    budget_blocked_misses: int = 0


class ImageEvaluator:
    """One image's run-local historical joint energy-force cache."""

    def __init__(
        self,
        backend: LEPSBackend,
        image_index: int,
        max_joint_evaluations: Optional[int] = None,
    ):
        self.backend = backend
        self.image_index = int(image_index)
        self.max_joint_evaluations = (
            None
            if max_joint_evaluations is None
            else int(max_joint_evaluations)
        )
        self.counters = EvaluationCounters()
        self._positions = None
        self._energy = None
        self._forces = None
        self._history = {}

    @staticmethod
    def _coordinate_key(positions_flat):
        positions = np.ascontiguousarray(
            np.asarray(positions_flat, dtype=np.float64).reshape(-1)
        )
        if positions.shape != (DIMENSION,):
            raise ValueError(f"Expected {DIMENSION} coordinates, got {positions.shape}")
        if not np.all(np.isfinite(positions)):
            raise ValueError("Image coordinates must be finite")
        return positions, positions.tobytes()

    def has_cached(self, positions_flat) -> bool:
        _, key = self._coordinate_key(positions_flat)
        return key in self._history

    def query(self, positions_flat, property_name="force"):
        if property_name not in {"energy", "force", "both"}:
            raise ValueError(f"Unsupported property request: {property_name}")
        # Validate before recording an API event. A NaN trial generated by an
        # unstable optimizer is rejected by the caller and is not a valid
        # energy/force request or backend evaluation.
        positions, key = self._coordinate_key(positions_flat)
        self.counters.total_query_events += 1
        if property_name in {"energy", "both"}:
            self.counters.energy_requests += 1
        if property_name in {"force", "both"}:
            self.counters.force_requests += 1
        if property_name == "both":
            self.counters.both_requests += 1

        cached = self._history.get(key)
        if cached is not None:
            self.counters.cache_hits += 1
            self._positions = positions.copy()
            self._energy = float(cached[0])
            self._forces = cached[1].copy()
        else:
            if (
                self.max_joint_evaluations is not None
                and self.counters.joint_evaluations
                >= self.max_joint_evaluations
            ):
                self.counters.budget_blocked_misses += 1
                raise EvaluationBudgetReached(
                    f"Image {self.image_index} reached its "
                    f"{self.max_joint_evaluations}-evaluation budget"
                )
            energy, forces = self.backend.evaluate_joint(positions)
            self._positions = positions.copy()
            self._energy = float(energy)
            self._forces = np.asarray(forces, dtype=float).copy()
            self._history[key] = (self._energy, self._forces.copy())
            self.counters.joint_evaluations += 1

        if property_name == "energy":
            return float(self._energy)
        if property_name == "force":
            return self._forces.copy()
        return float(self._energy), self._forces.copy()

    def snapshot(self):
        history = []
        for key, (energy, forces) in self._history.items():
            history.append(
                {
                    "positions": np.frombuffer(key, dtype=np.float64).tolist(),
                    "energy": float(energy),
                    "forces": forces.tolist(),
                }
            )
        return {
            "image_index": self.image_index,
            "counters": asdict(self.counters),
            "positions": None if self._positions is None else self._positions.tolist(),
            "energy": self._energy,
            "forces": None if self._forces is None else self._forces.tolist(),
            "history": history,
        }

    def restore(self, state):
        if int(state["image_index"]) != self.image_index:
            raise ValueError("Image evaluator checkpoint index mismatch")
        self.counters = EvaluationCounters(**state["counters"])
        self._positions = (
            None
            if state["positions"] is None
            else np.asarray(state["positions"], dtype=float)
        )
        self._energy = None if state["energy"] is None else float(state["energy"])
        self._forces = (
            None
            if state["forces"] is None
            else np.asarray(state["forces"], dtype=float)
        )
        self._history = {}
        for item in state.get("history", []):
            _, key = self._coordinate_key(item["positions"])
            self._history[key] = (
                float(item["energy"]),
                np.asarray(item["forces"], dtype=float),
            )


class AuditedLEPSCalculator(Calculator):
    """ASE adapter that records force and energy API requests separately."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, evaluator: ImageEvaluator):
        super().__init__()
        self.evaluator = evaluator

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            return super().get_property(name, atoms, allow_calculation)
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("Audited calculator requires Atoms")
        positions = atoms.get_positions().reshape(-1)
        if name == "energy":
            value = self.evaluator.query(positions, "energy")
            self.results["energy"] = value
            if self.evaluator._forces is not None:
                self.results["forces"] = self.evaluator._forces.reshape(-1, 3).copy()
            return value
        value = self.evaluator.query(positions, "force").reshape(-1, 3)
        self.results["forces"] = value.copy()
        if self.evaluator._energy is not None:
            self.results["energy"] = float(self.evaluator._energy)
        return value

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
        if atoms is None:
            atoms = self.atoms
        energy, forces = self.evaluator.query(
            atoms.get_positions().reshape(-1), "both"
        )
        self.atoms = atoms.copy()
        self.results = {
            "energy": energy,
            "forces": forces.reshape(-1, 3),
        }


def make_evaluators(backend, count=N_IMAGES, internal_limit=None):
    return [
        ImageEvaluator(
            backend,
            image_index=i,
            max_joint_evaluations=(
                internal_limit if 0 < i < count - 1 else None
            ),
        )
        for i in range(count)
    ]


def counter_arrays(evaluators):
    return {
        "joint_evaluations": np.asarray(
            [e.counters.joint_evaluations for e in evaluators], dtype=int
        ),
        "energy_requests": np.asarray(
            [e.counters.energy_requests for e in evaluators], dtype=int
        ),
        "force_requests": np.asarray(
            [e.counters.force_requests for e in evaluators], dtype=int
        ),
        "cache_hits": np.asarray(
            [e.counters.cache_hits for e in evaluators], dtype=int
        ),
        "total_query_events": np.asarray(
            [e.counters.total_query_events for e in evaluators], dtype=int
        ),
        "both_requests": np.asarray(
            [e.counters.both_requests for e in evaluators], dtype=int
        ),
        "budget_blocked_misses": np.asarray(
            [e.counters.budget_blocked_misses for e in evaluators], dtype=int
        ),
    }


def audit_counter_consistency(backend, evaluators):
    counters = counter_arrays(evaluators)
    total = int(np.sum(counters["joint_evaluations"]))
    request_identity = (
        counters["energy_requests"]
        + counters["force_requests"]
        - counters["both_requests"]
        == counters["total_query_events"]
    )
    outcome_identity = (
        counters["cache_hits"]
        + counters["joint_evaluations"]
        + counters["budget_blocked_misses"]
        == counters["total_query_events"]
    )
    return {
        "backend_label": backend.label,
        "backend_raw_joint_evaluations": int(backend.raw_joint_evaluations),
        "sum_image_joint_evaluations": total,
        "consistent": bool(
            total == backend.raw_joint_evaluations
            and np.all(request_identity)
            and np.all(outcome_identity)
        ),
        "per_image_joint_evaluations": counters["joint_evaluations"].tolist(),
        "per_image_energy_requests": counters["energy_requests"].tolist(),
        "per_image_force_requests": counters["force_requests"].tolist(),
        "per_image_cache_hits": counters["cache_hits"].tolist(),
        "per_image_total_query_events": counters["total_query_events"].tolist(),
        "per_image_both_requests": counters["both_requests"].tolist(),
        "per_image_budget_blocked_misses": counters[
            "budget_blocked_misses"
        ].tolist(),
    }


def atoms_from_flat(flat):
    atoms = Atoms("H2", positions=np.asarray(flat, dtype=float).reshape(2, 3))
    atoms.calc = None
    return atoms


def attach_audited_calculators(path, evaluators):
    images = []
    for flat, evaluator in zip(path, evaluators):
        atoms = atoms_from_flat(flat)
        atoms.calc = AuditedLEPSCalculator(evaluator)
        images.append(atoms)
    return images


def improved_tangents(path: np.ndarray, energies: np.ndarray, k=1.0):
    """Call ASE's ImprovedTangentMethod directly on known coordinates/energies."""
    path = np.asarray(path, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if path.shape != (N_IMAGES, DIMENSION):
        raise ValueError(f"Unexpected path shape: {path.shape}")
    if energies.shape != (N_IMAGES,):
        raise ValueError(f"Unexpected energy shape: {energies.shape}")
    images = [atoms_from_flat(flat) for flat in path]
    neb = NEB(
        images,
        k=float(k),
        climb=False,
        parallel=False,
        method="improvedtangent",
    )
    state = NEBState(neb, images, energies)
    spring1 = state.spring(0)
    tangents = []
    for image_index in range(1, N_IMAGES - 1):
        spring2 = state.spring(image_index)
        tangent = neb.neb_method.get_tangent(
            state, spring1, spring2, image_index
        )
        tangents.append(np.asarray(tangent, dtype=float).reshape(-1))
        spring1 = spring2
    return np.asarray(tangents, dtype=float)


def project_normal(vectors, tangents):
    vectors = np.asarray(vectors, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    return vectors - np.sum(vectors * tangents, axis=1, keepdims=True) * tangents


def local_path_weights(path):
    segment_lengths = np.linalg.norm(np.diff(np.asarray(path, dtype=float), axis=0), axis=1)
    return 0.5 * (segment_lengths[:-1] + segment_lengths[1:])


def equidistance_diagnostics(path):
    path = np.asarray(path, dtype=float)
    ds = 1.0 / float(path.shape[0] - 1)
    # Only internal vertices enter the discrete spring/equidistance condition.
    # This central stencil has the useful exact identity
    #   phi' . phi'' = (|dx_next|^2 - |dx_prev|^2) / (2 ds^3),
    # so an equal-chord polygon gives zero to round-off even when it is curved.
    first = (path[2:] - path[:-2]) / (2.0 * ds)
    second = (path[2:] - 2.0 * path[1:-1] + path[:-2]) / ds**2
    inner = np.sum(first * second, axis=1)
    speed = np.linalg.norm(first, axis=1)
    integral = float(np.sum(inner**2 * speed) * ds)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    mean = float(np.mean(lengths))
    minimum = float(np.min(lengths))
    maximum = float(np.max(lengths))
    unit_spring_force = np.abs(np.diff(lengths))
    return {
        # Discrete approximation to
        # I_arc = integral_0^1 (phi'(s) . phi''(s))^2 dphi.
        # Keep the historical key as an exact alias for backward compatibility.
        "i_arc": integral,
        "equidistance_integral": integral,
        # Standard NEB tangential spring-force magnitude with a common k=1.
        # This makes the spacing comparison independent of the scanned NEB k.
        "spring_force_max_unit": (
            float(np.max(unit_spring_force))
            if unit_spring_force.size
            else 0.0
        ),
        "equidistance_relative_defect": (
            float(integral / max(np.mean(speed) ** 5, 1e-300))
            if mean > 0.0
            else np.inf
        ),
        "segment_length_min": minimum,
        "segment_length_mean": mean,
        "segment_length_max": maximum,
        "segment_length_cv": (
            float(np.std(lengths) / mean) if mean > 0.0 else np.inf
        ),
        "segment_length_ratio": (
            float(maximum / minimum) if minimum > 0.0 else np.inf
        ),
    }


def path_diagnostics(path, energies, forces_internal):
    path = np.asarray(path, dtype=float)
    energies = np.asarray(energies, dtype=float)
    forces_internal = np.asarray(forces_internal, dtype=float)
    tangents = improved_tangents(path, energies)
    force_perp = project_normal(forces_internal, tangents)
    weights = local_path_weights(path)
    per_image_l2 = np.linalg.norm(force_perp, axis=1)
    diagnostics = {
        "tangents": tangents,
        "force_perp": force_perp,
        "per_image_l2": per_image_l2,
        "fmax": float(np.max(per_image_l2)),
        "i_perp": float(np.sum(weights * per_image_l2**2)),
        "barrier": float(np.max(energies) - energies[0]),
    }
    diagnostics.update(equidistance_diagnostics(path))
    return diagnostics


def evaluate_path(path, evaluators):
    energies = np.empty(N_IMAGES, dtype=float)
    forces = np.empty((N_IMAGES, DIMENSION), dtype=float)
    for i, (position, evaluator) in enumerate(zip(path, evaluators)):
        energies[i], forces[i] = evaluator.query(position, "both")
    return energies, forces


def load_common_path(path_file):
    with np.load(path_file, allow_pickle=False) as saved:
        path = np.asarray(saved["initial_path"], dtype=float)
    if path.shape != (N_IMAGES, DIMENSION):
        raise ValueError(f"Unexpected initial path shape: {path.shape}")
    return path


def build_common_idpp_path():
    """Reproduce the old notebooks' ASE IDPP initialization exactly."""
    start = physical_to_flat(REACTANT)
    end = physical_to_flat(PRODUCT)
    images = [atoms_from_flat(start)]
    images.extend(atoms_from_flat(start) for _ in range(N_INTERNAL))
    images.append(atoms_from_flat(end))
    neb = NEB(images, climb=False, parallel=False)
    try:
        neb.interpolate("idpp")
        source = "ASE IDPP"
    except Exception:
        neb.interpolate()
        source = "linear fallback after IDPP failure"
    path = np.asarray(
        [image.get_positions().reshape(-1) for image in neb.images],
        dtype=float,
    )
    path[0] = start
    path[-1] = end
    return path, source


def improved_hermite_equal_chord(path, frozen_tangents):
    """Equal-chord reparameterization on a frozen improved-tangent Hermite curve."""
    from scipy.interpolate import CubicHermiteSpline
    from scipy.optimize import least_squares

    path = np.asarray(path, dtype=float)
    frozen_tangents = np.asarray(frozen_tangents, dtype=float)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    if np.any(lengths <= 1e-14):
        raise ValueError("Hermite reparameterization received a zero segment")
    parameter = np.concatenate(([0.0], np.cumsum(lengths)))
    derivatives = np.empty_like(path)
    derivatives[1:-1] = frozen_tangents
    derivatives[0] = (path[1] - path[0]) / lengths[0]
    derivatives[-1] = (path[-1] - path[-2]) / lengths[-1]
    spline = CubicHermiteSpline(parameter, path, derivatives, axis=0)
    span = float(parameter[-1] - parameter[0])
    increments = np.diff(parameter)
    initial_logits = np.log(increments[:-1] / increments[-1])

    def positions_from_logits(logits):
        logits = np.concatenate((np.asarray(logits, dtype=float), [0.0]))
        logits -= np.max(logits)
        local = np.exp(logits)
        local *= span / np.sum(local)
        nodes = parameter[0] + np.concatenate(([0.0], np.cumsum(local)))
        return np.asarray(spline(nodes), dtype=float)

    def residual(logits):
        candidate = positions_from_logits(logits)
        candidate_lengths = np.linalg.norm(np.diff(candidate, axis=0), axis=1)
        return candidate_lengths[1:] - candidate_lengths[:-1]

    solution = least_squares(
        residual,
        initial_logits,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=2000,
    )
    result = positions_from_logits(solution.x)
    result[0] = path[0]
    result[-1] = path[-1]
    final_lengths = np.linalg.norm(np.diff(result, axis=0), axis=1)
    spring_residual = float(np.max(np.abs(np.diff(final_lengths))))
    return result, spring_residual


def rms_segment_trust_radius(path, fraction=2.0, maximum=1.0):
    segment_rms = np.linalg.norm(np.diff(path, axis=0), axis=1) / np.sqrt(
        float(N_ATOMS)
    )
    return min(float(fraction) * float(np.median(segment_rms)), float(maximum))


def clip_max_atomic_displacement(direction, max_step):
    direction = np.asarray(direction, dtype=float)
    largest = float(np.max(np.linalg.norm(direction.reshape(-1, 3), axis=1)))
    if largest > float(max_step):
        direction = direction * (float(max_step) / largest)
    return direction


def densified_path(path, subdivisions):
    path = np.asarray(path, dtype=float)
    subdivisions = int(subdivisions)
    fractions = np.arange(subdivisions, dtype=float) / float(subdivisions)
    segments = (
        path[:-1, None, :] * (1.0 - fractions[None, :, None])
        + path[1:, None, :] * fractions[None, :, None]
    )
    return np.concatenate(
        (segments.reshape(-1, path.shape[1]), path[-1:]), axis=0
    )


def densified_barrier(path, subdivisions=80):
    dense = densified_path(path, subdivisions)
    physical = flat_to_physical(dense)
    energies = leps_energy(physical[:, 0], physical[:, 1])
    return float(np.max(energies) - energies[0]), energies


def reference_saddle():
    """Locate and certify the first-order LEPS saddle between the endpoints."""
    from scipy.optimize import root

    def gradient(point):
        return np.asarray(
            leps_energy_gradient(point[0], point[1])[1], dtype=float
        )

    solution = root(gradient, np.array([1.15, 0.86], dtype=float))
    if not solution.success:
        raise RuntimeError(f"LEPS saddle solve failed: {solution.message}")
    point = np.asarray(solution.x, dtype=float)
    energy = float(leps_energy(point[0], point[1]))
    step = 1e-4
    hessian = np.empty((2, 2), dtype=float)
    for j in range(2):
        offset = np.zeros(2)
        offset[j] = step
        hessian[:, j] = (gradient(point + offset) - gradient(point - offset)) / (
            2.0 * step
        )
    eigvals = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))
    reactant_energy = float(leps_energy(REACTANT[0], REACTANT[1]))
    if not (eigvals[0] < 0.0 < eigvals[1]):
        raise RuntimeError("Located LEPS stationary point is not an index-1 saddle")
    return {
        "point": point,
        "energy": energy,
        "reactant_energy": reactant_energy,
        "forward_barrier": energy - reactant_energy,
        "hessian_eigenvalues": eigvals,
        "gradient_norm": float(np.linalg.norm(gradient(point))),
    }
