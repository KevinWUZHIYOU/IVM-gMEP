#!/usr/bin/env python3
"""Shared Alanine dipeptide IVM/NEB machinery.

Every production tangent is the normalized central difference
``x[i + 1] - x[i - 1]``.  OpenMM evaluations are jointly cached as
energy+force pairs and audited independently for every path image.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read
from ase.mep.neb import NEB, NEBMethod


EV_TO_KCAL = 23.0605
N_INTERNAL = 20
N_IMAGES = N_INTERNAL + 2
TANGENT_DEFINITION = "normalized central difference: x[i+1] - x[i-1]"
TOPOLOGY_FILENAME = "ala2_fixed.pdb"
ENDPOINT_CACHE_FILENAME = "ala2_scan_cache.npz"
FORCEFIELD_FILES = ("amber14-all.xml",)


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


def create_openmm_context(
    system,
    platform_preference: Iterable[str] = ("CPU",),
    precision: str = "double",
):
    from openmm import Context, Platform, VerletIntegrator
    from openmm.unit import femtoseconds

    available = {
        Platform.getPlatform(i).getName(): Platform.getPlatform(i)
        for i in range(Platform.getNumPlatforms())
    }
    errors = []
    for name in platform_preference:
        if name not in available:
            continue
        properties = {}
        if name in {"CUDA", "OpenCL"}:
            properties["Precision"] = precision
        try:
            integrator = VerletIntegrator(1.0 * femtoseconds)
            context = Context(system, integrator, available[name], properties)
            return context, integrator, available[name], properties
        except Exception as exc:  # pragma: no cover - platform dependent
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        f"No requested OpenMM platform available; requested={list(platform_preference)}, "
        f"available={list(available)}, errors={errors}"
    )


class OpenMMBackend:
    """One persistent OpenMM context with a raw joint energy-force call counter."""

    def __init__(
        self,
        topology_pdb: Path,
        platform_preference: Iterable[str] = ("CPU",),
        precision: str = "double",
        label: str = "optimization",
        forcefield_files: Iterable[str] = FORCEFIELD_FILES,
    ):
        from openmm.app import ForceField, NoCutoff, PDBFile

        pdb = PDBFile(str(topology_pdb))
        forcefield_files = tuple(str(item) for item in forcefield_files)
        forcefield = ForceField(*forcefield_files)
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff)
        context, integrator, platform, properties = create_openmm_context(
            system, platform_preference=platform_preference, precision=precision
        )
        self.context = context
        self.integrator = integrator
        self.system = system
        self.topology = pdb.topology
        self.n_atoms = system.getNumParticles()
        self.dimension = 3 * self.n_atoms
        self.platform_name = platform.getName()
        self.platform_properties = properties
        self.forcefield_files = forcefield_files
        self.label = str(label)
        self.raw_joint_evaluations = 0

    def evaluate_joint(self, positions_flat: np.ndarray):
        """Return energy (eV) and force (eV/A) in one OpenMM state request."""
        from openmm.unit import angstrom, kilocalorie_per_mole

        positions = np.asarray(positions_flat, dtype=float).reshape(self.n_atoms, 3)
        self.context.setPositions(positions * 0.1)
        state = self.context.getState(getEnergy=True, getForces=True)
        energy_kcal = state.getPotentialEnergy().value_in_unit(kilocalorie_per_mole)
        force_kcal_a = np.asarray(
            state.getForces(asNumpy=True).value_in_unit(
                kilocalorie_per_mole / angstrom
            )
        )
        self.raw_joint_evaluations += 1
        return (
            float(energy_kcal / EV_TO_KCAL),
            np.asarray(force_kcal_a / EV_TO_KCAL, dtype=float).reshape(-1),
        )


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
    """Run-local exact-coordinate cache and counters for one path image.

    The cache belongs to this evaluator instance (and therefore to one
    experiment).  It is deliberately not shared between runs.  Every cache
    miss obtains energy and force together, while any byte-identical
    coordinate evaluated earlier in the same run reuses that joint result.
    """

    def __init__(
        self,
        backend: OpenMMBackend,
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
        self._positions: Optional[np.ndarray] = None
        self._energy: Optional[float] = None
        self._forces: Optional[np.ndarray] = None
        self._history: dict[bytes, tuple[float, np.ndarray]] = {}

    @staticmethod
    def _coordinate_key(positions_flat: np.ndarray) -> tuple[np.ndarray, bytes]:
        positions = np.ascontiguousarray(
            np.asarray(positions_flat, dtype=np.float64).reshape(-1)
        )
        if not np.all(np.isfinite(positions)):
            raise ValueError("Image coordinates must be finite")
        return positions, positions.tobytes()

    def has_cached(self, positions_flat: np.ndarray) -> bool:
        _, key = self._coordinate_key(positions_flat)
        return key in self._history

    def has_current(self, positions_flat: np.ndarray) -> bool:
        """Backward-compatible alias: any run-local cached coordinate is valid."""
        return self.has_cached(positions_flat)

    def query(self, positions_flat: np.ndarray, property_name: str = "force"):
        """Query energy or force; a cache miss computes and stores both."""
        if property_name not in {"energy", "force", "both"}:
            raise ValueError(f"Unsupported property request: {property_name}")
        self.counters.total_query_events += 1
        if property_name in {"energy", "both"}:
            self.counters.energy_requests += 1
        if property_name in {"force", "both"}:
            self.counters.force_requests += 1
        if property_name == "both":
            self.counters.both_requests += 1
        positions, key = self._coordinate_key(positions_flat)
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

    def snapshot(self) -> dict:
        history = []
        for key, (energy, forces) in self._history.items():
            positions = np.frombuffer(key, dtype=np.float64).copy()
            history.append(
                {
                    "positions": positions.tolist(),
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

    def restore(self, state: dict) -> None:
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
        if "history" in state:
            for item in state["history"]:
                positions, key = self._coordinate_key(item["positions"])
                del positions
                self._history[key] = (
                    float(item["energy"]),
                    np.asarray(item["forces"], dtype=float),
                )
        elif self._positions is not None:
            # Backward compatibility for old one-entry checkpoints.
            _, key = self._coordinate_key(self._positions)
            self._history[key] = (float(self._energy), self._forces.copy())


def restore_evaluator_states(
    backend: OpenMMBackend,
    evaluators: list[ImageEvaluator],
    states: list[dict],
) -> None:
    """Restore all image caches and synchronize the shared backend counter."""
    if len(evaluators) != len(states):
        raise ValueError("Evaluator checkpoint length mismatch")
    if backend.raw_joint_evaluations != 0:
        raise ValueError("Checkpoint restore requires a fresh OpenMM backend")
    for evaluator, state in zip(evaluators, states):
        if evaluator.backend is not backend:
            raise ValueError("All restored evaluators must share the backend")
        evaluator.restore(state)
    backend.raw_joint_evaluations = int(
        sum(item.counters.joint_evaluations for item in evaluators)
    )


class AuditedOpenMMCalculator(Calculator):
    """ASE calculator backed by an ImageEvaluator.

    ASE commonly asks for force and then energy at the same coordinates.  This
    implementation records two API requests but performs one joint OpenMM
    evaluation.
    """

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
        forces = self.evaluator.query(positions, "force").reshape(-1, 3)
        self.results["forces"] = forces.copy()
        if self.evaluator._energy is not None:
            self.results["energy"] = float(self.evaluator._energy)
        return forces

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


def make_evaluators(
    backend: OpenMMBackend,
    count: int = N_IMAGES,
    internal_limit: Optional[int] = None,
):
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


def counter_arrays(evaluators: list[ImageEvaluator]) -> dict[str, np.ndarray]:
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


def audit_counter_consistency(
    backend: OpenMMBackend, evaluators: list[ImageEvaluator]
) -> dict:
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
        "per_image_total_query_events": counters[
            "total_query_events"
        ].tolist(),
        "per_image_both_requests": counters["both_requests"].tolist(),
        "per_image_budget_blocked_misses": counters[
            "budget_blocked_misses"
        ].tolist(),
        "per_image_request_identity_ok": request_identity.tolist(),
        "per_image_query_outcome_identity_ok": outcome_identity.tolist(),
    }


def central_tangents(path: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[0] < 3:
        raise ValueError(f"Invalid path shape: {path.shape}")
    tangent = path[2:] - path[:-2]
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("Central-difference tangent has zero norm")
    return tangent / norms


def project_normal(vectors: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    return vectors - np.sum(vectors * tangents, axis=1, keepdims=True) * tangents


def local_path_weights(path: np.ndarray) -> np.ndarray:
    segment_lengths = np.linalg.norm(path[1:] - path[:-1], axis=1)
    return 0.5 * (segment_lengths[:-1] + segment_lengths[1:])


def equidistance_diagnostics(path: np.ndarray) -> dict:
    r"""Measure constant-speed parameterization of a discrete path.

    With equally spaced labels ``s in [0, 1]``, constant image spacing is
    equivalent to ``d ||phi'(s)||^2 / ds = 2 phi'(s) . phi''(s) = 0``.
    The legacy raw monitor is discretized by centered differences as

        integral_0^1 (phi'(s) . phi''(s))^2 ds.

    The paper-facing monitor for this experiment uses the arclength measure:

        I_arc = integral_0^1 (phi'(s) . phi''(s))^2 |phi'(s)| ds.

    The centered stencil also gives the exact discrete identity

        phi'(s_i) . phi''(s_i)
        = (||x_{i+1}-x_i||^2 - ||x_i-x_{i-1}||^2) / (2 ds^3).

    Thus the raw integral measures adjacent squared-segment-length
    differences, not path curvature.  Segment CV, max/min, and a
    dimensionless relative defect are reported beside it.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[0] < 3:
        raise ValueError(f"Invalid path shape: {path.shape}")
    ds = 1.0 / float(path.shape[0] - 1)
    first = (path[2:] - path[:-2]) / (2.0 * ds)
    second = (path[2:] - 2.0 * path[1:-1] + path[:-2]) / (ds * ds)
    speed_derivative_half = np.sum(first * second, axis=1)
    integral = float(np.sum(speed_derivative_half**2) * ds)
    speeds = np.linalg.norm(first, axis=1)
    i_arc = float(np.sum(speed_derivative_half**2 * speeds) * ds)
    segment_lengths = np.linalg.norm(path[1:] - path[:-1], axis=1)
    mean_length = float(np.mean(segment_lengths))
    minimum = float(np.min(segment_lengths))
    maximum = float(np.max(segment_lengths))
    cv = (
        float(np.std(segment_lengths) / mean_length)
        if mean_length > 0.0
        else np.inf
    )
    relative_defect = (
        float(4.0 * ds**5 * integral / mean_length**4)
        if mean_length > 0.0
        else np.inf
    )
    ratio = maximum / minimum if minimum > 0.0 else np.inf
    max_unit_spring_force = float(
        np.max(np.abs(segment_lengths[1:] - segment_lengths[:-1]))
    )
    return {
        "equidistance_integral": integral,
        "i_arc": i_arc,
        "max_unit_spring_force": max_unit_spring_force,
        "equidistance_relative_defect": relative_defect,
        "segment_length_min": minimum,
        "segment_length_mean": mean_length,
        "segment_length_max": maximum,
        "segment_length_cv": cv,
        "segment_length_ratio": float(ratio),
    }


def path_diagnostics(
    path: np.ndarray, energies: np.ndarray, forces_internal: np.ndarray
) -> dict:
    path = np.asarray(path, dtype=float)
    energies = np.asarray(energies, dtype=float)
    forces_internal = np.asarray(forces_internal, dtype=float)
    tangents = central_tangents(path)
    force_perp = project_normal(forces_internal, tangents)
    weights = local_path_weights(path)
    per_image_l2 = np.linalg.norm(force_perp, axis=1)
    per_atom_norm = np.linalg.norm(
        force_perp.reshape(force_perp.shape[0], -1, 3), axis=2
    )
    diagnostics = {
        "tangents": tangents,
        "force_perp": force_perp,
        "per_image_l2": per_image_l2,
        "fmax": float(np.max(per_atom_norm)),
        "i_perp": float(np.sum(weights * per_image_l2**2)),
        "barrier_kcal": float((np.max(energies) - energies[0]) * EV_TO_KCAL),
    }
    diagnostics.update(equidistance_diagnostics(path))
    return diagnostics


def evaluate_path(
    path: np.ndarray,
    evaluators: list[ImageEvaluator],
    request_property: str = "force",
):
    path = np.asarray(path, dtype=float)
    if len(evaluators) != path.shape[0]:
        raise ValueError("Evaluator count does not match path image count")
    energies = np.empty(path.shape[0], dtype=float)
    forces = np.empty_like(path)
    for i, (position, evaluator) in enumerate(zip(path, evaluators)):
        energy, force = evaluator.query(position, "both")
        energies[i] = energy
        forces[i] = force
    return energies, forces


class CentralDifferenceNEBMethod(NEBMethod):
    """Standard NEB projection with one central-difference tangent definition."""

    tangent_definition = TANGENT_DEFINITION

    def get_tangent(self, state, spring1, spring2, i):
        del state, i
        tangent = spring1.t + spring2.t
        norm = np.linalg.norm(tangent)
        if norm <= 1e-30:
            raise ValueError("Central-difference NEB tangent has zero norm")
        return tangent / norm

    def add_image_force(
        self, state, tangential_force, tangent, imgforce, spring1, spring2, i
    ):
        del state, i
        imgforce -= tangential_force * tangent
        imgforce += (spring2.nt * spring2.k - spring1.nt * spring1.k) * tangent


class CentralDifferenceNEB(NEB):
    """ASE NEB whose runtime method is always CentralDifferenceNEBMethod."""

    def __init__(self, images, **kwargs):
        # ASE validates a closed list of method names.  We initialize its base
        # storage without evaluating a tangent, then replace the method before
        # any interpolation or force request can occur.
        kwargs.pop("method", None)
        super().__init__(images, method="aseneb", **kwargs)
        self.method = "centraldifference"
        self.neb_method = CentralDifferenceNEBMethod(self)


def atoms_from_flat(template: Atoms, flat: np.ndarray) -> Atoms:
    atoms = template.copy()
    atoms.set_positions(np.asarray(flat, dtype=float).reshape(len(template), 3))
    atoms.calc = None
    return atoms


def load_endpoints(input_dir: Path):
    input_dir = Path(input_dir)
    template = read(input_dir / TOPOLOGY_FILENAME)
    with np.load(input_dir / ENDPOINT_CACHE_FILENAME, allow_pickle=False) as cache:
        positions = np.asarray(cache["unique_positions"], dtype=float)
        energies = np.asarray(cache["unique_energy"], dtype=float)
    if positions.ndim != 3 or positions.shape[1:] != (len(template), 3):
        raise ValueError(
            "Endpoint cache shape does not match alanine topology: "
            f"{positions.shape} versus ({len(template)}, 3)"
        )
    if positions.shape[0] < 2 or energies.shape != (positions.shape[0],):
        raise ValueError("Endpoint cache must contain at least two ranked minima")
    ranking = np.argsort(energies, kind="stable")
    return (
        template,
        positions[ranking[0]].reshape(-1).astype(float),
        positions[ranking[1]].reshape(-1).astype(float),
    )


def build_central_idpp_path(
    input_dir: Path,
    n_internal: int = N_INTERNAL,
    idpp_fmax: float = 0.1,
    idpp_steps: int = 100,
) -> np.ndarray:
    """Build one common IDPP band while retaining the central tangent."""
    from ase.mep.neb import idpp_interpolate

    template, start, end = load_endpoints(input_dir)
    images = []
    for fraction in np.linspace(0.0, 1.0, n_internal + 2):
        flat = (1.0 - fraction) * start + fraction * end
        images.append(atoms_from_flat(template, flat))
    neb = CentralDifferenceNEB(
        images,
        k=0.1,
        climb=False,
        parallel=False,
        remove_rotation_and_translation=False,
    )
    idpp_interpolate(
        neb,
        traj=None,
        log=None,
        fmax=idpp_fmax,
        steps=idpp_steps,
    )
    path = np.asarray(
        [image.get_positions().reshape(-1) for image in neb.images], dtype=float
    )
    path[0] = start
    path[-1] = end
    return path


def spring_reparameterize(
    path: np.ndarray,
    spring_constant: float = 1.0,
    dtau: float = 0.05,
    max_iter: int = 500,
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, float]:
    """Geometry-only spring relaxation using central-difference tangents."""
    result = np.asarray(path, dtype=float).copy()
    residual = np.inf
    for _ in range(max_iter):
        segments = result[1:] - result[:-1]
        lengths = np.linalg.norm(segments, axis=1)
        tangent = central_tangents(result)
        spring_force = (
            spring_constant * (lengths[1:] - lengths[:-1])
        )[:, None] * tangent
        residual = float(np.max(np.linalg.norm(spring_force, axis=1)))
        if residual <= tolerance:
            break
        result[1:-1] += dtau * spring_force
    return result, residual


def clip_max_atomic_displacement(direction: np.ndarray, max_step: float):
    direction = np.asarray(direction, dtype=float)
    largest = float(
        np.max(np.linalg.norm(direction.reshape(-1, 3), axis=1))
    )
    if largest > max_step:
        direction = direction * (max_step / largest)
    return direction


def densified_path(path: np.ndarray, subdivisions: int) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    points = []
    for i in range(path.shape[0] - 1):
        for j in range(subdivisions):
            fraction = j / subdivisions
            points.append((1.0 - fraction) * path[i] + fraction * path[i + 1])
    points.append(path[-1].copy())
    return np.asarray(points, dtype=float)


def densified_barrier(
    path: np.ndarray,
    diagnostic_evaluator: ImageEvaluator,
    subdivisions: int,
) -> tuple[float, np.ndarray]:
    dense = densified_path(path, subdivisions)
    energies = np.asarray(
        [diagnostic_evaluator.query(point, "energy") for point in dense],
        dtype=float,
    )
    barrier = float((np.max(energies) - energies[0]) * EV_TO_KCAL)
    return barrier, energies


def load_common_path(path_file: Path) -> np.ndarray:
    with np.load(path_file, allow_pickle=False) as saved:
        if "initial_path" not in saved.files:
            raise KeyError(f"{path_file} does not contain initial_path")
        path = np.asarray(saved["initial_path"], dtype=float)
    if path.ndim != 2 or path.shape[0] < 3 or path.shape[1] % 3:
        raise ValueError(f"Unexpected initial path shape: {path.shape}")
    return path


def attach_audited_calculators(
    path: np.ndarray,
    template: Atoms,
    evaluators: list[ImageEvaluator],
) -> list[Atoms]:
    images = []
    for position, evaluator in zip(path, evaluators):
        atoms = atoms_from_flat(template, position)
        atoms.calc = AuditedOpenMMCalculator(evaluator)
        images.append(atoms)
    return images
