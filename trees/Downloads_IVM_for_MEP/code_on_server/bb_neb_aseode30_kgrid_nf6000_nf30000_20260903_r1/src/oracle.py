"""Uncached OpenMM Amber14/GBn2 joint energy--Force oracle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any

import numpy as np


KJ_PER_MOL_PER_EV = 96.48533212331002
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENMM_CPU_THREADS",
)


class OracleConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class OracleResult:
    energy_eV: float
    force_eV_per_A: np.ndarray

    def __iter__(self):
        # The isolated dense diagnostic accepts a two-item result and performs
        # exactly one call before parsing it.
        yield self.energy_eV
        yield self.force_eV_per_A


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require_single_thread_environment() -> dict[str, str]:
    """Refuse to create a Context unless every frozen thread cap is already 1."""

    values = {name: os.environ.get(name, "") for name in THREAD_ENVIRONMENT}
    invalid = {name: value for name, value in values.items() if value != "1"}
    if invalid:
        rendered = ", ".join(f"{name}={value!r}" for name, value in invalid.items())
        raise OracleConfigurationError(
            "all BLAS/OMP/VecLib/NumExpr/OpenMM thread caps must be fixed to 1 "
            f"before Python starts; invalid: {rendered}"
        )
    return values


class OpenMMJointOracle:
    """One ``getState`` joint energy/full-Force calculation per call.

    No coordinate/result cache exists in this object.  ``attempt_count`` is
    incremented immediately before the sole physical ``getState`` request, so
    a raised OpenMM exception is still visible to the paid ledger as an attempt.
    """

    forcefield_files = ("amber14-all.xml", "implicit/gbn2.xml")
    nonbonded_method = "NoCutoff"
    platform_name = "CPU"
    precision = "double"
    expected_openmm_version = "8.1.2"
    expected_system_xml_sha256 = (
        "ef5d6907fe238ed54be5dcd11d900a7488fe0949a479e544e645c9d2711a9969"
    )
    expected_system_xml_bytes = 3918711
    expected_force_class_sequence = (
        "HarmonicBondForce",
        "PeriodicTorsionForce",
        "NonbondedForce",
        "CMMotionRemover",
        "HarmonicAngleForce",
        "CustomGBForce",
    )

    def __init__(
        self,
        pdb_path: Path | str,
        *,
        expected_pdb_sha256: str = "26b58b4301c1e293c7fcd01878c3a208451621099419cbf5b41bd80eb96a0f49",
        expected_atom_count: int | None = 3132,
    ) -> None:
        self.thread_environment = require_single_thread_environment()
        self.pdb_path = Path(pdb_path).resolve(strict=True)
        if not re.fullmatch(r"[0-9a-f]{64}", expected_pdb_sha256):
            raise OracleConfigurationError("expected PDB hash is not lowercase SHA-256")
        actual_hash = file_sha256(self.pdb_path)
        if actual_hash != expected_pdb_sha256:
            raise OracleConfigurationError(
                f"neutral PDB SHA-256 mismatch: expected {expected_pdb_sha256}, got {actual_hash}"
            )
        self.pdb_sha256 = actual_hash

        try:
            import openmm
            from openmm import app, unit
        except ImportError as error:
            raise OracleConfigurationError("OpenMM is required for the physical oracle") from error

        self._openmm = openmm
        self._app = app
        self._unit = unit
        pdb = app.PDBFile(str(self.pdb_path))
        self.atom_count = sum(1 for _ in pdb.topology.atoms())
        if expected_atom_count is not None and self.atom_count != int(expected_atom_count):
            raise OracleConfigurationError(
                f"PDB atom count mismatch: expected {expected_atom_count}, got {self.atom_count}"
            )

        forcefield = app.ForceField(*self.forcefield_files)
        self._system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
        )
        # This is a zero-Force, cross-machine identity lock.  It happens
        # before the Context and before any paid reservation.  The serialized
        # System includes every force parameter used by later optimization
        # and by the call-free FF preconditioner construction.
        openmm_version = str(getattr(openmm, "__version__", "unknown"))
        system_xml = str(openmm.XmlSerializer.serialize(self._system))
        system_xml_bytes = system_xml.encode("utf-8")
        system_xml_sha256 = hashlib.sha256(system_xml_bytes).hexdigest()
        force_class_sequence = tuple(
            type(force).__name__ for force in self._system.getForces()
        )
        mismatches = []
        if openmm_version != self.expected_openmm_version:
            mismatches.append(
                f"OpenMM version {openmm_version!r} != {self.expected_openmm_version!r}"
            )
        if len(system_xml_bytes) != self.expected_system_xml_bytes:
            mismatches.append(
                f"System XML bytes {len(system_xml_bytes)} != {self.expected_system_xml_bytes}"
            )
        if system_xml_sha256 != self.expected_system_xml_sha256:
            mismatches.append(
                "System XML SHA-256 "
                f"{system_xml_sha256} != {self.expected_system_xml_sha256}"
            )
        if force_class_sequence != self.expected_force_class_sequence:
            mismatches.append(
                "force class sequence "
                f"{force_class_sequence!r} != {self.expected_force_class_sequence!r}"
            )
        if mismatches:
            raise OracleConfigurationError(
                "zero-Force OpenMM System identity lock failed: " + "; ".join(mismatches)
            )
        self.openmm_version = openmm_version
        self.system_xml_bytes = len(system_xml_bytes)
        self.system_xml_sha256 = system_xml_sha256
        self.force_class_sequence = force_class_sequence
        # The Integrator is never advanced.  It is required only to construct
        # a Context for deterministic potential-energy and Force evaluation.
        self._integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName(self.platform_name)
        try:
            self._context = openmm.Context(
                self._system,
                self._integrator,
                platform,
                {"Threads": "1"},
            )
        except Exception as error:
            raise OracleConfigurationError(
                "failed to construct the CPU OpenMM Context with Threads=1"
            ) from error
        if self._context.getPlatform().getName() != self.platform_name:
            raise OracleConfigurationError("OpenMM did not honor the frozen CPU platform")

        self.attempt_count = 0
        self.success_count = 0

    def evaluate(self, coordinates_A: Any) -> OracleResult:
        # A daemon monitor samples pressure independently of batch boundaries.
        # It may latch a permanent fail-stop while a preceding call is in C++;
        # this check prevents the next physical request from starting.
        from .resource_guard import assert_active_guard_safe

        assert_active_guard_safe()
        coordinate = np.asarray(coordinates_A, dtype=np.float64)
        if coordinate.ndim == 1:
            if coordinate.size != 3 * self.atom_count:
                raise ValueError("flattened coordinate has the wrong atom count")
            coordinate = coordinate.reshape((self.atom_count, 3))
        if coordinate.shape != (self.atom_count, 3):
            raise ValueError(
                f"coordinate must have shape ({self.atom_count},3), got {coordinate.shape}"
            )
        if not np.isfinite(coordinate).all():
            raise ValueError("coordinate contains a non-finite value")

        # setPositions itself performs no Force evaluation.  Exactly one joint
        # getState call follows it, with no groups omitted.
        self._context.setPositions(coordinate * self._unit.angstrom)
        self.attempt_count += 1
        state = self._context.getState(getEnergy=True, getForces=True)
        energy_kj_per_mol = state.getPotentialEnergy().value_in_unit(
            self._unit.kilojoule_per_mole
        )
        force_kj_per_mol_per_nm = np.asarray(
            state.getForces(asNumpy=True).value_in_unit(
                self._unit.kilojoule_per_mole / self._unit.nanometer
            ),
            dtype=np.float64,
        )
        energy_eV = float(energy_kj_per_mol) / KJ_PER_MOL_PER_EV
        force_eV_per_A = force_kj_per_mol_per_nm / (10.0 * KJ_PER_MOL_PER_EV)
        if not np.isfinite(energy_eV) or not np.isfinite(force_eV_per_A).all():
            raise FloatingPointError("OpenMM returned non-finite energy or Force")
        self.success_count += 1
        return OracleResult(
            energy_eV=energy_eV,
            force_eV_per_A=np.ascontiguousarray(force_eV_per_A, dtype="<f8"),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "OpenMM",
            "openmm_version": self.openmm_version,
            "forcefield_files": list(self.forcefield_files),
            "nonbonded_method": self.nonbonded_method,
            "platform": self._context.getPlatform().getName(),
            "precision": self.precision,
            "context_properties": {"Threads": "1"},
            "thread_environment": dict(self.thread_environment),
            "pdb_sha256": self.pdb_sha256,
            "atom_count": self.atom_count,
            "system_xml_sha256": self.system_xml_sha256,
            "system_xml_bytes": self.system_xml_bytes,
            "force_class_sequence": list(self.force_class_sequence),
            "system_identity_verified_before_context": True,
            "joint_energy_force": True,
            "cache": False,
        }

    def zero_force_system_for_preconditioner(self) -> Any:
        """Expose the verified immutable System without evaluating it."""

        if self.attempt_count != 0:
            raise OracleConfigurationError(
                "FF preconditioner must be built before the first physical Force attempt"
            )
        return self._system

    def close(self) -> None:
        # Explicitly release the Context before the Integrator.  OpenMM objects
        # do not expose a public close method.
        if hasattr(self, "_context"):
            del self._context
        if hasattr(self, "_integrator"):
            del self._integrator


# The production runner and isolated diagnostic factory intentionally receive
# the same physical implementation.
OpenMMOracle = OpenMMJointOracle


def create_oracle(
    pdb_path: Path | str,
    *,
    expected_pdb_sha256: str = "26b58b4301c1e293c7fcd01878c3a208451621099419cbf5b41bd80eb96a0f49",
    expected_atom_count: int | None = 3132,
) -> OpenMMJointOracle:
    return OpenMMJointOracle(
        pdb_path,
        expected_pdb_sha256=expected_pdb_sha256,
        expected_atom_count=expected_atom_count,
    )
