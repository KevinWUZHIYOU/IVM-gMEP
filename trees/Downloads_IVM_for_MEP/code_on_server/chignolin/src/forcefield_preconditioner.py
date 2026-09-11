"""Audited bonded force-field Hessian model shared by mechanism ablations.

Construction uses only force-field parameters already present in the OpenMM
System and never queries OpenMM energy or force.  This is the same numerical
initial inverse-Hessian model used by the selected IVM implementation.
"""

from __future__ import annotations

import numpy as np


class ForceFieldPreconditionerBank:
    KJ_MOL_PER_EV = 96.4853321233

    def __init__(self, backend, topology_pdb, c_stab: float = 1.0):
        from ase.io import read
        from ase.utils.ff import Angle, Bond, Dihedral
        from openmm.unit import kilojoule_per_mole, nanometer, radian

        self.template = read(str(topology_pdb))
        if 3 * len(self.template) != backend.dimension:
            raise ValueError("ASE/OpenMM atom counts differ")
        self.c_stab = float(c_stab)
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        for force in backend.system.getForces():
            force_name = type(force).__name__
            if force_name == "HarmonicBondForce":
                for index in range(force.getNumBonds()):
                    atom_i, atom_j, length, stiffness = (
                        force.getBondParameters(index)
                    )
                    self.bonds.append(
                        Bond(
                            int(atom_i),
                            int(atom_j),
                            stiffness.value_in_unit(
                                kilojoule_per_mole / nanometer**2
                            )
                            / self.KJ_MOL_PER_EV
                            / 100.0,
                            length.value_in_unit(nanometer) * 10.0,
                        )
                    )
            elif force_name == "HarmonicAngleForce":
                for index in range(force.getNumAngles()):
                    atom_i, atom_j, atom_k, angle, stiffness = (
                        force.getAngleParameters(index)
                    )
                    self.angles.append(
                        Angle(
                            int(atom_i),
                            int(atom_j),
                            int(atom_k),
                            stiffness.value_in_unit(
                                kilojoule_per_mole / radian**2
                            )
                            / self.KJ_MOL_PER_EV,
                            angle.value_in_unit(radian),
                        )
                    )
            elif force_name == "PeriodicTorsionForce":
                for index in range(force.getNumTorsions()):
                    (
                        atom_i,
                        atom_j,
                        atom_k,
                        atom_l,
                        periodicity,
                        phase,
                        stiffness,
                    ) = force.getTorsionParameters(index)
                    self.dihedrals.append(
                        Dihedral(
                            int(atom_i),
                            int(atom_j),
                            int(atom_k),
                            int(atom_l),
                            stiffness.value_in_unit(kilojoule_per_mole)
                            / self.KJ_MOL_PER_EV,
                            phase.value_in_unit(radian),
                            int(periodicity),
                        )
                    )
        if not self.bonds or not self.angles or not self.dihedrals:
            raise RuntimeError("Required bonded force-field terms are missing")
        self.preconditioners = None
        self.rebuilds = 0

    def build(self, path: np.ndarray) -> None:
        from ase.optimize.precon import FF

        self.preconditioners = []
        for position in np.asarray(path, dtype=float)[1:-1]:
            atoms = self.template.copy()
            atoms.set_positions(position.reshape(-1, 3))
            if abs(float(np.linalg.det(np.asarray(atoms.cell)))) < 1e-12:
                atoms.set_cell(np.eye(3) * 100.0)
                atoms.set_pbc(False)
            preconditioner = FF(
                c_stab=self.c_stab,
                apply_cell=False,
                hessian="reduced",
                bonds=self.bonds,
                angles=self.angles,
                dihedrals=self.dihedrals,
            )
            preconditioner.make_precon(atoms)
            self.preconditioners.append(preconditioner)
        self.rebuilds += 1

    def dense_hessian(self, internal_index: int) -> np.ndarray:
        if self.preconditioners is None:
            raise RuntimeError("Preconditioner bank has not been built")
        matrix = np.asarray(
            self.preconditioners[int(internal_index)].P.toarray(),
            dtype=float,
        )
        return 0.5 * (matrix + matrix.T)
