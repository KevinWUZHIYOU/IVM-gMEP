"""Call-free, nonperiodic BB acceleration for the frozen FFrbk matrices.

This module changes only sparse-matrix construction.  It does not change the
FF matrix entries, rebuild schedule, shifted solve, tangent constraint, or any
IR-IVM acceptance/reparameterization decision.  In particular it contains no
OpenMM energy/Force query.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import multiprocessing as mp
from types import MethodType

import numpy as np


POLICY_ID = "bb-9396d-nonperiodic-call-free-ffrbk-builder-v1"
_PROCESS_BUILD_CONTEXT = None


def _direct_rel_pos(atoms, i, j):
    positions = atoms.get_positions()
    return positions[i, :] - positions[j, :]


def _initialize_worker(template, c_stab, bonds, angles, dihedrals):
    from ase.utils import ff as ase_ff

    global _PROCESS_BUILD_CONTEXT
    ase_ff.rel_pos_pbc = _direct_rel_pos
    _PROCESS_BUILD_CONTEXT = (
        template,
        float(c_stab),
        bonds,
        angles,
        dihedrals,
    )


def _build_in_worker(position):
    from ase.optimize.precon import FF

    if _PROCESS_BUILD_CONTEXT is None:
        raise RuntimeError("FF worker was not initialized")
    template, c_stab, bonds, angles, dihedrals = _PROCESS_BUILD_CONTEXT
    atoms = template.copy()
    atoms.set_positions(np.asarray(position, dtype=float).reshape(-1, 3))
    operator = FF(
        c_stab=c_stab,
        apply_cell=False,
        hessian="reduced",
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
    )
    operator.make_precon(atoms)
    return operator


def _conservative_bonded_pairs(bank):
    pairs = set()
    for collection, names in (
        (bank.bonds, ("atomi", "atomj")),
        (bank.angles, ("atomi", "atomj", "atomk")),
        (bank.dihedrals, ("atomi", "atomj", "atomk", "atoml")),
    ):
        for term in collection:
            atoms = tuple(int(getattr(term, name)) for name in names)
            pairs.update(tuple(sorted(pair)) for pair in combinations(atoms, 2))
    if not pairs:
        raise RuntimeError("no conservative bonded pairs were extracted")
    return np.asarray(sorted(pairs), dtype=int)


def _call_free_fast_build(self, path):
    from ase.optimize.precon import FF
    from ase.utils import ff as ase_ff

    path = np.asarray(path, dtype=float)
    expected = (32, 3 * len(self.template))
    if path.shape != expected:
        raise ValueError(f"canonical BB path must have shape {expected}, got {path.shape}")
    if np.any(self.template.get_pbc()):
        raise RuntimeError("BB fast FFrbk builder requires a nonperiodic topology")
    coordinates = path.reshape(path.shape[0], len(self.template), 3)
    pairs = _conservative_bonded_pairs(self)
    pair_differences = np.abs(
        coordinates[:, pairs[:, 0], :] - coordinates[:, pairs[:, 1], :]
    )
    max_pair_component = float(np.max(pair_differences))
    # The inert legacy cell is 100 A.  ASE's rel_pos_pbc is exactly direct
    # subtraction only when every relevant fractional component lies in
    # (-1/2, 1/2).  Check every conservative pair from every bonded term on
    # every rebuild, not merely the whole-protein span.
    if not np.isfinite(max_pair_component) or max_pair_component >= 50.0:
        raise RuntimeError(
            "direct FFrbk displacement is not equivalent to the 100 A "
            f"legacy cell: max bonded-pair component={max_pair_component} A"
        )
    self.bb_max_conservative_bonded_pair_component_A = max_pair_component

    def build_one(position):
        atoms = self.template.copy()
        atoms.set_positions(np.asarray(position, dtype=float).reshape(-1, 3))
        operator = FF(
            c_stab=self.c_stab,
            apply_cell=False,
            hessian="reduced",
            bonds=self.bonds,
            angles=self.angles,
            dihedrals=self.dihedrals,
        )
        operator.make_precon(atoms)
        return operator

    internal_positions = list(path[1:-1])
    original_rel_pos = ase_ff.rel_pos_pbc
    ase_ff.rel_pos_pbc = _direct_rel_pos
    try:
        if self.bb_build_workers == 1:
            operators = [build_one(position) for position in internal_positions]
        else:
            with ProcessPoolExecutor(
                max_workers=min(self.bb_build_workers, len(internal_positions)),
                mp_context=mp.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(
                    self.template,
                    self.c_stab,
                    self.bonds,
                    self.angles,
                    self.dihedrals,
                ),
            ) as executor:
                operators = list(executor.map(_build_in_worker, internal_positions))
    finally:
        ase_ff.rel_pos_pbc = original_rel_pos
    if len(operators) != 30:
        raise RuntimeError("FFrbk builder did not return 30 ordered operators")
    self.preconditioners = operators
    self._shifted_solvers = {}
    self.reference_path = path.copy()
    self.rebuilds += 1


def install_call_free_fast_builder(bank, *, build_workers=1):
    """Install the explicit BB builder on one otherwise unchanged Chig bank."""
    workers = int(build_workers)
    if workers < 1:
        raise ValueError("build_workers must be at least one")
    if 3 * len(bank.template) != 9396:
        raise ValueError("BB FFrbk template must contain exactly 3132 atoms")
    if bank.template.cell.rank < 3:
        bank.template.set_cell([100.0, 100.0, 100.0])
    bank.template.set_pbc(False)
    bank.bb_build_workers = workers
    bank.bb_builder_policy_id = POLICY_ID
    bank.build = MethodType(_call_free_fast_build, bank)
    return bank
