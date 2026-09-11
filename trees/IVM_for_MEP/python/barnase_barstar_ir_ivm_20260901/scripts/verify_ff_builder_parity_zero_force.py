#!/usr/bin/env python3
"""Compare one canonical BB FFrbk matrix with/without the fast displacement.

The script constructs and verifies the OpenMM System but never calls
``getState``.  It is intentionally not run by the lightweight test suite.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_LOCK = "/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock"
DEFAULT_DOWNLOADS_LOCK = "/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock"
sys.path.insert(0, str(ROOT / "src"))

from ase.optimize.precon import FF  # noqa: E402
from ase.utils import ff as ase_ff  # noqa: E402
from bb_core import OpenMMBackend, load_common_path  # noqa: E402
from bb_ffprecon import _direct_rel_pos  # noqa: E402
from run_ivm_v9_spring_projected_production_20260730 import (  # noqa: E402
    ForceFieldPreconditionerBank,
)


@contextmanager
def acquire_all_host_locks(paths):
    """Acquire both BB checkout locks in fixed order, nonblocking."""
    ordered = sorted({Path(path).expanduser().resolve() for path in paths}, key=str)
    handles = []
    try:
        for path in ordered:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                raise RuntimeError(f"another BB task owns {path}") from error
            handles.append(handle)
        yield [str(path) for path in ordered]
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def build_one(bank, position, *, direct):
    atoms = bank.template.copy()
    if atoms.cell.rank < 3:
        atoms.set_cell([100.0, 100.0, 100.0])
    atoms.set_pbc(False)
    atoms.set_positions(position.reshape(-1, 3))
    original = ase_ff.rel_pos_pbc
    if direct:
        ase_ff.rel_pos_pbc = _direct_rel_pos
    try:
        operator = FF(
            c_stab=bank.c_stab,
            apply_cell=False,
            hessian="reduced",
            bonds=bank.bonds,
            angles=bank.angles,
            dihedrals=bank.dihedrals,
        )
        operator.make_precon(atoms)
    finally:
        ase_ff.rel_pos_pbc = original
    return operator.P.tocsr()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--internal-image", type=int, default=1)
    parser.add_argument("--global-lock-path", default=DEFAULT_CANONICAL_LOCK)
    parser.add_argument("--peer-global-lock-path", default=DEFAULT_DOWNLOADS_LOCK)
    args = parser.parse_args()
    root = Path(args.root).resolve(strict=True)
    if not 1 <= args.internal_image <= 30:
        raise ValueError("internal image must be in 1..30")
    with acquire_all_host_locks(
        (args.global_lock_path, args.peer_global_lock_path)
    ) as held_locks:
        backend = OpenMMBackend(root / "inputs" / "complex_fixed.pdb")
        path = load_common_path(root / "inputs" / "common_initial_path32.npz")
        bank = ForceFieldPreconditionerBank(
            backend, root / "inputs" / "complex_fixed.pdb", c_stab=1.0
        )
        before = int(backend.raw_joint_attempts)
        reference = build_one(bank, path[args.internal_image], direct=False)
        accelerated = build_one(bank, path[args.internal_image], direct=True)
        delta = (reference - accelerated).tocoo()
        after = int(backend.raw_joint_attempts)
        payload = {
            "status": "PASS" if delta.nnz == 0 and before == after == 0 else "FAIL",
            "held_locks": held_locks,
            "image": args.internal_image,
            "shape": list(reference.shape),
            "reference_nnz": int(reference.nnz),
            "accelerated_nnz": int(accelerated.nnz),
            "difference_nnz": int(delta.nnz),
            "max_abs_difference": float(np.max(np.abs(delta.data))) if delta.nnz else 0.0,
            "backend_raw_joint_attempts": after,
            "backend_raw_joint_evaluations": int(backend.raw_joint_evaluations),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
