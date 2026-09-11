#!/usr/bin/env python3
"""Precompute the call-free bonded OBSM-BFGS H0 bank for one initial path."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ala_core import (
    OpenMMBackend,
    TOPOLOGY_FILENAME,
    array_sha256,
    atomic_save_npz,
    load_common_path,
)
from forcefield_preconditioner import ForceFieldPreconditionerBank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--initial-path-file",
        default="inputs/common_initial_path_equidistant.npz",
    )
    parser.add_argument(
        "--output",
        default="inputs/obsm_bfgs_common_initial_bonded_h0.npz",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    initial_file = Path(args.initial_path_file).expanduser()
    if not initial_file.is_absolute():
        initial_file = root / initial_file
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = root / output
    initial_path = load_common_path(initial_file)
    backend = OpenMMBackend(
        root / "inputs" / TOPOLOGY_FILENAME,
        platform_preference=("CPU",),
        precision="double",
        label="obsm_bonded_h0_precompute",
    )
    bank = ForceFieldPreconditionerBank(
        backend, root / "inputs" / TOPOLOGY_FILENAME, c_stab=1.0
    )
    before = int(backend.raw_joint_evaluations)
    bank.build(initial_path)
    matrices = np.stack(
        [bank.dense_hessian(i) for i in range(initial_path.shape[0] - 2)]
    )
    after = int(backend.raw_joint_evaluations)
    if after != before:
        raise RuntimeError("Bonded H0 construction unexpectedly called OpenMM")
    atomic_save_npz(
        output,
        dense_hessians=matrices,
        initial_path_sha256=np.array(array_sha256(initial_path)),
        source_initial_path=np.array(str(initial_file.resolve())),
        c_stab=np.array(1.0),
        openmm_joint_evaluations=np.array(0, dtype=int),
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
