#!/usr/bin/env python3
"""Freeze the independent late-four ODE--NEB joint-E/F contract.

This command performs only sealed-file and NumPy validation.  It never
constructs an OpenMM backend or acquires a heavy-task lock.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
EVALUATOR_PATH = HERE / "evaluate_bb_ode_neb_late_four_joint_ef.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "bb_ode_neb_late_four_evaluator_for_freeze", EVALUATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load()


def write_exclusive(path: Path, payload: dict) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, destination)
        os.chmod(destination, 0o444)
    finally:
        temporary.unlink()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--topology", type=Path, default=EVALUATOR.ROOT / "inputs/complex_fixed.pdb"
    )
    parser.add_argument("--global-lock-path", type=Path, required=True)
    parser.add_argument("--peer-global-lock-path", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = EVALUATOR.build_contract(
        [args.global_lock_path, args.peer_global_lock_path],
        args.topology.expanduser().resolve(strict=True),
    )
    write_exclusive(args.output, payload)
    destination = args.output.expanduser().resolve(strict=True)
    print(
        json.dumps(
            {
                "status": "FROZEN_ZERO_PHYSICAL_CALLS",
                "contract": str(destination),
                "sha256": EVALUATOR.ENGINE.sha256_file(destination),
                "optimization_N_F": list(EVALUATOR.NF_VALUES),
                "new_discrete_calls": 0,
                "planned_dense_joint_EF_calls": EVALUATOR.MAXIMUM_ATTEMPT_STARTED,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
