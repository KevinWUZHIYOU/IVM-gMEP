#!/usr/bin/env python3
"""Freeze the densified BB figure with the joint-E/F v2 loader bound."""

from __future__ import annotations

import importlib.util
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "freeze_bb_densified_cost_figure_contract.py"
SPEC = importlib.util.spec_from_file_location("bb_densified_figure_freezer_v1", SOURCE)
FREEZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FREEZER)


def main() -> None:
    adapter = HERE / "plot_bb_cost_to_precision_densified_joint_v2.py"
    FREEZER.PLOTTER = adapter
    FREEZER.DIRECT_PLOT_DEPENDENCIES = (
        HERE / "plot_bb_cost_to_precision_densified.py",
        *FREEZER.DIRECT_PLOT_DEPENDENCIES,
        HERE / "evaluate_bb_dense_four_paths_joint_v2.py",
        HERE / "evaluate_bb_dense_four_ivm_joint_v2.py",
    )
    argv = list(sys.argv)
    try:
        output_index = argv.index("--output") + 1
    except (ValueError, IndexError) as error:
        raise ValueError("--output is required") from error
    destination = Path(argv[output_index]).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    stage = destination.with_name(destination.name + f".stage.{os.getpid()}")
    argv[output_index] = str(stage)
    original_argv = sys.argv
    try:
        sys.argv = argv
        with redirect_stdout(io.StringIO()):
            FREEZER.main()
    finally:
        sys.argv = original_argv
    payload = json.loads(stage.read_text(encoding="utf-8"))

    def argument(name: str) -> Path:
        index = original_argv.index(name) + 1
        return Path(original_argv[index]).expanduser().resolve(strict=True)

    def bind(path: Path):
        return FREEZER.bind(path)

    closure = {}
    for method, prefix in (("IVM", "--ivm"), ("ODE-NEB", "--ode")):
        result = argument(prefix + "-dense-result")
        audit_path = argument(prefix + "-dense-audit")
        contract_path = argument(prefix + "-dense-contract")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        manifest_path = Path(audit["shard_manifest"]).expanduser().resolve(strict=True)
        with np.load(result, allow_pickle=False) as saved:
            shard_paths = [Path(value).expanduser().resolve(strict=True) for value in np.asarray(saved["shard_path"]).astype(str)]
        wal_keys = ("online_paid_state_wal", "dense_joint_ef_wal") if method == "IVM" else ("discrete_joint_ef_wal", "dense_joint_ef_wal")
        closure[method] = {
            "result": bind(result), "audit": bind(audit_path), "contract": bind(contract_path),
            "manifest": bind(manifest_path), "authoritative_shards": [bind(path) for path in shard_paths],
            "WALs": [bind(Path(audit[key]).expanduser().resolve(strict=True)) for key in wal_keys],
        }
    payload["bindings"]["joint_EF_v2_direct_data_closure"] = closure
    payload["metric_protocol"]["arc500_stored_arrays_per_path"] = ["dense_coordinates_A", "dense_energy_eV", "dense_force_eV_per_A"]
    payload["metric_protocol"]["dense_physical_call"] = "one joint energy/full-force getState per point"
    payload["metric_protocol"]["authoritative_data_unit"] = "complete atomic per-path shard"
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if destination.exists():
        temporary.unlink()
        stage.unlink()
        raise FileExistsError(f"refusing to overwrite {destination}")
    os.rename(temporary, destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    stage.unlink()
    print(json.dumps({"status": payload["status"], "contract": str(destination), "sha256": FREEZER.sha256(destination), "joint_EF_v2_direct_data_closure": True}, indent=2))


if __name__ == "__main__":
    main()
