#!/usr/bin/env python3
"""Freeze the one allowed ODE--NEB four-path densification contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = ROOT / "scripts" / "evaluate_bb_dense_four_paths.py"
BB_CORE = ROOT / "src" / "bb_core.py"
DEFAULT_SELECTION = ROOT / "registrations" / "diagnostics" / "BB_ODE_NEB_TARGETED_ARC500_SELECTION_20260902_R2.json"
DEFAULT_TOPOLOGY = ROOT / "inputs" / "complex_fixed.pdb"
NF_VALUES = (60, 150, 300, 1320)
METRIC_POLICY = {
    "coordinate_space": "unweighted_3N_cartesian",
    "sampling": "uniform_cartesian_arclength",
    "interpolation": "piecewise_linear",
    "number_of_points": 500,
    "grid_formula": "s_j=j*L/499,j=0,...,499",
    "include_both_endpoints": True,
    "discrete_I_perp_implementation": "bb_core.path_diagnostics",
    "tangent": "normalized central difference: x[i+1] - x[i-1]",
    "weight": "half sum of adjacent Cartesian chord lengths",
    "fixed_endpoint_diagnostic_force_evaluations": 0,
    "optimization_N_F_contribution": 0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(path: Path):
    path = path.expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def raw_path_sha256(archive: Path) -> str:
    with np.load(archive, allow_pickle=False) as saved:
        path = np.concatenate(
            (saved["endpoint0_A"][None], saved["coordinates30_A"], saved["endpoint1_A"][None]),
            axis=0,
        ).reshape(32, -1)
    return hashlib.sha256(np.ascontiguousarray(path, dtype=np.float64).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--host-role", choices=("local", "remote"), default="local")
    parser.add_argument("--global-lock-path", type=Path, required=True)
    parser.add_argument("--peer-global-lock-path", type=Path, required=True)
    parser.add_argument("--reuse-result", type=Path, action="append", default=[])
    parser.add_argument("--reuse-audit", type=Path, action="append", default=[])
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    if len(args.reuse_result) != len(args.reuse_audit):
        raise ValueError("--reuse-result and --reuse-audit counts differ")
    selection_path = args.selection.expanduser().resolve(strict=True)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    official = {int(row["optimization_N_F"]): row for row in selection["official_paid_states"]}
    paths = []
    for nf in NF_VALUES:
        row = official.get(nf)
        if row is None:
            raise ValueError(f"targeted R2 selection lacks retained NF{nf}")
        archive = Path(row["coordinate_archive"]).expanduser().resolve(strict=True)
        if sha256(archive) != str(row["coordinate_archive_sha256"]):
            raise ValueError(f"targeted R2 NF{nf} coordinate archive hash mismatch")
        paths.append({
            "label": f"ODE-NEB_NF{nf}",
            "method": "ODE-NEB",
            "host_role": args.host_role,
            "optimization_N_F": nf,
            "path_identifier": {
                "optimization_N_F": nf,
                "paid_batch_id": int(row["paid_batch_id"]),
                "source_batch_id": int(row["source_batch_id"]),
                "disposition": str(row["disposition"]),
            },
            "source": bind(archive),
            "path_sha256": raw_path_sha256(archive),
            "official_path32_sha256": str(row["official_path32_sha256"]),
            "force_policy": "fresh_30_internal_joint_EF",
        })
    bundles = [
        {"result": bind(result), "audit": bind(audit)}
        for result, audit in zip(args.reuse_result, args.reuse_audit)
    ]
    locks = [args.global_lock_path.expanduser().resolve(), args.peer_global_lock_path.expanduser().resolve()]
    if len(set(locks)) != 2:
        raise ValueError("two distinct host-local lock paths are required")
    contract = {
        "schema": "bb-dense-four-path-dual-metric-contract-v1",
        "frozen": True,
        "purpose": "densify the manuscript ODE--NEB cost-to-precision trace at four preregistered retained paths",
        "jobs": 1,
        "evaluator": bind(EVALUATOR),
        "bb_core": bind(BB_CORE),
        "topology": bind(args.topology),
        "targeted_R2_selection": bind(selection_path),
        "host_local_lock_paths": [str(path) for path in locks],
        "metric_policy": METRIC_POLICY,
        "paths": paths,
        "allowed_pass_energy_reuse_bundles": bundles,
        "physical_call_policy": {
            "ODE_internal_joint_EF_per_path": 30,
            "IVM_diagnostic_force_per_path": 0,
            "arc500_energy_lineage_per_path": 500,
            "optimization_N_F_contribution": 0,
            "all_six_thread_caps": 1,
            "two_host_local_fcntl_locks": True,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    print(json.dumps({"status": "FROZEN", "contract": str(destination), "sha256": sha256(destination), "paths": [row["label"] for row in paths]}, indent=2))


if __name__ == "__main__":
    main()
