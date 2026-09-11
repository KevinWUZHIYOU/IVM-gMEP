#!/usr/bin/env python3
"""Evaluate the immutable current BB IVM baseline under the arc500 policy.

The discrete, unnormalized normal residual is reconstructed from the forces
already stored in each official milestone snapshot.  Only the densified
forward barrier makes fresh calls: exactly 500 energy-only evaluations on
equal-Cartesian-arclength points per frozen 32-image chain.  These diagnostic
calls contribute zero to optimization ``N_F``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from bb_core import path_diagnostics
from evaluate_discrete_iperp_densified_barrier import (
    ALPHA_SCOPE_R2_SHA256,
    COORDINATE_SPACE,
    GRID_FORMULA,
    INTERPOLATION,
    METRIC_AMENDMENT_SHA256,
    N_DENSE_POINTS,
    N_IMAGES,
    SAMPLING,
    acquire_host_locks,
    atomic_save_npz,
    atomic_write_json,
    evaluate_paths,
    resolve_path,
    sha256_array,
    sha256_file,
)


CONTRACT_SCHEMA = "bb-original-ivm-arc500-contract-v1"
MILESTONES = (30, 240, 480, 990, 9990)


def _require_hash(path: Path, expected: str, role: str) -> None:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{role} SHA-256 mismatch: {actual} != {expected}")


def _validate_contract(contract_path: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unsupported original-IVM arc500 contract")
    protocol = contract.get("diagnostic_protocol", {})
    expected = {
        "coordinate_space": COORDINATE_SPACE,
        "interpolation": INTERPOLATION,
        "sampling": SAMPLING,
        "number_of_points": N_DENSE_POINTS,
        "include_both_endpoints": True,
        "grid_formula": GRID_FORMULA,
        "force_evaluations_per_path": 0,
        "optimization_N_F_contribution": 0,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"diagnostic protocol mismatch for {key}")
    for role, record in contract["bindings"].items():
        path = resolve_path(contract_path, str(record["path"]))
        _require_hash(path, str(record["sha256"]), role)
    if contract["bindings"]["metric_policy_amendment"]["sha256"] != METRIC_AMENDMENT_SHA256:
        raise ValueError("metric-policy amendment binding mismatch")
    if contract["bindings"]["alpha_scope_R2"]["sha256"] != ALPHA_SCOPE_R2_SHA256:
        raise ValueError("alpha-scope R2 binding mismatch")
    if tuple(int(row["optimization_N_F"]) for row in contract["milestones"]) != MILESTONES:
        raise ValueError("original-IVM milestones are incomplete or unordered")
    return contract


def _load_rows(contract_path: Path, contract: dict) -> list[dict]:
    rows = []
    for record in contract["milestones"]:
        nf = int(record["optimization_N_F"])
        snapshot = resolve_path(contract_path, str(record["snapshot"]["path"]))
        release = resolve_path(contract_path, str(record["release"]["path"]))
        _require_hash(snapshot, str(record["snapshot"]["sha256"]), f"NF={nf} snapshot")
        _require_hash(release, str(record["release"]["sha256"]), f"NF={nf} release")
        release_data = json.loads(release.read_text(encoding="utf-8"))
        if (
            release_data.get("schema") != "bb-discrete-ivm-milestone-release-v1"
            or int(release_data["nf"]) != nf
            or str(release_data["method"]) != "ivm"
            or str(release_data["snapshot"]["sha256"]) != sha256_file(snapshot)
        ):
            raise ValueError(f"NF={nf} milestone release does not bind its snapshot")
        with np.load(snapshot, allow_pickle=False) as saved:
            path = np.asarray(saved["nodes"], dtype=np.float64)
            forces = np.asarray(saved["current_forces_eV_per_A"], dtype=np.float64)
            saved_nf = int(np.asarray(saved["nf"]).reshape(()))
            evaluated = bool(np.asarray(saved["evaluated"]).reshape(()))
            pending_ignored = bool(np.asarray(saved["dense_loader_must_ignore_pending_trial"]).reshape(()))
            method = str(np.asarray(saved["method"]).reshape(()))
            state_hash = str(np.asarray(saved["state_sha256"]).reshape(()))
            alpha = float(np.asarray(saved["alpha_A2_per_eV"]).reshape(()))
        if path.shape != (N_IMAGES, 9396) or forces.shape != (30, 9396):
            raise ValueError(f"NF={nf} snapshot has the wrong path/force shape")
        if saved_nf != nf or not evaluated or not pending_ignored or method != "ivm":
            raise ValueError(f"NF={nf} snapshot state is not an eligible evaluated image chain")
        if state_hash != str(release_data["snapshot"]["state_sha256"]):
            raise ValueError(f"NF={nf} snapshot state identity mismatch")
        if alpha != 0.02:
            raise ValueError(f"NF={nf} is not the immutable alpha=0.02 baseline")
        diagnostic = path_diagnostics(
            path,
            np.zeros(N_IMAGES, dtype=np.float64),
            forces,
        )
        rows.append(
            {
                "label": f"original_IVM_NF{nf}",
                "optimization_N_F": nf,
                "discrete_I_perp": float(diagnostic["i_perp"]),
                "path": np.ascontiguousarray(path),
                "path_sha256": sha256_array(path),
                "official_state_sha256": state_hash,
                "source_result": str(snapshot),
                "source_result_sha256": sha256_file(snapshot),
                "source_audit": str(release),
                "source_audit_sha256": sha256_file(release),
                "history_index": -1,
            }
        )
    return rows


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--global-lock-path", type=Path, default=Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock"))
    parser.add_argument("--peer-global-lock-path", type=Path, default=Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    contract_path = args.contract.expanduser().resolve()
    output = args.output.expanduser().resolve()
    audit_output = output.with_suffix(".audit.json")
    wal_output = output.with_suffix(".energy.wal.jsonl")
    if output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    contract = _validate_contract(contract_path)
    rows = _load_rows(contract_path, contract)
    if args.validate_only:
        print(json.dumps({"status": "PASS", "physical_calls": 0, "rows": [{"optimization_N_F": row["optimization_N_F"], "discrete_I_perp": row["discrete_I_perp"], "official_state_sha256": row["official_state_sha256"]} for row in rows]}, sort_keys=True))
        return
    if output.exists() or audit_output.exists() or wal_output.exists():
        raise FileExistsError("refusing to overwrite an original-IVM diagnostic artifact")
    topology_record = contract["topology"]
    topology = resolve_path(contract_path, str(topology_record["path"]))
    _require_hash(topology, str(topology_record["sha256"]), "topology")
    with acquire_host_locks((args.global_lock_path, args.peer_global_lock_path)) as held_locks:
        evaluated = evaluate_paths(topology, rows, wal_output)
    atomic_save_npz(
        output,
        label=np.asarray([row["label"] for row in evaluated]),
        optimization_N_F=np.asarray([row["optimization_N_F"] for row in evaluated], dtype=np.int64),
        discrete_I_perp=np.asarray([row["discrete_I_perp"] for row in evaluated], dtype=np.float64),
        densified_Delta_E_kcal_per_mol=np.asarray([row["densified_Delta_E_kcal_per_mol"] for row in evaluated], dtype=np.float64),
        energy_eV=np.asarray([row["energy_eV"] for row in evaluated]),
        selected_path_sha256=np.asarray([row["path_sha256"] for row in evaluated]),
        official_state_sha256=np.asarray([row["official_state_sha256"] for row in evaluated]),
        dense_coordinate_sha256=np.asarray([row["dense_coordinate_sha256"] for row in evaluated]),
        energy_sha256=np.asarray([row["energy_sha256"] for row in evaluated]),
        total_arclength_A=np.asarray([row["total_arclength_A"] for row in evaluated]),
        dense_spacing_A=np.asarray([row["dense_spacing_A"] for row in evaluated]),
        diagnostic_energy_attempts_per_path=np.full(len(evaluated), N_DENSE_POINTS, dtype=np.int64),
        diagnostic_force_requests_per_path=np.zeros(len(evaluated), dtype=np.int64),
        optimization_N_F_contribution=np.asarray(0, dtype=np.int64),
        coordinate_space=np.asarray(COORDINATE_SPACE),
        interpolation=np.asarray(INTERPOLATION),
        sampling=np.asarray(SAMPLING),
        number_of_points=np.asarray(N_DENSE_POINTS, dtype=np.int64),
        grid_formula=np.asarray(GRID_FORMULA),
        contract=np.asarray(str(contract_path)),
        contract_sha256=np.asarray(sha256_file(contract_path)),
        topology=np.asarray(str(topology)),
        topology_sha256=np.asarray(sha256_file(topology)),
        energy_wal=np.asarray(str(wal_output)),
        held_host_locks=np.asarray(held_locks),
    )
    audit = {
        "schema": "bb-original-ivm-arc500-audit-v1",
        "status": "PASS",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "energy_wal": str(wal_output),
        "energy_wal_sha256": sha256_file(wal_output),
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "path_count": len(evaluated),
        "diagnostic_energy_attempts": N_DENSE_POINTS * len(evaluated),
        "diagnostic_energy_successes": N_DENSE_POINTS * len(evaluated),
        "diagnostic_force_requests": 0,
        "optimization_N_F_contribution": 0,
        "held_host_locks": held_locks,
        "records": [{"optimization_N_F": row["optimization_N_F"], "discrete_I_perp": row["discrete_I_perp"], "densified_Delta_E_kcal_per_mol": row["densified_Delta_E_kcal_per_mol"], "official_state_sha256": row["official_state_sha256"]} for row in evaluated],
    }
    atomic_write_json(audit_output, audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
