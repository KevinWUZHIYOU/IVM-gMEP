#!/usr/bin/env python3
"""Re-evaluate official BB ODE--NEB milestones under the user metric policy.

The discrete, unnormalized normal residual is reconstructed from the paid
30-image force receipts, so it makes no physical request.  The densified
forward barrier uses exactly 500 fresh energy-only evaluations placed at
equal Cartesian arclength on each frozen piecewise-linear 32-image chain.
Those diagnostic energy calls contribute zero to optimization ``N_F``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from bb_core import path_diagnostics
from evaluate_discrete_iperp_densified_barrier import (
    ALPHA_SCOPE_R2_SHA256,
    COORDINATE_SPACE,
    DIMENSION,
    EV_TO_KCAL_PER_MOL,
    GRID_FORMULA,
    INTERPOLATION,
    METRIC_AMENDMENT_SHA256,
    N_DENSE_POINTS,
    N_IMAGES,
    N_INTERNAL,
    SAMPLING,
    acquire_host_locks,
    atomic_save_npz,
    atomic_write_json,
    evaluate_paths,
    resolve_path,
    sha256_array,
    sha256_file,
)


CONTRACT_SCHEMA = "bb-ode-neb-arc500-preview-contract-v1"


def _official_array_sha256(value: np.ndarray) -> str:
    """Hash an array exactly as the official ODE--NEB paid ledger does."""

    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    if not np.all(np.isfinite(array)):
        raise ValueError("official array contains a non-finite value")
    metadata = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, role: str) -> None:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{role} SHA-256 mismatch: {actual} != {expected}")


def _validate_contract(contract_path: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unsupported ODE--NEB preview contract")
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
    rows = contract.get("milestones", [])
    if [int(row["optimization_N_F"]) for row in rows] != [240, 480, 990, 9990]:
        raise ValueError("ODE--NEB milestones must be exactly 240, 480, 990, 9990")
    return contract


def _load_rows(contract_path: Path, contract: dict) -> list[dict]:
    rows = []
    for record in contract["milestones"]:
        nf = int(record["optimization_N_F"])
        batch_path = resolve_path(contract_path, str(record["coordinate_archive"]["path"]))
        force_path = resolve_path(contract_path, str(record["paid_force_receipt"]["path"]))
        _require_hash(batch_path, str(record["coordinate_archive"]["sha256"]), f"NF={nf} coordinate archive")
        _require_hash(force_path, str(record["paid_force_receipt"]["sha256"]), f"NF={nf} force receipt")
        with np.load(batch_path, allow_pickle=False) as saved:
            coordinates = np.asarray(saved["coordinates30_A"], dtype=np.float64)
            endpoint0 = np.asarray(saved["endpoint0_A"], dtype=np.float64)
            endpoint1 = np.asarray(saved["endpoint1_A"], dtype=np.float64)
            embedded_hash = str(np.asarray(saved["path32_sha256"]).item())
        with np.load(force_path, allow_pickle=False) as saved:
            paid_coordinates = np.asarray(saved["coordinates30_A"], dtype=np.float64)
            forces = np.asarray(saved["forces_eV_per_A"], dtype=np.float64)
            receipt_nf = int(np.asarray(saved["nf_after"]).item())
        if coordinates.shape != (N_INTERNAL, DIMENSION // 3, 3):
            raise ValueError(f"NF={nf} coordinate archive has the wrong shape")
        if endpoint0.shape != (DIMENSION // 3, 3) or endpoint1.shape != endpoint0.shape:
            raise ValueError(f"NF={nf} endpoints have the wrong shape")
        if paid_coordinates.shape != coordinates.shape or not np.array_equal(paid_coordinates, coordinates):
            raise ValueError(f"NF={nf} force receipt is not for the frozen image chain")
        if forces.shape != coordinates.shape or receipt_nf != nf:
            raise ValueError(f"NF={nf} force receipt has inconsistent accounting")
        path3 = np.concatenate((endpoint0[None], coordinates, endpoint1[None]), axis=0)
        official_hash = _official_array_sha256(path3)
        if embedded_hash != str(record["official_path32_sha256"]) or official_hash != embedded_hash:
            raise ValueError(f"NF={nf} official path identity mismatch")
        path = np.ascontiguousarray(path3.reshape(N_IMAGES, DIMENSION))
        diagnostic = path_diagnostics(
            path,
            np.zeros(N_IMAGES, dtype=np.float64),
            forces.reshape(N_INTERNAL, DIMENSION),
        )
        rows.append(
            {
                "label": f"ODE-NEB_NF{nf}",
                "optimization_N_F": nf,
                "discrete_I_perp": float(diagnostic["i_perp"]),
                "path": path,
                "path_sha256": sha256_array(path),
                "official_path32_sha256": official_hash,
                "source_result": str(batch_path),
                "source_result_sha256": sha256_file(batch_path),
                "source_audit": str(force_path),
                "source_audit_sha256": sha256_file(force_path),
                "history_index": -1,
            }
        )
    return rows


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--global-lock-path",
        type=Path,
        default=Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock"),
    )
    parser.add_argument(
        "--peer-global-lock-path",
        type=Path,
        default=Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock"),
    )
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
        print(json.dumps({"status": "PASS", "physical_calls": 0, "rows": [{"optimization_N_F": row["optimization_N_F"], "discrete_I_perp": row["discrete_I_perp"], "official_path32_sha256": row["official_path32_sha256"]} for row in rows]}, sort_keys=True))
        return
    if output.exists() or audit_output.exists() or wal_output.exists():
        raise FileExistsError("refusing to overwrite an ODE--NEB diagnostic artifact")
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
        official_path32_sha256=np.asarray([row["official_path32_sha256"] for row in evaluated]),
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
        "schema": "bb-ode-neb-arc500-preview-audit-v1",
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
        "records": [{"optimization_N_F": row["optimization_N_F"], "discrete_I_perp": row["discrete_I_perp"], "densified_Delta_E_kcal_per_mol": row["densified_Delta_E_kcal_per_mol"], "official_path32_sha256": row["official_path32_sha256"]} for row in evaluated],
    }
    atomic_write_json(audit_output, audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
