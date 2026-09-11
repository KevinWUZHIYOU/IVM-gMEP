#!/usr/bin/env python3
"""Evaluate only the frozen ODE--NEB path at optimization ``N_F=4080``.

The discrete normal residual is reconstructed from the already-paid force
receipt.  The densified forward barrier uses exactly 500 energy-only
evaluations, uniformly spaced in Cartesian arclength on the frozen
piecewise-linear 32-image chain.  Diagnostic energy calls request no force
and therefore contribute zero to the optimization Number of Forces.

``--validate-only`` performs a zero-OpenMM preflight: it verifies every bound
file, the ODE--NEB state identity, fixed endpoints, the paid-force coordinate
identity, the discrete residual, and the deterministic dense grid without
constructing an OpenMM backend or acquiring the physical-evaluation locks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from bb_core import path_diagnostics
from evaluate_discrete_iperp_densified_barrier import (
    ALPHA_SCOPE_R2_SHA256,
    COORDINATE_SPACE,
    DIMENSION,
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
    uniform_arclength_resample,
)


CONTRACT_SCHEMA = "bb-ode-neb-nf4080-arc500-contract-v1"
AUDIT_SCHEMA = "bb-ode-neb-nf4080-arc500-audit-v1"
EXPECTED_N_F = 4080
EXPECTED_BATCH_ID = 136
EXPECTED_DISCRETE_I_PERP = 4371.16533670238
EXPECTED_OFFICIAL_PATH32_SHA256 = (
    "eb21e326ccbefe8919c022bfc842ae0b6518de0136f4ef2336bb57a8af9a2461"
)
EXPECTED_RAW_PATH_SHA256 = (
    "b28a7222e61e457eae7e3b3dcfd90f83fec6b78728d64497914e1f607f4343d7"
)
EXPECTED_DENSE_COORDINATE_SHA256 = (
    "d84ddb9b2d0631a3c55e020fffb81e35faec3ddef37c40d17e4b52feb81ff874"
)


def _official_array_sha256(value: np.ndarray) -> str:
    """Use the metadata-qualified identity of the official ODE--NEB ledger."""

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


def _scalar(saved: np.lib.npyio.NpzFile, key: str):
    return np.asarray(saved[key]).item()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_hash(path: Path, expected: str, role: str) -> None:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{role} SHA-256 mismatch: {actual} != {expected}")


def _load_contract(contract_path: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(contract.get("schema") == CONTRACT_SCHEMA, "unsupported contract schema")
    _require(
        contract.get("status")
        == "FROZEN_BEFORE_ANY_NF4080_ARC500_PHYSICAL_CALL",
        "contract was not frozen prospectively",
    )
    _require(
        contract.get("scope") == "ODE-NEB_NF4080_SINGLE_RETAINED_PATH_ONLY",
        "contract scope is not the single N_F=4080 path",
    )
    _require(
        contract.get("all_path_ode_neb_arc500_task") == "REMAINS_PAUSED",
        "contract does not preserve the all-path pause",
    )

    protocol = contract.get("diagnostic_protocol", {})
    expected_protocol = {
        "coordinate_space": COORDINATE_SPACE,
        "interpolation": INTERPOLATION,
        "sampling": SAMPLING,
        "number_of_points": N_DENSE_POINTS,
        "include_both_endpoints": True,
        "grid_formula": GRID_FORMULA,
        "energy_evaluations_per_path": N_DENSE_POINTS,
        "force_evaluations_per_path": 0,
        "optimization_N_F_contribution": 0,
    }
    for key, expected in expected_protocol.items():
        _require(
            protocol.get(key) == expected,
            f"diagnostic protocol mismatch for {key}",
        )

    endpoint = contract.get("endpoint_policy", {})
    _require(endpoint.get("number_of_fixed_endpoints") == 2, "endpoint count mismatch")
    _require(
        endpoint.get("optimization_fixed_endpoint_N_F") == 0,
        "fixed endpoints must contribute zero optimization N_F",
    )
    _require(
        endpoint.get("dense_grid_includes_both_endpoints") is True,
        "dense grid must include both endpoints",
    )

    locks = contract.get("lock_policy", {})
    _require(locks.get("jobs_per_host") == 1, "host job count is not one")
    _require(
        locks.get("acquire_both_before_first_physical_call") is True,
        "lock acquisition policy is incomplete",
    )
    _require(
        locks.get("hold_both_through_last_physical_call") is True,
        "lock holding policy is incomplete",
    )

    bindings = contract.get("bindings", {})
    for role, record in bindings.items():
        path = resolve_path(contract_path, str(record["path"]))
        _require_hash(path, str(record["sha256"]), role)
    _require(
        bindings["metric_policy_amendment"]["sha256"]
        == METRIC_AMENDMENT_SHA256,
        "metric-policy amendment binding mismatch",
    )
    _require(
        bindings["alpha_scope_R2"]["sha256"] == ALPHA_SCOPE_R2_SHA256,
        "alpha-scope R2 binding mismatch",
    )
    return contract


def _load_row(contract_path: Path, contract: dict) -> tuple[dict, dict]:
    source = contract["source_path"]
    _require(int(source["optimization_N_F"]) == EXPECTED_N_F, "source N_F mismatch")
    _require(int(source["batch_id"]) == EXPECTED_BATCH_ID, "source batch mismatch")
    _require(
        str(source["official_path32_sha256"])
        == EXPECTED_OFFICIAL_PATH32_SHA256,
        "registered official path identity mismatch",
    )
    _require(
        str(source["raw_path_sha256"]) == EXPECTED_RAW_PATH_SHA256,
        "registered raw path identity mismatch",
    )
    _require(
        float(source["discrete_I_perp_eV2_per_A"])
        == EXPECTED_DISCRETE_I_PERP,
        "registered discrete I_perp mismatch",
    )

    snapshot_record = source["snapshot"]
    force_record = source["paid_force_receipt"]
    snapshot = resolve_path(contract_path, str(snapshot_record["path"]))
    force_receipt = resolve_path(contract_path, str(force_record["path"]))
    _require_hash(snapshot, str(snapshot_record["sha256"]), "N_F=4080 snapshot")
    _require_hash(
        force_receipt,
        str(force_record["sha256"]),
        "N_F=4080 paid-force receipt",
    )

    with np.load(snapshot, allow_pickle=False) as saved:
        path = np.asarray(saved["nodes"], dtype=np.float64)
        snapshot_nf = int(_scalar(saved, "nf"))
        snapshot_batch = int(_scalar(saved, "batch_id"))
        evaluated = bool(_scalar(saved, "evaluated"))
        method = str(_scalar(saved, "method"))
        disposition = str(_scalar(saved, "disposition"))
        representation = str(_scalar(saved, "representation"))
        state_sha = str(_scalar(saved, "state_sha256"))
        paid_path_sha = str(_scalar(saved, "paid_path_sha256"))
    _require(path.shape == (N_IMAGES, DIMENSION), "snapshot path shape mismatch")
    _require(snapshot_nf == EXPECTED_N_F, "snapshot N_F mismatch")
    _require(snapshot_batch == EXPECTED_BATCH_ID, "snapshot batch mismatch")
    _require(evaluated, "snapshot is not an evaluated state")
    _require(method == "neb", "snapshot is not ODE--NEB")
    _require(disposition == "accepted", "N_F=4080 state is not accepted")
    _require(
        representation == "frozen_piecewise_linear_32_images",
        "snapshot representation mismatch",
    )
    official_sha = _official_array_sha256(path.reshape(N_IMAGES, DIMENSION // 3, 3))
    _require(
        official_sha
        == state_sha
        == paid_path_sha
        == EXPECTED_OFFICIAL_PATH32_SHA256,
        "official snapshot path identity mismatch",
    )
    raw_sha = sha256_array(path)
    _require(raw_sha == EXPECTED_RAW_PATH_SHA256, "raw snapshot path hash mismatch")

    with np.load(force_receipt, allow_pickle=False) as saved:
        paid_coordinates = np.asarray(saved["coordinates30_A"], dtype=np.float64)
        paid_forces = np.asarray(saved["forces_eV_per_A"], dtype=np.float64)
        receipt_nf = int(_scalar(saved, "nf_after"))
        receipt_batch = int(_scalar(saved, "batch_id"))
    expected_shape = (N_INTERNAL, DIMENSION // 3, 3)
    _require(paid_coordinates.shape == expected_shape, "paid coordinate shape mismatch")
    _require(paid_forces.shape == expected_shape, "paid force shape mismatch")
    _require(receipt_nf == EXPECTED_N_F, "paid-force receipt N_F mismatch")
    _require(receipt_batch == EXPECTED_BATCH_ID, "paid-force receipt batch mismatch")
    _require(
        np.array_equal(
            path[1:-1].reshape(N_INTERNAL, DIMENSION // 3, 3),
            paid_coordinates,
        ),
        "paid-force coordinates differ from the snapshot internal images",
    )

    endpoint_record = contract["bindings"]["fixed_endpoint_cache"]
    endpoint_cache = resolve_path(contract_path, str(endpoint_record["path"]))
    with np.load(endpoint_cache, allow_pickle=False) as saved:
        endpoints = np.asarray(saved["coordinates_A"], dtype=np.float64)
    _require(endpoints.shape == (2, DIMENSION), "endpoint-cache shape mismatch")
    _require(
        np.array_equal(path[[0, -1]], endpoints),
        "snapshot endpoints differ from the frozen endpoint cache",
    )

    diagnostic = path_diagnostics(
        path,
        np.zeros(N_IMAGES, dtype=np.float64),
        paid_forces.reshape(N_INTERNAL, DIMENSION),
    )
    discrete_i_perp = float(diagnostic["i_perp"])
    _require(
        discrete_i_perp == EXPECTED_DISCRETE_I_PERP,
        "reconstructed discrete I_perp differs from the frozen value",
    )

    zero_force_report_record = contract["bindings"]["zero_force_extraction_report"]
    zero_force_report_path = resolve_path(
        contract_path, str(zero_force_report_record["path"])
    )
    zero_force_report = json.loads(zero_force_report_path.read_text(encoding="utf-8"))
    _require(
        zero_force_report.get("status")
        == "PASS_POSTHOC_EXTRACTION_UNDER_PREEXISTING_FROZEN_METRIC",
        "zero-force extraction report did not pass",
    )
    _require(
        int(zero_force_report["optimization_Number_of_Forces"]) == EXPECTED_N_F,
        "zero-force report N_F mismatch",
    )
    _require(
        float(zero_force_report["metric"]["value_eV2_per_A"])
        == discrete_i_perp,
        "zero-force report I_perp mismatch",
    )
    _require(
        str(zero_force_report["source_snapshot"]["state_sha256"])
        == official_sha,
        "zero-force report path identity mismatch",
    )
    validation = zero_force_report["validation"]
    _require(
        int(validation["new_OpenMM_energy_attempts"]) == 0
        and int(validation["new_OpenMM_force_attempts"]) == 0
        and int(validation["optimization_Number_of_Forces_contribution"]) == 0,
        "zero-force report contains a physical diagnostic call",
    )

    dense, geometry = uniform_arclength_resample(path)
    dense_sha = sha256_array(dense)
    _require(
        dense_sha == EXPECTED_DENSE_COORDINATE_SHA256,
        "deterministic arc-length-500 coordinate hash mismatch",
    )
    row = {
        "label": "ODE-NEB_NF4080",
        "optimization_N_F": EXPECTED_N_F,
        "batch_id": EXPECTED_BATCH_ID,
        "discrete_I_perp": discrete_i_perp,
        "path": np.ascontiguousarray(path),
        "path_sha256": raw_sha,
        "official_path32_sha256": official_sha,
        "source_result": str(snapshot),
        "source_result_sha256": sha256_file(snapshot),
        "source_audit": str(force_receipt),
        "source_audit_sha256": sha256_file(force_receipt),
    }
    preflight = {
        "optimization_N_F": EXPECTED_N_F,
        "batch_id": EXPECTED_BATCH_ID,
        "discrete_I_perp_eV2_per_A": discrete_i_perp,
        "official_path32_sha256": official_sha,
        "raw_path_sha256": raw_sha,
        "dense_coordinate_sha256": dense_sha,
        **geometry,
        "fixed_endpoints_match_cache_bytewise": True,
        "internal_images_match_paid_force_coordinates_bytewise": True,
    }
    return row, preflight


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

    contract_path = args.contract.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    _require(output.suffix == ".npz", "--output must end in .npz")
    contract = _load_contract(contract_path)
    row, preflight = _load_row(contract_path, contract)

    if args.validate_only:
        print(
            json.dumps(
                {
                    "schema": "bb-ode-neb-nf4080-arc500-zero-openmm-preflight-v1",
                    "status": "PASS",
                    "scope": "ODE-NEB_NF4080_SINGLE_RETAINED_PATH_ONLY",
                    "all_path_ode_neb_arc500_task": "REMAINS_PAUSED",
                    "new_OpenMM_energy_attempts": 0,
                    "new_OpenMM_force_attempts": 0,
                    "optimization_N_F_contribution": 0,
                    "physical_evaluation_locks_acquired": False,
                    "row": preflight,
                },
                sort_keys=True,
            )
        )
        return

    audit_output = output.with_suffix(".audit.json")
    wal_output = output.with_suffix(".energy.wal.jsonl")
    if output.exists() or audit_output.exists() or wal_output.exists():
        raise FileExistsError("refusing to overwrite an ODE--NEB N_F=4080 diagnostic")

    topology_record = contract["topology"]
    topology = resolve_path(contract_path, str(topology_record["path"]))
    _require_hash(topology, str(topology_record["sha256"]), "topology")
    with acquire_host_locks(
        (args.global_lock_path, args.peer_global_lock_path)
    ) as held_locks:
        evaluated = evaluate_paths(topology, [row], wal_output)
    result = evaluated[0]

    atomic_save_npz(
        output,
        label=np.asarray([result["label"]]),
        optimization_N_F=np.asarray([result["optimization_N_F"]], dtype=np.int64),
        batch_id=np.asarray([result["batch_id"]], dtype=np.int64),
        discrete_I_perp=np.asarray([result["discrete_I_perp"]], dtype=np.float64),
        densified_Delta_E_kcal_per_mol=np.asarray(
            [result["densified_Delta_E_kcal_per_mol"]], dtype=np.float64
        ),
        energy_eV=np.asarray([result["energy_eV"]], dtype=np.float64),
        selected_path_sha256=np.asarray([result["path_sha256"]]),
        official_path32_sha256=np.asarray([result["official_path32_sha256"]]),
        dense_coordinate_sha256=np.asarray([result["dense_coordinate_sha256"]]),
        energy_sha256=np.asarray([result["energy_sha256"]]),
        total_arclength_A=np.asarray([result["total_arclength_A"]], dtype=np.float64),
        dense_spacing_A=np.asarray([result["dense_spacing_A"]], dtype=np.float64),
        diagnostic_energy_attempts_per_path=np.asarray(
            [N_DENSE_POINTS], dtype=np.int64
        ),
        diagnostic_energy_successes_per_path=np.asarray(
            [N_DENSE_POINTS], dtype=np.int64
        ),
        diagnostic_force_requests_per_path=np.asarray([0], dtype=np.int64),
        optimization_N_F_contribution=np.asarray(0, dtype=np.int64),
        fixed_endpoint_optimization_N_F=np.asarray(0, dtype=np.int64),
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
        "schema": AUDIT_SCHEMA,
        "status": "PASS",
        "scope": "ODE-NEB_NF4080_SINGLE_RETAINED_PATH_ONLY",
        "all_path_ode_neb_arc500_task": "REMAINS_PAUSED",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "energy_wal": str(wal_output),
        "energy_wal_sha256": sha256_file(wal_output),
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "diagnostic_energy_attempts": N_DENSE_POINTS,
        "diagnostic_energy_successes": N_DENSE_POINTS,
        "diagnostic_force_requests": 0,
        "optimization_N_F_contribution": 0,
        "fixed_endpoint_optimization_N_F": 0,
        "held_host_locks": held_locks,
        "record": {
            **preflight,
            "densified_Delta_E_kcal_per_mol": float(
                result["densified_Delta_E_kcal_per_mol"]
            ),
            "energy_sha256": str(result["energy_sha256"]),
        },
    }
    atomic_write_json(audit_output, audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
