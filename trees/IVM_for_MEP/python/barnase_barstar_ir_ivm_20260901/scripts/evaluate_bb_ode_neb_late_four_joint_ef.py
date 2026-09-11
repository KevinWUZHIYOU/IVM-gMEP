#!/usr/bin/env python3
"""Audited late-four ODE--NEB arc500 joint energy/full-force diagnostic.

The original 32-image paths and their 30 already-paid internal-image energy/
force receipts are immutable inputs.  They define the article's discrete
``I_perp`` without any new physical call.  Each path is independently sampled
at 500 points uniformly in unweighted Cartesian arc length and evaluated with
one joint energy/full-force call per point.  Complete path shards are the only
restartable units; any failed, open, or incomplete path attempt fails closed
and requires a superseding registration.

This module imports OpenMM-capable code but constructs a backend only in
``run`` after both canonical host-local locks have been acquired.  Contract
loading and preflight are zero-physical-call operations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable, Sequence
import uuid

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENGINE_PATH = HERE / "evaluate_bb_dense_four_paths_joint_v2.py"
ODE_HELPER_PATH = HERE / "freeze_targeted_ode_neb_arc500.py"
FREEZER_PATH = HERE / "freeze_bb_ode_neb_late_four_joint_ef_contract.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ENGINE = _load("bb_late_four_joint_engine", ENGINE_PATH)
ODE = _load("bb_late_four_paid_ode_helper", ODE_HELPER_PATH)

SCHEMA = "bb-ode-neb-late-four-arc500-joint-ef-contract-v1"
SHARD_SCHEMA = "bb-ode-neb-late-path-arc500-joint-ef-shard-v1"
RESULT_SCHEMA = "bb-ode-neb-late-four-arc500-joint-ef-result-v1"
MANIFEST_SCHEMA = "bb-ode-neb-late-four-arc500-joint-ef-manifest-v1"
AUDIT_SCHEMA = "bb-ode-neb-late-four-arc500-joint-ef-audit-v1"
SOURCE_WAL_SCHEMA = "bb-ode-neb-late-four-paid-source-wal-v1"
DENSE_WAL_SCHEMA = "bb-ode-neb-late-four-arc500-joint-ef-wal-v1"

NF_VALUES = (2040, 4080, 8160, 9990)
N_IMAGES = 32
N_INTERNAL = 30
N_ATOMS = 3132
DIMENSION = 9396
N_DENSE = 500
MAXIMUM_ATTEMPT_STARTED = len(NF_VALUES) * N_DENSE
MINIMUM_FREE_BYTES_RESERVE = 8 * 1024**3
ESTIMATED_SHARD_BYTES = (
    (2 * N_DENSE * DIMENSION + N_INTERNAL * DIMENSION + N_IMAGES * DIMENSION)
    * 8
    + 8 * 1024**2
)

OFFICIAL_SOURCE_ROOT = (
    ROOT / "inputs/baselines/ode_neb_all_batches_k0p1"
).resolve()
RECEIPTS = {
    2040: Path(
        "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/"
        "bb_neb_aseode30_k0p1_nf9990_remote_20260817_r2/runs/"
        "fresh-nonresume-bb-neb-aseode-k0p1-nf9990-20260817-r2/"
        "milestone_force_receipts/nf02040_batch_0068_EF30.npz"
    ),
    4080: Path(
        "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/"
        "bb_neb_aseode30_k0p1_nf9990_remote_20260817_r2/runs/"
        "fresh-nonresume-bb-neb-aseode-k0p1-nf9990-20260817-r2/"
        "milestone_force_receipts/nf04080_batch_0136_EF30.npz"
    ),
    8160: Path(
        "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/"
        "bb_neb_aseode30_k0p1_nf9990_remote_20260817_r2/runs/"
        "fresh-nonresume-bb-neb-aseode-k0p1-nf9990-20260817-r2/"
        "milestone_force_receipts/nf08160_batch_0272_EF30.npz"
    ),
    9990: ROOT
    / "inputs/baselines/ode_neb_source_paths_nf240_480_990_9990/"
    "nf09990_batch_0333_EF30.npz",
}
CANONICAL_LOCKS = {
    Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock").resolve(),
    Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock").resolve(),
}
CANONICAL_RESULT = (
    ROOT
    / "runs/diagnostics/bb_ode_neb_late_four_arc500_joint_ef_20260902_r1.npz"
).resolve()

THREAD_CAPS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENMM_CPU_THREADS",
)
METRIC_POLICY = {
    "coordinate_space": "unweighted_3N_cartesian",
    "sampling": "uniform_cartesian_arclength",
    "interpolation": "piecewise_linear",
    "number_of_points": 500,
    "grid_formula": "s_j=j*L/499,j=0,...,499",
    "include_both_endpoints": True,
    "dense_physical_call": "one_joint_energy_full_force_getState_per_point",
    "dense_energy_and_force_same_call": True,
    "dense_force_storage": "complete_full_force_vector_for_every_point",
    "discrete_I_perp_source": "sealed_paid_30_internal_force_receipt",
    "discrete_I_perp_implementation": "bb_core.path_diagnostics",
    "original_fixed_endpoint_diagnostic_calls": 0,
    "dense_fixed_endpoint_joint_EF_calls": 2,
    "optimization_N_F_contribution": 0,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bind(path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": ENGINE.sha256_file(resolved)}


def scalar(saved, key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} must be scalar")
    return value.item()


def same_float(left, right) -> bool:
    return np.asarray(float(left), dtype="<f8").tobytes() == np.asarray(
        float(right), dtype="<f8"
    ).tobytes()


def _read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _load_receipt(path: Path, nf: int, row: dict) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "coordinates30_A",
            "energies_eV",
            "forces_eV_per_A",
            "batch_id",
            "nf_after",
        }
        require(required.issubset(saved.files), f"NF{nf} paid receipt is incomplete")
        coordinates = np.asarray(saved["coordinates30_A"], dtype=np.float64)
        energies = np.asarray(saved["energies_eV"], dtype=np.float64)
        forces = np.asarray(saved["forces_eV_per_A"], dtype=np.float64)
        batch_id = int(scalar(saved, "batch_id"))
        receipt_nf = int(scalar(saved, "nf_after"))
    require(receipt_nf == nf, f"NF{nf} receipt N_F mismatch")
    require(batch_id == int(row["paid_batch_id"]), f"NF{nf} receipt batch mismatch")
    require(
        coordinates.shape == forces.shape == (N_INTERNAL, N_ATOMS, 3),
        f"NF{nf} receipt coordinate/force shape mismatch",
    )
    require(energies.shape == (N_INTERNAL,), f"NF{nf} receipt energy shape mismatch")
    require(
        np.all(np.isfinite(coordinates))
        and np.all(np.isfinite(energies))
        and np.all(np.isfinite(forces)),
        f"NF{nf} receipt contains non-finite data",
    )
    return {
        "coordinates": np.ascontiguousarray(coordinates.reshape(N_INTERNAL, DIMENSION)),
        "energies": np.ascontiguousarray(energies),
        "forces": np.ascontiguousarray(forces.reshape(N_INTERNAL, DIMENSION)),
        "batch_id": batch_id,
    }


def discover_official_rows() -> tuple[list[dict], dict]:
    """Revalidate the official ledger and the four exact paid-force receipts."""

    paid, official_bindings = ODE.load_official_paid_ledger(OFFICIAL_SOURCE_ROOT)
    by_nf = {int(row["optimization_N_F"]): row for row in paid}
    receipt_paths = [RECEIPTS[nf].expanduser().resolve(strict=True) for nf in NF_VALUES]
    values, receipt_bindings, excluded = ODE.load_exact_force_receipts(
        receipt_paths, paid
    )
    require(not excluded, "one of the four fixed paid receipts is a rejected trial")
    receipt_by_nf = {int(row["optimization_N_F"]): row for row in receipt_bindings}
    rows = []
    for nf in NF_VALUES:
        source = by_nf.get(nf)
        receipt_record = receipt_by_nf.get(nf)
        require(source is not None and receipt_record is not None, f"missing NF{nf} evidence")
        require(source["disposition"] == "accepted", f"NF{nf} is not an accepted state")
        archive = Path(source["coordinate_archive"]).resolve(strict=True)
        path = ENGINE._load_ode_path(archive)
        receipt_path = RECEIPTS[nf].expanduser().resolve(strict=True)
        receipt = _load_receipt(receipt_path, nf, source)
        require(
            np.array_equal(receipt["coordinates"], path[1:-1]),
            f"NF{nf} paid receipt does not match retained path",
        )
        official_sha = ENGINE.official_path32_sha256(
            path.reshape(N_IMAGES, N_ATOMS, 3)
        )
        require(
            official_sha == source["official_path32_sha256"],
            f"NF{nf} official path identity mismatch",
        )
        i_perp = float(
            ENGINE.path_diagnostics(
                path,
                np.zeros(N_IMAGES, dtype=np.float64),
                receipt["forces"],
            )["i_perp"]
        )
        require(
            same_float(i_perp, receipt_record["discrete_I_perp_eV2_per_A"]),
            f"NF{nf} reconstructed I_perp mismatch",
        )
        rows.append(
            {
                "row_index": len(rows),
                "label": f"ODE-NEB_NF{nf}",
                "method": "ODE-NEB",
                "host_role": "local",
                "optimization_N_F": nf,
                "path_identifier": {
                    "optimization_N_F": nf,
                    "paid_batch_id": int(source["paid_batch_id"]),
                    "source_batch_id": int(source["source_batch_id"]),
                    "disposition": str(source["disposition"]),
                },
                "source": bind(archive),
                "paid_force_receipt": bind(receipt_path),
                "path_sha256": ENGINE.sha256_array(path),
                "official_path32_sha256": official_sha,
                "discrete_I_perp": i_perp,
                "paid_internal_energy_sha256": ENGINE.sha256_array(
                    receipt["energies"]
                ),
                "paid_internal_force_sha256": ENGINE.sha256_array(
                    receipt["forces"]
                ),
                "path": path,
                "receipt": receipt,
                "source_path": archive,
                "receipt_path": receipt_path,
            }
        )
    return rows, official_bindings


def frozen_row(row: dict) -> dict:
    keys = (
        "label",
        "method",
        "host_role",
        "optimization_N_F",
        "path_identifier",
        "source",
        "paid_force_receipt",
        "path_sha256",
        "official_path32_sha256",
        "discrete_I_perp",
        "paid_internal_energy_sha256",
        "paid_internal_force_sha256",
    )
    result = {key: row[key] for key in keys}
    result.update(
        {
            "original_discrete_policy": (
                "reuse_exact_paid_30_internal_joint_EF_receipt_zero_new_calls"
            ),
            "dense_policy": (
                "fresh_exactly_500_uniform_cartesian_arc_length_joint_EF_"
                "then_atomic_complete_path_shard"
            ),
        }
    )
    return result


def build_contract(locks: Sequence[Path], topology: Path) -> dict:
    rows, official_bindings = discover_official_rows()
    resolved_locks = [Path(path).expanduser().resolve() for path in locks]
    require(
        len(resolved_locks) == len(set(resolved_locks)) == 2
        and set(resolved_locks) == CANONICAL_LOCKS,
        "exactly the two canonical host-local BB locks are required",
    )
    return {
        "schema": SCHEMA,
        "frozen": True,
        "status": "FROZEN_BEFORE_ANY_LATE_FOUR_OPENMM_OR_PHYSICAL_CALL",
        "purpose": (
            "late ODE--NEB cost-to-precision support at NF2040/4080/8160/9990 "
            "with paid discrete I_perp and fresh same-call arc500 energy/full-force"
        ),
        "method": "ODE-NEB",
        "host_role": "local",
        "jobs": 1,
        "result_output": str(CANONICAL_RESULT),
        "host_local_lock_paths": [str(path) for path in resolved_locks],
        "evaluator": bind(Path(__file__)),
        "contract_freezer": bind(FREEZER_PATH),
        "joint_engine": bind(ENGINE_PATH),
        "ode_paid_ledger_validator": bind(ODE_HELPER_PATH),
        "bb_core": bind(ROOT / "src/bb_core.py"),
        "topology": bind(topology),
        "official_paid_ledger": official_bindings,
        "metric_policy": METRIC_POLICY,
        "paths": [frozen_row(row) for row in rows],
        "physical_call_policy": {
            "original_discrete_new_calls": 0,
            "paid_internal_joint_EF_values_reused_per_path": N_INTERNAL,
            "fresh_dense_joint_EF_successes_per_path": N_DENSE,
            "fresh_dense_joint_EF_successes_total": MAXIMUM_ATTEMPT_STARTED,
            "maximum_attempt_started_records": MAXIMUM_ATTEMPT_STARTED,
            "failed_or_open_attempt_retry_under_same_contract": False,
            "dense_full_force_storage": True,
            "optimization_N_F_contribution": 0,
            "jobs": 1,
            "all_six_thread_caps": 1,
            "two_canonical_host_local_fcntl_locks": True,
        },
        "durability_policy": {
            "separate_paid_source_and_dense_append_only_fsync_WAL": True,
            "atomic_complete_path_shard": True,
            "shard_contains_dense_coordinates_energy_full_force": True,
            "shard_contains_paid_internal_energy_full_force": True,
            "restart_reuse_unit": "complete_fully_validated_path_shard_only",
            "individual_incomplete_attempt_success_reuse": False,
            "failed_or_open_attempt_requires_superseding_registration": True,
            "no_overwrite": True,
        },
        "storage_policy": {
            "authoritative_unit": "complete_atomic_path_shard",
            "aggregate_role": (
                "index_metrics_and_plot_energy_only_no_duplicate_dense_coordinates_or_forces"
            ),
            "minimum_free_bytes_reserve": MINIMUM_FREE_BYTES_RESERVE,
            "estimated_uncompressed_shard_bytes_per_path": ESTIMATED_SHARD_BYTES,
        },
        "isolation": {
            "early_four_results_read": False,
            "early_four_results_modified": False,
            "late_four_namespace_only": True,
        },
    }


def load_contract(contract_path: Path, host_role: str) -> dict:
    contract_path = Path(contract_path).expanduser().resolve(strict=True)
    contract = _read_json(contract_path)
    require(
        contract.get("schema") == SCHEMA and contract.get("frozen") is True,
        "invalid late-four contract",
    )
    require(
        contract.get("status")
        == "FROZEN_BEFORE_ANY_LATE_FOUR_OPENMM_OR_PHYSICAL_CALL",
        "late-four contract status mismatch",
    )
    require(host_role == contract.get("host_role") == "local", "late-four is local-only")
    require(contract.get("method") == "ODE-NEB" and contract.get("jobs") == 1, "method/jobs mismatch")
    require(contract.get("metric_policy") == METRIC_POLICY, "metric policy mismatch")
    expected_files = {
        "evaluator": Path(__file__).resolve(),
        "contract_freezer": FREEZER_PATH.resolve(),
        "joint_engine": ENGINE_PATH.resolve(),
        "ode_paid_ledger_validator": ODE_HELPER_PATH.resolve(),
        "bb_core": (ROOT / "src/bb_core.py").resolve(),
    }
    for role, expected in expected_files.items():
        actual = ENGINE.require_file(contract[role], contract_path, role)
        require(actual == expected, f"{role} path mismatch")
    topology = ENGINE.require_file(contract["topology"], contract_path, "topology")
    rows, official_bindings = discover_official_rows()
    require(contract.get("official_paid_ledger") == official_bindings, "official ledger bindings drifted")
    frozen = contract.get("paths")
    require(isinstance(frozen, list) and len(frozen) == 4, "contract must bind four paths")
    require(tuple(int(row["optimization_N_F"]) for row in frozen) == NF_VALUES, "late NF order drifted")
    loaded = []
    for expected, actual in zip(frozen, rows):
        baseline = frozen_row(actual)
        require(expected == baseline, f"NF{actual['optimization_N_F']} frozen evidence drifted")
        loaded.append(actual)
    locks = [Path(value).expanduser().resolve() for value in contract["host_local_lock_paths"]]
    require(len(locks) == len(set(locks)) == 2 and set(locks) == CANONICAL_LOCKS, "canonical lock set drifted")
    require(Path(contract["result_output"]).expanduser().resolve() == CANONICAL_RESULT, "result path drifted")
    physical = contract.get("physical_call_policy", {})
    require(physical.get("original_discrete_new_calls") == 0, "discrete call policy drifted")
    require(physical.get("fresh_dense_joint_EF_successes_total") == MAXIMUM_ATTEMPT_STARTED, "dense call count drifted")
    require(physical.get("maximum_attempt_started_records") == MAXIMUM_ATTEMPT_STARTED, "attempt cap drifted")
    require(physical.get("failed_or_open_attempt_retry_under_same_contract") is False, "retry policy drifted")
    require(physical.get("dense_full_force_storage") is True, "full-force storage not required")
    require(physical.get("optimization_N_F_contribution") == 0, "diagnostics changed optimization N_F")
    durability = contract.get("durability_policy", {})
    require(durability.get("restart_reuse_unit") == "complete_fully_validated_path_shard_only", "restart policy drifted")
    require(durability.get("individual_incomplete_attempt_success_reuse") is False, "partial success reuse permitted")
    require(durability.get("failed_or_open_attempt_requires_superseding_registration") is True, "fail-close policy drifted")
    require(contract.get("isolation") == {"early_four_results_read": False, "early_four_results_modified": False, "late_four_namespace_only": True}, "early-four isolation drifted")
    return {
        "contract": contract,
        "contract_path": contract_path,
        "contract_sha256": ENGINE.sha256_file(contract_path),
        "topology": topology,
        "locks": locks,
        "rows": loaded,
        "result_output": CANONICAL_RESULT,
    }


def _shard_path(output: Path, nf: int) -> Path:
    return (
        output.with_suffix("")
        .with_name(output.stem + ".shards")
        / f"ode_neb_nf{nf:05d}_arc500_joint_ef_late_r1.npz"
    )


def _wal_paths(output: Path) -> tuple[Path, Path]:
    return output.with_suffix(".paid_source.wal.jsonl"), output.with_suffix(
        ".dense_joint_ef.wal.jsonl"
    )


def _final_paths(output: Path) -> tuple[Path, Path, Path]:
    return output, output.with_suffix(".manifest.json"), output.with_suffix(".audit.json")


def _read_wal(path: Path, schema: str, state: dict) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    require(lines, f"empty WAL: {path}")
    events = [json.loads(line) for line in lines]
    header = events[0]
    require(header.get("event") == "wal_header" and header.get("schema") == schema, f"invalid WAL header: {path}")
    require(header.get("contract_sha256") == state["contract_sha256"], f"WAL contract hash mismatch: {path}")
    require(Path(header["contract"]).resolve() == state["contract_path"], f"WAL contract path mismatch: {path}")
    return events


def _lifetime(events: list[dict]) -> dict:
    starts = [event for event in events if event.get("event") == "attempt_started"]
    successes = [event for event in events if event.get("event") == "attempt_succeeded"]
    failures = [event for event in events if event.get("event") == "attempt_failed"]
    start_ids = [str(event["attempt_id"]) for event in starts]
    require(len(start_ids) == len(set(start_ids)), "duplicate attempt id")
    success_ids = {str(event["attempt_id"]) for event in successes}
    failure_ids = {str(event["attempt_id"]) for event in failures}
    require(success_ids.isdisjoint(failure_ids), "attempt has conflicting terminal events")
    require(success_ids | failure_ids <= set(start_ids), "terminal attempt lacks start")
    logical = {
        (int(event["row_index"]), int(event["point_index"])) for event in starts
    }
    return {
        "attempt_started": len(starts),
        "attempt_succeeded": len(successes),
        "attempt_failed": len(failures),
        "attempt_open": len(set(start_ids) - success_ids - failure_ids),
        "retry_attempts": len(starts) - len(logical),
    }


def _validate_shard(path: Path, row: dict, state: dict) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "schema", "status", "method", "label", "optimization_N_F",
            "path_identifier_json", "contract_sha256", "evaluator_sha256",
            "joint_engine_sha256", "ode_validator_sha256", "bb_core_sha256",
            "topology_sha256", "selected_path", "selected_path_sha256",
            "official_path32_sha256", "paid_force_receipt",
            "paid_force_receipt_sha256", "paid_internal_energy_eV",
            "paid_internal_force_eV_per_A", "paid_internal_energy_sha256",
            "paid_internal_force_sha256", "discrete_I_perp",
            "dense_coordinates_A", "dense_energy_eV", "dense_force_eV_per_A",
            "dense_coordinate_sha256", "dense_energy_sha256",
            "dense_force_sha256", "densified_Delta_E_kcal_per_mol",
            "number_of_points", "original_discrete_new_calls",
            "dense_joint_EF_attempts", "optimization_N_F_contribution",
        }
        require(required.issubset(saved.files), f"incomplete shard {path}")
        require(str(scalar(saved, "schema")) == SHARD_SCHEMA and str(scalar(saved, "status")) == "PASS", "shard identity mismatch")
        require(str(scalar(saved, "method")) == "ODE-NEB", "shard method mismatch")
        require(int(scalar(saved, "optimization_N_F")) == row["optimization_N_F"], "shard N_F mismatch")
        require(str(scalar(saved, "label")) == row["label"], "shard label mismatch")
        require(json.loads(str(scalar(saved, "path_identifier_json"))) == row["path_identifier"], "shard identifier mismatch")
        bindings = {
            "contract_sha256": state["contract_sha256"],
            "evaluator_sha256": ENGINE.sha256_file(Path(__file__)),
            "joint_engine_sha256": ENGINE.sha256_file(ENGINE_PATH),
            "ode_validator_sha256": ENGINE.sha256_file(ODE_HELPER_PATH),
            "bb_core_sha256": ENGINE.sha256_file(ROOT / "src/bb_core.py"),
            "topology_sha256": ENGINE.sha256_file(state["topology"]),
        }
        for key, expected in bindings.items():
            require(str(scalar(saved, key)) == expected, f"shard {key} mismatch")
        selected = np.asarray(saved["selected_path"], dtype=np.float64)
        paid_e = np.asarray(saved["paid_internal_energy_eV"], dtype=np.float64)
        paid_f = np.asarray(saved["paid_internal_force_eV_per_A"], dtype=np.float64)
        dense = np.asarray(saved["dense_coordinates_A"], dtype=np.float64)
        dense_e = np.asarray(saved["dense_energy_eV"], dtype=np.float64)
        dense_f = np.asarray(saved["dense_force_eV_per_A"], dtype=np.float64)
        require(selected.shape == (N_IMAGES, DIMENSION), "shard path shape mismatch")
        require(paid_e.shape == (N_INTERNAL,) and paid_f.shape == (N_INTERNAL, DIMENSION), "shard paid E/F shape mismatch")
        require(dense.shape == dense_f.shape == (N_DENSE, DIMENSION) and dense_e.shape == (N_DENSE,), "shard dense E/F shape mismatch")
        require(all(np.all(np.isfinite(value)) for value in (selected, paid_e, paid_f, dense, dense_e, dense_f)), "shard has non-finite arrays")
        require(ENGINE.sha256_array(selected) == row["path_sha256"] == str(scalar(saved, "selected_path_sha256")), "shard selected path hash mismatch")
        require(str(scalar(saved, "official_path32_sha256")) == row["official_path32_sha256"], "shard official hash mismatch")
        require(str(scalar(saved, "paid_force_receipt")) == str(row["receipt_path"]), "shard receipt path mismatch")
        require(str(scalar(saved, "paid_force_receipt_sha256")) == row["paid_force_receipt"]["sha256"], "shard receipt SHA mismatch")
        require(np.array_equal(paid_e, row["receipt"]["energies"]) and np.array_equal(paid_f, row["receipt"]["forces"]), "shard paid E/F differs from receipt")
        require(ENGINE.sha256_array(paid_e) == row["paid_internal_energy_sha256"] == str(scalar(saved, "paid_internal_energy_sha256")), "shard paid energy hash mismatch")
        require(ENGINE.sha256_array(paid_f) == row["paid_internal_force_sha256"] == str(scalar(saved, "paid_internal_force_sha256")), "shard paid force hash mismatch")
        expected_dense, _ = ENGINE.uniform_arc500(selected)
        require(np.array_equal(dense, expected_dense), "shard dense coordinates are not deterministic arc500")
        require(ENGINE.sha256_array(dense) == str(scalar(saved, "dense_coordinate_sha256")), "shard dense coordinate hash mismatch")
        require(ENGINE.sha256_array(dense_e) == str(scalar(saved, "dense_energy_sha256")), "shard dense energy hash mismatch")
        require(ENGINE.sha256_array(dense_f) == str(scalar(saved, "dense_force_sha256")), "shard dense force hash mismatch")
        i_perp = float(ENGINE.path_diagnostics(selected, np.zeros(N_IMAGES), paid_f)["i_perp"])
        require(same_float(i_perp, row["discrete_I_perp"]) and same_float(i_perp, scalar(saved, "discrete_I_perp")), "shard I_perp mismatch")
        delta = float((np.max(dense_e) - dense_e[0]) * ENGINE.EV_TO_KCAL)
        require(same_float(delta, scalar(saved, "densified_Delta_E_kcal_per_mol")), "shard Delta E mismatch")
        require(int(scalar(saved, "number_of_points")) == N_DENSE, "shard point count mismatch")
        require(int(scalar(saved, "original_discrete_new_calls")) == 0, "shard claims new discrete calls")
        require(int(scalar(saved, "dense_joint_EF_attempts")) == N_DENSE, "shard attempt count mismatch")
        require(int(scalar(saved, "optimization_N_F_contribution")) == 0, "shard changes optimization N_F")
        return {key: np.asarray(saved[key]).copy() for key in saved.files}


def _validate_dense_closure(events: list[dict], row: dict, shard: dict, path: Path) -> None:
    row_events = [event for event in events if int(event.get("row_index", -1)) == row["row_index"]]
    starts = [event for event in row_events if event.get("event") == "attempt_started"]
    successes = [event for event in row_events if event.get("event") == "attempt_succeeded"]
    failures = [event for event in row_events if event.get("event") == "attempt_failed"]
    require(len(starts) == len(successes) == N_DENSE and not failures, f"NF{row['optimization_N_F']} WAL is not one complete path")
    start_by_id = {str(event["attempt_id"]): event for event in starts}
    success_by_point = {}
    for event in successes:
        attempt_id = str(event["attempt_id"])
        require(attempt_id in start_by_id, "dense success lacks start")
        started = start_by_id[attempt_id]
        for key in ("row_index", "label", "optimization_N_F", "path_sha256", "point_index", "coordinate_sha256", "session_id"):
            require(event.get(key) == started.get(key), f"dense start/success {key} mismatch")
        point = int(event["point_index"])
        require(point not in success_by_point, "duplicate dense point success")
        success_by_point[point] = event
    require(sorted(success_by_point) == list(range(N_DENSE)), "dense point coverage mismatch")
    dense = np.asarray(shard["dense_coordinates_A"])
    energy = np.asarray(shard["dense_energy_eV"])
    force = np.asarray(shard["dense_force_eV_per_A"])
    for point in range(N_DENSE):
        event = success_by_point[point]
        require(event["coordinate_sha256"] == ENGINE.sha256_array(dense[point]), "dense coordinate/WAL mismatch")
        require(same_float(event["energy_eV"], energy[point]), "dense energy/WAL mismatch")
        require(event["force_sha256"] == ENGINE.sha256_array(force[point]), "dense force/WAL mismatch")
    publications = [event for event in row_events if event.get("event") == "complete_shard_published"]
    require(any(event.get("shard") == str(path) and event.get("shard_sha256") == ENGINE.sha256_file(path) for event in publications), "dense WAL lacks matching shard publication")


def _validate_source_closure(events: list[dict], row: dict, path: Path) -> None:
    matches = [
        event for event in events
        if event.get("event") == "paid_source_reused"
        and int(event.get("row_index", -1)) == row["row_index"]
        and event.get("receipt_sha256") == row["paid_force_receipt"]["sha256"]
        and event.get("path_sha256") == row["path_sha256"]
        and event.get("new_physical_calls") == 0
    ]
    require(matches, f"NF{row['optimization_N_F']} source WAL closure missing")
    publications = [event for event in events if event.get("event") == "complete_shard_published" and int(event.get("row_index", -1)) == row["row_index"]]
    require(any(event.get("shard") == str(path) and event.get("shard_sha256") == ENGINE.sha256_file(path) for event in publications), "source WAL lacks matching shard publication")


def preflight(state: dict, output: Path, *, probe_locks: bool = True) -> dict:
    output = Path(output).expanduser().resolve()
    require(output == state["result_output"] or state.get("allow_test_output") is True, "output differs from frozen path")
    for final in _final_paths(output):
        if final.exists():
            raise FileExistsError(f"refusing to overwrite {final}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path, dense_path = _wal_paths(output)
    source_events = _read_wal(source_path, SOURCE_WAL_SCHEMA, state)
    dense_events = _read_wal(dense_path, DENSE_WAL_SCHEMA, state)
    lifetime = _lifetime(dense_events)
    require(lifetime["attempt_started"] <= MAXIMUM_ATTEMPT_STARTED, "physical-attempt cap exceeded")
    require(lifetime["attempt_failed"] == lifetime["attempt_open"] == lifetime["retry_attempts"] == 0, "failed/open/retried attempt requires a superseding registration")
    complete = []
    for row in state["rows"]:
        target = _shard_path(output, row["optimization_N_F"])
        row_starts = [event for event in dense_events if event.get("event") == "attempt_started" and int(event.get("row_index", -1)) == row["row_index"]]
        if target.exists():
            shard = _validate_shard(target, row, state)
            _validate_dense_closure(dense_events, row, shard, target)
            _validate_source_closure(source_events, row, target)
            complete.append(row["optimization_N_F"])
        else:
            require(not row_starts, f"NF{row['optimization_N_F']} has incomplete attempts; a superseding registration is required")
    expected_started = len(complete) * N_DENSE
    require(lifetime["attempt_started"] == expected_started, "WAL attempt count is not explained by complete shards")
    missing = len(NF_VALUES) - len(complete)
    required_free = MINIMUM_FREE_BYTES_RESERVE + missing * ESTIMATED_SHARD_BYTES
    observed_free = shutil.disk_usage(output.parent).free
    require(observed_free >= required_free, f"insufficient free disk: {observed_free} < {required_free}")
    if probe_locks:
        with ENGINE.acquire_two_locks(state["locks"]):
            pass
    return {
        "status": "PASS_ZERO_PHYSICAL_PREFLIGHT",
        "optimization_N_F": list(NF_VALUES),
        "paid_discrete_receipts_revalidated": len(NF_VALUES),
        "original_discrete_new_calls_planned": 0,
        "complete_shards_reusable": complete,
        "fresh_paths_planned": missing,
        "fresh_dense_joint_EF_attempts_planned": missing * N_DENSE,
        "fresh_full_force_vectors_planned": missing * N_DENSE,
        "maximum_attempt_started_records": MAXIMUM_ATTEMPT_STARTED,
        "lifetime": lifetime,
        "observed_free_bytes": observed_free,
        "required_free_bytes": required_free,
        "lock_probe": "PASS_NONBLOCKING" if probe_locks else "SKIPPED_FOR_TEST",
        "optimization_N_F_contribution": 0,
        "early_four_artifacts_touched": 0,
        "OpenMM_backend_constructed": False,
    }


def _attempt(wal, backend, row: dict, point: int, coordinate: np.ndarray):
    if not hasattr(wal, "late_attempt_started_count"):
        wal.late_attempt_started_count = _lifetime(
            [json.loads(line) for line in wal.path.read_text().splitlines()]
        )["attempt_started"]
    ordinal = int(wal.late_attempt_started_count) + 1
    require(ordinal <= MAXIMUM_ATTEMPT_STARTED, "physical-attempt authorization exhausted")
    coordinate = np.ascontiguousarray(coordinate, dtype=np.float64)
    attempt_id = f"{wal.session_id}:{row['row_index']}:{point}:{ordinal}:{uuid.uuid4().hex}"
    started = {
        "event": "attempt_started",
        "utc": utc_now(),
        "stage": "dense_arc500_joint_EF",
        "row_index": row["row_index"],
        "label": row["label"],
        "optimization_N_F": row["optimization_N_F"],
        "path_sha256": row["path_sha256"],
        "point_index": point,
        "coordinate_sha256": ENGINE.sha256_array(coordinate),
        "attempt_ordinal": ordinal,
        "session_id": wal.session_id,
        "attempt_id": attempt_id,
    }
    wal.append(started)
    wal.late_attempt_started_count = ordinal
    try:
        energy, force = backend.evaluate_joint(coordinate)
        energy = float(energy)
        force = np.ascontiguousarray(force, dtype=np.float64)
        require(force.shape == (DIMENSION,), "backend returned wrong force shape")
        require(np.isfinite(energy) and np.all(np.isfinite(force)), "backend returned non-finite E/F")
    except Exception as error:
        wal.append({**started, "event": "attempt_failed", "utc": utc_now(), "error": repr(error)})
        raise
    wal.append(
        {
            **started,
            "event": "attempt_succeeded",
            "utc": utc_now(),
            "energy_eV": energy,
            "energy_scalar_sha256": ENGINE.sha256_array(np.asarray(energy, dtype=np.float64)),
            "force_sha256": ENGINE.sha256_array(force),
        }
    )
    return energy, force


def _evaluate_one(state: dict, row: dict, output: Path, backend, source_wal, dense_wal) -> tuple[dict, Path]:
    target = _shard_path(output, row["optimization_N_F"])
    source_wal.append(
        {
            "event": "paid_source_reused",
            "utc": utc_now(),
            "row_index": row["row_index"],
            "label": row["label"],
            "optimization_N_F": row["optimization_N_F"],
            "path_sha256": row["path_sha256"],
            "receipt": str(row["receipt_path"]),
            "receipt_sha256": row["paid_force_receipt"]["sha256"],
            "paid_internal_joint_EF_values": N_INTERNAL,
            "new_physical_calls": 0,
        }
    )
    dense, geometry = ENGINE.uniform_arc500(row["path"])
    dense_wal.append(
        {
            "event": "path_stage_started",
            "utc": utc_now(),
            "stage": "dense_arc500_joint_EF",
            "row_index": row["row_index"],
            "label": row["label"],
            "optimization_N_F": row["optimization_N_F"],
            "path_sha256": row["path_sha256"],
            "dense_coordinate_sha256": ENGINE.sha256_array(dense),
            "number_of_points": N_DENSE,
        }
    )
    energy = np.empty(N_DENSE, dtype=np.float64)
    force = np.empty((N_DENSE, DIMENSION), dtype=np.float64)
    for point, coordinate in enumerate(dense):
        energy[point], force[point] = _attempt(dense_wal, backend, row, point, coordinate)
    delta = float((np.max(energy) - energy[0]) * ENGINE.EV_TO_KCAL)
    dense_wal.append(
        {
            "event": "path_stage_data_complete_uncommitted",
            "utc": utc_now(),
            "stage": "dense_arc500_joint_EF",
            "row_index": row["row_index"],
            "label": row["label"],
            "optimization_N_F": row["optimization_N_F"],
            "path_sha256": row["path_sha256"],
            "joint_attempts": N_DENSE,
            "joint_successes": N_DENSE,
            "dense_coordinate_sha256": ENGINE.sha256_array(dense),
            "dense_energy_sha256": ENGINE.sha256_array(energy),
            "dense_force_sha256": ENGINE.sha256_array(force),
            "densified_Delta_E_kcal_per_mol": delta,
        }
    )
    arrays = {
        "schema": np.asarray(SHARD_SCHEMA),
        "status": np.asarray("PASS"),
        "method": np.asarray("ODE-NEB"),
        "label": np.asarray(row["label"]),
        "optimization_N_F": np.asarray(row["optimization_N_F"], dtype=np.int64),
        "path_identifier_json": np.asarray(json.dumps(row["path_identifier"], sort_keys=True)),
        "contract_sha256": np.asarray(state["contract_sha256"]),
        "evaluator_sha256": np.asarray(ENGINE.sha256_file(Path(__file__))),
        "joint_engine_sha256": np.asarray(ENGINE.sha256_file(ENGINE_PATH)),
        "ode_validator_sha256": np.asarray(ENGINE.sha256_file(ODE_HELPER_PATH)),
        "bb_core_sha256": np.asarray(ENGINE.sha256_file(ROOT / "src/bb_core.py")),
        "topology_sha256": np.asarray(ENGINE.sha256_file(state["topology"])),
        "selected_path": row["path"],
        "selected_path_sha256": np.asarray(row["path_sha256"]),
        "official_path32_sha256": np.asarray(row["official_path32_sha256"]),
        "paid_force_receipt": np.asarray(str(row["receipt_path"])),
        "paid_force_receipt_sha256": np.asarray(row["paid_force_receipt"]["sha256"]),
        "paid_internal_energy_eV": row["receipt"]["energies"],
        "paid_internal_force_eV_per_A": row["receipt"]["forces"],
        "paid_internal_energy_sha256": np.asarray(row["paid_internal_energy_sha256"]),
        "paid_internal_force_sha256": np.asarray(row["paid_internal_force_sha256"]),
        "discrete_I_perp": np.asarray(row["discrete_I_perp"], dtype=np.float64),
        "dense_coordinates_A": dense,
        "dense_energy_eV": energy,
        "dense_force_eV_per_A": force,
        "dense_coordinate_sha256": np.asarray(ENGINE.sha256_array(dense)),
        "dense_energy_sha256": np.asarray(ENGINE.sha256_array(energy)),
        "dense_force_sha256": np.asarray(ENGINE.sha256_array(force)),
        "densified_Delta_E_kcal_per_mol": np.asarray(delta, dtype=np.float64),
        "total_arclength_A": np.asarray(geometry["total_arclength_A"]),
        "dense_spacing_A": np.asarray(geometry["dense_spacing_A"]),
        "number_of_points": np.asarray(N_DENSE, dtype=np.int64),
        "original_discrete_new_calls": np.asarray(0, dtype=np.int64),
        "paid_internal_joint_EF_values_reused": np.asarray(N_INTERNAL, dtype=np.int64),
        "dense_joint_EF_attempts": np.asarray(N_DENSE, dtype=np.int64),
        "dense_full_force_vectors_stored": np.asarray(N_DENSE, dtype=np.int64),
        "dense_endpoint_joint_EF_attempts": np.asarray(2, dtype=np.int64),
        "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
    }
    ENGINE._atomic_save_npz_no_overwrite(target, arrays)
    shard = _validate_shard(target, row, state)
    published = {
        "event": "complete_shard_published",
        "utc": utc_now(),
        "row_index": row["row_index"],
        "label": row["label"],
        "optimization_N_F": row["optimization_N_F"],
        "path_sha256": row["path_sha256"],
        "shard": str(target),
        "shard_sha256": ENGINE.sha256_file(target),
    }
    dense_wal.append(published)
    source_wal.append(published)
    return shard, target


def _publish_final(state: dict, output: Path, shards: list[dict], paths: list[Path], source_wal_path: Path, dense_wal_path: Path, held_locks: list[str], fresh: list[int], reused: list[int]) -> None:
    arrays = {
        "schema": np.asarray(RESULT_SCHEMA),
        "status": np.asarray("PASS"),
        "method": np.asarray("ODE-NEB"),
        "authoritative_data_unit": np.asarray("complete_atomic_path_shard"),
        "labels": np.asarray([str(scalar(shard, "label")) for shard in shards]),
        "optimization_N_F": np.asarray(NF_VALUES, dtype=np.int64),
        "selected_path_sha256": np.asarray([str(scalar(shard, "selected_path_sha256")) for shard in shards]),
        "official_path32_sha256": np.asarray([str(scalar(shard, "official_path32_sha256")) for shard in shards]),
        "discrete_I_perp": np.asarray([float(scalar(shard, "discrete_I_perp")) for shard in shards]),
        "energy_eV": np.asarray([shard["dense_energy_eV"] for shard in shards]),
        "energy_sha256": np.asarray([str(scalar(shard, "dense_energy_sha256")) for shard in shards]),
        "dense_coordinate_sha256": np.asarray([str(scalar(shard, "dense_coordinate_sha256")) for shard in shards]),
        "dense_force_sha256": np.asarray([str(scalar(shard, "dense_force_sha256")) for shard in shards]),
        "densified_Delta_E_kcal_per_mol": np.asarray([float(scalar(shard, "densified_Delta_E_kcal_per_mol")) for shard in shards]),
        "shard_path": np.asarray([str(path) for path in paths]),
        "shard_sha256": np.asarray([ENGINE.sha256_file(path) for path in paths]),
        "number_of_points": np.asarray(N_DENSE, dtype=np.int64),
        "original_discrete_new_calls": np.asarray(0, dtype=np.int64),
        "dense_joint_EF_attempts": np.asarray(MAXIMUM_ATTEMPT_STARTED, dtype=np.int64),
        "dense_full_force_vectors_stored": np.asarray(MAXIMUM_ATTEMPT_STARTED, dtype=np.int64),
        "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
        "contract": np.asarray(str(state["contract_path"])),
        "contract_sha256": np.asarray(state["contract_sha256"]),
    }
    ENGINE._atomic_save_npz_no_overwrite(output, arrays)
    dense_events = _read_wal(dense_wal_path, DENSE_WAL_SCHEMA, state)
    source_events = _read_wal(source_wal_path, SOURCE_WAL_SCHEMA, state)
    lifetime = _lifetime(dense_events)
    require(lifetime == {"attempt_started": MAXIMUM_ATTEMPT_STARTED, "attempt_succeeded": MAXIMUM_ATTEMPT_STARTED, "attempt_failed": 0, "attempt_open": 0, "retry_attempts": 0}, "final dense WAL accounting mismatch")
    for row, shard, path in zip(state["rows"], shards, paths):
        _validate_dense_closure(dense_events, row, shard, path)
        _validate_source_closure(source_events, row, path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "PASS",
        "created_utc": utc_now(),
        "contract": bind(state["contract_path"]),
        "aggregate": bind(output),
        "shards": [
            {
                "optimization_N_F": row["optimization_N_F"],
                "path": str(path),
                "sha256": ENGINE.sha256_file(path),
                "selected_path_sha256": row["path_sha256"],
                "paid_internal_energy_sha256": str(scalar(shard, "paid_internal_energy_sha256")),
                "paid_internal_force_sha256": str(scalar(shard, "paid_internal_force_sha256")),
                "dense_coordinate_sha256": str(scalar(shard, "dense_coordinate_sha256")),
                "dense_energy_sha256": str(scalar(shard, "dense_energy_sha256")),
                "dense_force_sha256": str(scalar(shard, "dense_force_sha256")),
            }
            for row, shard, path in zip(state["rows"], shards, paths)
        ],
    }
    ENGINE._atomic_write_json_no_overwrite(output.with_suffix(".manifest.json"), manifest)
    audit = {
        "schema": AUDIT_SCHEMA,
        "status": "PASS",
        "completed_utc": utc_now(),
        "contract": bind(state["contract_path"]),
        "output": bind(output),
        "manifest": bind(output.with_suffix(".manifest.json")),
        "paid_source_wal": bind(source_wal_path),
        "dense_joint_ef_wal": bind(dense_wal_path),
        "held_host_locks": held_locks,
        "fresh_paths_this_invocation": fresh,
        "complete_validated_shards_reused": reused,
        "path_count": 4,
        "paid_internal_joint_EF_values_reused": 4 * N_INTERNAL,
        "original_discrete_new_calls": 0,
        "dense_joint_EF_attempt_lifetime": lifetime,
        "dense_full_force_vectors_stored": MAXIMUM_ATTEMPT_STARTED,
        "optimization_N_F_contribution": 0,
        "early_four_artifacts_touched": 0,
    }
    ENGINE._atomic_write_json_no_overwrite(output.with_suffix(".audit.json"), audit)


def run(state: dict, output: Path) -> None:
    report = preflight(state, output, probe_locks=False)
    source_wal_path, dense_wal_path = _wal_paths(output)
    source_wal = ENGINE.DurableWal(source_wal_path, SOURCE_WAL_SCHEMA, state["contract_path"], state["contract_sha256"])
    dense_wal = ENGINE.DurableWal(dense_wal_path, DENSE_WAL_SCHEMA, state["contract_path"], state["contract_sha256"])
    shards, paths, fresh, reused = [], [], [], []
    backend = None
    try:
        with ENGINE.acquire_two_locks(state["locks"]) as held_locks:
            for row in state["rows"]:
                target = _shard_path(output, row["optimization_N_F"])
                if target.exists():
                    shard = _validate_shard(target, row, state)
                    reused.append(row["optimization_N_F"])
                else:
                    if backend is None:
                        backend = ENGINE.OpenMMBackend(
                            state["topology"],
                            platform_preference=("CPU",),
                            precision="double",
                            label="bb_ode_neb_late_four_arc500_joint_ef_r1",
                        )
                    before = int(backend.raw_joint_attempts)
                    shard, target = _evaluate_one(state, row, output, backend, source_wal, dense_wal)
                    require(int(backend.raw_joint_attempts) - before == N_DENSE, "backend attempt ledger mismatch")
                    fresh.append(row["optimization_N_F"])
                shards.append(shard)
                paths.append(target)
            _publish_final(state, output, shards, paths, source_wal_path, dense_wal_path, held_locks, fresh, reused)
    finally:
        source_wal.close()
        dense_wal.close()
    require(report["fresh_dense_joint_EF_attempts_planned"] == len(fresh) * N_DENSE, "preflight/run fresh count mismatch")


def check_thread_caps() -> None:
    wrong = {name: os.environ.get(name) for name in THREAD_CAPS if os.environ.get(name) != "1"}
    require(not wrong, f"all six thread caps must equal 1: {wrong}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host-role", choices=("local",), default="local")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-lock-probe-for-test", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    check_thread_caps()
    state = load_contract(args.contract, args.host_role)
    output = args.output.expanduser().resolve()
    require(output.suffix == ".npz", "--output must end in .npz")
    if args.preflight_only:
        report = preflight(
            state, output, probe_locks=not args.skip_lock_probe_for_test
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    require(not args.skip_lock_probe_for_test, "test-only lock bypass cannot execute")
    run(state, output)


if __name__ == "__main__":
    main()
