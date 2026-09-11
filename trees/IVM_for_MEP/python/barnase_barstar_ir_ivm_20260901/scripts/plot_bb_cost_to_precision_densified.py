#!/usr/bin/env python3
"""Plot the audited densified BB IVM versus ODE--NEB cost-to-precision figure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import plot_bb_cost_to_precision_final as FINAL
from bb_core import path_diagnostics
from evaluate_discrete_iperp_densified_barrier import uniform_arclength_resample
from plot_bb_cost_to_precision_preview import IVM_COLOR, NEB_COLOR, _force_formatter, _frontier, _setup_style


EXPECTED_STEM = "barnase_barstar_ivm_vs_ode_neb_cost_to_precision_densified"
CONTRACT_SCHEMA = "bb-densified-cost-to-precision-figure-contract-v1"
CONTRACT_STATUS = "FROZEN_DENSIFIED_INPUT_BINDINGS"
DENSE_AUDIT_SCHEMA = "bb-dense-four-path-dual-metric-audit-v1"
N_DENSE = 500
EXPECTED_NF = {"IVM": (60, 150, 300, 1329), "ODE-NEB": (60, 150, 300, 1320)}
DIRECT_PLOT_DEPENDENCIES = (
    Path(__file__).resolve().parent / "plot_bb_cost_to_precision_final.py",
    Path(__file__).resolve().parent / "plot_bb_cost_to_precision_preview.py",
    Path(__file__).resolve().parent / "evaluate_discrete_iperp_densified_barrier.py",
    Path(__file__).resolve().parent.parent / "src" / "bb_core.py",
)
TIMING_CORRECTION_SHA256 = "39998097ea221de11da491b5880e95e6293267e2a0a3dd33386516c19f303ac1"
NF30_RECEIPT_SHA256 = "afcdddd09457d74d20b3c6fd885b7eb0b92f96333003890a876c127223c2f4f6"
NF30_OFFICIAL_SHA256 = "8fadf37e320c936deeac7d7d24bfd8a9b789916d156c9ecf185b55fc6dbe8ae7"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def scalar(saved, key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def file_binding(path: Path):
    path = path.expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": FINAL.sha256_file(path)}


def official_path32_sha256(value: np.ndarray) -> str:
    import hashlib
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    metadata = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def validate_timing_correction(path: Path):
    correction_path = path.expanduser().resolve(strict=True)
    require(FINAL.sha256_file(correction_path) == TIMING_CORRECTION_SHA256, "timing correction R2 SHA mismatch")
    correction = read_json(correction_path)
    require(correction.get("schema") == "bb-irivm-ode-neb-cost-densification-selection-correction-v2", "timing correction schema mismatch")
    require(correction.get("status") == "R1_SUPERSEDED_TIMING_CORRECTION_NO_NEW_SELECTION", "timing correction status mismatch")
    require(correction.get("prospective_selection_authority") is False, "timing correction improperly claims authority")
    require(correction.get("created_after_new_metric_calls_started") is True, "timing correction chronology mismatch")
    superseded = correction["correction"]["superseded_file"]
    require(str(superseded["sha256"]) == "556661d6c899e7b8858fc455325478d8ab85c025dca219aedc20f72c31904647", "superseded R1 identity mismatch")
    authorities = correction.get("authoritative_pre_metric_materials", {})
    require(len(authorities) == 4, "timing correction must identify four earlier authorities")
    bound = {}
    for role, record in authorities.items():
        authority = Path(record["path"]).expanduser().resolve(strict=True)
        require(FINAL.sha256_file(authority) == str(record["sha256"]), f"timing authority drift: {role}")
        require(authority != Path(superseded["path"]).expanduser().resolve(), "superseded R1 used as authority")
        bound[role] = file_binding(authority)
    return {"correction_R2": file_binding(correction_path), "authoritative_pre_metric_materials": bound, "superseded_R1_is_authority": False}


def collect_bindings(value) -> set[tuple[str, str]]:
    found = set()
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            path = Path(value["path"]).expanduser().resolve(strict=True)
            require(FINAL.sha256_file(path) == value["sha256"], f"binding drift: {path}")
            found.add((str(path), str(value["sha256"])))
        for child in value.values():
            found.update(collect_bindings(child))
    elif isinstance(value, list):
        for child in value:
            found.update(collect_bindings(child))
    return found


def validate_figure_contract(path: Path, output: Path, inputs: Sequence[Path]):
    contract_path = path.expanduser().resolve(strict=True)
    contract = read_json(contract_path)
    require(contract.get("schema") == CONTRACT_SCHEMA, "figure contract schema mismatch")
    require(contract.get("status") == CONTRACT_STATUS, "figure contract is not frozen")
    require(Path(str(contract["figure_output"])).expanduser().resolve() == output, "figure output/contract mismatch")
    policy = contract.get("claim_policy", {})
    require(policy.get("main_curve") == "observed lower envelope", "main curve is not the observed lower envelope")
    require(policy.get("show_all_raw_points_faintly") is True, "raw-point display policy missing")
    require(policy.get("Delta_E_monotonicity_assumed") is False, "contract assumes Delta E monotonicity")
    require(policy.get("interpolation") == "forbidden", "contract permits interpolation")
    require(policy.get("all_cost_ratios") == "observed only", "contract permits an exact cost-ratio claim")
    require(policy.get("both_methods_have_incomplete_earlier_path_coverage") is True, "contract lacks two-method sparsity guard")
    bindings = collect_bindings(contract)
    for raw in inputs:
        item = raw.expanduser().resolve(strict=True)
        require((str(item), FINAL.sha256_file(item)) in bindings, f"contract does not bind {item}")
    this_plotter = Path(__file__).resolve(strict=True)
    require((str(this_plotter), FINAL.sha256_file(this_plotter)) in bindings, "contract does not bind this plotter")
    for dependency in DIRECT_PLOT_DEPENDENCIES:
        dependency = dependency.resolve(strict=True)
        require(
            (str(dependency), FINAL.sha256_file(dependency)) in bindings,
            f"contract does not bind direct plot dependency: {dependency.name}",
        )
    timing = contract.get("bindings", {}).get("timing_correction_R2")
    require(isinstance(timing, dict) and str(timing.get("sha256")) == TIMING_CORRECTION_SHA256, "contract lacks timing correction R2")
    require("superseded_R1" not in contract.get("bindings", {}), "superseded R1 may not be a figure authority")
    correction = read_json(Path(timing["path"]).expanduser().resolve(strict=True))
    contract_authorities = contract.get("bindings", {}).get("authoritative_pre_metric_materials", {})
    require(isinstance(contract_authorities, dict) and len(contract_authorities) == 4, "figure contract lacks four timing authorities")
    for role, record in correction["authoritative_pre_metric_materials"].items():
        bound = contract_authorities.get(role)
        require(isinstance(bound, dict) and bound.get("path") == record["path"] and bound.get("sha256") == record["sha256"], f"figure contract timing authority mismatch: {role}")
    nf30 = contract.get("bindings", {}).get("NF30_common_initial_reuse_authority", {})
    require(
        nf30.get("official_path32_sha256") == NF30_OFFICIAL_SHA256
        and int(nf30.get("optimization_N_F", -1)) == 30
        and nf30.get("paid_force_receipt", {}).get("sha256") == NF30_RECEIPT_SHA256,
        "figure contract lacks the exact NF30 reuse authority",
    )
    require(isinstance(nf30.get("topology"), dict) and isinstance(nf30.get("bb_core"), dict) and "System XML" in str(nf30.get("system_identity_enforcement")), "figure contract lacks NF30 topology/system enforcement")
    return {"contract": file_binding(contract_path), "claim_policy": policy}


def validate_wals(
    audit, method: str, contract_path: Path, paths: np.ndarray,
    path_hashes: np.ndarray, dense_paths: Sequence[np.ndarray],
    dense_hashes: np.ndarray, energies: np.ndarray,
    energy_hashes: np.ndarray, forces: np.ndarray,
    force_hashes: np.ndarray, joint_energies: np.ndarray,
    joint_energy_hashes: np.ndarray,
):
    row_count = len(paths)
    force_wal = Path(str(audit["force_wal"])).expanduser().resolve(strict=True)
    energy_wal = Path(str(audit["energy_wal"])).expanduser().resolve(strict=True)
    require(FINAL.sha256_file(force_wal) == str(audit["force_wal_sha256"]), "force WAL hash mismatch")
    require(FINAL.sha256_file(energy_wal) == str(audit["energy_wal_sha256"]), "energy WAL hash mismatch")
    force_events = [json.loads(line) for line in force_wal.read_text(encoding="utf-8").splitlines() if line]
    energy_events = [json.loads(line) for line in energy_wal.read_text(encoding="utf-8").splitlines() if line]
    require(force_events[0].get("schema") == "bb-dense-four-path-force-wal-v1", "force WAL schema mismatch")
    require(energy_events[0].get("schema") == "bb-dense-four-path-energy-wal-v1", "energy WAL schema mismatch")
    for header in (force_events[0], energy_events[0]):
        require(
            Path(str(header["contract"])).expanduser().resolve() == contract_path
            and str(header["contract_sha256"]) == FINAL.sha256_file(contract_path)
            and int(header["optimization_N_F_contribution"]) == 0,
            "diagnostic WAL header/contract mismatch",
        )
    require(not any(event.get("event") == "attempt_failed" for event in force_events + energy_events), "diagnostic WAL contains failure")
    force_started = [event for event in force_events if event.get("event") == "attempt_started"]
    force_success = [event for event in force_events if event.get("event") == "attempt_succeeded"]
    expected_force = 30 * row_count if method == "ODE-NEB" else 0
    require(len(force_started) == len(force_success) == expected_force, "force WAL attempt count mismatch")
    if method == "ODE-NEB":
        starts = {(int(event["row_index"]), int(event["internal_image_index"]), int(event["attempt"])): event for event in force_started}
        successes = {(int(event["row_index"]), int(event["internal_image_index"]), int(event["attempt"])): event for event in force_success}
        require(starts.keys() == successes.keys() and sorted(key[2] for key in starts) == list(range(1, expected_force + 1)), "force WAL attempt pairing/sequence mismatch")
        for (row, image, _attempt), start in starts.items():
            require(0 <= row < row_count and 1 <= image <= 30, "force WAL row/image out of range")
            success = successes[(row, image, _attempt)]
            coordinate_sha = FINAL.sha256_array(paths[row, image])
            require(start["path_sha256"] == success["path_sha256"] == path_hashes[row], "force WAL path SHA mismatch")
            require(start["coordinate_sha256"] == success["coordinate_sha256"] == coordinate_sha, "force WAL coordinate SHA mismatch")
            require(success["force_sha256"] == FINAL.sha256_array(forces[row, image - 1]), "force WAL force SHA mismatch")
            require(float(success["energy_eV"]) == float(joint_energies[row, image - 1]), "force WAL joint energy mismatch")
        require(all(FINAL.sha256_array(forces[index]) == force_hashes[index] for index in range(row_count)), "stored force aggregate SHA mismatch")
        require(all(FINAL.sha256_array(joint_energies[index]) == joint_energy_hashes[index] for index in range(row_count)), "stored joint-energy aggregate SHA mismatch")
    else:
        reused_force = [event for event in force_events if event.get("event") == "online_paid_history_reuse"]
        require(len(reused_force) == row_count, "IVM force-history reuse WAL mismatch")
        for row, event in enumerate(reused_force):
            require(int(event["row_index"]) == row and event["path_sha256"] == path_hashes[row] and int(event["diagnostic_force_attempts"]) == 0, "IVM force-history WAL identity mismatch")
    energy_started = [event for event in energy_events if event.get("event") == "attempt_started"]
    energy_success = [event for event in energy_events if event.get("event") == "attempt_succeeded"]
    reuse_events = [event for event in energy_events if event.get("event") == "exact_PASS_reuse"]
    require(len(energy_started) == len(energy_success), "energy WAL start/success mismatch")
    require(len(energy_started) + N_DENSE * len(reuse_events) == N_DENSE * row_count, "energy WAL lineage count mismatch")
    starts = {(int(event["row_index"]), int(event["point_index"]), int(event["attempt"])): event for event in energy_started}
    successes = {(int(event["row_index"]), int(event["point_index"]), int(event["attempt"])): event for event in energy_success}
    require(starts.keys() == successes.keys() and sorted(key[2] for key in starts) == list(range(1, len(starts) + 1)), "energy WAL attempt pairing/sequence mismatch")
    fresh_rows = set()
    for (row, point, _attempt), start in starts.items():
        require(0 <= row < row_count and 0 <= point < N_DENSE, "energy WAL row/point out of range")
        success = successes[(row, point, _attempt)]
        fresh_rows.add(row)
        coordinate_sha = FINAL.sha256_array(dense_paths[row][point])
        require(start["path_sha256"] == success["path_sha256"] == path_hashes[row], "energy WAL path SHA mismatch")
        require(start["coordinate_sha256"] == success["coordinate_sha256"] == coordinate_sha, "energy WAL coordinate SHA mismatch")
        require(float(success["energy_eV"]) == float(energies[row, point]), "energy WAL value mismatch")
    for row in fresh_rows:
        require(sum(key[0] == row for key in starts) == N_DENSE, "fresh energy row is incomplete")
    reused_rows = set()
    for event in reuse_events:
        row = int(event["row_index"])
        require(0 <= row < row_count and row not in fresh_rows and row not in reused_rows, "invalid/repeated energy reuse row")
        reused_rows.add(row)
        require(event["path_sha256"] == path_hashes[row] and event["dense_coordinate_sha256"] == dense_hashes[row] and event["energy_sha256"] == energy_hashes[row], "energy reuse lineage mismatch")
        source_result = Path(event["source_result"]).expanduser().resolve(strict=True)
        source_audit = Path(event["source_audit"]).expanduser().resolve(strict=True)
        require(FINAL.sha256_file(source_result) == event["source_result_sha256"] and FINAL.sha256_file(source_audit) == event["source_audit_sha256"], "energy reuse source hash mismatch")
    require(fresh_rows.union(reused_rows) == set(range(row_count)), "energy WAL does not cover every row")
    return {"force_wal": file_binding(force_wal), "energy_wal": file_binding(energy_wal), "fresh_force_attempts": len(force_started), "fresh_energy_attempts": len(energy_started), "exact_reused_energy_paths": len(reuse_events)}


def load_dense_four(result_raw: Path, audit_raw: Path, contract_raw: Path, method: str):
    result = result_raw.expanduser().resolve(strict=True)
    audit_path = audit_raw.expanduser().resolve(strict=True)
    contract_path = contract_raw.expanduser().resolve(strict=True)
    audit = read_json(audit_path)
    contract = read_json(contract_path)
    require(audit.get("schema") == DENSE_AUDIT_SCHEMA and audit.get("status") == "PASS", f"{method} dense audit did not pass")
    require(audit.get("method") == method, f"{method} dense audit method mismatch")
    require(Path(str(audit["output"])).expanduser().resolve() == result and audit["output_sha256"] == FINAL.sha256_file(result), f"{method} dense output binding mismatch")
    require(Path(str(audit["contract"])).expanduser().resolve() == contract_path and audit["contract_sha256"] == FINAL.sha256_file(contract_path), f"{method} dense contract binding mismatch")
    require(int(audit["path_count"]) == 4 and int(audit["optimization_N_F_contribution"]) == 0, f"{method} dense audit count/NoF mismatch")
    with np.load(result, allow_pickle=False) as saved:
        required = {"method", "labels", "path_identifier_json", "optimization_N_F", "selected_path", "selected_path_sha256", "discrete_I_perp", "densified_Delta_E_kcal_per_mol", "internal_diagnostic_force_eV_per_A", "internal_diagnostic_force_sha256", "internal_joint_energy_eV", "internal_joint_energy_sha256", "diagnostic_force_attempts_per_path", "fixed_endpoint_diagnostic_force_attempts_per_path", "dense_coordinate_sha256", "energy_eV", "energy_sha256", "diagnostic_energy_attempts_lineage_per_path", "diagnostic_energy_attempts_fresh_per_path", "number_of_points", "coordinate_space", "sampling", "interpolation", "grid_formula", "optimization_N_F_contribution", "contract_sha256", "topology_sha256", "evaluator_sha256", "bb_core_sha256"}
        require(required.issubset(saved.files), f"{method} dense result fields missing")
        require(str(scalar(saved, "method")) == method, f"{method} result method mismatch")
        require(int(scalar(saved, "number_of_points")) == N_DENSE and int(scalar(saved, "optimization_N_F_contribution")) == 0, f"{method} metric protocol mismatch")
        for key, expected in (("coordinate_space", "unweighted_3N_cartesian"), ("sampling", "uniform_cartesian_arclength"), ("interpolation", "piecewise_linear"), ("grid_formula", "s_j=j*L/499,j=0,...,499")):
            require(str(scalar(saved, key)) == expected, f"{method} result protocol mismatch: {key}")
        require(str(scalar(saved, "contract_sha256")) == FINAL.sha256_file(contract_path), f"{method} result contract hash mismatch")
        labels = np.asarray(saved["labels"]).astype(str)
        identifiers = [json.loads(value) for value in np.asarray(saved["path_identifier_json"]).astype(str)]
        nf = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        paths = np.asarray(saved["selected_path"], dtype=np.float64)
        path_hashes = np.asarray(saved["selected_path_sha256"]).astype(str)
        i_perp = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
        delta_e = np.asarray(saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64)
        force_hashes = np.asarray(saved["internal_diagnostic_force_sha256"]).astype(str)
        forces = np.asarray(saved["internal_diagnostic_force_eV_per_A"], dtype=np.float64)
        joint_energies = np.asarray(saved["internal_joint_energy_eV"], dtype=np.float64)
        joint_energy_hashes = np.asarray(saved["internal_joint_energy_sha256"]).astype(str)
        force_attempts = np.asarray(saved["diagnostic_force_attempts_per_path"], dtype=np.int64)
        endpoint_force = np.asarray(saved["fixed_endpoint_diagnostic_force_attempts_per_path"], dtype=np.int64)
        dense_hashes = np.asarray(saved["dense_coordinate_sha256"]).astype(str)
        energies = np.asarray(saved["energy_eV"], dtype=np.float64)
        energy_hashes = np.asarray(saved["energy_sha256"]).astype(str)
        energy_lineage = np.asarray(saved["diagnostic_energy_attempts_lineage_per_path"], dtype=np.int64)
        fresh_energy = np.asarray(saved["diagnostic_energy_attempts_fresh_per_path"], dtype=np.int64)
        result_topology_sha = str(scalar(saved, "topology_sha256"))
        result_bb_core_sha = str(scalar(saved, "bb_core_sha256"))
        result_evaluator_sha = str(scalar(saved, "evaluator_sha256"))
    require(tuple(int(value) for value in nf) == EXPECTED_NF[method], f"{method} dense N_F sequence mismatch")
    require(labels.shape == nf.shape and len(identifiers) == 4, f"{method} labels/identifier shape mismatch")
    require(paths.shape == (4, 32, 9396) and energies.shape == (4, N_DENSE), f"{method} path/energy shape mismatch")
    require(np.all(np.isfinite(paths)) and np.all(np.isfinite(energies)) and np.all(np.isfinite(i_perp)) and np.all(i_perp >= 0) and np.all(np.isfinite(delta_e)), f"{method} non-finite physical data")
    require(all(FINAL.sha256_array(path) == path_hashes[index] for index, path in enumerate(paths)), f"{method} path hash mismatch")
    dense_paths = []
    for index, path in enumerate(paths):
        dense, _geometry = uniform_arclength_resample(path, N_DENSE)
        require(FINAL.sha256_array(dense) == dense_hashes[index], f"{method} reconstructed arc500 dense SHA mismatch at row {index}")
        dense_paths.append(dense)
    require(all(FINAL.sha256_array(row) == energy_hashes[index] for index, row in enumerate(energies)), f"{method} energy hash mismatch")
    recomputed_delta = (np.max(energies, axis=1) - energies[:, 0]) * 23.0605
    require(np.array_equal(recomputed_delta, delta_e), f"{method} Delta E is not exactly reconstructed from arc500 energy")
    expected_force = 30 if method == "ODE-NEB" else 0
    require(np.all(force_attempts == expected_force) and np.all(endpoint_force == 0), f"{method} force accounting mismatch")
    require(np.all(energy_lineage == N_DENSE) and np.all((fresh_energy == 0) | (fresh_energy == N_DENSE)), f"{method} energy lineage mismatch")
    if method == "ODE-NEB":
        require(forces.shape == (4, 30, 9396) and joint_energies.shape == (4, 30), "ODE-NEB force/joint-energy shape mismatch")
        require(np.all(np.isfinite(forces)) and np.all(np.isfinite(joint_energies)), "ODE-NEB force/joint-energy non-finite")
        require(all(FINAL.sha256_array(row) == force_hashes[index] for index, row in enumerate(forces)), "ODE-NEB force SHA mismatch")
        require(all(FINAL.sha256_array(row) == joint_energy_hashes[index] for index, row in enumerate(joint_energies)), "ODE-NEB joint-energy SHA mismatch")
        for index in range(4):
            reconstructed = float(path_diagnostics(paths[index], np.zeros(32), forces[index])["i_perp"])
            require(reconstructed == float(i_perp[index]), f"ODE-NEB stored force does not reconstruct I_perp at row {index}")
    else:
        require(np.all(force_hashes == "") and np.all(joint_energy_hashes == "") and np.all(np.isnan(forces)) and np.all(np.isnan(joint_energies)), "IVM diagnostic force/joint-energy storage mismatch")
    records = audit.get("paths")
    require(isinstance(records, list) and len(records) == 4, f"{method} audit path records missing")
    for index, record in enumerate(records):
        require(int(record["optimization_N_F"]) == int(nf[index]) and record["path_sha256"] == path_hashes[index] and record["dense_coordinate_sha256"] == dense_hashes[index] and record["energy_sha256"] == energy_hashes[index], f"{method} audit/result row mismatch")
        require(float(record["discrete_I_perp"]) == float(i_perp[index]) and float(record["densified_Delta_E_kcal_per_mol"]) == float(delta_e[index]), f"{method} audit metric mismatch")
        require(int(record["diagnostic_force_attempts"]) == expected_force and int(record["diagnostic_force_successes"]) == expected_force and int(record["fixed_endpoint_diagnostic_force_attempts"]) == 0, f"{method} audit force accounting mismatch")
        require(str(record["force_sha256"]) == force_hashes[index] and str(record["joint_energy_sha256"]) == joint_energy_hashes[index], f"{method} audit force/joint-energy hash mismatch")
        require(int(record["diagnostic_energy_attempts_lineage"]) == 500 and int(record["diagnostic_energy_attempts_fresh"]) == int(fresh_energy[index]) and int(record["diagnostic_energy_attempts_reused"]) == 500 - int(fresh_energy[index]), f"{method} audit energy lineage mismatch")
        require(int(record["optimization_N_F_contribution"]) == 0, f"{method} audit contaminates optimization N_F")
    topology_binding = contract.get("topology", {})
    topology = Path(str(topology_binding.get("path", ""))).expanduser().resolve(strict=True)
    require(FINAL.sha256_file(topology) == str(topology_binding.get("sha256")) == result_topology_sha, f"{method} topology binding mismatch")
    bb_core_binding = contract.get("bb_core", {})
    bb_core_path = Path(str(bb_core_binding.get("path", ""))).expanduser().resolve(strict=True)
    require(FINAL.sha256_file(bb_core_path) == str(bb_core_binding.get("sha256")) == result_bb_core_sha, f"{method} bb_core binding mismatch")
    evaluator_record = contract.get("evaluator", contract.get("evaluator_engine", {}))
    evaluator_path = Path(str(evaluator_record.get("path", ""))).expanduser().resolve(strict=True)
    require(FINAL.sha256_file(evaluator_path) == str(evaluator_record.get("sha256")) == result_evaluator_sha, f"{method} evaluator binding mismatch")
    contract_paths = contract.get("paths")
    require(isinstance(contract_paths, list) and len(contract_paths) == 4, f"{method} dense contract path inventory missing")
    for index, record in enumerate(contract_paths):
        require(record.get("method") == method and str(record["label"]) == labels[index] and record["path_identifier"] == identifiers[index], f"{method} label/path identifier mismatch at row {index}")
        require(int(record["optimization_N_F"]) == int(nf[index]) and str(record["path_sha256"]) == str(path_hashes[index]), f"{method} dense contract/result identity mismatch at row {index}")
        source = Path(str(record["source"]["path"])).expanduser().resolve(strict=True)
        require(FINAL.sha256_file(source) == str(record["source"]["sha256"]), f"{method} source archive drift at row {index}")
        if method == "ODE-NEB":
            with np.load(source, allow_pickle=False) as archived:
                source_path = np.concatenate((archived["endpoint0_A"][None], archived["coordinates30_A"], archived["endpoint1_A"][None]), axis=0)
                embedded_official = str(np.asarray(archived["path32_sha256"]).item())
            require(np.array_equal(source_path.reshape(32, 9396), paths[index]), f"ODE-NEB source/path mismatch at row {index}")
            require(official_path32_sha256(source_path) == embedded_official == str(record["official_path32_sha256"]), f"ODE-NEB official metadata hash mismatch at row {index}")
    if method == "IVM":
        verification_record = contract.get("replay_verification", {})
        verification_path = Path(str(verification_record.get("path", ""))).expanduser().resolve(strict=True)
        require(FINAL.sha256_file(verification_path) == str(verification_record.get("sha256")), "IVM replay verification binding drift")
        verification = read_json(verification_path)
        require(verification.get("status") == "PASS_EXACT_OUTPUT_ONLY_PREFIX", "IVM replay verification not PASS")
        verified_rows = verification.get("new_reporting_rows")
        require(isinstance(verified_rows, list) and len(verified_rows) == 4, "IVM replay PASS rows missing")
        for index, verified in enumerate(verified_rows):
            require(int(verified["optimization_N_F"]) == int(nf[index]) and str(verified["selected_path_sha256"]) == path_hashes[index] and float(verified["discrete_I_perp_eV2_per_A"]) == float(i_perp[index]) and verified.get("fully_evaluated_current_path") is True and verified.get("terminal_tail") is False, f"IVM replay verification/result mismatch at row {index}")
    wal = validate_wals(audit, method, contract_path, paths, path_hashes, dense_paths, dense_hashes, energies, energy_hashes, forces, force_hashes, joint_energies, joint_energy_hashes)
    data = {
        "number_of_forces": nf,
        "i_perp": i_perp,
        "delta_e": delta_e,
        "selected_path_sha256": path_hashes,
    }
    if method == "ODE-NEB":
        official = []
        for index, record in enumerate(contract_paths):
            official_hash = str(record.get("official_path32_sha256", ""))
            require(len(official_hash) == 64, f"ODE-NEB dense contract lacks official path hash at row {index}")
            official.append(official_hash)
        data["official_path32_sha256"] = np.asarray(official)
    order = np.argsort(nf, kind="stable")
    data = {key: (value if key == "official_paid_states" else np.asarray(value)[order]) for key, value in data.items()}
    return data, {"result": file_binding(result), "audit": file_binding(audit_path), "contract": file_binding(contract_path), "topology": file_binding(topology), "bb_core": file_binding(bb_core_path), "evaluator": file_binding(evaluator_path), **wal}


def concatenate_unique(old, new, method: str):
    old_nf = np.asarray(old["number_of_forces"], dtype=np.int64)
    new_nf = np.asarray(new["number_of_forces"], dtype=np.int64)
    require(not set(old_nf).intersection(set(new_nf)), f"{method} old/new N_F overlap")
    order = np.argsort(np.concatenate((old_nf, new_nf)), kind="stable")
    result = {}
    for key in ("number_of_forces", "i_perp", "delta_e"):
        result[key] = np.concatenate((np.asarray(old[key]), np.asarray(new[key])))[order]
    for key in ("official_path32_sha256", "selected_path_sha256"):
        if key in old or key in new:
            left = np.asarray(old.get(key, np.full(len(old_nf), "", dtype="U64"))).astype(str)
            right = np.asarray(new.get(key, np.full(len(new_nf), "", dtype="U64"))).astype(str)
            result[key] = np.concatenate((left, right))[order]
    if "official_paid_states" in old:
        result["official_paid_states"] = old["official_paid_states"]
    if "targeted_selections" in old:
        result["targeted_selections"] = old["targeted_selections"]
    return result


def add_exact_common_initial_to_ode(ivm, ode, ivm_arc500_result: Path, ivm_terminal_result: Path):
    nf = np.asarray(ivm["number_of_forces"], dtype=np.int64)
    index = np.flatnonzero(nf == 30)
    require(index.size == 1, "IVM common initial NF30 row absent/nonunique")
    i = int(index[0])
    with np.load(ivm_terminal_result, allow_pickle=False) as saved:
        initial = np.ascontiguousarray(saved["initial_path"], dtype=np.float64)
    require(initial.shape == (32, 9396), "IVM initial path shape mismatch")
    with np.load(ivm_arc500_result, allow_pickle=False) as saved:
        arc_nf = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        arc_hashes = np.asarray(saved["selected_path_sha256"]).astype(str)
        arc_i = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
        arc_delta = np.asarray(saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64)
        arc_attempts = np.asarray(saved["diagnostic_energy_attempts_per_path"], dtype=np.int64)
        arc_forces = np.asarray(saved["diagnostic_force_requests_per_path"], dtype=np.int64)
        require(int(scalar(saved, "number_of_points")) == 500, "IVM common-initial result is not arc500")
        require(str(scalar(saved, "coordinate_space")) == "unweighted_3N_cartesian" and str(scalar(saved, "sampling")) == "uniform_cartesian_arclength" and str(scalar(saved, "grid_formula")) == "s_j=j*L/499,j=0,...,499", "IVM common-initial arc500 protocol mismatch")
    arc_index = np.flatnonzero(arc_nf == 30)
    require(arc_index.size == 1 and FINAL.sha256_array(initial) == arc_hashes[int(arc_index[0])], "IVM NF30 metric/path hash mismatch")
    ai = int(arc_index[0])
    require(int(arc_attempts[ai]) == 500 and int(arc_forces[ai]) == 0 and float(arc_i[ai]) == float(ivm["i_perp"][i]) and float(arc_delta[ai]) == float(ivm["delta_e"][i]), "IVM NF30 metric/lineage mismatch")
    paid_states = ode.get("official_paid_states")
    require(isinstance(paid_states, list), "complete ODE retained-path inventory required for NF30 reuse")
    records = [row for row in paid_states if int(row["optimization_N_F"]) == 30]
    require(len(records) == 1, "ODE NF30 retained path absent/nonunique")
    record = records[0]
    archive = Path(record["coordinate_archive"]).expanduser().resolve(strict=True)
    require(FINAL.sha256_file(archive) == str(record["coordinate_archive_sha256"]), "ODE NF30 archive hash drift")
    with np.load(archive, allow_pickle=False) as saved:
        ode_initial = np.concatenate((saved["endpoint0_A"][None], saved["coordinates30_A"], saved["endpoint1_A"][None]), axis=0).reshape(32, 9396)
    require(np.array_equal(initial, ode_initial), "ODE NF30 is not byte-identical to the IVM common initial path")
    require(official_path32_sha256(ode_initial.reshape(32, 3132, 3)) == NF30_OFFICIAL_SHA256 == str(record["official_path32_sha256"]), "ODE NF30 official metadata hash mismatch")
    selections = ode.get("targeted_selections")
    require(isinstance(selections, list) and selections, "targeted R2 selection binding required for NF30 reuse")
    selection_path = Path(selections[-1]["path"]).expanduser().resolve(strict=True)
    selection = read_json(selection_path)
    protocol = selection.get("diagnostic_protocol_for_each_selected_path", {})
    require(protocol == {"coordinate_space": "unweighted_3N_cartesian", "energy_evaluations_per_distinct_path": 500, "exact_hash_reuse_only": True, "force_evaluations_per_path": 0, "grid_formula": "s_j=j*L/499,j=0,...,499", "include_both_endpoints": True, "interpolation": "piecewise_linear", "number_of_points": 500, "optimization_N_F_contribution": 0, "sampling": "uniform_cartesian_arclength"}, "targeted R2 arc500 protocol mismatch")
    receipts = [item for item in selection["bindings"]["paid_force_receipts"] if int(item["optimization_N_F"]) == 30]
    require(len(receipts) == 1, "NF30 paid force receipt absent/nonunique")
    receipt_record = receipts[0]
    receipt = Path(receipt_record["path"]).expanduser().resolve(strict=True)
    require(FINAL.sha256_file(receipt) == str(receipt_record["sha256"]) == NF30_RECEIPT_SHA256, "NF30 paid receipt SHA mismatch")
    require(int(receipt_record["paid_batch_id"]) == 1 and str(receipt_record["official_path32_sha256"]) == NF30_OFFICIAL_SHA256, "NF30 receipt ledger identity mismatch")
    with np.load(receipt, allow_pickle=False) as saved:
        receipt_coordinates = np.asarray(saved["coordinates30_A"], dtype=np.float64)
        receipt_forces = np.asarray(saved["forces_eV_per_A"], dtype=np.float64)
        receipt_energies = np.asarray(saved["energies_eV"], dtype=np.float64)
        receipt_nf = int(np.asarray(saved["nf_after"]).item())
        receipt_batch = int(np.asarray(saved["batch_id"]).item())
    require(receipt_coordinates.shape == receipt_forces.shape == (30, 3132, 3) and receipt_energies.shape == (30,), "NF30 receipt array shape mismatch")
    require(np.all(np.isfinite(receipt_forces)) and np.all(np.isfinite(receipt_energies)) and np.array_equal(receipt_coordinates, ode_initial[1:-1].reshape(30, 3132, 3)), "NF30 receipt coordinate/physical data mismatch")
    require(receipt_nf == 30 and receipt_batch == 1, "NF30 receipt accounting mismatch")
    receipt_i = float(path_diagnostics(initial, np.zeros(32), receipt_forces.reshape(30, 9396))["i_perp"])
    require(receipt_i == float(receipt_record["discrete_I_perp_eV2_per_A"]) == float(ivm["i_perp"][i]), "NF30 receipt/ledger/IVM Iperp mismatch")
    optimization_audit_record = selection["bindings"]["optimization_audit"]
    optimization_audit_path = Path(optimization_audit_record["path"]).expanduser().resolve(strict=True)
    require(FINAL.sha256_file(optimization_audit_path) == str(optimization_audit_record["sha256"]), "ODE optimization audit drift")
    optimization_audit = read_json(optimization_audit_path)
    require(int(optimization_audit["endpoint_physical_calls"]) == 0, "ODE fixed endpoints incurred physical calls")
    require(30 not in set(np.asarray(ode["number_of_forces"], dtype=np.int64)), "ODE inputs already contain NF30")
    ode["number_of_forces"] = np.concatenate((np.asarray([30], dtype=np.int64), np.asarray(ode["number_of_forces"], dtype=np.int64)))
    ode["i_perp"] = np.concatenate((np.asarray([ivm["i_perp"][i]]), np.asarray(ode["i_perp"])))
    ode["delta_e"] = np.concatenate((np.asarray([ivm["delta_e"][i]]), np.asarray(ode["delta_e"])))
    ode["official_path32_sha256"] = np.concatenate((np.asarray([str(record["official_path32_sha256"])]), np.asarray(ode["official_path32_sha256"]).astype(str)))
    return {"status": "PASS_BYTE_IDENTICAL_COMMON_INITIAL_RECEIPT_AND_PROTOCOL_REUSE", "optimization_N_F": 30, "IVM_initial_path_sha256": FINAL.sha256_array(initial), "ODE_coordinate_archive": file_binding(archive), "ODE_paid_force_receipt": file_binding(receipt), "ODE_optimization_audit": file_binding(optimization_audit_path), "official_path32_sha256": str(record["official_path32_sha256"]), "reconstructed_discrete_I_perp": receipt_i, "reused_discrete_I_perp": float(ivm["i_perp"][i]), "reused_densified_Delta_E_kcal_per_mol": float(ivm["delta_e"][i]), "topology_and_system_policy": "same frozen topology and bb_core-enforced OpenMM system identity as both dense-four contracts", "arc500_protocol": protocol, "fixed_endpoint_optimization_N_F": 0, "additional_physical_calls": 0}


def render(ivm, ode, analysis, output: Path):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    _setup_style(mpl)
    figure = plt.figure(figsize=(180 / 25.4, 72 / 25.4), dpi=220)
    grid = GridSpec(1, 2, figure=figure, left=0.072, right=0.935, bottom=0.18, top=0.78, wspace=0.25)
    axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    for axis, key, xlabel, xscale in ((axes[0], "i_perp", r"$I_\perp$", "log"), (axes[1], "delta_e", r"$\Delta E$ (kcal mol$^{-1}$)", "linear")):
        axis.scatter(ivm[key], ivm["number_of_forces"], s=10, facecolors="none", edgecolors=IVM_COLOR, linewidths=0.45, alpha=0.28, zorder=3)
        axis.scatter(ode[key], ode["number_of_forces"], s=10, marker="s", facecolors="none", edgecolors=NEB_COLOR, linewidths=0.45, alpha=0.25, zorder=2)
        ivm_x, ivm_y = _frontier(ivm["number_of_forces"], ivm[key])
        ode_x, ode_y = _frontier(ode["number_of_forces"], ode[key])
        axis.plot(ivm_x, ivm_y, color=IVM_COLOR, linestyle="-", linewidth=1.85, marker="o", markersize=3.3, markerfacecolor="white", markeredgewidth=0.6, zorder=20)
        axis.plot(ode_x, ode_y, color=NEB_COLOR, linestyle="--", linewidth=1.65, marker="s", markersize=3.1, markerfacecolor="white", markeredgewidth=0.6, zorder=6)
        passage = analysis["first_passage_number_of_forces"][key]
        FINAL._add_arrow(axis, analysis["thresholds"][key], passage["IVM"], passage["ODE-NEB"], analysis["ODE_NEB_over_IVM"][key], analysis["ODE_NEB_passage_claim"][key]["claim"])
        all_x = np.concatenate((np.asarray(ivm[key]), np.asarray(ode[key])))
        if xscale == "log":
            axis.set_xscale("log")
            axis.set_xlim(float(np.max(all_x)) * 1.45, float(np.min(all_x)) / 1.45)
        else:
            span = max(float(np.ptp(all_x)), 1.0)
            axis.set_xlim(float(np.max(all_x)) + 0.06 * span, max(0.0, float(np.min(all_x)) - 0.06 * span))
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Number of Forces")
        axis.grid(True, which="major", color="#D9D9D6", linewidth=0.4, alpha=0.62)
        axis.yaxis.set_major_formatter(FuncFormatter(_force_formatter))
    ymax = max(11000.0, 1.08 * max(np.max(ivm["number_of_forces"]), np.max(ode["number_of_forces"])))
    for axis in axes:
        axis.set_ylim(0.0, ymax)
    figure.canvas.draw()
    for axis, letter in zip(axes, "ab"):
        position = axis.get_position()
        figure.text(position.x0 - 0.03, position.y1 + 0.012, letter, ha="left", va="bottom", fontsize=8.0, fontweight="bold")
    handles = [Line2D([0], [0], label="IVM", color=IVM_COLOR, linestyle="-", linewidth=1.85, marker="o", markersize=3.3, markerfacecolor="white", markeredgewidth=0.6), Line2D([0], [0], label="ODE-NEB", color=NEB_COLOR, linestyle="--", linewidth=1.65, marker="s", markersize=3.1, markerfacecolor="white", markeredgewidth=0.6)]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, columnspacing=1.5, handlelength=2.35, handletextpad=0.5, frameon=False)
    figure.savefig(output, format="pdf", dpi=600, metadata={"Title": "Barnase--Barstar densified IVM versus ODE-NEB cost to precision", "Subject": "Raw observations and observed lower envelopes for discrete I_perp and Cartesian arc500 Delta E"})
    plt.close(figure)


def analyze_observed_only(ivm, ode):
    analysis = FINAL.analyze(ivm, ode)
    analysis["both_method_coverage_guard"] = {
        "IVM": "incomplete earlier-path coverage",
        "ODE-NEB": "incomplete earlier distinct-retained-path coverage",
        "exact_first_passage_claim_authorized": False,
    }
    for key in ("i_perp", "delta_e"):
        analysis["ODE_NEB_passage_claim"][key]["claim"] = "earliest_observed_first_passage"
        analysis["ODE_NEB_passage_claim"][key]["two_method_incomplete_coverage_override"] = True
        analysis["ODE_NEB_over_IVM_interpretation"][key] = "observed-passage cost ratio only; neither method has complete earlier-path coverage"
    analysis["all_cost_ratios"] = "observed only"
    return analysis


def main(argv: Iterable[str] | None = None) -> None:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--ivm-result", type=Path, required=True)
    parser.add_argument("--ivm-audit", type=Path, required=True)
    parser.add_argument("--ivm-terminal-result", type=Path, required=True)
    parser.add_argument("--ivm-terminal-audit", type=Path, required=True)
    parser.add_argument("--ivm-long-verification", type=Path, required=True)
    parser.add_argument("--ivm-prefix-verification", type=Path, required=True)
    parser.add_argument("--ivm-dense-result", type=Path, required=True)
    parser.add_argument("--ivm-dense-audit", type=Path, required=True)
    parser.add_argument("--ivm-dense-contract", type=Path, required=True)
    parser.add_argument("--ode-result", type=Path, action="append", required=True)
    parser.add_argument("--ode-audit", type=Path, action="append", required=True)
    parser.add_argument("--ode-target-selection", type=Path)
    parser.add_argument("--ode-targeted-bundle", nargs=3, type=Path, action="append", default=[])
    parser.add_argument("--ode-dense-result", type=Path, required=True)
    parser.add_argument("--ode-dense-audit", type=Path, required=True)
    parser.add_argument("--ode-dense-contract", type=Path, required=True)
    parser.add_argument("--timing-correction", type=Path, required=True)
    parser.add_argument("--figure-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv_list)
    output = args.output.expanduser().resolve()
    require(output.suffix.lower() == ".pdf" and output.stem == EXPECTED_STEM, "output must use the registered new PDF stem")
    provenance_path = output.with_suffix(".json")
    require(not output.exists() and not provenance_path.exists(), "refusing to overwrite densified figure/provenance")
    required_inputs = [args.ivm_result, args.ivm_audit, args.ivm_terminal_result, args.ivm_terminal_audit, args.ivm_long_verification, args.ivm_prefix_verification, args.ivm_dense_result, args.ivm_dense_audit, args.ivm_dense_contract, *args.ode_result, *args.ode_audit, args.ode_dense_result, args.ode_dense_audit, args.ode_dense_contract, args.timing_correction]
    if args.ode_target_selection is not None:
        required_inputs.append(args.ode_target_selection)
    for bundle in args.ode_targeted_bundle:
        required_inputs.extend(bundle)
    figure_contract = validate_figure_contract(args.figure_contract, output, required_inputs)
    timing_correction = validate_timing_correction(args.timing_correction)
    ivm_old, ivm_old_bindings = FINAL.load_selected_ivm(args.ivm_result, args.ivm_audit, args.ivm_terminal_result, args.ivm_terminal_audit, args.ivm_long_verification, args.ivm_prefix_verification)
    ode_old, ode_old_bindings = FINAL.load_official_ode(args.ode_result, args.ode_audit, args.ode_target_selection, [tuple(bundle) for bundle in args.ode_targeted_bundle])
    require(len(ivm_old["number_of_forces"]) == 5, "existing IVM input must contain exactly five observations")
    require(len(ode_old["number_of_forces"]) == 7, "existing ODE-NEB input must contain exactly seven observations")
    ivm_new, ivm_new_binding = load_dense_four(args.ivm_dense_result, args.ivm_dense_audit, args.ivm_dense_contract, "IVM")
    ode_new, ode_new_binding = load_dense_four(args.ode_dense_result, args.ode_dense_audit, args.ode_dense_contract, "ODE-NEB")
    paid_states = ode_old.get("official_paid_states")
    require(isinstance(paid_states, list), "full official ODE-NEB retained-path ledger required")
    first_by_nf = {}
    for state in paid_states:
        first_by_nf.setdefault(int(state["optimization_N_F"]), state)
    for index, nf_value in enumerate(np.asarray(ode_new["number_of_forces"], dtype=np.int64)):
        state = first_by_nf.get(int(nf_value))
        require(state is not None and str(state["official_path32_sha256"]) == str(ode_new["official_path32_sha256"][index]), f"new ODE-NEB NF{int(nf_value)} is not the official retained path")
    require(ivm_new_binding["topology"] == ode_new_binding["topology"], "IVM/ODE-NEB dense-four topology mismatch")
    require(ivm_new_binding["bb_core"] == ode_new_binding["bb_core"], "IVM/ODE-NEB dense-four OpenMM system-enforcement source mismatch")
    with np.load(args.ivm_result.expanduser().resolve(strict=True), allow_pickle=False) as saved:
        old_ivm_topology_sha = str(scalar(saved, "topology_sha256"))
        for key, expected in (("coordinate_space", "unweighted_3N_cartesian"), ("sampling", "uniform_cartesian_arclength"), ("interpolation", "piecewise_linear"), ("grid_formula", "s_j=j*L/499,j=0,...,499")):
            require(str(scalar(saved, key)) == expected, f"existing IVM protocol mismatch: {key}")
    require(old_ivm_topology_sha == ivm_new_binding["topology"]["sha256"], "existing/new IVM topology mismatch")
    ivm = concatenate_unique(ivm_old, ivm_new, "IVM")
    ode = concatenate_unique(ode_old, ode_new, "ODE-NEB")
    nf30_reuse = add_exact_common_initial_to_ode(ivm, ode, args.ivm_result.expanduser().resolve(strict=True), args.ivm_terminal_result.expanduser().resolve(strict=True))
    analysis = analyze_observed_only(ivm, ode)
    provenance = {"schema": "bb-densified-cost-to-precision-provenance-v1", "status": "AUDITED_DENSIFIED_EVIDENCE", "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "figure": str(output), "command_argv": [sys.executable, str(Path(__file__).resolve()), *argv_list], "figure_contract": figure_contract, "timing_correction": timing_correction, "legend_labels": ["IVM", "ODE-NEB"], "vertical_axis": "Number of Forces", "main_curve": "observed lower envelope", "raw_points": "all audited observations shown faintly", "interpolation": "none", "Delta_E_monotonicity_assumed": False, "same_path_two_metrics": True, "arc_length_points": 500, "all_cost_ratios": "observed only", "diagnostic_calls_excluded_from_optimization_axis": True, "NF30_common_initial_reuse": nf30_reuse, "IVM_existing_bindings": ivm_old_bindings, "IVM_dense_four_binding": ivm_new_binding, "ODE_NEB_existing_bindings": ode_old_bindings, "ODE_NEB_dense_four_binding": ode_new_binding, "plotter": file_binding(Path(__file__).resolve()), **analysis}
    if args.validate_only:
        print(json.dumps(provenance, indent=2, sort_keys=True))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output.parent), prefix=".bb-densified-plot-") as temporary_dir:
        temporary_pdf = Path(temporary_dir) / output.name
        temporary_json = Path(temporary_dir) / provenance_path.name
        render(ivm, ode, analysis, temporary_pdf)
        provenance["figure_sha256"] = FINAL.sha256_file(temporary_pdf)
        temporary_json.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.link(temporary_json, provenance_path)
        os.link(temporary_pdf, output)
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
