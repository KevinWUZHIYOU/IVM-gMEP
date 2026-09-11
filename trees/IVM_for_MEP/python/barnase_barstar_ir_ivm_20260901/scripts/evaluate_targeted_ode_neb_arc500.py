#!/usr/bin/env python3
"""Evaluate only paths frozen by a targeted ODE--NEB arc500 selection.

The selector is deliberately not an execution authorization.  Physical
evaluation additionally requires a post-selection execution contract that
binds the immutable selection, topology, this evaluator, output pathname, and
both host-local Barnase--Barstar locks.  ``--validate-only`` runs every
preflight check without importing OpenMM, acquiring a lock, or writing an
artifact.

For each *distinct retained* 32-image path in ``paths_to_densify`` this entry
point evaluates exactly 500 potential energies on the common piecewise-linear,
uniform-Cartesian-arclength grid.  It never requests force and contributes zero
to the optimization Number of Forces ``N_F``.  A live append-only energy WAL
precedes the physical calls; the terminal NPZ and JSON audit are published
atomically and with no-overwrite semantics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

import bb_core
import evaluate_discrete_iperp_densified_barrier as common_arc500
import freeze_targeted_ode_neb_arc500 as target_freezer
from evaluate_discrete_iperp_densified_barrier import (
    COORDINATE_SPACE,
    DIMENSION,
    GRID_FORMULA,
    INTERPOLATION,
    N_DENSE_POINTS,
    N_IMAGES,
    SAMPLING,
    acquire_host_locks,
    sha256_array,
    sha256_file,
    uniform_arclength_resample,
)


SELECTION_SCHEMA = "bb-ode-neb-targeted-arc500-selection-v1"
EXECUTION_SCHEMA = "bb-ode-neb-targeted-arc500-execution-contract-v1"
AUDIT_SCHEMA = "bb-ode-neb-targeted-arc500-audit-v1"
PREFLIGHT_SCHEMA = "bb-ode-neb-targeted-arc500-zero-openmm-preflight-v1"
SELECTED_IVM_AUDIT_SCHEMA = (
    "bb-irivm-discrete-iperp-densified-barrier-audit-v1"
)
SELECTED_IVM_CONTRACT_SCHEMA = (
    "bb-irivm-discrete-iperp-densified-barrier-contract-v1"
)
SELECTED_IVM_LABEL = "bb_irivm_a1e0_r01_m05_inf9990_r1"
COMMON_IVM_LABEL = "common_initial_path"
EXPECTED_PAID_STATES = 333
EXPECTED_DISTINCT_RETAINED_PATHS = 284
EXPECTED_ACCEPTED_TRIALS = 283
EXPECTED_REJECTED_TRIALS = 49
CANONICAL_HOST_LOCKS = (
    Path(common_arc500.DEFAULT_CANONICAL_LOCK).expanduser().resolve(),
    Path(common_arc500.DEFAULT_DOWNLOADS_LOCK).expanduser().resolve(),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} does not contain a JSON object")
    return value


def scalar(saved: np.lib.npyio.NpzFile, key: str):
    require(key in saved.files, f"NPZ lacks {key}")
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def resolve_bound_path(value: object) -> Path:
    require(isinstance(value, str) and value, "bound path is empty")
    return Path(value).expanduser().resolve(strict=True)


def resolve_relative_to(reference: Path, value: object) -> Path:
    require(isinstance(value, str) and value, "relative binding path is empty")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = reference.parent / candidate
    return candidate.resolve(strict=True)


def validate_file_binding(record: object, role: str) -> Path:
    require(isinstance(record, dict), f"missing {role} binding")
    path = resolve_bound_path(record.get("path"))
    expected = str(record.get("sha256", ""))
    require(len(expected) == 64, f"{role} binding lacks SHA-256")
    require(sha256_file(path) == expected, f"{role} binding drift")
    return path


def validate_pair_binding(record: object, role: str) -> tuple[Path, Path]:
    require(isinstance(record, dict), f"missing {role} binding")
    result = resolve_bound_path(record.get("result"))
    audit = resolve_bound_path(record.get("audit"))
    require(
        sha256_file(result) == str(record.get("result_sha256", "")),
        f"{role} result binding drift",
    )
    require(
        sha256_file(audit) == str(record.get("audit_sha256", "")),
        f"{role} audit binding drift",
    )
    return result, audit


def validate_protocol(selection: Mapping[str, object]) -> None:
    require(
        selection.get("claim_mode") == "observed",
        "this energy-only evaluator accepts only observed-mode selections with exact paid-force I_perp",
    )
    protocol = selection.get("diagnostic_protocol_for_each_selected_path")
    require(isinstance(protocol, dict), "selection lacks diagnostic protocol")
    expected = {
        "coordinate_space": COORDINATE_SPACE,
        "interpolation": INTERPOLATION,
        "sampling": SAMPLING,
        "number_of_points": N_DENSE_POINTS,
        "include_both_endpoints": True,
        "grid_formula": GRID_FORMULA,
        "energy_evaluations_per_distinct_path": N_DENSE_POINTS,
        "force_evaluations_per_path": 0,
        "optimization_N_F_contribution": 0,
        "exact_hash_reuse_only": True,
    }
    for key, value in expected.items():
        require(protocol.get(key) == value, f"selection protocol mismatch: {key}")
    require(
        selection.get("physical_calls_made_by_selector")
        == {"energy": 0, "force": 0},
        "selector physical-call ledger is not exactly zero",
    )
    require(
        int(selection.get("optimization_N_F_contribution", -1)) == 0,
        "selector changed optimization N_F",
    )
    require(
        selection.get("execution_authorized_by_this_file") is False,
        "selection ambiguously claims execution authority",
    )
    guard = selection.get("claim_guard")
    require(isinstance(guard, dict), "selection lacks claim guard")
    require(
        guard.get("Delta_E_monotonicity_assumed") is False,
        "selection assumes Delta E monotonicity",
    )
    require(
        guard.get("unmeasured_earlier_paths_forbid_exact_first_passage_claim")
        is True,
        "selection lacks sparse first-passage guard",
    )


def _binding_identity(record: Mapping[str, object]) -> tuple[str, str]:
    return str(resolve_bound_path(record.get("path"))), str(record.get("sha256"))


def validate_selection_bindings(
    selection: Mapping[str, object],
) -> Dict[str, object]:
    """Validate every explicit immutable binding in the selection receipt."""

    bindings = selection.get("bindings")
    require(isinstance(bindings, dict), "selection lacks bindings")
    official_paths: Dict[str, Path] = {}
    for role in (
        "optimization_complete",
        "optimization_audit",
        "optimization_wal",
        "frozen_endpoints",
        "selector",
    ):
        official_paths[role] = validate_file_binding(bindings.get(role), role)
    require(
        official_paths["selector"]
        == Path(target_freezer.__file__).resolve(strict=True),
        "selection is not bound to the targeted selector implementation",
    )

    selected_result, selected_audit = validate_pair_binding(
        bindings.get("selected_IR_IVM_arc500"), "selected IR-IVM arc500"
    )

    force_receipts: list[Path] = []
    force_identities: list[tuple[str, str]] = []
    for group_name in ("paid_force_receipts", "excluded_paid_force_receipts"):
        group = bindings.get(group_name)
        require(isinstance(group, list), f"{group_name} binding is not a list")
        for index, record in enumerate(group):
            path = validate_file_binding(record, f"{group_name}[{index}]")
            force_receipts.append(path)
            force_identities.append((str(path), str(record["sha256"])))
    require(
        len(set(force_identities)) == len(force_identities),
        "selection repeats a paid-force receipt binding",
    )

    completed_records = bindings.get("completed_ODE_NEB_arc500")
    require(
        isinstance(completed_records, list) and completed_records,
        "selection lacks completed ODE-NEB arc500 bindings",
    )
    completed_pairs = [
        validate_pair_binding(record, f"completed ODE-NEB arc500[{index}]")
        for index, record in enumerate(completed_records)
    ]
    require(
        len(set(completed_pairs)) == len(completed_pairs),
        "selection repeats a completed ODE-NEB arc500 binding",
    )
    return {
        "official_paths": official_paths,
        "selected_result": selected_result,
        "selected_audit": selected_audit,
        "force_receipts": force_receipts,
        "completed_pairs": completed_pairs,
    }


def validate_selected_ivm_binding(
    selection: Mapping[str, object], result: Path, audit_path: Path
) -> dict:
    """Prove that the selection thresholds came from the final selected run."""

    audit = read_json(audit_path)
    require(
        audit.get("schema") == SELECTED_IVM_AUDIT_SCHEMA,
        "selected IR-IVM audit schema mismatch",
    )
    require(audit.get("status") == "PASS", "selected IR-IVM audit did not pass")
    require(
        resolve_bound_path(audit.get("output")) == result,
        "selected IR-IVM audit/output path mismatch",
    )
    require(
        audit.get("output_sha256") == sha256_file(result),
        "selected IR-IVM audit/output hash mismatch",
    )
    require(
        int(audit.get("diagnostic_force_requests", -1)) == 0,
        "selected IR-IVM diagnostic requested force",
    )
    require(
        int(audit.get("optimization_N_F_contribution", -1)) == 0,
        "selected IR-IVM diagnostic changed optimization N_F",
    )
    require(
        audit.get(
            "every_path_is_bound_to_exactly_500_successful_energy_evaluations"
        )
        is True,
        "selected IR-IVM audit lacks the 500-energy certificate",
    )
    with np.load(result, allow_pickle=False) as saved:
        required = {
            "label",
            "optimization_N_F",
            "discrete_I_perp",
            "densified_Delta_E_kcal_per_mol",
            "selected_path_sha256",
            "source_result",
            "source_result_sha256",
            "diagnostic_energy_attempts_per_path",
            "diagnostic_energy_successes_per_path",
            "diagnostic_force_requests_per_path",
            "optimization_N_F_contribution",
            "number_of_points",
            "coordinate_space",
            "interpolation",
            "sampling",
            "grid_formula",
            "contract",
            "contract_sha256",
            "history_index",
            "calls_per_image",
        }
        missing = sorted(required.difference(saved.files))
        require(not missing, f"selected IR-IVM result is missing {missing}")
        labels = np.asarray(saved["label"]).astype(str)
        nfs = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        i_values = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
        delta_values = np.asarray(
            saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64
        )
        path_hashes = np.asarray(saved["selected_path_sha256"]).astype(str)
        history_indices = np.asarray(saved["history_index"], dtype=np.int64)
        calls_per_image = np.asarray(saved["calls_per_image"], dtype=np.int64)
        source_results = np.asarray(saved["source_result"]).astype(str)
        source_hashes = np.asarray(saved["source_result_sha256"]).astype(str)
        attempts = np.asarray(
            saved["diagnostic_energy_attempts_per_path"], dtype=np.int64
        )
        successes = np.asarray(
            saved["diagnostic_energy_successes_per_path"], dtype=np.int64
        )
        forces = np.asarray(
            saved["diagnostic_force_requests_per_path"], dtype=np.int64
        )
        points = int(scalar(saved, "number_of_points"))
        optimization_contribution = int(
            scalar(saved, "optimization_N_F_contribution")
        )
        metric_contract_path = resolve_bound_path(str(scalar(saved, "contract")))
        metric_contract_sha = str(scalar(saved, "contract_sha256"))
        for key, expected in (
            ("coordinate_space", COORDINATE_SPACE),
            ("interpolation", INTERPOLATION),
            ("sampling", SAMPLING),
            ("grid_formula", GRID_FORMULA),
        ):
            require(str(scalar(saved, key)) == expected, f"selected {key} mismatch")

    require(
        sha256_file(metric_contract_path) == metric_contract_sha,
        "selected IR-IVM metric-contract hash drift",
    )
    require(
        resolve_bound_path(audit.get("contract")) == metric_contract_path
        and str(audit.get("contract_sha256")) == metric_contract_sha,
        "selected IR-IVM result/audit metric-contract binding mismatch",
    )
    metric_contract = read_json(metric_contract_path)
    require(
        metric_contract.get("schema") == SELECTED_IVM_CONTRACT_SCHEMA,
        "selected IR-IVM metric-contract schema mismatch",
    )
    require(
        metric_contract.get("contract_id")
        == "bb_irivm_selected_9990_arc500_20260902_r1",
        "selected IR-IVM metric contract is not the selected 9990 contract",
    )
    require(
        metric_contract.get("status")
        == "FROZEN_AFTER_TERMINAL_AND_EXACT_PREFIX_VERIFICATION_BEFORE_ANY_SELECTED_9990_DIAGNOSTIC_CALL",
        "selected IR-IVM metric contract was not frozen after terminal/prefix verification",
    )
    contract_runs = metric_contract.get("runs")
    require(
        isinstance(contract_runs, list)
        and len(contract_runs) == 1
        and contract_runs[0].get("label") == SELECTED_IVM_LABEL,
        "selected IR-IVM metric contract has the wrong run scope",
    )
    run = contract_runs[0]
    require(
        int(run.get("registered_optimization_N_F", -1)) == 9990,
        "selected IR-IVM metric contract does not have the 9990 N_F budget",
    )
    terminal_result = resolve_relative_to(metric_contract_path, run.get("source_result"))
    terminal_audit = resolve_relative_to(metric_contract_path, run.get("source_audit"))
    require(
        sha256_file(terminal_result) == str(run.get("source_result_sha256"))
        and sha256_file(terminal_audit) == str(run.get("source_audit_sha256")),
        "selected IR-IVM terminal result/audit binding drift",
    )
    long_verification = resolve_relative_to(
        metric_contract_path, run.get("terminal_verification")
    )
    prefix_verification = resolve_relative_to(
        metric_contract_path, run.get("exact_prefix_verification")
    )
    require(
        sha256_file(long_verification)
        == str(run.get("terminal_verification_sha256"))
        and sha256_file(prefix_verification)
        == str(run.get("exact_prefix_verification_sha256")),
        "selected IR-IVM terminal/prefix verification binding drift",
    )
    contract_bindings = metric_contract.get("bindings")
    require(isinstance(contract_bindings, dict), "selected IR-IVM metric contract lacks bindings")
    require(
        resolve_relative_to(
            metric_contract_path,
            contract_bindings.get("selected_long_member_verification", {}).get(
                "path"
            ),
        )
        == long_verification
        and str(
            contract_bindings.get("selected_long_member_verification", {}).get(
                "sha256"
            )
        )
        == sha256_file(long_verification),
        "selected metric contract long-member binding mismatch",
    )
    require(
        resolve_relative_to(
            metric_contract_path,
            contract_bindings.get("exact_prefix_verification", {}).get("path"),
        )
        == prefix_verification
        and str(
            contract_bindings.get("exact_prefix_verification", {}).get("sha256")
        )
        == sha256_file(prefix_verification),
        "selected metric contract exact-prefix binding mismatch",
    )
    long_receipt = read_json(long_verification)
    require(
        long_receipt.get("schema") == "bb-irivm-long-budget-member-verification-v2"
        and long_receipt.get("status") == "PASS"
        and long_receipt.get("label") == SELECTED_IVM_LABEL
        and int(long_receipt.get("endpoint_optimization_N_F", -1)) == 0,
        "selected IR-IVM terminal long-member verification did not pass",
    )
    require(
        validate_file_binding(
            long_receipt.get("bindings", {}).get("result"),
            "selected terminal verification result",
        )
        == terminal_result
        and validate_file_binding(
            long_receipt.get("bindings", {}).get("audit"),
            "selected terminal verification audit",
        )
        == terminal_audit,
        "selected terminal verification binds a different result/audit",
    )
    prefix_receipt = read_json(prefix_verification)
    require(
        prefix_receipt.get("schema")
        == "bb-irivm-selected-9990-exact-prefix-verification-v1"
        and prefix_receipt.get("status") == "PASS_EXACT_PREFIX"
        and prefix_receipt.get("tolerance_fallback_used") is False,
        "selected IR-IVM exact-prefix verification did not pass exactly",
    )
    require(
        validate_file_binding(
            prefix_receipt.get("bindings", {}).get("selected_result"),
            "exact-prefix selected result",
        )
        == terminal_result
        and validate_file_binding(
            prefix_receipt.get("bindings", {}).get("selected_verification"),
            "exact-prefix terminal verification",
        )
        == long_verification,
        "selected exact-prefix verification binds a different terminal lineage",
    )

    shape = labels.shape
    require(
        shape
        == nfs.shape
        == i_values.shape
        == delta_values.shape
        == path_hashes.shape
        == history_indices.shape
        == calls_per_image.shape
        == source_results.shape
        == source_hashes.shape
        == attempts.shape
        == successes.shape
        == forces.shape,
        "selected IR-IVM row arrays differ in length",
    )
    require(
        set(labels).issubset({COMMON_IVM_LABEL, SELECTED_IVM_LABEL}),
        "selected thresholds contain a candidate/preview IVM label",
    )
    selected_mask = labels == SELECTED_IVM_LABEL
    require(bool(np.any(selected_mask)), "final selected IR-IVM label is absent")
    require(
        points == N_DENSE_POINTS
        and bool(np.all(attempts == N_DENSE_POINTS))
        and bool(np.all(successes == N_DENSE_POINTS)),
        "selected IR-IVM rows do not each bind 500 successful energies",
    )
    require(
        bool(np.all(forces == 0)) and optimization_contribution == 0,
        "selected IR-IVM metric contains force or optimization N_F contamination",
    )
    require(
        bool(np.all(np.isfinite(i_values)))
        and bool(np.all(np.isfinite(delta_values))),
        "selected IR-IVM metric is non-finite",
    )
    terminal_bindings: set[tuple[str, str]] = set()
    for path_text, expected_sha in zip(
        source_results[selected_mask], source_hashes[selected_mask]
    ):
        terminal = resolve_bound_path(str(path_text))
        require(
            sha256_file(terminal) == str(expected_sha),
            "selected IR-IVM terminal result binding drift",
        )
        terminal_bindings.add((str(terminal), str(expected_sha)))
    require(
        len(terminal_bindings) == 1,
        "selected IR-IVM rows do not share one terminal result",
    )
    require(
        next(iter(terminal_bindings))[0] == str(terminal_result),
        "selected IR-IVM metric rows are not bound to the terminal verified result",
    )

    require(
        metric_contract.get("eligible_path_policy")
        == "post-initial reparameterization-committed stored paths only",
        "selected metric contract has the wrong eligible-path policy",
    )
    common_path, common_i, common_nf, eligible_rows = common_arc500.load_eligible_rows(
        metric_contract_path, metric_contract
    )
    frozen_states = run.get("eligible_states_at_freeze")
    require(
        isinstance(frozen_states, list) and len(frozen_states) == len(eligible_rows),
        "selected metric contract eligible-state inventory is incomplete",
    )
    require(
        [int(value) for value in run.get("eligible_committed_calls", [])]
        == [int(row["calls_per_image"]) for row in eligible_rows],
        "selected metric contract eligible-call inventory drift",
    )
    for index, (frozen, recovered) in enumerate(zip(frozen_states, eligible_rows)):
        for key in (
            "history_index",
            "calls_per_image",
            "optimization_N_F",
            "discrete_I_perp",
            "path_sha256",
        ):
            require(
                frozen.get(key) == recovered.get(key),
                f"selected metric contract frozen eligible state {index} drift: {key}",
            )

    expected_rows = [
        {
            "label": COMMON_IVM_LABEL,
            "history_index": 0,
            "calls_per_image": 1,
            "optimization_N_F": int(common_nf),
            "discrete_I_perp": float(common_i),
            "selected_path_sha256": sha256_array(common_path),
        },
        *[
            {
                "label": str(row["label"]),
                "history_index": int(row["history_index"]),
                "calls_per_image": int(row["calls_per_image"]),
                "optimization_N_F": int(row["optimization_N_F"]),
                "discrete_I_perp": float(row["discrete_I_perp"]),
                "selected_path_sha256": str(row["path_sha256"]),
            }
            for row in eligible_rows
        ],
    ]
    actual_rows = [
        {
            "label": str(label),
            "history_index": int(history),
            "calls_per_image": int(call),
            "optimization_N_F": int(nf),
            "discrete_I_perp": float(i_value),
            "selected_path_sha256": str(path_hash),
        }
        for label, history, call, nf, i_value, path_hash in zip(
            labels,
            history_indices,
            calls_per_image,
            nfs,
            i_values,
            path_hashes,
        )
    ]
    require(
        actual_rows == expected_rows,
        "selected IR-IVM arc500 rows are not exactly the frozen eligible path lineage",
    )
    audit_records = audit.get("records")
    require(
        isinstance(audit_records, list) and len(audit_records) == len(expected_rows),
        "selected IR-IVM audit record inventory is incomplete",
    )
    for index, (expected, record) in enumerate(zip(expected_rows, audit_records)):
        for result_key, audit_key in (
            ("label", "label"),
            ("history_index", "history_index"),
            ("optimization_N_F", "optimization_N_F"),
            ("discrete_I_perp", "discrete_I_perp"),
            ("selected_path_sha256", "selected_path_sha256"),
        ):
            require(
                record.get(audit_key) == expected[result_key],
                f"selected IR-IVM audit row {index} drift: {audit_key}",
            )
        require(
            int(record.get("diagnostic_energy_attempts", -1)) == N_DENSE_POINTS
            and int(record.get("diagnostic_force_requests", -1)) == 0,
            f"selected IR-IVM audit row {index} energy/force ledger mismatch",
        )
    row_count = len(expected_rows)
    require(
        int(audit.get("selected_path_count_including_common_initial", -1))
        == row_count
        and int(audit.get("diagnostic_energy_attempts", -1))
        == row_count * N_DENSE_POINTS
        and int(audit.get("diagnostic_energy_successes", -1))
        == row_count * N_DENSE_POINTS,
        "selected IR-IVM audit aggregate energy ledger mismatch",
    )
    reuse_preflight = metric_contract.get("exact_reuse_preflight")
    require(isinstance(reuse_preflight, dict), "selected metric contract lacks reuse preflight")
    unique_count = len({row["selected_path_sha256"] for row in expected_rows})
    require(
        int(
            reuse_preflight.get(
                "unique_selected_path_count_including_common_initial", -1
            )
        )
        == unique_count
        and int(reuse_preflight.get("exact_reused_unique_path_count", -1))
        + int(reuse_preflight.get("fresh_unique_path_count", -1))
        == unique_count
        and int(reuse_preflight.get("diagnostic_force_calls", -1)) == 0
        and int(reuse_preflight.get("optimization_N_F_contribution", -1)) == 0,
        "selected metric contract reuse/force ledger mismatch",
    )

    thresholds, _ = target_freezer.load_selected_thresholds(result, audit_path)
    require(
        selection.get("thresholds_from_selected_IR_IVM") == thresholds,
        "selection thresholds drift from the bound final selected IR-IVM result",
    )
    return {
        "selected_label": SELECTED_IVM_LABEL,
        "terminal_result": next(iter(terminal_bindings))[0],
        "thresholds": thresholds,
    }


def compare_paid_inventory(
    embedded: object, authoritative: Sequence[Mapping[str, object]]
) -> None:
    require(
        isinstance(embedded, list) and len(embedded) == EXPECTED_PAID_STATES,
        "selection must embed all 333 official paid states",
    )
    require(
        len(authoritative) == EXPECTED_PAID_STATES,
        "authoritative ODE-NEB ledger does not contain 333 paid states",
    )
    keys = (
        "paid_batch_id",
        "optimization_N_F",
        "disposition",
        "ase_current_residual_eV_per_A",
        "source_batch_id",
        "coordinate_archive",
        "coordinate_archive_sha256",
        "official_path32_sha256",
    )
    for index, (reported, actual) in enumerate(zip(embedded, authoritative), start=1):
        require(isinstance(reported, dict), f"paid state {index} is not an object")
        for key in keys:
            require(
                reported.get(key) == actual.get(key),
                f"paid state {index} authoritative mismatch: {key}",
            )
    nfs = [int(row["optimization_N_F"]) for row in embedded]
    batches = [int(row["paid_batch_id"]) for row in embedded]
    require(len(set(nfs)) == len(nfs), "paid-state N_F values are duplicated")
    require(len(set(batches)) == len(batches), "paid batch ids are duplicated")
    hashes = {str(row["official_path32_sha256"]) for row in embedded}
    require(
        len(hashes) == EXPECTED_DISTINCT_RETAINED_PATHS,
        "paid-state inventory does not contain 284 distinct retained paths",
    )
    dispositions = [str(row["disposition"]) for row in embedded]
    require(
        dispositions.count("accepted") == EXPECTED_ACCEPTED_TRIALS
        and dispositions.count("rejected") == EXPECTED_REJECTED_TRIALS
        and dispositions.count("current_initial") == 1,
        "paid-state disposition counts drift from the sealed ledger",
    )


def validate_target_records(
    selection: Mapping[str, object],
    paid_rows: Sequence[Mapping[str, object]],
    exact_i_by_hash: Mapping[str, float],
    measured_delta_by_hash: Mapping[str, float],
    exact_i_sources: Mapping[str, Mapping[str, object]],
) -> list[dict]:
    records = selection.get("paths_to_densify")
    require(isinstance(records, list) and records, "selection contains no target path")
    hashes = [str(record.get("official_path32_sha256", "")) for record in records]
    nfs = [int(record.get("optimization_N_F_first_occurrence", -1)) for record in records]
    require(len(set(hashes)) == len(hashes), "target selection repeats a path hash")
    require(len(set(nfs)) == len(nfs), "target selection repeats an optimization N_F")

    first_by_hash: dict[str, Mapping[str, object]] = {}
    for row in paid_rows:
        first_by_hash.setdefault(str(row["official_path32_sha256"]), row)

    validated: list[dict] = []
    for index, record in enumerate(records):
        require(isinstance(record, dict), f"target record {index} is not an object")
        path_hash = hashes[index]
        require(path_hash in first_by_hash, f"target path {index} is outside paid ledger")
        source = first_by_hash[path_hash]
        require(
            str(source["disposition"]) in {"current_initial", "accepted"},
            f"target path {index} first occurs as a rejected trial",
        )
        expected = {
            "optimization_N_F_first_occurrence": int(source["optimization_N_F"]),
            "paid_batch_id_first_occurrence": int(source["paid_batch_id"]),
            "source_batch_id": int(source["source_batch_id"]),
            "official_path32_sha256": path_hash,
            "coordinate_archive": str(source["coordinate_archive"]),
            "coordinate_archive_sha256": str(source["coordinate_archive_sha256"]),
        }
        for key, value in expected.items():
            require(record.get(key) == value, f"target record {index} mismatch: {key}")
        require(
            path_hash not in measured_delta_by_hash,
            f"target path {index} was already arc500-measured before selection",
        )
        require(
            path_hash in exact_i_by_hash,
            f"target path {index} lacks exact paid-force discrete I_perp",
        )
        require(
            path_hash in exact_i_sources
            and exact_i_sources[path_hash].get("kind") == "paid_force_receipt",
            f"target path {index} lacks a bound paid-force I_perp source",
        )
        exact_i = float(exact_i_by_hash[path_hash])
        require(
            math.isfinite(exact_i)
            and exact_i >= 0.0
            and record.get("exact_discrete_I_perp_eV2_per_A") == exact_i,
            f"target record {index} exact discrete I_perp mismatch",
        )
        archive = resolve_bound_path(str(record["coordinate_archive"]))
        require(
            sha256_file(archive) == str(record["coordinate_archive_sha256"]),
            f"target record {index} coordinate archive binding drift",
        )
        with np.load(archive, allow_pickle=False) as saved:
            coordinates = np.asarray(saved["coordinates30_A"], dtype=np.float64)
            endpoint0 = np.asarray(saved["endpoint0_A"], dtype=np.float64)
            endpoint1 = np.asarray(saved["endpoint1_A"], dtype=np.float64)
            embedded_hash = str(np.asarray(saved["path32_sha256"]).item())
        path3 = np.concatenate((endpoint0[None], coordinates, endpoint1[None]), axis=0)
        require(
            path3.shape == (N_IMAGES, DIMENSION // 3, 3),
            f"target record {index} path shape mismatch",
        )
        official = target_freezer.official_array_sha256(path3)
        require(
            official == embedded_hash == path_hash,
            f"target record {index} official path hash mismatch",
        )
        path = np.ascontiguousarray(path3.reshape(N_IMAGES, DIMENSION))
        dense, geometry = uniform_arclength_resample(path)
        validated.append(
            {
                "label": f"ODE-NEB_target_NF{int(source['optimization_N_F'])}",
                "optimization_N_F": int(source["optimization_N_F"]),
                "paid_batch_id": int(source["paid_batch_id"]),
                "source_batch_id": int(source["source_batch_id"]),
                "discrete_I_perp": exact_i,
                "path": path,
                "path_sha256": sha256_array(path),
                "official_path32_sha256": official,
                "source_result": str(archive),
                "source_result_sha256": sha256_file(archive),
                "exact_I_perp_source_kind": str(
                    exact_i_sources[path_hash]["kind"]
                ),
                "exact_I_perp_source": str(exact_i_sources[path_hash]["path"]),
                "exact_I_perp_source_sha256": str(
                    exact_i_sources[path_hash]["sha256"]
                ),
                "exact_I_perp_source_optimization_N_F": int(
                    exact_i_sources[path_hash]["optimization_N_F"]
                ),
                "preflight_dense_coordinate_sha256": sha256_array(dense),
                **geometry,
            }
        )
    return validated


def derive_bound_topology(completed_pairs: Sequence[tuple[Path, Path]]) -> Path:
    identities: set[tuple[str, str]] = set()
    for result, _audit in completed_pairs:
        with np.load(result, allow_pickle=False) as saved:
            topology = resolve_bound_path(str(scalar(saved, "topology")))
            topology_sha = str(scalar(saved, "topology_sha256"))
            require(
                str(scalar(saved, "coordinate_space")) == COORDINATE_SPACE,
                "completed ODE-NEB topology source coordinate-space mismatch",
            )
            require(
                str(scalar(saved, "interpolation")) == INTERPOLATION,
                "completed ODE-NEB topology source interpolation mismatch",
            )
            require(
                str(scalar(saved, "sampling")) == SAMPLING,
                "completed ODE-NEB topology source sampling mismatch",
            )
            require(
                int(scalar(saved, "number_of_points")) == N_DENSE_POINTS,
                "completed ODE-NEB topology source is not arc500",
            )
            require(
                str(scalar(saved, "grid_formula")) == GRID_FORMULA,
                "completed ODE-NEB topology source grid mismatch",
            )
        require(sha256_file(topology) == topology_sha, "bound topology hash drift")
        identities.add((str(topology), topology_sha))
    require(len(identities) == 1, "completed ODE-NEB bundles disagree on topology")
    return Path(next(iter(identities))[0])


def validate_prior_targeted_bundle(result: Path, audit_path: Path) -> tuple[dict, dict]:
    """Validate a prior targeted round before admitting it to the next round."""

    audit = read_json(audit_path)
    require(audit.get("schema") == AUDIT_SCHEMA, "prior targeted audit schema mismatch")
    require(audit.get("status") == "PASS", "prior targeted audit did not pass")
    require(
        resolve_bound_path(audit.get("output")) == result
        and str(audit.get("output_sha256")) == sha256_file(result),
        "prior targeted audit/output binding mismatch",
    )
    prior_selection = resolve_bound_path(audit.get("target_selection"))
    require(
        str(audit.get("target_selection_sha256")) == sha256_file(prior_selection),
        "prior targeted selection binding drift",
    )
    prior_selection_payload = read_json(prior_selection)
    require(
        prior_selection_payload.get("schema") == SELECTION_SCHEMA
        and prior_selection_payload.get("status")
        == "FROZEN_ZERO_OPENMM_TARGET_SELECTION"
        and prior_selection_payload.get("claim_mode") == "observed",
        "prior targeted selection is not a frozen observed-mode selection",
    )
    execution_contract = resolve_bound_path(audit.get("execution_contract"))
    require(
        str(audit.get("execution_contract_sha256"))
        == sha256_file(execution_contract),
        "prior targeted execution-contract binding drift",
    )
    contract = read_json(execution_contract)
    require(
        contract.get("schema") == EXECUTION_SCHEMA
        and contract.get("status") == "AUTHORIZED_AFTER_FROZEN_TARGET_SELECTION"
        and contract.get("physical_execution_authorized") is True,
        "prior targeted execution contract was not authorized",
    )
    require(
        resolve_bound_path(contract.get("target_selection", {}).get("path"))
        == prior_selection
        and str(contract.get("target_selection", {}).get("sha256"))
        == sha256_file(prior_selection)
        and Path(str(contract.get("output"))).expanduser().resolve() == result,
        "prior targeted execution contract lineage mismatch",
    )
    wal = resolve_bound_path(audit.get("energy_wal"))
    wal_validation = audit.get("energy_wal_terminal_validation")
    require(
        str(audit.get("energy_wal_sha256")) == sha256_file(wal)
        and isinstance(wal_validation, dict)
        and wal_validation.get("status") == "PASS"
        and wal_validation.get("sha256") == sha256_file(wal)
        and int(wal_validation.get("exact_path_energy_reuse_events", -1)) == 0
        and int(wal_validation.get("failed_energy_events", -1)) == 0,
        "prior targeted WAL did not pass the durable fresh-call audit",
    )
    require(
        int(audit.get("exact_path_energy_reuse_count", -1)) == 0
        and audit.get("current_round_allow_exact_energy_reuse") is False
        and audit.get(
            "every_discrete_I_perp_is_bound_to_an_exact_paid_force_receipt"
        )
        is True,
        "prior targeted round contains reuse or lacks exact I_perp bindings",
    )
    with np.load(result, allow_pickle=False) as saved:
        nfs = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        hashes = np.asarray(saved["official_path32_sha256"]).astype(str)
        selected_hashes = np.asarray(saved["selected_path_sha256"]).astype(str)
        dense_hashes = np.asarray(saved["dense_coordinate_sha256"]).astype(str)
        energies = np.asarray(saved["energy_eV"], dtype=np.float64)
        energy_hashes = np.asarray(saved["energy_sha256"]).astype(str)
        i_values = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
        delta_values = np.asarray(
            saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64
        )
        attempts = np.asarray(
            saved["diagnostic_energy_attempts_per_path"], dtype=np.int64
        )
        successes = np.asarray(
            saved["diagnostic_energy_successes_per_path"], dtype=np.int64
        )
        fresh_attempts = np.asarray(
            saved["diagnostic_energy_attempts_this_evaluation_per_path"],
            dtype=np.int64,
        )
        reused = np.asarray(
            saved["diagnostic_energy_reused_per_path"], dtype=bool
        )
        forces = np.asarray(
            saved["diagnostic_force_requests_per_path"], dtype=np.int64
        )
        source_kinds = np.asarray(
            saved["exact_discrete_I_perp_source_kind"]
        ).astype(str)
        source_paths = np.asarray(saved["exact_discrete_I_perp_source"]).astype(str)
        source_hashes = np.asarray(
            saved["exact_discrete_I_perp_source_sha256"]
        ).astype(str)
        source_nfs = np.asarray(
            saved["exact_discrete_I_perp_source_optimization_N_F"],
            dtype=np.int64,
        )
        prior_topology = resolve_bound_path(str(scalar(saved, "topology")))
        result_round_index = int(scalar(saved, "target_round_index"))
        result_prior_count = int(scalar(saved, "prior_targeted_bundle_count"))
        require(
            str(scalar(saved, "topology_sha256")) == sha256_file(prior_topology),
            "prior targeted topology binding drift",
        )
        require(
            str(scalar(saved, "target_selection")) == str(prior_selection)
            and str(scalar(saved, "target_selection_sha256"))
            == sha256_file(prior_selection)
            and str(scalar(saved, "execution_contract"))
            == str(execution_contract)
            and str(scalar(saved, "execution_contract_sha256"))
            == sha256_file(execution_contract)
            and str(scalar(saved, "energy_wal")) == str(wal)
            and str(scalar(saved, "energy_wal_sha256")) == sha256_file(wal),
            "prior targeted result lineage scalar mismatch",
        )
        require(
            int(scalar(saved, "number_of_points")) == N_DENSE_POINTS
            and str(scalar(saved, "coordinate_space")) == COORDINATE_SPACE
            and str(scalar(saved, "interpolation")) == INTERPOLATION
            and str(scalar(saved, "sampling")) == SAMPLING
            and str(scalar(saved, "grid_formula")) == GRID_FORMULA
            and int(scalar(saved, "optimization_N_F_contribution")) == 0
            and int(scalar(saved, "fixed_endpoint_optimization_N_F")) == 0
            and result_round_index == int(audit.get("target_round_index", -1))
            and result_prior_count
            == int(audit.get("prior_targeted_bundle_count", -1))
            and result_round_index == result_prior_count + 1,
            "prior targeted result protocol, N_F, or round scalar mismatch",
        )
        require(
            str(scalar(saved, "energy_wal_terminal_validation_schema"))
            == "bb-targeted-energy-wal-terminal-validation-v1"
            and int(scalar(saved, "energy_wal_exact_path_reuse_events")) == 0,
            "prior targeted result lacks terminal WAL/no-reuse evidence",
        )
    validate_execution_contract(
        execution_contract, prior_selection, prior_topology, result
    )
    target_records = prior_selection_payload.get("paths_to_densify")
    require(
        isinstance(target_records, list)
        and len(target_records) == len(nfs)
        and nfs.shape
        == hashes.shape
        == selected_hashes.shape
        == dense_hashes.shape
        == energy_hashes.shape
        == i_values.shape
        == delta_values.shape
        == attempts.shape
        == successes.shape
        == fresh_attempts.shape
        == reused.shape
        == forces.shape
        == source_kinds.shape
        == source_paths.shape
        == source_hashes.shape
        == source_nfs.shape,
        "prior targeted result/selection row shape mismatch",
    )
    require(
        energies.shape == (len(nfs), N_DENSE_POINTS),
        "prior targeted energy matrix shape mismatch",
    )
    expected_calls = len(nfs) * N_DENSE_POINTS
    require(
        bool(np.all(np.isfinite(i_values)))
        and bool(np.all(i_values >= 0.0))
        and bool(np.all(np.isfinite(delta_values)))
        and bool(np.all(np.isfinite(energies)))
        and bool(np.all(attempts == N_DENSE_POINTS))
        and bool(np.all(successes == N_DENSE_POINTS))
        and bool(np.all(fresh_attempts == N_DENSE_POINTS))
        and not bool(np.any(reused))
        and not bool(np.any(forces)),
        "prior targeted result contains invalid metrics or incomplete fresh-call ledgers",
    )
    require(
        int(audit.get("paid_N_F_state_count_verified", -1))
        == EXPECTED_PAID_STATES
        and int(audit.get("distinct_retained_path_count_verified", -1))
        == EXPECTED_DISTINCT_RETAINED_PATHS
        and int(audit.get("selected_distinct_retained_path_count", -1))
        == len(nfs)
        and int(audit.get("fresh_diagnostic_energy_attempts", -1))
        == expected_calls
        and int(audit.get("fresh_diagnostic_energy_successes", -1))
        == expected_calls
        and int(audit.get("lineage_energy_evaluations", -1)) == expected_calls
        and int(audit.get("diagnostic_force_requests", -1)) == 0
        and int(audit.get("optimization_N_F_contribution", -1)) == 0
        and int(audit.get("fixed_endpoint_optimization_N_F", -1)) == 0
        and audit.get(
            "every_included_path_has_exactly_500_successful_energy_evaluations"
        )
        is True
        and audit.get("every_included_path_was_freshly_evaluated_once") is True,
        "prior targeted audit fresh-call, force, N_F, or paid-ledger counts mismatch",
    )
    for index in range(len(nfs)):
        require(
            sha256_array(energies[index]) == str(energy_hashes[index]),
            f"prior targeted energy hash mismatch at row {index}",
        )
    prior_manifest = [
        {
            "row_index": index,
            "optimization_N_F": int(nfs[index]),
            "official_path32_sha256": str(hashes[index]),
            "selected_path_sha256": str(selected_hashes[index]),
            "dense_coordinate_sha256": str(dense_hashes[index]),
        }
        for index in range(len(nfs))
    ]
    replay = replay_targeted_energy_wal(
        wal,
        prior_manifest,
        [energies[index] for index in range(len(nfs))],
        {
            "target_selection": str(prior_selection),
            "target_selection_sha256": sha256_file(prior_selection),
            "execution_contract": str(execution_contract),
            "execution_contract_sha256": sha256_file(execution_contract),
            "topology": str(prior_topology),
            "topology_sha256": sha256_file(prior_topology),
            "target_manifest": prior_manifest,
            "durable_attempt_started_before_each_physical_call": True,
            "fsync_each_attempt_start_and_terminal_record": True,
        },
    )
    require(
        replay == wal_validation,
        "prior targeted WAL replay differs from its terminal validation receipt",
    )
    receipt_records = prior_selection_payload.get("bindings", {}).get(
        "paid_force_receipts"
    )
    require(
        isinstance(receipt_records, list),
        "prior targeted selection lacks paid-force receipt bindings",
    )
    receipt_identities = set()
    for receipt in receipt_records:
        require(
            isinstance(receipt, dict),
            "prior targeted paid-force receipt binding is not an object",
        )
        receipt_i = float(
            receipt.get("discrete_I_perp_eV2_per_A", float("nan"))
        )
        require(
            math.isfinite(receipt_i) and receipt_i >= 0.0,
            "prior targeted paid-force receipt lacks a finite exact I_perp",
        )
        receipt_identities.add(
            (
                str(resolve_bound_path(receipt.get("path"))),
                str(receipt.get("sha256")),
                int(receipt.get("optimization_N_F", -1)),
                str(receipt.get("official_path32_sha256", "")),
                receipt_i,
            )
        )
    require(
        len(receipt_identities) == len(receipt_records),
        "prior targeted selection repeats a paid-force receipt identity",
    )
    audit_records = audit.get("records")
    require(
        isinstance(audit_records, list) and len(audit_records) == len(nfs),
        "prior targeted audit/result row count mismatch",
    )
    for index, record in enumerate(target_records):
        require(
            int(record["optimization_N_F_first_occurrence"]) == int(nfs[index])
            and str(record["official_path32_sha256"]) == str(hashes[index])
            and float(record["exact_discrete_I_perp_eV2_per_A"])
            == float(i_values[index])
            and str(source_kinds[index]) == "paid_force_receipt",
            f"prior targeted result/selection identity mismatch at row {index}",
        )
        source = resolve_bound_path(str(source_paths[index]))
        require(
            sha256_file(source) == str(source_hashes[index]),
            f"prior targeted exact I_perp source drift at row {index}",
        )
        source_identity = (
            str(source),
            str(source_hashes[index]),
            int(source_nfs[index]),
            str(hashes[index]),
            float(i_values[index]),
        )
        require(
            source_identity in receipt_identities,
            f"prior targeted exact I_perp source is not exactly bound by the selection at row {index}",
        )
        audit_record = audit_records[index]
        require(
            int(audit_record.get("optimization_N_F", -1)) == int(nfs[index])
            and str(audit_record.get("official_path32_sha256", ""))
            == str(hashes[index])
            and str(audit_record.get("selected_path_sha256", ""))
            == str(selected_hashes[index])
            and str(audit_record.get("dense_coordinate_sha256", ""))
            == str(dense_hashes[index])
            and str(audit_record.get("energy_sha256", ""))
            == str(energy_hashes[index])
            and float(audit_record.get("discrete_I_perp", float("nan")))
            == float(i_values[index])
            and float(
                audit_record.get(
                    "densified_Delta_E_kcal_per_mol", float("nan")
                )
            )
            == float(delta_values[index])
            and int(audit_record.get("diagnostic_energy_attempts", -1))
            == N_DENSE_POINTS
            and int(audit_record.get("diagnostic_energy_successes", -1))
            == N_DENSE_POINTS
            and int(audit_record.get("diagnostic_force_requests", -1)) == 0
            and str(audit_record.get("exact_discrete_I_perp_source_kind", ""))
            == str(source_kinds[index])
            and str(
                resolve_bound_path(
                    audit_record.get("exact_discrete_I_perp_source")
                )
            )
            == str(source)
            and str(
                audit_record.get("exact_discrete_I_perp_source_sha256", "")
            )
            == str(source_hashes[index])
            and int(
                audit_record.get(
                    "exact_discrete_I_perp_source_optimization_N_F", -1
                )
            )
            == int(source_nfs[index]),
            f"prior targeted audit/result exact I_perp source mismatch at row {index}",
        )
    return prior_selection_payload, audit


def _targeted_lineage_identity(
    record: Mapping[str, object],
) -> tuple[str, str, str, str, tuple[int, ...], tuple[str, ...]]:
    """Normalize one immutable targeted-round binding for prefix comparison."""

    return (
        str(resolve_bound_path(record.get("result"))),
        str(record.get("result_sha256")),
        str(resolve_bound_path(record.get("audit"))),
        str(record.get("audit_sha256")),
        tuple(int(value) for value in record.get("optimization_N_F", [])),
        tuple(str(value) for value in record.get("official_path32_sha256", [])),
    )


def _selection_targeted_prefix(
    selection: Mapping[str, object],
) -> list[tuple[str, str, str, str, tuple[int, ...], tuple[str, ...]]]:
    """Extract and hash-check the targeted ancestry declared by a selection."""

    records = selection.get("bindings", {}).get("completed_ODE_NEB_arc500")
    require(isinstance(records, list), "prior selection lacks completed arc500 lineage")
    prefix = []
    for index, record in enumerate(records):
        require(isinstance(record, dict), "prior completed arc500 binding is not an object")
        result, audit_path = validate_pair_binding(
            record, f"prior completed ODE-NEB arc500[{index}]"
        )
        if read_json(audit_path).get("schema") != AUDIT_SCHEMA:
            continue
        identity = _targeted_lineage_identity(record)
        require(
            identity[0] == str(result) and identity[2] == str(audit_path),
            "prior targeted selection contains a non-canonical lineage path",
        )
        prefix.append(identity)
    return prefix


def _audit_targeted_prefix(
    audit: Mapping[str, object],
) -> list[tuple[str, str, str, str, tuple[int, ...], tuple[str, ...]]]:
    """Extract the targeted ancestry embedded in a prior round's audit."""

    records = audit.get("completed_prior_arc500_lineage")
    require(isinstance(records, list), "prior targeted audit lacks completed lineage")
    return [
        _targeted_lineage_identity(record)
        for record in records
        if isinstance(record, dict) and record.get("audit_schema") == AUDIT_SCHEMA
    ]


def completed_arc500_lineage(selection: Mapping[str, object]) -> tuple[list[dict], int]:
    lineage: list[dict] = []
    targeted_count = 0
    targeted_prefix: list[
        tuple[str, str, str, str, tuple[int, ...], tuple[str, ...]]
    ] = []
    current_selected_binding = selection["bindings"]["selected_IR_IVM_arc500"]
    current_paid_identity = [
        (
            int(row["paid_batch_id"]),
            int(row["optimization_N_F"]),
            str(row["official_path32_sha256"]),
        )
        for row in selection["official_paid_states"]
    ]
    for record in selection["bindings"]["completed_ODE_NEB_arc500"]:
        result = resolve_bound_path(record["result"])
        audit_path = resolve_bound_path(record["audit"])
        audit = read_json(audit_path)
        schema = str(audit.get("schema"))
        if schema == AUDIT_SCHEMA:
            prior_selection, prior_audit = validate_prior_targeted_bundle(
                result, audit_path
            )
            require(
                prior_selection.get("bindings", {}).get(
                    "selected_IR_IVM_arc500"
                )
                == current_selected_binding,
                "prior targeted round used a different selected IR-IVM threshold source",
            )
            prior_paid = prior_selection.get("official_paid_states")
            require(
                isinstance(prior_paid, list)
                and [
                    (
                        int(row["paid_batch_id"]),
                        int(row["optimization_N_F"]),
                        str(row["official_path32_sha256"]),
                    )
                    for row in prior_paid
                ]
                == current_paid_identity,
                "prior targeted round used a different official paid-state ledger",
            )
            require(
                int(prior_audit.get("target_round_index", -1))
                == targeted_count + 1
                and int(prior_audit.get("prior_targeted_bundle_count", -1))
                == targeted_count,
                "prior targeted bundles are not bound in chronological round order",
            )
            require(
                _selection_targeted_prefix(prior_selection) == targeted_prefix,
                "prior targeted selection does not contain the exact targeted ancestry prefix",
            )
            require(
                _audit_targeted_prefix(prior_audit) == targeted_prefix,
                "prior targeted audit does not contain the exact targeted ancestry prefix",
            )
            targeted_prefix.append(_targeted_lineage_identity(record))
            targeted_count += 1
        lineage.append(
            {
                "audit_schema": schema,
                "result": str(result),
                "result_sha256": str(record["result_sha256"]),
                "audit": str(audit_path),
                "audit_sha256": str(record["audit_sha256"]),
                "optimization_N_F": [
                    int(value) for value in record["optimization_N_F"]
                ],
                "official_path32_sha256": [
                    str(value) for value in record["official_path32_sha256"]
                ],
            }
        )
    require(
        len({(row["result"], row["result_sha256"]) for row in lineage})
        == len(lineage),
        "completed arc500 lineage repeats a result binding",
    )
    return lineage, targeted_count


def validate_execution_contract(
    contract_path: Path,
    selection_path: Path,
    topology: Path,
    output: Path,
) -> tuple[dict, tuple[Path, Path]]:
    contract_path = contract_path.expanduser().resolve(strict=True)
    selection_path = selection_path.expanduser().resolve(strict=True)
    topology = topology.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    contract = read_json(contract_path)
    require(contract.get("schema") == EXECUTION_SCHEMA, "execution contract schema mismatch")
    require(
        contract.get("status") == "AUTHORIZED_AFTER_FROZEN_TARGET_SELECTION",
        "execution contract is inert or not authorized",
    )
    require(
        contract.get("physical_execution_authorized") is True,
        "execution contract does not authorize physical energy evaluation",
    )
    selection_binding = contract.get("target_selection")
    require(isinstance(selection_binding, dict), "execution contract lacks selection binding")
    require(
        resolve_bound_path(selection_binding.get("path")) == selection_path
        and str(selection_binding.get("sha256")) == sha256_file(selection_path),
        "execution contract/selection binding mismatch",
    )
    topology_binding = contract.get("topology")
    require(isinstance(topology_binding, dict), "execution contract lacks topology binding")
    require(
        resolve_bound_path(topology_binding.get("path")) == topology
        and str(topology_binding.get("sha256")) == sha256_file(topology),
        "execution contract/topology binding mismatch",
    )
    evaluator_binding = contract.get("evaluator")
    evaluator = Path(__file__).resolve(strict=True)
    require(isinstance(evaluator_binding, dict), "execution contract lacks evaluator binding")
    require(
        resolve_bound_path(evaluator_binding.get("path")) == evaluator
        and str(evaluator_binding.get("sha256")) == sha256_file(evaluator),
        "execution contract/evaluator binding mismatch",
    )
    implementation_bindings = contract.get("implementation_bindings")
    require(
        isinstance(implementation_bindings, dict),
        "execution contract lacks implementation bindings",
    )
    expected_implementations = {
        "target_selector": Path(target_freezer.__file__).resolve(strict=True),
        "common_arc500_evaluator": Path(common_arc500.__file__).resolve(strict=True),
        "bb_core": Path(bb_core.__file__).resolve(strict=True),
    }
    for role, expected_path in expected_implementations.items():
        require(
            validate_file_binding(implementation_bindings.get(role), role)
            == expected_path,
            f"execution contract/{role} implementation mismatch",
        )
    require(
        Path(str(contract.get("output"))).expanduser().resolve() == output,
        "execution contract/output mismatch",
    )
    protocol = contract.get("execution_protocol")
    require(isinstance(protocol, dict), "execution contract lacks execution protocol")
    expected = {
        "claim_mode": "observed",
        "evaluate_only_paths_to_densify": True,
        "distinct_retained_paths_only": True,
        "energy_evaluations_per_path": N_DENSE_POINTS,
        "force_evaluations_per_path": 0,
        "optimization_N_F_contribution": 0,
        "fixed_endpoint_optimization_N_F": 0,
        "allow_exact_energy_reuse": False,
        "no_overwrite": True,
        "append_only_energy_WAL": True,
        "durable_attempt_started_before_each_physical_call": True,
        "fsync_each_attempt_start_and_terminal_record": True,
    }
    for key, value in expected.items():
        require(protocol.get(key) == value, f"execution protocol mismatch: {key}")
    lock_policy = contract.get("lock_policy")
    require(isinstance(lock_policy, dict), "execution contract lacks lock policy")
    require(lock_policy.get("jobs_per_host") == 1, "execution contract jobs_per_host is not one")
    require(
        lock_policy.get("acquire_both_before_first_physical_call") is True
        and lock_policy.get("hold_both_through_last_physical_call") is True,
        "execution contract does not hold both locks around all physical calls",
    )
    lock_values = lock_policy.get("host_local_BB_locks")
    require(isinstance(lock_values, list) and len(lock_values) == 2, "execution contract must name two BB locks")
    locks = tuple(Path(str(value)).expanduser().resolve() for value in lock_values)
    require(locks[0] != locks[1], "execution contract BB lock paths are identical")
    require(
        set(locks) == set(CANONICAL_HOST_LOCKS),
        "execution contract does not name both canonical host-local BB locks",
    )
    return contract, (locks[0], locks[1])


def _normalized_force_bindings(records: Sequence[Mapping[str, object]]) -> set[tuple[str, str, int, str]]:
    return {
        (
            str(resolve_bound_path(record.get("path"))),
            str(record.get("sha256")),
            int(record.get("optimization_N_F", -1)),
            str(record.get("official_path32_sha256", record.get("retained_path32_sha256", ""))),
        )
        for record in records
    }


def revalidate_metric_sources(
    selection: Mapping[str, object],
    paid_rows: Sequence[dict],
    force_receipts: Sequence[Path],
    completed_pairs: Sequence[tuple[Path, Path]],
) -> tuple[dict[str, float], dict[str, float], dict[str, dict]]:
    """Reconstruct exact I_perp and existing Delta E from bound evidence."""

    exact_i, included, excluded = target_freezer.load_exact_force_receipts(
        force_receipts, paid_rows
    )
    exact_sources: dict[str, dict] = {}
    for record in included:
        path_hash = str(record["official_path32_sha256"])
        source = {
            "kind": "paid_force_receipt",
            "path": str(resolve_bound_path(record["path"])),
            "sha256": str(record["sha256"]),
            "optimization_N_F": int(record["optimization_N_F"]),
            "paid_batch_id": int(record["paid_batch_id"]),
            "official_path32_sha256": path_hash,
        }
        previous = exact_sources.get(path_hash)
        require(
            previous is None or previous == source,
            "conflicting paid-force sources for one retained path",
        )
        exact_sources[path_hash] = source
    bindings = selection["bindings"]
    reported_force = list(bindings["paid_force_receipts"])
    reported_excluded = list(bindings["excluded_paid_force_receipts"])
    require(
        _normalized_force_bindings(included)
        == _normalized_force_bindings(reported_force),
        "included paid-force binding inventory drift",
    )
    require(
        _normalized_force_bindings(excluded)
        == _normalized_force_bindings(reported_excluded),
        "excluded/rejected paid-force binding inventory drift",
    )

    official_hash_by_nf = {
        int(row["optimization_N_F"]): str(row["official_path32_sha256"])
        for row in paid_rows
    }
    delta, diagnostic_i, completed = target_freezer.load_arc500_pairs(
        completed_pairs, official_hash_by_nf
    )
    reported_completed = bindings["completed_ODE_NEB_arc500"]
    normalized = {
        (
            str(resolve_bound_path(record["result"])),
            str(record["result_sha256"]),
            str(resolve_bound_path(record["audit"])),
            str(record["audit_sha256"]),
            tuple(int(value) for value in record["optimization_N_F"]),
            tuple(str(value) for value in record["official_path32_sha256"]),
        )
        for record in completed
    }
    reported_normalized = {
        (
            str(resolve_bound_path(record["result"])),
            str(record["result_sha256"]),
            str(resolve_bound_path(record["audit"])),
            str(record["audit_sha256"]),
            tuple(int(value) for value in record["optimization_N_F"]),
            tuple(str(value) for value in record["official_path32_sha256"]),
        )
        for record in reported_completed
    }
    require(normalized == reported_normalized, "completed arc500 binding inventory drift")
    for path_hash, value in diagnostic_i.items():
        require(
            path_hash not in exact_i or exact_i[path_hash] == value,
            "paid-force and completed arc500 discrete I_perp disagree",
        )
        exact_i[path_hash] = value
    return exact_i, delta, exact_sources


def validate_selection_decision(
    selection: Mapping[str, object],
    paid_rows: Sequence[dict],
    exact_i: Mapping[str, float],
    delta: Mapping[str, float],
) -> None:
    """Recompute the frozen selector decision from its immutable evidence."""

    expected_summary = target_freezer._inventory_summary(paid_rows, exact_i, delta)
    require(
        selection.get("official_ODE_NEB_inventory") == expected_summary,
        "selection ODE-NEB inventory summary drift",
    )
    thresholds = selection.get("thresholds_from_selected_IR_IVM")
    require(isinstance(thresholds, dict), "selection lacks final IR-IVM thresholds")
    i_threshold = float(thresholds["discrete_I_perp_eV2_per_A"])
    delta_threshold = float(thresholds["densified_Delta_E_kcal_per_mol"])
    require(
        selection.get("discrete_I_perp_first_passage")
        == target_freezer.classify_first_passage(paid_rows, exact_i, i_threshold),
        "selection discrete I_perp passage classification drift",
    )
    require(
        selection.get("densified_Delta_E_first_passage_before_followup")
        == target_freezer.classify_first_passage(paid_rows, delta, delta_threshold),
        "selection densified Delta E passage classification drift",
    )
    records = selection.get("paths_to_densify")
    require(isinstance(records, list) and records, "selection contains no target path")
    claim_mode = str(selection.get("claim_mode", ""))
    expected_rows, rationale = target_freezer.choose_paths_to_densify(
        paid_rows,
        delta,
        exact_i,
        i_threshold,
        delta_threshold,
        claim_mode,
        len(records),
    )
    expected_identity = [
        (
            int(row["optimization_N_F"]),
            str(row["official_path32_sha256"]),
        )
        for row in expected_rows
    ]
    reported_identity = [
        (
            int(record["optimization_N_F_first_occurrence"]),
            str(record["official_path32_sha256"]),
        )
        for record in records
    ]
    require(
        reported_identity == expected_identity,
        "paths_to_densify drift from the evidence-bound selector decision",
    )
    require(
        selection.get("selection_rationale") == rationale,
        "selection rationale drift from the recomputed decision",
    )


def preflight(selection_path: Path, execution_contract_path: Path, output: Path) -> dict:
    selection_path = selection_path.expanduser().resolve(strict=True)
    execution_contract_path = execution_contract_path.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    require(output.suffix == ".npz", "--output must end in .npz")
    selection = read_json(selection_path)
    require(selection.get("schema") == SELECTION_SCHEMA, "selection schema mismatch")
    require(
        selection.get("status") == "FROZEN_ZERO_OPENMM_TARGET_SELECTION",
        "target selection is not frozen",
    )
    validate_protocol(selection)
    bound = validate_selection_bindings(selection)

    source_root = bound["official_paths"]["optimization_complete"].parent
    paid_rows, authoritative_bindings = target_freezer.load_official_paid_ledger(
        source_root
    )
    for role in (
        "optimization_complete",
        "optimization_audit",
        "optimization_wal",
        "frozen_endpoints",
    ):
        reported = selection["bindings"][role]
        actual = authoritative_bindings[role]
        require(
            _binding_identity(reported) == _binding_identity(actual),
            f"authoritative official binding mismatch: {role}",
        )
    compare_paid_inventory(selection.get("official_paid_states"), paid_rows)
    selected_ivm = validate_selected_ivm_binding(
        selection, bound["selected_result"], bound["selected_audit"]
    )
    exact_i, measured_delta, exact_i_sources = revalidate_metric_sources(
        selection, paid_rows, bound["force_receipts"], bound["completed_pairs"]
    )
    validate_selection_decision(selection, paid_rows, exact_i, measured_delta)
    rows = validate_target_records(
        selection, paid_rows, exact_i, measured_delta, exact_i_sources
    )
    topology = derive_bound_topology(bound["completed_pairs"])
    prior_lineage, prior_targeted_count = completed_arc500_lineage(selection)
    _contract, locks = validate_execution_contract(
        execution_contract_path, selection_path, topology, output
    )
    audit_output = output.with_suffix(".audit.json")
    wal_output = output.with_suffix(".energy.wal.jsonl")
    require(
        not output.exists() and not audit_output.exists() and not wal_output.exists(),
        "refusing to overwrite a targeted ODE-NEB diagnostic artifact",
    )
    return {
        "selection_path": selection_path,
        "selection_sha256": sha256_file(selection_path),
        "execution_contract_path": execution_contract_path,
        "execution_contract_sha256": sha256_file(execution_contract_path),
        "output": output,
        "audit_output": audit_output,
        "wal_output": wal_output,
        "selection": selection,
        "topology": topology,
        "topology_sha256": sha256_file(topology),
        "locks": locks,
        "rows": rows,
        "selected_ivm": selected_ivm,
        "completed_prior_arc500_lineage": prior_lineage,
        "prior_targeted_bundle_count": prior_targeted_count,
        "target_round_index": prior_targeted_count + 1,
        "paid_state_count": len(paid_rows),
        "distinct_retained_path_count": len(
            {str(row["official_path32_sha256"]) for row in paid_rows}
        ),
    }


def _publish_exclusive(path: Path, writer) -> None:
    """Publish a fully flushed same-directory temporary via atomic hard link."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".tmp.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {destination}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_save_npz_exclusive(path: Path, **arrays) -> None:
    _publish_exclusive(path, lambda stream: np.savez_compressed(stream, **arrays))


def atomic_write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _publish_exclusive(path, lambda stream: stream.write(encoded))


def wal_context(preflight_state: Mapping[str, object]) -> dict:
    return {
        "target_selection": str(preflight_state["selection_path"]),
        "target_selection_sha256": str(preflight_state["selection_sha256"]),
        "execution_contract": str(preflight_state["execution_contract_path"]),
        "execution_contract_sha256": str(
            preflight_state["execution_contract_sha256"]
        ),
        "topology": str(preflight_state["topology"]),
        "topology_sha256": str(preflight_state["topology_sha256"]),
        "target_manifest": [
            {
                "row_index": index,
                "optimization_N_F": int(row["optimization_N_F"]),
                "official_path32_sha256": str(row["official_path32_sha256"]),
                "selected_path_sha256": str(row["path_sha256"]),
                "dense_coordinate_sha256": str(
                    row["preflight_dense_coordinate_sha256"]
                ),
            }
            for index, row in enumerate(preflight_state["rows"])
        ],
        "durable_attempt_started_before_each_physical_call": True,
        "fsync_each_attempt_start_and_terminal_record": True,
    }


def evaluate_targeted_paths(
    topology: Path,
    rows: Sequence[Mapping[str, object]],
    wal_path: Path,
    context: Mapping[str, object],
) -> list[dict]:
    """Freshly evaluate targeted paths with a durable pre-call WAL.

    This targeted execution path is deliberately local to this module.  The
    pre-frozen generic arc500 evaluator remains byte-for-byte unchanged and is
    used only for its energy-only backend and common arclength resampler.
    """

    reserved = {
        "schema",
        "number_of_points_per_path",
        "force_requests",
        "optimization_N_F_contribution",
        "exact_path_reuse_enabled",
    }.intersection(context)
    require(
        not reserved,
        f"WAL context attempts to replace reserved fields: {sorted(reserved)}",
    )
    evaluator = None
    evaluated: list[dict] = []
    with wal_path.open("x", encoding="utf-8", buffering=1) as wal:
        header = {
            "schema": "bb-densified-energy-wal-v1",
            "number_of_points_per_path": N_DENSE_POINTS,
            "force_requests": 0,
            "optimization_N_F_contribution": 0,
            "exact_path_reuse_enabled": False,
            **dict(context),
        }
        wal.write(json.dumps(header, sort_keys=True) + "\n")
        wal.flush()
        os.fsync(wal.fileno())
        for row_index, raw_row in enumerate(rows):
            row = dict(raw_row)
            dense, geometry = uniform_arclength_resample(row["path"])
            dense_sha = sha256_array(dense)
            require(
                dense_sha == str(row["preflight_dense_coordinate_sha256"]),
                f"target dense-coordinate hash drift at row {row_index}",
            )
            wal.write(
                json.dumps(
                    {
                        "event": "path_start",
                        "row_index": row_index,
                        "optimization_N_F": int(row["optimization_N_F"]),
                        "selected_path_sha256": str(row["path_sha256"]),
                        "official_path32_sha256": str(
                            row["official_path32_sha256"]
                        ),
                        "dense_coordinate_sha256": dense_sha,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            wal.flush()
            os.fsync(wal.fileno())
            if evaluator is None:
                evaluator = common_arc500.EnergyOnlyOpenMMEvaluator(topology)
            energies = np.empty(N_DENSE_POINTS, dtype=np.float64)
            for point_index, point in enumerate(dense):
                attempt = {
                    "row_index": row_index,
                    "point_index": point_index,
                    "attempt": evaluator.attempts + 1,
                }
                wal.write(
                    json.dumps(
                        {
                            **attempt,
                            "event": "attempt_started",
                            "selected_path_sha256": str(row["path_sha256"]),
                            "dense_coordinate_sha256": dense_sha,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                wal.flush()
                os.fsync(wal.fileno())
                try:
                    energies[point_index] = evaluator.evaluate(point)
                except Exception as error:
                    attempt.update(status="failed", error=repr(error))
                    wal.write(json.dumps(attempt, sort_keys=True) + "\n")
                    wal.flush()
                    os.fsync(wal.fileno())
                    raise
                attempt.update(
                    status="success", energy_eV=float(energies[point_index])
                )
                wal.write(json.dumps(attempt, sort_keys=True) + "\n")
                wal.flush()
                os.fsync(wal.fileno())
            require(
                evaluator.backend.raw_joint_attempts == 0,
                "targeted diagnostic requested a joint energy--force state",
            )
            energy_sha = sha256_array(energies)
            evaluated.append(
                {
                    **row,
                    **geometry,
                    "dense_coordinate_sha256": dense_sha,
                    "energy_eV": energies,
                    "energy_sha256": energy_sha,
                    "densified_Delta_E_kcal_per_mol": float(
                        (np.max(energies) - energies[0])
                        * common_arc500.EV_TO_KCAL_PER_MOL
                    ),
                    "diagnostic_energy_attempts": N_DENSE_POINTS,
                    "diagnostic_energy_successes": N_DENSE_POINTS,
                    "diagnostic_force_requests": 0,
                    "diagnostic_energy_reused": False,
                    "diagnostic_energy_attempts_this_evaluation": (
                        N_DENSE_POINTS
                    ),
                    "diagnostic_energy_source": "fresh_current_evaluation",
                }
            )
    expected_attempts = len(evaluated) * N_DENSE_POINTS
    actual_attempts = 0 if evaluator is None else evaluator.attempts
    require(
        actual_attempts == expected_attempts,
        "targeted diagnostic energy-attempt ledger is incomplete",
    )
    require(
        evaluator is not None
        and evaluator.successes == evaluator.attempts
        and evaluator.failures == 0,
        "targeted diagnostic energy evaluator recorded a failure",
    )
    return evaluated


def replay_targeted_energy_wal(
    wal_path: Path,
    target_manifest: Sequence[Mapping[str, object]],
    energy_rows: Sequence[np.ndarray],
    expected_context: Mapping[str, object],
) -> dict:
    """Replay a targeted WAL against independently loaded identities/energies."""

    with wal_path.open("r", encoding="utf-8") as stream:
        lines = [json.loads(line) for line in stream if line.strip()]
    require(bool(lines), "targeted energy WAL is empty")
    header = lines[0]
    require(header.get("schema") == "bb-densified-energy-wal-v1", "targeted WAL schema mismatch")
    require(
        int(header.get("number_of_points_per_path", -1)) == N_DENSE_POINTS
        and int(header.get("force_requests", -1)) == 0
        and int(header.get("optimization_N_F_contribution", -1)) == 0
        and header.get("exact_path_reuse_enabled") is False,
        "targeted WAL header permits reuse, force, or N_F contamination",
    )
    for key, value in expected_context.items():
        require(header.get(key) == value, f"targeted WAL context mismatch: {key}")

    cursor = 1
    expected_attempt = 0
    require(
        len(target_manifest) == len(energy_rows),
        "targeted WAL replay manifest/energy row count mismatch",
    )
    for row_index, (manifest, energies) in enumerate(
        zip(target_manifest, energy_rows)
    ):
        energies = np.asarray(energies, dtype=np.float64)
        require(
            energies.shape == (N_DENSE_POINTS,) and bool(np.all(np.isfinite(energies))),
            f"targeted WAL replay energy row {row_index} is invalid",
        )
        require(cursor < len(lines), "targeted WAL lacks path_start")
        start = lines[cursor]
        cursor += 1
        require(
            start
            == {
                "event": "path_start",
                "row_index": row_index,
                "optimization_N_F": int(manifest["optimization_N_F"]),
                "selected_path_sha256": str(manifest["selected_path_sha256"]),
                "official_path32_sha256": str(manifest["official_path32_sha256"]),
                "dense_coordinate_sha256": str(manifest["dense_coordinate_sha256"]),
            },
            f"targeted WAL path_start mismatch at row {row_index}",
        )
        for point_index in range(N_DENSE_POINTS):
            require(cursor + 1 < len(lines), "targeted WAL is truncated within a path")
            started = lines[cursor]
            terminal = lines[cursor + 1]
            cursor += 2
            expected_attempt += 1
            require(
                started
                == {
                    "event": "attempt_started",
                    "row_index": row_index,
                    "point_index": point_index,
                    "attempt": expected_attempt,
                    "selected_path_sha256": str(manifest["selected_path_sha256"]),
                    "dense_coordinate_sha256": str(manifest["dense_coordinate_sha256"]),
                },
                f"targeted WAL attempt_started mismatch at row {row_index}, point {point_index}",
            )
            require(
                terminal.get("row_index") == row_index
                and terminal.get("point_index") == point_index
                and terminal.get("attempt") == expected_attempt
                and terminal.get("status") == "success"
                and math.isfinite(float(terminal.get("energy_eV"))),
                f"targeted WAL terminal record mismatch at row {row_index}, point {point_index}",
            )
            require(
                set(terminal) == {
                    "row_index",
                    "point_index",
                    "attempt",
                    "status",
                    "energy_eV",
                },
                "targeted WAL terminal record contains an unexpected field",
            )
            require(
                float(terminal["energy_eV"])
                == float(energies[point_index]),
                f"targeted WAL/result energy mismatch at row {row_index}, point {point_index}",
            )
    require(cursor == len(lines), "targeted WAL contains an unexpected reuse/failure/tail event")
    require(
        expected_attempt == len(target_manifest) * N_DENSE_POINTS,
        "targeted WAL physical-attempt count mismatch",
    )
    return {
        "schema": "bb-targeted-energy-wal-terminal-validation-v1",
        "status": "PASS",
        "header_and_target_manifest_bound_before_first_physical_call": True,
        "attempt_started_fsynced_before_each_physical_call": True,
        "terminal_record_fsynced_after_each_success": True,
        "exact_path_energy_reuse_events": 0,
        "failed_energy_events": 0,
        "successful_energy_events": expected_attempt,
        "line_count": len(lines),
        "sha256": sha256_file(wal_path),
    }


def validate_terminal_energy_wal(
    preflight_state: Mapping[str, object], evaluated: Sequence[Mapping[str, object]]
) -> dict:
    """Replay the durable current-round WAL and prove no reuse or missing call."""

    manifest = [
        {
            "optimization_N_F": int(row["optimization_N_F"]),
            "selected_path_sha256": str(row["path_sha256"]),
            "official_path32_sha256": str(row["official_path32_sha256"]),
            "dense_coordinate_sha256": str(row["dense_coordinate_sha256"]),
        }
        for row in evaluated
    ]
    return replay_targeted_energy_wal(
        Path(preflight_state["wal_output"]),
        manifest,
        [np.asarray(row["energy_eV"], dtype=np.float64) for row in evaluated],
        wal_context(preflight_state),
    )


def save_terminal_artifacts(preflight_state: Mapping[str, object], evaluated: Sequence[dict], held_locks: Sequence[str]) -> dict:
    rows = list(preflight_state["rows"])
    require(len(evaluated) == len(rows), "evaluator returned a different path count")
    require(
        len({str(row["official_path32_sha256"]) for row in evaluated}) == len(rows),
        "evaluator returned duplicate target paths",
    )
    for index, (expected, actual) in enumerate(zip(rows, evaluated)):
        require(
            int(actual["optimization_N_F"]) == int(expected["optimization_N_F"])
            and str(actual["official_path32_sha256"])
            == str(expected["official_path32_sha256"])
            and str(actual["path_sha256"]) == str(expected["path_sha256"]),
            f"evaluated target order/identity mismatch at row {index}",
        )
        require(
            actual.get("diagnostic_energy_reused") is False
            and int(actual.get("diagnostic_energy_attempts_this_evaluation", -1))
            == N_DENSE_POINTS
            and int(actual.get("diagnostic_energy_attempts", -1)) == N_DENSE_POINTS
            and int(actual.get("diagnostic_energy_successes", -1)) == N_DENSE_POINTS
            and int(actual.get("diagnostic_force_requests", -1)) == 0,
            f"target row {index} does not contain exactly 500 fresh energy-only evaluations",
        )
        require(
            str(actual["dense_coordinate_sha256"])
            == str(expected["preflight_dense_coordinate_sha256"]),
            f"target row {index} dense-grid identity changed after preflight",
        )

    wal_validation = validate_terminal_energy_wal(preflight_state, evaluated)

    output = Path(preflight_state["output"])
    audit_output = Path(preflight_state["audit_output"])
    wal_output = Path(preflight_state["wal_output"])
    selection_path = Path(preflight_state["selection_path"])
    contract_path = Path(preflight_state["execution_contract_path"])
    count = len(evaluated)
    atomic_save_npz_exclusive(
        output,
        label=np.asarray([row["label"] for row in evaluated]),
        optimization_N_F=np.asarray(
            [row["optimization_N_F"] for row in evaluated], dtype=np.int64
        ),
        paid_batch_id_first_occurrence=np.asarray(
            [row["paid_batch_id"] for row in evaluated], dtype=np.int64
        ),
        source_batch_id=np.asarray(
            [row["source_batch_id"] for row in evaluated], dtype=np.int64
        ),
        discrete_I_perp=np.asarray(
            [row["discrete_I_perp"] for row in evaluated], dtype=np.float64
        ),
        densified_Delta_E_kcal_per_mol=np.asarray(
            [row["densified_Delta_E_kcal_per_mol"] for row in evaluated],
            dtype=np.float64,
        ),
        energy_eV=np.asarray([row["energy_eV"] for row in evaluated], dtype=np.float64),
        selected_path_sha256=np.asarray([row["path_sha256"] for row in evaluated]),
        official_path32_sha256=np.asarray(
            [row["official_path32_sha256"] for row in evaluated]
        ),
        dense_coordinate_sha256=np.asarray(
            [row["dense_coordinate_sha256"] for row in evaluated]
        ),
        energy_sha256=np.asarray([row["energy_sha256"] for row in evaluated]),
        source_result=np.asarray([row["source_result"] for row in evaluated]),
        source_result_sha256=np.asarray(
            [row["source_result_sha256"] for row in evaluated]
        ),
        exact_discrete_I_perp_source_kind=np.asarray(
            [row["exact_I_perp_source_kind"] for row in evaluated]
        ),
        exact_discrete_I_perp_source=np.asarray(
            [row["exact_I_perp_source"] for row in evaluated]
        ),
        exact_discrete_I_perp_source_sha256=np.asarray(
            [row["exact_I_perp_source_sha256"] for row in evaluated]
        ),
        exact_discrete_I_perp_source_optimization_N_F=np.asarray(
            [row["exact_I_perp_source_optimization_N_F"] for row in evaluated],
            dtype=np.int64,
        ),
        total_arclength_A=np.asarray(
            [row["total_arclength_A"] for row in evaluated], dtype=np.float64
        ),
        minimum_original_segment_length_A=np.asarray(
            [row["minimum_original_segment_length_A"] for row in evaluated],
            dtype=np.float64,
        ),
        maximum_original_segment_length_A=np.asarray(
            [row["maximum_original_segment_length_A"] for row in evaluated],
            dtype=np.float64,
        ),
        dense_spacing_A=np.asarray(
            [row["dense_spacing_A"] for row in evaluated], dtype=np.float64
        ),
        diagnostic_energy_attempts_per_path=np.full(
            count, N_DENSE_POINTS, dtype=np.int64
        ),
        diagnostic_energy_successes_per_path=np.full(
            count, N_DENSE_POINTS, dtype=np.int64
        ),
        diagnostic_energy_attempts_this_evaluation_per_path=np.full(
            count, N_DENSE_POINTS, dtype=np.int64
        ),
        diagnostic_energy_reused_per_path=np.zeros(count, dtype=bool),
        diagnostic_force_requests_per_path=np.zeros(count, dtype=np.int64),
        optimization_N_F_contribution=np.asarray(0, dtype=np.int64),
        fixed_endpoint_optimization_N_F=np.asarray(0, dtype=np.int64),
        coordinate_space=np.asarray(COORDINATE_SPACE),
        interpolation=np.asarray(INTERPOLATION),
        sampling=np.asarray(SAMPLING),
        number_of_points=np.asarray(N_DENSE_POINTS, dtype=np.int64),
        grid_formula=np.asarray(GRID_FORMULA),
        target_selection=np.asarray(str(selection_path)),
        target_selection_sha256=np.asarray(preflight_state["selection_sha256"]),
        execution_contract=np.asarray(str(contract_path)),
        execution_contract_sha256=np.asarray(
            preflight_state["execution_contract_sha256"]
        ),
        topology=np.asarray(str(preflight_state["topology"])),
        topology_sha256=np.asarray(preflight_state["topology_sha256"]),
        energy_wal=np.asarray(str(wal_output)),
        energy_wal_sha256=np.asarray(wal_validation["sha256"]),
        energy_wal_terminal_validation_schema=np.asarray(
            wal_validation["schema"]
        ),
        energy_wal_exact_path_reuse_events=np.asarray(0, dtype=np.int64),
        held_host_locks=np.asarray(list(held_locks)),
        target_round_index=np.asarray(
            preflight_state["target_round_index"], dtype=np.int64
        ),
        prior_targeted_bundle_count=np.asarray(
            preflight_state["prior_targeted_bundle_count"], dtype=np.int64
        ),
    )
    audit = {
        "schema": AUDIT_SCHEMA,
        "status": "PASS",
        "scope": "ONLY_DISTINCT_RETAINED_PATHS_IN_FROZEN_TARGET_SELECTION",
        "target_selection": str(selection_path),
        "target_selection_sha256": preflight_state["selection_sha256"],
        "execution_contract": str(contract_path),
        "execution_contract_sha256": preflight_state[
            "execution_contract_sha256"
        ],
        "output": str(output),
        "output_sha256": sha256_file(output),
        "energy_wal": str(wal_output),
        "energy_wal_sha256": sha256_file(wal_output),
        "energy_wal_terminal_validation": wal_validation,
        "paid_N_F_state_count_verified": preflight_state["paid_state_count"],
        "distinct_retained_path_count_verified": preflight_state[
            "distinct_retained_path_count"
        ],
        "selected_distinct_retained_path_count": count,
        "target_round_index": preflight_state["target_round_index"],
        "prior_targeted_bundle_count": preflight_state[
            "prior_targeted_bundle_count"
        ],
        "completed_prior_arc500_lineage": preflight_state[
            "completed_prior_arc500_lineage"
        ],
        "fresh_diagnostic_energy_attempts": count * N_DENSE_POINTS,
        "fresh_diagnostic_energy_successes": count * N_DENSE_POINTS,
        "lineage_energy_evaluations": count * N_DENSE_POINTS,
        "diagnostic_force_requests": 0,
        "optimization_N_F_contribution": 0,
        "fixed_endpoint_optimization_N_F": 0,
        "every_included_path_has_exactly_500_successful_energy_evaluations": True,
        "every_included_path_was_freshly_evaluated_once": True,
        "exact_path_energy_reuse_count": 0,
        "current_round_allow_exact_energy_reuse": False,
        "every_discrete_I_perp_is_bound_to_an_exact_paid_force_receipt": True,
        "held_host_locks": list(held_locks),
        "records": [
            {
                "optimization_N_F": int(row["optimization_N_F"]),
                "paid_batch_id_first_occurrence": int(row["paid_batch_id"]),
                "source_batch_id": int(row["source_batch_id"]),
                "official_path32_sha256": str(row["official_path32_sha256"]),
                "selected_path_sha256": str(row["path_sha256"]),
                "dense_coordinate_sha256": str(row["dense_coordinate_sha256"]),
                "energy_sha256": str(row["energy_sha256"]),
                "discrete_I_perp": float(row["discrete_I_perp"]),
                "densified_Delta_E_kcal_per_mol": float(
                    row["densified_Delta_E_kcal_per_mol"]
                ),
                "diagnostic_energy_attempts": N_DENSE_POINTS,
                "diagnostic_energy_successes": N_DENSE_POINTS,
                "diagnostic_force_requests": 0,
                "exact_discrete_I_perp_source_kind": str(
                    row["exact_I_perp_source_kind"]
                ),
                "exact_discrete_I_perp_source": str(
                    row["exact_I_perp_source"]
                ),
                "exact_discrete_I_perp_source_sha256": str(
                    row["exact_I_perp_source_sha256"]
                ),
                "exact_discrete_I_perp_source_optimization_N_F": int(
                    row["exact_I_perp_source_optimization_N_F"]
                ),
            }
            for row in evaluated
        ],
    }
    atomic_write_json_exclusive(audit_output, audit)
    return audit


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    state = preflight(args.selection, args.execution_contract, args.output)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "schema": PREFLIGHT_SCHEMA,
                    "status": "PASS",
                    "target_selection": str(state["selection_path"]),
                    "target_selection_sha256": state["selection_sha256"],
                    "execution_contract": str(state["execution_contract_path"]),
                    "execution_contract_sha256": state[
                        "execution_contract_sha256"
                    ],
                    "official_paid_states_verified": state["paid_state_count"],
                    "distinct_retained_paths_verified": state[
                        "distinct_retained_path_count"
                    ],
                    "selected_distinct_retained_paths": len(state["rows"]),
                    "planned_energy_evaluations": len(state["rows"])
                    * N_DENSE_POINTS,
                    "new_OpenMM_energy_attempts": 0,
                    "new_OpenMM_force_attempts": 0,
                    "optimization_N_F_contribution": 0,
                    "fixed_endpoint_optimization_N_F": 0,
                    "physical_evaluation_locks_acquired": False,
                    "paths": [
                        {
                            "optimization_N_F": int(row["optimization_N_F"]),
                            "paid_batch_id_first_occurrence": int(
                                row["paid_batch_id"]
                            ),
                            "official_path32_sha256": str(
                                row["official_path32_sha256"]
                            ),
                            "dense_coordinate_sha256": str(
                                row["preflight_dense_coordinate_sha256"]
                            ),
                        }
                        for row in state["rows"]
                    ],
                },
                sort_keys=True,
            )
        )
        return

    with acquire_host_locks(state["locks"]) as held_locks:
        evaluated = evaluate_targeted_paths(
            state["topology"],
            state["rows"],
            state["wal_output"],
            wal_context(state),
        )
    audit = save_terminal_artifacts(state, evaluated, held_locks)
    print(json.dumps(audit, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
