#!/usr/bin/env python3
"""Draw the terminal Barnase--Barstar IVM versus ODE--NEB cost-to-precision figure.

This entry point is intentionally stricter than the in-progress preview.  It
accepts only the selected 9990-N_F IVM lineage after terminal and exact-prefix
verification.  ODE--NEB data comprise the completed audited milestone bundles
plus any audit-passing targeted arc500 follow-up bound to a frozen selection
receipt.  Sparse observations are never promoted to an exact first passage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from evaluate_discrete_iperp_densified_barrier import (
    N_DENSE_POINTS,
    sha256_array,
    sha256_file,
)
from plot_bb_cost_to_precision_preview import (
    IVM_COLOR,
    NEB_COLOR,
    _first_passage,
    _force_formatter,
    _frontier,
    _setup_style,
)


SELECTED_LABEL = "bb_irivm_a1e0_r01_m05_inf9990_r1"
COMMON_LABEL = "common_initial_path"
OFFICIAL_ODE_CORE_MILESTONES = (240, 480, 990, 9990)
OFFICIAL_ODE_OPTIONAL_MILESTONES = (4080,)
TARGETED_ODE_AUDIT_SCHEMA = "bb-ode-neb-targeted-arc500-audit-v1"
TARGETED_SELECTION_SCHEMA = "bb-ode-neb-targeted-arc500-selection-v1"
TARGETED_EXECUTION_SCHEMA = "bb-ode-neb-targeted-arc500-execution-contract-v1"
TARGETED_WAL_VALIDATION_SCHEMA = "bb-targeted-energy-wal-terminal-validation-v1"
TARGETED_WAL_SCHEMA = "bb-densified-energy-wal-v1"
CANONICAL_BB_LOCKS = (
    "/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock",
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock",
)
ODE_AUDIT_SCHEMAS = {
    "bb-ode-neb-arc500-preview-audit-v1": OFFICIAL_ODE_CORE_MILESTONES,
    "bb-ode-neb-nf4080-arc500-audit-v1": OFFICIAL_ODE_OPTIONAL_MILESTONES,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def scalar(saved: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def binding_path(receipt: Dict[str, object], role: str) -> Path:
    record = receipt.get("bindings", {}).get(role)
    require(record is not None, f"receipt lacks {role} binding")
    path = Path(str(record["path"])).expanduser().resolve(strict=True)
    require(sha256_file(path) == str(record["sha256"]), f"{role} binding drift")
    return path


def file_binding(path: Path) -> Dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _collect_file_bindings(value: object) -> set[Tuple[str, str]]:
    records: set[Tuple[str, str]] = set()
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            path = Path(value["path"]).expanduser().resolve(strict=True)
            require(sha256_file(path) == value["sha256"], f"figure-contract binding drift: {path}")
            records.add((str(path), str(value["sha256"])))
        for child in value.values():
            records.update(_collect_file_bindings(child))
    elif isinstance(value, list):
        for child in value:
            records.update(_collect_file_bindings(child))
    return records


def _contains_placeholder(value: object) -> bool:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped.startswith("<") and stripped.endswith(">")
    if isinstance(value, dict):
        return any(_contains_placeholder(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(child) for child in value)
    return False


def validate_figure_contract(
    contract_raw: Path, output: Path, required_inputs: Sequence[Path]
) -> Dict[str, object]:
    """Validate the immutable final-figure contract and every input binding."""

    contract_path = contract_raw.expanduser().resolve(strict=True)
    contract = read_json(contract_path)
    require(
        contract.get("schema") == "bb-irivm-cost-to-precision-figure-contract-v1",
        "unsupported final figure contract schema",
    )
    require(
        contract.get("status") == "FROZEN_FINAL_INPUT_BINDINGS",
        "final figure contract is not activated and frozen",
    )
    require(not _contains_placeholder(contract), "final figure contract still contains a placeholder")
    require(
        Path(str(contract.get("figure_output"))).expanduser().resolve() == output,
        "figure contract/output mismatch",
    )
    claim_policy = contract.get("claim_policy", {})
    require(claim_policy.get("Delta_E_monotonicity_assumed") is False, "figure contract assumes Delta E monotonicity")
    require(
        claim_policy.get("allowed_sparse_claim") == "earliest observed first passage",
        "figure contract lacks the sparse-passage claim guard",
    )
    bindings = _collect_file_bindings(contract)
    for raw in required_inputs:
        path = raw.expanduser().resolve(strict=True)
        require(
            (str(path), sha256_file(path)) in bindings,
            f"final figure contract does not bind required input: {path}",
        )
    plotter = Path(__file__).resolve(strict=True)
    require(
        (str(plotter), sha256_file(plotter)) in bindings,
        "final figure contract does not bind this plotter",
    )
    return {"contract": file_binding(contract_path), "status": contract["status"]}


def _validate_selected_receipts(
    terminal_result: Path,
    terminal_audit: Path,
    long_verification_path: Path,
    prefix_verification_path: Path,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    long_verification = read_json(long_verification_path)
    require(
        long_verification.get("schema") == "bb-irivm-long-budget-member-verification-v2",
        "unsupported terminal long-member verification schema",
    )
    require(long_verification.get("status") == "PASS", "terminal long-member verification did not pass")
    require(long_verification.get("label") == SELECTED_LABEL, "terminal verification is not for the selected label")
    require(binding_path(long_verification, "result") == terminal_result, "terminal result binding mismatch")
    require(binding_path(long_verification, "audit") == terminal_audit, "terminal audit binding mismatch")
    require(int(long_verification.get("endpoint_optimization_N_F", -1)) == 0, "fixed endpoints incurred optimization NoF")

    prefix = read_json(prefix_verification_path)
    require(
        prefix.get("schema") == "bb-irivm-selected-9990-exact-prefix-verification-v1",
        "unsupported exact-prefix verification schema",
    )
    require(prefix.get("status") == "PASS_EXACT_PREFIX", "exact 4080-prefix verification did not pass")
    require(prefix.get("tolerance_fallback_used") is False, "exact-prefix verification used a tolerance fallback")
    require(binding_path(prefix, "selected_result") == terminal_result, "prefix/terminal result mismatch")
    require(
        binding_path(prefix, "selected_verification") == long_verification_path,
        "prefix/terminal verification mismatch",
    )
    return long_verification, prefix


def load_selected_ivm(
    arc500_result: Path,
    arc500_audit: Path,
    terminal_result: Path,
    terminal_audit: Path,
    long_verification_path: Path,
    prefix_verification_path: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    paths = [
        arc500_result,
        arc500_audit,
        terminal_result,
        terminal_audit,
        long_verification_path,
        prefix_verification_path,
    ]
    (
        arc500_result,
        arc500_audit,
        terminal_result,
        terminal_audit,
        long_verification_path,
        prefix_verification_path,
    ) = [path.expanduser().resolve(strict=True) for path in paths]
    long_verification, prefix = _validate_selected_receipts(
        terminal_result,
        terminal_audit,
        long_verification_path,
        prefix_verification_path,
    )
    audit = read_json(arc500_audit)
    require(
        audit.get("schema") == "bb-irivm-discrete-iperp-densified-barrier-audit-v1",
        "unsupported selected arc500 audit schema",
    )
    require(audit.get("status") == "PASS", "selected arc500 audit did not pass")
    require(audit.get("output_sha256") == sha256_file(arc500_result), "selected arc500 output hash mismatch")
    require(Path(str(audit.get("output"))).expanduser().resolve() == arc500_result, "selected arc500 audit/output path mismatch")
    require(int(audit.get("diagnostic_force_requests", -1)) == 0, "selected arc500 requested force")
    require(int(audit.get("optimization_N_F_contribution", -1)) == 0, "selected arc500 contaminated optimization NoF")

    with np.load(arc500_result, allow_pickle=False) as saved:
        required = {
            "label",
            "optimization_N_F",
            "discrete_I_perp",
            "densified_Delta_E_kcal_per_mol",
            "diagnostic_energy_successes_per_path",
            "diagnostic_force_requests_per_path",
            "optimization_N_F_contribution",
            "number_of_points",
            "source_result",
            "source_result_sha256",
        }
        missing = sorted(required.difference(saved.files))
        require(not missing, f"selected arc500 result is missing {missing}")
        labels = np.asarray(saved["label"]).astype(str)
        nf = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        i_perp = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
        delta_e = np.asarray(saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64)
        successes = np.asarray(saved["diagnostic_energy_successes_per_path"], dtype=np.int64)
        forces = np.asarray(saved["diagnostic_force_requests_per_path"], dtype=np.int64)
        source_results = np.asarray(saved["source_result"]).astype(str)
        source_hashes = np.asarray(saved["source_result_sha256"]).astype(str)
        points = int(scalar(saved, "number_of_points"))
        optimization_contribution = int(scalar(saved, "optimization_N_F_contribution"))
    require(set(labels).issubset({COMMON_LABEL, SELECTED_LABEL}), "candidate-B/preview rows are forbidden in the final IVM result")
    require(np.count_nonzero(labels == SELECTED_LABEL) > 0, "selected 9990 label has no arc500 row")
    require(labels.shape == nf.shape == i_perp.shape == delta_e.shape, "selected arc500 row arrays differ in length")
    require(successes.shape == nf.shape and forces.shape == nf.shape, "selected diagnostic ledgers differ in length")
    require(points == N_DENSE_POINTS and np.all(successes == N_DENSE_POINTS), "selected paths do not each bind 500 energies")
    require(np.all(forces == 0) and optimization_contribution == 0, "selected diagnostic calls contaminate optimization NoF")
    terminal_sha = sha256_file(terminal_result)
    selected_mask = labels == SELECTED_LABEL
    require(
        np.all(source_hashes[selected_mask] == terminal_sha),
        "selected arc500 row is not bound to the terminal 9990 result",
    )
    require(
        all(Path(value).expanduser().resolve() == terminal_result for value in source_results[selected_mask]),
        "selected arc500 row points to a different optimization result",
    )
    order = np.argsort(nf, kind="stable")
    data = {
        "number_of_forces": nf[order],
        "i_perp": i_perp[order],
        "delta_e": delta_e[order],
    }
    provenance = {
        "selected_label": SELECTED_LABEL,
        "arc500_result": file_binding(arc500_result),
        "arc500_audit": file_binding(arc500_audit),
        "terminal_result": file_binding(terminal_result),
        "terminal_audit": file_binding(terminal_audit),
        "terminal_long_member_verification": file_binding(long_verification_path),
        "exact_4080_prefix_verification": file_binding(prefix_verification_path),
        "terminal_actual_optimization_N_F": int(long_verification["actual_optimization_N_F"]),
        "exact_prefix_state_count": len(prefix["reference_eligible_prefix"]),
    }
    return data, provenance


def _validate_targeted_selection(path: Path) -> Tuple[Dict[str, object], Path]:
    selection_path = path.expanduser().resolve(strict=True)
    selection = read_json(selection_path)
    require(
        selection.get("schema") == TARGETED_SELECTION_SCHEMA,
        "unsupported targeted ODE selection schema",
    )
    require(
        selection.get("status") == "FROZEN_ZERO_OPENMM_TARGET_SELECTION",
        "targeted ODE selection is not frozen",
    )
    require(
        selection.get("physical_calls_made_by_selector") == {"energy": 0, "force": 0},
        "targeted selector made or ambiguously reported a physical call",
    )
    require(
        int(selection.get("optimization_N_F_contribution", -1)) == 0,
        "targeted selector changed optimization N_F",
    )
    guard = selection.get("claim_guard", {})
    require(guard.get("Delta_E_monotonicity_assumed") is False, "selection assumes Delta E monotonicity")
    require(
        guard.get("unmeasured_earlier_paths_forbid_exact_first_passage_claim") is True,
        "selection does not guard sparse first-passage claims",
    )
    paid_states = selection.get("official_paid_states")
    require(
        isinstance(paid_states, list) and len(paid_states) == 333,
        "final selection must embed all 333 official paid states for passage classification",
    )
    return selection, selection_path


def _selection_completed_pairs(selection: Dict[str, object]) -> set[Tuple[str, str, str, str]]:
    records = selection.get("bindings", {}).get("completed_ODE_NEB_arc500", [])
    require(isinstance(records, list) and records, "selection lacks completed ODE arc500 bindings")
    result = set()
    for record in records:
        result_path = Path(record["result"]).expanduser().resolve(strict=True)
        audit_path = Path(record["audit"]).expanduser().resolve(strict=True)
        require(str(record["result_sha256"]) == sha256_file(result_path), "selection completed-result binding drift")
        require(str(record["audit_sha256"]) == sha256_file(audit_path), "selection completed-audit binding drift")
        result.add((str(result_path), sha256_file(result_path), str(audit_path), sha256_file(audit_path)))
    require(len(result) == len(records), "selection repeats a completed ODE bundle")
    return result


def validate_selection_selected_ivm(
    selection_path: Path, selected_result: Path, selected_audit: Path
) -> None:
    """Reject a targeted selection frozen from any nonterminal/candidate IVM result."""

    selection, _ = _validate_targeted_selection(selection_path)
    record = selection.get("bindings", {}).get("selected_IR_IVM_arc500")
    require(isinstance(record, dict), "selection lacks selected IR-IVM arc500 binding")
    selected_result = selected_result.expanduser().resolve(strict=True)
    selected_audit = selected_audit.expanduser().resolve(strict=True)
    require(
        Path(str(record.get("result"))).expanduser().resolve(strict=True) == selected_result
        and str(record.get("result_sha256")) == sha256_file(selected_result),
        "targeted selection was not frozen from the final selected IR-IVM result",
    )


def _selection_selected_ivm_identity(selection: Mapping[str, object]) -> Tuple[str, str, str, str]:
    record = selection.get("bindings", {}).get("selected_IR_IVM_arc500")
    require(isinstance(record, dict), "selection lacks selected IR-IVM arc500 binding")
    result = Path(str(record["result"])).expanduser().resolve(strict=True)
    audit = Path(str(record["audit"])).expanduser().resolve(strict=True)
    require(str(record["result_sha256"]) == sha256_file(result), "selection selected-IVM result drift")
    require(str(record["audit_sha256"]) == sha256_file(audit), "selection selected-IVM audit drift")
    return str(result), sha256_file(result), str(audit), sha256_file(audit)


def _official_ledger_identity(selection: Mapping[str, object]) -> Tuple[Tuple[object, ...], ...]:
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
    return tuple(tuple(record.get(key) for key in keys) for record in selection["official_paid_states"])


def _pair_identity(result: Path, audit: Path) -> Tuple[str, str, str, str]:
    result = result.expanduser().resolve(strict=True)
    audit = audit.expanduser().resolve(strict=True)
    return str(result), sha256_file(result), str(audit), sha256_file(audit)


def _validate_targeted_execution_contract(
    contract_path: Path, selection_path: Path, result: Path
) -> Dict[str, object]:
    contract_path = contract_path.expanduser().resolve(strict=True)
    contract = read_json(contract_path)
    require(contract.get("schema") == TARGETED_EXECUTION_SCHEMA, "targeted execution-contract schema mismatch")
    require(
        contract.get("status") == "AUTHORIZED_AFTER_FROZEN_TARGET_SELECTION"
        and contract.get("physical_execution_authorized") is True,
        "targeted execution contract was not authorized",
    )
    target = contract.get("target_selection", {})
    require(
        Path(str(target.get("path"))).expanduser().resolve(strict=True) == selection_path
        and str(target.get("sha256")) == sha256_file(selection_path),
        "targeted execution contract/selection mismatch",
    )
    require(Path(str(contract.get("output"))).expanduser().resolve() == result, "targeted execution contract/output mismatch")
    protocol = contract.get("execution_protocol", {})
    expected_protocol = {
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
    for key, expected in expected_protocol.items():
        require(protocol.get(key) == expected, f"targeted execution protocol mismatch: {key}")
    lock_policy = contract.get("lock_policy", {})
    lock_paths = tuple(str(value) for value in lock_policy.get("host_local_BB_locks", []))
    require(
        int(lock_policy.get("jobs_per_host", -1)) == 1
        and lock_policy.get("acquire_both_before_first_physical_call") is True
        and lock_policy.get("hold_both_through_last_physical_call") is True
        and len(lock_paths) == 2
        and len(set(lock_paths)) == 2
        and all(Path(value).expanduser().is_absolute() for value in lock_paths),
        "targeted execution contract lock policy mismatch",
    )
    _collect_file_bindings(contract)
    return contract


def _validate_targeted_wal(
    wal_path: Path,
    selection_path: Path,
    contract_path: Path,
    topology: Path,
    nf: np.ndarray,
    official_hashes: np.ndarray,
    selected_hashes: np.ndarray,
    dense_hashes: np.ndarray,
    energies: np.ndarray,
) -> Dict[str, object]:
    with wal_path.open("r", encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    count = len(nf)
    require(len(events) == 1 + count * (1 + 2 * N_DENSE_POINTS), "targeted WAL line count mismatch")
    header = events[0]
    manifest = [
        {
            "row_index": index,
            "optimization_N_F": int(nf[index]),
            "official_path32_sha256": str(official_hashes[index]),
            "selected_path_sha256": str(selected_hashes[index]),
            "dense_coordinate_sha256": str(dense_hashes[index]),
        }
        for index in range(count)
    ]
    require(
        header.get("schema") == TARGETED_WAL_SCHEMA
        and int(header.get("number_of_points_per_path", -1)) == N_DENSE_POINTS
        and int(header.get("force_requests", -1)) == 0
        and int(header.get("optimization_N_F_contribution", -1)) == 0
        and header.get("exact_path_reuse_enabled") is False
        and Path(str(header.get("target_selection"))).expanduser().resolve() == selection_path
        and header.get("target_selection_sha256") == sha256_file(selection_path)
        and Path(str(header.get("execution_contract"))).expanduser().resolve() == contract_path
        and header.get("execution_contract_sha256") == sha256_file(contract_path)
        and Path(str(header.get("topology"))).expanduser().resolve() == topology
        and header.get("topology_sha256") == sha256_file(topology)
        and header.get("target_manifest") == manifest
        and header.get("durable_attempt_started_before_each_physical_call") is True
        and header.get("fsync_each_attempt_start_and_terminal_record") is True,
        "targeted WAL header or manifest mismatch",
    )
    cursor = 1
    attempt = 0
    for row_index in range(count):
        require(
            events[cursor]
            == {
                "event": "path_start",
                "row_index": row_index,
                "optimization_N_F": int(nf[row_index]),
                "selected_path_sha256": str(selected_hashes[row_index]),
                "official_path32_sha256": str(official_hashes[row_index]),
                "dense_coordinate_sha256": str(dense_hashes[row_index]),
            },
            f"targeted WAL path_start mismatch at row {row_index}",
        )
        cursor += 1
        for point_index in range(N_DENSE_POINTS):
            attempt += 1
            started = events[cursor]
            terminal = events[cursor + 1]
            cursor += 2
            require(
                started
                == {
                    "event": "attempt_started",
                    "row_index": row_index,
                    "point_index": point_index,
                    "attempt": attempt,
                    "selected_path_sha256": str(selected_hashes[row_index]),
                    "dense_coordinate_sha256": str(dense_hashes[row_index]),
                },
                f"targeted WAL attempt mismatch at row {row_index}, point {point_index}",
            )
            require(
                set(terminal) == {"row_index", "point_index", "attempt", "status", "energy_eV"}
                and terminal.get("row_index") == row_index
                and terminal.get("point_index") == point_index
                and terminal.get("attempt") == attempt
                and terminal.get("status") == "success"
                and float(terminal.get("energy_eV")) == float(energies[row_index, point_index]),
                f"targeted WAL terminal mismatch at row {row_index}, point {point_index}",
            )
    require(cursor == len(events), "targeted WAL has an unexpected tail")
    return {"successful_energy_events": attempt, "line_count": len(events)}


def _load_targeted_round(
    selection_raw: Path,
    result_raw: Path,
    audit_raw: Path,
    expected_round: int,
    available_prior_pairs: set[Tuple[str, str, str, str]],
    required_prior_targeted_pairs: set[Tuple[str, str, str, str]],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    selection, selection_path = _validate_targeted_selection(selection_raw)
    result = result_raw.expanduser().resolve(strict=True)
    audit_path = audit_raw.expanduser().resolve(strict=True)
    audit = read_json(audit_path)
    require(audit.get("schema") == TARGETED_ODE_AUDIT_SCHEMA, "targeted ODE audit schema mismatch")
    require(audit.get("status") == "PASS", "targeted ODE audit did not pass")
    require(
        Path(str(audit.get("output"))).expanduser().resolve() == result
        and audit.get("output_sha256") == sha256_file(result),
        "targeted ODE audit/output mismatch",
    )
    require(
        Path(str(audit.get("target_selection"))).expanduser().resolve() == selection_path
        and audit.get("target_selection_sha256") == sha256_file(selection_path),
        "targeted ODE audit/selection mismatch",
    )
    completed_pairs = _selection_completed_pairs(selection)
    require(completed_pairs.issubset(available_prior_pairs), "targeted selection cites a non-prior completed bundle")
    require(required_prior_targeted_pairs.issubset(completed_pairs), "targeted selection omits an earlier targeted round")
    lineage = audit.get("completed_prior_arc500_lineage")
    require(isinstance(lineage, list), "targeted audit lacks prior lineage")
    lineage_pairs = set()
    selection_records = {
        (
            str(Path(record["result"]).expanduser().resolve(strict=True)),
            str(Path(record["audit"]).expanduser().resolve(strict=True)),
        ): record
        for record in selection["bindings"]["completed_ODE_NEB_arc500"]
    }
    for record in lineage:
        identity = (
            str(Path(record["result"]).expanduser().resolve(strict=True)),
            str(record["result_sha256"]),
            str(Path(record["audit"]).expanduser().resolve(strict=True)),
            str(record["audit_sha256"]),
        )
        require(identity in completed_pairs, "targeted audit prior lineage is outside its selection")
        selected_record = selection_records[(identity[0], identity[2])]
        require(
            [int(value) for value in record.get("optimization_N_F", [])]
            == [int(value) for value in selected_record.get("optimization_N_F", [])]
            and [str(value) for value in record.get("official_path32_sha256", [])]
            == [str(value) for value in selected_record.get("official_path32_sha256", [])],
            "targeted audit prior-lineage row inventory mismatch",
        )
        lineage_pairs.add(identity)
    require(lineage_pairs == completed_pairs, "targeted audit prior lineage is incomplete")

    count_expected = len(selection.get("paths_to_densify", []))
    require(count_expected > 0, "targeted selection contains no path")
    expected_calls = count_expected * N_DENSE_POINTS
    require(
        int(audit.get("paid_N_F_state_count_verified", -1)) == 333
        and int(audit.get("distinct_retained_path_count_verified", -1)) == 284
        and int(audit.get("selected_distinct_retained_path_count", -1)) == count_expected
        and int(audit.get("target_round_index", -1)) == expected_round
        and int(audit.get("prior_targeted_bundle_count", -1)) == expected_round - 1,
        "targeted audit round or paid-ledger counts mismatch",
    )
    require(
        int(audit.get("fresh_diagnostic_energy_attempts", -1)) == expected_calls
        and int(audit.get("fresh_diagnostic_energy_successes", -1)) == expected_calls
        and int(audit.get("lineage_energy_evaluations", -1)) == expected_calls
        and int(audit.get("diagnostic_force_requests", -1)) == 0
        and int(audit.get("optimization_N_F_contribution", -1)) == 0
        and int(audit.get("fixed_endpoint_optimization_N_F", -1)) == 0
        and audit.get("every_included_path_has_exactly_500_successful_energy_evaluations") is True
        and audit.get("every_included_path_was_freshly_evaluated_once") is True
        and int(audit.get("exact_path_energy_reuse_count", -1)) == 0
        and audit.get("current_round_allow_exact_energy_reuse") is False
        and audit.get("every_discrete_I_perp_is_bound_to_an_exact_paid_force_receipt") is True,
        "targeted audit fresh-call, force, N_F, or reuse ledger mismatch",
    )

    contract_path = Path(str(audit.get("execution_contract"))).expanduser().resolve(strict=True)
    require(audit.get("execution_contract_sha256") == sha256_file(contract_path), "targeted execution-contract hash mismatch")
    contract = _validate_targeted_execution_contract(contract_path, selection_path, result)
    topology = Path(str(contract["topology"]["path"])).expanduser().resolve(strict=True)

    with np.load(result, allow_pickle=False) as saved:
        required = {
            "optimization_N_F", "discrete_I_perp", "densified_Delta_E_kcal_per_mol",
            "official_path32_sha256", "selected_path_sha256", "dense_coordinate_sha256",
            "energy_sha256", "energy_eV", "diagnostic_energy_attempts_per_path",
            "diagnostic_energy_successes_per_path",
            "diagnostic_energy_attempts_this_evaluation_per_path",
            "diagnostic_energy_reused_per_path", "diagnostic_force_requests_per_path",
            "exact_discrete_I_perp_source_kind", "exact_discrete_I_perp_source",
            "exact_discrete_I_perp_source_sha256",
            "exact_discrete_I_perp_source_optimization_N_F",
        }
        require(required.issubset(saved.files), f"targeted result lacks {sorted(required.difference(saved.files))}")
        nf = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        i_perp = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
        delta_e = np.asarray(saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64)
        official_hashes = np.asarray(saved["official_path32_sha256"]).astype(str)
        selected_hashes = np.asarray(saved["selected_path_sha256"]).astype(str)
        dense_hashes = np.asarray(saved["dense_coordinate_sha256"]).astype(str)
        energy_hashes = np.asarray(saved["energy_sha256"]).astype(str)
        energies = np.asarray(saved["energy_eV"], dtype=np.float64)
        attempts = np.asarray(saved["diagnostic_energy_attempts_per_path"], dtype=np.int64)
        successes = np.asarray(saved["diagnostic_energy_successes_per_path"], dtype=np.int64)
        fresh_attempts = np.asarray(saved["diagnostic_energy_attempts_this_evaluation_per_path"], dtype=np.int64)
        reused = np.asarray(saved["diagnostic_energy_reused_per_path"], dtype=bool)
        forces = np.asarray(saved["diagnostic_force_requests_per_path"], dtype=np.int64)
        source_kinds = np.asarray(saved["exact_discrete_I_perp_source_kind"]).astype(str)
        source_paths = np.asarray(saved["exact_discrete_I_perp_source"]).astype(str)
        source_hashes = np.asarray(saved["exact_discrete_I_perp_source_sha256"]).astype(str)
        source_nfs = np.asarray(saved["exact_discrete_I_perp_source_optimization_N_F"], dtype=np.int64)
        require(int(scalar(saved, "number_of_points")) == N_DENSE_POINTS, "targeted result is not arc500")
        for key, expected in (
            ("coordinate_space", "unweighted_3N_cartesian"),
            ("interpolation", "piecewise_linear"),
            ("sampling", "uniform_cartesian_arclength"),
            ("grid_formula", "s_j=j*L/499,j=0,...,499"),
        ):
            require(str(scalar(saved, key)) == expected, f"targeted result protocol mismatch: {key}")
        require(
            int(scalar(saved, "optimization_N_F_contribution")) == 0
            and int(scalar(saved, "fixed_endpoint_optimization_N_F")) == 0
            and int(scalar(saved, "target_round_index")) == expected_round
            and int(scalar(saved, "prior_targeted_bundle_count")) == expected_round - 1,
            "targeted result round or N_F scalar mismatch",
        )
        require(
            Path(str(scalar(saved, "target_selection"))).expanduser().resolve() == selection_path
            and str(scalar(saved, "target_selection_sha256")) == sha256_file(selection_path)
            and Path(str(scalar(saved, "execution_contract"))).expanduser().resolve() == contract_path
            and str(scalar(saved, "execution_contract_sha256")) == sha256_file(contract_path)
            and Path(str(scalar(saved, "topology"))).expanduser().resolve() == topology
            and str(scalar(saved, "topology_sha256")) == sha256_file(topology),
            "targeted result selection/contract/topology lineage mismatch",
        )
        wal_path = Path(str(scalar(saved, "energy_wal"))).expanduser().resolve(strict=True)
        require(
            str(scalar(saved, "energy_wal_sha256")) == sha256_file(wal_path)
            and str(scalar(saved, "energy_wal_terminal_validation_schema")) == TARGETED_WAL_VALIDATION_SCHEMA
            and int(scalar(saved, "energy_wal_exact_path_reuse_events")) == 0,
            "targeted result WAL binding mismatch",
        )

    row_shape = (count_expected,)
    for array in (nf, i_perp, delta_e, official_hashes, selected_hashes, dense_hashes, energy_hashes, attempts, successes, fresh_attempts, reused, forces, source_kinds, source_paths, source_hashes, source_nfs):
        require(array.shape == row_shape, "targeted result row-array shape mismatch")
    require(energies.shape == (count_expected, N_DENSE_POINTS), "targeted result energy shape mismatch")
    require(
        np.all(np.isfinite(i_perp))
        and np.all(i_perp >= 0.0)
        and np.all(np.isfinite(delta_e))
        and np.all(np.isfinite(energies)),
        "targeted result contains a non-finite or invalid physical metric",
    )
    require(
        np.all(attempts == N_DENSE_POINTS)
        and np.all(successes == N_DENSE_POINTS)
        and np.all(fresh_attempts == N_DENSE_POINTS)
        and not np.any(reused)
        and not np.any(forces),
        "targeted result contains reuse or incomplete fresh energy/force ledgers",
    )
    require(all(sha256_array(row) == energy_hashes[index] for index, row in enumerate(energies)), "targeted result energy hash mismatch")

    receipt_records = selection["bindings"].get("paid_force_receipts", [])
    receipts = {
        (str(Path(record["path"]).expanduser().resolve(strict=True)), str(record["sha256"])): record
        for record in receipt_records
    }
    audit_records = audit.get("records")
    require(isinstance(audit_records, list) and len(audit_records) == count_expected, "targeted audit record inventory mismatch")
    for index, target in enumerate(selection["paths_to_densify"]):
        require(
            int(target["optimization_N_F_first_occurrence"]) == int(nf[index])
            and str(target["official_path32_sha256"]) == str(official_hashes[index])
            and float(target["exact_discrete_I_perp_eV2_per_A"]) == float(i_perp[index]),
            f"targeted result/selection NF, path, or I_perp mismatch at row {index}",
        )
        source = Path(source_paths[index]).expanduser().resolve(strict=True)
        source_key = (str(source), str(source_hashes[index]))
        require(source_kinds[index] == "paid_force_receipt" and source_key in receipts, f"targeted I_perp source is not a bound paid-force receipt at row {index}")
        receipt = receipts[source_key]
        require(
            sha256_file(source) == str(source_hashes[index])
            and int(receipt["optimization_N_F"]) == int(source_nfs[index])
            and str(receipt["official_path32_sha256"]) == str(official_hashes[index])
            and float(receipt["discrete_I_perp_eV2_per_A"]) == float(i_perp[index]),
            f"targeted paid-force receipt identity or metric mismatch at row {index}",
        )
        record = audit_records[index]
        expected_record = {
            "optimization_N_F": int(nf[index]),
            "official_path32_sha256": str(official_hashes[index]),
            "selected_path_sha256": str(selected_hashes[index]),
            "dense_coordinate_sha256": str(dense_hashes[index]),
            "energy_sha256": str(energy_hashes[index]),
            "discrete_I_perp": float(i_perp[index]),
            "densified_Delta_E_kcal_per_mol": float(delta_e[index]),
            "diagnostic_energy_attempts": N_DENSE_POINTS,
            "diagnostic_energy_successes": N_DENSE_POINTS,
            "diagnostic_force_requests": 0,
            "exact_discrete_I_perp_source_kind": "paid_force_receipt",
            "exact_discrete_I_perp_source": str(source),
            "exact_discrete_I_perp_source_sha256": str(source_hashes[index]),
            "exact_discrete_I_perp_source_optimization_N_F": int(source_nfs[index]),
        }
        for key, expected in expected_record.items():
            require(record.get(key) == expected, f"targeted audit record mismatch at row {index}: {key}")

    require(
        Path(str(audit.get("energy_wal"))).expanduser().resolve() == wal_path
        and audit.get("energy_wal_sha256") == sha256_file(wal_path),
        "targeted audit WAL binding mismatch",
    )
    replay = _validate_targeted_wal(
        wal_path, selection_path, contract_path, topology, nf, official_hashes,
        selected_hashes, dense_hashes, energies,
    )
    terminal = audit.get("energy_wal_terminal_validation", {})
    require(
        terminal.get("schema") == TARGETED_WAL_VALIDATION_SCHEMA
        and terminal.get("status") == "PASS"
        and terminal.get("header_and_target_manifest_bound_before_first_physical_call") is True
        and terminal.get("attempt_started_fsynced_before_each_physical_call") is True
        and terminal.get("terminal_record_fsynced_after_each_success") is True
        and int(terminal.get("exact_path_energy_reuse_events", -1)) == 0
        and int(terminal.get("failed_energy_events", -1)) == 0
        and int(terminal.get("successful_energy_events", -1)) == replay["successful_energy_events"]
        and int(terminal.get("line_count", -1)) == replay["line_count"]
        and terminal.get("sha256") == sha256_file(wal_path),
        "targeted WAL terminal validation mismatch",
    )
    return {
        "number_of_forces": nf,
        "i_perp": i_perp,
        "delta_e": delta_e,
        "official_path32_sha256": official_hashes,
        "selection": selection,
        "selection_path": selection_path,
        "topology_identity": (str(topology), sha256_file(topology)),
        "completed_pairs": completed_pairs,
    }, {
        "audit_schema": TARGETED_ODE_AUDIT_SCHEMA,
        "target_round_index": expected_round,
        "selection": file_binding(selection_path),
        "execution_contract": file_binding(contract_path),
        "result": file_binding(result),
        "audit": file_binding(audit_path),
        "energy_wal": file_binding(wal_path),
        "optimization_N_F": [int(value) for value in nf],
        "official_path32_sha256": [str(value) for value in official_hashes],
    }
    require(
        Path(str(record.get("audit"))).expanduser().resolve(strict=True) == selected_audit
        and str(record.get("audit_sha256")) == sha256_file(selected_audit),
        "targeted selection was not frozen from the final selected IR-IVM audit",
    )


def load_official_ode(
    result_paths: Sequence[Path],
    audit_paths: Sequence[Path],
    targeted_selection_path: Path | None = None,
    targeted_bundles: Sequence[Tuple[Path, Path, Path]] = (),
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]]]:
    require(len(result_paths) == len(audit_paths), "each ODE result needs its explicit audit")
    old_pairs: List[Tuple[Path, Path]] = []
    rounds: List[Tuple[Path, Path, Path]] = []
    legacy_target: Tuple[Path, Path] | None = None
    for result_raw, audit_raw in zip(result_paths, audit_paths):
        result = result_raw.expanduser().resolve(strict=True)
        audit_path = audit_raw.expanduser().resolve(strict=True)
        schema = str(read_json(audit_path).get("schema"))
        if schema == TARGETED_ODE_AUDIT_SCHEMA:
            require(targeted_selection_path is not None, "legacy targeted ODE pair requires --ode-target-selection")
            require(legacy_target is None, "legacy one-selection syntax supports only one targeted round")
            legacy_target = (result, audit_path)
        else:
            old_pairs.append((result, audit_path))
    if targeted_selection_path is not None:
        require(legacy_target is not None, "--ode-target-selection lacks its legacy targeted result/audit pair")
        rounds.append((targeted_selection_path, legacy_target[0], legacy_target[1]))
    rounds.extend((Path(selection), Path(result), Path(audit)) for selection, result, audit in targeted_bundles)
    rows: List[Tuple[int, float, float, str]] = []
    bindings: List[Dict[str, object]] = []
    seen_schemas = set()
    available_prior_pairs: set[Tuple[str, str, str, str]] = set()
    for result, audit_path in old_pairs:
        audit = read_json(audit_path)
        schema = str(audit.get("schema"))
        require(schema in ODE_AUDIT_SCHEMAS, "ODE input is not an audited official milestone bundle")
        require(schema not in seen_schemas, "duplicate ODE milestone bundle schema")
        seen_schemas.add(schema)
        require(audit.get("status") == "PASS", "ODE milestone audit did not pass")
        require(audit.get("output_sha256") == sha256_file(result), "ODE milestone output hash mismatch")
        require(Path(str(audit.get("output"))).expanduser().resolve() == result, "ODE audit/output path mismatch")
        require(int(audit.get("diagnostic_force_requests", -1)) == 0, "ODE milestone diagnostic requested force")
        require(int(audit.get("optimization_N_F_contribution", -1)) == 0, "ODE milestone diagnostic contaminated optimization NoF")
        with np.load(result, allow_pickle=False) as saved:
            nf = np.asarray(saved["optimization_N_F"], dtype=np.int64)
            i_perp = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
            delta_e = np.asarray(saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64)
            force = np.asarray(saved["diagnostic_force_requests_per_path"], dtype=np.int64)
            attempts = np.asarray(saved["diagnostic_energy_attempts_per_path"], dtype=np.int64)
            if "diagnostic_energy_successes_per_path" in saved.files:
                successes = np.asarray(
                    saved["diagnostic_energy_successes_per_path"], dtype=np.int64
                )
            else:
                require(
                    schema == "bb-ode-neb-arc500-preview-audit-v1",
                    "only the audited legacy official milestone bundle may omit per-path successes",
                )
                successes = attempts.copy()
            points = int(scalar(saved, "number_of_points"))
            require("official_path32_sha256" in saved.files, "ODE result lacks official path identities")
            official_hashes = np.asarray(saved["official_path32_sha256"]).astype(str)
        expected = ODE_AUDIT_SCHEMAS[schema]
        require(tuple(sorted(int(value) for value in nf)) == expected, "ODE bundle contains a nonofficial or missing milestone")
        require(
            i_perp.shape
            == nf.shape
            == delta_e.shape
            == force.shape
            == attempts.shape
            == successes.shape,
            "ODE milestone row arrays differ in length",
        )
        require(official_hashes.shape == nf.shape, "ODE official path identities differ in length")
        require(
            points == N_DENSE_POINTS
            and np.all(attempts == N_DENSE_POINTS)
            and np.all(successes == N_DENSE_POINTS),
            "ODE paths do not each bind 500 successful energies",
        )
        require(np.all(force == 0), "ODE result contains a force request")
        expected_calls = int(nf.size) * N_DENSE_POINTS
        require(int(audit["diagnostic_energy_attempts"]) == expected_calls, "ODE diagnostic attempt ledger is incomplete")
        require(int(audit["diagnostic_energy_successes"]) == expected_calls, "ODE diagnostic success ledger is incomplete")
        rows.extend(
            (int(cost), float(residual), float(barrier), str(path_hash))
            for cost, residual, barrier, path_hash in zip(nf, i_perp, delta_e, official_hashes)
        )
        bindings.append(
            {
                "audit_schema": schema,
                "result": file_binding(result),
                "audit": file_binding(audit_path),
                "optimization_N_F": [int(value) for value in sorted(nf.tolist())],
                "official_path32_sha256": [str(value) for value in official_hashes],
            }
        )
        available_prior_pairs.add(_pair_identity(result, audit_path))

    common_selected_identity = None
    common_ledger_identity = None
    common_topology_identity = None
    required_prior_targeted_pairs: set[Tuple[str, str, str, str]] = set()
    seen_target_hashes: set[str] = set()
    latest_selection = None
    latest_selection_path = None
    for round_index, (selection_path, result, audit_path) in enumerate(rounds, start=1):
        round_data, round_binding = _load_targeted_round(
            selection_path, result, audit_path, round_index,
            available_prior_pairs, required_prior_targeted_pairs,
        )
        selection = round_data["selection"]
        selected_identity = _selection_selected_ivm_identity(selection)
        ledger_identity = _official_ledger_identity(selection)
        topology_identity = round_data["topology_identity"]
        if common_selected_identity is None:
            common_selected_identity = selected_identity
            common_ledger_identity = ledger_identity
            common_topology_identity = topology_identity
        else:
            require(selected_identity == common_selected_identity, "targeted rounds bind different selected IR-IVM arc500 results")
            require(ledger_identity == common_ledger_identity, "targeted rounds do not share the same full 333-state official ledger")
            require(topology_identity == common_topology_identity, "targeted rounds bind different topologies")
        hashes = set(np.asarray(round_data["official_path32_sha256"]).astype(str))
        require(not hashes.intersection(seen_target_hashes), "targeted rounds repeat an already densified path")
        seen_target_hashes.update(hashes)
        for cost, residual, barrier, path_hash in zip(
            round_data["number_of_forces"], round_data["i_perp"],
            round_data["delta_e"], round_data["official_path32_sha256"],
        ):
            rows.append((int(cost), float(residual), float(barrier), str(path_hash)))
        bindings.append(round_binding)
        identity = _pair_identity(Path(result), Path(audit_path))
        available_prior_pairs.add(identity)
        required_prior_targeted_pairs.add(identity)
        latest_selection = selection
        latest_selection_path = Path(selection_path).expanduser().resolve(strict=True)

    costs = [row[0] for row in rows]
    require(set(OFFICIAL_ODE_CORE_MILESTONES).issubset(costs), "official ODE core milestones are incomplete")
    require(len(costs) == len(set(costs)), "ODE milestone inputs contain duplicate optimization N_F")
    rows.sort(key=lambda row: row[0])
    data: Dict[str, object] = {
        "number_of_forces": np.asarray([row[0] for row in rows], dtype=np.int64),
        "i_perp": np.asarray([row[1] for row in rows], dtype=np.float64),
        "delta_e": np.asarray([row[2] for row in rows], dtype=np.float64),
        "official_path32_sha256": np.asarray([row[3] for row in rows]),
    }
    if latest_selection is not None and latest_selection_path is not None:
        data["official_paid_states"] = latest_selection["official_paid_states"]
        data["targeted_selections"] = [
            binding["selection"]
            for binding in bindings
            if binding.get("audit_schema") == TARGETED_ODE_AUDIT_SCHEMA
        ]
    return data, bindings


def raw_points(data: Dict[str, np.ndarray]) -> List[Dict[str, object]]:
    return [
        {
            "optimization_N_F": int(cost),
            "discrete_I_perp": float(residual),
            "densified_Delta_E_kcal_per_mol": float(barrier),
        }
        for cost, residual, barrier in zip(
            data["number_of_forces"], data["i_perp"], data["delta_e"]
        )
    ]


def classify_observed_passage(
    ode: Dict[str, object], metric_key: str, threshold: float
) -> Dict[str, object]:
    """Classify a sparse ODE passage without a monotonicity assumption.

    A plotted crossing is exact only if every distinct official retained path
    with an earlier first occurrence has a metric value.  Otherwise it is the
    earliest *observed* passage and the earliest missing path supplies the
    conservative lower endpoint of the possible exact-passage interval.
    """

    nf = np.asarray(ode["number_of_forces"], dtype=np.int64)
    metric = np.asarray(ode[metric_key], dtype=np.float64)
    passing = np.flatnonzero(metric <= threshold)
    require(passing.size > 0, f"ODE-NEB has no observed {metric_key} passage")
    observed_index = int(passing[np.argmin(nf[passing])])
    observed_nf = int(nf[observed_index])
    observed_value = float(metric[observed_index])

    paid_states = ode.get("official_paid_states")
    if not isinstance(paid_states, list):
        return {
            "claim": "sampled_first_passage_only",
            "observed_first_passage_N_F": observed_nf,
            "observed_value": observed_value,
            "possible_exact_first_passage_N_F_interval": None,
            "reason": "the complete official paid-state ledger was not supplied",
        }

    measured_hashes = set(np.asarray(ode["official_path32_sha256"]).astype(str))
    first_nf_by_hash: Dict[str, int] = {}
    for state in paid_states:
        path_hash = str(state["official_path32_sha256"])
        state_nf = int(state["optimization_N_F"])
        first_nf_by_hash.setdefault(path_hash, state_nf)
    earlier_missing = sorted(
        state_nf
        for path_hash, state_nf in first_nf_by_hash.items()
        if state_nf < observed_nf and path_hash not in measured_hashes
    )
    exact = not earlier_missing
    return {
        "claim": "exact_first_passage" if exact else "earliest_observed_first_passage",
        "observed_first_passage_N_F": observed_nf,
        "observed_value": observed_value,
        "possible_exact_first_passage_N_F_interval": (
            [observed_nf, observed_nf]
            if exact
            else [int(earlier_missing[0]), observed_nf]
        ),
        "unmeasured_distinct_paths_before_observed_passage": len(earlier_missing),
        "Delta_E_monotonicity_assumed": False,
    }


def analyze(ivm: Dict[str, np.ndarray], ode: Dict[str, np.ndarray]) -> Dict[str, object]:
    report: Dict[str, object] = {
        "raw_points": {"IVM": raw_points(ivm), "ODE-NEB": raw_points(ode)},
        "lower_envelopes": {},
        "thresholds": {},
        "first_passage_number_of_forces": {},
        "ODE_NEB_passage_claim": {},
        "ODE_NEB_over_IVM": {},
        "ODE_NEB_over_IVM_interpretation": {},
        "Delta_E_monotonicity_assumed": False,
    }
    for key in ("i_perp", "delta_e"):
        ivm_x, ivm_y = _frontier(ivm["number_of_forces"], ivm[key])
        ode_x, ode_y = _frontier(ode["number_of_forces"], ode[key])
        threshold = max(float(np.min(ivm_x)), float(np.min(ode_x)))
        ivm_cost = _first_passage(ivm["number_of_forces"], ivm[key], threshold)
        ode_cost = _first_passage(ode["number_of_forces"], ode[key], threshold)
        require(ivm_cost is not None and ode_cost is not None, f"both methods must reach the common {key} threshold")
        passage_evidence = classify_observed_passage(ode, key, threshold)
        require(
            int(passage_evidence["observed_first_passage_N_F"]) == int(ode_cost),
            "ODE observed-passage evidence disagrees with the plotted lower envelope",
        )
        report["lower_envelopes"][key] = {
            "IVM": [
                {"metric": float(value), "optimization_N_F": int(cost)}
                for value, cost in zip(ivm_x, ivm_y)
            ],
            "ODE-NEB": [
                {"metric": float(value), "optimization_N_F": int(cost)}
                for value, cost in zip(ode_x, ode_y)
            ],
        }
        report["thresholds"][key] = threshold
        report["first_passage_number_of_forces"][key] = {
            "IVM": int(ivm_cost),
            "ODE-NEB": int(ode_cost),
        }
        report["ODE_NEB_passage_claim"][key] = passage_evidence
        report["ODE_NEB_over_IVM"][key] = float(ode_cost) / float(ivm_cost)
        report["ODE_NEB_over_IVM_interpretation"][key] = (
            "exact first-passage cost ratio"
            if passage_evidence["claim"] == "exact_first_passage"
            else "observed-passage cost ratio; not an exact first-passage ratio"
        )
    return report


def _add_arrow(
    axis,
    threshold: float,
    ivm_cost: int,
    ode_cost: int,
    ratio: float,
    passage_claim: str,
) -> None:
    axis.axvline(threshold, color="#767676", linewidth=0.55, linestyle=":", zorder=1)
    midpoint = 0.5 * (ivm_cost + ode_cost)
    axis.annotate(
        "",
        xy=(threshold, ivm_cost),
        xytext=(threshold, ode_cost),
        arrowprops={"arrowstyle": "<->", "color": "#202020", "lw": 0.8},
        zorder=25,
    )
    axis.annotate(
        (
            rf"{ratio:.1f}$\times$"
            if passage_claim == "exact_first_passage"
            else rf"{ratio:.1f}$\times$ obs."
        ),
        xy=(threshold, midpoint),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color="#202020",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.8},
        zorder=26,
    )


def render(ivm: Dict[str, np.ndarray], ode: Dict[str, np.ndarray], analysis: Dict[str, object], output: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    _setup_style(mpl)
    figure = plt.figure(figsize=(180 / 25.4, 72 / 25.4), dpi=220)
    grid = GridSpec(1, 2, figure=figure, left=0.072, right=0.935, bottom=0.18, top=0.78, wspace=0.25)
    axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    for axis, key, xlabel, xscale in (
        (axes[0], "i_perp", r"$I_\perp$", "log"),
        (axes[1], "delta_e", r"$\Delta E$ (kcal mol$^{-1}$)", "linear"),
    ):
        ivm_x, ivm_y = _frontier(ivm["number_of_forces"], ivm[key])
        ode_x, ode_y = _frontier(ode["number_of_forces"], ode[key])
        axis.plot(ivm_x, ivm_y, color=IVM_COLOR, linestyle="-", linewidth=1.85, marker="o", markersize=3.3, markerfacecolor="white", markeredgewidth=0.6, zorder=20)
        axis.plot(ode_x, ode_y, color=NEB_COLOR, linestyle="--", linewidth=1.65, marker="s", markersize=3.1, markerfacecolor="white", markeredgewidth=0.6, zorder=6)
        passage = analysis["first_passage_number_of_forces"][key]
        _add_arrow(
            axis,
            analysis["thresholds"][key],
            passage["IVM"],
            passage["ODE-NEB"],
            analysis["ODE_NEB_over_IVM"][key],
            analysis["ODE_NEB_passage_claim"][key]["claim"],
        )
        all_x = np.concatenate((ivm_x, ode_x))
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
    handles = [
        Line2D([0], [0], label="IVM", color=IVM_COLOR, linestyle="-", linewidth=1.85, marker="o", markersize=3.3, markerfacecolor="white", markeredgewidth=0.6),
        Line2D([0], [0], label="ODE-NEB", color=NEB_COLOR, linestyle="--", linewidth=1.65, marker="s", markersize=3.1, markerfacecolor="white", markeredgewidth=0.6),
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, columnspacing=1.5, handlelength=2.35, handletextpad=0.5, frameon=False)
    figure.savefig(
        output,
        format="pdf",
        dpi=600,
        metadata={
            "Title": "Barnase--Barstar IVM versus ODE-NEB cost to precision",
            "Subject": "Discrete I_perp and equal-Cartesian-arclength 500-point Delta E",
        },
    )
    plt.close(figure)


def _exclusive_publish(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(temporary, destination)


def main(argv: Iterable[str] = None) -> None:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--ivm-result", type=Path, required=True)
    parser.add_argument("--ivm-audit", type=Path, required=True)
    parser.add_argument("--ivm-terminal-result", type=Path, required=True)
    parser.add_argument("--ivm-terminal-audit", type=Path, required=True)
    parser.add_argument("--ivm-long-verification", type=Path, required=True)
    parser.add_argument("--ivm-prefix-verification", type=Path, required=True)
    parser.add_argument("--ode-result", type=Path, action="append", required=True)
    parser.add_argument("--ode-audit", type=Path, action="append", required=True)
    parser.add_argument(
        "--ode-target-selection",
        type=Path,
        help="legacy one-round syntax: selection paired with the targeted result/audit in --ode-result/--ode-audit",
    )
    parser.add_argument(
        "--ode-targeted-bundle",
        nargs=3,
        type=Path,
        action="append",
        default=[],
        metavar=("SELECTION", "RESULT", "AUDIT"),
        help="repeat in chronological round order for iterative targeted densification",
    )
    parser.add_argument(
        "--figure-contract",
        type=Path,
        required=True,
        help="activated immutable final-figure contract binding every input",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv_list)
    require(
        args.ode_target_selection is not None or bool(args.ode_targeted_bundle),
        "final figure requires at least one completed targeted ODE bundle",
    )
    output = args.output.expanduser().resolve()
    require(output.suffix.lower() == ".pdf", "final output must be a PDF")
    provenance_path = output.with_suffix(".json")
    require(not output.exists(), "refusing to overwrite an existing final figure")
    require(not provenance_path.exists(), "refusing to overwrite existing final provenance")

    required_contract_inputs = [
        args.ivm_result,
        args.ivm_audit,
        args.ivm_terminal_result,
        args.ivm_terminal_audit,
        args.ivm_long_verification,
        args.ivm_prefix_verification,
        *args.ode_result,
        *args.ode_audit,
    ]
    if args.ode_target_selection is not None:
        required_contract_inputs.append(args.ode_target_selection)
    for selection, result, audit in args.ode_targeted_bundle:
        required_contract_inputs.extend((selection, result, audit))
    figure_contract = validate_figure_contract(
        args.figure_contract, output, required_contract_inputs
    )

    ivm, ivm_bindings = load_selected_ivm(
        args.ivm_result,
        args.ivm_audit,
        args.ivm_terminal_result,
        args.ivm_terminal_audit,
        args.ivm_long_verification,
        args.ivm_prefix_verification,
    )
    if args.ode_target_selection is not None:
        validate_selection_selected_ivm(
            args.ode_target_selection, args.ivm_result, args.ivm_audit
        )
    for selection, _result, _audit in args.ode_targeted_bundle:
        validate_selection_selected_ivm(
            selection, args.ivm_result, args.ivm_audit
        )
    ode, ode_bindings = load_official_ode(
        args.ode_result,
        args.ode_audit,
        args.ode_target_selection,
        [tuple(bundle) for bundle in args.ode_targeted_bundle],
    )
    analysis = analyze(ivm, ode)
    provenance = {
        "schema": "bb-terminal-cost-to-precision-provenance-v1",
        "status": "FINAL_TERMINAL_EVIDENCE",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "figure": str(output),
        "command_argv": [sys.executable, str(Path(__file__).resolve()), *argv_list],
        "figure_contract": figure_contract,
        "selected_IVM_label": SELECTED_LABEL,
        "legend_labels": ["IVM", "ODE-NEB"],
        "vertical_axis": "Number of Forces",
        "multiplier_annotation_position": "right of the vertical arrow at its arithmetic line midpoint",
        "diagnostic_energy_calls_excluded_from_optimization_axis": True,
        "speedup_uses_observed_ODE_NEB_passage_unless_exactly_certified": True,
        "IVM_bindings": ivm_bindings,
        "ODE_NEB_bindings": ode_bindings,
        "plotter": file_binding(Path(__file__).resolve()),
        **analysis,
    }
    if args.validate_only:
        print(json.dumps(provenance, indent=2, sort_keys=True))
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output.parent), prefix=".bb-final-plot-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        temporary_pdf = temporary_root / output.name
        temporary_provenance = temporary_root / provenance_path.name
        render(ivm, ode, analysis, temporary_pdf)
        provenance["figure_sha256"] = sha256_file(temporary_pdf)
        temporary_provenance.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _exclusive_publish(temporary_provenance, provenance_path)
        _exclusive_publish(temporary_pdf, output)
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
