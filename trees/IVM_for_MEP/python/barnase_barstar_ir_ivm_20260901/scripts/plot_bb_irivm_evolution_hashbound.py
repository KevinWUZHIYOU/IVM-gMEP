#!/usr/bin/env python3
"""Create the new hash-bound Barnase--Barstar IR-IVM evolution figure.

This is deliberately *not* an exact-reproduction entry point for the legacy
``barnase_branch_local_true0k_evolution.pdf``.  The legacy figure is a style
reference only: its exact one-off generator is not present in the repository.

The entry point is fail-closed.  A final contract must explicitly bind the
old calibrated path-CV definition, the old 0 K PES data, every ODE-NEB path
snapshot, and the selected 9990-upper-bound IR-IVM result/audit/verification.
The terminal JSON receipts and their SHA-256 bindings are validated before
the selected NPZ is hashed or opened, so an active/incomplete selected result
is never consumed.  Only stored paths following an accepted
reparameterization are eligible for an evolution panel.

No OpenMM context is imported or created by this script.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline, PchipInterpolator


CONTRACT_SCHEMA = "bb-irivm-evolution-figure-contract-v1"
FINAL_CONTRACT_STATUS = (
    "FINAL_INPUTS_FROZEN_AFTER_SELECTED_9990_TERMINAL_GATES"
)
TERMINAL_SCHEMA = "bb-irivm-long-budget-member-verification-v2"
PREFIX_SCHEMA = "bb-irivm-selected-9990-exact-prefix-verification-v1"
ODE_LEDGER_SCHEMA = "bb-ode-neb-all-paid-batches-arc500-contract-v1"
SELECTED_LABEL = "bb_irivm_a1e0_r01_m05_inf9990_r1"
N_IMAGES = 32
N_INTERNAL = 30

IVM_COLOR = "#0072B2"
ODE_NEB_COLOR = "#D55E00"
INITIAL_COLOR = "#7B3294"

FORBIDDEN_FORMAL_OUTPUTS = {
    Path(
        "/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/"
        "barnase_branch_local_true0k_evolution.pdf"
    ),
    Path(
        "/Users/wuzhiyou/IVM_for_MEP/Figures/barnase_barstar/"
        "barnase_branch_local_true0k_evolution.pdf"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verifier_dtype_shape_sha256(value: np.ndarray) -> str:
    """Match ``verify_long_budget_member.py::array_sha256`` exactly."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def canonical_le_float64_sha256(value: np.ndarray) -> str:
    """Match the selected-prefix/arc500 canonical float64 path identity."""

    array = np.asarray(value, dtype=np.dtype("<f8"), order="C")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def ode_ledger_array_sha256(value: np.ndarray) -> str:
    """Match the official ODE-NEB paid-path ledger identity."""

    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    require(np.all(np.isfinite(array)), "ODE-NEB path contains a non-finite value")
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


def scalar(saved: Mapping[str, np.ndarray], key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def resolve_path(contract_path: Path, value: str, role: str) -> Path:
    raw = Path(str(value)).expanduser()
    resolved = raw if raw.is_absolute() else contract_path.parent / raw
    try:
        return resolved.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"missing {role}: {resolved}") from error


def resolve_binding(
    contract_path: Path,
    record: Mapping[str, object],
    role: str,
    *,
    verify_hash: bool = True,
) -> Path:
    require(isinstance(record, Mapping), f"{role} binding is not an object")
    expected = str(record.get("sha256", ""))
    require(len(expected) == 64, f"{role} binding lacks a SHA-256")
    path = resolve_path(contract_path, str(record.get("path", "")), role)
    if verify_hash:
        actual = sha256_file(path)
        require(actual == expected, f"{role} SHA-256 mismatch: {actual} != {expected}")
    return path


def receipt_binding_path(
    receipt_path: Path,
    receipt: Mapping[str, object],
    role: str,
    *,
    verify_hash: bool = True,
) -> Path:
    bindings = receipt.get("bindings", {})
    require(isinstance(bindings, Mapping) and role in bindings, f"receipt lacks {role} binding")
    record = bindings[role]
    require(isinstance(record, Mapping), f"receipt {role} binding is not an object")
    raw = Path(str(record.get("path", ""))).expanduser()
    path = raw if raw.is_absolute() else receipt_path.parent / raw
    try:
        path = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"receipt-bound {role} is missing: {path}") from error
    if verify_hash:
        expected = str(record.get("sha256", ""))
        actual = sha256_file(path)
        require(actual == expected, f"receipt-bound {role} has drifted")
    return path


def _same_file(first: Path, second: Path, role: str) -> None:
    require(first == second, f"{role} path mismatch: {first} != {second}")


def validate_output_path(
    output: Path,
    identity: Mapping[str, object],
    output_contract: Mapping[str, object],
) -> Tuple[Path, Path]:
    destination = output.expanduser().resolve()
    require(destination.suffix.lower() == ".pdf", "evolution output must be a PDF")
    require(destination not in FORBIDDEN_FORMAL_OUTPUTS, "refusing to overwrite a formal evolution PDF")
    required_prefix = str(identity.get("required_output_stem_prefix", ""))
    require(required_prefix == "barnase_barstar_ir_ivm_evolution", "unexpected output-stem contract")
    require(destination.stem.startswith(required_prefix), "output stem does not identify the new IR-IVM figure")
    provenance = destination.with_suffix(".json")
    require(
        destination.name == str(output_contract.get("pdf_filename", "")),
        "CLI PDF filename differs from the final figure contract",
    )
    require(
        provenance.name == str(output_contract.get("provenance_filename", "")),
        "same-stem provenance filename differs from the final figure contract",
    )
    require(not destination.exists(), f"refusing to overwrite {destination}")
    require(not provenance.exists(), f"refusing to overwrite {provenance}")
    return destination, provenance


def validate_panel_contract(
    panel_contract: Sequence[Mapping[str, object]],
) -> List[Tuple[int, int]]:
    require(len(panel_contract) == 3, "evolution figure requires exactly three comparison panels")
    pairs: List[Tuple[int, int]] = []
    for row in panel_contract:
        require(isinstance(row, Mapping), "comparison panel is not an object")
        ivm_nf = int(row.get("ivm_optimization_N_F", -1))
        ode_nf = int(row.get("ode_neb_optimization_N_F", -1))
        require(ivm_nf > 0 and ode_nf > 0, "comparison panel has a nonpositive NoF")
        pairs.append((ivm_nf, ode_nf))
    require(len({ivm_nf for ivm_nf, _ in pairs}) == 3, "duplicate IVM NoF panel label")
    require(len({ode_nf for _, ode_nf in pairs}) == 3, "duplicate ODE-NEB NoF panel label")
    require(len(set(pairs)) == 3, "duplicate IVM/ODE-NEB comparison panel")
    return pairs


def invocation_record(
    raw_arguments: Sequence[str],
    process_argv: Sequence[str],
) -> Dict[str, object]:
    script = Path(__file__).resolve()
    return {
        "process_argv_exact": [str(value) for value in process_argv],
        "normalized_command": [
            str(Path(sys.executable).resolve()),
            str(script),
            *[str(value) for value in raw_arguments],
        ],
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "working_directory": str(Path.cwd().resolve()),
    }


def validate_audit(audit: Mapping[str, object], actual_nf: int) -> None:
    require(audit.get("consistent") is True, "selected producer NoF audit is inconsistent")
    for key in (
        "optimization_internal_nf",
        "backend_raw_joint_attempts",
        "backend_raw_joint_evaluations",
        "sum_image_joint_evaluations",
        "number_of_forces",
    ):
        require(int(audit.get(key, -1)) == actual_nf, f"audit counter mismatch: {key}")
    per_image = np.asarray(audit.get("per_image_joint_evaluations", []), dtype=np.int64)
    require(per_image.shape == (N_IMAGES,), "audit must contain 32 image counters")
    require(int(per_image[0]) == 0 and int(per_image[-1]) == 0, "fixed endpoints incurred NoF")
    require(int(per_image[1:-1].sum(dtype=np.int64)) == actual_nf, "internal-image NoF sum mismatch")
    require(int(audit.get("endpoint_getstate_attempts", -1)) == 0, "endpoint getState attempt detected")
    require(int(audit.get("endpoint_joint_ef_cache_misses", -1)) == 0, "endpoint cache miss detected")
    require(
        int(audit.get("endpoint_inherited_joint_ef_cache_entries", -1)) == 2,
        "two inherited endpoint E/F entries are required",
    )
    request_identity = np.asarray(audit.get("per_image_request_identity_ok", []), dtype=bool)
    query_identity = np.asarray(
        audit.get("per_image_query_outcome_identity_ok", []), dtype=bool
    )
    require(
        request_identity.shape == (N_IMAGES,) and bool(np.all(request_identity)),
        "request identity audit failed",
    )
    require(
        query_identity.shape == (N_IMAGES,) and bool(np.all(query_identity)),
        "query identity audit failed",
    )
    require(audit.get("discrete_all_returned_states_evaluated") is True, "unevaluated state returned")


def validate_registration(registration: Mapping[str, object]) -> None:
    require(registration.get("label") == SELECTED_LABEL, "selected registration label mismatch")
    require(int(registration.get("internal_nf_budget", -1)) == 9990, "selected budget is not 9990")
    require(float(registration.get("alpha", np.nan)) == 1.0, "selected alpha is not 1")
    require(float(registration.get("residual_ratio_r", np.nan)) == 0.1, "selected r is not 0.1")
    require(int(registration.get("inner_max_steps_m_max", -1)) == 5, "selected m_max is not 5")
    require(float(registration.get("physical_max_step_angstrom", np.nan)) == 1.0, "physical max step changed")
    require(float(registration.get("eta", np.nan)) == 1.1, "eta changed")
    require(int(registration.get("history", -1)) == 7, "history changed")
    require(
        registration.get("reparameterization")
        == "state_curve every outer iteration with per_image direct acceptance and recovery",
        "reparameterization identity changed",
    )
    require(isinstance(registration.get("input_sha256"), Mapping), "registration input closure missing")
    require(isinstance(registration.get("source_sha256"), Mapping), "registration source closure missing")


def validate_json_gates_before_selected_npz(
    contract_path: Path,
    contract: Mapping[str, object],
) -> Dict[str, object]:
    """Validate terminal gates before hashing or opening the selected NPZ."""

    require(contract.get("schema") == CONTRACT_SCHEMA, "unsupported evolution contract schema")
    require(contract.get("status") == FINAL_CONTRACT_STATUS, "contract is not a final frozen-input contract")
    identity = contract.get("figure_identity", {})
    require(isinstance(identity, Mapping), "figure identity is missing")
    require(identity.get("legacy_reproduction_claim") is False, "legacy exact-reproduction claim is forbidden")
    require(
        identity.get("scientific_identity") == "new Barnase--Barstar selected IR-IVM evolution",
        "figure scientific identity mismatch",
    )
    selected = contract.get("selected_run", {})
    require(isinstance(selected, Mapping), "selected_run is missing")

    # Resolve/hash JSON gates first.  selected_result is deliberately resolved
    # without reading or hashing until all terminal receipts pass below.
    verification_path = resolve_binding(
        contract_path, selected.get("terminal_verification", {}), "selected terminal verification"
    )
    audit_path = resolve_binding(contract_path, selected.get("audit", {}), "selected audit")
    registration_path = resolve_binding(
        contract_path, selected.get("registration", {}), "selected registration"
    )
    prefix_path = resolve_binding(
        contract_path, selected.get("exact_prefix_verification", {}), "selected exact-prefix verification"
    )
    result_path = resolve_binding(
        contract_path,
        selected.get("result", {}),
        "selected result",
        verify_hash=False,
    )

    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    require(verification.get("schema") == TERMINAL_SCHEMA, "unsupported terminal verification schema")
    require(verification.get("status") == "PASS", "selected terminal verification did not pass")
    require(verification.get("label") == SELECTED_LABEL, "terminal verification label mismatch")
    require(
        int(verification.get("registered_optimization_N_F_upper_bound", -1)) == 9990,
        "terminal verification does not bind the 9990 upper bound",
    )
    actual_nf = int(verification.get("actual_optimization_N_F", -1))
    require(0 < actual_nf <= 9990, "terminal actual NoF is outside the registered bound")
    require(int(verification.get("endpoint_optimization_N_F", -1)) == 0, "endpoint optimization NoF is not zero")
    if bool(verification.get("terminal_partial_tail_present", False)):
        require(
            verification.get("terminal_partial_tail_quarantined") is True,
            "terminal partial tail is not quarantined",
        )

    verified_result = receipt_binding_path(
        verification_path, verification, "result", verify_hash=False
    )
    verified_audit = receipt_binding_path(verification_path, verification, "audit")
    verified_registration = receipt_binding_path(verification_path, verification, "registration")
    _same_file(result_path, verified_result, "selected result")
    _same_file(audit_path, verified_audit, "selected audit")
    _same_file(registration_path, verified_registration, "selected registration")
    require(
        str(selected["result"]["sha256"]) == str(verification["bindings"]["result"]["sha256"]),
        "contract/result hash differs from the terminal receipt",
    )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    validate_audit(audit, actual_nf)
    validate_registration(registration)

    prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
    require(prefix.get("schema") == PREFIX_SCHEMA, "unsupported exact-prefix receipt schema")
    require(prefix.get("status") == "PASS_EXACT_PREFIX", "selected exact-prefix gate did not pass")
    require(prefix.get("tolerance_fallback_used") is False, "exact-prefix receipt used a tolerance fallback")
    prefix_result = receipt_binding_path(
        prefix_path, prefix, "selected_result", verify_hash=False
    )
    prefix_verification = receipt_binding_path(prefix_path, prefix, "selected_verification")
    _same_file(result_path, prefix_result, "exact-prefix selected result")
    _same_file(verification_path, prefix_verification, "exact-prefix terminal verification")
    require(
        str(prefix["bindings"]["selected_result"]["sha256"])
        == str(selected["result"]["sha256"]),
        "exact-prefix and contract result hashes differ",
    )

    # Only after the terminal and exact-prefix receipts pass may the selected
    # result bytes be read.  Hashing happens here, immediately before np.load.
    actual_result_sha = sha256_file(result_path)
    require(actual_result_sha == str(selected["result"]["sha256"]), "selected result SHA-256 mismatch")

    return {
        "identity": identity,
        "selected": selected,
        "verification": verification,
        "audit": audit,
        "registration": registration,
        "prefix": prefix,
        "paths": {
            "selected_result": result_path,
            "selected_audit": audit_path,
            "selected_registration": registration_path,
            "terminal_verification": verification_path,
            "exact_prefix_verification": prefix_path,
        },
        "actual_nf": actual_nf,
    }


def _as_band(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2:
        require(array.shape[1] % 3 == 0, f"flat path dimension is not divisible by 3: {array.shape}")
        return array.reshape(array.shape[0], -1, 3)
    require(array.ndim == 3 and array.shape[2] == 3, f"unexpected path shape: {array.shape}")
    return array


def collect_selected_committed_paths(
    saved: Mapping[str, np.ndarray],
    verification: Mapping[str, object],
    actual_nf: int,
) -> Dict[int, Dict[str, object]]:
    require(str(scalar(saved, "run_label")) == SELECTED_LABEL, "selected result label mismatch")
    require(float(scalar(saved, "alpha")) == 1.0, "selected result alpha mismatch")
    require(float(scalar(saved, "discrete_residual_ratio")) == 0.1, "selected result r mismatch")
    require(int(scalar(saved, "discrete_inner_max_steps")) == 5, "selected result m_max mismatch")
    for key in (
        "number_of_forces",
        "successful_joint_ef_evaluations",
        "optimization_internal_nf",
        "total_internal_nf_consumed",
    ):
        require(int(scalar(saved, key)) == actual_nf, f"selected result counter mismatch: {key}")
    require(int(scalar(saved, "total_internal_nf_budget")) == 9990, "selected result budget mismatch")
    require(bool(scalar(saved, "total_internal_nf_budget_no_overshoot")), "selected result overshot its budget")
    require(
        int(scalar(saved, "total_internal_nf_remaining")) == 9990 - actual_nf,
        "selected result remaining-budget mismatch",
    )
    require(
        bool(scalar(saved, "total_internal_nf_endpoint_evaluations_excluded")),
        "selected result does not exclude fixed endpoints",
    )
    require(
        int(scalar(saved, "total_internal_nf_endpoint_getstate_attempts_expected")) == 0,
        "selected result expects endpoint calls",
    )
    partial_tail = bool(scalar(saved, "total_internal_nf_terminal_partial_tail_present"))
    quarantined_tail = bool(
        scalar(saved, "total_internal_nf_terminal_partial_tail_quarantined")
    )
    require(not partial_tail or quarantined_tail, "selected result has an unquarantined partial tail")
    if actual_nf < 9990:
        require(0 < 9990 - actual_nf < N_INTERNAL, "sub-budget stop is not a 30-image atomic remainder")
        require(
            str(scalar(saved, "total_internal_nf_budget_stop_reason"))
            == "total_internal_joint_budget_insufficient_for_full_band",
            "sub-budget stop has a non-budget reason",
        )
        require(
            int(scalar(saved, "total_internal_nf_full_band_preflight_zero_step_stops")) > 0,
            "sub-budget stop lacks a zero-step full-band preflight",
        )
        require(
            int(scalar(saved, "total_internal_nf_complete_band_budget_stop_count")) > 0,
            "sub-budget stop lacks a complete-band budget stop",
        )
        require(partial_tail and quarantined_tail, "sub-budget terminal tail is not quarantined")
    require(
        int(scalar(saved, "total_internal_nf_last_complete_state_nf")) <= actual_nf,
        "last complete state exceeds actual NoF",
    )

    calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
    committed = np.asarray(saved["reparam_committed"], dtype=bool)
    counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
    stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
    stored_paths = np.asarray(saved["stored_paths"], dtype=np.float64)
    initial_path = np.asarray(saved["initial_path"], dtype=np.float64)
    require(calls.shape == committed.shape, "selected history/commit shape mismatch")
    require(counters.shape == (calls.size, N_IMAGES), "selected counter history shape mismatch")
    require(stored_paths.shape[0] == stored_calls.size, "selected stored path/call mismatch")
    require(_as_band(initial_path).shape[0] == N_IMAGES, "selected initial path is not 32 images")
    initial_matches = np.flatnonzero(stored_calls == 1)
    require(initial_matches.size == 1, "selected result lacks a unique stored initial path")
    require(
        np.array_equal(np.asarray(stored_paths[int(initial_matches[0])]), initial_path),
        "stored call-1 path differs from initial_path",
    )

    verifier_rows = {
        int(row["call_per_image"]): row
        for row in verification.get("post_initial_reparam_committed_stored_paths", [])
    }
    require(verifier_rows, "terminal receipt contains no committed stored path")
    eligible_receipt_calls = {
        int(value) for value in verification.get("eligible_stored_calls_per_image", [])
    }
    require(
        eligible_receipt_calls == {1, *verifier_rows},
        "terminal eligible stored-call set is inconsistent",
    )
    result: Dict[int, Dict[str, object]] = {}
    seen_calls = set()
    for stored_index, call_value in enumerate(stored_calls):
        call = int(call_value)
        matching = np.flatnonzero(calls == call)
        require(matching.size == 1, f"stored call {call} is not unique")
        history_index = int(matching[0])
        if call == 1 or not bool(committed[history_index]):
            continue
        require(call in verifier_rows, f"committed stored call {call} is absent from terminal receipt")
        path = np.asarray(stored_paths[stored_index], dtype=np.float64)
        nf = int(counters[history_index, 1:-1].sum(dtype=np.int64))
        receipt_row = verifier_rows[call]
        require(nf == int(receipt_row["actual_optimization_N_F"]), f"NoF mismatch at committed call {call}")
        verifier_hash = verifier_dtype_shape_sha256(path)
        require(verifier_hash == str(receipt_row["path_sha256"]), f"verifier path hash mismatch at NoF={nf}")
        require(nf not in result, f"duplicate committed path NoF={nf}")
        result[nf] = {
            "call_per_image": call,
            "history_index": history_index,
            "optimization_N_F": nf,
            "path": path,
            "verifier_dtype_shape_sha256": verifier_hash,
            "canonical_le_float64_sha256": canonical_le_float64_sha256(path),
        }
        seen_calls.add(call)
    require(seen_calls == set(verifier_rows), "eligible committed call set differs from terminal receipt")
    return {
        -1: {
            "call_per_image": 1,
            "optimization_N_F": int(counters[0, 1:-1].sum(dtype=np.int64)),
            "path": initial_path,
            "verifier_dtype_shape_sha256": verifier_dtype_shape_sha256(initial_path),
            "canonical_le_float64_sha256": canonical_le_float64_sha256(initial_path),
        },
        **result,
    }


def load_ode_neb_snapshot(
    contract_path: Path,
    record: Mapping[str, object],
    ledger_rows: Mapping[int, Mapping[str, object]],
) -> Tuple[Path, np.ndarray, Dict[str, object]]:
    nf = int(record.get("optimization_N_F", -1))
    require(nf > 0 and nf % N_INTERNAL == 0, f"ODE-NEB NoF={nf} is not a 30-image multiple")
    require(nf in ledger_rows, f"ODE-NEB NoF={nf} is absent from the official paid-path ledger")
    ledger_row = ledger_rows[nf]
    archive = resolve_binding(
        contract_path,
        record.get("path_archive", {}),
        f"ODE-NEB NoF={nf} path archive",
    )
    key = str(record.get("path_key", "coordinates30_A"))
    with np.load(archive, allow_pickle=False) as saved:
        require(key in saved.files, f"ODE-NEB NoF={nf} archive lacks {key}")
        data = np.asarray(saved[key], dtype=np.float64)
        if data.shape[0] == N_INTERNAL:
            endpoint_keys = record.get("endpoint_keys", ["endpoint0_A", "endpoint1_A"])
            require(len(endpoint_keys) == 2, f"ODE-NEB NoF={nf} needs two endpoint keys")
            require(all(str(name) in saved.files for name in endpoint_keys), f"ODE-NEB NoF={nf} endpoints missing")
            endpoint0 = np.asarray(saved[str(endpoint_keys[0])], dtype=np.float64)
            endpoint1 = np.asarray(saved[str(endpoint_keys[1])], dtype=np.float64)
            band = np.concatenate((endpoint0[None], _as_band(data), endpoint1[None]), axis=0)
        else:
            band = _as_band(data)
        require(band.shape[0] == N_IMAGES, f"ODE-NEB NoF={nf} path is not 32 images")
        embedded = str(np.asarray(saved["path32_sha256"]).item()) if "path32_sha256" in saved.files else None
    official_hash = ode_ledger_array_sha256(band)
    expected_official = str(record.get("official_path32_sha256", ""))
    require(len(expected_official) == 64, f"ODE-NEB NoF={nf} lacks official path identity")
    require(official_hash == expected_official, f"ODE-NEB NoF={nf} official path hash mismatch")
    if embedded is not None:
        require(embedded == official_hash, f"ODE-NEB NoF={nf} embedded path identity mismatch")
    ledger_archive = Path(str(ledger_row.get("coordinate_archive", ""))).expanduser().resolve()
    _same_file(archive, ledger_archive, f"ODE-NEB NoF={nf} official ledger archive")
    require(
        str(ledger_row.get("coordinate_archive_sha256", "")) == sha256_file(archive),
        f"ODE-NEB NoF={nf} archive hash differs from the official ledger",
    )
    require(
        str(ledger_row.get("official_path32_sha256", "")) == official_hash,
        f"ODE-NEB NoF={nf} path hash differs from the official ledger",
    )
    expected_batch = int(record.get("batch_id", -1))
    require(expected_batch == int(ledger_row.get("batch_id", -2)), f"ODE-NEB NoF={nf} batch mismatch")
    expected_role = str(record.get("ledger_state_role", ""))
    ledger_role = str(ledger_row.get("state_role", ""))
    require(expected_role in {"current_initial", "trial"}, f"ODE-NEB NoF={nf} state role is not explicit")
    require(expected_role == ledger_role, f"ODE-NEB NoF={nf} state-role mismatch")
    return archive, band, {
        "optimization_N_F": nf,
        "batch_id": expected_batch,
        "official_paid_path_state_role": ledger_role,
        "file_sha256": sha256_file(archive),
        "official_ode_ledger_sha256": official_hash,
        "canonical_le_float64_sha256": canonical_le_float64_sha256(band),
    }


def load_ode_neb_paid_path_ledger(
    contract_path: Path,
    record: Mapping[str, object],
) -> Tuple[Path, Dict[int, Mapping[str, object]], Dict[str, Path]]:
    ledger_path = resolve_binding(
        contract_path, record, "official ODE-NEB all-paid-path ledger"
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    require(ledger.get("schema") == ODE_LEDGER_SCHEMA, "unsupported ODE-NEB paid-path ledger schema")
    require(
        ledger.get("status") == "FROZEN_BEFORE_ANY_NEW_ALL_BATCH_ARC500_ENERGY_CALL",
        "ODE-NEB paid-path ledger is not frozen",
    )
    source_scope = ledger.get("source_scope", {})
    require(
        source_scope.get("run_label") == "bbcr30_neb_ode1e6_k0p1_prod_20260814_r1",
        "ODE-NEB paid-path ledger run identity mismatch",
    )
    require(int(source_scope.get("fixed_endpoint_optimization_N_F", -1)) == 0, "ODE-NEB ledger charges endpoints")
    require(int(source_scope.get("paid_path_count", -1)) == 333, "ODE-NEB ledger path count mismatch")
    paths = ledger.get("paths", [])
    require(len(paths) == 333, "ODE-NEB paid-path table is incomplete")
    rows: Dict[int, Mapping[str, object]] = {}
    for row in paths:
        nf = int(row.get("optimization_N_F", -1))
        require(nf > 0 and nf % N_INTERNAL == 0, f"invalid ODE-NEB ledger NoF={nf}")
        require(nf not in rows, f"duplicate ODE-NEB ledger NoF={nf}")
        rows[nf] = row
    require(sorted(rows) == list(range(30, 9991, 30)), "ODE-NEB ledger NoF sequence is incomplete")

    # Bind the official optimization provenance underneath the frozen mapping,
    # not only the mapping JSON itself.
    nested_paths: Dict[str, Path] = {}
    for role in ("optimization_audit", "optimization_complete", "optimization_wal"):
        nested_paths[role] = receipt_binding_path(ledger_path, ledger, role)
    return ledger_path, rows, nested_paths


class CalibratedPathCV:
    """Project Cartesian paths onto the old calibrated BB path CV."""

    def __init__(self, source: Path):
        with np.load(source, allow_pickle=False) as saved:
            self.center_path = _as_band(saved["extended_center_path"])
            self.labels = np.asarray(saved["extended_reference_labels"], dtype=float)
            self.center_features = np.asarray(saved["extended_center_features"], dtype=float)
            self.normals = np.asarray(saved["extended_normals"], dtype=float)
            self.path_lambda = float(saved["extended_path_lambda"])
            self.raw_bound = float(saved["raw_bound_progress"])
            self.progress_scale = float(saved["progress_scale"])
            self.baseline_s = np.asarray(saved["baseline_s"], dtype=float)
            self.baseline_n = np.asarray(saved["baseline_raw_normal"], dtype=float)
            self.map_old_s = np.asarray(saved["progress_map_old_s_nodes"], dtype=float)
            self.map_display_s = np.asarray(saved["progress_map_display_s_nodes"], dtype=float)
            self.map_left_slope = float(saved["progress_map_forward_left_slope"])
            self.map_right_slope = float(saved["progress_map_forward_right_slope"])
            self.heavy = np.asarray(saved["heavy_atom_indices"], dtype=int)
            self.heavy_weights = np.asarray(saved["heavy_atom_weights"], dtype=float)
            self.sqrt_heavy_weights = np.asarray(saved["sqrt_heavy_atom_weights"], dtype=float)
            self.chain_a_heavy = np.asarray(saved["A_heavy"], dtype=int)

        bound_index = int(np.argmin(np.abs(self.labels)))
        self.reference_a = self.center_path[bound_index, self.chain_a_heavy]
        heavy_lookup = {atom: local for local, atom in enumerate(self.heavy)}
        local_a = np.asarray([heavy_lookup[atom] for atom in self.chain_a_heavy], dtype=int)
        self.reference_a_weights = self.heavy_weights[local_a].copy()
        self.reference_a_weights /= self.reference_a_weights.sum()
        self.reference_a_center = np.sum(
            self.reference_a * self.reference_a_weights[:, None], axis=0
        )
        self.reference_a_centered = self.reference_a - self.reference_a_center
        self.baseline = CubicSpline(
            self.baseline_s, self.baseline_n, bc_type="natural", extrapolate=True
        )
        self.progress_map = PchipInterpolator(self.map_old_s, self.map_display_s)

    def align(self, position: np.ndarray) -> np.ndarray:
        pos = np.asarray(position, dtype=float).reshape(-1, 3)
        mobile = pos[self.chain_a_heavy]
        mobile_center = np.sum(mobile * self.reference_a_weights[:, None], axis=0)
        mobile_centered = mobile - mobile_center
        covariance = (
            mobile_centered * self.reference_a_weights[:, None]
        ).T @ self.reference_a_centered
        u, _, vt = np.linalg.svd(covariance)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        return (pos - mobile_center) @ rotation + self.reference_a_center

    def old_to_display(self, old_s: float) -> float:
        if old_s < self.map_old_s[0]:
            return float(
                self.map_display_s[0] + self.map_left_slope * (old_s - self.map_old_s[0])
            )
        if old_s > self.map_old_s[-1]:
            return float(
                self.map_display_s[-1] + self.map_right_slope * (old_s - self.map_old_s[-1])
            )
        return float(self.progress_map(old_s))

    def project_position(self, position: np.ndarray) -> np.ndarray:
        aligned = self.align(position)
        feature = (
            aligned[self.heavy] * self.sqrt_heavy_weights[:, None]
        ).reshape(-1)
        displacement = feature[None, :] - self.center_features
        squared_distances = np.sum(displacement * displacement, axis=1)
        log_weights = -self.path_lambda * squared_distances
        weights = np.exp(log_weights - np.max(log_weights))
        probabilities = weights / weights.sum()
        raw_progress = float(np.sum(probabilities * self.labels))
        old_s = (raw_progress - self.raw_bound) * self.progress_scale
        local_normals = np.sum(displacement * self.normals, axis=1)
        raw_normal = float(np.sum(probabilities * local_normals))
        signed_normal = raw_normal - float(self.baseline(old_s))
        return np.asarray([self.old_to_display(old_s), signed_normal], dtype=float)

    def project_band(self, band: np.ndarray) -> np.ndarray:
        return np.asarray([self.project_position(item) for item in _as_band(band)])


class LogQuadraticNorm(Normalize):
    """Chignolin-paper style: log1p allocation then concave quadratic."""

    def __init__(self, vmin: float, vmax: float, scale: float = 40.0, curvature: float = 2.5):
        super().__init__(vmin=vmin, vmax=vmax, clip=True)
        self.scale = float(scale)
        self.curvature = float(curvature)
        self._log_span = np.log1p((self.vmax - self.vmin) / self.scale)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        raw = np.clip(
            (result.data - self.vmin) / self.scale,
            0.0,
            (self.vmax - self.vmin) / self.scale,
        )
        logarithmic = np.log1p(raw) / self._log_span
        mapped = 1.0 - np.power(1.0 - logarithmic, self.curvature)
        output = np.ma.array(mapped, mask=result.mask, copy=False)
        return output[0] if is_scalar else output

    def inverse(self, value):
        mapped = np.asarray(value, dtype=float)
        logarithmic = 1.0 - np.power(
            np.clip(1.0 - mapped, 0.0, 1.0), 1.0 / self.curvature
        )
        return self.vmin + self.scale * np.expm1(logarithmic * self._log_span)


def load_surface(path: Path, surface_format: str) -> Dict[str, np.ndarray]:
    require(
        surface_format == "legacy_calibrated_cv_multistart_scattered",
        "only the registered old calibrated-CV multistart PES is supported",
    )
    with np.load(path, allow_pickle=False) as saved:
        required = {"achieved_s", "achieved_n_A", "energy_relative_bound_kcal", "accepted"}
        require(required.issubset(saved.files), "old BB PES archive has an unsupported layout")
        s = np.asarray(saved["achieved_s"], dtype=float)
        n = np.asarray(saved["achieved_n_A"], dtype=float)
        energy = np.asarray(saved["energy_relative_bound_kcal"], dtype=float)
        accepted = np.asarray(saved["accepted"], dtype=bool)
    mask = accepted & np.isfinite(s) & np.isfinite(n) & np.isfinite(energy)
    require(int(mask.sum()) >= 12, "old BB PES has too few accepted finite samples")
    return {
        "s": s[mask],
        "n": n[mask],
        "energy_kcal_per_mol": np.maximum(energy[mask], 0.0),
    }


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def compact_nf(value: int) -> str:
    return str(value)


def plot_evolution(
    surface: Mapping[str, np.ndarray],
    initial: np.ndarray,
    panels: Sequence[Mapping[str, object]],
    axis_limits: Mapping[str, Sequence[float]],
) -> plt.Figure:
    configure_style()
    energy = np.asarray(surface["energy_kcal_per_mol"], dtype=float)
    vmax = float(np.max(energy))
    require(vmax > 0.0, "PES energy range is degenerate")
    norm = LogQuadraticNorm(0.0, vmax, scale=40.0, curvature=2.5)
    filled_levels = norm.inverse(np.linspace(0.0, 1.0, 41))
    line_levels = norm.inverse(np.linspace(0.08, 0.95, 16))
    triangulation = mtri.Triangulation(surface["s"], surface["n"])

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.38), sharex=True, sharey=True)
    contour = None
    for panel_index, axis in enumerate(axes):
        contour = axis.tricontourf(
            triangulation,
            energy,
            levels=filled_levels,
            cmap="cividis",
            norm=norm,
            antialiased=True,
            extend="max",
        )
        axis.tricontour(
            triangulation,
            energy,
            levels=line_levels,
            colors="#F3F3F3",
            linewidths=0.32,
            alpha=0.58,
        )
        if panel_index == 0:
            axis.plot(
                initial[:, 0],
                initial[:, 1],
                color=INITIAL_COLOR,
                linewidth=1.55,
                marker="o",
                markersize=2.15,
                markevery=2,
                label="Initial path",
                zorder=5,
            )
            title = "Initial path"
        else:
            row = panels[panel_index - 1]
            ode_path = np.asarray(row["ode_neb_projected"], dtype=float)
            ivm_path = np.asarray(row["ivm_projected"], dtype=float)
            axis.plot(
                ode_path[:, 0],
                ode_path[:, 1],
                color=ODE_NEB_COLOR,
                linewidth=1.45,
                marker="o",
                markersize=2.15,
                markevery=2,
                label="ODE-NEB",
                zorder=5,
            )
            axis.plot(
                ivm_path[:, 0],
                ivm_path[:, 1],
                color=IVM_COLOR,
                linewidth=1.75,
                marker="s",
                markersize=2.15,
                markevery=2,
                label="IVM",
                zorder=6,
            )
            ivm_nf = int(row["ivm_optimization_N_F"])
            ode_nf = int(row["ode_neb_optimization_N_F"])
            title = (
                rf"IVM $N_F={compact_nf(ivm_nf)}$" + "\n"
                rf"ODE-NEB $N_F={compact_nf(ode_nf)}$"
            )
        axis.set_title(title, pad=4.0)
        axis.set_xlim(float(axis_limits["s"][0]), float(axis_limits["s"][1]))
        axis.set_ylim(float(axis_limits["n_A"][0]), float(axis_limits["n_A"][1]))
        axis.set_xticks((0.0, 0.5, 1.0))
        axis.tick_params(direction="out", length=2.8)
        axis.text(
            0.035,
            0.965,
            chr(97 + panel_index),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )
    axes[0].set_ylabel(r"Signed path-normal $n$ ($\AA$)")
    axes[-1].legend(
        frameon=True,
        facecolor="white",
        edgecolor="#B3B3B3",
        framealpha=0.90,
        loc="upper right",
        handlelength=2.0,
    )
    fig.supxlabel(r"Path progress $s$", x=0.48, y=0.035)
    color_axis = fig.add_axes([0.905, 0.23, 0.014, 0.67])
    colorbar = fig.colorbar(contour, cax=color_axis)
    ticks = [value for value in (0.0, 40.0, 240.0, 1.0e3, 1.0e4, 1.0e6) if value <= vmax]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(
        ["0" if value == 0 else f"{value:g}" if value < 1000 else rf"$10^{{{int(np.log10(value))}}}$" for value in ticks]
    )
    colorbar.set_label(r"$E_{\mathrm{0K}}-E_{\mathrm{bound}}$ (kcal mol$^{-1}$)")
    fig.subplots_adjust(left=0.075, right=0.885, bottom=0.23, top=0.90, wspace=0.08)
    return fig


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def render_pdf_exclusive(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".tmp.pdf", dir=str(output.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        fig.savefig(temporary, format="pdf", bbox_inches="tight")
        payload = temporary.read_bytes()
        require(payload.startswith(b"%PDF-"), "matplotlib did not create a PDF")
        write_exclusive(output, payload)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_inputs(contract_path: Path, output: Path) -> Dict[str, object]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    gates = validate_json_gates_before_selected_npz(contract_path, contract)
    destination, provenance_path = validate_output_path(
        output, gates["identity"], contract.get("outputs", {})
    )
    plot_contract = contract.get("plot", {})
    require(isinstance(plot_contract, Mapping), "plot contract is missing")
    method_labels = plot_contract.get("method_labels", [])
    require(
        method_labels == ["IVM", "ODE-NEB"] and len(set(method_labels)) == 2,
        "method labels must be the unique paper labels IVM and ODE-NEB",
    )
    panel_pairs = validate_panel_contract(contract.get("panels", []))

    inputs = contract.get("coordinate_surface", {})
    require(isinstance(inputs, Mapping), "coordinate_surface is missing")
    cv_path = resolve_binding(contract_path, inputs.get("cv_definition", {}), "old BB CV definition")
    pes_path = resolve_binding(contract_path, inputs.get("pes_data", {}), "old BB PES data")
    style_path = resolve_binding(
        contract_path,
        contract.get("style_reference_pdf", {}),
        "legacy evolution style reference PDF",
    )
    ode_ledger_path, ode_ledger_rows, ode_ledger_sources = load_ode_neb_paid_path_ledger(
        contract_path, contract.get("ode_neb_paid_path_ledger", {})
    )

    ode_records: Dict[int, Dict[str, object]] = {}
    ode_paths: Dict[int, np.ndarray] = {}
    for record in contract.get("ode_neb_snapshots", []):
        archive, path, metadata = load_ode_neb_snapshot(
            contract_path, record, ode_ledger_rows
        )
        nf = int(metadata["optimization_N_F"])
        require(nf not in ode_paths, f"duplicate ODE-NEB snapshot NoF={nf}")
        ode_paths[nf] = path
        ode_records[nf] = {"path": str(archive), **metadata}
    require(ode_paths, "contract contains no ODE-NEB path snapshots")

    # This is the first selected-result np.load in the entire program and it
    # occurs only after validate_json_gates_before_selected_npz has passed.
    with np.load(gates["paths"]["selected_result"], allow_pickle=False) as saved:
        selected_paths = collect_selected_committed_paths(
            saved, gates["verification"], int(gates["actual_nf"])
        )

    cv = CalibratedPathCV(cv_path)
    initial_projected = cv.project_band(selected_paths[-1]["path"])
    panels: List[Dict[str, object]] = []
    for ivm_nf, ode_nf in panel_pairs:
        require(ivm_nf in selected_paths and ivm_nf != -1, f"panel requests ineligible IVM NoF={ivm_nf}")
        require(ode_nf in ode_paths, f"panel requests unavailable ODE-NEB NoF={ode_nf}")
        ivm_band = _as_band(selected_paths[ivm_nf]["path"])
        ode_band = _as_band(ode_paths[ode_nf])
        require(ivm_band.shape == ode_band.shape, f"panel path shape mismatch at IVM NoF={ivm_nf}")
        require(
            np.array_equal(ivm_band[[0, -1]], ode_band[[0, -1]]),
            f"panel endpoints differ at IVM NoF={ivm_nf}/ODE-NEB NoF={ode_nf}",
        )
        panels.append(
            {
                "ivm_optimization_N_F": ivm_nf,
                "ode_neb_optimization_N_F": ode_nf,
                "ivm_projected": cv.project_band(ivm_band),
                "ode_neb_projected": cv.project_band(ode_band),
                "ivm_hashes": {
                    "verifier_dtype_shape_sha256": selected_paths[ivm_nf][
                        "verifier_dtype_shape_sha256"
                    ],
                    "canonical_le_float64_sha256": selected_paths[ivm_nf][
                        "canonical_le_float64_sha256"
                    ],
                },
                "ode_neb_hashes": {
                    "official_ode_ledger_sha256": ode_records[ode_nf][
                        "official_ode_ledger_sha256"
                    ],
                    "canonical_le_float64_sha256": ode_records[ode_nf][
                        "canonical_le_float64_sha256"
                    ],
                },
            }
        )

    surface = load_surface(pes_path, str(inputs.get("surface_format", "")))
    axis_limits = plot_contract.get("axis_limits", {})
    require(
        set(axis_limits) == {"s", "n_A"}
        and all(len(axis_limits[key]) == 2 for key in ("s", "n_A")),
        "plot axis limits are not fully registered",
    )
    return {
        "contract": contract,
        "gates": gates,
        "output": destination,
        "provenance_path": provenance_path,
        "cv_path": cv_path,
        "pes_path": pes_path,
        "style_path": style_path,
        "ode_ledger_path": ode_ledger_path,
        "ode_ledger_sources": ode_ledger_sources,
        "surface": surface,
        "initial_projected": initial_projected,
        "panels": panels,
        "ode_records": ode_records,
        "axis_limits": axis_limits,
    }


def provenance_payload(prepared: Mapping[str, object], output: Path) -> Dict[str, object]:
    gates = prepared["gates"]
    script = Path(__file__).resolve()
    contract_path = prepared["contract_path"]
    paths = gates["paths"]
    return {
        "schema": "bb-irivm-evolution-figure-provenance-v1",
        "status": "FINAL_NEW_BB_IRIVM_EVOLUTION_RENDERED",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scientific_identity": "new Barnase--Barstar selected IR-IVM evolution",
        "legacy_reproduction_claim": False,
        "legacy_reference_role": "style and calibrated coordinate/PES context only",
        "no_openmm_context_created": True,
        "invocation": prepared["invocation"],
        "outputs": {
            "pdf": str(output),
            "provenance_json": str(prepared["provenance_path"]),
        },
        "optimization_Number_of_Forces_definition": (
            "all physical joint getState attempts for 30 internal images; "
            "the two fixed endpoints contribute zero optimization N_F"
        ),
        "bindings": {
            "plot_script": {"path": str(script), "sha256": sha256_file(script)},
            "figure_contract": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
            },
            "old_cv_definition": {
                "path": str(prepared["cv_path"]),
                "sha256": sha256_file(prepared["cv_path"]),
            },
            "old_pes_data": {
                "path": str(prepared["pes_path"]),
                "sha256": sha256_file(prepared["pes_path"]),
            },
            "legacy_style_reference_pdf": {
                "path": str(prepared["style_path"]),
                "sha256": sha256_file(prepared["style_path"]),
            },
            "official_ode_neb_paid_path_ledger": {
                "path": str(prepared["ode_ledger_path"]),
                "sha256": sha256_file(prepared["ode_ledger_path"]),
            },
            **{
                f"official_ode_neb_{role}": {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for role, path in prepared["ode_ledger_sources"].items()
            },
            **{
                role: {"path": str(path), "sha256": sha256_file(path)}
                for role, path in paths.items()
            },
            "output_pdf": {"path": str(output), "sha256": sha256_file(output)},
        },
        "selected_setting": {
            "alpha": 1.0,
            "r": 0.1,
            "m_max": 5,
            "registered_optimization_N_F_upper_bound": 9990,
            "actual_optimization_N_F": int(gates["actual_nf"]),
            "eligible_path_policy": "post-initial reparameterization-committed stored paths only",
            "terminal_uncommitted_or_partial_tail_policy": "excluded",
        },
        "panel_paths": [
            {
                key: value
                for key, value in panel.items()
                if key not in {"ivm_projected", "ode_neb_projected"}
            }
            for panel in prepared["panels"]
        ],
        "ode_neb_snapshot_files": list(prepared["ode_records"].values()),
        "path_hash_conventions": {
            "verifier_dtype_shape_sha256": (
                "SHA256(dtype text + NUL + JSON shape + NUL + native contiguous bytes); "
                "used only against the terminal long-member receipt"
            ),
            "canonical_le_float64_sha256": (
                "SHA256(contiguous little-endian float64 bytes); used by exact-prefix/arc500 identity"
            ),
            "official_ode_ledger_sha256": (
                "SHA256(canonical dtype/shape JSON + NUL + little-endian float64 bytes)"
            ),
        },
        "coordinate_definition": (
            "old calibrated path progress s and signed calibrated path-normal n in Angstrom"
        ),
        "surface": {
            "kind": "old accepted multistart 0 K BB PES samples",
            "color_map": "cividis",
            "color_transform": "log1p(scale=40) then 1-(1-t)^2.5",
        },
        "plot_style": {
            "layout": "1x4: initial path plus three IVM/ODE-NEB comparisons",
            "ivm": {"label": "IVM", "color": IVM_COLOR, "line": "solid", "marker": "square"},
            "ode_neb": {
                "label": "ODE-NEB",
                "color": ODE_NEB_COLOR,
                "line": "solid",
                "marker": "circle",
            },
            "axis_limits": prepared["axis_limits"],
        },
    }


def main(argv: Sequence[str] = None) -> None:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    process_argv = list(sys.argv if argv is None else [str(Path(__file__).resolve()), *raw_arguments])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and project every frozen input but create no PDF/JSON",
    )
    args = parser.parse_args(raw_arguments)
    contract_path = args.contract.expanduser().resolve(strict=True)
    prepared = prepare_inputs(contract_path, args.output)
    prepared["contract_path"] = contract_path
    prepared["invocation"] = invocation_record(raw_arguments, process_argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "PASS_DRY_RUN_NO_OUTPUT_CREATED",
                    "actual_optimization_N_F": int(prepared["gates"]["actual_nf"]),
                    "panels": [
                        {
                            "ivm_optimization_N_F": int(row["ivm_optimization_N_F"]),
                            "ode_neb_optimization_N_F": int(row["ode_neb_optimization_N_F"]),
                        }
                        for row in prepared["panels"]
                    ],
                },
                sort_keys=True,
            )
        )
        return

    fig = plot_evolution(
        prepared["surface"],
        prepared["initial_projected"],
        prepared["panels"],
        prepared["axis_limits"],
    )
    pdf_created = False
    try:
        render_pdf_exclusive(fig, prepared["output"])
        pdf_created = True
    finally:
        plt.close(fig)
    try:
        provenance = provenance_payload(prepared, prepared["output"])
        write_exclusive(
            prepared["provenance_path"],
            (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    except Exception:
        # This PDF was created exclusively by this invocation.  Roll it back
        # if its mandatory same-stem provenance sidecar cannot be committed.
        if pdf_created and prepared["output"].exists():
            prepared["output"].unlink()
        raise
    print(f"Saved {prepared['output']}")
    print(f"Saved {prepared['provenance_path']}")


if __name__ == "__main__":
    main()
