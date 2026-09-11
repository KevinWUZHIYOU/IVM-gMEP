#!/usr/bin/env python3
"""Fail-closed, zero-OpenMM verifier for the selected IR-IVM legacy replay.

The replay exists only to retain a prospectively fixed set of current paths.
This verifier proves that its online history through call 50 is an exact prefix
of the sealed selected 9990-N_F trajectory, checks the force ledger and fixed
endpoints, and emits the only four newly reportable IVM rows.  The terminal
uncommitted charged tail is explicitly excluded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "bb-irivm-selected-legacy-replay-verification-v1"
REGISTRATION_SCHEMA = "bb-irivm-selected-output-only-legacy-replay-v1"
AUTHORIZATION_SCHEMA = "bb-irivm-selected-output-only-legacy-replay-authorization-v1"
AMENDMENT_SCHEMA = "bb-irivm-reporting-intermediate-states-amendment-v1"
AMENDMENT_STATUS = "FROZEN_AFTER_REPLAY_LAUNCH_BEFORE_ANY_NEW_REPORTING_METRIC_CALL"
LABEL = "bb_irivm_a1e0_r01_m05_inf1500_legacy_replay_r1"
FIXED_CALLS = (1, 2, 5, 10, 20, 50)
NEW_CALLS = (2, 5, 10, 50)
EXPECTED_NF = {1: 30, 2: 60, 5: 150, 10: 300, 20: 429, 50: 1329}
EXPECTED_TERMINAL_CALL = 55
EXPECTED_TERMINAL_NF = 1479
EXPECTED_BUDGET = 1500
EXPECTED_LAST_COMMITTED_CALL = 20
EXPECTED_LAST_COMMITTED_NF = 429
EXPECTED_TERMINAL_TAIL_NF = EXPECTED_TERMINAL_NF - EXPECTED_LAST_COMMITTED_NF
EXPECTED_REMAINING_NF = EXPECTED_BUDGET - EXPECTED_TERMINAL_NF
EXCLUDED_HISTORY_KEYS = frozenset({"wall_seconds"})
REQUIRED_EXACT_HISTORY_KEYS = frozenset(
    {
        "calls_per_image",
        "counter_vectors",
        "energy_request_vectors",
        "force_request_vectors",
        "i_perp",
        "reparam_committed",
        "total_internal_joint_evaluations",
    }
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_float64_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.dtype("<f8")))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def exact_array_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    first = np.asarray(left)
    second = np.asarray(right)
    return (
        first.shape == second.shape
        and first.dtype.str == second.dtype.str
        and np.ascontiguousarray(first).tobytes(order="C")
        == np.ascontiguousarray(second).tobytes(order="C")
    )


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def scalar(saved: Mapping[str, np.ndarray], key: str):
    require(key in saved, f"result lacks scalar {key}")
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def file_binding(path: Path) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def validate_binding(record: Mapping[str, object], role: str) -> Path:
    require(isinstance(record, Mapping), f"{role} binding is not an object")
    path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
    require(sha256_file(path) == str(record.get("sha256", "")), f"{role} binding drift")
    return path


def write_exclusive_json(path: Path, payload: dict) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def option_value(argv: Sequence[str], flag: str) -> str:
    values = [index for index, token in enumerate(argv) if token == flag]
    require(len(values) == 1, f"runtime argv must contain one {flag}")
    index = values[0]
    require(index + 1 < len(argv), f"runtime argv lacks value for {flag}")
    return str(argv[index + 1])


def validate_contracts(
    amendment_path: Path,
    registration_path: Path,
    authorization_path: Path,
    reference_result_path: Path,
) -> tuple[dict, dict, dict]:
    amendment = read_json(amendment_path)
    registration = read_json(registration_path)
    authorization = read_json(authorization_path)
    require(amendment.get("schema") == AMENDMENT_SCHEMA, "reporting amendment schema mismatch")
    require(amendment.get("status") == AMENDMENT_STATUS, "reporting amendment status mismatch")
    require(amendment.get("execution_authorized_by_this_file") is False, "amendment may not authorize replay")
    require(registration.get("schema") == REGISTRATION_SCHEMA, "replay registration schema mismatch")
    require(
        registration.get("status") == "FROZEN_BEFORE_FIRST_REPLAY_PHYSICAL_CALL",
        "replay registration was not prospectively frozen",
    )
    require(registration.get("label") == LABEL, "replay label mismatch")
    replay = registration.get("replay", {})
    require(replay.get("path_storage_mode") == "legacy", "replay storage is not legacy")
    require(replay.get("path_storage_affects_online_algorithm") is False, "storage is not output-only")
    require(int(replay.get("total_internal_optimization_N_F_budget", -1)) == EXPECTED_BUDGET, "replay budget mismatch")
    require(tuple(replay.get("prospective_reporting_calls", ())) == FIXED_CALLS, "fixed replay calls drift")
    runtime_argv = [str(value) for value in registration.get("runtime_argv", ())]
    require(option_value(runtime_argv, "--path-storage-mode") == "legacy", "runtime storage drift")
    require(option_value(runtime_argv, "--budget") == str(EXPECTED_BUDGET), "runtime safety budget drift")
    require(
        option_value(runtime_argv, "--total-internal-nf-budget") == str(EXPECTED_BUDGET),
        "runtime total budget drift",
    )
    require(option_value(runtime_argv, "--run-label") == LABEL, "runtime label drift")
    require(option_value(runtime_argv, "--alpha") == "1", "alpha drift")
    require(option_value(runtime_argv, "--discrete-residual-ratio") == "0.1", "r drift")
    require(option_value(runtime_argv, "--discrete-inner-max-steps") == "5", "m_max drift")
    require(option_value(runtime_argv, "--reparam-mode") == "state_curve", "reparameterization drift")

    amendment_bindings = amendment.get("bindings", {})
    require(
        validate_binding(amendment_bindings.get("replay_registration", {}), "amendment replay registration")
        == registration_path,
        "amendment/replay registration path mismatch",
    )
    require(
        validate_binding(amendment_bindings.get("selected_9990_result", {}), "amendment selected result")
        == reference_result_path,
        "amendment/reference result path mismatch",
    )
    registration_reference = validate_binding(
        registration.get("bindings", {}).get("selected_terminal_result", {}),
        "registration selected terminal result",
    )
    require(registration_reference == reference_result_path, "registration/reference result path mismatch")
    require(authorization.get("schema") == AUTHORIZATION_SCHEMA, "authorization schema mismatch")
    require(authorization.get("enabled") is True, "replay authorization is disabled")
    require(authorization.get("label") == LABEL, "authorization label mismatch")
    require(
        validate_binding(authorization.get("registration", {}), "authorization registration")
        == registration_path,
        "authorization/registration path mismatch",
    )
    return amendment, registration, authorization


def history_keys(reference, replay, cutoff_rows: int) -> list[str]:
    reference_rows = int(np.asarray(reference["calls_per_image"]).size)
    replay_rows = int(np.asarray(replay["calls_per_image"]).size)
    keys: list[str] = []
    for key in reference.files:
        if key in EXCLUDED_HISTORY_KEYS or key not in replay.files:
            continue
        left = np.asarray(reference[key])
        right = np.asarray(replay[key])
        if (
            left.ndim >= 1
            and right.ndim >= 1
            and left.shape[0] == reference_rows
            and right.shape[0] == replay_rows
            and left.shape[1:] == right.shape[1:]
            and left.dtype.str == right.dtype.str
        ):
            keys.append(key)
    missing = REQUIRED_EXACT_HISTORY_KEYS.difference(keys)
    require(not missing, "required exact history keys absent: " + ", ".join(sorted(missing)))
    for key in keys:
        require(
            exact_array_bytes(np.asarray(reference[key])[:cutoff_rows], np.asarray(replay[key])[:cutoff_rows]),
            f"online history exact-prefix mismatch: {key}",
        )
    return sorted(keys)


def compare_discrete_audit_prefix(reference, replay, cutoff_history_index: int) -> tuple[int, list[str]]:
    if "discrete_audit_outer" not in reference.files:
        return 0, []
    require("discrete_audit_outer" in replay.files, "replay lacks discrete audit outer indices")
    outer = np.asarray(reference["discrete_audit_outer"], dtype=np.int64)
    event_count = int(np.searchsorted(outer, cutoff_history_index + 1, side="left"))
    keys: list[str] = []
    for key in reference.files:
        if not key.startswith("discrete_") or key not in replay.files:
            continue
        left = np.asarray(reference[key])
        right = np.asarray(replay[key])
        if (
            left.ndim >= 1
            and right.ndim >= 1
            and left.shape[0] == outer.size
            and right.shape[0] >= event_count
            and left.shape[1:] == right.shape[1:]
            and left.dtype.str == right.dtype.str
        ):
            require(
                exact_array_bytes(left[:event_count], right[:event_count]),
                f"discrete audit exact-prefix mismatch: {key}",
            )
            keys.append(key)
    return event_count, sorted(keys)


def record_rows(saved, required_calls: Sequence[int]) -> list[dict]:
    calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
    counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
    totals = np.asarray(saved["total_internal_joint_evaluations"], dtype=np.int64)
    i_perp = np.asarray(saved["i_perp"], dtype=np.float64)
    committed = np.asarray(saved["reparam_committed"], dtype=bool)
    stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
    stored_paths = np.asarray(saved["stored_paths"], dtype=np.float64)
    require(counters.shape == (calls.size, 32), "counter history shape mismatch")
    require(totals.shape == calls.shape, "total N_F history shape mismatch")
    require(i_perp.shape == calls.shape, "I_perp history shape mismatch")
    require(committed.shape == calls.shape, "commit history shape mismatch")
    require(stored_paths.ndim == 3 and stored_paths.shape[1:] == (32, 9396), "stored paths have wrong shape")
    require(stored_paths.shape[0] == stored_calls.size, "stored path/call shape mismatch")
    require(np.all(np.isfinite(stored_paths)), "stored paths are non-finite")
    rows: list[dict] = []
    for call in required_calls:
        history_indices = np.flatnonzero(calls == call)
        path_indices = np.flatnonzero(stored_calls == call)
        require(history_indices.size == 1, f"history call {call} is absent or non-unique")
        require(path_indices.size == 1, f"stored call {call} is absent or non-unique")
        history_index = int(history_indices[0])
        path_index = int(path_indices[0])
        require(history_index < calls.size - 1, f"reporting call {call} is terminal")
        internal_nf = int(counters[history_index, 1:-1].sum(dtype=np.int64))
        require(internal_nf == int(totals[history_index]), f"counter-bound N_F mismatch at call {call}")
        require(internal_nf == EXPECTED_NF[call], f"unexpected N_F at call {call}")
        require(np.isfinite(i_perp[history_index]), f"non-finite I_perp at call {call}")
        path = stored_paths[path_index]
        rows.append(
            {
                "call_per_image": int(call),
                "history_index": history_index,
                "optimization_N_F": internal_nf,
                "discrete_I_perp_eV2_per_A": float(i_perp[history_index]),
                "reparameterization_committed": bool(committed[history_index]),
                "selected_path_sha256": canonical_float64_sha256(path),
                "terminal_tail": False,
                "fully_evaluated_current_path": True,
            }
        )
    return rows


def stored_path(saved, call: int) -> np.ndarray:
    stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
    indices = np.flatnonzero(stored_calls == int(call))
    require(indices.size == 1, f"stored path call {call} is absent or non-unique")
    return np.asarray(saved["stored_paths"], dtype=np.float64)[int(indices[0])]


def validate_terminal(replay, audit: dict) -> dict:
    calls = np.asarray(replay["calls_per_image"], dtype=np.int64)
    stored_calls = np.asarray(replay["stored_path_calls"], dtype=np.int64)
    require(int(calls[-1]) == EXPECTED_TERMINAL_CALL, "unexpected terminal call")
    require(int(stored_calls[-1]) == EXPECTED_TERMINAL_CALL, "terminal path was not retained as an excluded tail")
    require(
        exact_array_bytes(stored_path(replay, EXPECTED_TERMINAL_CALL), np.asarray(replay["final_path"])),
        "terminal stored path/final path mismatch",
    )
    require(scalar(replay, "path_storage_mode") == "legacy", "result storage mode is not legacy")
    require(scalar(replay, "path_storage_affects_online_algorithm") is False, "result says storage affects optimization")
    require(bool(scalar(replay, "discrete_all_returned_states_evaluated")), "runner returned unevaluated online states")
    require(int(scalar(replay, "total_internal_nf_budget")) == EXPECTED_BUDGET, "saved budget mismatch")
    require(int(scalar(replay, "total_internal_nf_consumed")) == EXPECTED_TERMINAL_NF, "saved consumed N_F mismatch")
    require(int(scalar(replay, "optimization_internal_nf")) == EXPECTED_TERMINAL_NF, "optimization N_F mismatch")
    require(int(scalar(replay, "number_of_forces")) == EXPECTED_TERMINAL_NF, "Number of Forces mismatch")
    require(bool(scalar(replay, "total_internal_nf_budget_no_overshoot")), "budget overshot")
    require(int(scalar(replay, "total_internal_nf_remaining")) == EXPECTED_REMAINING_NF, "remaining N_F mismatch")
    require(
        scalar(replay, "total_internal_nf_budget_stop_reason")
        == "total_internal_joint_budget_insufficient_for_full_band",
        "unexpected budget stop reason",
    )
    require(not bool(scalar(replay, "total_internal_nf_terminal_record_is_complete_outer_state")), "terminal tail is marked complete")
    require(bool(scalar(replay, "total_internal_nf_terminal_partial_tail_present")), "terminal tail is absent")
    require(bool(scalar(replay, "total_internal_nf_terminal_partial_tail_quarantined")), "terminal tail is not quarantined")
    require(
        int(scalar(replay, "total_internal_nf_terminal_uncommitted_charged_tail_nf"))
        == EXPECTED_TERMINAL_TAIL_NF,
        "terminal tail N_F mismatch",
    )
    last_complete_index = int(scalar(replay, "total_internal_nf_last_complete_state_index"))
    require(int(calls[last_complete_index]) == EXPECTED_LAST_COMMITTED_CALL, "last complete call mismatch")
    require(
        int(scalar(replay, "total_internal_nf_last_complete_state_nf")) == EXPECTED_LAST_COMMITTED_NF,
        "last complete N_F mismatch",
    )
    require(bool(audit.get("consistent")), "runtime audit is inconsistent")
    require(int(audit.get("number_of_forces", -1)) == EXPECTED_TERMINAL_NF, "audit Number of Forces mismatch")
    require(int(audit.get("optimization_internal_nf", -1)) == EXPECTED_TERMINAL_NF, "audit optimization N_F mismatch")
    require(int(audit.get("sum_image_joint_evaluations", -1)) == EXPECTED_TERMINAL_NF, "audit counter sum mismatch")
    require(int(audit.get("endpoint_getstate_attempts", -1)) == 0, "endpoint getState attempts are nonzero")
    require(int(audit.get("endpoint_joint_ef_cache_misses", -1)) == 0, "endpoint cache misses are nonzero")
    require(int(audit.get("endpoint_inherited_joint_ef_cache_entries", -1)) == 2, "endpoint inherited cache count mismatch")
    require(all(bool(value) for value in audit.get("per_image_query_outcome_identity_ok", ())), "query outcome identity failed")
    require(all(bool(value) for value in audit.get("per_image_request_identity_ok", ())), "request identity failed")
    return {
        "terminal_call_per_image": EXPECTED_TERMINAL_CALL,
        "terminal_consumed_optimization_N_F": EXPECTED_TERMINAL_NF,
        "registered_budget": EXPECTED_BUDGET,
        "remaining_budget": EXPECTED_REMAINING_NF,
        "last_complete_reparameterization_committed_call": EXPECTED_LAST_COMMITTED_CALL,
        "last_complete_reparameterization_committed_N_F": EXPECTED_LAST_COMMITTED_NF,
        "terminal_uncommitted_charged_tail_N_F": EXPECTED_TERMINAL_TAIL_NF,
        "terminal_path_excluded": True,
    }


def verify_arrays(reference, replay) -> dict:
    reference_calls = np.asarray(reference["calls_per_image"], dtype=np.int64)
    replay_calls = np.asarray(replay["calls_per_image"], dtype=np.int64)
    reference_cutoff = np.flatnonzero(reference_calls == max(FIXED_CALLS))
    replay_cutoff = np.flatnonzero(replay_calls == max(FIXED_CALLS))
    require(reference_cutoff.size == replay_cutoff.size == 1, "call-50 cutoff is absent or non-unique")
    require(int(reference_cutoff[0]) == int(replay_cutoff[0]), "call-50 history index drift")
    cutoff_index = int(reference_cutoff[0])
    compared_keys = history_keys(reference, replay, cutoff_index + 1)
    event_count, audit_keys = compare_discrete_audit_prefix(reference, replay, cutoff_index)
    rows = record_rows(replay, FIXED_CALLS)
    reference_rows = record_rows(reference, (1, 20))
    reference_by_call = {int(row["call_per_image"]): row for row in reference_rows}
    replay_by_call = {int(row["call_per_image"]): row for row in rows}
    for call in (1, 20):
        require(
            replay_by_call[call]["selected_path_sha256"]
            == reference_by_call[call]["selected_path_sha256"],
            f"stored path hash mismatch at call {call}",
        )
    new_rows = [row for row in rows if int(row["call_per_image"]) in NEW_CALLS]
    require(tuple(int(row["call_per_image"]) for row in new_rows) == NEW_CALLS, "new reporting row order drift")
    return {
        "exact_prefix_through_call": max(FIXED_CALLS),
        "exact_prefix_through_optimization_N_F": EXPECTED_NF[max(FIXED_CALLS)],
        "compared_online_history_record_count": cutoff_index + 1,
        "compared_online_history_keys": compared_keys,
        "excluded_online_history_keys": sorted(EXCLUDED_HISTORY_KEYS),
        "compared_discrete_audit_event_count": event_count,
        "compared_discrete_audit_keys": audit_keys,
        "fixed_reporting_rows": rows,
        "new_reporting_rows": new_rows,
        "reference_path_hash_checks": [
            {
                "call_per_image": call,
                "optimization_N_F": EXPECTED_NF[call],
                "selected_path_sha256": replay_by_call[call]["selected_path_sha256"],
                "exact_match": True,
            }
            for call in (1, 20)
        ],
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--replay-result", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    inputs = {
        name: path.expanduser().resolve(strict=True)
        for name, path in vars(args).items()
        if name != "output"
    }
    validate_contracts(
        inputs["amendment"],
        inputs["registration"],
        inputs["authorization"],
        inputs["reference_result"],
    )
    audit = read_json(inputs["replay_audit"])
    require(
        Path(str(audit.get("result_file", ""))).expanduser().resolve()
        == inputs["replay_result"],
        "audit/result path mismatch",
    )
    with np.load(inputs["reference_result"], allow_pickle=False) as reference, np.load(
        inputs["replay_result"], allow_pickle=False
    ) as replay:
        comparison = verify_arrays(reference, replay)
        terminal = validate_terminal(replay, audit)
    payload = {
        "schema": SCHEMA,
        "status": "PASS_EXACT_OUTPUT_ONLY_PREFIX",
        "created_utc": utc_now(),
        "scientific_scope": (
            "the output-only legacy replay exactly reproduces the selected IR-IVM "
            "online history through call 50 and contributes only the four prospectively "
            "registered, fully evaluated, nonterminal current paths"
        ),
        "tolerance_fallback_used": False,
        "fixed_endpoint_optimization_N_F": 0,
        "diagnostic_feedback_into_optimization": "forbidden",
        **comparison,
        "terminal_accounting": terminal,
        "bindings": {name: file_binding(path) for name, path in inputs.items()},
    }
    write_exclusive_json(args.output, payload)
    destination = args.output.expanduser().resolve(strict=True)
    print(destination)
    print(sha256_file(destination))


if __name__ == "__main__":
    main()
