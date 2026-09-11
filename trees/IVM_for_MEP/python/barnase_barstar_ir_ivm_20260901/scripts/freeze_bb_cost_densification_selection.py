#!/usr/bin/env python3
"""Freeze the metric-blind BB IR-IVM/ODE--NEB densification selection.

This is a zero-OpenMM preregistration utility.  It fixes which replay calls may
be reported and chooses cost-matched ODE--NEB retained paths without consulting
either newly densified barrier or newly recomputed discrete-residual values.
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


SCHEMA = "bb-irivm-ode-neb-cost-densification-selection-v1"
AMENDMENT_SCHEMA = "bb-irivm-reporting-intermediate-states-amendment-v1"
AMENDMENT_STATUS = "FROZEN_AFTER_REPLAY_LAUNCH_BEFORE_ANY_NEW_REPORTING_METRIC_CALL"
REGISTRATION_SCHEMA = "bb-irivm-selected-output-only-legacy-replay-v1"
REGISTRATION_STATUS = "FROZEN_BEFORE_FIRST_REPLAY_PHYSICAL_CALL"
ODE_LEDGER_SCHEMA = "bb-ode-neb-targeted-arc500-selection-v1"
ODE_LEDGER_STATUS = "FROZEN_ZERO_OPENMM_TARGET_SELECTION"
FIXED_IVM_CALLS = (1, 2, 5, 10, 20, 50)
NEW_IVM_CALLS = (2, 5, 10, 50)
EXPECTED_IVM_NF = {1: 30, 2: 60, 5: 150, 10: 300, 20: 429, 50: 1329}
EXPECTED_ODE_NF = (60, 150, 300, 1320)
EXPECTED_ODE_BATCH = (2, 5, 10, 44)
EXPECTED_PAID_STATES = 333
EXPECTED_DISTINCT_RETAINED_PATHS = 284
EXPECTED_ACCEPTED = 283
EXPECTED_REJECTED = 49


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def file_binding(path: Path) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def validate_binding(binding: Mapping[str, object], role: str) -> Path:
    require(isinstance(binding, Mapping), f"{role} binding is not an object")
    path = Path(str(binding.get("path", ""))).expanduser().resolve(strict=True)
    require(sha256_file(path) == str(binding.get("sha256", "")), f"{role} binding drift")
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


def scalar(saved: Mapping[str, np.ndarray], key: str):
    require(key in saved, f"selected reference result lacks {key}")
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def validate_contracts(
    amendment_path: Path,
    registration_path: Path,
    reference_path: Path,
    ode_ledger_path: Path,
    assigned_host: str,
) -> tuple[dict, dict, dict]:
    amendment = read_json(amendment_path)
    registration = read_json(registration_path)
    ledger = read_json(ode_ledger_path)
    require(amendment.get("schema") == AMENDMENT_SCHEMA, "reporting amendment schema mismatch")
    require(amendment.get("status") == AMENDMENT_STATUS, "reporting amendment status mismatch")
    require(amendment.get("execution_authorized_by_this_file") is False, "amendment may not authorize execution")
    require(registration.get("schema") == REGISTRATION_SCHEMA, "replay registration schema mismatch")
    require(registration.get("status") == REGISTRATION_STATUS, "replay registration was not prospectively frozen")
    require(ledger.get("schema") == ODE_LEDGER_SCHEMA, "ODE--NEB ledger schema mismatch")
    require(ledger.get("status") == ODE_LEDGER_STATUS, "ODE--NEB ledger status mismatch")
    selector_calls = ledger.get("physical_calls_made_by_selector", {})
    require(
        isinstance(selector_calls, Mapping)
        and int(selector_calls.get("energy", -1)) == 0
        and int(selector_calls.get("force", -1)) == 0,
        "source ODE--NEB selector was not zero-OpenMM",
    )
    require(ledger.get("execution_authorized_by_this_file") is False, "source ledger may not authorize execution")

    amendment_bindings = amendment.get("bindings", {})
    require(
        validate_binding(amendment_bindings.get("replay_registration", {}), "amendment replay registration")
        == registration_path,
        "amendment/replay registration path mismatch",
    )
    require(
        validate_binding(amendment_bindings.get("selected_9990_result", {}), "amendment selected result")
        == reference_path,
        "amendment/reference result path mismatch",
    )
    require(
        validate_binding(
            amendment_bindings.get("official_ODE_NEB_retained_path_ledger", {}),
            "amendment ODE--NEB ledger",
        )
        == ode_ledger_path,
        "amendment/ODE--NEB ledger path mismatch",
    )
    scope = amendment.get("scope", {})
    require(tuple(scope.get("fixed_reporting_calls", ())) == FIXED_IVM_CALLS, "fixed IVM calls drift")
    require(tuple(scope.get("new_reporting_calls", ())) == NEW_IVM_CALLS, "new IVM calls drift")
    registered = registration.get("replay", {})
    require(tuple(registered.get("prospective_reporting_calls", ())) == FIXED_IVM_CALLS, "registration calls drift")
    require(registered.get("path_storage_mode") == "legacy", "registration storage is not legacy")
    require(registered.get("path_storage_affects_online_algorithm") is False, "storage changes online algorithm")
    require(
        assigned_host == amendment.get("execution_assignment", {}).get("ODE_NEB_four_path_metric_evaluation_host"),
        "assigned ODE--NEB host differs from amendment",
    )
    return amendment, registration, ledger


def fixed_ivm_rows(reference_path: Path) -> list[dict]:
    with np.load(reference_path, allow_pickle=False) as saved:
        calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
        totals = np.asarray(saved["total_internal_joint_evaluations"], dtype=np.int64)
        i_perp = np.asarray(saved["i_perp"], dtype=np.float64)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        require(counters.shape == (calls.size, 32), "selected reference counter shape mismatch")
        require(totals.shape == calls.shape, "selected reference N_F history shape mismatch")
        require(i_perp.shape == calls.shape, "selected reference I_perp history shape mismatch")
        require(committed.shape == calls.shape, "selected reference commit history shape mismatch")
        require(bool(scalar(saved, "total_internal_nf_endpoint_evaluations_excluded")), "reference did not exclude endpoints")
        require(int(scalar(saved, "endpoint_joint_ef_cache_misses")) == 0, "reference charged endpoint cache misses")
        require(int(scalar(saved, "endpoint_inherited_joint_ef_cache_entries")) == 2, "reference endpoint cache drift")
        rows: list[dict] = []
        for call in FIXED_IVM_CALLS:
            indices = np.flatnonzero(calls == call)
            require(indices.size == 1, f"selected reference call {call} absent or non-unique")
            index = int(indices[0])
            require(index < calls.size - 1, f"selected reference call {call} is terminal")
            nf = int(counters[index, 1:-1].sum(dtype=np.int64))
            require(nf == int(totals[index]), f"selected reference counter mismatch at call {call}")
            require(nf == EXPECTED_IVM_NF[call], f"selected reference N_F drift at call {call}")
            require(np.isfinite(i_perp[index]), f"selected reference I_perp is non-finite at call {call}")
            rows.append(
                {
                    "call_per_image": call,
                    "optimization_N_F": nf,
                    "sealed_reference_discrete_I_perp_eV2_per_A": float(i_perp[index]),
                    "reparameterization_committed": bool(committed[index]),
                    "new_reporting_path": call in NEW_IVM_CALLS,
                    "eligibility": "contingent_on_PASS_EXACT_OUTPUT_ONLY_PREFIX_replay_verification",
                }
            )
    return rows


def first_occurrences(states: Sequence[Mapping[str, object]]) -> list[dict]:
    require(len(states) == EXPECTED_PAID_STATES, "official paid-state count drift")
    accepted = sum(record.get("disposition") == "accepted" for record in states)
    rejected = sum(record.get("disposition") == "rejected" for record in states)
    initial = sum(record.get("disposition") == "current_initial" for record in states)
    require((accepted, rejected, initial) == (EXPECTED_ACCEPTED, EXPECTED_REJECTED, 1), "ODE disposition counts drift")
    seen: set[str] = set()
    first: list[dict] = []
    previous_nf = 0
    for sequence_index, record in enumerate(states):
        nf = int(record.get("optimization_N_F", -1))
        batch = int(record.get("paid_batch_id", -1))
        path_hash = str(record.get("official_path32_sha256", ""))
        require(nf == 30 * batch, f"ODE paid-state N_F/batch mismatch at sequence {sequence_index}")
        require(nf > previous_nf, "ODE paid-state N_F is not strictly increasing")
        previous_nf = nf
        require(len(path_hash) == 64, f"ODE official path hash malformed at N_F={nf}")
        if path_hash in seen:
            continue
        seen.add(path_hash)
        archive = Path(str(record.get("coordinate_archive", ""))).expanduser().resolve(strict=True)
        require(
            sha256_file(archive) == str(record.get("coordinate_archive_sha256", "")),
            f"ODE coordinate archive drift at N_F={nf}",
        )
        first.append(dict(record))
    require(len(first) == EXPECTED_DISTINCT_RETAINED_PATHS, "distinct retained ODE path count drift")
    return first


def select_ode_rows(states: Sequence[Mapping[str, object]]) -> list[dict]:
    first = first_occurrences(states)
    selected: list[dict] = []
    for call in NEW_IVM_CALLS:
        ivm_nf = EXPECTED_IVM_NF[call]
        eligible = [record for record in first if int(record["optimization_N_F"]) <= ivm_nf]
        require(eligible, f"no retained ODE path is affordable at IVM N_F={ivm_nf}")
        record = max(eligible, key=lambda item: int(item["optimization_N_F"]))
        selected.append(
            {
                "matched_IR_IVM_call_per_image": call,
                "matched_IR_IVM_optimization_N_F": ivm_nf,
                "optimization_N_F_first_occurrence": int(record["optimization_N_F"]),
                "paid_batch_id_first_occurrence": int(record["paid_batch_id"]),
                "source_batch_id": int(record["source_batch_id"]),
                "disposition_at_first_occurrence": str(record["disposition"]),
                "official_path32_sha256": str(record["official_path32_sha256"]),
                "coordinate_archive": str(Path(str(record["coordinate_archive"])).expanduser().resolve(strict=True)),
                "coordinate_archive_sha256": str(record["coordinate_archive_sha256"]),
                "new_discrete_I_perp_seen_at_selection": record.get("discrete_I_perp_eV2_per_A") is not None,
                "new_densified_Delta_E_seen_at_selection": record.get("densified_Delta_E_kcal_per_mol") is not None,
            }
        )
    require(tuple(row["optimization_N_F_first_occurrence"] for row in selected) == EXPECTED_ODE_NF, "ODE cost match drift")
    require(tuple(row["paid_batch_id_first_occurrence"] for row in selected) == EXPECTED_ODE_BATCH, "ODE batch match drift")
    require(len({row["official_path32_sha256"] for row in selected}) == len(selected), "ODE selection repeats a path")
    require(
        not any(row["new_discrete_I_perp_seen_at_selection"] for row in selected),
        "an ODE path's new discrete I_perp was already visible at selection",
    )
    require(
        not any(row["new_densified_Delta_E_seen_at_selection"] for row in selected),
        "an ODE path's new densified Delta E was already visible at selection",
    )
    return selected


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--replay-registration", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--ode-ledger", type=Path, required=True)
    parser.add_argument("--assigned-host", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    amendment_path = args.amendment.expanduser().resolve(strict=True)
    registration_path = args.replay_registration.expanduser().resolve(strict=True)
    reference_path = args.reference_result.expanduser().resolve(strict=True)
    ode_ledger_path = args.ode_ledger.expanduser().resolve(strict=True)
    amendment, registration, ledger = validate_contracts(
        amendment_path,
        registration_path,
        reference_path,
        ode_ledger_path,
        str(args.assigned_host),
    )
    ivm_rows = fixed_ivm_rows(reference_path)
    ode_rows = select_ode_rows(ledger.get("official_paid_states", ()))
    payload = {
        "schema": SCHEMA,
        "status": "FROZEN_BEFORE_ANY_NEW_ARC500_OR_ODE_DIAGNOSTIC_FORCE_CALL",
        "created_utc": utc_now(),
        "decision_timing": (
            "after replay launch, using the replay registration's launch-time fixed call set, "
            "and before any new arc-length-500 energy or ODE--NEB diagnostic-force call"
        ),
        "physical_calls_made_by_freezer": 0,
        "execution_authorized_by_this_file": False,
        "selection_is_metric_blind": True,
        "IVM_reporting_rows": ivm_rows,
        "IVM_new_reporting_calls": list(NEW_IVM_CALLS),
        "ODE_NEB_cost_matched_rows": ode_rows,
        "ODE_NEB_rule": (
            "For each new IR-IVM actual optimization N_F, choose the latest first occurrence "
            "of a distinct official retained ODE--NEB current path with optimization N_F no greater "
            "than that IR-IVM N_F. Charge an identical retained path at its earliest occurrence."
        ),
        "scientific_protocol": {
            "same_frozen_path_for_both_metrics": True,
            "discrete_I_perp": "unnormalized normal residual on the 32-image path",
            "densified_Delta_E": "0 K forward barrier on 500 points uniformly spaced in unweighted Cartesian arc length along the same piecewise-linear path",
            "arc500_grid": "s_j=j*L/499,j=0,...,499",
            "arc500_includes_both_endpoints": True,
            "fixed_endpoint_optimization_N_F": 0,
            "endpoint_diagnostic_force_calls_per_path": 0,
            "IVM_new_diagnostic_force_calls_per_path": 0,
            "ODE_NEB_new_diagnostic_force_calls_per_path": 30,
            "diagnostic_calls_contribute_to_optimization_N_F": False,
            "diagnostic_calls_reported_separately": True,
            "diagnostic_feedback_into_optimization": "forbidden",
            "terminal_partial_or_uncommitted_tail": "excluded",
            "rejected_ODE_NEB_trial": "does not create a new retained reporting path",
        },
        "execution_assignment": {
            "ODE_NEB_metric_host": str(args.assigned_host),
            "jobs": 1,
            "one_BB_task_on_host": True,
            "host_local_BB_locks_required": True,
            "all_thread_caps": 1,
            "Mutagen_and_result_validation_gates_may_not_be_bypassed": True,
        },
        "planned_new_physical_calls": {
            "IR_IVM_replay_optimization_joint_EF_expected": 1479,
            "IR_IVM_replay_optimization_joint_EF_hard_upper_bound": 1500,
            "IR_IVM_arc500_energy": 4 * 500,
            "IR_IVM_diagnostic_force": 0,
            "ODE_NEB_arc500_energy": 4 * 500,
            "ODE_NEB_diagnostic_force": 4 * 30,
            "expected_total_including_replay": 5599,
            "hard_upper_bound_total_including_replay": 5620,
        },
        "eligibility_gate": {
            "IVM_replay_verification_required_status": "PASS_EXACT_OUTPUT_ONLY_PREFIX",
            "IVM_required_result_and_audit_paths": "must be passed explicitly to the replay verifier; runtime cfg hash may not be guessed",
            "fully_evaluated_nonterminal_current_state_required": True,
            "exact_selected_9990_online_prefix_required": True,
        },
        "bindings": {
            "reporting_amendment": file_binding(amendment_path),
            "replay_registration": file_binding(registration_path),
            "selected_9990_result": file_binding(reference_path),
            "official_ODE_NEB_retained_path_ledger": file_binding(ode_ledger_path),
            "freezer": file_binding(Path(__file__).resolve(strict=True)),
        },
    }
    require(
        tuple(row["call_per_image"] for row in payload["IVM_reporting_rows"] if row["new_reporting_path"])
        == NEW_IVM_CALLS,
        "final IVM row selection drift",
    )
    write_exclusive_json(args.output, payload)
    destination = args.output.expanduser().resolve(strict=True)
    print(destination)
    print(sha256_file(destination))


if __name__ == "__main__":
    main()
