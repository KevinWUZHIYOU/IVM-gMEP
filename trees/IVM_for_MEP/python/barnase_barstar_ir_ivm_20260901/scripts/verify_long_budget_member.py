#!/usr/bin/env python3
"""Zero-OpenMM verifier for one completed 4080/9990 BB IR-IVM member."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def scalar(saved, key):
    value = np.asarray(saved[key])
    if value.shape != ():
        raise RuntimeError(f"{key} is not scalar")
    return value.item()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_budget_atomicity(saved, actual: int, budget: int) -> tuple[bool, bool, str]:
    """Enforce the operative 30-internal-image band accounting amendment."""
    partial = bool(scalar(saved, "total_internal_nf_terminal_partial_tail_present"))
    quarantined = bool(scalar(saved, "total_internal_nf_terminal_partial_tail_quarantined"))
    require(not partial or quarantined, "terminal partial tail is not quarantined")
    if actual == budget:
        return partial, quarantined, "exact"

    remaining = budget - actual
    require(
        0 < remaining < 30,
        "sub-budget termination is not an atomic 30-image-band remainder",
    )
    require(
        str(scalar(saved, "total_internal_nf_budget_stop_reason"))
        == "total_internal_joint_budget_insufficient_for_full_band",
        "sub-budget termination has a non-budget stop reason",
    )
    require(
        int(scalar(saved, "total_internal_nf_full_band_preflight_zero_step_stops")) > 0,
        "sub-budget termination lacks a zero-step full-band preflight stop",
    )
    require(
        int(scalar(saved, "total_internal_nf_complete_band_budget_stop_count")) > 0,
        "sub-budget termination lacks a complete-band budget stop",
    )
    require(
        partial and quarantined,
        "sub-budget termination lacks a quarantined terminal partial tail",
    )
    return partial, quarantined, "atomic_remainder"


def collect_committed_stored_paths(saved) -> tuple[list[int], list[dict]]:
    """Return only physically stored paths that followed a committed reparameterization."""
    calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
    committed = np.asarray(saved["reparam_committed"], dtype=bool)
    stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
    stored_paths = np.asarray(saved["stored_paths"], dtype=np.float64)
    counter_vectors = np.asarray(saved["counter_vectors"], dtype=np.int64)
    require(calls.shape == committed.shape, "history/commit mask shape mismatch")
    require(
        counter_vectors.shape == (calls.size, 32),
        "counter history must contain one 32-image row per outer record",
    )
    require(stored_paths.shape[0] == stored_calls.size, "stored path/call count mismatch")
    eligible_calls = sorted({1, *calls[committed].tolist()})
    eligible_stored_calls = [int(value) for value in stored_calls if int(value) in eligible_calls]
    require(eligible_stored_calls, "no eligible stored path")
    post_initial_committed_calls = sorted(
        set(int(value) for value in calls[committed] if int(value) != 1)
        & set(int(value) for value in stored_calls)
    )
    require(
        post_initial_committed_calls,
        "no post-initial reparameterization-committed stored path",
    )
    committed_stored_paths = []
    for call in post_initial_committed_calls:
        call_indices = np.flatnonzero(calls == call)
        stored_indices = np.flatnonzero(stored_calls == call)
        require(call_indices.size == 1, f"non-unique outer record for call {call}")
        require(stored_indices.size == 1, f"non-unique stored path for call {call}")
        record_index = int(call_indices[0])
        path_index = int(stored_indices[0])
        committed_stored_paths.append(
            {
                "call_per_image": call,
                "actual_optimization_N_F": int(np.sum(counter_vectors[record_index, 1:-1])),
                "path_sha256": array_sha256(stored_paths[path_index]),
            }
        )
    return eligible_stored_calls, committed_stored_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--lock-receipt", type=Path, required=True)
    parser.add_argument("--host-observation", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: path.expanduser().resolve(strict=True)
        for name, path in vars(args).items()
        if name != "output" and path is not None
    }
    atomicity_path = (
        Path(__file__).resolve().parent.parent
        / "registrations"
        / "USER_BUDGET_ATOMICITY_AMENDMENT_20260901_R1.json"
    ).resolve(strict=True)
    paths["budget_atomicity_amendment"] = atomicity_path
    registration = json.loads(paths["registration"].read_text())
    authorization = json.loads(paths["authorization"].read_text())
    lock_receipt = json.loads(paths["lock_receipt"].read_text())
    audit = json.loads(paths["audit"].read_text())
    atomicity = json.loads(paths["budget_atomicity_amendment"].read_text())

    require(
        atomicity.get("schema") == "bb-irivm-budget-atomicity-amendment-v1",
        "budget-atomicity amendment schema mismatch",
    )
    require(
        atomicity.get("status") == "OPERATIVE_ACCOUNTING_CORRECTION",
        "budget-atomicity amendment is not operative",
    )

    label = registration["label"]
    budget = int(registration["internal_nf_budget"])
    require(authorization["label"] == label, "authorization label mismatch")
    require(
        authorization["member_registration_sha256"] == sha256(paths["registration"]),
        "authorization/registration hash mismatch",
    )
    require(
        authorization.get("controlled_scan_gate", {}).get("stage")
        in {
            "two_candidate_4080_N_F_reference_host_confirmation",
            "selected_9990_N_F_reference_host_confirmation",
        },
        "member lacks a long-budget promotion gate",
    )
    require(
        lock_receipt.get("status") == "CHILD_EXITED_LOCKS_RELEASED",
        "lock receipt is not terminal",
    )
    require(lock_receipt.get("returncode") == 0, "locked child did not exit successfully")
    require(
        lock_receipt.get("schema") == "bb-host-local-lock-launch-receipt-v1",
        "member was not launched by the pre-child lock wrapper",
    )
    command = " ".join(str(item) for item in lock_receipt.get("command", []))
    require(label.rsplit("_", 1)[-1] in command, "attempt tag absent from locked command")
    host_evidence = lock_receipt
    if "host" not in host_evidence:
        require(
            "host_observation" in paths,
            "legacy launch receipt lacks independent reference-host evidence",
        )
        host_evidence = json.loads(paths["host_observation"].read_text())
        require(
            host_evidence.get("schema") == "bb-running-host-observation-v1",
            "host-observation schema mismatch",
        )
        require(
            host_evidence.get("status")
            == "RUNNING_CHILD_AND_BOTH_LOCKS_OBSERVED",
            "host observation did not see the running child under both locks",
        )
        require(host_evidence.get("label") == label, "host-observation label mismatch")
        require(
            int(host_evidence.get("wrapper_pid", -1))
            == int(lock_receipt.get("wrapper_pid", -2)),
            "host-observation wrapper PID mismatch",
        )
    frozen_host = authorization["controlled_scan_gate"]["frozen_reference_host"]
    observed_host = host_evidence.get("host", {})
    for key in ("hostname", "architecture", "processor"):
        require(
            observed_host.get(key) == frozen_host[key],
            f"reference-host mismatch: {key}",
        )
    registered_command = registration["command"]
    root_index = registered_command.index("--root") + 1
    require(
        host_evidence.get("working_directory") == registered_command[root_index],
        "reference-host working directory mismatch",
    )
    launch_disk = host_evidence.get("launch_disk", {})
    minimum_free_bytes = int(
        authorization.get("controlled_scan_gate", {}).get(
            "minimum_free_disk_bytes_before_child", 5 * 1024**3
        )
    )
    require(
        int(launch_disk.get("free_bytes", -1)) >= minimum_free_bytes,
        "launch/observation disk gate was below the authorized minimum",
    )

    require(audit.get("consistent") is True, "producer audit is inconsistent")
    actual = int(audit["optimization_internal_nf"])
    for key in (
        "backend_raw_joint_attempts",
        "backend_raw_joint_evaluations",
        "sum_image_joint_evaluations",
        "number_of_forces",
    ):
        require(int(audit[key]) == actual, f"audit counter mismatch: {key}")
    require(0 < actual <= budget, "actual optimization N_F is outside the registered cap")
    per_image = np.asarray(audit["per_image_joint_evaluations"], dtype=np.int64)
    require(per_image.shape == (32,), "producer audit must contain 32 images")
    require(int(per_image[0]) == 0 and int(per_image[-1]) == 0, "fixed endpoints incurred NoF")
    require(int(np.sum(per_image[1:-1])) == actual, "internal-image counter sum mismatch")
    require(int(audit["endpoint_getstate_attempts"]) == 0, "endpoint getState attempt detected")
    require(int(audit["endpoint_joint_ef_cache_misses"]) == 0, "endpoint cache miss detected")
    require(
        int(audit["endpoint_inherited_joint_ef_cache_entries"]) == 2,
        "two inherited endpoint E/F entries are required",
    )
    require(all(audit["per_image_request_identity_ok"]), "request identity audit failed")
    require(all(audit["per_image_query_outcome_identity_ok"]), "query identity audit failed")
    require(audit["discrete_all_returned_states_evaluated"] is True, "unevaluated state returned")

    with np.load(paths["result"], allow_pickle=False) as saved:
        require(str(scalar(saved, "run_label")) == label, "result label mismatch")
        require(float(scalar(saved, "alpha")) == float(registration["alpha"]), "alpha mismatch")
        require(
            float(scalar(saved, "discrete_residual_ratio"))
            == float(registration["residual_ratio_r"]),
            "r mismatch",
        )
        require(
            int(scalar(saved, "discrete_inner_max_steps"))
            == int(registration["inner_max_steps_m_max"]),
            "m_max mismatch",
        )
        for key in (
            "number_of_forces",
            "successful_joint_ef_evaluations",
            "optimization_internal_nf",
            "total_internal_nf_consumed",
        ):
            require(int(scalar(saved, key)) == actual, f"result counter mismatch: {key}")
        require(int(scalar(saved, "total_internal_nf_budget")) == budget, "budget mismatch")
        require(bool(scalar(saved, "total_internal_nf_budget_no_overshoot")), "budget overshoot")
        require(
            int(scalar(saved, "total_internal_nf_remaining")) == budget - actual,
            "remaining budget mismatch",
        )
        require(
            bool(scalar(saved, "total_internal_nf_endpoint_evaluations_excluded")),
            "endpoint exclusion policy missing",
        )
        require(
            int(scalar(saved, "total_internal_nf_endpoint_getstate_attempts_expected")) == 0,
            "result expects endpoint calls",
        )
        partial, quarantined, atomicity_case = validate_budget_atomicity(
            saved, actual, budget
        )
        eligible_stored_calls, committed_stored_paths = collect_committed_stored_paths(
            saved
        )
        require(
            int(scalar(saved, "total_internal_nf_last_complete_state_nf")) <= actual,
            "last complete state exceeds actual N_F",
        )

    payload = {
        "schema": "bb-irivm-long-budget-member-verification-v2",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": label,
        "registered_optimization_N_F_upper_bound": budget,
        "actual_optimization_N_F": actual,
        "remaining_registered_N_F": budget - actual,
        "budget_atomicity_case": atomicity_case,
        "endpoint_optimization_N_F": 0,
        "diagnostic_energy_evaluations": 0,
        "authorized_minimum_launch_free_bytes": minimum_free_bytes,
        "observed_launch_free_bytes": int(launch_disk["free_bytes"]),
        "eligible_stored_calls_per_image": eligible_stored_calls,
        "post_initial_reparam_committed_stored_paths": committed_stored_paths,
        "terminal_partial_tail_present": partial,
        "terminal_partial_tail_quarantined": quarantined,
        "bindings": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", closefd=False) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    print(f"Saved {output}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
