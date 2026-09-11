#!/usr/bin/env python3
"""Seal the prospective cached-stall follow-up trigger decision.

This is a zero-OpenMM bookkeeping step.  It binds both long-budget
candidate outcomes and records whether the prospectively registered
``m_max=8`` follow-up is activated.  The decision is made from terminal
verification classes only; neither path diagnostic is read.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require(not output.exists(), f"refusing to overwrite {output}")

    paths = {
        "prospective_design": root / "registrations" / "BB_IRIVM_CACHED_STALL_TARGETED_FOLLOWUP_DESIGN_20260902_R1.json",
        "candidate_A_failure": root / "reports" / "failures" / "BB_IRIVM_A1E1_R05_M05_INF4080_R2_ALGORITHMIC_EARLY_STOP_20260902.json",
        "candidate_A_diagnosis": root / "reports" / "diagnostics" / "BB_IRIVM_A1E1_R05_M05_INF4080_R2_CACHED_STALL_DIAGNOSIS_20260902.json",
        "candidate_B_registration": root / "registrations" / "screen" / "bb_irivm_a1e0_r01_m05_inf4080_r1.json",
        "candidate_B_authorization": root / "registrations" / "execution_authorizations" / "bb_irivm_a1e0_r01_m05_inf4080_r1.json",
        "candidate_B_result": root / "runs" / "screen" / "ivm_bb_irivm_a1e0_r01_m05_inf4080_r1_central_state_ivm_cfg47c24114c93c_budget4080_storagecommitted.npz",
        "candidate_B_audit": root / "runs" / "screen" / "ivm_bb_irivm_a1e0_r01_m05_inf4080_r1_central_state_ivm_cfg47c24114c93c_budget4080_storagecommitted.audit.json",
        "candidate_B_lock_receipt": root / "runs" / "lock_receipts" / "bb_irivm_a1e0_r01_m05_inf4080_r1.launch.json",
        "candidate_B_verification": root / "reports" / "verifications" / "BB_IRIVM_A1E0_R01_M05_INF4080_R1_VERIFICATION_20260902.json",
    }
    for role, path in paths.items():
        require(path.is_file(), f"missing {role}: {path}")

    design = json.loads(paths["prospective_design"].read_text(encoding="utf-8"))
    verification = json.loads(paths["candidate_B_verification"].read_text(encoding="utf-8"))
    failure = json.loads(paths["candidate_A_failure"].read_text(encoding="utf-8"))
    require(
        design.get("status") == "CONDITIONAL_DESIGN_FROZEN_NON_EXECUTABLE",
        "prospective follow-up design is not frozen",
    )
    require(
        sha256(paths["candidate_A_failure"])
        == design["trigger"]["candidate_A_failure_receipt_sha256"],
        "candidate-A failure binding mismatch",
    )
    require(
        sha256(paths["candidate_A_diagnosis"])
        == design["trigger"]["candidate_A_cached_stall_diagnosis_sha256"],
        "candidate-A diagnosis binding mismatch",
    )
    require(
        sha256(paths["candidate_B_registration"])
        == design["trigger"]["candidate_B_registration_sha256"],
        "candidate-B registration binding mismatch",
    )
    require(
        sha256(paths["candidate_B_authorization"])
        == design["trigger"]["candidate_B_execution_authorization_sha256"],
        "candidate-B authorization binding mismatch",
    )
    require(verification.get("status") == "PASS", "candidate B did not verify")
    require(
        verification.get("budget_atomicity_case") in {"exact", "atomic_remainder"},
        "candidate B lacks a compliant budget class",
    )
    require(
        len(verification.get("post_initial_reparam_committed_stored_paths", [])) > 0,
        "candidate B has no eligible reparameterization-committed path",
    )
    require(
        int(verification.get("endpoint_optimization_N_F", -1)) == 0,
        "candidate B charged a fixed endpoint",
    )
    require(
        str(failure.get("status", failure.get("classification", "")))
        in {"ALGORITHMIC_EARLY_STOP", "FAIL", "FAILED"}
        or "ALGORITHMIC_EARLY_STOP" in json.dumps(failure, sort_keys=True),
        "candidate-A failure receipt does not preserve its early-stop class",
    )

    payload = {
        "schema": "bb-irivm-cached-stall-followup-trigger-decision-v1",
        "status": "A8_NOT_TRIGGERED",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "decision_basis": {
            "prospective_logical_rule": design["trigger"]["logical_rule"],
            "candidate_A_terminal_class": "ALGORITHMIC_EARLY_STOP",
            "candidate_B_terminal_class": "COMPLIANT_4080_UPPER_BOUND_WITH_ATOMIC_REMAINDER",
            "candidate_B_registered_optimization_N_F_upper_bound": int(
                verification["registered_optimization_N_F_upper_bound"]
            ),
            "candidate_B_actual_optimization_N_F": int(
                verification["actual_optimization_N_F"]
            ),
            "candidate_B_remaining_registered_N_F": int(
                verification["remaining_registered_N_F"]
            ),
            "candidate_B_post_initial_reparam_committed_path_count": len(
                verification["post_initial_reparam_committed_stored_paths"]
            ),
            "trigger_result": False,
            "interpretation": (
                "Candidate B supplies a compliant long-budget IR-IVM trajectory "
                "with at least one post-initial reparameterization-committed path; "
                "therefore neither prospective trigger clause is satisfied."
            ),
        },
        "authorization_consequence": {
            "bb_irivm_a1e1_r05_m08_inf4080_r1": "NOT_AUTHORIZED_AND_NOT_RUN",
            "m_max_10_followup": "NOT_AUTHORIZED_AND_NOT_RUN",
        },
        "diagnostic_information_used": False,
        "bindings": {
            role: {"path": str(path), "sha256": sha256(path)}
            for role, path in paths.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    print(f"Saved {output}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
