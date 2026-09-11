#!/usr/bin/env python3
"""Freeze the terminal-only, zero-OpenMM portion of the selected-9990 pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Sequence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main(argv: Sequence[str] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require(not output.exists(), f"refusing to overwrite {output}")
    bindings = {
        "selected_9990_design": root / "registrations" / "BB_IRIVM_SELECTED_9990_DESIGN_20260902_R1.json",
        "selected_registration": root / "registrations" / "production" / "bb_irivm_a1e0_r01_m05_inf9990_r1.json",
        "selected_authorization": root / "registrations" / "execution_authorizations" / "bb_irivm_a1e0_r01_m05_inf9990_r1.json",
        "prefix_criterion": root / "registrations" / "BB_IRIVM_LONG_PREFIX_REPRODUCIBILITY_CRITERION_20260902_R1.json",
        "reference_4080_registration": root / "registrations" / "screen" / "bb_irivm_a1e0_r01_m05_inf4080_r1.json",
        "reference_4080_result": root / "runs" / "screen" / "ivm_bb_irivm_a1e0_r01_m05_inf4080_r1_central_state_ivm_cfg47c24114c93c_budget4080_storagecommitted.npz",
        "reference_4080_verification": root / "reports" / "verifications" / "BB_IRIVM_A1E0_R01_M05_INF4080_R1_VERIFICATION_20260902.json",
        "reference_4080_arc500": root / "runs" / "diagnostics" / "bb_irivm_candidate_b_long_arc500_20260902_r1.npz",
        "reference_4080_arc500_audit": root / "runs" / "diagnostics" / "bb_irivm_candidate_b_long_arc500_20260902_r1.audit.json",
        "long_member_verifier": root / "scripts" / "verify_long_budget_member.py",
        "exact_prefix_verifier": root / "scripts" / "verify_selected_9990_prefix.py",
        "arc500_contract_freezer": root / "scripts" / "freeze_selected_9990_arc500_contract.py",
        "arc500_evaluator": root / "scripts" / "evaluate_discrete_iperp_densified_barrier.py",
    }
    for role, path in bindings.items():
        require(path.is_file(), f"missing {role}: {path}")
    selected_design = json.loads(bindings["selected_9990_design"].read_text())
    require(selected_design.get("status") == "FROZEN_BEFORE_FIRST_9990_PHYSICAL_CALL", "selected design status mismatch")
    selected_authorization = json.loads(bindings["selected_authorization"].read_text())
    require(selected_authorization.get("enabled") is True, "selected authorization is disabled")
    require(
        selected_authorization["controlled_scan_gate"].get("shorter_prefix_reproduction_required") is True,
        "selected authorization does not require exact prefix reproduction",
    )
    base = output.parent
    rel = lambda path: os.path.relpath(path.resolve(), start=base.resolve())
    payload: Dict[str, object] = {
        "schema": "bb-irivm-selected-9990-postrun-pipeline-v1",
        "status": "FROZEN_DURING_ACTIVE_OPTIMIZATION_WITHOUT_READING_AN_INCOMPLETE_RESULT",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "minimal terminal validation, exact 4080-prefix test, and arc-length-500 path/reuse freeze",
        "physical_E_or_F_calls_made_by_this_freeze": 0,
        "incomplete_9990_result_read": False,
        "ordered_terminal_gates": [
            {
                "stage": 1,
                "name": "zero-OpenMM terminal member verification",
                "requires": "terminal producer NPZ/audit and CHILD_EXITED_LOCKS_RELEASED receipt",
                "proves": "NoF four-ledger identity, zero fixed-endpoint calls, atomic budget accounting, reference host, pre-child locks, and authorized launch-disk gate",
            },
            {
                "stage": 2,
                "name": "exact reference-host 4080-prefix reproduction",
                "requires": "PASS stage 1 receipt",
                "proves": "eligible N_F sequence, committed paths, discrete I_perp, and online history are byte-exact with no tolerance fallback",
            },
            {
                "stage": 3,
                "name": "zero-OpenMM selected-9990 arc500 contract freeze",
                "requires": "PASS stages 1 and 2",
                "proves": "complete eligible path list and exact reused/fresh path partition before any new diagnostic energy call",
            },
            {
                "stage": 4,
                "name": "offline arc-length-500 energy diagnostic",
                "requires": "frozen stage 3 contract",
                "proves": "500 uniform-arclength energies per unique path, zero force requests, zero optimization N_F contribution",
            },
        ],
        "terminal_command_templates": {
            "stage_1": (
                "python scripts/verify_long_budget_member.py --registration registrations/production/"
                "bb_irivm_a1e0_r01_m05_inf9990_r1.json --authorization registrations/"
                "execution_authorizations/bb_irivm_a1e0_r01_m05_inf9990_r1.json --lock-receipt "
                "runs/lock_receipts/bb_irivm_a1e0_r01_m05_inf9990_r1.launch.json --result "
                "<TERMINAL_9990_RESULT.npz> --audit <TERMINAL_9990_RESULT.audit.json> --output "
                "reports/verifications/BB_IRIVM_A1E0_R01_M05_INF9990_R1_VERIFICATION_20260902.json"
            ),
            "stage_2": (
                "python scripts/verify_selected_9990_prefix.py --criterion registrations/"
                "BB_IRIVM_LONG_PREFIX_REPRODUCIBILITY_CRITERION_20260902_R1.json --reference-registration "
                "registrations/screen/bb_irivm_a1e0_r01_m05_inf4080_r1.json --reference-result "
                "runs/screen/ivm_bb_irivm_a1e0_r01_m05_inf4080_r1_central_state_ivm_"
                "cfg47c24114c93c_budget4080_storagecommitted.npz --reference-verification reports/"
                "verifications/BB_IRIVM_A1E0_R01_M05_INF4080_R1_VERIFICATION_20260902.json "
                "--selected-registration registrations/production/bb_irivm_a1e0_r01_m05_inf9990_r1.json "
                "--selected-result <TERMINAL_9990_RESULT.npz> --selected-verification reports/verifications/"
                "BB_IRIVM_A1E0_R01_M05_INF9990_R1_VERIFICATION_20260902.json --output reports/verifications/"
                "BB_IRIVM_SELECTED_9990_EXACT_4080_PREFIX_20260902_R1.json"
            ),
            "stage_3": (
                "python scripts/freeze_selected_9990_arc500_contract.py --selected-result "
                "<TERMINAL_9990_RESULT.npz> --selected-audit <TERMINAL_9990_RESULT.audit.json> "
                "--selected-verification reports/verifications/BB_IRIVM_A1E0_R01_M05_INF9990_R1_VERIFICATION_20260902.json "
                "--prefix-verification reports/verifications/BB_IRIVM_SELECTED_9990_EXACT_4080_PREFIX_20260902_R1.json "
                "--reuse-diagnostics runs/diagnostics/bb_irivm_candidate_b_long_arc500_20260902_r1.npz "
                "--reuse-audit runs/diagnostics/bb_irivm_candidate_b_long_arc500_20260902_r1.audit.json "
                "--output registrations/diagnostics/BB_IRIVM_SELECTED_9990_ARC500_20260902_R1.json"
            ),
            "stage_4": (
                "env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENMM_CPU_THREADS=1 "
                "PYTHONPATH=src:scripts /Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python "
                "scripts/evaluate_discrete_iperp_densified_barrier.py --contract registrations/diagnostics/"
                "BB_IRIVM_SELECTED_9990_ARC500_20260902_R1.json --output runs/diagnostics/"
                "bb_irivm_selected_9990_arc500_20260902_r1.npz --reuse-diagnostics runs/diagnostics/"
                "bb_irivm_candidate_b_long_arc500_20260902_r1.npz --reuse-diagnostic-audit runs/diagnostics/"
                "bb_irivm_candidate_b_long_arc500_20260902_r1.audit.json"
            ),
        },
        "bindings": {
            role: {"path": rel(path), "sha256": sha256(path)}
            for role, path in bindings.items()
        },
    }
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
