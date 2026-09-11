#!/usr/bin/env python3
"""Freeze the two 4080-N_F candidates and their common reference host.

This is a zero-OpenMM, zero-Force operation.  It pools the already sealed
controlled-scan and local-interaction selections, forms a cost--precision
Pareto set in (optimization N_F, discrete I_perp, densified Delta E_500),
and selects the first two settings by the preregistered final ranking.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


I_TARGET = 1468.5712347872072
DE_TARGET = 200.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dominates(left: dict, right: dict) -> bool:
    keys = ("optimization_N_F", "discrete_I_perp", "densified_Delta_E_kcal_per_mol")
    return all(left[key] <= right[key] for key in keys) and any(
        left[key] < right[key] for key in keys
    )


def ranking_key(row: dict) -> tuple:
    first_passage = (
        row["discrete_I_perp"] <= I_TARGET
        and row["densified_Delta_E_kcal_per_mol"] <= DE_TARGET
    )
    return (
        0 if first_passage else 1,
        row["optimization_N_F"] if first_passage else row["joint_target_score"],
        row["discrete_I_perp"] / I_TARGET,
        row["densified_Delta_E_kcal_per_mol"] / DE_TARGET,
        row["optimization_N_F"],
        row["alpha"],
        row["r"],
        row["m_max"],
    )


def normalize(row: dict, *, alpha: float, r: float, m_max: int, source: str) -> dict:
    result = {
        "alpha": float(alpha),
        "r": float(r),
        "m_max": int(m_max),
        "optimization_N_F": int(row["optimization_N_F"]),
        "discrete_I_perp": float(row["discrete_I_perp"]),
        "densified_Delta_E_kcal_per_mol": float(
            row["densified_Delta_E_kcal_per_mol"]
        ),
        "source_selection": source,
        "source_label": row["label"],
    }
    result["joint_target_score"] = max(
        result["discrete_I_perp"] / I_TARGET,
        result["densified_Delta_E_kcal_per_mol"] / DE_TARGET,
    )
    return result


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reference-hostname", required=True)
    parser.add_argument("--reference-host-address", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)

    names = {
        "alpha": "BB_IRIVM_ALPHA_CONFIRMATION_SELECTION_20260901_R1.json",
        "r": "BB_IRIVM_R_SCAN_SELECTION_20260901_R1.json",
        "m_max": "BB_IRIVM_M_MAX_SCAN_SELECTION_20260902_R1.json",
        "interaction": "BB_IRIVM_LOCAL_INTERACTION_SELECTION_20260902_R1.json",
    }
    paths = {key: root / "reports" / "selections" / name for key, name in names.items()}
    data = {key: json.loads(path.read_text()) for key, path in paths.items()}
    expected_status = {
        "alpha": "PASS_ALPHA_CONFIRMATION_TWO_METRIC_SELECTION",
        "r": "PASS_R_CONTROLLED_SCAN_TWO_METRIC_SELECTION",
        "m_max": "PASS_M_MAX_CONTROLLED_SCAN_TWO_METRIC_SELECTION",
        "interaction": "PASS_LOCAL_INTERACTION_TWO_METRIC_SELECTION",
    }
    for key, status in expected_status.items():
        if data[key].get("status") != status:
            raise RuntimeError(f"{key} selection has not passed")

    rows = []
    for row in data["alpha"]["per_alpha_selected_state_in_confirmation_order"]:
        rows.append(normalize(row, alpha=row["alpha"], r=0.5, m_max=5, source="alpha"))
    for row in data["r"]["per_r_selected_state"]:
        rows.append(normalize(row, alpha=10, r=row["r"], m_max=5, source="r"))
    for row in data["m_max"]["per_m_max_selected_state"]:
        rows.append(normalize(row, alpha=10, r=0.1, m_max=row["m_max"], source="m_max"))
    for row in data["interaction"]["per_setting_selected_state"]:
        rows.append(
            normalize(
                row,
                alpha=row["alpha"],
                r=row["r"],
                m_max=row["m_max"],
                source="interaction",
            )
        )

    # A setting can occur in several stages.  Retain its best sealed state by
    # the formal target-based ranking without mixing the two metrics.
    best = {}
    for row in rows:
        key = (row["alpha"], row["r"], row["m_max"])
        if key not in best or ranking_key(row) < ranking_key(best[key]):
            best[key] = row
    settings = sorted(best.values(), key=ranking_key)
    pareto = [
        row for row in settings if not any(dominates(other, row) for other in settings)
    ]
    pareto.sort(key=ranking_key)
    if len(pareto) < 2:
        raise RuntimeError("full cost--precision Pareto set has fewer than two settings")
    promoted = pareto[:2]

    payload = {
        "schema": "bb-irivm-long-budget-design-v1",
        "status": "FROZEN_BEFORE_FIRST_4080_PHYSICAL_CALL",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_bindings": {
            key: {"path": str(paths[key]), "sha256": sha256(paths[key])}
            for key in names
        },
        "efficient_scan_amendment": {
            "path": str(root / "registrations" / "USER_EFFICIENT_CONTROLLED_SCAN_AMENDMENT_20260901_R1.json"),
            "sha256": sha256(root / "registrations" / "USER_EFFICIENT_CONTROLLED_SCAN_AMENDMENT_20260901_R1.json"),
        },
        "pareto_definition": (
            "no other tested setting is no worse in optimization N_F, discrete I_perp, "
            "and arc-length-500 densified Delta E, and strictly better in at least one"
        ),
        "ranking_targets": {
            "discrete_I_perp": I_TARGET,
            "densified_Delta_E_kcal_per_mol": DE_TARGET,
        },
        "full_cost_precision_pareto_set": pareto,
        "promoted_4080_settings_in_order": promoted,
        "reference_host": {
            "hostname": args.reference_hostname,
            "address": args.reference_host_address,
            "architecture": "arm64",
            "processor": "Apple M4",
            "selection_basis": (
                "chosen before either 4080 call because measured throughput is about "
                "twice the local Mac; both candidates run serially on this same host"
            ),
        },
        "execution": {
            "tier": "screen",
            "optimization_Number_of_Forces_upper_bound": 4080,
            "fresh_from_common_initial_path": True,
            "candidate_count": 2,
            "same_reference_host": True,
            "one_BB_task_on_host": True,
            "jobs": 1,
            "all_thread_caps": 1,
            "host_local_fcntl_locks_required": True,
            "diagnostic_feedback_during_optimization": "forbidden",
        },
        "accounting": {
            "paid_images": 30,
            "fixed_endpoint_optimization_Number_of_Forces": 0,
            "all_getState_attempts_for_internal_images_count": True,
        },
        "after_both_4080_runs": (
            "seal and audit both runs, reproduce every fully closed shorter-budget prefix, "
            "then apply the same offline arc-length-500 diagnostic before selecting 9990"
        ),
    }
    output = args.output if args.output.is_absolute() else root / args.output
    write_exclusive(output, payload)
    print(f"Saved {output}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
