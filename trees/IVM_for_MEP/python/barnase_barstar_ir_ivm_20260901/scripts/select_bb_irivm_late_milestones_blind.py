#!/usr/bin/env python3
"""Preregister and later apply a metric-blind IR--IVM milestone selector.

The preregistration can be frozen before the remote all-path replay result
exists.  Selection is permitted only after the exact output-only replay
verifier emits a PASS receipt.  Candidate ranking reads only call number,
optimization N_F, reportability/transaction flags, reparameterization commit
state, and coordinate-path identity.  It never reads I_perp, energy, force, or
barrier values.

The highest eligible path is reserved first.  Five fixed cost anchors then
choose the nearest still-unused coordinate path, with ties resolved by lower
N_F and then lower call.  This keeps six distinct late milestones when at
least six are available, even when the highest legal path is also near the
8982 anchor.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

PREREG_SCHEMA = "bb-irivm-late-milestone-blind-selection-preregistration-v1"
SELECTION_SCHEMA = "bb-irivm-late-milestone-blind-selection-v1"
PREREG_STATUS = "FROZEN_BEFORE_REMOTE_RESULT_AND_BEFORE_ANY_NEW_METRIC_INSPECTION"
SELECTION_STATUS = "FROZEN_FROM_PASS_EXACT_REPLAY_WITHOUT_METRIC_RANKING"
VERIFICATION_SCHEMA = "bb-irivm-selected-all-paths-replay-verification-v2"
VERIFICATION_STATUS = "PASS_EXACT_FULL_HISTORY_OUTPUT_ONLY_REPLAY"

ANCHORS = (2000, 4000, 6000, 8000, 8982)
EARLY_FOUR_NF = (60, 150, 300, 1329)
MAXIMUM_NF = 9990
EXPECTED_PATH_SHAPE = (32, 9396)
N_DENSE = 500

DEFAULT_REPLAY_REGISTRATION = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_SELECTED_ALL_PATHS_REPLAY_NF9990_20260902_R2.json"
)
DEFAULT_REPLAY_AUTHORIZATION = (
    ROOT
    / "registrations/execution_authorizations/"
    "bb_irivm_a1e0_r01_m05_inf9990_allpaths_r2.json"
)
RUNNER = HERE / "run_selected_ivm_all_paths_replay_r2.py"
VERIFIER = HERE / "verify_selected_ivm_all_paths_replay_r2.py"
CANONICAL_PREREGISTRATION = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_LATE_MILESTONE_SELECTION_PREREG_20260902_R1.json"
)
CANONICAL_SELECTION = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_LATE_MILESTONE_SELECTION_20260902_R1.json"
)

ALLOWED_INVENTORY_FIELDS = {
    "history_index",
    "call_per_image",
    "optimization_N_F",
    "selected_path_sha256",
    "reparameterization_committed",
    "fully_evaluated_current_path",
    "formally_reportable",
    "terminal_uncommitted_charged_tail",
    "exclusion_reason",
}
FORBIDDEN_SELECTION_FIELDS = {
    "discrete_I_perp_eV2_per_A",
    "i_perp",
    "energy",
    "energies",
    "force",
    "forces",
    "barrier",
    "Delta_E",
    "densified_Delta_E_kcal_per_mol",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def write_exclusive(path: Path, payload: dict) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def freeze_preregistration(
    output: Path, replay_registration: Path, replay_authorization: Path
) -> None:
    registration = replay_registration.expanduser().resolve(strict=True)
    authorization = replay_authorization.expanduser().resolve(strict=True)
    registration_payload = read_json(registration)
    authorization_payload = read_json(authorization)
    require(
        registration_payload.get("schema")
        == "bb-irivm-selected-all-paths-output-only-replay-v2",
        "all-path registration schema mismatch",
    )
    require(
        registration_payload.get("status")
        == "FROZEN_BEFORE_FIRST_ALL_PATHS_REPLAY_PHYSICAL_CALL",
        "all-path registration status mismatch",
    )
    require(
        authorization_payload.get("schema")
        == "bb-irivm-selected-all-paths-output-only-replay-authorization-v2"
        and authorization_payload.get("enabled") is True,
        "all-path replay authorization is invalid",
    )
    require(
        authorization_payload.get("registration") == bind(registration),
        "authorization does not bind the all-path registration",
    )
    payload = {
        "schema": PREREG_SCHEMA,
        "status": PREREG_STATUS,
        "created_utc": utc_now(),
        "purpose": (
            "select approximately six late, fully evaluated, committed IR--IVM "
            "paths by optimization cost only, before any new diagnostic metric is read"
        ),
        "result_or_verification_bound_at_preregistration": False,
        "selection_may_freeze_only_after_exact_verification_PASS": True,
        "early_four_already_covered_optimization_N_F": list(EARLY_FOUR_NF),
        "late_eligibility": {
            "optimization_N_F_strictly_greater_than": max(EARLY_FOUR_NF),
            "optimization_N_F_less_than_or_equal_to": MAXIMUM_NF,
            "fully_evaluated_current_path": True,
            "reparameterization_committed": True,
            "formally_reportable": True,
            "terminal_uncommitted_charged_tail": False,
            "distinct_selected_path_sha256": True,
            "terminal_partial_tail_never_eligible": True,
        },
        "deterministic_selection": {
            "number_of_distinct_paths_required": 6,
            "role_order": [
                "highest_legal_optimization_N_F_not_above_9990",
                *[f"nearest_unused_to_NF{target}" for target in ANCHORS],
            ],
            "highest_role_rule": "maximum (optimization_N_F, call_per_image)",
            "anchor_targets_optimization_N_F": list(ANCHORS),
            "anchor_rule": (
                "minimum (absolute_NF_distance, optimization_N_F, call_per_image) "
                "among remaining distinct coordinate paths"
            ),
            "selection_fields_allowed": sorted(ALLOWED_INVENTORY_FIELDS),
            "metric_fields_forbidden": sorted(FORBIDDEN_SELECTION_FIELDS),
            "metric_values_consulted": False,
        },
        "future_joint_EF_protocol": {
            "coordinate_space": "unweighted_3N_cartesian",
            "sampling": "uniform_cartesian_arclength",
            "interpolation": "piecewise_linear",
            "number_of_points_per_selected_path": N_DENSE,
            "include_both_endpoints": True,
            "one_joint_energy_full_force_call_per_point": True,
            "store_dense_coordinates": True,
            "store_all_energies": True,
            "store_all_full_force_vectors": True,
            "maximum_dense_attempt_started_records": 6 * N_DENSE,
            "optimization_N_F_contribution": 0,
            "future_evaluation_contract_requires_frozen_selection": True,
        },
        "bindings": {
            "all_paths_replay_registration": bind(registration),
            "all_paths_replay_authorization": bind(authorization),
            "all_paths_runner": bind(RUNNER),
            "all_paths_exact_verifier": bind(VERIFIER),
            "selector": bind(Path(__file__)),
        },
    }
    write_exclusive(output, payload)


def validate_preregistration(path: Path) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    payload = read_json(resolved)
    require(
        payload.get("schema") == PREREG_SCHEMA
        and payload.get("status") == PREREG_STATUS,
        "invalid blind-selection preregistration",
    )
    for role, record in payload.get("bindings", {}).items():
        require(bind(Path(record["path"])) == record, f"preregistration binding drift: {role}")
    require(
        payload["bindings"]["selector"] == bind(Path(__file__)),
        "selector source drift",
    )
    require(
        payload["deterministic_selection"]["metric_values_consulted"] is False,
        "preregistration permits metric inspection",
    )
    return payload


def project_inventory_row(raw: dict) -> dict:
    """Copy only preregistered state/identity fields; never touch metric keys."""

    require(isinstance(raw, dict), "verification inventory row is not an object")
    missing = ALLOWED_INVENTORY_FIELDS - set(raw)
    require(not missing, f"verification inventory row lacks fields: {sorted(missing)}")
    return {key: raw[key] for key in ALLOWED_INVENTORY_FIELDS}


def eligible_rows(inventory: list[dict]) -> list[dict]:
    result = []
    for raw in inventory:
        row = project_inventory_row(raw)
        nf = int(row["optimization_N_F"])
        call = int(row["call_per_image"])
        if not (
            nf > max(EARLY_FOUR_NF)
            and nf <= MAXIMUM_NF
            and bool(row["fully_evaluated_current_path"])
            and bool(row["reparameterization_committed"])
            and bool(row["formally_reportable"])
            and not bool(row["terminal_uncommitted_charged_tail"])
        ):
            continue
        path_hash = str(row["selected_path_sha256"])
        require(len(path_hash) == 64, "eligible row has invalid path SHA-256")
        result.append(
            {
                "history_index": int(row["history_index"]),
                "call_per_image": call,
                "optimization_N_F": nf,
                "selected_path_sha256": path_hash,
                "reparameterization_committed": True,
                "fully_evaluated_current_path": True,
                "formally_reportable": True,
                "terminal_uncommitted_charged_tail": False,
            }
        )
    require(result, "exact verifier contains no eligible late committed path")
    return sorted(result, key=lambda row: (row["optimization_N_F"], row["call_per_image"]))


def select_rows(inventory: list[dict]) -> list[dict]:
    candidates = eligible_rows(inventory)
    unique_hashes = {row["selected_path_sha256"] for row in candidates}
    require(len(unique_hashes) >= 6, "fewer than six distinct eligible late paths")
    highest = max(candidates, key=lambda row: (row["optimization_N_F"], row["call_per_image"]))
    selected = [{**highest, "selection_role": "highest_legal_optimization_N_F_not_above_9990"}]
    used = {highest["selected_path_sha256"]}
    for target in ANCHORS:
        remaining = [row for row in candidates if row["selected_path_sha256"] not in used]
        require(remaining, f"no unused path remains for anchor NF{target}")
        chosen = min(
            remaining,
            key=lambda row: (
                abs(row["optimization_N_F"] - target),
                row["optimization_N_F"],
                row["call_per_image"],
            ),
        )
        selected.append({**chosen, "selection_role": f"nearest_unused_to_NF{target}"})
        used.add(chosen["selected_path_sha256"])
    require(len(selected) == len(used) == 6, "blind selector did not produce six distinct paths")
    return sorted(selected, key=lambda row: (row["optimization_N_F"], row["call_per_image"]))


def freeze_selection(
    preregistration_path: Path, verification_path: Path, output: Path
) -> None:
    preregistration_path = preregistration_path.expanduser().resolve(strict=True)
    verification_path = verification_path.expanduser().resolve(strict=True)
    preregistration = validate_preregistration(preregistration_path)
    verification = read_json(verification_path)
    require(
        verification.get("schema") == VERIFICATION_SCHEMA
        and verification.get("status") == VERIFICATION_STATUS,
        "all-path exact verification is not PASS v2",
    )
    bindings = verification.get("bindings", {})
    require(
        bindings.get("registration")
        == preregistration["bindings"]["all_paths_replay_registration"],
        "verification registration differs from preregistration",
    )
    require(
        bindings.get("authorization")
        == preregistration["bindings"]["all_paths_replay_authorization"],
        "verification authorization differs from preregistration",
    )
    for role in ("replay_result", "replay_audit"):
        record = bindings.get(role)
        require(isinstance(record, dict) and bind(Path(record["path"])) == record, f"verification {role} binding drift")
    inventory = verification.get("path_inventory")
    require(isinstance(inventory, list) and inventory, "verification lacks path inventory")
    selected = select_rows(inventory)
    payload = {
        "schema": SELECTION_SCHEMA,
        "status": SELECTION_STATUS,
        "created_utc": utc_now(),
        "selection_metric_values_consulted": False,
        "selection_inputs_used": sorted(ALLOWED_INVENTORY_FIELDS),
        "selected_path_count": len(selected),
        "selected_paths": selected,
        "future_joint_EF_protocol": preregistration["future_joint_EF_protocol"],
        "terminal_partial_tail_selected": False,
        "optimization_N_F_contribution": 0,
        "bindings": {
            "preregistration": bind(preregistration_path),
            "exact_all_paths_verification": bind(verification_path),
            "all_paths_replay_result": bindings["replay_result"],
            "all_paths_replay_audit": bindings["replay_audit"],
            "selector": bind(Path(__file__)),
        },
    }
    write_exclusive(output, payload)


def preflight(preregistration_path: Path) -> dict:
    payload = validate_preregistration(preregistration_path)
    return {
        "status": "PASS_ZERO_OPENMM_SELECTION_PREFLIGHT",
        "preregistration": bind(preregistration_path),
        "result_or_verification_required_now": False,
        "selection_frozen_now": False,
        "anchor_targets_optimization_N_F": list(ANCHORS),
        "highest_legal_cap_optimization_N_F": MAXIMUM_NF,
        "number_of_future_paths": 6,
        "future_joint_EF_points_per_path": N_DENSE,
        "future_joint_EF_attempts": 6 * N_DENSE,
        "terminal_partial_tail_eligible": False,
        "metric_values_consulted": False,
        "OpenMM_imported_or_called": False,
        "bindings_validated": sorted(payload["bindings"]),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    prereg = modes.add_parser("preregister")
    prereg.add_argument("--output", type=Path, required=True)
    prereg.add_argument("--replay-registration", type=Path, default=DEFAULT_REPLAY_REGISTRATION)
    prereg.add_argument("--replay-authorization", type=Path, default=DEFAULT_REPLAY_AUTHORIZATION)
    check = modes.add_parser("preflight")
    check.add_argument("--preregistration", type=Path, required=True)
    select = modes.add_parser("select")
    select.add_argument("--preregistration", type=Path, required=True)
    select.add_argument("--verification", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.mode == "preregister":
        freeze_preregistration(args.output, args.replay_registration, args.replay_authorization)
        print(args.output.expanduser().resolve(strict=True))
        print(sha256_file(args.output.expanduser().resolve(strict=True)))
    elif args.mode == "preflight":
        print(json.dumps(preflight(args.preregistration), indent=2, sort_keys=True))
    else:
        freeze_selection(args.preregistration, args.verification, args.output)
        destination = args.output.expanduser().resolve(strict=True)
        print(destination)
        print(sha256_file(destination))


if __name__ == "__main__":
    main()
