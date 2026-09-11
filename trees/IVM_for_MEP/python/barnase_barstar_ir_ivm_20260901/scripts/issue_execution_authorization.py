#!/usr/bin/env python3
"""Issue one immutable, hash-bound BB IR-IVM member authorization.

This script performs no OpenMM construction or physical E/F call.  It may
only authorize an already registered member under the frozen campaign,
migration, staged-search, FF-parity, and (for non-smoke runs) smoke gates.
An alpha-confirmation member must additionally bind the immutable two-metric
promotion receipt that selected it for an independent 990-N_F run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


METRIC_POLICY_AMENDMENT_SHA256 = (
    "ca547dd86a119e0dd4b6049e296ae9616d335248c3df1ce1f77f7fcb91500bc5"
)
ALPHA_SCOPE_R2_SHA256 = (
    "9dd4dda96d428df1e7acd07533061cc77f8e8fe55c1d38bdda14f24510be6e0f"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_alpha_promotion(receipt, registered):
    """Return the three promoted alpha values or fail closed."""

    if registered.get("tier") != "screen" or int(
        registered["internal_nf_budget"]
    ) != 990:
        raise RuntimeError(
            "an alpha-promotion receipt may authorize only a screen-tier "
            "independent 990-N_F member"
        )
    if receipt.get("status") != "PASS_ALPHA_SELECTION_FROM_TWO_PRINCIPAL_DIAGNOSTICS":
        raise RuntimeError("alpha-promotion receipt has not passed")
    if receipt.get("metric_policy_amendment_sha256") != METRIC_POLICY_AMENDMENT_SHA256:
        raise RuntimeError("alpha-promotion receipt metric-policy binding mismatch")
    if receipt.get("alpha_scope_R2_sha256") != ALPHA_SCOPE_R2_SHA256:
        raise RuntimeError("alpha-promotion receipt alpha-scope binding mismatch")
    promoted = tuple(
        float(value)
        for value in receipt.get(
            "promoted_alpha_values_for_independent_990_N_F_runs", []
        )
    )
    if len(promoted) != 3 or len(set(promoted)) != 3:
        raise RuntimeError("alpha-promotion receipt must name three distinct values")
    if float(registered["alpha"]) not in promoted:
        raise RuntimeError("member alpha was not promoted by the bound receipt")
    return promoted


def validate_infrastructure_retry(receipt, registered, replacement_label):
    """Validate that a new label replaces only a recorded non-result attempt."""

    if receipt.get("status") != "RECORDED_INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_RESULT":
        raise RuntimeError("infrastructure-retry receipt has not recorded a non-result")
    if receipt.get("replacement_label") != replacement_label:
        raise RuntimeError("infrastructure-retry replacement label mismatch")
    controls = receipt.get("registered_controls", {})
    expected = {
        "alpha": float(registered["alpha"]),
        "r": float(registered["residual_ratio_r"]),
        "m_max": int(registered["inner_max_steps_m_max"]),
        "optimization_Number_of_Forces": int(registered["internal_nf_budget"]),
    }
    observed = {
        "alpha": float(controls.get("alpha")),
        "r": float(controls.get("r")),
        "m_max": int(controls.get("m_max")),
        "optimization_Number_of_Forces": int(
            controls.get("optimization_Number_of_Forces")
        ),
    }
    if observed != expected:
        raise RuntimeError("infrastructure-retry controls mismatch")
    if int(receipt.get("returncode", 0)) == 0:
        raise RuntimeError("infrastructure-retry receipt has no failed return code")
    if receipt.get("scientific_result_exists") is not False:
        raise RuntimeError("infrastructure-retry receipt does not prove absence of result")


def validate_r_controlled_scan(selection, efficiency, registered):
    """Validate one prospectively registered 480-N_F r-scan member."""

    if selection.get("status") != "PASS_ALPHA_CONFIRMATION_TWO_METRIC_SELECTION":
        raise RuntimeError("alpha confirmation selection has not passed")
    alpha_star = float(selection["alpha_star_for_r_controlled_scan"])
    if efficiency.get("status") != "OPERATIVE_PROSPECTIVE_EFFICIENCY_AMENDMENT":
        raise RuntimeError("efficient controlled-scan amendment is not operative")
    stage = efficiency.get("r_controlled_scan", {})
    if registered.get("tier") != "screen" or int(
        registered["internal_nf_budget"]
    ) != int(stage.get("registered_budget_upper_bound", -1)):
        raise RuntimeError("r-controlled member must use the registered screen budget")
    if float(registered["alpha"]) != alpha_star or alpha_star != float(
        stage.get("fixed_alpha")
    ):
        raise RuntimeError("r-controlled member alpha mismatch")
    if int(registered["inner_max_steps_m_max"]) != int(stage.get("fixed_m_max")):
        raise RuntimeError("r-controlled member m_max mismatch")
    new_values = tuple(float(record["r"]) for record in stage.get("new_members", []))
    if float(registered["residual_ratio_r"]) not in new_values:
        raise RuntimeError("r-controlled member is not a registered new r value")
    return new_values


def validate_m_max_controlled_scan(selection, efficiency, registered):
    """Validate one prospectively registered 480-N_F m_max-scan member."""

    if selection.get("status") != "PASS_R_CONTROLLED_SCAN_TWO_METRIC_SELECTION":
        raise RuntimeError("r controlled-scan selection has not passed")
    if efficiency.get("status") != "OPERATIVE_PROSPECTIVE_EFFICIENCY_AMENDMENT":
        raise RuntimeError("efficient controlled-scan amendment is not operative")
    stage = efficiency.get("m_max_controlled_scan", {})
    if registered.get("tier") != "screen" or int(
        registered["internal_nf_budget"]
    ) != int(stage.get("first_budget_upper_bound", -1)):
        raise RuntimeError("m_max-controlled member must use the registered screen budget")
    if float(registered["alpha"]) != float(selection["fixed_alpha"]):
        raise RuntimeError("m_max-controlled member alpha mismatch")
    if float(registered["residual_ratio_r"]) != float(
        selection["r_star_for_m_max_controlled_scan"]
    ):
        raise RuntimeError("m_max-controlled member r mismatch")
    candidates = tuple(int(value) for value in stage.get("candidate_order", []))
    cap = int(registered["inner_max_steps_m_max"])
    if cap not in candidates or cap == candidates[0]:
        raise RuntimeError("m_max-controlled member is not a registered new cap")
    return candidates


def validate_local_interaction(design, registered):
    """Validate one new member of the frozen 2 x 2 x 2 interaction."""

    if design.get("status") != "FROZEN_BEFORE_FIRST_NEW_LOCAL_INTERACTION_PHYSICAL_CALL":
        raise RuntimeError("local-interaction design is not frozen")
    controls = (
        float(registered["alpha"]),
        float(registered["residual_ratio_r"]),
        int(registered["inner_max_steps_m_max"]),
    )
    new_members = {
        (float(row["alpha"]), float(row["r"]), int(row["m_max"]))
        for row in design.get("new_members", [])
    }
    if controls not in new_members:
        raise RuntimeError("member is not a registered new local-interaction setting")
    budget = int(design.get("fixed_controls", {}).get(
        "optimization_Number_of_Forces_upper_bound", -1
    ))
    if registered.get("tier") != "screen" or int(
        registered["internal_nf_budget"]
    ) != budget:
        raise RuntimeError("local-interaction member budget/tier mismatch")
    return sorted(new_members)


def validate_long_budget(design, registered):
    """Validate one of the two frozen 4080-N_F reference-host members."""

    if design.get("status") != "FROZEN_BEFORE_FIRST_4080_PHYSICAL_CALL":
        raise RuntimeError("long-budget design is not frozen")
    execution = design.get("execution", {})
    if registered.get("tier") != execution.get("tier"):
        raise RuntimeError("long-budget tier mismatch")
    if int(registered["internal_nf_budget"]) != int(
        execution.get("optimization_Number_of_Forces_upper_bound", -1)
    ):
        raise RuntimeError("long-budget optimization N_F mismatch")
    controls = (
        float(registered["alpha"]),
        float(registered["residual_ratio_r"]),
        int(registered["inner_max_steps_m_max"]),
    )
    promoted = [
        (float(row["alpha"]), float(row["r"]), int(row["m_max"]))
        for row in design.get("promoted_4080_settings_in_order", [])
    ]
    if len(promoted) != 2 or len(set(promoted)) != 2:
        raise RuntimeError("long-budget design must promote two distinct settings")
    if controls not in promoted:
        raise RuntimeError("member is not promoted by the long-budget design")
    host = design.get("reference_host", {})
    if not host.get("hostname") or not host.get("address"):
        raise RuntimeError("long-budget reference host was not frozen")
    return promoted, host


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--tier", choices=("smoke", "screen", "production"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--alpha-promotion-receipt",
        type=Path,
        help=(
            "immutable alpha-selection receipt required for a promoted "
            "independent 990-N_F alpha-confirmation member"
        ),
    )
    parser.add_argument(
        "--infrastructure-retry-receipt",
        type=Path,
        help="immutable failure receipt required for a replacement attempt label",
    )
    parser.add_argument("--r-scan-selection-receipt", type=Path)
    parser.add_argument("--m-scan-selection-receipt", type=Path)
    parser.add_argument("--local-interaction-design", type=Path)
    parser.add_argument("--long-budget-design", type=Path)
    parser.add_argument("--efficiency-amendment", type=Path)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve(strict=True)
    registration = root / "registrations" / args.tier / f"{args.label}.json"
    registered = json.loads(registration.read_text(encoding="utf-8"))
    if registered.get("label") != args.label or registered.get("tier") != args.tier:
        raise RuntimeError("member registration label/tier mismatch")

    campaign = root / "registrations" / "CAMPAIGN_PREREGISTRATION.json"
    contract = root / "contracts" / "MIGRATION_CONTRACT.json"
    staged = root / "registrations" / "STAGED_SEARCH_PROTOCOL.json"
    ff_gate = root / "reports" / "gates" / "FF_BUILDER_PARITY_ZERO_FORCE.json"
    ff_payload = json.loads(ff_gate.read_text(encoding="utf-8"))
    if ff_payload.get("status") != "PASS":
        raise RuntimeError("FF builder parity gate has not passed")

    payload = {
        "schema": "bb-irivm-per-member-execution-authorization-v1",
        "issued_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "enabled": True,
        "label": args.label,
        "tier": args.tier,
        "registered_controls": {
            "alpha": registered["alpha"],
            "r": registered["residual_ratio_r"],
            "m_max": registered["inner_max_steps_m_max"],
            "optimization_Number_of_Forces": registered["internal_nf_budget"],
        },
        "member_registration_sha256": sha256(registration),
        "campaign_preregistration_sha256": sha256(campaign),
        "migration_contract_sha256": sha256(contract),
        "staged_search_protocol_sha256": sha256(staged),
        "ff_builder_parity_gate_sha256": sha256(ff_gate),
        "fixed_endpoint_policy": (
            "hash-bound inherited endpoint E/F; endpoint getState attempts, "
            "misses, and optimization NoF must remain zero"
        ),
        "concurrency": "one BB task on this host, two host-local fcntl locks, jobs=1",
        "diagnostic_feedback": "forbidden during optimization",
        "authorization_scope": "one exact fresh non-resumed registered member",
    }
    if args.tier != "smoke":
        smoke_gate = root / "reports" / "gates" / "SMOKE_240_GATE.json"
        smoke_payload = json.loads(smoke_gate.read_text(encoding="utf-8"))
        if smoke_payload.get("status") != "PASS":
            raise RuntimeError("smoke gate has not passed")
        payload["smoke_240_gate_sha256"] = sha256(smoke_gate)

    if args.alpha_promotion_receipt is not None:
        receipt_path = args.alpha_promotion_receipt.expanduser().resolve(strict=True)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        promoted = validate_alpha_promotion(receipt, registered)
        metric_policy = root / "registrations" / "USER_METRIC_POLICY_AMENDMENT_20260901_R1.json"
        alpha_scope = root / "registrations" / "USER_ALPHA_SCOPE_AMENDMENT_20260901_R2.json"
        if sha256(metric_policy) != METRIC_POLICY_AMENDMENT_SHA256:
            raise RuntimeError("operative metric-policy amendment hash mismatch")
        if sha256(alpha_scope) != ALPHA_SCOPE_R2_SHA256:
            raise RuntimeError("operative alpha-scope amendment hash mismatch")
        payload["promotion_gate"] = {
            "stage": "alpha_controlled_scan_independent_990_N_F_confirmation",
            "selection_receipt": str(receipt_path),
            "selection_receipt_sha256": sha256(receipt_path),
            "promoted_alpha_values": list(promoted),
            "metric_policy_amendment_sha256": METRIC_POLICY_AMENDMENT_SHA256,
            "alpha_scope_R2_sha256": ALPHA_SCOPE_R2_SHA256,
            "same_state_two_metric_selection_required": True,
        }

    if args.infrastructure_retry_receipt is not None:
        retry_path = args.infrastructure_retry_receipt.expanduser().resolve(strict=True)
        retry_receipt = json.loads(retry_path.read_text(encoding="utf-8"))
        validate_infrastructure_retry(retry_receipt, registered, args.label)
        payload["infrastructure_retry_gate"] = {
            "failure_receipt": str(retry_path),
            "failure_receipt_sha256": sha256(retry_path),
            "original_label": retry_receipt["original_label"],
            "replacement_label": args.label,
            "rule": "fresh non-resumed replacement; original attempt remains excluded",
        }

    if args.r_scan_selection_receipt is not None:
        if args.efficiency_amendment is None:
            raise RuntimeError("r scan authorization requires the efficiency amendment")
        selection_path = args.r_scan_selection_receipt.expanduser().resolve(strict=True)
        efficiency_path = args.efficiency_amendment.expanduser().resolve(strict=True)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        efficiency = json.loads(efficiency_path.read_text(encoding="utf-8"))
        if efficiency.get("alpha_confirmation_selection_sha256") != sha256(
            selection_path
        ):
            raise RuntimeError("efficiency amendment/alpha selection hash mismatch")
        new_values = validate_r_controlled_scan(selection, efficiency, registered)
        payload["controlled_scan_gate"] = {
            "stage": "r_controlled_scan_480_N_F",
            "alpha_confirmation_selection": str(selection_path),
            "alpha_confirmation_selection_sha256": sha256(selection_path),
            "efficiency_amendment": str(efficiency_path),
            "efficiency_amendment_sha256": sha256(efficiency_path),
            "registered_new_r_values": list(new_values),
            "same_state_two_metric_selection_required": True,
        }

    if args.m_scan_selection_receipt is not None:
        if args.efficiency_amendment is None:
            raise RuntimeError("m_max scan authorization requires the efficiency amendment")
        if args.r_scan_selection_receipt is not None:
            raise RuntimeError("one authorization cannot combine r and m_max scan gates")
        selection_path = args.m_scan_selection_receipt.expanduser().resolve(strict=True)
        efficiency_path = args.efficiency_amendment.expanduser().resolve(strict=True)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        efficiency = json.loads(efficiency_path.read_text(encoding="utf-8"))
        candidates = validate_m_max_controlled_scan(selection, efficiency, registered)
        payload["controlled_scan_gate"] = {
            "stage": "m_max_controlled_scan_480_N_F",
            "r_controlled_scan_selection": str(selection_path),
            "r_controlled_scan_selection_sha256": sha256(selection_path),
            "efficiency_amendment": str(efficiency_path),
            "efficiency_amendment_sha256": sha256(efficiency_path),
            "registered_m_max_candidate_order": list(candidates),
            "sequential_exact_alias_rule_required": True,
            "same_state_two_metric_selection_required": True,
        }

    if args.local_interaction_design is not None:
        if args.r_scan_selection_receipt is not None or args.m_scan_selection_receipt is not None:
            raise RuntimeError("one authorization cannot combine controlled-scan and interaction gates")
        design_path = args.local_interaction_design.expanduser().resolve(strict=True)
        design = json.loads(design_path.read_text(encoding="utf-8"))
        members = validate_local_interaction(design, registered)
        bindings = design.get("selection_bindings", {})
        verified = {}
        for name in ("alpha", "r", "m_max"):
            record = bindings.get(name, {})
            bound_path = (design_path.parent / record.get("path", "")).resolve(strict=True)
            if sha256(bound_path) != record.get("sha256"):
                raise RuntimeError(f"local-interaction {name} selection hash mismatch")
            verified[name] = {
                "path": str(bound_path),
                "sha256": record["sha256"],
            }
        efficiency = root / "registrations" / "USER_EFFICIENT_CONTROLLED_SCAN_AMENDMENT_20260901_R1.json"
        if sha256(efficiency) != bindings.get("efficient_scan_amendment_sha256"):
            raise RuntimeError("local-interaction efficiency-amendment hash mismatch")
        if sha256(staged) != bindings.get("staged_search_protocol_sha256"):
            raise RuntimeError("local-interaction staged-protocol hash mismatch")
        payload["controlled_scan_gate"] = {
            "stage": "registered_local_interaction_2x2x2_480_N_F",
            "design": str(design_path),
            "design_sha256": sha256(design_path),
            "new_parameter_settings": [list(values) for values in members],
            "verified_selection_bindings": verified,
            "same_state_two_metric_selection_required": True,
        }

    if args.long_budget_design is not None:
        if any(
            value is not None
            for value in (
                args.alpha_promotion_receipt,
                args.r_scan_selection_receipt,
                args.m_scan_selection_receipt,
                args.local_interaction_design,
            )
        ):
            raise RuntimeError("one authorization cannot combine long-budget and earlier-stage gates")
        design_path = args.long_budget_design.expanduser().resolve(strict=True)
        design = json.loads(design_path.read_text(encoding="utf-8"))
        promoted, host = validate_long_budget(design, registered)
        for record in design.get("selection_bindings", {}).values():
            bound_path = Path(record["path"]).expanduser().resolve(strict=True)
            if sha256(bound_path) != record["sha256"]:
                raise RuntimeError("long-budget selection binding hash mismatch")
        amendment_record = design.get("efficient_scan_amendment", {})
        amendment_path = Path(amendment_record["path"]).expanduser().resolve(strict=True)
        if sha256(amendment_path) != amendment_record["sha256"]:
            raise RuntimeError("long-budget efficiency-amendment hash mismatch")
        payload["controlled_scan_gate"] = {
            "stage": "two_candidate_4080_N_F_reference_host_confirmation",
            "design": str(design_path),
            "design_sha256": sha256(design_path),
            "promoted_parameter_settings": [list(values) for values in promoted],
            "frozen_reference_host": host,
            "fresh_from_common_initial_path": True,
            "same_state_two_metric_selection_required_after_both_runs": True,
            "shorter_prefix_reproduction_required": True,
        }

    destination = (
        root / "registrations" / "execution_authorizations" / f"{args.label}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    print(f"issued immutable authorization: {destination}")


if __name__ == "__main__":
    main()
