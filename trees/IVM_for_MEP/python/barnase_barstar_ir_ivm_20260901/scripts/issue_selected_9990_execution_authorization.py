#!/usr/bin/env python3
"""Issue one immutable authorization for the selected 9990-N_F run."""

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
    parser.add_argument("--label", required=True)
    parser.add_argument("--design", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    design_path = args.design.expanduser().resolve(strict=True)
    design = json.loads(design_path.read_text(encoding="utf-8"))
    require(design.get("status") == "FROZEN_BEFORE_FIRST_9990_PHYSICAL_CALL", "9990 design is not frozen")
    registration = root / "registrations" / "production" / f"{args.label}.json"
    require(registration.is_file(), f"missing production registration: {registration}")
    registered = json.loads(registration.read_text(encoding="utf-8"))
    controls = design["selected_setting"]
    require(registered.get("label") == args.label, "label mismatch")
    require(registered.get("tier") == "production", "tier mismatch")
    require(int(registered["internal_nf_budget"]) == 9990, "budget mismatch")
    require(float(registered["alpha"]) == float(controls["alpha"]), "alpha mismatch")
    require(float(registered["residual_ratio_r"]) == float(controls["r"]), "r mismatch")
    require(int(registered["inner_max_steps_m_max"]) == int(controls["m_max"]), "m_max mismatch")
    for name, expected in design["source_closure_sha256"].items():
        require(sha256(root / "src" / name) == expected, f"source drift: {name}")
        require(registered["source_sha256"][name] == expected, f"registration source drift: {name}")
    for name, expected in design["input_closure_sha256"].items():
        require(sha256(root / "inputs" / name) == expected, f"input drift: {name}")
        require(registered["input_sha256"][name] == expected, f"registration input drift: {name}")

    campaign = root / "registrations" / "CAMPAIGN_PREREGISTRATION.json"
    staged = root / "registrations" / "STAGED_SEARCH_PROTOCOL.json"
    migration = root / "contracts" / "MIGRATION_CONTRACT.json"
    parity = root / "reports" / "gates" / "FF_BUILDER_PARITY_ZERO_FORCE.json"
    smoke = root / "reports" / "gates" / "SMOKE_240_GATE.json"
    selection_record = design["bindings"]["selection"]
    selection_path = Path(selection_record["path"]).resolve(strict=True)
    require(sha256(selection_path) == selection_record["sha256"], "selection binding mismatch")
    prefix_record = design["bindings"]["prefix_reproducibility_criterion"]
    prefix_path = Path(prefix_record["path"]).resolve(strict=True)
    require(sha256(prefix_path) == prefix_record["sha256"], "prefix criterion binding mismatch")
    execution = design["execution"]
    payload = {
        "schema": "bb-irivm-selected-9990-execution-authorization-v1",
        "enabled": True,
        "issued_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": args.label,
        "tier": "production",
        "authorization_scope": "one exact fresh non-resumed selected 9990-N_F member",
        "member_registration_sha256": sha256(registration),
        "campaign_preregistration_sha256": sha256(campaign),
        "staged_search_protocol_sha256": sha256(staged),
        "migration_contract_sha256": sha256(migration),
        "ff_builder_parity_gate_sha256": sha256(parity),
        "smoke_240_gate_sha256": sha256(smoke),
        "fixed_endpoint_policy": "hash-bound inherited endpoint E/F; endpoint getState attempts, misses, and optimization NoF must remain zero",
        "diagnostic_feedback": "forbidden during optimization",
        "concurrency": "one BB task on this host, two host-local fcntl locks, jobs=1",
        "registered_controls": {
            "alpha": float(registered["alpha"]),
            "r": float(registered["residual_ratio_r"]),
            "m_max": int(registered["inner_max_steps_m_max"]),
            "optimization_Number_of_Forces": 9990,
        },
        "controlled_scan_gate": {
            "stage": "selected_9990_N_F_reference_host_confirmation",
            "design": str(design_path),
            "design_sha256": sha256(design_path),
            "selection": str(selection_path),
            "selection_sha256": sha256(selection_path),
            "selected_parameter_setting": [
                float(registered["alpha"]),
                float(registered["residual_ratio_r"]),
                int(registered["inner_max_steps_m_max"]),
            ],
            "frozen_reference_host": execution["reference_host"],
            "fresh_from_common_initial_path": True,
            "shorter_prefix_reproduction_required": True,
            "prefix_reproducibility_criterion": str(prefix_path),
            "prefix_reproducibility_criterion_sha256": sha256(prefix_path),
            "minimum_free_disk_bytes_before_child": int(
                execution["minimum_free_disk_bytes_before_child"]
            ),
        },
    }
    destination = root / "registrations" / "execution_authorizations" / f"{args.label}.json"
    require(not destination.exists(), f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
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
    print(f"Saved {destination}")
    print(f"sha256={sha256(destination)}")


if __name__ == "__main__":
    main()
