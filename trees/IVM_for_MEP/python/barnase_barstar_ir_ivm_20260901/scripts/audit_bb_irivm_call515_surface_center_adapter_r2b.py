#!/usr/bin/env python3
"""Publish the zero-physical call515 surface-center adapter audit."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
RUNS = ROOT / "runs/diagnostics"
RESULT = RUNS / (
    "bb_irivm_call515_nf14983_terminal_uncommitted_tail_0k_section_"
    "energy_59x28_20260903_r2b.npz"
)
SURFACE_AUDIT = RESULT.with_suffix(".audit.json")
WAL = RESULT.with_suffix(".energy.wal.jsonl")
COORDINATE = RUNS / (
    "bb_irivm_call515_nf14983_terminal_uncommitted_tail_0k_section_"
    "coordinates_59x28_20260903_r2.npz"
)
OUTPUT = RUNS / (
    "bb_irivm_call515_nf14983_terminal_uncommitted_tail_0k_section_"
    "surface_center_adapter_20260903_r2b.audit.json"
)
SCHEMA = "bb-irivm-call515-surface-center-adapter-audit-v1"
STATUS = "PASS_ZERO_PHYSICAL_SURFACE_CENTER_BINDING"
PATH_SHA256 = "5a4ba3c5dc8f7063c1963f0bfd951f82356e60d0ad9c4fc17c0bc8b0d36d8569"
CENTER_STATE = {
    "method": "IVM", "call_per_image": 515, "optimization_N_F": 14983,
    "path_sha256": PATH_SHA256,
    "state_role": "terminal_uncommitted_charged_tail",
    "formally_reportable": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size}


def scalar(saved: Mapping[str, np.ndarray], key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    result = RESULT.resolve(strict=True)
    surface_audit = SURFACE_AUDIT.resolve(strict=True)
    wal = WAL.resolve(strict=True)
    coordinate = COORDINATE.resolve(strict=True)
    audit = json.loads(surface_audit.read_text(encoding="utf-8"))
    require(audit.get("schema") == "bb-irivm-call515-terminal-tail-section-energy-r2-v1-audit"
            and audit.get("status") == "PASS" and audit.get("stage") == "surface",
            "R2B surface audit identity mismatch")
    require(int(audit.get("center_source_call_per_image", -1)) == 515
            and int(audit.get("center_source_optimization_N_F", -1)) == 14983
            and int(audit.get("force_calls", -1)) == 0
            and int(audit.get("optimization_N_F_contribution", -1)) == 0,
            "R2B surface audit state/call mismatch")
    with np.load(coordinate, allow_pickle=False) as saved:
        require(str(scalar(saved, "returned_terminal_path32_sha256")) == PATH_SHA256,
                "coordinate manifest path SHA mismatch")
    with np.load(result, allow_pickle=False) as saved:
        require(str(scalar(saved, "schema"))
                == "bb-irivm-call515-terminal-tail-section-energy-r2-v1-result"
                and str(scalar(saved, "status")) == "PASS"
                and int(scalar(saved, "center_source_call_per_image")) == 515
                and int(scalar(saved, "center_source_optimization_N_F")) == 14983
                and bool(scalar(saved, "returned_terminal_state"))
                and bool(scalar(saved, "terminal_uncommitted_charged_tail"))
                and not bool(scalar(saved, "formally_reportable")),
                "R2B result semantics mismatch")
        require(saved["surface_coordinates_A"].shape[:2] == (59, 28)
                and saved["surface_energy_eV"].shape == (59, 28)
                and np.all(np.isfinite(saved["surface_energy_eV"]))
                and int(scalar(saved, "fresh_energy_only_attempts")) == 1650
                and int(scalar(saved, "inherited_endpoint_energy_values")) == 2
                and int(scalar(saved, "force_calls")) == 0
                and int(scalar(saved, "optimization_N_F_contribution")) == 0,
                "R2B result grid/call mismatch")
    records = [json.loads(line) for line in wal.read_text(encoding="utf-8").splitlines()]
    require(records and int(records[0].get("maximum_attempt_started_records", -1)) == 1650,
            "R2B WAL header mismatch")
    starts = sum(item.get("event") == "attempt_started" for item in records[1:])
    successes = sum(item.get("event") == "attempt_succeeded" for item in records[1:])
    failures = sum(item.get("event") == "attempt_failed" for item in records[1:])
    publications = sum(item.get("event") == "result_published" for item in records[1:])
    require((starts, successes, failures, publications) == (1650, 1650, 0, 1),
            "R2B WAL lifetime mismatch")
    payload = {
        "schema": SCHEMA, "status": STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "auditor": binding(THIS_FILE),
        "surface_result": binding(result),
        "surface_audit": binding(surface_audit),
        "surface_energy_wal": binding(wal),
        "surface_coordinate_manifest": binding(coordinate),
        "center_state": CENTER_STATE,
        "stored_surface_energy_values": 1652,
        "fresh_energy_only_calls": 1650,
        "inherited_endpoint_energy_values": 2,
        "physical_force_calls": 0, "openmm_contexts_constructed": 0,
        "optimization_N_F_contribution": 0,
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps({"status": STATUS, "adapter_audit": binding(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
