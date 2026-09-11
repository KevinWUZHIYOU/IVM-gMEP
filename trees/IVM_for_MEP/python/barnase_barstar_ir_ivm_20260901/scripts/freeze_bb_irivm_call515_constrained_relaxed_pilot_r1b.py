#!/usr/bin/env python3
"""R1B zero-physics freezer for the call-515 constrained pilot.

R1 correctly rejected publication because the source ``linspace`` values are
not bitwise identical to the human-readable index/58 representation.  R1B
first verifies numerical identity, then stores the explicitly registered s/n
values that its validator compares bit-for-bit.  No OpenMM module is imported.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
BASE = THIS_FILE.with_name("freeze_bb_irivm_call515_constrained_relaxed_pilot_r1.py")
spec = importlib.util.spec_from_file_location("bb_constrained_pilot_r1_base", BASE)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

base.THIS_FILE = THIS_FILE
base.CONTRACT_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-coordinate-contract-r1b-v1"
base.MANIFEST_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-coordinate-manifest-r1b-v1"
base.AUDIT_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-coordinate-audit-r1b-v1"
base.CONTRACT = base.REGISTRATION_DIR / (
    "BB_IRIVM_CALL515_TERMINAL_UNCOMMITTED_TAIL_CONSTRAINED_RELAXED_"
    "PILOT_COORDINATE_CONTRACT_R1B_20260903.json"
)
base.OUTPUT = base.OUTPUT_DIR / (
    "bb_irivm_call515_nf14983_terminal_uncommitted_tail_constrained_relaxed_"
    "pilot_coordinates_15x9_20260903_r1b.npz"
)
base.AUDIT = base.OUTPUT.with_suffix(".audit.json")

_source_validate = base.validate_source
REGISTERED_S = base.S_INDICES.astype(np.float64) / 58.0
REGISTERED_N = np.asarray(
    [-0.12, -0.06, 0.0, 0.12, 0.24, 0.36, 0.48, 0.60, 0.69],
    dtype=np.float64,
)


def validate_source_registered_values():
    data = _source_validate()
    base.require(np.allclose(data["s"], REGISTERED_S, atol=2e-16, rtol=0.0),
                 "source s differs numerically from registered grid")
    base.require(np.allclose(data["n"], REGISTERED_N, atol=2e-16, rtol=0.0),
                 "source n differs numerically from registered grid")
    data["s"] = REGISTERED_S.copy()
    data["n"] = REGISTERED_N.copy()
    return data


base.validate_source = validate_source_registered_values


def prepare_contract() -> None:
    data = base.validate_source()
    del data
    base.require(not base.CONTRACT.exists() and not base.OUTPUT.exists()
                 and not base.AUDIT.exists(), "R1B pilot coordinate chain exists")
    payload = {
        "schema": base.CONTRACT_SCHEMA, "status": base.CONTRACT_STATUS,
        "created_utc": base.utc_now(), "builder": base.binding(THIS_FILE),
        "frozen_base_builder": base.binding(BASE),
        "r1_superseded_zero_physics_evidence": {
            "contract": str((base.REGISTRATION_DIR / (
                "BB_IRIVM_CALL515_TERMINAL_UNCOMMITTED_TAIL_CONSTRAINED_RELAXED_"
                "PILOT_COORDINATE_CONTRACT_R1_20260903.json")).resolve()),
            "coordinate_manifest_without_audit": str((base.OUTPUT_DIR / (
                "bb_irivm_call515_nf14983_terminal_uncommitted_tail_constrained_relaxed_"
                "pilot_coordinates_15x9_20260903_r1.npz")).resolve()),
            "physical_force_calls": 0,
            "reason": "bitwise s-grid representation mismatch caught after coordinate-only write",
        },
        "source_coordinate_manifest": base.binding(base.SOURCE),
        "source_coordinate_audit": base.binding(base.SOURCE_AUDIT),
        "center_state": {
            "method": "IVM", "call_per_image": 515,
            "optimization_N_F": 14983, "path_sha256": base.SOURCE_PATH_SHA256,
            "state_role": "terminal_uncommitted_charged_tail",
            "formally_reportable": False,
        },
        "source_s_indices": base.S_INDICES.tolist(),
        "source_n_indices": base.N_INDICES.tolist(),
        "registered_s_values": REGISTERED_S.tolist(),
        "registered_n_values_rms_A": REGISTERED_N.tolist(),
        "n_execution_order_within_pilot": base.N_EXECUTION_ORDER.tolist(),
        "surface_shape": [15, 9],
        "planned_coordinate_manifest": str(base.OUTPUT.resolve()),
        "planned_coordinate_audit": str(base.AUDIT.resolve()),
        "openmm_contexts_constructed": 0, "physical_force_calls": 0,
        "optimization_N_F_contribution": 0,
    }
    base.atomic_json(base.CONTRACT, payload)
    print(json.dumps({"status": base.CONTRACT_STATUS,
                      "contract": base.binding(base.CONTRACT),
                      "physical_force_calls": 0}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-contract", action="store_true")
    modes.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.prepare_contract:
        prepare_contract()
    else:
        base.build()


if __name__ == "__main__":
    main()
