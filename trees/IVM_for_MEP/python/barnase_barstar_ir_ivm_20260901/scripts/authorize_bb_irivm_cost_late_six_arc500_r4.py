#!/usr/bin/env python3
"""Issue one host-specific authorization from its immutable R4 preflight."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "run_bb_irivm_cost_late_six_arc500_r4.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = _load("bb_cost_late_six_runner_r4_for_authorizer", RUNNER_PATH)
FREEZER = RUNNER.FREEZER


def validate_preflight(state: dict, path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    expected_path = Path(state["contract"]["preflight_receipt_output"]).resolve()
    FREEZER.require(resolved == expected_path, "noncanonical host preflight receipt")
    payload = FREEZER.read_json(resolved)
    expected = {
        "schema": FREEZER.PREFLIGHT_SCHEMA,
        "status": FREEZER.PREFLIGHT_STATUS,
        "host_role": state["host_role"],
        "selected_path_count": len(state["rows"]),
        "selection_roles": [row["selection_role"] for row in state["rows"]],
        "fresh_joint_EF_attempts_planned": len(state["rows"])
        * FREEZER.FRESH_PER_PATH,
        "stored_arc500_joint_EF_values_planned": len(state["rows"])
        * FREEZER.N_DENSE,
        "inherited_endpoint_joint_EF_values_planned": len(state["rows"]) * 2,
        "original32_joint_EF_reused_per_path": FREEZER.N_IMAGES,
        "original32_new_physical_calls": 0,
        "optimization_N_F_contribution": 0,
        "OpenMM_imported_or_called": False,
        "result_output": str(state["result_output"]),
    }
    for key, value in expected.items():
        FREEZER.require(
            payload.get(key) == value, f"preflight field drift: {key}"
        )
    bindings = payload.get("bindings", {})
    required_bindings = {
        "preparation": FREEZER.bind(FREEZER.PREPARATION),
        "bundle": FREEZER.bind(FREEZER.BUNDLE),
        "publication_Mutagen_gate": FREEZER.bind(FREEZER.PUBLICATION_GATE),
        "contract": FREEZER.bind(state["contract_path"]),
        "authorized_runner": FREEZER.bind(RUNNER_PATH),
    }
    FREEZER.require(bindings == required_bindings, "preflight binding drift")
    physical = payload.get("physical_preflight", {})
    lifetime = physical.get("lifetime", {})
    FREEZER.require(
        physical.get("lock_probe") == "PASS_NONBLOCKING"
        and physical.get("OpenMM_backend_constructed") is False
        and physical.get("fresh_joint_EF_attempts_planned")
        == expected["fresh_joint_EF_attempts_planned"]
        and lifetime.get("attempt_started") == 0
        and lifetime.get("attempt_failed") == 0
        and lifetime.get("attempt_open") == 0
        and lifetime.get("retry_attempts") == 0,
        "preflight lock/OpenMM/WAL lifetime gate failed",
    )
    return payload


def build_authorization(state: dict, preflight_path: Path) -> dict:
    validate_preflight(state, preflight_path)
    return {
        "schema": FREEZER.AUTH_SCHEMA,
        "status": FREEZER.AUTH_STATUS,
        "enabled": True,
        "created_utc": FREEZER.utc_now(),
        "host_role": state["host_role"],
        "preparation": FREEZER.bind(FREEZER.PREPARATION),
        "bundle": FREEZER.bind(FREEZER.BUNDLE),
        "publication_Mutagen_gate": FREEZER.bind(FREEZER.PUBLICATION_GATE),
        "contract": FREEZER.bind(state["contract_path"]),
        "authorized_runner": FREEZER.bind(RUNNER_PATH),
        "authorizer": FREEZER.bind(Path(__file__).resolve()),
        "zero_physical_preflight": FREEZER.bind(preflight_path),
        "result_output": str(state["result_output"]),
        "maximum_fresh_joint_EF_attempts": len(state["rows"])
        * FREEZER.FRESH_PER_PATH,
        "optimization_N_F_contribution": 0,
        "OpenMM_calls_before_authorization": 0,
        "jobs": 1,
        "thread_caps": dict(FREEZER.SELECTOR.THREAD_CAPS),
        "host_local_lock_paths": list(FREEZER.SELECTOR.LOCKS),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--publication-gate", type=Path, required=True)
    parser.add_argument("--host-role", choices=("remote", "local"), required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    FREEZER.require(
        args.preparation.resolve(strict=True) == FREEZER.PREPARATION,
        "noncanonical preparation",
    )
    preparation = FREEZER.validate_preparation(args.preparation)
    contract = Path(
        preparation["host_plans"][args.host_role]["contract_output"]
    )
    state = FREEZER.load_host_state(
        contract,
        args.bundle,
        args.publication_gate,
        args.host_role,
    )
    expected_output = Path(
        preparation["host_plans"][args.host_role]["authorization_output"]
    ).resolve()
    FREEZER.require(
        args.output.resolve() == expected_output,
        "noncanonical authorization output",
    )
    FREEZER.write_exclusive(
        args.output, build_authorization(state, args.preflight)
    )
    print(args.output.resolve(strict=True))
    print(FREEZER.SELECTOR.sha256_file(args.output.resolve(strict=True)))


if __name__ == "__main__":
    main()
