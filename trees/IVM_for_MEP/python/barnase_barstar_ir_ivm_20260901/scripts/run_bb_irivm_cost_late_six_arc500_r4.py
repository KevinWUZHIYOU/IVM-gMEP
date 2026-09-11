#!/usr/bin/env python3
"""Authorized remote4/local2 runner for R5B-selected IR--IVM arc500 data.

Preflight is fail-closed and performs no OpenMM construction.  A physical run
requires the canonical host contract, bundle, publication Mutagen gate, and a
separate authorization bound to the canonical host preflight receipt.

The dense scientific implementation delegates to the previously audited R1
late-six evaluator.  Only its cardinality, output schemas, disk reserve, and
host-identity callback are scoped to the frozen host contract; the arc-length
construction and all 498 interior joint E/F calls per path are unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FREEZER_PATH = HERE / "freeze_bb_irivm_cost_late_six_arc500_r4.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FREEZER = _load("bb_cost_late_six_freezer_r4_for_runner", FREEZER_PATH)
DELEGATE = FREEZER.DELEGATE

SHARD_SCHEMA = "bb-irivm-cost-late-host-arc500-shard-v4"
RESULT_SCHEMA = "bb-irivm-cost-late-host-arc500-result-v4"
MANIFEST_SCHEMA = "bb-irivm-cost-late-host-arc500-manifest-v4"
AUDIT_SCHEMA = "bb-irivm-cost-late-host-arc500-audit-v4"
SOURCE_WAL_SCHEMA = "bb-irivm-cost-late-host-source-wal-v4"
DENSE_WAL_SCHEMA = "bb-irivm-cost-late-host-dense-wal-v4"


def require(condition: bool, message: str) -> None:
    FREEZER.require(condition, message)


def bind(path: Path) -> dict:
    return FREEZER.bind(path)


@contextmanager
def configured_delegate(state: dict):
    count = len(state["rows"])
    replacements = {
        "N_PATHS": count,
        "TOTAL_STORED_JOINT_EF": count * FREEZER.N_DENSE,
        "MAXIMUM_FRESH_ATTEMPT_STARTED": count * FREEZER.FRESH_PER_PATH,
        "MINIMUM_FREE_BYTES_RESERVE": FREEZER.MINIMUM_FREE_BYTES_RESERVE,
        "ESTIMATED_SHARD_BYTES": FREEZER.ESTIMATED_SHARD_BYTES,
        "SHARD_SCHEMA": SHARD_SCHEMA,
        "RESULT_SCHEMA": RESULT_SCHEMA,
        "MANIFEST_SCHEMA": MANIFEST_SCHEMA,
        "AUDIT_SCHEMA": AUDIT_SCHEMA,
        "SOURCE_WAL_SCHEMA": SOURCE_WAL_SCHEMA,
        "DENSE_WAL_SCHEMA": DENSE_WAL_SCHEMA,
        "check_remote_host": FREEZER.check_host,
    }
    original = {name: getattr(DELEGATE, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(DELEGATE, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(DELEGATE, name, value)


def _authorization_path(state: dict) -> Path:
    return Path(state["contract"]["execution_authorization_output"]).resolve()


def _preflight_path(state: dict) -> Path:
    return Path(state["contract"]["preflight_receipt_output"]).resolve()


def validate_authorization(state: dict, path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    require(resolved == _authorization_path(state), "noncanonical host authorization")
    payload = FREEZER.read_json(resolved)
    host_role = state["host_role"]
    expected = {
        "schema": FREEZER.AUTH_SCHEMA,
        "status": FREEZER.AUTH_STATUS,
        "enabled": True,
        "host_role": host_role,
        "preparation": bind(FREEZER.PREPARATION),
        "bundle": bind(FREEZER.BUNDLE),
        "publication_Mutagen_gate": bind(FREEZER.PUBLICATION_GATE),
        "contract": bind(state["contract_path"]),
        "authorized_runner": bind(Path(__file__).resolve()),
        "authorizer": bind(FREEZER.AUTHORIZER_PATH),
        "zero_physical_preflight": bind(_preflight_path(state)),
        "result_output": str(state["result_output"]),
        "maximum_fresh_joint_EF_attempts": len(state["rows"])
        * FREEZER.FRESH_PER_PATH,
        "optimization_N_F_contribution": 0,
        "OpenMM_calls_before_authorization": 0,
        "jobs": 1,
        "thread_caps": dict(FREEZER.SELECTOR.THREAD_CAPS),
        "host_local_lock_paths": list(FREEZER.SELECTOR.LOCKS),
    }
    for key, value in expected.items():
        require(payload.get(key) == value, f"authorization field drift: {key}")
    return payload


def build_preflight_receipt(state: dict, output: Path) -> dict:
    output = Path(output).expanduser().resolve()
    require(output == state["result_output"], "preflight output differs from contract")
    require(
        not _authorization_path(state).exists(),
        "preauthorization preflight requires absent host authorization",
    )
    with configured_delegate(state):
        physical = DELEGATE.preflight(
            state,
            output,
            probe_locks=True,
            probe_remote_host=True,
        )
    require(
        physical.get("OpenMM_backend_constructed") is False
        and physical.get("lock_probe") == "PASS_NONBLOCKING"
        and physical.get("fresh_joint_EF_attempts_planned")
        == len(state["rows"]) * FREEZER.FRESH_PER_PATH
        and physical.get("original_discrete_new_calls") == 0
        and physical.get("optimization_N_F_contribution") == 0,
        "delegate zero-physical preflight accounting drift",
    )
    return {
        "schema": FREEZER.PREFLIGHT_SCHEMA,
        "status": FREEZER.PREFLIGHT_STATUS,
        "created_utc": FREEZER.utc_now(),
        "host_role": state["host_role"],
        "host_identity": physical["remote_host_probe"],
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
        "physical_preflight": physical,
        "bindings": {
            "preparation": bind(FREEZER.PREPARATION),
            "bundle": bind(FREEZER.BUNDLE),
            "publication_Mutagen_gate": bind(FREEZER.PUBLICATION_GATE),
            "contract": bind(state["contract_path"]),
            "authorized_runner": bind(Path(__file__).resolve()),
        },
        "result_output": str(state["result_output"]),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--publication-gate", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-gate", type=Path)
    parser.add_argument("--host-role", choices=("remote", "local"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    DELEGATE.check_thread_caps()
    state = FREEZER.load_host_state(
        args.contract,
        args.bundle,
        args.publication_gate,
        args.host_role,
    )
    output = args.output.expanduser().resolve()
    require(output == state["result_output"], "noncanonical host result output")
    require(output.suffix == ".npz", "result output must end in .npz")
    if args.preflight_only:
        require(
            args.authorization is None
            and args.authorization_gate is None
            and args.preflight_receipt is not None
            and args.preflight_receipt.resolve() == _preflight_path(state),
            "preflight requires canonical receipt and no authorization",
        )
        payload = build_preflight_receipt(state, output)
        FREEZER.write_exclusive(args.preflight_receipt, payload)
        print(args.preflight_receipt.resolve(strict=True))
        print(FREEZER.SELECTOR.sha256_file(args.preflight_receipt.resolve(strict=True)))
        return
    require(
        args.preflight_receipt is None
        and args.authorization is not None
        and args.authorization_gate is not None,
        "physical mode requires authorization+gate and forbids preflight receipt",
    )
    validate_authorization(state, args.authorization)
    preparation = FREEZER.validate_preparation(FREEZER.PREPARATION)
    FREEZER.validate_gate(
        args.authorization_gate,
        "authorization",
        FREEZER.PREPARATION,
        bundle=FREEZER.BUNDLE,
        remote_authorization=Path(
            preparation["host_plans"]["remote"]["authorization_output"]
        ),
        local_authorization=Path(
            preparation["host_plans"]["local"]["authorization_output"]
        ),
    )
    with configured_delegate(state):
        DELEGATE.run(state, output)


if __name__ == "__main__":
    main()
