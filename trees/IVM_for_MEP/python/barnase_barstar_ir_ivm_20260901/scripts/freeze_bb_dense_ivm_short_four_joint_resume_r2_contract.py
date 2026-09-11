#!/usr/bin/env python3
"""Freeze the local R2 continuation of the short BB IVM arc500 diagnostic."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FREEZER = Path(__file__).resolve()
EVALUATOR_PATH = ROOT / "scripts/evaluate_bb_dense_ivm_short_four_joint_resume_r2.py"


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("bb_ivm_short_resume_r2_freezer_eval", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load_evaluator()


def bind(path: Path) -> dict:
    path = Path(path).expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": EVALUATOR.ENGINE.sha256_file(path)}


def atomic_json_no_overwrite(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, default=EVALUATOR.CANONICAL_RESULT)
    parser.add_argument("--global-lock-path", type=Path, required=True)
    parser.add_argument("--peer-global-lock-path", type=Path, required=True)
    args = parser.parse_args()

    destination = args.output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    result_output = args.result_output.expanduser().resolve()
    if result_output != EVALUATOR.CANONICAL_RESULT:
        raise ValueError("R2 result output must be the canonical local path")
    locks = [
        args.global_lock_path.expanduser().resolve(),
        args.peer_global_lock_path.expanduser().resolve(),
    ]
    if len(set(locks)) != 2 or set(locks) != EVALUATOR.CANONICAL_LOCKS:
        raise ValueError("the two canonical host-local BB locks are required")

    evidence = EVALUATOR.load_r1_evidence()
    rows = []
    for source in evidence["rows"]:
        call = int(source["path_identifier"]["calls_per_image"])
        row = {
            "label": source["label"],
            "method": "IVM",
            "host_role": "local",
            "optimization_N_F": int(source["optimization_N_F"]),
            "path_identifier": source["path_identifier"],
            "short_replay_source": source["short_replay_source"],
            "path_sha256": source["path_sha256"],
            "online_discrete_I_perp": float(source["i_perp"]),
            "original_discrete_policy": "verified_short_replay_path_and_sealed_online_Iperp_zero_new_calls",
        }
        if call in EVALUATOR.IMPORTED_CALLS:
            row.update(
                {
                    "R2_action": "import_complete_R1_shard",
                    "dense_policy": "hash_revalidate_complete_R1_arc500_joint_EF_shard_zero_new_calls",
                    "authoritative_shard": evidence["imported"][call]["binding"],
                }
            )
        else:
            row.update(
                {
                    "R2_action": "fresh_R2_complete_path",
                    "dense_policy": "fresh_exactly_500_uniform_cartesian_arc_length_joint_EF_then_atomic_shard",
                }
            )
        rows.append(row)

    contract = {
        "schema": EVALUATOR.SCHEMA,
        "frozen": True,
        "status": "FROZEN_BEFORE_ANY_R2_OPENMM_OR_PHYSICAL_CALL",
        "method": "IVM",
        "host_role": "local",
        "jobs": 1,
        "purpose": "continue the interrupted short BB IVM diagnostic by importing two complete R1 paths and freshly evaluating only calls 10 and 50",
        "result_output": str(result_output),
        "contract_freezer": bind(FREEZER),
        "evaluator": bind(EVALUATOR_PATH),
        "joint_engine": bind(EVALUATOR.ENGINE_PATH),
        "bb_core": bind(ROOT / "src/bb_core.py"),
        "topology": bind(evidence["topology"]),
        "r1_evaluator": bind(EVALUATOR.R1_EVALUATOR_PATH),
        "r1_contract": bind(evidence["r1_contract"]),
        "r1_pause_receipt": bind(evidence["pause_receipt"]),
        "r1_dense_wal_at_pause": bind(evidence["r1_dense_wal"]),
        "r1_source_wal_at_pause": bind(evidence["r1_source_wal"]),
        "host_local_lock_paths": [str(path) for path in locks],
        "metric_policy": EVALUATOR.METRIC_POLICY,
        "paths": rows,
        "R1_quarantine": {
            **evidence["partial"],
            "source_registration": bind(evidence["r1_contract"]),
            "pause_receipt": bind(evidence["pause_receipt"]),
            "complete_shard_published": False,
            "may_be_used_as_a_metric_result": False,
            "may_be_reused_by_R2": False,
        },
        "physical_call_policy": {
            "imported_R1_complete_paths": 2,
            "imported_R1_dense_joint_EF_values": 1000,
            "imported_R1_new_physical_calls": 0,
            "quarantined_R1_call10_successes": 35,
            "quarantined_R1_call10_open_attempts": 1,
            "quarantined_R1_values_accepted": 0,
            "fresh_R2_paths": 2,
            "fresh_R2_dense_joint_EF_per_path": 500,
            "fresh_R2_endpoint_joint_EF_per_path": 2,
            "fresh_R2_accepted_successes": 1000,
            "fresh_R2_maximum_attempt_started_records": 1000,
            "failed_or_open_R2_attempt_policy": "fail_closed_require_new_superseding_registration",
            "dense_full_force_storage": True,
            "original_discrete_new_calls_per_path": 0,
            "optimization_N_F_contribution": 0,
            "jobs": 1,
            "all_six_thread_caps": 1,
            "two_host_local_fcntl_locks": True,
        },
        "storage_policy": {
            "authoritative_unit": "complete_atomic_path_shard",
            "import_R1_shards_by_reference_no_copy": True,
            "R2_fresh_shard_contains_dense_coordinates_energy_full_force": True,
            "aggregate_role": "index_metrics_and_plot_compatible_dense_energy_not_dense_coordinates_or_forces",
            "restart_reuse_unit": "complete_fully_validated_path_shard_only",
            "reuse_individual_incomplete_WAL_successes": False,
            "minimum_free_bytes_reserve": EVALUATOR.MINIMUM_FREE_BYTES_RESERVE,
            "estimated_uncompressed_shard_bytes_per_path": EVALUATOR.ESTIMATED_SHARD_BYTES,
        },
        "durability_policy": {
            "append_only_fsync_R2_source_and_dense_WAL": True,
            "unique_R1_quarantine_marker_in_each_R2_WAL": True,
            "atomic_no_overwrite_R2_complete_path_shard": True,
            "manifest_hash_binds_all_imported_and_fresh_shards": True,
            "aggregate_publication_footer_in_each_R2_WAL": True,
            "failed_or_open_R2_attempt_requires_new_registration": True,
            "same_R2_registration_retry_forbidden": True,
        },
        "optimization_accounting": {
            "optimization_N_F_source": "sealed_short_replay_counter_vectors",
            "reported_optimization_N_F": list(EVALUATOR.EXPECTED_NF),
            "all_R1_and_R2_arc500_joint_EF_calls_are_diagnostics": True,
            "optimization_N_F_contribution": 0,
        },
    }
    atomic_json_no_overwrite(destination, contract)
    print(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_ANY_R2_OPENMM_OR_PHYSICAL_CALL",
                "contract": str(destination),
                "sha256": EVALUATOR.ENGINE.sha256_file(destination),
                "imported_R1_calls": list(EVALUATOR.IMPORTED_CALLS),
                "fresh_R2_calls": list(EVALUATOR.FRESH_CALLS),
                "fresh_R2_joint_EF_attempt_maximum": EVALUATOR.MAXIMUM_R2_ATTEMPT_STARTED,
                "optimization_N_F_contribution": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
