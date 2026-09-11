#!/usr/bin/env python3
"""Resume the short BB IVM arc500 joint-E/F diagnostic without redoing R1.

Calls 2 and 5 are immutable imports from two fully closed R1 shards.  The
interrupted R1 call-10 tail is evidence only: its 35 successes and one open
attempt are explicitly quarantined and never become metric data.  This R2
registration may make exactly 500 fresh joint energy/full-force attempts for
each of calls 10 and 50.  A failed/open R2 attempt fails closed and requires a
new superseding registration; the only restart unit is a complete path shard.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Iterable, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENGINE_PATH = HERE / "evaluate_bb_dense_four_paths_joint_v2.py"
R1_EVALUATOR_PATH = HERE / "evaluate_bb_dense_ivm_short_four_joint.py"
FREEZER_PATH = HERE / "freeze_bb_dense_ivm_short_four_joint_resume_r2_contract.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ENGINE = _load_module("bb_joint_engine_ivm_resume_r2", ENGINE_PATH)
R1 = _load_module("bb_ivm_short_joint_r1_for_resume_r2", R1_EVALUATOR_PATH)

SCHEMA = "bb-dense-ivm-short-four-joint-ef-resume-contract-v2"
SHARD_SCHEMA = "bb-dense-ivm-short-path-joint-ef-resume-shard-v2"
RESULT_SCHEMA = "bb-dense-ivm-short-four-joint-ef-resume-result-v2"
MANIFEST_SCHEMA = "bb-dense-ivm-short-four-joint-ef-resume-manifest-v2"
AUDIT_SCHEMA = "bb-dense-ivm-short-four-joint-ef-resume-audit-v2"
SOURCE_WAL_SCHEMA = "bb-dense-ivm-short-four-source-resume-wal-v2"
DENSE_WAL_SCHEMA = "bb-dense-ivm-short-four-joint-ef-resume-wal-v2"

CALLS = (2, 5, 10, 50)
EXPECTED_NF = (60, 150, 300, 1329)
IMPORTED_CALLS = (2, 5)
FRESH_CALLS = (10, 50)
N_IMAGES = 32
DIMENSION = 9396
N_DENSE = 500
FRESH_ACCEPTED_SUCCESSES = len(FRESH_CALLS) * N_DENSE
MAXIMUM_R2_ATTEMPT_STARTED = FRESH_ACCEPTED_SUCCESSES
MINIMUM_FREE_BYTES_RESERVE = 8 * 1024**3
ESTIMATED_SHARD_BYTES = (
    (2 * N_DENSE * DIMENSION + N_IMAGES * DIMENSION) * 8 + 8 * 1024**2
)

CANONICAL_LOCKS = {
    Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock").resolve(),
    Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock").resolve(),
}
CANONICAL_RESULT = (
    ROOT
    / "runs/diagnostics/bb_irivm_short_four_arc500_joint_ef_20260902_r2.npz"
).resolve()
R1_CONTRACT = (
    ROOT
    / "registrations/diagnostics/BB_DENSE_COST_IVM_FOUR_JOINT_EF_SHORT_PREFIX_20260902_R1.json"
)
R1_CONTRACT_SHA256 = "c0350c283e53f81a3446918b92319f1bbeb7cbd80cf5e7493e5a635f8ac5810a"
R1_EVALUATOR_SHA256 = "020e70db46e52529cebfc01481541b8b76a767d0e5372dd14c83dcf6e99f16c6"
PAUSE_RECEIPT = (
    ROOT
    / "registrations/diagnostics/BB_DENSE_COST_IVM_SHORT_R1_PLOT_PRIORITY_PAUSE_20260902.json"
)
PAUSE_RECEIPT_SHA256 = "c35831d5e9f560ff82afb0c0903c23f3344f0604fc248643a3d6aed2e3480a8a"
R1_OUTPUT = (
    ROOT
    / "runs/diagnostics/bb_irivm_short_four_arc500_joint_ef_20260902_r1.npz"
)
R1_DENSE_WAL = R1_OUTPUT.with_suffix(".dense_joint_ef.wal.jsonl")
R1_SOURCE_WAL = R1_OUTPUT.with_suffix(".source.wal.jsonl")
R1_DENSE_WAL_SHA256 = "7384ee9babf516e6470a4db308a14964dcda68c2a978628ca422ed68b51411ba"
R1_SOURCE_WAL_SHA256 = "bef4608247b4543bfbc1ba7d685b84dbaafe4f516d41c43ef7941a336933a160"
R1_IMPORTED_SHARD_SHA256 = {
    2: "3f844cfbaebe2e1a4b03359e533de9cc3c806275ba1a4f72b13f5dadf124a1af",
    5: "0122d171f8e4336642445b73d53a4d89d632f28b5ed0fe521716b5e808b7c465",
}

METRIC_POLICY = {
    **R1.METRIC_POLICY,
    "R1_complete_shard_reuse": "calls_2_and_5_exact_hash_validated_zero_new_calls",
    "R1_partial_tail_policy": "call_10_35_successes_plus_1_open_attempt_quarantined_not_reused",
}


def require(condition: bool, message: str) -> None:
    ENGINE.require(condition, message)


def scalar(saved, key: str):
    return np.asarray(saved[key]).item()


def same_float(left, right) -> bool:
    return (
        np.asarray(float(left), dtype=np.float64).tobytes()
        == np.asarray(float(right), dtype=np.float64).tobytes()
    )


def binding(path: Path) -> dict:
    path = Path(path).expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": ENGINE.sha256_file(path)}


def _require_exact_file(path: Path, expected_sha256: str, role: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    require(ENGINE.sha256_file(resolved) == expected_sha256, f"{role} SHA-256 mismatch")
    return resolved


def _r1_shard_path(call: int) -> Path:
    return R1.shard_path(R1_OUTPUT, call)


def _r2_shard_path(output: Path, call: int) -> Path:
    return (
        output.with_suffix("")
        .with_name(output.stem + ".shards")
        / f"ivm_call{call:04d}_arc500_joint_ef_r2.npz"
    )


def _wal_paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_suffix(".source.wal.jsonl"),
        output.with_suffix(".dense_joint_ef.wal.jsonl"),
    )


def _final_paths(output: Path) -> tuple[Path, Path, Path]:
    return output, output.with_suffix(".manifest.json"), output.with_suffix(".audit.json")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _r1_partial_tail_evidence(events: list[dict], row: dict) -> dict:
    starts = [
        event
        for event in events
        if event.get("event") == "attempt_started"
        and int(event.get("row_index", -1)) == 2
        and event.get("stage") == "dense_arc500_joint_EF"
    ]
    successes = [
        event
        for event in events
        if event.get("event") == "attempt_succeeded"
        and int(event.get("row_index", -1)) == 2
        and event.get("stage") == "dense_arc500_joint_EF"
    ]
    failures = [
        event
        for event in events
        if event.get("event") == "attempt_failed"
        and int(event.get("row_index", -1)) == 2
        and event.get("stage") == "dense_arc500_joint_EF"
    ]
    started = {str(event["attempt_id"]): event for event in starts}
    terminal = {str(event["attempt_id"]) for event in successes + failures}
    open_ids = sorted(set(started) - terminal)
    require(len(starts) == 36 and len(successes) == 35 and not failures, "R1 partial-tail attempt counts drifted")
    require(len(open_ids) == 1, "R1 partial tail must contain exactly one open attempt")
    require(sorted(int(event["point_index"]) for event in successes) == list(range(35)), "R1 partial-tail success coverage drifted")
    open_event = started[open_ids[0]]
    require(int(open_event["point_index"]) == 35, "R1 open attempt point drifted")
    require(open_event.get("label") == row["label"], "R1 partial-tail label drifted")
    require(open_event.get("path_sha256") == row["path_sha256"], "R1 partial-tail path drifted")
    require(
        not any(
            event.get("event") in {"path_stage_data_complete_uncommitted", "complete_shard_published"}
            and int(event.get("row_index", -1)) == 2
            for event in events
        ),
        "R1 partial tail was unexpectedly published",
    )
    require(
        not any(
            event.get("event") in {"attempt_started", "attempt_succeeded", "attempt_failed"}
            and int(event.get("row_index", -1)) == 3
            for event in events
        ),
        "R1 call-50 unexpectedly has physical attempts",
    )
    return {
        "calls_per_image": 10,
        "optimization_N_F": 300,
        "successful_joint_energy_force_attempts": 35,
        "open_interrupted_attempts": 1,
        "open_attempt_id": open_ids[0],
        "open_attempt_coordinate_sha256": str(open_event["coordinate_sha256"]),
        "accepted_metric_values": 0,
    }


def load_r1_evidence() -> dict:
    """Fully revalidate immutable R1 imports and the quarantined tail."""

    r1_contract_path = _require_exact_file(R1_CONTRACT, R1_CONTRACT_SHA256, "R1 contract")
    _require_exact_file(R1_EVALUATOR_PATH, R1_EVALUATOR_SHA256, "R1 evaluator")
    pause_path = _require_exact_file(PAUSE_RECEIPT, PAUSE_RECEIPT_SHA256, "R1 pause receipt")
    dense_wal_path = _require_exact_file(R1_DENSE_WAL, R1_DENSE_WAL_SHA256, "R1 dense WAL")
    source_wal_path = _require_exact_file(R1_SOURCE_WAL, R1_SOURCE_WAL_SHA256, "R1 source WAL")

    r1_contract, topology, locks, rows = R1.load_contract(r1_contract_path, "local")
    require(tuple(int(row["path_identifier"]["calls_per_image"]) for row in rows) == CALLS, "R1 calls drifted")
    require(tuple(int(row["optimization_N_F"]) for row in rows) == EXPECTED_NF, "R1 N_F drifted")
    require(set(locks) == CANONICAL_LOCKS, "R1 canonical locks drifted")

    contract_sha = ENGINE.sha256_file(r1_contract_path)
    topology_sha = ENGINE.sha256_file(topology)
    source_events = R1._read_validated_wal(
        source_wal_path, R1.SOURCE_WAL_SCHEMA, r1_contract_path, contract_sha
    )
    dense_events = R1._read_validated_wal(
        dense_wal_path, R1.DENSE_WAL_SCHEMA, r1_contract_path, contract_sha
    )
    lifetime = R1._strict_wal_lifetime(dense_events, "R1 dense")
    require(
        lifetime
        == {
            "attempt_started": 1036,
            "attempt_succeeded": 1035,
            "attempt_failed": 0,
            "attempt_open": 1,
            "retry_attempts": 0,
        },
        "R1 dense WAL lifetime drifted",
    )

    imported = {}
    for row_index, call in enumerate(IMPORTED_CALLS):
        target = _require_exact_file(
            _r1_shard_path(call), R1_IMPORTED_SHARD_SHA256[call], f"R1 call-{call} shard"
        )
        shard = R1.validate_shard(target, rows[row_index], contract_sha, topology_sha)
        R1._validate_source_wal_closure(source_events, row_index, rows[row_index], target, shard)
        closure = R1._validate_dense_wal_closure(
            dense_events, row_index, rows[row_index], target, shard
        )
        require(int(closure["accepted_stage_successes"]) == N_DENSE, f"R1 call-{call} closure count mismatch")
        imported[call] = {
            "path": target,
            "binding": binding(target),
            "shard": shard,
        }

    require(not _r1_shard_path(10).exists(), "R1 call-10 partial tail must not have a shard")
    require(not _r1_shard_path(50).exists(), "R1 call-50 must not have a shard")
    partial = _r1_partial_tail_evidence(dense_events, rows[2])

    pause = _read_json(pause_path)
    require(
        pause.get("schema") == "bb-dense-cost-ivm-short-r1-plot-priority-pause-v1"
        and pause.get("status") == "PAUSED_WITH_TWO_COMPLETE_ATOMIC_SHARDS",
        "invalid R1 pause receipt",
    )
    require(pause.get("contract") == binding(r1_contract_path), "pause receipt R1 contract drifted")
    require(pause.get("dense_wal_at_pause") == binding(dense_wal_path), "pause receipt dense WAL drifted")
    require(pause.get("source_wal_at_pause") == binding(source_wal_path), "pause receipt source WAL drifted")
    require(pause.get("dense_wal_lifetime_at_pause") == {k: lifetime[k] for k in ("attempt_started", "attempt_succeeded", "attempt_failed", "attempt_open")}, "pause receipt lifetime drifted")
    pause_tail = pause.get("unaccepted_partial_tail", {})
    expected_pause_tail = {
        key: value for key, value in partial.items() if key != "accepted_metric_values"
    }
    expected_pause_tail.update(
        {
            "complete_shard_published": False,
            "may_be_used_as_a_metric_result": False,
            "must_not_be_silently_retried_under_the_same_registration": True,
        }
    )
    require(pause_tail == expected_pause_tail, "pause receipt partial-tail evidence drifted")
    accepted = pause.get("accepted_complete_shards")
    require(isinstance(accepted, list) and len(accepted) == 2, "pause receipt accepted shard list drifted")
    for item, call, nf in zip(accepted, IMPORTED_CALLS, EXPECTED_NF[:2]):
        require(
            item
            == {
                "calls_per_image": call,
                "optimization_N_F": nf,
                **binding(_r1_shard_path(call)),
            },
            f"pause receipt call-{call} binding drifted",
        )
    require(
        pause.get("accounting")
        == {
            "optimization_N_F_contribution": 0,
            "accepted_dense_joint_energy_force_values": 1000,
            "partial_tail_excluded_from_accepted_results": True,
        },
        "pause receipt accounting drifted",
    )
    require(
        pause.get("process_termination_observation", {}).get("both_canonical_host_local_BB_locks_released") is True,
        "pause receipt does not prove R1 lock release",
    )
    return {
        "r1_contract": r1_contract_path,
        "pause_receipt": pause_path,
        "topology": topology,
        "locks": locks,
        "rows": rows,
        "imported": imported,
        "partial": partial,
        "r1_dense_wal": dense_wal_path,
        "r1_source_wal": source_wal_path,
        "r1_dense_lifetime": lifetime,
        "r1_contract_payload": r1_contract,
    }


def load_contract(contract_path: Path, host_role: str) -> dict:
    contract = _read_json(contract_path)
    require(contract.get("schema") == SCHEMA and contract.get("frozen") is True, "invalid R2 contract")
    require(
        contract.get("status") == "FROZEN_BEFORE_ANY_R2_OPENMM_OR_PHYSICAL_CALL",
        "R2 contract status mismatch",
    )
    require(host_role == "local" and contract.get("host_role") == "local", "R2 is local-only")
    require(contract.get("method") == "IVM" and int(contract.get("jobs", -1)) == 1, "R2 method/jobs mismatch")
    require(contract.get("metric_policy") == METRIC_POLICY, "R2 metric policy mismatch")
    require(
        ENGINE.require_file(contract["evaluator"], contract_path, "R2 evaluator") == Path(__file__).resolve(),
        "R2 evaluator binding mismatch",
    )
    require(
        ENGINE.require_file(contract["contract_freezer"], contract_path, "R2 freezer") == FREEZER_PATH.resolve(),
        "R2 freezer binding mismatch",
    )
    require(
        ENGINE.require_file(contract["joint_engine"], contract_path, "joint engine") == ENGINE_PATH.resolve(),
        "joint engine binding mismatch",
    )
    require(
        ENGINE.require_file(contract["bb_core"], contract_path, "bb_core") == (ROOT / "src/bb_core.py").resolve(),
        "bb_core binding mismatch",
    )
    require(
        ENGINE.require_file(contract["r1_evaluator"], contract_path, "R1 evaluator") == R1_EVALUATOR_PATH.resolve(),
        "R1 evaluator binding mismatch",
    )

    evidence = load_r1_evidence()
    require(
        ENGINE.require_file(contract["topology"], contract_path, "topology")
        == evidence["topology"],
        "R2 topology binding mismatch",
    )
    require(contract.get("r1_contract") == binding(evidence["r1_contract"]), "R2/R1 contract binding mismatch")
    require(contract.get("r1_pause_receipt") == binding(evidence["pause_receipt"]), "R2 pause binding mismatch")
    require(contract.get("r1_dense_wal_at_pause") == binding(evidence["r1_dense_wal"]), "R2 dense WAL binding mismatch")
    require(contract.get("r1_source_wal_at_pause") == binding(evidence["r1_source_wal"]), "R2 source WAL binding mismatch")

    result_output = Path(str(contract.get("result_output", ""))).expanduser().resolve()
    require(result_output == CANONICAL_RESULT, "R2 result path mismatch")
    locks = [Path(value).expanduser().resolve() for value in contract.get("host_local_lock_paths", [])]
    require(len(locks) == len(set(locks)) == 2 and set(locks) == CANONICAL_LOCKS, "R2 canonical two-lock set required")

    rows_payload = contract.get("paths")
    require(isinstance(rows_payload, list) and len(rows_payload) == 4, "R2 must bind four paths")
    rows = []
    for index, (frozen, source) in enumerate(zip(rows_payload, evidence["rows"])):
        call = int(source["path_identifier"]["calls_per_image"])
        require(int(frozen.get("optimization_N_F", -1)) == EXPECTED_NF[index], f"call-{call} N_F drifted")
        require(frozen.get("path_identifier") == source["path_identifier"], f"call-{call} identifier drifted")
        require(frozen.get("path_sha256") == source["path_sha256"], f"call-{call} path drifted")
        require(same_float(frozen.get("online_discrete_I_perp"), source["i_perp"]), f"call-{call} I_perp drifted")
        require(frozen.get("short_replay_source") == source["short_replay_source"], f"call-{call} replay source drifted")
        require(frozen.get("host_role") == "local" and frozen.get("method") == "IVM", f"call-{call} host/method drifted")
        require(
            frozen.get("original_discrete_policy")
            == "verified_short_replay_path_and_sealed_online_Iperp_zero_new_calls",
            f"call-{call} discrete policy mismatch",
        )
        mode = "import_complete_R1_shard" if call in IMPORTED_CALLS else "fresh_R2_complete_path"
        require(frozen.get("R2_action") == mode, f"call-{call} R2 action mismatch")
        if call in IMPORTED_CALLS:
            require(
                frozen.get("dense_policy")
                == "hash_revalidate_complete_R1_arc500_joint_EF_shard_zero_new_calls",
                f"call-{call} import policy mismatch",
            )
            require(
                frozen.get("authoritative_shard") == evidence["imported"][call]["binding"],
                f"call-{call} imported shard binding mismatch",
            )
            shard_path = evidence["imported"][call]["path"]
            shard = evidence["imported"][call]["shard"]
        else:
            require(
                frozen.get("dense_policy")
                == "fresh_exactly_500_uniform_cartesian_arc_length_joint_EF_then_atomic_shard",
                f"call-{call} fresh policy mismatch",
            )
            require("authoritative_shard" not in frozen, f"call-{call} cannot preclaim an R2 shard")
            shard_path = None
            shard = None
        rows.append(
            {
                **frozen,
                "path": source["path"],
                "i_perp": source["i_perp"],
                "source_path": source["source_path"],
                "shard_path": shard_path,
                "shard": shard,
                "row_index": index,
            }
        )

    require(
        contract.get("R1_quarantine")
        == {
            **evidence["partial"],
            "source_registration": binding(evidence["r1_contract"]),
            "pause_receipt": binding(evidence["pause_receipt"]),
            "complete_shard_published": False,
            "may_be_used_as_a_metric_result": False,
            "may_be_reused_by_R2": False,
        },
        "R2 quarantine declaration mismatch",
    )

    physical = contract.get("physical_call_policy", {})
    require(
        physical
        == {
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
        "R2 physical-call policy mismatch",
    )
    storage = contract.get("storage_policy", {})
    require(storage.get("authoritative_unit") == "complete_atomic_path_shard", "R2 shard authority mismatch")
    require(storage.get("import_R1_shards_by_reference_no_copy") is True, "R1 imports must be by reference")
    require(storage.get("reuse_individual_incomplete_WAL_successes") is False, "partial WAL reuse forbidden")
    require(storage.get("R2_fresh_shard_contains_dense_coordinates_energy_full_force") is True, "full R2 storage required")
    require(
        storage.get("aggregate_role")
        == "index_metrics_and_plot_compatible_dense_energy_not_dense_coordinates_or_forces",
        "R2 aggregate role mismatch",
    )
    require(
        storage.get("restart_reuse_unit")
        == "complete_fully_validated_path_shard_only",
        "R2 restart unit mismatch",
    )
    require(int(storage.get("minimum_free_bytes_reserve", -1)) == MINIMUM_FREE_BYTES_RESERVE, "disk reserve mismatch")
    require(int(storage.get("estimated_uncompressed_shard_bytes_per_path", -1)) == ESTIMATED_SHARD_BYTES, "shard estimate mismatch")
    require(
        contract.get("durability_policy")
        == {
            "append_only_fsync_R2_source_and_dense_WAL": True,
            "unique_R1_quarantine_marker_in_each_R2_WAL": True,
            "atomic_no_overwrite_R2_complete_path_shard": True,
            "manifest_hash_binds_all_imported_and_fresh_shards": True,
            "aggregate_publication_footer_in_each_R2_WAL": True,
            "failed_or_open_R2_attempt_requires_new_registration": True,
            "same_R2_registration_retry_forbidden": True,
        },
        "R2 durability policy mismatch",
    )
    require(
        contract.get("optimization_accounting")
        == {
            "optimization_N_F_source": "sealed_short_replay_counter_vectors",
            "reported_optimization_N_F": list(EXPECTED_NF),
            "all_R1_and_R2_arc500_joint_EF_calls_are_diagnostics": True,
            "optimization_N_F_contribution": 0,
        },
        "R2 optimization accounting mismatch",
    )
    return {
        "contract": contract,
        "contract_path": contract_path,
        "contract_sha256": ENGINE.sha256_file(contract_path),
        "topology": evidence["topology"],
        "locks": locks,
        "rows": rows,
        "result_output": result_output,
        "evidence": evidence,
    }


def _read_r2_wal(path: Path, schema: str, state: dict) -> list[dict]:
    events = ENGINE._read_wal_events(path)
    require(events and events[0].get("event") == "wal_header", "R2 WAL header missing")
    require(events[0].get("schema") == schema, "R2 WAL schema mismatch")
    require(events[0].get("contract_sha256") == state["contract_sha256"], "R2 WAL contract SHA mismatch")
    require(Path(str(events[0].get("contract", ""))).resolve() == state["contract_path"], "R2 WAL contract path mismatch")
    return events


def _r2_lifetime(events: list[dict]) -> dict:
    report = R1._strict_wal_lifetime(events, "R2 dense")
    require(report["attempt_started"] <= MAXIMUM_R2_ATTEMPT_STARTED, "R2 attempt authorization exceeded")
    require(report["retry_attempts"] == 0, "same-registration R2 retry is forbidden")
    return report


def _stable_quarantine_event(state: dict, wal_role: str) -> dict:
    partial = state["evidence"]["partial"]
    return {
        "event": "R1_partial_tail_quarantined",
        "wal_role": wal_role,
        "pause_receipt": str(state["evidence"]["pause_receipt"]),
        "pause_receipt_sha256": PAUSE_RECEIPT_SHA256,
        "R1_dense_wal": str(state["evidence"]["r1_dense_wal"]),
        "R1_dense_wal_sha256": R1_DENSE_WAL_SHA256,
        "calls_per_image": 10,
        "optimization_N_F": 300,
        "successful_attempts_quarantined": partial["successful_joint_energy_force_attempts"],
        "open_attempts_quarantined": partial["open_interrupted_attempts"],
        "open_attempt_id": partial["open_attempt_id"],
        "open_attempt_coordinate_sha256": partial["open_attempt_coordinate_sha256"],
        "accepted_metric_values": 0,
        "reused_physical_values": 0,
        "optimization_N_F_contribution": 0,
    }


def _matching_stable_event(event: dict, expected: dict) -> bool:
    return all(event.get(key) == value for key, value in expected.items())


def _ensure_once(wal, events: list[dict], stable: dict) -> None:
    matches = [event for event in events if _matching_stable_event(event, stable)]
    require(len(matches) <= 1, f"duplicate {stable['event']} event")
    if not matches:
        wal.append({**stable, "utc": ENGINE.utc_now()})


def _stable_import_event(state: dict, row: dict, wal_role: str) -> dict:
    call = int(row["path_identifier"]["calls_per_image"])
    return {
        "event": "complete_R1_shard_imported",
        "wal_role": wal_role,
        "row_index": int(row["row_index"]),
        "calls_per_image": call,
        "optimization_N_F": int(row["optimization_N_F"]),
        "label": row["label"],
        "path_sha256": row["path_sha256"],
        "shard": str(row["shard_path"]),
        "shard_sha256": ENGINE.sha256_file(row["shard_path"]),
        "dense_coordinate_sha256": str(scalar(row["shard"], "dense_coordinate_sha256")),
        "dense_energy_sha256": str(scalar(row["shard"], "dense_energy_sha256")),
        "dense_force_sha256": str(scalar(row["shard"], "dense_force_sha256")),
        "new_physical_calls": 0,
        "optimization_N_F_contribution": 0,
    }


def _stable_source_event(row: dict) -> dict:
    call = int(row["path_identifier"]["calls_per_image"])
    return {
        "event": "verified_short_replay_path_and_Iperp_reused",
        "row_index": int(row["row_index"]),
        "calls_per_image": call,
        "optimization_N_F": int(row["optimization_N_F"]),
        "label": row["label"],
        "path_sha256": row["path_sha256"],
        "discrete_I_perp": float(row["i_perp"]),
        "source": str(row["source_path"]),
        "source_sha256": row["short_replay_source"]["sha256"],
        "new_physical_calls": 0,
        "optimization_N_F_contribution": 0,
    }


def _fresh_publication_event(row: dict, target: Path, shard: dict, wal_role: str) -> dict:
    return {
        "event": "complete_R2_shard_published",
        "wal_role": wal_role,
        "row_index": int(row["row_index"]),
        "calls_per_image": int(row["path_identifier"]["calls_per_image"]),
        "optimization_N_F": int(row["optimization_N_F"]),
        "label": row["label"],
        "path_sha256": row["path_sha256"],
        "shard": str(target),
        "shard_sha256": ENGINE.sha256_file(target),
        "dense_coordinate_sha256": str(scalar(shard, "dense_coordinate_sha256")),
        "dense_energy_sha256": str(scalar(shard, "dense_energy_sha256")),
        "dense_force_sha256": str(scalar(shard, "dense_force_sha256")),
        "optimization_N_F_contribution": 0,
    }


def validate_r2_shard(path: Path, row: dict, state: dict) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "schema", "status", "method", "label", "optimization_N_F",
            "path_identifier_json", "contract_sha256", "evaluator_sha256",
            "joint_engine_sha256", "bb_core_sha256", "topology_sha256",
            "short_replay_source", "short_replay_source_sha256", "selected_path",
            "selected_path_sha256", "discrete_I_perp", "discrete_I_perp_source",
            "dense_coordinates_A", "dense_energy_eV", "dense_force_eV_per_A",
            "dense_coordinate_sha256", "dense_energy_sha256", "dense_force_sha256",
            "densified_Delta_E_kcal_per_mol", "number_of_points",
            "dense_joint_EF_attempts", "dense_endpoint_joint_EF_attempts",
            "original_discrete_new_calls", "optimization_N_F_contribution",
            "R1_partial_values_reused",
        }
        require(required.issubset(saved.files), f"incomplete R2 shard {path}")
        require(str(scalar(saved, "schema")) == SHARD_SCHEMA and str(scalar(saved, "status")) == "PASS", "R2 shard identity mismatch")
        require(str(scalar(saved, "method")) == "IVM", "R2 shard method mismatch")
        require(str(scalar(saved, "contract_sha256")) == state["contract_sha256"], "R2 shard contract mismatch")
        require(str(scalar(saved, "evaluator_sha256")) == ENGINE.sha256_file(Path(__file__)), "R2 shard evaluator mismatch")
        require(str(scalar(saved, "joint_engine_sha256")) == ENGINE.sha256_file(ENGINE_PATH), "R2 shard engine mismatch")
        require(str(scalar(saved, "bb_core_sha256")) == ENGINE.sha256_file(ROOT / "src/bb_core.py"), "R2 shard core mismatch")
        require(str(scalar(saved, "topology_sha256")) == ENGINE.sha256_file(state["topology"]), "R2 shard topology mismatch")
        require(str(scalar(saved, "label")) == row["label"] and int(scalar(saved, "optimization_N_F")) == int(row["optimization_N_F"]), "R2 shard row mismatch")
        require(json.loads(str(scalar(saved, "path_identifier_json"))) == row["path_identifier"], "R2 shard identifier mismatch")
        require(str(scalar(saved, "short_replay_source")) == str(row["source_path"]), "R2 shard replay source mismatch")
        require(str(scalar(saved, "short_replay_source_sha256")) == row["short_replay_source"]["sha256"], "R2 shard replay source SHA mismatch")
        selected = np.asarray(saved["selected_path"], dtype=np.float64)
        dense = np.asarray(saved["dense_coordinates_A"], dtype=np.float64)
        energy = np.asarray(saved["dense_energy_eV"], dtype=np.float64)
        force = np.asarray(saved["dense_force_eV_per_A"], dtype=np.float64)
        require(selected.shape == (N_IMAGES, DIMENSION), "R2 selected path shape mismatch")
        require(dense.shape == force.shape == (N_DENSE, DIMENSION), "R2 dense coordinate/force shape mismatch")
        require(energy.shape == (N_DENSE,), "R2 dense energy shape mismatch")
        require(np.all(np.isfinite(selected)) and np.all(np.isfinite(dense)) and np.all(np.isfinite(energy)) and np.all(np.isfinite(force)), "R2 shard has non-finite values")
        require(np.array_equal(selected, row["path"]), "R2 selected path bytes mismatch")
        require(ENGINE.sha256_array(selected) == str(scalar(saved, "selected_path_sha256")) == row["path_sha256"], "R2 selected path hash mismatch")
        expected_dense, _ = ENGINE.uniform_arc500(selected)
        require(np.array_equal(dense, expected_dense), "R2 dense coordinates differ from arc500")
        require(ENGINE.sha256_array(dense) == str(scalar(saved, "dense_coordinate_sha256")), "R2 dense coordinate SHA mismatch")
        require(ENGINE.sha256_array(energy) == str(scalar(saved, "dense_energy_sha256")), "R2 energy SHA mismatch")
        require(ENGINE.sha256_array(force) == str(scalar(saved, "dense_force_sha256")), "R2 force SHA mismatch")
        require(str(scalar(saved, "discrete_I_perp_source")) == "sealed_verified_short_replay_online_history", "R2 I_perp source mismatch")
        require(same_float(scalar(saved, "discrete_I_perp"), row["i_perp"]), "R2 I_perp mismatch")
        delta = float((np.max(energy) - energy[0]) * ENGINE.EV_TO_KCAL)
        require(same_float(delta, scalar(saved, "densified_Delta_E_kcal_per_mol")), "R2 Delta E mismatch")
        require(int(scalar(saved, "number_of_points")) == N_DENSE, "R2 point count mismatch")
        require(int(scalar(saved, "dense_joint_EF_attempts")) == N_DENSE, "R2 attempt count mismatch")
        require(int(scalar(saved, "dense_endpoint_joint_EF_attempts")) == 2, "R2 endpoint diagnostic count mismatch")
        require(int(scalar(saved, "original_discrete_new_calls")) == 0, "R2 discrete call mismatch")
        require(int(scalar(saved, "optimization_N_F_contribution")) == 0, "R2 diagnostic entered optimization N_F")
        require(int(scalar(saved, "R1_partial_values_reused")) == 0, "R2 shard reused quarantined R1 values")
        return {key: np.asarray(saved[key]).copy() for key in saved.files}


def _validate_fresh_wal_closure(events: list[dict], row: dict, target: Path, shard: dict) -> dict:
    row_index = int(row["row_index"])
    report = ENGINE._validate_stage_wal_closure(
        events,
        row_index,
        row,
        "dense_arc500_joint_EF",
        np.asarray(shard["dense_coordinates_A"]),
        np.asarray(shard["dense_energy_eV"]),
        np.asarray(shard["dense_force_eV_per_A"]),
    )
    stable = _fresh_publication_event(row, target, shard, "dense")
    publications = [
        (index, event)
        for index, event in enumerate(events)
        if _matching_stable_event(event, stable)
    ]
    require(len(publications) == 1, f"{row['label']} needs one R2 dense publication")
    complete_index = max(
        index
        for index, event in enumerate(events)
        if event.get("event") == "path_stage_data_complete_uncommitted"
        and int(event.get("row_index", -1)) == row_index
        and event.get("stage") == "dense_arc500_joint_EF"
    )
    require(publications[0][0] > complete_index, f"{row['label']} publication precedes completion")
    return report


def _validate_source_closure(events: list[dict], row: dict, target: Path, shard: dict) -> None:
    source = _stable_source_event(row)
    publication = _fresh_publication_event(row, target, shard, "source")
    source_at = [index for index, event in enumerate(events) if _matching_stable_event(event, source)]
    publish_at = [index for index, event in enumerate(events) if _matching_stable_event(event, publication)]
    require(len(source_at) == len(publish_at) == 1 and source_at[0] < publish_at[0], f"{row['label']} source closure mismatch")


def _preflight_existing_r2(state: dict, output: Path) -> dict:
    source_path, dense_path = _wal_paths(output)
    require(source_path.exists() == dense_path.exists(), "R2 source and dense WAL must coexist")
    source_events: list[dict] = []
    dense_events: list[dict] = []
    lifetime = {
        "attempt_started": 0,
        "attempt_succeeded": 0,
        "attempt_failed": 0,
        "attempt_open": 0,
        "retry_attempts": 0,
    }
    if source_path.exists():
        source_events = _read_r2_wal(source_path, SOURCE_WAL_SCHEMA, state)
        dense_events = _read_r2_wal(dense_path, DENSE_WAL_SCHEMA, state)
        lifetime = _r2_lifetime(dense_events)
        require(lifetime["attempt_failed"] == lifetime["attempt_open"] == 0, "R2 failed/open attempt requires a new superseding registration")

    existing = []
    accepted_attempts = 0
    for row in state["rows"]:
        call = int(row["path_identifier"]["calls_per_image"])
        if call in IMPORTED_CALLS:
            continue
        target = _r2_shard_path(output, call)
        row_attempts = [
            event
            for event in dense_events
            if event.get("event") == "attempt_started"
            and int(event.get("row_index", -1)) == int(row["row_index"])
        ]
        if target.exists():
            require(source_path.exists(), f"call-{call} R2 shard lacks WAL")
            shard = validate_r2_shard(target, row, state)
            report = _validate_fresh_wal_closure(dense_events, row, target, shard)
            _validate_source_closure(source_events, row, target, shard)
            require(len(row_attempts) == N_DENSE, f"call-{call} R2 lifetime attempts mismatch")
            accepted_attempts += int(report["accepted_stage_attempts"])
            existing.append(call)
        else:
            require(not row_attempts, f"call-{call} has an incomplete R2 WAL tail; require R3")
    require(lifetime["attempt_started"] == accepted_attempts, "R2 WAL contains attempts outside complete shards")
    return {
        "source_events": source_events,
        "dense_events": dense_events,
        "lifetime": lifetime,
        "existing_complete_R2_shards": existing,
    }


def preflight(state: dict, output: Path, *, probe_locks: bool = True) -> dict:
    require(output == state["result_output"] or state.get("allow_test_output") is True, "output differs from frozen result path")
    for final in _final_paths(output):
        if final.exists():
            raise FileExistsError(f"refusing to overwrite {final}")
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = _preflight_existing_r2(state, output)
    missing = len(FRESH_CALLS) - len(existing["existing_complete_R2_shards"])
    required_free = MINIMUM_FREE_BYTES_RESERVE + missing * ESTIMATED_SHARD_BYTES
    observed_free = shutil.disk_usage(output.parent).free
    require(observed_free >= required_free, f"insufficient free disk: {observed_free} < {required_free}")
    if probe_locks:
        with ENGINE.acquire_two_locks(state["locks"]):
            pass
    return {
        "imported_R1_complete_shards": list(IMPORTED_CALLS),
        "imported_R1_new_physical_calls": 0,
        "quarantined_R1_successes": 35,
        "quarantined_R1_open_attempts": 1,
        "quarantined_R1_values_reused": 0,
        "existing_complete_R2_shards": existing["existing_complete_R2_shards"],
        "fresh_R2_paths_planned": missing,
        "fresh_R2_joint_EF_attempts_planned": missing * N_DENSE,
        "fresh_R2_full_force_vectors_planned": missing * N_DENSE,
        "R2_attempt_lifetime": existing["lifetime"],
        "observed_free_bytes": observed_free,
        "required_free_bytes": required_free,
        "canonical_two_locks_available": probe_locks,
        "optimization_N_F_contribution": 0,
    }


def _attempt(wal, backend, row: dict, point_index: int, coordinate: np.ndarray):
    if not hasattr(wal, "R2_attempt_started_count"):
        existing_events = ENGINE._read_wal_events(wal.path)
        wal.R2_attempt_started_count = sum(
            event.get("event") == "attempt_started" for event in existing_events
        )
    starts = int(wal.R2_attempt_started_count)
    require(starts < MAXIMUM_R2_ATTEMPT_STARTED, "R2 physical-attempt authorization exhausted")
    coordinate = np.ascontiguousarray(coordinate, dtype=np.float64)
    attempt_id = (
        f"{wal.session_id}:{row['row_index']}:dense_arc500_joint_EF:"
        f"{point_index}:{starts + 1}"
    )
    started = {
        "event": "attempt_started",
        "utc": ENGINE.utc_now(),
        "stage": "dense_arc500_joint_EF",
        "row_index": int(row["row_index"]),
        "label": row["label"],
        "optimization_N_F": int(row["optimization_N_F"]),
        "path_identifier": row["path_identifier"],
        "path_sha256": row["path_sha256"],
        "point_index": point_index,
        "coordinate_sha256": ENGINE.sha256_array(coordinate),
        "R2_attempt_ordinal": starts + 1,
        "session_id": wal.session_id,
        "attempt_id": attempt_id,
    }
    wal.append(started)
    wal.R2_attempt_started_count = starts + 1
    try:
        energy, force = backend.evaluate_joint(coordinate)
        energy = float(energy)
        force = np.ascontiguousarray(force, dtype=np.float64)
        require(force.shape == (DIMENSION,), "R2 backend returned wrong force shape")
        require(np.isfinite(energy) and np.all(np.isfinite(force)), "R2 backend returned non-finite E/F")
    except Exception as error:
        wal.append({**started, "event": "attempt_failed", "utc": ENGINE.utc_now(), "error": repr(error)})
        raise
    wal.append(
        {
            **started,
            "event": "attempt_succeeded",
            "utc": ENGINE.utc_now(),
            "energy_eV": energy,
            "energy_scalar_sha256": ENGINE.sha256_array(np.asarray(energy, dtype=np.float64)),
            "force_sha256": ENGINE.sha256_array(force),
        }
    )
    return energy, force


def _evaluate_fresh_path(state: dict, row: dict, output: Path, backend, source_wal, dense_wal) -> dict:
    call = int(row["path_identifier"]["calls_per_image"])
    target = _r2_shard_path(output, call)
    source_events = ENGINE._read_wal_events(source_wal.path)
    _ensure_once(source_wal, source_events, _stable_source_event(row))
    dense, _ = ENGINE.uniform_arc500(row["path"])
    dense_wal.append(
        {
            "event": "path_stage_started",
            "utc": ENGINE.utc_now(),
            "stage": "dense_arc500_joint_EF",
            "row_index": int(row["row_index"]),
            "calls_per_image": call,
            "label": row["label"],
            "optimization_N_F": int(row["optimization_N_F"]),
            "path_sha256": row["path_sha256"],
            "dense_coordinate_sha256": ENGINE.sha256_array(dense),
            "number_of_points": N_DENSE,
            "R1_partial_values_reused": 0,
        }
    )
    energy = np.empty(N_DENSE, dtype=np.float64)
    force = np.empty((N_DENSE, DIMENSION), dtype=np.float64)
    for point_index, coordinate in enumerate(dense):
        energy[point_index], force[point_index] = _attempt(
            dense_wal, backend, row, point_index, coordinate
        )
    require(np.all(np.isfinite(energy)) and np.all(np.isfinite(force)), "R2 dense E/F is non-finite")
    delta = float((np.max(energy) - energy[0]) * ENGINE.EV_TO_KCAL)
    dense_wal.append(
        {
            "event": "path_stage_data_complete_uncommitted",
            "utc": ENGINE.utc_now(),
            "stage": "dense_arc500_joint_EF",
            "row_index": int(row["row_index"]),
            "calls_per_image": call,
            "label": row["label"],
            "optimization_N_F": int(row["optimization_N_F"]),
            "path_sha256": row["path_sha256"],
            "joint_attempts": N_DENSE,
            "joint_successes": N_DENSE,
            "dense_coordinate_sha256": ENGINE.sha256_array(dense),
            "dense_energy_sha256": ENGINE.sha256_array(energy),
            "dense_force_sha256": ENGINE.sha256_array(force),
            "densified_Delta_E_kcal_per_mol": delta,
            "R1_partial_values_reused": 0,
        }
    )
    arrays = {
        "schema": np.asarray(SHARD_SCHEMA),
        "status": np.asarray("PASS"),
        "method": np.asarray("IVM"),
        "label": np.asarray(row["label"]),
        "optimization_N_F": np.asarray(row["optimization_N_F"], dtype=np.int64),
        "path_identifier_json": np.asarray(json.dumps(row["path_identifier"], sort_keys=True)),
        "contract_sha256": np.asarray(state["contract_sha256"]),
        "evaluator_sha256": np.asarray(ENGINE.sha256_file(Path(__file__))),
        "joint_engine_sha256": np.asarray(ENGINE.sha256_file(ENGINE_PATH)),
        "bb_core_sha256": np.asarray(ENGINE.sha256_file(ROOT / "src/bb_core.py")),
        "topology_sha256": np.asarray(ENGINE.sha256_file(state["topology"])),
        "short_replay_source": np.asarray(str(row["source_path"])),
        "short_replay_source_sha256": np.asarray(row["short_replay_source"]["sha256"]),
        "selected_path": row["path"],
        "selected_path_sha256": np.asarray(row["path_sha256"]),
        "discrete_I_perp": np.asarray(row["i_perp"], dtype=np.float64),
        "discrete_I_perp_source": np.asarray("sealed_verified_short_replay_online_history"),
        "dense_coordinates_A": dense,
        "dense_energy_eV": energy,
        "dense_force_eV_per_A": force,
        "dense_coordinate_sha256": np.asarray(ENGINE.sha256_array(dense)),
        "dense_energy_sha256": np.asarray(ENGINE.sha256_array(energy)),
        "dense_force_sha256": np.asarray(ENGINE.sha256_array(force)),
        "densified_Delta_E_kcal_per_mol": np.asarray(delta, dtype=np.float64),
        "number_of_points": np.asarray(N_DENSE, dtype=np.int64),
        "dense_joint_EF_attempts": np.asarray(N_DENSE, dtype=np.int64),
        "dense_endpoint_joint_EF_attempts": np.asarray(2, dtype=np.int64),
        "original_discrete_new_calls": np.asarray(0, dtype=np.int64),
        "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
        "R1_partial_values_reused": np.asarray(0, dtype=np.int64),
    }
    ENGINE._atomic_save_npz_no_overwrite(target, arrays)
    shard = validate_r2_shard(target, row, state)
    dense_wal.append(
        {**_fresh_publication_event(row, target, shard, "dense"), "utc": ENGINE.utc_now()}
    )
    source_wal.append(
        {**_fresh_publication_event(row, target, shard, "source"), "utc": ENGINE.utc_now()}
    )
    return shard


def _authoritative_shards(state: dict, output: Path, dense_events: list[dict], source_events: list[dict]) -> tuple[list[dict], list[Path], list[str]]:
    shards = []
    paths = []
    origins = []
    for row in state["rows"]:
        call = int(row["path_identifier"]["calls_per_image"])
        if call in IMPORTED_CALLS:
            shard = row["shard"]
            path = row["shard_path"]
            origin = "imported_complete_R1_shard"
        else:
            path = _r2_shard_path(output, call)
            shard = validate_r2_shard(path, row, state)
            _validate_fresh_wal_closure(dense_events, row, path, shard)
            _validate_source_closure(source_events, row, path, shard)
            origin = "fresh_complete_R2_shard"
        shards.append(shard)
        paths.append(path)
        origins.append(origin)
    return shards, paths, origins


def _validate_aggregate(output: Path, state: dict, shards: list[dict], paths: list[Path], origins: list[str]) -> None:
    with np.load(output, allow_pickle=False) as saved:
        required = {
            "schema", "status", "method", "authoritative_data_unit", "labels",
            "optimization_N_F", "selected_path", "selected_path_sha256",
            "discrete_I_perp", "energy_eV", "energy_sha256",
            "dense_coordinate_sha256", "dense_force_sha256",
            "densified_Delta_E_kcal_per_mol", "shard_path", "shard_sha256",
            "shard_origin", "contract_sha256", "imported_R1_new_physical_calls",
            "quarantined_R1_values_reused", "fresh_R2_joint_EF_attempts",
            "optimization_N_F_contribution",
        }
        require(required.issubset(saved.files), "R2 aggregate fields missing")
        require(str(scalar(saved, "schema")) == RESULT_SCHEMA and str(scalar(saved, "status")) == "PASS", "R2 aggregate identity mismatch")
        require(str(scalar(saved, "method")) == "IVM", "R2 aggregate method mismatch")
        require(str(scalar(saved, "authoritative_data_unit")) == "complete_atomic_path_shard", "R2 aggregate authority mismatch")
        require(str(scalar(saved, "contract_sha256")) == state["contract_sha256"], "R2 aggregate contract mismatch")
        require("dense_coordinates_A" not in saved.files and "dense_force_eV_per_A" not in saved.files, "R2 aggregate duplicates full dense arrays")
        expected = {
            "labels": np.asarray([row["label"] for row in state["rows"]]),
            "optimization_N_F": np.asarray(EXPECTED_NF, dtype=np.int64),
            "selected_path": np.asarray([row["path"] for row in state["rows"]]),
            "selected_path_sha256": np.asarray([row["path_sha256"] for row in state["rows"]]),
            "discrete_I_perp": np.asarray([row["i_perp"] for row in state["rows"]], dtype=np.float64),
            "energy_eV": np.asarray([shard["dense_energy_eV"] for shard in shards]),
            "energy_sha256": np.asarray([str(scalar(shard, "dense_energy_sha256")) for shard in shards]),
            "dense_coordinate_sha256": np.asarray([str(scalar(shard, "dense_coordinate_sha256")) for shard in shards]),
            "dense_force_sha256": np.asarray([str(scalar(shard, "dense_force_sha256")) for shard in shards]),
            "densified_Delta_E_kcal_per_mol": np.asarray([float(scalar(shard, "densified_Delta_E_kcal_per_mol")) for shard in shards], dtype=np.float64),
            "shard_path": np.asarray([str(path) for path in paths]),
            "shard_sha256": np.asarray([ENGINE.sha256_file(path) for path in paths]),
            "shard_origin": np.asarray(origins),
        }
        for key, wanted in expected.items():
            actual = np.asarray(saved[key])
            require(actual.dtype.str == wanted.dtype.str and actual.shape == wanted.shape and actual.tobytes(order="C") == wanted.tobytes(order="C"), f"R2 aggregate {key} mismatch")
        require(int(scalar(saved, "imported_R1_new_physical_calls")) == 0, "R1 import call accounting mismatch")
        require(int(scalar(saved, "quarantined_R1_values_reused")) == 0, "R1 quarantine accounting mismatch")
        require(int(scalar(saved, "fresh_R2_joint_EF_attempts")) == FRESH_ACCEPTED_SUCCESSES, "R2 fresh attempt accounting mismatch")
        require(int(scalar(saved, "optimization_N_F_contribution")) == 0, "R2 diagnostics entered optimization N_F")


def run(state: dict, output: Path) -> None:
    initial = preflight(state, output)
    source_wal_path, dense_wal_path = _wal_paths(output)
    manifest_path = output.with_suffix(".manifest.json")
    audit_path = output.with_suffix(".audit.json")
    backend = None
    fresh_this_invocation = []
    reused_complete_R2 = []
    with ENGINE.acquire_two_locks(state["locks"]) as held:
        for final in _final_paths(output):
            if final.exists():
                raise FileExistsError(f"refusing to overwrite {final}")
        existing = _preflight_existing_r2(state, output)
        source_wal = ENGINE.DurableWal(
            source_wal_path, SOURCE_WAL_SCHEMA, state["contract_path"], state["contract_sha256"]
        )
        dense_wal = ENGINE.DurableWal(
            dense_wal_path, DENSE_WAL_SCHEMA, state["contract_path"], state["contract_sha256"]
        )
        try:
            source_events = ENGINE._read_wal_events(source_wal_path)
            dense_events = ENGINE._read_wal_events(dense_wal_path)
            _ensure_once(source_wal, source_events, _stable_quarantine_event(state, "source"))
            _ensure_once(dense_wal, dense_events, _stable_quarantine_event(state, "dense"))
            for row in state["rows"]:
                call = int(row["path_identifier"]["calls_per_image"])
                if call in IMPORTED_CALLS:
                    source_events = ENGINE._read_wal_events(source_wal_path)
                    dense_events = ENGINE._read_wal_events(dense_wal_path)
                    _ensure_once(source_wal, source_events, _stable_import_event(state, row, "source"))
                    _ensure_once(dense_wal, dense_events, _stable_import_event(state, row, "dense"))
                    continue
                target = _r2_shard_path(output, call)
                if target.exists():
                    reused_complete_R2.append(call)
                    continue
                if backend is None:
                    backend = ENGINE.OpenMMBackend(
                        state["topology"],
                        platform_preference=("CPU",),
                        precision="double",
                        label="bb_ivm_short_dense_joint_ef_resume_r2",
                    )
                _evaluate_fresh_path(state, row, output, backend, source_wal, dense_wal)
                fresh_this_invocation.append(call)

            if backend is not None:
                require(
                    int(backend.raw_joint_attempts)
                    == len(fresh_this_invocation) * N_DENSE,
                    "backend/WAL fresh-attempt count mismatch",
                )
                require(
                    int(backend.raw_joint_evaluations)
                    == len(fresh_this_invocation) * N_DENSE,
                    "backend/WAL fresh-success count mismatch",
                )

            source_events = _read_r2_wal(source_wal_path, SOURCE_WAL_SCHEMA, state)
            dense_events = _read_r2_wal(dense_wal_path, DENSE_WAL_SCHEMA, state)
            lifetime = _r2_lifetime(dense_events)
            require(
                lifetime
                == {
                    "attempt_started": FRESH_ACCEPTED_SUCCESSES,
                    "attempt_succeeded": FRESH_ACCEPTED_SUCCESSES,
                    "attempt_failed": 0,
                    "attempt_open": 0,
                    "retry_attempts": 0,
                },
                "R2 final WAL must contain exactly 1000 clean attempts",
            )
            shards, paths, origins = _authoritative_shards(
                state, output, dense_events, source_events
            )
            result = {
                "schema": np.asarray(RESULT_SCHEMA),
                "status": np.asarray("PASS"),
                "method": np.asarray("IVM"),
                "authoritative_data_unit": np.asarray("complete_atomic_path_shard"),
                "labels": np.asarray([row["label"] for row in state["rows"]]),
                "optimization_N_F": np.asarray(EXPECTED_NF, dtype=np.int64),
                "selected_path": np.asarray([row["path"] for row in state["rows"]]),
                "selected_path_sha256": np.asarray([row["path_sha256"] for row in state["rows"]]),
                "discrete_I_perp": np.asarray([row["i_perp"] for row in state["rows"]], dtype=np.float64),
                "energy_eV": np.asarray([shard["dense_energy_eV"] for shard in shards]),
                "energy_sha256": np.asarray([str(scalar(shard, "dense_energy_sha256")) for shard in shards]),
                "dense_coordinate_sha256": np.asarray([str(scalar(shard, "dense_coordinate_sha256")) for shard in shards]),
                "dense_force_sha256": np.asarray([str(scalar(shard, "dense_force_sha256")) for shard in shards]),
                "densified_Delta_E_kcal_per_mol": np.asarray([float(scalar(shard, "densified_Delta_E_kcal_per_mol")) for shard in shards], dtype=np.float64),
                "shard_path": np.asarray([str(path) for path in paths]),
                "shard_sha256": np.asarray([ENGINE.sha256_file(path) for path in paths]),
                "shard_origin": np.asarray(origins),
                "contract_sha256": np.asarray(state["contract_sha256"]),
                "imported_R1_new_physical_calls": np.asarray(0, dtype=np.int64),
                "quarantined_R1_values_reused": np.asarray(0, dtype=np.int64),
                "fresh_R2_joint_EF_attempts": np.asarray(FRESH_ACCEPTED_SUCCESSES, dtype=np.int64),
                "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
            }
            ENGINE._atomic_save_npz_no_overwrite(output, result)
            _validate_aggregate(output, state, shards, paths, origins)
            aggregate_event = {
                "event": "aggregate_published",
                "aggregate": str(output),
                "aggregate_sha256": ENGINE.sha256_file(output),
                "authoritative_shard_paths": [str(path) for path in paths],
                "authoritative_shard_sha256": [ENGINE.sha256_file(path) for path in paths],
                "optimization_N_F_contribution": 0,
            }
            source_wal.append({**aggregate_event, "wal_role": "source", "utc": ENGINE.utc_now()})
            dense_wal.append({**aggregate_event, "wal_role": "dense", "utc": ENGINE.utc_now()})
        finally:
            source_wal.close()
            dense_wal.close()

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "PASS",
            "contract": str(state["contract_path"]),
            "contract_sha256": state["contract_sha256"],
            "aggregate": binding(output),
            "source_wal": binding(source_wal_path),
            "dense_joint_ef_wal": binding(dense_wal_path),
            "shards": [
                {
                    "calls_per_image": call,
                    "optimization_N_F": nf,
                    "origin": origin,
                    **binding(path),
                    "selected_path_sha256": row["path_sha256"],
                    "dense_coordinate_sha256": str(scalar(shard, "dense_coordinate_sha256")),
                    "dense_energy_sha256": str(scalar(shard, "dense_energy_sha256")),
                    "dense_force_sha256": str(scalar(shard, "dense_force_sha256")),
                }
                for call, nf, row, shard, path, origin in zip(
                    CALLS, EXPECTED_NF, state["rows"], shards, paths, origins
                )
            ],
        }
        ENGINE._atomic_write_json_no_overwrite(manifest_path, manifest)
        audit = {
            "schema": AUDIT_SCHEMA,
            "status": "PASS",
            "method": "IVM",
            "contract": binding(state["contract_path"]),
            "output": binding(output),
            "manifest": binding(manifest_path),
            "source_wal": binding(source_wal_path),
            "dense_joint_ef_wal": binding(dense_wal_path),
            "imported_R1_complete_shards": list(IMPORTED_CALLS),
            "imported_R1_dense_joint_EF_values": len(IMPORTED_CALLS) * N_DENSE,
            "imported_R1_new_physical_calls": 0,
            "quarantined_R1_partial_tail": {
                **state["evidence"]["partial"],
                "values_reused": 0,
                "pause_receipt": binding(state["evidence"]["pause_receipt"]),
            },
            "fresh_R2_paths_this_invocation": fresh_this_invocation,
            "complete_R2_shards_reused_this_invocation": reused_complete_R2,
            "fresh_R2_joint_EF_attempts_lineage": FRESH_ACCEPTED_SUCCESSES,
            "fresh_R2_joint_EF_successes_lineage": FRESH_ACCEPTED_SUCCESSES,
            "fresh_R2_full_force_vectors_stored": len(FRESH_CALLS) * N_DENSE,
            "total_accepted_dense_joint_EF_values": len(CALLS) * N_DENSE,
            "total_full_force_vectors_stored_in_authoritative_shards": len(CALLS) * N_DENSE,
            "R2_WAL_lifetime": lifetime,
            "original_discrete_new_calls": 0,
            "optimization_N_F_contribution": 0,
            "canonical_two_locks_held": held,
            "initial_preflight": initial,
        }
        ENGINE._atomic_write_json_no_overwrite(audit_path, audit)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host-role", choices=("local",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    ENGINE.check_thread_caps()
    contract_path = args.contract.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    require(output.suffix == ".npz", "R2 output must be .npz")
    state = load_contract(contract_path, args.host_role)
    require(output == state["result_output"], "CLI output differs from frozen R2 result path")
    if args.preflight_only:
        report = preflight(state, output)
        print(
            json.dumps(
                {
                    "status": "ZERO_OPENMM_PREFLIGHT_PASS",
                    "contract": str(contract_path),
                    "contract_sha256": state["contract_sha256"],
                    "host_role": "local",
                    "method": "IVM",
                    "paths": list(CALLS),
                    "optimization_N_F": list(EXPECTED_NF),
                    "imported_R1_complete_shards": list(IMPORTED_CALLS),
                    "fresh_R2_paths": list(FRESH_CALLS),
                    "fresh_R2_joint_EF_attempts_planned": report["fresh_R2_joint_EF_attempts_planned"],
                    "quarantined_R1_values_reused": 0,
                    "optimization_N_F_contribution": 0,
                    "physical_preflight": report,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    run(state, output)


if __name__ == "__main__":
    main()
