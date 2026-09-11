#!/usr/bin/env python3
"""Zero-OpenMM exact-prefix verifier and per-path eligibility index."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA = "bb-irivm-selected-full-history-replay-verification-v1"
REGISTRATION_SCHEMA = "bb-irivm-selected-full-history-output-only-replay-v1"
AUTH_SCHEMA = "bb-irivm-selected-full-history-output-only-replay-authorization-v1"
LABEL = "bb_irivm_a1e0_r01_m05_inf9990_fullhistory_r1"
ROWS = 352
LAST_COMPLETE_CALL = 318
LAST_COMPLETE_NF = 8982
TERMINAL_NF = 9985
TAIL_NF = 1003
EXCLUDED_HISTORY_KEYS = {"wall_seconds"}
REQUIRED_HISTORY_KEYS = {
    "calls_per_image", "counter_vectors", "energy_request_vectors",
    "force_request_vectors", "i_perp", "reparam_committed",
    "total_internal_joint_evaluations",
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


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def exact(left, right) -> bool:
    first, second = np.asarray(left), np.asarray(right)
    return first.shape == second.shape and first.dtype.str == second.dtype.str and first.tobytes() == second.tobytes()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not an object")
    return value


def bind(path: Path) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def scalar(saved, key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def classify_inventory(calls, nf, i_perp, committed, paths) -> list[dict]:
    result = []
    for index, call in enumerate(np.asarray(calls, dtype=np.int64)):
        eligible = int(call) <= LAST_COMPLETE_CALL
        result.append({
            "history_index": index,
            "call_per_image": int(call),
            "optimization_N_F": int(nf[index]),
            "discrete_I_perp_eV2_per_A": float(i_perp[index]),
            "selected_path_sha256": sha256_array(paths[index]),
            "reparameterization_committed": bool(committed[index]),
            "fully_evaluated_current_path": True,
            "formally_reportable": eligible,
            "terminal_uncommitted_charged_tail": not eligible,
            "exclusion_reason": None if eligible else "after_last_complete_state_in_terminal_uncommitted_transaction",
        })
    return result


def verify(reference_path: Path, replay_path: Path, replay_audit_path: Path) -> tuple[dict, list[dict]]:
    audit = read_json(replay_audit_path)
    require(audit.get("consistent") is True, "replay force audit inconsistent")
    require(Path(str(audit.get("result_file", ""))).expanduser().resolve() == replay_path, "audit/result mismatch")
    with np.load(reference_path, allow_pickle=False) as reference, np.load(replay_path, allow_pickle=False) as replay:
        reference_calls = np.asarray(reference["calls_per_image"])
        replay_calls = np.asarray(replay["calls_per_image"])
        require(exact(reference_calls, replay_calls), "calls are not exact selected prefix")
        require(replay_calls.shape == (ROWS,) and np.array_equal(replay_calls, np.arange(1, ROWS + 1)), "replay calls are not 1..352")
        compared = []
        for key in reference.files:
            if key in EXCLUDED_HISTORY_KEYS or key not in replay.files:
                continue
            left, right = np.asarray(reference[key]), np.asarray(replay[key])
            if left.ndim >= 1 and right.ndim >= 1 and left.shape[0] == right.shape[0] == ROWS and left.shape[1:] == right.shape[1:] and left.dtype.str == right.dtype.str:
                require(exact(left, right), f"online exact-history mismatch: {key}")
                compared.append(key)
        require(not REQUIRED_HISTORY_KEYS.difference(compared), "required online histories were not compared")
        require(str(scalar(reference, "configuration_sha256")) == str(scalar(replay, "configuration_sha256")), "algorithm configuration hash changed")
        require(str(scalar(replay, "path_storage_mode")) == "legacy", "storage mode is not legacy")
        require(scalar(replay, "path_storage_affects_online_algorithm") is False, "storage affected online algorithm")
        stored_calls = np.asarray(replay["stored_path_calls"], dtype=np.int64)
        stored_paths = np.asarray(replay["stored_paths"], dtype=np.float64)
        require(exact(stored_calls, replay_calls), "not every external-iteration path was retained")
        require(stored_paths.shape == (ROWS, 32, 9396) and np.all(np.isfinite(stored_paths)), "full stored path tensor invalid")
        reference_stored_calls = np.asarray(reference["stored_path_calls"], dtype=np.int64)
        reference_stored_paths = np.asarray(reference["stored_paths"], dtype=np.float64)
        for ref_index, call in enumerate(reference_stored_calls):
            replay_index = np.flatnonzero(stored_calls == int(call))
            require(replay_index.size == 1 and exact(reference_stored_paths[ref_index], stored_paths[int(replay_index[0])]), f"stored reference path mismatch at call {call}")
        require(exact(reference["initial_path"], replay["initial_path"]), "initial path mismatch")
        require(exact(reference["final_path"], replay["final_path"]), "terminal path mismatch")
        require(int(scalar(replay, "total_internal_nf_last_complete_state_index")) + 1 == LAST_COMPLETE_CALL, "last complete call mismatch")
        require(int(scalar(replay, "total_internal_nf_last_complete_state_nf")) == LAST_COMPLETE_NF, "last complete N_F mismatch")
        require(int(scalar(replay, "total_internal_nf_consumed")) == TERMINAL_NF, "terminal N_F mismatch")
        require(int(scalar(replay, "total_internal_nf_terminal_uncommitted_charged_tail_nf")) == TAIL_NF, "tail N_F mismatch")
        require(bool(scalar(replay, "total_internal_nf_terminal_partial_tail_present")), "terminal tail absent")
        require(bool(scalar(replay, "total_internal_nf_terminal_partial_tail_quarantined")), "terminal tail not quarantined")
        inventory = classify_inventory(
            replay_calls,
            np.asarray(replay["total_internal_joint_evaluations"]),
            np.asarray(replay["i_perp"]),
            np.asarray(replay["reparam_committed"]),
            stored_paths,
        )
        summary = {
            "compared_exact_history_keys": sorted(compared),
            "configuration_sha256": str(scalar(replay, "configuration_sha256")),
            "stored_path_count": int(stored_calls.size),
            "reportable_path_count": sum(row["formally_reportable"] for row in inventory),
            "excluded_tail_path_count": sum(row["terminal_uncommitted_charged_tail"] for row in inventory),
            "last_complete_call": LAST_COMPLETE_CALL,
            "last_complete_optimization_N_F": LAST_COMPLETE_NF,
            "terminal_call": ROWS,
            "terminal_optimization_N_F": TERMINAL_NF,
            "terminal_uncommitted_charged_tail_N_F": TAIL_NF,
        }
    return summary, inventory


def write_exclusive(path: Path, payload: dict) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--replay-result", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    paths = {name: value.expanduser().resolve(strict=True) for name, value in vars(args).items() if name != "output"}
    registration = read_json(paths["registration"])
    authorization = read_json(paths["authorization"])
    require(registration.get("schema") == REGISTRATION_SCHEMA and registration.get("status") == "FROZEN_BEFORE_FIRST_FULL_HISTORY_REPLAY_PHYSICAL_CALL", "registration invalid")
    require(registration.get("label") == LABEL, "registration label mismatch")
    require(authorization.get("schema") == AUTH_SCHEMA and authorization.get("enabled") is True, "authorization invalid")
    require(authorization.get("registration") == bind(paths["registration"]), "authorization/registration mismatch")
    require(registration["bindings"]["selected_terminal_result"] == bind(paths["reference_result"]), "reference result binding mismatch")
    require(registration["source_sha256"][Path(__file__).name] == sha256_file(Path(__file__).resolve()), "verifier source drift")
    summary, inventory = verify(paths["reference_result"], paths["replay_result"], paths["replay_audit"])
    payload = {
        "schema": SCHEMA, "status": "PASS_EXACT_FULL_HISTORY_OUTPUT_ONLY_REPLAY",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tolerance_fallback_used": False,
        "scientific_scope": "all 352 coordinate paths retained; only calls 1..318 reportable; calls 319..352 are stored but quarantined terminal-tail states",
        **summary, "path_inventory": inventory,
        "bindings": {name: bind(path) for name, path in paths.items()},
    }
    write_exclusive(args.output, payload)
    destination = args.output.expanduser().resolve(strict=True)
    print(destination)
    print(sha256_file(destination))


if __name__ == "__main__":
    main()
