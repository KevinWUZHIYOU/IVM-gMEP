#!/usr/bin/env python3
"""IVM path-selection adapter for the frozen four-path metric engine.

Calls 2/5/10/50 are prospectively registered, fully evaluated nonterminal
current paths.  They need not be reparameterization-committed.  This adapter
binds the exact-prefix verification and delegates only physical arc500 work
and durable ledgers to ``evaluate_bb_dense_four_paths.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
ENGINE_PATH = HERE / "evaluate_bb_dense_four_paths.py"
SPEC = importlib.util.spec_from_file_location("bb_dense_four_engine_frozen", ENGINE_PATH)
ENGINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ENGINE)

SCHEMA = "bb-dense-four-ivm-dual-metric-contract-v1"
VERIFICATION_SCHEMA = "bb-irivm-selected-legacy-replay-verification-v1"
CALLS = (2, 5, 10, 50)
EXPECTED_NF = (60, 150, 300, 1329)


def require_file(record, contract_path: Path, role: str) -> Path:
    return ENGINE.require_file(record, contract_path, role)


def load_reuse(contract, contract_path: Path, topology_sha: str):
    reuse = {}
    for bundle in contract.get("allowed_pass_energy_reuse_bundles", []):
        result = require_file(bundle["result"], contract_path, "energy reuse result")
        audit_path = require_file(bundle["audit"], contract_path, "energy reuse audit")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        ENGINE.require(audit.get("status") == "PASS", "energy reuse audit is not PASS")
        ENGINE.require(audit.get("output_sha256") == ENGINE.sha256_file(result), "energy reuse audit/output mismatch")
        with np.load(result, allow_pickle=False) as saved:
            required = {"selected_path_sha256", "dense_coordinate_sha256", "energy_eV", "energy_sha256", "number_of_points", "topology_sha256", "diagnostic_force_requests_per_path", "diagnostic_energy_attempts_per_path", "diagnostic_energy_successes_per_path"}
            ENGINE.require(required.issubset(saved.files), "energy reuse result fields missing")
            ENGINE.require(int(ENGINE._scalar(saved, "number_of_points")) == ENGINE.N_DENSE, "energy reuse point count mismatch")
            ENGINE.require(str(ENGINE._scalar(saved, "topology_sha256")) == topology_sha, "energy reuse topology mismatch")
            force_counts = np.asarray(saved["diagnostic_force_requests_per_path"], dtype=np.int64)
            attempts = np.asarray(saved["diagnostic_energy_attempts_per_path"], dtype=np.int64)
            successes = np.asarray(saved["diagnostic_energy_successes_per_path"], dtype=np.int64)
            ENGINE.require(np.all(force_counts == 0), "energy reuse requested force")
            ENGINE.require(np.all(attempts == ENGINE.N_DENSE) and np.all(successes == ENGINE.N_DENSE), "energy reuse lacks 500 successful calls")
            for index, path_sha in enumerate(np.asarray(saved["selected_path_sha256"]).astype(str)):
                energies = np.asarray(saved["energy_eV"][index], dtype=np.float64)
                energy_sha = str(np.asarray(saved["energy_sha256"])[index])
                ENGINE.require(energies.shape == (ENGINE.N_DENSE,) and ENGINE.sha256_array(energies) == energy_sha, "energy reuse vector hash mismatch")
                entry = {
                    "dense_sha256": str(np.asarray(saved["dense_coordinate_sha256"])[index]),
                    "energies": energies, "energy_sha256": energy_sha,
                    "source_result": str(result), "source_result_sha256": ENGINE.sha256_file(result),
                    "source_audit": str(audit_path), "source_audit_sha256": ENGINE.sha256_file(audit_path),
                    "source_row_index": index,
                }
                previous = reuse.get(path_sha)
                ENGINE.require(previous is None or (previous["dense_sha256"] == entry["dense_sha256"] and previous["energy_sha256"] == entry["energy_sha256"]), "conflicting energy reuse")
                reuse[path_sha] = entry
    return reuse


def validate_and_load(contract_path: Path, host_role: str):
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    ENGINE.require(contract.get("schema") == SCHEMA, "unsupported IVM contract schema")
    ENGINE.require(contract.get("frozen") is True, "IVM contract is not frozen")
    ENGINE.require(int(contract.get("jobs", -1)) == 1, "jobs must equal 1")
    ENGINE.require(contract.get("metric_policy") == ENGINE.METRIC_POLICY, "metric policy mismatch")
    adapter = require_file(contract["ivm_path_adapter"], contract_path, "IVM path adapter")
    ENGINE.require(adapter == Path(__file__).resolve(), "contract binds another IVM adapter")
    engine = require_file(contract["evaluator_engine"], contract_path, "metric engine")
    ENGINE.require(engine == ENGINE_PATH.resolve(), "contract binds another metric engine")
    bb_core = require_file(contract["bb_core"], contract_path, "bb_core")
    ENGINE.require(bb_core == (HERE.parent / "src" / "bb_core.py").resolve(), "contract binds another bb_core")
    topology = require_file(contract["topology"], contract_path, "topology")
    replay_result = require_file(contract["replay_result"], contract_path, "replay result")
    replay_audit = require_file(contract["replay_audit"], contract_path, "replay audit")
    verification_path = require_file(contract["replay_verification"], contract_path, "replay verification")
    audit = json.loads(replay_audit.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    ENGINE.require(audit.get("consistent") is True, "replay NoF audit is inconsistent")
    ENGINE.require(verification.get("schema") == VERIFICATION_SCHEMA and verification.get("status") == "PASS_EXACT_OUTPUT_ONLY_PREFIX", "replay verification did not pass")
    bindings = verification.get("bindings", {})
    ENGINE.require(str(bindings["replay_result"]["sha256"]) == ENGINE.sha256_file(replay_result), "verification/replay result mismatch")
    ENGINE.require(str(bindings["replay_audit"]["sha256"]) == ENGINE.sha256_file(replay_audit), "verification/replay audit mismatch")
    verification_rows = {int(row["call_per_image"]): row for row in verification["new_reporting_rows"]}
    paths = contract.get("paths")
    ENGINE.require(isinstance(paths, list) and len(paths) == 4, "IVM contract must contain four paths")
    ENGINE.require(tuple(int(row["path_identifier"]["calls_per_image"]) for row in paths) == CALLS, "IVM calls must be 2/5/10/50")
    ENGINE.require(tuple(int(row["optimization_N_F"]) for row in paths) == EXPECTED_NF, "IVM N_F must be 60/150/300/1329")
    ENGINE.require(all(row.get("method") == "IVM" and row.get("host_role") == host_role for row in paths), "IVM method/host assignment mismatch")
    with np.load(replay_result, allow_pickle=False) as saved:
        stored_paths = np.asarray(saved["stored_paths"], dtype=np.float64)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
        history_calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
        i_perp = np.asarray(saved["i_perp"], dtype=np.float64)
    loaded = []
    for row in paths:
        call = int(row["path_identifier"]["calls_per_image"])
        si = np.flatnonzero(stored_calls == call)
        hi = np.flatnonzero(history_calls == call)
        ENGINE.require(si.size == hi.size == 1, f"replay call {call} absent/nonunique")
        path = np.ascontiguousarray(stored_paths[int(si[0])], dtype=np.float64)
        index = int(hi[0])
        nf = int(counters[index].sum(dtype=np.int64))
        value = float(i_perp[index])
        verified = verification_rows.get(call)
        ENGINE.require(verified is not None and verified.get("fully_evaluated_current_path") is True and verified.get("terminal_tail") is False, f"call {call} is not verified eligible")
        ENGINE.require(nf == int(row["optimization_N_F"]) == int(verified["optimization_N_F"]), f"call {call} N_F mismatch")
        ENGINE.require(value == float(row["online_discrete_I_perp"]) == float(verified["discrete_I_perp_eV2_per_A"]), f"call {call} I_perp mismatch")
        path_sha = ENGINE.sha256_array(path)
        ENGINE.require(path_sha == str(row["path_sha256"]) == str(verified["selected_path_sha256"]), f"call {call} path SHA mismatch")
        ENGINE.require(row.get("force_policy") == "online_paid_history_zero_diagnostic_force", "IVM force policy mismatch")
        loaded.append({**row, "source_path": replay_result, "path": path, "online_i_perp": value, "computed_nf": nf})
    locks = [Path(value).expanduser().resolve() for value in contract["host_local_lock_paths"]]
    ENGINE.require(len(locks) == 2 and len(set(locks)) == 2, "two distinct lock paths required")
    return contract, topology, locks, loaded, load_reuse(contract, contract_path, ENGINE.sha256_file(topology))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host-role", choices=("local", "remote"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    ENGINE.check_thread_caps()
    contract_path = args.contract.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    ENGINE.require(output.suffix == ".npz", "--output must end in .npz")
    artifacts = (output, output.with_suffix(".audit.json"), output.with_suffix(".force.wal.jsonl"), output.with_suffix(".energy.wal.jsonl"))
    if any(path.exists() for path in artifacts):
        raise FileExistsError("refusing to overwrite an output/audit/WAL artifact")
    contract, topology, locks, rows, reuse = validate_and_load(contract_path, args.host_role)
    summary = {"status": "ZERO_OPENMM_PREFLIGHT_PASS", "contract": str(contract_path), "contract_sha256": ENGINE.sha256_file(contract_path), "method": "IVM", "host_role": args.host_role, "path_count": 4, "calls_per_image": list(CALLS), "optimization_N_F": list(EXPECTED_NF), "path_sha256": [str(row["path_sha256"]) for row in rows], "fresh_force_attempts_planned": 0, "fresh_energy_attempts_planned": sum(str(row["path_sha256"]) not in reuse for row in rows) * ENGINE.N_DENSE, "optimization_N_F_contribution": 0}
    if args.preflight_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    ENGINE.run_evaluation(contract_path, contract, "IVM", topology, locks, rows, reuse, output)


if __name__ == "__main__":
    main()
