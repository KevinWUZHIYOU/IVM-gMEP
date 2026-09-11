#!/usr/bin/env python3
"""Freeze the four replay-path IVM arc500 diagnostic contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "scripts" / "evaluate_bb_dense_four_ivm_paths.py"
ENGINE = ROOT / "scripts" / "evaluate_bb_dense_four_paths.py"
BB_CORE = ROOT / "src" / "bb_core.py"
TOPOLOGY = ROOT / "inputs" / "complex_fixed.pdb"
CALLS = (2, 5, 10, 50)
METRIC_POLICY = {
    "coordinate_space": "unweighted_3N_cartesian",
    "sampling": "uniform_cartesian_arclength",
    "interpolation": "piecewise_linear",
    "number_of_points": 500,
    "grid_formula": "s_j=j*L/499,j=0,...,499",
    "include_both_endpoints": True,
    "discrete_I_perp_implementation": "bb_core.path_diagnostics",
    "tangent": "normalized central difference: x[i+1] - x[i-1]",
    "weight": "half sum of adjacent Cartesian chord lengths",
    "fixed_endpoint_diagnostic_force_evaluations": 0,
    "optimization_N_F_contribution": 0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).hexdigest()


def bind(path: Path):
    path = path.expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-result", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--replay-verification", type=Path, required=True)
    parser.add_argument("--host-role", choices=("local", "remote"), default="remote")
    parser.add_argument("--global-lock-path", type=Path, default=Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock"))
    parser.add_argument("--peer-global-lock-path", type=Path, default=Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock"))
    parser.add_argument("--reuse-result", type=Path, action="append", default=[])
    parser.add_argument("--reuse-audit", type=Path, action="append", default=[])
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    if len(args.reuse_result) != len(args.reuse_audit):
        raise ValueError("--reuse-result and --reuse-audit counts differ")
    replay_result = args.replay_result.expanduser().resolve(strict=True)
    replay_audit = args.replay_audit.expanduser().resolve(strict=True)
    verification_path = args.replay_verification.expanduser().resolve(strict=True)
    audit = json.loads(replay_audit.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if audit.get("consistent") is not True:
        raise ValueError("replay audit is inconsistent")
    if verification.get("schema") != "bb-irivm-selected-legacy-replay-verification-v1" or verification.get("status") != "PASS_EXACT_OUTPUT_ONLY_PREFIX":
        raise ValueError("replay verification did not pass")
    if str(verification["bindings"]["replay_result"]["sha256"]) != sha256(replay_result):
        raise ValueError("verification does not bind replay result")
    if str(verification["bindings"]["replay_audit"]["sha256"]) != sha256(replay_audit):
        raise ValueError("verification does not bind replay audit")
    verified = {int(row["call_per_image"]): row for row in verification["new_reporting_rows"]}
    if tuple(verified) != CALLS:
        raise ValueError("verification does not provide calls 2/5/10/50 in order")
    with np.load(replay_result, allow_pickle=False) as saved:
        stored_paths = np.asarray(saved["stored_paths"], dtype=np.float64)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
        history_calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
        values = np.asarray(saved["i_perp"], dtype=np.float64)
    paths = []
    for call in CALLS:
        si = np.flatnonzero(stored_calls == call)
        hi = np.flatnonzero(history_calls == call)
        if si.size != 1 or hi.size != 1:
            raise ValueError(f"replay call {call} absent/nonunique")
        path = np.ascontiguousarray(stored_paths[int(si[0])], dtype=np.float64)
        history_index = int(hi[0])
        nf = int(counters[history_index].sum(dtype=np.int64))
        i_perp = float(values[history_index])
        row = verified[call]
        path_sha = array_sha(path)
        if path_sha != str(row["selected_path_sha256"]):
            raise ValueError(f"call {call} path hash differs from verification")
        if nf != int(row["optimization_N_F"]) or i_perp != float(row["discrete_I_perp_eV2_per_A"]):
            raise ValueError(f"call {call} online history differs from verification")
        if row.get("fully_evaluated_current_path") is not True or row.get("terminal_tail") is not False:
            raise ValueError(f"call {call} is not an eligible current path")
        paths.append({
            "label": f"IVM_call{call}_NF{nf}", "method": "IVM", "host_role": args.host_role,
            "optimization_N_F": nf,
            "path_identifier": {"calls_per_image": call, "history_index": history_index, "fully_evaluated_current_path": True, "terminal_tail": False},
            "source": bind(replay_result), "path_sha256": path_sha,
            "online_discrete_I_perp": i_perp,
            "reparameterization_committed": bool(row["reparameterization_committed"]),
            "force_policy": "online_paid_history_zero_diagnostic_force",
        })
    locks = [args.global_lock_path.expanduser().resolve(), args.peer_global_lock_path.expanduser().resolve()]
    if len(set(locks)) != 2:
        raise ValueError("two distinct lock paths required")
    contract = {
        "schema": "bb-dense-four-ivm-dual-metric-contract-v1", "frozen": True,
        "purpose": "arc500 Delta E for four prospectively retained exact-prefix IVM current paths; reuse online paid I_perp",
        "jobs": 1, "ivm_path_adapter": bind(ADAPTER), "evaluator_engine": bind(ENGINE), "bb_core": bind(BB_CORE), "topology": bind(TOPOLOGY),
        "replay_result": bind(replay_result), "replay_audit": bind(replay_audit), "replay_verification": bind(verification_path),
        "host_local_lock_paths": [str(path) for path in locks], "metric_policy": METRIC_POLICY, "paths": paths,
        "allowed_pass_energy_reuse_bundles": [{"result": bind(result), "audit": bind(audit_path)} for result, audit_path in zip(args.reuse_result, args.reuse_audit)],
        "physical_call_policy": {"diagnostic_force_per_path": 0, "arc500_energy_lineage_per_path": 500, "optimization_N_F_contribution": 0, "jobs": 1, "all_six_thread_caps": 1, "two_host_local_fcntl_locks": True},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    print(json.dumps({"status": "FROZEN", "contract": str(destination), "sha256": sha256(destination), "calls": list(CALLS), "optimization_N_F": [row["optimization_N_F"] for row in paths]}, indent=2))


if __name__ == "__main__":
    main()
