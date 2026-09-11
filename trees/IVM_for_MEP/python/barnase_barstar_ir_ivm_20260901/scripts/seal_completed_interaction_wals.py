#!/usr/bin/env python3
"""Seal only complete unique paths from interrupted interaction WALs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from evaluate_discrete_iperp_densified_barrier import (
    COORDINATE_SPACE, GRID_FORMULA, INTERPOLATION, N_DENSE_POINTS, SAMPLING,
    _validate_contract, atomic_save_npz, atomic_write_json,
    load_eligible_rows, load_exact_reuse_rows, resolve_path, sha256_array,
    sha256_file, uniform_arclength_resample,
)


def assigned_rows(contract_path: Path, contract: dict, reuse: dict, shard: int, count: int):
    common, common_i, common_nf, committed = load_eligible_rows(contract_path, contract)
    rows = [{
        "label": "common_initial_path", "path": common,
        "path_sha256": sha256_array(common), "optimization_N_F": int(common_nf),
        "discrete_I_perp": float(common_i),
    }, *committed]
    fresh_hashes = []
    for row in rows:
        digest = str(row["path_sha256"])
        if digest not in reuse and digest not in fresh_hashes:
            fresh_hashes.append(digest)
    assigned = {digest for index, digest in enumerate(fresh_hashes) if index % count == shard}
    return [row for row in rows if str(row["path_sha256"]) in assigned or (shard == 0 and str(row["path_sha256"]) in reuse)]


def read_complete_rows(wal: Path, rows: list[dict]):
    lines = wal.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    if int(header["number_of_points_per_path"]) != N_DENSE_POINTS or int(header["force_requests"]) != 0:
        raise ValueError("interrupted WAL protocol mismatch")
    points: dict[int, dict[int, float]] = {}
    failed = 0
    for line in lines[1:]:
        record = json.loads(line)
        if record.get("status") == "failed":
            failed += 1
        if record.get("status") != "success":
            continue
        row = int(record["row_index"]); point = int(record["point_index"])
        if row >= len(rows) or point in points.setdefault(row, {}):
            raise ValueError("interrupted WAL row/point mismatch")
        points[row][point] = float(record["energy_eV"])
    if failed:
        raise ValueError("interrupted WAL contains a failed energy attempt")
    complete, partial = [], []
    expected = set(range(N_DENSE_POINTS))
    for row_index, values in points.items():
        if set(values) == expected:
            complete.append((rows[row_index], np.asarray([values[i] for i in range(N_DENSE_POINTS)], dtype=np.float64), row_index))
        else:
            partial.append({"row_index": row_index, "successful_points": len(values), "path_sha256": str(rows[row_index]["path_sha256"])})
    return complete, partial, sum(len(values) for values in points.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--base-diagnostics", type=Path, required=True)
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--wal", type=Path, action="append", required=True)
    parser.add_argument("--shard-index", type=int, action="append", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.wal) != len(args.shard_index):
        raise ValueError("provide one shard index per WAL")
    contract_path = args.contract.expanduser().resolve(strict=True)
    contract = _validate_contract(contract_path)
    topology = resolve_path(contract_path, str(contract["topology"]["path"]))
    topology_sha = sha256_file(topology)
    base = args.base_diagnostics.expanduser().resolve(strict=True)
    base_audit = args.base_audit.expanduser().resolve(strict=True)
    reusable = load_exact_reuse_rows(base, base_audit, topology_sha)
    recovered = dict(reusable)
    wal_bindings, partial_rows = [], []
    interrupted_successes = 0
    newly_sealed = 0
    for wal_arg, shard in zip(args.wal, args.shard_index):
        wal = wal_arg.expanduser().resolve(strict=True)
        rows = assigned_rows(contract_path, contract, reusable, int(shard), int(args.shard_count))
        complete, partial, successes = read_complete_rows(wal, rows)
        interrupted_successes += successes
        for row, energies, row_index in complete:
            digest = str(row["path_sha256"])
            dense, _ = uniform_arclength_resample(row["path"])
            record = {
                "dense_coordinate_sha256": sha256_array(dense),
                "energy_eV": energies,
                "energy_sha256": sha256_array(energies),
                "source_diagnostic": str(wal),
                "source_diagnostic_sha256": sha256_file(wal),
                "source_row_index": row_index,
            }
            previous = recovered.get(digest)
            if previous is not None:
                if previous["dense_coordinate_sha256"] != record["dense_coordinate_sha256"] or previous["energy_sha256"] != record["energy_sha256"]:
                    raise ValueError("complete WAL conflicts with an exact reused path")
            else:
                recovered[digest] = record; newly_sealed += 1
        partial_rows.extend([{**row, "wal": str(wal)} for row in partial])
        wal_bindings.append({"path": str(wal), "sha256": sha256_file(wal), "shard_index": int(shard), "successful_energy_attempts": successes})
    output = args.output.expanduser().resolve()
    audit_output = output.with_suffix(".audit.json")
    if output.exists() or audit_output.exists():
        raise FileExistsError("refusing to overwrite recovery bundle")
    items = sorted(recovered.items())
    atomic_save_npz(
        output,
        selected_path_sha256=np.asarray([key for key, _ in items]),
        dense_coordinate_sha256=np.asarray([row["dense_coordinate_sha256"] for _, row in items]),
        energy_eV=np.asarray([row["energy_eV"] for _, row in items], dtype=np.float64),
        energy_sha256=np.asarray([row["energy_sha256"] for _, row in items]),
        diagnostic_energy_attempts_per_path=np.full(len(items), N_DENSE_POINTS, dtype=np.int64),
        diagnostic_energy_successes_per_path=np.full(len(items), N_DENSE_POINTS, dtype=np.int64),
        diagnostic_force_requests_per_path=np.zeros(len(items), dtype=np.int64),
        coordinate_space=np.asarray(COORDINATE_SPACE), interpolation=np.asarray(INTERPOLATION),
        sampling=np.asarray(SAMPLING), number_of_points=np.asarray(N_DENSE_POINTS, dtype=np.int64),
        grid_formula=np.asarray(GRID_FORMULA), topology_sha256=np.asarray(topology_sha),
    )
    audit = {
        "schema": "bb-irivm-interrupted-wal-complete-path-reuse-bundle-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "PASS", "output": str(output), "output_sha256": sha256_file(output),
        "diagnostic_force_requests": 0, "optimization_N_F_contribution": 0,
        "base_diagnostics": str(base), "base_diagnostics_sha256": sha256_file(base),
        "base_audit": str(base_audit), "base_audit_sha256": sha256_file(base_audit),
        "interrupted_wals": wal_bindings,
        "interrupted_successful_energy_attempts": interrupted_successes,
        "newly_sealed_complete_unique_path_count": newly_sealed,
        "partial_duplicate_rows_excluded": partial_rows,
        "partial_duplicate_successful_energy_attempts_excluded": sum(row["successful_points"] for row in partial_rows),
        "sealed_unique_path_count_including_base": len(items),
        "every_included_path_has_exactly_500_successful_energy_evaluations": True,
    }
    atomic_write_json(audit_output, audit)
    print(f"Saved {output}"); print(f"Saved {audit_output}")


if __name__ == "__main__":
    main()
