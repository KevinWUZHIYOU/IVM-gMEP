#!/usr/bin/env python3
"""Merge hash-bound arc-length-500 diagnostic shards without OpenMM calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from evaluate_discrete_iperp_densified_barrier import (
    N_DENSE_POINTS,
    atomic_save_npz,
    atomic_write_json,
    load_eligible_rows,
    sha256_array,
    sha256_file,
)


ROW_FIELDS = (
    "label",
    "alpha",
    "r",
    "m_max",
    "source_result",
    "source_result_sha256",
    "source_audit",
    "source_audit_sha256",
    "history_index",
    "calls_per_image",
    "optimization_N_F",
    "discrete_I_perp",
    "selected_path_sha256",
    "dense_coordinate_sha256",
    "energy_eV",
    "energy_sha256",
    "densified_Delta_E_kcal_per_mol",
    "total_arclength_A",
    "minimum_original_segment_length_A",
    "maximum_original_segment_length_A",
    "dense_spacing_A",
    "diagnostic_energy_attempts_per_path",
    "diagnostic_energy_successes_per_path",
    "diagnostic_force_requests_per_path",
    "diagnostic_energy_reused_per_path",
    "diagnostic_energy_attempts_this_evaluation_per_path",
    "diagnostic_energy_source",
)


def row_key(arrays: dict[str, np.ndarray], index: int) -> tuple[str, int, str]:
    return (
        str(arrays["label"][index]),
        int(arrays["history_index"][index]),
        str(arrays["selected_path_sha256"][index]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--audits", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.shards) != len(args.audits) or len(args.shards) < 2:
        raise ValueError("provide one audit per diagnostic shard")
    contract = args.contract.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    audit_output = output.with_suffix(".audit.json")
    if output.exists() or audit_output.exists():
        raise FileExistsError("refusing to overwrite merged diagnostics")
    contract_payload = json.loads(contract.read_text(encoding="utf-8"))
    common_path, common_i, common_nf, committed = load_eligible_rows(
        contract, contract_payload
    )
    expected = [
        ("common_initial_path", 0, sha256_array(common_path)),
        *[
            (
                str(row["label"]),
                int(row["history_index"]),
                str(row["path_sha256"]),
            )
            for row in committed
        ],
    ]
    records: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    shard_bindings = []
    scalar_reference = None
    held_locks = []
    for shard_arg, audit_arg in zip(args.shards, args.audits):
        shard = shard_arg.expanduser().resolve(strict=True)
        audit_path = audit_arg.expanduser().resolve(strict=True)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS":
            raise ValueError(f"diagnostic shard audit did not pass: {audit_path}")
        if audit.get("output_sha256") != sha256_file(shard):
            raise ValueError("diagnostic shard audit/output hash mismatch")
        if audit.get("contract_sha256") != sha256_file(contract):
            raise ValueError("diagnostic shard contract mismatch")
        if int(audit.get("diagnostic_force_requests", -1)) != 0:
            raise ValueError("diagnostic shard requested a force")
        with np.load(shard, allow_pickle=False) as saved:
            missing = sorted(set(ROW_FIELDS).difference(saved.files))
            if missing:
                raise ValueError(f"diagnostic shard is missing {missing}")
            arrays = {name: np.asarray(saved[name]) for name in ROW_FIELDS}
            n_rows = len(arrays["label"])
            if any(len(value) != n_rows for value in arrays.values()):
                raise ValueError("diagnostic shard row arrays differ in length")
            scalars = {
                name: np.asarray(saved[name]).copy()
                for name in (
                    "optimization_N_F_contribution",
                    "coordinate_space",
                    "interpolation",
                    "sampling",
                    "number_of_points",
                    "grid_formula",
                    "contract",
                    "contract_sha256",
                    "metric_policy_amendment_sha256",
                    "alpha_scope_R2_sha256",
                    "topology",
                    "topology_sha256",
                )
            }
            held_locks.extend(np.asarray(saved["held_host_locks"]).astype(str).tolist())
        if scalar_reference is None:
            scalar_reference = scalars
        else:
            for name, value in scalars.items():
                if not np.array_equal(value, scalar_reference[name]):
                    raise ValueError(f"diagnostic shard scalar mismatch: {name}")
        for index in range(n_rows):
            key = row_key(arrays, index)
            if key in records:
                raise ValueError(f"duplicate diagnostic row across shards: {key}")
            records[key] = {name: arrays[name][index].copy() for name in ROW_FIELDS}
        shard_bindings.append(
            {
                "output": str(shard),
                "output_sha256": sha256_file(shard),
                "audit": str(audit_path),
                "audit_sha256": sha256_file(audit_path),
            }
        )
    if set(records) != set(expected) or len(records) != len(expected):
        missing = [key for key in expected if key not in records]
        extra = [key for key in records if key not in set(expected)]
        raise ValueError(f"merged shard coverage mismatch; missing={missing}, extra={extra}")
    ordered = [records[key] for key in expected]
    merged = {
        name: np.asarray([row[name] for row in ordered]) for name in ROW_FIELDS
    }
    if merged["energy_eV"].shape != (len(expected), N_DENSE_POINTS):
        raise ValueError("merged energy array has the wrong shape")
    if np.any(merged["diagnostic_energy_attempts_per_path"] != N_DENSE_POINTS):
        raise ValueError("merged path lacks a 500-energy lineage")
    if np.any(merged["diagnostic_energy_successes_per_path"] != N_DENSE_POINTS):
        raise ValueError("merged path lacks 500 successful energies")
    if np.any(merged["diagnostic_force_requests_per_path"] != 0):
        raise ValueError("merged diagnostics contain a force request")
    assert scalar_reference is not None
    atomic_save_npz(
        output,
        **merged,
        **scalar_reference,
        held_host_locks=np.asarray(sorted(set(held_locks))),
        diagnostic_shard_outputs=np.asarray([item["output"] for item in shard_bindings]),
        diagnostic_shard_output_sha256=np.asarray(
            [item["output_sha256"] for item in shard_bindings]
        ),
        diagnostic_shard_audits=np.asarray([item["audit"] for item in shard_bindings]),
        diagnostic_shard_audit_sha256=np.asarray(
            [item["audit_sha256"] for item in shard_bindings]
        ),
    )
    current_attempts = int(
        np.sum(
            merged["diagnostic_energy_attempts_this_evaluation_per_path"],
            dtype=np.int64,
        )
    )
    audit = {
        "schema": "bb-irivm-merged-diagnostic-shards-audit-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "PASS",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "contract": str(contract),
        "contract_sha256": sha256_file(contract),
        "source_shards": shard_bindings,
        "selected_path_count_including_common_initial": len(expected),
        "diagnostic_energy_attempts": N_DENSE_POINTS * len(expected),
        "diagnostic_energy_successes": N_DENSE_POINTS * len(expected),
        "diagnostic_energy_attempts_this_evaluation": current_attempts,
        "diagnostic_energy_successes_this_evaluation": current_attempts,
        "diagnostic_force_requests": 0,
        "optimization_N_F_contribution": 0,
        "exact_reused_path_record_count": int(
            np.count_nonzero(merged["diagnostic_energy_reused_per_path"])
        ),
        "freshly_evaluated_path_record_count": int(
            np.count_nonzero(~merged["diagnostic_energy_reused_per_path"])
        ),
        "every_path_is_bound_to_exactly_500_successful_energy_evaluations": True,
        "same_state_row_coverage_complete": True,
        "records": [
            {
                "label": str(row["label"]),
                "history_index": int(row["history_index"]),
                "optimization_N_F": int(row["optimization_N_F"]),
                "discrete_I_perp": float(row["discrete_I_perp"]),
                "selected_path_sha256": str(row["selected_path_sha256"]),
                "dense_coordinate_sha256": str(row["dense_coordinate_sha256"]),
                "energy_sha256": str(row["energy_sha256"]),
                "densified_Delta_E_kcal_per_mol": float(
                    row["densified_Delta_E_kcal_per_mol"]
                ),
                "diagnostic_energy_reused": bool(
                    row["diagnostic_energy_reused_per_path"]
                ),
            }
            for row in ordered
        ],
    }
    atomic_write_json(audit_output, audit)
    print(f"Saved {output}")
    print(f"Saved {audit_output}")


if __name__ == "__main__":
    main()
