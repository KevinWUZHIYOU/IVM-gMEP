#!/usr/bin/env python3
"""Select r_star and r_2 from the controlled r scan using both metrics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from evaluate_discrete_iperp_densified_barrier import N_DENSE_POINTS, sha256_file
from select_alpha_from_principal_diagnostics import pareto_nondominated


R_ORDER = (0.1, 0.3, 0.5, 0.7, 0.8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnostics = args.diagnostics.expanduser().resolve(strict=True)
    audit_path = args.audit.expanduser().resolve(strict=True)
    contract = args.contract.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ValueError("r diagnostic audit did not pass")
    if audit.get("output_sha256") != sha256_file(diagnostics):
        raise ValueError("r diagnostic audit/output hash mismatch")
    if audit.get("contract_sha256") != sha256_file(contract):
        raise ValueError("r diagnostic contract hash mismatch")
    if int(audit.get("diagnostic_force_requests", -1)) != 0:
        raise ValueError("r diagnostics requested a force")
    if int(audit.get("optimization_N_F_contribution", -1)) != 0:
        raise ValueError("r diagnostics altered optimization N_F")
    with np.load(diagnostics, allow_pickle=False) as saved:
        labels = np.asarray(saved["label"]).astype(str)
        ratio = np.asarray(saved["r"], dtype=np.float64)
        nf = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        i_perp = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
        delta_e = np.asarray(
            saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64
        )
        attempts = np.asarray(
            saved["diagnostic_energy_attempts_per_path"], dtype=np.int64
        )
        forces = np.asarray(
            saved["diagnostic_force_requests_per_path"], dtype=np.int64
        )
    n_rows = len(labels)
    if any(len(array) != n_rows for array in (ratio, nf, i_perp, delta_e, attempts, forces)):
        raise ValueError("r diagnostic arrays differ in length")
    if np.any(attempts != N_DENSE_POINTS) or np.any(forces != 0):
        raise ValueError("r per-path diagnostic ledger mismatch")
    initial = np.flatnonzero(labels == "common_initial_path")
    if initial.size != 1:
        raise ValueError("r diagnostics require one common initial path")
    initial_index = int(initial[0])
    i0 = float(i_perp[initial_index])
    e0 = float(delta_e[initial_index])
    all_states = []
    per_r = []
    for order, value in enumerate(R_ORDER):
        indices = np.flatnonzero(np.isclose(ratio, value, rtol=0.0, atol=0.0))
        if not indices.size:
            raise ValueError(f"r={value:g} has no eligible diagnostic state")
        candidates = []
        for index in indices:
            index = int(index)
            row = {
                "r": value,
                "label": str(labels[index]),
                "diagnostic_record_index": index,
                "optimization_N_F": int(nf[index]),
                "discrete_I_perp": float(i_perp[index]),
                "densified_Delta_E_kcal_per_mol": float(delta_e[index]),
                "normalized_discrete_I_perp": float(i_perp[index]) / i0,
                "normalized_densified_Delta_E": float(delta_e[index]) / e0,
                "registered_r_order": order,
            }
            row["joint_quantity"] = max(
                row["normalized_discrete_I_perp"],
                row["normalized_densified_Delta_E"],
            )
            candidates.append(row)
            all_states.append(row)
        per_r.append(
            dict(
                min(
                    candidates,
                    key=lambda row: (
                        row["joint_quantity"],
                        row["normalized_discrete_I_perp"],
                        row["normalized_densified_Delta_E"],
                        row["optimization_N_F"],
                    ),
                )
            )
        )
    for row, flag in zip(per_r, pareto_nondominated(per_r)):
        row["Pareto_nondominated_among_r_values"] = bool(flag)
    ordered = sorted(
        per_r,
        key=lambda row: (
            row["joint_quantity"],
            row["normalized_discrete_I_perp"],
            row["normalized_densified_Delta_E"],
            row["optimization_N_F"],
            row["registered_r_order"],
        ),
    )
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    top_ratio = float(ordered[1]["joint_quantity"]) / float(
        ordered[0]["joint_quantity"]
    )
    pareto_incomparable = bool(
        ordered[0]["Pareto_nondominated_among_r_values"]
        and ordered[1]["Pareto_nondominated_among_r_values"]
    )
    extension_required = bool(top_ratio <= 1.10 or pareto_incomparable)
    receipt = {
        "schema": "bb-irivm-r-controlled-scan-selection-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "PASS_R_CONTROLLED_SCAN_TWO_METRIC_SELECTION",
        "contract": str(contract),
        "contract_sha256": sha256_file(contract),
        "diagnostics": str(diagnostics),
        "diagnostics_sha256": sha256_file(diagnostics),
        "diagnostic_audit": str(audit_path),
        "diagnostic_audit_sha256": sha256_file(audit_path),
        "fixed_alpha": 10.0,
        "fixed_m_max": 5,
        "common_initial_normalization": {
            "discrete_I_perp": i0,
            "densified_Delta_E_kcal_per_mol": e0,
        },
        "per_state_results": all_states,
        "per_r_selected_state": ordered,
        "r_star_for_m_max_controlled_scan": float(ordered[0]["r"]),
        "r_2_for_registered_interaction": float(ordered[1]["r"]),
        "top_two_joint_quantity_ratio": top_ratio,
        "top_two_are_Pareto_incomparable": pareto_incomparable,
        "independent_990_N_F_extension_required": extension_required,
        "extension_rule": "required only when top two joint quantities differ by at most 10 percent or are Pareto-incomparable",
        "raw_diagnostics_are_primary": True,
        "same_state_rule_satisfied": True,
        "sparse_barrier_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    output.chmod(0o444)
    print(f"Saved {output}")
    print(f"sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
