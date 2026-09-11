#!/usr/bin/env python3
"""Select and rank the frozen 2 x 2 x 2 local interaction by both metrics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from evaluate_discrete_iperp_densified_barrier import N_DENSE_POINTS, sha256_file
from select_alpha_from_principal_diagnostics import pareto_nondominated


SETTINGS = (
    (10.0, 0.1, 5), (10.0, 0.1, 8),
    (10.0, 0.7, 5), (10.0, 0.7, 8),
    (1.0, 0.1, 5), (1.0, 0.1, 8),
    (1.0, 0.7, 5), (1.0, 0.7, 8),
)


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
    if audit.get("status") != "PASS" or audit.get("output_sha256") != sha256_file(diagnostics):
        raise ValueError("interaction diagnostics failed their audit")
    if audit.get("contract_sha256") != sha256_file(contract):
        raise ValueError("interaction diagnostic contract mismatch")
    if int(audit.get("diagnostic_force_requests", -1)) != 0 or int(
        audit.get("optimization_N_F_contribution", -1)
    ) != 0:
        raise ValueError("interaction diagnostics contaminate optimization Number of Forces")
    with np.load(diagnostics, allow_pickle=False) as saved:
        labels = np.asarray(saved["label"]).astype(str)
        alpha = np.asarray(saved["alpha"], dtype=np.float64)
        ratio = np.asarray(saved["r"], dtype=np.float64)
        cap = np.asarray(saved["m_max"], dtype=np.int64)
        nf = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        i_perp = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
        delta_e = np.asarray(saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64)
        attempts = np.asarray(saved["diagnostic_energy_attempts_per_path"], dtype=np.int64)
        forces = np.asarray(saved["diagnostic_force_requests_per_path"], dtype=np.int64)
    if np.any(attempts != N_DENSE_POINTS) or np.any(forces != 0):
        raise ValueError("per-path diagnostic ledger mismatch")
    initial = np.flatnonzero(labels == "common_initial_path")
    if initial.size != 1:
        raise ValueError("one common initial path is required")
    i0, e0 = float(i_perp[int(initial[0])]), float(delta_e[int(initial[0])])
    per_state, per_setting = [], []
    for order, (a, r, m) in enumerate(SETTINGS):
        indices = np.flatnonzero(
            np.isclose(alpha, a, rtol=0.0, atol=0.0)
            & np.isclose(ratio, r, rtol=0.0, atol=0.0)
            & (cap == m)
        )
        if not indices.size:
            raise ValueError(f"setting {(a, r, m)} has no eligible state")
        candidates = []
        for index in indices:
            index = int(index)
            row = {
                "alpha": a, "r": r, "m_max": m,
                "label": str(labels[index]),
                "diagnostic_record_index": index,
                "optimization_N_F": int(nf[index]),
                "discrete_I_perp": float(i_perp[index]),
                "densified_Delta_E_kcal_per_mol": float(delta_e[index]),
                "normalized_discrete_I_perp": float(i_perp[index]) / i0,
                "normalized_densified_Delta_E": float(delta_e[index]) / e0,
                "registered_interaction_order": order,
            }
            row["joint_quantity"] = max(
                row["normalized_discrete_I_perp"], row["normalized_densified_Delta_E"]
            )
            candidates.append(row); per_state.append(row)
        per_setting.append(dict(min(candidates, key=lambda row: (
            row["joint_quantity"], row["normalized_discrete_I_perp"],
            row["normalized_densified_Delta_E"], row["optimization_N_F"],
        ))))
    for row, flag in zip(per_setting, pareto_nondominated(per_setting)):
        row["Pareto_nondominated_among_interaction_settings"] = bool(flag)
    ordered = sorted(per_setting, key=lambda row: (
        row["joint_quantity"], row["normalized_discrete_I_perp"],
        row["normalized_densified_Delta_E"], row["optimization_N_F"],
        row["registered_interaction_order"],
    ))
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    payload = {
        "schema": "bb-irivm-local-interaction-selection-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "PASS_LOCAL_INTERACTION_TWO_METRIC_SELECTION",
        "contract": str(contract), "contract_sha256": sha256_file(contract),
        "diagnostics": str(diagnostics), "diagnostics_sha256": sha256_file(diagnostics),
        "diagnostic_audit": str(audit_path), "diagnostic_audit_sha256": sha256_file(audit_path),
        "common_initial_normalization": {"discrete_I_perp": i0, "densified_Delta_E_kcal_per_mol": e0},
        "per_state_results": per_state,
        "per_setting_selected_state": ordered,
        "interaction_winner": {key: ordered[0][key] for key in ("alpha", "r", "m_max")},
        "interaction_runner_up": {key: ordered[1][key] for key in ("alpha", "r", "m_max")},
        "Pareto_nondominated_settings": [
            {key: row[key] for key in ("alpha", "r", "m_max")}
            for row in ordered if row["Pareto_nondominated_among_interaction_settings"]
        ],
        "raw_diagnostics_are_primary": True,
        "same_state_rule_satisfied": True,
        "sparse_barrier_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output.chmod(0o444)
    print(f"Saved {output}")
    print(f"sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
