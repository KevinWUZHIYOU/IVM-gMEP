#!/usr/bin/env python3
"""Select alpha from the two complementary principal diagnostics.

The raw quantities are the discrete unnormalized normal residual ``I_perp``
and the densified 0 K forward barrier ``Delta E``.  The dimensionless joint
quantity is used only to make the registered promotion decision; both raw
diagnostics and their Pareto relation remain primary in the report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from evaluate_discrete_iperp_densified_barrier import (
    ALPHA_SCOPE_R2_SHA256,
    METRIC_AMENDMENT_SHA256,
    N_DENSE_POINTS,
    sha256_file,
)


ALPHA_ORDER = (0.1, 1.0, 10.0, 100.0, 1000.0)


def pareto_nondominated(records: Sequence[Dict[str, object]]) -> List[bool]:
    flags = []
    for index, record in enumerate(records):
        dominated = False
        for other_index, other in enumerate(records):
            if other_index == index:
                continue
            i_no_worse = float(other["discrete_I_perp"]) <= float(record["discrete_I_perp"])
            e_no_worse = float(other["densified_Delta_E_kcal_per_mol"]) <= float(
                record["densified_Delta_E_kcal_per_mol"]
            )
            one_strict = (
                float(other["discrete_I_perp"]) < float(record["discrete_I_perp"])
                or float(other["densified_Delta_E_kcal_per_mol"])
                < float(record["densified_Delta_E_kcal_per_mol"])
            )
            if i_no_worse and e_no_worse and one_strict:
                dominated = True
                break
        flags.append(not dominated)
    return flags


def select_alpha_records(
    *,
    labels: np.ndarray,
    alpha: np.ndarray,
    optimization_nf: np.ndarray,
    discrete_i_perp: np.ndarray,
    densified_delta_e: np.ndarray,
) -> Tuple[Dict[str, float], List[Dict[str, object]], List[Dict[str, object]]]:
    initial = np.flatnonzero(labels == "common_initial_path")
    if initial.size != 1:
        raise ValueError("diagnostic output must contain one common initial path")
    initial_index = int(initial[0])
    i0 = float(discrete_i_perp[initial_index])
    e0 = float(densified_delta_e[initial_index])
    if not np.isfinite(i0) or not np.isfinite(e0) or i0 <= 0.0 or e0 <= 0.0:
        raise ValueError("common-initial normalization must be positive and finite")

    all_states: List[Dict[str, object]] = []
    selected: List[Dict[str, object]] = []
    for order_index, value in enumerate(ALPHA_ORDER):
        indices = np.flatnonzero(np.isclose(alpha, value, rtol=0.0, atol=0.0))
        if not indices.size:
            raise ValueError(f"alpha={value:g} has no densified eligible state")
        candidate_states = []
        for index in indices:
            index = int(index)
            u_i = float(discrete_i_perp[index]) / i0
            u_e = float(densified_delta_e[index]) / e0
            record: Dict[str, object] = {
                "alpha": float(value),
                "label": str(labels[index]),
                "diagnostic_record_index": index,
                "optimization_N_F": int(optimization_nf[index]),
                "discrete_I_perp": float(discrete_i_perp[index]),
                "densified_Delta_E_kcal_per_mol": float(densified_delta_e[index]),
                "normalized_discrete_I_perp": u_i,
                "normalized_densified_Delta_E": u_e,
                "joint_quantity": max(u_i, u_e),
                "registered_alpha_order": order_index,
            }
            candidate_states.append(record)
            all_states.append(record)
        best = min(
            candidate_states,
            key=lambda row: (
                float(row["joint_quantity"]),
                float(row["normalized_discrete_I_perp"]),
                float(row["normalized_densified_Delta_E"]),
                int(row["optimization_N_F"]),
            ),
        )
        selected.append(dict(best))
    pareto = pareto_nondominated(selected)
    for record, flag in zip(selected, pareto):
        record["Pareto_nondominated_among_alpha_values"] = bool(flag)
    ordered = sorted(
        selected,
        key=lambda row: (
            float(row["joint_quantity"]),
            float(row["normalized_discrete_I_perp"]),
            float(row["normalized_densified_Delta_E"]),
            int(row["optimization_N_F"]),
            int(row["registered_alpha_order"]),
        ),
    )
    for rank, record in enumerate(ordered, start=1):
        record["promotion_order"] = rank
    return {"discrete_I_perp": i0, "densified_Delta_E_kcal_per_mol": e0}, all_states, ordered


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--diagnostic-audit", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    diagnostics = args.diagnostics.expanduser().resolve()
    audit_path = args.diagnostic_audit.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ValueError("densified diagnostic audit did not pass")
    if audit.get("output_sha256") != sha256_file(diagnostics):
        raise ValueError("densified diagnostic audit/output hash mismatch")
    if audit.get("contract_sha256") != sha256_file(contract_path):
        raise ValueError("densified diagnostic contract hash mismatch")
    if audit.get("metric_policy_amendment_sha256") != METRIC_AMENDMENT_SHA256:
        raise ValueError("metric-policy amendment hash mismatch")
    if audit.get("alpha_scope_R2_sha256") != ALPHA_SCOPE_R2_SHA256:
        raise ValueError("alpha-scope R2 hash mismatch")
    if int(audit.get("diagnostic_force_requests", -1)) != 0:
        raise ValueError("densified diagnostic requested a force")
    if int(audit.get("optimization_N_F_contribution", -1)) != 0:
        raise ValueError("densified diagnostic was added to optimization N_F")

    with np.load(diagnostics, allow_pickle=False) as saved:
        required = {
            "label",
            "alpha",
            "optimization_N_F",
            "discrete_I_perp",
            "densified_Delta_E_kcal_per_mol",
            "diagnostic_energy_attempts_per_path",
            "diagnostic_force_requests_per_path",
            "number_of_points",
            "metric_policy_amendment_sha256",
            "alpha_scope_R2_sha256",
        }
        missing = sorted(required.difference(saved.files))
        if missing:
            raise ValueError(f"diagnostic output is missing {missing}")
        labels = np.asarray(saved["label"]).astype(str)
        alpha = np.asarray(saved["alpha"], dtype=np.float64)
        optimization_nf = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        discrete_i_perp = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
        densified_delta_e = np.asarray(
            saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64
        )
        attempts = np.asarray(saved["diagnostic_energy_attempts_per_path"], dtype=np.int64)
        force_requests = np.asarray(saved["diagnostic_force_requests_per_path"], dtype=np.int64)
        if int(np.asarray(saved["number_of_points"]).item()) != N_DENSE_POINTS:
            raise ValueError("diagnostic output does not use 500 points")
        if str(np.asarray(saved["metric_policy_amendment_sha256"]).item()) != METRIC_AMENDMENT_SHA256:
            raise ValueError("diagnostic metric-policy binding mismatch")
        if str(np.asarray(saved["alpha_scope_R2_sha256"]).item()) != ALPHA_SCOPE_R2_SHA256:
            raise ValueError("diagnostic alpha-scope binding mismatch")
    n = len(labels)
    if not all(len(array) == n for array in (alpha, optimization_nf, discrete_i_perp, densified_delta_e, attempts, force_requests)):
        raise ValueError("diagnostic arrays differ in length")
    if np.any(attempts != N_DENSE_POINTS) or np.any(force_requests != 0):
        raise ValueError("diagnostic per-path ledger is inconsistent")
    if not np.all(np.isfinite(discrete_i_perp)) or not np.all(np.isfinite(densified_delta_e)):
        raise ValueError("principal diagnostics contain non-finite values")

    normalization, all_states, ordered = select_alpha_records(
        labels=labels,
        alpha=alpha,
        optimization_nf=optimization_nf,
        discrete_i_perp=discrete_i_perp,
        densified_delta_e=densified_delta_e,
    )
    receipt = {
        "schema": "bb-irivm-alpha-principal-diagnostic-selection-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "PASS_ALPHA_SELECTION_FROM_TWO_PRINCIPAL_DIAGNOSTICS",
        "scientific_terms": {
            "stationarity_diagnostic": "discrete unnormalized normal residual I_perp",
            "energetic_diagnostic": "densified 0 K forward barrier Delta E",
            "cost_axis": "optimization Number of Forces N_F",
        },
        "diagnostics": str(diagnostics),
        "diagnostics_sha256": sha256_file(diagnostics),
        "diagnostic_audit": str(audit_path),
        "diagnostic_audit_sha256": sha256_file(audit_path),
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "metric_policy_amendment_sha256": METRIC_AMENDMENT_SHA256,
        "alpha_scope_R2_sha256": ALPHA_SCOPE_R2_SHA256,
        "common_initial_normalization": normalization,
        "per_state_results": all_states,
        "per_alpha_selected_state_in_promotion_order": ordered,
        "promoted_alpha_values_for_independent_990_N_F_runs": [
            float(record["alpha"]) for record in ordered[:3]
        ],
        "raw_diagnostics_are_primary": True,
        "sparse_barrier_used": False,
        "same_state_rule_satisfied": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    output.chmod(0o444)
    print(f"Saved {output}")
    print(f"sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
