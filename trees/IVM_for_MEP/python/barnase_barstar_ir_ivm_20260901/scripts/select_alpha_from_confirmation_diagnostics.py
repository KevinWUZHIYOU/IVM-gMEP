#!/usr/bin/env python3
"""Select ``alpha_star`` and ``alpha_2`` from the 990-N_F confirmations."""

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
from select_alpha_from_principal_diagnostics import pareto_nondominated


ALPHA_CONFIRMATION_ORDER = (1.0, 0.1, 10.0)


def select_confirmation_records(
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
    per_alpha: List[Dict[str, object]] = []
    for registered_order, value in enumerate(ALPHA_CONFIRMATION_ORDER):
        indices = np.flatnonzero(np.isclose(alpha, value, rtol=0.0, atol=0.0))
        if not indices.size:
            raise ValueError(f"alpha={value:g} has no eligible densified state")
        candidates = []
        for index in indices:
            index = int(index)
            u_i = float(discrete_i_perp[index]) / i0
            u_e = float(densified_delta_e[index]) / e0
            row: Dict[str, object] = {
                "alpha": float(value),
                "label": str(labels[index]),
                "diagnostic_record_index": index,
                "optimization_N_F": int(optimization_nf[index]),
                "discrete_I_perp": float(discrete_i_perp[index]),
                "densified_Delta_E_kcal_per_mol": float(densified_delta_e[index]),
                "normalized_discrete_I_perp": u_i,
                "normalized_densified_Delta_E": u_e,
                "joint_quantity": max(u_i, u_e),
                "registered_confirmation_order": registered_order,
            }
            candidates.append(row)
            all_states.append(row)
        per_alpha.append(
            dict(
                min(
                    candidates,
                    key=lambda row: (
                        float(row["joint_quantity"]),
                        float(row["normalized_discrete_I_perp"]),
                        float(row["normalized_densified_Delta_E"]),
                        int(row["optimization_N_F"]),
                    ),
                )
            )
        )

    for row, flag in zip(per_alpha, pareto_nondominated(per_alpha)):
        row["Pareto_nondominated_among_confirmed_alpha_values"] = bool(flag)
    ordered = sorted(
        per_alpha,
        key=lambda row: (
            float(row["joint_quantity"]),
            float(row["normalized_discrete_I_perp"]),
            float(row["normalized_densified_Delta_E"]),
            int(row["optimization_N_F"]),
            int(row["registered_confirmation_order"]),
        ),
    )
    for rank, row in enumerate(ordered, start=1):
        row["confirmation_rank"] = rank
    return {
        "discrete_I_perp": i0,
        "densified_Delta_E_kcal_per_mol": e0,
    }, all_states, ordered


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--diagnostic-audit", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--principal-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    diagnostics = args.diagnostics.expanduser().resolve()
    audit_path = args.diagnostic_audit.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    principal_path = args.principal_selection.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    principal = json.loads(principal_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ValueError("alpha-confirmation diagnostic audit did not pass")
    if audit.get("output_sha256") != sha256_file(diagnostics):
        raise ValueError("diagnostic audit/output hash mismatch")
    if audit.get("contract_sha256") != sha256_file(contract_path):
        raise ValueError("diagnostic contract hash mismatch")
    if audit.get("metric_policy_amendment_sha256") != METRIC_AMENDMENT_SHA256:
        raise ValueError("diagnostic metric-policy binding mismatch")
    if audit.get("alpha_scope_R2_sha256") != ALPHA_SCOPE_R2_SHA256:
        raise ValueError("diagnostic alpha-scope binding mismatch")
    if int(audit.get("diagnostic_force_requests", -1)) != 0:
        raise ValueError("confirmation diagnostics requested a force")
    if int(audit.get("optimization_N_F_contribution", -1)) != 0:
        raise ValueError("confirmation diagnostics altered optimization N_F")
    principal_binding = contract.get("bindings", {}).get("principal_alpha_selection", {})
    if principal_binding.get("sha256") != sha256_file(principal_path):
        raise ValueError("confirmation contract/principal-selection hash mismatch")
    promoted = tuple(
        float(value)
        for value in principal.get(
            "promoted_alpha_values_for_independent_990_N_F_runs", []
        )
    )
    if promoted != ALPHA_CONFIRMATION_ORDER:
        raise ValueError("principal selection does not match confirmation order")

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
            raise ValueError("confirmation diagnostics do not use 500 points")
    arrays: Sequence[np.ndarray] = (
        alpha,
        optimization_nf,
        discrete_i_perp,
        densified_delta_e,
        attempts,
        force_requests,
    )
    if any(len(array) != len(labels) for array in arrays):
        raise ValueError("confirmation diagnostic arrays differ in length")
    if np.any(attempts != N_DENSE_POINTS) or np.any(force_requests != 0):
        raise ValueError("confirmation per-path diagnostic ledger is inconsistent")
    if not np.all(np.isfinite(discrete_i_perp)) or not np.all(np.isfinite(densified_delta_e)):
        raise ValueError("confirmation diagnostics contain non-finite values")

    normalization, all_states, ordered = select_confirmation_records(
        labels=labels,
        alpha=alpha,
        optimization_nf=optimization_nf,
        discrete_i_perp=discrete_i_perp,
        densified_delta_e=densified_delta_e,
    )
    receipt = {
        "schema": "bb-irivm-alpha-990-nf-confirmation-selection-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "PASS_ALPHA_CONFIRMATION_TWO_METRIC_SELECTION",
        "metric_policy_amendment_sha256": METRIC_AMENDMENT_SHA256,
        "alpha_scope_R2_sha256": ALPHA_SCOPE_R2_SHA256,
        "principal_selection": str(principal_path),
        "principal_selection_sha256": sha256_file(principal_path),
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "diagnostics": str(diagnostics),
        "diagnostics_sha256": sha256_file(diagnostics),
        "diagnostic_audit": str(audit_path),
        "diagnostic_audit_sha256": sha256_file(audit_path),
        "common_initial_normalization": normalization,
        "per_state_results": all_states,
        "per_alpha_selected_state_in_confirmation_order": ordered,
        "alpha_star_for_r_controlled_scan": float(ordered[0]["alpha"]),
        "alpha_2_for_registered_interaction": float(ordered[1]["alpha"]),
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
