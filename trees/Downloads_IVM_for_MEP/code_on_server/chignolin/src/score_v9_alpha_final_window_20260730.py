#!/usr/bin/env python3
"""Score the audited V9 grid with the frozen final-window contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


REFERENCE_BARRIER_KCAL = 20.373739515742795
REFERENCE_FINAL_I_PERP = 0.03885242073674519
REFERENCE_ENDPOINT_BARRIER_ERROR_KCAL = 0.798519886390451
EQUIDISTANCE_NONINFERIORITY_FACTOR = 1.20
SELECTED_ALPHA_TARGET = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometric_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Geometric-mean values must be positive and finite")
    return float(np.exp(np.mean(np.log(values))))


def strict_single_well(values, center) -> bool:
    values = np.asarray(values, dtype=float)
    return bool(
        np.all(np.diff(values[: center + 1]) < 0.0)
        and np.all(np.diff(values[center:]) > 0.0)
    )


def moderate_unique_interior_well(values, alphas) -> bool:
    values = np.asarray(values, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    local_minima = [
        index
        for index in range(1, len(values) - 1)
        if values[index] < values[index - 1]
        and values[index] < values[index + 1]
    ]
    global_index = int(np.argmin(values))
    return bool(
        len(local_minima) == 1
        and local_minima[0] == global_index
        and 0.75 <= alphas[global_index] <= 1.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-summary", type=Path, required=True)
    parser.add_argument("--dense-manifest", type=Path, required=True)
    parser.add_argument("--method-summary", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    grid_path = args.grid_summary.expanduser().resolve()
    dense_path = args.dense_manifest.expanduser().resolve()
    method_path = args.method_summary.expanduser().resolve()
    contract_path = args.analysis_contract.expanduser().resolve()
    grid = json.loads(grid_path.read_text())
    dense = json.loads(dense_path.read_text())
    methods = json.loads(method_path.read_text())["methods"]
    budgets = tuple(
        int(value) for value in grid["budgets_total_number_of_forces"]
    )
    dense_by_alpha = {
        float(record["alpha"]): record for record in dense["records"]
    }
    ode_baseline_eq = float(
        methods["ODE-NEB"]["final_equidistance_integral"]
    )
    obsm_baseline_eq = float(
        methods["OBSM-FIRE"]["final_equidistance_integral"]
    )
    better_baseline_i = min(
        float(methods["ODE-NEB"]["final_i_perp"]),
        float(methods["OBSM-FIRE"]["final_i_perp"]),
    )
    better_baseline_barrier = min(
        float(methods["ODE-NEB"]["final_dense_barrier_kcal"]),
        float(methods["OBSM-FIRE"]["final_dense_barrier_kcal"]),
    )

    rows = []
    detailed = []
    for record in sorted(
        grid["records"], key=lambda item: float(item["alpha"])
    ):
        alpha = float(record["alpha"])
        dense_record = dense_by_alpha[alpha]
        result = Path(record["result"]).resolve()
        dense_result = Path(dense_record["dense_result"]).resolve()
        if (
            sha256(result) != record["result_sha256"]
            or sha256(dense_result)
            != dense_record["dense_result_sha256"]
        ):
            raise RuntimeError(f"Source changed for alpha={alpha:g}")
        budget_records = record["budget_records"]
        if tuple(
            int(item["requested_total_number_of_forces"])
            for item in budget_records
        ) != budgets:
            raise RuntimeError(f"Budget mismatch for alpha={alpha:g}")
        anytime_i = np.asarray(
            [
                item["anytime_integral_mean_unnormalized_i_perp"]
                for item in budget_records
            ],
            dtype=float,
        )
        trailing_i = np.asarray(
            [
                item["trailing_integral_mean_unnormalized_i_perp"]
                for item in budget_records
            ],
            dtype=float,
        )
        endpoint_i = np.asarray(
            [item["endpoint_unnormalized_i_perp"] for item in budget_records],
            dtype=float,
        )
        endpoint_eq = np.asarray(
            [item["endpoint_equidistance_integral"] for item in budget_records],
            dtype=float,
        )
        anytime_dense = np.asarray(
            dense_record["anytime_rms_barrier_error_kcal_by_budget"],
            dtype=float,
        )
        trailing_dense = np.asarray(
            dense_record["trailing_rms_barrier_error_kcal_by_budget"],
            dtype=float,
        )
        endpoint_barrier = np.asarray(
            dense_record["endpoint_dense_barrier_kcal_by_budget"],
            dtype=float,
        )
        final_endpoint_error = abs(
            float(endpoint_barrier[-1]) - REFERENCE_BARRIER_KCAL
        )
        gates = {
            "prior_cap15_i_perp": bool(
                endpoint_i[-1] <= REFERENCE_FINAL_I_PERP
            ),
            "prior_cap15_dense_endpoint_error": bool(
                final_endpoint_error
                <= REFERENCE_ENDPOINT_BARRIER_ERROR_KCAL
            ),
            "ode_neb_equidistance_noninferiority": bool(
                endpoint_eq[-1]
                <= EQUIDISTANCE_NONINFERIORITY_FACTOR * ode_baseline_eq
            ),
            "ode_obsm_i_perp": bool(endpoint_i[-1] <= better_baseline_i),
            "ode_obsm_dense_barrier": bool(
                endpoint_barrier[-1] <= better_baseline_barrier
            ),
        }
        eligible = bool(all(gates.values()))
        row = {
            "alpha": alpha,
            "production_eligible": eligible,
            "trailing_stationarity_score_geometric_mean": geometric_mean(
                trailing_i
            ),
            "trailing_dense_barrier_score_geometric_mean_kcal": (
                geometric_mean(trailing_dense)
            ),
            "endpoint_i_perp_nf_le_10000": float(endpoint_i[-1]),
            "endpoint_dense_barrier_kcal_nf_le_10000": float(
                endpoint_barrier[-1]
            ),
            "endpoint_dense_barrier_error_kcal_nf_le_10000": (
                final_endpoint_error
            ),
            "endpoint_equidistance_integral_nf_le_10000": float(
                endpoint_eq[-1]
            ),
            "endpoint_total_number_of_forces_nf_le_10000": int(
                budget_records[-1][
                    "attained_complete_total_number_of_forces"
                ]
            ),
            "endpoint_max_call_per_internal_image_nf_le_10000": int(
                budget_records[-1]["source_max_call_per_internal_image"]
            ),
            "optimization_result": str(result),
            "optimization_result_sha256": record["result_sha256"],
            "optimization_audit": record["audit"],
            "optimization_audit_sha256": record["audit_sha256"],
            "dense_result": str(dense_result),
            "dense_result_sha256": dense_record[
                "dense_result_sha256"
            ],
            "dense_audit": dense_record["dense_audit"],
            "dense_audit_sha256": dense_record["dense_audit_sha256"],
            **{f"gate_{key}": value for key, value in gates.items()},
        }
        for index, budget in enumerate(budgets):
            row[f"i_perp_anytime_mean_nf_le_{budget}"] = float(
                anytime_i[index]
            )
            row[f"i_perp_trailing_mean_nf_le_{budget}"] = float(
                trailing_i[index]
            )
            row[f"i_perp_endpoint_nf_le_{budget}"] = float(
                endpoint_i[index]
            )
            row[f"dense_barrier_anytime_rms_error_nf_le_{budget}"] = (
                float(anytime_dense[index])
            )
            row[f"dense_barrier_trailing_rms_error_nf_le_{budget}"] = (
                float(trailing_dense[index])
            )
            row[f"dense_barrier_endpoint_nf_le_{budget}"] = float(
                endpoint_barrier[index]
            )
        rows.append(row)
        detailed.append(
            {
                **row,
                "optimization_budget_records": budget_records,
                "dense_checkpoints": dense_record["checkpoints"],
            }
        )

    eligible = [row for row in rows if row["production_eligible"]]
    selected = (
        min(
            eligible,
            key=lambda row: (
                row["trailing_stationarity_score_geometric_mean"],
                row[
                    "trailing_dense_barrier_score_geometric_mean_kcal"
                ],
                row["endpoint_equidistance_integral_nf_le_10000"],
                row["endpoint_total_number_of_forces_nf_le_10000"],
            ),
        )
        if eligible
        else None
    )
    alphas = np.asarray([row["alpha"] for row in rows], dtype=float)
    center_candidates = np.flatnonzero(
        np.isclose(alphas, SELECTED_ALPHA_TARGET)
    )
    if center_candidates.size != 1:
        raise RuntimeError("Alpha=1 is absent or duplicated")
    center = int(center_candidates[0])
    well = {}
    for budget in budgets:
        i_values = [
            row[f"i_perp_trailing_mean_nf_le_{budget}"] for row in rows
        ]
        barrier_values = [
            row[f"dense_barrier_trailing_rms_error_nf_le_{budget}"]
            for row in rows
        ]
        well[str(budget)] = {
            "i_perp_values": i_values,
            "dense_barrier_values": barrier_values,
            "i_perp_strict_single_well_at_alpha_1": strict_single_well(
                i_values, center
            ),
            "i_perp_unique_interior_well_at_alpha_1": bool(
                moderate_unique_interior_well(i_values, alphas)
                and np.isclose(
                    alphas[int(np.argmin(i_values))],
                    SELECTED_ALPHA_TARGET,
                )
            ),
            "dense_barrier_strict_single_well_at_alpha_1": (
                strict_single_well(barrier_values, center)
            ),
            "dense_barrier_unique_moderate_interior_well": (
                moderate_unique_interior_well(barrier_values, alphas)
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "V9_ALPHA_FINAL_WINDOW_SCORES.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "source_grid_summary": str(grid_path),
        "source_grid_summary_sha256": sha256(grid_path),
        "source_dense_manifest": str(dense_path),
        "source_dense_manifest_sha256": sha256(dense_path),
        "source_method_summary": str(method_path),
        "source_method_summary_sha256": sha256(method_path),
        "analysis_contract": str(contract_path),
        "analysis_contract_sha256": sha256(contract_path),
        "selection_rule": (
            "Eligible candidates are ranked lexicographically by the "
            "across-budget final-window stationarity score, across-budget real-"
            "OpenMM final-window dense-barrier score, endpoint equidistance, "
            "and N_F."
        ),
        "sparse_22_image_barrier_used": False,
        "diagnostic_calls_excluded_from_optimization_axis": True,
        "reference_values": {
            "barrier_kcal": REFERENCE_BARRIER_KCAL,
            "prior_cap15_i_perp": REFERENCE_FINAL_I_PERP,
            "prior_cap15_endpoint_barrier_error_gate_kcal": (
                REFERENCE_ENDPOINT_BARRIER_ERROR_KCAL
            ),
            "equidistance_noninferiority_factor": (
                EQUIDISTANCE_NONINFERIORITY_FACTOR
            ),
            "ode_neb_equidistance_integral": ode_baseline_eq,
            "obsm_fire_equidistance_integral": obsm_baseline_eq,
            "selected_to_ode_neb_equidistance_ratio": (
                None
                if selected is None
                else selected[
                    "endpoint_equidistance_integral_nf_le_10000"
                ]
                / ode_baseline_eq
            ),
            "selected_to_obsm_fire_equidistance_ratio": (
                None
                if selected is None
                else selected[
                    "endpoint_equidistance_integral_nf_le_10000"
                ]
                / obsm_baseline_eq
            ),
            "better_ode_obsm_i_perp": better_baseline_i,
            "better_ode_obsm_dense_barrier_kcal": better_baseline_barrier,
        },
        "selected_alpha": None if selected is None else selected["alpha"],
        "selection_rejected": selected is None,
        "selected_record": selected,
        "well_assessment": well,
        "i_perp_strict_single_well_all_budgets": all(
            item["i_perp_strict_single_well_at_alpha_1"]
            for item in well.values()
        ),
        "i_perp_unique_interior_well_at_alpha_1_all_budgets": all(
            item["i_perp_unique_interior_well_at_alpha_1"]
            for item in well.values()
        ),
        "dense_barrier_strict_single_well_all_budgets": all(
            item["dense_barrier_strict_single_well_at_alpha_1"]
            for item in well.values()
        ),
        "dense_barrier_unique_moderate_interior_well_all_budgets": all(
            item["dense_barrier_unique_moderate_interior_well"]
            for item in well.values()
        ),
        "records": detailed,
    }
    output = args.output_dir / "V9_ALPHA_FINAL_WINDOW_SCORES.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, output)
    print(csv_path)
    print(output)
    if selected is None:
        raise RuntimeError("No alpha passed the frozen production gates")


if __name__ == "__main__":
    main()
