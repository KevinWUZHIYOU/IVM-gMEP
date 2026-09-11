#!/usr/bin/env python3
"""Score the preregistered V2 alpha grid without fitting or cherry-picking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


FORCE_BUDGETS = (1000, 3000, 6000, 10000)
REFERENCE_BARRIER_KCAL = 20.373739515742795
REFERENCE_FINAL_I_PERP = 0.03885242073674519
REFERENCE_FINAL_EQUIDISTANCE = 5.885353699911007e-20
REFERENCE_B10000_DENSE_RMS_ERROR = 0.798519886390451
NONINFERIORITY_FACTOR = 1.05
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometric_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Geometric-mean inputs must be positive and finite")
    return float(np.exp(np.mean(np.log(values))))


def local_minima(values: np.ndarray) -> list[int]:
    """Return raw connected-curve local minima; no fit or smoothing."""
    values = np.asarray(values, dtype=float)
    output = []
    for index in range(1, len(values) - 1):
        left = values[index - 1]
        center = values[index]
        right = values[index + 1]
        if center <= left and center <= right and (
            center < left or center < right
        ):
            output.append(index)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dense-manifest", type=Path, required=True)
    parser.add_argument("--method-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    dense_manifest_path = args.dense_manifest.expanduser().resolve()
    method_summary_path = args.method_summary.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    dense_manifest = json.loads(dense_manifest_path.read_text())
    method_summary = json.loads(method_summary_path.read_text())
    method_records = method_summary["methods"]
    ode_final_i = float(method_records["ODE-NEB"]["final_i_perp"])
    obsm_final_i = float(method_records["OBSM-FIRE"]["final_i_perp"])
    ode_final_dense_barrier = float(
        method_records["ODE-NEB"]["final_dense_barrier_kcal"]
    )
    obsm_final_dense_barrier = float(
        method_records["OBSM-FIRE"]["final_dense_barrier_kcal"]
    )
    dense_by_alpha = {
        float(record["alpha"]): record
        for record in dense_manifest["records"]
    }
    rows = []
    detailed = []
    for record in manifest["records"]:
        alpha = float(record["alpha"])
        dense_record = dense_by_alpha[alpha]
        result = Path(record["result_file"]).resolve()
        dense_result = Path(dense_record["dense_result"]).resolve()
        if (
            sha256(result) != record["result_sha256"]
            or sha256(dense_result) != dense_record["dense_result_sha256"]
        ):
            raise RuntimeError(f"Source changed for alpha={alpha:g}")
        i_windows = record["i_perp_trailing_windows"]
        if tuple(
            int(window["requested_total_force_budget"])
            for window in i_windows
        ) != FORCE_BUDGETS:
            raise RuntimeError(f"I_perp budget mismatch for alpha={alpha:g}")
        i_means = np.asarray(
            [window["trailing_arithmetic_mean"] for window in i_windows],
            dtype=float,
        )
        i_endpoints = np.asarray(
            [window["endpoint_value"] for window in i_windows], dtype=float
        )
        energy_rms = np.asarray(
            dense_record["rms_barrier_error_kcal_by_budget"], dtype=float
        )
        if energy_rms.shape != (len(FORCE_BUDGETS),):
            raise RuntimeError(
                f"Dense RMS budget mismatch for alpha={alpha:g}"
            )
        with np.load(result, allow_pickle=False) as saved:
            final_i = float(saved["i_perp"][-1])
            final_eq = float(saved["equidistance_integral"][-1])
            final_spacing_cv = float(saved["segment_length_cv"][-1])
            final_total_force = int(
                np.asarray(saved["counter_vectors"][-1], dtype=np.int64).sum()
            )
        with np.load(dense_result, allow_pickle=False) as saved:
            barriers = np.asarray(
                saved["adaptive_barrier_kcal"], dtype=float
            )
            errors = np.asarray(saved["barrier_error_kcal"], dtype=float)
            actual_force = np.asarray(
                saved["actual_total_force"], dtype=np.int64
            )
            subdivisions = np.asarray(
                saved["selected_subdivisions"], dtype=np.int64
            )
        if barriers.shape != (len(FORCE_BUDGETS), 7):
            raise RuntimeError(
                f"Dense trailing matrix mismatch for alpha={alpha:g}"
            )
        final_dense_barrier = float(barriers[-1, -1])
        final_dense_error = float(errors[-1, -1])
        i_gate = final_i <= (
            NONINFERIORITY_FACTOR * REFERENCE_FINAL_I_PERP
        )
        energy_gate = energy_rms[-1] <= (
            NONINFERIORITY_FACTOR * REFERENCE_B10000_DENSE_RMS_ERROR
        )
        equidistance_gate = final_eq <= (
            NONINFERIORITY_FACTOR * REFERENCE_FINAL_EQUIDISTANCE
        )
        common_i_gate = final_i <= min(
            ode_final_i, obsm_final_i
        )
        common_energy_gate = final_dense_error <= min(
            abs(ode_final_dense_barrier - REFERENCE_BARRIER_KCAL),
            abs(obsm_final_dense_barrier - REFERENCE_BARRIER_KCAL),
        )
        eligible = bool(
            i_gate
            and energy_gate
            and equidistance_gate
            and common_i_gate
            and common_energy_gate
        )
        stationarity_score = geometric_mean(i_means)
        energy_score = geometric_mean(energy_rms)
        row = {
            "alpha": alpha,
            "production_eligible": eligible,
            "stationarity_score_geometric_mean": stationarity_score,
            "dense_barrier_score_geometric_mean_kcal": energy_score,
            "final_i_perp": final_i,
            "final_dense_barrier_kcal": final_dense_barrier,
            "final_dense_barrier_error_kcal": final_dense_error,
            "b10000_dense_barrier_rms_error_kcal": float(energy_rms[-1]),
            "final_equidistance_integral": final_eq,
            "final_segment_length_cv": final_spacing_cv,
            "final_total_number_of_forces": final_total_force,
            "cap15_final_i_perp_gate": bool(i_gate),
            "cap15_dense_rms_gate": bool(energy_gate),
            "cap15_equidistance_gate": bool(equidistance_gate),
            "ode_obsm_common_i_perp_gate": bool(common_i_gate),
            "ode_obsm_common_dense_barrier_gate": bool(common_energy_gate),
            "optimization_result": str(result),
            "optimization_result_sha256": record["result_sha256"],
            "dense_result": str(dense_result),
            "dense_result_sha256": dense_record["dense_result_sha256"],
        }
        for budget_index, budget in enumerate(FORCE_BUDGETS):
            row[f"i_perp_trailing_mean_nf_le_{budget}"] = float(
                i_means[budget_index]
            )
            row[f"i_perp_endpoint_nf_le_{budget}"] = float(
                i_endpoints[budget_index]
            )
            row[f"dense_barrier_rms_error_nf_le_{budget}"] = float(
                energy_rms[budget_index]
            )
        rows.append(row)
        detailed.append(
            {
                **row,
                "dense_barriers_by_budget_and_sample_kcal": barriers.tolist(),
                "dense_barrier_errors_by_budget_and_sample_kcal": errors.tolist(),
                "dense_actual_total_force_by_budget_and_sample": (
                    actual_force.tolist()
                ),
                "dense_subdivisions_by_budget_and_sample": (
                    subdivisions.tolist()
                ),
            }
        )

    rows.sort(key=lambda row: row["alpha"])
    detailed.sort(key=lambda row: row["alpha"])
    eligible = [row for row in rows if row["production_eligible"]]
    selected = (
        min(
            eligible,
            key=lambda row: (
                row["stationarity_score_geometric_mean"],
                row["dense_barrier_score_geometric_mean_kcal"],
                row["final_equidistance_integral"],
                row["final_total_number_of_forces"],
            ),
        )
        if eligible
        else None
    )

    alphas = np.asarray([row["alpha"] for row in rows], dtype=float)
    well_assessment = {}
    for metric_prefix in (
        "i_perp_trailing_mean_nf_le_",
        "dense_barrier_rms_error_nf_le_",
    ):
        metric_rows = {}
        for budget in FORCE_BUDGETS:
            values = np.asarray(
                [row[f"{metric_prefix}{budget}"] for row in rows],
                dtype=float,
            )
            minima = local_minima(values)
            global_index = int(np.argmin(values))
            metric_rows[str(budget)] = {
                "global_minimum_alpha": float(alphas[global_index]),
                "global_minimum_value": float(values[global_index]),
                "raw_local_minimum_alphas": [
                    float(alphas[index]) for index in minima
                ],
                "raw_local_minimum_values": [
                    float(values[index]) for index in minima
                ],
                "single_interior_local_minimum": bool(len(minima) == 1),
                "global_minimum_is_moderate": bool(
                    0.2 <= alphas[global_index] <= 20.0
                ),
            }
        well_assessment[metric_prefix.rstrip("_")] = metric_rows
    i_single_well_all_budgets = all(
        row["single_interior_local_minimum"]
        and row["global_minimum_is_moderate"]
        for row in well_assessment[
            "i_perp_trailing_mean_nf_le"
        ].values()
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "V2_ALPHA_SCORES.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "source_dense_manifest": str(dense_manifest_path),
        "source_dense_manifest_sha256": sha256(dense_manifest_path),
        "source_method_summary": str(method_summary_path),
        "source_method_summary_sha256": sha256(method_summary_path),
        "preregistered_selection": (
            "Eligible candidates are ranked lexicographically by "
            "across-budget stationarity score, across-budget dense-barrier "
            "score, final equal-speed integral, and total Number of Forces."
        ),
        "sparse_22_image_barrier_used": False,
        "reference_values": {
            "dense_barrier_kcal": REFERENCE_BARRIER_KCAL,
            "final_i_perp": REFERENCE_FINAL_I_PERP,
            "final_equidistance_integral": REFERENCE_FINAL_EQUIDISTANCE,
            "b10000_dense_rms_error_kcal": (
                REFERENCE_B10000_DENSE_RMS_ERROR
            ),
            "noninferiority_factor": NONINFERIORITY_FACTOR,
        },
        "selected_alpha": (
            None if selected is None else selected["alpha"]
        ),
        "selection_rejected": selected is None,
        "selected_record": selected,
        "i_perp_strict_single_well_all_budgets": (
            i_single_well_all_budgets
        ),
        "single_well_assessment": well_assessment,
        "records": detailed,
    }
    json_path = output_dir / "V2_ALPHA_SCORES.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, json_path)
    print(csv_path)
    print(json_path)
    if selected is None:
        raise RuntimeError(
            "No V2 alpha passes the preregistered production gates"
        )


if __name__ == "__main__":
    main()
