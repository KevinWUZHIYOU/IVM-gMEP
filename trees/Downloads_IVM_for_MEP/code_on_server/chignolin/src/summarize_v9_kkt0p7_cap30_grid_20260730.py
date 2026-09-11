#!/usr/bin/env python3
"""Audit the spline-free V9 KKT=0.7, cap=30 fixed-alpha grid.

Only complete operator-split macrostates enter the budget slices.  Before the
late free-polish phase every recorded state is already on the spring-flow
gauge.  During free polish, a physical IVM half-step is complete only after
the corresponding geometry-only relative-defect spring step is committed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ALPHAS = (0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
BUDGETS = (1000, 3000, 6000, 10000)
TRAILING_FRACTION = 0.30
TANGENT = "normalized central difference: x[i+1] - x[i-1]"
SPRING_PROJECTION = (
    "identical nearest-neighbour spring flow with normalized central "
    "difference tangent"
)
RELATIVE_KKT = 0.7
INNER_CAP = 30


def alpha_token(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(saved, name: str):
    value = np.asarray(saved[name])
    return value.item() if value.ndim == 0 else value[-1].item()


def unique_curve(x, y):
    order = np.argsort(x, kind="stable")
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    unique_x = []
    unique_y = []
    for value_x, value_y in zip(x, y):
        if unique_x and value_x == unique_x[-1]:
            unique_y[-1] = float(value_y)
        else:
            unique_x.append(float(value_x))
            unique_y.append(float(value_y))
    return np.asarray(unique_x), np.asarray(unique_y)


def complete_macrostate_mask(saved):
    free_polish = np.asarray(saved["state_free_polish_active"], dtype=bool)
    spring_committed = np.asarray(saved["reparam_committed"], dtype=bool)
    mask = (~free_polish) | spring_committed
    if not mask[0]:
        raise RuntimeError("Initial state must be a complete macrostate")
    return mask


def trailing_integral_mean(x, y, budget, fraction=TRAILING_FRACTION):
    eligible = np.flatnonzero(np.asarray(x) <= int(budget))
    if not eligible.size:
        raise RuntimeError(f"No complete macrostate at N_F <= {budget}")
    endpoint_index = int(eligible[-1])
    endpoint = int(x[endpoint_index])
    curve_x, curve_y = unique_curve(
        x[: endpoint_index + 1], y[: endpoint_index + 1]
    )
    start = max(float(curve_x[0]), endpoint - fraction * float(budget))
    inside = (curve_x > start) & (curve_x < endpoint)
    sample_x = np.concatenate(([start], curve_x[inside], [endpoint]))
    sample_y = np.interp(sample_x, curve_x, curve_y)
    width = endpoint - start
    mean = (
        float(np.trapz(sample_y, sample_x) / width)
        if width > 0.0
        else float(sample_y[-1])
    )
    return endpoint_index, endpoint, start, mean


def anytime_integral_mean(x, y, budget):
    """Cost-axis mean after the shared initial path.

    The first record is the common unevolved initial band.  The integration
    therefore starts at the first complete macrostate produced by the
    algorithm and uses every subsequent complete macrostate through B.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    eligible = np.flatnonzero(x <= int(budget))
    if eligible.size < 2:
        raise RuntimeError(
            f"No algorithm-produced complete macrostate at N_F <= {budget}"
        )
    endpoint_index = int(eligible[-1])
    curve_x, curve_y = unique_curve(
        x[1 : endpoint_index + 1], y[1 : endpoint_index + 1]
    )
    start = float(curve_x[0])
    endpoint = float(curve_x[-1])
    width = endpoint - start
    mean = (
        float(np.trapz(curve_y, curve_x) / width)
        if width > 0.0
        else float(curve_y[-1])
    )
    return endpoint_index, int(endpoint), start, mean


def strict_single_well(values):
    values = np.asarray(values, dtype=float)
    center = ALPHAS.index(1.0)
    return bool(
        np.all(np.diff(values[: center + 1]) < 0.0)
        and np.all(np.diff(values[center:]) > 0.0)
    )


def locate_result(root: Path, alpha: float) -> Path:
    pattern = (
        "ivm_fixed_alpha_v9_kkt0p7_cap30_"
        f"a{alpha_token(alpha)}_central_state_ivm_*_budget*.npz"
    )
    matches = sorted((root / "runs" / "ivm").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one KKT=0.7/cap=30 result for alpha={alpha:g}; "
            f"got {matches}"
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    records = []
    csv_rows = []
    initial_hashes = set()
    source_hashes = set()
    for alpha in ALPHAS:
        result = locate_result(root, alpha)
        audit_path = result.with_suffix(".audit.json")
        audit = json.loads(audit_path.read_text())
        if not (
            audit["consistent"]
            and audit["backend_raw_joint_evaluations"]
            == audit["sum_image_joint_evaluations"]
            and audit["state_curve_projection_method"]
            == SPRING_PROJECTION
            and audit["state_curve_projection_uses_spline"] is False
            and audit["state_curve_projection_openmm_calls"] == 0
        ):
            raise RuntimeError(f"Counter/projection audit failed: {result}")

        with np.load(result, allow_pickle=False) as saved:
            if not (
                str(scalar(saved, "tangent_definition")) == TANGENT
                and str(scalar(saved, "alpha_mode")) == "fixed"
                and np.isclose(float(scalar(saved, "alpha")), alpha)
                and str(scalar(saved, "state_curve_projection_method"))
                == SPRING_PROJECTION
                and not bool(
                    scalar(saved, "state_curve_projection_uses_spline")
                )
                and int(
                    scalar(saved, "state_curve_projection_openmm_calls")
                )
                == 0
                and str(
                    scalar(saved, "state_free_polish_reparam_strategy")
                )
                == "relative_defect"
                and np.isclose(
                    float(
                        scalar(
                            saved,
                            "state_curve_relock_escape_trust_factor",
                        )
                    ),
                    1.0,
                )
                and np.isclose(
                    float(scalar(saved, "plbfgs_kkt_rtol")),
                    RELATIVE_KKT,
                )
                and int(scalar(saved, "inner_calls")) == INNER_CAP
                and int(
                    scalar(saved, "state_curve_direct_trial_calls")
                )
                == INNER_CAP
            ):
                raise RuntimeError(f"Algorithm metadata failed: {result}")

            initial_hashes.add(str(scalar(saved, "initial_path_sha256")))
            source_hashes.add(str(scalar(saved, "configuration_sha256")))
            total_force_all = np.asarray(
                saved["counter_vectors"], dtype=np.int64
            ).sum(axis=1)
            complete = complete_macrostate_mask(saved)
            total_force = total_force_all[complete]
            n_internal = (
                np.asarray(saved["counter_vectors"]).shape[1] - 2
            )
            if int(total_force[-1]) < max(BUDGETS) - n_internal:
                raise RuntimeError(
                    f"{result} does not cover the N_F<={max(BUDGETS)} "
                    "slice to within one synchronized 20-image macrostep; "
                    f"last complete N_F is {int(total_force[-1])}"
                )
            i_perp = np.asarray(saved["i_perp"], dtype=float)[complete]
            equidistance = np.asarray(
                saved["equidistance_integral"], dtype=float
            )[complete]
            relative_defect = np.asarray(
                saved["equidistance_relative_defect"], dtype=float
            )[complete]
            spacing_cv = np.asarray(
                saved["segment_length_cv"], dtype=float
            )[complete]
            calls_per_image = np.asarray(
                saved["calls_per_image"], dtype=int
            )[complete]
            complete_source_indices = np.flatnonzero(complete)

            budget_rows = []
            flat = {
                "alpha": alpha,
                "result": str(result),
                "result_sha256": sha256(result),
                "audit": str(audit_path),
                "audit_sha256": sha256(audit_path),
                "stop_reason": str(scalar(saved, "stop_reason")),
                "final_total_number_of_forces_all_records": int(
                    total_force_all[-1]
                ),
                "final_complete_total_number_of_forces": int(total_force[-1]),
            }
            for budget in BUDGETS:
                index, endpoint, start, mean = trailing_integral_mean(
                    total_force, i_perp, budget
                )
                (
                    anytime_index,
                    anytime_endpoint,
                    anytime_start,
                    anytime_mean,
                ) = anytime_integral_mean(total_force, i_perp, budget)
                if anytime_index != index or anytime_endpoint != endpoint:
                    raise RuntimeError(
                        "Trailing and anytime endpoint policies disagree"
                    )
                row = {
                    "requested_total_number_of_forces": budget,
                    "attained_complete_total_number_of_forces": endpoint,
                    "source_record_index": int(
                        complete_source_indices[index]
                    ),
                    "trailing_window_start_total_number_of_forces": start,
                    "trailing_integral_mean_unnormalized_i_perp": mean,
                    "anytime_window_start_total_number_of_forces": (
                        anytime_start
                    ),
                    "anytime_integral_mean_unnormalized_i_perp": (
                        anytime_mean
                    ),
                    "endpoint_unnormalized_i_perp": float(i_perp[index]),
                    "minimum_complete_unnormalized_i_perp_before_budget": (
                        float(np.min(i_perp[: index + 1]))
                    ),
                    "endpoint_equidistance_integral": float(
                        equidistance[index]
                    ),
                    "endpoint_relative_equidistance_defect": float(
                        relative_defect[index]
                    ),
                    "endpoint_segment_length_cv": float(
                        spacing_cv[index]
                    ),
                    "source_max_call_per_internal_image": int(
                        calls_per_image[index]
                    ),
                }
                budget_rows.append(row)
                for key, value in row.items():
                    if key != "requested_total_number_of_forces":
                        flat[f"{key}_nf_le_{budget}"] = value
            records.append(
                {
                    **{
                        key: flat[key]
                        for key in (
                            "alpha",
                            "result",
                            "result_sha256",
                            "audit",
                            "audit_sha256",
                            "stop_reason",
                            "final_total_number_of_forces_all_records",
                            "final_complete_total_number_of_forces",
                        )
                    },
                    "budget_records": budget_rows,
                }
            )
            csv_rows.append(flat)

    if len(initial_hashes) != 1:
        raise RuntimeError(f"Initial paths differ: {initial_hashes}")

    assessments = {}
    for budget_index, budget in enumerate(BUDGETS):
        anytime = [
            record["budget_records"][budget_index][
                "anytime_integral_mean_unnormalized_i_perp"
            ]
            for record in records
        ]
        trailing = [
            record["budget_records"][budget_index][
                "trailing_integral_mean_unnormalized_i_perp"
            ]
            for record in records
        ]
        endpoint = [
            record["budget_records"][budget_index][
                "endpoint_unnormalized_i_perp"
            ]
            for record in records
        ]
        assessments[str(budget)] = {
            "anytime_values": anytime,
            "trailing_values": trailing,
            "endpoint_values": endpoint,
            "anytime_strict_single_well_at_alpha_1": strict_single_well(
                anytime
            ),
            "trailing_strict_single_well_at_alpha_1": strict_single_well(
                trailing
            ),
            "endpoint_strict_single_well_at_alpha_1": strict_single_well(
                endpoint
            ),
            "trailing_best_alpha": float(ALPHAS[int(np.argmin(trailing))]),
            "anytime_best_alpha": float(ALPHAS[int(np.argmin(anytime))]),
            "endpoint_best_alpha": float(ALPHAS[int(np.argmin(endpoint))]),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "v9_kkt0p7_cap30_fixed_alpha_grid.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = {
        "algorithm": (
            "Fixed-alpha frozen-normal proximal IVM with one alpha-independent "
            "relative KKT forcing factor 0.7, at most 30 new joint "
            "energy-force evaluations per image including rejected trials, "
            "largest-normal-force block continuation at "
            "deterministic projected stalling, and optional normalized-"
            "central-tangent spring-flow parameterization."
        ),
        "alpha_grid": list(ALPHAS),
        "budgets_total_number_of_forces": list(BUDGETS),
        "trailing_fraction": TRAILING_FRACTION,
        "primary_alpha_sensitivity_statistic": (
            "Force-axis arithmetic mean of unnormalized I_perp over the final "
            "30% of each requested budget through the latest complete "
            "operator-split macrostate at or before B."
        ),
        "budget_state_policy": (
            "Latest complete operator-split macrostate at or before B; raw "
            "late free-polish half-steps are excluded."
        ),
        "initial_path_sha256": next(iter(initial_hashes)),
        "configuration_hashes": sorted(source_hashes),
        "tangent_definition": TANGENT,
        "projection_method": SPRING_PROJECTION,
        "projection_uses_spline": False,
        "relative_kkt_forcing_factor": RELATIVE_KKT,
        "maximum_inner_evaluations": INNER_CAP,
        "cost_axis": (
            "sum of audited joint energy-force cache misses over all 22 images"
        ),
        "dense_barrier_in_this_stage": False,
        "sparse_barrier_used_for_scoring": False,
        "dense_barrier_requirement": (
            "Real OpenMM Cartesian-linear 10/20 subdivisions per segment, "
            "then 40/80 while successive barriers differ by more than "
            "0.05 kcal/mol; diagnostic calls excluded from optimization N_F."
        ),
        "records": records,
        "assessments": assessments,
        "all_anytime_slices_strict_single_well_at_alpha_1": all(
            item["anytime_strict_single_well_at_alpha_1"]
            for item in assessments.values()
        ),
        "all_trailing_slices_strict_single_well_at_alpha_1": all(
            item["trailing_strict_single_well_at_alpha_1"]
            for item in assessments.values()
        ),
        "all_endpoint_slices_strict_single_well_at_alpha_1": all(
            item["endpoint_strict_single_well_at_alpha_1"]
            for item in assessments.values()
        ),
    }
    json_path = output_dir / "v9_kkt0p7_cap30_grid_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json_path)


if __name__ == "__main__":
    main()
