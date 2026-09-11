#!/usr/bin/env python3
"""Build a strict audited manifest for the unified V2 alpha grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ALPHAS = (
    0.03,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    50.0,
    70.0,
    100.0,
    300.0,
    1000.0,
)
REUSABLE_ALPHAS = (0.03, 0.05, 0.1)
FORCE_BUDGETS = (1000, 3000, 6000, 10000)
TRAILING_FRACTION = 0.30
TANGENT = "normalized central difference: x[i+1] - x[i-1]"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alpha_tag(alpha: float) -> str:
    return f"{float(alpha):g}".replace(".", "p")


def one_match(paths, description: str) -> Path:
    paths = sorted(paths)
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected one {description}, found {len(paths)}: {paths}"
        )
    return paths[0].resolve()


def trailing_statistics(
    total_force: np.ndarray,
    values: np.ndarray,
    budget: int,
) -> dict:
    eligible = np.flatnonzero(total_force <= int(budget))
    if not eligible.size:
        index = 0
    else:
        index = int(eligible[-1])
    endpoint_force = int(total_force[index])
    lower = (1.0 - TRAILING_FRACTION) * endpoint_force
    window = np.flatnonzero(
        (np.arange(len(total_force)) <= index)
        & (total_force >= lower)
    )
    if not window.size:
        window = np.asarray([index])
    return {
        "requested_total_force_budget": int(budget),
        "actual_endpoint_total_force": endpoint_force,
        "source_endpoint_record": index,
        "endpoint_value": float(values[index]),
        "trailing_window_lower_total_force": float(lower),
        "trailing_record_count": int(len(window)),
        "trailing_arithmetic_mean": float(np.mean(values[window])),
        "trailing_median": float(np.median(values[window])),
        "trailing_minimum": float(np.min(values[window])),
    }


def audit_result(
    result: Path,
    provenance: str,
    expected_alpha: float,
    require_v2_fields: bool,
) -> dict:
    audit_path = result.with_suffix(".audit.json")
    if not audit_path.exists():
        raise RuntimeError(f"Missing audit file: {audit_path}")
    audit = json.loads(audit_path.read_text())
    with np.load(result, allow_pickle=False) as saved:
        alpha = float(saved["alpha"])
        total_force = np.asarray(
            saved["counter_vectors"], dtype=np.int64
        ).sum(axis=1)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        equidistance = np.asarray(
            saved["equidistance_integral"], dtype=float
        )
        fmax = np.asarray(saved["physical_fmax"], dtype=float)
        accepted = np.asarray(saved["accepted_fraction"], dtype=float)
        backtracks = np.asarray(
            saved["subproblem_line_search_backtracks"], dtype=np.int64
        )
        state_free = (
            np.asarray(saved["state_free_polish_active"], dtype=bool)
            if "state_free_polish_active" in saved.files
            else np.zeros(len(i_perp), dtype=bool)
        )
        initial_hash = str(saved["initial_path_sha256"])
        tangent = str(saved["tangent_definition"])
        alpha_mode = str(saved["alpha_mode"])
        globalization = str(saved["state_globalization"])
        reparam_mode = str(saved["reparam_mode"])
        counter_final = np.asarray(
            saved["counter_vectors"][-1], dtype=np.int64
        )
        direct_retired = bool(saved["state_curve_direct_retired_final"])
        budget = int(saved["budget"])
        stop_reason = str(saved["stop_reason"])
        v2_fields = {}
        if require_v2_fields:
            v2_fields = {
                "state_curve_immediate_block_relock": bool(
                    saved["state_curve_immediate_block_relock"]
                ),
                "failure_line_search_trials": int(
                    saved[
                        "state_curve_direct_failure_line_search_trials"
                    ]
                ),
                "state_curve_stall_free_polish": bool(
                    saved["state_curve_stall_free_polish"]
                ),
                "relock_escape_trust_factor": float(
                    saved["state_curve_relock_escape_trust_factor"]
                ),
                "relock_strategy": str(
                    saved["state_free_polish_reparam_strategy"]
                ),
            }
    if not np.isclose(alpha, expected_alpha, rtol=0.0, atol=0.0):
        raise RuntimeError(
            f"Alpha mismatch in {result}: {alpha} != {expected_alpha}"
        )
    if tangent != TANGENT:
        raise RuntimeError(f"Tangent mismatch in {result}: {tangent}")
    if alpha_mode != "fixed" or globalization != "proximal_only":
        raise RuntimeError(f"Non-production alpha/globalization in {result}")
    if reparam_mode != "state_curve_projected":
        raise RuntimeError(f"Unexpected reparameterization in {result}")
    if np.any(np.diff(total_force) < 0):
        raise RuntimeError(f"Nonmonotone total-force axis in {result}")
    if not (
        audit["consistent"]
        and int(audit["backend_raw_joint_evaluations"])
        == int(audit["sum_image_joint_evaluations"])
        == int(counter_final.sum())
    ):
        raise RuntimeError(f"Counter audit failed in {result}")
    if int(audit["backend_raw_joint_evaluations"]) != int(
        counter_final.sum()
    ):
        raise RuntimeError(f"Backend/NPZ counter mismatch in {result}")
    if (
        not all(audit["per_image_request_identity_ok"])
        or not all(audit["per_image_query_outcome_identity_ok"])
        or any(audit["per_image_budget_blocked_misses"])
    ):
        raise RuntimeError(f"Per-image request audit failed in {result}")
    if require_v2_fields and (
        not v2_fields["state_curve_immediate_block_relock"]
        or v2_fields["failure_line_search_trials"] != 4
        or not v2_fields["state_curve_stall_free_polish"]
        or not np.isclose(
            v2_fields["relock_escape_trust_factor"], 1e-4
        )
        or v2_fields["relock_strategy"] != "hermite_relock"
    ):
        raise RuntimeError(f"V2 configuration mismatch in {result}")
    if not require_v2_fields and (
        direct_retired
        or np.any(accepted != 1.0)
        or np.any(backtracks != 0)
    ):
        raise RuntimeError(
            f"Archived run is not provably branch-unreachable: {result}"
        )

    i_windows = [
        trailing_statistics(total_force, i_perp, budget_value)
        for budget_value in FORCE_BUDGETS
    ]
    return {
        "alpha": float(alpha),
        "provenance": provenance,
        "result_file": str(result),
        "result_sha256": sha256(result),
        "audit_file": str(audit_path),
        "audit_sha256": sha256(audit_path),
        "initial_path_sha256": initial_hash,
        "tangent_definition": tangent,
        "budget_calls_per_image": budget,
        "stop_reason": stop_reason,
        "final_total_number_of_forces": int(total_force[-1]),
        "final_per_image_joint_evaluations": counter_final.tolist(),
        "final_i_perp": float(i_perp[-1]),
        "minimum_i_perp": float(np.min(i_perp)),
        "final_physical_fmax": float(fmax[-1]),
        "final_equidistance_integral": float(equidistance[-1]),
        "maximum_equidistance_integral": float(np.max(equidistance)),
        "direct_solver_retired": direct_retired,
        "first_state_free_polish_total_force": (
            int(total_force[np.flatnonzero(state_free)[0]])
            if np.any(state_free)
            else None
        ),
        "maximum_aggregate_line_search_backtracks": int(
            np.max(backtracks)
        ),
        "i_perp_trailing_windows": i_windows,
        "v2_configuration": v2_fields if require_v2_fields else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    old_manifest = json.loads(
        args.old_manifest.expanduser().resolve().read_text()
    )
    old_by_alpha = {
        float(record["alpha"]): record
        for record in old_manifest["records"]
    }

    records = []
    for alpha in ALPHAS:
        if alpha in REUSABLE_ALPHAS:
            archived_record = old_by_alpha[alpha]
            source = Path(archived_record["source_result"]).resolve()
            archived_sha256 = archived_record.get(
                "source_result_sha256",
                archived_record.get("source_sha256"),
            )
            if archived_sha256 is None:
                raise RuntimeError(
                    "Archived branch-reuse record has no source SHA-256: "
                    f"alpha={alpha:g}"
                )
            if sha256(source) != archived_sha256:
                raise RuntimeError(f"Archived result changed: {source}")
            record = audit_result(
                source,
                "reused_exactly: new V2 branches and four-trial limit "
                "provably unreachable (all bands accepted; zero backtracks)",
                alpha,
                require_v2_fields=False,
            )
        elif np.isclose(alpha, 1.0):
            source = one_match(
                (root / "runs" / "ivm").glob(
                    "ivm_production_a1_early_block_relock_ls4_v2_"
                    "central_state_ivm_*_budget500.npz"
                ),
                "V2 alpha=1 result",
            )
            record = audit_result(
                source, "new_unified_v2_run", alpha, True
            )
        else:
            source = one_match(
                (root / "runs" / "ivm").glob(
                    f"ivm_production_v2_universal_a{alpha_tag(alpha)}_"
                    "central_state_ivm_*_budget500.npz"
                ),
                f"V2 alpha={alpha:g} result",
            )
            record = audit_result(
                source, "new_unified_v2_run", alpha, True
            )
        records.append(record)

    hashes = sorted({record["initial_path_sha256"] for record in records})
    if len(hashes) != 1:
        raise RuntimeError(f"Mixed initial paths: {hashes}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": (
            "Unified fixed-alpha V2 IVM. Every new run differs only in "
            "alpha; archived alpha<=0.1 trajectories are reused only after "
            "an exact branch-unreachability proof."
        ),
        "alphas": list(ALPHAS),
        "total_force_budgets": list(FORCE_BUDGETS),
        "trailing_fraction": TRAILING_FRACTION,
        "trailing_statistic": "arithmetic mean over all measured records",
        "cost_axis": "counter_vectors.sum(axis=1), all 22 images",
        "common_initial_path_sha256": hashes[0],
        "tangent_definition": TANGENT,
        "records": records,
    }
    json_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    rows = []
    for record in records:
        row = {
            "alpha": record["alpha"],
            "provenance": record["provenance"],
            "final_total_number_of_forces": record[
                "final_total_number_of_forces"
            ],
            "final_i_perp": record["final_i_perp"],
            "minimum_i_perp": record["minimum_i_perp"],
            "final_physical_fmax": record["final_physical_fmax"],
            "final_equidistance_integral": record[
                "final_equidistance_integral"
            ],
            "direct_solver_retired": record["direct_solver_retired"],
            "first_state_free_polish_total_force": record[
                "first_state_free_polish_total_force"
            ],
            "result_file": record["result_file"],
            "result_sha256": record["result_sha256"],
        }
        for window in record["i_perp_trailing_windows"]:
            budget = window["requested_total_force_budget"]
            row[f"i_perp_endpoint_nf_le_{budget}"] = window[
                "endpoint_value"
            ]
            row[f"i_perp_trailing_mean_nf_le_{budget}"] = window[
                "trailing_arithmetic_mean"
            ]
            row[f"actual_endpoint_nf_le_{budget}"] = window[
                "actual_endpoint_total_force"
            ]
        rows.append(row)
    csv_path = output_dir / "manifest.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
