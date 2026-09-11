#!/usr/bin/env python3
"""Audit the four-condition stopping point of the fixed production NEB."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def forward_index(calls, call):
    return int(
        np.clip(np.searchsorted(calls, call, side="right") - 1, 0, len(calls) - 1)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--ivm", required=True)
    parser.add_argument("--neb", required=True)
    parser.add_argument("--dense-ivm", required=True)
    parser.add_argument("--dense-neb", required=True)
    parser.add_argument(
        "--ivm-stop-call",
        type=int,
        default=0,
        help="Use the IVM state at or before this declared budget.",
    )
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    ivm_path = Path(args.ivm).expanduser().resolve()
    neb_path = Path(args.neb).expanduser().resolve()
    dense_ivm_path = Path(args.dense_ivm).expanduser().resolve()
    dense_neb_path = Path(args.dense_neb).expanduser().resolve()

    with np.load(ivm_path, allow_pickle=False) as saved:
        ivm_calls = np.asarray(saved["calls_per_image"], dtype=int)
        requested_ivm_call = (
            int(args.ivm_stop_call)
            if args.ivm_stop_call > 0
            else int(ivm_calls[-1])
        )
        ivm_index = forward_index(ivm_calls, requested_ivm_call)
        ivm_i_target = 1.05 * float(saved["i_perp"][ivm_index])
        ivm_call = int(ivm_calls[ivm_index])
    with np.load(dense_ivm_path, allow_pickle=False) as saved:
        dense_ivm_calls = np.asarray(saved["calls"], dtype=int)
        dense_ivm_index = forward_index(dense_ivm_calls, ivm_call)
        ivm_dense_at_budget = float(
            saved["adaptive_barrier_kcal"][dense_ivm_index]
        )
    with np.load(neb_path, allow_pickle=False) as saved:
        neb = {name: np.asarray(saved[name]) for name in saved.files}
    with np.load(dense_neb_path, allow_pickle=False) as saved:
        dense_neb = {name: np.asarray(saved[name]) for name in saved.files}

    reference = float(dense_neb["adaptive_barrier_kcal"][-1])
    ivm_barrier_error_target = abs(ivm_dense_at_budget - reference)
    raw_calls = np.asarray(neb["calls_per_image"], dtype=int)
    dense_calls = np.asarray(dense_neb["calls"], dtype=int)
    dense_barriers = np.asarray(
        dense_neb["adaptive_barrier_kcal"], dtype=float
    )
    rows = []
    for dense_index, call in enumerate(dense_calls):
        if call < 600:
            continue
        raw_index = forward_index(raw_calls, call)
        previous_call = int(call - 100)
        previous_raw_index = forward_index(raw_calls, previous_call)
        previous_dense_index = forward_index(dense_calls, previous_call)
        i_perp = float(neb["i_perp"][raw_index])
        previous_i_perp = float(neb["i_perp"][previous_raw_index])
        fmax = float(neb["physical_fmax"][raw_index])
        dense_barrier = float(dense_barriers[dense_index])
        previous_dense_barrier = float(
            dense_barriers[previous_dense_index]
        )
        barrier_error = abs(dense_barrier - reference)
        i_change = abs(i_perp - previous_i_perp) / max(
            previous_i_perp, np.finfo(float).tiny
        )
        barrier_drift = abs(dense_barrier - previous_dense_barrier)
        checks = {
            "fmax_ok": fmax <= 0.05,
            "i_perp_target_ok": i_perp <= ivm_i_target,
            "barrier_error_target_ok": (
                barrier_error <= ivm_barrier_error_target
            ),
            "i_perp_100_call_change_ok": i_change < 0.02,
            "barrier_100_call_drift_ok": barrier_drift < 0.05,
        }
        rows.append(
            {
                "call": int(call),
                "fmax_eV_per_A": fmax,
                "i_perp": i_perp,
                "dense_barrier_kcal": dense_barrier,
                "barrier_error_kcal": barrier_error,
                "previous_raw_call_label": int(raw_calls[previous_raw_index]),
                "previous_dense_call_label": int(
                    dense_calls[previous_dense_index]
                ),
                "i_perp_100_call_endpoint_change_fraction": i_change,
                "dense_barrier_100_call_drift_kcal": barrier_drift,
                **checks,
                "all_conditions_met": bool(all(checks.values())),
            }
        )
    satisfying = [row for row in rows if row["all_conditions_met"]]
    earliest = satisfying[0] if satisfying else None
    status = (
        "criteria_met"
        if earliest is not None
        else "hard_budget_cutoff_without_all_conditions"
    )
    stop_call = (
        int(earliest["call"])
        if earliest is not None
        else int(raw_calls[-1])
    )
    stop_index = forward_index(raw_calls, stop_call)
    stop_counter_vector = np.asarray(
        neb["counter_vectors"][stop_index], dtype=int
    )
    internal = stop_counter_vector[1:-1]
    report = {
        "status": status,
        "stopping_call": stop_call,
        "earliest_satisfying_record": earliest,
        "ivm_budget_call": ivm_call,
        "ivm_i_perp_1p05_target": ivm_i_target,
        "dense_barrier_reference_kcal": reference,
        "ivm_dense_barrier_at_budget_kcal": ivm_dense_at_budget,
        "ivm_barrier_error_target_kcal": ivm_barrier_error_target,
        "i_perp_stability_definition": (
            "absolute relative endpoint change between call and call-100"
        ),
        "dense_barrier_stability_definition": (
            "absolute drift from the final stored dense checkpoint at or "
            "before call-100"
        ),
        "actual_neb_run_budget": int(neb["budget"]),
        "actual_neb_final_call": int(raw_calls[-1]),
        "actual_neb_stop_reason": str(neb["stop_reason"]),
        "post_stop_segment_role": (
            "reference and stability validation; excluded from the declared "
            "minimum stopping cost"
        ),
        "stop_counter_vector": stop_counter_vector.tolist(),
        "stop_internal_min": int(np.min(internal)),
        "stop_internal_mean": float(np.mean(internal)),
        "stop_internal_max": int(np.max(internal)),
        "stop_internal_sum": int(np.sum(internal)),
        "stop_total_openmm_evaluations_including_endpoints": int(
            np.sum(stop_counter_vector)
        ),
        "n_satisfying_dense_checkpoints": len(satisfying),
        "source_ivm": str(ivm_path),
        "source_neb": str(neb_path),
        "source_dense_ivm": str(dense_ivm_path),
        "source_dense_neb": str(dense_neb_path),
    }
    report_path = root / "reports" / "neb_production_stopping_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    csv_path = root / "reports" / "neb_production_stopping_trace.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"{status} at call {stop_call}: {report_path}")


if __name__ == "__main__":
    main()
