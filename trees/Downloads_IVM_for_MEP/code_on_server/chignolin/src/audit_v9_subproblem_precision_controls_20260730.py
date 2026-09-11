#!/usr/bin/env python3
"""Audit the cap/KKT controls behind the final-window production choice."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


BUDGETS = (1000, 3000, 6000, 10000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(saved, key):
    value = np.asarray(saved[key])
    return value.item() if value.ndim == 0 else value[-1].item()


def complete_mask(saved):
    return (
        ~np.asarray(saved["state_free_polish_active"], dtype=bool)
    ) | np.asarray(saved["reparam_committed"], dtype=bool)


def unique_curve(x, y):
    order = np.argsort(x, kind="stable")
    x = np.asarray(x, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    output_x = []
    output_y = []
    for value_x, value_y in zip(x, y):
        if output_x and value_x == output_x[-1]:
            output_y[-1] = float(value_y)
        else:
            output_x.append(float(value_x))
            output_y.append(float(value_y))
    return np.asarray(output_x), np.asarray(output_y)


def integral_mean(x, y, start, endpoint):
    x, y = unique_curve(x, y)
    inside = (x > start) & (x < endpoint)
    sample_x = np.concatenate(([start], x[inside], [endpoint]))
    sample_y = np.interp(sample_x, x, y)
    width = endpoint - start
    return (
        float(np.trapz(sample_y, sample_x) / width)
        if width > 0.0
        else float(sample_y[-1])
    )


def inspect(path: Path, label: str) -> tuple[dict, list[dict]]:
    audit_path = path.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text())
    if not (
        audit["consistent"]
        and audit["backend_raw_joint_evaluations"]
        == audit["sum_image_joint_evaluations"]
    ):
        raise RuntimeError(f"Counter audit failed: {path}")
    with np.load(path, allow_pickle=False) as saved:
        complete = complete_mask(saved)
        source_indices = np.flatnonzero(complete)
        total_force = np.asarray(
            saved["counter_vectors"], dtype=np.int64
        ).sum(axis=1)[complete]
        i_perp = np.asarray(saved["i_perp"], dtype=float)[complete]
        hit = np.asarray(
            saved["subproblem_hit_call_limit_images"], dtype=float
        )[complete]
        relative_kkt = np.asarray(
            saved["subproblem_relative_kkt_max"], dtype=float
        )[complete]
        metadata = {
            "label": label,
            "alpha": float(scalar(saved, "alpha")),
            "relative_kkt_forcing_factor": float(
                scalar(saved, "plbfgs_kkt_rtol")
            ),
            "inner_call_cap": int(scalar(saved, "inner_calls")),
            "result": str(path),
            "result_sha256": sha256(path),
            "audit": str(audit_path),
            "audit_sha256": sha256(audit_path),
            "initial_path_sha256": str(
                scalar(saved, "initial_path_sha256")
            ),
            "projection_uses_spline": bool(
                scalar(saved, "state_curve_projection_uses_spline")
            ),
            "projection_openmm_calls": int(
                scalar(saved, "state_curve_projection_openmm_calls")
            ),
        }
        rows = []
        for budget in BUDGETS:
            eligible = np.flatnonzero(total_force <= budget)
            if not eligible.size:
                raise RuntimeError(f"No state through {budget}: {path}")
            endpoint_index = int(eligible[-1])
            endpoint_force = int(total_force[endpoint_index])
            trailing_start = max(
                float(total_force[0]), endpoint_force - 0.30 * budget
            )
            anytime_start = float(total_force[1])
            raw_source = int(source_indices[endpoint_index])
            rows.append(
                {
                    "label": label,
                    "alpha": metadata["alpha"],
                    "relative_kkt_forcing_factor": metadata[
                        "relative_kkt_forcing_factor"
                    ],
                    "inner_call_cap": metadata["inner_call_cap"],
                    "requested_total_force_budget": budget,
                    "actual_endpoint_total_force": endpoint_force,
                    "endpoint_i_perp": float(i_perp[endpoint_index]),
                    "trailing_mean_i_perp": integral_mean(
                        total_force[: endpoint_index + 1],
                        i_perp[: endpoint_index + 1],
                        trailing_start,
                        endpoint_force,
                    ),
                    "anytime_mean_i_perp": integral_mean(
                        total_force[1 : endpoint_index + 1],
                        i_perp[1 : endpoint_index + 1],
                        anytime_start,
                        endpoint_force,
                    ),
                    "endpoint_images_hitting_inner_cap": int(hit[endpoint_index]),
                    "endpoint_relative_kkt_max": float(
                        relative_kkt[endpoint_index]
                    ),
                    "source_record_index": raw_source,
                }
            )
    return metadata, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    run_dir = root / "runs" / "ivm"
    specifications = (
        ("kkt0.7_cap30_alpha1.5", "ivm_fixed_alpha_v9_kkt0p7_cap30_a1p5*budget650.npz"),
        ("kkt0.7_cap30_alpha2", "ivm_fixed_alpha_v9_kkt0p7_cap30_a2*budget650.npz"),
        ("kkt0.7_cap60_alpha1.5", "ivm_fixed_alpha_v9_kkt0p7_cap60_a1p5*budget650.npz"),
        ("kkt0.7_cap60_alpha2", "ivm_fixed_alpha_v9_kkt0p7_cap60_a2*budget650.npz"),
        (
            "kkt0.5_cap60_alpha1",
            "ivm_probe_v9_kkt0p5_cap60_a1_central_state_ivm_*budget500.npz",
        ),
        ("kkt0.5_cap60_alpha1.5", "ivm_probe_v9_kkt0p5_cap60_a1p5*budget500.npz"),
        ("kkt0.5_cap60_alpha2", "ivm_probe_v9_kkt0p5_cap60_a2*budget500.npz"),
    )
    metadata = []
    rows = []
    for label, pattern in specifications:
        matches = sorted(run_dir.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"{label}: expected one result, got {matches}")
        record, budget_rows = inspect(matches[0], label)
        metadata.append(record)
        rows.extend(budget_rows)
    hashes = {record["initial_path_sha256"] for record in metadata}
    if len(hashes) != 1:
        raise RuntimeError(f"Control initial paths differ: {hashes}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "V9_SUBPROBLEM_PRECISION_CONTROLS.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "purpose": (
            "Determine whether a larger uniform inner cap or tighter relative "
            "KKT criterion removes late fixed-alpha phase reversals without "
            "damaging alpha=1 production efficiency."
        ),
        "common_initial_path_sha256": next(iter(hashes)),
        "cost_axis": (
            "sum of audited joint energy-force cache misses over all 22 images"
        ),
        "complete_macrostate_policy": (
            "Raw late physical half-steps are excluded until their "
            "geometry-only spring step is committed."
        ),
        "conclusion": (
            "Rejected as production-shaping controls. Cap=60 does not remove "
            "the alpha=1.5 versus alpha=2 late phase reversal. Tightening to "
            "relative KKT=0.5 also leaves a late reversal and degrades the "
            "alpha=1 N_F<=10000 trailing mean by more than an order of "
            "magnitude. The cap=30/KKT=0.7 alpha=1 production is retained, "
            "and the user-requested final-30%-of-budget mean is used as the "
            "primary finite-budget precision statistic. Cumulative anytime "
            "burden remains an archived secondary audit."
        ),
        "records": metadata,
        "budget_records": rows,
        "csv": str(csv_path),
    }
    json_path = args.output_dir / "V9_SUBPROBLEM_PRECISION_CONTROLS.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
