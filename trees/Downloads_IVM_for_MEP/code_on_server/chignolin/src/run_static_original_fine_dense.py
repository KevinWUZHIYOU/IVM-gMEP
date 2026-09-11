#!/usr/bin/env python3
"""Dense-barrier diagnostics for the independently selected fine productions."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import run_selected_dense_diagnostics as dense


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--outer-workers", type=int, default=2)
    parser.add_argument("--diagnostic-workers", type=int, default=2)
    parser.add_argument("--maximum-points", type=int, default=48)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    report_name = "paper_static_original_fine_dense_20260727e"
    source_manifest = (
        root
        / "reports"
        / "paper_static_original_fine_20260727e"
        / "manifest.json"
    )
    payload = json.loads(source_manifest.read_text())
    selected = {
        row["name"]: Path(row["result_file"])
        for row in payload["records"]
    }
    jobs = [
        (
            args.python.expanduser().resolve(),
            root,
            name,
            result,
            500,
            int(args.maximum_points),
            int(args.diagnostic_workers),
        )
        for name, result in selected.items()
    ]
    records = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(args.outer_workers))
    ) as executor:
        futures = {
            executor.submit(dense.run_one, *job): job[2] for job in jobs
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            records.append(record)
            print(
                f"[{completed}/{len(futures)}] {record['name']} "
                f"{record['status']}",
                flush=True,
            )
    if any(record["status"] == "failed" for record in records):
        raise RuntimeError("At least one dense diagnostic failed")
    for record in records:
        audit = json.loads(Path(record["audit_file"]).read_text())
        if (
            audit["diagnostic_backend_raw_joint_evaluations"]
            != audit["diagnostic_evaluator_cache_misses"]
            or not audit["counts_excluded_from_optimization_axis"]
        ):
            raise RuntimeError(
                f"Dense diagnostic audit failed for {record['name']}"
            )
        record["diagnostic_joint_evaluations"] = audit[
            "diagnostic_backend_raw_joint_evaluations"
        ]
        record["selected_subdivision_counts"] = audit[
            "selected_subdivision_counts"
        ]
    report_dir = root / "reports" / report_name
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "manifest.json"
    output.write_text(
        json.dumps(
            {
                "source_production_manifest": str(source_manifest),
                "checkpoint_selection": (
                    "union of logarithmic early and uniform whole-curve "
                    "indices, at most 48 paths per method"
                ),
                "optimization_cost_axis": "counter_vectors.sum(axis=1)",
                "diagnostic_calls_excluded_from_optimization_axis": True,
                "records": sorted(records, key=lambda row: row["name"]),
            },
            indent=2,
        )
        + "\n"
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
