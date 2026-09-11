#!/usr/bin/env python3
"""Classify archived alpha runs by reachability of the new stall branch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived-runner", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    runner_source = args.archived_runner.read_text()
    assignments = re.findall(
        r"^\s*curve_direct_retired\s*=\s*(True|False)\s*$",
        runner_source,
        flags=re.MULTILINE,
    )
    if assignments != ["False", "True"]:
        raise RuntimeError(
            "Archived direct-retirement state is not monotone: "
            f"{assignments}"
        )
    manifest = json.loads(args.manifest.read_text())
    rows = []
    for record in manifest["records"]:
        result = Path(record["source_result"])
        with np.load(result, allow_pickle=False) as saved:
            mode = str(saved["reparam_mode"])
            retired = bool(saved["state_curve_direct_retired_final"])
            accepted = np.asarray(saved["accepted_fraction"], dtype=float)
            force_axis = np.asarray(
                saved["counter_vectors"], dtype=np.int64
            ).sum(axis=1)
            zero = np.flatnonzero(accepted == 0.0)
            tangent = str(saved["tangent_definition"])
            initial_hash = str(saved["initial_path_sha256"])
            final_nf = int(force_axis[-1])
            final_iperp = float(saved["i_perp"][-1])
        if mode != "state_curve_projected":
            raise RuntimeError(f"Unexpected mode in {result}: {mode}")
        if tangent != "normalized central difference: x[i+1] - x[i-1]":
            raise RuntimeError(f"Unexpected tangent in {result}: {tangent}")

        # In the archived runner curve_direct_retired has exactly two
        # assignments: False at initialization and True at retirement.  Thus
        # a False final value proves the new branch's prerequisite was never
        # reachable anywhere in the trajectory.
        branch_unreachable = not retired
        rows.append(
            {
                "alpha": float(record["alpha"]),
                "source_result": str(result),
                "source_sha256": sha256(result),
                "initial_path_sha256": initial_hash,
                "reparam_mode": mode,
                "tangent_definition": tangent,
                "direct_solver_retired_final": retired,
                "zero_displacement_records": int(zero.size),
                "first_zero_displacement_total_nf": (
                    int(force_axis[zero[0]]) if zero.size else None
                ),
                "final_total_nf": final_nf,
                "final_i_perp": final_iperp,
                "new_stall_branch_unreachable": branch_unreachable,
                "disposition": (
                    "reuse_by_unreachable_branch_proof"
                    if branch_unreachable
                    else "rerun_with_escape_relock"
                ),
            }
        )

    initial_hashes = sorted({row["initial_path_sha256"] for row in rows})
    if len(initial_hashes) != 1:
        raise RuntimeError(f"Mixed initial paths: {initial_hashes}")
    payload = {
        "classification_rule": (
            "Archived runner's curve_direct_retired is monotone False→True; "
            "when final False, the new escape–relock branch prerequisite was "
            "unreachable and the archived trajectory is conditionally exact."
        ),
        "archived_runner": str(args.archived_runner.resolve()),
        "archived_runner_sha256": sha256(args.archived_runner),
        "verified_direct_retirement_assignments": assignments,
        "common_initial_path_sha256": initial_hashes[0],
        "records": rows,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
    with args.csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(
        {
            "reuse": [r["alpha"] for r in rows if r[
                "new_stall_branch_unreachable"
            ]],
            "rerun": [r["alpha"] for r in rows if not r[
                "new_stall_branch_unreachable"
            ]],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
