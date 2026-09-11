#!/usr/bin/env python3
"""Produce a fail-closed cache/call audit for a selected legacy IVM run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result_path = Path(args.result).expanduser().resolve()
    audit_path = result_path.with_suffix(".audit.json")
    with np.load(result_path, allow_pickle=False) as saved:
        counter = np.asarray(saved["counter_vectors"][-1], dtype=int)
        energy = np.asarray(saved["energy_request_vectors"][-1], dtype=int)
        force = np.asarray(saved["force_request_vectors"][-1], dtype=int)
        hits = np.asarray(saved["cache_hit_vectors"][-1], dtype=int)
        calls = np.asarray(saved["calls_per_image"], dtype=int)
        initial_hash = str(saved["initial_path_sha256"])
        tangent = str(saved["tangent_definition"])
    backend_audit = json.loads(audit_path.read_text())
    backend_calls = int(backend_audit["backend_raw_joint_evaluations"])

    # This run used evaluator.query(..., "both") for every optimization
    # property request. Therefore energy and force request vectors are equal,
    # and each query outcome is exactly one cache hit or one joint evaluation.
    both_only = energy == force
    outcome_closed = energy == hits + counter
    backend_closed = int(np.sum(counter)) == backend_calls
    rows = []
    for image_index in range(len(counter)):
        rows.append(
            {
                "image_index": image_index,
                "energy_api_requests": int(energy[image_index]),
                "force_api_requests": int(force[image_index]),
                "cache_hits": int(hits[image_index]),
                "joint_openmm_evaluations": int(counter[image_index]),
                "both_only_request_identity_ok": bool(
                    both_only[image_index]
                ),
                "query_outcome_identity_ok": bool(
                    outcome_closed[image_index]
                ),
            }
        )
    payload = {
        "passed": bool(
            np.all(both_only)
            and np.all(outcome_closed)
            and backend_closed
        ),
        "result": str(result_path),
        "source_audit": str(audit_path),
        "initial_path_sha256": initial_hash,
        "tangent_definition": tangent,
        "counter_semantics": (
            "run-local exact-coordinate history cache; every miss evaluates "
            "and caches energy+force jointly, and no cache entry is shared "
            "between experiments"
        ),
        "backend_raw_joint_evaluations": backend_calls,
        "sum_22_image_joint_evaluations": int(np.sum(counter)),
        "backend_sum_identity_ok": bool(backend_closed),
        "all_requests_were_both": bool(np.all(both_only)),
        "all_query_outcomes_close": bool(np.all(outcome_closed)),
        "source_history_records": int(len(calls)),
        "unique_max_call_labels": int(len(np.unique(calls))),
        "duplicate_record_policy_for_postprocessing": (
            "retain the final state recorded at each max-call label"
        ),
        "per_image": rows,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    if not payload["passed"]:
        raise RuntimeError(f"Selected-run audit failed: {output}")
    print(f"Selected-run audit passed: {output}")


if __name__ == "__main__":
    main()
