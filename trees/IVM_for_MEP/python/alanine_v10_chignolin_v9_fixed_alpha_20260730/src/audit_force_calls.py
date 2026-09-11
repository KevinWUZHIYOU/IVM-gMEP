#!/usr/bin/env python3
"""Fail-closed white-box audit of per-image joint OpenMM call accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit_result(result_path: Path) -> dict:
    with np.load(result_path, allow_pickle=False) as saved:
        data = {name: np.asarray(saved[name]) for name in saved.files}
    audit_path = result_path.with_suffix(".audit.json")
    backend_audit = json.loads(audit_path.read_text())

    calls = np.asarray(data["calls_per_image"], dtype=int)
    joint = np.asarray(data["counter_vectors"], dtype=int)
    energy = np.asarray(data["energy_request_vectors"], dtype=int)
    force = np.asarray(data["force_request_vectors"], dtype=int)
    hits = np.asarray(data["cache_hit_vectors"], dtype=int)
    internal = joint[:, 1:-1]
    image_count = int(np.asarray(data["initial_path"]).shape[0])
    failures = []

    def require(condition, message):
        if not bool(condition):
            failures.append(message)

    require(
        joint.shape[1] == image_count,
        "counter vector image count differs from the stored initial path",
    )
    require(
        energy.shape == joint.shape
        and force.shape == joint.shape
        and hits.shape == joint.shape,
        "request/cache counter arrays have inconsistent image dimensions",
    )
    require(np.all(np.diff(joint, axis=0) >= 0), "joint counters decrease")
    require(np.all(np.diff(energy, axis=0) >= 0), "energy requests decrease")
    require(np.all(np.diff(force, axis=0) >= 0), "force requests decrease")
    require(np.all(np.diff(hits, axis=0) >= 0), "cache hits decrease")
    require(
        np.array_equal(calls, np.max(internal, axis=1)),
        "horizontal-axis calls are not max internal joint evaluations",
    )
    require(
        np.allclose(
            data["mean_calls_per_internal_image"],
            np.mean(internal, axis=1),
            rtol=0,
            atol=1e-12,
        ),
        "stored internal mean is inconsistent",
    )
    require(
        np.array_equal(
            data["total_internal_joint_evaluations"],
            np.sum(internal, axis=1),
        ),
        "stored internal sum is inconsistent",
    )
    require(
        np.array_equal(
            joint[-1],
            backend_audit["per_image_joint_evaluations"],
        ),
        "final per-image joint vector differs from backend audit",
    )
    require(
        int(np.sum(joint[-1]))
        == int(backend_audit["backend_raw_joint_evaluations"]),
        "sum of image cache misses differs from raw OpenMM backend calls",
    )

    method = str(data["method"])
    if method == "IVM":
        require(np.array_equal(energy, force), "IVM both-requests diverge")
        require(
            np.array_equal(energy, joint + hits),
            "IVM request != joint evaluation + cache hit",
        )
    elif method == "NEB":
        require(
            np.array_equal(energy + force, joint + hits),
            "NEB energy+force requests != joint evaluation + cache hit",
        )
    else:
        failures.append(f"unsupported method {method}")

    unchanged_path_with_new_calls = 0
    if "stored_paths" in data and "stored_path_calls" in data:
        paths = np.asarray(data["stored_paths"])
        path_calls = np.asarray(data["stored_path_calls"], dtype=int)
        call_to_history = {int(call): index for index, call in enumerate(calls)}
        for index in range(1, len(path_calls)):
            previous_call = int(path_calls[index - 1])
            current_call = int(path_calls[index])
            if previous_call not in call_to_history or current_call not in call_to_history:
                continue
            before = joint[call_to_history[previous_call]]
            after = joint[call_to_history[current_call]]
            if np.array_equal(paths[index], paths[index - 1]) and np.any(after > before):
                unchanged_path_with_new_calls += 1

    report = {
        "result": str(result_path.resolve()),
        "method": method,
        "passed": not failures,
        "failures": failures,
        "records": int(len(calls)),
        "horizontal_axis_definition": "max joint evaluations per internal image",
        "final_internal_min": int(np.min(internal[-1])),
        "final_internal_mean": float(np.mean(internal[-1])),
        "final_internal_max": int(np.max(internal[-1])),
        "final_internal_sum": int(np.sum(internal[-1])),
        "image_count": image_count,
        "internal_image_count": image_count - 2,
        "final_total_path_joint_evaluations": int(np.sum(joint[-1])),
        "backend_raw_joint_evaluations": int(
            backend_audit["backend_raw_joint_evaluations"]
        ),
        "unchanged_saved_path_intervals_with_new_real_evaluations": (
            unchanged_path_with_new_calls
        ),
        "interpretation": (
            "Such unchanged-path intervals prove that rejected evaluations "
            "remain in the counters rather than disappearing on rollback."
        ),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = [audit_result(Path(item).expanduser().resolve()) for item in args.result]
    payload = {
        "passed": all(item["passed"] for item in reports),
        "results": reports,
    }
    output = Path(args.root).expanduser().resolve() / "reports" / args.output
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
