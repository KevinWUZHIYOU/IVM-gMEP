#!/usr/bin/env python3
"""Freeze all 333 paid ODE--NEB paths before any new arc500 energy call."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from evaluate_discrete_iperp_densified_barrier import (
    COORDINATE_SPACE,
    GRID_FORMULA,
    INTERPOLATION,
    N_DENSE_POINTS,
    SAMPLING,
    sha256_file,
)


EXPECTED_LABEL = "bbcr30_neb_ode1e6_k0p1_prod_20260814_r1"
EXPECTED_BATCHES = 333
EXPECTED_IMAGES = 32
EXPECTED_INTERNAL = 30
EXPECTED_DIMENSION = 9396
SCHEMA = "bb-ode-neb-all-paid-current-paths-arc500-contract-v2"


def official_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    if not np.all(np.isfinite(array)):
        raise ValueError("official path contains a non-finite value")
    metadata = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--reuse-diagnostics", type=Path, required=True)
    parser.add_argument("--reuse-audit", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--precall-failure", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_root.expanduser().resolve(strict=True)
    topology = args.topology.expanduser().resolve(strict=True)
    reuse = args.reuse_diagnostics.expanduser().resolve(strict=True)
    reuse_audit = args.reuse_audit.expanduser().resolve(strict=True)
    evaluator = args.evaluator.expanduser().resolve(strict=True)
    precall_failure = (
        args.precall_failure.expanduser().resolve(strict=True)
        if args.precall_failure is not None
        else None
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    complete_path = source / "OPTIMIZATION_COMPLETE.json"
    audit_path = source / "OPTIMIZATION_AUDIT.json"
    wal_path = source / "optimization_force_wal.jsonl"
    endpoints_path = source / "frozen_endpoints_A.npy"
    for path in (complete_path, audit_path, wal_path, endpoints_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required_complete = {
        "label": EXPECTED_LABEL,
        "paid_batches": EXPECTED_BATCHES,
        "paid_nf": EXPECTED_BATCHES * EXPECTED_INTERNAL,
        "images": EXPECTED_IMAGES,
        "interior_images": EXPECTED_INTERNAL,
        "endpoint_physical_calls": 0,
    }
    for key, expected in required_complete.items():
        if complete.get(key) != expected:
            raise ValueError(f"terminal metadata mismatch for {key}")
    if complete.get("optimization_wal_sha256") != sha256_file(wal_path):
        raise ValueError("optimization WAL hash mismatch")
    if (
        audit.get("status") != "PASS"
        and audit.get("consistent") is not True
        and audit.get("pass") is not True
    ):
        raise ValueError("optimization audit is not a passing consistency receipt")

    reserves = {}
    commits = {}
    previous_sequence = -1
    with wal_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            sequence = int(event["sequence"])
            if sequence != previous_sequence + 1:
                raise ValueError("optimization WAL sequence is not contiguous")
            previous_sequence = sequence
            kind = event.get("event")
            if kind not in {"batch_reserve", "batch_commit"}:
                continue
            batch_id = int(event["batch_id"])
            target = reserves if kind == "batch_reserve" else commits
            if batch_id in target:
                raise ValueError(f"duplicate {kind}: {batch_id}")
            target[batch_id] = event
    if sorted(reserves) != list(range(1, EXPECTED_BATCHES + 1)):
        raise ValueError("optimization WAL does not contain all 333 batch reserves")
    if sorted(commits) != list(range(1, EXPECTED_BATCHES + 1)):
        raise ValueError("optimization WAL does not contain all 333 batch commits")
    dispositions = [str(commits[index]["disposition"]) for index in range(1, EXPECTED_BATCHES + 1)]
    if dispositions.count("current_initial") != 1:
        raise ValueError("expected exactly one current_initial disposition")
    if dispositions.count("accepted") != 283 or dispositions.count("rejected") != 49:
        raise ValueError("accepted/rejected disposition counts do not match the sealed audit")

    frozen_endpoints = np.asarray(np.load(endpoints_path, allow_pickle=False), dtype=np.float64)
    if frozen_endpoints.shape != (2, EXPECTED_DIMENSION // 3, 3):
        raise ValueError("frozen endpoint array has the wrong shape")
    archives_by_hash = {}
    coordinate_root = source / "paid_batch_coordinates"
    for batch_id in range(1, EXPECTED_BATCHES + 1):
        event = reserves[batch_id]
        relative = Path(str(event["batch_coordinate_file"]))
        archive = (source / relative).resolve(strict=True)
        expected_name = f"batch_{batch_id:04d}_paid_coordinates30_A.npz"
        if archive.parent != coordinate_root or archive.name != expected_name:
            raise ValueError(f"batch {batch_id} coordinate routing mismatch")
        archive_sha = sha256_file(archive)
        if archive_sha != str(event["batch_coordinate_file_sha256"]):
            raise ValueError(f"batch {batch_id} coordinate archive hash mismatch")
        with np.load(archive, allow_pickle=False) as saved:
            coordinates = np.asarray(saved["coordinates30_A"], dtype=np.float64)
            endpoint0 = np.asarray(saved["endpoint0_A"], dtype=np.float64)
            endpoint1 = np.asarray(saved["endpoint1_A"], dtype=np.float64)
            embedded_hash = str(np.asarray(saved["path32_sha256"]).item())
            state_role = str(np.asarray(saved["state_role"]).item())
        if coordinates.shape != (EXPECTED_INTERNAL, EXPECTED_DIMENSION // 3, 3):
            raise ValueError(f"batch {batch_id} coordinate shape mismatch")
        if not np.array_equal(endpoint0, frozen_endpoints[0]) or not np.array_equal(
            endpoint1, frozen_endpoints[1]
        ):
            raise ValueError(f"batch {batch_id} endpoint mismatch")
        path32 = np.concatenate((endpoint0[None], coordinates, endpoint1[None]), axis=0)
        official_hash = official_array_sha256(path32)
        if official_hash != embedded_hash or official_hash != str(event["path32_sha256"]):
            raise ValueError(f"batch {batch_id} path identity mismatch")
        nf_after = int(event["nf_after"])
        if nf_after != EXPECTED_INTERNAL * batch_id:
            raise ValueError(f"batch {batch_id} optimization N_F mismatch")
        archives_by_hash[official_hash] = {
            "source_batch_id": batch_id,
            "source_state_role": state_role,
            "coordinate_archive": str(archive),
            "coordinate_archive_sha256": archive_sha,
            "official_path32_sha256": official_hash,
        }

    paths = []
    previous_current_hash = None
    for paid_batch_id in range(1, EXPECTED_BATCHES + 1):
        commit = commits[paid_batch_id]
        nf_after = int(commit["nf_after"])
        if nf_after != EXPECTED_INTERNAL * paid_batch_id:
            raise ValueError(f"batch {paid_batch_id} committed N_F mismatch")
        current_hash = str(commit["current_path32_sha256"])
        disposition = str(commit["disposition"])
        if disposition == "rejected" and current_hash != previous_current_hash:
            raise ValueError(f"rejected batch {paid_batch_id} did not retain the previous current path")
        if disposition in {"current_initial", "accepted"} and current_hash != str(commit["paid_path32_sha256"]):
            raise ValueError(f"accepted/current batch {paid_batch_id} current/paid identity mismatch")
        if current_hash not in archives_by_hash:
            raise ValueError(f"current path for batch {paid_batch_id} has no frozen coordinate archive")
        source_record = archives_by_hash[current_hash]
        paths.append(
            {
                "paid_batch_id": paid_batch_id,
                "optimization_N_F": nf_after,
                "disposition": disposition,
                **source_record,
            }
        )
        previous_current_hash = current_hash

    unique_current_hashes = {record["official_path32_sha256"] for record in paths}
    if len(unique_current_hashes) != 284:
        raise ValueError("expected 284 distinct retained paths (initial plus 283 accepted trials)")

    payload = {
        "schema": SCHEMA,
        "status": "FROZEN_BEFORE_ANY_NEW_ALL_BATCH_ARC500_ENERGY_CALL",
        "purpose": "Apply the common equal-Cartesian-arclength 500-point densified Delta E diagnostic to the actual retained ODE--NEB path at every paid N_F value; rejected trial proposals retain the preceding accepted path.",
        "source_scope": {
            "run_label": EXPECTED_LABEL,
            "paid_N_F_state_count": EXPECTED_BATCHES,
            "distinct_retained_path_count": len(unique_current_hashes),
            "accepted_trial_count": 283,
            "rejected_trial_count": 49,
            "optimization_N_F": [EXPECTED_INTERNAL, EXPECTED_BATCHES * EXPECTED_INTERNAL],
            "fixed_endpoint_optimization_N_F": 0,
        },
        "diagnostic_protocol": {
            "coordinate_space": COORDINATE_SPACE,
            "interpolation": INTERPOLATION,
            "sampling": SAMPLING,
            "number_of_points": N_DENSE_POINTS,
            "include_both_endpoints": True,
            "grid_formula": GRID_FORMULA,
            "energy_evaluations_per_path": N_DENSE_POINTS,
            "force_evaluations_per_path": 0,
            "optimization_N_F_contribution": 0,
            "identical_path_policy": "exact dense-grid and energy-vector reuse only after exact 32-image path hash equality",
            "rejected_trial_policy": "at that paid N_F, diagnose the retained pre-trial current path rather than the rejected proposal",
        },
        "bindings": {
            "optimization_complete": {"path": str(complete_path), "sha256": sha256_file(complete_path)},
            "optimization_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
            "optimization_wal": {"path": str(wal_path), "sha256": sha256_file(wal_path)},
            "frozen_endpoints": {"path": str(endpoints_path), "sha256": sha256_file(endpoints_path)},
            "topology": {"path": str(topology), "sha256": sha256_file(topology)},
            "reuse_diagnostics": {"path": str(reuse), "sha256": sha256_file(reuse)},
            "reuse_audit": {"path": str(reuse_audit), "sha256": sha256_file(reuse_audit)},
            "evaluator": {"path": str(evaluator), "sha256": sha256_file(evaluator)},
        },
        "paths": paths,
    }
    if precall_failure is not None:
        payload["bindings"]["superseded_precall_failure"] = {
            "path": str(precall_failure),
            "sha256": sha256_file(precall_failure),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(output)
    print(sha256_file(output))


if __name__ == "__main__":
    main()
