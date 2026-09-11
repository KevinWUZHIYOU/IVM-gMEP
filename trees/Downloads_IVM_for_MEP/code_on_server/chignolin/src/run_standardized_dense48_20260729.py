#!/usr/bin/env python3
"""Run one universal 48-checkpoint dense-barrier protocol for all methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


N_CHECKPOINTS = 48
N_EARLY = 12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def standardized_indices(total_force: np.ndarray, count: int = 48) -> np.ndarray:
    """Select early records plus force-axis coverage, then fill largest gaps."""
    total_force = np.asarray(total_force, dtype=np.int64)
    if np.any(np.diff(total_force) < 0):
        raise ValueError("total-force axis must be monotone")
    n_records = len(total_force)
    if n_records <= int(count):
        return np.arange(n_records, dtype=int)
    selected = set(range(min(N_EARLY, n_records)))
    remaining_targets = max(0, int(count) - len(selected))
    targets = np.linspace(
        float(total_force[max(selected)]),
        float(total_force[-1]),
        remaining_targets + 1,
    )[1:]
    for target in targets:
        eligible = np.flatnonzero(total_force <= target)
        selected.add(int(eligible[-1]) if eligible.size else 0)
    while len(selected) < int(count):
        ordered = np.asarray(sorted(selected), dtype=int)
        candidates = np.asarray(
            [index for index in range(n_records) if index not in selected],
            dtype=int,
        )
        nearest = np.min(
            np.abs(
                total_force[candidates, None]
                - total_force[ordered][None, :]
            ),
            axis=1,
        )
        selected.add(int(candidates[int(np.argmax(nearest))]))
    if len(selected) > int(count):
        # This can occur only through an early/target overlap edge case.
        ordered = sorted(selected)
        protected = set(range(min(N_EARLY, n_records)))
        while len(ordered) > int(count):
            removable = [
                index for index in ordered if index not in protected
            ]
            best = min(
                removable,
                key=lambda index: min(
                    abs(total_force[index] - total_force[other])
                    for other in ordered
                    if other != index
                ),
            )
            ordered.remove(best)
        selected = set(ordered)
    return np.asarray(sorted(selected), dtype=int)


def parse_record(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--record must be formatted as NAME=/absolute/result.npz"
        )
    name, path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("record name may not be empty")
    return name.strip(), Path(path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--record", action="append", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--python", type=Path, default=Path(sys.executable)
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    report_dir = root / "reports" / args.namespace
    log_dir = root / "logs" / args.namespace
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for value in args.record:
        name, result = parse_record(value)
        if not result.exists():
            raise FileNotFoundError(result)
        with np.load(result, allow_pickle=False) as saved:
            calls = np.asarray(saved["calls_per_image"], dtype=int)
            stored_calls = np.asarray(
                saved["stored_path_calls"], dtype=int
            )
            total_force = np.asarray(
                saved["counter_vectors"], dtype=np.int64
            ).sum(axis=1)
            initial_hash = str(saved["initial_path_sha256"])
        if not np.array_equal(calls, stored_calls):
            raise RuntimeError(
                f"{name} does not store one path per measured state"
            )
        indices = standardized_indices(total_force, N_CHECKPOINTS)
        selected_calls = calls[indices]
        if len(np.unique(selected_calls)) != len(selected_calls):
            # The postprocessor selects by call label, so retain the final
            # state at each label and fill with further unique labels.
            unique_final = {}
            for index in range(len(calls)):
                unique_final[int(calls[index])] = index
            eligible_indices = np.asarray(
                sorted(unique_final.values()), dtype=int
            )
            relative = standardized_indices(
                total_force[eligible_indices],
                min(N_CHECKPOINTS, len(eligible_indices)),
            )
            indices = eligible_indices[relative]
            selected_calls = calls[indices]
        label = f"{args.namespace}_{name.lower().replace(' ', '_')}"
        output = (
            root / "runs" / "diagnostics" / f"dense_barrier_{label}.npz"
        )
        audit = output.with_suffix(".audit.json")
        command = [
            str(args.python.expanduser().resolve()),
            str(root / "src" / "postprocess_dense_barrier.py"),
            "--root",
            str(root),
            "--result",
            str(result),
            "--output-label",
            label,
            "--workers",
            str(max(1, int(args.workers))),
            "--calls",
            ",".join(str(int(call)) for call in selected_calls),
        ]
        log = log_dir / f"{name.lower().replace(' ', '_')}.log"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")
        if not (output.exists() and audit.exists()):
            with log.open("w") as stream:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if completed.returncode:
                raise RuntimeError(
                    f"Dense diagnostic failed for {name}; see {log}"
                )
        audit_payload = json.loads(audit.read_text())
        with np.load(output, allow_pickle=False) as dense:
            diagnostic_backend = int(
                dense["diagnostic_backend_raw_joint_evaluations"]
            )
            diagnostic_misses = int(
                dense["diagnostic_evaluator_cache_misses"]
            )
            dense_calls = np.asarray(dense["calls"], dtype=int)
        if (
            diagnostic_backend != diagnostic_misses
            or diagnostic_backend
            != int(
                audit_payload[
                    "diagnostic_backend_raw_joint_evaluations"
                ]
            )
            or not audit_payload["counts_excluded_from_optimization_axis"]
            or not np.array_equal(dense_calls, selected_calls)
        ):
            raise RuntimeError(f"Dense audit mismatch for {name}")
        records.append(
            {
                "name": name,
                "source_result": str(result),
                "source_result_sha256": sha256(result),
                "initial_path_sha256": initial_hash,
                "selected_record_indices": indices.tolist(),
                "selected_calls_per_image": selected_calls.tolist(),
                "selected_total_number_of_forces": total_force[
                    indices
                ].tolist(),
                "dense_result": str(output),
                "dense_result_sha256": sha256(output),
                "dense_audit": str(audit),
                "dense_audit_sha256": sha256(audit),
                "diagnostic_joint_evaluations": diagnostic_backend,
                "command": command,
            }
        )
    initial_hashes = sorted(
        {record["initial_path_sha256"] for record in records}
    )
    if len(initial_hashes) != 1:
        raise RuntimeError(f"Methods use mixed initial paths: {initial_hashes}")
    manifest = {
        "selection_protocol": (
            "The first 12 measured states plus total-Number-of-Forces "
            "coverage, filled by farthest force-axis gaps to exactly 48 "
            "unique stored call labels per method when available."
        ),
        "checkpoints_per_method": N_CHECKPOINTS,
        "cost_axis": "counter_vectors.sum(axis=1), all 22 images",
        "barrier_protocol": (
            "Real OpenMM energies after 10 and 20 linear subdivisions per "
            "segment; adaptively double to 40/80 if successive barriers "
            "differ by more than 0.05 kcal/mol."
        ),
        "diagnostic_calls_excluded_from_optimization_axis": True,
        "common_initial_path_sha256": initial_hashes[0],
        "records": records,
    }
    output_manifest = report_dir / "dense48_manifest.json"
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(output_manifest)


if __name__ == "__main__":
    main()
