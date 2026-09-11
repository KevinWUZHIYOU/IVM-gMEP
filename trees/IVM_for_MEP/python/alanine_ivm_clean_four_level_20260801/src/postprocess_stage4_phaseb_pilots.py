#!/usr/bin/env python3
"""Independent dense barriers at raw/committed checkpoints of phase-B pilots."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def select_calls(result: Path) -> list[int]:
    with np.load(result, allow_pickle=False) as saved:
        calls = np.asarray(saved["calls_per_image"], dtype=int)
        stored = np.unique(np.asarray(saved["stored_path_calls"], dtype=int))
        iperp = np.asarray(saved["i_perp"], dtype=float)
        committed = calls[np.asarray(saved["reparam_committed"], dtype=bool)]
    selected = set(int(value) for value in committed)
    for target in np.geomspace(1, stored[-1], 12):
        selected.add(int(stored[np.argmin(np.abs(stored - target))]))
    for threshold in (0.002, 0.001, 5e-4, 3e-4, 2.1e-4, 1.5e-4, 1.2e-4):
        indices = np.flatnonzero(iperp <= threshold)
        if indices.size:
            selected.add(int(calls[indices[0]]))
    selected.update((int(stored[0]), int(stored[-1])))
    return sorted(selected)


def run_one(root: Path, result: Path):
    variant = result.name.split("pilot_", 1)[1].split("_a20_", 1)[0]
    label = f"clean_stage4_phaseB_pilot_{variant}_dense"
    output = root / "runs/diagnostics" / f"dense_barrier_{label}.npz"
    calls = select_calls(result)
    if output.exists():
        return {
            "variant": variant,
            "status": "skipped_existing",
            "result": str(result),
            "dense": str(output),
            "calls": calls,
        }
    command = [
        sys.executable,
        str(root / "src/postprocess_dense_barrier.py"),
        "--root", str(root),
        "--result", str(result),
        "--output-label", label,
        "--workers", "1",
        "--calls", ",".join(str(value) for value in calls),
    ]
    completed = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log_dir = root / "logs/stage4_phaseb_pilots" / variant
    (log_dir / "dense_command.json").write_text(json.dumps(command, indent=2) + "\n")
    (log_dir / "dense_stdout.txt").write_text(completed.stdout)
    (log_dir / "dense_stderr.txt").write_text(completed.stderr)
    return {
        "variant": variant,
        "status": "completed" if completed.returncode == 0 and output.exists() else "failed",
        "result": str(result),
        "dense": str(output),
        "calls": calls,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = sorted((root / "runs/stage4_inexact_optimized_resample/pilots").glob("*.npz"))
    payloads = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_one, root, result) for result in results]
        for future in concurrent.futures.as_completed(futures):
            payload = future.result()
            payloads.append(payload)
            print(json.dumps(payload, sort_keys=True), flush=True)
    output = root / "reports/stage4_phaseb_policy_pilots_dense.json"
    output.write_text(json.dumps(payloads, indent=2) + "\n")
    if any(item["status"] == "failed" for item in payloads):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
