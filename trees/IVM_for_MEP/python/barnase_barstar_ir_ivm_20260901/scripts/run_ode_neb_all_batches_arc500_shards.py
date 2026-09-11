#!/usr/bin/env python3
"""Run the prospectively frozen ODE--NEB all-batch arc500 shards serially."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from evaluate_discrete_iperp_densified_barrier import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=16)
    args = parser.parse_args()
    contract = args.contract.expanduser().resolve(strict=True)
    contract_sha = sha256_file(contract)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Path(__file__).resolve().parent / "evaluate_ode_neb_all_batches_arc500.py"
    for index in range(args.shard_count):
        output = output_dir / f"shard_{index:02d}_of_{args.shard_count:02d}.npz"
        audit = output.with_suffix(".audit.json")
        wal = output.with_suffix(".energy.wal.jsonl")
        present = [path.exists() for path in (output, audit, wal)]
        if all(present):
            receipt = json.loads(audit.read_text(encoding="utf-8"))
            if (
                receipt.get("status") != "PASS"
                or receipt.get("contract_sha256") != contract_sha
                or receipt.get("output_sha256") != sha256_file(output)
                or receipt.get("energy_wal_sha256") != sha256_file(wal)
            ):
                raise RuntimeError(f"existing shard {index} failed validation")
            print(f"shard {index}/{args.shard_count}: existing PASS", flush=True)
            continue
        if any(present):
            raise RuntimeError(f"shard {index} has partial artifacts; refusing overwrite")
        command = [
            sys.executable,
            str(evaluator),
            "--contract",
            str(contract),
            "--output",
            str(output),
            "--shard-index",
            str(index),
            "--shard-count",
            str(args.shard_count),
        ]
        print(f"shard {index}/{args.shard_count}: starting", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        print(f"shard {index}/{args.shard_count}: PASS", flush=True)


if __name__ == "__main__":
    main()
