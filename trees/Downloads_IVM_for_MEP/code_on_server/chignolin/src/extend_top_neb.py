#!/usr/bin/env python3
"""Independently rerun the three selected NEB configurations to 2000 calls."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_one(root, item, cpu_threads):
    script = root / "src" / "run_neb.py"
    name = (
        f"k{item['spring_k']:g}_a{item['alpha']:g}_{item['optimizer']}"
    ).replace(".", "p")
    log = root / "logs" / "neb-top3-2000" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(Path(sys.executable).resolve()),
        str(script),
        "--root",
        str(root),
        "--run-label",
        "top3",
        "--spring-k",
        str(item["spring_k"]),
        "--alpha",
        str(item["alpha"]),
        "--optimizer",
        item["optimizer"],
        "--budget",
        "2000",
    ]
    environment = os.environ.copy()
    environment["OPENMM_CPU_THREADS"] = str(cpu_threads)
    environment["OMP_NUM_THREADS"] = str(cpu_threads)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(root / "src")
    start = time.time()
    with log.open("w") as handle:
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
        )
    saved = ""
    for line in reversed(log.read_text().splitlines()):
        if line.startswith("Saved "):
            saved = line.split("Saved ", 1)[1]
            break
    return {
        "name": name,
        "returncode": result.returncode,
        "wall_seconds": time.time() - start,
        "log": str(log),
        "result": saved,
        "configuration": item,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    analysis = json.loads(
        (root / "reports" / "neb_scan_analysis.json").read_text()
    )
    top_three = analysis["top_three"]
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = [
            executor.submit(run_one, root, item, args.cpu_threads)
            for item in top_three
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{'ok' if result['returncode'] == 0 else 'FAILED'} "
                f"{result['name']} {result['wall_seconds']:.1f}s",
                flush=True,
            )
    report = {
        "source_scan_analysis": str(
            root / "reports" / "neb_scan_analysis.json"
        ),
        "results": sorted(results, key=lambda item: item["name"]),
    }
    output = root / "reports" / "neb_top3_2000_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    if any(result["returncode"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
